"""Preregister source-only checkpoints, then author and verify passive references."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from context_policy import ContextBudgetError
from freeze_passive_dataset import Bindings, bind_source_assets, project_path
from passive_contract import (
    PassiveContextBuilder,
    PassivePredictionSession,
    apply_passive_edit,
    canonical_messages,
    parse_snapshot,
    token_ids,
)
from passive_dataset import ROOT, RESERVED, compact_transaction, digest, read
from replay_session import replay, session_hash
from verify_passive_dataset import rows, tokenizers
from verify_release_export import Exporter, verify_export_delta, verify_operation

VERSION = "passive-source-checkpoints-v2"
SEED = "passive-checkpoints-v2-2026-09-26"
POLICY = {
    "checkpoints_per_recorded_group": 2,
    "source_selection": "sha256_rank_source_id_distinct_when_available",
    "cursor_selection": "sha256_uniform_time_in_first_and_second_timeline_halves",
    "selection": "first_cursor_only_second_up_to_three_forward_items_in_same_run",
    "nominal_window_items": 64,
    "history": "empty_on_unmodified_registered_document",
    "label_inspection": "after_immutable_preregistration_and_context_materialization",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    return str(project_path(path).relative_to(ROOT))


def save_new(path, value):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def save_rows(path, values):
    with project_path(path).open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def rank(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()


def structural_items(document):
    """Project timing/identity only: wording is never consulted by the schedule."""
    result, paragraph, previous, run, seen = [], -1, None, 0, set()
    clock = 0.0
    for index, item in enumerate(document["content"]):
        require(item["uuid"] not in seen, "Repeated source identity.")
        seen.add(item["uuid"])
        if item["type"] == "paragraph_start":
            paragraph += 1
        elif item["type"] in {"text", "non_text", "artificial_silence"}:
            length = item["length"]
            require(math.isfinite(length) and length > 0, "Invalid source duration.")
            require(paragraph >= 0, "Timed item outside a paragraph.")
            if previous is not None and (
                "source" not in item
                or "source" not in previous
                or item["source"] != previous["source"]
                or abs(
                    item["sourceStart"] - previous["sourceStart"] - previous["length"]
                )
                > 1e-6
            ):
                run += 1
            result.append(
                {
                    "alias": len(result),
                    "index": index,
                    "uuid": item["uuid"],
                    "paragraph": paragraph,
                    "source_run": run,
                    "source": item.get("source"),
                    "source_start": item.get("sourceStart"),
                    "timeline_start": clock,
                    "timeline_end": clock + length,
                }
            )
            clock += length
            previous = item
    require(bool(result), "Empty source document.")
    return result


def registered_sources(registries, bindings):
    sources, groups = {}, {}
    for name in registries:
        registry = read(ROOT / bindings.bind(name))
        for row in registry["sources"]:
            group, split = row["source_group_id"], row["split"]
            require(
                group not in RESERVED and split in {"train", "dev"},
                "Reserved or unknown source split; do not open its body.",
            )
            require(
                groups.setdefault(group, split) == split,
                "Source family crosses splits.",
            )
            require(row["id"] not in sources, "Duplicate source identity.")
            if row.get("media_kind") == "authored_symbolic_timeline":
                continue
            require(
                bool(row.get("media_path")), "A recorded checkpoint needs actual media."
            )
            sources[row["id"]] = row
    return sources


def source_document(source, bindings):
    name = relative(source["document_path"])
    expected = source.get("document_sha256") or source.get("file_sha256", {}).get(name)
    require(bool(expected), "Source registry lacks its physical document hash.")
    return read(ROOT / bindings.bind(name, expected))


def schedule(registries, seed, bindings):
    sources = registered_sources(registries, bindings)
    groups = defaultdict(list)
    for source in sources.values():
        groups[source["source_group_id"]].append(source)
    checkpoints = []
    for group, members in sorted(groups.items()):
        ranked = sorted(members, key=lambda s: rank(seed, group, "source", s["id"]))
        for slot in range(2):
            source = ranked[slot % len(ranked)]
            structure = structural_items(source_document(source, bindings))
            draw = (int(rank(seed, group, "time", slot)[:16], 16) >> 11) / 2**53
            moment = structure[-1]["timeline_end"] * (slot + draw) / 2
            focus = next(i for i in structure if i["timeline_end"] > moment)
            selection = []
            if slot == 1:
                for item in structure[focus["alias"] : focus["alias"] + 3]:
                    if (item["paragraph"], item["source_run"]) != (
                        focus["paragraph"],
                        focus["source_run"],
                    ):
                        break
                    selection.append(item["uuid"])
            low = max(0, min(focus["alias"] - 32, len(structure) - 64))
            high = min(len(structure), low + 64)
            region = None
            if (
                structure[low]["source"] is not None
                and structure[low]["source_run"] == structure[high - 1]["source_run"]
            ):
                last = structure[high - 1]
                region = [
                    structure[low]["source_start"],
                    last["source_start"]
                    + last["timeline_end"]
                    - last["timeline_start"],
                ]
            identifier = (
                f"{source['id']}-checkpoint-{slot}-{rank(seed, group, slot)[:12]}"
            )
            checkpoints.append(
                {
                    "id": identifier,
                    "source_id": source["id"],
                    "source_group_id": group,
                    "split": source["split"],
                    "language": source["language"],
                    "media_kind": source.get("media_kind", "recording"),
                    "document_path": source["document_path"],
                    "document_sha256": bindings.hashes[
                        relative(source["document_path"])
                    ],
                    "structural_sha256": session_hash(structure),
                    "slot": slot,
                    "timeline_time": moment,
                    "focus_uuid": focus["uuid"],
                    "focus_alias": focus["alias"],
                    "selection_uuids": selection,
                    "timeline_region": [
                        structure[low]["timeline_start"],
                        structure[high - 1]["timeline_end"],
                    ],
                    "source_region": region,
                    "source_opportunity_id": identifier,
                    "paired_source_family": group,
                }
            )
    return checkpoints


def bind_code(bindings):
    old = ROOT / "artifacts/datasets/nep-release-v1/frozen/manifest.json"
    manifest = read(ROOT / bindings.bind(old))
    for name, expected in manifest["code_and_recipe_sha256"].items():
        bindings.bind(name, expected)
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).resolve().is_relative_to(ROOT):
            bindings.bind(name)
    for name in (
        "ml/prompt-passive-v1.txt",
        "ml/export_release_states.cjs",
        "app/src/core/webvtt.ts",
        "app/package.json",
        "app/yarn.lock",
    ):
        if (ROOT / name).exists():
            bindings.bind(name)


def preregister(registries, output, seed=SEED):
    bindings = Bindings()
    registries = [relative(p) for p in registries]
    checkpoints = schedule(registries, seed, bindings)
    value = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "policy": POLICY,
        "source_registries": registries,
        "source_files_sha256": bindings.hashes,
        "schedule_code_sha256": {relative(__file__): digest(__file__)},
        "checkpoints": checkpoints,
        "scope": "Reference-independent sampling in previously registered train/dev recordings; not a claim that the corpus itself was never inspected.",
    }
    bindings.stable()
    save_new(output, value)
    return value


def validate_preregistration(path, bindings):
    value = read(ROOT / bindings.bind(path))
    require(
        value["version"] == VERSION and value["policy"] == POLICY,
        "Unknown checkpoint schedule.",
    )
    for name, expected in {
        **value["source_files_sha256"],
        **value["schedule_code_sha256"],
    }.items():
        bindings.bind(name, expected)
    require(
        schedule(value["source_registries"], value["seed"], bindings)
        == value["checkpoints"],
        "Preregistered focus/selection differs from the metadata-only schedule.",
    )
    return value


def make_builder(bindings):
    dependencies = {}
    loaded = tokenizers(dependencies)
    for name, expected in dependencies.items():
        bindings.bind(name, expected)
    return PassiveContextBuilder(loaded["qwen06"], budget_tokenizers=loaded)


def checkpoint_meta(checkpoint, bundle):
    return {
        **{
            k: checkpoint[k]
            for k in (
                "source_id",
                "source_group_id",
                "split",
                "language",
                "media_kind",
                "source_region",
                "source_opportunity_id",
                "paired_source_family",
            )
        },
        "episode_id": checkpoint["id"],
        "episode_step": 0,
        "actor": "synthetic",
        "history_origin": "empty_on_registered_current_document",
        "kind": "reference_independent_source_checkpoint",
        "context": bundle["context"],
        "input_sha256": session_hash(canonical_messages(bundle["input"])),
        "document_before_hash": bundle["document_hash"],
        "training_eligible": False,
        "review_status": "unlabeled_checkpoint",
        "cursor_sampling": VERSION,
        "registered_focus_alias": checkpoint["focus_alias"],
    }


def prepare_contexts(preregistration, output, builder=None):
    bindings = Bindings()
    registration = validate_preregistration(preregistration, bindings)
    builder = builder or make_builder(bindings)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    contexts, exclusions = [], []
    for checkpoint in registration["checkpoints"]:
        document = read(ROOT / checkpoint["document_path"])
        try:
            bundle = builder.build(
                document,
                checkpoint["focus_uuid"],
                as_of=0,
                selection=checkpoint["selection_uuids"],
            )
        except ContextBudgetError as error:
            exclusions.append(
                {
                    "checkpoint_id": checkpoint["id"],
                    "status": "unsupported_context_budget",
                    "reason": str(error),
                }
            )
            continue
        identifier = checkpoint["id"] + "-s00"
        saved = {
            "checkpoint": checkpoint,
            "bundle": bundle,
            "messages": canonical_messages(bundle["input"]),
            "token_ids": {
                name: token_ids(t, bundle["input"])
                for name, t in (
                    builder.budget_tokenizers or {"primary": builder.tokenizer}
                ).items()
            },
        }
        name = f"contexts/{identifier}.json"
        save_new(output / name, saved)
        bindings.bind(output / name)
        contexts.append(
            {
                "id": identifier,
                "input": bundle["input"],
                "metadata": checkpoint_meta(checkpoint, bundle),
                "reproduction": {"prepared_context": name},
            }
        )
    save_rows(output / "unlabeled-contexts.jsonl", contexts)
    save_rows(output / "context-exclusions.jsonl", exclusions)
    for name in ("unlabeled-contexts.jsonl", "context-exclusions.jsonl"):
        bindings.bind(output / name)
    bind_code(bindings)
    bindings.stable()
    save_new(
        output / "manifest.json",
        {
            "version": VERSION,
            "preregistration": relative(preregistration),
            "preregistration_sha256": digest(project_path(preregistration)),
            "source_registries": registration["source_registries"],
            "scheduled_checkpoints": len(registration["checkpoints"]),
            "prepared_contexts": len(contexts),
            "context_exclusions": len(exclusions),
            "dependencies_sha256": bindings.hashes,
        },
    )
    return contexts


def load_contexts(directory, bindings):
    directory = project_path(directory)
    manifest = read(ROOT / bindings.bind(directory / "manifest.json"))
    for name, expected in manifest["dependencies_sha256"].items():
        bindings.bind(name, expected)
    registration = validate_preregistration(manifest["preregistration"], bindings)
    require(
        digest(project_path(manifest["preregistration"]))
        == manifest["preregistration_sha256"],
        "Preregistration changed.",
    )
    contexts = rows(directory / "unlabeled-contexts.jsonl")
    scheduled = {c["id"] for c in registration["checkpoints"]}
    excluded = rows(directory / "context-exclusions.jsonl")
    require(
        {c["metadata"]["episode_id"] for c in contexts}
        | {c["checkpoint_id"] for c in excluded}
        == scheduled,
        "Incomplete checkpoint census.",
    )
    require(
        len(contexts) + len(excluded) == len(scheduled), "Duplicated checkpoint census."
    )
    require(
        manifest["scheduled_checkpoints"] == len(scheduled)
        and manifest["prepared_contexts"] == len(contexts)
        and manifest["context_exclusions"] == len(excluded),
        "Prepared-context counts differ from their complete census.",
    )
    by_id = {c["id"]: c for c in registration["checkpoints"]}
    for context in contexts:
        require(
            set(context) == {"id", "input", "metadata", "reproduction"},
            "Unexpected context fields.",
        )
        checkpoint = by_id[context["metadata"]["episode_id"]]
        require(context["id"] == checkpoint["id"] + "-s00", "Context identity changed.")
        expected_path = f"contexts/{context['id']}.json"
        require(
            context["reproduction"] == {"prepared_context": expected_path},
            "Prepared context path changed.",
        )
        prepared = read(ROOT / bindings.bind(directory / expected_path))
        require(
            prepared["checkpoint"] == checkpoint,
            "Prepared cursor differs from preregistration.",
        )
        require(
            prepared["bundle"]["input"] == context["input"],
            "Prepared predictor input changed.",
        )
        require(
            parse_snapshot(context["input"]).model_dump() == context["input"],
            "Noncanonical predictor input.",
        )
        require(
            prepared["messages"] == canonical_messages(context["input"]),
            "Prepared messages changed.",
        )
        require(
            context["metadata"] == checkpoint_meta(checkpoint, prepared["bundle"]),
            "Checkpoint metadata changed.",
        )
    return manifest, contexts


def undo_redo_trace(before, after, action, group):
    events = [
        {
            "kind": "start",
            "seq": 0,
            "schema_version": 2,
            "hash_format": "typed-json-f64-v1",
            "actor": "synthetic",
            "collection_mode": "synthetic",
            "generator": VERSION,
            "document": before,
            "document_hash": session_hash(before),
            "sources": [group],
        }
    ]
    if before != after:
        events.extend(
            [
                compact_transaction(before, after, 1, 1, action),
                compact_transaction(after, before, 2, 2, action, origin="undo"),
                compact_transaction(before, after, 3, 3, action, origin="redo"),
            ]
        )
        undone = events[:3] + [
            {
                "kind": "end",
                "seq": 3,
                "revision": 2,
                "document_hash": session_hash(before),
                "transactions": 2,
            }
        ]
        require(
            replay(undone)[0] == before, "Undo did not restore the exact source state."
        )
    events.append(
        {
            "kind": "end",
            "seq": len(events),
            "revision": len(events) - 1,
            "document_hash": session_hash(after),
            "transactions": len(events) - 1,
        }
    )
    actual, summary = replay(events)
    require(actual == after and summary["complete"], "Reference replay/redo failed.")
    events[0].pop("document")
    return events


def checked_labels(contexts, labels):
    require(
        set(labels) == {"version", "contexts_manifest_sha256", "cases"}
        and labels["version"] == VERSION,
        "Unknown label document; keep teacher metadata outside the predictor.",
    )
    indexed = {r["case_id"]: r for r in labels["cases"]}
    require(
        len(indexed) == len(labels["cases"])
        and set(indexed) == {c["id"] for c in contexts},
        "Every observed context needs one explicit resolution, not an inferred no_action.",
    )
    for context in contexts:
        label = indexed[context["id"]]
        common = {"case_id", "input_sha256", "resolution", "reason"}
        require(
            label["resolution"] in {"reference", "unresolved", "unsupported"},
            "Unknown label resolution.",
        )
        expected_keys = common | (
            {"primary", "alternatives"} if label["resolution"] == "reference" else set()
        )
        require(
            set(label) == expected_keys
            and isinstance(label["reason"], str)
            and bool(label["reason"].strip()),
            "Reference/exclusion fields must be explicit and complete.",
        )
        require(
            label["input_sha256"] == session_hash(canonical_messages(context["input"])),
            "Label used another input.",
        )
    return indexed


def materialize_reviews(context_directory, labels_path, output):
    bindings = Bindings()
    manifest, contexts = load_contexts(context_directory, bindings)
    labels = read(ROOT / bindings.bind(labels_path))
    require(
        labels["contexts_manifest_sha256"]
        == digest(project_path(context_directory) / "manifest.json"),
        "Labels used another context manifest.",
    )
    indexed = checked_labels(contexts, labels)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases, exclusions, specs = [], [], []
    for context in contexts:
        saved = read(
            project_path(context_directory)
            / context["reproduction"]["prepared_context"]
        )
        checkpoint, bundle = saved["checkpoint"], saved["bundle"]
        require(bundle["input"] == context["input"], "Prepared context changed.")
        label = indexed[context["id"]]
        spec = {
            "id": checkpoint["id"],
            "source_id": checkpoint["source_id"],
            "kind": context["metadata"]["kind"],
            "source_region": checkpoint["source_region"],
            "corruptions": [],
            "checkpoint": checkpoint,
            "resolution": label["resolution"],
            "rationale": label["reason"],
        }
        document = read(ROOT / checkpoint["document_path"])
        journals, after_hashes = [], []
        if label["resolution"] == "reference":
            references = [label["primary"], *label["alternatives"]]
            require(
                len({session_hash(p) for p in references}) == len(references),
                "Duplicated reference.",
            )
            for prediction in references:
                after = apply_passive_edit(document, bundle, prediction)
                verify_operation(document, after, bundle["mapping"], prediction)
                journals.append(
                    undo_redo_trace(
                        document,
                        after,
                        prediction["action"],
                        checkpoint["source_group_id"],
                    )
                )
                after_hashes.append(session_hash(after))
            spec["references"] = references
            case = {
                "id": context["id"],
                "input": context["input"],
                "expectedOutput": references[0],
                "acceptableOutputs": references[1:],
                "metadata": {
                    **context["metadata"],
                    "decision_rationale": label["reason"],
                    "review_status": "pending_independent_admission",
                    "document_after_hash": after_hashes[0],
                },
                "reproduction": {
                    "episode": f"episodes/{checkpoint['id']}.json",
                    "step": 0,
                },
            }
            cases.append(case)
        else:
            exclusions.append(
                {
                    "case_id": context["id"],
                    "episode_id": checkpoint["id"],
                    "input_sha256": label["input_sha256"],
                    "status": label["resolution"],
                    "reason": label["reason"],
                }
            )
        specs.append(spec)
        save_new(
            output / "episodes" / f"{checkpoint['id']}.json",
            {
                "spec": spec,
                "source_document": checkpoint["document_path"],
                "source_document_sha256": checkpoint["document_sha256"],
                "prepared_context": relative(
                    project_path(context_directory)
                    / context["reproduction"]["prepared_context"]
                ),
                "reference_journals": journals,
                "after_hashes": after_hashes,
            },
        )
    for split in ("train", "dev"):
        save_rows(
            output / f"{split}-candidates.jsonl",
            [c for c in cases if c["metadata"]["split"] == split],
        )
    save_rows(output / "exclusions.jsonl", exclusions)
    save_new(output / "recipe.json", {"version": VERSION, "episodes": specs})
    bindings.stable()
    save_new(
        output / "manifest.json",
        {
            "version": VERSION,
            "context_directory": relative(context_directory),
            "labels_path": relative(labels_path),
            "labels_sha256": digest(project_path(labels_path)),
            "source_registries": manifest["source_registries"],
            "candidate_count": len(cases),
            "unresolved_or_unsupported": len(exclusions),
            "dependencies_sha256": bindings.hashes,
        },
    )
    return cases


def verify(directory, output, builder=None):
    bindings = Bindings()
    directory = project_path(directory)
    manifest = read(ROOT / bindings.bind(directory / "manifest.json"))
    for name, expected in manifest["dependencies_sha256"].items():
        bindings.bind(name, expected)
    context_manifest, contexts = load_contexts(manifest["context_directory"], bindings)
    labels = read(
        ROOT / bindings.bind(manifest["labels_path"], manifest["labels_sha256"])
    )
    require(
        labels["contexts_manifest_sha256"]
        == digest(project_path(manifest["context_directory"]) / "manifest.json"),
        "Labels used another context manifest.",
    )
    indexed = checked_labels(contexts, labels)
    builder = builder or make_builder(bindings)
    bind_code(bindings)
    candidate_hashes = {}
    for name in (
        "train-candidates.jsonl",
        "dev-candidates.jsonl",
        "recipe.json",
        "exclusions.jsonl",
    ):
        candidate_hashes[name] = bindings.hashes[bindings.bind(directory / name)]
    candidates = []
    for split in ("train", "dev"):
        batch = rows(directory / f"{split}-candidates.jsonl")
        require(
            all(c["metadata"]["split"] == split for c in batch),
            "Candidate split differs from file placement.",
        )
        candidates.extend(batch)
    by_id = {c["id"]: c for c in candidates}
    require(
        manifest["candidate_count"] == len(candidates)
        and manifest["unresolved_or_unsupported"] == len(contexts) - len(candidates),
        "Candidate manifest counts differ from explicit resolutions.",
    )
    require(
        len(by_id) == len(candidates)
        and set(by_id)
        == {k for k, v in indexed.items() if v["resolution"] == "reference"},
        "Candidate resolution census differs.",
    )
    require(
        {e["case_id"] for e in rows(directory / "exclusions.jsonl")}
        == set(indexed) - set(by_id),
        "Exclusion census differs.",
    )
    specs = {s["id"]: s for s in read(directory / "recipe.json")["episodes"]}
    require(
        set(specs) == {c["metadata"]["episode_id"] for c in contexts},
        "Episode census differs.",
    )
    asset_cases, outcomes, errors, parity = [], [], [], 0
    exporter = Exporter()
    try:
        for context in contexts:
            identifier = context["id"]
            episode_path = (
                directory / "episodes" / f"{context['metadata']['episode_id']}.json"
            )
            saved = read(ROOT / bindings.bind(episode_path))
            require(
                saved["spec"] == specs[context["metadata"]["episode_id"]],
                "Episode differs from recipe.",
            )
            checkpoint = saved["spec"]["checkpoint"]
            require(
                saved["prepared_context"]
                == relative(
                    project_path(manifest["context_directory"])
                    / context["reproduction"]["prepared_context"]
                ),
                "Episode changed its prepared context path.",
            )
            prepared = read(ROOT / bindings.bind(saved["prepared_context"]))
            require(
                prepared["checkpoint"] == checkpoint,
                "Episode moved the registered cursor.",
            )
            asset_cases.append(
                {
                    "metadata": context["metadata"],
                    "reproduction": {
                        "candidate_directory": relative(directory),
                        "episode": str(episode_path.relative_to(directory)),
                    },
                }
            )
            document = read(
                ROOT
                / bindings.bind(
                    saved["source_document"], saved["source_document_sha256"]
                )
            )
            runtime = PassivePredictionSession(identifier, builder).prepare(
                document,
                checkpoint["focus_uuid"],
                as_of=0,
                selection=checkpoint["selection_uuids"],
            )
            require(
                runtime == prepared["bundle"] and runtime["input"] == context["input"],
                "Runtime checkpoint differs from the preregistered observation.",
            )
            require(
                canonical_messages(runtime["input"]) == prepared["messages"],
                "Canonical messages changed.",
            )
            actual_ids = {
                name: token_ids(t, runtime["input"])
                for name, t in (
                    builder.budget_tokenizers or {"primary": builder.tokenizer}
                ).items()
            }
            require(
                actual_ids == prepared["token_ids"], "Tokenizer prompt IDs changed."
            )
            parity += 1
            if identifier not in by_id:
                require(
                    not saved["reference_journals"] and not saved["after_hashes"],
                    "Unresolved input fabricated an action.",
                )
                continue
            case, label = by_id[identifier], indexed[identifier]
            require(
                case["reproduction"]
                == {"episode": str(episode_path.relative_to(directory)), "step": 0},
                "Candidate reproduction path changed.",
            )
            require(
                case["input"] == context["input"]
                and case["metadata"]["input_sha256"]
                == session_hash(canonical_messages(case["input"])),
                "Candidate input changed.",
            )
            for key, value in context["metadata"].items():
                if key != "review_status":
                    require(
                        case["metadata"][key] == value,
                        "Candidate source/checkpoint metadata changed.",
                    )
            references = [case["expectedOutput"], *case["acceptableOutputs"]]
            require(
                references
                == [label["primary"], *label["alternatives"]]
                == saved["spec"]["references"],
                "Candidate references differ from authored decisions.",
            )
            require(
                len(references)
                == len(saved["reference_journals"])
                == len(saved["after_hashes"]),
                "Reference trace census differs.",
            )
            require(
                case["metadata"]["document_after_hash"] == saved["after_hashes"][0],
                "Candidate after hash changed.",
            )
            require(
                case["metadata"]["decision_rationale"]
                == label["reason"]
                == saved["spec"]["rationale"],
                "Authoring rationale changed.",
            )
            line_length = runtime["input"]["caption_profile"]["export_line_length"]
            before_export = exporter.export(document, line_length)
            for index, prediction in enumerate(references):
                after = apply_passive_edit(document, runtime, prediction)
                operation = verify_operation(
                    document, after, runtime["mapping"], prediction
                )
                require(
                    session_hash(after) == saved["after_hashes"][index],
                    "Reference after hash changed.",
                )
                require(
                    saved["reference_journals"][index]
                    == undo_redo_trace(
                        document,
                        after,
                        prediction["action"],
                        checkpoint["source_group_id"],
                    ),
                    "Recorded replay/undo trace changed.",
                )
                audit = verify_export_delta(
                    before_export,
                    exporter.export(after, line_length),
                    operation,
                    prediction["action"],
                )
                outcomes.append(
                    {
                        "case_id": identifier,
                        "reference_index": index,
                        "status": "passed",
                        "undo_redo": "exact_source_restoration_passed",
                        **audit,
                    }
                )
    except (ValueError, AssertionError, RuntimeError) as error:
        errors.append({"case_id": identifier, "error": str(error)})
    finally:
        exporter.close()
    source_ids = bind_source_assets(
        asset_cases, manifest["source_registries"], bindings
    )
    bindings.stable()
    report = {
        "version": VERSION,
        "status": "failed" if errors else "passed",
        "errors": errors,
        "directory": relative(directory),
        "source_registries": manifest["source_registries"],
        "source_ids": source_ids,
        "candidate_hashes": candidate_hashes,
        "candidate_files_sha256": {
            relative(directory / name): candidate_hashes[name]
            for name in ("train-candidates.jsonl", "dev-candidates.jsonl")
        },
        "dependencies_sha256": bindings.hashes,
        "case_ids": sorted(by_id),
        "observed_case_ids": sorted(indexed),
        "excluded_case_ids": sorted(set(indexed) - set(by_id)),
        "scheduled_checkpoints": context_manifest["scheduled_checkpoints"],
        "observed_contexts": len(contexts),
        "actual_serving_message_parity_checks": parity,
        "candidates": len(candidates),
        "unresolved_or_unsupported": len(contexts) - len(candidates),
        "replayed_episodes": len({o["case_id"] for o in outcomes}),
        "accepted_outcomes_exported": len(outcomes),
        "outcomes": outcomes,
        "primary_actions": dict(
            Counter(c["expectedOutput"]["action"] for c in candidates)
        ),
        "semantic_admission": "pending_independent_review",
        "scope": "Shared passive runtime messages and pinned tokenizer IDs; no live editor, remote server, acoustic cut qualification or model ranking.",
    }
    save_new(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preregister")
    p.add_argument("--source-registry", nargs="+", required=True)
    p.add_argument("--seed", default=SEED)
    p.add_argument("--output", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--preregistration", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("materialize")
    p.add_argument("--contexts", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("verify")
    p.add_argument("--directory", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "preregister":
        result = preregister(args.source_registry, args.output, args.seed)
        print(
            json.dumps(
                {
                    "checkpoints": len(result["checkpoints"]),
                    "preregistration_sha256": digest(project_path(args.output)),
                }
            )
        )
    elif args.command == "prepare":
        print(
            json.dumps(
                {"contexts": len(prepare_contexts(args.preregistration, args.output))}
            )
        )
    elif args.command == "materialize":
        print(
            json.dumps(
                {
                    "candidates": len(
                        materialize_reviews(args.contexts, args.labels, args.output)
                    )
                }
            )
        )
    else:
        report = verify(args.directory, args.output)
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "status",
                        "errors",
                        "candidates",
                        "accepted_outcomes_exported",
                    )
                }
            )
        )
        if report["errors"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
