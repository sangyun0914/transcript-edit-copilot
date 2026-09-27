"""Package admitted passive data without changing its immutable parent release."""

import argparse
import bisect
import copy
import gzip
import hashlib
import json
import math
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from freeze_passive_dataset import Bindings, bind_source_assets, project_path
from passive_contract import PROMPT, canonical_messages
from passive_dataset import ROOT, RESERVED, digest, jsonl, read, write
from passive_formats_v2 import (
    INPUT_LIMIT, OUTPUT_LIMIT, REWRITE_PROMPT, VERSION, eligibility, encode,
    messages, parse, serialize,
)
from passive_training import training_messages
from replay_session import session_hash
from verify_passive_dataset import rows, tokenizers


def balanced_weights(cases):
    """Overlapping regions of one source share mass, including dependent replays."""
    splits = defaultdict(list)
    for case in cases:
        splits[case["metadata"]["split"]].append(case)
    for split_cases in splits.values():
        sources = defaultdict(list)
        for case in split_cases:
            m = case["metadata"]
            sources[(m["language"], m["source_group_id"], m["source_id"])].append(case)
        clusters = defaultdict(lambda: defaultdict(list))
        for (language, group, source), members in sorted(sources.items()):
            if any(c["metadata"]["source_region"] is None for c in members):
                if any(c["metadata"].get("media_kind") != "authored_symbolic_timeline" for c in members):
                    raise ValueError("A recording opportunity needs its actual source interval.")
                clusters[language][group].append(members)
                continue
            ordered = sorted(members, key=lambda c: (*c["metadata"]["source_region"], c["id"]))
            end, cluster = -math.inf, None
            for case in ordered:
                start, stop = case["metadata"]["source_region"]
                if not all(math.isfinite(n) for n in (start, stop)) or stop <= start:
                    raise ValueError("An opportunity needs a positive finite source interval.")
                if cluster is None or start >= end:
                    cluster = []
                    clusters[language][group].append(cluster)
                    end = stop
                else:
                    end = max(end, stop)
                cluster.append(case)
        for language, groups in clusters.items():
            for group, opportunities in groups.items():
                for index, members in enumerate(opportunities):
                    for case in members:
                        case["metadata"].update(
                            sampling_cluster=f"{split_cases[0]['metadata']['split']}:{language}:{group}:{index}",
                            sampling_weight=1 / len(clusters) / len(groups) / len(opportunities) / len(members),
                        )


def weighted_indices(cases, count, seed):
    if type(count) is not int or count < 1 or not cases:
        raise ValueError("Specify a positive sample count and a nonempty training set.")
    cumulative, total = [], 0.0
    for case in cases:
        m = case["metadata"]
        if m["split"] != "train" or m.get("training_eligible") is not True:
            raise ValueError("Development or pending rows cannot enter a training sampler.")
        weight = m["sampling_weight"]
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Sampling weights must be positive and finite.")
        total += weight
        cumulative.append(total)
    rng = random.Random(seed)
    return [bisect.bisect_right(cumulative, rng.random() * total) for _ in range(count)]


def bind_parent(directory, bindings):
    manifest = read(ROOT / bindings.bind(directory / "manifest.json"))
    for name, sha in manifest["files_sha256"].items():
        bindings.bind(directory / name, sha)
    for name, sha in manifest["dependencies_sha256"].items():
        bindings.bind(name, sha)
    result = []
    for split in ("train", "dev"):
        for c in rows(directory / f"{split}-cases.jsonl"):
            if c["metadata"]["split"] != split or c["metadata"].get("training_eligible") != (split == "train"):
                raise ValueError("Parent split/admission flags changed.")
            c["metadata"]["release_stratum"] = "curriculum_v1"
            result.append(c)
    return result


def checkpoint_dispositions(verification, directory, bindings, candidates):
    path = directory / "exclusions.jsonl"
    relative = str(path.relative_to(ROOT))
    sha = verification["dependencies_sha256"].get(relative)
    if not sha or verification.get("candidate_hashes", {}).get("exclusions.jsonl") != sha:
        raise ValueError("Checkpoint exclusions must be physically bound by verification.")
    bindings.bind(path, sha)
    exclusions = rows(path)
    accepted_ids = {c["id"] for c in candidates}
    excluded_ids = {c["case_id"] for c in exclusions}
    observed_ids = verification.get("observed_case_ids", [])
    recipe = read(directory / "recipe.json")
    recipe_ids = {e["id"] + "-s00" for e in recipe["episodes"]}
    if (len(excluded_ids) != len(exclusions) or accepted_ids & excluded_ids
            or len(observed_ids) != len(set(observed_ids))
            or accepted_ids | excluded_ids != set(observed_ids)
            or set(observed_ids) != recipe_ids
            or verification.get("scheduled_checkpoints") != len(recipe["episodes"])
            or verification.get("observed_contexts") != len(observed_ids)
            or verification.get("unresolved_or_unsupported") != len(exclusions)
            or set(verification.get("excluded_case_ids", [])) != excluded_ids
            or any(e.get("status") not in {"unresolved", "unsupported"}
                   or not e.get("reason", "").strip() or not e.get("input_sha256") for e in exclusions)):
        raise ValueError("Checkpoint admissions/exclusions must preserve every registered observation.")
    return [{**e, "decision": "exclude", "exclusion_stage": "checkpoint_qualification",
             "release_stratum": "source_first_checkpoints",
             "directory": str(directory.relative_to(ROOT))} for e in exclusions]


def reviewed_supplement(spec, bindings):
    directory = project_path(spec["directory"])
    verification = read(ROOT / bindings.bind(spec["verification"]))
    if verification.get("status") != "passed" or verification.get("errors") != []:
        raise ValueError("A supplement needs a passing mechanical verification.")
    dependencies = verification.get("dependencies_sha256", {})
    if not dependencies:
        raise ValueError("Supplement verification must bind reconstruction dependencies.")
    if project_path(verification["directory"]) != directory:
        raise ValueError("Mechanical verification belongs to another draft.")
    verifiers = {
        "passive-sessions-v2": "ml/passive_sessions_v2.py",
        "passive-source-checkpoints-v2": "ml/passive_checkpoint_v2.py",
    }
    if verification.get("version") not in verifiers:
        raise ValueError("Use a recognized source-bound supplement verifier.")
    required = {
        str((directory / "recipe.json").relative_to(ROOT)),
        verifiers[verification["version"]],
        *(str(project_path(p).relative_to(ROOT)) for p in spec["source_registries"]),
    }
    for path, sha in dependencies.items():
        bindings.bind(path, sha)
    candidates, file_hashes = [], {}
    for split in ("train", "dev"):
        path = directory / f"{split}-candidates.jsonl"
        name = bindings.bind(path)
        file_hashes[name] = bindings.hashes[name]
        if name not in dependencies or dependencies[name] != file_hashes[name]:
            raise ValueError("Unbound supplement candidate file.")
        for case in rows(path):
            if case["metadata"]["split"] != split:
                raise ValueError("Supplement split differs from file placement.")
            candidates.append(case)
    for case in candidates:
        episode = project_path(directory / case["reproduction"]["episode"])
        if not episode.is_relative_to(directory):
            raise ValueError("Supplement reproduction escapes its draft.")
        required.add(str(episode.relative_to(ROOT)))
    for name in ("train-candidates.jsonl", "dev-candidates.jsonl", "recipe.json"):
        relative = str((directory / name).relative_to(ROOT))
        if verification.get("candidate_hashes", {}).get(name) != dependencies.get(relative):
            raise ValueError("Candidate/recipe hashes disagree with the audited dependencies.")
        required.add(relative)
    review = read(ROOT / bindings.bind(spec["review"]))
    if review.get("candidate_files_sha256") != file_hashes:
        raise ValueError("Independent review must bind both exact candidate files.")
    decisions = {d["case_id"]: d for d in review["cases"]}
    if len(decisions) != len(review["cases"]) or set(decisions) != {c["id"] for c in candidates}:
        raise ValueError("Require exactly one independent review per candidate.")
    expected = {(c["id"], n) for c in candidates for n in range(1 + len(c["acceptableOutputs"]))}
    outcomes = verification.get("outcomes", [])
    if len(outcomes) != len(expected) or {(o["case_id"], o["reference_index"]) for o in outcomes} != expected or any(o["status"] != "passed" for o in outcomes):
        raise ValueError("Mechanical evidence must cover all primary and alternative outcomes.")
    count = len(candidates)
    checkpoint_excluded = []
    parity_count = count
    if verification["version"] == "passive-source-checkpoints-v2":
        checkpoint_excluded = checkpoint_dispositions(verification, directory, bindings, candidates)
        parity_count += len(checkpoint_excluded)
    if (verification.get("candidates") != count
            or len(verification.get("case_ids", [])) != count
            or set(verification["case_ids"]) != {c["id"] for c in candidates}
            or verification.get("actual_serving_message_parity_checks") != parity_count
            or verification.get("replayed_episodes") != len({c["metadata"]["episode_id"] for c in candidates})
            or verification.get("accepted_outcomes_exported") != len(expected)
            or (verification["version"] == "passive-sessions-v2"
                and verification.get("full_ledger_as_of_parity_checks") != count)):
        raise ValueError("Mechanical proof lacks the complete parity/replay/export census.")
    sources = {}
    for path in spec["source_registries"]:
        registry = read(ROOT / bindings.bind(path))
        for source in registry["sources"]:
            if source["id"] in sources:
                raise ValueError("Duplicate registered supplement source.")
            sources[source["id"]] = source
    admitted, excluded = [], checkpoint_excluded
    for c in candidates:
        m, d = c["metadata"], decisions[c["id"]]
        signature = session_hash(canonical_messages(c["input"]))
        if m["input_sha256"] != signature or d["input_sha256"] != signature:
            raise ValueError("Review or case refers to different predictor evidence.")
        if d["decision"] not in {"admit", "exclude"} or not d.get("reason", "").strip():
            raise ValueError("Unresolved or unexplained review decisions cannot be admitted.")
        source = sources[m["source_id"]]
        for key in ("source_group_id", "split", "language"):
            if m[key] != source[key]:
                raise ValueError("Supplement metadata differs from its source registry.")
        if source["source_group_id"] in RESERVED:
            raise ValueError("Reserved test source in a supplement.")
        if m["media_kind"] != source.get("media_kind", "recording"):
            raise ValueError("Media provenance differs from the registered source.")
        document_path = str(project_path(source["document_path"]).relative_to(ROOT))
        required.add(document_path)
        document_sha = source.get("document_sha256") or source.get("file_sha256", {}).get(document_path)
        if not document_sha:
            raise ValueError("Supplement registry needs a physical document hash.")
        bindings.bind(document_path, document_sha)
        if source.get("media_path"):
            media_path = str(project_path(source["media_path"]).relative_to(ROOT))
            required.add(media_path)
            media_sha = (source.get("media_sha256") or source.get("media_download", {}).get("sha256")
                         or source.get("file_sha256", {}).get(media_path))
            if not media_sha:
                raise ValueError("Supplement registry needs a physical media hash.")
            bindings.bind(media_path, media_sha)
        elif source.get("media_kind") != "authored_symbolic_timeline":
            raise ValueError("A recording supplement needs its physical media binding.")
        if d["decision"] == "exclude":
            excluded.append({**d, "directory": spec["directory"],
                             "exclusion_stage": "independent_candidate_review",
                             "release_stratum": spec["stratum"]})
            continue
        m.update(training_eligible=m["split"] == "train",
                 review_status="independent_observable_input_review",
                 review_file=spec["review"], review_reason=d["reason"],
                 release_stratum=spec["stratum"])
        c["reproduction"]["candidate_directory"] = spec["directory"]
        admitted.append(c)
    if not required <= dependencies.keys():
        raise ValueError(f"Mechanical proof omits reconstruction evidence: {sorted(required - dependencies.keys())}")
    asset_cases = copy.deepcopy(candidates)
    for c in asset_cases:
        c["reproduction"]["candidate_directory"] = spec["directory"]
    bind_source_assets(asset_cases, spec["source_registries"], bindings)
    return admitted, excluded


def validate_rows(cases):
    ids, signatures, groups = set(), {}, {}
    for c in cases:
        m = c["metadata"]
        if c["id"] in ids:
            raise ValueError("Duplicate release case ID.")
        ids.add(c["id"])
        if m["split"] not in {"train", "dev"} or m["source_group_id"] in RESERVED:
            raise ValueError("Invalid release split or reserved source.")
        if groups.setdefault(m["source_group_id"], m["split"]) != m["split"]:
            raise ValueError("Source parent crosses train/dev.")
        signature = session_hash(canonical_messages(c["input"]))
        if signature in signatures:
            raise ValueError(f"Duplicate canonical input: {signatures[signature]} / {c['id']}")
        signatures[signature] = c["id"]
        if m["input_sha256"] != signature:
            raise ValueError("Stale predictor hash.")
        normalized = []
        for action in [c["expectedOutput"], *c["acceptableOutputs"]]:
            for fmt in ("json", "rewrite"):
                checked = parse(serialize(c["input"], action, fmt), c["input"], fmt).model_dump()
                if checked != action:
                    raise ValueError("Output formats no longer express the same transaction.")
            normalized.append(checked)
        c["expectedOutput"], c["acceptableOutputs"] = normalized[0], normalized[1:]
    return groups


def screening_view(cases, count=36, seed="passive-b1-screen-v2"):
    dev = sorted((c for c in cases if c["metadata"]["split"] == "dev"),
                 key=lambda c: hashlib.sha256((seed + c["id"]).encode()).hexdigest())
    selected, gaps = {}, []
    for language in ("en", "ko"):
        for action in ("no_action", "correct_text", "split_paragraph", "merge_paragraph"):
            candidates = [c for c in dev if c["metadata"]["language"] == language and c["expectedOutput"]["action"] == action]
            if candidates:
                selected[candidates[0]["id"]] = candidates[0]
            else:
                gaps.append(f"{language}:{action}")
    # Independent source-first checkpoints stay visible instead of being overwhelmed by curriculum rows.
    for c in dev:
        if c["metadata"]["release_stratum"] == "source_first_checkpoints":
            selected[c["id"]] = c
    for group in sorted({c["metadata"]["source_group_id"] for c in dev}):
        candidates = [c for c in dev if c["metadata"]["source_group_id"] == group]
        if not any(c["metadata"]["source_group_id"] == group for c in selected.values()):
            selected[candidates[0]["id"]] = candidates[0]
    for c in dev:
        if len(selected) >= count:
            break
        selected[c["id"]] = c
    if len(selected) > count:
        raise ValueError("Screening coverage exceeds its preregistered request budget.")
    return {
        "version": "passive-b1-screen-v2", "seed": seed, "request_budget": count,
        "selected_case_ids": list(selected), "unfilled_slots": count - len(selected),
        "missing_action_language_cells": gaps, "model_predictions_consulted": False,
        "scope": "Purposive exposed-development screening, not a population prevalence sample or final ranking. All source-first dev checkpoints plus action/language and source-family coverage. Full B2 uses the complete dev set.",
        "gated_actions": ["delete_span", "trim_silence", "remove_silence"],
    }


def export_views(directory, cases, loaded):
    comparisons, exclusions, maxima = [], [], defaultdict(Counter)
    token_mass = {}
    for c in cases:
        ok, counts = eligibility(c["input"], loaded)
        if ok:
            comparisons.append(copy.deepcopy(c))
        else:
            exclusions.append({"case_id": c["id"], "reason": "input_budget_overflow_no_reference_used",
                               "input_token_counts": counts, "retained_in_json_curriculum": True})
    balanced_weights(comparisons)
    (directory / "matched").mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        jsonl(directory / "matched" / f"{split}-cases.jsonl",
              [c for c in comparisons if c["metadata"]["split"] == split])
    for name, tokenizer in loaded.items():
        for variant, fmt in (("json-curriculum", "json"), ("json", "json"), ("rewrite", "rewrite")):
            for split in ("train", "dev"):
                path = directory / "tokenized" / name / f"{variant}-{split}.jsonl.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                selected = [c for c in (cases if variant == "json-curriculum" else comparisons) if c["metadata"]["split"] == split]
                mass = defaultdict(Counter)
                with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
                    for c in selected:
                        actions = [c["expectedOutput"], *c["acceptableOutputs"]]
                        encoded = [encode(tokenizer, c["input"], a, fmt) for a in actions]
                        prefix = encoded[0]["completion_start"]
                        maxima[name][f"{fmt}_input"] = max(maxima[name][f"{fmt}_input"], prefix)
                        maxima[name][f"{fmt}_output"] = max(maxima[name][f"{fmt}_output"], max(len(e["input_ids"])-prefix for e in encoded))
                        payload = {"id": c["id"], "split": split,
                                   "sampling_weight": c["metadata"]["sampling_weight"],
                                   "input_sha256": c["metadata"]["input_sha256"],
                                   "messages": messages(c["input"], fmt),
                                   "references": [serialize(c["input"], a, fmt) for a in actions],
                                   "primary": encoded[0],
                                   "alternative_completion_ids": [e["input_ids"][prefix:] for e in encoded[1:]]}
                        archive.write((json.dumps(payload, ensure_ascii=False, separators=(",", ":"))+"\n").encode())
                        if split == "train":
                            contribution = c["metadata"]["sampling_weight"] * (len(encoded[0]["input_ids"]) - prefix)
                            for category in ("language", "source_group_id", "episode_id", "kind", "release_stratum"):
                                mass[category][c["metadata"][category]] += contribution
                if split == "train":
                    total = sum(mass["language"].values())
                    token_mass[f"{name}:{variant}"] = {
                        "unique_rows": len(selected),
                        "mean_completion_tokens_under_sampler": total,
                        "expected_primary_loss_token_fractions": {
                            category: {key: value / total for key, value in values.items()}
                            for category, values in mass.items()
                        },
                    }
    write(directory / "matched" / "eligibility.json", {
        "policy": "Input-only qualification; both formats use the exact same user message. No recropping or relabeling. JSON rows outside the matched view remain curriculum-only for a format comparison.",
        "input_limit": INPUT_LIMIT, "output_limit_each_format": OUTPUT_LIMIT,
        "qualified": dict(Counter(c["metadata"]["split"] for c in comparisons)),
        "excluded": exclusions, "measured_maxima": dict(maxima),
    })
    write(directory / "training-token-mass.json", {
        "method": "Expected completion-token mass under the frozen row sampler, including EOS. These are counts, not observed model loss/gradients. Actual padding/batch reduction and training updates remain trainer concerns.",
        "views": token_mass,
    })
    return comparisons, exclusions


def freeze(config_path, output):
    config_path, output = project_path(config_path), project_path(output)
    if output.exists():
        raise ValueError("Never overwrite a dataset release.")
    bindings = Bindings()
    config = read(ROOT / bindings.bind(config_path))
    parent = project_path(config["parent_release"])
    cases = bind_parent(parent, bindings)
    excluded = []
    for spec in config["supplements"]:
        accepted, declined = reviewed_supplement(spec, bindings)
        cases.extend(accepted)
        excluded.extend(declined)
    for name in config["provenance_files"]:
        bindings.bind(name)
    for name in ("ml/passive_bundle_v2.py", "ml/passive_formats_v2.py", "ml/passive_training_v2.py"):
        bindings.bind(name)
    groups = validate_rows(cases)
    balanced_weights(cases)
    dependencies = {}
    loaded = tokenizers(dependencies)
    if len(loaded) != 5:
        raise ValueError("All five pinned comparison tokenizers are required.")
    for name, sha in dependencies.items():
        bindings.bind(name, sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        stage = Path(tmp) / "release"
        stage.mkdir()
        for split in ("train", "dev"):
            selected = [c for c in cases if c["metadata"]["split"] == split]
            jsonl(stage / f"{split}-cases.jsonl", selected)
            jsonl(stage / f"{split}-messages.jsonl", [
                {"id": c["id"], "messages": training_messages(c) if split == "train" else messages(c["input"]),
                 "sampling_weight": c["metadata"]["sampling_weight"]} for c in selected])
        jsonl(stage / "exclusions.jsonl", excluded)
        write(stage / "source-groups.json", groups)
        (stage / "prompt-json.txt").write_text(PROMPT + "\n")
        (stage / "prompt-rewrite.txt").write_text(REWRITE_PROMPT + "\n")
        comparisons, format_excluded = export_views(stage, cases, loaded)
        write(stage / "screening-protocol.json", screening_view(cases))
        train = [c for c in cases if c["metadata"]["split"] == "train"]
        schedule = weighted_indices(train, len(train), config["sampling_seed"])
        write(stage / "training-sampling-example.json", {
            "kind": "reproducible_example_not_a_training_run", "seed": config["sampling_seed"],
            "policy": "Sample with replacement using frozen weights. Do not also multiply sampled row losses by the weight.",
            "case_ids": [train[i]["id"] for i in schedule],
        })
        census = {
            "version": "passive-dataset-completion-v2", "status": "frozen_supported_scope",
            "release_id": config["release_id"],
            "contract_version": "passive-nep-v1", "format_version": VERSION,
            "rows": len(cases), "splits": dict(Counter(c["metadata"]["split"] for c in cases)),
            "stratum_splits": dict(Counter(f"{c['metadata']['release_stratum']}:{c['metadata']['split']}" for c in cases)),
            "action_splits": dict(Counter(f"{c['expectedOutput']['action']}:{c['metadata']['split']}" for c in cases)),
            "language_splits": dict(Counter(f"{c['metadata']['language']}:{c['metadata']['split']}" for c in cases)),
            "source_group_splits": dict(Counter(groups.values())),
            "registered_sources": len({c["metadata"]["source_id"] for c in cases}),
            "episodes": len({c["metadata"]["episode_id"] for c in cases}),
            "known_outcomes": sum(1+len(c["acceptableOutputs"]) for c in cases),
            "feedback_rows": sum(bool(c["input"]["recent_feedback"]) for c in cases),
            "selection_rows": sum(bool(c["input"]["selection"]) for c in cases),
            "multi_target_corrections": sum(c["expectedOutput"]["action"] == "correct_text" and len(c["expectedOutput"]["target"]) > 1 for c in cases),
            "matched_format_splits": dict(Counter(c["metadata"]["split"] for c in comparisons)),
            "format_input_exclusions": len(format_excluded),
            "supplement_review_exclusions": sum(e["exclusion_stage"] == "independent_candidate_review" for e in excluded),
            "checkpoint_registered_observations": sum(c["metadata"]["release_stratum"] == "source_first_checkpoints" for c in cases) + sum(e["release_stratum"] == "source_first_checkpoints" for e in excluded),
            "checkpoint_unresolved_exclusions": sum(e["exclusion_stage"] == "checkpoint_qualification" for e in excluded),
            "audio_cut_rows": 0, "test_rows": 0, "training_run": False,
            "full_seven_action_coverage_complete": False,
            "sampling_policy": "Equal language, source group, connected overlapping source opportunities, then dependent states. Sampling with replacement implements weights; this does not make dependent rows independent.",
            "limitations": config["limitations"],
        }
        write(stage / "census.json", census)
        files = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        write(stage / "manifest.json", {
            "version": "passive-dataset-completion-v2", "status": "frozen_supported_scope",
            "release_id": config["release_id"],
            "files_sha256": files, "dependencies_sha256": bindings.hashes,
            "parent_manifest_sha256": digest(parent / "manifest.json"),
            "test_protocol": config["test_protocol"], "coverage_audit": config["coverage_audit"],
            "scope": "Passive text/caption training and exposed development. Audio actions gated. Final test deferred by protocol.",
        })
        bindings.stable()
        if output.exists():
            raise ValueError("Release destination appeared during packaging.")
        stage.rename(output)
    print(json.dumps(census, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.config, args.output)
