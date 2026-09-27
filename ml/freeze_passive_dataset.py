"""Admit independently reviewed passive rows and bind all reconstruction evidence."""

import argparse
import copy
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from passive_contract import PROMPT, canonical_messages
from passive_dataset import ROOT, RESERVED, digest, jsonl, read, write
from passive_training import training_messages, training_row
from replay_session import session_hash
from verify_passive_dataset import rows, tokenizers


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Release evidence must remain inside this project.")
    return path


def file_version(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class Bindings:
    """Hash each physical asset once, then reject any changed file version."""

    def __init__(self):
        self.hashes = {}
        self.versions = {}

    def bind(self, value, expected=None):
        path = project_path(value)
        name = str(path.relative_to(ROOT))
        version = file_version(path)
        if name not in self.hashes:
            actual = digest(path)
            if file_version(path) != version:
                raise ValueError(f"Evidence changed while being hashed: {name}")
            self.hashes[name], self.versions[name] = actual, version
        elif version != self.versions[name]:
            raise ValueError(f"Evidence changed during freeze: {name}")
        if expected is not None and self.hashes[name] != expected:
            raise ValueError(f"Evidence hash differs from its binding: {name}")
        return name

    def stable(self):
        for name, version in self.versions.items():
            if file_version(ROOT / name) != version:
                raise ValueError(f"Evidence changed during freeze: {name}")


def bind_source_assets(cases, registries, bindings):
    if not registries:
        raise ValueError("Freezing requires explicit source_registries.")
    sources = {}

    def links(value):
        for key, path in value.items():
            if key.endswith("_path") and path:
                expected = value.get(key[:-5] + "_sha256")
                if expected is not None:
                    bindings.bind(path, expected)

    for name in registries:
        registry = read(ROOT / bindings.bind(name))
        links(registry)
        for source in registry["sources"]:
            if source["id"] in sources:
                raise ValueError("Duplicate registered source identity.")
            sources[source["id"]] = source
    used, episodes = {}, {}
    for case in cases:
        meta = case["metadata"]
        source = sources.get(meta["source_id"])
        if source is None:
            raise ValueError("An admitted source is absent from source_registries.")
        if source["source_group_id"] in RESERVED or any(
            meta[key] != source[key] for key in ("source_group_id", "split", "language")
        ):
            raise ValueError("Admitted source identity differs from its registry.")
        if meta["media_kind"] != source.get("media_kind", "recording"):
            raise ValueError("Admitted media provenance differs from its registry.")
        directory = project_path(case["reproduction"]["candidate_directory"])
        episode_path = project_path(directory / case["reproduction"]["episode"])
        if not episode_path.is_relative_to(directory):
            raise ValueError("Admitted episode path escapes its candidate draft.")
        if episode_path not in episodes:
            episodes[episode_path] = read(ROOT / bindings.bind(episode_path))
        episode = episodes[episode_path]
        if (
            project_path(episode["source_document"])
            != project_path(source["document_path"])
            or episode["spec"]["source_id"] != source["id"]
            or meta["source_region"] != episode["spec"].get("source_region")
            or meta["kind"] != episode["spec"]["kind"]
        ):
            raise ValueError(
                "Source provenance or sampling region differs from its episode."
            )
        bindings.bind(source["document_path"], episode["source_document_sha256"])
        used[source["id"]] = source
    for source in used.values():
        links(source)
        for key in ("file_sha256", "dependency_sha256"):
            for name, expected in source.get(key, {}).items():
                bindings.bind(name, expected)
        for key, path in source.items():
            if key.endswith("_path") and path:
                bindings.bind(path)
        document = project_path(source["document_path"])
        expected_document = source.get("document_sha256") or source.get(
            "file_sha256", {}
        ).get(str(document.relative_to(ROOT)))
        if expected_document is None:
            raise ValueError("Registered source has no physical document hash.")
        bindings.bind(document, expected_document)
        if source.get("media_path"):
            media = project_path(source["media_path"])
            expected_media = (
                source.get("media_sha256")
                or source.get("media_download", {}).get("sha256")
                or source.get("file_sha256", {}).get(str(media.relative_to(ROOT)))
            )
            if expected_media is None:
                raise ValueError("Registered recording has no physical media hash.")
            bindings.bind(media, expected_media)
        elif source.get("media_kind") != "authored_symbolic_timeline":
            raise ValueError("A recorded source must retain its actual media asset.")
    return sorted(used)


def freeze(config_path, output):
    config_path, output = project_path(config_path), project_path(output)
    if output.exists():
        raise ValueError("A passive release must use a new output directory.")
    bindings = Bindings()
    bindings.bind(config_path)
    for name in ("ml/freeze_passive_dataset.py", "ml/passive_training.py"):
        bindings.bind(name)
    config = read(config_path)
    for name, expected in config.get("provenance_files_sha256", {}).items():
        bindings.bind(name, expected)
    cases, excluded, reviews_seen, ids = [], [], set(), set()
    for draft in config["drafts"]:
        directory = project_path(draft["path"])
        verification_path = ROOT / bindings.bind(draft["verification"])
        verification = read(verification_path)
        if (
            verification["status"] != "passed"
            or not verification.get("dependencies_sha256")
            or verification.get("errors") != []
        ):
            raise ValueError(
                "Use complete source-bound verification, not an early diagnostic audit."
            )
        if Path(verification["directory"]).resolve() != directory.resolve():
            raise ValueError("Verification is bound to another draft.")
        if (
            not {
                "train-candidates.jsonl",
                "dev-candidates.jsonl",
                "recipe.json",
            }
            <= verification["candidate_hashes"].keys()
        ):
            raise ValueError("Verification lacks the complete candidate-file binding.")
        for name, expected in verification["candidate_hashes"].items():
            path = directory / name
            if path.resolve().parent != directory:
                raise ValueError("Verification candidate path escapes its draft.")
            bindings.bind(path, expected)
        for name, expected in verification["dependencies_sha256"].items():
            bindings.bind(name, expected)
        candidates, candidate_files = [], {}
        for split in ("train", "dev"):
            path = directory / f"{split}-candidates.jsonl"
            for case in rows(path):
                if case["id"] in candidate_files:
                    raise ValueError("Duplicate candidate identity.")
                meta = case["metadata"]
                if meta["split"] != split or meta["source_group_id"] in RESERVED:
                    raise ValueError("Candidate split or reserved source is invalid.")
                if meta["input_sha256"] != session_hash(
                    canonical_messages(case["input"])
                ):
                    raise ValueError("Candidate predictor-input hash changed.")
                candidate_files[case["id"]] = path
                candidates.append(case)
        bound_candidate_files = {
            directory / f"{split}-candidates.jsonl": verification["candidate_hashes"][
                f"{split}-candidates.jsonl"
            ]
            for split in ("train", "dev")
        }
        required_dependencies = {
            str((directory / name).relative_to(ROOT))
            for name in (
                "train-candidates.jsonl",
                "dev-candidates.jsonl",
                "recipe.json",
                "exclusions.jsonl",
                *(c["reproduction"]["episode"] for c in candidates),
            )
        }
        if not required_dependencies <= verification["dependencies_sha256"].keys():
            raise ValueError("Verification lacks candidate reconstruction evidence.")
        expected_outcomes = {
            (c["id"], index)
            for c in candidates
            for index in range(1 + len(c["acceptableOutputs"]))
        }
        outcomes = verification.get("outcomes", [])
        if (
            verification.get("candidates") != len(candidates)
            or verification.get("actual_serving_message_parity_checks")
            != len(candidates)
            or verification.get("replayed_episodes")
            != len({c["metadata"]["episode_id"] for c in candidates})
            or verification.get("accepted_outcomes_exported") != len(expected_outcomes)
            or len(outcomes) != len(expected_outcomes)
            or {(o["case_id"], o["reference_index"]) for o in outcomes}
            != expected_outcomes
            or any(o["status"] != "passed" for o in outcomes)
        ):
            raise ValueError(
                "Verification does not cover every candidate and reference."
            )
        decisions = {}
        for name in draft["reviews"]:
            path = ROOT / bindings.bind(name)
            review = read(path)
            if not review.get("candidate_files_sha256"):
                raise ValueError(
                    "Semantic review must bind the candidate files it read."
                )
            reviewed_files = {}
            for candidate_name, expected in review["candidate_files_sha256"].items():
                candidate_path = project_path(candidate_name)
                if candidate_path in reviewed_files:
                    raise ValueError("Duplicated semantic candidate-file binding.")
                reviewed_files[candidate_path] = expected
            if reviewed_files != bound_candidate_files:
                raise ValueError(
                    "Semantic review must bind both exact candidate files."
                )
            reviews_seen.add(name)
            for decision in review["cases"]:
                if decision["case_id"] not in candidate_files:
                    raise ValueError("Semantic review names an unknown candidate.")
                if decision["case_id"] in decisions:
                    raise ValueError("Duplicated semantic review decision.")
                if decision["decision"] not in {"admit", "exclude"}:
                    raise ValueError("Unresolved semantic review cannot be frozen.")
                if (
                    not isinstance(decision.get("reason"), str)
                    or not decision["reason"].strip()
                ):
                    raise ValueError("Every semantic decision needs its actual reason.")
                decisions[decision["case_id"]] = {
                    **decision,
                    "review_file": name,
                    "review_source_ids": review.get("source_ids"),
                }
        if set(decisions) != {c["id"] for c in candidates}:
            raise ValueError(
                "Every candidate needs exactly one bound independent decision."
            )
        for case in candidates:
            decision = decisions[case["id"]]
            meta = case["metadata"]
            if decision["input_sha256"] != session_hash(
                canonical_messages(case["input"])
            ):
                raise ValueError("Semantic review read a different predictor input.")
            if (
                decision.get("source_id", meta["source_id"]) != meta["source_id"]
                or (
                    decision["review_source_ids"] is not None
                    and meta["source_id"] not in decision["review_source_ids"]
                )
                or (
                    decision.get("candidate_file") is not None
                    and project_path(decision["candidate_file"])
                    != candidate_files[case["id"]]
                )
            ):
                raise ValueError(
                    "Semantic review source or candidate-file identity differs."
                )
            if case["id"] in ids:
                raise ValueError("Duplicate cross-draft identity.")
            ids.add(case["id"])
            if decision["decision"] == "exclude":
                excluded.append(
                    {"case_id": case["id"], "draft": draft["path"], **decision}
                )
                continue
            case = copy.deepcopy(case)
            meta = case["metadata"]
            if meta["source_group_id"] in RESERVED or meta["split"] not in {
                "train",
                "dev",
            }:
                raise ValueError("Reserved material cannot enter this release.")
            meta.update(
                review_status="independent_observable_input_review",
                training_eligible=meta["split"] == "train",
                review_file=decision["review_file"],
                review_reason=decision["reason"],
                release_scope="passive_text_caption_curriculum_audio_actions_gated",
            )
            case["reproduction"]["candidate_directory"] = draft["path"]
            cases.append(case)
    if not cases:
        raise ValueError("A frozen curriculum must contain an admitted case.")
    source_ids = bind_source_assets(cases, config.get("source_registries"), bindings)
    groups, texts = {}, {}
    for case in cases:
        meta = case["metadata"]
        group, split = meta["source_group_id"], meta["split"]
        if group in groups and groups[group] != split:
            raise ValueError("Admitted source family crosses train/dev.")
        groups[group] = split
        signature = session_hash(canonical_messages(case["input"]))
        if signature in texts:
            raise ValueError(
                "Exact repeated canonical input must be adjudicated, not silently counted twice."
            )
        texts[signature] = split
    # History controls and multiple steps retain one underlying source opportunity's mass.
    by_split = defaultdict(list)
    for case in cases:
        by_split[case["metadata"]["split"]].append(case)
    for split_cases in by_split.values():
        clusters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for case in split_cases:
            meta = case["metadata"]
            region = json.dumps(
                [meta["source_id"], meta["source_region"]], sort_keys=True
            )
            clusters[meta["language"]][meta["source_group_id"]][region].append(case)
        for language_groups in clusters.values():
            for opportunities in language_groups.values():
                for members in opportunities.values():
                    weight = (
                        1
                        / len(clusters)
                        / len(language_groups)
                        / len(opportunities)
                        / len(members)
                    )
                    for case in members:
                        case["metadata"]["sampling_weight"] = weight
    tokenizer_dependencies = {}
    loaded = tokenizers(tokenizer_dependencies)
    if len(loaded) != 5:
        raise ValueError("The release requires all five pinned tokenizer candidates.")
    for name, expected in tokenizer_dependencies.items():
        bindings.bind(name, expected)
    max_tokens = Counter()
    for case in cases:
        if case["metadata"]["training_eligible"]:
            for name, tokenizer in loaded.items():
                encoded = training_row(tokenizer, case)
                max_tokens[name] = max(max_tokens[name], len(encoded["input_ids"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as tmp:
        staged = Path(tmp) / "release"
        staged.mkdir()
        write_release(
            staged,
            cases,
            excluded,
            by_split,
            groups,
            ids,
            source_ids,
            max_tokens,
            bindings.hashes,
            reviews_seen,
        )
        bindings.stable()
        if output.exists():
            raise ValueError("The release destination appeared during freeze.")
        staged.rename(output)
    print(json.dumps(read(output / "census.json"), ensure_ascii=False, indent=2))


def write_release(
    output,
    cases,
    excluded,
    by_split,
    groups,
    ids,
    source_ids,
    max_tokens,
    dependencies,
    reviews_seen,
):
    for split in ("train", "dev"):
        jsonl(output / f"{split}-cases.jsonl", by_split[split])
    jsonl(
        output / "train-messages.jsonl",
        [
            {
                "id": c["id"],
                "messages": training_messages(c),
                "sampling_weight": c["metadata"]["sampling_weight"],
            }
            for c in by_split["train"]
        ],
    )
    jsonl(output / "exclusions.jsonl", excluded)
    (output / "prompt.txt").write_text(PROMPT + "\n")
    write(output / "source-groups.json", groups)
    census = {
        "status": "frozen_scoped_curriculum",
        "contract_version": "passive-nep-v1",
        "rows": len(cases),
        "splits": dict(Counter(c["metadata"]["split"] for c in cases)),
        "languages": dict(Counter(c["metadata"]["language"] for c in cases)),
        "actions": dict(Counter(c["expectedOutput"]["action"] for c in cases)),
        "kinds": dict(Counter(c["metadata"]["kind"] for c in cases)),
        "source_groups": dict(Counter(c["metadata"]["source_group_id"] for c in cases)),
        "source_group_splits": dict(Counter(groups.values())),
        "media_kinds": dict(Counter(c["metadata"]["media_kind"] for c in cases)),
        "episodes": len({c["metadata"]["episode_id"] for c in cases}),
        "registered_sources": len(source_ids),
        "independent_semantic_exclusions": len(excluded),
        "candidate_rows_reviewed": len(ids),
        "test_rows": 0,
        "audio_cut_rows": 0,
        "positive_rows_at_cursor": sum(
            bool(c["expectedOutput"].get("target"))
            and c["expectedOutput"]["target"][0] == c["input"]["cursor_before"]
            for c in cases
        ),
        "feedback_context_rows": sum(bool(c["input"]["recent_feedback"]) for c in cases),
        "cursor_sampling": "Target-centered synthetic opportunities; not a reference-independent live cursor benchmark.",
        "passive_final_test_protocol_frozen": False,
        "known_acceptable_alternatives": sum(
            len(c["acceptableOutputs"]) for c in cases
        ),
        "training_run": False,
        "runtime_server_chat_template_verified": False,
        "all_admitted_train_rows_tokenized_on_five_candidates": True,
        "max_training_tokens": dict(max_tokens),
        "sampling_weight_sums": {
            split: sum(c["metadata"]["sampling_weight"] for c in split_cases)
            for split, split_cases in by_split.items()
        },
        "history_action_counts": dict(
            Counter(
                f"{c['metadata']['split']}:{c['expectedOutput']['action']}:"
                f"{'with_history' if c['input']['recent_edits'] else 'empty_history'}"
                for c in cases
            )
        ),
        "sampling_policy": "Equal language, then source family, then underlying source opportunity, then dependent rows. Weights accompany messages; a trainer must actually use them.",
        "limitations": [
            "Source-derived display corruptions are synthetic augmentation, not a measured natural ASR error distribution.",
            "Symbolic challenge timelines are not evidence of acoustic edit quality.",
            "A known-reference mismatch is not automatically an incorrect model suggestion; other useful alternatives may need adjudication.",
            "Positive opportunities use target-centered synthetic cursor sampling; arbitrary next-location and when-to-show performance require separately registered cursor states.",
            "The seed's 36 reserved test request slots are inherited planning evidence, not a passive-nep-v1-bound final-test protocol.",
            "The three audio-cut actions remain gated; live editor/server integration and final held-out evaluation are not certified.",
        ],
    }
    write(output / "census.json", census)
    manifest = {
        "version": "passive-curriculum-release-v1",
        "status": census["status"],
        "files_sha256": {
            p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()
        },
        "dependencies_sha256": dependencies,
        "source_ids": source_ids,
        "asset_integrity": "Physical SHA256 once per unique path; unchanged device/inode/size/mtime/ctime required through final publication.",
        "independent_review_files": sorted(reviews_seen),
    }
    write(output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.config, args.output)
