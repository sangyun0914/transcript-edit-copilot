"""Assemble immutable, individually verified v3 data components into one release."""

import argparse
import copy
import json
import shutil
from collections import Counter
from collections import defaultdict
from pathlib import Path

import prepare_passive_actions_v3 as dataset


def load_component(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        dataset.require(path.is_relative_to(directory) and dataset.sha256(path) == expected, "Component file binding changed.")
    return manifest


def stratum(case):
    lineage, source = case["lineage"], case["source"]
    if source.get("generated_audio"):
        return "constructed_audio"
    if lineage.get("old_release") == "completion-v2":
        if lineage["selection_method"] == "source_first":
            return "legacy_source_first"
        if lineage["scenario_family"] == "authored_feedback_session":
            return "legacy_feedback"
        return "legacy_curriculum"
    if source.get("natural_audio") and (source.get("acoustically_verified") or lineage["selection_method"] == "source_first"):
        return "natural_recorded"
    return "authored_symbolic"


def source_neighbors(cases):
    groups = defaultdict(list)
    for case in cases:
        if not case["source"].get("natural_audio"):
            continue
        items = [item for item in case["input"]["items"] if item["source_start_ms"] is not None]
        if not items:
            continue
        low, high = min(item["source_start_ms"] for item in items), max(item["source_end_ms"] for item in items)
        groups[(case["source"]["source_group"], case["source"]["id"])].append((low, high, case))
    neighbors = []
    for (group, source_id), windows in sorted(groups.items()):
        windows.sort(key=lambda value: (value[0], value[1], value[2]["id"]))
        for index, (left_start, left_end, left) in enumerate(windows):
            for right_start, right_end, right in windows[index + 1:]:
                if right_start >= left_end:
                    break
                overlap = min(left_end, right_end) - max(left_start, right_start)
                fraction = overlap / min(left_end - left_start, right_end - right_start)
                if fraction < 0.8:
                    continue
                dataset.require(left["split"] == right["split"], "Neighboring source windows cross splits.")
                neighbors.append({"left_id": left["id"], "right_id": right["id"], "source_group": group,
                                  "source_id": source_id, "split": left["split"], "overlap_ms": overlap,
                                  "smaller_window_overlap_fraction": fraction,
                                  "interpretation": "Correlated source material, potentially different edited states; not independent recordings."})
    return neighbors


def assemble(output, inherited, authored, audio, natural, source_first=None):
    output = Path(output).resolve()
    dataset.require(not output.exists(), "Use a fresh release directory.")
    directories = {"inherited": Path(inherited).resolve(), "authored": Path(authored).resolve(),
                   "constructed_audio": Path(audio).resolve(), "natural": Path(natural).resolve()}
    if source_first:
        directories["natural_source_first"] = Path(source_first).resolve()
    components = {name: {"path": str(path.relative_to(dataset.ROOT)), "sha256": dataset.sha256(path / "manifest.json"),
                         "manifest": load_component(path)} for name, path in directories.items()}
    cases, migration, exclusions = [], dataset.rows(directories["inherited"] / "migration.jsonl"), []
    for name, path in directories.items():
        if name in ("inherited", "authored"):
            incoming = dataset.rows(path / "cases/train.jsonl") + dataset.rows(path / "cases/dev.jsonl")
        else:
            incoming = dataset.rows(path / "cases.jsonl")
        if name == "inherited":
            incoming = [case for case in incoming if case["lineage"].get("old_id") is not None]
        cases.extend(incoming)
        exclusion = path / "exclusions.jsonl"
        if exclusion.exists():
            rows = dataset.rows(exclusion)
            if name == "inherited":
                rows = [row for row in rows if row.get("old_id")]
            exclusions.extend(rows)
    natural_dispositions = dataset.rows(directories["natural"] / "acoustic-dispositions.jsonl")
    exclusions.extend({"id": "natural-evidence-" + row["qualification_id"], **row} for row in natural_dispositions if row["status"] != "admitted")
    dataset.require(len(migration) == 1154, "All 809/289 plus 42/14 old candidates must have a disposition.")
    dataset.require(Counter(row["old_split"] for row in migration) == {"train": 851, "dev": 303}, "Historical split accounting differs.")
    splits = dataset.source_splits()
    override_path = dataset.FIXTURES / "admission-overrides-revision-003.json"
    overrides = {row["id"]: row for row in json.loads(override_path.read_text())["overrides"]}
    seen, unique, duplicates = {}, [], []
    for original in cases:
        case = copy.deepcopy(original)
        if case["id"] in overrides:
            override = overrides[case["id"]]
            case["rationale"] = override["rationale"]
            case["lineage"].update(contrast_type=override["contrast_type"], admission_override=str(override_path.relative_to(dataset.ROOT)),
                                   admission_override_sha256=dataset.sha256(override_path))
        accepted_actions = {prediction["action"] for prediction in case["accepted_predictions"]}
        dataset.require(
            (accepted_actions == {"no_action"}) if case["judgment"]["status"] == "preserve"
            else "no_action" not in accepted_actions,
            "Mixed necessity is ambiguous and cannot enter supervised admission.",
        )
        case["lineage"]["stratum"] = stratum(case)
        case["source"].setdefault("generated_audio", False)
        dataset.require(splits[case["source"]["source_group"]] == case["split"], "Source split changed during assembly.")
        key = dataset.digest(case["input"])
        if key in seen:
            kept = seen[key]
            dataset.require(case["split"] == kept["split"], "Identical predictor input crosses splits.")
            dataset.require({dataset.packed(p) for p in case["accepted_predictions"]} == {dataset.packed(p) for p in kept["accepted_predictions"]}, "Identical predictor input has conflicting supervision.")
            duplicates.append({"id": case["id"], "kept_id": kept["id"], "input_sha256": key,
                               "reason": "Identical predictor input and accepted outcomes; counted once."})
            for entry in migration:
                if entry.get("new_id") == case["id"]:
                    entry.update(status="revised", new_id=kept["id"], reason=entry["reason"] + " Identical input merged into the retained v3 case.")
            continue
        seen[key] = case
        unique.append(case)
    cases = unique
    dataset.require(len({case["id"] for case in cases}) == len(cases), "Duplicate case IDs.")
    output.mkdir(parents=True)
    for split in ("train", "dev"):
        dataset.write_rows(output / "cases" / f"{split}.jsonl", [case for case in cases if case["split"] == split])
    dataset.write_rows(output / "migration.jsonl", migration)
    dataset.write_rows(output / "exclusions.jsonl", exclusions)
    dataset.write_rows(output / "duplicate-inputs.jsonl", duplicates)
    neighbors = source_neighbors(cases)
    dataset.write_rows(output / "source-neighbors.jsonl", neighbors)
    dataset.write_rows(output / "acoustic-dispositions.jsonl", natural_dispositions)
    dataset.write_rows(output / "runtime-inputs.jsonl", [{"id": case["id"], "split": case["split"],
                      "document_sha256": dataset.session_hash(case["document"]), "hydrate_document_from": "case.document", **case["runtime"]} for case in cases])
    # Reuse immutable replay proofs; the independent verifier rechecks execution/export.
    proofs = {}
    for name in directories:
        if not (directories[name] / "reference-executions.jsonl").exists():
            continue
        for proof in dataset.rows(directories[name] / "reference-executions.jsonl"):
            proofs[proof["id"]] = proof
    tokenizer, tokenizer_identity = dataset.legacy.tokenizer()
    builder = dataset.contract.PassiveContextBuilder(tokenizer)
    for case in cases:
        if case["id"] not in proofs:
            proofs[case["id"]] = {"id": case["id"], **dataset.validate_case(case, builder)}
        bound = {key: value for key, value in proofs[case["id"]].items() if key != "id"}
        dataset.require(dataset.digest(bound) == case["execution_sha256"], "Assembly changed a reference proof.")
    dataset.write_rows(output / "reference-executions.jsonl", [proofs[case["id"]] for case in cases])
    dataset.write(output / "source-splits.json", splits)
    dataset.write(output / "components.json", components)
    dataset.write(output / "tokenizer.json", tokenizer_identity)
    report = dataset.coverage(cases, migration, exclusions)
    report["strata"] = {split: dict(Counter(case["lineage"]["stratum"] for case in cases if case["split"] == split)) for split in ("train", "dev")}
    report["input_duplicates_collapsed"] = len(duplicates)
    report["natural_source_neighbor_pairs_at_80pct_overlap"] = len(neighbors)
    report["authored_scenario_families"] = dict(Counter(case["lineage"]["scenario_family"] for case in cases if case["lineage"]["stratum"] == "authored_symbolic"))
    report["exact_natural_audio_operations_admitted"] = sum(case["source"].get("natural_audio") and case["source"].get("acoustically_verified") for case in cases)
    report["exact_generated_audio_restorations"] = sum(case["source"].get("generated_audio") for case in cases)
    report["fresh_natural_source_first"] = {"sampled": components.get("natural_source_first", {}).get("manifest", {}).get("sampled_count", 0),
                                           "admitted": sum(case["lineage"].get("old_id") is None and case["lineage"]["selection_method"] == "source_first" for case in cases),
                                           "unknown_excluded": components.get("natural_source_first", {}).get("manifest", {}).get("unknown_excluded", 0)}
    report["natural_unverified_cut_previews"] = dict(Counter(case["primary"]["action"] for case in cases
                                                           if case["source"].get("natural_audio") and not case["source"].get("acoustically_verified")
                                                           and case["primary"]["action"] in {"delete_span", "trim_silence", "remove_silence"}))
    dataset.write(output / "coverage.json", report)
    source_paths = [Path(__file__), Path(dataset.__file__), dataset.ROOT / "ml/prepare_audio_actions_v3.py",
                    dataset.ROOT / "ml/prepare_natural_actions_v3.py", dataset.ROOT / "ml/verify_passive_actions_v3.py",
                    dataset.ROOT / "ml/prepare_source_first_actions_v3.py",
                    dataset.ROOT / "ml/project_audio_identity_actions_v3.py",
                    dataset.contract.PROMPT_PATH, *sorted(dataset.FIXTURES.rglob("*"))]
    closure = dataset.legacy.source_closure([p for p in source_paths if p.is_file()])
    for name, expected in closure.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset.ROOT / name, target)
        dataset.require(dataset.sha256(target) == expected, "Saved preparation source changed.")
    manifest = {"version": dataset.VERSION, "status": "immutable_release_pending_independent_validation",
                "contract_version": dataset.contract.VERSION, "counts": report["counts"],
                "case_schema": "Self-contained document/input/runtime; acceptable and accepted_predictions include primary first.",
                "strata": ["legacy_curriculum", "legacy_feedback", "legacy_source_first", "authored_symbolic", "constructed_audio", "natural_recorded"],
                "source_sha256": closure, "components": {name: {key: value for key, value in data.items() if key != "manifest"} for name, data in components.items()},
                "files_sha256": {str(p.relative_to(output)): dataset.sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
                "coverage": report, "final_test_bodies_read": 0, "model_predictions_consulted": False,
                "semantic_review_scope": "Historical labels retain their recorded earlier review; current policy migration is author qualification, not independent re-review. Parent reviewed authored preservation revisions; training_v3 independently reviewed all100 authored inputs and references, leading to three caption alternatives regenerated in revision005. Fresh source-first review is recorded separately. No claim that all1154 historical candidates received a fresh independent semantic review.",
                "provenance_scope": "source/ snapshots bind the assembly and verification implementation at release time. Original generation versions and hashes remain separately bound in each immutable component manifest; current source closure is not represented as the historical generator for every component.",
                "claims": "AI-authored references and controlled challenges. Neither synthetic balance nor exact audio reconstruction measures natural editing prevalence or human preference accuracy."}
    review = dataset.ROOT / "artifacts/audits/project-completion-2026-10-11/v3-label-review-002.json"
    if review.exists():
        manifest["independent_semantic_review"] = {"path": str(review.relative_to(dataset.ROOT)), "sha256": dataset.sha256(review),
                                                  "scope": "All100 authored inputs, three regenerated caption alternatives, all24 fresh source-first inputs and22 admitted outcomes; historical migration policy/exclusion review, not a fresh semantic review of every old row."}
    dataset.write(output / "manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "inherited", "authored", "audio", "natural"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-first", type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble(args.output, args.inherited, args.authored, args.audio, args.natural, args.source_first), ensure_ascii=False, indent=2))
