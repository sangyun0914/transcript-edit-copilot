"""Freeze the exact admitted 22 TRAIN states for a future unchanged-B002 run.

This stdlib-only command prepares inputs, not candidates or preference pairs.
Large parent case files are not needed: their pinned append/readback evidence is
reused explicitly, while the admitted suffix is checked and copied byte-for-byte.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


VERSION = "actions-v3-dpo-supplement-inputs-1"
DATASET = "artifacts/datasets/nep-passive-actions-v3/"
TRAINING = "artifacts/training/gemma-actions-v3/"
COMPONENT = DATASET + "natural-preservation-supplement-attempt-001/admission-003/"
FILES = {
    "release": DATASET + "release-attempt-003/manifest.json",
    "parent": DATASET + "release-attempt-002/manifest.json",
    "component": COMPONENT + "manifest.json",
    "cases": COMPONENT + "cases.jsonl",
    "append": DATASET + "release-attempt-003/append-integrity.json",
    "splits": DATASET + "release-attempt-003/source-splits.json",
    "verification": DATASET + "release-attempt-003-verification.json",
    "admission": "artifacts/audits/project-completion-2026-10-11/"
    "v3-natural-preservation-supplement-parent-admission-001.json",
    "selection": "artifacts/experiments/passive-actions-v3/dpo-anchor-selection-001.json",
    "preparation": TRAINING + "substantive-preparation-attempt-002-json-full/manifest.json",
    "configuration": TRAINING + "substantive-preparation-attempt-002-json-full/config-json.json",
    "training": TRAINING + "substantive-json-full-attempt-001/train/manifest.json",
    "exposure": TRAINING + "substantive-json-full-attempt-001/train/exposure.json",
    "reload": TRAINING + "substantive-json-full-attempt-001/reload/manifest.json",
    "b003_training": TRAINING + "substantive-json-targeted-attempt-001/train/manifest.json",
    "b003_exposure": TRAINING + "substantive-json-targeted-attempt-001/train/exposure.json",
}
TRUSTED_SHA256 = {
    "release": "f91b3b49856d918d32958b653bc48b7170422f867504bb802995ffd421ca4a3d",
    "verification": "e0a62620f45d9f382f796df5186b3dfe8b5f99540f2ee6c8aa2ee5857cfb96ee",
    "selection": "1199065b8f6275b17cd16d2670e5f1a374d27aef7344c1e7460dce34190d8057",
    "b003_training": "df8087947afe4ae1192c5f84aaba8824ae485fda593a6b0baae7639d22ca1b7a",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bytes_hash(value):
    return hashlib.sha256(value).hexdigest()


def stored_hash(value):
    return bytes_hash(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_evidence(input_root):
    root = Path(input_root).resolve()
    paths = {key: (root / name).resolve() for key, name in FILES.items()}
    require(all(path.is_relative_to(root) and path.is_file() for path in paths.values()),
            "All fixed input paths must exist inside input-root.")
    raw = {key: path.read_bytes() for key, path in paths.items() if key != "cases"}
    hashes = {key: bytes_hash(value) for key, value in raw.items()}
    for key, expected in TRUSTED_SHA256.items():
        require(hashes[key] == expected, "Pinned evidence changed: " + key)
    values = {key: json.loads(value) for key, value in raw.items()}

    def binding(key):
        return {"path": FILES[key], "sha256": hashes[key]}

    def bound(actual, key):
        require(actual == binding(key), "Evidence binding differs: " + key)

    def payload(owner, name, key):
        require(values[owner]["files_sha256"][name] == hashes[key],
                "Bound payload changed: " + key)

    release, parent, component = [values[key] for key in ("release", "parent", "component")]
    admission, receipt = values["admission"], values["verification"]
    for actual, key in (
        (release["parent_release"], "parent"),
        (release["components"]["parent"], "parent"),
        (release["components"]["supplement"], "component"),
        (release["parent_admission"], "admission"),
        (admission["parent_release"], "parent"),
        (admission["component"], "component"),
        (receipt["component_admission"], "admission"),
        (receipt["append_integrity"], "append"),
    ):
        bound(actual, key)
    for owner in ("release", "parent", "component"):
        payload(owner, "source-splits.json", "splits")
    payload("release", "append-integrity.json", "append")
    require(parent["counts"] == {"train": 848, "dev": 322}
            and release["counts"] == {"train": 870, "dev": 322}
            and component["counts"] == {"sampled": 24, "admitted": 22, "excluded": 2}
            and admission["status"] == "admitted_for_train_only_append_release",
            "This preparation only accepts the approved 22-case append.")
    require(receipt["status"] == "passed"
            and receipt["release_manifest_sha256"] == hashes["release"]
            and receipt["parent_release_manifest_sha256"] == hashes["parent"]
            and receipt["cases"] == 1192 and receipt["references"] == 1340
            and receipt["added_cases"] == 22 and receipt["added_references"] == 28
            and receipt["inputs_and_primaries_changed"] == 0
            and receipt["counts_by_split"] == release["counts"]
            and receipt["final_test_bodies_read"] is False
            and "no exact input or source-split collision" in receipt["checks"],
            "Missing frozen append/readback evidence.")
    dev_hash = parent["files_sha256"]["cases/dev.jsonl"]
    require(release["files_sha256"]["cases/dev.jsonl"] == dev_hash
            == receipt["exact_old_bytes"]["cases/dev.jsonl"]
            == admission["unchanged_development_cases"]["sha256"],
            "The DEV body binding changed.")
    appended = values["append"]["cases/train.jsonl"]
    require(appended["parent"] == {
        "path": DATASET + "release-attempt-002/cases/train.jsonl",
        "sha256": parent["files_sha256"]["cases/train.jsonl"],
    } and appended["suffix"] == {
        "path": FILES["cases"], "sha256": component["files_sha256"]["cases.jsonl"],
    } and appended["exact_prefix_and_suffix"] is True and appended["parent_unchanged"] is True
            and receipt["exact_old_bytes"]["cases/train.jsonl"] == appended["parent"]["sha256"],
            "The admitted component is not the verified release003 suffix.")
    require(paths["cases"].stat().st_size == appended["suffix_bytes"], "Admitted suffix size changed.")

    selection, config = values["selection"], values["configuration"]
    require(selection["status"] == "frozen" and selection["selected_anchor"] == "B002_full_json"
            and selection["test_bodies_read"] is False, "Use the original frozen B002 selection.")
    for field, key in (("dataset", "parent"), ("configuration", "configuration"),
                       ("preparation", "preparation"), ("training", "training"), ("reloaded", "reload")):
        bound(selection[field], key)
    for owner, name, key in (("preparation", "config-json.json", "configuration"),
                             ("training", "configuration.json", "configuration"),
                             ("reload", "configuration.json", "configuration"),
                             ("training", "exposure.json", "exposure"),
                             ("b003_training", "exposure.json", "b003_exposure")):
        payload(owner, name, key)
    identity = selection["identity"]
    require(identity["configuration_sha256"] == hashes["configuration"]
            and identity["training_manifest_sha256"] == hashes["training"]
            and identity["reload_manifest_sha256"] == hashes["reload"]
            and identity["adapter_sha256"] == values["training"]["files_sha256"]["adapter/adapter_model.safetensors"]
            and values["preparation"]["files_sha256"]["train-cases.jsonl"] == appended["parent"]["sha256"]
            and config["format"] == "json" and config["generation"]["max_new_tokens"] == 512
            and config["generation"]["do_sample"] is False and config["generation"]["num_beams"] == 1,
            "B002 identity, original train payload or generation policy changed.")
    return paths, raw, hashes, values


def exposure_counts(exposure, unique, draws):
    require(exposure["optimizer_commit_uncertain"] is False
            and exposure["pending_backward_rows_not_committed"] == 0, "Incomplete optimizer exposure evidence.")
    counts = exposure["phases"]["optimizer_committed"]["counts"]["case_id"]
    require(len(counts) == unique and sum(counts.values()) == draws
            and all(type(count) is int and count > 0 for count in counts.values()), "Exposure roster/counts changed.")
    for phase in ("forward", "backward", "optimizer_committed"):
        info = exposure["phases"][phase]
        require(info["rows"] == draws and info["unique_cases"] == unique
                and info["counts"]["case_id"] == counts, "Exposure phase rosters disagree.")
    return counts


def read_roster(path, values):
    admission, component, release = [values[key] for key in ("admission", "component", "release")]
    expected = admission["case_ids"]
    require(len(expected) == len(set(expected)) == 22, "Admission membership must contain exactly 22 unique cases.")
    b002 = exposure_counts(values["exposure"], 848, 848)
    b003 = exposure_counts(values["b003_exposure"], 870, 1530)
    require(Counter(values["configuration"]["training"]["sample_order"]) == b002,
            "B002 configuration and completed exposure disagree.")
    require(set(expected).isdisjoint(b002) and set(b003) == set(b002) | set(expected),
            "Supplement is already in B002 or is not the complete B003 addition.")
    splits = values["splits"]
    require(set(splits.values()) == {"train", "dev"}, "Unexpected source split.")
    recorded = release["coverage"]["recorded_source_case_counts"]
    train_sources = {(row["source_group"], row["source_id"]) for row in recorded if row["split"] == "train"}
    dev_sources = {row["source_id"] for row in recorded if row["split"] == "dev"}
    receipt_rows = values["verification"]["results"]
    verified_ids = {row["id"] for row in receipt_rows}
    require(len(verified_ids) == 1192 and set(b003) <= verified_ids,
            "Verified release membership differs from the exposure rosters.")
    dev_ids = verified_ids - set(b003)
    require(len(dev_ids) == 322 and set(expected).isdisjoint(dev_ids), "Supplement overlaps DEV membership.")
    executed = Counter(row["id"] for row in receipt_rows if row["status"] == "passed")
    rows, inputs, digest, references = [], set(), hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            require(raw.strip() and raw.endswith(b"\n"), "Suffix contains a blank or partial JSONL line.")
            case = json.loads(raw)
            source, lineage = case["source"], case["lineage"]
            require(case["split"] == source["split"] == splits[source["source_group"]] == "train"
                    and (source["source_group"], source["id"]) in train_sources
                    and source["id"] not in dev_sources, "Non-TRAIN or overlapping source in supplement.")
            require(case["language"] == "en" and source["natural_audio"] is True
                    and source["acoustically_verified"] is False and lineage["stratum"] == "natural_recorded"
                    and lineage["parent_id"] in b002 and lineage["format_eligibility"]["json"]["eligible"] is True,
                    "Supplement origin or JSON eligibility changed.")
            status = case["judgment"]["status"]
            require(status in ("preserve", "edit_required")
                    and (status == "preserve") == (case["primary"]["action"] == "no_action"),
                    "Unknown or contradictory judgment cannot enter the roster.")
            input_hash = stored_hash(case["input"])
            require(input_hash not in inputs, "Duplicate stored predictor input in supplement.")
            inputs.add(input_hash)
            reference_count = len(case["accepted_predictions"])
            require(reference_count > 0 and executed[case["id"]] == reference_count,
                    "Admitted case lacks the exact verified reference count.")
            references += reference_count
            rows.append({
                "case_id": case["id"], "source_line": line_number,
                "raw_case_line_sha256": bytes_hash(raw), "stored_input_sha256": input_hash,
                "provenance_sha256": stored_hash({"source": source, "lineage": lineage}),
                "split": case["split"], "language": case["language"], "judgment": status,
                "primary_action": case["primary"]["action"], "accepted_references": reference_count,
                "source_id": source["id"], "source_group": source["source_group"],
                "stratum": lineage["stratum"], "parent_stratum": lineage["parent_stratum"],
                "parent_case_id": lineage["parent_id"], "parent_provenance": source.get("parent_provenance"),
                "b002_optimizer_exposures": b002.get(case["id"], 0),
                "b003_optimizer_exposures": b003.get(case["id"], 0),
            })
    require(digest.hexdigest() == component["files_sha256"]["cases.jsonl"], "Admitted case bytes changed.")
    require(len(rows) == len({row["case_id"] for row in rows}) == 22
            and {row["case_id"] for row in rows} == set(expected), "Admitted membership changed.")
    counts = {key: dict(sorted(Counter(row[key] for row in rows).items())) for key in
              ("split", "language", "judgment", "primary_action", "source_id", "source_group", "stratum", "parent_stratum")}
    require(references == 28 and counts["judgment"] == {"edit_required": 8, "preserve": 14}
            and counts["primary_action"] == component["by_action"]
            and counts["source_group"] == component["by_source_group"]
            and counts["parent_stratum"] == component["by_parent_stratum"], "Approved supplement census changed.")
    counts["language"] = {"en": counts["language"].get("en", 0), "ko": counts["language"].get("ko", 0)}
    counts["accepted_references"] = references
    counts["b003_optimizer_exposures"] = sum(row["b003_optimizer_exposures"] for row in rows)
    return rows, counts


def prepare(input_root, output):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), "Use a fresh output directory; reuse is forbidden.")
    output = output.resolve()
    paths, raw, hashes, values = load_evidence(input_root)
    require(all(not output.is_relative_to(path.parent) and not path.is_relative_to(output) for path in paths.values()),
            "Output must be outside each immutable input directory.")
    roster, counts = read_roster(paths["cases"], values)
    output.mkdir(parents=True)
    shutil.copyfile(paths["cases"], output / "cases.jsonl")
    case_hash = values["component"]["files_sha256"]["cases.jsonl"]
    require(file_hash(output / "cases.jsonl") == case_hash, "Suffix changed while copying.")
    (output / "evidence").mkdir()
    (output / "anchor").mkdir()
    evidence = {}
    for key, content in raw.items():
        destination = "anchor/configuration.json" if key == "configuration" else "evidence/" + key + ".json"
        (output / destination).write_bytes(content)
        evidence[key] = {"path": FILES[key], "sha256": hashes[key], "bytes": len(content), "snapshot": destination}
    evidence["cases"] = {"path": FILES["cases"], "sha256": case_hash,
                         "bytes": (output / "cases.jsonl").stat().st_size, "snapshot": "cases.jsonl"}
    with (output / "case-roster.jsonl").open("x", encoding="utf-8") as stream:
        for row in roster:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    identity = values["selection"]["identity"]
    plan = {
        "version": VERSION, "status": "inputs_frozen_candidates_not_generated",
        "selection": "All and only the admitted release003-minus-release002 TRAIN suffix; no output-based selection.",
        "case_order": [row["case_id"] for row in roster], "cases": 22, "counts": counts,
        "anchor": {"anchor_" + key: identity[key] for key in
                   ("training_manifest_sha256", "reload_manifest_sha256", "configuration_sha256", "adapter_sha256")},
        "original_anchor_identity": identity,
        "anchor_configuration": {"path": "anchor/configuration.json", "sha256": hashes["configuration"]},
        "input_cases": {"path": "cases.jsonl", "sha256": case_hash},
        "case_roster": {"path": "case-roster.jsonl", "sha256": file_hash(output / "case-roster.jsonl")},
        "planned_generation": {"format": "json", "candidates_per_case": 1, "sampling": "deterministic_greedy",
                               "maximum_new_tokens": 512, "repeat_draws": False,
                               "retain_all_raw_outputs_failures_and_ties": True},
        "exposure_scope": {"b002_trained_on_these_states": False, "b003_trained_on_these_states": True,
                           "globally_unseen": False, "new_independent_recordings": False,
                           "basis": "Frozen optimizer-committed case counts, original config and exact append evidence."},
        "verification_scope": {
            "current_checks": "Suffix bytes, all22 IDs/inputs, source splits/recordings, census, exposure and metadata bindings.",
            "reused_evidence": "Pinned release003 independent readback proves exact old prefix, unchanged DEV and no exact input/source collision.",
            "parent_case_files_rehashed": False, "native_assets_or_tokenizer_loaded": False,
            "adapter_and_reload_execution_reverified": False,
            "later_generation_requires": "Original SFT preflight/verify_reload plus a new verified supplementary-input path.",
        },
        "hash_conventions": {"raw_case_line_sha256": "Exact UTF-8 JSONL bytes including newline.",
                             "stored_input_sha256": "SHA256 of compact ensure_ascii=False JSON preserving stored key order; not wire_snapshot hash.",
                             "provenance_sha256": "Same encoding for the object with source then lineage, unchanged from the case."},
        "evidence": evidence, "limitations": values["admission"]["limitations"],
        "generated_candidates": 0, "generated_preference_pairs": 0,
        "preference_policy": "Only independently justified strict preferences containing actual candidates; ties/uncertain stay excluded, no invented rejected or quota.",
        "existing_generator_accepts_this_plan": False,
        "dev_bodies_read": False, "final_test_bodies_read": False, "model_predictions_consulted": False,
    }
    write_json(output / "candidate-input-plan.json", plan)
    shutil.copyfile(__file__, output / "preparation-source.py")
    manifest = {"version": VERSION, "status": plan["status"], "cases": 22, "train_only": True,
                "counts": counts, "anchor": plan["anchor"], "final_test_bodies_read": False,
                "files_sha256": {str(path.relative_to(output)): file_hash(path)
                                 for path in sorted(output.rglob("*")) if path.is_file()}}
    write_json(output / "manifest.json", manifest)
    return {"status": manifest["status"], "cases": 22, "counts": counts,
            "manifest_sha256": file_hash(output / "manifest.json"), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Root preserving the fixed artifact-relative paths.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_root, args.output), ensure_ascii=False, indent=2))
