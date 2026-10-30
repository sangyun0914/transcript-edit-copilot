"""Append the admitted TRAIN preservation component without rewriting parent rows."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import finalize_passive_actions_v3 as finalizer
import prepare_passive_actions_v3 as data

ROOT = data.ROOT
PARENT = ROOT / "artifacts/datasets/nep-passive-actions-v3/release-attempt-002"
PARENT_SHA = "343214096a7f8d2cd5079436880d609af3b7c4dd3442c013aec117d64ba141c1"
BASELINE = ROOT / "artifacts/datasets/nep-passive-actions-v3/release-attempt-002-verification.json"
BASELINE_SHA = "bd39876f3220461f0880b44badccfb457dac048693f93bd9fd879122ddde1cc4"
SUPPLEMENT = ROOT / "artifacts/datasets/nep-passive-actions-v3/natural-preservation-supplement-attempt-001"
COMPONENT = SUPPLEMENT / "admission-003"
COMPONENT_SHA = "0b8cd3874dac27e5d48d679579cd6992fcb48264ea3c1fd74846d9955d0fa6f7"
ADMISSION = ROOT / "artifacts/audits/project-completion-2026-10-11/v3-natural-preservation-supplement-parent-admission-001.json"
ADMISSION_SHA = "ed7be00da0ec709fffb9d0c38d2576d01f29869251a8eca05cf4c40bd0f19f41"


def bound(path):
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": data.sha256(path)}


def load_bound(binding):
    path = ROOT / binding["path"]
    data.require(path.resolve().is_relative_to(ROOT) and data.sha256(path) == binding["sha256"], "Bound artifact changed.")
    return json.loads(path.read_text())


def verify_files(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        data.require(path.is_relative_to(directory) and data.sha256(path) == expected, "Immutable payload changed: " + name)
    return manifest


def segment_sha256(path, offset=0, size=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining is None or remaining:
            block = stream.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not block:
                data.require(remaining in (None, 0), "Append prefix was truncated.")
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def clone(source, target):
    source, target = Path(source), Path(target)
    data.require(sys.platform == "darwin", "This release requires APFS cloning; no full-copy fallback is permitted.")
    data.require(source.is_file() and not target.exists(), "Clone requires a fresh target and existing source.")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["/bin/cp", "-c", str(source), str(target)], check=True)
    data.require(not os.path.samefile(source, target), "Clone shares the frozen parent inode.")


def clone_append(source, target, suffix):
    before = source.stat()
    expected, suffix_hash = data.sha256(source), data.sha256(suffix)
    data.require(before.st_size == 0 or segment_sha256(source, before.st_size - 1, 1) == hashlib.sha256(b"\n").hexdigest(),
                 "Cannot append JSONL after a partial parent line.")
    clone(source, target)
    target_before = target.stat()
    data.require(not os.path.samefile(source, target), "Appending would mutate the parent inode.")
    with target.open("ab") as output, suffix.open("rb") as incoming:
        while block := incoming.read(1024 * 1024):
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    after, target_after = source.stat(), target.stat()
    data.require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                 == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                 and not os.path.samefile(source, target) and data.sha256(source) == expected,
                 "Frozen parent changed during append.")
    data.require(target_after.st_size == before.st_size + suffix.stat().st_size
                 and segment_sha256(target, 0, before.st_size) == expected
                 and segment_sha256(target, before.st_size) == suffix_hash,
                 "Appended file is not the exact parent prefix plus admitted suffix.")
    return {"storage": "apfs_clone_then_append_no_copy_fallback", "parent": bound(source), "suffix": bound(suffix),
            "parent_bytes": before.st_size, "suffix_bytes": suffix.stat().st_size,
            "parent_inode_before": [before.st_dev, before.st_ino], "parent_inode_after": [after.st_dev, after.st_ino],
            "clone_inode_before_append": [target_before.st_dev, target_before.st_ino],
            "clone_inode_after_append": [target_after.st_dev, target_after.st_ino],
            "exact_prefix_and_suffix": True, "parent_unchanged": True}


def summaries(paths, splits):
    rows, identities, inputs, references, judgments = [], set(), set(), Counter(), Counter()
    for path in paths:
        with path.open() as stream:
            for line in stream:
                case = json.loads(line)
                digest = data.digest(case["input"])
                data.require(case["id"] not in identities and digest not in inputs, "Duplicate case or exact predictor input.")
                data.require(case["split"] == case["source"]["split"] == splits[case["source"]["source_group"]],
                             "Existing source split changed.")
                identities.add(case["id"])
                inputs.add(digest)
                references[case["split"]] += len(case["accepted_predictions"])
                judgments[(case["split"], case["judgment"]["status"])] += 1
                rows.append({key: case[key] for key in ("id", "split", "language", "source", "primary", "input", "lineage")})
    return rows, references, judgments


def assemble(output):
    output = Path(output).resolve()
    data.require(not output.exists(), "Use a fresh immutable release directory.")
    data.require(data.sha256(PARENT / "manifest.json") == PARENT_SHA and data.sha256(BASELINE) == BASELINE_SHA
                 and data.sha256(COMPONENT / "manifest.json") == COMPONENT_SHA and data.sha256(ADMISSION) == ADMISSION_SHA,
                 "The specifically approved release or component changed.")
    parent, component = verify_files(PARENT), verify_files(COMPONENT)
    baseline, admission = json.loads(BASELINE.read_text()), json.loads(ADMISSION.read_text())
    data.require(baseline["status"] == "passed" and baseline["release_manifest_sha256"] == PARENT_SHA
                 and baseline["cases"] == 1170 and baseline["references"] == 1312, "Parent replay evidence is missing.")
    data.require(admission["status"] == "admitted_for_train_only_append_release"
                 and admission["parent_release"] == bound(PARENT / "manifest.json")
                 and admission["component"] == bound(COMPONENT / "manifest.json"), "Parent did not admit this exact append.")
    for name, expected in parent["source_sha256"].items():
        data.require(data.sha256(ROOT / name) == expected, "Parent implementation changed; cached proof reuse is invalid.")
    evidence = load_bound(admission["execution_readback"])
    qualified = load_bound(admission["native_and_state_readback"])
    data.require(evidence["status"] == qualified["status"] == "passed" and evidence["cases"] == 22
                 and evidence["references"] == 28 and evidence["release_manifest_sha256"] == COMPONENT_SHA,
                 "Added cases lack exact executable evidence.")
    for name in ("semantic_parent_review", "independent_semantic_review", "independent_consistency_addendum"):
        load_bound(admission[name])
    cases = data.rows(COMPONENT / "cases.jsonl")
    splits = json.loads((PARENT / "source-splits.json").read_text())
    data.require({case["id"] for case in cases} == set(admission["case_ids"]) and len(cases) == 22,
                 "The admitted candidate set changed.")
    data.require(all(case["split"] == case["source"]["split"] == splits[case["source"]["source_group"]] == "train"
                     and case["language"] == "en" and case["source"].get("natural_audio")
                     and case["lineage"]["stratum"] == "natural_recorded" for case in cases),
                 "This path only appends the approved existing-source English TRAIN states.")
    rows, refs, judgments = summaries([PARENT / "cases/train.jsonl", PARENT / "cases/dev.jsonl", COMPONENT / "cases.jsonl"], splits)
    data.require(Counter(row["split"] for row in rows) == {"train": 870, "dev": 322}
                 and refs == {"train": 981, "dev": 359}
                 and judgments[("train", "edit_required")] == 700 and judgments[("train", "preserve")] == 170,
                 "Release membership, necessity or reference counts differ from the approved append.")
    print("Verified parent/component bindings and 1,192 unique inputs; cloning immutable payloads.", flush=True)
    output.mkdir(parents=True)
    modified = {"cases/train.jsonl", "reference-executions.jsonl", "runtime-inputs.jsonl", "exclusions.jsonl",
                "coverage.json", "source-neighbors.jsonl"}
    for name in parent["files_sha256"]:
        if name in modified or name.startswith("source/"):
            continue
        clone(PARENT / name, output / name)
    for name in ("manifest.json", "coverage.json", "source-neighbors.jsonl"):
        clone(PARENT / name, output / "history/parent-release" / name)
    clone(BASELINE, output / "history/parent-release/verification.json")
    clone(ADMISSION, output / "supplement/parent-admission.json")
    clone(COMPONENT / "manifest.json", output / "supplement/component-manifest.json")
    clone(SUPPLEMENT / "manifest-revision-003.json", output / "supplement/handoff.json")
    for name in ("execution_readback", "native_and_state_readback", "semantic_parent_review", "independent_semantic_review",
                 "independent_consistency_addendum"):
        clone(ROOT / admission[name]["path"], output / "supplement" / (name + ".json"))
    handoff = json.loads((SUPPLEMENT / "manifest-revision-003.json").read_text())
    load_bound(handoff["previous_handoff"])
    clone(ROOT / handoff["previous_handoff"]["path"], output / "supplement/superseded-handoff-001.json")
    clone(SUPPLEMENT / "revision-002-status.json", output / "supplement/superseded-status-002.json")
    for revision in ("001", "002"):
        clone(SUPPLEMENT / f"admission-{revision}/manifest.json", output / "supplement" / f"superseded-admission-{revision}.json")
        clone(SUPPLEMENT / f"admission-{revision}-verification.json", output / "supplement" / f"superseded-verification-{revision}.json")
    for revision in ("001", "002", "003"):
        labels = ROOT / f"ml/fixtures/passive-actions-v3/natural-preservation-supplement-judgments-revision-{revision}.json"
        clone(labels, output / "supplement" / f"judgments-{revision}.json")
    prior_labels = json.loads((output / "supplement/judgments-002.json").read_text())
    load_bound(prior_labels["parent_objection"])
    clone(ROOT / prior_labels["parent_objection"]["path"], output / "supplement/superseded-parent-review-001.json")
    historical_proofs = ROOT / handoff["historical_symmetry_proofs"]["path"]
    data.require(data.sha256(historical_proofs) == handoff["historical_symmetry_proofs"]["sha256"], "Historical reviewer proofs changed.")
    clone(historical_proofs, output / "supplement/historical-symmetric-deletion-proofs.jsonl.gz")
    load_bound(handoff["previous_87_operation_evidence"])
    clone(ROOT / handoff["previous_87_operation_evidence"]["path"], output / "supplement/prior-history-verification.json")
    clone(ROOT / handoff["independent_verifier_source"]["path"], output / "supplement/independent-verifier-source.py")
    data.write_rows(output / "supplement/runtime-suffix.jsonl", ({"id": case["id"], "split": "train",
        "document_sha256": data.session_hash(case["document"]), "hydrate_document_from": "case.document", **case["runtime"]} for case in cases))
    suffixes = {"cases/train.jsonl": COMPONENT / "cases.jsonl", "reference-executions.jsonl": COMPONENT / "reference-executions.jsonl",
                "runtime-inputs.jsonl": output / "supplement/runtime-suffix.jsonl", "exclusions.jsonl": COMPONENT / "exclusions.jsonl"}
    clones = {name: clone_append(PARENT / name, output / name, suffix) for name, suffix in suffixes.items()}
    data.write(output / "append-integrity.json", clones)
    migration = data.rows(PARENT / "migration.jsonl")
    excluded = data.rows(output / "exclusions.jsonl")
    neighbors = finalizer.source_neighbors(rows)
    data.write_rows(output / "source-neighbors.jsonl", neighbors)
    coverage = data.coverage(rows, migration, excluded)
    coverage.update(
        strata={split: dict(Counter(row["lineage"]["stratum"] for row in rows if row["split"] == split)) for split in ("train", "dev")},
        accepted_reference_count=sum(refs.values()), accepted_reference_count_by_split=dict(refs),
        primary_actions={split: dict(Counter(row["primary"]["action"] for row in rows if row["split"] == split)) for split in ("train", "dev")},
        judgment_counts={split: {status: judgments[(split, status)] for status in ("edit_required", "preserve")} for split in ("train", "dev")},
        input_duplicates_collapsed=0, natural_source_neighbor_pairs_at_80pct_overlap=len(neighbors),
        authored_scenario_families=dict(Counter(row["lineage"]["scenario_family"] for row in rows if row["lineage"]["stratum"] == "authored_symbolic")),
        exact_natural_audio_operations_admitted=sum(bool(row["source"].get("natural_audio") and row["source"].get("acoustically_verified")) for row in rows),
        exact_generated_audio_restorations=sum(bool(row["source"].get("generated_audio")) for row in rows),
        natural_unverified_cut_previews=dict(Counter(row["primary"]["action"] for row in rows if row["source"].get("natural_audio")
            and not row["source"].get("acoustically_verified") and row["primary"]["action"] in {"delete_span", "trim_silence", "remove_silence"})),
        fresh_natural_source_first=parent["coverage"]["fresh_natural_source_first"],
        natural_preservation_supplement={"candidates": 24, "train_admitted": 22, "unknown_excluded": 2,
            "accepted_references": 28, "preserve": 14, "edit_required": 8, "format_eligible": component["format_eligible"],
            "preservation_source_recordings": handoff["preservation_source_recordings"],
            "preservation_parent_provenance": handoff["preservation_parent_provenance"],
            "limits": handoff["claims"]},
        historical_coverage=bound(PARENT / "coverage.json"))
    coverage["recorded_source_case_counts"] = [
        {"split": split, "source_group": group, "source_id": recording, "cases": count}
        for (split, group, recording), count in sorted(Counter((row["split"], row["source"]["source_group"], row["source"]["id"])
            for row in rows if row["source"].get("natural_audio")).items())]
    data.write(output / "coverage.json", coverage)
    entries = [ROOT / name for name in parent["source_sha256"]]
    entries += [ROOT / name for name in component["source_sha256"]]
    entries += [Path(__file__), ROOT / "ml/verify_append_natural_preservation_actions_v3.py",
                ROOT / "ml/test_append_natural_preservation_actions_v3.py"]
    entries += [ROOT / binding["path"] for binding in handoff["sources"]]
    entries += [ROOT / f"ml/fixtures/passive-actions-v3/natural-preservation-supplement-judgments-revision-{revision}.json" for revision in ("001", "002", "003")]
    closure = data.legacy.source_closure(entries)
    for name, expected in closure.items():
        clone(ROOT / name, output / "source" / name)
        data.require(data.sha256(output / "source" / name) == expected, "Preparation snapshot changed.")
    manifest = {"version": "passive-actions-v3-natural-preservation-append-1", "status": "immutable_release_pending_independent_validation",
        "contract_version": parent["contract_version"], "counts": coverage["counts"], "case_schema": parent["case_schema"],
        "strata": parent["strata"], "source_sha256": closure, "parent_release": bound(PARENT / "manifest.json"),
        "parent_verification": bound(BASELINE), "components": {"parent": bound(PARENT / "manifest.json"), "supplement": bound(COMPONENT / "manifest.json")},
        "parent_admission": bound(ADMISSION), "supplement_handoff": bound(SUPPLEMENT / "manifest-revision-003.json"),
        "validation_basis": "Exact unchanged parent prefixes and independently verified component suffixes; no replay of the old 1,170 states.",
        "semantic_review_scope": parent["semantic_review_scope"] + " The TRAIN supplement has separate full 24-candidate review, a same-context consistency addendum, 22-case/28-reference execution and parent admission.",
        "provenance_scope": "Current append/verification source snapshot is distinct from immutable historical generation and review closures.",
        "claims": handoff["claims"], "final_test_bodies_read": 0, "model_predictions_consulted": False,
        "coverage": coverage, "files_sha256": {str(p.relative_to(output)): data.sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    data.write(output / "manifest.json", manifest)
    print(json.dumps({"manifest": bound(output / "manifest.json"), "counts": coverage["counts"], "references": sum(refs.values()),
                      "natural_source_neighbor_pairs_at_80pct_overlap": len(neighbors)}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    assemble(parser.parse_args().output)
