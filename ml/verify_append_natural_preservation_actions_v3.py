"""Verify exact parent bytes and execute only the newly admitted TRAIN references."""

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path

import append_natural_preservation_actions_v3 as append
import finalize_passive_actions_v3 as finalizer
import prepare_passive_actions_v3 as data
import verify_passive_actions_v3 as execute


def verify(directory, output, component_receipt):
    directory, output, component_receipt = [Path(path).resolve() for path in (directory, output, component_receipt)]
    data.require(not output.exists() and not component_receipt.exists(), "Use fresh immutable verification receipts.")
    manifest = append.verify_files(directory)
    parent = append.verify_files(append.PARENT)
    data.require(manifest["parent_release"] == append.bound(append.PARENT / "manifest.json")
                 and manifest["parent_release"]["sha256"] == append.PARENT_SHA
                 and manifest["parent_admission"] == append.bound(append.ADMISSION)
                 and manifest["parent_admission"]["sha256"] == append.ADMISSION_SHA,
                 "Appended release is not bound to the admitted parent.")
    for name, expected in manifest["source_sha256"].items():
        data.require(data.sha256(data.ROOT / name) == expected
                     and data.sha256(directory / "source" / name) == expected, "Frozen verification/preparation source changed.")
    for name, expected in parent["source_sha256"].items():
        data.require(data.sha256(data.ROOT / name) == expected, "Parent proof implementation changed.")
    admission = json.loads(append.ADMISSION.read_text())
    baseline = append.load_bound(manifest["parent_verification"])
    data.require(manifest["parent_verification"]["sha256"] == append.BASELINE_SHA
                 and baseline["release_manifest_sha256"] == append.PARENT_SHA and baseline["status"] == "passed",
                 "Parent replay proof chain changed.")
    data.require(manifest["components"]["supplement"] == append.bound(append.COMPONENT / "manifest.json")
                 and manifest["components"]["supplement"]["sha256"] == append.COMPONENT_SHA,
                 "The appended component changed.")
    clones = json.loads((directory / "append-integrity.json").read_text())
    expected_suffixes = {
        "cases/train.jsonl": append.COMPONENT / "cases.jsonl",
        "reference-executions.jsonl": append.COMPONENT / "reference-executions.jsonl",
        "runtime-inputs.jsonl": directory / "supplement/runtime-suffix.jsonl",
        "exclusions.jsonl": append.COMPONENT / "exclusions.jsonl",
    }
    data.require(set(clones) == set(expected_suffixes), "Unexpected append target.")
    for name, suffix in expected_suffixes.items():
        source, target, record = append.PARENT / name, directory / name, clones[name]
        source_stat, target_stat = source.stat(), target.stat()
        data.require(record["parent"] == append.bound(source) and record["suffix"] == append.bound(suffix), "Append sources changed.")
        data.require(not os.path.samefile(source, target)
                     and record["parent_inode_before"] == record["parent_inode_after"] == [source_stat.st_dev, source_stat.st_ino]
                     and record["clone_inode_before_append"] == record["clone_inode_after_append"] == [target_stat.st_dev, target_stat.st_ino],
                     "Parent and append inode separation was not preserved.")
        data.require(record["storage"] == "apfs_clone_then_append_no_copy_fallback"
                     and target_stat.st_size == source_stat.st_size + suffix.stat().st_size
                     and append.segment_sha256(target, 0, source_stat.st_size) == parent["files_sha256"][name]
                     and append.segment_sha256(target, source_stat.st_size) == data.sha256(suffix),
                     "The release is not the exact old prefix plus admitted suffix.")
    regenerated = {"coverage.json", "source-neighbors.jsonl", *expected_suffixes}
    for name, expected in parent["files_sha256"].items():
        if name in regenerated or name.startswith("source/"):
            continue
        data.require(data.sha256(directory / name) == expected and not os.path.samefile(append.PARENT / name, directory / name),
                     "An inherited payload changed or shares a mutable parent inode.")
    splits = json.loads((directory / "source-splits.json").read_text())
    rows, refs, judgments = append.summaries([directory / "cases/train.jsonl", directory / "cases/dev.jsonl"], splits)
    data.require(Counter(row["split"] for row in rows) == manifest["counts"] == {"train": 870, "dev": 322}
                 and refs == {"train": 981, "dev": 359} and judgments[("train", "edit_required")] == 700
                 and judgments[("train", "preserve")] == 170, "Current release counts are inconsistent.")
    component_cases = data.rows(append.COMPONENT / "cases.jsonl")
    data.require({row["id"] for row in component_cases} == set(admission["case_ids"])
                 and all(row["split"] == row["source"]["split"] == splits[row["source"]["source_group"]] == "train"
                         for row in component_cases), "The TRAIN suffix contains an unadmitted source or case.")
    runtime = data.rows(directory / "supplement/runtime-suffix.jsonl")
    data.require(runtime == [{"id": case["id"], "split": "train", "document_sha256": data.session_hash(case["document"]),
                              "hydrate_document_from": "case.document", **case["runtime"]} for case in component_cases],
                 "A new runtime record differs from its unchanged admitted case.")
    coverage = json.loads((directory / "coverage.json").read_text())
    calculated = data.coverage(rows, data.rows(directory / "migration.jsonl"), data.rows(directory / "exclusions.jsonl"))
    data.require(all(coverage[key] == value for key, value in calculated.items()), "Current coverage retains stale parent counts.")
    neighbors = finalizer.source_neighbors(rows)
    data.require(data.rows(directory / "source-neighbors.jsonl") == neighbors
                 and coverage["natural_source_neighbor_pairs_at_80pct_overlap"] == len(neighbors),
                 "Source correlation was not recomputed for the appended states.")
    data.require(coverage["primary_actions"] == {split: dict(Counter(row["primary"]["action"] for row in rows if row["split"] == split))
                                               for split in ("train", "dev")}, "Primary action counts changed.")
    print("Exact old prefixes, distinct clone inodes, 1,192 inputs and coverage passed; executing only 22 added cases.", flush=True)
    execute.verify(append.COMPONENT, component_receipt)
    added = json.loads(component_receipt.read_text())
    accepted = append.load_bound(admission["execution_readback"])
    data.require(added["status"] == "passed" and added["cases"] == 22 and added["references"] == 28
                 and added["results"] == accepted["results"], "Fresh component execution differs from the admitted outcomes.")
    result = copy.deepcopy(baseline)
    result.update(version="passive-actions-v3-append-only-independent-readback-1", status="passed",
                  release_manifest_sha256=data.sha256(directory / "manifest.json"), cases=len(rows), references=sum(refs.values()),
                  parent_verification_sha256=data.sha256(append.BASELINE), parent_release_manifest_sha256=append.PARENT_SHA,
                  verification_source_sha256=data.legacy.source_closure([Path(__file__), Path(append.__file__)]),
                  component_verification=append.bound(component_receipt), component_admission=append.bound(append.ADMISSION),
                  results=baseline["results"] + added["results"], added_cases=22, added_references=28,
                  inputs_and_primaries_changed=0, old_results_reused=1312, new_results_reexecuted=28,
                  max_native_input_tokens=max(baseline["max_native_input_tokens"], added["max_native_input_tokens"]),
                  exact_old_bytes={name: parent["files_sha256"][name] for name in (
                      "cases/train.jsonl", "cases/dev.jsonl", "reference-executions.jsonl", "reference-additions.jsonl", "runtime-inputs.jsonl")},
                  append_integrity=append.bound(directory / "append-integrity.json"), counts_by_split=manifest["counts"],
                  references_by_split=dict(refs), source_neighbor_pairs=len(neighbors),
                  final_test_bodies_read=False, model_predictions_consulted=False)
    data.require(len({(row["id"], row["reference_index"]) for row in result["results"]}) == len(result["results"]) == 1340,
                 "Combined execution result identity is duplicated or incomplete.")
    result["actions"] = dict(Counter(row["action"] for row in result["results"]))
    result["checks"] += ["byte-exact parent case/proof/runtime prefixes and unchanged DEV", "distinct APFS clone inodes before/after append",
                         "no exact input or source-split collision", "recomputed source overlap and action coverage",
                         "all22 appended cases/28 refs freshly executed and matched to admitted evidence"]
    data.write(output, result)
    print(json.dumps({"status": "passed", "cases": len(rows), "references": 1340,
                      "release_manifest": append.bound(directory / "manifest.json"), "verification": append.bound(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "output", "component-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(args.directory, args.output, args.component_receipt)
