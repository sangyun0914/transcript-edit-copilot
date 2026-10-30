"""Add reviewed references to a frozen release while retaining its proven inputs."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import prepare_passive_actions_v3 as dataset
import passive_formats_actions_v3 as formats
from verify_release_export import Exporter, verify_export_delta, verify_operation


def clone_and_append(source, target, values):
    before_hash = dataset.sha256(source)
    if sys.platform == "darwin":
        subprocess.run(["/bin/cp", "-c", str(source), str(target)], check=True)
        storage = "apfs_clone_then_append"
    else:
        shutil.copyfile(source, target)
        storage = "independent_copy_then_append"
    dataset.require(not os.path.samefile(source, target), "Appending would mutate the frozen parent inode.")
    with target.open("a") as stream:
        for value in values:
            stream.write(dataset.packed(value) + "\n")
    dataset.require(not os.path.samefile(source, target) and dataset.sha256(source) == before_hash, "Frozen parent changed during append.")
    return {"storage": storage, "original_sha256_before_and_after": before_hash, "distinct_inodes_before_and_after": True}


def add_references(original, additions, revision):
    dataset.require(original["judgment"]["status"] == "edit_required", "An added edit cannot preserve an abstention label.")
    case = copy.deepcopy(original)
    for addition in additions:
        dataset.require(addition["input_sha256"] == dataset.digest(original["input"]), "Reviewed addition names a different predictor input.")
        prediction = addition["prediction"]
        dataset.require(prediction["action"] != "no_action", "Added references cannot change edit necessity.")
        dataset.require(prediction not in case["accepted_predictions"], "Added reference is already accepted.")
        case["accepted_predictions"].append(prediction)
    case["acceptable"] = copy.deepcopy(case["accepted_predictions"])
    case["lineage"]["accepted_reference_revision"] = revision
    case["lineage"]["added_reference_rationales"] = [addition["rationale"] for addition in additions]
    return case


def revise(original, revision_path, baseline_path, output, receipt_path, component=None, component_review=None):
    original, revision_path, baseline_path, output, receipt_path = [Path(path).resolve() for path in
                                                                   (original, revision_path, baseline_path, output, receipt_path)]
    dataset.require(not output.exists() and not receipt_path.exists(), "Use fresh release and receipt paths.")
    manifest = json.loads((original / "manifest.json").read_text())
    parent_hash = dataset.sha256(original / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        path = (original / name).resolve()
        dataset.require(path.is_relative_to(original) and dataset.sha256(path) == expected, "Frozen parent file changed.")
    revision = json.loads(revision_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    dataset.require(revision["parent_manifest_sha256"] == parent_hash, "Reference revision names a different release.")
    dataset.require(baseline["status"] == "passed" and baseline["release_manifest_sha256"] == parent_hash, "Full parent verification is required.")
    for name, expected in baseline["verification_source_sha256"].items():
        dataset.require(dataset.sha256(dataset.ROOT / name) == expected, "Baseline execution implementation changed.")
    review = revision["independent_review"]
    dataset.require(dataset.sha256(dataset.ROOT / review["path"]) == review["sha256"], "Independent review receipt changed.")
    extra_reviews = revision.get("additional_independent_reviews", [])
    for extra_review in extra_reviews:
        dataset.require(dataset.sha256(dataset.ROOT / extra_review["path"]) == extra_review["sha256"], "Additional independent review changed.")
    by_id = {}
    for addition in revision["additions"]:
        by_id.setdefault(addition["id"], []).append(addition)
    dataset.require(bool(by_id), "No reviewed reference additions.")
    revision_binding = {"path": str(revision_path.relative_to(dataset.ROOT)), "sha256": dataset.sha256(revision_path)}
    component_cases, component_manifest, component_binding = [], {}, None
    if component:
        dataset.require(component_review is not None, "Added component review is required.")
        component = Path(component).resolve()
        component_manifest = json.loads((component / "manifest.json").read_text())
        for name, expected in component_manifest["files_sha256"].items():
            path = (component / name).resolve()
            dataset.require(path.is_relative_to(component) and dataset.sha256(path) == expected, "Added component file changed.")
        component_review = Path(component_review).resolve()
        component_review_data = json.loads(component_review.read_text())
        component_hash = dataset.sha256(component / "manifest.json")
        dataset.require(component_review_data["status"] == "passed" and component_review_data["manifest_sha256"] == component_hash
                        and component_review_data["model_predictions_consulted"] is False, "Added component requires pre-output semantic review.")
        component_cases = dataset.rows(component / "cases.jsonl")
        component_binding = {"path": str(component.relative_to(dataset.ROOT)), "manifest_sha256": component_hash,
                             "semantic_review": {"path": str(component_review.relative_to(dataset.ROOT)), "sha256": dataset.sha256(component_review)}}
        dataset.require(all(not case["source"].get("natural_audio") and case["lineage"]["stratum"] == "authored_symbolic" for case in component_cases),
                        "This addition path is for the independently reviewed symbolic history component.")
    tokenizer, identity = dataset.legacy.tokenizer()
    builder = dataset.contract.PassiveContextBuilder(tokenizer)
    output.mkdir(parents=True)
    (output / "cases").mkdir()
    exporter, proofs, results, changed, references, counts = Exporter(), [], [], set(), 0, Counter()
    summaries, seen_ids, seen_inputs, component_proofs = [], set(), set(), []
    max_input_tokens = baseline["max_native_input_tokens"]
    splits = json.loads((original / "source-splits.json").read_text())

    def register(case):
        input_hash = dataset.digest(case["input"])
        dataset.require(case["id"] not in seen_ids and input_hash not in seen_inputs, "Added component duplicates a case or predictor input.")
        dataset.require(case["split"] == case["source"]["split"] == splits[case["source"]["source_group"]], "Added component changes an existing source split.")
        seen_ids.add(case["id"])
        seen_inputs.add(input_hash)
        summaries.append({key: case[key] for key in ("split", "language", "source", "primary", "input", "lineage")})

    try:
        for split in ("train", "dev"):
            with (original / "cases" / f"{split}.jsonl").open() as source, (output / "cases" / f"{split}.jsonl").open("x") as target:
                for line in source:
                    if not line.strip():
                        continue
                    old = json.loads(line)
                    case = old
                    if old["id"] in by_id:
                        case = add_references(old, by_id[old["id"]], revision_binding)
                        bundle = dataset.runtime_bundle(case, builder)
                        dataset.validate_case(case, builder, bundle=bundle)
                        prefix = dataset.contract.token_ids(tokenizer, case["input"])
                        dataset.require(len(prefix) <= 3584, "Native input budget exceeded.")
                        before_export = exporter.export(case["document"], case["input"]["caption_profile"]["export_line_length"])
                        for index, addition in enumerate(by_id[old["id"]], start=len(old["accepted_predictions"])):
                            prediction = addition["prediction"]
                            completion = tokenizer.encode(dataset.packed(prediction), add_special_tokens=False)
                            dataset.require(len(completion) + 1 <= 512 and len(prefix) + len(completion) + 1 <= 4096, "Native completion budget exceeded.")
                            proof = dataset.execution(case["document"], bundle, prediction, case["source"])
                            after = dataset.contract.apply_passive_edit(case["document"], bundle, prediction)
                            operation = verify_operation(case["document"], after, bundle["mapping"], prediction)
                            after_export = exporter.export(after, case["input"]["caption_profile"]["export_line_length"])
                            export = verify_export_delta(before_export, after_export, operation, prediction["action"])
                            proofs.append({"id": case["id"], "reference_index": index, **proof})
                            results.append({"id": case["id"], "reference_index": index, "action": prediction["action"],
                                            "after_sha256": dataset.session_hash(after), "caption_export": export, "status": "passed"})
                        changed.add(case["id"])
                    target.write(dataset.packed(case) + "\n" if case is not old else line)
                    register(case)
                    references += len(case["accepted_predictions"])
                    counts[split] += 1
                for case in component_cases:
                    if case["split"] != split:
                        continue
                    bundle = dataset.runtime_bundle(case, builder)
                    primary_proof = dataset.validate_case(case, builder, bundle=bundle)
                    prefix = dataset.contract.token_ids(tokenizer, case["input"])
                    dataset.require(len(prefix) <= 3584, "Added case native input budget exceeded.")
                    max_input_tokens = max(max_input_tokens, len(prefix))
                    before_export = exporter.export(case["document"], case["input"]["caption_profile"]["export_line_length"])
                    for index, prediction in enumerate(case["accepted_predictions"]):
                        for format_name in formats.FORMATS:
                            formats.encode(tokenizer, case["input"], prediction, format_name)
                        completion = tokenizer.encode(dataset.packed(prediction), add_special_tokens=False)
                        dataset.require(len(completion) + 1 <= 512 and len(prefix) + len(completion) + 1 <= 4096, "Added case completion budget exceeded.")
                        proof = primary_proof if index == 0 else dataset.execution(case["document"], bundle, prediction, case["source"])
                        after = dataset.contract.apply_passive_edit(case["document"], bundle, prediction)
                        operation = verify_operation(case["document"], after, bundle["mapping"], prediction)
                        after_export = exporter.export(after, case["input"]["caption_profile"]["export_line_length"])
                        export = verify_export_delta(before_export, after_export, operation, prediction["action"])
                        results.append({"id": case["id"], "reference_index": index, "action": prediction["action"],
                                        "after_sha256": dataset.session_hash(after), "caption_export": export, "status": "passed"})
                        if index == 0:
                            component_proofs.append({"id": case["id"], **proof})
                    register(case)
                    target.write(dataset.packed(case) + "\n")
                    references += len(case["accepted_predictions"])
                    counts[split] += 1
    finally:
        exporter.close()
    dataset.require(changed == set(by_id), "A reviewed case is absent from the parent release.")
    expected_counts = Counter(manifest["counts"]) + Counter(case["split"] for case in component_cases)
    dataset.require(counts == expected_counts and sum(counts.values()) == baseline["cases"] + len(component_cases), "Revision changed case membership.")
    dataset.require(references == baseline["references"] + len(results), "Revision changed existing accepted references.")
    clone_receipts = {}
    for path in original.rglob("*"):
        relative = path.relative_to(original)
        if not path.is_file() or relative.parts[0] in {"source", "cases"} or relative.as_posix() in {"manifest.json", "coverage.json", "migration.jsonl"}:
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if component_cases and relative.as_posix() in {"reference-executions.jsonl", "runtime-inputs.jsonl"}:
            values = component_proofs if relative.name == "reference-executions.jsonl" else [
                {"id": case["id"], "split": case["split"], "document_sha256": dataset.session_hash(case["document"]),
                 "hydrate_document_from": "case.document", **case["runtime"]} for case in component_cases]
            clone_receipts[relative.as_posix()] = clone_and_append(path, target, values)
        else:
            os.link(path, target)
    migration = dataset.rows(original / "migration.jsonl")
    for row in migration:
        if row.get("new_id") in changed:
            row.update(status="revised", prior_release_status=row["status"], accepted_reference_revision=revision_binding)
    dataset.write_rows(output / "migration.jsonl", migration)
    dataset.write_rows(output / "reference-additions.jsonl", proofs)
    dataset.write(output / "admission-revision.json", revision)
    os.link(baseline_path, output / "baseline-verification.json")
    os.link(dataset.ROOT / review["path"], output / "targeted-semantic-review.json")
    for index, extra_review in enumerate(extra_reviews, start=1):
        os.link(dataset.ROOT / extra_review["path"], output / f"extra-semantic-review-{index:03d}.json")
    if component_binding:
        dataset.write(output / "added-component.json", {**component_binding, "manifest": component_manifest})
        os.link(component_review, output / "added-component-semantic-review.json")
        dataset.write(output / "clone-integrity.json", clone_receipts)
    coverage = json.loads((original / "coverage.json").read_text())
    coverage.update(dataset.coverage(summaries, migration, dataset.rows(original / "exclusions.jsonl")))
    coverage.update(accepted_reference_count=references, added_reviewed_references=len(proofs),
                    added_history_cases=len(component_cases))
    for case in component_cases:
        strata = coverage["strata"][case["split"]]
        strata["authored_symbolic"] = strata.get("authored_symbolic", 0) + 1
        family = case["lineage"]["scenario_family"]
        coverage["authored_scenario_families"][family] = coverage["authored_scenario_families"].get(family, 0) + 1
    dataset.write(output / "coverage.json", coverage)
    entries = [dataset.ROOT / name for name in manifest["source_sha256"]] + [Path(__file__), revision_path]
    entries += [dataset.ROOT / name for name in component_manifest.get("source_sha256", {})]
    closure = dataset.legacy.source_closure(entries)
    for name, expected in closure.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset.ROOT / name, target)
        dataset.require(dataset.sha256(target) == expected, "Revision source changed during freeze.")
    revised = copy.deepcopy(manifest)
    revised.update(status="immutable_release_validated", source_sha256=closure, coverage=coverage, counts=dict(counts),
                   parent_release={"path": str(original.relative_to(dataset.ROOT)), "manifest_sha256": parent_hash},
                   reference_revision=revision_binding, targeted_semantic_review=review,
                   additional_independent_reviews=extra_reviews,
                   validation_basis="Full parent readback/export plus independently executed added references and declared history cases; existing inputs, primary references, runtime and source splits unchanged.",
                   provenance_scope="Current source snapshots bind this reference-only revision. The parent release and immutable component manifests retain their actual earlier generation and verification versions.",
                   files_sha256={str(path.relative_to(output)): dataset.sha256(path) for path in sorted(output.rglob("*")) if path.is_file()})
    if component_binding:
        revised["added_component"] = component_binding
    dataset.write(output / "manifest.json", revised)
    combined = copy.deepcopy(baseline)
    combined.update(version="passive-actions-v3-incremental-reference-readback-1", release_manifest_sha256=dataset.sha256(output / "manifest.json"),
                    cases=sum(counts.values()), references=references, tokenizer=identity, parent_verification_sha256=dataset.sha256(baseline_path),
                    verification_source_sha256=dataset.legacy.source_closure([Path(__file__)]),
                    results=baseline["results"] + results, added_references=len(proofs), added_cases=len(component_cases),
                    inputs_and_primaries_changed=0, max_native_input_tokens=max_input_tokens)
    combined["actions"] = dict(Counter(row["action"] for row in combined["results"]))
    combined["checks"] += ["exact parent-to-revision membership/input/runtime/primary preservation", "added references native budgets and actual replay/export"]
    dataset.write(receipt_path, combined)
    return {"counts": dict(counts), "references": references, "added_references": len(proofs), "added_cases": len(component_cases),
            "manifest_sha256": dataset.sha256(output / "manifest.json"), "verification_sha256": dataset.sha256(receipt_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "revision", "baseline-verification", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--component", type=Path)
    parser.add_argument("--component-review", type=Path)
    args = parser.parse_args()
    print(json.dumps(revise(args.original, args.revision, args.baseline_verification, args.output, args.receipt,
                           args.component, args.component_review), indent=2))
