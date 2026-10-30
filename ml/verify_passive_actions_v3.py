"""Read back a v3 release and verify every reference through actual caption export."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import prepare_passive_actions_v3 as dataset
from replay_session import replay, session_hash
from verify_release_export import Exporter, verify_export_delta, verify_operation


def iter_cases(directory):
    paths = ([directory / "cases.jsonl"] if (directory / "cases.jsonl").exists()
             else [directory / "cases/train.jsonl", directory / "cases/dev.jsonl"])
    for path in paths:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


class IndexedProofs:
    def __init__(self, path):
        self.stream = Path(path).open("rb")
        self.offsets = {}
        try:
            while True:
                offset = self.stream.tell()
                line = self.stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                identity = json.loads(line)["id"]
                dataset.require(identity not in self.offsets, "Duplicate execution proof identity.")
                self.offsets[identity] = offset
        except BaseException:
            self.stream.close()
            raise

    def __getitem__(self, identity):
        self.stream.seek(self.offsets[identity])
        proof = json.loads(self.stream.readline())
        dataset.require(proof["id"] == identity, "Execution proof index changed.")
        return proof

    def close(self):
        self.stream.close()


def verify(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    dataset.require(not output.exists(), "Use a fresh validation receipt.")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        dataset.require(path.is_relative_to(directory) and dataset.sha256(path) == expected, "Frozen release file changed: " + name)
    tokenizer, identity = dataset.legacy.tokenizer()
    proofs = IndexedProofs(directory / "reference-executions.jsonl")
    splits = (json.loads((directory / "source-splits.json").read_text()) if (directory / "source-splits.json").exists()
              else dataset.source_splits())
    exporter, results, seen, maxima = Exporter(), [], {}, Counter()
    try:
        for index, case in enumerate(iter_cases(directory)):
            dataset.require(case["split"] == case["source"]["split"] == splits[case["source"]["source_group"]], "Source split mismatch.")
            dataset.require(case["judgment"]["status"] in ("edit_required", "preserve"), "Unknown in supervised corpus.")
            snapshot = dataset.contract.parse_snapshot(case["input"]).model_dump()
            dataset.require(dataset.contract.canonical_messages(snapshot) == case["messages"], "Observable message changed.")
            signature = dataset.digest(snapshot)
            dataset.require(signature not in seen, "Duplicate predictor input.")
            seen[signature] = case["id"]
            prefix = dataset.contract.token_ids(tokenizer, snapshot)
            dataset.require(len(prefix) <= 3584, "Native input budget exceeded.")
            maxima["input_tokens"] = max(maxima["input_tokens"], len(prefix))
            document, runtime = case["document"], case["runtime"]
            records, mapping = dataset.contract._records(document)
            del records
            bundle = {"input": snapshot, "mapping": {str(item["id"]): mapping[str(item["id"])] for item in snapshot["items"]},
                      "document_hash": session_hash(document), "context": case["lineage"]["context"],
                      "caption_authority": {"version": 1, "protected_uuids": [], "profile": runtime["caption_profile"]}}
            proof = proofs[case["id"]]
            dataset.require(dataset.digest({key: value for key, value in proof.items() if key != "id"}) == case["execution_sha256"], "Reference proof binding changed.")
            if proof["journal"]:
                journal = copy.deepcopy(proof["journal"])
                journal[0]["document"] = document
                restored, summary = replay(journal)
                dataset.require(summary["transactions"] == 3 and session_hash(restored) == proof["after_sha256"], "Actual apply/undo/redo mismatch.")
            dataset.require(case["acceptable"] == case["accepted_predictions"] and case["primary"] == case["acceptable"][0], "Reference set changed.")
            accepted_actions = {prediction["action"] for prediction in case["accepted_predictions"]}
            dataset.require(
                (accepted_actions == {"no_action"}) if case["judgment"]["status"] == "preserve"
                else "no_action" not in accepted_actions,
                "A supervised reference cannot mix edit necessity and preservation.",
            )
            left = exporter.export(document, snapshot["caption_profile"]["export_line_length"])
            for reference_index, reference in enumerate(case["accepted_predictions"]):
                completion = tokenizer.encode(dataset.packed(reference), add_special_tokens=False)
                dataset.require(len(completion) + 1 <= 512 and len(prefix) + len(completion) + 1 <= 4096, "Native completion budget exceeded.")
                after = dataset.contract.apply_passive_edit(document, bundle, reference)
                operation = verify_operation(document, after, bundle["mapping"], reference)
                right = exporter.export(after, snapshot["caption_profile"]["export_line_length"])
                export = verify_export_delta(left, right, operation, reference["action"])
                if reference_index:
                    alternate = dataset.execution(document, bundle, reference, case["source"])
                    dataset.require(alternate["after_sha256"] == session_hash(after), "Alternative replay mismatch.")
                results.append({"id": case["id"], "reference_index": reference_index, "action": reference["action"],
                                "after_sha256": session_hash(after), "caption_export": export, "status": "passed"})
            if (index + 1) % 100 == 0:
                print(f"Verified {index + 1}/{len(proofs.offsets)} current inputs and actual caption exports", flush=True)
        dataset.require(len(seen) == len(proofs.offsets), "Unreferenced execution proof.")
    finally:
        exporter.close()
        proofs.close()
    result = {"version": "passive-actions-v3-independent-execution-readback-2", "release_manifest_sha256": dataset.sha256(directory / "manifest.json"),
              "verification_source_sha256": dataset.legacy.source_closure([Path(__file__)]),
              "status": "passed", "cases": len(seen), "references": len(results), "max_native_input_tokens": maxima["input_tokens"],
              "actions": dict(Counter(row["action"] for row in results)), "tokenizer": identity,
              "checks": ["bound files", "source splits", "no unknown-as-no_action", "unique predictor inputs", "native input/completion budgets", "exact source operations", "primary and alternate replay/undo", "actual Audapolis caption export"],
              "results": results}
    dataset.write(output, result)
    return {key: value for key, value in result.items() if key != "results"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.output), indent=2))
