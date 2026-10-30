"""Register untouched natural-source cursors before inspecting and labeling inputs."""

import argparse
import json
from collections import Counter
from pathlib import Path

import prepare_passive_actions_v3 as dataset
from passive_checkpoint_v2 import structural_items
from passive_dataset import RESERVED

VERSION = "actions-v3-natural-source-first-1"
SEED = "actions-v3-2026-10-11-source-first-before-labels"
REGISTRY = dataset.ROOT / "data/manifests/passive-existing-sources-v1.json"
POLICY = {
    "cursors_per_source": 3,
    "cursor": "SHA256 uniform timeline position in each third; first item ending after that time",
    "document": "Untouched registered source document; no correction, paragraph regrouping or retiming",
    "selection": [],
    "history": [],
    "label_order": "Save all cursor registrations, then all native inputs, then inspect and author references",
    "scope": "Previously exposed train/development recordings, not unseen source claims",
}


def register(output):
    output = Path(output).resolve()
    dataset.require(not output.exists(), "Use a fresh immutable source-first attempt.")
    output.mkdir(parents=True)
    registry = json.loads(REGISTRY.read_text())
    sources = registry["sources"]
    splits = dataset.source_splits()
    checkpoints = []
    for source in sources:
        dataset.require(source["split"] in ("train", "dev") and source["source_group_id"] not in RESERVED,
                        "Reserved source bodies cannot be opened.")
        dataset.require(splits[source["source_group_id"]] == source["split"], "Source group split changed.")
        path = dataset.ROOT / source["document_path"]
        dataset.require(dataset.sha256(path) == source["document_sha256"], "Natural source document changed.")
        structure = structural_items(json.loads(path.read_text()))
        for slot in range(POLICY["cursors_per_source"]):
            hashed = dataset.digest([SEED, source["id"], slot])
            draw = (int(hashed[:16], 16) >> 11) / 2**53
            moment = structure[-1]["timeline_end"] * (slot + draw) / POLICY["cursors_per_source"]
            focus = next(item for item in structure if item["timeline_end"] > moment)
            checkpoints.append({"id": f"actions-v3-source-first-{source['id']}-{slot}", "source": source,
                                "slot": slot, "hash_draw": hashed, "timeline_moment_seconds": moment,
                                "focus_uuid": focus["uuid"], "focus_alias": focus["alias"],
                                "selection": [], "history": [], "sequence": 0})
    dataset.write(output / "registration.json", {"version": VERSION, "seed": SEED, "policy": POLICY,
                  "source_registry": str(REGISTRY.relative_to(dataset.ROOT)), "source_registry_sha256": dataset.sha256(REGISTRY),
                  "code_sha256": dataset.sha256(Path(__file__)), "checkpoints": checkpoints,
                  "references_authored": False, "model_predictions_consulted": False})
    tokenizer, identity = dataset.legacy.tokenizer()
    builder = dataset.contract.PassiveContextBuilder(tokenizer)
    for checkpoint in checkpoints:
        source = checkpoint["source"]
        document = json.loads((dataset.ROOT / source["document_path"]).read_text())
        bundle = builder.build(document, checkpoint["focus_uuid"], as_of=0, caption_profile=dataset.PROFILE)
        dataset.write(output / "contexts" / f"{checkpoint['id']}.json", {"checkpoint": checkpoint, "bundle": bundle,
                      "input_sha256": dataset.digest(bundle["input"]), "messages": dataset.contract.canonical_messages(bundle["input"])})
        print(json.dumps({"prepared": checkpoint["id"], "items": len(bundle["input"]["items"])}), flush=True)
    dataset.write(output / "context-manifest.json", {"version": VERSION, "stage": "contexts_before_references",
                  "tokenizer": identity, "checkpoint_count": len(checkpoints),
                  "files_sha256": {str(path.relative_to(output)): dataset.sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}})
    return {"registered": len(checkpoints), "labelled": 0}


def materialize(contexts, labels, output):
    contexts, labels, output = Path(contexts).resolve(), Path(labels).resolve(), Path(output).resolve()
    dataset.require(not output.exists(), "Use a fresh immutable materialization directory.")
    manifest = json.loads((contexts / "context-manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        dataset.require(dataset.sha256(contexts / name) == expected, "Preregistered input changed.")
    judgments = json.loads(labels.read_text())
    dataset.require(judgments["context_manifest_sha256"] == dataset.sha256(contexts / "context-manifest.json"), "Labels bound another context set.")
    by_id = {row["id"]: row for row in judgments["judgments"]}
    prepared = [json.loads(path.read_text()) for path in sorted((contexts / "contexts").glob("*.json"))]
    dataset.require(set(by_id) == {row["checkpoint"]["id"] for row in prepared}, "Every sampled checkpoint needs a disposition.")
    tokenizer, identity = dataset.legacy.tokenizer()
    builder = dataset.contract.PassiveContextBuilder(tokenizer)
    cases, exclusions, proofs = [], [], []
    for saved in prepared:
        checkpoint = saved["checkpoint"]
        source, judgment = checkpoint["source"], by_id[checkpoint["id"]]
        dataset.require(judgment["input_sha256"] == saved["input_sha256"], "Judgment used a different visible input.")
        dataset.require(judgment["status"] in ("edit_required", "preserve", "unknown"), "Unknown judgment status.")
        if judgment["status"] == "unknown":
            dataset.require(not judgment.get("primary"), "Unknown may not fabricate a no_action label.")
            exclusions.append({"id": checkpoint["id"], "split": source["split"], "source_group": source["source_group_id"],
                               "status": "deferred", "judgment_status": "unknown", "reason": judgment["reason"],
                               "input_sha256": saved["input_sha256"], "sampled_context": str((contexts / "contexts" / f"{checkpoint['id']}.json").relative_to(dataset.ROOT))})
            continue
        document = json.loads((dataset.ROOT / source["document_path"]).read_text())
        dataset.require(dataset.sha256(dataset.ROOT / source["document_path"]) == source["document_sha256"], "Original document changed.")
        runtime = {"focus_uuid": checkpoint["focus_uuid"], "selection": [], "sequence": 0, "history": [],
                   "feedback_journal": None, "caption_profile": dataset.PROFILE}
        source_row = {"id": source["id"], "source_group": source["source_group_id"], "split": source["split"],
                      "language": source["language"], "provenance": source["transcript_origin"], "natural_audio": True,
                      "generated_audio": False, "acoustically_verified": False, "media_path": source["media_path"],
                      "media_sha256": source["media_sha256"], "source_terms": source["license"],
                      "project_use_authorization": "data/manifests/project-source-use-2026-09-25.json"}
        row = dataset.case_row(checkpoint["id"], source["split"], source["language"], source_row, document, saved["bundle"],
                               judgment["primary"], judgment.get("acceptable", []), runtime, judgment["reason"],
                               {"old_id": None, "selection_method": "source_first", "cursor_selection_before_judgment": True,
                                "scenario_family": judgment["family"], "stratum": "natural_recorded", "context": saved["bundle"]["context"],
                                "registration": str((contexts / "registration.json").relative_to(dataset.ROOT)),
                                "registration_sha256": dataset.sha256(contexts / "registration.json"),
                                "source_document": source["document_path"], "source_document_sha256": source["document_sha256"],
                                "judgments": str(labels.relative_to(dataset.ROOT)), "judgments_sha256": dataset.sha256(labels),
                                "label_actor": "synthetic", "independent_review": "pending_parent_review"})
        dataset.require(row["judgment"]["status"] == judgment["status"], "Necessity and primary action disagree.")
        proof = dataset.validate_case(row, builder)
        row["execution_sha256"] = dataset.digest(proof)
        cases.append(row)
        proofs.append({"id": row["id"], **proof})
    output.mkdir(parents=True)
    dataset.write_rows(output / "cases.jsonl", cases)
    dataset.write_rows(output / "exclusions.jsonl", exclusions)
    dataset.write_rows(output / "reference-executions.jsonl", proofs)
    dataset.write(output / "manifest.json", {"version": VERSION, "status": "prepared_pending_parent_semantic_review",
                  "sampled_count": len(prepared), "case_count": len(cases), "counts": dict(Counter(row["split"] for row in cases)),
                  "unknown_excluded": len(exclusions), "tokenizer": identity, "code_sha256": dataset.sha256(Path(__file__)),
                  "context_manifest": str((contexts / "context-manifest.json").relative_to(dataset.ROOT)),
                  "context_manifest_sha256": dataset.sha256(contexts / "context-manifest.json"), "labels_sha256": dataset.sha256(labels),
                  "files_sha256": {str(path.relative_to(output)): dataset.sha256(path) for path in sorted(output.rglob("*")) if path.is_file()},
                  "claims": "Natural unmodified source contexts with AI-authored labels, not acoustic certification or natural seven-action coverage."})
    return {"sampled": len(prepared), "admitted": len(cases), "unknown": len(exclusions)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "materialize"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    result = register(args.output) if args.command == "register" else materialize(args.contexts, args.labels, args.output)
    print(json.dumps(result, indent=2))
