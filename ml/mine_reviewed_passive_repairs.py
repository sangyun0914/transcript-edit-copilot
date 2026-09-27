"""Reconstruct concrete review findings on unchanged registered source documents."""

import argparse
import json
from pathlib import Path

from passive_contract import canonical_messages
from passive_dataset import ROOT, digest, load_sources, read, write
from replay_session import session_hash
from session_cases import document_records
from verify_passive_dataset import rows


def repair_recipes(config):
    sources = load_sources([ROOT / p for p in config["sources"]])
    specs, skipped, seen = [], [], {}
    for batch in config["drafts"]:
        directory = ROOT / batch["path"]
        candidates = {
            c["id"]: c
            for split in ("train", "dev")
            for c in rows(directory / f"{split}-candidates.jsonl")
        }
        for name in batch["reviews"]:
            review = read(ROOT / name)
            for candidate_path, expected in review["candidate_files_sha256"].items():
                path = ROOT / candidate_path
                if (
                    path.resolve().parent != directory.resolve()
                    or digest(path) != expected
                ):
                    raise ValueError("Review no longer binds its candidate draft.")
            for decision in review["cases"]:
                suggestions = decision.get("suggested_native_fixes", [])
                if decision.get("suggested_native_fix"):
                    suggestions = [decision["suggested_native_fix"], *suggestions]
                if not suggestions or decision.get("duplicate_of_case_id"):
                    continue
                case = candidates[decision["case_id"]]
                if decision["input_sha256"] != session_hash(
                    canonical_messages(case["input"])
                ):
                    raise ValueError("Finding refers to another predictor input.")
                if case["metadata"]["kind"] != "source_derived_restoration":
                    raise ValueError(
                        "Only source-derived display restoration drafts are mined here."
                    )
                source = sources[case["metadata"]["source_id"]]
                records, mapping = document_records(source["document"])
                record_by_id = {r["id"]: r for r in records}
                visible = {i["id"]: i for i in case["input"]["items"]}
                for prediction in suggestions:
                    if prediction["action"] not in {
                        "correct_text",
                        "split_paragraph",
                        "merge_paragraph",
                    }:
                        raise ValueError(
                            "A text review cannot qualify an audio action."
                        )
                    targets = prediction["target"]
                    if any(t not in visible or t not in record_by_id for t in targets):
                        raise ValueError(
                            "Suggested targets differ from visible native source text."
                        )
                    if any(
                        record_by_id[t]["text"] != visible[t]["text"] for t in targets
                    ):
                        skipped.append(
                            {
                                "case_id": case["id"],
                                "reason": "finding_target_still_contains_injected_display_corruption",
                            }
                        )
                        continue
                    identities = [mapping[str(t)]["uuid"] for t in targets]
                    opportunity = (
                        source["id"],
                        prediction["action"],
                        tuple(identities),
                    )
                    if opportunity in seen:
                        if seen[opportunity] != prediction.get("text"):
                            raise ValueError(
                                "Conflicting repairs for one native opportunity."
                            )
                        skipped.append(
                            {
                                "case_id": case["id"],
                                "reason": "repeated_native_opportunity",
                            }
                        )
                        continue
                    seen[opportunity] = prediction.get("text")
                    step = {
                        "action": prediction["action"],
                        "focus_uuid": identities[0],
                        "target_uuids": identities,
                    }
                    if "text" in prediction:
                        step["text"] = prediction["text"]
                    fingerprint = session_hash(
                        [*opportunity[:2], identities, prediction.get("text")]
                    )[:12]
                    specs.append(
                        {
                            "id": f"{source['id']}-native-{fingerprint}",
                            "source_id": source["id"],
                            "kind": "native_source_repair",
                            "source_region": case["metadata"]["source_region"],
                            "rationale": decision["reason"],
                            "corruptions": [],
                            "steps": [step],
                            "finding": {
                                "review": name,
                                "review_sha256": digest(ROOT / name),
                                "candidate_id": case["id"],
                                "input_sha256": decision["input_sha256"],
                                "fresh_input_review_required": True,
                            },
                        }
                    )
    return {
        "version": "reviewed-native-repair-recipe-v1",
        "scope": "Candidate positives on unmodified source text; no automatic stop labels or acoustic claims.",
        "episodes": specs,
        "skipped_findings": skipped,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new native recipe path.")
    result = repair_recipes(read(args.config))
    result["mining_config_sha256"] = digest(args.config)
    write(args.output, result)
    print(
        json.dumps(
            {
                "episodes": len(result["episodes"]),
                "skipped_findings": len(result["skipped_findings"]),
            }
        )
    )
