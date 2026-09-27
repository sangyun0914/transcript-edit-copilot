"""Cross-check episode causality and AMI timeline conservation outside row grading."""

import argparse
from pathlib import Path

from dataset_release_inventory import digest, read, rows, write
from release_episodes import committed_history


def verify(directory):
    executions = {r["id"]: r for r in rows(directory / "executions.jsonl")}
    results = []
    for episode in read(directory / "episode-summary.json")["episodes"]:
        journal = rows(directory / episode["session"])
        feedback = read(directory / episode["feedback"])
        transactions = [e for e in journal if e["kind"] == "transaction"]
        state, consumed = journal[0]["document_hash"], 0
        for step in feedback["steps"]:
            if step["before_hash"] != state:
                raise ValueError("Episode step does not follow the previous state.")
            if step["recorded_case"]:
                settings = executions[step["id"]]["reconstruction"]
                before = settings["as_of"]
                past = [e for e in feedback["events"] if e["sequence"] <= before]
                proposals = {
                    k: p
                    for k, p in feedback["proposals"].items()
                    if p["sequence"] <= before
                }
                if (
                    settings["feedback"]["events"] != past
                    or settings["feedback"]["proposals"] != proposals
                ):
                    raise ValueError(
                        "Stored prediction feedback differs from the actual past."
                    )
                latest = {e["proposal_id"]: e for e in past}
                expected_history = [
                    {
                        "sequence": e["sequence"],
                        "edit": committed_history(
                            {"input": proposals[e["proposal_id"]]["snapshot"]},
                            e["applied"],
                        ),
                    }
                    for e in sorted(latest.values(), key=lambda e: e["sequence"])
                    if e["outcome"] in {"accepted", "modified"}
                ]
                if settings["history"] != expected_history:
                    raise ValueError(
                        "Narrated history does not match committed, non-undone actions."
                    )
            needed = (
                (2 if step["undo"] else 1)
                if step["outcome"] in {"accepted", "modified"}
                else 0
            )
            for tx in transactions[consumed : consumed + needed]:
                if tx["before_hash"] != state:
                    raise ValueError(
                        "Journal transaction does not match the episode step."
                    )
                state = tx["after_hash"]
            consumed += needed
            if state != step["after_hash"]:
                raise ValueError(
                    "Accepted or undone outcome does not produce its saved state."
                )
        if consumed != len(transactions) or state != journal[-1]["document_hash"]:
            raise ValueError(
                "Episode has unaccounted transactions or a different final state."
            )
        results.append(
            {
                "id": episode["id"],
                "steps": len(feedback["steps"]),
                "transactions": consumed,
                "state_continuity": "passed",
                "past_only_history": "passed",
            }
        )
    ami = read(directory / "evidence/ami-import.json")
    positive = {a["id"] for a in ami["native_annotations"] if a["end"] > a["start"]}
    represented = [
        identity
        for item in ami["item_annotations"].values()
        for identity in item["annotation_ids"]
    ]
    if set(represented) != positive or len(represented) != len(set(represented)):
        raise ValueError(
            "A positive-duration AMI annotation was omitted or duplicated."
        )
    timeline = sorted(ami["item_annotations"].values(), key=lambda i: i["start"])
    reversed_annotations = [
        a for a in ami["native_annotations"] if a["end"] < a["start"]
    ]
    cases = rows(directory / "train-cases.jsonl") + rows(directory / "dev-cases.jsonl")
    for annotation in reversed_annotations:
        for case in cases:
            meta = case["metadata"]
            start, end = meta["source_context_seconds"]
            if meta["parent_source_id"] == "ami-es2002a" and max(
                start, annotation["end"]
            ) < min(end, annotation["start"]):
                raise ValueError(
                    "A case intersects an AMI annotation with reversed timing."
                )
    if timeline[0]["start"] != 0 or any(
        abs(a["end"] - b["start"]) > 1e-8 for a, b in zip(timeline, timeline[1:])
    ):
        raise ValueError(
            "Mixed audio timeline contains a duplicate interval or missing time."
        )
    return {
        "audit_kind": "episode_causality",
        "status": "passed",
        "episodes": results,
        "ami_positive_annotations_represented_once": len(positive),
        "ami_preserved_duration_seconds": timeline[-1]["end"],
        "zero_duration_annotations_retained": sum(
            a["end"] == a["start"] for a in ami["native_annotations"]
        ),
        "reversed_timing_annotations_retained_without_timed_items": reversed_annotations,
        "reversed_timing_intersects_case_context": False,
        "perceptual_audio_cut_quality": "see_exact_operation_listening_evidence"
        if (directory / "evidence/reviewed-audio.json").exists()
        else "not_verified",
        "human_preference_evidence": False,
        "manifest_sha256": digest(directory / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    write(args.output, result)
    print(result)
