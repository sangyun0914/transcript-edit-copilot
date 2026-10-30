"""Admit strict train-only preferences and retain ambiguous candidates for review.

A valid prediction outside the accepted reference set is not automatically worse.
Contract failures, missed required edits and equivalent dismissed proposals have
explicit reasons; other semantic differences require a separate AI judgment.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import passive_actions as contract
import passive_formats_actions_v3 as formats
import prepare_passive_actions_v3 as dataset
import train_gemma_actions_v3 as sft
from grading import transcript_media

VERSION = "passive-actions-v3-preferences-1"
require, digest, sha256, rows, write, write_rows = (
    dataset.require,
    dataset.digest,
    dataset.sha256,
    dataset.rows,
    dataset.write,
    dataset.write_rows,
)


def verify_strict_preference(case, candidate, bundle, chosen):
    chosen_after = contract.apply_passive_edit(case["document"], bundle, chosen)
    try:
        action = formats.parse(
            candidate["text"], case["input"], candidate["format"]
        ).model_dump()
        candidate_after = contract.apply_passive_edit(case["document"], bundle, action)
    except (ValueError, KeyError):
        return
    require(
        transcript_media(chosen_after) != transcript_media(candidate_after),
        "Equivalent executed outcomes are ties regardless of formatting or target representation.",
    )


def classify(case, candidate, bundle):
    require(
        case["split"] == candidate["split"] == "train",
        "Preferences require a train input.",
    )
    require(
        case["id"] == candidate["case_id"]
        and candidate["input_sha256"] == digest(contract.wire_snapshot(case["input"])),
        "Candidate and reference belong to different inputs.",
    )
    require(
        case["judgment"]["status"] in ("edit_required", "preserve"),
        "Uncertain labels are not preferences.",
    )
    if candidate["finish_reason"] != "stop" or not candidate["text"].strip():
        return {"status": "excluded", "reason": "incomplete_or_empty_candidate"}
    try:
        action = formats.parse(
            candidate["text"], case["input"], candidate["format"]
        ).model_dump()
    except ValueError as error:
        return {
            "status": "preferred",
            "category": "contract_violation",
            "chosen": case["primary"],
            "reason": f"The complete sampled response fails the executable action contract: {error}",
        }
    require(
        action == candidate["parsed_action"],
        "Saved candidate parsing differs from current validation.",
    )
    before = case["document"]
    try:
        after = contract.apply_passive_edit(before, bundle, action)
    except (ValueError, KeyError) as error:
        return {
            "status": "preferred",
            "category": "execution_violation",
            "chosen": case["primary"],
            "reason": f"The candidate cannot execute against its actual source mapping: {error}",
        }
    accepted = [
        transcript_media(contract.apply_passive_edit(before, bundle, reference))
        for reference in case["accepted_predictions"]
    ]
    if transcript_media(after) in accepted:
        return {
            "status": "tie",
            "reason": "Candidate executes to an accepted text, grouping and media result.",
        }
    if (
        action["action"] == "no_action"
        and case["judgment"]["status"] == "edit_required"
    ):
        return {
            "status": "preferred",
            "category": "missed_required_edit",
            "chosen": case["primary"],
            "reason": "The independently admitted reference requires a useful edit now. The candidate abstains. "
            + case["rationale"],
        }
    journal = case["runtime"].get("feedback_journal")
    if (
        case["judgment"]["status"] == "preserve"
        and journal
        and contract.PassiveFeedbackLedger.restore(journal).suppressed(
            before, bundle, action, case["runtime"]["sequence"]
        )
    ):
        return {
            "status": "preferred",
            "category": "repeats_dismissed_or_undone",
            "chosen": case["primary"],
            "reason": "The exact local proposal repeats a dismissed or undone edit in unchanged relevant context. The display gate suppresses it; explicit abstention follows the policy.",
        }
    return {
        "status": "needs_review",
        "reason": "Valid non-reference output; exact reference mismatch alone does not establish preference.",
        "parsed_action": action,
    }


def prepare(cases_path, candidate_path, output, judgments_path=None):
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh preference release directory.")
    candidate_path = Path(candidate_path)
    _, candidate_info = sft.verified_run(
        candidate_path, "candidates_complete", version="gemma-actions-v3-dpo"
    )
    candidate_sha = sha256(candidate_path / "manifest.json")
    cases = rows(cases_path)
    indexed = {case["id"]: case for case in cases}
    require(len(indexed) == len(cases), "Duplicate case IDs.")
    candidates = rows(candidate_path / "candidates.jsonl")
    require(
        {candidate["case_id"] for candidate in candidates} == set(indexed)
        and len(candidates) == len(indexed),
        "Candidates must cover each admitted train state exactly once.",
    )
    judgments = rows(judgments_path) if judgments_path else []
    decisions = {value["candidate_id"]: value for value in judgments}
    require(
        len(decisions) == len(judgments)
        and set(decisions) <= {value["candidate_id"] for value in candidates},
        "Duplicate or foreign semantic judgments.",
    )
    tokenizer, tokenizer_identity = dataset.legacy.tokenizer()
    builder = contract.PassiveContextBuilder(tokenizer)
    ledger, pairs = [], []
    for candidate in candidates:
        case = indexed[candidate["case_id"]]
        bundle = dataset.runtime_bundle(case, builder)
        require(bundle["input"] == case["input"], "Training runtime window changed.")
        decision = classify(case, candidate, bundle)
        if candidate["candidate_id"] in decisions:
            manual = decisions[candidate["candidate_id"]]
            require(
                decision["status"] == "needs_review"
                and manual["candidate_run_sha256"] == candidate_sha
                and manual["input_sha256"] == candidate["input_sha256"]
                and manual["actor"] == "ai_generated"
                and manual["rationale"].strip()
                and manual["status"] in ("preferred", "tie", "uncertain"),
                "Semantic judgments must bind an unresolved same-input candidate and state their evidence.",
            )
            decision = {
                "status": manual["status"],
                "category": manual.get("category", "ai_semantic_review"),
                "reason": manual["rationale"],
                **(
                    {"chosen": manual["chosen"]}
                    if manual["status"] == "preferred"
                    else {}
                ),
            }
        event = {
            "case_id": case["id"],
            "candidate_id": candidate["candidate_id"],
            "input_sha256": candidate["input_sha256"],
            "source_group": case["source"]["source_group"],
            "language": case["language"],
            "reference_action": case["primary"]["action"],
            **decision,
        }
        ledger.append(event)
        if decision["status"] == "preferred":
            chosen = decision["chosen"]
            verify_strict_preference(case, candidate, bundle, chosen)
            require(
                formats.serialize(case["input"], chosen, candidate["format"])
                != candidate["text"],
                "Identical strings are ties.",
            )
            pairs.append(
                {
                    "pair_id": f"{candidate['candidate_id']}:preference-v3",
                    "case_id": case["id"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_run_sha256": candidate_sha,
                    "input_sha256": candidate["input_sha256"],
                    "chosen": chosen,
                    "rejected_text": candidate["text"],
                    "judgment": {
                        "actor": "ai_generated",
                        "status": "preferred",
                        "category": decision["category"],
                        "rationale": decision["reason"],
                    },
                }
            )
    output.mkdir(parents=True)
    write_rows(output / "pairs.jsonl", pairs)
    write_rows(output / "candidate-decisions.jsonl", ledger)
    write_rows(
        output / "review-queue.jsonl",
        [value for value in ledger if value["status"] == "needs_review"],
    )
    report = {
        "version": VERSION,
        "status": "qualified_preferences_with_explicit_remaining_review",
        "candidate_manifest_sha256": candidate_sha,
        "candidate_info": candidate_info,
        "cases_sha256": sha256(cases_path),
        "tokenizer": tokenizer_identity,
        "manual_judgments_sha256": sha256(judgments_path) if judgments_path else None,
        "candidates": len(candidates),
        "pairs": len(pairs),
        "decision_counts": dict(Counter(value["status"] for value in ledger)),
        "chosen_actions": dict(Counter(value["chosen"]["action"] for value in pairs)),
        "language_pairs": dict(
            Counter(indexed[value["case_id"]]["language"] for value in pairs)
        ),
        "train_only": True,
        "human_preferences": False,
        "final_test_bodies_read": False,
        "files_sha256": {path.name: sha256(path) for path in output.iterdir()},
        "implementation_sha256": sha256(__file__),
    }
    write(output / "manifest.json", report)
    return {
        key: report[key]
        for key in ("candidates", "pairs", "decision_counts", "chosen_actions")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "candidates", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.cases, args.candidates, args.output, args.judgments), indent=2
        )
    )


if __name__ == "__main__":
    main()
