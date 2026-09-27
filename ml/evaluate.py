"""Regrade saved predictions and retain per-case failures for the next generation cycle."""

import argparse
import json
from collections import Counter
from pathlib import Path

from grading import grade, load_executions
from nep_v2 import Envelope, Snapshot, validate_prediction


def evaluate(cases, results, executions):
    cases = {case["id"]: case for case in cases}
    rows = []
    for result in results:
        case = cases[result["case_id"]]
        request = Envelope(
            document_id=case["id"],
            request_id="evaluation",
            revision=0,
            interaction_epoch=0,
        )
        row = {
            "case_id": case["id"],
            "repeat": result.get("repeat", 0),
            "valid": False,
            "reference_known": case.get("expectedOutput") is not None,
            "reference_action": (case.get("expectedOutput") or {}).get("action"),
            "scenario_variant": case["metadata"].get("scenario_variant"),
            "source_group_id": case["metadata"]["source_group_id"],
            "language": case["input"]["language"],
            "history_condition": case["metadata"].get("history_condition"),
        }
        try:
            if case["input"].get("schema_version") in {3, 4}:
                from model_input import (
                    parse_snapshot,
                    validate_prediction as validate_case,
                )

                snapshot = parse_snapshot(case["input"])
            else:
                validate_case = validate_prediction
                snapshot = Snapshot.model_validate(case["input"])
            if not result.get("valid"):
                raise ValueError(
                    result.get("error", "Prediction failed in its original run.")
                )
            prediction = validate_case(
                json.dumps(result["prediction"]),
                snapshot,
                request,
                request,
                "stop",
            )
            row.update(valid=True, prediction=prediction.model_dump())
            row.update(grade(case, row["prediction"], executions.get(case["id"])))
        except (ValueError, KeyError) as error:
            row.update(
                valid=False, error=str(error), exact_match=False, outcome_match=False
            )
        rows.append(row)
    known = [r for r in rows if r["reference_known"]]
    proposed = [
        r for r in rows if r["valid"] and r["prediction"]["action"] != "no_action"
    ]

    def breakdown(field):
        groups = {}
        for row in known:
            key = row[field] or "unspecified"
            group = groups.setdefault(
                key,
                {
                    "cases": 0,
                    "valid": 0,
                    "outcome_matches": 0,
                    "unnecessary_proposals": 0,
                    "missed_edits": 0,
                },
            )
            group["cases"] += 1
            group["valid"] += row["valid"]
            group["outcome_matches"] += row["outcome_match"] is True
            action = row.get("prediction", {}).get("action")
            group["unnecessary_proposals"] += bool(
                row["valid"]
                and row["reference_action"] == "no_action"
                and action != "no_action"
            )
            group["missed_edits"] += bool(
                row["reference_action"] != "no_action" and not row["outcome_match"]
            )
        return groups

    return {
        "scope": "Generated-reference agreement, not human preference precision",
        "requests": len(rows),
        "known_references": len(known),
        "missing_case_ids": sorted(set(cases) - {r["case_id"] for r in rows}),
        "valid": sum(r["valid"] for r in rows),
        "exact_matches": sum(r["exact_match"] is True for r in known),
        "outcome_matches": sum(r["outcome_match"] is True for r in known),
        "proposed": len(proposed),
        "proposed_outcome_matches": sum(r["outcome_match"] is True for r in proposed),
        "proposed_unknown": sum(not r["reference_known"] for r in proposed),
        "failure_actions": dict(
            Counter(r["reference_action"] for r in known if not r["outcome_match"])
        ),
        "by_action": breakdown("reference_action"),
        "by_scenario": breakdown("scenario_variant"),
        "by_source_group": breakdown("source_group_id"),
        "by_language": breakdown("language"),
        "by_history_condition": breakdown("history_condition"),
        "rows": rows,
    }


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--executions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        read_rows(args.cases), read_rows(args.results), load_executions(args.executions)
    )
    with args.output.open("x") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
