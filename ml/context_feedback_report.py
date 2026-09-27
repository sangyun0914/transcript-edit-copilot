"""Audit an exposed context/feedback batch and compare raw model and display decisions."""

import argparse
import copy
import json
import statistics
from pathlib import Path

from context_policy import FeedbackLedger
from evaluate import evaluate, read_rows
from grading import load_executions
from iteration import sha
from replay_session import replay, session_hash


def history_rule(snapshot):
    corrections = {}
    pauses = set()
    for edit in snapshot["recent_edits"]:
        if edit["action"] == "correct_text" and edit["before"] and edit["after"]:
            corrections.setdefault(edit["before"], set()).add(edit["after"])
        if edit["action"] == "trim_silence" and edit["after_ms"]:
            pauses.add(edit["after_ms"])
    for item in snapshot["items"]:
        if item["id"] in snapshot["protected_ids"]:
            continue
        values = corrections.get(item["text"], set())
        if len(values) == 1 and "correct_text" in snapshot["allowed_actions"]:
            return {
                "action": "correct_text",
                "target": [item["id"]],
                "text": next(iter(values)),
            }
        if (
            item["audio_class"] == "verified_silence"
            and item["boundary_status"] == "verified"
            and 3000 <= item["duration_ms"] <= 5000
            and len(pauses) == 1
            and "trim_silence" in snapshot["allowed_actions"]
        ):
            return {
                "action": "trim_silence",
                "target": [item["id"]],
                "keep_ms": next(iter(pauses)),
            }
    return {"action": "no_action"}


def report(root, runs, output):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"Dataset file mismatch: {name}")
    cases = read_rows(root / "dev-cases.jsonl")
    all_cases = read_rows(root / "train-cases.jsonl") + cases
    executions = load_executions(root / "executions.jsonl")
    journals = list((root / "sessions").glob("*.jsonl"))
    for path in journals:
        _, summary = replay(read_rows(path))
        if not summary["complete"]:
            raise ValueError(f"Incomplete session: {path}")
    ledgers = {
        c["id"]: FeedbackLedger.restore(
            json.loads((root / c["metadata"]["feedback_journal"]).read_text())
        )
        for c in all_cases
    }
    known = {c["id"]: c for c in cases}
    baselines = {
        "always_no_action": [
            {"case_id": c["id"], "valid": True, "prediction": {"action": "no_action"}}
            for c in cases
        ],
        "consistent_history_rule": [
            {"case_id": c["id"], "valid": True, "prediction": history_rule(c["input"])}
            for c in cases
        ],
    }
    baselines.update(
        {
            name: read_rows(path if path.is_file() else path / "results.jsonl")
            for name, path in runs.items()
        }
    )
    reports = {}
    for name, results in baselines.items():
        if len(results) != len(cases) or {r["case_id"] for r in results} != set(known):
            raise ValueError(
                f"Expected exactly one prediction per development case: {name}"
            )
        raw = evaluate(cases, results, executions)
        gated, suppressed = [], []
        for row in raw["rows"]:
            value = copy.deepcopy(row)
            if row["valid"]:
                case, execution = known[row["case_id"]], executions[row["case_id"]]
                bundle = {
                    "input": case["input"],
                    "mapping": execution["mapping"],
                    "document_hash": session_hash(execution["before"]),
                }
                if ledgers[case["id"]].suppressed(
                    execution["before"],
                    bundle,
                    row["prediction"],
                    case["metadata"]["context"]["as_of"],
                ):
                    value["prediction"] = {"action": "no_action"}
                    suppressed.append(case["id"])
            gated.append(value)
        reports[name] = {
            "model": raw,
            "display": evaluate(cases, gated, executions),
            "suppressed_case_ids": suppressed,
        }

    def spread(values):
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    sweeps = read_rows(root / "context-sweep.jsonl")
    result = {
        "scope": "Exposed AI-reference development; raw model behavior is separate from deterministic display suppression. Timing is symbolic. Ablations use full-evidence references.",
        "evaluation_code_sha256": {
            name: sha(Path(__file__).parent / name)
            for name in (
                "context_feedback_report.py",
                "context_policy.py",
                "evaluate.py",
                "grading.py",
            )
        },
        "dataset_manifest_sha256": sha(root / "manifest.json"),
        "prediction_sha256": {
            name: sha(path if path.is_file() else path / "results.jsonl")
            for name, path in runs.items()
        },
        "verified_files": len(manifest["files"]),
        "replayed_journals": len(journals),
        "restored_feedback_journals": len(ledgers),
        "counts": manifest["counts"],
        "input_items": spread([len(c["input"]["items"]) for c in all_cases]),
        "input_tokens": spread(
            [c["metadata"]["context"]["input_tokens"] for c in all_cases]
        ),
        "visible_paragraphs": spread(
            [c["metadata"]["context"]["visible_paragraphs"] for c in all_cases]
        ),
        "total_document_items": spread(
            [c["metadata"]["context"]["total_items"] for c in all_cases]
        ),
        "context_sweep": [
            {
                "input_budget": budget,
                "history_limit": history,
                "feedback_limit": history,
                "cases": sum(
                    r["input_budget"] == budget and r["history_limit"] == history
                    for r in sweeps
                ),
                "reference_targets_outside_window": sum(
                    r["input_budget"] == budget
                    and r["history_limit"] == history
                    and not r["reference_targets_visible"]
                    for r in sweeps
                ),
                "visible_items": spread(
                    [
                        r["visible_items"]
                        for r in sweeps
                        if r["input_budget"] == budget and r["history_limit"] == history
                    ]
                ),
            }
            for budget in (2048, 3840)
            for history in (0, 3, 8)
        ],
        "baselines": reports,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "items": result["input_items"],
                "tokens": result["input_tokens"],
                "baselines": {
                    n: {
                        "raw": r["model"]["outcome_matches"],
                        "display": r["display"]["outcome_matches"],
                        "valid": r["model"]["valid"],
                        "cases": r["model"]["requests"],
                        "suppressed": len(r["suppressed_case_ids"]),
                    }
                    for n, r in reports.items()
                },
            },
            ensure_ascii=False,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="name=probe directory or predictions JSONL",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(
        args.dataset,
        {name: Path(path) for name, path in (s.split("=", 1) for s in args.run)},
        args.output,
    )
