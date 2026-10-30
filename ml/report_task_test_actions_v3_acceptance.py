"""Report registered final-test point criteria; never select or promote a model."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import compare_task_test_actions_v3 as comparison
import recompute_actions_v3_results as evidence

TARGETS_SHA256 = "91167faaa8cf7a432252f140cc4c20550a1b846283c85260e3f5e7d8d5a5329a"
require = evidence.require


def conjunction(statuses):
    require(set(statuses) <= {"pass", "fail", "unassessed"}, "Unknown verdict.")
    if "fail" in statuses:
        return "fail"
    return "unassessed" if not statuses or "unassessed" in statuses else "pass"


def verdict(passed, **details):
    return {
        "status": "unassessed" if passed is None else "pass" if passed else "fail",
        **details,
    }


def counts(records):
    return {
        "N": len(records),
        "E": sum(row["judgment"] == "edit_required" for row in records),
        "B": sum(row["judgment"] == "preserve" for row in records),
        "D": sum(row["displayed"] for row in records),
        "U": sum(row["useful"] for row in records),
        "M": sum(row["preserved"] and row["model_abstained"] for row in records),
        "S": sum(row["preserved"] for row in records),
    }


def compare_counts(base, selected, minimum_useful_gain):
    for value in (base, selected):
        require(
            all(type(n) is int and n >= 0 for n in value.values())
            and value["N"] == value["E"] + value["B"]
            and value["U"] <= min(value["D"], value["E"])
            and value["D"] <= value["N"]
            and value["M"] <= value["S"] <= value["B"],
            "Invalid integer metric counts.",
        )
    require(
        all(base[k] == selected[k] for k in ("N", "E", "B")), "Compared cohorts differ."
    )
    left = selected["U"] * base["D"]
    right = base["U"] * selected["D"]
    criteria = {
        "precision": verdict(
            left >= right if base["D"] > 0 and selected["D"] > 0 else None,
            selected_useful_times_base_displayed=left,
            base_useful_times_selected_displayed=right,
            minimum_rate_difference=0,
            denominators={"base": base["D"], "selected": selected["D"]},
        ),
        "model_preservation": verdict(
            selected["M"] >= base["M"] if base["B"] > 0 else None,
            base_successes=base["M"],
            selected_successes=selected["M"],
            shared_denominator=base["B"],
            minimum_rate_difference=0,
        ),
        "system_preservation": verdict(
            selected["S"] >= base["S"] if base["B"] > 0 else None,
            base_successes=base["S"],
            selected_successes=selected["S"],
            shared_denominator=base["B"],
            minimum_rate_difference=0,
        ),
        "useful_count": verdict(
            selected["U"] - base["U"] >= minimum_useful_gain if base["N"] > 0 else None,
            difference=selected["U"] - base["U"],
            minimum=minimum_useful_gain,
            shared_requests=base["N"],
        ),
    }
    return {
        "status": conjunction([r["status"] for r in criteria.values()]),
        "base_counts": base,
        "selected_counts": selected,
        "criteria": criteria,
    }


def cohort_verdict(base, selected, identities, specification):
    groups = {}
    for group in specification["groups"]:
        ids = [
            i for i in identities if group == "overall" or base[i]["language"] == group
        ]
        groups[group] = {
            "case_ids": ids,
            **compare_counts(
                counts([base[i] for i in ids]),
                counts([selected[i] for i in ids]),
                specification["minimum_useful_count_difference_by_group"][group],
            ),
        }
    return {
        "status": conjunction([r["status"] for r in groups.values()]),
        "groups": groups,
    }


def functionality(cases, records, specification):
    cells = {}
    for language in specification["languages"]:
        for action in specification["actions"]:
            opportunities = sorted(
                identity
                for identity, case in cases.items()
                if case["language"] == language
                and action in {p["action"] for p in case["accepted_predictions"]}
                and (action != "no_action" or case["judgment"]["status"] == "preserve")
            )
            successes = [
                identity
                for identity in opportunities
                if records[identity]["predicted_action"] == action
                and records[identity]["outcome_match"]
                and (
                    records[identity]["useful"]
                    or records[identity]["preserved"]
                    and records[identity]["model_abstained"]
                )
            ]
            minimum = specification[
                "minimum_actual_correct_predictions_per_available_cell"
            ]
            cells[f"{action}:{language}"] = verdict(
                len(successes) >= minimum if opportunities else None,
                opportunities=len(opportunities),
                successful_predictions=len(successes),
                minimum=minimum,
                opportunity_case_ids=opportunities,
                successful_case_ids=successes,
            )
    return {
        "status": conjunction([r["status"] for r in cells.values()]),
        "cells": cells,
        "cell_status_counts": dict(Counter(r["status"] for r in cells.values())),
    }


def verify_comparison(path, root, recorded_workspace=None):
    path = evidence.resolve_path(root, path, recorded_workspace)
    manifest = evidence.verify_manifest(path)
    require(
        manifest.get("version") == "task-test-actions-v3-comparison-1"
        and manifest["scope"] == "final_test"
        and manifest["status"] == "compared"
        and "comparison.json" in manifest["files_sha256"],
        "Require a complete final-test comparison manifest.",
    )
    saved = evidence.read(path.parent / "comparison.json")
    paths = [
        evidence.bound(root, row["grade_manifest"], recorded_workspace)
        for row in saved["runs"].values()
    ]
    canonical_key = saved.get("cohort_keys", {}).get("canonical", "canonical_full_five_roles")
    sampling = next(iter(saved[canonical_key]["pairs"].values()))[
        "source_resampling"
    ]
    rebuilt, arms = comparison.build_comparison(
        paths,
        root=root,
        recorded_workspace=recorded_workspace,
        resamples=sampling["resamples"],
        seed=sampling["seed"],
    )
    for role in arms:
        # Relocation changes resolved paths, not the original receipt's identity.
        rebuilt["runs"][role]["grade_manifest"] = saved["runs"][role]["grade_manifest"]
    require(
        saved == rebuilt
        and manifest["implementation_sha256"] == rebuilt["implementation_sha256"],
        "Saved final-test comparison differs from fully verified evidence.",
    )
    return saved, arms


def natural_c0(dispositions, roster, cases):
    slots = {row["slot_id"]: row for row in roster}
    require(len(slots) == len(roster), "Repeated registered slot.")
    result = []
    for row in dispositions:
        slot = slots[row["id"]]
        require(
            all(
                row.get(k) == slot.get(k)
                for k in ("stratum", "checkpoint", "language", "window_id")
            ),
            "Disposition differs from its registered natural checkpoint identity.",
        )
        stratum, checkpoint = row["stratum"], row.get("checkpoint")
        require(
            stratum != "natural_source_first" or checkpoint == 0,
            "Natural C0 stratum requires checkpoint zero.",
        )
        require(
            stratum != "natural_source_simulated_session" or checkpoint in (1, 2),
            "Natural simulated sessions require checkpoint one or two.",
        )
        if row["id"] in cases:
            case = cases[row["id"]]
            require(
                case["lineage"]["stratum"] == stratum
                and case["language"] == row["language"],
                "Admitted case differs from its disposition provenance.",
            )
            if stratum == "natural_source_first" and checkpoint == 0:
                result.append(row["id"])
    return sorted(result)


def report(
    targets_path,
    comparison_path,
    output,
    *,
    root=comparison.ROOT,
    recorded_workspace=None,
):
    output = Path(output)
    require(not output.exists(), "Use a fresh acceptance report.")

    def bound(value):
        return evidence.bound(root, value, recorded_workspace)

    targets_path = evidence.resolve_path(root, targets_path, recorded_workspace)
    require(
        evidence.sha256(targets_path) == TARGETS_SHA256,
        "Registered targets001 receipt changed.",
    )
    targets = evidence.read(targets_path)
    bound(targets["original_policy"])
    saved, arms = verify_comparison(comparison_path, root, recorded_workspace)
    freeze = evidence.read(bound(saved["freeze"]))
    selection = evidence.read(bound(freeze["selection_receipt"]))
    selected_role = selection.get("selected_model")
    require(
        selection.get("status") == "frozen"
        and selection.get("selection_scope") == "development"
        and selection.get("test_bodies_read") is False
        and selected_role in {"sft", "sft_dpo"}
        and selected_role == freeze["selected_model"],
        "Require an explicit immutable pre-test development selection.",
    )
    metrics = evidence.read(bound(freeze["policies"]["metrics"]))
    thresholds = metrics["decision_thresholds"]
    require(
        metrics["status"] == "frozen"
        and metrics["scope"] == "final_test"
        and bound(thresholds["registered_targets"]) == targets_path
        and thresholds["numerical_targets"] == targets["numerical_targets"]
        and thresholds["exact_arithmetic"] == targets["exact_arithmetic"]
        and datetime.fromisoformat(targets["created_at"])
        <= datetime.fromisoformat(freeze["frozen_at"]),
        "Final freeze does not bind the unchanged pre-body numerical targets.",
    )
    implementation = {
        **saved["implementation_sha256"],
        "ml/report_task_test_actions_v3_acceptance.py": evidence.sha256(__file__),
    }
    require(
        metrics.get("reporting_implementation_sha256") == implementation,
        "Frozen metrics policy does not bind the exact reporting implementation.",
    )
    base, selected = arms["native_base"], arms[selected_role]
    cases = base["cases"]
    identities = sorted(cases)
    c0 = natural_c0(
        base["dispositions"], list(evidence.rows(bound(freeze["roster"]))), cases
    )
    specifications = targets["numerical_targets"]
    canonical = cohort_verdict(
        base["records"], selected["records"], identities, specifications["canonical"]
    )
    natural = cohort_verdict(
        base["records"], selected["records"], c0, specifications["natural_C0"]
    )
    actions = functionality(
        cases, selected["records"], specifications["final_functionality"]
    )
    failures = {
        role: saved["runs"][role]["full_issued"]["source_integrity_failures"]
        for role in saved["runs"]
    }
    comparison_binding = comparison.evaluation.file_binding(
        evidence.resolve_path(root, comparison_path, recorded_workspace)
    )
    accounting = {
        role: {
            "requested": len(arm["requests"]),
            "observed": len(arm["observations"]),
            "graded": len(arm["records"]),
            "ordered_case_ids_match": list(arm["requests"])
            == list(arm["observations"])
            == list(arm["records"]),
        }
        for role, arm in arms.items()
    }
    engineering = {
        "source_integrity": verdict(
            sum(failures.values())
            <= targets["engineering_targets"]["maximum_source_integrity_failures"],
            failures_by_role=failures,
            maximum=0,
            evidence=[comparison_binding],
        ),
        "every_registered_eligible_request_accounted": verdict(
            all(
                row["ordered_case_ids_match"]
                and row["requested"] == row["observed"] == row["graded"]
                for row in accounting.values()
            ),
            counts_by_role=accounting,
            evidence=[comparison_binding],
        ),
        "exact_model_reload_runtime_input_bindings": verdict(
            None,
            reason="Requires separately assessed, bound model reload and runtime receipts; frozen assertions are not sufficient for this engineering claim.",
        ),
        "local_preview_apply_undo_export_and_durable_feedback_verified": verdict(
            None,
            reason="Requires separately assessed actual local UI and durable-feedback receipts. Native final results do not establish this.",
        ),
    }
    result = {
        "version": "task-test-actions-v3-acceptance-report-1",
        "scope": "final_test",
        "status": "reported",
        "promotion_authorized": False,
        "model_selection_changed": False,
        "engineering_fixture_only": saved["engineering_fixture_only"],
        "inputs": {
            "targets": comparison.evaluation.file_binding(targets_path),
            "comparison": comparison_binding,
            "freeze": saved["freeze"],
            "selection": freeze["selection_receipt"],
            "metrics_policy": freeze["policies"]["metrics"],
        },
        "selected_model": selected_role,
        "selected_identity": selected["verified"]["model_identity"],
        "comparator": "native_base",
        "canonical": canonical,
        "natural_C0": natural,
        "final_functionality": actions,
        "engineering": engineering,
        "verdicts": {
            "canonical_relative": canonical["status"],
            "natural_C0_non_regression": natural["status"],
            "final_functionality": actions["status"],
            "native_quality_targets": conjunction(
                [canonical["status"], natural["status"], actions["status"]]
            ),
            "engineering_targets": conjunction(
                [r["status"] for r in engineering.values()]
            ),
            "original_DEV_functionality": "unassessed",
            "project_completion": "unassessed",
        },
        "reporting_only": {
            ("six_role_final_accounting" if len(saved["runs"]) == 6 else "five_role_final_accounting"):
                "verified; retained failures do not mean successful model execution",
            "execution_failure_attribution_by_role": {
                role: saved["runs"][role]["failure_attribution"]
                for role in saved["runs"]
            },
            "dependent_natural_C1_C2_case_ids": sorted(
                i
                for i, c in cases.items()
                if c["lineage"]["stratum"] == "natural_source_simulated_session"
            ),
            "authored_challenge_case_ids": sorted(
                i
                for i, c in cases.items()
                if c["lineage"]["stratum"] == "authored_challenge"
            ),
            "all_strata_metrics": "Remain in the bound final comparison; no extra acceptance threshold is applied.",
        },
        "implementation_sha256": implementation,
        "limits": [
            "Only registered targets001 are evaluated, using integer counts and crossproducts. N>0, as registered, determines whether useful-count comparison is assessed; no E>0 rule is substituted.",
            "Fail takes precedence over unassessed; absent denominators or action opportunities never pass by default. Every model failure stays in its registered case denominator.",
            "The selected model comes only from the bound pre-test selection. A better final arm cannot replace it. Original DEV min14, local quality and DPO adoption remain separate prerequisites, not assessed here.",
            "These are relative research point criteria, not absolute reliability floors, population superiority, acoustic-quality certificates, engineering completion or automatic promotion.",
            "Canonical full-cohort, natural C0 and final functionality verdicts remain separate. The ICL subset cannot substitute for their canonical cases.",
            "Registered pipeline failures stay in the quality arithmetic, with startup/interruption/generation attribution reported separately. A pre-generation failure is not an observed answer or a measure of intrinsic model capability. The unchanged source-integrity category also includes safe action-application rejection.",
        ],
    }
    if saved.get("dpo_omission") is not None:
        result["dpo_omission"] = saved["dpo_omission"]
    if saved.get("inference_policy") is not None:
        result["inference_policy"] = saved["inference_policy"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"output": str(output), "verdicts": result["verdicts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument(
        "--comparison", required=True, help="Verified complete frozen-role comparison manifest.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=str(comparison.ROOT))
    parser.add_argument("--recorded-workspace")
    args = parser.parse_args()
    print(
        json.dumps(
            report(
                args.targets,
                args.comparison,
                args.output,
                root=args.root,
                recorded_workspace=args.recorded_workspace,
            )
        )
    )


if __name__ == "__main__":
    main()
