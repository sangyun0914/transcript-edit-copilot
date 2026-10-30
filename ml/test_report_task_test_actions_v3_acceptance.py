"""Exact acceptance arithmetic and synthetic sealed-evidence integration only."""

import copy
import json
from pathlib import Path
import unittest

import report_task_test_actions_v3_acceptance as acceptance
import test_compare_task_test_actions_v3 as comparison_fixtures
import test_prepare_task_test_actions_v3 as fixtures

TARGETS = (
    acceptance.comparison.ROOT
    / "artifacts/experiments/passive-actions-v3/final-acceptance-targets-001.json"
)


class AcceptanceTests(unittest.TestCase):
    def test_precision_crossproduct_retains_difference_lost_by_float_rounding(self):
        size = 2**60
        base = dict(N=size + 1, E=size, B=1, D=size, U=size - 1, M=1, S=1)
        selected = {**base, "D": size - 1, "U": size - 2}
        self.assertEqual(base["U"] / base["D"], selected["U"] / selected["D"])
        result = acceptance.compare_counts(base, selected, 0)["criteria"]["precision"]
        self.assertEqual(result["status"], "fail")
        self.assertEqual(
            result["selected_useful_times_base_displayed"] + 1,
            result["base_useful_times_selected_displayed"],
        )
        base = dict(N=5, E=4, B=1, D=2, U=1, M=1, S=1)
        selected = {**base, "D": 4, "U": 2}
        self.assertEqual(acceptance.compare_counts(base, selected, 1)["status"], "pass")

    def test_undefined_fail_precedence_and_separate_model_system_preservation(self):
        abstaining = dict(N=2, E=1, B=1, D=0, U=0, M=1, S=1)
        result = acceptance.compare_counts(abstaining, abstaining, 0)
        self.assertEqual(result["status"], "unassessed")
        self.assertEqual(result["criteria"]["precision"]["status"], "unassessed")
        self.assertEqual(
            acceptance.compare_counts(abstaining, abstaining, 1)["status"], "fail"
        )
        empty = dict.fromkeys(abstaining, 0)
        self.assertEqual(
            acceptance.compare_counts(empty, empty, 0)["status"], "unassessed"
        )
        preserve_only = dict(N=1, E=0, B=1, D=0, U=0, M=1, S=1)
        self.assertEqual(
            acceptance.compare_counts(preserve_only, preserve_only, 0)["criteria"][
                "useful_count"
            ]["status"],
            "pass",
        )
        base = dict(N=2, E=1, B=1, D=1, U=1, M=1, S=1)
        result = acceptance.compare_counts(base, {**base, "M": 0}, 0)
        self.assertEqual(result["criteria"]["model_preservation"]["status"], "fail")
        self.assertEqual(result["criteria"]["system_preservation"]["status"], "pass")

    @unittest.skipUnless(
        TARGETS.is_file(), "Requires the registered local acceptance targets."
    )
    def test_language_regression_cannot_hide_behind_overall_useful_gain(self):
        def row(language, useful):
            return {
                "language": language,
                "judgment": "edit_required",
                "displayed": True,
                "useful": useful,
                "preserved": False,
                "model_abstained": False,
            }

        base = {"en": row("en", True), "ko1": row("ko", False), "ko2": row("ko", False)}
        selected = {
            "en": row("en", False),
            "ko1": row("ko", True),
            "ko2": row("ko", True),
        }
        spec = acceptance.evidence.read(TARGETS)["numerical_targets"]["canonical"]
        result = acceptance.cohort_verdict(base, selected, list(base), spec)
        self.assertEqual(
            result["groups"]["overall"]["criteria"]["useful_count"]["status"], "pass"
        )
        self.assertEqual(
            result["groups"]["en"]["criteria"]["useful_count"]["status"], "fail"
        )
        self.assertEqual(result["status"], "fail")

    @unittest.skipUnless(
        TARGETS.is_file(), "Requires the registered local acceptance targets."
    )
    def test_actual_alternative_actions_count_once_and_suppression_is_not_abstention(
        self,
    ):
        cases = {
            "alternative": {
                "language": "en",
                "judgment": {"status": "edit_required"},
                "accepted_predictions": [
                    {"action": "split_paragraph", "target": [1]},
                    {"action": "merge_paragraph", "target": [1]},
                    {"action": "merge_paragraph", "target": [2]},
                ],
            },
            "suppressed": {
                "language": "ko",
                "judgment": {"status": "preserve"},
                "accepted_predictions": [{"action": "no_action"}],
            },
        }
        records = {
            "alternative": {
                "predicted_action": "merge_paragraph",
                "outcome_match": True,
                "useful": True,
                "preserved": False,
                "model_abstained": False,
            },
            "suppressed": {
                "predicted_action": "correct_text",
                "outcome_match": True,
                "useful": False,
                "preserved": True,
                "model_abstained": False,
            },
        }
        spec = acceptance.evidence.read(TARGETS)["numerical_targets"][
            "final_functionality"
        ]
        result = acceptance.functionality(cases, records, spec)
        self.assertEqual(len(result["cells"]), 14)
        self.assertEqual(result["cells"]["merge_paragraph:en"]["opportunities"], 1)
        self.assertEqual(result["cells"]["merge_paragraph:en"]["status"], "pass")
        self.assertEqual(result["cells"]["split_paragraph:en"]["status"], "fail")
        self.assertEqual(result["cells"]["delete_span:en"]["status"], "unassessed")
        self.assertEqual(result["cells"]["no_action:ko"]["status"], "fail")
        self.assertEqual(result["status"], "fail")

    def test_natural_c0_requires_both_registered_stratum_and_checkpoint(self):
        rows = [
            {
                "id": "c0",
                "stratum": "natural_source_first",
                "checkpoint": 0,
                "language": "en",
                "window_id": "w",
            },
            {
                "id": "c1",
                "stratum": "natural_source_simulated_session",
                "checkpoint": 1,
                "language": "en",
                "window_id": "w",
            },
        ]
        roster = [
            {"slot_id": row["id"], **{k: v for k, v in row.items() if k != "id"}}
            for row in rows
        ]
        cases = {
            row["id"]: {"language": "en", "lineage": {"stratum": row["stratum"]}}
            for row in rows
        }
        self.assertEqual(acceptance.natural_c0(rows, roster, cases), ["c0"])
        for index, checkpoint in ((0, 1), (1, 0)):
            changed, changed_roster = copy.deepcopy(rows), copy.deepcopy(roster)
            changed[index]["checkpoint"] = changed_roster[index]["checkpoint"] = (
                checkpoint
            )
            with (
                self.subTest(index=index),
                self.assertRaisesRegex(ValueError, "checkpoint"),
            ):
                acceptance.natural_c0(changed, changed_roster, cases)

    def prepare_comparison(self, *, pin_reporting=True):
        fixture = fixtures.SealedTaskTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        targets = acceptance.evidence.read(TARGETS)
        fixture.freeze["frozen_at"] = fixtures.sealed.now()
        policy_path = fixture.root / "metrics-policy.json"
        policy = fixtures.sealed.read_json(policy_path)
        policy["decision_thresholds"] = {
            "registered_targets": fixtures.sealed.binding(TARGETS),
            "numerical_targets": targets["numerical_targets"],
            "exact_arithmetic": targets["exact_arithmetic"],
        }
        if pin_reporting:
            comparator = acceptance.comparison
            sources = [
                Path(module.__file__).resolve()
                for module in (
                    comparator.paired,
                    comparator.evidence,
                    comparator.evaluation,
                    comparator.conditions,
                    comparator.evaluation.formats,
                    comparator.evaluation.contract,
                    comparator,
                    acceptance,
                )
            ]
            sources.append(comparator.evaluation.contract.PROMPT_PATH)
            policy["reporting_implementation_sha256"] = {
                str(path.relative_to(comparator.ROOT)): fixtures.sealed.sha256(path)
                for path in sources
            }
        policy_path.write_text(json.dumps(policy))
        fixture.freeze["policies"]["metrics"] = fixtures.sealed.binding(policy_path)
        fixture.replace_freeze()
        release = fixture.seal(fixture.prepare())
        helper = comparison_fixtures.FinalComparisonTests()
        grades = [
            helper.grade(fixture, release, role, role)
            for role in acceptance.comparison.ROLES
        ]
        comparison_dir = fixture.root / "comparison"
        acceptance.comparison.compare(grades, comparison_dir, resamples=30)
        return fixture, comparison_dir, helper

    @unittest.skipUnless(
        TARGETS.is_file(), "Requires the registered local acceptance targets."
    )
    def test_bound_selection_exact_targets_and_comparison_integrity_end_to_end(self):
        fixture, comparison_dir, helper = self.prepare_comparison()
        output = fixture.root / "acceptance.json"
        acceptance.report(TARGETS, comparison_dir / "manifest.json", output)
        result = fixtures.sealed.read_json(output)
        self.assertEqual(result["selected_model"], "sft")
        self.assertFalse(result["promotion_authorized"])
        self.assertEqual(result["verdicts"]["canonical_relative"], "fail")
        self.assertEqual(result["verdicts"]["natural_C0_non_regression"], "unassessed")
        self.assertEqual(result["engineering"]["source_integrity"]["status"], "pass")
        self.assertEqual(
            result["engineering"]["every_registered_eligible_request_accounted"][
                "status"
            ],
            "pass",
        )
        self.assertEqual(result["verdicts"]["engineering_targets"], "unassessed")
        self.assertEqual(result["verdicts"]["original_DEV_functionality"], "unassessed")
        with self.assertRaisesRegex(ValueError, "fresh"):
            acceptance.report(TARGETS, comparison_dir / "manifest.json", output)
        changed_targets = fixture.root / "changed-targets.json"
        changed_targets.write_bytes(TARGETS.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "targets001 receipt changed"):
            acceptance.report(
                changed_targets,
                comparison_dir / "manifest.json",
                fixture.root / "bad-targets.json",
            )
        manifest_path = comparison_dir / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        missing_payload = fixtures.sealed.read_json(manifest_path)
        missing_payload["files_sha256"].pop("comparison.json")
        manifest_path.write_text(json.dumps(missing_payload))
        with self.assertRaisesRegex(ValueError, "complete final-test comparison"):
            acceptance.report(
                TARGETS, manifest_path, fixture.root / "unbound-comparison.json"
            )
        self.assertFalse((fixture.root / "unbound-comparison.json").exists())
        manifest_path.write_bytes(manifest_bytes)
        comparison_path = comparison_dir / "comparison.json"
        changed = fixtures.sealed.read_json(comparison_path)
        changed["canonical_full_five_roles"]["arms"]["sft"]["precision"][
            "numerator"
        ] += 1
        helper.rewrite(comparison_path, changed)
        with self.assertRaisesRegex(ValueError, "differs from fully verified evidence"):
            acceptance.report(
                TARGETS,
                comparison_dir / "manifest.json",
                fixture.root / "bad-comparison.json",
            )
        self.assertFalse((fixture.root / "bad-comparison.json").exists())
        unpinned, unpinned_comparison, _ = self.prepare_comparison(pin_reporting=False)
        unpinned_output = unpinned.root / "unpinned-report.json"
        with self.assertRaisesRegex(ValueError, "exact reporting implementation"):
            acceptance.report(
                TARGETS, unpinned_comparison / "manifest.json", unpinned_output
            )
        self.assertFalse(unpinned_output.exists())


if __name__ == "__main__":
    unittest.main()
