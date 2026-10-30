"""Comparison engineering only; all final-test-shaped bodies are synthetic."""

import copy
import json
import unittest
from unittest import mock

import compare_task_test_actions_v3 as comparison
import test_prepare_task_test_actions_v3 as fixtures
from test_prepare_task_test_actions_v3 import (
    FixtureTokenizer,
    OverflowICLTokenizer,
    sealed,
)


class PartialICLTokenizer(FixtureTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if (
            len(messages) > 2
            and json.loads(messages[-1]["content"])["language"] == "en"
        ):
            return [2] * 4000
        return super().apply_chat_template(messages, **kwargs)


class FinalComparisonTests(unittest.TestCase):
    def fixture(self, tokenizer=None):
        fixture = fixtures.SealedTaskTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.tokenizer = tokenizer or PartialICLTokenizer()
        inputs = fixture.prepare()
        release = fixture.seal(inputs)
        paths = {
            role: self.grade(fixture, release, role, role) for role in comparison.ROLES
        }
        return fixture, inputs, release, paths

    def grade(self, fixture, release, role, name):
        observed = fixture.root / (name + "-observed")
        output = fixture.root / (name + "-graded")

        def predictor(request):
            if request["case_id"] == "engineering-edit":
                if role == "native_base":
                    raise RuntimeError("Synthetic registered generation failure")
                if role == "sft_dpo":
                    return {
                        "text": '{"action":"correct_text","target":[1],"text":"we"}',
                        "finish_reason": "stop",
                    }
            return {"text": '{"action":"no_action"}', "finish_reason": "stop"}

        args = (
            {}
            if role in {"always_no_action", "consistent_history_rule"}
            else {
                "predictor": predictor,
                "predictor_identity": {
                    "model": "native_base" if role == "base_icl" else role,
                    "engineering_fixture_only": True,
                },
            }
        )
        sealed.record_run(
            fixture.freeze_path,
            release,
            role,
            observed,
            tokenizer=fixture.tokenizer,
            **args,
        )
        sealed.grade(fixture.freeze_path, release, observed, output)
        return output / "manifest.json"

    def rewrite(self, path, value, *, jsonl=False):
        if jsonl:
            path.write_text("".join(json.dumps(row) + "\n" for row in value))
        else:
            path.write_text(json.dumps(value) + "\n")
        manifest_path = path.parent / "manifest.json"
        manifest = sealed.read_json(manifest_path)
        manifest["files_sha256"][path.name] = sealed.sha256(path)
        manifest_path.write_text(json.dumps(manifest) + "\n")

    def test_full_canonical_and_common_icl_keep_failures_and_true_action_counts(self):
        fixture, _, _, paths = self.fixture()
        output = fixture.root / "comparison"
        result = comparison.compare(paths.values(), output, resamples=30)
        self.assertEqual((result["canonical_cases"], result["common_cases"]), (2, 1))
        report = sealed.read_json(output / "comparison.json")
        full, common = (
            report["canonical_full_five_roles"],
            report["icl_common_six_roles"],
        )
        self.assertEqual(len(full["pairs"]), 10)
        self.assertEqual(len(common["pairs"]), 15)
        self.assertEqual(
            full["arms"]["native_base"]["status_counts"],
            {"request_failure": 1, "completed": 1},
        )
        self.assertEqual(full["arms"]["native_base"]["recall"]["denominator"], 1)
        self.assertEqual(
            report["runs"]["native_base"]["failed_outcomes"][0]["case_id"],
            "engineering-edit",
        )
        self.assertEqual(
            report["runs"]["native_base"]["failure_attribution"]["request_failures"],
            1,
        )
        success = full["arms"]["sft"]["successful_predicted_action_language"]
        self.assertEqual(
            (len(success), success["correct_text:en"], success["no_action:ko"]),
            (14, 0, 1),
        )
        self.assertEqual(
            full["arms"]["sft_dpo"]["successful_predicted_action_language"][
                "correct_text:en"
            ],
            1,
        )
        self.assertEqual(
            full["pairs"]["sft__sft_dpo"]["paired_outcomes"]["right_only"]["case_ids"],
            ["engineering-edit"],
        )
        pair = common["pairs"]["native_base__base_icl"]
        self.assertEqual(pair["left_only_case_ids"], ["engineering-edit"])
        self.assertEqual(pair["right_only_case_ids"], [])
        self.assertEqual(
            common["registered_admitted_compared_denominators"]["language"]["en"][
                "compared"
            ],
            0,
        )
        self.assertEqual(
            common["registered_admitted_compared_denominators"]["language"]["en"][
                "registered"
            ],
            2,
        )
        self.assertEqual(
            common["arms"]["always_no_action"]["precision"]["denominator"], 0
        )
        self.assertIsNone(common["arms"]["always_no_action"]["recall"]["value"])
        interval = pair["source_resampling"]["percentile95_descriptive"]["precision"]
        self.assertEqual(interval["defined_resamples"], 0)
        self.assertIsNone(interval["low"])
        with self.assertRaisesRegex(ValueError, "fresh"):
            comparison.compare(paths.values(), output, resamples=30)
        with mock.patch.object(
            comparison,
            "full_cohort_policy",
            return_value={"engineering_fixture_only": True},
        ):
            with self.assertRaisesRegex(ValueError, "full admitted cohort"):
                comparison.compare(
                    paths.values(),
                    fixture.root / "small-must-not-drop-icl",
                    resamples=30,
                )

    def test_full_cohort_policy_reports_one_table_and_retains_failures(self):
        import gemma_final_test_runtime as gemma
        import small_final_test_runtime as small

        fixture, _, _, paths = self.fixture(FixtureTokenizer())
        output = fixture.root / "small-full-cohort"
        policy = {
            "engineering_fixture_only": True,
            "version": small.INFERENCE_POLICY_VERSION,
        }
        with mock.patch.object(comparison, "full_cohort_policy", return_value=policy):
            result = comparison.compare(paths.values(), output, resamples=30)
        self.assertEqual((result["canonical_cases"], result["common_cases"]), (2, 2))
        report = sealed.read_json(output / "comparison.json")
        self.assertNotIn("canonical_full_five_roles", report)
        self.assertNotIn("icl_common_six_roles", report)
        self.assertEqual(
            report["cohort_keys"],
            {
                "canonical": "full_cohort_all_roles",
                "icl_common": "full_cohort_all_roles",
            },
        )
        table = report["full_cohort_all_roles"]
        self.assertEqual(table["cases"], 2)
        self.assertEqual(set(table["arms"]), set(comparison.ROLES))
        self.assertEqual(len(table["pairs"]), 15)
        for role in comparison.ROLES:
            self.assertEqual(report["runs"][role]["full_issued"]["requests"], 2)
            self.assertEqual(report["runs"][role]["run_eligibility_exclusions"], [])
            self.assertEqual(report["runs"][role]["outside_common_case_ids"], [])
        self.assertEqual(
            table["arms"]["native_base"]["status_counts"],
            {"request_failure": 1, "completed": 1},
        )
        self.assertEqual(table["arms"]["native_base"]["recall"]["denominator"], 1)
        with mock.patch.object(
            comparison,
            "full_cohort_policy",
            return_value={**policy, "version": gemma.INFERENCE_POLICY_VERSION},
        ):
            comparison.compare(
                paths.values(), fixture.root / "gemma-full-cohort", resamples=30
            )
        gemma_report = sealed.read_json(
            fixture.root / "gemma-full-cohort/comparison.json"
        )
        self.assertEqual(gemma_report["full_cohort_all_roles"], table)
        self.assertEqual(
            gemma_report["implementation_sha256"]["ml/gemma_final_test_runtime.py"],
            gemma.sha(gemma.__file__),
        )

    def test_empty_icl_intersection_retains_full_cohort_and_undefined_denominators(
        self,
    ):
        fixture, _, _, paths = self.fixture(OverflowICLTokenizer())
        output = fixture.root / "empty-common"
        comparison.compare(paths.values(), output, resamples=20)
        report = sealed.read_json(output / "comparison.json")
        self.assertEqual(report["canonical_full_five_roles"]["cases"], 2)
        common = report["icl_common_six_roles"]
        self.assertEqual(common["cases"], 0)
        self.assertEqual(report["runs"]["base_icl"]["full_issued"]["requests"], 0)
        self.assertEqual(
            len(report["runs"]["base_icl"]["run_eligibility_exclusions"]), 2
        )
        for summary in common["arms"].values():
            self.assertEqual(summary["recall"]["denominator"], 0)
            self.assertIsNone(summary["precision"]["value"])
        for pair in common["pairs"].values():
            self.assertEqual(pair["source_resampling"]["source_groups"], 0)
            self.assertEqual(pair["paired_outcomes"]["right_only"]["count"], 0)

    def test_missing_outcome_and_changed_observation_prefix_are_rejected(self):
        fixture, _, _, paths = self.fixture()
        path = paths["native_base"].parent / "graded.jsonl"
        original = sealed.rows(path)
        self.rewrite(path, original[1:], jsonl=True)
        with self.assertRaisesRegex(ValueError, "Every registered request"):
            comparison.load_grade(paths["native_base"], sealed.ROOT)
        self.rewrite(path, original, jsonl=True)
        observed = fixture.root / "native_base-observed"
        observations = sealed.rows(observed / "observations.jsonl")
        observations[0]["prefix_ids_sha256"] = "different prefix"
        self.rewrite(observed / "observations.jsonl", observations, jsonl=True)
        manifest = sealed.read_json(paths["native_base"])
        manifest["observation_manifest"] = sealed.binding(observed / "manifest.json")
        paths["native_base"].write_text(json.dumps(manifest) + "\n")
        with self.assertRaisesRegex(
            ValueError, "Observed model, input or native prefix"
        ):
            comparison.load_grade(paths["native_base"], sealed.ROOT)

    def test_changed_accepted_references_cannot_mix_sealed_releases(self):
        fixture, inputs, _, paths = self.fixture()
        references = copy.deepcopy(fixture.references(inputs))
        reference = next(row for row in references if row["id"] == "engineering-edit")
        reference["primary"]["text"] = "new"
        reference["accepted_predictions"][0]["text"] = "new"
        path = fixture.root / "changed-references.jsonl"
        sealed.write_rows(path, references)
        changed = fixture.root / "changed-sealed"
        sealed.seal_references(
            fixture.freeze_path, inputs, path, changed, tokenizer=fixture.tokenizer
        )
        paths["sft"] = self.grade(fixture, changed, "sft", "changed-sft")
        output = fixture.root / "must-not-publish"
        with self.assertRaisesRegex(
            ValueError, "mixes final-test freezes or sealed releases"
        ):
            comparison.compare(paths.values(), output, resamples=20)
        self.assertFalse(output.exists())

    def test_resigned_scored_provenance_must_still_match_sealed_case(self):
        fixture, _, _, paths = self.fixture()
        path = paths["sft"].parent / "graded.jsonl"
        records = sealed.rows(path)
        records[0]["stratum"] = "different_stratum"
        self.rewrite(path, records, jsonl=True)
        report_path = path.parent / "report.json"
        report = sealed.read_json(report_path)
        report["by"]["stratum"] = {
            key: comparison.evaluation.summarize(
                [r for r in records if r["stratum"] == key]
            )
            for key in {r["stratum"] for r in records}
        }
        self.rewrite(report_path, report)
        with self.assertRaisesRegex(ValueError, "scored provenance"):
            comparison.load_grade(paths["sft"], sealed.ROOT)

    def test_resigned_consistent_grade_must_match_actual_observation(self):
        fixture, _, _, paths = self.fixture()
        path = paths["sft"].parent / "graded.jsonl"
        records = sealed.rows(path)
        alternate = sealed.rows(paths["sft_dpo"].parent / "graded.jsonl")
        self.assertFalse(records[0]["useful"])
        self.assertTrue(alternate[0]["useful"])
        records[0] = alternate[0]
        self.rewrite(path, records, jsonl=True)
        report_path = path.parent / "report.json"
        report = sealed.read_json(report_path)
        report["summary"] = comparison.evaluation.summarize(records)
        for dimension in report["by"]:
            groups = comparison.defaultdict(list)
            for row in records:
                groups[
                    row[dimension] if row[dimension] is not None else "unavailable"
                ].append(row)
            report["by"][dimension] = {
                key: comparison.evaluation.summarize(values)
                for key, values in groups.items()
            }
        self.rewrite(report_path, report)
        with self.assertRaisesRegex(ValueError, "deterministic execution"):
            comparison.load_grade(paths["sft"], sealed.ROOT)

    def test_failures_before_and_during_generation_are_attributed_without_exclusion(
        self,
    ):
        observations = {
            "startup": {
                "status": "request_failure",
                "failure_stage": "preflight_or_model_loading",
                "generation_attempted": False,
            },
            "generation": {
                "status": "request_failure",
                "failure_stage": "generation",
                "generation_attempted": True,
            },
            "fixture": {"status": "request_failure"},
            "invalid": {"status": "completed"},
        }
        records = {key: {"status": "request_failure"} for key in observations}
        records["invalid"]["status"] = "invalid_output"
        result = comparison.failure_attribution(
            {"observations": observations, "records": records}
        )
        self.assertEqual(result["request_failures"], 3)
        self.assertEqual(
            result["request_failure_stages"],
            {"preflight_or_model_loading": 1, "generation": 1, "unspecified": 1},
        )
        self.assertEqual(result["pre_generation_request_failures"], 1)
        self.assertEqual(result["generation_attempted_request_failures"], 1)
        self.assertEqual(result["unspecified_generation_attempt_request_failures"], 1)
        self.assertEqual(
            result["completed_response_grade_status_counts"], {"invalid_output": 1}
        )


if __name__ == "__main__":
    unittest.main()
