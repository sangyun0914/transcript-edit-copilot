"""Regressions for the combined history diagnostic's evidence and paired unit."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import evaluate_passive_actions_v3 as evaluation
import prepare_history_length_actions_v3 as history
import summarize_final_history_actions_v3 as summary
from test_prepare_history_length_actions_v3 import Tokenizer


def replace(path, value, *, rows=False):
    Path(path).write_text(
        "".join(json.dumps(row) + "\n" for row in value) if rows else json.dumps(value)
    )


class FinalHistorySummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not history.FIXTURE.is_file():
            raise unittest.SkipTest(
                "History diagnostics require local experiment fixtures."
            )
        fixture = json.loads(history.FIXTURE.read_text())
        cls.cases, cls.requests = [], []
        for episode in fixture["episodes"]:
            cases, _, _ = history.build_episode(
                fixture, episode, Tokenizer(), history.digest(fixture)
            )
            for case in cases:
                case = copy.deepcopy(case)
                length = case["history_length"]["length"]
                case["id"] += f"-h{length}"
                case.pop("condition")
                case.pop("condition_sha256")
                case["messages"] = evaluation.formats.messages(case["input"], "json")
                cls.cases.append(case)
                prefix = Tokenizer().apply_chat_template(case["messages"])
                cls.requests.append(
                    {
                        "case_id": case["id"],
                        "split": "dev",
                        "format": "json",
                        "input": case["input"],
                        "messages": case["messages"],
                        "input_sha256": evaluation.semantic_input_hash(case["input"]),
                        "messages_sha256": evaluation.digest(case["messages"]),
                        "prefix_ids": prefix,
                        "prefix_ids_sha256": evaluation.digest(prefix),
                    }
                )

    def evidence(self, root):
        preparation, run = root / "preparation", root / "run"
        preparation.mkdir()
        run.mkdir()
        base = {"repository": "unit/native", "revision": "unit-pinned"}
        evaluation.write(root / "configuration.json", {"base": base})
        evaluation.write(
            root / "dataset-manifest.json", {"scope": "synthetic_unit_fixture"}
        )
        evaluation.write_rows(preparation / "cases.jsonl", self.cases)
        evaluation.write_rows(preparation / "requests-json.jsonl", self.requests)
        evaluation.write_rows(preparation / "exclusions.jsonl", [])
        manifest = {
            "version": evaluation.VERSION,
            "final_builder_version": summary.final.VERSION,
            "final_test_bodies_read": False,
            "cohort": "history",
            "excluded": 0,
            "requests": 24,
            "history_condition_cases": 24,
            "history_episodes": 8,
            "unique_editor_anchors": 8,
            "original_development_cases": 24,
            "eligibility_format": "json",
            "case_ids": [case["id"] for case in self.cases],
            "history_anchor_bindings": {
                case["history_length"]["episode_id"]: [
                    case["bundle"]["document_hash"],
                    case["history_length"]["non_history_sha256"],
                    case["history_length"]["history_chain_sha256"],
                ]
                for case in self.cases
                if case["history_length"]["length"] == 8
            },
            "dataset_manifest": evaluation.file_binding(root / "dataset-manifest.json"),
            "model_preparation": evaluation.file_binding(root / "configuration.json"),
            "native_model": base,
            "sources_sha256": {
                evaluation.__file__: evaluation.sha256(evaluation.__file__)
            },
            "files_sha256": {
                path.name: evaluation.sha256(path) for path in preparation.iterdir()
            },
        }
        evaluation.write(preparation / "manifest.json", manifest)
        identity = {
            "kind": "unit_native_sft",
            "repository": base["repository"],
            "revision": base["revision"],
            "configuration_sha256": manifest["model_preparation"]["sha256"],
            "adapter_sha256": "unit_adapter",
        }
        observations = []
        for index, (case, request) in enumerate(
            zip(self.cases, self.requests, strict=True)
        ):
            row = {
                key: request[key]
                for key in (
                    "case_id",
                    "format",
                    "input_sha256",
                    "messages_sha256",
                    "prefix_ids_sha256",
                )
            }
            row.update(
                model_identity=identity,
                status="completed",
                finish_reason="stop",
                generation_ms=1.0,
                text=evaluation.formats.serialize(
                    case["input"], case["primary"], "json"
                ),
            )
            if index == 0:
                row["text"] = "invalid JSON"
            elif index == 5:
                row.update(
                    status="request_failure",
                    error="synthetic transport failure",
                    generation_ms=None,
                )
            observations.append(row)
        evaluation.write_rows(run / "observations.jsonl", observations)
        evaluation.write(
            run / "run.json",
            {
                "version": "unit-native",
                "status": "inference_complete",
                "format": "json",
                "configuration_sha256": manifest["model_preparation"]["sha256"],
                "requests_sha256": evaluation.sha256(
                    preparation / "requests-json.jsonl"
                ),
                "requested": 24,
                "observed": 24,
                "final_test_bodies_read": False,
                "model_identity": identity,
                "schema_constrained": False,
                "numerical_regime": "native_bfloat16_with_optional_float32_lora",
            },
        )
        evaluation.write(
            run / "manifest.json",
            {
                "version": "unit-native",
                "status": "inference_complete",
                "files_sha256": {
                    path.name: evaluation.sha256(path) for path in run.iterdir()
                },
            },
        )
        evaluation.grade(preparation, run, root / "graded", "json")
        return root / "graded/report.json"

    def rebind_preparation(self, report_path):
        report = summary.read(report_path)
        manifest_path = Path(report["preparation"]["path"])
        manifest = summary.read(manifest_path)
        manifest["files_sha256"] = {
            name: evaluation.sha256(manifest_path.parent / name)
            for name in manifest["files_sha256"]
        }
        replace(manifest_path, manifest)
        report["preparation"] = evaluation.file_binding(manifest_path)
        replace(report_path, report)

    def test_complete_summary_keeps_invalid_and_failure_and_conditional_references(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = self.evidence(root)
            result = summary.summarize(report_path, root / "summary")
            self.assertEqual(
                (result["anchors"], result["overall"]["requests"]), (8, 24)
            )
            self.assertEqual(
                result["overall"]["status_counts"],
                {"invalid_output": 1, "completed": 22, "request_failure": 1},
            )
            self.assertEqual(
                {
                    key: value["outcome_accuracy"]["numerator"]
                    for key, value in result["per_arm"].items()
                },
                {"H0": 7, "H3": 8, "H8": 7},
            )
            self.assertEqual(
                {
                    key: len(value["changed_reference_effect_or_judgment"])
                    for key, value in result["paired"].items()
                },
                {"H0_vs_H3": 4, "H0_vs_H8": 6, "H3_vs_H8": 2},
            )
            self.assertEqual(
                result["paired"]["H0_vs_H3"]["outcomes"]["right_wins"]["count"], 1
            )
            self.assertEqual(result["by"]["language"]["en"]["anchors"], 4)
            self.assertEqual(len(result["by"]["source_group"]), 8)
            errors = [
                arm["grade"]["error"]
                for anchor in result["anchor_results"].values()
                for arm in anchor["arms"].values()
                if "error" in arm["grade"]
            ]
            self.assertIn("synthetic transport failure", errors)

    def test_missing_or_duplicate_arm_is_rejected(self):
        for cases in (self.cases[:-1], [*self.cases[:-1], self.cases[0]]):
            with (
                self.subTest(count=len(cases)),
                self.assertRaisesRegex(ValueError, "24 unique"),
            ):
                summary.group_anchors(cases, {})

    def test_rebound_model_or_configuration_mismatch_is_rejected(self):
        for change in ("model", "configuration"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                report_path = self.evidence(root)
                report = summary.read(report_path)
                if change == "model":
                    report["model_identity"]["adapter_sha256"] = "different_adapter"
                    replace(report_path, report)
                else:
                    config_path = root / "configuration.json"
                    replace(
                        config_path,
                        {
                            "base": {
                                "repository": "unit/native",
                                "revision": "unit-pinned",
                            },
                            "changed": True,
                        },
                    )
                    manifest_path = Path(report["preparation"]["path"])
                    manifest = summary.read(manifest_path)
                    manifest["model_preparation"] = evaluation.file_binding(config_path)
                    replace(manifest_path, manifest)
                    self.rebind_preparation(report_path)
                with self.assertRaisesRegex(
                    ValueError, "keep model|different configuration"
                ):
                    summary.load_history(report_path)

    def test_rebound_prompt_or_executable_reference_mismatch_is_rejected(self):
        for change in ("prompt", "reference"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                report_path = self.evidence(root)
                cases_path = root / "preparation/cases.jsonl"
                cases = evaluation.rows(cases_path)
                if change == "prompt":
                    cases[0]["messages"][0]["content"] += " Changed prompt."
                else:
                    cases[0]["accepted_result_sha256"] = [
                        "different executable reference"
                    ]
                replace(cases_path, cases, rows=True)
                self.rebind_preparation(report_path)
                with self.assertRaisesRegex(
                    ValueError, "canonical messages|Reference execution"
                ):
                    summary.load_history(report_path)

    def test_one_observation_cannot_change_model_within_an_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = self.evidence(root)
            observations_path = root / "run/observations.jsonl"
            observations = evaluation.rows(observations_path)
            observations[0]["model_identity"]["adapter_sha256"] = "other_model"
            replace(observations_path, observations, rows=True)
            run_manifest_path = root / "run/manifest.json"
            run_manifest = summary.read(run_manifest_path)
            run_manifest["files_sha256"]["observations.jsonl"] = evaluation.sha256(
                observations_path
            )
            replace(run_manifest_path, run_manifest)
            report = summary.read(report_path)
            report["observations"] = evaluation.file_binding(observations_path)
            report["inference_manifest"] = evaluation.file_binding(run_manifest_path)
            replace(report_path, report)
            with self.assertRaisesRegex(ValueError, "mixes or omits model"):
                summary.load_history(report_path)


if __name__ == "__main__":
    unittest.main()
