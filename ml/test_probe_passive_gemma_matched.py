import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import probe_passive_gemma_matched as runner
import test_probe_passive_matched as matched_tests


class GemmaMatchedProbeTests(unittest.TestCase):
    def case(self):
        return matched_tests.MatchedProbeTests().case()

    def response(self, text, *, raw=None, terminal=1, finish="stop"):
        ids = [9, terminal] if finish != "length" else [9] * 512
        return {
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": len(ids)},
            "__verbose": {
                "content": text if raw is None else raw,
                "tokens": ids,
                "stop_type": "limit" if finish == "length" else "eos",
            },
        }

    def test_equal_verbose_capture_and_no_primary_grammar_without_global_mutation(self):
        case = self.case()
        original_request, original_grade = (
            runner.probe.request_body,
            runner.probe.evaluate_result,
        )
        for condition, (fmt, constrained) in runner.CONDITIONS.items():
            runtime = runner.isolated_runtime(fmt)
            text = runner.formats.serialize(case["input"], case["expectedOutput"], fmt)
            data = self.response(text)
            body = runtime.request_body(
                case,
                condition,
                constrained=constrained,
                messages=runner.formats.messages(case["input"], fmt),
            )
            self.assertEqual(
                "response_format" in body, condition == "matched-json-constrained"
            )
            self.assertNotIn("grammar", body)
            self.assertTrue(body["verbose"])
            self.assertTrue(body["return_tokens"])
            self.assertEqual(body["max_tokens"], 512)
            self.assertEqual(body["temperature"], 0)
            self.assertNotIn("expectedOutput", json.dumps(body))
            self.assertEqual(
                body["messages"][1], runner.formats.messages(case["input"], "json")[1]
            )
            response = SimpleNamespace(json=lambda: data)
            with patch.object(runner.probe.httpx.Client, "post", return_value=response):
                with runtime.httpx.Client(base_url="http://unused.invalid") as client:
                    client.post("/v1/chat/completions", json=body)
                    row = runtime.evaluate_result(case, text, "stop")
            self.assertTrue(row["raw_valid"])
            self.assertTrue(row["sampled_format_valid"])
            self.assertEqual(row["sampled_token_ids"], [9, 1])
            with self.assertRaisesRegex(ValueError, "No response"):
                runtime.evaluate_result(case, text, "stop")
        self.assertIs(runner.probe.request_body, original_request)
        self.assertIs(runner.probe.evaluate_result, original_grade)

    def test_api_capture_and_raw_fences_are_separate_with_no_output_repair(self):
        case = self.case()
        for fmt in ("json", "rewrite"):
            text = runner.formats.serialize(case["input"], case["expectedOutput"], fmt)
            fenced = "```json\n" + text + "\n```"
            row = runner.grade_response(case, self.response(text, raw=fenced), fmt)
            self.assertTrue(row["raw_valid"])
            self.assertFalse(row["sampled_format_valid"])
            self.assertFalse(row["sampled_source_preserving_parse_valid"])
            self.assertEqual(row["sampled_content"], fenced)
            both = runner.grade_response(case, self.response(fenced), fmt)
            self.assertFalse(both["raw_valid"])
            self.assertFalse(both["format_valid"])

    def test_tool_handoff_truncation_and_bad_token_evidence_are_uncredited(self):
        case = self.case()
        text = runner.formats.serialize(case["input"], case["expectedOutput"], "json")
        for finish in ("stop", "tool_calls"):
            row = runner.grade_response(
                case, self.response(text, terminal=50, finish=finish), "json"
            )
            self.assertFalse(row["raw_valid"])
            self.assertFalse(row["displayed"])
            self.assertIsNone(row["raw_grade"])
            self.assertEqual(row["error_type"], "tool_handoff")
        row = runner.grade_response(case, self.response(text, finish="length"), "json")
        self.assertFalse(row["raw_valid"])
        self.assertFalse(row["sampled_source_preserving_parse_valid"])
        for mutation in ("missing", "count", "terminal", "vocabulary"):
            value = self.response(text)
            if mutation == "missing":
                value.pop("__verbose")
            if mutation == "count":
                value["usage"]["completion_tokens"] = 3
            if mutation == "terminal":
                value["__verbose"]["tokens"] = [9, 10]
            if mutation == "vocabulary":
                value["__verbose"]["tokens"] = [262144, 1]
            with (
                self.subTest(mutation=mutation),
                self.assertRaises((ValueError, KeyError)),
            ):
                runner.grade_response(case, value, "json")

    def test_saved_grades_and_prefixes_are_verified_and_failure_denominators_retained(
        self,
    ):
        case = self.case()
        text = runner.formats.serialize(
            case["input"], case["expectedOutput"], "rewrite"
        )
        response = self.response(text)
        row = {
            **runner.grade_response(case, response, "rewrite"),
            "raw_text": text,
            "finish_reason": "stop",
            "response": response,
            "http_response_ms": 10,
        }
        prefixes = {case["id"]: [1, 2, 3]}
        runner.validate_results([case], [row], prefixes, "rewrite")
        for field, value in (
            ("sampled_format_valid", False),
            ("displayed", False),
            ("sampled_token_ids", [9, 106]),
        ):
            changed = {**row, field: value}
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "grading or token"),
            ):
                runner.validate_results([case], [changed], prefixes, "rewrite")
        changed = copy.deepcopy(row)
        changed["response"]["usage"]["prompt_tokens"] = 4
        with self.assertRaisesRegex(ValueError, "prefix"):
            runner.validate_results([case], [changed], prefixes, "rewrite")
        other = copy.deepcopy(case)
        other["id"] = "failed-case"
        failed = {
            "case_id": other["id"],
            "request_failed": True,
            "raw_valid": False,
            "displayed": False,
            "error_type": "timeout",
            "http_response_ms": 40,
        }
        summary = runner.summarize([case, other], [row, failed])
        self.assertEqual(summary["expected_requests"], 2)
        self.assertEqual(summary["request_errors"], 1)
        self.assertEqual(summary["output_tokens"]["missing"], 1)
        self.assertEqual(summary["fresh_http_response_ms_p50_p95"]["p50_ms"], 10)
        self.assertEqual(summary["fresh_http_response_ms_p50_p95"]["p95_ms"], 40)

    def test_smoke_wrong_arm_partial_and_resumed_runs_fail_final_gate(self):
        config = {
            "version": runner.TRAINING_VERSION,
            "phase": "substantive",
            "format": "json",
        }
        run = {
            **config,
            "status": "training_complete_fresh_reload_pending",
            "mode": "train",
            "training_updates": 201,
            "optimizer_committed_rows": 802,
            "configuration_sha256": "config",
            "evaluation_protocol_sha256": "protocol",
            "initialization": "original_native_base_with_fresh_lora",
            "optimizer_state_resumed": False,
            "final_test_bodies_read": 0,
        }
        runner.validate_run_identity(config, run, "json", "config", "protocol")
        for change in (
            {"phase": "smoke"},
            {"format": "rewrite"},
            {"training_updates": 203},
            {"optimizer_committed_rows": 801},
            {"optimizer_state_resumed": True},
            {"status": "recovery_checkpoint_complete"},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "final201/802"),
            ):
                runner.validate_run_identity(
                    config, {**run, **change}, "json", "config", "protocol"
                )

    def test_malformed_failed_responses_remain_uncredited_and_reportable(self):
        case = self.case()
        case["expectedOutput"] = {"action": "no_action"}
        case["acceptableOutputs"] = []
        prefixes = {case["id"]: [1, 2, 3]}
        for fmt in ("json", "rewrite"):
            text = runner.formats.serialize(case["input"], case["expectedOutput"], fmt)
            for fault in ("verbose", "token_count", "usage", "count_type", "response"):
                response = self.response(text)
                if fault == "verbose":
                    response.pop("__verbose")
                elif fault == "token_count":
                    response["__verbose"]["tokens"] = [1]
                elif fault == "usage":
                    response["usage"] = None
                elif fault == "count_type":
                    response["usage"]["completion_tokens"] = "two"
                else:
                    response = []
                row = {
                    "case_id": case["id"],
                    "raw_text": text,
                    "finish_reason": "stop",
                    "response": response,
                    "request_failed": True,
                    "raw_valid": False,
                    "displayed": False,
                    "error_type": "request_failure",
                    "error": "Synthetic malformed response evidence",
                    "http_response_ms": 17,
                }
                original = copy.deepcopy(row)
                with self.subTest(format=fmt, fault=fault):
                    runner.validate_results([case], [row], prefixes, fmt)
                    summary = runner.summarize([case], [row])
                    self.assertEqual(summary["expected_requests"], 1)
                    self.assertEqual(summary["observed_results"], 1)
                    self.assertEqual(summary["request_errors"], 1)
                    self.assertEqual(summary["invalid_results"], 1)
                    self.assertEqual(summary["preserve_cases"], 1)
                    self.assertEqual(summary["correct_abstentions"], 0)
                    self.assertEqual(summary["displayed_useful_edits"], 0)
                    self.assertEqual(summary["token_evidence_observations"], 0)
                    self.assertEqual(
                        summary["output_tokens"]["missing"],
                        int(fault in ("usage", "count_type", "response")),
                    )
                    self.assertEqual(
                        summary["fresh_http_response_ms_p50_p95"]["p95_ms"], 17
                    )
                    self.assertEqual(row, original)
                    success = {
                        key: value
                        for key, value in row.items()
                        if key not in ("request_failed", "error", "error_type")
                    }
                    with self.assertRaises((ValueError, KeyError, TypeError)):
                        runner.validate_results([case], [success], prefixes, fmt)
                    for credit in (
                        {"raw_valid": True},
                        {"displayed": True},
                        {"raw_grade": {"outcome_match": True}},
                        {"displayed_grade": {"outcome_match": True}},
                        {"source_preserving_valid": True},
                        {"sampled_format_valid": True},
                        {"token_evidence_valid": True},
                    ):
                        with self.assertRaisesRegex(ValueError, "Failed requests"):
                            runner.validate_results(
                                [case], [{**row, **credit}], prefixes, fmt
                            )

    def test_final_exposure_and_bf16_expert_census_are_required(self):
        recipe = {
            "max_updates": 201,
            "draw_count": 802,
            "sample_order": ["a"] * 802,
            "max_total_sequence_tokens": 8020,
            "max_total_completion_tokens": 1604,
        }
        rows = [
            {
                "case_id": "a",
                "source_id": "source",
                "source_group_id": "group",
                "episode_id": "episode",
                "language": "en",
                "action": "no_action",
                "input_ids": [1] * 10,
                "completion_start": 8,
            }
        ] * 802
        expected = runner.gemma.trainer.curriculum.Exposure()
        for row in rows:
            for phase in runner.gemma.trainer.curriculum.PHASES:
                expected.record(phase, row)
        expected.updates = 201
        exposure = expected.summary()
        runner.validate_exposure(exposure, recipe, rows)
        with self.assertRaisesRegex(ValueError, "optimizer exposure"):
            runner.validate_exposure(
                {**exposure, "pending_backward_rows_not_committed": 1}, recipe, rows
            )
        for phase in runner.gemma.trainer.curriculum.PHASES:
            changed = copy.deepcopy(exposure)
            changed["phases"][phase]["counts"]["episode_id"] = {"other": 802}
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "Every"):
                runner.validate_exposure(changed, recipe, rows)
        initialization = {
            "initialization": "original_native_base_with_fresh_lora",
            "smoke_adapter_loaded": False,
            "t1_adapter_loaded": False,
            "other_format_adapter_loaded": False,
            "format": "json",
            "trainable_tensors": {str(i): 1 for i in range(410)},
            "trainable_parameters": 18585600,
        }
        runner.validate_initialization(initialization, "json")
        for key, value in (
            ("smoke_adapter_loaded", True),
            ("t1_adapter_loaded", True),
            ("other_format_adapter_loaded", True),
            ("format", "rewrite"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "fresh205"):
                runner.validate_initialization({**initialization, key: value}, "json")
        model = {
            "target_modules": runner.gemma.trainer.target_names(),
            "gpu_name_contains": "GPU",
            "minimum_gpu_bytes": 75_000_000_000,
            "attn_implementation": "sdpa",
            "experts_implementation": "eager",
        }
        config = {"model": model, "generation": {"eos_token_id": [1, 106, 50]}}
        runtime = {
            "numerical_regime": "native_bfloat16_lora_float32",
            "parameter_dtypes": ["torch.bfloat16"],
            "ignored_native_tensor_names": [],
            "targets": model["target_modules"],
            "effective_generation": config["generation"],
            "native_generation_config": {"eos_token_id": [1, 106, 50]},
            "completion_label_eos_token_id": 1,
            "gpu": "GPU",
            "gpu_total_bytes": 80_000_000_000,
            "text_attn_implementation": "sdpa",
            "text_experts_implementation": "eager",
            "base_parameter_census": {
                "parameter_elements_by_dtype": {"torch.bfloat16": 100}
            },
            "after_adapter_parameter_census": {
                "parameter_elements_by_dtype": {
                    "torch.bfloat16": 100,
                    "torch.float32": 18585600,
                }
            },
        }
        runner.validate_runtime(runtime, config, trained=True)
        for field, value in (
            ("completion_label_eos_token_id", 106),
            ("text_experts_implementation", "unknown"),
            ("targets", model["target_modules"][:-1]),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "runtime census"),
            ):
                runner.validate_runtime({**runtime, field: value}, config, trained=True)
        changed = copy.deepcopy(runtime)
        changed["base_parameter_census"]["parameter_elements_by_dtype"] = {
            "torch.float32": 100
        }
        with self.assertRaisesRegex(ValueError, "runtime census"):
            runner.validate_runtime(changed, config, trained=True)

    def test_protocol_budget_and_missing_source_pins_fail_before_data_access(self):
        protocol = {
            "version": runner.VERSION,
            "conditions": list(runner.CONDITIONS),
            "formats": ["json", "rewrite"],
            "registered_requests_per_condition": 286,
            "full_dev_registered_requests": 289,
            "excluded_from_matched_comparison": 3,
            "training_rows": 802,
            "training_exclusions": 7,
            "final_test_bodies_read": False,
            "training_plan": {"candidate": "gemma26", "updates": 202},
        }
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as d:
            path = Path(d) / "protocol.json"
            path.write_text(json.dumps(protocol))
            with patch.object(runner, "read_bound") as read:
                with self.assertRaisesRegex(ValueError, "require 201 updates"):
                    runner.load_protocol(path)
                read.assert_not_called()
        with self.assertRaisesRegex(ValueError, "complete exact"):
            runner.verify_source_pins({"evaluation_sources_sha256": {}}, {})

    def test_actual_export_normalized_snapshot_and_gemma_census_contract(self):
        root = runner.ROOT / "artifacts/training/passive-t1-gemma26-sft-v1"
        receipt_path = root / "export-attempt-01/receipt.json"
        if not receipt_path.is_file():
            self.skipTest(
                "Requires the local Gemma model export and training artifacts."
            )
        receipt = runner.load_json(receipt_path)
        config = runner.load_json(root / "input/configuration-attempt-01.json")
        registry = runner.load_json(runner.ROOT / "ml/passive-models-v2.json")
        anchor = next(r for r in registry["candidates"] if r["id"] == "gemma26")
        manifest = runner.load_json(root / "sft-attempt-01/train/manifest.json")
        registration = {"inputs": {}, "anchor": anchor, "native_eos": 1}
        runner.validate_export(
            registration,
            config,
            receipt_path,
            receipt,
            root / "sft-attempt-01/train/run.json",
            manifest,
        )
        altered = copy.deepcopy(receipt)
        altered["native_generation_stop_ids"] = [1, 106]
        with self.assertRaisesRegex(ValueError, "EOS/stop-token"):
            runner.validate_export(
                {"inputs": {}, "anchor": anchor, "native_eos": 1},
                config,
                receipt_path,
                altered,
                root / "sft-attempt-01/train/run.json",
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
