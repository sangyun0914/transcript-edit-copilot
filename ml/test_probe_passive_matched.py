import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_passive_matched as runner
from passive_contract import OUTPUT_SCHEMA
from test_passive_contract import CORRECTION, builder, document


class MatchedProbeTests(unittest.TestCase):
    def case(self):
        source = document()
        bundle = builder().build(source, "b", as_of=0)
        return {
            "id": "matched-case",
            "input": bundle["input"],
            "expectedOutput": CORRECTION,
            "acceptableOutputs": [],
            "metadata": {
                "source_group_id": "source",
                "language": "en",
                "episode_id": "episode",
            },
            "_execution": {
                "before": source,
                "bundle": bundle,
                "feedback_ledger": None,
                "as_of": 0,
            },
        }

    def test_isolated_callback_preserves_shared_runner_and_equivalent_execution(self):
        original_grader = runner.probe.evaluate_result
        case = self.case()
        results = []
        for fmt in ("json", "rewrite"):
            runtime = runner.isolated_runtime(fmt)
            self.assertIsNot(runtime, runner.probe)
            self.assertIs(runner.probe.evaluate_result, original_grader)
            text = runner.formats.serialize(case["input"], CORRECTION, fmt)
            row = runtime.evaluate_result(case, text)
            self.assertTrue(row["format_valid"])
            self.assertTrue(row["source_preserving_valid"])
            results.append(row)
        for field in (
            "prediction",
            "raw_grade",
            "displayed",
            "displayed_grade",
            "suppression_reason",
        ):
            self.assertEqual(results[0][field], results[1][field])
        self.assertIs(runner.probe.evaluate_result, original_grader)

    def test_primary_arms_have_no_grammar_and_keep_private_evidence_out_of_requests(
        self,
    ):
        case = self.case()
        for condition, (fmt, constrained) in runner.CONDITIONS.items():
            messages = runner.formats.messages(case["input"], fmt)
            body = runner.probe.request_body(
                case, condition, constrained=constrained, messages=messages
            )
            self.assertEqual(
                "response_format" in body, condition == "matched-json-constrained"
            )
            self.assertEqual(body["max_tokens"], 512)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(
                body["messages"][1], runner.formats.messages(case["input"], "json")[1]
            )
            self.assertNotIn("_execution", body)
            self.assertNotIn("expectedOutput", json.dumps(body))
            if constrained:
                self.assertEqual(
                    body["response_format"]["json_schema"]["schema"],
                    runner.probe.runtime_schema(OUTPUT_SCHEMA),
                )

    def test_rewrite_extra_changes_and_truncation_are_not_credited(self):
        case = self.case()
        valid = runner.formats.serialize(case["input"], CORRECTION, "rewrite")
        changed = json.loads(valid)
        changed["after"][0]["boundary_before"] = not changed["after"][0][
            "boundary_before"
        ]
        result = runner.grade(case, json.dumps(changed), format_name="rewrite")
        self.assertTrue(result["format_valid"])
        self.assertTrue(result["extra_change"])
        self.assertFalse(result["raw_valid"])
        self.assertFalse(result["displayed"])
        truncated = runner.grade(case, valid, "length", format_name="rewrite")
        self.assertEqual(truncated["error_type"], "truncated")
        self.assertFalse(truncated["raw_valid"])
        self.assertFalse(truncated["displayed"])
        failed = runner.grade(case, "not json", format_name="rewrite")
        self.assertEqual(failed["error_type"], "format_invalid")
        self.assertFalse(failed["format_valid"])

    def test_recipes_require_identical_draws_and_allow_only_format_token_totals(self):
        config = {
            "training": {
                "sample_order": ["a", "b"],
                "seed": 3407,
                "max_total_sequence_tokens": 100,
                "max_total_completion_tokens": 10,
            },
            "bindings": {"preparation": {"sha256": "same"}},
            "candidate_id": "qwen35-9b",
            "model": {},
            "libraries": {},
            "generation": {"eos_token_id": 248046},
            "base": {"files_sha256": {"base": "same"}},
            "tokenizer": {"files_sha256": {"tokenizer": "same"}},
        }
        arms = {
            fmt: {"configuration": copy.deepcopy(config)} for fmt in ("json", "rewrite")
        }
        arms["rewrite"]["configuration"]["training"].update(
            max_total_sequence_tokens=200, max_total_completion_tokens=20
        )
        runner.validate_matched_recipes(arms)
        for key, value in (("sample_order", ["b", "a"]), ("seed", 42)):
            changed = copy.deepcopy(arms)
            changed["rewrite"]["configuration"]["training"][key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "share sampling order"),
            ):
                runner.validate_matched_recipes(changed)

    def test_old_202_update_preregistration_is_rejected_before_any_payload_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
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
                "training_plan": {"updates": 202},
            }
            path.write_text(json.dumps(protocol))
            with (
                patch.object(
                    runner, "project_path", side_effect=lambda value: Path(value)
                ),
                patch.object(runner, "read_bound") as read,
            ):
                with self.assertRaisesRegex(ValueError, "require 201 updates"):
                    runner.load_protocol(path)
                read.assert_not_called()

    def test_arm_gate_rejects_smoke_partial_and_wrong_format_before_export_usage(self):
        config = {"version": "passive-sft-matched-v1", "format": "rewrite"}
        export = {"training_record": {"path": "run", "sha256": "run-sha"}}
        reload_manifest = {
            "version": "passive-sft-matched-v1",
            "status": "fresh_reload_passed_local_export_pending",
        }
        run = {
            "version": "passive-sft-matched-v1",
            "format": "rewrite",
            "status": "training_complete_fresh_reload_pending",
            "mode": "train",
            "training_updates": 201,
            "optimizer_committed_rows": 802,
            "final_test_bodies_read": 0,
            "configuration_sha256": "config-sha",
            "evaluation_protocol_sha256": "protocol-sha",
        }
        config_path, protocol_path = Path("config"), Path("protocol")
        registration = {
            "inputs": {config_path: "config-sha", protocol_path: "protocol-sha"},
            "execution_path": Path("execution"),
            "protocol_path": protocol_path,
        }
        for change in (
            {"training_updates": 2},
            {"training_updates": 202},
            {"optimizer_committed_rows": 801},
            {"format": "json"},
            {"status": "recovery_checkpoint_complete"},
        ):
            values = [
                (config_path, config),
                (Path("receipt"), export),
                (Path("reload"), reload_manifest),
                (Path("run"), {**run, **change}),
            ]
            with (
                patch.object(runner, "bound_json", side_effect=values),
                self.subTest(change=change),
            ):
                with self.assertRaisesRegex(ValueError, "final 201-update"):
                    runner.validate_arm(
                        registration,
                        "rewrite",
                        dict.fromkeys(
                            (
                                "training_configuration",
                                "export_receipt",
                                "fresh_reload",
                            ),
                            {},
                        ),
                    )

    def test_raw_completion_prefix_and_format_grades_are_rechecked(self):
        case = self.case()
        text = runner.formats.serialize(case["input"], CORRECTION, "rewrite")
        row = {
            **runner.grade(case, text, format_name="rewrite"),
            "raw_text": text,
            "finish_reason": "stop",
            "response": {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 50},
            },
        }
        prefixes = {case["id"]: [1, 2, 3]}
        runner.validate_results([case], [row], prefixes, "rewrite")
        for field, value in (("prompt_tokens", 4), ("completion_tokens", 513)):
            changed = copy.deepcopy(row)
            changed["response"]["usage"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "512-output budget"),
            ):
                runner.validate_results([case], [changed], prefixes, "rewrite")
        changed = {**row, "displayed": False}
        with self.assertRaisesRegex(ValueError, "format/source grade"):
            runner.validate_results([case], [changed], prefixes, "rewrite")
        failed = {
            "case_id": case["id"],
            "request_failed": True,
            "raw_valid": False,
            "displayed": False,
            "error_type": "timeout",
        }
        runner.validate_results([case], [failed], prefixes, "rewrite")
        runner.validate_results([case], [], prefixes, "rewrite")
        report = runner.summarize([case], [failed])
        self.assertEqual(report["request_errors"], 1)
        self.assertEqual(report["correct_abstentions"], 0)
        self.assertEqual(report["output_tokens"]["missing"], 1)


if __name__ == "__main__":
    unittest.main()
