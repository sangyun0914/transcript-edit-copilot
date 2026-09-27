import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_passive_gemma_smoke as runner


def rows_and_native():
    rows = [{"case": {"id": f"case-{i}"}, "prefix": [2, i]} for i in range(8)]
    native = [
        {
            "case_id": row["case"]["id"],
            "prompt_ids_sha256": runner.prompt_digest(row["prefix"]),
            "text": '{"action":"no_action"}',
            "text_raw": '{"action":"no_action"}<eos>',
            "output_ids": [9, 1],
            "terminal_token_id": 1,
            "finish_reason": "stop",
        }
        for row in rows
    ]
    return rows, native


class GemmaSmokeTests(unittest.TestCase):
    def test_three_native_stops_keep_actual_ids_and_eight_exact_prefixes(self):
        rows, native = rows_and_native()
        for terminal in (1, 106, 50):
            native[0].update(output_ids=[9, terminal], terminal_token_id=terminal)
            runner.validate_native_smoke(native, rows)
        changed = copy.deepcopy(native)
        changed[0]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "terminal behavior"):
            runner.validate_native_smoke(changed, rows)
        changed = copy.deepcopy(native)
        changed[0]["prompt_ids_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "registered prefix"):
            runner.validate_native_smoke(changed, rows)

    def test_real_frozen_train_only_input_and_native_shapes_are_supported(self):
        directory = runner.ROOT / "artifacts/training/passive-t1-gemma26-smoke-v1"
        config = directory / "input/configuration-attempt-01.json"
        native = directory / "native-acquisition/native"
        if not config.is_file() or not native.is_dir():
            self.skipTest("Exact local Gemma smoke input not staged.")
        training, _, rows, template = runner.load_inputs(config, native)
        self.assertEqual(training["phase"], "smoke")
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [len(row["prefix"]) for row in rows],
            [3373, 3130, 3582, 3178, 3197, 3136, 3574, 986],
        )
        self.assertEqual(
            runner.sha256(template), training["tokenizer"]["template_sha256"]
        )

    def test_diagnostic_runner_is_isolated_and_sampling_is_unchanged(self):
        original = runner.shared.make_body
        isolated = runner.isolated_runner()
        self.assertIs(runner.shared.make_body, original)
        self.assertIsNot(isolated.make_body, original)
        self.assertEqual(len(runner.shared.CONDITIONS), 5)
        self.assertEqual(len(runner.CONDITIONS), 3)
        body = {
            "temperature": 0,
            "max_tokens": 512,
            "ignore_eos": False,
            "response_format": {"type": "json_schema"},
        }
        with patch.object(runner.shared, "make_body", return_value=body):
            actual = isolated.make_body({}, "model", {}, True)
        self.assertEqual(actual, {**body, "verbose": True, "return_tokens": True})
        self.assertEqual(body["max_tokens"], 512)

    def test_raw_token_failure_and_tool_handoff_never_receive_edit_credit(self):
        value = {
            "case_id": "case-0",
            "raw_valid": True,
            "displayed": True,
            "contract_valid": True,
            "raw_grade": {"outcome_match": True},
            "response": {
                "usage": {"completion_tokens": 2},
                "__verbose": {
                    "tokens": [9, 50],
                    "content": "{}<|tool_response>",
                    "stop_type": "eos",
                },
            },
        }
        qualified = runner.qualify_local(value)
        self.assertFalse(qualified["raw_valid"])
        self.assertFalse(qualified["displayed"])
        self.assertEqual(qualified["terminal_token_id"], 50)
        self.assertEqual(qualified["output_ids"], [9, 50])
        self.assertEqual(qualified["error_type"], "tool_handoff")
        self.assertTrue(value["raw_valid"])
        missing = runner.qualify_local({"case_id": "case-0", "response": {}})
        self.assertTrue(missing["request_failed"])
        self.assertFalse(missing["raw_valid"])
        self.assertIn("token_evidence_error", missing)
        failed = runner.qualify_local({**value, "request_failed": True})
        self.assertFalse(failed["displayed"])

    def test_failed_serving_attempt_keeps_manifest_and_cannot_be_reused(self):
        rows, native = rows_and_native()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "serving.json"
            config.write_text("{}")
            prepared = {
                "config": {},
                "rows": rows,
                "native": native,
                "models": {},
                "sampling": {},
                "eos": 1,
                "timeout": 120,
                "startup_timeout": 180,
                "execution_unavailable": {},
                "execution_files_sha256": {},
                "protected_directories": [root / "input"],
            }
            output = root / "new-run"

            def failure(prepared, condition, directory):
                directory.mkdir()
                (directory / "results.jsonl").write_text(
                    json.dumps({"case_id": "case-0", "request_failed": True}) + "\n"
                )
                raise TimeoutError("simulated timeout after cleanup")

            with (
                patch.object(runner, "prepare", return_value=prepared),
                patch.object(runner, "run_condition", side_effect=failure) as run,
                patch.object(
                    runner,
                    "grade",
                    return_value={
                        "raw_valid": False,
                        "displayed": False,
                        "contract_valid": False,
                    },
                ),
            ):
                with self.assertRaises(TimeoutError):
                    runner.run(config, output)
                summary = json.loads((output / "summary.json").read_text())
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(
                    summary["counts"]["native-base-q4-k-m"]["request_failures"], 1
                )
                self.assertEqual(summary["counts"]["adapted-q4-k-m"]["missing"], 8)
                self.assertIn("source-execution.jsonl", manifest["files_sha256"])
                with self.assertRaises(FileExistsError):
                    runner.run(config, output)
                self.assertEqual(run.call_count, 1)

    def test_shared_predictor_lock_is_checked_before_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts").mkdir()
            with (root / "artifacts/nep-passive-b1.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    patch.object(runner, "ROOT", root),
                    patch.object(runner, "run") as run,
                ):
                    with self.assertRaises(SystemExit):
                        runner.main(
                            [
                                "--config",
                                str(root / "config.json"),
                                "--output",
                                str(root / "new"),
                            ]
                        )
                    run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
