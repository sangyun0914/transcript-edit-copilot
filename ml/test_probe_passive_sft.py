import copy
import fcntl
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import httpx

import probe_passive_sft as runner


class PassiveSftProbeTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "case": {"id": f"train-{index}"},
                "prefix": [10, index + 20],
                "messages": [
                    {"role": "system", "content": "policy"},
                    {"role": "user", "content": "visible input"},
                ],
            }
            for index in range(8)
        ]

    def native(self, rows):
        return [
            {
                "case_id": row["case"]["id"],
                "prompt_ids_sha256": runner.prompt_digest(row["prefix"]),
                "output_ids": [100, 3],
                "text": '{"action":"no_action"}',
                "finish_reason": "stop",
            }
            for row in rows
        ]

    def prepared(self):
        return {
            "rows": self.rows(),
            "sampling": {"temperature": 0, "seed": 3407, "max_tokens": 512},
            "server": "/mock/server",
            "models": {"base": "/mock/base.gguf"},
            "template": "/mock/template.jinja",
            "timeout": 1,
            "startup_timeout": 1,
            "eos": 3,
        }

    def test_prediction_requests_never_include_targets_and_grammar_is_separate(self):
        prepared = self.prepared()
        row = prepared["rows"][0]
        row["case"].update(
            expectedOutput={"action": "correct_text", "text": "private answer"},
            _execution={"before": "private source"},
        )
        for constrained in (False, True):
            body = runner.make_body(row, "smoke", prepared, constrained)
            self.assertEqual(body["messages"], row["messages"])
            self.assertEqual(body["ignore_eos"], False)
            self.assertEqual(body["max_tokens"], 512)
            self.assertEqual("response_format" in body, constrained)
            self.assertNotIn("private answer", json.dumps(body))
            self.assertNotIn("private source", json.dumps(body))
        self.assertEqual(
            [condition[2] for condition in runner.CONDITIONS], [False] * 4 + [True]
        )
        self.assertEqual(runner.CONDITIONS[-1][1], runner.CONDITIONS[-2][1])

    def test_native_comparator_requires_exact_eight_prefixes_and_stopping(self):
        rows = self.rows()
        values = self.native(rows)
        runner.validate_native_smoke(values, rows, 3)
        for field, value in (
            ("prompt_ids_sha256", "0" * 64),
            ("finish_reason", "length"),
            ("output_ids", [1] * 513),
            ("case_id", "dev-row"),
        ):
            changed = copy.deepcopy(values)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_native_smoke(changed, rows, 3)
        with self.assertRaises(ValueError):
            runner.validate_native_smoke(values[:-1], rows, 3)

    def exercise_failure(self, mismatch=False, stubborn=False):
        prepared = self.prepared()
        state = {"alive": True, "terminated": False, "killed": False, "completions": 0}

        class Process:
            def poll(self):
                return None if state["alive"] else 0

            def terminate(self):
                state["terminated"] = True
                if not stubborn:
                    state["alive"] = False

            def kill(self):
                state.update(alive=False, killed=True)

            def wait(self, timeout):
                if state["alive"]:
                    raise subprocess.TimeoutExpired("server", timeout)
                return 0

        def response(request):
            if request.url.path == "/props":
                data = {
                    "model_alias": "native-base-f16",
                    "build_info": "b10816-427291b5b",
                    "eos_token": "<eos>",
                }
            elif request.url.path == "/apply-template":
                body = json.loads(request.content)
                self.assertEqual(
                    [message["role"] for message in body["messages"]],
                    ["system", "user"],
                )
                data = {"prompt": "frozen"}
            elif request.url.path == "/tokenize":
                body = json.loads(request.content)
                if body["content"] == "<eos>":
                    data = {"tokens": [3]}
                else:
                    position = state.setdefault("prefix", 0)
                    data = {
                        "tokens": [999]
                        if mismatch
                        else prepared["rows"][position]["prefix"]
                    }
                    state["prefix"] += 1
            elif request.url.path == "/v1/chat/completions":
                state["completions"] += 1
                raise httpx.ReadTimeout("simulated inference timeout", request=request)
            else:
                data = {}
            return httpx.Response(200, json=data)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "condition"
            client = httpx.Client(
                base_url="http://127.0.0.1", transport=httpx.MockTransport(response)
            )
            with (
                patch.object(runner.subprocess, "Popen", return_value=Process()),
                patch.object(runner.httpx, "Client", return_value=client),
                patch.object(runner.socket, "socket") as socket_factory,
            ):
                socket_factory.return_value.__enter__.return_value.getsockname.return_value = (
                    "127.0.0.1",
                    12345,
                )
                with self.assertRaises(ValueError if mismatch else httpx.ReadTimeout):
                    runner.run_condition(prepared, runner.CONDITIONS[0], output)
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertTrue(receipt["server_stopped"])
            self.assertTrue(state["terminated"])
            if stubborn:
                self.assertTrue(state["killed"])
            if mismatch:
                self.assertEqual(state["completions"], 0)
                self.assertEqual(receipt["missing_requests"], 8)
            else:
                self.assertEqual(state["completions"], 1)
                self.assertEqual(receipt["missing_requests"], 7)
                rows = [
                    json.loads(line)
                    for line in (output / "results.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]["request_failed"])
                self.assertIn("ReadTimeout", rows[0]["error"])
            with self.assertRaises(FileExistsError):
                runner.run_condition(prepared, runner.CONDITIONS[0], output)

    def test_prefix_failure_blocks_every_generation_and_releases_server(self):
        self.exercise_failure(mismatch=True)

    def test_timeout_retains_failed_row_and_force_kills_a_stubborn_server(self):
        self.exercise_failure(stubborn=True)

    def test_lock_prevents_even_preparation_while_another_predictor_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            with (root / "artifacts/nep-passive-b1.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (
                    patch.object(runner, "ROOT", root),
                    patch.object(runner, "run") as run,
                    patch("sys.stderr"),
                ):
                    with self.assertRaises(SystemExit):
                        runner.main(
                            [
                                "--config",
                                "unused.json",
                                "--output",
                                str(root / "output"),
                            ]
                        )
                run.assert_not_called()

    def test_comparison_preserves_missing_conditions_and_reports_quantized_changes(
        self,
    ):
        rows = self.rows()
        native = self.native(rows)

        def graded(case, text, finish):
            return {
                "case_id": case["id"],
                "format_valid": True,
                "contract_valid": True,
                "raw_valid": None,
                "raw_grade": None,
                "execution_available": False,
                "prediction": json.loads(text),
            }

        different = {
            **graded(
                rows[0]["case"], '{"action":"split_paragraph","target":[1]}', "stop"
            ),
            "raw_text": '{"action":"split_paragraph","target":[1]}',
        }
        with patch.object(runner, "grade", side_effect=graded):
            result = runner.comparison(rows, native, {"adapted-f16": [different]})
        self.assertEqual(result["counts"]["adapted-f16"]["missing"], 7)
        self.assertEqual(result["counts"]["adapted-q4-k-m"]["missing"], 8)
        self.assertEqual(result["counts"]["adapted-f16"]["execution_unavailable"], 1)
        comparison = result["pairs"][0]["paired_observations"][0]
        self.assertFalse(comparison["same_text"])
        self.assertFalse(comparison["same_valid_prediction"])
        self.assertIsNone(comparison["after_outcome_match"])


if __name__ == "__main__":
    unittest.main()
