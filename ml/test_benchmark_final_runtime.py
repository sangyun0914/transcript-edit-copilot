"""Synthetic metadata and HTTP fixtures only; no model process or real dataset bodies."""

import contextlib
import copy
import io
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import benchmark_final_runtime as benchmark
from test_passive_actions import builder, document


class FinalRuntimeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshots = {
            language: builder().build(document(language), "b", as_of=0)["input"]
            for language in ("en", "ko")
        }
        self.rows = [
            {
                "id": f"engineering-{language}-{action}-{index}",
                "split": "train",
                "language": language,
                "primary": {"action": action},
                "input": self.snapshots[language],
            }
            for language in ("en", "ko")
            for action in benchmark.contract.ALLOWED_ACTIONS
            for index in range(4)
        ]
        self.train, self.live = self.root / "train.jsonl", self.root / "live.json"
        self.write_train(self.rows)
        self.live.write_text(json.dumps({"input": self.snapshots["ko"]}))

    def write_train(self, rows):
        self.train.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_hash_roster_is_stratified_order_independent_and_train_only(self):
        selected = benchmark.roster(self.train, self.live, 3)
        self.assertEqual(len(selected), 43)
        self.assertEqual(len({row["id"] for row in selected}), 43)
        cells = {(row["language"], row["reference_action"]) for row in selected[:-1]}
        self.assertEqual(len(cells), 14)
        self.assertTrue(
            all(
                sum(
                    (row["language"], row["reference_action"]) == cell
                    for row in selected[:-1]
                )
                == 3
                for cell in cells
            )
        )
        self.write_train(list(reversed(self.rows)))
        self.assertEqual(benchmark.roster(self.train, self.live, 3), selected)
        self.assertEqual(len(benchmark.roster(self.train, self.live)), 15)
        for invalid in (
            self.rows + [self.rows[0]],
            [{**self.rows[0], "split": "dev"}, *self.rows[1:]],
            [row for row in self.rows if row["primary"]["action"] != "no_action"],
            self.rows[1:3] + self.rows[4:],
        ):
            self.write_train(invalid)
            with self.assertRaises(ValueError):
                benchmark.roster(self.train, self.live, 3)

    def test_sampling_profiles_and_legacy_defaults_are_explicit(self):
        args = benchmark.arguments(
            ["--model", "fixture", "--gguf", "fixture", "--output", "fixture"]
        )
        self.assertEqual(
            (args.per_cell, args.repeats, args.seed, args.sampling_profile),
            (1, 1, 42, "greedy"),
        )
        expected = {
            "greedy": (0, 40, 0.95, 0.05, 1.0),
            "lfm_vendor": (0.1, 50, 1.0, 0.0, 1.05),
            "lfm_unsloth": (0.1, 50, 0.1, 0.0, 1.05),
            "gemma_vendor": (1.0, 64, 0.95, 0.0, 1.0),
        }
        for profile, values in expected.items():
            settings = benchmark.settings("fixture", profile, 43)
            self.assertEqual(
                tuple(
                    settings[key]
                    for key in (
                        "temperature",
                        "top_k",
                        "top_p",
                        "min_p",
                        "repeat_penalty",
                    )
                ),
                values,
            )
            self.assertEqual(settings["seed"], 43)
            self.assertFalse(settings["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(settings["max_tokens"], 512)

    def response(self, text='{"action":"no_action"}', elapsed=10):
        raw = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
        return {
            "http_status": 200,
            "elapsed_ms": elapsed,
            "raw": raw,
            "raw_text": json.dumps(raw),
        }

    def test_warmup_failures_raw_outputs_and_latency_classes_are_not_dropped(self):
        row = benchmark.roster(self.train, self.live)[0]
        output = self.root / "output"
        output.mkdir()
        replies = [
            {
                "http_status": 503,
                "elapsed_ms": 12,
                "raw_text": '{"error":"busy"}',
                "error": "HTTP 503",
                "error_type": "http_error",
            },
            self.response('{"action":"delete_span","target":[1]}', 20),
            self.response(),
            self.response(elapsed=30),
            self.response("not JSON", 40),
            self.response(elapsed=50),
        ]
        with (
            patch.object(benchmark, "request", side_effect=replies) as request,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results = benchmark.run_cases(
                "fixture",
                [row],
                benchmark.settings("fixture"),
                ["identical_repeat", "cache_off"],
                2,
                output,
            )
        self.assertEqual(len(results), 6)
        self.assertEqual(
            [row["outcome"] for row in results],
            [
                "request_error",
                "proposal",
                "no_action",
                "no_action",
                "validation_error",
                "no_action",
            ],
        )
        self.assertFalse(results[1]["condition_prerequisites_met"])
        self.assertIn("raw_text", results[0])
        self.assertIn("raw", results[4])
        self.assertEqual(
            [
                json.loads(line)
                for line in (output / "results.jsonl").read_text().splitlines()
            ],
            results,
        )
        bodies = [call.args[2] for call in request.call_args_list]
        self.assertEqual(bodies[0], bodies[1])
        self.assertFalse(bodies[-1]["cache_prompt"])
        summary = benchmark.summarize(results, ["identical_repeat", "cache_off"])
        self.assertEqual(summary["identical_repeat"]["n"], 2)
        self.assertEqual(
            summary["identical_repeat"]["warmups"]["outcomes"]["request_error"], 1
        )
        self.assertEqual(
            summary["identical_repeat"]["eligible_latency"]["proposal"]["n"], 0
        )
        self.assertEqual(
            summary["identical_repeat"]["eligible_latency"]["no_action"]["p50_ms"], 30
        )
        self.assertEqual(
            summary["cache_off"]["eligible_latency"]["validation_error"]["p50_ms"], 40
        )

    def test_http_error_body_and_invalid_json_are_retained(self):
        failure = urllib.error.HTTPError(
            "http://fixture/route", 503, "busy", {}, io.BytesIO(b'{"error":"busy"}')
        )
        with patch.object(benchmark.urllib.request, "urlopen", side_effect=failure):
            result = benchmark.request("http://fixture", "/route")
        self.assertEqual(
            (result["http_status"], result["raw"]), (503, {"error": "busy"})
        )
        self.assertEqual(result["raw_text"], '{"error":"busy"}')
        response = io.BytesIO(b"not json")
        response.status = 200
        with patch.object(benchmark.urllib.request, "urlopen", return_value=response):
            result = benchmark.request("http://fixture", "/route")
        self.assertEqual(result["error_type"], "invalid_response_json")
        self.assertEqual(result["raw_text"], "not json")

        class Partial(io.BytesIO):
            status = 200

            def read(self, *args):
                raise http.client.IncompleteRead(b'{"partial":', 20)

        with patch.object(benchmark.urllib.request, "urlopen", return_value=Partial()):
            result = benchmark.request("http://fixture", "/route")
        self.assertEqual(result["error_type"], "IncompleteRead")
        self.assertEqual(result["raw_text"], '{"partial":')

    def test_separate_lora_requires_exact_loaded_identity_and_preserves_late_failure(
        self,
    ):
        gguf, adapter, gateway = (
            self.root / name for name in ("base.gguf", "lora.gguf", "gateway.json")
        )
        gguf.write_bytes(b"synthetic base")
        adapter.write_bytes(b"synthetic adapter")
        benchmark.write(
            gateway,
            {
                "version": 2,
                "model_path": str(gguf),
                "model_sha256": benchmark.sha(gguf),
                "adapter_path": str(adapter),
                "adapter_sha256": benchmark.sha(adapter),
                "adapter_scale": 1.0,
            },
        )
        loaded = [
            {"id": 0, "path": str(adapter), "scale": 1.0, "alora_invocation_tokens": []}
        ]
        props = {"model_alias": "gemma", "model_path": str(gguf), "total_slots": 1}
        for scenario in (
            "correct",
            "missing",
            "wrong_path",
            "wrong_scale",
            "alora",
            "late_change",
        ):
            with self.subTest(scenario=scenario):
                output = self.root / scenario
                bad = copy.deepcopy(loaded)
                if scenario == "missing":
                    bad = []
                elif scenario == "wrong_path":
                    bad[0]["path"] = str(self.root / "other.gguf")
                elif scenario == "wrong_scale":
                    bad[0]["scale"] = 0.0
                elif scenario == "alora":
                    bad[0]["alora_invocation_tokens"] = [42]
                calls, lora_reads = [], []

                def request(endpoint, route, body=None, **kwargs):
                    if route == "/health":
                        return {"raw": {"status": "ok"}}
                    if route == "/props":
                        return {"raw": props}
                    if route == "/lora-adapters":
                        lora_reads.append(route)
                        value = (
                            []
                            if scenario == "late_change" and len(lora_reads) == 2
                            else bad
                        )
                        return {"raw": value}
                    calls.append(copy.deepcopy(body))
                    return self.response()

                with (
                    patch.object(benchmark, "request", side_effect=request),
                    patch.object(benchmark, "host_context", return_value={}),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    argv = [
                        "--model",
                        "gemma",
                        "--gguf",
                        str(gguf),
                        "--lora-gateway",
                        str(gateway),
                        "--train",
                        str(self.train),
                        "--live",
                        str(self.live),
                        "--output",
                        str(output),
                        "--conditions",
                        "cache_off",
                    ]
                    if scenario == "correct":
                        benchmark.main(argv)
                    else:
                        with self.assertRaises(SystemExit):
                            benchmark.main(argv)
                protocol = json.loads((output / "protocol.json").read_text())
                expected = (
                    "completed"
                    if scenario == "correct"
                    else "runtime_identity_changed"
                    if scenario == "late_change"
                    else "preflight_failed"
                )
                self.assertEqual(protocol["status"], expected)
                if scenario in ("correct", "late_change"):
                    self.assertEqual(len(calls), 15)
                    self.assertTrue(
                        all(body["lora"] == [{"id": 0, "scale": 1.0}] for body in calls)
                    )
                    self.assertTrue((output / "results.jsonl").is_file())
                else:
                    self.assertEqual(calls, [])
        adapter.write_bytes(b"changed adapter")
        with self.assertRaises(ValueError):
            benchmark.lora_binding(gateway, gguf)

    def test_main_records_native_props_code_hashes_and_same_roster_for_seeds(self):
        gguf = self.root / "fixture.gguf"
        gguf.write_bytes(b"engineering fixture, no model weights")
        props = {
            "model_alias": "fixture",
            "model_path": str(gguf),
            "total_slots": 1,
            "build_info": "engineering-no-runtime",
            "n_batch": 2048,
            "n_ubatch": 512,
        }
        requests = []

        def request(endpoint, route, body=None, **kwargs):
            if route == "/health":
                return {"http_status": 200, "elapsed_ms": 1, "raw": {"status": "ok"}}
            if route == "/props":
                return {"http_status": 200, "elapsed_ms": 1, "raw": props}
            requests.append(copy.deepcopy(body))
            return self.response()

        protocols = []
        for seed in (42, 43):
            output = self.root / str(seed)
            with (
                patch.object(benchmark, "request", side_effect=request),
                patch.object(benchmark, "host_context", return_value={"fixture": True}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                benchmark.main(
                    [
                        "--model",
                        "fixture",
                        "--gguf",
                        str(gguf),
                        "--train",
                        str(self.train),
                        "--live",
                        str(self.live),
                        "--output",
                        str(output),
                        "--per-cell",
                        "3",
                        "--seed",
                        str(seed),
                        "--sampling-profile",
                        "lfm_vendor",
                        "--conditions",
                        "cache_off",
                    ]
                )
            protocol = json.loads((output / "protocol.json").read_text())
            protocols.append(protocol)
            self.assertEqual(protocol["status"], "completed")
            self.assertEqual(protocol["runtime_before"]["props"]["raw"], props)
            self.assertEqual(protocol["settings"]["seed"], seed)
            self.assertEqual(protocol["scope"], benchmark.SCOPE)
            self.assertIn(
                str(Path(benchmark.__file__).resolve()), protocol["source_sha256"]
            )
            summary = json.loads((output / "summary.json").read_text())["cache_off"]
            self.assertEqual(summary["groups"]["train_action_language"]["n"], 42)
            self.assertEqual(summary["groups"]["live_diagnostic"]["n"], 1)
        self.assertEqual(protocols[0]["roster"], protocols[1]["roster"])
        self.assertEqual(len(requests), 86)
        self.assertTrue(
            all(body["min_p"] == 0 and not body["cache_prompt"] for body in requests)
        )
        props["model_alias"] = "wrong-model"
        output = self.root / "wrong-model"
        with (
            patch.object(benchmark, "request", side_effect=request),
            patch.object(benchmark, "host_context", return_value={"fixture": True}),
            self.assertRaises(SystemExit),
        ):
            benchmark.main(
                [
                    "--model",
                    "fixture",
                    "--gguf",
                    str(gguf),
                    "--train",
                    str(self.train),
                    "--live",
                    str(self.live),
                    "--output",
                    str(output),
                ]
            )
        self.assertEqual(len(requests), 86)
        protocol = json.loads((output / "protocol.json").read_text())
        self.assertEqual(protocol["status"], "preflight_failed")
        self.assertIn("model_alias_mismatch", protocol["preflight_errors"])


if __name__ == "__main__":
    unittest.main()
