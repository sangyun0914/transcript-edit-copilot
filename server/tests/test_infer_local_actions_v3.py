import asyncio
import fcntl
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import test_evaluate_local_actions_v3 as fixtures

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import infer_local_actions_v3 as runner  # noqa: E402

core = runner.core


class SequentialSessionTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.LocalEvaluationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.requests = [fixture.request, fixture.make_request("ko")]
        self.requests_path = self.root / "session-requests.jsonl"
        core.write_rows(self.requests_path, self.requests)
        self.config = fixture.config
        self.adapter = core.FrozenRequestAdapter(self.config, self.requests)
        self.info = fixture.info | {
            "version": runner.VERSION,
            "requests": core.deployment.binding(self.requests_path),
            "requested": 2,
            "cache_policy": runner.CACHE_POLICY,
        }
        self.protocol = self.root / "session-protocol.json"
        core.write_new(self.protocol, self.info)
        self.process = SimpleNamespace(pid=123, returncode=None)
        self.process.poll = lambda: self.process.returncode

    def owned_runtime(self):
        def stop(process):
            self.assertIs(process, self.process)
            process.returncode = 0

        stack = ExitStack()
        stack.enter_context(
            patch.object(runner, "check", return_value=(self.info, self.requests, self.adapter))
        )
        stack.enter_context(patch.object(core.lifecycle, "LOCK", self.root / "shared.lock"))
        stack.enter_context(patch.object(core.lifecycle, "refuse_occupied"))
        stack.enter_context(patch.object(core.lifecycle, "wait_model"))
        stack.enter_context(patch.object(core.lifecycle, "stop_owned", side_effect=stop))
        self.popen = stack.enter_context(
            patch.object(runner.subprocess, "Popen", return_value=self.process)
        )
        stack.enter_context(patch.object(self.adapter, "health", return_value={"status": "ready"}))
        stack.enter_context(patch.object(self.adapter, "_json", side_effect=[{}, []]))
        self.tokenizer = stack.enter_context(
            patch.object(
                self.adapter,
                "apply_chat_template",
                side_effect=[r["prefix_ids"] for r in self.requests],
            )
        )
        return stack

    def execute(self, folder):
        return runner.run(
            self.protocol, self.root / folder, protocol_sha256=core.sha256(self.protocol)
        )

    def test_scope_and_protocol_hash_reject_before_request_access(self):
        with patch.object(core.deployment, "bound") as bound:
            with self.assertRaisesRegex(ValueError, "Frozen protocol hash"):
                runner.check(self.protocol, "wrong")
            core.write_state(
                self.protocol, self.info | {"status": "registered_not_inferred", "split": "test"}
            )
            with self.assertRaisesRegex(ValueError, "held-out"):
                runner.check(self.protocol)
        bound.assert_not_called()

    def test_cache_evidence_accepts_observed_partial_counts_without_warm_claim(self):
        for count in (0, 17, 99):
            result = runner.cache_evidence({"runtime_cached_tokens": count}, 100)
            self.assertTrue(result["verified"])
            self.assertEqual(result["status"], "observed_sequential_session")
            self.assertFalse(result["cold_or_warm_claimed"])
        for timing in (
            {},
            {"runtime_cached_tokens": True},
            {"runtime_cached_tokens": 101},
            {"runtime_cached_tokens": 17, "runtime_usage_cached_tokens": 16},
        ):
            self.assertFalse(runner.cache_evidence(timing, 100)["verified"])

    def test_registration_freezes_app_defaults_and_rejects_changed_gateway(self):
        deployment = self.root / "deployment"
        deployment.mkdir()
        core.write_new(deployment / "deployment.json", {})
        receipt = {
            "historical_tooling_only": False,
            "numerical_regime": "native_derived_q4_base",
            "runtime_files_sha256": {},
            "runtime_symlink_targets": {},
        }
        registration = self.root / "registered"
        with patch.object(
            core.deployment, "check", return_value=(self.config, None, ["fixture-server"], receipt)
        ):
            result = runner.register(
                self.requests_path,
                core.sha256(self.requests_path),
                "train",
                "base",
                deployment,
                registration,
            )
            protocol = registration / "protocol.json"
            info, rows, adapter = runner.check(protocol, result["protocol"]["sha256"])
            self.assertEqual(rows, self.requests)
            self.assertFalse(adapter.config.capture_runtime_evidence)
            self.assertTrue(adapter.config.cache_prompt)
            self.assertEqual(info["cache_policy"], runner.CACHE_POLICY)
            info["gateway"]["capture_runtime_evidence"] = True
            core.write_state(protocol, info)
            with self.assertRaisesRegex(ValueError, "Registered source, request, runtime"):
                runner.check(protocol, core.sha256(protocol))

    def test_one_process_two_ordered_requests_capture_actual_app_default_http(self):
        bodies = []

        def handler(request):
            if request.url.path == "/lora-adapters":
                return httpx.Response(200, json=[])
            body = json.loads(request.content)
            bodies.append(body)
            cached = 0 if len(bodies) == 1 else 1
            return httpx.Response(
                200,
                json={
                    "model": self.config.model_id,
                    "choices": [
                        {"message": {"content": '{"action":"no_action"}'}, "finish_reason": "stop"}
                    ],
                    "timings": {"cache_n": cached},
                    "usage": {"prompt_tokens_details": {"cached_tokens": cached}},
                },
            )

        actual_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return actual_client(*args, **kwargs, transport=httpx.MockTransport(handler))

        with (
            self.owned_runtime(),
            patch.object(core.copilot.httpx, "AsyncClient", side_effect=client),
        ):
            result = self.execute("ordered")
        self.assertTrue(result["runtime_preflight_passed"])
        self.popen.assert_called_once()
        self.assertEqual(self.tokenizer.call_count, 2)
        rows = core.deployment.preparation.rows(self.root / "ordered/observations.jsonl")
        self.assertEqual([r["case_id"] for r in rows], [r["case_id"] for r in self.requests])
        self.assertEqual([r["cache_evidence"]["cached_tokens"] for r in rows], [0, 1])
        self.assertTrue(all(r["native_prefix"]["identical"] for r in rows))
        self.assertEqual(
            [body["messages"] for body in bodies], [r["messages"] for r in self.requests]
        )
        for body in bodies:
            self.assertNotIn("verbose", body)
            self.assertNotIn("return_tokens", body)
            self.assertEqual(body["lora"], [])
            self.assertTrue(body["cache_prompt"])
        state = core.read(self.root / "ordered/run.json")
        self.assertEqual(state["model_requests"], 2)
        self.assertEqual(state["planned_priming_requests"], 0)
        self.assertTrue(state["owned_processes_stopped"])

    def test_cancel_preserves_attempt_unrun_case_and_owned_process_exit(self):
        with (
            self.owned_runtime(),
            patch.object(
                self.adapter, "generate", new=AsyncMock(side_effect=asyncio.CancelledError)
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                self.execute("cancelled")
        rows = core.deployment.preparation.rows(self.root / "cancelled/observations.jsonl")
        self.assertEqual([r["status"] for r in rows], ["cancelled", "not_run"])
        self.assertEqual(rows[0]["model_identity"], self.info["model_identity"])
        state = core.read(self.root / "cancelled/run.json")
        self.assertEqual(state["observed"], 2)
        self.assertTrue(state["owned_processes_stopped"])
        self.assertFalse(state["runtime_preflight_passed"])

    def test_prefix_mismatch_preserves_failure_and_never_generates(self):
        with self.owned_runtime():
            self.tokenizer.side_effect = [[999]]
            with patch.object(self.adapter, "generate", new=AsyncMock()) as generate:
                with self.assertRaisesRegex(ValueError, "Runtime/native"):
                    self.execute("prefix-failure")
            generate.assert_not_called()
        rows = core.deployment.preparation.rows(self.root / "prefix-failure/observations.jsonl")
        self.assertEqual([r["status"] for r in rows], ["error", "not_run"])
        self.assertFalse(rows[0]["native_prefix"]["identical"])

    def test_shared_lock_refusal_never_starts_or_stops_another_process(self):
        lock_path = self.root / "shared.lock"
        with lock_path.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(
                    runner, "check", return_value=(self.info, self.requests, self.adapter)
                ),
                patch.object(core.lifecycle, "LOCK", lock_path),
                patch.object(runner.subprocess, "Popen") as popen,
                patch.object(core.lifecycle, "stop_owned") as stop,
            ):
                with self.assertRaises(BlockingIOError):
                    self.execute("busy")
        popen.assert_not_called()
        stop.assert_not_called()
        rows = core.deployment.preparation.rows(self.root / "busy/observations.jsonl")
        self.assertEqual([r["status"] for r in rows], ["not_run", "not_run"])


if __name__ == "__main__":
    unittest.main()
