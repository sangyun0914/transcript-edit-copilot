import asyncio
import copy
import fcntl
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from test_copilot import FakeAdapter, document

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import evaluate_local_actions_v3 as runner  # noqa: E402


class LocalEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.builder = runner.formats.contract.PassiveContextBuilder(FakeAdapter())
        self.request = self.make_request("en")
        self.requests_path = self.root / "requests.jsonl"
        runner.write_rows(self.requests_path, [self.request])
        self.config = runner.copilot.LocalConfig(
            version=2,
            contract_version="passive-nep-actions-v1",
            endpoint="http://127.0.0.1:18091",
            model_id="fixture-base",
            model_path=str(self.root / "base.gguf"),
            model_sha256="a" * 64,
            chat_template_path=str(self.root / "template.jinja"),
            chat_template_sha256="b" * 64,
            runtime_build="fixture-build",
            cache_prompt=True,
        )
        self.info = {
            "version": runner.VERSION,
            "split": "train",
            "model_condition": "base",
            "cache_modes": ["cold", "warm"],
            "timeout_seconds": 5,
            "format": "json",
            "model_identity": {"kind": "base", "adapter_sha256": None, "adapter_scale": None},
            "runtime_sha256": "c" * 64,
            "requests": runner.deployment.binding(self.requests_path),
            "command": ["fixture-owned-server"],
            "numerical_regime": "native_derived_q4_base",
            "schema_constrained": True,
            "final_test_bodies_read": False,
        }
        self.protocol = self.root / "protocol.json"
        runner.write_new(self.protocol, self.info)

    def make_request(self, language, format_name="json"):
        snapshot = self.builder.build(document(language), "w1", as_of=0)["input"]
        messages = runner.formats.messages(snapshot, format_name)
        tokens = [2, 11, 13]
        return {
            "case_id": f"fixture-{language}",
            "split": "train",
            "format": format_name,
            "input": snapshot,
            "input_sha256": runner.conditions.input_hash(snapshot),
            "messages": messages,
            "messages_sha256": runner.value_hash(messages),
            "prefix_ids": tokens,
            "prefix_ids_sha256": runner.value_hash(tokens),
        }

    def adapter(self):
        return runner.FrozenRequestAdapter(self.config, [self.request])

    def test_held_out_scope_rejected_before_any_request_file_access(self):
        with patch.object(runner.deployment, "bound") as bound:
            with self.assertRaisesRegex(ValueError, "TRAIN/DEV"):
                runner.load_requests({"path": "sealed-test.jsonl"}, "test", "base")
        bound.assert_not_called()
        with patch.object(runner, "load_requests") as loaded:
            runner.write_state(
                self.protocol,
                self.info
                | {
                    "version": runner.VERSION,
                    "status": "registered_not_inferred",
                    "split": "test",
                },
            )
            with self.assertRaisesRegex(ValueError, "held-out"):
                runner.check(self.protocol)
        loaded.assert_not_called()

    def test_registered_protocol_hash_is_checked_before_opening_its_references(self):
        with patch.object(runner, "read") as read:
            with self.assertRaisesRegex(ValueError, "Frozen protocol hash"):
                runner.check(self.protocol, "wrong")
        read.assert_not_called()

    def test_request_admission_rejects_labels_wrong_split_prompt_and_prefix_changes(self):
        for update in (
            {"primary": {"action": "no_action"}},
            {"split": "dev"},
            {"messages": self.request["messages"] + [{"role": "user", "content": "extra"}]},
            {"prefix_ids": [2, 99]},
            {"input_sha256": "wrong"},
        ):
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                runner.validate_requests([self.request | update], "train", "base")
        runner.validate_requests([self.request], "train", "base")
        runner.validate_requests([self.request | {"split": "dev"}], "dev", "base")

    def test_icl_is_train_bound_and_only_exact_registered_messages_are_permitted(self):
        demos = []
        for language in ("en", "ko"):
            request = self.make_request(language)
            action = (
                {"action": "no_action"}
                if language == "ko"
                else {
                    "action": "correct_text",
                    "target": [request["input"]["items"][0]["id"]],
                    "text": "corrected fixture text",
                }
            )
            outputs = {
                name: runner.value_hash(runner.formats.serialize(request["input"], action, name))
                for name in runner.formats.FORMATS
            }
            demos.append(
                {
                    "case_id": f"demo-{language}",
                    "split": "train",
                    "language": language,
                    "source_group": f"unit-{language}",
                    "input": request["input"],
                    "input_sha256": request["input_sha256"],
                    "action": action,
                    "outputs_sha256": outputs,
                }
            )
        spec = runner.conditions.condition_specs(demos)["train_icl"]
        messages = runner.conditions.expected_messages(self.request["input"], "json", spec)
        request = self.request | {
            "messages": messages,
            "messages_sha256": runner.value_hash(messages),
            "condition": spec,
            "condition_sha256": runner.value_hash(spec),
        }
        runner.validate_requests([request], "train", "base_icl")
        with self.assertRaisesRegex(ValueError, "Only base_icl"):
            runner.validate_requests([request], "train", "sft")
        adapter = runner.FrozenRequestAdapter(self.config, [request])
        with adapter.request(messages):
            self.assertEqual(adapter._model_messages(messages), messages)
            with self.assertRaisesRegex(ValueError, "Active request"):
                adapter._model_messages(messages[:-1])
        with self.assertRaises(ValueError):
            adapter._model_messages(messages)
        changed = copy.deepcopy(messages)
        changed[2]["content"] = '{"action":"no_action"}'
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            with adapter.request(changed):
                pass

    def test_named_dpo_condition_cannot_use_an_sft_training_receipt(self):
        training_path = self.root / "training.json"
        runner.write_new(training_path, {"version": "gemma-actions-v3-sft"})
        export_config = self.root / "configuration.json"
        runner.write_new(
            export_config, {"training_record": runner.deployment.binding(training_path)}
        )
        export_path = self.root / "export.json"
        runner.write_new(
            export_path,
            {
                "configuration_sha256": runner.sha256(export_config),
                "training_manifest_sha256": runner.sha256(training_path),
                "gguf_sha256": "d" * 64,
            },
        )
        config = self.config.model_copy(
            update={
                "adapter_path": str(self.root / "lora.gguf"),
                "adapter_sha256": "d" * 64,
            }
        )
        receipt = {
            "historical_tooling_only": False,
            "adapter_export_receipt": runner.deployment.binding(export_path),
        }
        with self.assertRaisesRegex(ValueError, "named condition"):
            runner.model_identity(config, receipt, "dpo")

    def test_warm_cache_requires_near_complete_actual_prefix_and_successful_priming(self):
        def timing(count, usage=None):
            return {
                "runtime_cached_tokens": count,
                "runtime_usage_cached_tokens": count if usage is None else usage,
            }

        for evidence, primed in (
            ({}, True),
            (timing(0), True),
            (timing(2), True),
            (timing(99), False),
            (timing(99, 98), True),
        ):
            with self.subTest(evidence=evidence, primed=primed):
                self.assertFalse(runner.cache_evidence(evidence, "warm", 100, primed)["verified"])
        self.assertTrue(runner.cache_evidence(timing(99), "warm", 100, True)["verified"])
        self.assertTrue(runner.cache_evidence(timing(0), "cold", 100, False)["verified"])
        self.assertFalse(runner.cache_evidence(timing(1), "cold", 100, False)["verified"])

    def test_generation_reuses_adapter_default_flags_and_records_non_2xx_raw_failure(self):
        original = runner.copilot.httpx
        adapter = self.adapter()
        bodies = []

        def handler(request):
            if request.url.path == "/lora-adapters":
                return httpx.Response(200, json=[])
            bodies.append(json.loads(request.content))
            return httpx.Response(503, json={"error": "fixture capacity failure"})

        actual_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return actual_client(*args, **kwargs, transport=httpx.MockTransport(handler))

        record = runner.observation(self.request, self.info, "cold")
        with patch.object(original, "AsyncClient", side_effect=client):
            asyncio.run(
                runner.generate(adapter, self.request, self.info, record, self.root / "raw.json")
            )
        self.assertIs(runner.copilot.httpx, original)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["http_records"][-1]["status_code"], 503)
        self.assertEqual(
            record["http_records"][-1]["response"]["error"], "fixture capacity failure"
        )
        self.assertNotIn("verbose", bodies[0])
        self.assertNotIn("return_tokens", bodies[0])
        self.assertTrue(bodies[0]["cache_prompt"])
        self.assertEqual(bodies[0]["lora"], [])

    def test_cancel_preserves_partial_raw_attempt_and_restores_http_hook(self):
        adapter = self.adapter()
        record = runner.observation(self.request, self.info, "cold")
        original = runner.copilot.httpx
        with patch.object(adapter, "generate", new=AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(
                    runner.generate(
                        adapter, self.request, self.info, record, self.root / "cancel.json"
                    )
                )
        self.assertIs(runner.copilot.httpx, original)
        self.assertIsNone(adapter.active_messages)
        self.assertEqual(runner.read(self.root / "cancel.json")["status"], "cancelled")
        self.assertEqual(record["model_identity"], self.info["model_identity"])

    def test_busy_lock_records_all_unrun_denominators_without_touching_processes(self):
        lock_path = self.root / "shared.lock"
        with lock_path.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(
                    runner, "check", return_value=(self.info, [self.request], self.adapter())
                ),
                patch.object(runner.lifecycle, "LOCK", lock_path),
                patch.object(runner.subprocess, "Popen") as popen,
                patch.object(runner.lifecycle, "stop_owned") as stop,
                self.assertRaises(BlockingIOError),
            ):
                runner.run(
                    self.protocol, self.root / "busy", protocol_sha256=runner.sha256(self.protocol)
                )
        popen.assert_not_called()
        stop.assert_not_called()
        state = runner.read(self.root / "busy/run.json")
        self.assertEqual(state["planned_scored_requests"], 2)
        self.assertEqual(state["observed"], 2)
        self.assertFalse(state["runtime_preflight_passed"])
        for mode in ("cold", "warm"):
            rows = runner.deployment.preparation.rows(
                self.root / "busy" / mode / "observations.jsonl"
            )
            self.assertEqual([row["status"] for row in rows], ["not_run"])

    def test_explicit_priming_is_separate_and_failure_prevents_warm_scoring(self):
        adapter = self.adapter()
        process = SimpleNamespace(pid=123, returncode=None)
        process.poll = lambda: process.returncode

        def stop(owned):
            self.assertIs(owned, process)
            owned.returncode = 0

        async def generate(_adapter, _request, _info, record, path, primed=False):
            mode = record["cache_mode"]
            record.update(
                status="completed" if mode == "cold" else "error",
                started_at=runner.now(),
                generation_ms=10,
                valid_output=mode == "cold",
                cache_evidence={"verified": mode == "cold"},
            )
            runner.write_state(path, record)
            return record

        with (
            patch.object(runner, "check", return_value=(self.info, [self.request], adapter)),
            patch.object(runner.lifecycle, "LOCK", self.root / "lock"),
            patch.object(runner.lifecycle, "refuse_occupied"),
            patch.object(runner.lifecycle, "wait_model"),
            patch.object(runner.lifecycle, "stop_owned", side_effect=stop),
            patch.object(runner.subprocess, "Popen", return_value=process),
            patch.object(adapter, "health", return_value={"status": "ready"}),
            patch.object(adapter, "_json", side_effect=[{}, []]),
            patch.object(adapter, "apply_chat_template", return_value=self.request["prefix_ids"]),
            patch.object(runner, "generate", side_effect=generate) as generated,
        ):
            result = runner.run(
                self.protocol,
                self.root / "prime-fail",
                protocol_sha256=runner.sha256(self.protocol),
            )
        self.assertEqual(generated.call_count, 2)
        self.assertFalse(result["runtime_preflight_passed"])
        self.assertEqual(result["by_cache"]["cold"]["completed_latency_ms"]["count"], 1)
        self.assertEqual(result["by_cache"]["warm"]["status_counts"], {"not_run": 1})
        self.assertEqual(result["priming"]["status_counts"], {"error": 1})
        self.assertEqual(
            runner.read(self.root / "prime-fail/cases/00000/startup.json")["process_returncode"], 0
        )

    def test_deterministic_baseline_roundtrip_has_zero_model_requests(self):
        for format_name in runner.formats.FORMATS:
            request = self.make_request("ko", format_name)
            path = self.root / f"{format_name}.jsonl"
            runner.write_rows(path, [request])
            registration = self.root / f"register-{format_name}"
            runner.register(path, runner.sha256(path), "train", "always_no_action", registration)
            with patch.object(runner.subprocess, "Popen") as popen:
                result = runner.run(
                    registration / "protocol.json",
                    self.root / f"baseline-{format_name}",
                    protocol_sha256=runner.sha256(registration / "protocol.json"),
                )
            popen.assert_not_called()
            self.assertTrue(result["runtime_preflight_passed"])
            info = runner.read(self.root / f"baseline-{format_name}/run.json")
            self.assertEqual(info["model_requests"], 0)
            self.assertEqual(result["by_cache"]["not_applicable"]["valid_outputs"], 1)


if __name__ == "__main__":
    unittest.main()
