import asyncio
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "copilot_p6_bridge", Path(__file__).with_name("copilot_p6_bridge.py")
)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_fixture_seed_survives_restart_without_local_evaluation_artifacts(self):
        fixture = {"case_id": "unit-case", "runtime": {"feedback_journal": None}}
        roster = SimpleNamespace(
            manifest_sha256="unit-roster",
            fixture=lambda *_: copy.deepcopy(fixture),
            media=lambda _: [],
            cases={"unit-case": {"runtime": fixture["runtime"], "source": {}}},
            requests={"unit-case": {}},
        )
        adapter = bridge.ScriptedAdapter()
        adapter.arm = Mock()
        with tempfile.TemporaryDirectory() as directory:
            service = bridge.FixtureService(adapter, roster, Mock(), feedback_directory=directory)
            service.seed("unit-case", "unit-document")
            restarted = bridge.FixtureService(adapter, roster, Mock(), feedback_directory=directory)
            self.assertEqual(
                restarted._session("unit-document").feedback.dump(),
                service._session("unit-document").feedback.dump(),
            )
            self.assertEqual(
                restarted.journals["unit-document"][0]["roster_manifest_sha256"],
                "unit-roster",
            )

    def test_invalid_deployment_records_failure_without_starting_a_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root, deployment=root / "invalid-deployment")
            deployment = SimpleNamespace(check=Mock(side_effect=ValueError("Deployment changed.")))
            with patch.dict("sys.modules", {"run_local_copilot_actions_v3": deployment}):
                with patch.object(bridge.subprocess, "Popen") as launch:
                    with self.assertRaisesRegex(ValueError, "Deployment changed"):
                        with bridge.runtime(args):
                            self.fail("Invalid deployment yielded an adapter.")
                    launch.assert_not_called()
            receipt = json.loads(next(root.glob("runtime-*.json")).read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["error_type"], "ValueError")
            self.assertTrue(receipt["model_process_stopped"])

    def test_failed_health_stops_owned_process_and_releases_shared_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root, deployment=root / "deployment")
            process = Mock(pid=77, returncode=None)
            process.poll.side_effect = lambda: process.returncode

            def require(value, message):
                if not value:
                    raise ValueError(message)

            def stop(owned):
                self.assertIs(owned, process)
                owned.returncode = 0

            lifecycle = SimpleNamespace(
                LOCK=root / "runtime.lock",
                refuse_occupied=Mock(),
                wait_model=Mock(),
                stop_owned=Mock(side_effect=stop),
            )
            adapter = SimpleNamespace(health=lambda: {"status": "error"})
            deployment = SimpleNamespace(
                check=lambda _: (
                    SimpleNamespace(
                        version=2, endpoint="http://127.0.0.1:18091", capture_runtime_evidence=False
                    ),
                    adapter,
                    ["unused-model-command"],
                    {"historical_tooling_only": False},
                ),
                require=require,
                lifecycle=lifecycle,
            )
            with patch.dict("sys.modules", {"run_local_copilot_actions_v3": deployment}):
                with patch.object(bridge.subprocess, "Popen", return_value=process):
                    with self.assertRaisesRegex(ValueError, "Runtime identity verification failed"):
                        with bridge.runtime(args):
                            self.fail("Failed health yielded an adapter.")
            lifecycle.stop_owned.assert_called_once_with(process)
            with lifecycle.LOCK.open("a") as lock:
                bridge.fcntl.flock(lock, bridge.fcntl.LOCK_EX | bridge.fcntl.LOCK_NB)
            receipt = json.loads(next(root.glob("runtime-*.json")).read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertTrue(receipt["model_process_stopped"])
            self.assertEqual(receipt["model_returncode"], 0)

    def test_diagnostic_capture_cannot_enter_ui_performance_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root, deployment=root / "deployment")

            def require(value, message):
                if not value:
                    raise ValueError(message)

            deployment = SimpleNamespace(
                check=lambda _: (
                    SimpleNamespace(version=2, capture_runtime_evidence=True),
                    Mock(),
                    [],
                    {"historical_tooling_only": False},
                ),
                require=require,
            )
            with patch.dict("sys.modules", {"run_local_copilot_actions_v3": deployment}):
                with patch.object(bridge.subprocess, "Popen") as launch:
                    with self.assertRaisesRegex(ValueError, "capture_runtime_evidence=false"):
                        with bridge.runtime(args):
                            self.fail("Diagnostic deployment reached performance mode.")
                    launch.assert_not_called()


class RegisteredHydrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (bridge.ROOT / "artifacts/ui/passive-actions-v3/registered-roster-001").is_dir():
            raise unittest.SkipTest("Local registered evaluation roster is not distributed.")
        cls.roster = bridge.RegisteredRoster(
            bridge.ROOT / "artifacts/ui/passive-actions-v3/registered-roster-001",
            "55b4e22f2c27c12ce59baa7f3e21e973b5b3e4cf6f26124f225eacf1c9da9663",
        )

    def service(self, root, case_id, prediction=None):
        expected = self.roster.requests[case_id]

        class NativeFixture(bridge.ScriptedAdapter):
            def apply_chat_template(self, messages, **kwargs):
                return expected["prefix_ids"]

        native = NativeFixture()
        native.prediction = prediction or {"action": "no_action"}
        audit = []
        adapter = bridge.RegisteredAdapter(native, audit.append)
        service = bridge.FixtureService(adapter, self.roster, audit.append, feedback_directory=root)
        return service, native, audit

    @staticmethod
    def request(fixture):
        from app.copilot import PredictRequest

        rt = fixture["runtime"]
        return PredictRequest.model_validate(
            {
                "request": {
                    "document_id": fixture["document_id"],
                    "request_id": "live-request",
                    "revision": 0,
                    "interaction_epoch": 0,
                },
                "document": fixture["session_document"],
                "focus_uuid": rt["focus_uuid"],
                "selection": rt["selection"],
                "history": rt["history"],
                "sequence": fixture["prediction_sequence"],
                "origin": "synthetic",
                "trigger": "explicit",
                "deadline_ms": 30000,
            }
        )

    def test_past_feedback_seed_is_distinct_and_survives_restart_with_suppression(self):
        case_id = "v3-legacy-sessions-v2-museum-dismissed-formality-s00"
        original = self.roster.cases[case_id]["runtime"]["feedback_journal"]
        prediction = original["records"][0]["prediction"]
        with tempfile.TemporaryDirectory() as directory:
            service, _, _ = self.service(directory, case_id)
            seeded = service.seed(case_id, "fixture-document")
            before, _ = service.evidence.read("fixture-document")
            self.assertEqual([r["kind"] for r in before], ["fixture_context_seed"])
            self.assertFalse(before[0]["model_attempted"])
            self.assertNotIn("ledger_record", before[0])
            self.assertEqual(before[0]["original_runtime"]["feedback_journal"], original)
            restarted, native, audit = self.service(directory, case_id, prediction)
            self.assertEqual(
                restarted._session("fixture-document").feedback.dump(),
                service._session("fixture-document").feedback.dump(),
            )
            result = asyncio.run(restarted.predict(self.request(seeded["fixture"])))
            self.assertEqual(result["status"], "suppressed")
            self.assertEqual(native.started, 1)
            self.assertEqual([r["kind"] for r in audit], ["fixture_generation_started"])
            self.assertFalse(audit[0]["model_attempted"])

    def test_history_or_source_mutation_is_rejected_before_any_model_call(self):
        case_id = "v3-legacy-authored-en-oral-history-answer-s01"
        for mutation in ("history", "source"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                service, native, audit = self.service(directory, case_id)
                fixture = service.seed(case_id, "fixture-document")["fixture"]
                original = self.roster.cases[case_id]
                self.assertEqual(
                    fixture["runtime"]["history"][0]["prediction"],
                    original["runtime"]["history"][0]["prediction"],
                )
                self.assertNotEqual(
                    fixture["projection"]["original_document_sha256"],
                    fixture["projection"]["projected_document_sha256"],
                )
                request = self.request(fixture)
                if mutation == "history":
                    request.history = []
                else:
                    next(i for i in request.document.content if "source" in i)["sourceStart"] += (
                        0.01
                    )
                with self.assertRaisesRegex(
                    bridge.GatewayError, "differs from the registered fixture"
                ):
                    asyncio.run(service.predict(request))
                self.assertEqual(native.started, 0)
                self.assertFalse(any(r.get("model_attempted") for r in audit))
                records, _ = service.evidence.read("fixture-document")
                self.assertEqual([r["kind"] for r in records], ["fixture_context_seed"])

    def test_native_prefix_or_prompt_change_never_reaches_generation(self):
        case_id = "v3-legacy-sessions-v2-oral-history-expired-merge-s00"
        for mutation in ("prefix", "prompt"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                service, native, audit = self.service(directory, case_id)
                fixture = service.seed(case_id, "fixture-document")["fixture"]
                if mutation == "prefix":
                    native.apply_chat_template = lambda *a, **kw: [111] * 100
                else:
                    service.roster = copy.copy(self.roster)
                    service.roster.requests = copy.deepcopy(self.roster.requests)
                    service.roster.requests[case_id]["messages"][0]["content"] += " modified"
                with self.assertRaises(bridge.GatewayError):
                    asyncio.run(service.predict(self.request(fixture)))
                self.assertEqual(native.started, 0)
                self.assertFalse(any(r.get("model_attempted") for r in audit))
                records, _ = service.evidence.read("fixture-document")
                self.assertIn("prediction_failed", [r["kind"] for r in records])
                self.assertNotIn("prediction_generated", [r["kind"] for r in records])

    def test_rewrite_counterpart_verifies_native_prefix_and_uses_normal_gateway_validation(self):
        from copilot_p6_roster import formats

        registered = bridge.RegisteredRoster(
            self.roster.directory,
            self.roster.manifest_sha256,
            bridge.ROOT / "artifacts/ui/passive-actions-v3/registered-roster-rewrite-001",
            "637ba353a58f63b9999b6080afd1e7b20ca25395ac9cae96213dec7094721d7d",
        )
        case_id = "v3-legacy-authored-en-oral-history-answer-s01"
        expected = registered.requests[case_id]

        class RewriteFixture(bridge.ScriptedAdapter):
            output_format = "rewrite"

            def request_messages(self, snapshot):
                return formats.messages(snapshot, "rewrite")

            def apply_chat_template(self, messages, **kwargs):
                return expected["prefix_ids"]

            async def generate(self, messages, output_schema, timeout):
                self.started += 1
                self.seen = messages
                return bridge.Completion('{"format_version":1,"scope":[],"after":[]}', "stop")

        native, audit = RewriteFixture(), []
        with tempfile.TemporaryDirectory() as directory:
            service = bridge.FixtureService(
                bridge.RegisteredAdapter(native, audit.append),
                registered,
                audit.append,
                feedback_directory=directory,
            )
            fixture = service.seed(case_id, "fixture-rewrite")["fixture"]
            result = asyncio.run(service.predict(self.request(fixture)))
            self.assertEqual(result["prediction"], {"action": "no_action"})
            self.assertEqual(native.seen, expected["messages"])
            self.assertEqual(native.started, 1)
            with self.assertRaisesRegex(ValueError, "exceeds the selected format"):
                service.seed("v3-legacy-shuka-restore-006-s00", "over-budget")
            self.assertEqual(native.started, 1)


if __name__ == "__main__":
    unittest.main()
