import asyncio
import fcntl
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import copilot_runtime as runtime
from app.copilot import (
    CopilotService,
    DisabledAdapter,
    LocalConfig,
    LocalLlamaAdapter,
    adapter_from_environment,
)


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.deployment = self.root / "deployment"
        self.deployment.mkdir()
        self.receipt = {"historical_tooling_only": False, "purpose": "ui_performance"}
        self.receipt_path = self.deployment / "deployment.json"
        self.receipt_path.write_text(json.dumps(self.receipt))
        self.config = LocalConfig(
            version=2,
            contract_version="passive-nep-actions-v1",
            endpoint="http://127.0.0.1:18091",
            model_id="small-test-fixture",
            model_path=str(self.root / "fixture.gguf"),
            model_sha256="a" * 64,
            chat_template_path=str(self.root / "fixture.jinja"),
            chat_template_sha256="b" * 64,
            runtime_build="test-only",
        )
        self.gateway = self.deployment / "gateway.json"
        self.gateway.write_text(self.config.model_dump_json())
        self.selection = self.root / "selection.json"
        self.selection.write_text(
            json.dumps(
                {
                    "version": 1,
                    "deployment": "deployment",
                    "deployment_sha256": runtime.digest(self.receipt_path),
                }
            )
        )
        self.addCleanup(patch.stopall)
        patch.object(runtime, "ROOT", self.root).start()
        self.environ = {
            "AUDAPOLIS_NEP_AUTOSTART": "1",
            "AUDAPOLIS_NEP_DEFAULT_DEPLOYMENT": str(self.selection),
        }
        patch.dict(os.environ, self.environ, clear=True).start()
        self.popen = patch.object(runtime.subprocess, "Popen").start()


class DefaultRuntimeTests(RuntimeFixture):
    def test_default_is_disabled_in_tests_without_an_explicit_startup_test_opt_in(self):
        self.assertIsInstance(adapter_from_environment(), DisabledAdapter)
        self.popen.assert_not_called()

    def test_selected_default_constructs_a_waiting_adapter_without_starting_a_process(self):
        with patch.dict(os.environ, {"AUDAPOLIS_NEP_AUTOSTART_TEST": "1"}):
            adapter = adapter_from_environment()
        self.assertIsInstance(adapter, runtime.ManagedLocalAdapter)
        self.assertEqual(adapter.health()["status"], "starting")
        self.assertEqual(adapter.config, self.config)
        self.popen.assert_not_called()

    def test_external_gateway_takes_precedence_without_claiming_process_ownership(self):
        with (
            patch.dict(os.environ, {"AUDAPOLIS_NEP_CONFIG": str(self.gateway)}),
            patch.object(runtime, "default_adapter") as default,
        ):
            adapter = adapter_from_environment()
        self.assertIs(type(adapter), LocalLlamaAdapter)
        default.assert_not_called()
        self.popen.assert_not_called()

    def test_missing_or_modified_selection_keeps_the_editor_gateway_available(self):
        for change in [lambda: self.receipt_path.write_text("changed"), self.selection.unlink]:
            with (
                self.subTest(change=change),
                patch.dict(os.environ, {"AUDAPOLIS_NEP_AUTOSTART_TEST": "1"}),
            ):
                change()
                adapter = adapter_from_environment()
                self.assertIsInstance(adapter, DisabledAdapter)
                self.assertIn("unavailable", adapter.health()["message"])
        self.popen.assert_not_called()

    def test_selection_cannot_point_outside_the_workspace(self):
        value = json.loads(self.selection.read_text())
        value["deployment"] = "../outside"
        self.selection.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "workspace"):
            runtime.selected_deployment(self.selection)
        self.popen.assert_not_called()


class OwnedRuntimeTests(RuntimeFixture):
    def setUp(self):
        super().setUp()
        self.model = Mock(pid=12345)
        self.model.returncode = None
        self.model.poll.side_effect = lambda: self.model.returncode
        self.popen.return_value = self.model

        def stopped(model):
            model.returncode = -15

        self.lifecycle = SimpleNamespace(
            LOCK=self.root / "shared-model.lock",
            refuse_occupied=Mock(),
            wait_model=Mock(),
            stop_owned=Mock(side_effect=stopped),
        )
        self.launcher = SimpleNamespace(
            fcntl=fcntl,
            lifecycle=self.lifecycle,
            check=Mock(
                return_value=(
                    self.config,
                    SimpleNamespace(_verified_files=None),
                    ["fixture-model-command"],
                    self.receipt,
                )
            ),
        )
        patch.object(runtime, "load_launcher", return_value=self.launcher).start()
        self.health = patch.object(
            LocalLlamaAdapter, "health", return_value={"status": "ready", "model": {}}
        ).start()
        self.adapter = runtime.ManagedLocalAdapter(
            self.config, self.deployment, runtime.digest(self.receipt_path)
        )
        self.addCleanup(self.adapter.stop)

    def wait_status(self, status):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.adapter._readiness[0] == status:
                return
            time.sleep(0.005)
        self.fail(f"Expected {status}, got {self.adapter._readiness}")

    def test_gateway_exposes_startup_readiness_without_waiting_for_model_loading(self):
        service = CopilotService(adapter=self.adapter, feedback_directory=self.root / "feedback")
        health = asyncio.run(service.health())
        self.assertEqual(health["status"], "starting")
        self.assertEqual(health["process_lifecycle"], "app_owned")
        self.assertEqual(health["contract_version"], "passive-nep-actions-v1")
        self.assertIn("Checking", health["message"])
        self.popen.assert_not_called()

    def test_ready_session_starts_once_holds_lock_and_releases_only_its_owned_model(self):
        self.adapter.start()
        self.wait_status("ready")
        self.adapter.start()
        self.popen.assert_called_once()
        self.assertTrue(self.popen.call_args.kwargs["start_new_session"])
        self.health.assert_called_once()
        with self.lifecycle.LOCK.open("a") as second:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.adapter.stop()
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lifecycle.stop_owned.assert_called_once_with(self.model)
        self.assertEqual(self.adapter.health()["status"], "unavailable")

    def test_busy_lock_does_not_check_start_or_stop_another_session(self):
        with self.lifecycle.LOCK.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.adapter._run()
        self.launcher.check.assert_not_called()
        self.popen.assert_not_called()
        self.lifecycle.stop_owned.assert_not_called()
        self.assertIn("shared model lock", self.adapter.health()["message"])

    def test_occupied_port_or_invalid_deployment_never_starts_or_stops_a_process(self):
        for target, message in [
            (self.launcher.check, "Pinned deployment changed"),
            (self.lifecycle.refuse_occupied, "Model endpoint occupied"),
        ]:
            with self.subTest(message=message):
                target.side_effect = ValueError(message)
                self.adapter._run()
                target.side_effect = None
                self.assertIn(message, self.adapter.health()["message"])
        self.popen.assert_not_called()
        self.lifecycle.stop_owned.assert_not_called()

    def test_failed_loaded_adapter_verification_cleans_up_the_owned_model_and_lock(self):
        self.health.side_effect = ValueError("Loaded adapter differs")
        self.adapter._run()
        self.lifecycle.stop_owned.assert_called_once_with(self.model)
        self.assertIn("Loaded adapter differs", self.adapter.health()["message"])
        with self.lifecycle.LOCK.open("a") as second:
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_shutdown_during_preflight_cannot_spawn_a_model_after_the_app_has_closed(self):
        entered, finish = threading.Event(), threading.Event()
        result = self.launcher.check.return_value

        def check(_deployment):
            entered.set()
            finish.wait(timeout=2)
            return result

        self.launcher.check.side_effect = check
        self.adapter.start()
        self.assertTrue(entered.wait(timeout=2))
        threading.Timer(0.05, finish.set).start()
        self.adapter.stop()
        self.assertFalse(self.adapter._thread.is_alive())
        self.popen.assert_not_called()
        self.lifecycle.stop_owned.assert_not_called()

    def test_an_exited_model_becomes_unavailable_without_restarting_or_falling_back(self):
        self.adapter.start()
        self.wait_status("ready")
        self.model.returncode = 1
        self.wait_status("unavailable")
        self.assertIn("Runtime log:", self.adapter.health()["message"])
        self.adapter.stop()
        self.popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
