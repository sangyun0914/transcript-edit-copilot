import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_local_copilot.py"
SPEC = importlib.util.spec_from_file_location("local_copilot_launcher", SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


class LocalCopilotLauncherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.deployment, self.execution = self.root / "deployment", self.root / "execution"
        (self.root / "app/node_modules/electron").mkdir(parents=True)
        (self.root / "server/.venv/bin").mkdir(parents=True)
        (self.root / "server/.venv/bin/python").touch()
        model = self.root / "model.gguf"
        model.write_bytes(b"fixture")
        parity = put(self.deployment / "native-token-parity.json", {"fixture": True})
        self.config = {
            "model_id": launcher.MODEL_ID,
            "model_path": str(model),
            "model_sha256": "a" * 64,
            "contract_version": launcher.CONTRACT,
            "chat_template_sha256": "b" * 64,
            "runtime_build": "b10816-427291b5b",
            "token_parity_path": parity["path"],
            "token_parity_sha256": parity["sha256"],
            "endpoint": "http://127.0.0.1:18088",
        }
        arm = {
            "condition": launcher.CONDITION,
            "configuration": put(self.deployment / "gateway.json", self.config),
            "launcher": put(self.deployment / "start-model.sh", "fixture"),
            "parity": parity,
        }
        protocol = put(self.root / "protocol.json", {"requests": 14, "arms": [arm]})
        self.pipeline = {
            "status": "completed",
            "protocol": protocol,
            "arms": [
                {
                    "condition": launcher.CONDITION,
                    "status": "completed",
                    "requests_completed": 14,
                    "process_exit_code": 0,
                }
            ],
        }
        put(self.execution / "pipeline.json", self.pipeline)
        health = {
            "status": "ready",
            "model": {
                "id": self.config["model_id"],
                "sha256": self.config["model_sha256"],
                "training_token_parity": "verified_fixed_bilingual_checks",
                **{
                    key: self.config[key]
                    for key in (
                        "chat_template_sha256",
                        "runtime_build",
                        "contract_version",
                        "token_parity_sha256",
                    )
                },
            },
        }
        put(self.execution / launcher.CONDITION / "health.json", health)
        self.addCleanup(patch.stopall)
        patch.object(launcher, "ROOT", self.root).start()
        patch.object(launcher.shutil, "which", return_value="installed").start()

    def test_uses_exact_completed_runtime_without_quality_threshold(self):
        self.assertEqual(launcher.validate_deployment(self.deployment, self.execution), self.config)
        self.assertFalse((self.execution / "summary.json").exists())
        put(self.deployment / "gateway.json", {**self.config, "contract_version": "passive-nep-v1"})
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            launcher.validate_deployment(self.deployment, self.execution)

    def test_incomplete_or_foreign_evaluation_rejects_before_start(self):
        for key, value in (
            ("status", "running"),
            ("requests_completed", 13),
            ("process_exit_code", 1),
            ("condition", "base"),
        ):
            changed = copy.deepcopy(self.pipeline)
            changed["arms"][0][key] = value
            put(self.execution / "pipeline.json", changed)
            with (
                self.subTest(key=key),
                self.assertRaises(ValueError),
                patch.object(launcher.subprocess, "Popen") as popen,
            ):
                launcher.start(self.deployment, self.execution)
                popen.assert_not_called()

    def test_busy_lock_never_starts_or_stops_any_process(self):
        import fcntl

        lock = self.root / "shared.lock"
        with lock.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(launcher, "LOCK", lock),
                patch.object(launcher.subprocess, "Popen") as popen,
                patch.object(launcher, "stop_owned") as stop,
            ):
                with self.assertRaisesRegex(ValueError, "shared machine lock"):
                    launcher.start(self.deployment, self.execution)
                popen.assert_not_called()
                stop.assert_not_called()

    def test_occupied_endpoint_is_never_replaced(self):
        with (
            patch.object(launcher.socket, "create_connection"),
            patch.object(launcher, "stop_owned") as stop,
        ):
            with self.assertRaisesRegex(ValueError, "occupied"):
                launcher.refuse_occupied(self.config["endpoint"])
            stop.assert_not_called()

    def test_owned_model_is_stopped_if_startup_fails_before_app(self):
        lock = self.root / "shared.lock"
        model = object()
        with (
            patch.object(launcher, "LOCK", lock),
            patch.object(launcher, "refuse_occupied"),
            patch.object(launcher.subprocess, "Popen", return_value=model) as popen,
            patch.object(launcher, "wait_model", side_effect=TimeoutError("startup")),
            patch.object(launcher, "stop_owned") as stop,
        ):
            with self.assertRaises(TimeoutError):
                launcher.start(self.deployment, self.execution)
            popen.assert_called_once()
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(stop.call_args_list[-1].args, (model,))

    def test_stop_terminates_only_its_owned_process_group(self):
        children = [
            subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
            )
            for _ in range(2)
        ]
        try:
            launcher.stop_owned(children[0])
            self.assertIsNotNone(children[0].poll())
            self.assertIsNone(children[1].poll())
        finally:
            for process in children:
                launcher.stop_owned(process)


if __name__ == "__main__":
    unittest.main()
