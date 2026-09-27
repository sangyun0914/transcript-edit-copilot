import fcntl
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "cached_launcher", SCRIPTS / "run_local_copilot_cached.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


class CachedLauncherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.deployment = self.root / "cached"
        self.original = {"model_id": "fixed-model", "endpoint": "http://127.0.0.1:18088"}
        self.config = {**self.original, "cache_prompt": True, "capture_runtime_evidence": False}
        self.model_launcher = self.root / "original/start-model.sh"
        protocol = {
            "source_pins": [],
            "configs": {
                "on": put(
                    self.root / "measured-on.json",
                    {
                        **self.config,
                        "capture_runtime_evidence": True,
                    },
                )
            },
            "bindings": {"launcher": put(self.model_launcher, "fixture")},
        }
        self.report = {
            "status": "completed",
            "selected_cache_prompt": True,
            "normal_requests": 24,
            "all_requests": 28,
            "checks": {str(i): True for i in range(5)},
            "protocol": put(self.root / "protocol.json", protocol),
        }
        self.selection = {
            "version": "local-gemma-measured-cache-v1",
            "measurement": put(self.root / "report.json", self.report),
            "independent_verification": put(self.root / "verification.json", {"passed": True}),
            "configuration": put(self.deployment / "gateway.json", self.config),
        }
        put(self.deployment / "selection.json", self.selection)
        self.addCleanup(patch.stopall)
        patch.object(launcher.base, "ROOT", self.root).start()
        patch.object(launcher.base, "DEPLOYMENT", self.model_launcher.parent).start()
        patch.object(launcher.base, "LOCK", self.root / "shared.lock").start()
        patch.object(launcher.base, "validate_deployment", return_value=self.original).start()
        patch.object(launcher, "REPORT_SHA", self.selection["measurement"]["sha256"]).start()
        patch.object(
            launcher, "VERIFICATION_SHA", self.selection["independent_verification"]["sha256"]
        ).start()

    def test_cached_config_preserves_model_and_rejects_other_changes(self):
        self.assertEqual(launcher.validate_deployment(self.deployment)[0], self.config)
        self.selection["configuration"] = put(
            self.deployment / "gateway.json",
            {
                **self.config,
                "endpoint": "http://127.0.0.1:19000",
            },
        )
        put(self.deployment / "selection.json", self.selection)
        with self.assertRaisesRegex(ValueError, "Only prompt reuse"):
            launcher.validate_deployment(self.deployment)

    def test_failed_cache_decision_prevents_any_process(self):
        self.report["selected_cache_prompt"] = False
        self.selection["measurement"] = put(self.root / "report.json", self.report)
        put(self.deployment / "selection.json", self.selection)
        with (
            patch.object(launcher, "REPORT_SHA", self.selection["measurement"]["sha256"]),
            patch.object(launcher.subprocess, "Popen") as popen,
            self.assertRaisesRegex(ValueError, "did not select"),
        ):
            launcher.start(self.deployment)
        popen.assert_not_called()

    def test_busy_shared_lock_never_starts_or_stops_processes(self):
        with launcher.base.LOCK.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(launcher.subprocess, "Popen") as popen,
                patch.object(launcher.base, "stop_owned") as stop,
                self.assertRaisesRegex(ValueError, "shared machine lock"),
            ):
                launcher.start(self.deployment)
            popen.assert_not_called()
            stop.assert_not_called()

    def test_health_failure_stops_owned_model_before_app_start(self):
        model = object()
        with (
            patch.object(launcher.base, "refuse_occupied"),
            patch.object(launcher.base, "wait_model"),
            patch.object(
                launcher.base, "existing_gateway_health", side_effect=ValueError("health")
            ),
            patch.object(launcher.subprocess, "Popen", return_value=model) as popen,
            patch.object(launcher.base, "stop_owned") as stop,
            self.assertRaisesRegex(ValueError, "health"),
        ):
            launcher.start(self.deployment)
        popen.assert_called_once()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(stop.call_args_list[-1].args, (model,))


if __name__ == "__main__":
    unittest.main()
