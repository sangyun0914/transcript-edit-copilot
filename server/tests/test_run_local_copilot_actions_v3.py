import fcntl
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "actions_v3_launcher", SCRIPTS / "run_local_copilot_actions_v3.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class ActionsPreparationTests(unittest.TestCase):
    def test_ui_performance_option_freezes_capture_setting_without_starting_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return path

            model = write("model.gguf", "small base fixture")
            template = write("template.jinja", "template fixture")
            adapter = write("adapter.gguf", "small adapter fixture")
            bundle = write("bundle/manifest.json", {}).parent
            for name in (
                "server/scripts/run_local_copilot.py",
                "server/app/copilot.py",
                "server/app/copilot_runtime.py",
                "server/app/copilot_store.py",
            ):
                write(name, "source fixture")
            base_receipt = write(
                "base-receipt.json",
                {
                    "base_repository": launcher.REPOSITORY,
                    "base_revision": launcher.REVISION,
                    "artifacts": {"base_quantized": {"sha256": launcher.digest(model)}},
                },
            )
            revision = "427291b5b34cd914a31b3fd3b61a68f6184f4b9f"
            registry = write(
                "registry.json",
                {
                    "candidates": [
                        {
                            "id": launcher.BASE_ID,
                            "base_repository": launcher.REPOSITORY,
                            "base_revision": launcher.REVISION,
                            "export_receipt": launcher.binding(base_receipt),
                            "model_path": "model.gguf",
                            "model_sha256": launcher.digest(model),
                            "template_path": "template.jinja",
                            "template_sha256": launcher.digest(template),
                        }
                    ],
                    "runtime": {
                        "revision": revision,
                        "tag": "test",
                        "server": "server-bin",
                        "files_sha256": {},
                    },
                },
            )
            export = write(
                "adapter-receipt.json",
                {
                    "version": "gemma-actions-v3-lora-export",
                    "status": "adapter_conversion_verified_serving_pending",
                    "base_repository": launcher.REPOSITORY,
                    "base_revision": launcher.REVISION,
                    "runtime_revision": revision,
                    "all_tensors_exact": True,
                    "adapter_tensor_count": 410,
                    "contract_version": launcher.formats.contract.VERSION,
                    "format_version": launcher.formats.VERSION,
                    "output_format": "json",
                    "gguf_sha256": launcher.digest(adapter),
                },
            )
            records = [
                {
                    "id": language,
                    "messages": [
                        {"content": "system"},
                        {"content": json.dumps({"language": language})},
                    ],
                    "primary": {"input_ids": [1, 2], "completion_start": 1},
                }
                for language in ("en", "ko")
            ]
            with (
                patch.object(launcher, "ROOT", root),
                patch.object(
                    launcher.preparation,
                    "verify_bundle",
                    return_value={"tokens": {"json": records}},
                ),
                patch.object(launcher.preparation, "source_closure", return_value={}),
                patch.object(
                    launcher.formats.contract,
                    "parse_snapshot",
                    side_effect=lambda value: SimpleNamespace(**value),
                ),
                patch.object(launcher.subprocess, "Popen") as popen,
            ):
                for purpose, options, capture in (
                    ("runtime_evidence", {}, True),
                    ("ui_performance", {"ui_performance": True}, False),
                ):
                    with self.subTest(purpose=purpose):
                        output = root / purpose
                        launcher.prepare(
                            bundle, registry, adapter, export, output, "fixture-model", **options
                        )
                        gateway = json.loads((output / "gateway.json").read_text())
                        receipt = json.loads((output / "deployment.json").read_text())
                        self.assertIs(gateway["capture_runtime_evidence"], capture)
                        self.assertEqual(receipt["purpose"], purpose)
                        self.assertEqual(
                            receipt["files_sha256"]["gateway.json"],
                            launcher.digest(output / "gateway.json"),
                        )
                        self.assertFalse(receipt["app_default_changed"])
                popen.assert_not_called()

    def test_ui_performance_rejects_historical_tooling_before_reading_assets(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(launcher.preparation, "verify_bundle") as verify,
        ):
            with self.assertRaisesRegex(ValueError, "current v3 deployment"):
                launcher.prepare(
                    None,
                    None,
                    None,
                    None,
                    Path(temporary) / "new",
                    "fixture",
                    ui_performance=True,
                    historical_tooling_only=True,
                )
            verify.assert_not_called()


class ActionsLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.deployment = self.root / "deployment"
        self.deployment.mkdir()
        self.receipt = {
            "historical_tooling_only": False,
            "numerical_regime": "native_derived_q4_base_fp32_lora",
        }
        (self.deployment / "deployment.json").write_text(json.dumps(self.receipt))
        self.config = SimpleNamespace(endpoint="http://127.0.0.1:18091")
        self.adapter = SimpleNamespace(health=Mock(side_effect=ValueError("wrong adapter")))
        self.addCleanup(patch.stopall)
        patch.object(launcher.lifecycle, "LOCK", self.root / "shared.lock").start()
        patch.object(
            launcher, "check", return_value=(self.config, self.adapter, ["model"], self.receipt)
        ).start()

    def test_busy_lock_never_starts_or_stops_another_process(self):
        with launcher.lifecycle.LOCK.open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(launcher.subprocess, "Popen") as popen,
                patch.object(launcher.lifecycle, "stop_owned") as stop,
                self.assertRaises(BlockingIOError),
            ):
                launcher.run(self.deployment, self.root / "run", smoke_only=True)
        popen.assert_not_called()
        stop.assert_not_called()
        self.assertEqual(json.loads((self.root / "run/run.json").read_text())["status"], "failed")

    def test_failed_adapter_health_stops_owned_model_before_editor_start(self):
        model = SimpleNamespace(pid=123, returncode=None)
        model.poll = lambda: model.returncode

        def stop(process):
            if process is not None:
                process.returncode = -15

        with (
            patch.object(launcher.lifecycle, "refuse_occupied"),
            patch.object(launcher.lifecycle, "wait_model"),
            patch.object(launcher.lifecycle, "stop_owned", side_effect=stop) as stopped,
            patch.object(launcher.subprocess, "Popen", return_value=model) as popen,
            self.assertRaisesRegex(ValueError, "wrong adapter"),
        ):
            launcher.run(self.deployment, self.root / "run", with_app=True)
        popen.assert_called_once()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(stopped.call_args_list[-1].args, (model,))
        result = json.loads((self.root / "run/run.json").read_text())
        self.assertTrue(result["model_process_stopped"])
        self.assertEqual(result["model_returncode"], -15)
        self.assertFalse(result["app_started"])

    def test_historical_tooling_adapter_cannot_launch_editor(self):
        self.receipt["historical_tooling_only"] = True
        with (
            patch.object(launcher.subprocess, "Popen") as popen,
            self.assertRaisesRegex(ValueError, "tooling-only"),
        ):
            launcher.run(self.deployment, self.root / "run", with_app=True)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
