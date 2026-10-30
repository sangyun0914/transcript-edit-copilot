import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_local_copilot_small_final as launcher


class SmallDeploymentTests(unittest.TestCase):
    def test_merged_model_uses_native_checks_and_dispatch_without_external_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return path

            for name in ("run_local_copilot_actions_v3.py", "run_local_copilot.py"):
                write("server/scripts/" + name, "source")
            for name in ("copilot.py", "copilot_runtime.py", "copilot_store.py"):
                write("server/app/" + name, "source")
            exported = root / "export"
            model = write("export/model-Q4_K_M.gguf", "small model fixture")
            template = write("export/chat_template.jinja", "native template fixture")
            config = {
                "context_tokens": 4096,
                "model": "lfm25-1.2b",
                "base": {"revision": "pinned"},
                "assets": {"files_sha256": {"chat_template.jinja": launcher.digest(template)}},
            }
            preparation = write("preparation/configuration.json", config).parent
            write("export/configuration.json", config)
            write(
                "export/run.json",
                {
                    "status": "gguf_exported_runtime_validation_pending",
                    "base": config["base"],
                    "quantization": "Q4_K_M",
                    "gguf_sha256": launcher.digest(model),
                    "adapter_sha256": "a" * 64,
                },
            )
            registry = write(
                "registry.json",
                {
                    "runtime": {
                        "revision": "427291b5b34cd914a31b3fd3b61a68f6184f4b9f",
                        "tag": "b10816",
                        "files_sha256": {},
                        "server": "server-bin",
                    }
                },
            )
            rows = [
                {
                    "case_id": language,
                    "language": language,
                    "messages": [],
                    "input_ids": [1, 2, 7],
                    "completion_start": 2,
                }
                for language in ("en", "ko")
            ]
            with (
                patch.object(launcher, "ROOT", root),
                patch.object(launcher.data, "prepared", return_value=(config, rows, None)),
                patch.object(launcher.legacy.preparation, "source_closure", return_value=[]),
            ):
                output = root / "deployment"
                launcher.prepare(preparation, exported, output, "final-lfm", registry)
                gateway, _, command, receipt = launcher.legacy.check(output)
                self.assertIsNone(gateway.adapter_path)
                self.assertTrue(gateway.cache_prompt)
                self.assertFalse(gateway.capture_runtime_evidence)
                self.assertNotIn("--lora", command)
                self.assertEqual(receipt["training_adapter_sha256"], "a" * 64)
                self.assertEqual(receipt["kind"], "sft")
                parity = json.loads((output / "native-token-parity.json").read_text())
                self.assertEqual({row["case_id"] for row in parity["checks"]}, {"en", "ko"})
                self.assertFalse(receipt["app_default_changed"])
                model.write_text("different model")
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    launcher.legacy.check(output)

    def test_mismatched_export_stops_before_creating_a_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run.json").write_text(
                json.dumps(
                    {
                        "status": "gguf_exported_runtime_validation_pending",
                        "base": {"revision": "other"},
                    }
                )
            )
            with patch.object(
                launcher.data,
                "prepared",
                return_value=({"context_tokens": 4096, "base": {"revision": "expected"}}, [], None),
            ):
                with self.assertRaisesRegex(ValueError, "matching native-derived"):
                    launcher.prepare(root, root, root / "deployment", "model", root / "unused.json")
            self.assertFalse((root / "deployment").exists())

    def test_dpo_export_preserves_and_checks_its_actual_sft_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configuration.json").write_text('{"model":"lfm25-1.2b"}')
            (root / "plan.md").write_text("Registered before training.")
            frozen = {
                "plan_sha256": launcher.digest(root / "plan.md"),
                "parent": {
                    "configuration_sha256": launcher.digest(root / "configuration.json"),
                    "adapter_sha256": "a" * 64,
                },
            }
            (root / "dpo_configuration.json").write_text(json.dumps(frozen))
            report = {
                "kind": "dpo",
                "parent_adapter_sha256": "a" * 64,
                "dpo_configuration_sha256": launcher.digest(root / "dpo_configuration.json"),
            }
            self.assertEqual(launcher.export_identity(root, report), report)
            report["parent_adapter_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "registered SFT parent"):
                launcher.export_identity(root, report)


if __name__ == "__main__":
    unittest.main()
