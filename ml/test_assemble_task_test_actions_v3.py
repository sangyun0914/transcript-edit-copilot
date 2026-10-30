"""Model-specific source assembly fixtures; no held-out bodies or model weights."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import assemble_task_test_actions_v3 as assembly


class FrozenRuntimeTests(unittest.TestCase):
    def setUp(self):
        (assembly.ROOT / "artifacts").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="final-runtime-engineering-", dir=assembly.ROOT / "artifacts"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        preparation = self.root / "preparation"
        (preparation / "source").mkdir(parents=True)
        names = (*assembly.SHARED, "train_gemma_actions_v3.py")
        for name in names:
            body = (assembly.ROOT / "ml" / name).read_text()
            if name == "train_gemma_actions_v3.py":
                body += (
                    '\nENGINEERING_FROZEN_SOURCE_MARKER = "original_model_runtime"\n'
                )
            (preparation / "source" / name).write_text(body)
        hashes = {
            name: assembly.file_sha(preparation / "source" / name) for name in names
        }
        self.config = preparation / "config-json.json"
        assembly.write(self.config, {"sources_sha256": hashes})
        assembly.write(
            preparation / "manifest.json",
            {
                "train_only": True,
                "final_test_bodies_read": False,
                "files_sha256": {
                    "config-json.json": assembly.file_sha(self.config),
                    **{"source/" + name: value for name, value in hashes.items()},
                },
            },
        )
        self.runtime = self.root / "runtime"
        self.identity = assembly.assemble(self.config, self.runtime)

    def test_original_model_sources_win_and_normal_import_uses_them(self):
        expected = assembly.read(self.config)["sources_sha256"][
            "train_gemma_actions_v3.py"
        ]
        self.assertEqual(
            self.identity["source_sha256"]["ml/train_gemma_actions_v3.py"], expected
        )
        self.assertNotEqual(
            expected, assembly.file_sha(assembly.ROOT / "ml/train_gemma_actions_v3.py")
        )
        environment = {
            **os.environ,
            "PYTHONPATH": str(self.runtime / "ml"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import train_gemma_actions_v3 as s; "
                "print(s.ENGINEERING_FROZEN_SOURCE_MARKER)",
            ],
            cwd=self.runtime,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "original_model_runtime")
        self.assertFalse(any(self.runtime.rglob("*.safetensors")))
        self.assertFalse(any(self.runtime.rglob("*.jsonl")))

    def test_runtime_source_tampering_and_unregistered_imports_fail_closed(self):
        extra = self.runtime / "ml/unregistered_module.py"
        extra.write_text("VALUE = 1\n")
        with self.assertRaisesRegex(ValueError, "Unregistered Python"):
            assembly.load_runtime(self.runtime)
        extra.unlink()
        (self.runtime / "ml/train_gemma_actions_v3.py").write_text("VALUE = 1\n")
        with self.assertRaisesRegex(ValueError, "runtime source changed"):
            assembly.load_runtime(self.runtime)

    def test_changed_archival_source_is_not_replaced_by_current_code(self):
        target = self.config.parent / "source/train_gemma_actions_v3.py"
        target.write_text("VALUE = 1\n")
        with self.assertRaisesRegex(ValueError, "Frozen preparation source changed"):
            assembly.assemble(self.config, self.root / "never-created")
        self.assertFalse((self.root / "never-created").exists())


if __name__ == "__main__":
    unittest.main()
