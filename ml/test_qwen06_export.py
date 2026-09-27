import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from export_qwen06 import (
    BASE_REPO,
    check_hash,
    validate_adapter_config,
    verify_download,
    write_new,
    run,
)
from verify_qwen06_export import mapped_adapter_name


class Qwen06ExportTests(unittest.TestCase):
    def test_unsupported_adapter_semantics_are_rejected(self):
        original = {
            "base_model_name_or_path": BASE_REPO,
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 16,
            "bias": "none",
            "lora_bias": False,
            "fan_in_fan_out": False,
            "use_dora": False,
            "use_rslora": False,
        }
        validate_adapter_config(original)
        for key, value in (
            ("base_model_name_or_path", "Qwen/wrong-revision"),
            ("use_dora", True),
            ("use_rslora", True),
            ("rank_pattern", {"mlp": 32}),
            ("modules_to_save", ["lm_head"]),
        ):
            with self.subTest(key=key):
                config = copy.deepcopy(original)
                config[key] = value
                with self.assertRaises(ValueError):
                    validate_adapter_config(config)

    def test_mapping_preserves_attention_output_and_rejects_unexpected_modules(self):
        prefix = "base_model.model.model.language_model.layers.3."
        self.assertEqual(
            mapped_adapter_name(prefix + "self_attn.o_proj.lora_B.weight"),
            (
                "blk.3.attn_output.weight",
                "blk.3.attn_output.weight.lora_b",
            ),
        )
        for name in (
            prefix + "linear_attn.out_proj.lora_A.weight",
            prefix.replace("layers.3.", "layers.24.") + "mlp.down_proj.lora_A.weight",
        ):
            with self.assertRaises(ValueError):
                mapped_adapter_name(name)

    def test_download_accepts_only_published_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(b"original")
            expected = hashlib.sha256(b"original").hexdigest()
            verify_download(path, {"size": 8, "lfs": {"sha256": expected}})
            path.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                verify_download(path, {"size": 8, "lfs": {"sha256": expected}})
            with self.assertRaises(ValueError):
                check_hash(path, expected)

    def test_existing_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            write_new(path, {"status": "failed"})
            with self.assertRaises(FileExistsError):
                write_new(path, {"status": "passed"})
            self.assertEqual(json.loads(path.read_text()), {"status": "failed"})

    def test_dotted_model_names_keep_separate_command_attempts(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("export_qwen06.ROOT", Path(directory)),
        ):
            for _ in range(2):
                run("quantize-Qwen3.5", [sys.executable, "-c", "pass"])
            commands = sorted((Path(directory) / "logs").glob("*.command.json"))
            self.assertEqual(
                [p.name for p in commands],
                [
                    "quantize-Qwen3-5-001.command.json",
                    "quantize-Qwen3-5-002.command.json",
                ],
            )


if __name__ == "__main__":
    unittest.main()
