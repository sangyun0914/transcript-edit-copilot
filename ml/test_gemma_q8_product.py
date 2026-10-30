"""CPU metadata/transport regressions; no real model, network or conversion."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest import mock

import evaluate_gemma_q8_final as evaluation
import gemma_q8_export as export
import gemma_q8_product as product
import infer_deployed_task_test_actions_v3 as auxiliary
import test_gemma_final_deployment as deployed_fixture


def inventory():
    tensors = [
        {
            "name": f"fixture.{i}.weight",
            "shape": [32, 32],
            "type": "Q8_0" if i else "F32",
            "bytes": 1088 if i else 4096,
        }
        for i in range(658)
    ]
    return {
        "architecture": "gemma4",
        "file_type": 7,
        "tensor_count": 658,
        "semantic_metadata_sha256": "fixture-tokenizer-template-terminal",
        "metadata_keys": ["tokenizer.fixture"],
        "tensor_types": {"F32": 1, "Q8_0": 657},
        "tensors": tensors,
    }


class Q8ExportTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "Run the real tiny-GGUF reader check with the pinned conversion Python.",
    )
    def test_pinned_reader_records_actual_tiny_gguf_types_shapes_and_bytes(self):
        converter = Path(export.read(export.CONFIG)["runtime"]["source_path"])
        sys.path.insert(0, str(converter / "gguf-py"))
        import gguf
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.gguf"
            writer = gguf.GGUFWriter(path, "gemma4")
            writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
            writer.add_tokenizer_model("synthetic")
            writer.add_tensor("fixture.norm", np.zeros(32, dtype=np.float32))
            writer.add_tensor(
                "fixture.weight",
                np.zeros((32, 34), dtype=np.uint8),
                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            actual = export.gguf_inventory(path, converter)
            self.assertEqual(actual["file_type"], 7)
            self.assertEqual(actual["tensor_types"], {"F32": 1, "Q8_0": 1})
            self.assertEqual(
                actual["tensors"][1],
                {
                    "name": "fixture.weight",
                    "shape": [32, 32],
                    "type": "Q8_0",
                    "bytes": 1088,
                },
            )

    def test_q8_census_rejects_q4_masquerade_fallback_and_inventory_or_terminal_change(
        self,
    ):
        expected = inventory()
        export.validate_inventory(expected, expected)
        mutations = []
        for key, value in (
            ("file_type", 15),
            ("tensor_count", 657),
            ("semantic_metadata_sha256", "different-eos"),
        ):
            bad = copy.deepcopy(expected)
            bad[key] = value
            mutations.append(bad)
        for quant in ("Q4_K", "F16"):
            bad = copy.deepcopy(expected)
            bad["tensors"][1]["type"] = quant
            mutations.append(bad)
        bad = copy.deepcopy(expected)
        bad["tensors"][1]["name"] = bad["tensors"][2]["name"]
        mutations.append(bad)
        bad = copy.deepcopy(expected)
        bad["tensors"][1]["shape"] = [64, 32]
        mutations.append(bad)
        for bad in mutations:
            with self.subTest(bad=bad.get("file_type")), self.assertRaises(ValueError):
                export.validate_inventory(bad, expected)

    def test_file_hash_path_escape_and_failed_publication_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.gguf").write_bytes(b"synthetic never loaded")
            files = {"weights.gguf": export.sha(root / "weights.gguf")}
            export.verify_tree(root, files)
            (root / "weights.gguf").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "input changed"):
                export.verify_tree(root, files)
            with self.assertRaisesRegex(ValueError, "escapes"):
                export.verify_tree(root, {"../escape": "0" * 64})
            export.write(
                root / "receipt.json", {"version": export.VERSION, "status": "failed"}
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                export.verify_export(root / "receipt.json")

    def test_matched_quantization_requires_same_adapter_and_native_identity(self):
        native = {"kind": "sft", "adapter_sha256": "native-sft"}
        summary = {
            "model_identity": {
                "kind": "sft",
                "training_adapter_sha256": "native-sft",
                "native_identity_sha256": product.gemma.digest(native),
            }
        }
        product.matched_q4_identity(summary, native)
        for different in (
            {**native, "kind": "dpo"},
            {**native, "adapter_sha256": "dpo-weights"},
        ):
            with self.assertRaisesRegex(ValueError, "same-adapter"):
                product.matched_q4_identity(summary, different)

    def test_q8_command_preserves_full_swa_cache_and_batch_contract(self):
        command = [
            "fixture-server",
            "-c",
            "4096",
            "-np",
            "1",
            "-ngl",
            "99",
            "--cache-ram",
            "0",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "-b",
            "2048",
            "-ub",
            "512",
            "-fa",
            "auto",
            "-ctk",
            "f16",
            "-ctv",
            "f16",
            "--swa-full",
            "--jinja",
            "--no-ui",
            "--lora",
            "adapter",
            "-m",
            "q8",
        ]
        product.fixed_command(command)
        for altered in (
            command + ["-c", "4096"],
            [v for v in command if v != "--swa-full"],
            ["q8_0" if v == "f16" else v for v in command],
        ):
            with self.assertRaises(ValueError):
                product.fixed_command(altered)


class Q8HTTPTests(unittest.TestCase):
    def setUp(self):
        self.fixture = deployed_fixture.GemmaDeployedTests(
            methodName="test_real_loaded_adapter_validator_keeps_id_path_scale_and_ordinary_lora"
        )
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_q8_records_every_failed_request_and_rejects_post_child_outer_failure(self):
        fixture = self.fixture
        prepared = fixture.root / "prepared"
        rows = [
            {
                "case_id": f"fixture-{i}",
                "split": "dev",
                "format": "json",
                "input_sha256": "input",
                "messages_sha256": "messages",
                "prefix_ids_sha256": "prefix",
                "prefix_ids": [2, 20, 106],
                "messages": [{"role": "user", "content": "Synthetic fixture"}],
            }
            for i in range(2)
        ]
        evaluation.gemma.evidence.prep.write_rows(
            prepared / "requests-json.jsonl", rows
        )
        for path in (
            prepared / "manifest.json",
            fixture.root / "model.json",
            fixture.root / "deployment.json",
        ):
            evaluation.write(path, {"engineering_fixture": True})
        model = {"identity": {"kind": "sft", "adapter_sha256": "native-sft"}}
        context = {
            "gateway": fixture.gateway,
            "policy": {
                "context_tokens": 4096,
                "request_settings": auxiliary.request_settings(fixture.gateway),
            },
            "model": model,
            "identity": evaluation.development.deployed.product_identity(
                fixture.gateway, model
            ),
            "requests": rows,
        }
        with (
            mock.patch.object(evaluation, "deployed_context", return_value=context),
            mock.patch.object(auxiliary.http, "request", side_effect=fixture.transport),
        ):
            result = evaluation.infer_deployed(
                prepared,
                fixture.root / "model.json",
                fixture.root / "deployment.json",
                fixture.root / "q8-success",
            )
            self.assertEqual(result["numerical_regime"], export.REGIME)
            path = fixture.root / "q8-success/observations.jsonl"
            observed = evaluation.gemma.evidence.prep.rows(path)
            self.assertTrue(all(r["status"] == "completed" for r in observed))
            observed[0]["http_evidence"]["completion"]["request"]["lora"] = []
            path.write_text("".join(json.dumps(row) + "\n" for row in observed))
            evaluation.gemma.evidence.sft.manifest(
                path.parent, "inference_complete", evaluation.VERSION
            )
            with self.assertRaisesRegex(ValueError, "different request or LoRA"):
                evaluation.verify_deployed_run(prepared, path.parent)
            fixture.loaded = []
            result = evaluation.infer_deployed(
                prepared,
                fixture.root / "model.json",
                fixture.root / "deployment.json",
                fixture.root / "q8-failed-requests",
            )
            self.assertEqual(result["numerical_regime"], export.REGIME)
            observed = evaluation.gemma.evidence.prep.rows(
                fixture.root / "q8-failed-requests/observations.jsonl"
            )
            self.assertEqual(len(observed), 2)
            self.assertTrue(all(r["status"] == "request_failure" for r in observed))
        outer = fixture.root / "late-failure"
        evaluation.write(
            outer / "manifest.json", {"version": evaluation.VERSION, "status": "failed"}
        )
        evaluation.write(
            outer / "run.json",
            {"version": evaluation.VERSION, "status": "inference_complete"},
        )
        with self.assertRaisesRegex(ValueError, "incomplete or failed"):
            evaluation.verify_full_run(outer)


if __name__ == "__main__":
    unittest.main()
