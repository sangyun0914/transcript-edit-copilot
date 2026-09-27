import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import train_passive_sft as trainer


class SmokePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.preparation = self.root / "preparation"
        self.native = self.root / "native"
        self.tokenizer = self.root / "tokenizer"
        for path in (self.preparation, self.native, self.tokenizer):
            path.mkdir()
        tokenizer_files = {
            "config.json": {"text_config": {"vocab_size": 64, "eos_token_id": 2}},
            "tokenizer.json": {"test": True},
            "tokenizer_config.json": {"eos_token": "<end>"},
            "chat_template.jinja": "test template",
        }
        for name, value in tokenizer_files.items():
            trainer.write(self.tokenizer / name, value)
        tokenizer_hashes = {name: trainer.digest(self.tokenizer / name) for name in tokenizer_files}
        (self.native / "config.json").write_bytes((self.tokenizer / "config.json").read_bytes())
        (self.native / "LICENSE").write_text("test license")
        (self.native / "weights.safetensors").write_bytes(b"fixture only; never load")
        trainer.write(
            self.native / "model.safetensors.index.json",
            {
                "weight_map": {
                    "model.weight": "weights.safetensors",
                    "mtp.test.weight": "weights.safetensors",
                },
            },
        )
        self.config = {
            "version": trainer.VERSION,
            "candidate_id": "qwen35-9b",
            "trainer_sha256": trainer.digest(Path(trainer.__file__)),
            "bindings": {},
            "base": {
                "repository": "Qwen/Qwen3.5-9B",
                "revision": "a" * 40,
                "path": "native",
                "absent_files": ["generation_config.json"],
                "files_sha256": {p.name: trainer.digest(p) for p in self.native.iterdir()},
            },
            "tokenizer": {
                "path": "tokenizer",
                "files_sha256": tokenizer_hashes,
                "template_sha256": tokenizer_hashes["chat_template.jinja"],
            },
            "libraries": {key: "1.0.0" for key in trainer.LIBRARIES},
            "model": {
                "class": "Qwen3_5ForConditionalGeneration",
                "dtype": "bfloat16",
                "quantization": "nf4",
                "bnb_4bit_use_double_quant": True,
                "attn_implementation": "eager",
                "gradient_checkpointing": True,
                "target_modules": ["model.language_model.layers.3.self_attn.q_proj"],
                "ignored_native_tensor_names": ["mtp.test.weight"],
            },
            "training": {
                "seed": 3407,
                "micro_batch_size": 1,
                "gradient_accumulation": 4,
                "max_updates": 2,
                "rank": 16,
                "alpha": 16,
                "dropout": 0,
                "bias": "none",
                "optimizer": "adamw_torch",
                "scheduler": "constant",
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "max_sequence_tokens": 4096,
                "max_completion_tokens": 512,
                "loss_implementation": "completion_positions_cross_entropy",
                "normalization": "mean_completion_loss_per_row_then_mean_microbatches",
                "truncation": False,
                "packing": False,
            },
            "generation": {
                "max_new_tokens": 512,
                "do_sample": False,
                "use_cache": True,
                "num_beams": 1,
                "eos_token_id": 1,
                "pad_token_id": 1,
            },
        }
        self.cases, self.messages, self.tokens, self.inputs = [], [], [], []
        for index, (action, language) in enumerate(trainer.SLOTS):
            identity = f"row-{index}"
            answer = {"action": action}
            prefix = index + 2
            primary = {
                "input_ids": [5] * prefix + [9, 1],
                "labels": [-100] * prefix + [9, 1],
                "attention_mask": [1] * (prefix + 2),
                "completion_start": prefix,
            }
            messages = [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": identity},
            ]
            reference = json.dumps(answer)
            self.cases.append(
                {
                    "id": identity,
                    "expectedOutput": answer,
                    "metadata": {
                        "split": "train",
                        "training_eligible": True,
                        "language": language,
                        "source_group_id": language,
                        "sampling_cluster": f"train:{language}",
                        "input_sha256": "b" * 64,
                    },
                }
            )
            self.messages.append(
                {
                    "id": identity,
                    "messages": [*messages, {"role": "assistant", "content": reference}],
                }
            )
            self.tokens.append(
                {
                    "id": identity,
                    "split": "train",
                    "input_sha256": "b" * 64,
                    "messages": messages,
                    "references": [reference],
                    "primary": primary,
                }
            )
            self.inputs.append(
                {key: primary[key] for key in ("input_ids", "labels", "attention_mask")}
            )
        order = [row["id"] for row in self.cases]
        self.config["training"].update(
            sample_order=order,
            max_total_sequence_tokens=sum(len(r["input_ids"]) for r in self.inputs),
            max_total_completion_tokens=16,
        )
        self.prepared = {
            "native_base": self.config["base"],
            "tokenizer": {
                "files_sha256": tokenizer_hashes,
                "eos_token_id": 1,
                "vocab_size": 64,
            },
            "proposed_smoke": {"order": order},
        }
        self.selection = {
            "selected_t1_candidate_ids": ["qwen35-9b"],
            "bindings": {"release_manifest_sha256": "c" * 64},
            "selected_native_identity": {
                "base_repository": self.config["base"]["repository"],
                "base_revision": self.config["base"]["revision"],
                "tokenizer_files_sha256": tokenizer_hashes,
            },
        }
        self.persist()

    def persist(self):
        trainer.write(self.preparation / "configuration.json", self.prepared)
        for name, rows in (
            ("smoke-cases.jsonl", self.cases),
            ("smoke-messages.jsonl", self.messages),
            ("smoke-tokens.jsonl", self.tokens),
            ("smoke-model-inputs.jsonl", self.inputs),
        ):
            (self.preparation / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        trainer.write(
            self.preparation / "manifest.json",
            {
                "candidate_id": "qwen35-9b",
                "release_manifest_sha256": "c" * 64,
                "files_sha256": {
                    p.name: trainer.digest(p)
                    for p in self.preparation.iterdir()
                    if p.name != "manifest.json"
                },
            },
        )
        self.selection["bindings"]["train_only_preparation"] = {
            "sha256": trainer.digest(self.preparation / "manifest.json")
        }
        trainer.write(self.root / "selection.json", self.selection)
        self.config["bindings"] = {
            "selection": {
                "path": "selection.json",
                "sha256": trainer.digest(self.root / "selection.json"),
            },
            "preparation": {
                "path": "preparation/manifest.json",
                "sha256": trainer.digest(self.preparation / "manifest.json"),
            },
        }
        self.save_config()

    def save_config(self):
        trainer.write(self.root / "configuration.json", self.config)

    def preflight(self):
        return trainer.preflight(self.root / "configuration.json")

    def test_preflight_is_portable_and_retains_explicit_eos_difference(self):
        _, _, rows, _, report = self.preflight()
        self.assertEqual(len(rows), 8)
        self.assertEqual(report["native_config_eos_token_id"], 2)
        self.assertEqual(report["explicit_training_and_generation_eos_token_id"], 1)
        self.assertEqual(report["training_updates"], 0)
        self.assertFalse(report["environment"]["matched"])

    def test_set_valued_loading_diagnostics_are_stable_json_arrays(self):
        diagnostics = {
            "missing_keys": set(),
            "unexpected_keys": {"mtp.z.weight", "mtp.a.weight"},
            "mismatched_keys": [("x", (2, 3), (3, 2))],
            "nested": {"keys": frozenset({"b", "a"})},
        }
        normalized = trainer.loading_diagnostics(diagnostics)
        self.assertEqual(json.loads(json.dumps(normalized)), normalized)
        self.assertEqual(normalized["missing_keys"], [])
        self.assertEqual(normalized["unexpected_keys"], ["mtp.a.weight", "mtp.z.weight"])
        self.assertEqual(normalized["nested"]["keys"], ["a", "b"])
        self.assertIsInstance(diagnostics["unexpected_keys"], set)

    def test_absent_library_gate_detects_incompatible_preinstalled_package(self):
        self.config["libraries"]["python"] = trainer.platform.python_version()
        self.config["absent_libraries"] = ["torchao"]
        trainer.validate_configuration(self.config)

        def installed(name):
            return "0.10.0" if name == "torchao" else "1.0.0"

        with patch.object(trainer.importlib.metadata, "version", side_effect=installed):
            report = trainer.environment(self.config)
        self.assertFalse(report["matched"])
        self.assertEqual(report["differences"], {"torchao": {"expected": None, "actual": "0.10.0"}})

        def removed(name):
            if name == "torchao":
                raise trainer.importlib.metadata.PackageNotFoundError(name)
            return "1.0.0"

        with patch.object(trainer.importlib.metadata, "version", side_effect=removed):
            report = trainer.environment(self.config)
        self.assertTrue(report["matched"])
        self.assertIsNone(report["actual"]["torchao"])
        for invalid in (["torch"], ["torchao", "torchao"], "torchao"):
            self.config["absent_libraries"] = invalid
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "Absent libraries"),
            ):
                trainer.validate_configuration(self.config)

    def test_bound_native_payload_generation_absence_and_source_drift_fail(self):
        for path in (self.native / "weights.safetensors", self.preparation / "smoke-tokens.jsonl"):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Bound file changed"):
                self.preflight()
            path.write_bytes(original)
        (self.native / "generation_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "absent file appeared"):
            self.preflight()
        (self.native / "generation_config.json").unlink()
        self.config["trainer_sha256"] = "f" * 64
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Bound file changed"):
            self.preflight()

    def test_rehashed_masks_nontrain_and_sample_order_are_rejected(self):
        originals = copy.deepcopy((self.cases, self.tokens, self.inputs, self.config))
        for mutation in ("mask", "attention", "eos", "dev", "order", "budget"):
            self.cases, self.tokens, self.inputs, self.config = copy.deepcopy(originals)
            if mutation == "mask":
                self.tokens[0]["primary"]["labels"][0] = 5
            elif mutation == "attention":
                self.tokens[0]["primary"]["attention_mask"][0] = 0
            elif mutation == "eos":
                self.tokens[0]["primary"]["input_ids"][-1] = 2
                self.tokens[0]["primary"]["labels"][-1] = 2
            elif mutation == "dev":
                self.cases[0]["metadata"]["split"] = "dev"
            elif mutation == "order":
                self.config["training"]["sample_order"] = list(
                    reversed(self.config["training"]["sample_order"])
                )
            else:
                self.config["training"]["max_total_completion_tokens"] += 1
            self.persist()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.preflight()

    def test_vision_gdn_unpinned_mtp_and_escape_paths_are_rejected(self):
        original = copy.deepcopy(self.config)
        for target in (
            "q_proj",
            "model.visual.blocks.0.self_attn.q_proj",
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
        ):
            self.config = copy.deepcopy(original)
            self.config["model"]["target_modules"] = [target]
            self.save_config()
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "module names"):
                self.preflight()
        self.config = copy.deepcopy(original)
        self.config["model"]["ignored_native_tensor_names"] = []
        self.save_config()
        with self.assertRaisesRegex(ValueError, "MTP tensor census"):
            self.preflight()
        self.config = copy.deepcopy(original)
        self.config["base"]["path"] = "../native"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.preflight()


class SmokeLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(
                "Torch CPU tests require the existing export environment."
            ) from error
        cls.torch = torch

    def model(self):
        torch = self.torch

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(16, 7)
                self.head = torch.nn.Linear(7, 16)

            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep, return_dict):
                from types import SimpleNamespace

                return SimpleNamespace(
                    logits=self.head(self.embedding(input_ids)[:, logits_to_keep])
                )

        torch.manual_seed(1)
        return ToyModel()

    def rows(self):
        return [
            {
                "case_id": f"row-{i}",
                "input_ids": [2, 3, *([4 + i] * (i % 3 + 1)), 1],
                "labels": [-100, -100, *([4 + i] * (i % 3 + 1)), 1],
                "completion_start": 2,
            }
            for i in range(8)
        ]

    def test_two_updates_equal_manual_means_of_variable_length_completion_losses(self):
        torch = self.torch
        actual, expected = self.model(), self.model()
        rows = self.rows()
        settings = {"gradient_accumulation": 4, "max_updates": 2, "max_grad_norm": 1.0}
        actual_optimizer = torch.optim.SGD(actual.parameters(), lr=0.02)
        expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.02)
        updates, seen = [], []
        trainer.train_updates(
            actual, rows, actual_optimizer, torch, settings, "cpu", updates.append, seen.append
        )
        for offset in (0, 4):
            expected_optimizer.zero_grad()
            sum(
                trainer.completion_loss(expected, row, torch, "cpu") / 4
                for row in rows[offset : offset + 4]
            ).backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
            expected_optimizer.step()
        self.assertEqual([event["case_id"] for event in seen], [row["case_id"] for row in rows])
        self.assertEqual([event["consumed_rows"] for event in updates], [4, 8])
        for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(left, right)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            trainer.train_updates(
                actual, rows[:-1], actual_optimizer, torch, settings, "cpu", updates.append
            )

    def test_nonfinite_loss_stops_before_optimizer_update(self):
        torch = self.torch
        model = self.model()
        with torch.no_grad():
            model.head.weight.fill_(float("nan"))
        updates = []
        with self.assertRaisesRegex(ValueError, "Non-finite loss"):
            trainer.train_updates(
                model,
                self.rows(),
                torch.optim.SGD(model.parameters(), lr=0.02),
                torch,
                {"gradient_accumulation": 4, "max_updates": 2, "max_grad_norm": 1.0},
                "cpu",
                updates.append,
            )
        self.assertEqual(updates, [])

    def test_real_qwen_selected_logits_equal_full_masked_loss_and_gradients(self):
        try:
            from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        except ImportError as error:
            self.skipTest(f"Qwen3.5 Transformers required: {error}")
        torch = self.torch
        torch.manual_seed(19)
        config = Qwen3_5Config(
            text_config={
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "layer_types": ["full_attention"],
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [1, 1, 2],
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
            },
        )
        config._attn_implementation = "eager"
        model = Qwen3_5ForConditionalGeneration(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            model, diagnostics = Qwen3_5ForConditionalGeneration.from_pretrained(
                directory,
                local_files_only=True,
                output_loading_info=True,
                attn_implementation="eager",
            )
            trainer.write(
                Path(directory) / "loading-info.json", trainer.loading_diagnostics(diagnostics)
            )
        model.eval()
        row = {
            "input_ids": [3, 4, 5, 6, 7, 1],
            "labels": [-100, -100, -100, 6, 7, 1],
            "completion_start": 3,
        }
        ids = torch.tensor([row["input_ids"]])
        full = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=torch.tensor([row["labels"]]),
            use_cache=False,
        ).loss
        full.backward()
        expected = {
            name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad(set_to_none=True)
        selected = trainer.completion_loss(model, row, torch, "cpu")
        selected.backward()
        torch.testing.assert_close(selected, full)
        for name, parameter in model.named_parameters():
            if name in expected:
                torch.testing.assert_close(parameter.grad, expected[name], atol=2e-6, rtol=2e-5)

    def test_hybrid_qwen_peft_save_reload_preserves_tensors_and_greedy_outputs(self):
        try:
            from peft import PeftModel, get_peft_model_state_dict
            from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        except ImportError as error:
            self.skipTest(f"Qwen3.5 and PEFT required: {error}")
        torch = self.torch
        torch.manual_seed(23)
        native_config = Qwen3_5Config(
            text_config={
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [1, 1, 2],
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
            },
        )
        native_config._attn_implementation = "eager"
        base = Qwen3_5ForConditionalGeneration(native_config)
        original = copy.deepcopy(base.state_dict())
        config = {
            "base": {"repository": "Qwen/Qwen3.5-9B", "revision": "a" * 40},
            "model": {
                "quantization": "none",
                "target_modules": [
                    name
                    for name, _ in base.named_modules()
                    if trainer.MODULE_PATTERN.fullmatch(name)
                ],
            },
            "training": {
                "rank": 16,
                "alpha": 16,
                "dropout": 0,
                "bias": "none",
                "gradient_accumulation": 4,
                "max_updates": 2,
                "max_grad_norm": 1.0,
            },
            "generation": {
                "max_new_tokens": 4,
                "do_sample": False,
                "use_cache": True,
                "num_beams": 1,
                "eos_token_id": 1,
                "pad_token_id": 1,
            },
        }
        model, trainable = trainer.enable_training(base, config)
        self.assertEqual(len(trainable), 2 * len(config["model"]["target_modules"]))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        trainer.train_updates(
            model, self.rows(), optimizer, torch, config["training"], "cpu", lambda _: None
        )

        class Tokenizer:
            bos_token_id = None

            @staticmethod
            def decode(ids, skip_special_tokens):
                return json.dumps(ids)

        saved_tensors = trainer.tensor_records(
            get_peft_model_state_dict(model, save_embedding_layers=False), torch
        )
        saved_outputs = trainer.generate_smoke(
            model, Tokenizer(), self.rows()[:2], config, torch, "cpu"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
            adapter_config = trainer.read(output / "adapter_config.json")
            self.assertEqual(
                adapter_config["base_model_name_or_path"], config["base"]["repository"]
            )
            self.assertEqual(adapter_config["revision"], config["base"]["revision"])
            fresh_base = Qwen3_5ForConditionalGeneration(native_config)
            fresh_base.load_state_dict(original)
            fresh_base = trainer.prepare_reload_base(fresh_base, config)
            fresh = PeftModel.from_pretrained(
                fresh_base, output, is_trainable=False, local_files_only=True
            )
            self.assertEqual(
                trainer.tensor_records(
                    get_peft_model_state_dict(fresh, save_embedding_layers=False), torch
                ),
                saved_tensors,
            )
            self.assertEqual(
                trainer.generate_smoke(fresh, Tokenizer(), self.rows()[:2], config, torch, "cpu"),
                saved_outputs,
            )


if __name__ == "__main__":
    unittest.main()
