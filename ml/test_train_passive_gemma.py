import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import train_passive_gemma as trainer
import train_passive_sft as smoke


def configuration(phase="smoke"):
    return {
        "version": trainer.VERSION,
        "candidate_id": "gemma26",
        "phase": phase,
        **{key: "a" * 64 for _, key, _ in trainer.SOURCE_MODULES},
        "bindings": {
            name: {"path": f"{name}.json", "sha256": "b" * 64}
            for name in ("preparation", "native_pin", "evaluation_protocol")
        },
        "libraries": {key: "1.0.0" for key in smoke.LIBRARIES},
        "absent_libraries": ["torchao"],
        "base": {
            "path": "native",
            "repository": trainer.REPOSITORY,
            "revision": trainer.REVISION,
            "files_sha256": {},
            "absent_files": [],
        },
        "tokenizer": {
            "path": "tokenizer",
            "files_sha256": {},
            "template_sha256": "c" * 64,
        },
        "model": {
            "class": "Gemma4ForConditionalGeneration",
            "dtype": "bfloat16",
            "adapter_dtype": "float32",
            "quantization": "none",
            "attn_implementation": "sdpa",
            "experts_implementation": "eager",
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "ignored_native_tensor_names": [],
            "expected_trainable_parameters": 18585600,
            "gpu_name_contains": "H100",
            "minimum_gpu_bytes": 75000000000,
            "target_modules": trainer.target_names(),
        },
        "training": {
            "seed": 3407,
            "sampling_seed": "3407",
            "micro_batch_size": 1,
            "gradient_accumulation": 4,
            "max_updates": 2 if phase == "smoke" else 203,
            "draw_count": 8 if phase == "smoke" else 809,
            "rank": 16,
            "alpha": 16,
            "dropout": 0,
            "bias": "none",
            "learning_rate": 1e-4,
            "optimizer": "adamw_torch",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "scheduler": "constant",
            "wall_limit_seconds": 21600,
            "max_sequence_tokens": 4096,
            "max_completion_tokens": 512,
            "loss_implementation": "completion_positions_cross_entropy",
            "normalization": "mean_completion_loss_per_row_then_mean_microbatches",
            "final_partial_accumulation": "actual_microbatch_count",
            "fresh_native_initialization": True,
            "truncation": False,
            "packing": False,
            "recovery_updates": [] if phase == "smoke" else [50, 100, 150],
            "sample_order": [
                f"case-{i}" for i in range(8 if phase == "smoke" else 809)
            ],
        },
        "generation": {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
            "num_beams": 1,
            "eos_token_id": [1, 106, 50],
            "pad_token_id": 0,
            "bos_token_id": 2,
        },
    }


class GemmaBindingTests(unittest.TestCase):
    def test_configuration_snapshot_preserves_original_bytes_and_rejects_late_change(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, saved = root / "input.json", root / "configuration.json"
            original = b'{ "name" : "\\uac00", "nested": {"value":1} }\n\n'
            source.write_bytes(original)
            expected = smoke.digest(source)
            trainer.snapshot_configuration(source, saved, expected)
            self.assertEqual(saved.read_bytes(), original)
            self.assertEqual(smoke.digest(saved), expected)
            source.write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(ValueError, "changed after preflight"):
                trainer.snapshot_configuration(source, root / "invalid.json", expected)
            self.assertFalse((root / "invalid.json").exists())

    def test_phase_and_precision_recipe_are_fixed(self):
        for phase in ("smoke", "substantive"):
            config = configuration(phase)
            trainer.validate_configuration(config)
            for section, key, value in (
                ("model", "quantization", "nf4"),
                ("model", "adapter_dtype", "bfloat16"),
                ("model", "experts_implementation", "grouped_mm"),
                ("model", "target_modules", trainer.target_names()[:-1]),
                ("training", "max_updates", 202),
                ("training", "fresh_native_initialization", False),
                ("generation", "eos_token_id", [1, 106]),
            ):
                changed = copy.deepcopy(config)
                changed[section][key] = value
                with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                    trainer.validate_configuration(changed)

    def test_substantive_gate_requires_bound_completed_smoke_and_exact_reload(self):
        original, config = configuration(), configuration("substantive")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def bind(name, value):
                path = root / f"{name}.json"
                smoke.write(path, value)
                config["bindings"][name] = {
                    "path": path.name,
                    "sha256": smoke.digest(path),
                }
                return smoke.digest(path)

            original_sha = bind("smoke_configuration", original)
            run = {
                "version": trainer.VERSION,
                "status": "training_complete_fresh_reload_pending",
                "phase": "smoke",
                "mode": "train",
                "training_updates": 2,
                "optimizer_committed_rows": 8,
                "configuration_sha256": original_sha,
                "release_manifest_sha256": "r" * 64,
                "final_test_bodies_read": 0,
                "optimizer_state_resumed": False,
                "pid": 12,
            }
            run_sha = bind("smoke_run", run)
            train_sha = bind(
                "smoke_training_manifest",
                {
                    "version": trainer.VERSION,
                    "status": run["status"],
                    "files_sha256": {
                        "run.json": run_sha,
                        "configuration.json": original_sha,
                    },
                },
            )
            reload_run = {
                "version": trainer.VERSION,
                "status": "fresh_reload_passed_local_export_pending",
                "phase": "smoke",
                "mode": "fresh_reload",
                "reloaded_training_updates": 2,
                "previous_manifest_sha256": train_sha,
                "configuration_sha256": original_sha,
                "pid": 13,
            }
            reload_sha = bind("smoke_reload_run", reload_run)
            parity = {
                "adapter_tensors_identical": True,
                "optimizer_state_resumed": False,
                "cases": [
                    {"case_id": name, "identical": True}
                    for name in original["training"]["sample_order"]
                ],
            }
            parity_sha = bind("smoke_reload_parity", parity)
            bind(
                "smoke_reload_manifest",
                {
                    "version": trainer.VERSION,
                    "status": reload_run["status"],
                    "files_sha256": {
                        "run.json": reload_sha,
                        "reload-parity.json": parity_sha,
                    },
                },
            )
            trainer.validate_smoke_gate(config, root, "r" * 64)
            run["training_updates"] = 1
            run_sha = bind("smoke_run", run)
            bind(
                "smoke_training_manifest",
                {
                    "version": trainer.VERSION,
                    "status": run["status"],
                    "files_sha256": {
                        "run.json": run_sha,
                        "configuration.json": original_sha,
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "actual completed"):
                trainer.validate_smoke_gate(config, root, "r" * 64)

    def test_reload_rejects_manifest_omission_before_reading_unbound_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke.write(
                root / "manifest.json",
                {
                    "version": trainer.VERSION,
                    "status": "training_complete_fresh_reload_pending",
                    "files_sha256": {"run.json": "a" * 64},
                },
            )
            with self.assertRaisesRegex(ValueError, "all consumed files hash-bound"):
                trainer.verify_previous(root, configuration(), "a" * 64)

    def test_tool_terminal_is_retained_and_recognized_without_hiding_raw_text(self):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                return "answer" if skip_special_tokens else "answer<|tool_response>"

        class Tensor:
            def __init__(self, values):
                self.values = values

            def __getitem__(self, key):
                return Tensor(self.values[key[0]][key[1]])

            def tolist(self):
                return self.values

        class Model:
            def eval(self):
                pass

            def gradient_checkpointing_disable(self):
                pass

            def generate(self, **kwargs):
                return Tensor([[2, 3, 12, 50]])

        try:
            import torch
            from transformers import GenerationConfig  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(
                "Use the pinned Gemma CPU test environment."
            ) from error
        result = trainer.generate_smoke(
            Model(),
            Tokenizer(),
            [
                {"case_id": "case", "input_ids": [2, 3, 11, 1], "completion_start": 2},
            ],
            configuration(),
            torch,
            "cpu",
        )[0]
        self.assertEqual(result["output_ids"], [12, 50])
        self.assertEqual(result["terminal_token_id"], 50)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertIn("<|tool_response>", result["text_raw"])


class GemmaModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import (
                Gemma4Config,
                Gemma4TextConfig,
                Gemma4ForConditionalGeneration,
            )
        except ImportError as error:
            raise unittest.SkipTest(
                "Use Transformers5.8 and Torch in the Gemma CPU environment."
            ) from error
        cls.torch = torch
        cls.Config, cls.TextConfig, cls.Model = (
            Gemma4Config,
            Gemma4TextConfig,
            Gemma4ForConditionalGeneration,
        )
        torch.set_num_threads(1)

    def model(self):
        self.torch.manual_seed(3407)
        text = self.TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            hidden_size_per_layer_input=0,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=16,
            max_position_embeddings=64,
            attention_k_eq_v=True,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=16,
            final_logit_softcapping=30.0,
            use_bidirectional_attention="vision",
        )
        config = self.Config(
            text_config=text,
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "patch_size": 2,
                "position_embedding_size": 16,
            },
        )
        config._attn_implementation = "sdpa"
        config._experts_implementation = "eager"
        return self.Model(config).to(self.torch.bfloat16)

    def settings(self, model):
        config = configuration()
        targets = {
            n: m
            for n, m in model.named_modules()
            if trainer.MODULE_PATTERN.fullmatch(n)
        }
        config["model"]["target_modules"] = sorted(targets)
        config["training"].update(rank=2, alpha=2)
        config["model"]["expected_trainable_parameters"] = sum(
            2 * (m.in_features + m.out_features) for m in targets.values()
        )
        return config

    def rows(self):
        return [
            {
                "case_id": f"case-{i}",
                "draw_index": i,
                "source_id": "source",
                "source_group_id": "group",
                "episode_id": "episode",
                "language": "en",
                "action": "no_action",
                "input_ids": [2, 7 + i, 9, 12, 1],
                "attention_mask": [1] * 5,
                "labels": [-100, -100, -100, 12, 1],
                "completion_start": 3,
            }
            for i in range(8)
        ]

    def test_indexed_softcapped_completion_loss_matches_full_causal_loss(self):
        torch = self.torch
        model = self.model().eval()
        row = self.rows()[0]
        ids = torch.tensor([row["input_ids"]])
        full = model(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        ).logits
        expected = torch.nn.functional.cross_entropy(
            full[:, :-1].float().reshape(-1, full.shape[-1]),
            torch.tensor([row["labels"]])[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        actual = smoke.completion_loss(model, row, torch, "cpu")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_architecture_meta_census_matches_205_targets_and_rank16_size(self):
        from accelerate import init_empty_weights

        text = self.TextConfig(
            hidden_size=2816,
            intermediate_size=2112,
            num_hidden_layers=30,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_global_key_value_heads=2,
            head_dim=256,
            global_head_dim=512,
            hidden_size_per_layer_input=0,
            attention_k_eq_v=True,
            enable_moe_block=True,
            num_experts=128,
            top_k_experts=8,
            moe_intermediate_size=704,
        )
        with init_empty_weights():
            model = self.Model(self.Config(text_config=text))
        modules = {
            n: m
            for n, m in model.named_modules()
            if trainer.MODULE_PATTERN.fullmatch(n)
        }
        self.assertEqual(sorted(modules), trainer.target_names())
        self.assertEqual(
            sum(16 * (m.in_features + m.out_features) for m in modules.values()),
            18585600,
        )
        self.assertEqual(
            len([n for n, _ in model.named_parameters() if ".experts." in n]), 60
        )
        self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))

    def test_two_updates_keep_base_bf16_and_fresh_process_reloads_exact_adapter_and_outputs(
        self,
    ):
        torch = self.torch
        from peft import get_peft_model_state_dict

        model = self.model()
        settings = self.settings(model)
        rows = self.rows()
        base = {name: p.detach().clone() for name, p in model.named_parameters()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model.save_pretrained(root / "base")
            model, trainable = trainer.enable_training(model, settings, torch)
            initial = smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            exposure, events, updates = trainer.curriculum.Exposure(), [], []
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=0.01,
            )
            trainer.curriculum.train_draws(
                model,
                rows,
                optimizer,
                torch,
                settings["training"],
                "cpu",
                exposure,
                events.append,
                updates.append,
            )
            self.assertEqual([u["microbatches"] for u in updates], [4, 4])
            self.assertEqual([event["draw_index"] for event in events], list(range(8)))
            self.assertEqual(
                exposure.summary()["phases"]["optimizer_committed"]["rows"], 8
            )
            self.assertEqual(len(trainable), 26)
            actual_base = model.get_base_model()
            for name, tensor in base.items():
                module, _, suffix = name.rpartition(".")
                if module in settings["model"]["target_modules"]:
                    name = module + ".base_layer." + suffix
                torch.testing.assert_close(
                    actual_base.get_parameter(name), tensor, rtol=0, atol=0
                )
            final = trainer.curriculum.save_training_state(
                root, model, optimizer, torch, "c" * 64, exposure
            )
            self.assertTrue(any(final[name] != initial[name] for name in final))
            model.eval()
            model.gradient_checkpointing_disable()
            prefix = torch.tensor([[2, 7, 9]])
            generation = {**settings["generation"], "max_new_tokens": 5}
            with torch.inference_mode():
                expected_ids = model.generate(
                    input_ids=prefix,
                    attention_mask=torch.ones_like(prefix),
                    **generation,
                ).tolist()
            code = """
import json,sys,torch
from pathlib import Path
from transformers import Gemma4ForConditionalGeneration
from peft import PeftModel,get_peft_model_state_dict
import train_passive_sft as smoke
torch.set_num_threads(1)
root=Path(sys.argv[1])
model,loading=Gemma4ForConditionalGeneration.from_pretrained(root/'base',dtype=torch.bfloat16,attn_implementation='sdpa',experts_implementation='eager',local_files_only=True,output_loading_info=True)
json.dumps(smoke.loading_diagnostics(loading))
assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
model.requires_grad_(False)
model=PeftModel.from_pretrained(model,root/'adapter',is_trainable=False,local_files_only=True,autocast_adapter_dtype=True).eval()
assert all(p.dtype == (torch.float32 if '.lora_' in n else torch.bfloat16) for n,p in model.named_parameters())
records=smoke.tensor_records(get_peft_model_state_dict(model,save_embedding_layers=False),torch)
assert records == smoke.read(root/'adapter-tensors.json')
prefix=torch.tensor([[2,7,9]])
with torch.inference_mode():
 ids=model.generate(input_ids=prefix,attention_mask=torch.ones_like(prefix),max_new_tokens=5,do_sample=False,use_cache=True,num_beams=1,eos_token_id=[1,106,50],pad_token_id=0,bos_token_id=2).tolist()
smoke.write(root/'fresh.json',{'ids':ids,'tensors':records,'pid':__import__('os').getpid()})
"""
            result = subprocess.run(
                [sys.executable, "-c", code, str(root)],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
                env={**os.environ, "PYTHONPATH": str(Path(trainer.__file__).parent)},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            fresh = smoke.read(root / "fresh.json")
            self.assertEqual(fresh["ids"], expected_ids)
            self.assertEqual(fresh["tensors"], final)
            self.assertNotEqual(fresh["pid"], os.getpid())


if __name__ == "__main__":
    unittest.main()
