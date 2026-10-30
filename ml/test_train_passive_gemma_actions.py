"""Train-only continuation identity, partial batches and real PEFT round trips."""

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import train_passive_gemma_actions as trainer


def configuration(phase="smoke"):
    return {
        "version": trainer.VERSION,
        "candidate_id": "gemma26",
        "contract_version": trainer.CONTRACT_VERSION,
        "phase": phase,
        **{key: "a" * 64 for _, key, _ in trainer.SOURCE_MODULES},
        "libraries": {key: "1.0.0" for key in trainer.smoke.LIBRARIES},
        "absent_libraries": ["torchao"],
        "base": {
            "path": "native",
            "repository": trainer.REPOSITORY,
            "revision": trainer.REVISION,
            "files_sha256": {},
            "absent_files": [],
        },
        "tokenizer": {
            "path": "native",
            "files_sha256": {},
            "template_sha256": "b" * 64,
        },
        "anchor": {
            "path": "anchor/adapter",
            "files_sha256": {
                "adapter_config.json": "c" * 64,
                "adapter_model.safetensors": "d" * 64,
            },
        },
        "bindings": {},
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
            "target_modules": trainer.gemma.target_names(),
            "gpu_name_contains": "Registered GPU",
            "minimum_gpu_bytes": 75000000000,
        },
        "training": {
            **copy.deepcopy(trainer.COMMON_TRAINING),
            **trainer.phase_budget(phase),
            "sample_order": [f"case-{i}" for i in range(8 if phase == "smoke" else 42)],
        },
        "generation": copy.deepcopy(trainer.GENERATION),
    }


class BindingTests(unittest.TestCase):
    def test_minimized_targets_preserve_all_205_native_modules_only(self):
        try:
            from peft.tuners.tuners_utils import _find_minimal_target_modules
        except ImportError as error:
            raise unittest.SkipTest(
                "Use the pinned PEFT CPU test environment."
            ) from error

        targets = trainer.gemma.target_names()
        other = {
            "model.vision_tower.layers.0.self_attn.q_proj",
            "model.audio_tower.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.0.mlp.experts.down_proj",
            "lm_head",
        }
        names = set(targets) | other
        minimized = _find_minimal_target_modules(targets, other)
        self.assertEqual(len(targets), 205)
        self.assertLess(len(minimized), len(targets))
        for representation in (targets, minimized):
            trainer.validate_adapter_targets(
                SimpleNamespace(target_modules=representation), targets, names
            )
        for representation, inventory in (
            (targets[:-1], names),
            ([*targets, "lm_head"], names),
            ([*minimized, "q_proj"], names),
            (minimized, names - {targets[0]}),
        ):
            with (
                self.subTest(targets=representation),
                self.assertRaisesRegex(
                    ValueError, "exact configured native module set"
                ),
            ):
                trainer.validate_adapter_targets(
                    SimpleNamespace(target_modules=representation), targets, inventory
                )

    def test_only_explicit_fixed_anchor_continuation_and_phase_budget_are_valid(self):
        for phase in ("smoke", "substantive"):
            config = configuration(phase)
            trainer.validate_configuration(config)
            for section, key, value in (
                ("training", "initialization", "fresh_native"),
                ("training", "optimizer_state_resumed", True),
                ("training", "sampling_seed", "3407"),
                ("training", "draw_count", 43),
                ("training", "max_updates", 12),
                ("training", "max_total_completion_tokens", 599),
                ("training", "replacement", True),
                ("model", "quantization", "nf4"),
                ("generation", "eos_token_id", [1, 106]),
            ):
                wrong = copy.deepcopy(config)
                wrong[section][key] = value
                with self.subTest(phase=phase, key=key), self.assertRaises(ValueError):
                    trainer.validate_configuration(wrong)

    def test_protocol_binds_both_exact_phases_and_forbids_dev_selection(self):
        config = configuration()
        config["bindings"]["preparation"] = {"sha256": "e" * 64}
        prepared, phases = {}, {}
        for phase, key in (("smoke", "smoke"), ("substantive", "training")):
            value = configuration(phase)["training"]
            prepared[key] = {
                **value,
                "optimizer_updates": value["max_updates"],
                "final_update_rows": 4 if phase == "smoke" else 2,
            }
            phases[phase] = {
                "draw_count": value["draw_count"],
                "updates": value["max_updates"],
                "final_accumulation": prepared[key]["final_update_rows"],
                "sequence_tokens": value["max_total_sequence_tokens"],
                "completion_tokens": value["max_total_completion_tokens"],
                "sample_order": value["sample_order"],
            }
        protocol = {
            "version": "passive-actions-gemma-training-protocol-v1",
            "candidate_id": "gemma26",
            "contract_version": trainer.CONTRACT_VERSION,
            "preparation_sha256": "e" * 64,
            "anchor": {
                name.removeprefix("anchor_") + "_sha256": digest
                for name, digest in trainer.ANCHOR_PINS.items()
            },
            "phases": phases,
            "training_recipe": copy.deepcopy(trainer.COMMON_TRAINING),
            "generation": copy.deepcopy(trainer.GENERATION),
            "label_eos_token_id": 1,
            "development": {
                "during_training": False,
                "case_count": 14,
                "checkpoint_selection": "final_update_11_only",
                "no_auto_extra_training": True,
            },
            "final_test_bodies_read": False,
        }
        trainer.validate_protocol(protocol, config, prepared)
        for mutate in (
            lambda p: p["anchor"].update(configuration_sha256="f" * 64),
            lambda p: p["phases"]["substantive"].update(final_accumulation=4),
            lambda p: p["development"].update(during_training=True),
            lambda p: p.update(label_eos_token_id=106),
        ):
            wrong = copy.deepcopy(protocol)
            mutate(wrong)
            with self.assertRaises(ValueError):
                trainer.validate_protocol(wrong, config, prepared)

    def test_reload_refuses_incomplete_manifest_before_reading_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer.smoke.write(
                root / "manifest.json",
                {
                    "version": trainer.VERSION,
                    "status": "training_complete_fresh_reload_pending",
                    "files_sha256": {"run.json": "a" * 64},
                },
            )
            with self.assertRaisesRegex(ValueError, "every consumed file bound"):
                trainer.verify_previous(root, configuration(), "a" * 64, [])

    def test_substantive_requires_same_anchor_complete_smoke_and_independent_reload(
        self,
    ):
        original, config = configuration(), configuration("substantive")
        for name in (
            *trainer.ANCHOR_BINDINGS,
            "preparation",
            "native_pin",
            "evaluation_protocol",
        ):
            original["bindings"][name] = {"path": name, "sha256": "a" * 64}
        config["bindings"] = copy.deepcopy(original["bindings"])
        config["anchor"]["path"] = "portable-anchor/adapter"
        hashes = {
            name: f"{i:064x}" for i, name in enumerate(trainer.gemma.SMOKE_BINDINGS, 1)
        }
        config["bindings"].update(
            {name: {"path": name, "sha256": sha} for name, sha in hashes.items()}
        )
        run = {
            "status": "training_complete_fresh_reload_pending",
            "phase": "smoke",
            "mode": "train",
            "training_updates": 2,
            "optimizer_committed_rows": 8,
            "initialization": trainer.INITIALIZATION,
            "optimizer_state_resumed": False,
            "configuration_sha256": hashes["smoke_configuration"],
            "pid": 10,
            "finished_unix": 2,
        }
        fresh = {
            "status": "fresh_reload_passed_local_export_pending",
            "mode": "fresh_reload",
            "phase": "smoke",
            "configuration_sha256": hashes["smoke_configuration"],
            "previous_manifest_sha256": hashes["smoke_training_manifest"],
            "reloaded_training_updates": 2,
            "pid": 11,
            "started_unix": 3,
        }
        evidence = {
            "smoke_configuration": original,
            "smoke_run": run,
            "smoke_reload_run": fresh,
            "smoke_training_manifest": {
                "version": trainer.VERSION,
                "status": run["status"],
                "files_sha256": {
                    "configuration.json": hashes["smoke_configuration"],
                    "run.json": hashes["smoke_run"],
                },
            },
            "smoke_reload_manifest": {
                "version": trainer.VERSION,
                "status": fresh["status"],
                "files_sha256": {
                    "run.json": hashes["smoke_reload_run"],
                    "reload-parity.json": hashes["smoke_reload_parity"],
                },
            },
            "smoke_reload_parity": {
                "adapter_tensors_identical": True,
                "optimizer_state_resumed": False,
                "cases": [
                    {"case_id": name, "identical": True}
                    for name in original["training"]["sample_order"]
                ],
            },
        }

        def check(values):
            with patch.object(
                trainer,
                "bound",
                side_effect=lambda binding, root: (
                    Path(binding["path"]),
                    values[binding["path"]],
                ),
            ):
                trainer.validate_smoke_gate(config, Path("/unused"))

        check(evidence)
        for mutate in (
            lambda e: e["smoke_run"].update(initialization="fresh_native"),
            lambda e: e["smoke_run"].update(training_updates=1),
            lambda e: e["smoke_reload_run"].update(pid=10),
            lambda e: e["smoke_reload_run"].update(previous_manifest_sha256="f" * 64),
            lambda e: e["smoke_reload_parity"]["cases"].pop(),
            lambda e: e["smoke_configuration"]["anchor"]["files_sha256"].update(
                {"adapter_model.safetensors": "f" * 64}
            ),
        ):
            wrong = copy.deepcopy(evidence)
            mutate(wrong)
            with self.assertRaises(ValueError):
                check(wrong)


class TinyContinuationTests(unittest.TestCase):
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
                "Use the pinned Gemma CPU test environment."
            ) from error
        cls.torch, cls.Config, cls.TextConfig, cls.Model = (
            torch,
            Gemma4Config,
            Gemma4TextConfig,
            Gemma4ForConditionalGeneration,
        )
        torch.set_num_threads(1)

    def model(self):
        self.torch.manual_seed(3407)
        config = self.Config(
            text_config=self.TextConfig(
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
            ),
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
        config._attn_implementation, config._experts_implementation = "sdpa", "eager"
        return self.Model(config).to(self.torch.bfloat16)

    def rows(self, count):
        return [
            {
                "case_id": f"case-{i}",
                "draw_index": i,
                "source_id": f"source-{i}",
                "source_group_id": "symbolic",
                "episode_id": f"case-{i}",
                "language": "en",
                "action": "delete_span",
                "input_ids": [2, 7 + i, 9, 12, 1],
                "attention_mask": [1] * 5,
                "labels": [-100, -100, -100, 12, 1],
                "completion_start": 3,
            }
            for i in range(count)
        ]

    def test_partial_two_normalization_matches_direct_adamw_and_exact_exposure(self):
        torch = self.torch
        left = torch.nn.Linear(1, 1, bias=False)
        right = copy.deepcopy(left)
        rows = self.rows(42)
        actual = torch.optim.AdamW(left.parameters(), lr=1e-4, weight_decay=0.01)
        expected = torch.optim.AdamW(right.parameters(), lr=1e-4, weight_decay=0.01)

        def loss(model, row, *_):
            return (model.weight.sum() - row["draw_index"] / 42).square()

        for offset in range(0, 42, 4):
            group = rows[offset : offset + 4]
            expected.zero_grad(set_to_none=True)
            sum(loss(right, row) for row in group).div(len(group)).backward()
            torch.nn.utils.clip_grad_norm_(right.parameters(), 1.0)
            expected.step()
        exposure, row_events, updates = trainer.curriculum.Exposure(), [], []
        settings = configuration("substantive")["training"]
        with patch.object(trainer.smoke, "completion_loss", side_effect=loss):
            trainer.curriculum.train_draws(
                left,
                rows,
                actual,
                torch,
                settings,
                "cpu",
                exposure,
                row_events.append,
                updates.append,
            )
        torch.testing.assert_close(left.weight, right.weight, rtol=0, atol=0)
        self.assertEqual([e["accumulation_divisor"] for e in row_events[-2:]], [2, 2])
        self.assertEqual([e["microbatches"] for e in updates], [4] * 10 + [2])
        summary = exposure.summary()
        self.assertEqual(summary["completed_updates"], 11)
        self.assertFalse(summary["optimizer_commit_uncertain"])
        self.assertEqual(summary["pending_backward_rows_not_committed"], 0)
        for phase in trainer.curriculum.PHASES:
            self.assertEqual(summary["phases"][phase]["rows"], 42)
            self.assertEqual(summary["phases"][phase]["unique_cases"], 42)
            self.assertEqual(summary["phases"][phase]["completion_tokens"], 84)
        self.assertEqual(
            {int(state["step"].item()) for state in actual.state.values()}, {11}
        )

    def test_all_205_targets_reload_from_equivalent_suffix_metadata(self):
        from peft import get_peft_model_state_dict
        from peft.tuners.tuners_utils import _find_minimal_target_modules

        torch = self.torch
        tiny = self.model().config
        tiny.text_config.num_hidden_layers = 30
        tiny.text_config.layer_types = (
            ["sliding_attention"] * 5 + ["full_attention"]
        ) * 5
        base = self.Model(tiny).to(dtype=torch.bfloat16)
        config = configuration()
        targets = {
            name: module
            for name, module in base.named_modules()
            if trainer.gemma.MODULE_PATTERN.fullmatch(name)
        }
        self.assertEqual(set(targets), set(trainer.gemma.target_names()))
        config["model"]["expected_trainable_parameters"] = sum(
            2 * (module.in_features + module.out_features)
            for module in targets.values()
        )
        config["training"].update(rank=2, alpha=2)
        names = {name for name, _ in base.named_modules()}
        suffixes = _find_minimal_target_modules(set(targets), names - set(targets))
        self.assertNotEqual(suffixes, set(targets))
        model, _ = trainer.gemma.enable_training(base, config, torch)
        expected = trainer.smoke.tensor_records(
            get_peft_model_state_dict(model, save_embedding_layers=False), torch
        )
        self.assertEqual(len(expected), 410)
        with tempfile.TemporaryDirectory() as temporary:
            model.save_pretrained(temporary, save_embedding_layers=False)
            adapter_path = Path(temporary) / "adapter_config.json"
            adapter = trainer.smoke.read(adapter_path)
            adapter["target_modules"] = sorted(suffixes)
            trainer.smoke.write(adapter_path, adapter)
            reloaded, trainable, actual = trainer.load_adapter(
                self.Model(tiny).to(dtype=torch.bfloat16),
                temporary,
                expected,
                config,
                torch,
                False,
            )
            self.assertEqual(actual, expected)
            self.assertFalse(trainable)
            self.assertFalse(reloaded.training)

    def test_real_saved_anchor_warmstart_fresh_optimizer_and_independent_reload(self):
        torch = self.torch
        from peft import get_peft_model_state_dict

        base = self.model()
        config = configuration()
        targets = {
            n: m
            for n, m in base.named_modules()
            if trainer.gemma.MODULE_PATTERN.fullmatch(n)
        }
        config["model"]["target_modules"] = sorted(targets)
        config["model"]["expected_trainable_parameters"] = sum(
            2 * (m.in_features + m.out_features) for m in targets.values()
        )
        config["training"].update(rank=2, alpha=2, draw_count=6, max_updates=2)
        config["generation"]["max_new_tokens"] = 3
        rows = self.rows(8)
        native = {name: p.detach().clone() for name, p in base.named_parameters()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base.save_pretrained(root / "base")
            anchor, _ = trainer.gemma.enable_training(base, config, torch)
            anchor_optimizer = torch.optim.AdamW(
                [p for p in anchor.parameters() if p.requires_grad], lr=1e-4
            )
            trainer.curriculum.train_draws(
                anchor,
                rows[:6],
                anchor_optimizer,
                torch,
                config["training"],
                "cpu",
                trainer.curriculum.Exposure(),
                lambda _: None,
                lambda _: None,
            )
            anchor.save_pretrained(root / "anchor", save_embedding_layers=False)
            expected = trainer.smoke.tensor_records(
                get_peft_model_state_dict(anchor, save_embedding_layers=False), torch
            )

            fresh_base = self.Model.from_pretrained(
                root / "base",
                dtype=torch.bfloat16,
                local_files_only=True,
                attn_implementation="sdpa",
                experts_implementation="eager",
            )
            model, trainable, initial = trainer.load_adapter(
                fresh_base, root / "anchor", expected, config, torch, True
            )
            self.assertEqual(initial, expected)
            self.assertEqual(
                sum(trainable.values()),
                config["model"]["expected_trainable_parameters"],
            )
            with self.assertRaisesRegex(ValueError, "exactly one adapter"):
                trainer.load_adapter(
                    model, root / "anchor", expected, config, torch, True
                )
            wrong_base = self.Model.from_pretrained(
                root / "base",
                dtype=torch.bfloat16,
                local_files_only=True,
                attn_implementation="sdpa",
                experts_implementation="eager",
            )
            wrong = copy.deepcopy(expected)
            wrong.pop(next(iter(wrong)))
            with self.assertRaisesRegex(ValueError, "exact bound tensor records"):
                trainer.load_adapter(
                    wrong_base, root / "anchor", wrong, config, torch, True
                )

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4,
                weight_decay=0.01,
            )
            self.assertEqual(len(optimizer.state), 0)
            exposure = trainer.curriculum.Exposure()
            trainer.curriculum.train_draws(
                model,
                rows[:6],
                optimizer,
                torch,
                config["training"],
                "cpu",
                exposure,
                lambda _: None,
                lambda _: None,
            )
            saved = root / "saved"
            saved.mkdir()
            final = trainer.curriculum.save_training_state(
                saved, model, optimizer, torch, "a" * 64, exposure
            )
            self.assertNotEqual(initial, final)
            self.assertEqual(
                {int(s["step"].item()) for s in optimizer.state.values()}, {2}
            )
            for name, parameter in model.get_base_model().named_parameters():
                if ".lora_" not in name:
                    self.assertFalse(parameter.requires_grad)
                    self.assertEqual(parameter.dtype, torch.bfloat16)
                    torch.testing.assert_close(
                        parameter,
                        native[name.replace(".base_layer", "")],
                        rtol=0,
                        atol=0,
                    )
            tokenizer = type(
                "Tokenizer", (), {"decode": lambda _, ids, **kwargs: str(ids)}
            )()
            outputs = trainer.gemma.generate_smoke(
                model, tokenizer, rows, config, torch, "cpu"
            )
            trainer.smoke.write(root / "settings.json", config)
            trainer.smoke.write(root / "rows.json", rows)
            script = """
import json,os,sys
from pathlib import Path
import torch
from transformers import Gemma4ForConditionalGeneration
import train_passive_gemma_actions as t
torch.set_num_threads(1)
r=Path(sys.argv[1]); c=t.smoke.read(r/'settings.json')
m=Gemma4ForConditionalGeneration.from_pretrained(r/'base',dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',experts_implementation='eager')
m,trainable,records=t.load_adapter(m,r/'saved/adapter',t.smoke.read(r/'saved/adapter-tensors.json'),c,torch,False)
assert not trainable
tokenizer=type('Tokenizer',(),{'decode':lambda _,ids,**kwargs:str(ids)})()
outputs=t.gemma.generate_smoke(m,tokenizer,t.smoke.read(r/'rows.json'),c,torch,'cpu')
t.smoke.write(r/'child.json',{'pid':os.getpid(),'records':records,'outputs':outputs})
"""
            environment = {
                **os.environ,
                "PYTHONPATH": str(Path(trainer.__file__).parent),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OMP_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script, str(root)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            child = trainer.smoke.read(root / "child.json")
            self.assertNotEqual(child["pid"], os.getpid())
            self.assertEqual(child["records"], final)
            self.assertEqual(child["outputs"], outputs)


if __name__ == "__main__":
    unittest.main()
