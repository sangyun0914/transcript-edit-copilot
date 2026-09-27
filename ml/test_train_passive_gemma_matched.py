import copy
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_train_passive_gemma as gemma_tests
import test_train_passive_sft as smoke_tests
import train_passive_gemma_matched as trainer


smoke, curriculum = trainer.smoke, trainer.curriculum
RELEASE = "d" * 64
T1_PROTOCOL = "e" * 64


def configuration(fmt="json", phase="substantive"):
    original = gemma_tests.configuration("substantive")
    config = copy.deepcopy(original)
    config.update(version=trainer.VERSION, format=fmt, phase=phase)
    config.update({key: "a" * 64 for _, key, _ in trainer.SOURCE_MODULES})
    count = 8 if phase == "smoke" else 802
    config["training"].update(
        max_updates=2 if phase == "smoke" else 201,
        draw_count=count,
        sample_order=[f"case-{i}" for i in range(count)],
        max_total_sequence_tokens=count * (5 if fmt == "json" else 6),
        max_total_completion_tokens=count * (2 if fmt == "json" else 3),
        recovery_updates=[] if phase == "smoke" else [50, 100, 150],
    )
    return config, original


def protocol(config):
    return {
        "version": "passive-gemma-matched-evaluation-v1",
        "candidate_id": "gemma26",
        "formats": ["json", "rewrite"],
        "training_rows": 802,
        "registered_requests_per_condition": 286,
        "same_case_sampling_order_required": True,
        "training_plan": {
            "candidate": "gemma26",
            "draw_count": 802,
            "sampling_seed": "3407",
            "model_seed": 3407,
            "effective_batch": 4,
            "updates": 201,
            "final_accumulation": 2,
            "independent_fresh_native_runs": True,
            "development_during_training": False,
            "checkpoint_selection": "final_update_201_only",
            "recovery_updates": [50, 100, 150],
            "max_wall_seconds": 21600,
        },
        "numerical_regime": {
            "base_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "quantization": "none",
        },
        "engineering_smoke": {
            "draw_count": 8,
            "updates": 2,
            "fresh_reload_required": True,
            "one_smoke_per_format": True,
            "actual_peak_memory_must_be_measured": True,
            "native_gpu": config["model"]["gpu_name_contains"],
            "minimum_gpu_bytes": config["model"]["minimum_gpu_bytes"],
        },
        "b2": {"release_manifest": {"sha256": RELEASE}},
        "t1_protocol": {"sha256": T1_PROTOCOL},
        "t1_evidence": {name: config["bindings"][name] for name in trainer.T1_BINDINGS},
        "conditions": [
            "matched-json-unconstrained",
            "matched-rewrite-unconstrained",
            "matched-json-constrained",
        ],
        "primary_format_comparison": {
            "left": "matched-json-unconstrained",
            "right": "matched-rewrite-unconstrained",
        },
        "generation_policy": {
            **config["generation"],
            "label_eos_token_id": 1,
            "thinking": False,
            "raw_tokens_and_terminal_id_retained": True,
        },
        "final_test_bodies_read": False,
        "learning_signal_policy": {
            "no_auto_extra_training": True,
            "deployment_promotion": False,
        },
    }


def bind(root, config, name, value):
    path = root / f"{name}.json"
    smoke.write(path, value)
    config["bindings"][name] = {"path": path.name, "sha256": smoke.digest(path)}
    return smoke.digest(path)


def t1_evidence(root, config, original, *, updates=203, observed=100):
    original_sha = bind(root, config, "t1_configuration", original)
    run = {
        "version": trainer.gemma.VERSION,
        "status": "training_complete_fresh_reload_pending",
        "phase": "substantive",
        "mode": "train",
        "training_updates": updates,
        "optimizer_committed_rows": 809,
        "configuration_sha256": original_sha,
        "release_manifest_sha256": RELEASE,
        "evaluation_protocol_sha256": T1_PROTOCOL,
        "native_pin_sha256": original["bindings"]["native_pin"]["sha256"],
        "preparation_sha256": original["bindings"]["preparation"]["sha256"],
        "trainer_sha256": config["gemma_trainer_sha256"],
        "initialization": "original_native_base_with_fresh_lora",
        "optimizer_state_resumed": False,
        "final_test_bodies_read": 0,
        "pid": 10,
        "started_unix": 1,
        "finished_unix": 2,
    }
    run_sha = bind(root, config, "t1_run", run)
    train_sha = bind(
        root,
        config,
        "t1_training_manifest",
        {
            "version": trainer.gemma.VERSION,
            "status": run["status"],
            "files_sha256": {"run.json": run_sha, "configuration.json": original_sha},
        },
    )
    re_run = {
        "version": trainer.gemma.VERSION,
        "status": "fresh_reload_passed_local_export_pending",
        "phase": "substantive",
        "mode": "fresh_reload",
        "reloaded_training_updates": 203,
        "configuration_sha256": original_sha,
        "previous_manifest_sha256": train_sha,
        "evaluation_protocol_sha256": T1_PROTOCOL,
        "optimizer_state_resumed": False,
        "pid": 11,
        "started_unix": 3,
    }
    re_sha = bind(root, config, "t1_reload_run", re_run)
    parity_sha = bind(
        root,
        config,
        "t1_reload_parity",
        {
            "adapter_tensors_identical": True,
            "optimizer_state_resumed": False,
            "cases": [{"case_id": f"case-{i}", "identical": True} for i in range(8)],
        },
    )
    re_manifest_sha = bind(
        root,
        config,
        "t1_reload_manifest",
        {
            "version": trainer.gemma.VERSION,
            "status": re_run["status"],
            "files_sha256": {
                "run.json": re_sha,
                "configuration.json": original_sha,
                "reload-parity.json": parity_sha,
            },
        },
    )
    assets = {
        name: {"path": name + ".gguf", "sha256": str(i) * 64, "bytes": i + 1}
        for i, name in enumerate(
            ("base", "adapter", "merged", "base_quantized", "merged_quantized")
        )
    }
    export_sha = bind(
        root,
        config,
        "t1_export_receipt",
        {
            "version": "passive-gemma-sft-export-v1",
            "status": "tensor_verified_paired_export_runtime_preflight_pending",
            "training_record": {"sha256": run_sha},
            "base_repository": trainer.REPOSITORY,
            "base_revision": trainer.REVISION,
            "native_generation_stop_ids": [1, 106, 50],
            "artifacts": assets,
        },
    )
    execution_sha = bind(
        root,
        config,
        "t1_evaluation_execution",
        {
            "version": "passive-gemma-execution-v1",
            "protocol_sha256": T1_PROTOCOL,
            "export_receipt": {"sha256": export_sha},
            "training_configuration": {"sha256": original_sha},
            "fresh_reload": {"sha256": re_manifest_sha},
        },
    )
    counts = {
        "native-base-full": 289,
        "native-base-icl": 100,
        "sft-full": 289,
        "sft-icl": 100,
    }
    summary = {
        "version": "passive-gemma-evaluation-v1",
        "protocol_sha256": T1_PROTOCOL,
        "execution_sha256": execution_sha,
        "registered_new_requests": 778,
        "registered_conditions": list(counts),
        "paired_export": {
            "artifacts": {
                "native_base": assets["base_quantized"],
                "sft": assets["merged_quantized"],
            }
        },
        "runs": {
            name: {
                "run_status": "completed",
                "expected_fresh_requests": count,
                "observed_fresh_results": count,
                "missing_requests": 0,
                "request_errors": 0,
            }
            for name, count in counts.items()
        },
        "provenance": {
            name: {"status": "completed", "receipt_sha256": "9" * 64} for name in counts
        },
        "learning_signal": {
            "full": {"classification": "criteria_not_met"},
            "icl": {"classification": "criteria_not_met"},
        },
    }
    summary["runs"]["sft-icl"]["observed_fresh_results"] = observed
    summary_sha = bind(root, config, "t1_evaluation_summary", summary)
    bind(
        root,
        config,
        "t1_evaluation_verification",
        {
            "version": "passive-gemma-final-evaluation-verification-v1",
            "status": "passed",
            "registered_requests": 778,
            "completed_verified_requests": 778,
            "root_exec_exit_code": 0,
            "final_report": {"sha256": summary_sha},
            "final_test_bodies_read": 0,
            "deployment_promotion": False,
            "condition_checks": [
                {"condition": name, "requests": count, "receipt": {"sha256": "9" * 64}}
                for name, count in counts.items()
            ],
        },
    )
    bind(root, config, "evaluation_protocol", protocol(config))


def smoke_evidence(root, config, *, fmt=None, updates=2, reload_pid=13):
    original = copy.deepcopy(config)
    original["phase"] = "smoke"
    original["format"] = fmt or config["format"]
    original["training"].update(
        max_updates=2,
        draw_count=8,
        sample_order=[f"case-{i}" for i in range(8)],
        recovery_updates=[],
        max_total_sequence_tokens=40,
        max_total_completion_tokens=16,
    )
    original_sha = bind(root, config, "smoke_configuration", original)
    run = {
        "version": trainer.VERSION,
        "status": "training_complete_fresh_reload_pending",
        "phase": "smoke",
        "format": original["format"],
        "mode": "train",
        "training_updates": updates,
        "optimizer_committed_rows": 8,
        "consumed_rows": 8,
        "configuration_sha256": original_sha,
        "release_manifest_sha256": RELEASE,
        "preparation_sha256": config["bindings"]["preparation"]["sha256"],
        "native_pin_sha256": config["bindings"]["native_pin"]["sha256"],
        "evaluation_protocol_sha256": config["bindings"]["evaluation_protocol"][
            "sha256"
        ],
        "optimizer_state_resumed": False,
        "final_test_bodies_read": 0,
        "peak_allocated_gpu_bytes": 100,
        "peak_reserved_gpu_bytes": 120,
        "changed_adapter_tensors": ["lora_B"],
        "pid": 12,
        "finished_unix": 20,
    }
    run_sha = bind(root, config, "smoke_run", run)
    train_sha = bind(
        root,
        config,
        "smoke_training_manifest",
        {
            "version": trainer.VERSION,
            "status": run["status"],
            "files_sha256": {"run.json": run_sha, "configuration.json": original_sha},
        },
    )
    re_run = {
        "version": trainer.VERSION,
        "status": "fresh_reload_passed_local_export_pending",
        "phase": "smoke",
        "format": original["format"],
        "mode": "fresh_reload",
        "reloaded_training_updates": 2,
        "configuration_sha256": original_sha,
        "previous_manifest_sha256": train_sha,
        "optimizer_state_resumed": False,
        "pid": reload_pid,
        "started_unix": 21,
    }
    re_sha = bind(root, config, "smoke_reload_run", re_run)
    parity_sha = bind(
        root,
        config,
        "smoke_reload_parity",
        {
            "adapter_tensors_identical": True,
            "optimizer_state_resumed": False,
            "cases": [
                {"case_id": name, "identical": True}
                for name in original["training"]["sample_order"]
            ],
        },
    )
    bind(
        root,
        config,
        "smoke_reload_manifest",
        {
            "version": trainer.VERSION,
            "status": re_run["status"],
            "files_sha256": {
                "run.json": re_sha,
                "configuration.json": original_sha,
                "reload-parity.json": parity_sha,
            },
        },
    )


def prepared_bundle():
    cases = [
        {
            "id": f"case-{i}",
            "metadata": {
                "source_id": f"source-{i % 8}",
                "source_group_id": f"group-{i % 8}",
                "episode_id": f"episode-{i % 11}",
                "language": "en" if i % 2 else "ko",
                "split": "train",
                "training_eligible": True,
                "sampling_weight": 1.0,
            },
            "expectedOutput": {"action": "no_action"},
        }
        for i in range(802)
    ]
    indices = trainer.gemma.curriculum_preparation.weighted_draw_indices(
        cases, 802, "3407"
    )
    plan = [
        {"draw_index": pos, "case_id": cases[index]["id"], "train_row_index": index}
        for pos, index in enumerate(indices)
    ]
    views, view_config = {}, {}
    for fmt in trainer.FORMATS:
        tokens = []
        for case in cases:
            completion = [9, 1] if fmt == "json" else [9, 10, 1]
            primary = {
                "input_ids": [2, 3, 4] + completion,
                "attention_mask": [1] * (3 + len(completion)),
                "labels": [-100] * 3 + completion,
                "completion_start": 3,
            }
            tokens.append({"id": case["id"], "primary": primary})
        rows = [{"case_id": r["id"], **r["primary"]} for r in tokens[:8]]
        views[fmt] = {
            "tokens": tokens,
            "diagnostics": {"cases": cases[:8], "tokens": tokens[:8], "rows": rows},
        }
        view_config[fmt] = {
            "training_view": fmt,
            "max_total_sequence_tokens": 802 * len(tokens[0]["primary"]["input_ids"]),
            "max_total_completion_tokens": 802 * len(completion),
            "diagnostic_case_order": [r["case_id"] for r in rows],
            "diagnostic_sequence_tokens": 8 * len(tokens[0]["primary"]["input_ids"]),
            "diagnostic_completion_tokens": 8 * len(completion),
        }
    return {
        "cases": cases,
        "draw_plan": plan,
        "views": views,
        "configuration": {
            "candidate_id": "gemma26",
            "frozen_train_rows": 802,
            "native_base": {
                "repository": trainer.REPOSITORY,
                "revision": trainer.REVISION,
            },
            "tokenizer": {"files_sha256": {}, "eos_token_id": 1, "vocab_size": 262144},
            "sampling": {
                "algorithm": "weighted_indices_v2",
                "seed": "3407",
                "seed_type": "str",
                "replacement": True,
                "draw_count": 802,
                "weights_applied_to_loss": False,
            },
            "generation_metadata": {
                "completion_label_eos_token_id": 1,
                "native_generation_stop_ids": [1, 106, 50],
                "bos_token_id": 2,
                "pad_token_id": 0,
                "actual_generation_policy_verified_by_preparation": False,
            },
            "training": {
                "micro_batch_size": 1,
                "gradient_accumulation": 4,
                "optimizer_updates": 201,
                "final_update_rows": 2,
                "final_update_normalization": "mean_over_actual_microbatch_count",
                "sample_order": [r["case_id"] for r in plan],
                "initialization": "fresh_adapter_on_exact_native_base_not_smoke_or_T1_adapter",
            },
            "views": view_config,
        },
    }


def select(config, bundle):
    view = bundle["configuration"]["views"][config["format"]]
    is_smoke = config["phase"] == "smoke"
    config["training"].update(
        sample_order=view["diagnostic_case_order"]
        if is_smoke
        else bundle["configuration"]["training"]["sample_order"],
        max_total_sequence_tokens=view[
            "diagnostic_sequence_tokens" if is_smoke else "max_total_sequence_tokens"
        ],
        max_total_completion_tokens=view[
            "diagnostic_completion_tokens"
            if is_smoke
            else "max_total_completion_tokens"
        ],
    )
    return trainer.validate_prepared(config, bundle)


class MatchedBindingTests(unittest.TestCase):
    def test_static_import_needs_only_stdlib(self):
        code = "import sys; import train_passive_gemma_matched; assert not {'torch','transformers','peft'} & set(sys.modules)"
        subprocess.run(
            [sys.executable, "-B", "-S", "-c", code],
            cwd=Path(trainer.__file__).parent,
            check=True,
            capture_output=True,
        )

    def test_fixed_native_recipe_and_phase_arithmetic(self):
        for fmt in trainer.FORMATS:
            for phase in ("smoke", "substantive"):
                config, original = configuration(fmt, phase)
                trainer.validate_configuration(config)
                trainer.validate_configuration(config, original)
                for section, key, value in (
                    ("training", "max_updates", 202),
                    ("training", "sampling_seed", 3407),
                    ("training", "learning_rate", 0.0002),
                    ("training", "fresh_native_initialization", False),
                    ("model", "dtype", "float32"),
                    ("model", "quantization", "nf4"),
                    ("generation", "eos_token_id", [1, 106]),
                ):
                    changed = copy.deepcopy(config)
                    changed[section][key] = value
                    with (
                        self.subTest(fmt=fmt, phase=phase, key=key),
                        self.assertRaises(ValueError),
                    ):
                        trainer.validate_configuration(changed, original)

    def test_t1_metadata_quality_failure_is_allowed_but_incomplete_lineage_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, original = configuration()
            t1_evidence(root, config, original)
            trainer.validate_t1_evidence(config, root, RELEASE)
            trainer.validate_protocol(
                smoke.read(root / "evaluation_protocol.json"), config, RELEASE
            )
            for changes in ({"updates": 202}, {"observed": 99}):
                t1_evidence(root, config, original, **changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    trainer.validate_t1_evidence(config, root, RELEASE)
            t1_evidence(root, config, original)
            (root / "t1_evaluation_verification.json").write_text("{}")
            with self.assertRaises(ValueError):
                trainer.validate_t1_evidence(config, root, RELEASE)

    def test_substantive_requires_same_format_completed_smoke_and_later_independent_reload(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, original = configuration()
            t1_evidence(root, config, original)
            order = [f"case-{i}" for i in range(8)]
            smoke_evidence(root, config)
            trainer.validate_smoke_gate(config, root, RELEASE, order)
            for changes in ({"fmt": "rewrite"}, {"updates": 1}, {"reload_pid": 12}):
                smoke_evidence(root, config, **changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    trainer.validate_smoke_gate(config, root, RELEASE, order)

    def test_protocol_cannot_extend_updates_drop_a_format_smoke_or_enable_dev_selection(
        self,
    ):
        config, _ = configuration()
        config["bindings"].update(
            {name: {"path": name, "sha256": "a" * 64} for name in trainer.T1_BINDINGS}
        )
        original = protocol(config)
        trainer.validate_protocol(original, config, RELEASE)
        for section, key, value in (
            ("training_plan", "updates", 202),
            ("training_plan", "development_during_training", True),
            ("engineering_smoke", "one_smoke_per_format", False),
            ("generation_policy", "eos_token_id", [1, 106]),
        ):
            changed = copy.deepcopy(original)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                trainer.validate_protocol(changed, config, RELEASE)

    def test_both_formats_select_same_draws_and_eight_diagnostics_with_distinct_tokens(
        self,
    ):
        bundle = prepared_bundle()
        result = {}
        for fmt in trainer.FORMATS:
            for phase, count in (("smoke", 8), ("substantive", 802)):
                config, _ = configuration(fmt, phase)
                rows, diagnostics = select(config, bundle)
                self.assertEqual(len(rows), count)
                self.assertEqual(len(diagnostics), 8)
                self.assertEqual(rows[-1]["input_ids"][-1], 1)
                result[fmt, phase] = rows
        for phase in ("smoke", "substantive"):
            self.assertEqual(
                [r["case_id"] for r in result["json", phase]],
                [r["case_id"] for r in result["rewrite", phase]],
            )
            self.assertNotEqual(
                result["json", phase][0]["labels"],
                result["rewrite", phase][0]["labels"],
            )
        config, _ = configuration("rewrite")
        select(config, bundle)
        changed = copy.deepcopy(bundle)
        chosen = changed["draw_plan"][0]["train_row_index"]
        changed["views"]["rewrite"]["tokens"][chosen]["primary"]["labels"][0] = 2
        with self.assertRaisesRegex(ValueError, "mask"):
            trainer.validate_prepared(config, changed)
        changed = copy.deepcopy(bundle)
        changed["configuration"]["views"]["rewrite"]["diagnostic_case_order"] = list(
            reversed(
                changed["configuration"]["views"]["rewrite"]["diagnostic_case_order"]
            )
        )
        with self.assertRaisesRegex(ValueError, "diagnostics"):
            trainer.validate_prepared(config, changed)

    def test_reload_checks_full_inventory_sources_format_and_committed_exposure(self):
        config, _ = configuration(phase="smoke")
        rows, diagnostics = select(config, prepared_bundle())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, key, source in trainer.SOURCE_MODULES:
                (root / name).write_bytes(Path(source).read_bytes())
                config[key] = smoke.digest(root / name)
            smoke.write(root / "configuration.json", config)
            config_sha = smoke.digest(root / "configuration.json")
            exposure = curriculum.Exposure()
            for row in rows:
                for phase in curriculum.PHASES:
                    exposure.record(phase, row)
            exposure.updates = 2
            smoke.write(root / "exposure.json", exposure.summary())
            run = {
                "configuration_sha256": config_sha,
                "status": "training_complete_fresh_reload_pending",
                "mode": "train",
                "phase": "smoke",
                "format": "json",
                "stage": "matched_format_sft",
                "pid": os.getpid() + 1,
                "finished_unix": 1,
                "optimizer_state_resumed": False,
                "initialization": "original_native_base_with_fresh_lora",
                "training_updates": 2,
                "optimizer_committed_rows": 8,
            }
            smoke.write(root / "run.json", run)
            smoke.write(
                root / "smoke-generation.json",
                [{"case_id": r["case_id"]} for r in diagnostics],
            )
            for name in (
                "runtime.json",
                "adapter/adapter_config.json",
                "adapter/adapter_model.safetensors",
                "adapter-tensors.json",
                "training-state.pt",
                "initialization.json",
                "training-rows.jsonl",
                "training-events.jsonl",
            ):
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"unit-test-placeholder")
            trainer.manifest(root, run["status"])
            trainer.verify_previous(root, config, config_sha, rows, diagnostics)
            (root / "unbound.txt").write_text("must not be trusted")
            with self.assertRaisesRegex(ValueError, "inventory"):
                trainer.verify_previous(root, config, config_sha, rows, diagnostics)
            (root / "unbound.txt").unlink()
            run["format"] = "rewrite"
            smoke.write(root / "run.json", run)
            trainer.manifest(root, run["status"])
            with self.assertRaisesRegex(ValueError, "exact fully committed phase"):
                trainer.verify_previous(root, config, config_sha, rows, diagnostics)
            run["format"] = "json"
            smoke.write(root / "run.json", run)
            changed = exposure.summary()
            changed["phases"]["backward"]["counts"]["episode_id"] = {"different": 8}
            smoke.write(root / "exposure.json", changed)
            trainer.manifest(root, run["status"])
            with self.assertRaisesRegex(ValueError, "Source/episode"):
                trainer.verify_previous(root, config, config_sha, rows, diagnostics)

    def test_checkpoint_preserves_new_version_refuses_overwrite_and_uncertain_commit(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exposure = curriculum.Exposure()
            with patch.object(curriculum, "save_training_state"):
                trainer.save_checkpoint(
                    root,
                    "checkpoint-000050",
                    None,
                    None,
                    None,
                    "a" * 64,
                    exposure,
                    "unit_test",
                )
                self.assertEqual(
                    smoke.read(root / "checkpoint-000050/manifest.json")["version"],
                    trainer.VERSION,
                )
                with self.assertRaises(FileExistsError):
                    trainer.save_checkpoint(
                        root,
                        "checkpoint-000050",
                        None,
                        None,
                        None,
                        "a" * 64,
                        exposure,
                        "unit_test",
                    )
                exposure.optimizer_commit_uncertain = True
                with self.assertRaisesRegex(ValueError, "interrupted optimizer"):
                    trainer.save_checkpoint(
                        root,
                        "uncertain",
                        None,
                        None,
                        None,
                        "a" * 64,
                        exposure,
                        "unit_test",
                    )
                self.assertFalse((root / "uncertain").exists())


class MatchedGradientTests(unittest.TestCase):
    def test_802_draws_match_manual_adamw_and_final_two_row_mean(self):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(
                "Use the existing Gemma CPU test environment."
            ) from error
        torch.set_num_threads(1)
        fixture = smoke_tests.SmokeLossTests()
        fixture.torch = torch
        actual, expected = fixture.model(), fixture.model()
        bundle = prepared_bundle()
        config, _ = configuration("rewrite")
        rows, _ = select(config, bundle)
        left = torch.optim.AdamW(actual.parameters(), lr=1e-4, weight_decay=0.01)
        right = torch.optim.AdamW(expected.parameters(), lr=1e-4, weight_decay=0.01)
        exposure, row_events, updates = curriculum.Exposure(), [], []
        curriculum.train_draws(
            actual,
            rows,
            left,
            torch,
            config["training"],
            "cpu",
            exposure,
            row_events.append,
            updates.append,
        )
        for offset in range(0, 802, 4):
            group = rows[offset : offset + 4]
            right.zero_grad(set_to_none=True)
            sum(
                smoke.completion_loss(expected, row, torch, "cpu") / len(group)
                for row in group
            ).backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
            right.step()
        for a, b in zip(actual.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        self.assertEqual(len(updates), 201)
        self.assertEqual([u["microbatches"] for u in updates], [4] * 200 + [2])
        self.assertEqual([r["draw_index"] for r in row_events], list(range(802)))
        self.assertEqual([r["accumulation_divisor"] for r in row_events[-2:]], [2, 2])
        summary = exposure.summary()
        self.assertFalse(summary["optimizer_commit_uncertain"])
        self.assertEqual(summary["pending_backward_rows_not_committed"], 0)
        for phase in curriculum.PHASES:
            self.assertEqual(
                summary["phases"][phase]["counts"]["case_id"],
                dict(Counter(config["training"]["sample_order"])),
            )
            self.assertEqual(
                summary["phases"][phase]["sequence_tokens"],
                config["training"]["max_total_sequence_tokens"],
            )
            self.assertEqual(
                summary["phases"][phase]["completion_tokens"],
                config["training"]["max_total_completion_tokens"],
            )


if __name__ == "__main__":
    unittest.main()
