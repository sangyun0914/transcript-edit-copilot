import copy
from pathlib import Path
import subprocess
import sys
import unittest

import test_train_passive_curriculum as curriculum_tests
import test_train_passive_sft as smoke_tests
import train_passive_matched as trainer
import train_passive_sft as smoke


def protocol():
    return {
        "version": "passive-matched-evaluation-v1",
        "formats": ["json", "rewrite"],
        "b2": {"release_manifest": {"sha256": "a" * 64}},
        "t1_protocol": {"sha256": "b" * 64},
        "training_plan": {
            "candidate": "qwen35-9b",
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
        "conditions": [
            "matched-json-unconstrained",
            "matched-rewrite-unconstrained",
            "matched-json-constrained",
        ],
        "primary_format_comparison": {
            "left": "matched-json-unconstrained",
            "right": "matched-rewrite-unconstrained",
        },
        "registered_requests_per_condition": 286,
        "training_rows": 802,
        "same_case_sampling_order_required": True,
        "final_test_bodies_read": False,
        "learning_signal_policy": {"no_auto_extra_training": True, "deployment_promotion": False},
        "launch_gate": {
            "t1_final_updates": 203,
            "t1_committed_draws": 809,
            "fresh_reload_adapter_and_outputs": "exact",
            "paired_export": "tensor_verified_paired_export_runtime_preflight_pending",
            "local_preflight_status": "template_preflight_passed_generation_unrun",
            "full_t1_development_evaluation_may_run_in_parallel": True,
        },
    }


class MatchedGatesTests(unittest.TestCase):
    def setUp(self):
        fixture = smoke_tests.SmokePreflightTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.original = fixture.config
        self.config = copy.deepcopy(self.original)
        self.config.update(
            version=trainer.VERSION, format="json", curriculum_trainer_sha256="c" * 64
        )
        self.config["training"].update(
            sampling_seed="3407",
            max_updates=201,
            draw_count=802,
            sample_order=["fixed-draw"] * 802,
            final_partial_accumulation="actual_microbatch_count",
            fresh_native_initialization=True,
            recovery_updates=[50, 100, 150],
            wall_limit_seconds=21600,
        )
        self.config["smoke_trainer_sha256"] = self.original["trainer_sha256"]

    def test_both_formats_require_the_fixed_201_update_fresh_native_recipe(self):
        for fmt in trainer.FORMATS:
            self.config["format"] = fmt
            trainer.validate_configuration(self.config, self.original)
        for key, value in (
            ("max_updates", 202),
            ("max_updates", 203),
            ("draw_count", 809),
            ("sampling_seed", 3407),
            ("learning_rate", 0.0002),
            ("fresh_native_initialization", False),
            ("recovery_updates", [10, 20]),
        ):
            changed = copy.deepcopy(self.config)
            changed["training"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                trainer.validate_configuration(changed, self.original)
        changed = copy.deepcopy(self.config)
        changed["format"] = "json-curriculum"
        with self.assertRaisesRegex(ValueError, "matched format"):
            trainer.validate_configuration(changed, self.original)

    def test_protocol_keeps_same_constraints_and_corrected_budget(self):
        original = protocol()
        trainer.validate_protocol(original, self.config, "a" * 64)
        for section, key, value in (
            ("training_plan", "updates", 202),
            ("training_plan", "development_during_training", True),
            ("training_plan", "independent_fresh_native_runs", False),
            ("primary_format_comparison", "left", "matched-json-constrained"),
            ("launch_gate", "local_preflight_status", "prepared_generation_unrun"),
        ):
            changed = copy.deepcopy(original)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "protocol differs"):
                trainer.validate_protocol(changed, self.config, "a" * 64)

    def evidence(self):
        hashes = {name: f"{index:064x}" for index, name in enumerate(trainer.T1_BINDINGS, 1)}
        config_sha = "d" * 64
        values = {
            "t1_training_manifest": {
                "version": trainer.curriculum.VERSION,
                "status": "training_complete_fresh_reload_pending",
                "files_sha256": {"run.json": hashes["t1_run"], "configuration.json": config_sha},
            },
            "t1_run": {
                "version": trainer.curriculum.VERSION,
                "status": "training_complete_fresh_reload_pending",
                "mode": "train",
                "training_updates": 203,
                "optimizer_committed_rows": 809,
                "configuration_sha256": config_sha,
                "pid": 123,
                "trainer_sha256": self.config["curriculum_trainer_sha256"],
                "smoke_trainer_sha256": self.config["smoke_trainer_sha256"],
                "release_manifest_sha256": "a" * 64,
                "evaluation_protocol_sha256": "b" * 64,
                "final_test_bodies_read": 0,
            },
            "t1_reload_manifest": {
                "version": trainer.curriculum.VERSION,
                "status": "fresh_reload_passed_local_export_pending",
                "files_sha256": {
                    "run.json": hashes["t1_reload_run"],
                    "reload-parity.json": hashes["t1_reload_parity"],
                },
            },
            "t1_reload_run": {
                "version": trainer.curriculum.VERSION,
                "status": "fresh_reload_passed_local_export_pending",
                "mode": "fresh_reload",
                "previous_manifest_sha256": hashes["t1_training_manifest"],
                "configuration_sha256": config_sha,
                "evaluation_protocol_sha256": "b" * 64,
                "pid": 124,
                "reloaded_training_updates": 203,
                "optimizer_state_resumed": False,
            },
            "t1_reload_parity": {
                "optimizer_state_resumed": False,
                "adapter_tensors_identical": True,
                "cases": [{"case_id": f"case-{index}", "identical": True} for index in range(8)],
            },
            "t1_export_receipt": {
                "version": "passive-sft-export-v1",
                "status": "tensor_verified_paired_export_runtime_preflight_pending",
                "training_record": {"sha256": hashes["t1_run"]},
                "base_repository": self.config["base"]["repository"],
                "base_revision": self.config["base"]["revision"],
                "artifacts": dict.fromkeys(
                    ("base", "adapter", "merged", "base_quantized", "merged_quantized")
                ),
            },
            "t1_local_preflight": {
                "version": "passive-curriculum-evaluation-v1",
                "condition": "sft-full",
                "model_role": "sft",
                "status": "template_preflight_passed_generation_unrun",
                "error": None,
                "export_receipt_sha256": hashes["t1_export_receipt"],
                "training_configuration_sha256": config_sha,
                "fresh_reload_manifest_sha256": hashes["t1_reload_manifest"],
                "protocol_sha256": "b" * 64,
                "registered_requests": 289,
                "requested_cases": 289,
                "files_sha256": {"manifest.json": "e" * 64},
            },
        }
        return values, hashes

    def test_launch_requires_exact_t1_chain_and_actual_template_preflight(self):
        evidence, hashes = self.evidence()
        trainer.validate_t1_evidence(self.config, protocol(), evidence, hashes, "a" * 64)
        for name, key, value in (
            ("t1_run", "training_updates", 2),
            ("t1_reload_run", "previous_manifest_sha256", "f" * 64),
            ("t1_reload_run", "configuration_sha256", "f" * 64),
            ("t1_reload_run", "pid", evidence["t1_run"]["pid"]),
            ("t1_reload_parity", "adapter_tensors_identical", False),
            ("t1_export_receipt", "base_revision", "different-revision"),
            ("t1_local_preflight", "status", "prepared_generation_unrun"),
            ("t1_local_preflight", "status", "completed"),
            ("t1_local_preflight", "export_receipt_sha256", "f" * 64),
        ):
            changed = copy.deepcopy(evidence)
            changed[name][key] = value
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                trainer.validate_t1_evidence(self.config, protocol(), changed, hashes, "a" * 64)

    def test_import_and_binding_validation_do_not_import_training_libraries(self):
        script = (
            "import sys; import train_passive_matched; "
            "assert not any(name in sys.modules for name in ('torch','transformers','peft','bitsandbytes'))"
        )
        subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).parent, check=True)


class MatchedUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest("Use the existing ML environment for CPU gradients.") from error
        cls.torch = torch

    def fixture(self):
        fixture = curriculum_tests.CurriculumUpdateTests()
        fixture.torch = self.torch
        return fixture

    def test_actual_802_draws_make_201_updates_and_202_is_rejected_before_forward(self):
        torch, fixture = self.torch, self.fixture()
        rows, model = fixture.rows(802), fixture.model()
        settings = {
            "draw_count": 802,
            "max_updates": 201,
            "gradient_accumulation": 4,
            "max_grad_norm": 1.0,
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        exposure, events, updates = trainer.Exposure(), [], []
        trainer.train_draws(
            model, rows, optimizer, torch, settings, "cpu", exposure, events.append, updates.append
        )
        self.assertEqual(len(events), 802)
        self.assertEqual(exposure.updates, 201)
        self.assertEqual([event["microbatches"] for event in updates], [4] * 200 + [2])
        self.assertEqual([event["accumulation_divisor"] for event in events[-2:]], [2, 2])
        self.assertEqual(exposure.phases["optimizer_committed"]["rows"], 802)
        self.assertFalse(exposure.optimizer_commit_uncertain)
        rejected = []
        with self.assertRaisesRegex(ValueError, "Draw/update budget mismatch"):
            trainer.train_draws(
                model,
                rows,
                optimizer,
                torch,
                {**settings, "max_updates": 202},
                "cpu",
                trainer.Exposure(),
                rejected.append,
                rejected.append,
            )
        self.assertEqual(rejected, [])

    def test_last_two_rows_equal_manual_mean_completion_loss_gradient(self):
        torch, fixture = self.torch, self.fixture()
        actual, expected = fixture.model(), fixture.model()
        rows = fixture.rows(6)
        left = torch.optim.SGD(actual.parameters(), lr=0.03)
        right = torch.optim.SGD(expected.parameters(), lr=0.03)
        trainer.train_draws(
            actual,
            rows,
            left,
            torch,
            {"draw_count": 6, "max_updates": 2, "gradient_accumulation": 4, "max_grad_norm": 1.0},
            "cpu",
            trainer.Exposure(),
            lambda _: None,
            lambda _: None,
        )
        for group in (rows[:4], rows[4:]):
            right.zero_grad()
            sum(
                smoke.completion_loss(expected, row, torch, "cpu") / len(group) for row in group
            ).backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
            right.step()
        for observed, reference in zip(actual.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(observed, reference)


if __name__ == "__main__":
    unittest.main()
