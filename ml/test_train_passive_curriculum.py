import copy
import json
from pathlib import Path
import tempfile
import unittest

import test_train_passive_sft as smoke_tests
import train_passive_curriculum as trainer
import train_passive_sft as smoke


class CurriculumConfigurationTests(unittest.TestCase):
    def setUp(self):
        fixture = smoke_tests.SmokePreflightTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.original = fixture.config
        self.config = copy.deepcopy(self.original)
        self.config["version"] = trainer.VERSION
        self.config["training"].update(
            sampling_seed="3407",
            max_updates=203,
            draw_count=809,
            sample_order=["fixed-draw"] * 809,
            final_partial_accumulation="actual_microbatch_count",
            fresh_native_initialization=True,
            recovery_updates=[50, 100, 150],
            wall_limit_seconds=21600,
        )

    def test_curriculum_keeps_verified_identity_and_fixed_nonadaptive_recipe(self):
        trainer.validate_configuration(self.config, self.original)
        for key, value in (
            ("max_updates", 204),
            ("draw_count", 808),
            ("sampling_seed", 3407),
            ("learning_rate", 0.0002),
            ("fresh_native_initialization", False),
            ("recovery_updates", [10, 20]),
            ("wall_limit_seconds", 22000),
            ("final_partial_accumulation", "always_four"),
        ):
            changed = copy.deepcopy(self.config)
            changed["training"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                trainer.validate_configuration(changed, self.original)
        changed = copy.deepcopy(self.config)
        changed["model"]["quantization"] = "none"
        with self.assertRaisesRegex(ValueError, "settings changed"):
            trainer.validate_configuration(changed, self.original)

    def test_evaluation_protocol_cannot_enable_dev_during_training_or_extend_budget(self):
        protocol = {
            "version": "passive-curriculum-evaluation-v1",
            "training_plan": {
                "candidate": "qwen35-9b",
                "view": "json-curriculum",
                "draw_count": 809,
                "sampling_seed": "3407",
                "model_seed": 3407,
                "effective_batch": 4,
                "updates": 203,
                "final_accumulation": 1,
                "development_during_training": False,
            },
            "conditions": ["native-base-full", "native-base-icl", "sft-full", "sft-icl"],
            "b2": {"release_manifest": {"sha256": "a" * 64}},
            "final_test_bodies_read": False,
            "learning_signal_policy": {
                "no_auto_extra_training": True,
                "deployment_promotion": False,
            },
        }
        trainer.validate_protocol(protocol, self.config, "a" * 64)
        for key, value in (("development_during_training", True), ("updates", 405)):
            changed = copy.deepcopy(protocol)
            changed["training_plan"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "protocol differs"):
                trainer.validate_protocol(changed, self.config, "a" * 64)


class CurriculumUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(
                "Use the existing ML environment for meaningful CPU gradients."
            ) from error
        cls.torch = torch

    def model(self):
        fixture = smoke_tests.SmokeLossTests()
        fixture.torch = self.torch
        return fixture.model()

    def rows(self, count=9):
        fixture = smoke_tests.SmokeLossTests()
        originals = fixture.rows()
        rows = []
        for index in range(count):
            row = copy.deepcopy(originals[index % len(originals)])
            row.update(
                draw_index=index,
                source_id=f"source-{index % 2}",
                source_group_id=f"group-{index % 2}",
                episode_id=f"episode-{index % 3}",
                language="en" if index % 2 else "ko",
                action="correct_text",
            )
            rows.append(row)
        return rows

    def test_last_single_draw_uses_one_row_mean_and_all_exposure_counts_are_exact(self):
        torch = self.torch
        actual, expected = self.model(), self.model()
        rows = self.rows()
        left = torch.optim.SGD(actual.parameters(), lr=0.03)
        right = torch.optim.SGD(expected.parameters(), lr=0.03)
        settings = {
            "draw_count": 9,
            "max_updates": 3,
            "gradient_accumulation": 4,
            "max_grad_norm": 1.0,
        }
        exposure, row_events, updates = trainer.Exposure(), [], []
        trainer.train_draws(
            actual, rows, left, torch, settings, "cpu", exposure, row_events.append, updates.append
        )
        for offset in (0, 4, 8):
            group = rows[offset : offset + 4]
            right.zero_grad()
            sum(
                smoke.completion_loss(expected, row, torch, "cpu") / len(group) for row in group
            ).backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
            right.step()
        for a, b in zip(actual.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(a, b)
        self.assertEqual([event["microbatches"] for event in updates], [4, 4, 1])
        self.assertEqual(row_events[-1]["accumulation_divisor"], 1)
        self.assertEqual([event["draw_index"] for event in row_events], list(range(9)))
        summary = exposure.summary()
        self.assertEqual(summary["completed_updates"], 3)
        self.assertEqual(summary["pending_backward_rows_not_committed"], 0)
        for phase in trainer.PHASES:
            self.assertEqual(summary["phases"][phase]["rows"], 9)
            self.assertEqual(summary["phases"][phase]["unique_cases"], 8)
            self.assertEqual(summary["phases"][phase]["counts"]["case_id"]["row-0"], 2)
            self.assertEqual(
                summary["phases"][phase]["sequence_tokens"],
                sum(len(row["input_ids"]) for row in rows),
            )

    def test_wall_stop_distinguishes_partial_backward_from_committed_rows(self):
        torch = self.torch
        model, unchanged = self.model(), self.model()
        rows = self.rows(4)
        exposure, events, updates = trainer.Exposure(), [], []
        checks = 0

        def deadline():
            nonlocal checks
            checks += 1
            trainer.check_deadline(0, 6, lambda: checks)

        with self.assertRaises(trainer.WallLimitExceeded):
            trainer.train_draws(
                model,
                rows,
                torch.optim.SGD(model.parameters(), lr=0.03),
                torch,
                {
                    "draw_count": 4,
                    "max_updates": 1,
                    "gradient_accumulation": 4,
                    "max_grad_norm": 1.0,
                },
                "cpu",
                exposure,
                events.append,
                updates.append,
                deadline,
            )
        self.assertEqual(updates, [])
        summary = exposure.summary()
        self.assertEqual(summary["phases"]["forward"]["rows"], 3)
        self.assertEqual(summary["phases"]["backward"]["rows"], 3)
        self.assertEqual(summary["phases"]["optimizer_committed"]["rows"], 0)
        self.assertEqual(summary["pending_backward_rows_not_committed"], 3)
        for before, after in zip(unchanged.parameters(), model.parameters(), strict=True):
            torch.testing.assert_close(before, after)

    def test_nonfinite_forward_is_logged_and_never_counted_as_backward_or_update(self):
        torch = self.torch
        model = self.model()
        with torch.no_grad():
            model.head.weight.fill_(float("nan"))
        exposure, events, updates = trainer.Exposure(), [], []
        with self.assertRaisesRegex(ValueError, "Non-finite loss"):
            trainer.train_draws(
                model,
                self.rows(4),
                torch.optim.SGD(model.parameters(), lr=0.03),
                torch,
                {
                    "draw_count": 4,
                    "max_updates": 1,
                    "gradient_accumulation": 4,
                    "max_grad_norm": 1.0,
                },
                "cpu",
                exposure,
                events.append,
                updates.append,
            )
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["forward_complete"])
        self.assertFalse(events[0]["backward_complete"])
        self.assertEqual(exposure.phases["forward"]["rows"], 1)
        self.assertEqual(exposure.phases["backward"]["rows"], 0)
        self.assertEqual(updates, [])

    def test_interrupted_optimizer_mutation_cannot_be_saved_as_safe_recovery(self):
        torch = self.torch
        model, unchanged = self.model(), self.model()

        class InterruptedSGD(torch.optim.SGD):
            def step(self, closure=None):
                super().step(closure)
                raise RuntimeError("interrupted after parameter mutation")

        optimizer = InterruptedSGD(model.parameters(), lr=0.03)
        exposure, events, updates = trainer.Exposure(), [], []
        with self.assertRaisesRegex(RuntimeError, "after parameter mutation"):
            trainer.train_draws(
                model,
                self.rows(4),
                optimizer,
                torch,
                {
                    "draw_count": 4,
                    "max_updates": 1,
                    "gradient_accumulation": 4,
                    "max_grad_norm": 1.0,
                },
                "cpu",
                exposure,
                events.append,
                updates.append,
            )
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(unchanged.parameters(), model.parameters(), strict=True)
            )
        )
        summary = exposure.summary()
        self.assertTrue(summary["optimizer_commit_uncertain"])
        self.assertEqual(summary["completed_updates"], 0)
        self.assertEqual(summary["phases"]["backward"]["rows"], 4)
        self.assertEqual(summary["phases"]["optimizer_committed"]["rows"], 0)
        self.assertEqual(updates, [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "interrupted optimizer step"):
                trainer.save_checkpoint(
                    root, "unsafe", model, optimizer, torch, "a" * 64, exposure, "failure"
                )
            self.assertFalse((root / "unsafe").exists())

    def test_recovery_checkpoint_is_immutable_and_records_uncommitted_exposure(self):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as error:
            self.skipTest(f"PEFT is required: {error}")
        torch = self.torch
        native = torch.nn.Sequential(torch.nn.Linear(4, 4))
        model = get_peft_model(native, LoraConfig(r=2, lora_alpha=2, target_modules=["0"]))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        exposure = trainer.Exposure()
        for row in self.rows(2):
            exposure.record("forward", row)
            exposure.record("backward", row)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = trainer.save_checkpoint(
                root,
                "recovery-after-000000",
                model,
                optimizer,
                torch,
                "a" * 64,
                exposure,
                "wall_limit",
            )
            path = root / relative
            receipt = json.loads((path / "manifest.json").read_text())
            original = (path / "manifest.json").read_bytes()
            self.assertEqual(receipt["status"], "recovery_checkpoint_complete")
            for name, expected in receipt["files_sha256"].items():
                self.assertEqual(smoke.digest(path / name), expected)
            checkpoint = smoke.read(path / "checkpoint.json")
            self.assertEqual(checkpoint["exposure"]["pending_backward_rows_not_committed"], 2)
            self.assertFalse(checkpoint["partial_gradients_saved"])
            self.assertFalse(checkpoint["resume_supported"])
            with self.assertRaises(FileExistsError):
                trainer.save_checkpoint(
                    root, relative, model, optimizer, torch, "a" * 64, exposure, "repeat"
                )
            self.assertEqual((path / "manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
