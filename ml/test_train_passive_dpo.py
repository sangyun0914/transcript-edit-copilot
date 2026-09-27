import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_train_passive_sft as smoke_tests
import train_passive_dpo as trainer


class DpoGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(
                "Use the existing ML environment for CPU gradient parity."
            ) from error
        cls.torch = torch

    def model(self):
        fixture = smoke_tests.SmokeLossTests()
        fixture.torch = self.torch
        return fixture.model()

    def pairs(self, count=6):
        rows = smoke_tests.SmokeLossTests().rows()
        return [
            {
                "pair_id": f"pair-{index}",
                "case_id": f"case-{index}",
                "metadata": {
                    "source_id": f"source-{index % 2}",
                    "source_group_id": f"group-{index % 2}",
                    "episode_id": f"episode-{index % 3}",
                    "language": "en" if index % 2 else "ko",
                    "chosen_action": "no_action" if index % 2 else "correct_text",
                },
                "chosen": copy.deepcopy(rows[index % 8]),
                "rejected": copy.deepcopy(rows[(index + 3) % 8]),
            }
            for index in range(count)
        ]

    def settings(self, count):
        return {
            "pair_count": count,
            "max_updates": math.ceil(count / 4),
            "gradient_accumulation": 4,
            "max_grad_norm": 1.0,
            "beta": 0.1,
            "chosen_recompute_atol": 1e-4,
            "chosen_recompute_rtol": 1e-6,
        }

    def references(self, pairs):
        return {
            pair["pair_id"]: {"chosen_logprob": -3.2 - index, "rejected_logprob": -6.4 + index / 2}
            for index, pair in enumerate(pairs)
        }

    def joint_loss(self, model, pair, reference):
        torch = self.torch
        values = []
        for side in ("chosen", "rejected"):
            row = pair[side]
            ids = torch.tensor([row["input_ids"]])
            logits = model(ids, torch.ones_like(ids), False, slice(None), True).logits
            labels = torch.tensor([row["labels"]])
            # This independent full-logit reference includes the EOS target and masks the prompt.
            values.append(
                -torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            )
        refs = torch.tensor([reference["chosen_logprob"], reference["rejected_logprob"]])
        return -torch.nn.functional.logsigmoid(
            0.1 * ((values[0] - values[1]) - (refs[0] - refs[1]))
        )

    def test_three_forward_gradient_equals_ordinary_joint_graph_with_summed_eos_loss(self):
        torch = self.torch
        actual, expected = self.model(), self.model()
        pair = self.pairs(1)[0]
        reference = self.references([pair])[pair["pair_id"]]
        event, phases = {}, []
        trainer.sequential_backward(
            actual,
            pair,
            reference,
            torch,
            "cpu",
            self.settings(1),
            1,
            event,
            lambda phase, *_: phases.append(phase),
        )
        reference_loss = self.joint_loss(expected, pair, reference)
        reference_loss.backward()
        self.assertAlmostEqual(event["loss"], float(reference_loss.detach()), places=6)
        self.assertEqual(event["chosen_recompute_absolute_error"], 0)
        self.assertEqual(
            phases,
            [
                "chosen_scored",
                "rejected_forward",
                "rejected_backward",
                "chosen_forward",
                "chosen_backward",
            ],
        )
        for observed, target in zip(actual.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(observed.grad, target.grad, rtol=1e-6, atol=1e-7)

    def test_adamw_updates_equal_joint_dpo_with_actual_final_pair_group(self):
        torch = self.torch
        actual, expected = self.model(), self.model()
        pairs = self.pairs(6)
        references = self.references(pairs)
        left = torch.optim.AdamW(actual.parameters(), lr=1e-5, weight_decay=0.01)
        right = torch.optim.AdamW(expected.parameters(), lr=1e-5, weight_decay=0.01)
        exposure, events, updates = trainer.Exposure(), [], []
        trainer.train_pairs(
            actual,
            pairs,
            references,
            left,
            torch,
            self.settings(6),
            "cpu",
            exposure,
            events.append,
            updates.append,
        )
        for group in (pairs[:4], pairs[4:]):
            right.zero_grad(set_to_none=True)
            sum(
                self.joint_loss(expected, pair, references[pair["pair_id"]]) / len(group)
                for pair in group
            ).backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
            right.step()
        for observed, target in zip(actual.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(observed, target, rtol=1e-6, atol=1e-7)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(
                    left.state[observed][key], right.state[target][key], rtol=1e-6, atol=1e-8
                )
        self.assertEqual([row["microbatches"] for row in updates], [4, 2])
        self.assertEqual([row["accumulation_divisor"] for row in events[-2:]], [2, 2])
        summary = exposure.summary()
        self.assertEqual(summary["completed_updates"], 2)
        self.assertEqual(summary["phases"]["optimizer_committed"]["rows"], 6)
        self.assertEqual(summary["pending_complete_backward_pairs_not_committed"], 0)
        self.assertEqual(summary["partial_backward_pairs_not_completed"], 0)
        self.assertEqual(
            summary["actual_policy_forward_sequence_tokens"],
            sum(
                2 * len(pair["chosen"]["input_ids"]) + len(pair["rejected"]["input_ids"])
                for pair in pairs
            ),
        )

    def test_recompute_drift_stops_before_update_and_records_partial_backward(self):
        torch = self.torch
        model = self.model()
        original = model.forward

        def drifting(*args, **kwargs):
            output = original(*args, **kwargs)
            if not torch.is_grad_enabled():
                output.logits = output.logits * 1.1
            return output

        model.forward = drifting
        pairs, exposure, events = self.pairs(1), trainer.Exposure(), []
        with self.assertRaisesRegex(ValueError, "Chosen scoring/recomputation differs"):
            trainer.train_pairs(
                model,
                pairs,
                self.references(pairs),
                torch.optim.AdamW(model.parameters(), lr=1e-5),
                torch,
                self.settings(1),
                "cpu",
                exposure,
                events.append,
                lambda _: None,
            )
        self.assertEqual(exposure.updates, 0)
        self.assertEqual(exposure.summary()["partial_backward_pairs_not_completed"], 1)
        self.assertEqual(exposure.phases["optimizer_committed"]["rows"], 0)
        self.assertEqual(events[0]["status"], "failed")
        self.assertIn("rejected_backward", events[0]["completed_phases"])
        self.assertNotIn("chosen_backward", events[0]["completed_phases"])
        self.assertEqual(exposure.attempted_compute_calls["chosen_forward"], 1)
        self.assertEqual(exposure.attempted_compute_calls["chosen_backward"], 0)

    def test_profile_uses_fixed_eight_largest_pairs_without_reordering_main_pass(self):
        pairs = self.pairs(10)
        original = [pair["pair_id"] for pair in pairs]
        pairs[-1]["chosen"]["input_ids"] += [1] * 20
        selected = trainer.profile_pairs(pairs)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0]["pair_id"], "pair-9")
        self.assertEqual([pair["pair_id"] for pair in pairs], original)
        self.assertEqual(trainer.profile_pairs(copy.deepcopy(pairs)), selected)


class DpoBindingTests(unittest.TestCase):
    def pairs(self):
        return DpoGradientTests().pairs(6)

    def verified(self, pairs):
        return {
            "configuration_sha256": "a" * 64,
            "preparation_sha256": "b" * 64,
            "anchor_training_manifest_sha256": "c" * 64,
            "anchor_adapter_sha256": "d" * 64,
            "anchor_configuration_sha256": "e" * 64,
            "sequence_tokens": sum(
                len(pair[side]["input_ids"]) for pair in pairs for side in trainer.SIDES
            ),
            "completion_tokens": sum(
                len(pair[side]["input_ids"]) - pair[side]["completion_start"]
                for pair in pairs
                for side in trainer.SIDES
            ),
        }

    def cache(self, root, pairs, verified):
        run = {
            **verified,
            "mode": "reference_precompute",
            "training_updates": 0,
            "reference_updates": 0,
            "reference_adapter_unchanged": True,
            "actual_tokenizer_parity": True,
            "completed_pairs": len(pairs),
            "reference_source": "exact_final_T1_adapter",
            "log_probability": "summed_completion_including_eos",
            "reference_forward_attempts": 2 * len(pairs),
            "reference_forward_completions": 2 * len(pairs),
            "reference_forward_sequence_tokens": verified["sequence_tokens"],
            "reference_forward_completion_tokens": verified["completion_tokens"],
        }
        rows = [
            {
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "chosen_logprob": -10.0,
                "rejected_logprob": -20.0,
                **{
                    side + "_completion_tokens": len(pair[side]["input_ids"])
                    - pair[side]["completion_start"]
                    for side in trainer.SIDES
                },
            }
            for pair in pairs
        ]
        trainer.smoke.write(root / "run.json", run)
        (root / "reference-logps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        trainer.manifest(root, "reference_cache_complete")
        return run, rows

    def test_cache_rejects_changed_anchor_disabled_adapter_reduction_and_eos_budget(self):
        pairs = self.pairs()
        verified = self.verified(pairs)
        config = {"training": {"pair_order": [pair["pair_id"] for pair in pairs]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, rows = self.cache(root, pairs, verified)
            actual, _, _ = trainer.cache_rows(root, config, verified, pairs)
            self.assertEqual(list(actual), config["training"]["pair_order"])
            receipt = trainer.smoke.read(root / "manifest.json")
            del receipt["files_sha256"]["reference-logps.jsonl"]
            trainer.smoke.write(root / "manifest.json", receipt)
            with self.assertRaisesRegex(ValueError, "payload must be hash-bound"):
                trainer.cache_rows(root, config, verified, pairs)
            trainer.manifest(root, "reference_cache_complete")
            for key, value in (
                ("anchor_adapter_sha256", "f" * 64),
                ("reference_source", "disabled_adapter_native_base"),
                ("log_probability", "mean_completion_logprob"),
                ("training_updates", 1),
                ("actual_tokenizer_parity", False),
                ("reference_forward_completions", 2 * len(pairs) - 1),
            ):
                trainer.smoke.write(root / "run.json", {**run, key: value})
                trainer.manifest(root, "reference_cache_complete")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    trainer.cache_rows(root, config, verified, pairs)
            trainer.smoke.write(root / "run.json", run)
            rows[0]["chosen_completion_tokens"] -= 1
            (root / "reference-logps.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            trainer.manifest(root, "reference_cache_complete")
            with self.assertRaisesRegex(ValueError, "completion/EOS"):
                trainer.cache_rows(root, config, verified, pairs)

    def test_actual_tokenizer_verification_fails_before_model_loading(self):
        with (
            patch.object(trainer, "verify_actual_tokens", return_value=[]),
            patch.object(trainer.smoke, "load_runtime") as load,
        ):
            with self.assertRaisesRegex(ValueError, "Verified preference pairs changed"):
                trainer.load_adapter(
                    {}, Path("/unused"), self.pairs(), Path("/unused/adapter"), {}, True
                )
            load.assert_not_called()
        code = "import sys; import train_passive_dpo; assert not any(k in sys.modules for k in ('torch','transformers','peft','bitsandbytes'))"
        subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).parent, check=True)

    def test_substantive_profile_gate_binds_exact_cache_order_and_fresh_reload(self):
        pairs = DpoGradientTests().pairs(8)
        order = [pair["pair_id"] for pair in pairs]
        verified = self.verified(pairs)
        config = {"profile": {"pair_order": order}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, reload = root / "profile", root / "reload"
            profile.mkdir()
            reload.mkdir()
            run = {
                **verified,
                "mode": "train",
                "experiment_phase": "engineering_profile",
                "reference_cache_sha256": "f" * 64,
                "training_updates": 2,
                "optimizer_committed_pairs": 8,
                "pair_order": order,
                "actual_tokenizer_parity": True,
                "changed_adapter_tensors": ["lora_A"],
                "pid": os.getpid() + 1,
            }
            trainer.smoke.write(profile / "run.json", run)
            exposure = trainer.Exposure()
            for pair in pairs:
                for phase, side in (
                    ("rejected_backward", "rejected"),
                    ("chosen_backward", "chosen"),
                    ("pair_backward_complete", None),
                    ("optimizer_committed", None),
                ):
                    exposure.record(phase, pair, side)
            exposure.updates = 2
            trainer.smoke.write(profile / "exposure.json", exposure.summary())
            (profile / "adapter").mkdir()
            trainer.smoke.write(profile / "adapter-tensors.json", {})
            trainer.smoke.write(profile / "adapter/adapter_config.json", {})
            (profile / "adapter/adapter_model.safetensors").write_bytes(b"unit-test bound payload")
            output = [{"case_id": f"diagnostic-{index}", "output_ids": [1]} for index in range(8)]
            trainer.smoke.write(profile / "smoke-generation.json", output)
            trainer.manifest(profile, "training_complete_fresh_reload_pending")
            reload_run = {
                "mode": "fresh_reload",
                "experiment_phase": "engineering_profile",
                "configuration_sha256": verified["configuration_sha256"],
                "previous_manifest_sha256": trainer.smoke.digest(profile / "manifest.json"),
                "reloaded_training_updates": 2,
                "pid": os.getpid() + 2,
            }
            trainer.smoke.write(reload / "run.json", reload_run)
            trainer.smoke.write(reload / "smoke-generation.json", output)
            trainer.smoke.write(
                reload / "reload-parity.json",
                {
                    "adapter_tensors_identical": True,
                    "optimizer_state_resumed": False,
                    "cases": [{"case_id": row["case_id"], "identical": True} for row in output],
                },
            )
            trainer.manifest(reload, "fresh_reload_passed_local_export_pending")
            result = trainer.validate_profile(profile, reload, config, verified, "f" * 64)
            self.assertEqual(
                result["profile_manifest_sha256"], trainer.smoke.digest(profile / "manifest.json")
            )
            receipt = trainer.smoke.read(reload / "manifest.json")
            del receipt["files_sha256"]["reload-parity.json"]
            trainer.smoke.write(reload / "manifest.json", receipt)
            with self.assertRaisesRegex(ValueError, "payload must be hash-bound"):
                trainer.validate_profile(profile, reload, config, verified, "f" * 64)
            trainer.manifest(reload, "fresh_reload_passed_local_export_pending")
            with self.assertRaisesRegex(ValueError, "two-update profile"):
                trainer.validate_profile(profile, reload, config, verified, "0" * 64)
            reload_run["previous_manifest_sha256"] = "0" * 64
            trainer.smoke.write(reload / "run.json", reload_run)
            trainer.manifest(reload, "fresh_reload_passed_local_export_pending")
            with self.assertRaisesRegex(ValueError, "two-update profile"):
                trainer.validate_profile(profile, reload, config, verified, "f" * 64)

    def test_output_cannot_enter_bound_inputs_or_an_external_reference_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, cache = root / "inputs", root / "reference-cache"
            inputs.mkdir()
            cache.mkdir()
            for output in (
                inputs / "anchor/train/new-output",
                inputs / "candidates/new-output",
                cache / "new-output",
            ):
                with (
                    self.subTest(output=output),
                    self.assertRaisesRegex(ValueError, "fresh output"),
                ):
                    trainer.new_output(output, inputs, cache)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
