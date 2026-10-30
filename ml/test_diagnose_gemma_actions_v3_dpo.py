"""Synthetic diagnostic binding/census regressions; run remotely, without models."""

import copy
from contextlib import nullcontext
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import diagnose_gemma_actions_v3_dpo as diagnostic


def pairs(count=3):
    return [
        {
            "pair_id": f"pair-{index}",
            "case_id": f"case-{index}",
            "input_sha256": str(index) * 64,
            "messages": [],
            "chosen": {"input_ids": [2, 3, 1], "completion_start": 1},
            "rejected": {"input_ids": [2, 4, 1], "completion_start": 1},
        }
        for index in range(count)
    ]


def score(values, stage, emitted=None):
    prepared = pairs(len(values))
    references = {
        pair["pair_id"]: {"chosen_logprob": -10.0, "rejected_logprob": -5.0}
        for pair in prepared
    }

    def loss(chosen, rejected, reference, beta, _torch):
        margin = beta * (
            (chosen - rejected)
            - (reference["chosen_logprob"] - reference["rejected_logprob"])
        )
        return math.log1p(math.exp(-margin)), margin

    with (
        patch.object(
            diagnostic.dpo.objective,
            "summed_logprob",
            side_effect=[v for row in values for v in row],
        ),
        patch.object(diagnostic.dpo.objective, "dpo_loss", side_effect=loss),
    ):
        records = diagnostic.score_pairs(
            object(),
            prepared,
            references,
            {"training": {"beta": 0.1}},
            SimpleNamespace(no_grad=nullcontext),
            "fixture",
            stage,
            emitted.append if emitted is not None else lambda _row: None,
        )
    return prepared, records


class DpoDiagnosticTests(unittest.TestCase):
    def test_reference_relative_accuracy_is_distinct_from_raw_policy_preference(self):
        prepared, initial = score([(-10, -5)], "initial")
        _, final = score([(-9, -6)], "final")
        result = diagnostic.compare_scores(prepared, initial, final, 0.1)
        self.assertEqual(result["initial"]["preference_accuracy"]["value"], 0)
        self.assertEqual(result["initial"]["near_zero_margin_pairs"], 1)
        self.assertEqual(result["final"]["preference_accuracy"]["value"], 1)
        self.assertEqual(result["final"]["policy_raw_preference_accuracy"]["value"], 0)
        self.assertAlmostEqual(
            result["final"]["mean_dpo_loss"], math.log1p(math.exp(-0.2))
        )
        self.assertLess(result["mean_loss_change_final_minus_initial"], 0)
        self.assertEqual(result["pairs_with_higher_final_margin"], 1)

    def test_anchor_parity_checks_the_third_pair_and_retains_the_failing_score(self):
        emitted = []
        with self.assertRaisesRegex(ValueError, "Initial anchor differs"):
            score([(-10, -5), (-10, -5), (-9.999, -5)], "initial", emitted)
        self.assertEqual(
            [row["pair_id"] for row in emitted], ["pair-0", "pair-1", "pair-2"]
        )
        self.assertFalse(emitted[-1]["reference_parity_passed"])

    def test_missing_reordered_or_rebound_pairs_cannot_publish_complete_comparison(
        self,
    ):
        prepared, initial = score([(-10, -5)] * 3, "initial")
        _, final = score([(-9, -6)] * 3, "final")
        rebound = copy.deepcopy(final)
        rebound[-1]["pair_binding"] = "changed"
        for changed in (final[:-1], list(reversed(final)), rebound):
            with (
                self.subTest(rows=changed),
                self.assertRaisesRegex(ValueError, "Every frozen pair"),
            ):
                diagnostic.compare_scores(prepared, initial, changed, 0.1)

    def test_completed_training_and_reload_must_use_this_reference_and_full_pair_budget(
        self,
    ):
        prepared = pairs()
        config = {"training": {"max_updates": 1}}
        trained = {
            "configuration_sha256": "configuration",
            "reference_manifest_sha256": "reference",
            "mode": "train",
            "training_updates": 1,
            "optimizer_committed_rows": 3,
            "initialization": "frozen_sft_adapter_fresh_optimizer",
        }
        reloaded = {
            **trained,
            "mode": "fresh_reload",
            "reloaded_training_updates": 1,
        }
        diagnostic.validate_completed_run(
            config, prepared, trained, reloaded, "configuration", "reference"
        )
        for changes in (
            {"configuration_sha256": "other"},
            {"reference_manifest_sha256": "other"},
            {"training_updates": 0},
            {"optimizer_committed_rows": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                diagnostic.validate_completed_run(
                    config,
                    prepared,
                    {**trained, **changes},
                    reloaded,
                    "configuration",
                    "reference",
                )
        with self.assertRaisesRegex(ValueError, "another DPO"):
            diagnostic.validate_completed_run(
                config,
                prepared,
                trained,
                {**reloaded, "reference_manifest_sha256": "other"},
                "configuration",
                "reference",
            )


if __name__ == "__main__":
    unittest.main()
