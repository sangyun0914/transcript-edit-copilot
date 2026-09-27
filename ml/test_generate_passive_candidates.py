import copy
import unittest

import generate_passive_candidates as candidates


class CandidateBoundaryTests(unittest.TestCase):
    def test_only_completed_t1_and_its_exact_fresh_reload_can_supply_candidates(self):
        training = {
            "mode": "train",
            "status": "training_complete_fresh_reload_pending",
            "training_updates": 203,
            "optimizer_committed_rows": 809,
            "configuration_sha256": "config",
            "pid": 1,
        }
        reloaded = {
            "mode": "fresh_reload",
            "status": "fresh_reload_passed_local_export_pending",
            "reloaded_training_updates": 203,
            "previous_manifest_sha256": "manifest",
            "configuration_sha256": "config",
            "pid": 2,
        }
        parity = {
            "adapter_tensors_identical": True,
            "cases": [{"case_id": str(index), "identical": True} for index in range(8)],
        }
        candidates.validate_anchor(training, reloaded, parity, "manifest", "config")
        for target, field, value in (
            ("train", "training_updates", 2),
            ("train", "optimizer_committed_rows", 808),
            ("reload", "previous_manifest_sha256", "different-run"),
            ("reload", "pid", 1),
        ):
            left, right = copy.deepcopy(training), copy.deepcopy(reloaded)
            (left if target == "train" else right)[field] = value
            with (
                self.subTest(target=target, field=field),
                self.assertRaises(ValueError),
            ):
                candidates.validate_anchor(left, right, parity, "manifest", "config")
        parity["cases"][-1]["identical"] = False
        with self.assertRaises(ValueError):
            candidates.validate_anchor(training, reloaded, parity, "manifest", "config")

    def test_generation_uses_unique_train_states_instead_of_replacement_draws(self):
        bundle = {
            "cases": [
                {
                    "id": str(index),
                    "metadata": {"split": "train", "training_eligible": True},
                }
                for index in range(809)
            ],
            "tokens": [
                {
                    "id": str(index),
                    "split": "train",
                    "primary": {
                        "input_ids": [1, 2, 3],
                        "completion_start": 2,
                        "labels": [-100, -100, 3],
                    },
                }
                for index in range(809)
            ],
        }
        rows = candidates.unique_rows(bundle)
        self.assertEqual(
            [row["case_id"] for row in rows], [str(index) for index in range(809)]
        )
        for kind in ("dev", "ineligible", "duplicate", "different_token_row"):
            changed = copy.deepcopy(bundle)
            if kind == "dev":
                changed["cases"][0]["metadata"]["split"] = "dev"
            elif kind == "ineligible":
                changed["cases"][0]["metadata"]["training_eligible"] = False
            elif kind == "duplicate":
                changed["cases"][0]["id"] = "1"
            else:
                changed["tokens"][0]["id"] = "different"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                candidates.unique_rows(changed)


if __name__ == "__main__":
    unittest.main()
