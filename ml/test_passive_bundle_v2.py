import copy
import unittest
from collections import Counter
from unittest.mock import Mock, patch

from passive_bundle_v2 import ROOT, balanced_weights, checkpoint_dispositions, weighted_indices


def case(identity, language, group, region):
    return {"id": identity, "metadata": {
        "split": "train", "training_eligible": True, "language": language,
        "source_group_id": group, "source_id": group, "source_region": region,
    }}


class PassiveSamplingTests(unittest.TestCase):
    def test_unresolved_checkpoints_remain_in_the_registered_denominator(self):
        directory = ROOT / "example-checkpoints"
        excluded = [{"case_id": "second-s00", "input_sha256": "input-hash",
                     "status": "unresolved", "reason": "Ambiguous caption benefit."}]
        proof = {"dependencies_sha256": {"example-checkpoints/exclusions.jsonl": "file-hash"},
                 "candidate_hashes": {"exclusions.jsonl": "file-hash"},
                 "observed_case_ids": ["first-s00", "second-s00"],
                 "excluded_case_ids": ["second-s00"], "scheduled_checkpoints": 2,
                 "observed_contexts": 2, "unresolved_or_unsupported": 1}
        recipe = {"episodes": [{"id": "first"}, {"id": "second"}]}
        with patch("passive_bundle_v2.rows", return_value=excluded), patch("passive_bundle_v2.read", return_value=recipe):
            result = checkpoint_dispositions(proof, directory, Mock(), [{"id": "first-s00"}])
            self.assertEqual(result[0]["decision"], "exclude")
            self.assertEqual(result[0]["status"], "unresolved")
            for change in ({"excluded_case_ids": []}, {"observed_case_ids": ["first-s00"]},
                           {"scheduled_checkpoints": 1}, {"candidate_hashes": {}}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    checkpoint_dispositions({**proof, **change}, directory, Mock(), [{"id": "first-s00"}])

    def test_symbolic_full_document_with_no_media_region_is_one_opportunity(self):
        cases = [case("authored-1", "ko", "script", None), case("authored-2", "ko", "script", None)]
        for c in cases:
            c["metadata"]["media_kind"] = "authored_symbolic_timeline"
        balanced_weights(cases)
        self.assertEqual([c["metadata"]["sampling_weight"] for c in cases], [0.5, 0.5])
        self.assertEqual(cases[0]["metadata"]["sampling_cluster"], cases[1]["metadata"]["sampling_cluster"])
        cases[0]["metadata"]["media_kind"] = "recording"
        with self.assertRaises(ValueError):
            balanced_weights(cases)

    def test_overlapping_replays_share_mass_and_sampler_consumes_weights(self):
        cases = [case(f"en-{i}", "en", "lecture", [0, 10]) for i in range(20)]
        cases += [case("ko-1", "ko", "interview", [0, 8]),
                  case("ko-2", "ko", "interview", [7, 15])]
        balanced_weights(cases)
        by_language = Counter()
        for c in cases:
            by_language[c["metadata"]["language"]] += c["metadata"]["sampling_weight"]
        self.assertAlmostEqual(by_language["en"], 0.5)
        self.assertAlmostEqual(by_language["ko"], 0.5)
        self.assertEqual(cases[-1]["metadata"]["sampling_cluster"], cases[-2]["metadata"]["sampling_cluster"])
        indices = weighted_indices(cases, 4000, "frozen-seed")
        self.assertEqual(indices, weighted_indices(cases, 4000, "frozen-seed"))
        selected_en = sum(cases[i]["metadata"]["language"] == "en" for i in indices)
        self.assertTrue(1800 < selected_en < 2200)
        # Uniform rows would put about 91% of samples in the twenty English replays.
        self.assertLess(selected_en, 2500)

    def test_pending_or_development_rows_cannot_be_sampled_for_training(self):
        original = [case("x", "en", "lecture", [0, 10])]
        balanced_weights(original)
        for change in ({"split": "dev"}, {"training_eligible": False}, {"sampling_weight": float("nan")}):
            altered = copy.deepcopy(original)
            altered[0]["metadata"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                weighted_indices(altered, 1, "seed")


if __name__ == "__main__":
    unittest.main()
