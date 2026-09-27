import unittest
from importlib.util import find_spec
from itertools import pairwise

if find_spec("numpy") is None or find_spec("scipy") is None:
    raise unittest.SkipTest(
        "Paired audio alignment requires optional numpy and scipy libraries."
    )

import numpy as np  # noqa: E402
from paired_video_alignment import (  # noqa: E402
    align_audio,
    classify_project_cuts,
    conservative_mapping,
    teacher_timeline,
    waveform_match,
)


class PairedAlignmentTests(unittest.TestCase):
    @staticmethod
    def signal(seconds=32, rate=8000):
        rng = np.random.default_rng(817)
        signal = rng.normal(0, 0.1, round(seconds * rate)).astype(np.float32)
        envelope = np.repeat(rng.uniform(0.2, 1, round(seconds * 10)), rate // 10)
        return signal * envelope

    def test_exact_waveform_shift_and_gain(self):
        raw = self.signal(4)
        edited = raw[4321:12321] * 0.7
        result = waveform_match(raw, edited, 8000, 0, 0.5, 1, 0.1)
        self.assertEqual(result["raw_start"], 4321 / 8000)
        self.assertGreater(result["pcm_correlation"], 0.999)

    def test_recovers_short_and_long_cuts_without_subdividing(self):
        raw = self.signal()
        edited = np.concatenate((raw[:64000], raw[68800:144000], raw[184000:]))
        result = align_audio(raw, edited, end_seconds=len(edited) / 8000)
        candidates = [
            r for r in result["transitions"] if r["status"] == "forward_cut_candidate"
        ]
        for edited_time, duration in ((8, 0.6), (17.4, 5)):
            matched = [
                r for r in candidates if abs(r["edited_boundary"] - edited_time) < 0.1
            ]
            self.assertEqual(len(matched), 1)
            self.assertAlmostEqual(
                matched[0]["source_jump_seconds"], duration, places=3
            )
        self.assertEqual(len(candidates), 2)
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["generated_dataset_rows"], 0)
        document, segments, cuts = teacher_timeline(result, "test-source", 1, 24)
        self.assertEqual(len(cuts), 2)
        self.assertAlmostEqual(sum(s["raw_end"] - s["raw_start"] for s in segments), 23)
        self.assertFalse(document["metadata"]["training_eligible"])
        mapping = conservative_mapping(result, 1, 24)
        for left, right in pairwise(mapping):
            self.assertLessEqual(left["edited_end"], right["edited_start"])

    def test_reordered_block_is_not_reported_as_a_forward_deletion(self):
        raw = self.signal()
        edited = np.concatenate(
            (raw[:64000], raw[144000:192000], raw[64000:144000], raw[192000:])
        )
        result = align_audio(raw, edited, end_seconds=32)
        backwards = [
            r
            for r in result["transitions"]
            if r["status"] == "reorder_or_repeat_candidate"
        ]
        self.assertEqual(len(backwards), 1)
        self.assertAlmostEqual(backwards[0]["source_jump_seconds"], -16, places=3)
        self.assertIsNone(backwards[0]["raw_removed_interval_estimate"])

    def test_silent_unmatched_audio_stays_uncertain(self):
        result = align_audio(
            self.signal(8), np.zeros(32000, dtype=np.float32), end_seconds=4
        )
        self.assertEqual(result["supported_runs"], [])
        self.assertEqual(result["transitions"], [])
        self.assertEqual(result["uncertain_edited_intervals"], [[0, 4]])

    def test_native_item_mismatch_and_long_cut_are_not_admitted(self):
        document = {
            "content": [
                {
                    "type": "text",
                    "uuid": "one",
                    "text": "original",
                    "sourceStart": 0,
                    "length": 1,
                },
                {
                    "type": "text",
                    "uuid": "two",
                    "text": "words",
                    "sourceStart": 1,
                    "length": 1,
                },
            ]
        }
        cut = {
            "id": "cut",
            "status": "forward_cut_candidate",
            "edited_boundary": 1,
            "edited_bracket": [0.995, 1.005],
            "raw_removed_interval_estimate": [0.5, 1.5],
            "change_point_profile": {"status": "localized"},
        }
        rows = classify_project_cuts(
            document,
            [cut, {**cut, "id": "long", "raw_removed_interval_estimate": [0, 3]}],
        )
        self.assertEqual(
            [r["status"] for r in rows],
            ["native_item_boundary_mismatch", "long_cut_outside_delete_span_policy"],
        )
        self.assertTrue(
            all(
                not r["training_eligible"] and not r["natural_cut_qualified"]
                for r in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
