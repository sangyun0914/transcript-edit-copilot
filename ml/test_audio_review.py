import importlib.util
import unittest
import copy

from audio_review import (
    RATE,
    annotation_conflicts,
    cut_pcm,
    removal_interval,
    source_selection,
)
from reviewed_audio import bind_observation, verify_media_edit
from synthetic_pilot import document_for, symbolic_evidence
from session_cases import apply_edit, snapshot_for


class AudioReviewTests(unittest.TestCase):
    def test_symbolic_media_checks_cannot_certify_recorded_audio(self):
        from release_media import verify_media_edit as verify_release_media

        document = document_for(
            {
                "language": "en",
                "paragraphs": [
                    [
                        {"id": "left", "text": "Before.", "ms": 1000},
                        {"id": "gap", "text": "", "ms": 2000},
                        {"id": "right", "text": "After.", "ms": 1000},
                    ]
                ],
            }
        )
        document["metadata"] = {
            "provenance": "authored_symbolic_timeline",
            "script_sha256": "unit-test-script",
        }
        for item in document["content"]:
            if "source" in item:
                item["source"] = "symbolic:unit-test-script"
        gap = next(i for i in document["content"] if i["uuid"] == "gap")
        bundle = snapshot_for(
            document,
            gap["uuid"],
            "Remove the idle stall.",
            evidence=symbolic_evidence(document),
        )
        alias = next(
            int(k) for k, v in bundle["mapping"].items() if v["uuid"] == gap["uuid"]
        )
        action = {"action": "remove_silence", "target": [alias]}
        after = apply_edit(document, bundle, action)
        self.assertIsNone(
            verify_release_media(
                document,
                after,
                bundle["mapping"],
                action,
                [],
                "authored_symbolic_timeline",
            )
        )
        with self.assertRaisesRegex(ValueError, "exact reviewed"):
            verify_release_media(document, after, bundle["mapping"], action, [])
        forged = copy.deepcopy(document)
        next(i for i in forged["content"] if "sourceStart" in i)["source"] = (
            "real-recording-hash"
        )
        with self.assertRaisesRegex(ValueError, "cannot authorize"):
            verify_release_media(
                forged,
                after,
                bundle["mapping"],
                action,
                [],
                "authored_symbolic_timeline",
            )

    def test_review_binding_rejects_a_different_source_or_render(self):
        evidence = {
            "id": "gap",
            "source_sha256": "original",
            "role": "natural_candidate",
        }
        observation = {
            "id": "gap",
            "source_sha256": "original",
            "evidence_sha256": "render",
            "decision": "accept",
            "human_listened": True,
        }
        bind_observation(observation, evidence, "render")
        for change in [
            {"source_sha256": "other"},
            {"human_listened": False},
            {"decision": "reject"},
        ]:
            with self.assertRaisesRegex(ValueError, "exact source/render"):
                bind_observation({**observation, **change}, evidence, "render")
        with self.assertRaisesRegex(ValueError, "exact source/render"):
            bind_observation(observation, evidence, "different-render")

    def test_reviewed_trim_cannot_authorize_another_parameter_or_damage_other_media(
        self,
    ):
        document = document_for(
            {
                "language": "en",
                "paragraphs": [
                    [
                        {"id": "left", "text": "Before.", "ms": 1000},
                        {"id": "gap", "text": "", "ms": 2000},
                        {"id": "right", "text": "After.", "ms": 1000},
                    ]
                ],
            }
        )
        bundle = snapshot_for(
            document, "gap", "Tighten the pause.", evidence=symbolic_evidence(document)
        )
        action = {"action": "trim_silence", "target": [1], "keep_ms": 900}
        proof = {
            "id": "symbolic-test-only",
            "action": "trim_silence",
            "keep_ms": 900,
            "items": [document["content"][2]],
            "removed_source_interval": [1.45, 2.55],
        }
        after = apply_edit(document, bundle, action)
        self.assertEqual(
            verify_media_edit(document, after, bundle["mapping"], action, [proof]),
            proof,
        )
        other = {**action, "keep_ms": 500}
        with self.assertRaisesRegex(ValueError, "exact reviewed"):
            verify_media_edit(
                document,
                apply_edit(document, bundle, other),
                bundle["mapping"],
                other,
                [proof],
            )
        damaged = copy.deepcopy(after)
        damaged["content"][-2]["length"] -= 0.1
        with self.assertRaisesRegex(ValueError, "outside"):
            verify_media_edit(document, damaged, bundle["mapping"], action, [proof])

    def test_rounding_cannot_turn_a_noop_into_a_zero_sample_cut(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            removal_interval(600, 600.9000000000001, "trim_silence", 900)
        self.assertEqual(removal_interval(10, 12, "trim_silence", 1500), [10.75, 11.25])

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"), "Audio review runtime required"
    )
    def test_trim_preserves_both_source_edges_and_can_restore_all_samples(self):
        import numpy as np

        source = np.arange(4 * RATE).astype(np.int16)
        after, removed, indices = cut_pcm(source, [601.25, 602.75], 600)
        self.assertEqual(indices, [20000, 44000])
        self.assertTrue(np.array_equal(after[:20000], source[:20000]))
        self.assertTrue(np.array_equal(after[20000:], source[44000:]))
        self.assertTrue(
            np.array_equal(
                np.concatenate((after[:20000], removed, after[20000:])), source
            )
        )
        with self.assertRaisesRegex(ValueError, "context"):
            cut_pcm(source, [600, 601], 600)

    def test_zero_duration_and_reversed_events_are_not_silence(self):
        def event(identity, start, end, tag, text=""):
            return {
                "id": identity,
                "start": start,
                "end": end,
                "tag": tag,
                "text": text,
            }

        annotations = [
            event("laugh", 1, 1, "vocalsound"),
            event("reversed", 2, 1.5, "vocalsound"),
            event("punctuation", 1.2, 1.2, "w", "."),
            event("untimed-word", 1.3, 1.3, "w", "yes"),
        ]
        self.assertEqual(
            [v["id"] for v in annotation_conflicts(annotations, [0.9, 1.8])],
            ["laugh", "reversed", "untimed-word"],
        )

    def test_review_target_must_use_real_item_edges_without_source_jumps(self):
        source = {
            "document": {
                "content": [
                    {"sourceStart": 1, "length": 0.5, "type": "text", "text": "Um"},
                    {"sourceStart": 1.5, "length": 0.5, "type": "text", "text": "well"},
                ]
            }
        }
        self.assertEqual(
            len(source_selection(source, {"target_start": 1, "target_end": 1.5})), 1
        )
        with self.assertRaisesRegex(ValueError, "native item"):
            source_selection(source, {"target_start": 1, "target_end": 1.75})
        source["document"]["content"][1]["sourceStart"] = 1.6
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            source_selection(source, {"target_start": 1, "target_end": 2.1})


if __name__ == "__main__":
    unittest.main()
