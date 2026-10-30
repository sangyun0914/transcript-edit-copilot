import struct
import unittest

import prepare_audio_actions_v3 as audio


class AudioActionsV3Tests(unittest.TestCase):
    def test_exact_interval_render_removes_only_constructed_gap(self):
        original = struct.pack("<4h", 120, -90, 260, -200)
        constructed = original[:4] + b"\0\0" * 3 + original[4:]
        document = {
            "content": [
                {"source": "derived", "sourceStart": 0, "length": 2 / audio.RATE},
                {
                    "source": "derived",
                    "sourceStart": 5 / audio.RATE,
                    "length": 2 / audio.RATE,
                },
            ]
        }
        self.assertEqual(audio.render(document, constructed, "derived"), original)
        with self.assertRaisesRegex(ValueError, "another source"):
            audio.render(document, constructed, "unrelated")

    def test_boundary_selection_requires_contiguous_words_and_known_speaker(self):
        def document(speaker, discontinuity=0):
            return {
                "content": [{"type": "paragraph_start", "speaker": speaker}]
                + [
                    {
                        "type": "text",
                        "uuid": str(i),
                        "source": "source",
                        "sourceStart": 40 + i + (discontinuity if i > 2 else 0),
                        "length": 1,
                        "text": str(i),
                    }
                    for i in range(8)
                ]
                + [{"type": "paragraph_end"}]
            }

        self.assertTrue(audio.candidate_boundaries(document("speaker")))
        self.assertEqual(audio.candidate_boundaries(document(None)), [])
        selected = audio.candidate_boundaries(document("speaker", 3))
        for value in selected:
            items = value["items"]
            for left, right in zip(items, items[1:]):
                self.assertAlmostEqual(
                    left["sourceStart"] + left["length"], right["sourceStart"]
                )


if __name__ == "__main__":
    unittest.main()
