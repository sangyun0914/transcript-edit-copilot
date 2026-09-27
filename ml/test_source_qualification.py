import tempfile
import unittest
from pathlib import Path

from source_qualification import cue_document, focus_at


class SourceQualificationTests(unittest.TestCase):
    def test_cursor_on_shared_boundary_selects_following_item(self):
        items = [
            {"uuid": "previous", "sourceStart": 1.0, "length": 1.5},
            {"uuid": "following", "sourceStart": 2.5, "length": 0.8},
        ]
        self.assertEqual(focus_at(items, 2.5)["uuid"], "following")
        self.assertEqual(focus_at(items, 2.49)["uuid"], "previous")

    def test_caption_import_preserves_cue_units_and_rejects_overlap(self):
        source = {"id": "lecture", "language": "en", "video_sha256": "source"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captions.vtt"
            path.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n"
                "LECTURER: Two whole words.\n\n"
                "00:00:02.000 --> 00:00:03.000\nAnother cue.\n"
            )
            document = cue_document(source, path)
            items = [i for i in document["content"] if "sourceStart" in i]
            self.assertEqual(len(items), 3)
            self.assertEqual(items[0]["text"], "LECTURER: Two whole words.")
            self.assertEqual(items[0]["length"], 1.5)
            self.assertEqual(items[1]["type"], "non_text")
            self.assertEqual(items[1]["length"], 0.5)
            self.assertEqual(items[2]["sourceStart"], 2.0)
            path.write_text(path.read_text().replace("00:00:02.000", "00:00:01.000"))
            with self.assertRaisesRegex(ValueError, "Overlapping cues"):
                cue_document(source, path)


if __name__ == "__main__":
    unittest.main()
