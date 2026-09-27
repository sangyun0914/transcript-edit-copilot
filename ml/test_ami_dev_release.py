import unittest

from ami_dev_release import dev_document, source_audit


def native(identity, speaker, start, end, text="", tag="w"):
    return {
        "id": identity,
        "speaker": speaker,
        "position": int(identity[-1]),
        "start": start,
        "end": end,
        "text": text,
        "display": text,
        "tag": tag,
        "attributes": {},
    }


class AmiDevTimelineTests(unittest.TestCase):
    def test_overlap_audio_once_and_nonpositive_events_never_retimed(self):
        annotations = [
            native("word1", "A", 0.0, 1.0, "hello"),
            native("word2", "B", 0.5, 1.5, "yes"),
            native("point3", "A", 1.5, 1.5, tag="vocalsound"),
            native("reversed4", "B", 2.0, 1.8, tag="vocalsound"),
        ]
        document, evidence, lineage = dev_document(annotations, "source", 2.5)
        source = {
            "document": document,
            "native_annotations": annotations,
            "boundary_evidence": evidence,
            "item_annotations": lineage,
        }
        audit = source_audit(source, 2.5)
        items = [i for i in document["content"] if "sourceStart" in i]
        self.assertEqual(items[0]["text"], "A: hello\nB: yes")
        self.assertEqual(items[0]["length"], 1.5)
        self.assertEqual(audit["audio_interval_multiplicity"], 1)
        self.assertEqual(audit["positive_duration_annotations_retained_once"], 2)
        self.assertEqual(
            audit["zero_duration_annotations_retained_in_native_records"], 1
        )
        self.assertEqual(audit["reversed_annotations_retained_in_native_records"], 1)
        self.assertTrue(
            all(e["boundary_status"] != "verified" for e in evidence.values())
        )

    def test_unusable_spoken_word_alignment_is_rejected(self):
        for start, end in ((1.0, 1.0), (1.2, 1.0)):
            with (
                self.subTest(start=start, end=end),
                self.assertRaisesRegex(ValueError, "do not invent timing"),
            ):
                dev_document([native("word1", "A", start, end, "yes")], "source", 2.0)


if __name__ == "__main__":
    unittest.main()
