import copy
import tempfile
import unicodedata
import unittest
from pathlib import Path

from context_policy import ContextBudgetError
from dataset_release import action_for, audit_cases
from dataset_release_inventory import register_group, resolve_local
from session_cases import document_records
from release_sources import scribe_document
from synthetic_pilot import document_for
from ami_release import ami_document
from dataset_release import restore_bundle
from context_policy import ContextBuilder, FeedbackLedger
from session_cases import apply_edit
from test_context_policy import Tokenizer
from release_audit import dependence_blocks


class ReleaseTests(unittest.TestCase):
    def test_ami_overlap_is_one_audio_interval_and_events_are_not_silence(self):
        def annotation(identity, speaker, start, end, text="", tag="w"):
            return {
                "id": identity,
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
                "display": text,
                "tag": tag,
                "position": int(identity),
                "attributes": {},
            }

        records = [
            annotation("0", "A", 1, 2, "question?"),
            annotation("1", "B", 1.5, 2.5, "Yeah."),
            annotation("2", "B", 2.5, 3, tag="vocalsound"),
            annotation("3", "B", 3, 3, tag="vocalsound"),
        ]
        document, evidence, lineage = ami_document(records, "mixed", 4)
        items, _ = document_records(document, evidence)
        intervals = [i for i in document["content"] if "sourceStart" in i]
        self.assertEqual(sum(i["length"] for i in intervals), 4)
        self.assertEqual(
            [(i["sourceStart"], i["length"]) for i in intervals],
            [(0, 1), (1, 1.5), (2.5, 0.5), (3, 1)],
        )
        self.assertIn("B: Yeah.", items[1]["text"])
        self.assertIsNone(items[1]["speaker"])
        self.assertEqual(items[1]["boundary_status"], "overlap")
        self.assertEqual(items[2]["audio_class"], "audio_event")
        self.assertEqual(items[3]["audio_class"], "unknown")
        self.assertTrue(all(i["boundary_status"] != "verified" for i in items))
        self.assertEqual(lineage[intervals[1]["uuid"]]["annotation_ids"], ["0", "1"])

    def test_serialized_episode_feedback_is_past_only_and_reconstructs_exactly(self):
        document = document_for(
            {
                "language": "en",
                "paragraphs": [[{"id": "word", "text": "control", "ms": 500}]],
            }
        )
        builder = ContextBuilder(Tokenizer(), "rules")
        bundle = builder.build(document, "word", "Keep verbatim.", as_of=0)
        ledger = FeedbackLedger("episode")
        candidate = {"action": "correct_text", "target": [0], "text": "controls"}
        ledger.propose("p", 1, document, bundle, candidate)
        ledger.record("p", "shown", 2)
        ledger.record("p", "dismissed", 4, reason="target")
        saved = {
            "document_id": ledger.document_id,
            "proposals": ledger.proposals,
            "events": ledger.events,
        }
        early = restore_bundle(
            builder, document, "word", "Keep verbatim.", {"as_of": 3, "feedback": saved}
        )
        later = restore_bundle(
            builder, document, "word", "Keep verbatim.", {"as_of": 5, "feedback": saved}
        )
        self.assertEqual(early["input"]["recent_feedback"], [])
        self.assertEqual(later["input"]["recent_feedback"][0]["reason"], "target")
        self.assertTrue(ledger.suppressed(document, later, candidate, 5))
        changed = builder.build(
            document, "word", "Use plural nouns.", as_of=5, feedback=ledger
        )
        self.assertFalse(ledger.suppressed(document, changed, candidate, 5))
        after = apply_edit(document, changed, candidate)
        self.assertEqual(after["content"][1]["text"], "controls")

    def test_excerpt_import_preserves_offsets_and_never_qualifies_gaps(self):
        content = [
            {
                "speaker": "speaker_1",
                "content": [
                    {"type": "word", "word": "X", "sourceStart": 0, "length": 0.2},
                    {"type": "silence", "sourceStart": 0.2, "length": 0.3},
                    {
                        "type": "word",
                        "word": "하나가",
                        "sourceStart": 0.5,
                        "length": 0.4,
                    },
                ],
            }
        ]
        document = scribe_document(content, "original-video", 600, "excerpt", "ko")
        timed = [i for i in document["content"] if "source" in i]
        self.assertEqual([i["sourceStart"] for i in timed], [600, 600.2, 600.5])
        self.assertEqual([i["length"] for i in timed], [0.2, 0.3, 0.4])
        records, _ = document_records(document)
        self.assertEqual(records[1]["audio_class"], "unknown")
        self.assertTrue(all(i["boundary_status"] == "unverified" for i in records))
        self.assertEqual(records[0]["speaker"], "speaker_1")

    def test_existing_and_reserved_groups_cannot_be_reassigned(self):
        groups = {}
        register_group(groups, "lecture", "train", "original", "real_recording")
        register_group(groups, "lecture", "train", "retranscription", "real_recording")
        self.assertEqual(groups["lecture"]["origins"], ["original", "retranscription"])
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            register_group(groups, "lecture", "dev", "translated", "real_recording")
        register_group(groups, "holdout", "reserved_test", "plan", "real_recording")
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            register_group(groups, "holdout", "train", "new-recipe", "real_recording")

    def test_overlapping_windows_and_episode_steps_share_sampling_blocks(self):
        def row(identity, start, end, episode):
            return {
                "id": identity,
                "metadata": {
                    "source_group_id": "recording",
                    "video_sha256": "media",
                    "source_context_seconds": [start, end],
                    "parent_episode_id": episode,
                },
            }

        cases = [
            row("original", 0, 10, "one"),
            row("nearby", 9, 15, "two"),
            row("later-step", 90, 100, "two"),
            row("independent", 150, 160, "three"),
        ]
        blocks = dependence_blocks(cases)
        self.assertEqual(blocks["original"], blocks["nearby"])
        self.assertEqual(blocks["nearby"], blocks["later-step"])
        self.assertNotEqual(blocks["original"], blocks["independent"])

    def test_unicode_media_paths_resolve_without_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = unicodedata.normalize("NFD", "미분.MP4")
            (root / name).write_bytes(b"media")
            self.assertEqual(resolve_local(root, "미분.MP4").read_bytes(), b"media")
            with self.assertRaises(ValueError):
                resolve_local(root, "../outside")

    def test_multi_item_reference_includes_gap_and_does_not_recenter_context(self):
        document = document_for(
            {
                "language": "ko",
                "paragraphs": [
                    [
                        {"id": "a", "text": "제", "ms": 200},
                        {"id": "gap", "text": "", "ms": 100},
                        {"id": "b", "text": "일", "ms": 200},
                        {"id": "c", "text": "덕목은", "ms": 400},
                    ]
                ],
            }
        )
        _, mapping = document_records(document)
        spec = {
            "action": "correct_text",
            "target_start": 0,
            "target_end": 0.5,
            "text": "제일",
        }
        self.assertEqual(action_for(spec, document, mapping)["target"], [0, 1, 2])
        with self.assertRaises(ContextBudgetError):
            action_for(spec, document, {"0": mapping["0"]})
        with self.assertRaisesRegex(ValueError, "endpoint"):
            action_for({**spec, "target_end": 0.45}, document, mapping)

    def case(self, **metadata):
        return {
            "id": "one",
            "input": {"items": [{"text": "meaningful speech"}]},
            "expectedOutput": {"action": "no_action"},
            "metadata": {
                "source_group_id": "lecture",
                "split": "train",
                "label_status": "known_preserve",
                "training_eligible": False,
                "video_sha256": "same-recording",
                "source_context_seconds": [0, 10],
                **metadata,
            },
        }

    def test_exact_conflicts_unknown_labels_and_duplicate_media_lineage_fail(self):
        case = self.case()
        self.assertEqual(audit_cases([case], ["test"])["unique_pairs"], 1)
        other = copy.deepcopy(case)
        other["id"] = "two"
        with self.assertRaisesRegex(ValueError, "Duplicate input/reference"):
            audit_cases([case, other], [])
        other["expectedOutput"] = {
            "action": "correct_text",
            "target": [0],
            "text": "different",
        }
        other["metadata"]["label_status"] = "known_edit"
        with self.assertRaisesRegex(ValueError, "conflicting references"):
            audit_cases([case, other], [])
        with self.assertRaisesRegex(ValueError, "Unknown labels"):
            audit_cases([self.case(label_status="unsupported")], [])
        with self.assertRaisesRegex(ValueError, "Reserved-test"):
            audit_cases([self.case(source_group_id="test")], ["test"])
        other["input"]["items"][0]["text"] = "different evidence"
        other["metadata"]["source_group_id"] = "disguised-retranscription"
        with self.assertRaisesRegex(ValueError, "Identical media"):
            audit_cases([case, other], [])


if __name__ == "__main__":
    unittest.main()
