import copy
import unittest

from context_policy import ContextBuilder, ContextPolicy
from session_cases import apply_edit
from synthetic_pilot import document_for, symbolic_evidence
from test_context_policy import Tokenizer
from verify_release_export import Exporter, verify_export_delta, verify_operation


def fixture():
    document = document_for(
        {
            "language": "en",
            "paragraphs": [
                [
                    {"id": "a", "text": "A: hello.\nB: Hi.", "ms": 600},
                    {"id": "gap", "text": "", "ms": 1400},
                    {"id": "b", "text": "Less < more.", "ms": 900},
                    {"id": "c", "text": "Today.", "ms": 700},
                ],
                [{"id": "event", "text": "", "ms": 1000}],
                [{"id": "d", "text": "Keep this utterance.", "ms": 2000}],
            ],
        }
    )
    builder = ContextBuilder(
        Tokenizer(),
        "rules",
        ContextPolicy(schema_version=4, input_tokens=6000, context_tokens=6256),
    )
    bundle = builder.build(
        document,
        "b",
        "Preserve outside content.",
        as_of=0,
        evidence=symbolic_evidence(document),
    )
    return document, bundle


class ExportAuditTests(unittest.TestCase):
    def setUp(self):
        self.document, self.bundle = fixture()
        self.exporter = Exporter()

    def tearDown(self):
        self.exporter.close()

    def test_actual_export_retains_event_cues_multiline_speech_and_abstention(self):
        before = self.exporter.export(self.document, None)
        self.assertEqual(len(before["cues"]), 3)
        self.assertEqual(before["cues"][1]["text_ids"], [])
        self.assertEqual(before["cues"][0]["text_ids"], ["a", "b", "c"])
        operation = verify_operation(
            self.document,
            copy.deepcopy(self.document),
            self.bundle["mapping"],
            {"action": "no_action"},
        )
        result = verify_export_delta(before, before, operation, "no_action")
        self.assertEqual(result["baseline_empty_cues"], 1)
        self.assertEqual(result["retained_empty_cues"], 1)
        self.assertEqual(result["introduced_empty_cues"], 0)

    def test_source_trim_duration_and_outside_utterances_are_checked(self):
        action = {"action": "trim_silence", "target": [1], "keep_ms": 725}
        after = apply_edit(self.document, self.bundle, action)
        operation = verify_operation(
            self.document, after, self.bundle["mapping"], action
        )
        result = verify_export_delta(
            self.exporter.export(self.document, None),
            self.exporter.export(after, None),
            operation,
            action["action"],
        )
        self.assertAlmostEqual(result["removed_source_seconds"], 0.675)
        self.assertEqual(result["untouched_cues_checked"], 2)
        damaged = copy.deepcopy(after)
        damaged["content"][3]["sourceStart"] -= 0.05
        with self.assertRaisesRegex(ValueError, "outer source boundary"):
            verify_operation(self.document, damaged, self.bundle["mapping"], action)

    def test_a_correction_cannot_drop_an_unrelated_utterance(self):
        action = {
            "action": "correct_text",
            "target": [2, 3],
            "text": "Less < more today.",
        }
        after = apply_edit(self.document, self.bundle, action)
        operation = verify_operation(
            self.document, after, self.bundle["mapping"], action
        )
        verify_export_delta(
            self.exporter.export(self.document, None),
            self.exporter.export(after, None),
            operation,
            action["action"],
        )
        after["content"][-2]["text"] = "An accidental overwrite."
        with self.assertRaisesRegex(ValueError, "outside its target"):
            verify_operation(self.document, after, self.bundle["mapping"], action)

    def test_new_empty_cue_is_not_excused_by_an_existing_event_cue(self):
        before = self.exporter.export(self.document, None)
        after = copy.deepcopy(before)
        after["cues"][0]["empty"] = True
        operation = {
            "removed_seconds": 0.0,
            "target_end_seconds": 0.6,
            "affected_paragraphs": ["p0"],
        }
        with self.assertRaisesRegex(ValueError, "introduced empty"):
            verify_export_delta(before, after, operation, "correct_text")
        after = copy.deepcopy(before)
        after["export_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Abstention changed"):
            verify_export_delta(before, after, operation, "no_action")


if __name__ == "__main__":
    unittest.main()
