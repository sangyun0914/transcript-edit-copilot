import copy
import json
import unittest

from passive_contract import Envelope, apply_passive_edit, canonical_messages
from passive_formats_v2 import messages, parse, serialize
from test_passive_contract import CORRECTION, builder, document


class PassiveFormatTests(unittest.TestCase):
    def setUp(self):
        self.document = document()
        self.builder = builder()
        self.bundle = self.builder.build(self.document, "b", as_of=0)

    def test_same_evidence_and_single_transaction_roundtrip(self):
        snapshot = self.bundle["input"]
        self.assertEqual(messages(snapshot), canonical_messages(snapshot))
        self.assertEqual(messages(snapshot)[1], messages(snapshot, "rewrite")[1])
        self.assertNotIn("task_brief", messages(snapshot, "rewrite")[1]["content"])
        for action in (
            {"action": "no_action"}, CORRECTION,
            {"action": "correct_text", "target": [0, 1, 2],
             "text": "We may require another review."},
            {"action": "split_paragraph", "target": [4]},
        ):
            with self.subTest(action=action):
                result = parse(serialize(snapshot, action, "rewrite"), snapshot, "rewrite")
                self.assertEqual(result.model_dump(), action)
        shortened = copy.deepcopy(self.document)
        for item in shortened["content"]:
            if "length" in item:
                item["length"] *= 0.5
                item["sourceStart"] *= 0.5
        before_split = self.builder.build(shortened, "b", as_of=0)
        split = {"action": "split_paragraph", "target": [4]}
        after = apply_passive_edit(shortened, before_split, split)
        state = self.builder.build(after, "d", as_of=1)["input"]
        merge = {"action": "merge_paragraph", "target": [4]}
        self.assertEqual(parse(serialize(state, merge, "rewrite"), state, "rewrite").model_dump(), merge)

    def test_rewrite_cannot_change_audio_or_add_a_second_edit(self):
        snapshot = self.bundle["input"]
        value = json.loads(serialize(snapshot, CORRECTION, "rewrite"))
        for field, replacement in (
            ("ids", [999]), ("audio", {"mode": "edges", "keep_ms": 500}),
            ("boundary_before", True),
        ):
            bad = copy.deepcopy(value)
            bad["after"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse(json.dumps(bad), snapshot, "rewrite")
        value["after"] = []
        with self.assertRaises(ValueError):
            parse(json.dumps(value), snapshot, "rewrite")

    def test_stale_or_truncated_rewrite_is_not_executable(self):
        snapshot = self.bundle["input"]
        value = serialize(snapshot, CORRECTION, "rewrite")
        current = Envelope(document_id="changed", request_id="new", revision=2, interaction_epoch=1)
        with self.assertRaises(ValueError):
            parse(value, snapshot, "rewrite", current=current)
        with self.assertRaises(ValueError):
            parse(value, snapshot, "rewrite", finish_reason="length")


if __name__ == "__main__":
    unittest.main()
