import copy
import json
import unittest
from pathlib import Path

from replay_session import canonical, read_events, replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.events = read_events(
            Path(__file__).parent / "fixtures/editing-session.jsonl"
        )

    def test_replays_javascript_ui_fixture_and_reports_completion(self):
        document, summary = replay(self.events)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["transactions"], 4)
        self.assertEqual(
            [i["text"] for i in document["content"] if i["type"] == "text"],
            ["Hello", "world"],
        )
        self.assertAlmostEqual(
            sum(i.get("length", 0) for i in document["content"]), 2.725
        )
        self.assertEqual(
            sum(i["type"] == "paragraph_start" for i in document["content"]), 2
        )
        _, prefix = replay(self.events[:-1])
        self.assertFalse(prefix["complete"])

    def test_canonical_keeps_json_escaping_literals_and_javascript_key_order(self):
        for value in (
            "",
            'quote" slash\\ control\b\n\t\x00',
            "한글 🙂 é\u2028",
            None,
            True,
            False,
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    canonical(value),
                    json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                )
        self.assertEqual(
            canonical({"\ue000": None, "𐀀": True}), '{"𐀀":true,"\ue000":null}'
        )
        self.assertEqual(canonical(-0.0), canonical(0))
        self.assertNotEqual(canonical(True), canonical(1))
        for value in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                canonical(value)

    def test_corrupt_before_image_missing_events_and_post_end_are_rejected(self):
        changed = copy.deepcopy(self.events)
        transaction = next(e for e in changed if e["kind"] == "transaction")
        transaction["patch"]["remove"][0]["sourceStart"] = 99
        for events in (
            changed,
            self.events[1:],
            self.events[:3] + self.events[4:],
            self.events + [self.events[-1]],
        ):
            with self.subTest(events=len(events)), self.assertRaises(ValueError):
                replay(events)


if __name__ == "__main__":
    unittest.main()
