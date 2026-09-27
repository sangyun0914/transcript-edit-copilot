import json
import unittest
from pathlib import Path

from nep import EDIT, Envelope, Snapshot, validate_prediction

CASES = [
    json.loads(line)
    for line in Path(__file__).with_name("seed-cases.jsonl").read_text().splitlines()
]
REQUEST = Envelope(
    document_id="test", request_id="request-1", revision=1, interaction_epoch=2
)


class ContractTests(unittest.TestCase):
    def validate(self, prediction, case=0, **kwargs):
        return validate_prediction(
            json.dumps(prediction),
            Snapshot.model_validate(CASES[case]["input"]),
            REQUEST,
            kwargs.get("current", REQUEST),
            kwargs.get("finish_reason", "stop"),
        )

    def test_draft_expectations_obey_the_contract(self):
        for index, case in enumerate(CASES):
            with self.subTest(case=case["id"]):
                self.assertEqual(
                    self.validate(case["expectedOutput"], index).model_dump(),
                    case["expectedOutput"],
                )

    def test_schema_file_matches_validator(self):
        self.assertEqual(
            json.loads(Path(__file__).with_name("edit.schema.json").read_text()),
            EDIT.json_schema(),
        )

    def test_unknown_extra_malformed_and_incomplete_outputs_are_rejected(self):
        for raw in (
            '{"action":"no_action"',
            '{"action":"no_action"} extra',
            "[]",
            '{"action":"delete_span","target":[true]}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_prediction(
                    raw,
                    Snapshot.model_validate(CASES[0]["input"]),
                    REQUEST,
                    REQUEST,
                    "stop",
                )
        for prediction in (
            {"action": "no_action", "confidence": 1},
            {"action": "delete_span", "target": []},
            {"action": "delete_span", "target": [999]},
            {"action": "correct_text", "target": [1], "text": " "},
        ):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                self.validate(prediction)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.validate({"action": "no_action"}, finish_reason="length")

    def test_stale_revision_cursor_epoch_and_request_are_rejected(self):
        for field, value in (
            ("revision", 2),
            ("interaction_epoch", 3),
            ("document_id", "other"),
            ("request_id", "other"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Stale"):
                self.validate(
                    {"action": "no_action"},
                    current=REQUEST.model_copy(update={field: value}),
                )

    def test_audio_semantic_guards(self):
        for prediction, case in (
            ({"action": "delete_span", "target": [0]}, 8),
            ({"action": "delete_span", "target": [0]}, 9),
            ({"action": "delete_span", "target": [0]}, 11),
            ({"action": "delete_span", "target": [0, 1]}, 4),
            ({"action": "delete_span", "target": [0, 2]}, 0),
            ({"action": "delete_span", "target": [1, 0]}, 0),
            ({"action": "delete_span", "target": [0, 0]}, 0),
            ({"action": "trim_silence", "target": [1], "keep_ms": 300}, 7),
            ({"action": "trim_silence", "target": [0], "keep_ms": 500}, 0),
        ):
            with (
                self.subTest(prediction=prediction, case=case),
                self.assertRaises(ValueError),
            ):
                self.validate(prediction, case)

    def test_same_speaker_deletion_cannot_join_discontinuous_source_runs(self):
        snapshot = Snapshot.model_validate(CASES[0]["input"])
        snapshot.items[1].source_run = 1
        with self.assertRaisesRegex(ValueError, "source-run"):
            validate_prediction(
                '{"action":"delete_span","target":[0,1]}',
                snapshot,
                REQUEST,
                REQUEST,
                "stop",
            )


if __name__ == "__main__":
    unittest.main()
