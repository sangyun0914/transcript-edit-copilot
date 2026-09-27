import json
import tempfile
import unittest
from pathlib import Path

from context_feedback_iteration import FAMILIES, case_state
from context_policy import ContextBuilder, ContextPolicy, FeedbackLedger
from grading import grade
from nep_v2 import Envelope
from nep_v3 import Snapshot, validate_prediction
from replay_session import replay
from session_cases import apply_edit


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]


class GenerationTests(unittest.TestCase):
    def test_bilingual_references_feedback_and_replay(self):
        root = Path(__file__).parent
        groups = json.loads((root / "fixtures/context-feedback-v1.json").read_text())[
            "groups"
        ]
        chosen = [
            next(
                g for g in groups if g["language"] == language and g["split"] == "train"
            )
            for language in ("en", "ko")
        ]
        actions = set()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "sessions").mkdir()
            builder = ContextBuilder(
                Tokenizer(), (root / "prompt-v3.txt").read_text(), ContextPolicy()
            )
            for group in chosen:
                for family in FAMILIES:
                    with self.subTest(group=group["id"], family=family):
                        state = case_state(group, family, builder, output)
                        bundle, expected = state["bundle"], state["expected"]
                        actions.add(expected["action"])
                        request = Envelope(
                            document_id=state["id"],
                            request_id="test",
                            revision=0,
                            interaction_epoch=0,
                        )
                        validate_prediction(
                            json.dumps(expected),
                            Snapshot.model_validate(bundle["input"]),
                            request,
                            request,
                            "stop",
                        )
                        apply_edit(state["document"], bundle, expected)
                        restored = FeedbackLedger.restore(state["ledger"].dump())
                        self.assertEqual(restored.dump(), state["ledger"].dump())
                        if family == "future_rejection":
                            self.assertEqual(bundle["input"]["recent_feedback"], [])
                        if family == "two_opportunities":
                            alternate = state["alternatives"][0]
                            case = {
                                "input": bundle["input"],
                                "expectedOutput": expected,
                                "acceptableOutputs": [alternate],
                            }
                            self.assertTrue(grade(case, alternate)["outcome_match"])
                            self.assertFalse(grade(case, alternate)["exact_match"])
            for path in (output / "sessions").glob("*.jsonl"):
                _, report = replay(
                    [json.loads(s) for s in path.read_text().splitlines()]
                )
                self.assertTrue(report["complete"])
        self.assertEqual(
            actions,
            {
                "no_action",
                "correct_text",
                "trim_silence",
                "remove_silence",
                "delete_span",
                "split_paragraph",
                "merge_paragraph",
            },
        )


if __name__ == "__main__":
    unittest.main()
