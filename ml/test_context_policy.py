import json
import unittest

from context_policy import (
    ContextBudgetError,
    ContextBuilder,
    ContextPolicy,
    FeedbackLedger,
    PredictionSession,
    messages,
)
from nep_v2 import Envelope
from nep_v3 import Snapshot
from session_cases import apply_edit
from synthetic_pilot import document_for


class Tokenizer:
    def apply_chat_template(self, conversation, **kwargs):
        return list(
            range(len(json.dumps(conversation, ensure_ascii=False).encode()) // 5 + 7)
        )


def document():
    return document_for(
        {
            "language": "en",
            "paragraphs": [
                [
                    {"id": f"w{p}-{n}", "text": f"word{p}-{n}", "ms": 300}
                    for n in range(30)
                ]
                for p in range(5)
            ],
        }
    )


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.doc = document()
        self.builder = ContextBuilder(
            Tokenizer(), "rules", ContextPolicy(input_tokens=2000)
        )
        self.brief = "Preserve meaning."

    def bundle(self, **kwargs):
        return self.builder.build(
            self.doc, "w2-10", self.brief, as_of=kwargs.pop("as_of", 20), **kwargs
        )

    def proposal(self, ledger, bundle=None, prediction=None):
        bundle = bundle or self.bundle()
        prediction = prediction or {
            "action": "correct_text",
            "target": [70],
            "text": "revised",
        }
        ledger.propose("p", 1, self.doc, bundle, prediction)
        ledger.record("p", "shown", 2)
        return bundle, prediction

    def test_runtime_and_offline_bytes_and_budget_match(self):
        session = PredictionSession("doc", self.builder)
        offline = self.bundle()
        runtime = session.prepare(self.doc, "w2-10", self.brief, as_of=20)
        self.assertEqual(
            messages(offline["input"], "rules"), messages(runtime["input"], "rules")
        )
        self.assertEqual(offline["context"], runtime["context"])
        self.assertTrue(offline["context"]["budget_reduced"])
        self.assertLessEqual(offline["context"]["input_tokens"], 2000)
        self.assertIn(70, [i["id"] for i in offline["input"]["items"]])
        for paragraph in offline["input"]["complete_paragraphs"]:
            self.assertEqual(
                sum(i["paragraph"] == paragraph for i in offline["input"]["items"]), 30
            )

    def test_common_view_respects_the_most_restrictive_tokenizer(self):
        class LargerTokenizer(Tokenizer):
            def apply_chat_template(self, conversation, **kwargs):
                return super().apply_chat_template(conversation, **kwargs) * 2

        larger = LargerTokenizer()
        policy = ContextPolicy(input_tokens=2500)
        common = ContextBuilder(Tokenizer(), "rules", policy, {"larger": larger})
        restrictive = ContextBuilder(larger, "rules", policy)
        left = common.build(self.doc, "w2-10", self.brief, as_of=0)
        right = restrictive.build(self.doc, "w2-10", self.brief, as_of=0)
        self.assertEqual(left["input"], right["input"])
        self.assertEqual(left["mapping"], right["mapping"])
        self.assertLessEqual(left["context"]["comparison_token_counts"]["larger"], 2500)
        self.assertLess(
            left["context"]["input_tokens"], right["context"]["input_tokens"]
        )

    def test_item_window_and_selection_are_not_answer_centered(self):
        b = ContextBuilder(
            Tokenizer(),
            "rules",
            ContextPolicy(input_tokens=20000, context_tokens=21000),
        )
        bundle = b.build(self.doc, "w2-10", self.brief, as_of=20)
        self.assertEqual(
            [i["id"] for i in bundle["input"]["items"]], list(range(38, 102))
        )
        selected = b.build(
            self.doc,
            "w2-10",
            self.brief,
            as_of=20,
            selection=["w3-20"],
            protected=["w3-20"],
        )
        self.assertEqual(selected["input"]["selection"], [110])
        self.assertEqual(selected["input"]["protected_ids"], [110])
        with self.assertRaises(ContextBudgetError):
            b.build(self.doc, "w0-0", self.brief, as_of=20, selection=["w4-29"])
        tiny = ContextBuilder(Tokenizer(), "rules", ContextPolicy(input_tokens=10))
        with self.assertRaises(ContextBudgetError):
            tiny.build(self.doc, "w2-10", self.brief, as_of=20)

    def test_past_only_history_and_feedback_without_mutating_earlier_snapshot(self):
        ledger = FeedbackLedger("doc")
        self.proposal(ledger)
        ledger.record(
            "p",
            "accepted",
            3,
            applied={"action": "correct_text", "target": [70], "text": "revised"},
            transaction_id="tx",
        )
        ledger.record("p", "undone", 12, transaction_id="tx")
        history = [
            {
                "sequence": seq,
                "edit": {
                    "action": "correct_text",
                    "origin": "synthetic",
                    "before": "x",
                    "after": str(seq),
                },
            }
            for seq in [2, 15]
        ]
        early = self.bundle(as_of=5, history=history, feedback=ledger)
        late = self.bundle(as_of=20, history=history, feedback=ledger)
        self.assertEqual([e["after"] for e in early["input"]["recent_edits"]], ["2"])
        self.assertEqual(early["input"]["recent_feedback"][0]["outcome"], "accepted")
        self.assertEqual(late["input"]["recent_feedback"][0]["outcome"], "undone")
        self.assertFalse(
            ledger.suppressed(
                self.doc,
                early,
                {"action": "correct_text", "target": [70], "text": "revised"},
                5,
            )
        )
        self.assertTrue(
            ledger.suppressed(
                self.doc,
                late,
                {"action": "correct_text", "target": [70], "text": "revised"},
                20,
            )
        )

    def test_rejection_survives_unrelated_edits_but_not_changed_evidence(self):
        ledger = FeedbackLedger("doc")
        bundle, prediction = self.proposal(ledger)
        ledger.record("p", "dismissed", 3)
        self.assertTrue(ledger.suppressed(self.doc, bundle, prediction, 20))
        next(i for i in self.doc["content"] if i["uuid"] == "w0-0")["text"] = (
            "unrelated"
        )
        bundle = self.bundle(feedback=ledger)
        self.assertTrue(ledger.suppressed(self.doc, bundle, prediction, 20))
        self.assertFalse(
            ledger.suppressed(self.doc, bundle, {**prediction, "target": [71]}, 20)
        )
        self.assertFalse(
            ledger.suppressed(self.doc, bundle, {**prediction, "text": "different"}, 20)
        )
        self.brief += " New spelling evidence is available."
        self.assertFalse(ledger.suppressed(self.doc, self.bundle(), prediction, 20))
        self.brief = "Preserve meaning."
        next(i for i in self.doc["content"] if i["uuid"] == "w2-9")["text"] = (
            "changed evidence"
        )
        self.assertFalse(ledger.suppressed(self.doc, self.bundle(), prediction, 20))

    def test_equivalent_correction_is_suppressed_and_silent_outcomes_are_not_rejections(
        self,
    ):
        ledger = FeedbackLedger("doc")
        bundle, prediction = self.proposal(ledger)
        ledger.record("p", "dismissed", 3)
        equivalent = {
            "action": "correct_text",
            "target": [69, 70],
            "text": "word2-9 revised",
        }
        self.assertTrue(ledger.suppressed(self.doc, bundle, equivalent, 20))
        for outcome in ("expired", "invalidated"):
            other = FeedbackLedger("doc")
            self.proposal(other)
            other.record("p", outcome, 3)
            self.assertFalse(other.suppressed(self.doc, bundle, prediction, 20))

    def test_feedback_requires_exposure_and_valid_transaction_and_refs(self):
        ledger = FeedbackLedger("doc")
        bundle = self.bundle()
        prediction = {"action": "correct_text", "target": [70], "text": "revised"}
        ledger.propose("p", 1, self.doc, bundle, prediction)
        with self.assertRaises(ValueError):
            ledger.record("p", "dismissed", 2)
        ledger.record("p", "shown", 2)
        with self.assertRaises(ValueError):
            ledger.record("p", "accepted", 3)
        applied = {**prediction, "text": "alternative"}
        ledger.record("p", "modified", 3, applied=applied, transaction_id="tx")
        ledger.record("p", "undone", 4, transaction_id="tx")
        value = self.bundle(feedback=ledger)["input"]
        self.assertEqual(value["recent_feedback"][0]["applied_text"], "alternative")
        self.assertTrue(ledger.suppressed(self.doc, bundle, applied, 20))
        self.assertFalse(ledger.suppressed(self.doc, bundle, prediction, 20))
        self.assertTrue(
            ledger.suppressed(
                self.doc,
                bundle,
                {
                    "action": "correct_text",
                    "target": [69, 70],
                    "text": "word2-9 alternative",
                },
                20,
            )
        )
        value["recent_feedback"][0]["visible_targets"] = [9999]
        with self.assertRaises(ValueError):
            Snapshot.model_validate(value)

    def test_zero_history_conditions_and_runtime_freshness(self):
        session = PredictionSession(
            "doc",
            ContextBuilder(
                Tokenizer(), "rules", ContextPolicy(history_limit=0, feedback_limit=0)
            ),
        )
        bundle, prediction = self.proposal(session.feedback)
        session.feedback.record("p", "dismissed", 3)
        actual = session.prepare(self.doc, "w2-10", self.brief, as_of=20)
        self.assertEqual(actual["input"]["recent_feedback"], [])
        request = Envelope(
            document_id="doc", request_id="req", revision=2, interaction_epoch=1
        )
        result = session.validate(
            json.dumps(prediction), self.doc, actual, request, request, "stop", 20
        )
        self.assertTrue(result["suppressed"])
        self.assertFalse(result["show"])
        stale = request.model_copy(update={"revision": 3})
        with self.assertRaises(ValueError):
            session.validate(
                json.dumps(prediction), self.doc, actual, request, stale, "stop", 20
            )
        changed = apply_edit(self.doc, actual, prediction)
        with self.assertRaises(ValueError):
            session.validate(
                json.dumps(prediction), changed, actual, request, request, "stop", 20
            )


if __name__ == "__main__":
    unittest.main()
