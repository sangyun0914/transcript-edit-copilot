import copy
import json
import unittest

from context_policy import ContextBudgetError, FeedbackLedger
from passive_contract import (
    ACTIONS,
    ALLOWED_ACTIONS,
    EDIT,
    GATED_ACTIONS,
    PROMPT,
    VERSION,
    Envelope,
    PassiveContextBuilder,
    PassiveFeedbackLedger,
    PassivePolicy,
    PassivePredictionSession,
    apply_passive_edit,
    canonical_messages,
    committed_transition,
    parse_snapshot,
    token_ids,
    validate_prediction,
    wire_snapshot,
)
from replay_session import session_hash
from synthetic_pilot import document_for


class Tokenizer:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        text = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return list(text.encode()) * self.multiplier


def builder(max_items=64, **kwargs):
    return PassiveContextBuilder(
        Tokenizer(),
        policy=PassivePolicy(
            max_items=max_items,
            input_tokens=30000,
            context_tokens=31024,
            output_tokens=1024,
        ),
        **kwargs,
    )


def document():
    return document_for(
        {
            "language": "en",
            "paragraphs": [
                [
                    {"id": "a", "text": "We may", "ms": 1000},
                    {"id": "b", "text": "requier", "ms": 1000},
                    {"id": "c", "text": "another review.", "ms": 1000},
                    {"id": "gap", "text": "", "ms": 1400},
                    {"id": "d", "text": "We may require", "ms": 1600},
                    {"id": "e", "text": "another review.", "ms": 1500},
                ]
            ],
        }
    )


REQUEST = Envelope(document_id="doc", request_id="r", revision=0, interaction_epoch=0)
CORRECTION = {"action": "correct_text", "target": [1], "text": "require"}


class PassiveContractTests(unittest.TestCase):
    def setUp(self):
        self.doc = document()
        self.builder = builder()

    def bundle(self, document=None, **kwargs):
        return self.builder.build(
            document or self.doc, "b", as_of=kwargs.pop("as_of", 0), **kwargs
        )

    def test_runtime_and_offline_messages_tokens_and_compact_roundtrip_are_identical(
        self,
    ):
        original = self.bundle()
        after = apply_passive_edit(self.doc, original, CORRECTION)
        transaction = committed_transition(self.doc, after, CORRECTION, 3)
        session = PassivePredictionSession("doc", self.builder)
        session.feedback.propose("p", 1, self.doc, original, CORRECTION)
        session.feedback.record("p", "shown", 2)
        session.feedback.record("p", "accepted", 3, transition=transaction)
        offline = self.builder.build(
            after, "b", as_of=3, history=[transaction], feedback=session.feedback
        )
        runtime = session.prepare(after, "b", as_of=3, history=[transaction])
        self.assertEqual(offline, runtime)
        self.assertEqual(
            canonical_messages(offline["input"]), canonical_messages(runtime["input"])
        )
        self.assertEqual(
            token_ids(self.builder.tokenizer, offline["input"]),
            token_ids(self.builder.tokenizer, runtime["input"]),
        )
        messages, options = self.builder.tokenizer.calls[-1]
        self.assertEqual(messages[0], {"role": "system", "content": PROMPT})
        self.assertEqual(
            options,
            {
                "tokenize": True,
                "return_dict": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )
        wire = json.loads(messages[1]["content"])
        self.assertEqual(parse_snapshot(wire).model_dump(), offline["input"])
        self.assertEqual(wire["contract_version"], VERSION)
        self.assertNotIn("task_brief", wire)
        self.assertNotIn("boundary_status", wire["item_fields"])
        self.assertNotIn("audio_class", wire["item_fields"])
        self.assertNotIn("transaction_id", messages[1]["content"])
        self.assertEqual(offline["input"]["recent_edits"][0]["before"], "requier")
        self.assertEqual(offline["input"]["recent_feedback"][0]["outcome"], "accepted")
        restored = PassiveFeedbackLedger.restore(session.feedback.dump())
        self.assertEqual(
            offline,
            self.builder.build(
                after, "b", as_of=3, history=[transaction], feedback=restored
            ),
        )

    def test_authoring_and_acoustic_fields_are_rejected_in_both_input_encodings(self):
        value = self.bundle()["input"]
        for key in (
            "task_brief",
            "goal",
            "teacher_rationale",
            "reference",
            "source_final",
            "evidence",
            "schema_version",
        ):
            for encode in (lambda x: x, wire_snapshot):
                with (
                    self.subTest(key=key, encoding=encode),
                    self.assertRaises(ValueError),
                ):
                    parse_snapshot({**encode(value), key: "privileged"})
        for key in (
            "boundary_status",
            "audio_class",
            "reviewed_boundary",
            "reference_text",
        ):
            forged = copy.deepcopy(value)
            forged["items"][0][key] = "verified"
            with self.subTest(key=key), self.assertRaises(ValueError):
                canonical_messages(forged)
        for key in ("task_brief", "evidence", "allowed_actions", "source_final"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                self.builder.build(self.doc, "b", as_of=0, **{key: "override"})
        with self.assertRaises(ValueError):
            self.bundle(feedback=FeedbackLedger("doc"))

    def test_all_seven_actions_have_shapes_but_sound_cuts_cannot_be_enabled(self):
        shapes = [
            CORRECTION,
            {"action": "no_action"},
            {"action": "delete_span", "target": [1]},
            {"action": "trim_silence", "target": [3], "keep_ms": 500},
            {"action": "remove_silence", "target": [3]},
            {"action": "split_paragraph", "target": [4]},
            {"action": "merge_paragraph", "target": [4]},
        ]
        self.assertEqual({EDIT.validate_python(e).action for e in shapes}, set(ACTIONS))
        bundle = self.bundle()
        self.assertEqual(bundle["input"]["allowed_actions"], list(ALLOWED_ACTIONS))
        for edit in shapes:
            if edit["action"] not in GATED_ACTIONS:
                continue
            with (
                self.subTest(action=edit["action"]),
                self.assertRaisesRegex(ValueError, "not allowed"),
            ):
                apply_passive_edit(self.doc, bundle, edit)
            forged = copy.deepcopy(bundle)
            forged["input"]["allowed_actions"].append(edit["action"])
            with self.assertRaisesRegex(ValueError, "gated"):
                apply_passive_edit(self.doc, forged, edit)

    def test_correction_is_display_only_and_no_action_is_exactly_invariant(self):
        bundle = self.bundle()
        edit = {
            "action": "correct_text",
            "target": [0, 1, 2],
            "text": "We may require another review.",
        }
        after = apply_passive_edit(self.doc, bundle, edit)
        self.assertEqual(after["content"][1]["sourceStart"], 0.0)
        self.assertEqual(after["content"][1]["length"], 3.0)
        self.assertEqual(after["content"][2:], self.doc["content"][4:])
        self.assertEqual(after["metadata"], self.doc["metadata"])
        self.assertEqual(
            apply_passive_edit(self.doc, bundle, {"action": "no_action"}), self.doc
        )
        with self.assertRaises(ValueError):
            apply_passive_edit(
                self.doc,
                bundle,
                {"action": "correct_text", "target": [1], "text": "requier"},
            )

    def test_cropped_caption_facts_allow_local_split_without_claiming_completeness(
        self,
    ):
        doc = document_for(
            {
                "language": "ko",
                "paragraphs": [
                    [
                        {
                            "id": f"w{n}",
                            "text": "다음 문장.😀" if n % 2 else "앞 문장입니다.",
                            "ms": 300,
                        }
                        for n in range(120)
                    ],
                    [{"id": "next", "text": "다음 단락입니다.", "ms": 1800}],
                ],
            }
        )
        narrow = builder(max_items=12)
        bundle = narrow.build(doc, "w60", as_of=0)
        self.assertEqual(bundle["input"]["complete_paragraphs"], [])
        self.assertEqual(
            parse_snapshot(wire_snapshot(bundle["input"])).model_dump(), bundle["input"]
        )
        after = apply_passive_edit(
            doc, bundle, {"action": "split_paragraph", "target": [60]}
        )
        self.assertEqual(
            [i for i in after["content"] if "source" in i],
            [i for i in doc["content"] if "source" in i],
        )
        at_edge = narrow.build(doc, "next", as_of=0)
        with self.assertRaisesRegex(ValueError, "Insufficient visible context"):
            apply_passive_edit(
                doc, at_edge, {"action": "merge_paragraph", "target": [120]}
            )

    def test_caption_minimum_and_speaker_constraints_survive_conversion(self):
        short = document_for(
            {
                "language": "en",
                "paragraphs": [
                    [
                        {"id": "a", "text": "One.", "ms": 400},
                        {"id": "b", "text": "Two.", "ms": 1500},
                    ]
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "below the profile minimum"):
            apply_passive_edit(
                short, self.bundle(short), {"action": "split_paragraph", "target": [1]}
            )
        doc = document_for(
            {
                "language": "en",
                "paragraphs": [
                    [{"id": "a", "text": "A short", "ms": 1200}],
                    [{"id": "b", "text": "sentence.", "ms": 1200}],
                ],
            }
        )
        merged = apply_passive_edit(
            doc, self.bundle(doc), {"action": "merge_paragraph", "target": [1]}
        )
        self.assertEqual(
            sum(i["type"] == "paragraph_start" for i in merged["content"]), 1
        )
        doc["content"][3]["speaker"] = "other"
        with self.assertRaisesRegex(ValueError, "speakers"):
            apply_passive_edit(
                doc, self.bundle(doc), {"action": "merge_paragraph", "target": [1]}
            )

    def test_rebinding_rejects_forged_display_facts_caption_metrics_and_stale_sources(
        self,
    ):
        bundle = self.bundle()
        for kind in ("text", "duration", "caption", "mapping", "profile"):
            forged = copy.deepcopy(bundle)
            if kind == "text":
                forged["input"]["items"][1]["text"] = "requirx"
            elif kind == "duration":
                forged["input"]["items"][1]["duration_ms"] += 1
            elif kind == "caption":
                forged["input"]["caption_facts"][0]["duration_ms"] += 100
            elif kind == "mapping":
                forged["mapping"]["1"]["index"] += 1
            else:
                forged["input"]["caption_profile"]["min_cue_ms"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                apply_passive_edit(self.doc, forged, {"action": "no_action"})
        changed = copy.deepcopy(self.doc)
        changed["content"][2]["text"] = "require"
        with self.assertRaisesRegex(ValueError, "Stale"):
            apply_passive_edit(changed, bundle, CORRECTION)

    def test_history_is_verified_past_state_not_an_authored_edit_narrative(self):
        baseline = self.bundle()
        after = apply_passive_edit(self.doc, baseline, CORRECTION)
        first = committed_transition(self.doc, after, CORRECTION, 1)
        later_bundle = self.bundle(after, as_of=1, history=[first])
        second_edit = {"action": "correct_text", "target": [4], "text": "We may need"}
        second_after = apply_passive_edit(after, later_bundle, second_edit)
        second = committed_transition(after, second_after, second_edit, 2)
        self.assertEqual(
            later_bundle, self.bundle(after, as_of=1, history=[first, second])
        )
        with self.assertRaisesRegex(ValueError, "final committed history state"):
            self.bundle(self.doc, as_of=1, history=[first])
        with self.assertRaises(ValueError):
            self.bundle(
                after,
                as_of=1,
                history=[
                    {
                        "sequence": 1,
                        "edit": {
                            "action": "correct_text",
                            "origin": "human",
                            "before": "may",
                            "after": "will",
                        },
                    }
                ],
            )
        forged = copy.deepcopy(first)
        forged["after"]["content"][-2]["length"] += 0.5
        with self.assertRaisesRegex(ValueError, "outside"):
            self.bundle(after, as_of=1, history=[forged])
        with self.assertRaises(ValueError):
            self.bundle(second_after, as_of=2, history=[second, first])
        self.assertEqual(self.bundle(after)["input"]["recent_edits"], [])

    def test_feedback_acceptance_requires_committed_delta_and_matching_history(self):
        bundle = self.bundle()
        ledger = PassiveFeedbackLedger("doc")
        ledger.propose("p", 1, self.doc, bundle, CORRECTION)
        ledger.record("p", "shown", 2)
        with self.assertRaisesRegex(ValueError, "actual committed transition"):
            ledger.record(
                "p", "accepted", 3, applied=CORRECTION, transaction_id="fiction"
            )
        after = apply_passive_edit(self.doc, bundle, CORRECTION)
        transition = committed_transition(self.doc, after, CORRECTION, 3)
        with self.assertRaisesRegex(ValueError, "actor"):
            ledger.record("p", "accepted", 3, origin="human", transition=transition)
        different = {**CORRECTION, "text": "need"}
        modified = committed_transition(
            self.doc, apply_passive_edit(self.doc, bundle, different), different, 3
        )
        with self.assertRaisesRegex(ValueError, "modified acceptance"):
            ledger.record("p", "accepted", 3, transition=modified)
        ledger.record("p", "accepted", 3, transition=transition)
        with self.assertRaisesRegex(ValueError, "absent from actual committed history"):
            self.bundle(after, as_of=3, feedback=ledger)
        self.bundle(after, as_of=3, history=[transition], feedback=ledger)

    def test_dismissal_is_local_and_undo_restores_real_state_and_suppression(self):
        session = PassivePredictionSession("doc", self.builder)
        initial = session.prepare(self.doc, "b", as_of=0)
        session.feedback.propose("p", 1, self.doc, initial, CORRECTION)
        session.feedback.record("p", "shown", 2)
        session.feedback.record("p", "dismissed", 3)
        current = session.prepare(self.doc, "b", as_of=3)
        response = session.validate(
            json.dumps(CORRECTION), self.doc, current, REQUEST, REQUEST, "stop", 3
        )
        self.assertTrue(response["suppressed"])
        other = {"action": "correct_text", "target": [4], "text": "We may need"}
        self.assertTrue(
            session.validate(
                json.dumps(other), self.doc, current, REQUEST, REQUEST, "stop", 3
            )["show"]
        )
        self.assertIn(
            "whole visible window", canonical_messages(current["input"])[0]["content"]
        )

        ledger = PassiveFeedbackLedger("doc")
        ledger.propose("p", 1, self.doc, initial, CORRECTION)
        ledger.record("p", "shown", 2)
        after = apply_passive_edit(self.doc, initial, CORRECTION)
        commit = committed_transition(self.doc, after, CORRECTION, 3)
        ledger.record("p", "accepted", 3, transition=commit)
        undo = committed_transition(
            after, self.doc, CORRECTION, 4, undo_of=commit["transaction_id"]
        )
        ledger.record("p", "undone", 4, transition=undo)
        restored = self.bundle(as_of=4, history=[commit, undo], feedback=ledger)
        self.assertEqual(restored["input"]["recent_edits"], [])
        self.assertEqual(restored["input"]["recent_feedback"][0]["outcome"], "undone")
        self.assertTrue(ledger.suppressed(self.doc, restored, CORRECTION, 4))
        self.assertEqual(
            PassiveFeedbackLedger.restore(ledger.dump()).dump(), ledger.dump()
        )

    def test_runtime_rejects_changed_context_stale_request_and_incomplete_output(self):
        session = PassivePredictionSession("doc", self.builder)
        bundle = session.prepare(self.doc, "b", as_of=0)
        forged = copy.deepcopy(bundle)
        forged["input"]["recent_edits"] = [
            {
                "action": "correct_text",
                "origin": "human",
                "before": "may",
                "after": "will",
            }
        ]
        with self.assertRaisesRegex(ValueError, "unissued"):
            session.validate(
                json.dumps(CORRECTION), self.doc, forged, REQUEST, REQUEST, "stop", 0
            )
        newer = REQUEST.model_copy(update={"revision": 1})
        with self.assertRaisesRegex(ValueError, "Stale"):
            session.validate(
                json.dumps(CORRECTION), self.doc, bundle, REQUEST, newer, "stop", 0
            )
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_prediction(
                json.dumps(CORRECTION), bundle["input"], REQUEST, REQUEST, "length"
            )
        result = session.validate(
            '{"action":"no_action"}', self.doc, bundle, REQUEST, REQUEST, "stop", 0
        )
        self.assertFalse(result["show"])

    def test_full_prompt_budget_uses_all_tokenizers_and_keeps_cursor_selection(self):
        doc = document_for(
            {
                "language": "en",
                "paragraphs": [
                    [
                        {"id": f"w{n}", "text": f"Context word {n}.", "ms": 400}
                        for n in range(100)
                    ]
                ],
            }
        )
        tokenizer = Tokenizer()
        larger = Tokenizer(2)
        policy = PassivePolicy(input_tokens=12000, context_tokens=12512)
        shared = PassiveContextBuilder(tokenizer, {"larger": larger}, policy)
        bundle = shared.build(doc, "w50", as_of=0, selection=["w48", "w52"])
        self.assertTrue(bundle["context"]["budget_reduced"])
        self.assertLessEqual(
            bundle["context"]["comparison_token_counts"]["larger"], 12000
        )
        self.assertTrue({48, 50, 52} <= {i["id"] for i in bundle["input"]["items"]})
        self.assertEqual(
            bundle["context"]["prompt_hash"],
            session_hash(canonical_messages(bundle["input"])),
        )
        with self.assertRaises(ContextBudgetError):
            PassiveContextBuilder(
                tokenizer, policy=PassivePolicy(input_tokens=20)
            ).build(doc, "w50", as_of=0)


if __name__ == "__main__":
    unittest.main()
