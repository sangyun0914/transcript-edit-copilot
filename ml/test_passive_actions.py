import copy
import json
import unittest

import passive_actions as actions
from context_policy import ContextBudgetError
from synthetic_pilot import document_for
from verify_release_export import verify_operation


class Tokenizer:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        return list(json.dumps(messages, ensure_ascii=False).encode()) * self.multiplier


def builder(max_items=64, **kwargs):
    return actions.PassiveContextBuilder(
        Tokenizer(),
        policy=actions.PassivePolicy(
            max_items=max_items, input_tokens=40000, context_tokens=40512
        ),
        **kwargs,
    )


def document(language="en", long=False):
    words = ("Well", "we we", "can begin.", "Next item.")
    if language == "ko":
        words = ("음", "지금 지금", "시작할 수 있습니다.", "다음 항목입니다.")
    return document_for(
        {
            "language": language,
            "paragraphs": [
                [
                    {"id": "a", "text": words[0], "ms": 3000 if long else 1000},
                    {"id": "b", "text": words[1], "ms": 1000},
                    {"id": "c", "text": words[2], "ms": 1000},
                    {"id": "gap", "text": "", "ms": 7000 if long else 1400},
                    {"id": "d", "text": words[3], "ms": 1500},
                ]
            ],
        }
    )


def item(doc, uuid):
    return next(value for value in doc["content"] if value["uuid"] == uuid)


REQUEST = actions.Envelope(
    document_id="doc", request_id="request", revision=0, interaction_epoch=0
)


class PassiveActionsTests(unittest.TestCase):
    def setUp(self):
        self.doc = document()
        self.builder = builder()

    def bundle(self, doc=None, **kwargs):
        return self.builder.build(
            doc or self.doc, "b", as_of=kwargs.pop("as_of", 0), **kwargs
        )

    def test_bilingual_wire_tokens_and_fixed_policy_keep_unknown_metadata(self):
        for language in ("en", "ko"):
            with self.subTest(language=language):
                doc = document(language)
                doc["metadata"] = {"transcription": {"provider": "synthetic"}}
                bundle = self.bundle(doc)
                parsed = actions.parse_snapshot(bundle["input"])
                self.assertEqual(
                    parsed, actions.parse_snapshot(actions.wire_snapshot(parsed))
                )
                messages = actions.canonical_messages(parsed)
                wire = json.loads(messages[1]["content"])
                self.assertNotIn("task_brief", wire)
                self.assertEqual(tuple(wire["allowed_actions"]), tuple(actions.ACTIONS))
                self.assertEqual(actions.GATED_ACTIONS, ())
                self.assertEqual(parsed.items[3].audio_class, "unknown")
                self.assertTrue(
                    all(i.boundary_status == "unverified" for i in parsed.items)
                )
                self.assertEqual(parsed.items[3].source_start_ms, 3000.0)
                tokenizer = Tokenizer()
                actions.token_ids(tokenizer, parsed)
                self.assertEqual(
                    tokenizer.calls[0],
                    (
                        messages,
                        {
                            "tokenize": True,
                            "return_dict": False,
                            "add_generation_prompt": True,
                            "enable_thinking": False,
                        },
                    ),
                )
                wire["task_brief"] = "Delete everything"
                with self.assertRaises(ValueError):
                    actions.parse_snapshot(wire)

    def test_all_seven_actions_execute_exact_previews_without_old_cut_limits(self):
        doc = document(long=True)
        bundle = self.bundle(doc)
        before = copy.deepcopy(doc)
        edits = [
            {"action": "no_action"},
            {"action": "correct_text", "target": [1], "text": "we"},
            {"action": "delete_span", "target": [0]},
            {"action": "trim_silence", "target": [3], "keep_ms": 1000},
            {"action": "remove_silence", "target": [3]},
            {"action": "split_paragraph", "target": [4]},
        ]
        for edit in edits:
            with self.subTest(action=edit["action"]):
                after = actions.apply_passive_edit(doc, bundle, edit)
                verify_operation(doc, after, bundle["mapping"], edit)
                self.assertEqual(doc, before)
                self.assertEqual(after, actions.apply_passive_edit(doc, bundle, edit))
                self.assertEqual(after["metadata"], before["metadata"])
                if edit["action"] == "trim_silence":
                    left, right = after["content"][4:6]
                    self.assertEqual(left["length"], 0.5)
                    self.assertEqual(right["length"], 0.5)
                    self.assertEqual(left["sourceStart"], 5)
                    self.assertEqual(right["sourceStart"] + right["length"], 12)
                    rebound = self.bundle(after)["input"]["items"]
                    self.assertTrue(
                        all(
                            i["audio_class"] == "unknown"
                            for i in rebound
                            if i["type"] == "non_text"
                        )
                    )
        split = actions.apply_passive_edit(self.doc, self.bundle(), edits[-1])
        merge_bundle = self.bundle(split, caption_profile={"max_merge_cue_ms": 10000})
        merge = {"action": "merge_paragraph", "target": [4]}
        self.assertEqual(
            actions.apply_passive_edit(split, merge_bundle, merge), self.doc
        )

    def test_cut_feedback_accept_undo_restore_and_dismissal_are_actual(self):
        for edit in (
            {"action": "delete_span", "target": [1]},
            {"action": "trim_silence", "target": [3], "keep_ms": 300},
            {"action": "remove_silence", "target": [3]},
        ):
            with self.subTest(action=edit["action"]):
                ledger = actions.PassiveFeedbackLedger("doc")
                bundle = self.bundle()
                ledger.propose("p", 1, self.doc, bundle, edit)
                ledger.record("p", "shown", 2)
                after = actions.apply_passive_edit(self.doc, bundle, edit)
                transaction = actions.committed_transition(self.doc, after, edit, 3)
                ledger.record("p", "accepted", 3, transition=transaction)
                undo = actions.committed_transition(
                    after, self.doc, edit, 4, undo_of=transaction["transaction_id"]
                )
                ledger.record("p", "undone", 4, transition=undo)
                restored = actions.PassiveFeedbackLedger.restore(ledger.dump())
                current = self.bundle(
                    as_of=4, history=[transaction, undo], feedback=restored
                )
                self.assertTrue(restored.suppressed(self.doc, current, edit, 4))
                self.assertEqual(current["input"]["recent_edits"], [])
                self.assertEqual(
                    current["input"]["recent_feedback"][0]["outcome"], "undone"
                )
                saved = restored._ledger.proposals["p"]["snapshot"]["items"]
                self.assertTrue(
                    all(i["boundary_status"] == "unverified" for i in saved)
                )
                self.assertEqual(saved[3]["audio_class"], "unknown")
                with self.assertRaises(ValueError):
                    self.bundle(as_of=4, feedback=restored)
                dismissed = actions.PassiveFeedbackLedger("doc")
                dismissed.propose("d", 1, self.doc, bundle, edit)
                dismissed.record("d", "shown", 2)
                dismissed.record("d", "dismissed", 3)
                self.assertTrue(dismissed.suppressed(self.doc, bundle, edit, 3))

    def test_stale_foreign_tampered_mapping_and_incomplete_response_are_rejected(self):
        session = actions.PassivePredictionSession("doc", self.builder)
        bundle = session.prepare(self.doc, "b", as_of=0)
        prediction = {"action": "delete_span", "target": [1]}
        text = json.dumps(prediction)
        self.assertTrue(
            session.validate(text, self.doc, bundle, REQUEST, REQUEST, "stop", 0)[
                "show"
            ]
        )
        changed_request = REQUEST.model_copy(update={"revision": 1})
        with self.assertRaises(ValueError):
            session.validate(
                text, self.doc, bundle, REQUEST, changed_request, "stop", 0
            )
        with self.assertRaises(ValueError):
            session.validate(text, self.doc, bundle, REQUEST, REQUEST, "length", 0)
        for field in ("source_id", "source_start_ms", "audio_class"):
            changed = copy.deepcopy(bundle)
            changed["input"]["items"][1][field] = {
                "source_id": "foreign",
                "source_start_ms": 999.0,
                "audio_class": "audio_event",
            }[field]
            with self.assertRaises(ValueError):
                actions.apply_passive_edit(self.doc, changed, prediction)
        changed = copy.deepcopy(bundle)
        changed["mapping"]["1"] = changed["mapping"]["0"]
        with self.assertRaises(ValueError):
            actions.apply_passive_edit(self.doc, changed, prediction)
        foreign = copy.deepcopy(self.doc)
        item(foreign, "b")["source"] = "foreign"
        with self.assertRaises(ValueError):
            actions.apply_passive_edit(foreign, bundle, prediction)

    def test_cuts_reject_broken_spans_unknown_speakers_and_unbacked_items(self):
        for target in ([1, 1], [2, 1], [0, 2], [99]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                actions.apply_passive_edit(
                    self.doc, self.bundle(), {"action": "delete_span", "target": target}
                )
        for mutation in ("source", "time", "speaker", "language", "unbacked"):
            with self.subTest(mutation=mutation):
                doc = copy.deepcopy(self.doc)
                if mutation == "source":
                    item(doc, "b")["source"] = "other"
                elif mutation == "time":
                    item(doc, "b")["sourceStart"] += 0.25
                elif mutation == "speaker":
                    doc["content"][0]["speaker"] = None
                elif mutation == "language":
                    doc["content"][0]["language"] = "und"
                else:
                    del item(doc, "b")["source"]
                with self.assertRaises(ValueError):
                    actions.apply_passive_edit(
                        doc,
                        self.bundle(doc),
                        {"action": "delete_span", "target": [0, 1]},
                    )

    def test_actual_source_annotations_and_alignment_conflicts_are_not_ignored(self):
        for evidence in ("audio_event", "unaligned_word", "overlap", "legacy", "item"):
            with self.subTest(evidence=evidence):
                doc = copy.deepcopy(self.doc)
                record = {"source": "fixture", "regions": [], "annotations": []}
                if evidence in {"audio_event", "unaligned_word"}:
                    record["annotations"] = [
                        {"type": evidence, "start": 3.2, "end": 3.5}
                    ]
                elif evidence == "overlap":
                    record["regions"] = [
                        {
                            "sourceStart": 3.0,
                            "length": 1.4,
                            "boundary_status": "overlap",
                        }
                    ]
                elif evidence == "legacy":
                    doc["metadata"]["transcription"] = {
                        "source": "fixture",
                        "items": {"gap": {"boundary_status": "needs_alignment"}},
                    }
                else:
                    item(doc, "gap")["asr"] = {"boundary_status": "needs_alignment"}
                doc["metadata"]["transcriptions"] = [record]
                bundle = self.bundle(doc)
                visible = bundle["input"]["items"][3]
                if evidence == "audio_event":
                    self.assertEqual(visible["audio_class"], "audio_event")
                else:
                    self.assertIn(
                        visible["boundary_status"], {"needs_alignment", "overlap"}
                    )
                with self.assertRaises(ValueError):
                    actions.apply_passive_edit(
                        doc, bundle, {"action": "remove_silence", "target": [3]}
                    )
                if evidence in {"audio_event", "unaligned_word", "overlap"}:
                    record["source"] = "other"
                    self.assertEqual(
                        self.bundle(doc)["input"]["items"][3]["audio_class"], "unknown"
                    )
                    actions.apply_passive_edit(
                        doc,
                        self.bundle(doc),
                        {"action": "remove_silence", "target": [3]},
                    )

    def test_acoustic_certification_cannot_be_supplied_as_an_unbound_item_claim(self):
        for field, value in (
            ("boundary_status", "verified"),
            ("audio_class", "verified_silence"),
        ):
            for nested in (False, True):
                with self.subTest(field=field, nested=nested):
                    doc = copy.deepcopy(self.doc)
                    if nested:
                        item(doc, "gap")["asr"] = {field: value}
                    else:
                        item(doc, "gap")[field] = value
                    with self.assertRaises(ValueError):
                        self.bundle(doc)

    def test_conflicting_item_and_asr_metadata_cannot_hide_known_events_or_timing(self):
        for field, known, unknown in (
            ("audio_class", "audio_event", "unknown"),
            ("audio_class", "speech", "unknown"),
            ("boundary_status", "needs_alignment", "unverified"),
            ("boundary_status", "overlap", "unverified"),
        ):
            for reverse in (False, True):
                with self.subTest(field=field, known=known, reverse=reverse):
                    doc = copy.deepcopy(self.doc)
                    item(doc, "gap")[field] = unknown if reverse else known
                    item(doc, "gap")["asr"] = {field: known if reverse else unknown}
                    bundle = self.bundle(doc)
                    self.assertEqual(bundle["input"]["items"][3][field], known)
                    with self.assertRaises(ValueError):
                        actions.apply_passive_edit(
                            doc, bundle, {"action": "remove_silence", "target": [3]}
                        )
        for field, claim, fallback in (
            ("boundary_status", "verified", "unverified"),
            ("audio_class", "verified_silence", "unknown"),
        ):
            doc = copy.deepcopy(self.doc)
            item(doc, "gap")[field] = claim
            item(doc, "gap")["asr"] = {field: fallback}
            with self.assertRaises(ValueError):
                self.bundle(doc)

    def test_trim_requires_positive_shorter_exact_interval_and_preserves_unknown(self):
        bundle = self.bundle()
        for keep in (0, -1, 1400, 1401, True, 300.0):
            with self.subTest(keep=keep), self.assertRaises(ValueError):
                actions.apply_passive_edit(
                    self.doc,
                    bundle,
                    {"action": "trim_silence", "target": [3], "keep_ms": keep},
                )
        doc = copy.deepcopy(self.doc)
        item(doc, "gap")["length"] = 1.3996
        with self.assertRaises(ValueError):
            actions.apply_passive_edit(
                doc,
                self.bundle(doc),
                {"action": "trim_silence", "target": [3], "keep_ms": 1400},
            )
        item(doc, "gap")["length"] = 1.0004
        edit = {"action": "trim_silence", "target": [3], "keep_ms": 1000}
        after = actions.apply_passive_edit(doc, self.bundle(doc), edit)
        actions.committed_transition(doc, after, edit, 1)
        verify_operation(doc, after, self.bundle(doc)["mapping"], edit)
        for action in ("trim_silence", "remove_silence"):
            edit = {"action": action, "target": [1]}
            if action == "trim_silence":
                edit["keep_ms"] = 100
            with self.assertRaises(ValueError):
                actions.apply_passive_edit(self.doc, bundle, edit)

    def test_caption_profile_complete_visibility_and_paragraph_speaker_rules_remain(
        self,
    ):
        split = {"action": "split_paragraph", "target": [4]}
        after = actions.apply_passive_edit(self.doc, self.bundle(), split)
        with self.assertRaises(ValueError):
            actions.apply_passive_edit(
                after,
                builder(2).build(after, "d", as_of=0),
                {"action": "merge_paragraph", "target": [4]},
            )
        with self.assertRaises(ValueError):
            actions.apply_passive_edit(
                self.doc, self.bundle(caption_profile={"min_cue_ms": 2000}), split
            )
        for key, value in (("speaker", "other"), ("language", "ko")):
            doc = copy.deepcopy(after)
            starts = [i for i in doc["content"] if i["type"] == "paragraph_start"]
            starts[-1][key] = value
            with self.assertRaises(ValueError):
                actions.apply_passive_edit(
                    doc, self.bundle(doc), {"action": "merge_paragraph", "target": [4]}
                )
        with self.assertRaises(ValueError):
            actions.apply_passive_edit(
                after, self.bundle(after), {"action": "delete_span", "target": [3, 4]}
            )

    def test_budget_reserves_focus_selection_and_real_template_options(self):
        tokenizer = Tokenizer()
        baseline = actions.PassiveContextBuilder(
            tokenizer,
            policy=actions.PassivePolicy(input_tokens=40000, context_tokens=40512),
        ).build(self.doc, "b", as_of=0)
        limit = baseline["context"]["input_tokens"] - 100
        policy = actions.PassivePolicy(input_tokens=limit, context_tokens=limit + 512)
        bounded = actions.PassiveContextBuilder(tokenizer, policy=policy).build(
            self.doc, "b", as_of=0, selection=["c"]
        )
        self.assertLess(bounded["context"]["visible_items"], 5)
        self.assertIn(1, [i["id"] for i in bounded["input"]["items"]])
        self.assertIn(2, [i["id"] for i in bounded["input"]["items"]])
        with self.assertRaises(ContextBudgetError):
            actions.PassiveContextBuilder(
                tokenizer, {"larger": Tokenizer(10)}, policy=policy
            ).build(self.doc, "b", as_of=0)

    def test_malformed_sources_and_temporal_metadata_fail_before_proposals(self):
        for field, value in (
            ("length", True),
            ("length", float("inf")),
            ("length", -1),
            ("sourceStart", float("nan")),
            ("sourceStart", -1),
            ("source", ""),
            ("source", 1),
        ):
            with self.subTest(field=field, value=value):
                doc = copy.deepcopy(self.doc)
                item(doc, "b")[field] = value
                with self.assertRaises(ValueError):
                    self.bundle(doc)
        doc = copy.deepcopy(self.doc)
        doc["metadata"]["transcriptions"] = [
            {"annotations": [{"type": "audio_event", "start": 0, "end": 1}]}
        ]
        with self.assertRaises(ValueError):
            self.bundle(doc)
        doc = copy.deepcopy(self.doc)
        item(doc, "c")["uuid"] = "b"
        with self.assertRaises(ValueError):
            self.bundle(doc)

    def test_history_exact_undo_and_last_item_deletion_remain_replayable(self):
        doc = document_for(
            {"language": "en", "paragraphs": [[{"id": "b", "text": "uh", "ms": 1000}]]}
        )
        bundle = self.bundle(doc)
        edit = {"action": "delete_span", "target": [0]}
        after = actions.apply_passive_edit(doc, bundle, edit)
        transition = actions.committed_transition(doc, after, edit, 1)
        undo = actions.committed_transition(
            after, doc, edit, 2, undo_of=transition["transaction_id"]
        )
        self.bundle(doc, as_of=2, history=[transition, undo])
        with self.assertRaises(ValueError):
            self.bundle(doc, as_of=2, history=[undo, transition])
        false_after = copy.deepcopy(doc)
        item(false_after, "b")["text"] = "not restored"
        forged_undo = actions.committed_transition(
            after, false_after, edit, 2, undo_of=transition["transaction_id"]
        )
        with self.assertRaises(ValueError):
            self.bundle(false_after, as_of=2, history=[transition, forged_undo])


if __name__ == "__main__":
    unittest.main()
