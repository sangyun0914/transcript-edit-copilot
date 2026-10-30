import copy
import json
import unittest
from unittest.mock import patch

import passive_actions as actions
from context_policy import ContextBudgetError, correction_effect
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

    def test_product_ledger_receipts_match_legacy_acceptance_undo_and_context(self):
        edit = {"action": "delete_span", "target": [1]}
        bundle = self.bundle()
        after = actions.apply_passive_edit(self.doc, bundle, edit)
        accepted = actions.committed_transition(self.doc, after, edit, 3)
        undone = actions.committed_transition(
            after, self.doc, edit, 4, undo_of=accepted["transaction_id"]
        )
        legacy = actions.PassiveFeedbackLedger("doc")
        compact = actions.CompactFeedbackLedger("doc")
        receipts = []
        for ledger in (legacy, compact):
            ledger.propose("p", 1, self.doc, bundle, edit)
            if ledger is compact:
                receipts.append(ledger._records[-1])
            for outcome, sequence, transition in (
                ("shown", 2, None),
                ("accepted", 3, accepted),
                ("undone", 4, undone),
            ):
                if ledger is compact:
                    original = ledger.dump()
                    candidate = ledger.fork()
                    candidate.record("p", outcome, sequence, transition=transition)
                    self.assertEqual(ledger.dump(), original)
                    receipts.append(candidate._records[-1])
                    compact = ledger = candidate
                else:
                    ledger.record("p", outcome, sequence, transition=transition)
        restored = actions.CompactFeedbackLedger("doc")
        for receipt in receipts:
            restored.restore_record(receipt)
        self.assertEqual(restored.dump(), compact.dump())
        self.assertNotIn('"content":', json.dumps(compact.dump()))
        self.assertEqual(len(compact._records), 1)
        for ledger in (
            compact,
            restored,
            actions.CompactFeedbackLedger.from_ledger(legacy),
        ):
            actual = self.bundle(as_of=4, history=[accepted, undone], feedback=ledger)
            expected = self.bundle(as_of=4, history=[accepted, undone], feedback=legacy)
            self.assertEqual(actual, expected)
            self.assertTrue(ledger.suppressed(self.doc, actual, edit, 4))
            with self.assertRaises(ValueError):
                self.bundle(as_of=4, history=[], feedback=ledger)
        self.assertIn("document", legacy.dump()["records"][0])
        self.assertEqual(
            actions.PassiveFeedbackLedger.restore(legacy.dump()).dump(), legacy.dump()
        )

    def test_product_suppression_keeps_equivalent_corrections_without_repeating_execution(
        self,
    ):
        legacy = actions.PassiveFeedbackLedger("doc")
        bundle = self.bundle()
        for index in range(12):
            edit = {
                "action": "correct_text",
                "target": [1],
                "text": f"proposal {index}",
            }
            sequence = index * 3 + 1
            legacy.propose(str(index), sequence, self.doc, bundle, edit)
            legacy.record(str(index), "shown", sequence + 1)
            legacy.record(str(index), "dismissed", sequence + 2)
        compact = actions.CompactFeedbackLedger.from_ledger(legacy)
        values = [
            {"action": "correct_text", "target": [1], "text": "another correction"},
            {
                "action": "correct_text",
                "target": [1, 2],
                "text": "proposal 11 can begin.",
            },
        ]
        for prediction in values:
            with patch.object(
                actions, "_correction_effect", wraps=actions._correction_effect
            ) as effect:
                self.assertEqual(
                    compact.suppressed(self.doc, bundle, prediction, 40),
                    legacy.suppressed(self.doc, bundle, prediction, 40),
                )
                self.assertEqual(effect.call_count, 1)
        changed = copy.deepcopy(self.doc)
        item(changed, "a")["text"] = "Changed neighboring evidence"
        changed_bundle = self.bundle(changed)
        self.assertFalse(compact.suppressed(changed, changed_bundle, values[-1], 40))

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

    def test_session_reuses_validation_without_trusting_mutated_inputs(self):
        session = actions.PassivePredictionSession("doc", self.builder)
        self.doc["metadata"]["transcriptions"] = [{"source": "fixture", "regions": []}]
        with patch.object(
            actions, "document_records", wraps=actions.document_records
        ) as records:
            original = session.prepare(self.doc, "b", as_of=0)
            bundle = session.prepare(copy.deepcopy(self.doc), "b", as_of=0)
            self.assertEqual(bundle, original)
            self.assertEqual(records.call_count, 1)
            prediction = {"action": "delete_span", "target": [1]}
            with session.validation_scope():
                session.validate(
                    json.dumps(prediction),
                    self.doc,
                    bundle,
                    REQUEST,
                    REQUEST,
                    "stop",
                    0,
                )
                actions.apply_passive_edit(self.doc, bundle, prediction)
                session.feedback.propose("cached", 0, self.doc, bundle, prediction)
            self.assertEqual(records.call_count, 1)
            bundle["mapping"]["1"]["uuid"] = "forged"
            self.assertEqual(session.prepare(self.doc, "b", as_of=0), original)

        item(self.doc, "b")["length"] = True
        with self.assertRaises(ValueError):
            session.prepare(self.doc, "b", as_of=0)
        item(self.doc, "b")["length"] = 1.0
        self.doc["metadata"]["transcriptions"][0]["regions"] = ()
        with self.assertRaises(ValueError):
            session.prepare(self.doc, "b", as_of=0)
        self.doc["metadata"]["transcriptions"][0]["regions"] = []
        item(self.doc, "b")["source"] = "different-source"
        with self.assertRaises(ValueError):
            session.validate(
                json.dumps(prediction), self.doc, original, REQUEST, REQUEST, "stop", 0
            )

    def test_cached_history_validates_new_deltas_and_rechecks_changed_old_deltas(self):
        edit = {"action": "correct_text", "target": [1], "text": "we"}
        after = actions.apply_passive_edit(self.doc, self.bundle(), edit)
        first = actions.committed_transition(self.doc, after, edit, 1)
        second_edit = {"action": "delete_span", "target": [0]}
        second_bundle = self.bundle(after)
        final = actions.apply_passive_edit(after, second_bundle, second_edit)
        second = actions.committed_transition(after, final, second_edit, 2)
        session = actions.PassivePredictionSession("doc", self.builder)
        with patch(
            "verify_release_export.verify_operation", wraps=verify_operation
        ) as verify:
            session.prepare(after, "b", as_of=1, history=[first])
            session.prepare(
                copy.deepcopy(after), "b", as_of=1, history=copy.deepcopy([first])
            )
            self.assertEqual(verify.call_count, 1)
            actual = session.prepare(final, "b", as_of=2, history=[first, second])
            self.assertEqual(verify.call_count, 2)
        self.assertEqual(actual, self.bundle(final, as_of=2, history=[first, second]))
        item(first["before"], "b")["length"] = True
        with self.assertRaises(ValueError):
            session.prepare(final, "b", as_of=2, history=[first, second])

        empty = {"content": [], "metadata": {}}
        with session.validation_scope():
            actions._records(empty, allow_empty=True)
            with self.assertRaises(ValueError):
                actions._records(empty)

    def test_correction_effect_uses_the_same_paragraph_and_media_equivalence(self):
        for language in ("en", "ko"):
            doc = document(language)
            split = {"action": "split_paragraph", "target": [4]}
            doc = actions.apply_passive_edit(doc, self.bundle(doc), split)
            bundle = self.bundle(doc)
            for target in ([0, 1], [4]):
                prediction = {
                    "action": "correct_text",
                    "target": target,
                    "text": "수정 🙂",
                }
                with self.subTest(language=language, target=target):
                    self.assertEqual(
                        actions._correction_effect(doc, bundle, prediction),
                        correction_effect(
                            doc, actions._legacy_bundle(doc, bundle), prediction
                        ),
                    )

    def test_incremental_history_matches_full_context_and_owns_acknowledged_states(
        self,
    ):
        correction = {"action": "correct_text", "target": [1], "text": "we"}
        after = actions.apply_passive_edit(self.doc, self.bundle(), correction)
        first = actions.committed_transition(self.doc, after, correction, 1)
        deletion = {"action": "delete_span", "target": [0]}
        deleted = actions.apply_passive_edit(after, self.bundle(after), deletion)
        second = actions.committed_transition(after, deleted, deletion, 2)
        undo = actions.committed_transition(
            deleted, after, deletion, 3, undo_of=second["transaction_id"]
        )
        session = actions.PassivePredictionSession("doc", self.builder)
        base = None
        history = []
        with patch(
            "verify_release_export.verify_operation", wraps=verify_operation
        ) as verify:
            for transition in (first, second, undo):
                history.append(transition)
                bundle, candidate = session.prepare_incremental(
                    transition["after"],
                    "b",
                    as_of=transition["sequence"],
                    history=[transition],
                    history_base=base,
                )
                self.assertEqual(candidate.tip["count"], len(history))
                session.commit_history(candidate)
                base = candidate.tip["transaction_id"]
                expected = self.bundle(
                    transition["after"], as_of=transition["sequence"], history=history
                )
                self.assertEqual(bundle, expected)
            self.assertEqual(verify.call_count, 9)
            repeated, candidate = session.prepare_incremental(
                after, "b", as_of=3, history_base=base
            )
            self.assertEqual(repeated, bundle)
            self.assertEqual(verify.call_count, 9)
            session.commit_history(candidate)
            reordered = json.loads(json.dumps(history, sort_keys=True))
            synced, candidate = session.prepare_incremental(
                after, "b", as_of=3, history=reordered
            )
            self.assertEqual(synced, bundle)
            self.assertEqual(verify.call_count, 9)
            session.commit_history(candidate)
        owned = session._history_state.entries
        self.assertIs(owned[1].transition.before, owned[0].transition.after)
        self.assertIs(
            owned[1].transition.after["metadata"], owned[0].transition.after["metadata"]
        )
        self.assertIs(
            owned[1].transition.after["content"][0],
            owned[0].transition.after["content"][0],
        )
        item(first["before"], "b")["length"] = True
        self.assertEqual(
            session.prepare_incremental(after, "b", as_of=3, history_base=base)[0],
            bundle,
        )
        with self.assertRaises(ValueError):
            session.prepare_incremental(after, "b", as_of=3, history=history)

    def test_incremental_history_resync_cancellation_stale_document_and_future_guards(
        self,
    ):
        edit = {"action": "correct_text", "target": [1], "text": "we"}
        after = actions.apply_passive_edit(self.doc, self.bundle(), edit)
        transition = actions.committed_transition(self.doc, after, edit, 1)
        session = actions.PassivePredictionSession("doc", self.builder)
        with self.assertRaises(actions.HistoryResyncRequired):
            session.prepare_incremental(after, "b", as_of=1, history_base="unknown")
        with self.assertRaises(ValueError):
            session.prepare_incremental(after, "b", as_of=0, history=[transition])
        _, abandoned = session.prepare_incremental(
            after, "b", as_of=1, history=[transition]
        )
        self.assertEqual(session._history_state.tip["count"], 0)
        _, accepted = session.prepare_incremental(
            after, "b", as_of=1, history=[transition]
        )
        session.commit_history(accepted)
        with self.assertRaises(actions.HistoryResyncRequired):
            session.commit_history(abandoned)
        changed = copy.deepcopy(after)
        changed["metadata"]["changed"] = True
        with self.assertRaises(ValueError):
            session.prepare_incremental(
                changed, "b", as_of=1, history_base=transition["transaction_id"]
            )
        next_edit = {"action": "correct_text", "target": [1], "text": "we can"}
        next_document = actions.apply_passive_edit(after, self.bundle(after), next_edit)
        next_transition = actions.committed_transition(
            after, next_document, next_edit, 2
        )
        with (
            patch.object(actions._SessionHistory, "limit", 1),
            self.assertRaises(actions.HistoryLimitError),
        ):
            session.prepare_incremental(
                next_document,
                "b",
                as_of=2,
                history=[next_transition],
                history_base=transition["transaction_id"],
            )

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

    def test_temporal_lookup_preserves_reordered_overlapping_source_evidence(self):
        doc = copy.deepcopy(self.doc)
        for uuid, start in (("a", 5), ("b", 3), ("c", 2), ("gap", 10), ("d", 5)):
            item(doc, uuid)["sourceStart"] = start
            item(doc, uuid)["length"] = 1
        item(doc, "d")["source"] = "other"
        doc["metadata"]["transcriptions"] = [
            {
                "source": "fixture",
                "regions": [
                    {"sourceStart": 8, "length": 1, "boundary_status": "overlap"},
                    {
                        "sourceStart": 2,
                        "length": 1,
                        "boundary_status": "needs_alignment",
                    },
                ],
                "annotations": [
                    {"type": "audio_event", "start": 4, "end": 4.5},
                    {"type": "audio_event", "start": 5, "end": 5},
                    {"type": "unaligned_word", "start": 6, "end": 6},
                ],
            },
            {
                "source": "other",
                "regions": [
                    {
                        "sourceStart": 6,
                        "length": 1,
                        "boundary_status": "needs_alignment",
                    }
                ],
                "annotations": [{"type": "audio_event", "start": 6, "end": 6}],
            },
            {
                "source": "fixture",
                "regions": [
                    {"sourceStart": 0, "length": 10, "boundary_status": "overlap"},
                    {
                        "sourceStart": 4,
                        "length": 1,
                        "boundary_status": "needs_alignment",
                    },
                ],
            },
        ]
        before = copy.deepcopy(doc)

        class LinearIntervals:
            def __init__(self, intervals):
                self.intervals = intervals

            def overlaps(self, start, end):
                return any(
                    actions._overlaps(start, end, a, b) for a, b in self.intervals
                )

        with patch.object(actions, "_IntervalIndex", LinearIntervals):
            expected = actions._records(doc)
            expected_messages = actions.canonical_messages(self.bundle(doc)["input"])
        records, mapping = actions._records(doc)
        self.assertEqual((records, mapping), expected)
        self.assertEqual(
            actions.canonical_messages(self.bundle(doc)["input"]), expected_messages
        )
        self.assertEqual(
            [(record["boundary_status"], record["audio_class"]) for record in records],
            [
                ("needs_alignment", "audio_event"),
                ("overlap", "speech"),
                ("needs_alignment", "speech"),
                ("unverified", "unknown"),
                ("unverified", "audio_event"),
            ],
        )
        self.assertEqual(doc, before)

    def test_long_source_does_not_scan_every_region_for_every_item(self):
        count = 2000
        doc = document_for(
            {
                "language": "en",
                "paragraphs": [
                    [
                        {"id": f"word-{n}", "text": "word", "ms": 1000}
                        for n in range(count)
                    ]
                ],
            }
        )
        doc["metadata"]["transcriptions"] = [
            {
                "source": "fixture",
                "regions": [
                    {
                        "sourceStart": n,
                        "length": 0.5,
                        "boundary_status": "overlap" if n % 4 == 0 else "unverified",
                    }
                    for n in range(0, count, 2)
                ],
            }
        ]
        overlaps = actions._overlaps
        comparisons = 0

        def bounded_overlap(*args):
            nonlocal comparisons
            comparisons += 1
            self.assertLess(
                comparisons, count * 8, "Repeated whole-source metadata scan"
            )
            return overlaps(*args)

        with patch.object(actions, "_overlaps", bounded_overlap):
            records, mapping = actions._records(doc)
        self.assertEqual(len(mapping), count)
        self.assertEqual(
            [record["boundary_status"] for record in records],
            ["overlap" if n % 4 == 0 else "unverified" for n in range(count)],
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

    def test_shrinking_caption_window_matches_full_document_metrics(self):
        doc = document_for(
            {
                "language": "ko",
                "paragraphs": [
                    [
                        {"id": f"a{i}", "text": text, "ms": 333}
                        for i, text in enumerate(("앞", "🙂", "", "가 나", "é", "끝"))
                    ],
                    [
                        {"id": f"b{i}", "text": text, "ms": 701}
                        for i, text in enumerate(("다음", "", "자막", "마지막"))
                    ],
                ],
            }
        )
        # Empty text items still contribute a separator to UTF-16 text metrics.
        item(doc, "a2").update(type="text", text="")
        records, _ = actions._records(doc)
        for line_length in (None, 3, 20):
            profile = actions.CaptionProfile(
                export_line_length=line_length
            ).model_dump()
            initial = records[1:-1]
            window = actions._CaptionWindow(doc, records, initial, profile)
            for start in range(len(initial)):
                for end in range(start + 1, len(initial) + 1):
                    visible = initial[start:end]
                    with self.subTest(line_length=line_length, start=start, end=end):
                        self.assertEqual(
                            window.select(visible),
                            actions.caption_facts(doc, records, visible, [], profile),
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
        for evidence in (
            {
                "regions": [
                    {"sourceStart": 100, "length": 0, "boundary_status": "overlap"}
                ]
            },
            {"annotations": [{"type": "audio_event", "start": 100, "end": 99}]},
        ):
            with self.subTest(unused_source_evidence=evidence):
                doc = copy.deepcopy(self.doc)
                doc["metadata"]["transcriptions"] = [{"source": "unused", **evidence}]
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
