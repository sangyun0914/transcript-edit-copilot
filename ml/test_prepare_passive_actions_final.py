import copy
import unittest

import passive_actions as actions
import passive_contract as old
import prepare_passive_actions_final as preparation


class Tokens:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


def document():
    return {"content": [
        {"type": "paragraph_start", "uuid": "p", "speaker": "narrator", "language": "en"},
        {"type": "text", "uuid": "a", "text": "Check the ligt.", "source": "symbolic:delete-reference-en-train", "sourceStart": 0., "length": 2.},
        {"type": "non_text", "uuid": "g", "source": "symbolic:delete-reference-en-train", "sourceStart": 2., "length": 1.2},
        {"type": "text", "uuid": "b", "text": "Then stop.", "source": "symbolic:delete-reference-en-train", "sourceStart": 3.2, "length": 2.},
        {"type": "paragraph_end", "uuid": "e"}], "metadata": {"transcriptions": [{
            "source": "symbolic:delete-reference-en-train", "regions": [], "annotations": []}]}}


class PreparationTests(unittest.TestCase):
    def test_source_projection_is_bijective_and_preserves_timing_text_and_metadata(self):
        original = document()
        mapping = preparation.source_map([original])
        projected = preparation.project_document(original, mapping)
        self.assertNotIn("delete-reference", preparation.packed(projected))
        self.assertEqual(projected["content"][1]["text"], original["content"][1]["text"])
        self.assertEqual(projected["content"][3]["sourceStart"], 3.2)
        self.assertEqual(projected["metadata"]["transcriptions"][0]["source"], projected["content"][1]["source"])
        self.assertEqual(preparation.project(projected, {v: k for k, v in mapping.items()}), original)

    def test_structural_cut_audit_matches_actual_validator_and_does_not_certify_usefulness(self):
        request = actions.Envelope(document_id="d", request_id="r", revision=0, interaction_epoch=0)
        variants = ({}, {"audio_class": "audio_event"}, {"audio_class": "speech"},
                    {"boundary_status": "needs_alignment"}, {"boundary_status": "overlap"})
        for variant in variants:
            doc = document()
            doc["content"][2].update(variant)
            for speaker in ("narrator", None):
                doc["content"][0]["speaker"] = speaker
                bundle = actions.PassiveContextBuilder(Tokens()).build(doc, "a", as_of=0)
                audited = preparation.cut_evidence(bundle["input"])
                for item in audited:
                    prediction = {"action": item["action"], "target": [item["alias"]]}
                    try:
                        actions.validate_prediction(preparation.packed(prediction), bundle["input"], request, request, "stop")
                        actual = True
                    except ValueError:
                        actual = False
                    self.assertEqual(item["structurally_eligible"], actual)

    def test_feedback_migration_keeps_actual_dismissal_and_no_future_record(self):
        original = document()
        builder = old.PassiveContextBuilder(Tokens())
        bundle = builder.build(original, "a", as_of=0)
        ledger = old.PassiveFeedbackLedger("episode")
        ledger.propose("one", 0, original, bundle, {"action": "correct_text", "target": [0], "text": "Check the light."})
        ledger.record("one", "shown", 1, origin="synthetic")
        ledger.record("one", "dismissed", 2, origin="synthetic")
        mapping = preparation.source_map([original])
        projected = preparation.project_document(original, mapping)
        migrated = preparation.migrate_feedback(ledger.dump(), mapping, 2)
        actual = actions.PassiveContextBuilder(Tokens()).build(projected, "a", as_of=2, feedback=migrated)
        expected = builder.build(original, "a", as_of=2, feedback=ledger)
        self.assertEqual(actual["input"]["recent_feedback"], expected["input"]["recent_feedback"])
        self.assertEqual(len(preparation.migrate_feedback(ledger.dump(), mapping, 0).dump()["records"]), 1)
        self.assertTrue(migrated.suppressed(projected, actual, {"action": "correct_text", "target": [0], "text": "Check the light."}, 2))

    def test_projected_split_acceptance_and_undo_recompute_only_derived_marker_ids(self):
        original = document()
        bundle = old.PassiveContextBuilder(Tokens()).build(original, "b", as_of=0)
        prediction = {"action": "split_paragraph", "target": [2]}
        after = old.apply_passive_edit(original, bundle, prediction)
        commit = old.committed_transition(original, after, prediction, 2)
        undo = old.committed_transition(after, original, prediction, 3, undo_of=commit["transaction_id"])
        ledger = old.PassiveFeedbackLedger("episode")
        ledger.propose("one", 0, original, bundle, prediction)
        ledger.record("one", "shown", 1, origin="synthetic")
        ledger.record("one", "accepted", 2, transition=commit)
        ledger.record("one", "undone", 3, transition=undo)
        mapping = preparation.source_map([original, after])
        generated = preparation.projected_history_uuids([commit, undo], mapping)
        self.assertEqual(len(generated), 2)
        self.assertTrue(all(old_id != new_id for old_id, new_id in generated.items()))
        projected = preparation.project_document(original, mapping, generated)
        projected_after = preparation.project_document(after, mapping, generated)
        upgraded = preparation.upgrade_bundle(bundle, projected)
        self.assertEqual(actions.apply_passive_edit(projected, upgraded, prediction), projected_after)
        history = preparation.project([commit, undo], mapping, generated)
        self.assertEqual(history[0]["transaction_id"], commit["transaction_id"])
        self.assertEqual(history[1]["before"], projected_after)
        self.assertEqual(history[1]["after"], projected)
        migrated = preparation.migrate_feedback(ledger.dump(), mapping, 3, generated)
        current = actions.PassiveContextBuilder(Tokens()).build(projected, "b", as_of=3, history=history, feedback=migrated)
        old_current = old.PassiveContextBuilder(Tokens()).build(original, "b", as_of=3, history=[commit, undo], feedback=ledger)
        self.assertEqual(current["input"]["recent_edits"], old_current["input"]["recent_edits"])
        self.assertEqual(current["input"]["recent_feedback"], old_current["input"]["recent_feedback"])

    def test_ambiguous_preserve_is_not_relabelled_even_if_cropping_removes_a_cut(self):
        doc = document()
        snapshot = actions.PassiveContextBuilder(Tokens()).build(doc, "a", as_of=0)["input"]
        case = {"id": "x", "expectedOutput": {"action": "no_action"}, "metadata": {"source_group_id": "arbitrary"}, "input": snapshot}
        review = {"case_id": "x", "review": "scope_ambiguous", "reason": "Actual conversational restarts were reviewed before predictions."}
        result = preparation.applicability(case, snapshot, [{"available": True}], [], review)
        self.assertEqual(result["status"], "legacy_no_action_cut_scope_ambiguous")
        self.assertFalse(result["full_action_preservation_admitted"])
        self.assertFalse(result["reference_altered"])
        with self.assertRaises(ValueError):
            preparation.applicability(case, snapshot, [{"available": True}], [], None)

    def test_new_preserve_window_cannot_silently_inherit_original_semantic_review(self):
        doc = document()
        original = old.PassiveContextBuilder(Tokens()).build(doc, "a", as_of=0)["input"]
        snapshot = actions.PassiveContextBuilder(Tokens()).build(doc, "a", as_of=0)["input"]
        case = {"id": "x", "expectedOutput": {"action": "no_action"}, "metadata": {}, "input": copy.deepcopy(original)}
        review = {"case_id": "x", "review": "preservation_applicable_if_current_window_unchanged"}
        snapshot["items"] = snapshot["items"][:1]
        with self.assertRaisesRegex(ValueError, "window changed"):
            preparation.applicability(case, snapshot, [{"available": True}], [{"structurally_eligible": True}], review)


if __name__ == "__main__":
    unittest.main()
