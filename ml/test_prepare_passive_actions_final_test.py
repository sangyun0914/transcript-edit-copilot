import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import prepare_passive_actions_final_test as final
import passive_actions_final_sources as sources


class FinalTestPreparationGuards(unittest.TestCase):
    def binding(self):
        protocol = {"models": {"native_base": {"sha256": "a" * 64},
                               "final11": {"sha256": "b" * 64}},
                    "source_plan": {"natural": {"kind": "natural_recording"},
                                    "authored": {"kind": "authored_symbolic", "parent_uri": "authored://unit"}}}
        binding = {
            "status": "frozen_for_sealed_generation", "protocol_sha256": final.digest(protocol),
            "models": copy.deepcopy(protocol["models"]), "parent_authorization": "unit-only",
            "test_bodies_generated_before_freeze": 0, "test_bodies_inspected_before_freeze": 0,
            "author_actor": "author-unit", "reviewer_actor": "reviewer-unit",
            "sources": {"natural": {"status": "source_unavailable", "reason_code": "fixture_only"},
                        "authored": {"status": "generation_authorized", "parent_uri": "authored://unit",
                                     "generator_identity": "unit", "seed": "unit"}},
            "effective_policy": {
                "contract_version": final.contract.VERSION, "allowed_actions": list(final.contract.ALLOWED_ACTIONS),
                "gated_actions": [], "input_tokens": 3584, "max_tokens": 512, "context_tokens": 4096,
                "history_limit": 3, "feedback_limit": 3, "temperature": 0, "seed": 42,
                "native_stop_ids": [1, 106, 50], "tool_handoff_credited": False, "schema_constrained": True,
                "cache_prompt": False, "cache_reset_rule": "reset_each_request", "timeout_seconds": 120,
                "retry_policy": "none", "icl": False, "truncation": False,
            },
        }
        return protocol, binding

    def test_pending_runtime_wrong_checkpoint_and_cut_gate_reject(self):
        protocol, binding = self.binding()
        final.validate_freeze(protocol, binding)
        mutations = [lambda b: b.update(status="pending_L2"),
                     lambda b: b["models"]["final11"].update(sha256="c" * 64),
                     lambda b: b["effective_policy"].update(gated_actions=["delete_span"])]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                bad = copy.deepcopy(binding)
                mutate(bad)
                with self.assertRaises(ValueError):
                    final.validate_freeze(protocol, bad)

    def test_natural_windows_need_original_metadata_not_target_selection(self):
        protocol, binding = self.binding()
        natural = {"status": "available_uninspected", "uri": "https://example.invalid/unit.mp4",
                   "duration_seconds": 600, "media_sha256": "c" * 64, "captions_sha256": "d" * 64,
                   "windows": final.recording_windows(600), "selected_before_content_inspection": True}
        binding["sources"]["natural"] = natural
        final.validate_freeze(protocol, binding)
        self.assertEqual(natural["windows"][0], {"start": 75.0, "end": 165.0})
        natural["windows"][0]["start"] += 1
        with self.assertRaisesRegex(ValueError, "natural_metadata"):
            final.validate_freeze(protocol, binding)
        with self.assertRaisesRegex(ValueError, "too_short"):
            final.recording_windows(449)

    def test_focus_survives_then_uses_source_time_neighbor(self):
        document = {"content": [{"type": "text", "uuid": str(n), "source": "same", "sourceStart": n}
                                for n in range(3)]}
        self.assertEqual(final.scheduled_focus(document), "1")
        previous = {"document": document, "focus_uuid": "1"}
        current = copy.deepcopy(document)
        current["content"].pop(1)
        self.assertEqual(final.scheduled_focus(current, previous), "0")

    def test_initial_checkpoint_rejects_future_or_target_shift(self):
        runtime = {"split": "test", "document_id": "unit-document", "selection": [], "document": {"content": [
            {"type": "text", "uuid": "a"}, {"type": "text", "uuid": "b"},
            {"type": "text", "uuid": "c"}]}, "focus_uuid": "b", "history": [],
                   "sequence": 0, "feedback_journal": None}
        final.validate_checkpoint(runtime, 0, None)
        runtime["focus_uuid"] = "a"
        with self.assertRaisesRegex(ValueError, "focus_changed"):
            final.validate_checkpoint(runtime, 0, None)
        runtime["focus_uuid"] = "b"
        runtime["history"] = [{"sequence": 1}]
        with self.assertRaisesRegex(ValueError, "future_history"):
            final.validate_checkpoint(runtime, 0, None)

    def test_unknown_reference_never_becomes_no_action(self):
        _, binding = self.binding()
        reference = {"id": "unit", "label_state": "ambiguous", "reason_code": "unit_only",
                     "accepted_predictions": [{"action": "no_action"}],
                     "audit": {"author_actor": "author-unit", "reviewer_actor": "reviewer-unit",
                               "input_first": True, "predictions_seen": False,
                               "all_visible_opportunities_reviewed": True, "novelty_rationale": "unit"}}
        with self.assertRaisesRegex(ValueError, "unknown_reference"):
            final.seal_reference_records([reference], [{"id": "unit", "status": "unlabeled_input_frozen"}],
                                         [], [], binding, None)

    def test_visible_text_duplicate_normalization_ignores_aliases(self):
        left = {"input": {"items": [{"id": 1, "text": "Unit  text"}]}}
        right = {"input": {"item_fields": ["id", "text"], "items": [[9, "Unit text"]]}}
        self.assertEqual(final.corpus_text(left), final.corpus_text(right))

    def test_source_cues_preserve_exact_times_unknown_gap_and_text(self):
        cues = sources.cues_from_srt("1\n00:00:01,100 --> 00:00:02,300\nUnit caption.\n\n2\n00:00:03,700 --> 00:00:04,900\nNext unit.\n")
        window = {"start": 0, "end": 6}
        document = sources.initial_document(cues, window, "unit-source")
        self.assertEqual(document["content"][2]["type"], "non_text")
        self.assertEqual(document["content"][2]["asr"], {"boundary_status": "unverified", "audio_class": "unknown"})
        sources.verify_initial_document(document, cues, window, "unit-source", "en")
        document["content"][1]["text"] = "Different unit."
        with self.assertRaisesRegex(ValueError, "differs_from_frozen"):
            sources.verify_initial_document(document, cues, window, "unit-source", "en")
        cues[1]["start"] = 2.0
        with self.assertRaisesRegex(ValueError, "overlapping"):
            sources.initial_document(cues, window, "unit-source")

    def test_public_manifest_hides_per_case_gold_and_binds_unlabeled_inputs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory).resolve()
            protocol, binding, inputs = (root / name for name in ("protocol.json", "binding.json", "inputs.json"))
            for path in (protocol, binding, inputs):
                path.write_text("{}\n")
            output = root / "artifacts/datasets/nep-passive-actions-final-test-v1/unit"
            with patch.object(final, "ROOT", root):
                final.publish(output, "unit", protocol, binding, {},
                              [{"id": "unit", "status": "known_preserve"}], {"admitted": 1}, input_manifest=inputs)
            value = json.loads((output / "manifest.json").read_text())
            self.assertEqual(value["slot_census"], [{"id": "unit", "status": "admitted"}])
            self.assertEqual(value["input_manifest"]["sha256"], final.file_sha(inputs))


if __name__ == "__main__":
    unittest.main()
