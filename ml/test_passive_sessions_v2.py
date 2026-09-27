import copy
import json
import tempfile
import unittest
from pathlib import Path

from passive_contract import PassiveContextBuilder, PassivePolicy
from passive_dataset import write
from passive_sessions_v2 import build_session, make_document, scenes


class CountingTokenizer:
    def apply_chat_template(self, messages, **_):
        return list(json.dumps(messages, ensure_ascii=False).encode())


def builder(max_items=64):
    return PassiveContextBuilder(
        CountingTokenizer(),
        policy=PassivePolicy(
            max_items=max_items,
            input_tokens=30000,
            output_tokens=1024,
            context_tokens=31024,
        ),
    )


class PassiveSessionTests(unittest.TestCase):
    def run_scene(self, name, *, context=None):
        scene = next(copy.deepcopy(s) for s in scenes() if s["name"] == name)
        language = "ko" if "-ko-" in scene["parent"] else "en"
        document, keys, duration = make_document(scene, name, language)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "document.json"
            write(path, document)
            source = {
                "id": name,
                "source_group_id": "authored-test-family",
                "split": "dev",
                "language": language,
                "document": document,
                "document_path": str(path),
                "source_base": {"source_id": scene["parent"]},
            }
            spec = {
                **scene,
                "id": name,
                "source_id": name,
                "source_region": [0, duration],
                "item_keys": keys,
            }
            return build_session(spec, source, context or builder())

    def test_modified_acceptance_exposes_actual_hedge_not_proposed_certainty(self):
        cases, artifact, _ = self.run_scene("count-modified-retains-hedge")
        feedback = cases[0]["input"]["recent_feedback"][0]
        self.assertEqual(feedback["outcome"], "modified")
        self.assertEqual(feedback["proposed_text"], "세 번 확인했습니다.")
        self.assertEqual(feedback["applied_text"], "아마 세 번쯤 확인했습니다.")
        self.assertEqual(cases[0]["expectedOutput"]["text"], "기록을 찾아보겠습니다.")
        self.assertEqual(
            artifact["committed_transitions"][0]["after"],
            artifact["checkpoints"][0]["before"],
        )

    def test_undo_restores_state_and_suppresses_only_reversed_effect(self):
        cases, artifact, _ = self.run_scene("recollection-undo-restores-uncertainty")
        self.assertEqual(cases[0]["input"]["recent_edits"], [])
        self.assertEqual(cases[0]["input"]["recent_feedback"][0]["outcome"], "undone")
        self.assertEqual(
            artifact["initial_document"], artifact["checkpoints"][0]["before"]
        )
        self.assertTrue(
            artifact["checkpoints"][0]["suppression_probes"][0]["suppressed"]
        )
        self.assertEqual(cases[0]["expectedOutput"]["action"], "correct_text")
        self.assertTrue(artifact["replay_summary"]["complete"])

    def test_changed_local_evidence_reopens_target_without_future_leakage(self):
        cases, artifact, _ = self.run_scene("changed-local-evidence-sign")
        first, changed = cases[:2]
        self.assertEqual(first["expectedOutput"], {"action": "no_action"})
        self.assertTrue(first["input"]["recent_feedback"][0]["same_context"])
        self.assertFalse(changed["input"]["recent_feedback"][0]["same_context"])
        self.assertEqual(changed["expectedOutput"]["text"], "Check the sign.")
        self.assertEqual(first["input"]["recent_edits"], [])
        self.assertEqual(len(changed["input"]["recent_edits"]), 1)
        self.assertEqual(
            [
                c["suppression_probes"][0]["suppressed"]
                for c in artifact["checkpoints"][:2]
            ],
            [True, False],
        )

    def test_multi_item_repair_preserves_intervals_and_does_not_guess_truncation(self):
        cases, artifact, _ = self.run_scene("oral-history-word-fragments")
        self.assertEqual(cases[0]["expectedOutput"]["target"], [0, 1])
        before = artifact["checkpoints"][0]["before"]["content"]
        after = artifact["checkpoints"][0]["after_primary"]["content"]
        timed_before = [i for i in before if "length" in i]
        timed_after = [i for i in after if "length" in i]
        self.assertAlmostEqual(
            timed_after[0]["length"], sum(i["length"] for i in timed_before[:2])
        )
        self.assertEqual(timed_before[-1], timed_after[-1])
        self.assertEqual(cases[-1]["expectedOutput"], {"action": "no_action"})

    def test_context_crop_cannot_create_a_clean_stop(self):
        with self.assertRaisesRegex(ValueError, "entire authored window"):
            self.run_scene("lab-expired-reassurance", context=builder(max_items=1))

    def test_feedback_never_carries_between_authored_sessions(self):
        self.run_scene("catalogue-dismissal-is-local")
        cases, artifact, _ = self.run_scene("museum-word-fragments")
        self.assertEqual(cases[0]["input"]["recent_feedback"], [])
        self.assertEqual(cases[0]["input"]["recent_edits"], [])
        self.assertEqual(artifact["feedback_ledger"]["records"], [])

    def test_useful_spelling_dismissal_changes_decision_without_hiding_other_error(
        self,
    ):
        cases, artifact, _ = self.run_scene("timing-dismissal-of-useful-spelling")
        before, dismissed, repaired_other = cases
        self.assertEqual(before["input"]["items"], dismissed["input"]["items"])
        self.assertEqual(before["input"]["recent_feedback"], [])
        self.assertEqual(dismissed["input"]["recent_feedback"][0]["reason"], "timing")
        self.assertEqual(before["expectedOutput"]["target"], [0])
        self.assertEqual(dismissed["expectedOutput"]["target"], [1])
        self.assertEqual(before["acceptableOutputs"], [dismissed["expectedOutput"]])
        self.assertEqual(repaired_other["expectedOutput"], {"action": "no_action"})
        self.assertIn("probbaly", repaired_other["input"]["items"][0]["text"])
        self.assertNotIn("comitting", repaired_other["input"]["items"][1]["text"])
        self.assertTrue(
            artifact["checkpoints"][2]["suppression_probes"][0]["suppressed"]
        )


if __name__ == "__main__":
    unittest.main()
