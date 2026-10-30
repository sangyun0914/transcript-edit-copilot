import copy
import unittest
from collections import Counter
from unittest.mock import patch

import prepare_passive_actions_v3 as preparation
from replay_session import replay, session_hash


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        return [8] * 100 if tokenize else preparation.packed(messages)


class ActionsV3PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (preparation.FIXTURES / "revision-005").is_dir():
            raise unittest.SkipTest("Local authored research data is not distributed.")
        cls.builder = preparation.contract.PassiveContextBuilder(Tokenizer())
        with patch.object(
            preparation, "FIXTURES", preparation.FIXTURES / "revision-005"
        ):
            cls.cases, cls.excluded = preparation.build_authored_cases(cls.builder)
        cls.by_id = {case["id"]: case for case in cls.cases}

    def test_complete_authored_references_replay_and_source_integrity(self):
        counts = Counter(
            (case["split"], case["language"], case["primary"]["action"])
            for case in self.cases
        )
        for split in ("train", "dev"):
            for language in ("en", "ko"):
                self.assertEqual(
                    {
                        action
                        for s, lang, action in counts
                        if s == split and lang == language
                    },
                    set(preparation.contract.ALLOWED_ACTIONS),
                )
        for case in self.cases:
            proof = preparation.validate_case(case, self.builder)
            if proof["changed"]:
                journal = copy.deepcopy(proof["journal"])
                journal[0]["document"] = case["document"]
                after, summary = replay(journal)
                self.assertEqual(summary["transactions"], 3)
                self.assertEqual(session_hash(after), proof["after_sha256"])
            self.assertFalse(case["source"]["natural_audio"])
            self.assertFalse(case["source"]["acoustically_verified"])
            self.assertFalse(case["judgment"]["observed_action"])

    def test_unknown_is_not_a_supervised_abstention(self):
        self.assertEqual(len(self.excluded), 2)
        self.assertTrue(
            all(case["judgment_status"] == "unknown" for case in self.excluded)
        )
        self.assertFalse(
            set(case["id"] for case in self.excluded).intersection(self.by_id)
        )
        damaged = copy.deepcopy(self.cases[0])
        damaged["judgment"]["status"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown is not no_action"):
            preparation.validate_case(damaged, self.builder)

    def test_feedback_changes_reference_without_fabricating_an_acceptance(self):
        pairs = (
            ("river-expired-correction", "river-dismissed-correction"),
            ("ko-radio-correction-expired", "ko-radio-correction-dismissed"),
        )
        for expired_id, dismissed_id in pairs:
            expired = self.by_id["actions-v3-" + expired_id]
            dismissed = self.by_id["actions-v3-" + dismissed_id]
            self.assertEqual(expired["primary"]["action"], "correct_text")
            self.assertEqual(dismissed["primary"]["action"], "no_action")
            self.assertEqual(
                expired["document"]["content"][1]["text"],
                dismissed["document"]["content"][1]["text"],
            )
            self.assertEqual(
                expired["input"]["recent_feedback"][-1]["outcome"], "expired"
            )
            self.assertEqual(
                dismissed["input"]["recent_feedback"][-1]["outcome"], "dismissed"
            )
            self.assertEqual(expired["runtime"]["history"], [])
            self.assertEqual(dismissed["runtime"]["history"], [])

    def test_full_cut_transaction_rejects_timing_tampering(self):
        case = self.by_id["actions-v3-repair-restart"]
        self.assertEqual(len(case["primary"]["target"]), 3)
        broken = copy.deepcopy(case)
        broken["document"]["content"][2]["length"] += 0.2
        with self.assertRaises(ValueError):
            preparation.validate_case(broken, self.builder)

    def test_reference_history_is_executed_and_split_is_registered(self):
        splits = preparation.source_splits()
        self.assertEqual(splits["shuka-local-recording"], "dev")
        for case in self.cases:
            self.assertEqual(splits[case["source"]["source_group"]], case["split"])
            if case["runtime"]["history"]:
                self.assertEqual(
                    case["runtime"]["history"][-1]["after"], case["document"]
                )
            user_message = case["messages"][1]["content"]
            self.assertNotIn(case["rationale"], user_message)
            self.assertNotIn(case["id"], user_message)
            self.assertNotIn(case["source"]["source_group"], user_message)

    def test_audio_preservation_does_not_hide_needed_caption_boundaries(self):
        identities = (
            "ferry-horn",
            "radio-question-pause",
            "ko-seed-wait",
            "ko-balloon-countdown",
        )
        for identity in identities:
            case = self.by_id["actions-v3-" + identity]
            texts = [item for item in case["input"]["items"] if item["type"] == "text"]
            self.assertNotEqual(texts[-1]["paragraph"], texts[0]["paragraph"])
            bundle = preparation.runtime_bundle(case, self.builder)
            with self.assertRaises(ValueError):
                preparation.contract.apply_passive_edit(
                    case["document"],
                    bundle,
                    {"action": "split_paragraph", "target": [texts[-1]["id"]]},
                )


if __name__ == "__main__":
    unittest.main()
