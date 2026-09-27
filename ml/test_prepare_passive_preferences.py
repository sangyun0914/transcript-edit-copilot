import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_passive_preferences as preferences
import test_passive_benchmark as benchmark_tests
from test_passive_contract import CORRECTION
from passive_contract import PassiveFeedbackLedger


class PreferenceQualificationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = benchmark_tests.PassiveBenchmarkTests()
        self.fixture.setUp()

    def candidate(self, value, finish="stop"):
        return {
            "text": json.dumps(value) if not isinstance(value, str) else value,
            "finish_reason": finish,
        }

    def case(self, **kwargs):
        case = self.fixture.case(**kwargs)
        case["metadata"]["split"] = "train"
        return case

    def test_acceptable_alternatives_are_ties_and_valid_unmatched_edits_are_excluded(
        self,
    ):
        case = self.case()
        alternative = {
            "action": "correct_text",
            "target": [2],
            "text": "another check.",
        }
        case["acceptableOutputs"] = [alternative]
        for value in (CORRECTION, alternative):
            self.assertEqual(
                preferences.classify(case, self.candidate(value))["status"], "tie"
            )
        uncertain = {
            "action": "correct_text",
            "target": [2],
            "text": "another assessment.",
        }
        result = preferences.classify(case, self.candidate(uncertain))
        self.assertEqual(
            result,
            {
                "status": "excluded",
                "reason": "valid_unmatched_edit_requires_semantic_review",
            },
        )
        result = preferences.classify(
            case, self.candidate({"action": "no_action"}, "length")
        )
        self.assertEqual(result["status"], "excluded")

    def test_preferences_allow_chosen_edits_and_abstentions_without_length_shortcuts(
        self,
    ):
        needed = self.case()
        missed = preferences.classify(needed, self.candidate({"action": "no_action"}))
        self.assertEqual(missed["reason"], "missed_required_edit")
        self.assertEqual(missed["status"], "qualified")
        preserve = self.case(expected={"action": "no_action"})
        overedit = preferences.classify(preserve, self.candidate(CORRECTION))
        self.assertEqual(overedit["reason"], "unnecessary_edit_on_preserve_state")
        self.assertEqual(overedit["status"], "qualified")
        invalid = preferences.classify(
            needed,
            self.candidate({"action": "correct_text", "target": [999], "text": "x"}),
        )
        self.assertEqual(invalid["reason"], "semantic_invalid")
        malformed = preferences.classify(needed, self.candidate("not JSON"))
        self.assertEqual(malformed["reason"], "format_invalid")

    def test_conflicting_accepted_reference_and_repeat_gate_are_excluded(self):
        ledger = PassiveFeedbackLedger("doc")
        ledger.propose("p", 1, self.fixture.document, self.fixture.bundle, CORRECTION)
        ledger.record("p", "shown", 2)
        ledger.record("p", "dismissed", 3)
        current = self.fixture.builder.build(
            self.fixture.document, "b", as_of=3, feedback=ledger
        )
        case = self.case(
            expected={"action": "no_action"}, bundle=current, ledger=ledger.dump()
        )
        case["acceptableOutputs"] = [CORRECTION]
        result = preferences.classify(case, self.candidate(CORRECTION))
        self.assertEqual(
            result,
            {
                "status": "excluded",
                "reason": "accepted_reference_conflicts_with_repeat_gate",
            },
        )
        case["metadata"]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "development/test"):
            preferences.classify(case, self.candidate(CORRECTION))


class PairIntegrityTests(unittest.TestCase):
    def test_unknown_payload_is_rejected_before_any_payload_content_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": preferences.VERSION,
                        "files_sha256": {"dev-cases.jsonl": "a" * 64},
                    }
                )
            )
            with patch.object(
                preferences.smoke,
                "check_file",
                side_effect=AssertionError("read a payload"),
            ):
                with self.assertRaisesRegex(ValueError, "inventory"):
                    preferences.verify_bundle(
                        root, root / "anchor", root / "candidates"
                    )

    def test_all_candidates_are_validated_even_when_a_row_would_not_be_selected(self):
        bundle = {
            "cases": [{"id": str(index)} for index in range(809)],
            "tokens": [
                {"primary": {"input_ids": [1, 2, 3, 7], "completion_start": 2}}
                for _ in range(809)
            ],
            "configuration": {"tokenizer": {"eos_token_id": 7, "vocab_size": 10}},
        }
        original = [
            {
                "case_id": str(index),
                "prompt_ids_sha256": hashlib.sha256(b"[1,2]").hexdigest(),
                "output_ids": [3, 7],
                "text": "chosen",
                "finish_reason": "stop",
            }
            for index in range(809)
        ]

        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                return "chosen"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "generator.py").write_text("test fixture only")
            generator_sha = preferences.smoke.digest(root / "generator.py")
            (root / "runtime.json").write_text('{"test_fixture_only":true}')
            preferences.smoke.write(
                root / "protocol.json",
                {
                    "qualification_policy_sha256": "policy",
                    "generator_sha256": generator_sha,
                },
            )
            preferences.smoke.write(
                root / "run.json",
                {
                    "configuration_sha256": "config",
                    "preparation_sha256": "preparation",
                    "completed_candidates": 809,
                    "requested_candidates": 809,
                    "training_updates": 0,
                    "candidate_source": "final_T1_NF4_adapter",
                    "generator_sha256": generator_sha,
                    "protocol_sha256": preferences.smoke.digest(root / "protocol.json"),
                },
            )

            def publish(rows):
                (root / "candidates.jsonl").write_bytes(
                    preferences.data.jsonl_bytes(rows)
                )
                preferences.smoke.write(
                    root / "manifest.json",
                    {
                        "version": "passive-preference-candidates-v1",
                        "status": "completed_train_candidates_unqualified",
                        "files_sha256": {
                            name: preferences.smoke.digest(root / name)
                            for name in preferences.CANDIDATE_PAYLOADS
                        },
                    },
                )

            publish(original)
            _, loaded = preferences.load_candidates(
                root, bundle, "config", "policy", "preparation", Tokenizer()
            )
            self.assertEqual(len(loaded), 809)
            for fault in ("wrong_text", "false_eos", "wrong_prefix"):
                changed = copy.deepcopy(original)
                if fault == "wrong_text":
                    changed[-1]["text"] = "text not generated by those IDs"
                elif fault == "false_eos":
                    changed[-1]["output_ids"][-1] = 6
                else:
                    changed[-1]["prompt_ids_sha256"] = "different-state"
                publish(changed)
                with self.subTest(fault=fault), self.assertRaises(ValueError):
                    preferences.load_candidates(
                        root, bundle, "config", "policy", "preparation", Tokenizer()
                    )

    def pair(self, index=0, action="correct_text", language="en"):
        messages = [{"role": "user", "content": str(index)}]
        chosen = (
            '{"action":"no_action"}'
            if action == "no_action"
            else "longer useful correction"
        )
        rejected = "wrong"
        identity = preferences.pair_identity(messages, chosen, rejected)
        return {
            "pair_id": identity,
            "case_id": str(index),
            "messages": messages,
            "metadata": {
                "split": "train",
                "source_id": f"source-{index}",
                "source_group_id": f"group-{index % 2}",
                "episode_id": str(index),
                "chosen_action": action,
                "language": language,
            },
            "chosen_text": chosen,
            "rejected_text": rejected,
            "preference_origin": "AI",
            "reason": "format_invalid",
            "prompt_ids_sha256": hashlib.sha256(b"[1,2]").hexdigest(),
            "chosen": {
                "input_ids": [1, 2, 3, 4, 7],
                "attention_mask": [1] * 5,
                "labels": [-100, -100, 3, 4, 7],
                "completion_start": 2,
            },
            "rejected": {
                "input_ids": [1, 2, 5, 7],
                "attention_mask": [1] * 4,
                "labels": [-100, -100, 5, 7],
                "completion_start": 2,
            },
        }

    def test_repeated_same_state_pairs_are_removed_before_fixed_balanced_selection(
        self,
    ):
        pairs = [
            self.pair(
                index,
                "no_action" if index % 3 == 0 else "correct_text",
                "en" if index % 2 else "ko",
            )
            for index in range(80)
        ]
        duplicate = copy.deepcopy(pairs[0])
        duplicate["case_id"] = "another-identity-same-prompt-and-pair"
        selected, excluded = preferences.select_pairs([*pairs, duplicate], maximum=32)
        again, _ = preferences.select_pairs([*pairs, duplicate], maximum=32)
        self.assertEqual(selected, again)
        self.assertEqual(len(selected), 32)
        self.assertEqual(len({pair["pair_id"] for pair in selected}), 32)
        self.assertEqual(
            sum(row["reason"] == "exact_same_state_pair_duplicate" for row in excluded),
            1,
        )
        self.assertTrue(preferences.readiness(selected)["passed"])
        self.assertFalse(
            preferences.readiness(
                [
                    pair
                    for pair in selected
                    if pair["metadata"]["chosen_action"] != "no_action"
                ]
            )["passed"]
        )

    def test_masks_and_eos_are_checked_and_both_sides_require_the_identical_state(self):
        pair = self.pair()
        preferences.validate_pairs([pair], 7, 10)
        for fault in ("prefix", "prompt_loss", "eos", "dev", "duplicate"):
            changed = copy.deepcopy(pair)
            batch = [changed]
            if fault == "prefix":
                changed["rejected"]["input_ids"][0] = 6
            elif fault == "prompt_loss":
                changed["chosen"]["labels"][0] = 1
            elif fault == "eos":
                changed["rejected"]["input_ids"][-1] = changed["rejected"]["labels"][
                    -1
                ] = 6
            elif fault == "dev":
                changed["metadata"]["split"] = "dev"
            else:
                batch.append(copy.deepcopy(changed))
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                preferences.validate_pairs(batch, 7, 10)


if __name__ == "__main__":
    unittest.main()
