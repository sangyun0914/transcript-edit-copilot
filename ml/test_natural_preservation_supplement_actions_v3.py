import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from replay_session import replay, session_hash
from verify_release_export import Exporter

PATH = (
    Path(__file__).parent
    / "fixtures/passive-actions-v3/prepare_natural_preservation_supplement_actions_v3.py"
)
if not PATH.is_file():
    raise unittest.SkipTest("Local research authoring helpers are not distributed.")
SPEC = importlib.util.spec_from_file_location("natural_supplement", PATH)
supplement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supplement)
data = supplement.data


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [8] * 100


class NaturalPreservationSupplementTests(unittest.TestCase):
    def state(self):
        document, keys, _ = data.authored_document(
            {
                "id": "supplement-test",
                "language": "en",
                "paragraphs": [
                    [{"key": "first", "ms": 1500, "text": "Please read teh label."}],
                    [
                        {
                            "key": "focus",
                            "ms": 2000,
                            "text": "Leave this sentence intact.",
                        }
                    ],
                ],
            }
        )
        state = {
            "source": {"id": "test-only"},
            "document": document,
            "runtime": {
                "focus_uuid": keys["focus"],
                "selection": [],
                "sequence": 0,
                "history": [],
                "feedback_journal": None,
                "caption_profile": data.PROFILE,
            },
        }
        return state, data.contract.PassiveContextBuilder(Tokenizer())

    def test_prior_edit_cannot_move_cursor_after_deleting_it(self):
        state, builder = self.state()
        with self.assertRaisesRegex(ValueError, "deleted the fixed original focus"):
            supplement.apply_prior(
                state, {"action": "delete_span", "target": [1]}, "test", builder, None
            )
        self.assertEqual(state["runtime"]["history"], [])

    def test_full_document_history_and_prior_proof_hydrate_from_before(self):
        state, builder = self.state()
        exporter = Exporter()
        try:
            edited = supplement.apply_prior(
                state,
                {
                    "action": "correct_text",
                    "target": [0],
                    "text": "Please read the label.",
                },
                "Explicit test typo repair",
                builder,
                exporter,
            )
        finally:
            exporter.close()
        self.assertEqual(
            edited["runtime"]["focus_uuid"], state["runtime"]["focus_uuid"]
        )
        self.assertEqual(
            len(edited["document"]["content"]), len(state["document"]["content"])
        )
        history = edited["runtime"]["history"][0]
        self.assertEqual(history["before"], state["document"])
        self.assertEqual(history["after"], edited["document"])
        proof = edited["prior_execution"][0]["proof"]
        self.assertEqual(
            proof["hydrate_start_document_from"], "case.runtime.history[0].before"
        )
        journal = copy.deepcopy(proof["journal"])
        journal[0]["document"] = history["before"]
        restored, summary = replay(journal)
        self.assertEqual(summary["transactions"], 3)
        self.assertEqual(session_hash(restored), proof["after_sha256"])

    def test_metadata_sampler_rejects_dev_and_generated_audio_before_ranking(self):
        normal = {
            "id": "real-train",
            "split": "train",
            "language": "en",
            "natural_audio": True,
            "generated_audio": False,
            "source_group": "mit-6006-s2020",
            "source_id": "recording",
            "stratum": "natural_recorded",
            "action": "merge_paragraph",
            "selection_rank": "9",
        }
        pool = [
            normal,
            {**normal, "id": "dev", "split": "dev", "selection_rank": "0"},
            {
                **normal,
                "id": "generated",
                "generated_audio": True,
                "selection_rank": "0",
            },
        ]
        with patch.dict(supplement.POLICY, {"buckets": {"mit_source_first": 1}}):
            self.assertEqual(
                [row["id"] for row in supplement.choose(pool)], ["real-train"]
            )


if __name__ == "__main__":
    unittest.main()
