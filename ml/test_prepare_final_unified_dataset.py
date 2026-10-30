"""Admission regressions: unknown is not preservation and sources stay split."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import prepare_final_unified_dataset as final


class ReviewAdmissionTest(unittest.TestCase):
    def review(self, decision, **overrides):
        (final.ROOT / "artifacts").mkdir(exist_ok=True)
        value = {
            "reviewer_id": "independent-test-reviewer",
            "actor": "ai_generated",
            "model_predictions_consulted": False,
            "decisions": [decision],
            **overrides,
        }
        with tempfile.TemporaryDirectory(dir=final.ROOT / "artifacts") as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps(value))
            return final.read_reviews([path])

    def test_unknown_cannot_fabricate_abstention(self):
        decision = {
            "id": "unknown",
            "status": "unknown",
            "reason": "The visible cue is ambiguous.",
            "input_sha256": "input",
            "primary": {"action": "no_action"},
        }
        with self.assertRaisesRegex(ValueError, "Unknown must not become no_action"):
            self.review(decision)
        decision.pop("primary")
        rows, _ = self.review(decision)
        self.assertEqual(rows["unknown"]["status"], "unknown")

    def test_whole_window_and_alternative_necessity_are_required(self):
        decision = {
            "id": "preserve",
            "status": "preserve",
            "reason": "No action is useful in this visible window.",
            "input_sha256": "input",
            "primary": {"action": "no_action"},
        }
        with self.assertRaisesRegex(ValueError, "whole-window"):
            self.review(decision)
        decision["whole_visible_window_reviewed"] = True
        decision["acceptable"] = [{"action": "split_paragraph", "target": [1]}]
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            self.review(decision)

    def test_author_or_prediction_exposure_is_not_independent_review(self):
        decision = {
            "id": "unknown",
            "status": "unknown",
            "reason": "Unresolved wording.",
            "input_sha256": "input",
        }
        with self.assertRaisesRegex(ValueError, "author cannot"):
            self.review(decision, reviewer_id="final_dataset_agent")
        with self.assertRaisesRegex(ValueError, "prediction-blind"):
            self.review(decision, model_predictions_consulted=True)

    def test_symbolic_source_renaming_does_not_create_a_unique_input(self):
        left = {
            "items": [
                {
                    "source_id": "symbolic:one",
                    "text": "Keep this uncertainty.",
                    "duration_ms": 1200,
                }
            ]
        }
        right = copy.deepcopy(left)
        right["items"][0]["source_id"] = "symbolic:two"
        self.assertEqual(
            final.semantic_input_hash(left), final.semantic_input_hash(right)
        )
        right["items"][0]["text"] = "This is certain."
        self.assertNotEqual(
            final.semantic_input_hash(left), final.semantic_input_hash(right)
        )


if __name__ == "__main__":
    unittest.main()
