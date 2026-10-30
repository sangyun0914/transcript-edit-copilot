import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_history_length_actions_v3 as controls


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [8] * 120

    def encode(self, text, **kwargs):
        return [256 + ord(character) for character in text]


class HistoryLengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not controls.FIXTURE.is_file():
            raise unittest.SkipTest(
                "History diagnostics require local experiment fixtures."
            )
        cls.fixture = json.loads(controls.FIXTURE.read_text())
        controls.validate_fixture(cls.fixture)
        cls.tokenizer = Tokenizer()
        cls.built = {
            episode["id"]: controls.build_episode(
                cls.fixture, episode, cls.tokenizer, controls.digest(cls.fixture)
            )
            for episode in cls.fixture["episodes"]
        }
        cls.cases = [case for cases, _, _ in cls.built.values() for case in cases]
        cls.requests = [
            request for _, requests, _ in cls.built.values() for request in requests
        ]

    def review(self):
        return {
            "version": controls.VERSION,
            "draft_manifest_sha256": "draft-manifest",
            "reviewer_id": "independent-unit-fixture-reviewer",
            "actor": "ai_generated",
            "model_predictions_consulted": False,
            "decisions": [
                {
                    **controls.review_binding(case, request),
                    "status": "qualified",
                    "whole_visible_window_reviewed": True,
                    "rationale": "Unit fixture: all visible items and condition-specific evidence checked.",
                }
                for case, request in zip(self.cases, self.requests, strict=True)
            ],
        }

    def test_executed_chain_and_fixed_window_have_three_exact_views(self):
        for episode in self.fixture["episodes"]:
            cases, requests, proofs = self.built[episode["id"]]
            anchor = cases[-1]
            history = anchor["runtime"]["history"]
            with self.subTest(episode=episode["id"]):
                self.assertEqual(len(history), 8)
                self.assertEqual(len({item["transaction_id"] for item in history}), 8)
                self.assertEqual(
                    [item["sequence"] for item in history], list(range(1, 9))
                )
                for before, after in zip(history, history[1:]):
                    self.assertEqual(before["after"], after["before"])
                self.assertEqual(history[-1]["after"], anchor["document"])
                self.assertTrue(
                    all(item["before"] != item["after"] for item in history)
                )
                self.assertTrue(
                    all(
                        proof["replay"]["transactions"] == 3
                        for proof in proofs["history"]
                    )
                )
                for length, case, request in zip(
                    controls.LENGTHS, cases, requests, strict=True
                ):
                    controls.validate_request(request)
                    self.assertEqual(
                        case["primary"]["action"],
                        episode["expected_actions"][str(length)],
                    )
                    self.assertEqual(
                        case["input"], controls.projection(anchor["input"], length)
                    )
                    self.assertEqual(
                        case["bundle"]["mapping"], anchor["bundle"]["mapping"]
                    )
                    self.assertEqual(case["runtime"], anchor["runtime"])
                    self.assertEqual(case["document"], anchor["document"])
                    self.assertFalse(
                        {"primary", "rationale", "judgment"}.intersection(request)
                    )
                controls.validate_pair(requests[0], requests[1])
                controls.validate_pair(requests[1], requests[2])

    def test_latest_correction_revises_the_actual_same_item_twice(self):
        for episode in self.fixture["episodes"]:
            if episode["mechanism"] != "latest_correction_supersedes_older":
                continue
            cases, _, _ = self.built[episode["id"]]
            history = cases[-1]["runtime"]["history"]
            self.assertEqual(
                history[1]["prediction"]["target"], history[6]["prediction"]["target"]
            )
            self.assertEqual(
                history[1]["prediction"]["text"],
                episode["critical_edits"]["7"]["before"],
            )
            self.assertEqual(
                history[6]["prediction"]["text"],
                episode["critical_edits"]["7"]["after"],
            )
            self.assertEqual(cases[1]["primary"]["text"], episode["replacement"])
            self.assertEqual(cases[2]["primary"], cases[1]["primary"])

    def test_projection_rejects_missing_eighth_edit_or_nonempty_feedback(self):
        anchor = copy.deepcopy(self.cases[2]["input"])
        anchor["recent_edits"].pop()
        with self.assertRaisesRegex(ValueError, "eight actual edits"):
            controls.projection(anchor, 3)
        anchor = copy.deepcopy(self.cases[2]["input"])
        anchor["recent_feedback"] = ["unexpected feedback"]
        with self.assertRaisesRegex(ValueError, "Feedback"):
            controls.projection(anchor, 0)

    def test_hidden_name_cannot_leak_through_non_history_fields(self):
        for episode in self.fixture["episodes"]:
            cases, _, _ = self.built[episode["id"]]
            controls.assert_no_visible_name_leak(
                cases[-1]["input"], episode["hidden_names"]
            )
            changed = copy.deepcopy(cases[-1]["input"])
            changed["items"][-1]["text"] += episode["hidden_names"][0]
            with self.assertRaisesRegex(ValueError, "leaked"):
                controls.assert_no_visible_name_leak(changed, episode["hidden_names"])

    def test_admission_requires_complete_bound_independent_review(self):
        manifest = {"author_id": self.fixture["provenance"]["author_id"]}
        review = self.review()
        self.assertEqual(
            len(
                controls.validate_review(
                    review, manifest, "draft-manifest", self.cases, self.requests
                )
            ),
            24,
        )
        changes = []
        wrong = copy.deepcopy(review)
        wrong["decisions"].pop()
        changes.append(wrong)
        wrong = copy.deepcopy(review)
        wrong["reviewer_id"] = manifest["author_id"]
        changes.append(wrong)
        wrong = copy.deepcopy(review)
        wrong["model_predictions_consulted"] = True
        changes.append(wrong)
        for field, value in (
            ("input_sha256", "wrong"),
            ("accepted_result_sha256", ["wrong"]),
            ("whole_visible_window_reviewed", False),
            ("status", "uncertain"),
        ):
            wrong = copy.deepcopy(review)
            wrong["decisions"][0][field] = value
            changes.append(wrong)
        for wrong in changes:
            with self.assertRaises(ValueError):
                controls.validate_review(
                    wrong, manifest, "draft-manifest", self.cases, self.requests
                )

    def test_only_admitted_artifacts_get_runnable_request_names(self):
        def already_built(fixture, episode, tokenizer, fixture_sha256):
            result = copy.deepcopy(self.built[episode["id"]])
            for rows in result[:2]:
                for row in rows:
                    row["history_length"]["fixture_sha256"] = fixture_sha256
            return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft, admitted = root / "draft", root / "admitted"
            with patch.object(controls, "build_episode", side_effect=already_built):
                controls.prepare(
                    controls.FIXTURE, draft, self.tokenizer, {"kind": "unit_fixture"}
                )
            self.assertFalse(list(draft.rglob("requests-json.jsonl")))
            with self.assertRaises(ValueError):
                controls.admit(draft, draft / "review-template.json", admitted)
            self.assertFalse(admitted.exists())
            review = json.loads((draft / "review-template.json").read_text())
            review["reviewer_id"] = "independent-unit-fixture-reviewer"
            for decision in review["decisions"]:
                decision.update(
                    status="qualified",
                    whole_visible_window_reviewed=True,
                    rationale="Unit fixture: the full visible view is qualified.",
                )
            review_path = root / "review.json"
            review_path.write_text(json.dumps(review))
            controls.admit(draft, review_path, admitted)
            for length in controls.LENGTHS:
                directory = admitted / controls.arm_name(length)
                requests = controls.dataset.rows(directory / "requests-json.jsonl")
                self.assertEqual(len(requests), 8)
                self.assertTrue(
                    all(row["history_length"]["length"] == length for row in requests)
                )
                manifest = json.loads((directory / "manifest.json").read_text())
                self.assertEqual(manifest["version"], controls.conditions.VERSION)
                self.assertEqual(manifest["status"], "prepared_not_inferred")


if __name__ == "__main__":
    unittest.main()
