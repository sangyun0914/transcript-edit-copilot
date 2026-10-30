import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_passive_actions_v3 as evaluation
import infer_gemma_actions_v3 as inference
import prepare_actions_v3_preferences as preferences
from test_passive_actions import actions, builder, document


class EvaluationTests(unittest.TestCase):
    def case(self, primary=None, ledger=None):
        doc = document()
        context = builder().build(doc, "b", as_of=3 if ledger else 0, feedback=ledger)
        primary = primary or {"action": "correct_text", "target": [1], "text": "we"}
        return {
            "id": "case",
            "language": "en",
            "source": {"source_group": "fixture", "kind": "symbolic"},
            "document": doc,
            "bundle": context,
            "input": context["input"],
            "primary": primary,
            "acceptable": [primary],
            "accepted_predictions": [primary],
            "accepted_result_sha256": [
                evaluation.result_hash(
                    actions.apply_passive_edit(doc, context, primary)
                )
            ],
            "judgment": {
                "status": "preserve"
                if primary["action"] == "no_action"
                else "edit_required"
            },
            "runtime": {
                "sequence": 3 if ledger else 0,
                "feedback_journal": ledger.dump() if ledger else None,
            },
        }

    def observation(self, case, prediction, **changes):
        return {
            "case_id": case["id"],
            "format": "json",
            "input_sha256": evaluation.semantic_input_hash(case["input"]),
            "messages_sha256": evaluation.digest(
                evaluation.formats.messages(case["input"])
            ),
            "status": "completed",
            "text": evaluation.packed(prediction),
            "finish_reason": "stop",
            "generation_ms": 123.0,
            **changes,
        }

    def score(self, case, prediction, **changes):
        return evaluation.evaluate_one(
            case, self.observation(case, prediction, **changes)
        )

    def test_equivalent_rendered_result_can_use_a_different_correction_span(self):
        case = self.case()
        row = self.score(
            case, {"action": "correct_text", "target": [1, 2], "text": "we can begin."}
        )
        self.assertTrue(row["useful"])
        self.assertEqual(row["source_integrity"], "passed")

    def test_accepted_alternative_counts_as_the_action_actually_predicted(self):
        case = self.case()
        alternative = {"action": "delete_span", "target": [0]}
        case["accepted_predictions"].append(alternative)
        case["acceptable"] = case["accepted_predictions"]
        case["accepted_result_sha256"].append(
            evaluation.result_hash(
                actions.apply_passive_edit(
                    case["document"], case["bundle"], alternative
                )
            )
        )
        row = self.score(case, alternative)
        self.assertTrue(row["useful"])
        self.assertEqual(row["reference_action"], "correct_text")
        self.assertEqual(row["predicted_action_language"], "delete_span:en")
        self.assertEqual(
            evaluation.summarize([row])["successful_model_actions"], {"delete_span": 1}
        )

    def test_invalid_or_truncated_no_action_is_not_preservation(self):
        case = self.case({"action": "no_action"})
        for prediction, changes in (
            ({"action": "no_action", "target": [1]}, {}),
            ({"action": "no_action"}, {"finish_reason": "length"}),
        ):
            row = self.score(case, prediction, **changes)
            self.assertEqual(row["status"], "invalid_output")
            self.assertFalse(row["preserved"])
            self.assertFalse(row["model_abstained"])
        self.assertTrue(self.score(case, {"action": "no_action"})["preserved"])

    def test_request_failure_is_retained_in_every_request_denominator(self):
        case = self.case({"action": "no_action"})
        failed = self.score(
            case, None, status="timeout", error="deadline", generation_ms=1000
        )
        good = self.score(case, {"action": "no_action"})
        result = evaluation.summarize([failed, good])
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["preservation"]["value"], 0.5)
        self.assertEqual(result["validity"]["value"], 0.5)
        self.assertIsNone(result["precision"]["value"])

    def test_valid_wrong_edit_counts_against_precision_and_recall(self):
        case = self.case()
        wrong = self.score(case, {"action": "delete_span", "target": [2]})
        good = self.score(case, case["primary"])
        abstained = self.score(case, {"action": "no_action"})
        result = evaluation.summarize([wrong, good, abstained])
        self.assertTrue(wrong["valid"])
        self.assertFalse(wrong["useful"])
        self.assertEqual(result["precision"]["value"], 0.5)
        self.assertEqual(result["recall"]["value"], 1 / 3)
        self.assertEqual(result["f1"], 0.4)

    def test_suppression_preserves_but_is_not_model_abstention(self):
        doc = document()
        context = builder().build(doc, "b", as_of=0)
        edit = {"action": "delete_span", "target": [0]}
        ledger = actions.PassiveFeedbackLedger("document")
        ledger.propose("proposal", 1, doc, context, edit)
        ledger.record("proposal", "shown", 2)
        ledger.record("proposal", "dismissed", 3)
        case = self.case({"action": "no_action"}, ledger)
        row = self.score(case, edit)
        self.assertTrue(row["suppressed"])
        self.assertTrue(row["preserved"])
        self.assertFalse(row["model_abstained"])
        self.assertFalse(row["displayed"])
        self.assertEqual(evaluation.summarize([row])["model_preservation"]["value"], 0)

    def test_wrong_input_or_prompt_and_unknown_reference_fail_closed(self):
        case = self.case()
        for field in ("input_sha256", "messages_sha256", "case_id"):
            observation = self.observation(case, case["primary"], **{field: "other"})
            with self.assertRaises(ValueError):
                evaluation.evaluate_one(case, observation)
        case["judgment"]["status"] = "unknown"
        with self.assertRaises(ValueError):
            self.score(case, {"action": "no_action"})

    def test_changed_reference_cannot_relabel_a_saved_prediction(self):
        case = self.case()
        changed = copy.deepcopy(case)
        changed["primary"]["text"] = "we can"
        with self.assertRaises(ValueError):
            self.score(changed, case["primary"])

    def test_saved_run_cannot_mix_models_or_change_native_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "case_id": "case",
                "format": "json",
                "input": self.case()["input"],
                "input_sha256": "input",
                "messages_sha256": "messages",
                "prefix_ids_sha256": "prefix",
            }
            requests = root / "requests-json.jsonl"
            evaluation.write_rows(requests, [request])
            run = root / "baseline"
            evaluation.baseline(root, run)
            original = evaluation.rows(run / "observations.jsonl")
            self.assertEqual(
                len(evaluation.verify_observations(run, requests, "json")[0]), 1
            )
            for field, value in (
                ("model_identity", {"kind": "another_model"}),
                ("prefix_ids_sha256", "another_prefix"),
            ):
                changed = copy.deepcopy(original)
                changed[0][field] = value
                (run / "observations.jsonl").unlink()
                evaluation.write_rows(run / "observations.jsonl", changed)
                manifest = json.loads((run / "manifest.json").read_text())
                manifest["files_sha256"]["observations.jsonl"] = evaluation.sha256(
                    run / "observations.jsonl"
                )
                (run / "manifest.json").unlink()
                evaluation.write(run / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    evaluation.verify_observations(run, requests, "json")

    def test_preference_cannot_reject_equivalent_executed_result(self):
        case = self.case()
        candidate = {
            "text": json.dumps(
                {"action": "correct_text", "target": [1, 2], "text": "we can begin."},
                indent=2,
            ),
            "format": "json",
        }
        with self.assertRaises(ValueError):
            preferences.verify_strict_preference(
                case, candidate, case["bundle"], case["primary"]
            )
        candidate["text"] = '{"action":"no_action"}'
        preferences.verify_strict_preference(
            case, candidate, case["bundle"], case["primary"]
        )

    def test_registered_condition_survives_request_baseline_and_grade(self):
        case = self.case({"action": "no_action"})
        condition = evaluation.conditions.condition_specs([])["canonical_full_context"]
        case.update(condition=condition, condition_sha256=evaluation.digest(condition))
        messages = evaluation.conditions.expected_messages(
            case["input"], "json", condition
        )
        request = {
            "case_id": case["id"],
            "split": "dev",
            "format": "json",
            "input": case["input"],
            "input_sha256": evaluation.semantic_input_hash(case["input"]),
            "messages": messages,
            "messages_sha256": evaluation.digest(messages),
            "prefix_ids": [2, 3],
            "prefix_ids_sha256": evaluation.digest([2, 3]),
            "condition": condition,
            "condition_sha256": case["condition_sha256"],
        }
        inference.validate_requests([request], "json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            evaluation.write_rows(prepared / "cases.jsonl", [case])
            evaluation.write_rows(prepared / "requests-json.jsonl", [request])
            evaluation.write(
                prepared / "manifest.json",
                {
                    "version": evaluation.conditions.VERSION,
                    "final_test_bodies_read": False,
                    "excluded": 0,
                    "sources_sha256": {},
                    "files_sha256": {
                        p.name: evaluation.sha256(p) for p in prepared.iterdir()
                    },
                },
            )
            evaluation.baseline(prepared, root / "run")
            result = evaluation.grade(prepared, root / "run", root / "graded", "json")
            self.assertEqual(result["preservation"]["value"], 1)
            self.assertEqual(result["requests"], 1)

    def test_rewrite_overflow_does_not_remove_json_eligible_cases(self):
        short = self.case({"action": "no_action"})
        long = copy.deepcopy(short)
        short["id"], long["id"] = "short", "long"
        long["input"]["cursor_before"] = 0
        long["bundle"]["input"] = long["input"]
        for case in (short, long):
            case["split"] = "dev"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            cases_path = release / "cases/dev.jsonl"
            evaluation.write_rows(cases_path, [short, long])
            evaluation.write(
                release / "manifest.json",
                {"files_sha256": {"cases/dev.jsonl": evaluation.sha256(cases_path)}},
            )
            evaluation.write(
                root / "review.json",
                {
                    "status": "passed",
                    "manifest_sha256": evaluation.sha256(release / "manifest.json"),
                    "model_predictions_consulted": False,
                },
            )

            def prefixes(tokenizer, snapshot, name):
                size = (
                    3634
                    if snapshot["cursor_before"] == 0 and name == "rewrite"
                    else 3580
                )
                return [2] * size

            with (
                patch.object(
                    evaluation.dataset.legacy, "tokenizer", return_value=(None, {})
                ),
                patch.object(evaluation.contract, "PassiveContextBuilder"),
                patch.object(
                    evaluation.conditions, "verify_reference_proofs", return_value={}
                ),
                patch.object(
                    evaluation.dataset,
                    "runtime_bundle",
                    side_effect=lambda case, builder: case["bundle"],
                ),
                patch.object(evaluation.formats, "prefix_ids", side_effect=prefixes),
            ):
                common = evaluation.prepare(
                    release, root / "review.json", root / "common"
                )
                full_json = evaluation.prepare(
                    release, root / "review.json", root / "json", "json"
                )
            self.assertEqual((common["requests"], common["excluded"]), (1, 1))
            self.assertEqual((full_json["requests"], full_json["excluded"]), (2, 0))
            self.assertFalse((root / "json/requests-rewrite.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
