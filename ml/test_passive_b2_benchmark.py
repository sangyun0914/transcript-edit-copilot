import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import passive_b2_benchmark as b2
import passive_benchmark as b1
from passive_contract import PassiveFeedbackLedger, canonical_messages
from replay_session import session_hash
from session_cases import replayable_session
from test_passive_contract import CORRECTION, builder, document


class PassiveB2Tests(unittest.TestCase):
    def setUp(self):
        self.document = document()
        self.builder = builder()
        self.bundle = self.builder.build(self.document, "b", as_of=0)

    def case(self, identity="case", *, expected=CORRECTION, bundle=None, ledger=None):
        bundle = bundle or self.bundle
        return {
            "id": identity,
            "input": bundle["input"],
            "expectedOutput": expected,
            "acceptableOutputs": [],
            "metadata": {
                "split": "dev",
                "source_group_id": "dev-source",
                "language": "en",
                "episode_id": identity,
                "episode_step": 0,
                "release_stratum": "feedback_sessions",
                "document_before_hash": session_hash(self.document),
                "input_sha256": session_hash(canonical_messages(bundle["input"])),
                "context": bundle["context"],
            },
            "_execution": {
                "before": self.document,
                "bundle": bundle,
                "feedback_ledger": ledger,
                "as_of": bundle["context"]["as_of"],
            },
        }

    def test_full_loader_keeps_order_and_verifies_split_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            release = root / "frozen"
            release.mkdir()
            cases = [self.case("second"), self.case("first")]
            for case in cases:
                case.pop("_execution")
                case["reproduction"] = {
                    "candidate_directory": ".",
                    "episode": "episode.json",
                }
            journal = replayable_session(self.document, self.document, "no_action", [])
            journal[0].pop("document")
            (root / "source.json").write_text(json.dumps(self.document))
            (root / "episode.json").write_text(
                json.dumps(
                    {
                        "spec": {"corruptions": []},
                        "source_document": "source.json",
                        "journal": journal,
                    }
                )
            )
            (release / "dev-cases.jsonl").write_text(
                "\n".join(json.dumps(case) for case in cases)
            )
            (release / "dev-messages.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": case["id"],
                            "messages": canonical_messages(case["input"]),
                        }
                    )
                    for case in cases
                )
            )
            train_cases = copy.deepcopy(cases)
            for case in train_cases:
                case["metadata"].update(split="train", source_group_id="train-source")
            (release / "train-cases.jsonl").write_text(
                "\n".join(json.dumps(case) for case in train_cases)
            )
            (release / "train-messages.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": case["id"],
                            "messages": canonical_messages(case["input"])
                            + [
                                {
                                    "role": "assistant",
                                    "content": b2.serialize(
                                        case["input"], case["expectedOutput"]
                                    ),
                                }
                            ],
                        }
                    )
                    for case in train_cases
                )
            )
            groups = release / "source-groups.json"
            groups.write_text(
                json.dumps({"dev-source": "dev", "train-source": "train"})
            )

            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            manifest = {
                "version": "passive-dataset-completion-v2",
                "status": "frozen_supported_scope",
                "files_sha256": {path.name: digest(path) for path in release.iterdir()},
                "dependencies_sha256": {
                    path.name: digest(path)
                    for path in (root / "source.json", root / "episode.json")
                },
            }
            manifest_path = release / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(b2, "ROOT", root), patch.object(b1, "ROOT", root):
                loaded = b2.load_dev(release, expected_count=2)
                self.assertEqual([case["id"] for case in loaded], ["second", "first"])
                self.assertTrue(all("_execution" in case for case in loaded))
                train, _ = b2._load_split(release, "train")
                self.assertEqual([case["id"] for case in train], ["second", "first"])
                with self.assertRaisesRegex(ValueError, "full 289-case"):
                    b2.load_dev(release)
                groups.write_text(json.dumps({"dev-source": "train"}))
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    b2.load_dev(release, expected_count=2)
                manifest["files_sha256"][groups.name] = digest(groups)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "registered exposed split"):
                    b2.load_dev(release, expected_count=2)
                with self.assertRaisesRegex(ValueError, "exposed train/dev"):
                    b2._load_split(release, "test")

    def test_removed_feedback_is_unscored_but_real_repeat_gate_survives(self):
        ledger = PassiveFeedbackLedger("document")
        ledger.propose("p", 1, self.document, self.bundle, CORRECTION)
        ledger.record("p", "shown", 2)
        ledger.record("p", "dismissed", 3)
        bundle = self.builder.build(self.document, "b", as_of=3, feedback=ledger)
        original = self.case(
            expected={"action": "no_action"}, bundle=bundle, ledger=ledger.dump()
        )
        preserved = copy.deepcopy(original)
        view = b2.condition_case(original, "context_only")
        self.assertEqual(original, preserved)
        self.assertEqual(view["input"]["recent_feedback"], [])
        self.assertIsNone(view["expectedOutput"])
        self.assertFalse(view["_b2"]["reference_comparable"])
        self.assertEqual(view["_execution"], original["_execution"])
        row = b1.evaluate_result(view, json.dumps(CORRECTION))
        self.assertFalse(row["displayed"])
        self.assertIsNotNone(row["suppression_reason"])
        self.assertIsNone(row["raw_grade"]["outcome_match"])
        report = b2.summarize_b2([view], [row])
        self.assertEqual(report["quality_unscored_cases"], 1)
        self.assertEqual(report["correct_abstentions"], 0)
        self.assertEqual(
            report["by_action_language"]["no_action|en"]["expected_requests"], 1
        )
        self.assertIn("dismissed:same", report["by_feedback_condition"])
        self.assertEqual(b2.condition_case(view, "context_only"), view)
        with self.assertRaisesRegex(ValueError, "original frozen case"):
            b2.condition_case(view, "full")

    def test_icl_messages_include_only_fixed_train_inputs_and_primary_answers(self):
        target = self.case()
        target["expectedOutput"] = {
            "action": "correct_text",
            "target": [1],
            "text": "SECRET_DEV_ANSWER",
        }
        target["metadata"]["decision_rationale"] = "SECRET_DEV_ANSWER"
        target["_execution"]["private"] = "SECRET_EXECUTION"
        demos = [
            self.case("train-keep", expected={"action": "no_action"}),
            self.case("train-edit"),
        ]
        for demo in demos:
            demo["metadata"].update(split="train", source_group_id="train-source")
            demo["metadata"]["review_reason"] = "SECRET_TRAIN_REVIEW"
        messages = b2.condition_messages(target, "icl", demos)
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )
        self.assertEqual(json.loads(messages[2]["content"]), {"action": "no_action"})
        self.assertEqual(json.loads(messages[4]["content"]), CORRECTION)
        self.assertEqual(messages[-1], canonical_messages(target["input"])[1])
        self.assertNotIn("SECRET_", json.dumps(messages))
        demos[0]["metadata"]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "train-only"):
            b2.condition_messages(target, "icl", demos)

    def test_common_budget_intersection_is_symmetric_and_missing_counts_fail(self):
        counts = {
            "first": {
                "full": {"a": 100, "b": 200, "c": 300},
                "icl": {"a": 200, "b": 3585, "c": 400},
            },
            "second": {
                "full": {"a": 100, "b": 200, "c": 3585},
                "icl": {"a": 200, "b": 300, "c": 400},
            },
        }
        result = b2.common_input_eligibility(["c", "a", "b"], counts)
        self.assertEqual(result["eligible_case_ids"], ["a"])
        self.assertEqual([row["case_id"] for row in result["excluded"]], ["c", "b"])
        self.assertEqual(result["registered_requests"], 3)
        del counts["second"]["icl"]["c"]
        with self.assertRaisesRegex(ValueError, "every registered case"):
            b2.common_input_eligibility(["a", "b", "c"], counts)

    def test_unknown_quality_and_all_request_failures_have_distinct_denominators(self):
        cases = [
            self.case("correct"),
            self.case("unknown"),
            self.case("timeout"),
            self.case("missing"),
        ]
        cases[1]["input"] = copy.deepcopy(cases[1]["input"])
        cases[1]["input"]["recent_edits"] = [
            {
                "action": "correct_text",
                "origin": "synthetic",
                "before": "requier",
                "after": "require",
            }
        ]
        cases = [b2.condition_case(case, "context_only") for case in cases]
        rows = [
            dict(b1.evaluate_result(case, json.dumps(CORRECTION)), complete_json_ms=10)
            for case in cases[:2]
        ]
        rows.append(
            {
                "case_id": "timeout",
                "error_type": "timeout",
                "error": "deadline",
                "complete_json_ms": 1000,
            }
        )
        report = b2.summarize_b2(cases, rows)
        self.assertEqual(report["expected_requests"], 4)
        self.assertEqual(report["quality_scored_cases"], 3)
        self.assertEqual(report["quality_unscored_cases"], 1)
        self.assertEqual(report["displayed_unknown"], 1)
        self.assertEqual(report["displayed_useful_precision"], 1)
        self.assertEqual(report["useful_edit_recall"], 1 / 3)
        self.assertEqual(report["all_request_p95_ms"], 1000)
        self.assertEqual(report["missing_latency_samples"], 1)
        self.assertEqual(report["errors_by_type"], {"timeout": 1, "missing_result": 1})

    def test_paired_reports_keep_source_clusters_failures_and_comparability(self):
        cases = [self.case("a"), self.case("b"), self.case("c")]
        cases[2]["metadata"]["source_group_id"] = "second-source"
        cases[1]["input"] = copy.deepcopy(cases[1]["input"])
        cases[1]["input"]["recent_edits"] = [
            {
                "action": "correct_text",
                "origin": "synthetic",
                "before": "requier",
                "after": "require",
            }
        ]
        full = [b2.condition_case(case, "full") for case in cases]
        ablated = [b2.condition_case(case, "context_only") for case in cases]
        left = [b1.evaluate_result(case, json.dumps(CORRECTION)) for case in full[:2]]
        right = [
            b1.evaluate_result(ablated[0], '{"action":"no_action"}'),
            b1.evaluate_result(ablated[2], json.dumps(CORRECTION)),
        ]
        result = b2.compare_runs(
            cases,
            {"left": left, "right": right},
            cases_by_name={"left": full, "right": ablated},
        )
        pair = result["pairs"]["left__vs__right"]
        self.assertEqual(pair["registered_requests"], 3)
        self.assertEqual(pair["common_reference_cases"], 2)
        self.assertEqual(pair["by_source"]["dev-source"]["left_only_match"], 1)
        self.assertEqual(pair["by_source"]["second-source"]["right_only_match"], 1)
        self.assertEqual(pair["source_macro_agreement_delta"], 0)
        self.assertEqual(pair["leave_one_source_out_macro_delta_range"], [-1, 1])
        self.assertEqual(result["runs"]["left"]["missing_case_ids"], ["c"])
        self.assertEqual(result["runs"]["right"]["quality_unscored_case_ids"], ["b"])
        with self.assertRaisesRegex(ValueError, "same registered case order"):
            b2.compare_runs(
                cases, {"left": left}, cases_by_name={"left": list(reversed(full))}
            )

    def test_sequence_report_orders_checkpoints_without_counting_new_requests(self):
        cases = [self.case("later"), self.case("earlier")]
        for case, step in zip(cases, (1, 0), strict=True):
            case["metadata"].update(episode_id="episode", episode_step=step)
        rows = [b1.evaluate_result(case, json.dumps(CORRECTION)) for case in cases]
        result = b2.sequence_report(cases, rows)
        episode = result["episodes"]["episode"]
        self.assertEqual(episode["case_ids"], ["earlier", "later"])
        self.assertEqual(episode["expected_requests"], 2)
        self.assertTrue(episode["all_checkpoint_outcomes_match"])
        self.assertIn("human_time_savings", result["unmeasured"])
        cases[0]["metadata"]["episode_step"] = 0
        with self.assertRaisesRegex(ValueError, "unique episode steps"):
            b2.sequence_report(cases, rows)

    def test_input_exclusions_and_reused_quality_are_not_fresh_model_failures(self):
        cases = [self.case("reused"), self.case("overflow"), self.case("missing")]
        eligible = {
            "registered_requests": 3,
            "eligible_case_ids": ["reused", "missing"],
            "excluded": [
                {"case_id": "overflow", "reason": "common_input_budget_exceeded"}
            ],
        }
        reused = {
            **b1.evaluate_result(cases[0], json.dumps(CORRECTION)),
            "complete_json_ms": 123,
            "inherited_complete_json_ms": 99,
            "reused_from": "frozen-full-run",
        }
        result = b2.summarize_b2(cases, [reused], eligibility=eligible)
        self.assertEqual(result["registered_requests"], 3)
        self.assertEqual(result["quality_scored_cases"], 2)
        self.assertEqual(result["input_budget_excluded_case_ids"], ["overflow"])
        self.assertEqual(result["missing_case_ids"], ["missing"])
        self.assertEqual(result["invalid_results"], 0)
        self.assertEqual(result["reused_quality_rows"], 1)
        self.assertEqual(result["expected_fresh_requests"], 1)
        self.assertEqual(result["observed_fresh_results"], 0)
        self.assertEqual(result["latency_samples"], 0)
        self.assertIsNone(result["all_request_p50_ms"])
        self.assertEqual(result["errors_by_type"], {"missing_result": 1})
        self.assertEqual(reused["complete_json_ms"], 123)
        with self.assertRaisesRegex(ValueError, "observed prior result"):
            b2.summarize_b2(
                cases,
                [{**reused, "error_type": "missing_result"}],
                eligibility=eligible,
            )
        with self.assertRaisesRegex(ValueError, "must not contain a model result"):
            b2.summarize_b2(
                cases, [reused, {"case_id": "overflow"}], eligibility=eligible
            )


if __name__ == "__main__":
    unittest.main()
