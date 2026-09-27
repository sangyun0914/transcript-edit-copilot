import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import passive_benchmark as benchmark
from passive_contract import PassiveFeedbackLedger, canonical_messages
from replay_session import session_hash
from session_cases import replayable_session
from test_passive_contract import CORRECTION, builder, document


class PassiveBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.document = document()
        self.builder = builder()
        self.bundle = self.builder.build(self.document, "b", as_of=0)

    def case(self, identity="case", expected=None, bundle=None, ledger=None):
        bundle = bundle or self.bundle
        return {
            "id": identity,
            "input": bundle["input"],
            "expectedOutput": CORRECTION if expected is None else expected,
            "acceptableOutputs": [],
            "metadata": {
                "split": "dev", "source_group_id": "authored-dev", "language": "en",
                "episode_id": identity, "release_stratum": "feedback_sessions",
                "document_before_hash": session_hash(self.document),
                "input_sha256": session_hash(canonical_messages(bundle["input"])),
                "context": bundle["context"],
            },
            "_execution": {"before": self.document, "bundle": bundle,
                           "feedback_ledger": ledger, "as_of": bundle["context"]["as_of"]},
        }

    def test_equivalent_span_and_acceptable_alternative_preserve_sources(self):
        case = self.case()
        equivalent = {"action": "correct_text", "target": [0, 1, 2],
                      "text": "We may require another review."}
        grade = benchmark.grade_prediction(case, equivalent)
        self.assertFalse(grade["exact_match"])
        self.assertTrue(grade["outcome_match"])
        self.assertEqual(grade["method"], "transcript_media_equivalence")
        alternative = {"action": "correct_text", "target": [2], "text": "another check."}
        case["acceptableOutputs"] = [alternative]
        self.assertEqual(benchmark.grade_prediction(case, alternative)["matched_reference_index"], 1)
        changed_meaning = {**equivalent, "text": "We require another review."}
        self.assertFalse(benchmark.grade_prediction(case, changed_meaning)["outcome_match"])
        with self.assertRaises(ValueError):
            benchmark.grade_prediction(case, {"action": "delete_span", "target": [1]})

    def test_repeat_suppression_and_invalid_output_never_become_correct_abstention(self):
        ledger = PassiveFeedbackLedger("doc")
        ledger.propose("p", 1, self.document, self.bundle, CORRECTION)
        ledger.record("p", "shown", 2)
        ledger.record("p", "dismissed", 3)
        current = self.builder.build(self.document, "b", as_of=3, feedback=ledger)
        case = self.case(expected={"action": "no_action"}, bundle=current, ledger=ledger.dump())
        row = benchmark.evaluate_result(case, json.dumps(CORRECTION))
        self.assertTrue(row["raw_valid"])
        self.assertFalse(row["raw_grade"]["outcome_match"])
        self.assertFalse(row["displayed"])
        self.assertIsNone(row["displayed_grade"])
        self.assertIsNotNone(row["suppression_reason"])
        report = benchmark.summarize([case], [row])
        self.assertEqual(report["correct_abstentions"], 0)
        self.assertEqual(report["suppressed"], 1)
        for value, finish, error_type in (("broken", "stop", "format_invalid"),
                                          (json.dumps({"action": "delete_span", "target": [1]}),
                                           "stop", "semantic_invalid"),
                                          ('{"action":"no_action"}', "length", "truncated")):
            with self.subTest(error_type=error_type):
                row = benchmark.evaluate_result(case, value, finish)
                self.assertFalse(row["raw_valid"])
                self.assertEqual(row["error_type"], error_type)
                self.assertEqual(benchmark.summarize([case], [row])["correct_abstentions"], 0)

    def test_missing_failures_and_all_response_latency_use_fixed_denominators(self):
        cases = [self.case("good"), self.case("timeout"), self.case("missing"),
                 self.case("unnecessary", expected={"action": "no_action"})]
        good = {**benchmark.evaluate_result(cases[0], json.dumps(CORRECTION)), "latency_ms": 10}
        timeout = {"case_id": "timeout", "error": "deadline", "error_type": "timeout", "latency_ms": 1000}
        unnecessary = {**benchmark.evaluate_result(cases[3], json.dumps(CORRECTION)), "latency_ms": 20}
        result = benchmark.summarize(cases, [good, timeout, unnecessary])
        self.assertEqual(result["expected_requests"], 4)
        self.assertEqual(result["missing_case_ids"], ["missing"])
        self.assertEqual(result["missed_useful_edits"], 2)
        self.assertEqual(result["displayed_unnecessary_edits"], 1)
        self.assertEqual(result["displayed_useful_precision"], 0.5)
        self.assertEqual(result["useful_edit_recall"], 1 / 3)
        self.assertEqual(result["all_request_p95_ms"], 1000)
        self.assertEqual(result["missing_latency_samples"], 1)
        self.assertEqual(result["errors_by_type"], {"timeout": 1, "missing_result": 1})
        self.assertEqual(result["by_action"]["correct_text"]["expected_requests"], 3)
        for rows in ([good, good], [{"case_id": "unregistered"}]):
            with self.assertRaises(ValueError):
                benchmark.summarize(cases, rows)

    def test_baselines_and_serving_messages_do_not_consume_reference_metadata(self):
        case = self.case()
        case["input"] = copy.deepcopy(case["input"])
        case["input"]["recent_edits"] = [{"action": "correct_text", "origin": "synthetic",
                                          "before": "requier", "after": "require"}]
        original = canonical_messages(case["input"])
        case["expectedOutput"] = {"action": "correct_text", "target": [2], "text": "SECRET_ANSWER"}
        case["metadata"]["decision_rationale"] = "SECRET_ANSWER"
        case["_execution"] = "SECRET_ANSWER"
        self.assertEqual(benchmark.baseline_prediction(case, "consistent_history_rule"), CORRECTION)
        self.assertEqual(benchmark.baseline_prediction(case, "always_no_action"), {"action": "no_action"})
        self.assertEqual(canonical_messages(case["input"]), original)
        self.assertNotIn("SECRET_ANSWER", json.dumps(original))

    def test_frozen_screen_order_hashes_and_message_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            release = root / "frozen"
            release.mkdir()
            cases = [self.case(f"case-{i:02d}") for i in range(36)]
            for case in cases:
                case.pop("_execution")
                case["reproduction"] = {"candidate_directory": ".", "episode": "episode.json", "step": 0}
            journal = replayable_session(self.document, self.document, "no_action", [])
            journal[0].pop("document")
            episode = {"spec": {"corruptions": []}, "source_document": "source.json", "journal": journal}
            source_path, episode_path = root / "source.json", root / "episode.json"
            source_path.write_text(json.dumps(self.document))
            episode_path.write_text(json.dumps(episode))
            protocol = {"version": "passive-b1-screen-v2", "request_budget": 36,
                        "selected_case_ids": [c["id"] for c in reversed(cases)],
                        "model_predictions_consulted": False}
            message_rows = [{"id": c["id"], "messages": canonical_messages(c["input"])} for c in cases]
            (release / "dev-cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases))
            message_path = release / "dev-messages.jsonl"
            message_path.write_text("\n".join(json.dumps(m) for m in message_rows))
            (release / "screening-protocol.json").write_text(json.dumps(protocol))
            def digest(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {"version": "passive-dataset-completion-v2", "status": "frozen_supported_scope",
                        "files_sha256": {p.name: digest(p) for p in release.iterdir()},
                        "dependencies_sha256": {p.name: digest(p) for p in (source_path, episode_path)}}
            manifest_path = release / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(benchmark, "ROOT", root):
                loaded = benchmark.load_screen(release)
                self.assertEqual([c["id"] for c in loaded], protocol["selected_case_ids"])
                self.assertTrue(all("_execution" in c for c in loaded))
                message_rows[0]["messages"][1]["content"] = "teacher answer"
                message_path.write_text("\n".join(json.dumps(m) for m in message_rows))
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    benchmark.load_screen(release)
                manifest["files_sha256"][message_path.name] = digest(message_path)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "serving input"):
                    benchmark.load_screen(release)


if __name__ == "__main__":
    unittest.main()
