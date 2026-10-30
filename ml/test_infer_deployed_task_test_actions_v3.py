"""Synthetic sealed fixtures and mock HTTP only; no reserved TEST body or model."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import compare_deployed_task_test_actions_v3 as comparison
import infer_deployed_task_test_actions_v3 as deployed
import prepare_task_test_actions_v3 as core
import test_prepare_task_test_actions_v3 as fixtures


def capture(raw, *, status=200):
    text = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    result = {
        "elapsed_ms": 12.5,
        "http_status": status,
        "raw_text": text,
        "raw_body_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "raw": raw,
    }
    if status != 200:
        result.update(error=f"HTTP {status}", error_type="http_error")
    return result


class DeployedSealedTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SealedTaskTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        model = self.root / "synthetic.gguf"
        model.write_bytes(b"Synthetic identity only; never loaded.")
        self.gateway = {
            "endpoint": "http://127.0.0.1:1",
            "model_id": "synthetic-deployed-q8",
            "model_path": str(model),
            "chat_template_path": str(self.root / "tokenizer/chat_template.jinja"),
            "runtime_build": "synthetic-build",
            "cache_prompt": True,
        }
        selected = self.fixture.freeze["models"]["sft"]
        self.policy = {
            "selected_native_run_id": "sft",
            "native_core_registration": self.fixture.evidence,
            "protocol_id": "synthetic-only",
            "selected_native_model": selected,
            "selected_native_identity": core.read_json(Path(selected["path"]))[
                "identity"
            ],
            "model_identity": {
                "runtime_family": "small-actions-final-gguf",
                "role": "deployed_selected",
                "model_id": self.gateway["model_id"],
                "model_sha256": core.sha256(model),
                "adapter_sha256": None,
                "engineering_fixture_only": True,
            },
            "source_sha256": deployed.sources(),
            "numerical_regime": "native_bf16_merged_q8_0",
            "context_tokens": 4096,
            "request_deadline_seconds": 30,
            "request_settings": deployed.request_settings(self.gateway),
        }
        self.fixture.freeze.update(
            deployment_evaluation_policy=self.fixture.save(
                "synthetic-deployed-policy.json", self.policy
            ),
            registration=self.fixture.evidence,
            protocol_id="synthetic-only",
        )
        self.fixture.replace_freeze()
        validation = mock.patch.object(
            deployed, "validate_policy", return_value=self.gateway
        )
        validation.start()
        self.addCleanup(validation.stop)
        self.sealed = self.fixture.seal(self.fixture.prepare())
        self.context = deployed.load_authorized(self.fixture.freeze_path, self.sealed)
        self.requests = self.context["requests"]
        self.calls = []
        self.behavior = {}

    def transport(self, endpoint, route, body=None, timeout=120):
        self.assertEqual(endpoint, self.gateway["endpoint"])
        self.assertEqual(timeout, 30)
        self.calls.append((route, copy.deepcopy(body)))
        if route == "/health":
            return capture({"status": "ok"})
        if route == "/props":
            return capture(
                {
                    "model_alias": getattr(
                        self, "runtime_alias", self.gateway["model_id"]
                    ),
                    "model_path": self.gateway["model_path"],
                    "build_info": self.gateway["runtime_build"],
                    "total_slots": 1,
                    "default_generation_settings": {
                        "n_ctx": 4096,
                        "params": {
                            "reasoning_format": "none",
                            "reasoning_in_content": False,
                        },
                    },
                    "chat_template": Path(
                        self.gateway["chat_template_path"]
                    ).read_text(),
                }
            )
        if route == "/lora-adapters":
            return capture([])
        messages = (
            json.loads(body["content"]) if route == "/tokenize" else body["messages"]
        )
        row = next(row for row in self.requests if row["messages"] == messages)
        behavior = self.behavior.get(row["case_id"])
        if route == "/apply-template":
            return capture({"prompt": json.dumps(messages)})
        if route == "/tokenize":
            tokens = list(row["prefix_ids"])
            if behavior == "prefix_mismatch":
                tokens[-1] += 1
            return capture({"tokens": tokens})
        self.assertEqual(route, "/v1/chat/completions")
        if behavior == "http_failure":
            return capture({"error": "synthetic unavailable"}, status=503)
        if behavior == "interrupt":
            raise KeyboardInterrupt("synthetic interruption")
        prediction = (
            {"action": "correct_text", "target": [1], "text": "we"}
            if row["case_id"] == "engineering-edit"
            else {"action": "no_action"}
        )
        return capture(
            {
                "model": self.gateway["model_id"],
                "choices": [
                    {
                        "message": {"content": json.dumps(prediction)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 10},
            }
        )

    def execute(self, name="raw"):
        output = self.root / name
        with (
            mock.patch.object(deployed.http, "request", side_effect=self.transport),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            deployed.run(self.fixture.freeze_path, self.sealed, output)
        return output

    def admit_and_grade(self, raw):
        admitted, graded = self.root / "admitted", self.root / "graded"
        deployed.admit(
            self.fixture.freeze_path,
            self.sealed,
            raw,
            admitted,
            tokenizer=self.fixture.tokenizer,
        )
        comparison.grade(
            self.fixture.freeze_path,
            self.sealed,
            admitted,
            graded,
            tokenizer=self.fixture.tokenizer,
        )
        return graded, core.read_json(graded / "report.json")

    def rewrite_rows(self, raw, values):
        path = raw / "observations.jsonl"
        path.write_text("".join(json.dumps(value) + "\n" for value in values))
        manifest = core.read_json(raw / "manifest.json")
        manifest["files_sha256"][path.name] = core.sha256(path)
        (raw / "manifest.json").write_text(json.dumps(manifest))

    def test_actual_http_outputs_are_admitted_graded_and_kept_distinct_from_native_time(
        self,
    ):
        raw = self.execute()
        _, report = self.admit_and_grade(raw)
        observations = core.rows(raw / "observations.jsonl")
        self.assertEqual(
            [row["case_id"] for row in observations],
            [row["case_id"] for row in self.requests],
        )
        self.assertTrue(
            all(
                row["split"] == "final_test" and row["run_id"] == "sft"
                for row in observations
            )
        )
        self.assertTrue(all(row["generation_ms"] is None for row in observations))
        self.assertEqual(report["summary"]["recall"]["value"], 1)
        self.assertEqual(report["summary"]["precision"]["value"], 1)
        self.assertEqual(report["http_latency"]["valid_displayable_proposal"]["n"], 1)
        self.assertEqual(report["http_latency"]["model_no_action"]["n"], 1)
        self.assertEqual(report["numerical_regime"], "native_bf16_merged_q8_0")
        self.assertTrue(report["engineering_fixture_only"])

    def test_parity_and_http_failures_remain_in_the_same_full_denominator(self):
        self.behavior = {
            "engineering-edit": "prefix_mismatch",
            "engineering-preserve": "http_failure",
        }
        raw = self.execute()
        _, report = self.admit_and_grade(raw)
        observations = core.rows(raw / "observations.jsonl")
        self.assertEqual(
            [row["generation_attempted"] for row in observations], [False, True]
        )
        self.assertEqual(
            sum(route == "/v1/chat/completions" for route, _ in self.calls), 1
        )
        self.assertEqual(report["failure_attribution"]["request_failures"], 2)
        self.assertEqual(report["http_latency"]["all_completed_http"]["n"], 0)
        self.assertEqual(report["summary"]["recall"]["value"], 0)
        self.assertTrue(all("text" not in row for row in observations))

    def test_interrupt_preserves_attempted_request_and_all_remaining_rows(self):
        self.behavior = {"engineering-edit": "interrupt"}
        raw = self.root / "raw"
        with self.assertRaises(KeyboardInterrupt):
            self.execute()
        _, report = self.admit_and_grade(raw)
        observations = core.rows(raw / "observations.jsonl")
        self.assertEqual(
            [row["generation_attempted"] for row in observations], [True, False]
        )
        self.assertEqual(
            [row["failure_stage"] for row in observations],
            ["generation", "aborted_before_request"],
        )
        self.assertEqual(report["failure_attribution"]["request_failures"], 2)

    def test_wrong_loaded_model_fails_the_entire_cohort_before_generation(self):
        self.runtime_alias = "another-model"
        raw = self.execute()
        _, report = self.admit_and_grade(raw)
        self.assertEqual(
            [route for route, _ in self.calls], ["/health", "/props", "/lora-adapters"]
        )
        self.assertEqual(
            report["failure_attribution"]["pre_generation_request_failures"], 2
        )
        self.assertTrue(
            all(
                row["model_identity_verified"] is False
                for row in core.rows(raw / "observations.jsonl")
            )
        )

    def test_rehashed_output_request_and_census_tampering_are_rejected(self):
        raw = self.execute()
        original = core.rows(raw / "observations.jsonl")
        mutations = []
        changed = copy.deepcopy(original)
        changed[0]["text"] = '{"action":"no_action"}'
        mutations.append(changed)
        changed = copy.deepcopy(original)
        changed[0]["http_evidence"]["completion"]["request"]["temperature"] = False
        mutations.append(changed)
        mutations.append(original[:-1])
        for values in mutations:
            with self.subTest(values=len(values)):
                self.rewrite_rows(raw, values)
                with self.assertRaises(ValueError):
                    deployed.verify_observations(
                        self.context, raw, tokenizer=self.fixture.tokenizer
                    )

    def test_approval_and_whole_cohort_overflow_fail_before_any_http(self):
        self.fixture.freeze["body_access_authorized"] = False
        self.fixture.replace_freeze()
        with mock.patch.object(deployed.http, "request") as request:
            with self.assertRaisesRegex(ValueError, "body access"):
                deployed.run(
                    self.fixture.freeze_path,
                    fixtures.ForbiddenBodyPath(),
                    self.root / "never",
                )
        request.assert_not_called()
        self.fixture.freeze["body_access_authorized"] = True
        self.fixture.replace_freeze()
        rows = copy.deepcopy(self.requests)
        rows[0]["prefix_ids"] = [2] * 3585
        rows[0]["prefix_ids_sha256"] = core.digest(rows[0]["prefix_ids"])
        gate = copy.deepcopy(self.context["gate"])
        gate["policies"]["runtime"]["tokenizer"]["runtime_family"] = (
            "small-actions-final-v1"
        )
        original_rows = core.rows
        request_path = self.sealed / "requests-sft.jsonl"
        with (
            mock.patch.object(core, "load_gate", return_value=gate),
            mock.patch.object(
                core,
                "rows",
                side_effect=lambda path: (
                    rows if Path(path) == request_path else original_rows(path)
                ),
            ),
            mock.patch.object(deployed.http, "request") as request,
        ):
            with self.assertRaisesRegex(ValueError, "Entire deployed cohort"):
                deployed.run(self.fixture.freeze_path, self.sealed, self.root / "never")
        request.assert_not_called()
        self.assertFalse((self.root / "never").exists())

    def test_auxiliary_append_keeps_native_comparison_and_exact_selected_requests(self):
        self.behavior = {"engineering-preserve": "http_failure"}
        raw = self.execute()
        graded, report = self.admit_and_grade(raw)
        records = {row["case_id"]: row for row in core.rows(graded / "graded.jsonl")}
        arms = {
            run["role"]: {
                "run": run,
                "requests": {row["case_id"]: row for row in self.requests},
                "records": records,
            }
            for run in self.fixture.freeze["runs"]
        }
        native = {
            "freeze": self.context["gate"]["binding"],
            "sealed_manifest": self.context["sealed_binding"],
            "cohort_keys": {"canonical": "full_cohort_all_roles"},
            "runs": {role: {"full_issued": report["summary"]} for role in arms},
        }
        with (
            mock.patch.object(
                core, "load_tokenizer", return_value=self.fixture.tokenizer
            ),
            mock.patch.object(
                comparison.native_comparison,
                "build_comparison",
                return_value=(native, arms),
            ),
        ):
            output = self.root / "combined"
            comparison.aggregate(
                self.fixture.freeze_path, self.sealed, graded, [], output
            )
            actual = core.read_json(output / "comparison.json")
            self.assertEqual(len(actual["rows"]), 7)
            self.assertEqual(actual["native_core_comparison"], native)
            self.assertEqual(actual["rows"][-1]["role"], "deployed_selected")
            self.assertFalse(actual["causal_native_comparisons_include_deployed_gguf"])
            pair = actual["selected_native_vs_deployed"]
            self.assertEqual(pair["cases"], 2)
            self.assertEqual(
                pair["case_ids"], [row["case_id"] for row in self.requests]
            )
            self.assertEqual(
                pair["paired_outcomes"]["both_correct"]["case_ids"],
                ["engineering-edit"],
            )
            self.assertEqual(
                pair["paired_outcomes"]["both_wrong"]["case_ids"],
                ["engineering-preserve"],
            )
            self.assertEqual(
                sum(value["count"] for value in pair["paired_outcomes"].values()),
                pair["cases"],
            )
            self.assertEqual(
                pair["metric_deltas"]["outcome_accuracy"]["product_minus_native"], 0
            )

            native_observations = core.rows(raw / "observations.jsonl")
            for observation in native_observations:
                observation.update(
                    status="completed",
                    text='{"action":"no_action"}',
                    finish_reason="stop",
                )
            native_records = comparison.grade_rows(
                self.context, self.sealed, native_observations
            )
            arms["sft"]["records"] = {row["case_id"]: row for row in native_records}
            native["runs"]["sft"]["full_issued"] = (
                comparison.native_comparison.summarize(native_records)
            )
            crossed = self.root / "crossed-pair"
            comparison.aggregate(
                self.fixture.freeze_path, self.sealed, graded, [], crossed
            )
            actual = core.read_json(crossed / "comparison.json")
            self.assertEqual(actual["native_core_comparison"], native)
            pair = actual["selected_native_vs_deployed"]
            self.assertEqual(
                {
                    name: value["count"]
                    for name, value in pair["paired_outcomes"].items()
                },
                {
                    "both_correct": 0,
                    "native_only": 1,
                    "product_only": 1,
                    "both_wrong": 0,
                },
            )
            self.assertEqual(
                pair["paired_outcomes"]["native_only"]["case_ids"],
                ["engineering-preserve"],
            )
            self.assertEqual(
                pair["paired_outcomes"]["product_only"]["case_ids"],
                ["engineering-edit"],
            )
            self.assertEqual(
                sum(
                    value["count"]
                    for group in pair["by_source_group"].values()
                    for value in group.values()
                ),
                2,
            )
            self.assertEqual(
                pair["counts"]["native"]["requests"],
                pair["counts"]["product"]["requests"],
            )
            self.assertEqual(pair["metric_deltas"]["recall"]["product_minus_native"], 1)
            self.assertEqual(
                pair["metric_deltas"]["model_preservation"]["product_minus_native"], -1
            )
            self.assertEqual(
                pair["metric_deltas"]["validity"]["product_minus_native"], -0.5
            )
            self.assertIsNone(
                pair["metric_deltas"]["precision"]["product_minus_native"]
            )
            arms["sft"]["requests"] = {
                key: {**value, "messages_sha256": "altered"}
                for key, value in arms["sft"]["requests"].items()
            }
            with self.assertRaisesRegex(ValueError, "exact frozen requests"):
                comparison.aggregate(
                    self.fixture.freeze_path,
                    self.sealed,
                    graded,
                    [],
                    self.root / "invalid-combined",
                )

    def test_prefix_and_capture_verification_preserve_json_types(self):
        self.assertFalse(deployed.same_prefix([True, 2], [1, 2]))
        value = capture({"tokens": [1, 2]})
        value["raw"]["tokens"][0] = True
        with self.assertRaisesRegex(ValueError, "Parsed HTTP response"):
            deployed.verify_http_capture(value)


if __name__ == "__main__":
    unittest.main()
