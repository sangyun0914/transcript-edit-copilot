import copy
import fcntl
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import probe_passive_gemma_caption_prompt as runner
import test_probe_passive_gemma_matched as matched_tests


class CaptionPromptTests(unittest.TestCase):
    def case(self, identity="dev-case", split="dev"):
        case = matched_tests.GemmaMatchedProbeTests().case()
        case["id"] = identity
        case["metadata"]["split"] = split
        return case

    def response(self, text, **kwargs):
        response = matched_tests.GemmaMatchedProbeTests().response(text, **kwargs)
        response["timings"] = {"cache_n": 0, "prompt_ms": 6, "predicted_ms": 4}
        return response

    def bundle(self, directory):
        cases = {"dev": [self.case()], "diagnostic": [self.case("train-case", "train")]}
        original = runner.canonical_messages(cases["dev"][0]["input"])[0]["content"]
        prompts = {
            "original": original,
            "candidate": "Require an observable grouping defect and a compliant result.",
        }
        config = {
            "prompt_variants": {},
            "cohorts": {},
            "payloads": {name: {} for name in prompts},
        }
        for variant, prompt in prompts.items():
            config["prompt_variants"][variant] = f"{variant}-prompt.txt"
            (directory / f"{variant}-prompt.txt").write_text(prompt)
        original_prefixes = {}
        for cohort, cohort_cases in cases.items():
            identities = [case["id"] for case in cohort_cases]
            config["cohorts"][cohort] = {
                "roster": f"{cohort}-roster.json",
                "case_count": len(identities),
                "case_ids": identities,
            }
            roster = []
            for case in cohort_cases:
                user = runner.canonical_messages(case["input"])[1]["content"]
                roster.append(
                    {
                        "case_id": case["id"],
                        **case["metadata"],
                        "user_sha256": runner.text_hash(user),
                    }
                )
            runner.write_new(
                directory / f"{cohort}-roster.json",
                {
                    "split": cohort_cases[0]["metadata"]["split"],
                    "case_ids": identities,
                    "cases": roster,
                },
            )
            original_prefixes[cohort] = {identity: [1, 2, 3] for identity in identities}
            for variant in prompts:
                names = {
                    "messages": f"{variant}-{cohort}-messages.jsonl",
                    "prefixes": f"{variant}-{cohort}-prefixes.jsonl.gz",
                }
                config["payloads"][variant][cohort] = names
                with (
                    (directory / names["messages"]).open("w") as message_file,
                    gzip.open(directory / names["prefixes"], "wt") as prefix_file,
                ):
                    for case in cohort_cases:
                        messages = runner.canonical_messages(case["input"])
                        messages[0]["content"] = prompts[variant]
                        tokens = [1, 2, 3] if variant == "original" else [1, 7]
                        message_file.write(
                            json.dumps(
                                {
                                    "case_id": case["id"],
                                    "messages": messages,
                                    "messages_sha256": runner.json_hash(messages),
                                    "user_sha256": runner.text_hash(
                                        messages[1]["content"]
                                    ),
                                }
                            )
                            + "\n"
                        )
                        prefix_file.write(
                            json.dumps(
                                {
                                    "case_id": case["id"],
                                    "input_ids": tokens,
                                    "input_tokens": len(tokens),
                                    "rendered_prompt": variant + " rendered",
                                    "messages_sha256": runner.json_hash(messages),
                                    "prefix_sha256": runner.json_hash(tokens),
                                }
                            )
                            + "\n"
                        )
        return (
            {"configuration": config, "directory": directory},
            cases,
            original_prefixes,
        )

    def test_only_system_prompt_can_change_and_both_cohorts_retain_frozen_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle, cases, prefixes = self.bundle(directory)
            views = runner.validate_views(bundle, cases, prefixes)
            self.assertEqual(views["original"]["dev"]["prefixes"], prefixes["dev"])
            self.assertEqual(
                views["candidate"]["diagnostic"]["prefixes"]["train-case"], [1, 7]
            )
            name = bundle["configuration"]["payloads"]["candidate"]["dev"]["messages"]
            path = directory / name
            row = json.loads(path.read_text())
            for change in ("user", "assistant", "system"):
                value = copy.deepcopy(row)
                if change == "user":
                    value["messages"][1]["content"] += "leaked reference"
                elif change == "assistant":
                    value["messages"].append({"role": "assistant", "content": "{}"})
                else:
                    value["messages"][0]["content"] += "unregistered revision"
                value["messages_sha256"] = runner.json_hash(value["messages"])
                path.write_text(json.dumps(value) + "\n")
                with (
                    self.subTest(change=change),
                    self.assertRaisesRegex(ValueError, "Only the system"),
                ):
                    runner.validate_views(bundle, cases, prefixes)
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Original prefix"):
                runner.validate_views(
                    bundle, cases, {**prefixes, "dev": {"dev-case": [9, 2, 3]}}
                )

    def test_input_budget_is_a_gate_without_silent_truncation_or_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle, cases, prefixes = self.bundle(directory)
            path = (
                directory
                / bundle["configuration"]["payloads"]["candidate"]["dev"]["prefixes"]
            )
            row = runner.original.read_rows(path)[0]
            row.update(
                input_ids=[1] * 3585,
                input_tokens=3585,
                prefix_sha256=runner.json_hash([1] * 3585),
            )
            with gzip.open(path, "wt") as stream:
                stream.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "bounded token"):
                runner.validate_views(bundle, cases, prefixes)

    def test_isolated_constrained_capture_preserves_frozen_request_and_source_grading(
        self,
    ):
        case = self.case()
        messages = runner.canonical_messages(case["input"])
        prepared = {
            "messages": {case["id"]: messages},
            "prefixes": {case["id"]: [1, 2, 3]},
            "rendered": {case["id"]: "frozen"},
        }
        old_request, old_grader, old_template = (
            runner.probe.request_body,
            runner.probe.evaluate_result,
            runner.probe.template_check,
        )
        runtime = runner.isolated_runtime(prepared)
        body = runtime.request_body(case, "model", messages=messages)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 512)
        self.assertFalse(body["cache_prompt"])
        self.assertTrue(body["verbose"])
        self.assertTrue(body["return_tokens"])
        self.assertEqual(len(body["messages"]), 2)
        self.assertNotIn("expectedOutput", json.dumps(body))
        text = json.dumps(case["expectedOutput"])
        response = self.response(text)
        with patch.object(
            runner.probe.httpx.Client,
            "post",
            return_value=SimpleNamespace(json=lambda: response),
        ):
            with runtime.httpx.Client(base_url="http://unused.invalid") as client:
                client.post("/v1/chat/completions", json=body)
                result = runtime.evaluate_result(case, text, "stop")
        self.assertTrue(result["raw_valid"])
        self.assertTrue(result["raw_grade"]["outcome_match"])
        self.assertEqual(result["sampled_token_ids"], [9, 1])
        self.assertIs(runner.probe.request_body, old_request)
        self.assertIs(runner.probe.evaluate_result, old_grader)
        self.assertIs(runner.probe.template_check, old_template)

    def test_server_prefix_gate_requires_rendered_text_and_exact_ids(self):
        case = self.case()
        messages = runner.canonical_messages(case["input"])
        prepared = {
            "messages": {case["id"]: messages},
            "prefixes": {case["id"]: [1, 2, 3]},
            "rendered": {case["id"]: "frozen"},
        }
        for text, tokens, passed in (
            ("frozen", [1, 2, 3], True),
            ("different", [1, 2, 3], False),
            ("frozen", [1, 9, 3], False),
        ):
            fake = SimpleNamespace(
                template_check=lambda *args: {
                    "passed": tokens == [1, 2, 3],
                    "rendered_prompt": text,
                    "runtime_token_ids": tokens,
                    "frozen_tokens": 3,
                    "runtime_tokens": 3,
                    "within_budget": True,
                }
            )
            with patch.object(runner.evidence, "isolated_runtime", return_value=fake):
                runtime = runner.isolated_runtime(prepared)
            check = runtime.template_check(None, {"messages": messages}, [1, 2, 3])
            self.assertEqual(check["passed"], passed)
            row = {"case_id": case["id"], **check}
            runner.validate_template_checks([case], [row], prepared, completed=False)
            if passed:
                runner.validate_template_checks([case], [row], prepared, completed=True)
            else:
                with self.assertRaisesRegex(ValueError, "mismatched"):
                    runner.validate_template_checks(
                        [case], [row], prepared, completed=True
                    )
                with self.assertRaisesRegex(ValueError, "parity evidence"):
                    runner.validate_template_checks(
                        [case], [{**row, "passed": True}], prepared, completed=False
                    )

    def test_malformed_and_native_handoff_outputs_never_become_abstentions(self):
        case = self.case()
        case["expectedOutput"] = {"action": "no_action"}
        case["acceptableOutputs"] = []
        text = '{"action":"no_action"}'
        for terminal in (1, 106, 50):
            response = self.response(text, terminal=terminal)
            row = {
                **runner.evidence.grade_response(case, response, "json"),
                "response": response,
                "raw_text": text,
                "finish_reason": "stop",
                "http_response_ms": 10,
            }
            runner.validate_results([case], [row], {case["id"]: [1, 2, 3]})
            summary = runner.summarize([case], [row])
            self.assertEqual(summary["correct_abstentions"], int(terminal != 50))
            self.assertEqual(summary["tool_handoffs"], int(terminal == 50))
        failed = {
            "case_id": case["id"],
            "request_failed": True,
            "raw_valid": False,
            "displayed": False,
            "error_type": "request_failure",
            "response": [],
            "http_response_ms": 27,
        }
        runner.validate_results([case], [failed], {case["id"]: [1, 2, 3]})
        summary = runner.summarize([case], [failed])
        self.assertEqual(summary["correct_abstentions"], 0)
        self.assertEqual(summary["request_errors"], 1)
        self.assertEqual(summary["invalid_results"], 1)
        self.assertEqual(summary["fresh_http_response_ms_p50_p95"]["p95_ms"], 27)
        self.assertEqual(
            summary["server_prefill_decode_ms_p50_p95"]["prefill"][
                "unmeasured_requests"
            ],
            1,
        )
        for credit in (
            {"raw_valid": True},
            {"token_evidence_valid": True},
            {"raw_grade": {"outcome_match": True}},
        ):
            with self.assertRaisesRegex(ValueError, "Failed requests"):
                runner.validate_results(
                    [case], [{**failed, **credit}], {case["id"]: [1, 2, 3]}
                )

    def test_timing_census_keeps_failures_and_missing_measurements(self):
        rows = [
            {"response": {"timings": {"prompt_ms": 6, "predicted_ms": 4}}},
            {
                "request_failed": True,
                "response": {"timings": {"prompt_ms": 20, "predicted_ms": 2}},
            },
            {
                "response": {
                    "timings": {"prompt_ms": float("nan"), "predicted_ms": True}
                }
            },
            {"response": None},
        ]
        value = runner.server_timings(rows, 5)
        self.assertEqual(value["prefill"]["p95_ms"], 20)
        self.assertEqual(value["decode"]["p95_ms"], 4)
        self.assertEqual(value["prefill"]["measured_requests"], 2)
        self.assertEqual(value["prefill"]["unmeasured_requests"], 3)

    def test_stricter_preservation_floor_and_all_request_evidence_are_required(self):
        summary = {
            "run_status": "completed",
            "observed_results": 289,
            "request_errors": 0,
            "token_evidence_observations": 289,
            "correct_abstentions": 65,
            "displayed_useful_edits": 183,
            "displayed_useful_precision": 0.85,
            "invalid_results": 19,
        }
        summaries = {name: dict(summary) for name in runner.CONDITIONS}
        summaries["base-candidate"]["correct_abstentions"] = 50
        summaries["sft-candidate"]["correct_abstentions"] = 60
        result = runner.decision(summaries, "dev")
        self.assertEqual(result["native_preservation_floor"], 65)
        self.assertEqual(result["status"], "failed")
        summaries["sft-candidate"]["correct_abstentions"] = 65
        self.assertEqual(runner.decision(summaries, "dev")["status"], "passed")
        self.assertEqual(
            runner.decision(summaries, "diagnostic")["status"], "not_assessable"
        )
        summaries["sft-original"]["token_evidence_observations"] = 288
        self.assertEqual(runner.decision(summaries, "dev")["status"], "not_assessable")

    def test_predictor_lock_blocks_concurrent_models(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(runner, "ROOT", Path(temporary)),
        ):
            (Path(temporary) / "artifacts").mkdir()
            with (Path(temporary) / "artifacts/nep-passive-b1.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "Another passive benchmark"):
                    with runner.predictor_lock():
                        self.fail("Predictor lock was bypassed")
            with runner.predictor_lock():
                pass

    def test_cli_defaults_to_prepare_and_rejects_nonfresh_source_pins(self):
        with (
            patch.object(runner, "prepare", return_value={}) as prepare,
            patch.object(runner, "run_condition") as run,
        ):
            runner.main(
                [
                    "run",
                    "--protocol",
                    "protocol.json",
                    "--condition",
                    "base-original",
                    "--output",
                    "out",
                ]
            )
        prepare.assert_called_once_with(Path("protocol.json"), "base-original", "dev")
        self.assertEqual(run.call_args.args[3], "prepare")
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary:
            path = Path(temporary) / "protocol.json"
            runner.write_new(
                path,
                {
                    "version": runner.VERSION,
                    "conditions": list(runner.CONDITIONS),
                    "registered_requests": 289,
                    "diagnostic_requests": 8,
                    "final_test_bodies_read": False,
                    "runtime": runner.POLICY,
                    "success_criteria": runner.SUCCESS_CRITERIA,
                    "evaluation_sources_sha256": {},
                },
            )
            with (
                patch.object(
                    runner, "source_paths", return_value=[Path(runner.__file__)]
                ),
                self.assertRaisesRegex(ValueError, "complete exact evaluator"),
            ):
                runner.prepare(path, "base-original")

    def test_report_retains_failed_and_missing_runs_and_rechecks_source_execution(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary:
            directory = Path(temporary)
            bundle = directory / "bundle"
            bundle.mkdir()
            protocol = directory / "protocol.json"
            preparation = bundle / "manifest.json"
            configuration = bundle / "configuration.json"
            registry_path = directory / "registry.json"
            tokenizer = {"id": "gemma26", "path": "native"}
            registry = {
                "runtime": {"sha256": "runtime"},
                "candidates": [{"id": "base"}, {"id": "sft"}],
            }
            for path, content in (
                (protocol, {}),
                (preparation, {}),
                (
                    configuration,
                    {
                        "tokenizer": {
                            **tokenizer,
                            "eos_token_id": 1,
                            "vocab_size": 262144,
                        }
                    },
                ),
                (registry_path, registry),
            ):
                runner.write_new(path, content)
            inputs = {
                path: runner.probe.digest(path)
                for path in (protocol, preparation, configuration, registry_path)
            }
            initial_inputs = dict(inputs)
            case = self.case()
            messages = runner.canonical_messages(case["input"])
            selected = {
                "messages": {case["id"]: messages},
                "prefixes": {case["id"]: [1, 2, 3]},
                "rendered": {case["id"]: "prompt"},
            }
            binding = {
                "condition": "base-original",
                "model_role": "native_base",
                "prompt_variant": "original",
                "registry_sha256": inputs[registry_path],
                "registered_case_ids": [case["id"]],
                "snapshot_sha256": {},
            }
            registration = {
                "protocol_path": protocol,
                "preparation_path": preparation,
                "bundle": {
                    "directory": bundle,
                    "configuration": runner.load_json(configuration),
                },
                "inputs": inputs,
                "registry_path": registry_path,
                "release": directory / "release",
                "cases": [case],
                "binding": binding,
                "views": {
                    variant: {"dev": selected} for variant in ("original", "candidate")
                },
                "t1": {
                    "registry": registry,
                    "execution": {
                        "candidate_ids": {"native_base": "base", "sft": "sft"}
                    },
                },
            }
            runs = directory / "runs"
            for index, (condition, (role, variant)) in enumerate(
                runner.CONDITIONS.items()
            ):
                if condition == "sft-original":
                    continue
                output = runs / condition
                (output / "implementation").mkdir(parents=True)
                (output / "implementation/model-registry.json").write_bytes(
                    registry_path.read_bytes()
                )
                expected = {
                    **binding,
                    "condition": condition,
                    "model_role": role,
                    "prompt_variant": variant,
                    "messages_sha256": {case["id"]: runner.json_hash(messages)},
                    "prefixes_sha256": {case["id"]: runner.json_hash([1, 2, 3])},
                    "rendered_prompts_sha256": {case["id"]: runner.text_hash("prompt")},
                }
                text = json.dumps(case["expectedOutput"])
                response = self.response(text)
                row = {
                    **runner.evidence.grade_response(case, response, "json"),
                    "raw_text": text,
                    "finish_reason": "stop",
                    "response": response,
                    "http_response_ms": 10,
                }
                if condition == "base-candidate":
                    row = {
                        "case_id": case["id"],
                        "raw_valid": False,
                        "displayed": False,
                        "request_failed": True,
                        "error_type": "request_failure",
                        "http_response_ms": 40,
                        "response": [],
                    }
                model = registry["candidates"][int(role == "sft")]
                manifest = {
                    "benchmark": expected,
                    "candidate": model["id"],
                    "candidate_binding": model,
                    "runtime_binding": registry["runtime"],
                    "tokenizer_binding": tokenizer,
                    "registry_sha256": inputs[registry_path],
                    "stage": "screen",
                    "case_ids": [case["id"]],
                    "sampling": runner.probe.sampling_options(registry_path),
                    "schema_constrained": True,
                    "cache_requested": False,
                    "context_tokens": 4096,
                    "reused_results": 0,
                    "completed_requests": 1,
                    "status": "completed",
                    "started_at": f"{index}:00",
                    "finished_at": f"{index}:01",
                }
                runner.write_new(output / "manifest.json", manifest)
                (output / "results.jsonl").write_text(json.dumps(row) + "\n")
                checks = {
                    "case_id": case["id"],
                    "passed": True,
                    "within_budget": True,
                    "runtime_token_ids": [1, 2, 3],
                    "rendered_prompt": "prompt",
                    "rendered_prompt_matches_frozen": True,
                    "frozen_tokens": 3,
                    "runtime_tokens": 3,
                }
                (output / "template-checks.jsonl").write_text(json.dumps(checks) + "\n")
                execution = list(runner.shared.source_execution([case], [row]))
                (output / "source-execution.jsonl").write_text(
                    "".join(json.dumps(value) + "\n" for value in execution)
                )
                files = {
                    str(path.relative_to(output)): runner.probe.digest(path)
                    for path in output.rglob("*")
                    if path.is_file()
                }
                runner.write_new(
                    output / "caption-prompt-receipt.json",
                    {**expected, "status": "completed", "files_sha256": files},
                )
            with patch.object(runner, "prepare", return_value=registration):
                runner.report(protocol, runs, directory / "report")
                summary = runner.load_json(directory / "report/summary.json")
                self.assertEqual(summary["runs"]["base-candidate"]["request_errors"], 1)
                self.assertEqual(summary["runs"]["sft-original"]["missing_requests"], 1)
                self.assertEqual(summary["decision"]["status"], "not_assessable")
                source_path = runs / "base-original/source-execution.jsonl"
                source_path.write_text("{}\n")
                receipt_path = source_path.parent / "caption-prompt-receipt.json"
                receipt = runner.load_json(receipt_path)
                receipt["files_sha256"]["source-execution.jsonl"] = runner.probe.digest(
                    source_path
                )
                receipt_path.write_text(json.dumps(receipt))
                registration["inputs"] = initial_inputs
                with self.assertRaisesRegex(ValueError, "source-execution"):
                    runner.report(protocol, runs, directory / "tampered-report")


if __name__ == "__main__":
    unittest.main()
