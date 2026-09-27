import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_passive_curriculum as runner
from passive_benchmark import evaluate_result
from test_passive_contract import builder, document


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": runner.probe.digest(path)}


class CurriculumProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for module in (runner, runner.probe):
            mocked = patch.object(module, "ROOT", self.root)
            mocked.start()
            self.addCleanup(mocked.stop)
        native = self.root / "native"
        native_files = {}
        for name, value in (
            ("config.json", {"text_config": {"eos_token_id": 40}}),
            ("tokenizer_config.json", {"eos_token": "<eos>"}),
            (
                "tokenizer.json",
                {"added_tokens": [{"id": 42, "content": "<eos>", "special": True}]},
            ),
            ("chat_template.jinja", "native"),
        ):
            native_files[name] = write(native / name, value)["sha256"]
        tokenizer = {
            "id": "qwen35-9b",
            "path": str(native),
            "files_sha256": native_files,
        }
        token_registry = write(
            self.root / "tokenizers.json", {"candidates": [tokenizer]}
        )
        server = write(self.root / "server", {"mock": "binary"})
        self.anchor = {
            "id": "qwen35-9b",
            "tokenizer_id": "qwen35-9b",
            "base_repository": runner.BASE_REPOSITORY,
            "base_revision": runner.BASE_REVISION,
            "tokenizer_path": str(native),
            "tokenizer_files_sha256": native_files,
            "template_path": str(native / "chat_template.jinja"),
            "template_sha256": native_files["chat_template.jinja"],
        }
        self.old_registry = {
            "runtime": {
                "revision": runner.RUNTIME_REVISION,
                "server": server["path"],
                "sha256": server["sha256"],
                "files_sha256": {server["path"]: server["sha256"]},
            },
            "tokenizers": {
                "registry_path": token_registry["path"],
                "registry_sha256": token_registry["sha256"],
            },
            "candidates": [self.anchor],
            "sampling": {"temperature": 0, "max_tokens": 512},
        }
        release = write(self.root / "release/manifest.json", {})
        old_protocol = write(
            self.root / "b2/protocol.json",
            {"release_manifest_sha256": release["sha256"]},
        )
        self.eligibility = {
            "registered_requests": 289,
            "eligible_case_ids": [f"case-{index}" for index in range(100)],
            "excluded": [{"case_id": f"case-{index}"} for index in range(100, 289)],
        }
        eligibility = write(self.root / "b2/prefixes/common.json", self.eligibility)
        prefixes = write(
            self.root / "b2/prefixes/manifest.json",
            {
                "common_icl_eligibility_file": "common.json",
                "common_icl_eligibility_sha256": eligibility["sha256"],
            },
        )
        self.protocol = {
            "version": runner.VERSION,
            "conditions": list(runner.CONDITIONS),
            "registered_requests": 289,
            "icl_eligible_requests": 100,
            "icl_excluded_requests": 189,
            "final_test_bodies_read": False,
            "training_plan": {
                "candidate": "qwen35-9b",
                "view": "json-curriculum",
                "updates": 203,
                "draw_count": 809,
                "sampling_seed": "3407",
                "model_seed": 3407,
            },
            "b2": {
                "release_manifest": release,
                "protocol": old_protocol,
                "execution": write(self.root / "b2/execution.json", {}),
                "registry": write(self.root / "b2/registry.json", self.old_registry),
                "prefix_manifest": prefixes,
            },
            "external_comparators": [],
        }
        for index, (candidate, condition) in enumerate(
            (
                ("qwen35-9b", "full"),
                ("qwen35-9b", "icl"),
                ("always_no_action", "full"),
                ("consistent_history_rule", "full"),
            )
        ):
            self.protocol["external_comparators"].append(
                {
                    "id": f"external-{index}",
                    "candidate": candidate,
                    "condition": condition,
                    "manifest": write(
                        self.root / f"external-{index}/manifest.json", {}
                    ),
                    "results": write(self.root / f"external-{index}/results.jsonl", {}),
                }
            )
        self.protocol_path = self.root / "new/protocol.json"
        protocol_binding = write(self.protocol_path, self.protocol)
        self.registry = copy.deepcopy(self.old_registry)
        self.registry["candidates"] = []
        artifacts = {}
        for role, key in runner.ARTIFACT_KEYS.items():
            artifact = write(self.root / f"export/{key}.gguf", {"model_role": role})
            artifacts[key] = artifact
            self.registry["candidates"].append(
                {
                    **self.anchor,
                    "id": role,
                    "model_path": artifact["path"],
                    "model_sha256": artifact["sha256"],
                    "status": "ready",
                }
            )
        tensor = write(
            self.root / "export/tensor-verification.json",
            {
                "status": "passed",
                "template_sha256": self.anchor["template_sha256"],
                "eos_token_id": 42,
            },
        )
        training, config, reload_manifest = self.training_evidence(protocol_binding)
        self.export = {
            "version": "passive-sft-export-v1",
            "status": "tensor_verified_paired_export_runtime_preflight_pending",
            "base_repository": runner.BASE_REPOSITORY,
            "base_revision": runner.BASE_REVISION,
            "runtime_revision": runner.RUNTIME_REVISION,
            "artifacts": artifacts,
            "tensor_verification_sha256": tensor["sha256"],
            "training_record": training,
        }
        self.registry_path = self.root / "new/registry.json"
        self.execution_path = self.root / "new/execution.json"
        self.execution = {
            "version": runner.EXECUTION_VERSION,
            "protocol_sha256": protocol_binding["sha256"],
            "registry": write(self.registry_path, self.registry),
            "export_receipt": write(self.root / "export/receipt.json", self.export),
            "training_configuration": config,
            "fresh_reload": reload_manifest,
            "candidate_ids": {"native_base": "native_base", "sft": "sft"},
        }
        write(self.execution_path, self.execution)

    def training_evidence(self, protocol_binding):
        prepared_file = write(
            self.root / "preparation/configuration.json",
            {"training_view": "json-curriculum"},
        )
        preparation = write(
            self.root / "preparation/manifest.json",
            {
                "version": "passive-sft-curriculum-preparation-v1",
                "status": "frozen_train_only_preparation",
                "dev_test_record_bodies_copied": 0,
                "release_manifest_sha256": self.protocol["b2"]["release_manifest"][
                    "sha256"
                ],
                "files_sha256": {"configuration.json": prepared_file["sha256"]},
            },
        )
        configuration = {
            "version": "passive-sft-curriculum-v1",
            "candidate_id": "qwen35-9b",
            "generation": {"eos_token_id": 42},
            "base": {
                "repository": runner.BASE_REPOSITORY,
                "revision": runner.BASE_REVISION,
            },
            "bindings": {
                "evaluation_protocol": protocol_binding,
                "preparation": preparation,
            },
            "training": {
                "max_updates": 203,
                "draw_count": 809,
                "sample_order": ["train-case"] * 809,
                "sampling_seed": "3407",
                "seed": 3407,
                "fresh_native_initialization": True,
                "max_total_sequence_tokens": 8090,
                "max_total_completion_tokens": 809,
            },
        }
        sources = {}
        for filename, key in (
            ("trainer.py", "trainer_sha256"),
            ("train_passive_sft.py", "smoke_trainer_sha256"),
            ("prepare_passive_curriculum.py", "preparer_sha256"),
            ("prepare_passive_sft.py", "smoke_preparer_sha256"),
        ):
            binding = write(self.root / "training" / filename, {"source": filename})
            configuration[key] = binding["sha256"]
            sources[filename] = binding["sha256"]
        config = write(self.root / "input/configuration-attempt.json", configuration)
        train_files = {
            **sources,
            "configuration.json": write(
                self.root / "training/configuration.json", configuration
            )["sha256"],
        }
        self.training = {
            "version": "passive-sft-curriculum-v1",
            "status": "training_complete_fresh_reload_pending",
            "mode": "train",
            "training_updates": 203,
            "optimizer_committed_rows": 809,
            "configuration_sha256": config["sha256"],
            "evaluation_protocol_sha256": protocol_binding["sha256"],
            "preparation_sha256": preparation["sha256"],
            "final_test_bodies_read": 0,
            "pid": 1,
        }
        train_run = write(self.root / "training/run.json", self.training)
        train_files["run.json"] = train_run["sha256"]
        train_files["exposure.json"] = write(
            self.root / "training/exposure.json",
            {
                "completed_updates": 203,
                "optimizer_commit_uncertain": False,
                "pending_backward_rows_not_committed": 0,
                "phases": {
                    "optimizer_committed": {
                        "rows": 809,
                        "counts": {"case_id": {"train-case": 809}},
                        "sequence_tokens": 8090,
                        "completion_tokens": 809,
                    }
                },
            },
        )["sha256"]
        generations = [
            {"case_id": f"diagnostic-{index}", "output_ids": [42]} for index in range(8)
        ]
        train_files["smoke-generation.json"] = write(
            self.root / "training/smoke-generation.json", generations
        )["sha256"]
        train_manifest = write(
            self.root / "training/manifest.json",
            {
                "version": "passive-sft-curriculum-v1",
                "status": "training_complete_fresh_reload_pending",
                "files_sha256": train_files,
            },
        )
        reload_files = {}
        for name, expected in sources.items():
            write(self.root / "reload" / name, {"source": name})
            reload_files[name] = expected
        values = {
            "configuration.json": configuration,
            "smoke-generation.json": generations,
            "run.json": {
                "status": "fresh_reload_passed_local_export_pending",
                "mode": "fresh_reload",
                "pid": 2,
                "configuration_sha256": config["sha256"],
                "evaluation_protocol_sha256": protocol_binding["sha256"],
                "reloaded_training_updates": 203,
                "previous_manifest_sha256": train_manifest["sha256"],
            },
            "reload-parity.json": {
                "adapter_tensors_identical": True,
                "cases": [
                    {"case_id": row["case_id"], "identical": True}
                    for row in generations
                ],
            },
        }
        for name, value in values.items():
            reload_files[name] = write(self.root / "reload" / name, value)["sha256"]
        reload_manifest = write(
            self.root / "reload/manifest.json",
            {
                "version": "passive-sft-curriculum-v1",
                "status": "fresh_reload_passed_local_export_pending",
                "files_sha256": reload_files,
            },
        )
        return train_run, config, reload_manifest

    def registration(self):
        return runner.load_registration(self.protocol_path, self.execution_path)

    def prepared(self, count=289):
        return {
            "cases": [{"id": f"case-{index}"} for index in range(count)],
            "messages": {},
            "prefixes": {},
            "source_files": [],
            "reused_results": {},
            "binding": {"snapshot_sha256": {}},
        }

    def test_protocol_requires_fixed_cohort_and_preserves_external_hash_checks(self):
        self.assertEqual(len(runner.load_protocol(self.protocol_path)["inputs"]), 14)
        for key, value in (
            ("icl_eligible_requests", 101),
            ("final_test_bodies_read", True),
            ("conditions", list(runner.CONDITIONS)[:-1]),
        ):
            with self.subTest(key=key):
                write(self.protocol_path, {**self.protocol, key: value})
                with self.assertRaises(ValueError):
                    runner.load_protocol(self.protocol_path)
        write(self.protocol_path, self.protocol)
        write(self.root / "external-0/results.jsonl", {"changed": True})
        with self.assertRaisesRegex(ValueError, "Changed bound input"):
            runner.load_protocol(self.protocol_path)

    def test_native_pair_cannot_be_swapped_or_change_runtime_or_template(self):
        registration = self.registration()
        runner.validate_pair(registration)
        self.assertEqual(registration["training"]["training_updates"], 203)
        changes = []
        swapped = copy.deepcopy(self.registry)
        swapped["candidates"][0].update(
            {
                key: swapped["candidates"][1][key]
                for key in ("model_path", "model_sha256")
            }
        )
        changes.append(swapped)
        changed = copy.deepcopy(self.registry)
        changed["runtime"]["revision"] = "different"
        changes.append(changed)
        changed = copy.deepcopy(self.registry)
        changed["candidates"][1]["template_sha256"] = "0" * 64
        changes.append(changed)
        for value in changes:
            write(self.registry_path, value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_pair(registration)
        write(self.registry_path, self.registry)
        Path(self.registry["candidates"][0]["model_path"]).write_text("changed bytes")
        with self.assertRaisesRegex(ValueError, "Pinned file differs"):
            runner.validate_pair(registration)

    def test_final_training_and_fresh_reload_are_required_for_this_protocol(self):
        registration = self.registration()
        path = self.root / "training/run.json"
        runner.validate_training(registration, path, self.training)
        for change in (
            {"training_updates": 2},
            {"optimizer_committed_rows": 808},
            {"status": "recovery_checkpoint_complete"},
            {"evaluation_protocol_sha256": "0" * 64},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "final 203-update"),
            ):
                runner.validate_training(
                    registration, path, {**self.training, **change}
                )
        write(
            self.root / "reload/reload-parity.json",
            {"adapter_tensors_identical": False, "cases": []},
        )
        with self.assertRaisesRegex(ValueError, "Changed bound input"):
            runner.validate_training(registration, path, self.training)

    def test_all_four_conditions_reuse_original_anchor_and_exact_prepared_messages(
        self,
    ):
        for condition, (_, view) in runner.CONDITIONS.items():
            prepared = self.prepared(100 if view == "icl" else 289)
            prepared["messages"] = {
                "case-0": [{"role": "user", "content": "bound original B2 input"}]
            }
            with (
                self.subTest(condition=condition),
                patch.object(
                    runner.original, "prepare", return_value=prepared
                ) as original,
            ):
                registration = runner.prepare(
                    self.protocol_path, self.execution_path, condition
                )
                called = original.call_args.args[0]
                self.assertEqual(called.candidate, "qwen35-9b")
                self.assertEqual(called.condition, view)
                self.assertEqual(called.registry, self.root / "b2/registry.json")
                self.assertIs(
                    registration["prepared"]["messages"], prepared["messages"]
                )
                self.assertEqual(registration["binding"]["condition"], condition)
        with patch.object(runner.original, "prepare", return_value=self.prepared(99)):
            with self.assertRaisesRegex(ValueError, "denominator"):
                runner.prepare(self.protocol_path, self.execution_path, "sft-icl")

    def case_and_row(self):
        source = document()
        bundle = builder().build(source, "b", as_of=0)
        case = {
            "id": "case-0",
            "input": bundle["input"],
            "expectedOutput": {"action": "no_action"},
            "acceptableOutputs": [],
            "_execution": {
                "before": source,
                "bundle": bundle,
                "feedback_ledger": None,
                "as_of": 0,
            },
        }
        text = '{"action":"no_action"}'
        row = {
            **evaluate_result(case, text),
            "raw_text": text,
            "finish_reason": "stop",
            "response": {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3},
            },
        }
        return case, row

    def test_regrades_source_execution_and_never_treats_failures_as_abstentions(self):
        case, row = self.case_and_row()
        prefixes = {case["id"]: [1, 2, 3]}
        runner.validate_results([case], [row], prefixes)
        changed = copy.deepcopy(row)
        changed["raw_grade"]["outcome_match"] = False
        with self.assertRaisesRegex(ValueError, "quality differs"):
            runner.validate_results([case], [changed], prefixes)
        changed = copy.deepcopy(row)
        changed["response"]["usage"]["prompt_tokens"] = 4
        with self.assertRaisesRegex(ValueError, "frozen token prefix"):
            runner.validate_results([case], [changed], prefixes)
        failed = {
            "case_id": case["id"],
            "error_type": "timeout",
            "raw_valid": False,
            "displayed": False,
        }
        runner.validate_results([case], [failed], prefixes)
        runner.validate_results([case], [], prefixes)
        with self.assertRaisesRegex(ValueError, "cannot receive"):
            runner.validate_results([case], [{**failed, "raw_valid": True}], prefixes)
        evidence = list(runner.source_execution([case], [row]))[0]
        self.assertEqual(evidence["predicted_after"], evidence["execution"]["before"])

    def test_run_failure_preserves_write_once_receipt_and_execution_evidence(self):
        with patch.object(runner.original, "prepare", return_value=self.prepared()):
            registration = runner.prepare(
                self.protocol_path, self.execution_path, "sft-full"
            )
        case, _ = self.case_and_row()
        registration["prepared"].update(cases=[case], prefixes={case["id"]: [1, 2, 3]})
        output = self.root / "runs/sft-full"

        def failed(args, *, prepared):
            args.output.mkdir(parents=True)
            write(args.output / "manifest.json", {"status": "failed"})
            row = {
                "case_id": case["id"],
                "error_type": "timeout",
                "raw_valid": False,
                "displayed": False,
            }
            (args.output / "results.jsonl").write_text(json.dumps(row) + "\n")
            self.assertEqual(args.candidate, "sft")
            self.assertIs(prepared, registration["prepared"])
            raise RuntimeError("mock request timeout")

        with (
            patch.object(runner.probe, "run", side_effect=failed),
            self.assertRaisesRegex(RuntimeError, "timeout"),
        ):
            runner.run_condition(registration, "sft-full", output, "screen", "idle")
        receipt = runner.load_json(output / "curriculum-receipt.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("results.jsonl", receipt["files_sha256"])
        self.assertIn("source-execution.jsonl", receipt["files_sha256"])
        with self.assertRaisesRegex(ValueError, "write-once"):
            runner.run_condition(registration, "sft-full", output, "screen", "idle")

    def test_lock_is_acquired_before_preparation_or_server_start(self):
        path = self.root / "artifacts/nep-passive-b1.lock"
        path.parent.mkdir()
        with path.open("a") as lock, patch.object(runner, "prepare") as prepared:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "owns the local predictor"):
                runner.main(
                    [
                        "--protocol",
                        str(self.protocol_path),
                        "--execution",
                        str(self.execution_path),
                        "--condition",
                        "sft-full",
                        "--output",
                        str(self.root / "output"),
                    ]
                )
            prepared.assert_not_called()

    def test_learning_signal_is_descriptive_and_incomplete_runs_cannot_qualify(self):
        base = {
            "displayed_useful_precision": 0.5,
            "displayed_useful_edits": 5,
            "correct_abstentions": 4,
            "invalid_results": 2,
            "missing_requests": 0,
        }
        adapted = {**base, "displayed_useful_edits": 6}
        self.assertEqual(
            runner.learning_signal(base, adapted)["classification"],
            "descriptive_learning_signal",
        )
        for change in (
            {"displayed_useful_precision": 0.4},
            {"correct_abstentions": 3},
            {"invalid_results": 3},
            {"displayed_useful_edits": 5},
            {"displayed_useful_precision": None},
        ):
            self.assertEqual(
                runner.learning_signal(base, {**adapted, **change})["classification"],
                "criteria_not_met",
            )
        for change in ({"missing_requests": 1}, {"run_status": "failed"}):
            self.assertEqual(
                runner.learning_signal(base, {**adapted, **change})["classification"],
                "incomplete",
            )

    def test_report_keeps_all_778_requests_and_icl_exclusions_when_runs_are_missing(
        self,
    ):
        prepared = self.prepared()
        case, _ = self.case_and_row()
        cases = []
        for index in range(289):
            value = copy.deepcopy(case)
            value["id"] = f"case-{index}"
            value["metadata"] = {
                "source_group_id": "source",
                "language": "en",
                "episode_id": value["id"],
            }
            cases.append(value)
        prepared.update(cases=cases, summarize=runner.b2.summarize_b2)
        with patch.object(runner.original, "prepare", return_value=prepared):
            registration = runner.prepare(
                self.protocol_path, self.execution_path, "native-base-full"
            )
        icl = {
            **prepared,
            "cases": cases[:100],
            "summarize": lambda _, rows: runner.b2.summarize_b2(
                cases, rows, eligibility=self.eligibility
            ),
        }
        output = self.root / "report"
        with (
            patch.object(runner, "prepare", return_value=registration),
            patch.object(runner.original, "prepare", return_value=icl),
            patch.object(runner, "historical_results", return_value=({}, {})),
        ):
            runner.report(
                self.protocol_path,
                self.execution_path,
                self.root / "absent-runs",
                output,
            )
        report = runner.load_json(output / "summary.json")
        self.assertEqual(report["registered_new_requests"], 778)
        for name, (_, condition) in runner.CONDITIONS.items():
            summary = report["runs"][name]
            self.assertEqual(
                summary["missing_requests"], 100 if condition == "icl" else 289
            )
            self.assertEqual(
                summary["input_budget_excluded"], 189 if condition == "icl" else 0
            )
            self.assertEqual(summary["correct_preserves"], 0)
        self.assertEqual(
            report["paired"]["icl"]["learning_signal"]["classification"], "incomplete"
        )


if __name__ == "__main__":
    unittest.main()
