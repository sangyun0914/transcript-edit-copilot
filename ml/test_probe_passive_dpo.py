import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import probe_passive_dpo as runner
from passive_benchmark import evaluate_result
from test_passive_contract import CORRECTION, builder, document


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": runner.probe.digest(path)}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return runner.probe.digest(path)


class DPOProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for module in (runner, runner.curriculum, runner.probe):
            mocked = patch.object(module, "ROOT", self.root)
            mocked.start()
            self.addCleanup(mocked.stop)

    def case(self, identity="case"):
        source = document()
        bundle = builder().build(source, "b", as_of=0)
        return {
            "id": identity,
            "input": bundle["input"],
            "expectedOutput": CORRECTION,
            "acceptableOutputs": [],
            "metadata": {
                "source_group_id": "source",
                "source_id": "source",
                "language": "en",
                "episode_id": "episode",
            },
            "_execution": {
                "before": source,
                "bundle": bundle,
                "feedback_ledger": None,
                "as_of": 0,
            },
        }

    def pair(self, index):
        return {
            "pair_id": f"pair-{index}",
            "case_id": f"case-{index}",
            "chosen": {"input_ids": [1, 2, 3, 4], "completion_start": 2},
            "rejected": {"input_ids": [1, 2, 5, 6, 4], "completion_start": 2},
            "metadata": {
                "source_group_id": "source",
                "source_id": "source",
                "episode_id": "episode",
                "language": "en",
                "chosen_action": "correct_text",
            },
        }

    def training_run(self, pairs):
        return {
            "mode": "train",
            "status": runner.TRAIN_COMPLETE,
            "experiment_phase": "substantive",
            "training_updates": math.ceil(len(pairs) / 4),
            "optimizer_committed_pairs": len(pairs),
            "pair_count": len(pairs),
            "pair_order": [pair["pair_id"] for pair in pairs],
            "initialization": "exact_final_T1_adapter",
            "reference_source": "exact_final_T1_adapter",
            "optimizer_state_resumed": False,
            "actual_tokenizer_parity": True,
            "changed_adapter_tensors": ["lora_B"],
        }

    def result(self, case, text=None):
        text = json.dumps(CORRECTION) if text is None else text
        return {
            **evaluate_result(case, text),
            "raw_text": text,
            "finish_reason": "stop",
            "response": {
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 50},
            },
            "complete_json_ms": 100,
        }

    def test_final_checkpoint_gate_rejects_profile_partial_and_reordered_pairs(self):
        pairs = [self.pair(index) for index in range(33)]
        run = self.training_run(pairs)
        runner.validate_final_run(run, pairs, "substantive")
        for change in (
            {"experiment_phase": "engineering_profile"},
            {"training_updates": 8},
            {"optimizer_committed_pairs": 32},
            {"pair_order": list(reversed(run["pair_order"]))},
            {"initialization": "profile_adapter"},
            {"optimizer_state_resumed": True},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "final exact pair"),
            ):
                runner.validate_final_run({**run, **change}, pairs, "substantive")

    def test_reference_requires_all_pairs_unchanged_anchor_and_complete_eos_scores(
        self,
    ):
        pairs = [self.pair(index) for index in range(2)]
        tensors = {"lora_B": {"sha256": "anchor"}}
        runtime = self.root / "reference/runtime.json"
        write(runtime, {"initial_adapter_tensors": tensors})
        scores = self.root / "reference/reference-logps.jsonl"
        rows = [
            {
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "chosen_logprob": -2.0,
                "rejected_logprob": -3.0,
                "chosen_completion_tokens": 2,
                "rejected_completion_tokens": 3,
            }
            for pair in pairs
        ]
        write_rows(scores, rows)
        run = {
            "mode": "reference_precompute",
            "training_updates": 0,
            "reference_updates": 0,
            "reference_adapter_unchanged": True,
            "completed_pairs": 2,
            "reference_forward_attempts": 4,
            "reference_forward_completions": 4,
            "reference_forward_sequence_tokens": 18,
            "reference_forward_completion_tokens": 10,
            "reference_source": "exact_final_T1_adapter",
            "log_probability": "summed_completion_including_eos",
        }
        evidence = {
            "run": run,
            "files": {"runtime.json": runtime, "reference-logps.jsonl": scores},
        }
        runner.validate_reference(evidence, pairs, tensors)
        for field, value in (
            ("reference_updates", 1),
            ("reference_source", "adapter_disabled"),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "unchanged T1"),
            ):
                runner.validate_reference(
                    {**evidence, "run": {**run, field: value}}, pairs, tensors
                )
        for change in (
            {"chosen_logprob": float("nan")},
            {"chosen_completion_tokens": 1},
        ):
            write_rows(scores, [{**rows[0], **change}, rows[1]])
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "completion/EOS"),
            ):
                runner.validate_reference(evidence, pairs, tensors)
        write_rows(scores, list(reversed(rows)))
        with self.assertRaisesRegex(ValueError, "fixed order"):
            runner.validate_reference(evidence, pairs, tensors)

    def test_exposure_checks_three_forwards_two_backwards_and_final_partial_group(self):
        pairs = [self.pair(index) for index in range(5)]
        exposure = runner.trainer.Exposure()
        sides = {
            "chosen_scored": "chosen",
            "rejected_forward": "rejected",
            "rejected_backward": "rejected",
            "chosen_forward": "chosen",
            "chosen_backward": "chosen",
        }
        for pair in pairs:
            for phase in runner.trainer.PHASES:
                if phase in sides:
                    exposure.attempted_compute_calls[phase] += 1
                exposure.record(phase, pair, sides.get(phase))
        exposure.updates = 2
        exposure_path = self.root / "training/exposure.json"
        write(exposure_path, exposure.summary())
        events = []
        for index, group in enumerate((pairs[:4], pairs[4:])):
            events.append(
                {
                    "update": index + 1,
                    "pair_ids": [pair["pair_id"] for pair in group],
                    "microbatches": len(group),
                    "committed_pairs": min((index + 1) * 4, 5),
                    "pair_losses": [0.5] * len(group),
                    "mean_pair_loss": 0.5,
                    "gradient_norm_before_clip": 1.2,
                    "learning_rate": 1e-5,
                }
            )
        events_path = self.root / "training/training-events.jsonl"
        write_rows(events_path, events)
        evidence = {
            "files": {
                "exposure.json": exposure_path,
                "training-events.jsonl": events_path,
            }
        }
        runner.validate_exposure(evidence, pairs)
        write_rows(events_path, [events[0], {**events[1], "microbatches": 4}])
        with self.assertRaisesRegex(ValueError, "partial accumulation"):
            runner.validate_exposure(evidence, pairs)
        write_rows(events_path, events)
        changed = exposure.summary()
        changed["phases"]["chosen_forward"]["sequence_tokens"] -= 1
        write(exposure_path, changed)
        with self.assertRaisesRegex(ValueError, "three forwards"):
            runner.validate_exposure(evidence, pairs)

    def test_raw_validation_keeps_failures_uncredited_and_rejects_output_budget_drift(
        self,
    ):
        case = self.case()
        row = self.result(case)
        prefixes = {case["id"]: [1, 2, 3]}
        runner.validate_results([case], [row], prefixes)
        changed = copy.deepcopy(row)
        changed["response"]["usage"]["completion_tokens"] = 513
        with self.assertRaisesRegex(ValueError, "512-token"):
            runner.validate_results([case], [changed], prefixes)
        failure = {
            "case_id": case["id"],
            "request_failed": True,
            "error_type": "timeout",
            "raw_valid": False,
            "displayed": False,
        }
        runner.validate_results([case], [failure], prefixes)
        summary = runner.summarize([case], [failure])
        self.assertEqual(summary["correct_abstentions"], 0)
        self.assertEqual(summary["request_errors"], 1)
        self.assertEqual(summary["outcome_reasons"]["request_failure"]["count"], 1)
        missing = runner.summarize([case], [])
        self.assertEqual(missing["missing_requests"], 1)
        self.assertEqual(missing["outcome_reasons"]["missing_result"]["count"], 1)
        with self.assertRaisesRegex(ValueError, "cannot receive"):
            runner.validate_results([case], [{**failure, "displayed": True}], prefixes)

    def test_fresh_reload_requires_separate_process_same_final_tensors_and_eight_outputs(
        self,
    ):
        manifest = self.root / "train/manifest.json"
        write(manifest, {})
        ids = [f"diagnostic-{index}" for index in range(8)]
        outputs = [{"case_id": identity, "output_ids": [4]} for identity in ids]
        tensors = {"lora_B": {"sha256": "final"}}
        training = {
            "path": manifest,
            "run": {
                "experiment_phase": "substantive",
                "reference_cache_sha256": "cache",
                "training_updates": 9,
                "pid": 1,
            },
            "files": {},
        }
        reloaded = {
            "run": {
                "mode": "fresh_reload",
                "experiment_phase": "substantive",
                "previous_manifest_sha256": runner.probe.digest(manifest),
                "reference_cache_sha256": "cache",
                "reloaded_training_updates": 9,
                "optimizer_state_resumed": False,
                "pid": 2,
            },
            "files": {},
        }
        values = (
            (training, "train", "smoke-generation.json", outputs),
            (training, "train", "adapter-tensors.json", tensors),
            (reloaded, "reload", "smoke-generation.json", outputs),
            (reloaded, "reload", "runtime.json", {"initial_adapter_tensors": tensors}),
            (
                reloaded,
                "reload",
                "reload-parity.json",
                {
                    "adapter_tensors_identical": True,
                    "optimizer_state_resumed": False,
                    "cases": [
                        {"case_id": identity, "identical": True} for identity in ids
                    ],
                },
            ),
        )
        for evidence, directory, name, value in values:
            path = self.root / directory / name
            write(path, value)
            evidence["files"][name] = path
        runner.validate_reload(training, reloaded, ids)
        with self.assertRaisesRegex(ValueError, "Fresh reload"):
            runner.validate_reload(
                training, {**reloaded, "run": {**reloaded["run"], "pid": 1}}, ids
            )
        write(
            reloaded["files"]["runtime.json"],
            {"initial_adapter_tensors": {"lora_B": {"sha256": "profile"}}},
        )
        with self.assertRaisesRegex(ValueError, "Fresh reload"):
            runner.validate_reload(training, reloaded, ids)

    def test_saved_sft_failure_receipt_is_hash_bound_without_success_only_filtering(
        self,
    ):
        cases = [self.case("one"), self.case("two")]
        row = {
            "case_id": "one",
            "request_failed": True,
            "raw_valid": False,
            "displayed": False,
            "error_type": "timeout",
        }
        directory = self.root / "sft"
        registry = self.root / "registry.json"
        registry_binding = write(registry, {})
        write(directory / "implementation/model-registry.json", {})
        binding = {"condition": "sft-full", "snapshot_sha256": {}}
        manifest = {"benchmark": binding, "status": "failed", "completed_requests": 1}
        manifest_binding = write(directory / "manifest.json", manifest)
        results_sha = write_rows(directory / "results.jsonl", [row])
        receipt_path = directory / "curriculum-receipt.json"
        receipt = {
            "condition": "sft-full",
            "configuration_sha256": "fixed",
            "status": "failed",
            "files_sha256": {
                "manifest.json": manifest_binding["sha256"],
                "results.jsonl": results_sha,
            },
        }
        write(receipt_path, receipt)
        prepared = {
            "cases": cases,
            "prefixes": {case["id"]: [1, 2, 3] for case in cases},
            "binding": binding,
        }
        expected = {"condition": "sft-full", "configuration_sha256": "fixed"}
        registration = {"inputs": {registry: registry_binding["sha256"]}}
        with patch.object(runner.curriculum, "validate_runtime_manifest"):
            result = runner.saved_results(
                registration, receipt_path, expected, prepared, "sft", registry
            )
            self.assertEqual(result["rows"], [row])
            self.assertEqual(result["status"], "failed")
            with self.assertRaisesRegex(ValueError, "no reselection"):
                runner.saved_results(
                    registration,
                    receipt_path,
                    {**expected, "configuration_sha256": "other"},
                    prepared,
                    "sft",
                    registry,
                )
            write_rows(directory / "results.jsonl", [])
            with self.assertRaisesRegex(ValueError, "Changed bound input"):
                runner.saved_results(
                    registration, receipt_path, expected, prepared, "sft", registry
                )

    def test_runner_uses_only_dpo_and_retains_raw_failure_receipt_on_exception(self):
        case = self.case()
        failure = {
            "case_id": case["id"],
            "request_failed": True,
            "raw_valid": False,
            "displayed": False,
            "error_type": "timeout",
        }
        registration = {
            "inputs": {},
            "paths": {"registry": self.root / "registry.json"},
            "execution": {"dpo_candidate_id": "dpo-final"},
            "t1": {"args": SimpleNamespace(release=self.root / "release")},
            "prepared": {"cases": [case], "prefixes": {case["id"]: [1, 2, 3]}},
            "binding": {"condition": "dpo-full", "registered_new_requests": 289},
        }
        output = self.root / "run"

        def interrupted(args, prepared):
            self.assertEqual(args.candidate, "dpo-final")
            self.assertFalse(args.unconstrained)
            self.assertEqual(args.timeout, 120)
            self.assertIs(prepared, registration["prepared"])
            output.mkdir()
            write_rows(output / "results.jsonl", [failure])
            write(output / "manifest.json", {"status": "failed"})
            raise TimeoutError("server deadline")

        with (
            patch.object(runner, "snapshot"),
            patch.object(runner.probe, "run", side_effect=interrupted),
        ):
            with self.assertRaisesRegex(TimeoutError, "server deadline"):
                runner.run_condition(registration, output, "screen", "uncontrolled")
        receipt = runner.load_json(output / "dpo-receipt.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("results.jsonl", receipt["files_sha256"])
        self.assertIn("source-execution.jsonl", receipt["files_sha256"])
        self.assertEqual(runner.original.read_rows(output / "results.jsonl"), [failure])
        with self.assertRaisesRegex(ValueError, "write-once"):
            runner.run_condition(registration, output, "screen", "uncontrolled")

    def test_worse_dpo_quality_completes_fixed_experiment_and_reuses_sft_timing(self):
        cases = [self.case(str(index)) for index in range(289)]
        sft_rows = [self.result(case) for case in cases]
        dpo_rows = [self.result(case, "invalid") for case in cases]
        receipt = self.root / "dpo-run/dpo-receipt.json"
        write(receipt, {})
        registration = {
            "inputs": {},
            "prepared": {"cases": cases},
            "t1": {"args": SimpleNamespace(release=self.root / "release")},
            "execution": {"dpo_candidate_id": "dpo"},
            "paths": {"registry": self.root / "registry.json"},
            "binding": {"registered_new_requests": 289},
            "protocol": {"scope": "Exposed development with AI preferences"},
            "preferences": {"pairs": [{"reason": "format_invalid"}]},
            "sft": {
                "rows": sft_rows,
                "status": "completed",
                "timing_scope": runner.TIMING_SCOPE,
            },
        }
        with (
            patch.object(runner, "prepare", return_value=registration),
            patch.object(runner, "snapshot"),
            patch.object(
                runner,
                "saved_results",
                return_value={"rows": dpo_rows, "status": "completed"},
            ),
        ):
            runner.report("protocol", "execution", receipt.parent, self.root / "report")
        result = runner.load_json(self.root / "report/summary.json")
        self.assertEqual(result["experiment_status"], "completed")
        self.assertEqual(result["registered_new_requests"], 289)
        self.assertEqual(result["registered_reused_sft_requests"], 289)
        self.assertGreater(
            result["runs"]["sft-full"]["displayed_useful_edits"],
            result["runs"]["dpo-full"]["displayed_useful_edits"],
        )
        self.assertEqual(result["runs"]["dpo-full"]["invalid_results"], 289)
        self.assertIn(
            "not a fresh paired latency", result["runs"]["sft-full"]["timing_scope"]
        )


if __name__ == "__main__":
    unittest.main()
