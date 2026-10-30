"""Tiny export files and mocked processes only; no model, network or real data."""

import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/final-completion/20261025-001/run_runtime_experiments.py"
)
if not PATH.is_file():
    raise unittest.SkipTest("Requires the archived runtime experiment runner.")
SPEC = importlib.util.spec_from_file_location("final_runtime_experiments", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RuntimeExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train, self.live = self.root / "train.jsonl", self.root / "live.json"
        self.train.write_text('{"synthetic":true}\n')
        self.live.write_text('{"synthetic":true}')
        self.roster = [{"id": f"synthetic-{index}"} for index in range(43)]

    def export(self, quantization="Q8_0", model_name="lfm25-1.2b", name="export"):
        exported = self.root / name
        exported.mkdir()
        model = exported / "receipt-selected.gguf"
        model.write_bytes(b"synthetic file, not model weights:" + quantization.encode())
        (exported / "model-Q4_K_M.gguf").write_bytes(
            b"misleading filename must not be selected"
        )
        (exported / "chat_template.jinja").write_text("synthetic template")
        config = {
            "version": runner.data.VERSION,
            "model": model_name,
            "base": copy.deepcopy(runner.data.PROFILES[model_name]),
            "assets": {
                "files_sha256": {
                    "chat_template.jinja": runner.data.sha(
                        exported / "chat_template.jinja"
                    )
                }
            },
        }
        runner.data.write(exported / "configuration.json", config)
        report = {
            "version": runner.data.VERSION,
            "status": "gguf_exported_runtime_validation_pending",
            "kind": "sft",
            "base": config["base"],
            "quantization": quantization,
            "gguf_sha256": runner.data.sha(model),
            "gguf_bytes": model.stat().st_size,
            "adapter_sha256": "a" * 64,
            "training_configuration_sha256": runner.data.sha(
                exported / "configuration.json"
            ),
        }
        runner.data.write(exported / "run.json", report)
        runner.data.write(
            exported / "commands.json",
            [
                ["converter"],
                [
                    "quantizer",
                    "/content/old/model-BF16.gguf",
                    "/content/old/receipt-selected.gguf",
                    quantization,
                ],
            ],
        )
        return exported, model, report

    def test_receipt_selects_q4_or_q8_after_relocation_without_filename_guessing(self):
        for index, quantization in enumerate(("Q4_K_M", "Q8_0")):
            exported, model, report = self.export(quantization, name=f"export-{index}")
            found, template, identity = runner.resolve_export(exported, "lfm25-1.2b")
            self.assertEqual(found, model.resolve())
            self.assertEqual(template, exported.resolve() / "chat_template.jinja")
            self.assertEqual(identity["quantization"], quantization)
            self.assertEqual(identity["model_file"]["sha256"], report["gguf_sha256"])
            self.assertEqual(identity["training_adapter_sha256"], "a" * 64)
            self.assertEqual(
                identity["recorded_export_path"], "/content/old/receipt-selected.gguf"
            )

    def test_missing_or_mismatched_export_identity_is_rejected_before_process_or_output(
        self,
    ):
        for index, mutation in enumerate(
            (
                "missing_model",
                "wrong_bytes",
                "missing_identity",
                "wrong_size",
                "wrong_quantization",
                "wrong_config",
                "wrong_template",
                "missing_commands",
                "wrong_model",
                "escaping_symlink",
            )
        ):
            with self.subTest(mutation=mutation):
                exported, model, report = self.export(name=f"export-{index}")
                selected = "lfm25-1.2b"
                if mutation == "missing_model":
                    model.unlink()
                elif mutation == "wrong_bytes":
                    model.write_bytes(b"x" * model.stat().st_size)
                elif mutation == "missing_identity":
                    report.pop("adapter_sha256")
                elif mutation == "wrong_size":
                    report["gguf_bytes"] += 1
                elif mutation == "wrong_quantization":
                    report["quantization"] = "Q4_K_M"
                elif mutation == "wrong_config":
                    report["training_configuration_sha256"] = "b" * 64
                elif mutation == "wrong_template":
                    (exported / "chat_template.jinja").write_text("changed")
                elif mutation == "missing_commands":
                    (exported / "commands.json").unlink()
                elif mutation == "wrong_model":
                    selected = "gemma4-e2b"
                else:
                    external = self.root / "external.gguf"
                    model.replace(external)
                    model.symlink_to(external)
                runner.data.write(exported / "run.json", report)
                output = self.root / f"output-{index}"
                with (
                    patch.object(runner.subprocess, "Popen") as process,
                    self.assertRaises(ValueError),
                ):
                    runner.main(
                        [
                            "--model",
                            selected,
                            "--exported",
                            str(exported),
                            "--output",
                            str(output),
                        ]
                    )
                process.assert_not_called()
                self.assertFalse(output.exists())

    def test_dpo_export_retains_and_validates_same_parent_identity(self):
        exported, _, report = self.export()
        (exported / "plan.md").write_text("synthetic DPO plan")
        frozen = {
            "plan_sha256": runner.data.sha(exported / "plan.md"),
            "parent": {
                "configuration_sha256": report["training_configuration_sha256"],
                "adapter_sha256": "b" * 64,
            },
        }
        runner.data.write(exported / "dpo_configuration.json", frozen)
        report.update(
            kind="dpo",
            parent_adapter_sha256="b" * 64,
            dpo_configuration_sha256=runner.data.sha(
                exported / "dpo_configuration.json"
            ),
        )
        runner.data.write(exported / "run.json", report)
        _, _, identity = runner.resolve_export(exported, "lfm25-1.2b")
        self.assertEqual(
            (
                identity["kind"],
                identity["training_adapter_sha256"],
                identity["parent_adapter_sha256"],
            ),
            ("dpo", "a" * 64, "b" * 64),
        )
        (exported / "plan.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "DPO export"):
            runner.resolve_export(exported, "lfm25-1.2b")

    def write_child_result(self, argv):
        def arg(name):
            return argv[argv.index(name) + 1]

        conditions = argv[argv.index("--conditions") + 1 : argv.index("--repeats")]
        output = Path(arg("--output"))
        output.mkdir()
        protocol = {
            "status": "completed",
            "model": arg("--model"),
            "gguf_sha256": runner.data.sha(arg("--gguf")),
            "train_sha256": runner.data.sha(arg("--train")),
            "live_sha256": runner.data.sha(arg("--live")),
            "order": conditions,
            "repeats": 1,
            "per_cell": 3,
            "sampling_profile": arg("--sampling-profile"),
            "settings": runner.benchmark.settings(
                arg("--model"), arg("--sampling-profile"), int(arg("--seed"))
            ),
            "run_label": arg("--run-label"),
            "roster": self.roster,
            "runtime_launch": {
                "sha256": runner.data.sha(arg("--runtime-command-file"))
            },
            "final_test_bodies_opened": False,
        }
        if "--lora-gateway" in argv:
            protocol["lora"] = runner.benchmark.lora_binding(
                arg("--lora-gateway"), arg("--gguf")
            )
            protocol["settings"]["lora"] = [{"id": 0, "scale": 1.0}]
        rows = [
            {
                "id": row["id"],
                "condition": condition,
                "repeat": 1,
                "phase": phase,
                "group": "train_action_language",
                "condition_prerequisites_met": True,
                "outcome": "request_error",
                "error": "synthetic HTTP failure",
                "elapsed_ms": 10,
                "raw_text": '{"error":"synthetic"}',
            }
            for condition in conditions
            for row in self.roster
            for phase in (
                ("warmup", "measured")
                if condition == "identical_repeat"
                else ("measured",)
            )
        ]
        runner.data.write(output / "protocol.json", protocol)
        runner.data.write_rows(output / "results.jsonl", rows)
        runner.data.write(
            output / "summary.json", runner.benchmark.summarize(rows, conditions)
        )
        return output, protocol, rows

    def simulate(
        self,
        exported,
        *,
        model_name="lfm25-1.2b",
        flags=("--sampling",),
        fail_first=False,
        startup_failure=False,
        interrupt=False,
    ):
        output = self.root / "output"
        calls, processes, lifecycle = [], [], []

        def popen(argv, **kwargs):
            if processes:
                self.assertTrue(processes[-1].wait.called)
            process = Mock()
            process.poll.return_value = None
            process.wait.return_value = 0
            processes.append(process)
            lifecycle.append(argv)
            return process

        def run(argv, **kwargs):
            calls.append(argv)
            child, protocol, rows = self.write_child_result(argv)
            kwargs["stdout"].write("preserved synthetic subprocess log\n")
            if (fail_first or interrupt) and len(calls) == 1:
                runner.data.write_rows(child / "results.jsonl", rows[:1])
                protocol["status"] = "interrupted"
                runner.data.write(child / "protocol.json", protocol)
                if interrupt:
                    raise KeyboardInterrupt()
                return subprocess.CompletedProcess(argv, 1)
            return subprocess.CompletedProcess(argv, 0)

        socket = Mock()
        socket.__enter__ = Mock(return_value=socket)
        socket.__exit__ = Mock(return_value=False)
        socket.connect_ex.return_value = 1
        readiness = (
            [RuntimeError("synthetic startup failure"), None, None, None, None]
            if startup_failure
            else None
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "TRAIN", self.train))
            stack.enter_context(patch.object(runner, "LIVE", self.live))
            stack.enter_context(
                patch.object(runner.benchmark, "roster", return_value=self.roster)
            )
            stack.enter_context(
                patch.object(runner.socket, "socket", return_value=socket)
            )
            stack.enter_context(
                patch.object(runner.subprocess, "Popen", side_effect=popen)
            )
            stack.enter_context(patch.object(runner.subprocess, "run", side_effect=run))
            stack.enter_context(
                patch.object(runner, "wait_ready", side_effect=readiness)
            )
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            argv = [
                "--model",
                model_name,
                "--deployment" if model_name == "gemma26" else "--exported",
                str(exported),
                "--output",
                str(output),
                *flags,
            ]
            if model_name == "gemma26":
                argv.extend(["--model-metadata", str(exported / "model-metadata.json")])
            if interrupt:
                with self.assertRaises(KeyboardInterrupt):
                    runner.main(argv)
                code = None
            else:
                code = runner.main(argv)
        return code, runner.data.read(output / "run.json"), calls, processes, lifecycle

    def test_failed_condition_keeps_raw_output_and_runs_all_later_variants(self):
        exported, model, _ = self.export()
        code, state, calls, processes, commands = self.simulate(
            exported, fail_first=True
        )
        self.assertEqual(
            (code, state["status"], len(calls), len(processes)),
            (1, "completed_with_failures", 11, 5),
        )
        self.assertFalse(state["registered_plan_completed"])
        self.assertEqual(state["jobs"][0]["status"], "failed")
        self.assertTrue(all(job["status"] == "completed" for job in state["jobs"][1:]))
        self.assertEqual(state["jobs"][-1]["variant"], "baseline-repeat")
        self.assertIn("greedy-seed42/results.jsonl", state["jobs"][0]["files_sha256"])
        self.assertIn("preserved", Path(state["jobs"][0]["log"]["path"]).read_text())
        self.assertEqual(
            len(runner.data.rows(Path(state["jobs"][0]["output"]) / "results.jsonl")), 1
        )
        self.assertEqual(
            [(c[c.index("-ub") + 1], c[c.index("-fa") + 1]) for c in commands],
            [
                ("512", "auto"),
                ("1024", "auto"),
                ("512", "off"),
                ("2048", "auto"),
                ("512", "auto"),
            ],
        )
        self.assertTrue(
            all(
                c[c.index("-m") + 1] == str(model.resolve())
                and c[c.index("-b") + 1] == "2048"
                for c in commands
            )
        )
        self.assertTrue(
            all(
                p.terminate.call_count == 1 and p.wait.call_count == 1
                for p in processes
            )
        )
        self.assertEqual(
            [(job["sampling"], job["seed"]) for job in state["jobs"][1:7]],
            [
                (profile, seed)
                for profile in ("lfm_vendor", "lfm_unsloth")
                for seed in (42, 43, 44)
            ],
        )

    def test_e2b_full_grid_records_every_observation_without_asserting_quality_or_ui_success(
        self,
    ):
        exported, _, _ = self.export("Q4_K_M", "gemma4-e2b")
        code, state, calls, processes, _ = self.simulate(
            exported, model_name="gemma4-e2b"
        )
        self.assertEqual(
            (code, state["status"], len(calls), len(processes)), (0, "completed", 8, 5)
        )
        self.assertTrue(state["registered_plan_completed"])
        self.assertEqual(sum(job["observations"] for job in state["jobs"]), 989)
        self.assertFalse(state["quality_or_UI_success_asserted"])
        self.assertFalse(state["final_test_bodies_opened"])
        self.assertEqual(state["selected_export"]["quantization"], "Q4_K_M")

    def test_gemma_grid_keeps_checked_base_adapter_and_all_eight_conditions(self):
        directory, model, _ = self.export()
        template = directory / "chat_template.jinja"
        adapter = directory / "adapter.gguf"
        adapter.write_bytes(b"synthetic ordinary LoRA")
        gateway = {
            "version": 2,
            "model_path": str(model),
            "model_sha256": runner.data.sha(model),
            "adapter_path": str(adapter),
            "adapter_sha256": runner.data.sha(adapter),
            "adapter_scale": 1.0,
            "chat_template_path": str(template),
        }
        runner.data.write(directory / "gateway.json", gateway)
        identity = {
            "model_file": runner.binding(model),
            "lora": runner.benchmark.lora_binding(directory / "gateway.json", model),
            "runtime_command": [
                "synthetic-server",
                "-m",
                str(model),
                "--alias",
                "old",
                "--port",
                "18000",
                "--lora",
                str(adapter),
                "-c",
                "4096",
                "-np",
                "1",
            ],
        }
        import evaluate_gemma_final as evaluation
        import gemma_final_deployment as gemma

        native = {
            "inference_configuration": gemma.binding(directory / "configuration.json"),
            "identity": {"adapter_sha256": "a" * 64},
        }
        metadata = directory / "model-metadata.json"
        runner.data.write(metadata, native)
        runner.data.write(directory / "deployment.json", {"synthetic": True})
        receipt = {
            "numerical_regime": "native_derived_q4_base_fp32_lora",
            "adapter_export_receipt": {},
            "base_export_receipt": {},
        }
        with (
            patch.object(evaluation, "verify_model", return_value=native),
            patch.object(gemma, "validate_deployment"),
            patch.object(
                gemma,
                "deployment_check",
                return_value=(gateway, identity["runtime_command"], receipt),
            ),
        ):
            code, state, calls, processes, commands = self.simulate(
                directory, model_name="gemma26"
            )
        self.assertEqual((code, len(calls), len(processes)), (0, 8, 5))
        self.assertTrue(state["registered_plan_completed"])
        self.assertTrue(
            all(
                command[command.index("--lora") + 1] == str(adapter)
                for command in commands
            )
        )
        self.assertTrue(all("--lora-gateway" in command for command in calls))
        self.assertEqual(sum(job["observations"] for job in state["jobs"]), 989)
        self.assertEqual(
            state["selected_export"]["native_model_identity"], native["identity"]
        )

    def test_gemma_runtime_rejects_a_different_training_parent_before_launch(self):
        import evaluate_gemma_final as evaluation
        import gemma_final_deployment as gemma

        directory = self.root / "deployment"
        metadata = self.root / "model.json"
        config = self.root / "configuration.json"
        runner.data.write(config, {"synthetic": True})
        model = {
            "inference_configuration": gemma.binding(config),
            "identity": {"adapter_sha256": "a" * 64},
        }
        directory.mkdir()
        runner.data.write(directory / "deployment.json", {"old_adapter": "B003"})
        with (
            patch.object(evaluation, "verify_model", return_value=model) as verify,
            patch.object(
                gemma,
                "validate_deployment",
                side_effect=ValueError("different adapter parent"),
            ) as validate,
            patch.object(gemma, "deployment_check") as check,
            patch.object(runner.subprocess, "Popen") as process,
        ):
            with self.assertRaisesRegex(ValueError, "different adapter parent"):
                runner.main(
                    [
                        "--model",
                        "gemma26",
                        "--deployment",
                        str(directory),
                        "--model-metadata",
                        str(metadata),
                        "--output",
                        str(self.root / "output"),
                    ]
                )
        verify.assert_called_once_with(metadata)
        self.assertEqual(validate.call_args.args[1], model)
        check.assert_not_called()
        process.assert_not_called()
        self.assertFalse((self.root / "output").exists())

    def test_partial_scope_is_not_reported_as_registered_plan_complete(self):
        exported, _, _ = self.export()
        code, state, calls, _, _ = self.simulate(exported, flags=("--baseline-only",))
        self.assertEqual(
            (code, state["status"], len(calls)), (0, "completed_partial_scope", 1)
        )
        self.assertTrue(state["requested_grid_completed"])
        self.assertFalse(state["registered_plan_completed"])
        self.assertEqual(
            sum(job["status"] == "not_requested" for job in state["jobs"]), 10
        )

    def test_startup_failure_is_preserved_and_does_not_drop_remaining_variants(self):
        exported, _, _ = self.export()
        code, state, calls, processes, _ = self.simulate(exported, startup_failure=True)
        self.assertEqual((code, len(calls), len(processes)), (1, 4, 5))
        self.assertTrue(all(job["status"] == "not_issued" for job in state["jobs"][:7]))
        self.assertTrue(all(job["status"] == "completed" for job in state["jobs"][7:]))
        self.assertIn("server.log", state["variant_evidence"]["baseline"])

    def test_interruption_stops_owned_server_and_preserves_partial_observations(self):
        exported, _, _ = self.export()
        _, state, calls, processes, _ = self.simulate(exported, interrupt=True)
        self.assertEqual(
            (state["status"], len(calls), len(processes)), ("interrupted", 1, 1)
        )
        self.assertEqual(state["jobs"][0]["status"], "interrupted")
        self.assertTrue(all(job["status"] == "not_issued" for job in state["jobs"][1:]))
        self.assertFalse(state["requested_grid_completed"])
        processes[0].terminate.assert_called_once()
        self.assertIn("greedy-seed42/results.jsonl", state["jobs"][0]["files_sha256"])

    def test_false_completed_child_with_missing_observation_or_changed_identity_is_rejected(
        self,
    ):
        exported, model, _ = self.export()
        _, _, identity = runner.resolve_export(exported, "lfm25-1.2b")
        launch = self.root / "command.txt"
        launch.write_text("synthetic command")
        job = runner.experiment_grid("lfm25-1.2b")[0]
        state = {
            "selected_export": identity,
            "inputs": {
                "train": runner.binding(self.train),
                "live": runner.binding(self.live),
            },
            "roster_ids": [row["id"] for row in self.roster],
        }
        for index, mutation in enumerate(
            ("missing", "duplicate", "identity", "settings", "summary")
        ):
            with self.subTest(mutation=mutation):
                output, protocol, rows = self.write_child_result(
                    [
                        "--output",
                        str(self.root / f"child-{index}"),
                        "--model",
                        "fixture",
                        "--gguf",
                        str(model),
                        "--train",
                        str(self.train),
                        "--live",
                        str(self.live),
                        "--conditions",
                        *job["conditions"],
                        "--repeats",
                        "1",
                        "--sampling-profile",
                        "greedy",
                        "--seed",
                        "42",
                        "--run-label",
                        "baseline",
                        "--runtime-command-file",
                        str(launch),
                    ]
                )
                if mutation == "missing":
                    runner.data.write_rows(output / "results.jsonl", rows[:-1])
                elif mutation == "duplicate":
                    runner.data.write_rows(
                        output / "results.jsonl", [rows[0], *rows[:-1]]
                    )
                elif mutation == "identity":
                    protocol["gguf_sha256"] = "f" * 64
                elif mutation == "settings":
                    protocol["settings"]["seed"] = 43
                else:
                    runner.data.write(output / "summary.json", {})
                runner.data.write(output / "protocol.json", protocol)
                with self.assertRaises(ValueError):
                    runner.validate_result(output, job, "fixture", state, launch)

    def test_loading_http_200_waits_before_next_readiness_poll(self):
        process = Mock()
        process.poll.return_value = None
        client = Mock()
        client.request.side_effect = [{"status": "loading"}, {"status": "ok"}]
        with (
            patch.object(runner.serving, "LocalClient", return_value=client),
            patch.object(runner.time, "monotonic", side_effect=[0, 0, 0.25]),
            patch.object(runner.time, "sleep") as sleep,
        ):
            runner.wait_ready(process, "http://127.0.0.1:18093")
        sleep.assert_called_once_with(0.25)
        self.assertEqual(client.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
