import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import probe_passive_gemma as runner
from test_train_passive_gemma import configuration


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": runner.probe.digest(path)}


class GemmaProbeTests(unittest.TestCase):
    def setUp(self):
        (runner.ROOT / "artifacts").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=runner.ROOT / "artifacts")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def training_evidence(self):
        root = self.root
        release = write(root / "release.json", {})
        protocol_path = (
            runner.ROOT
            / "artifacts/benchmarks/passive-t1-gemma26-v1/protocol-attempt-02.json"
        )
        if not protocol_path.is_file():
            self.skipTest("Requires the local Gemma benchmark protocol.")
        protocol = runner.load_json(protocol_path)
        protocol["b2"]["release_manifest"] = release
        protocol_binding = write(root / "protocol.json", protocol)
        preparation = write(
            root / "preparation/manifest.json",
            {
                "version": "passive-sft-gemma-preparation-v1",
                "status": "frozen_train_only_preparation",
                "release_manifest_sha256": release["sha256"],
                "files_sha256": {},
            },
        )
        pin = write(
            root / "pin.json",
            {
                "repository": runner.BASE_REPOSITORY,
                "revision": runner.BASE_REVISION,
                "files": {},
            },
        )
        selection = write(
            root / "selection.json",
            {
                "selected_t1_candidate_ids": ["gemma26"],
                "bindings": {
                    "train_only_preparation": preparation,
                    "release_manifest_sha256": release["sha256"],
                },
            },
        )
        config = configuration("substantive")
        config["model"]["gpu_name_contains"] = protocol["engineering_smoke"][
            "native_gpu"
        ]
        config["model"]["minimum_gpu_bytes"] = protocol["engineering_smoke"][
            "minimum_gpu_bytes"
        ]
        config["training"].update(
            max_total_sequence_tokens=protocol["training_plan"]["sequence_tokens"],
            max_total_completion_tokens=protocol["training_plan"]["completion_tokens"],
        )
        config["bindings"] = {
            "preparation": preparation,
            "native_pin": pin,
            "evaluation_protocol": protocol_binding,
            "selection": selection,
        }
        config["bindings"] = {
            key: {**value, "path": str(Path(value["path"]).relative_to(root))}
            for key, value in config["bindings"].items()
        }
        source_bytes = {}
        for filename, key, source in runner.trainer.SOURCE_MODULES:
            source_bytes[filename] = Path(source).read_bytes()
            config[key] = runner.probe.digest(Path(source))
        smoke_config = copy.deepcopy(config)
        smoke_config["phase"] = "smoke"
        smoke_config["training"].update(
            max_updates=2,
            draw_count=8,
            recovery_updates=[],
            sample_order=config["training"]["sample_order"][:8],
        )

        def bind_smoke(name, value):
            result = write(root / "smoke" / f"{name}.json", value)
            result["path"] = str(Path(result["path"]).relative_to(root))
            config["bindings"][name] = result
            return result["sha256"]

        smoke_sha = bind_smoke("smoke_configuration", smoke_config)
        smoke_run = {
            "version": runner.trainer.VERSION,
            "phase": "smoke",
            "mode": "train",
            "status": "training_complete_fresh_reload_pending",
            "training_updates": 2,
            "optimizer_committed_rows": 8,
            "configuration_sha256": smoke_sha,
            "release_manifest_sha256": release["sha256"],
            "final_test_bodies_read": 0,
            "optimizer_state_resumed": False,
            "pid": 11,
        }
        smoke_run_sha = bind_smoke("smoke_run", smoke_run)
        smoke_manifest_sha = bind_smoke(
            "smoke_training_manifest",
            {
                "version": runner.trainer.VERSION,
                "status": smoke_run["status"],
                "files_sha256": {
                    "run.json": smoke_run_sha,
                    "configuration.json": smoke_sha,
                },
            },
        )
        smoke_reload = {
            "version": runner.trainer.VERSION,
            "phase": "smoke",
            "mode": "fresh_reload",
            "status": "fresh_reload_passed_local_export_pending",
            "reloaded_training_updates": 2,
            "configuration_sha256": smoke_sha,
            "previous_manifest_sha256": smoke_manifest_sha,
            "pid": 12,
        }
        smoke_reload_sha = bind_smoke("smoke_reload_run", smoke_reload)
        parity = {
            "adapter_tensors_identical": True,
            "optimizer_state_resumed": False,
            "cases": [
                {"case_id": name, "identical": True}
                for name in smoke_config["training"]["sample_order"]
            ],
        }
        parity_sha = bind_smoke("smoke_reload_parity", parity)
        bind_smoke(
            "smoke_reload_manifest",
            {
                "version": runner.trainer.VERSION,
                "status": smoke_reload["status"],
                "files_sha256": {
                    "run.json": smoke_reload_sha,
                    "reload-parity.json": parity_sha,
                },
            },
        )
        config_binding = write(root / "configuration-portable-name.json", config)
        self.bundle_readback = {
            "manifest": runner.load_json(Path(preparation["path"])),
            "verification": {
                "status": "passed_train_only_gemma_readback",
                "train_rows": 809,
                "draws": 809,
                "optimizer_updates": 203,
                "sequence_tokens": config["training"]["max_total_sequence_tokens"],
                "completion_tokens": config["training"]["max_total_completion_tokens"],
                "dev_test_record_bodies_read": 0,
                "training_updates_executed": 0,
            },
        }
        verifier = patch.object(
            runner.trainer.preparation,
            "verify_bundle",
            return_value=self.bundle_readback,
        )
        self.bundle_verifier = verifier.start()
        self.addCleanup(verifier.stop)
        prepared_validator = patch.object(runner.trainer, "validate_prepared")
        self.prepared_validator = prepared_validator.start()
        self.addCleanup(prepared_validator.stop)
        common = {
            "version": runner.trainer.VERSION,
            "phase": "substantive",
            "configuration_sha256": config_binding["sha256"],
            "evaluation_protocol_sha256": protocol_binding["sha256"],
            "native_pin_sha256": pin["sha256"],
            "preparation_sha256": preparation["sha256"],
            "final_test_bodies_read": 0,
            "optimizer_state_resumed": False,
            "peak_allocated_gpu_bytes": 100,
            "peak_reserved_gpu_bytes": 200,
        }
        training = {
            **common,
            "status": "training_complete_fresh_reload_pending",
            "mode": "train",
            "pid": 21,
            "training_updates": 203,
            "optimizer_committed_rows": 809,
            "initialization": "original_native_base_with_fresh_lora",
        }
        training_binding = write(root / "train/run.json", training)
        runtime = {
            "numerical_regime": "native_bfloat16_lora_float32",
            "parameter_dtypes": ["torch.bfloat16"],
            "ignored_native_tensor_names": [],
            "targets": {name: {} for name in config["model"]["target_modules"]},
            "effective_generation": config["generation"],
            "native_generation_config": {"eos_token_id": [1, 106, 50]},
            "completion_label_eos_token_id": 1,
            "gpu": config["model"]["gpu_name_contains"],
            "gpu_total_bytes": config["model"]["minimum_gpu_bytes"],
            "base_parameter_census": {
                "parameter_elements_by_dtype": {"torch.bfloat16": 123}
            },
            "after_adapter_parameter_census": {
                "parameter_elements_by_dtype": {
                    "torch.bfloat16": 123,
                    "torch.float32": 18585600,
                }
            },
        }
        generated = [
            {
                "case_id": row["case_id"],
                "text_raw": "<eos>",
                "output_ids": [1],
                "terminal_token_id": 1,
            }
            for row in parity["cases"]
        ]
        for directory in (root / "train", root / "reload"):
            for filename, content in source_bytes.items():
                directory.mkdir(parents=True, exist_ok=True)
                (directory / filename).write_bytes(content)
            write(directory / "configuration.json", config)
            write(directory / "runtime.json", runtime)
            write(directory / "smoke-generation.json", generated)
        write(
            root / "train/exposure.json",
            {
                "completed_updates": 203,
                "optimizer_commit_uncertain": False,
                "pending_backward_rows_not_committed": 0,
                "phases": {
                    "optimizer_committed": {
                        "rows": 809,
                        "counts": {
                            "case_id": dict(
                                runner.Counter(config["training"]["sample_order"])
                            )
                        },
                        "sequence_tokens": config["training"][
                            "max_total_sequence_tokens"
                        ],
                        "completion_tokens": config["training"][
                            "max_total_completion_tokens"
                        ],
                    }
                },
            },
        )
        write(
            root / "train/initialization.json",
            {
                "initialization": "original_native_base_with_fresh_lora",
                "smoke_adapter_loaded": False,
                "trainable_tensors": {str(i): 1 for i in range(410)},
                "trainable_parameters": 18585600,
            },
        )

        def manifest(directory, status):
            return write(
                directory / "manifest.json",
                {
                    "version": runner.trainer.VERSION,
                    "status": status,
                    "files_sha256": {
                        p.name: runner.probe.digest(p)
                        for p in directory.iterdir()
                        if p.name != "manifest.json"
                    },
                },
            )

        previous = manifest(root / "train", training["status"])
        write(
            root / "reload/run.json",
            {
                **common,
                "mode": "fresh_reload",
                "pid": 22,
                "status": "fresh_reload_passed_local_export_pending",
                "reloaded_training_updates": 203,
                "previous_manifest_sha256": previous["sha256"],
            },
        )
        write(root / "reload/reload-parity.json", parity)
        reload_manifest = manifest(
            root / "reload", "fresh_reload_passed_local_export_pending"
        )
        registration = {
            "protocol": protocol,
            "inputs": {
                Path(release["path"]): release["sha256"],
                Path(protocol_binding["path"]): protocol_binding["sha256"],
                Path(config_binding["path"]): config_binding["sha256"],
                Path(reload_manifest["path"]): reload_manifest["sha256"],
            },
            "paths": {
                "release_manifest": Path(release["path"]),
                "protocol_new": Path(protocol_binding["path"]),
                "new_training_configuration": Path(config_binding["path"]),
                "new_fresh_reload": Path(reload_manifest["path"]),
            },
        }
        return registration, Path(training_binding["path"]), training

    def test_full_training_reload_and_native_stop_evidence_are_required(self):
        registration, path, training = self.training_evidence()
        checked = runner.validate_training(registration, path, training)
        self.bundle_verifier.assert_called_once_with(self.root / "preparation")
        self.prepared_validator.assert_called_once_with(checked, self.bundle_readback)
        self.assertNotIn(
            "dev_test_record_bodies_copied", self.bundle_readback["manifest"]
        )
        self.assertEqual(checked["generation"]["eos_token_id"], [1, 106, 50])
        for key, value in (
            ("phase", "smoke"),
            ("training_updates", 2),
            ("optimizer_committed_rows", 808),
            ("configuration_sha256", "0" * 64),
            ("optimizer_state_resumed", True),
            ("initialization", "smoke_adapter"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_training(registration, path, {**training, key: value})
        config_path = registration["paths"]["new_training_configuration"]
        checked["generation"]["eos_token_id"] = [1, 106]
        registration["inputs"][config_path] = write(config_path, checked)["sha256"]
        with self.assertRaisesRegex(ValueError, "native terminal"):
            runner.validate_training(registration, path, training)

    def test_preparation_readback_rejects_wrong_scope_counts_or_source(self):
        registration, path, training = self.training_evidence()
        for key, bad in (
            ("status", "failed"),
            ("dev_test_record_bodies_read", 1),
            ("draws", 808),
            ("optimizer_updates", 202),
            ("sequence_tokens", 1),
            ("training_updates_executed", 1),
        ):
            original = self.bundle_readback["verification"][key]
            with self.subTest(key=key):
                self.bundle_readback["verification"][key] = bad
                with self.assertRaisesRegex(ValueError, "preparation readback"):
                    runner.validate_training(registration, path, training)
            self.bundle_readback["verification"][key] = original
        with patch.object(runner.trainer.preparation, "__file__", __file__):
            with self.assertRaisesRegex(ValueError, "preparation verifier"):
                runner.validate_training(registration, path, training)

    def test_own_icl_prefixes_must_match_fixed_eligibility_and_messages(self):
        ids = [f"case-{i}" for i in range(289)]
        eligibility = {
            "eligible_case_ids": ids[:100],
            "excluded": [{"case_id": i} for i in ids[100:]],
        }
        eligibility_binding = write(self.root / "prefix/common.json", eligibility)
        counts = {i: 1 if n < 100 else 3585 for n, i in enumerate(ids)}
        rows = [
            {"case_id": i, "input_ids": [1] * counts[i], "messages_sha256": "msg"}
            for i in ids
        ]
        path = self.root / "prefix/gemma26/icl.jsonl.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(gzip.compress("\n".join(json.dumps(r) for r in rows).encode()))
        manifest_path = self.root / "prefix/manifest.json"
        manifest = {
            "common_icl_eligibility_file": "common.json",
            "common_icl_eligibility_sha256": eligibility_binding["sha256"],
            "files_sha256": {"gemma26/icl.jsonl.gz": runner.probe.digest(path)},
            "prefix_counts": {"gemma26": {"icl": counts}},
        }
        write(manifest_path, manifest)
        prepared = {
            "cases": [{"id": i} for i in ids[:100]],
            "prefixes": {i: [999] for i in ids},
            "binding": {
                "registered_case_ids": ids,
                "messages_sha256": {i: "msg" for i in ids},
            },
        }
        registration = {
            "inputs": {},
            "paths": {
                "release_manifest": self.root / "release.json",
                "registry": self.root / "registry.json",
                "protocol": self.root / "protocol.json",
                "execution": self.root / "execution.json",
                "prefix_manifest": manifest_path,
            },
        }
        with patch.object(
            runner.original, "prepare", return_value=prepared
        ) as original:
            args, result, _ = runner.prepare_inputs(registration, "icl")
            self.assertEqual(args.candidate, "qwen35-9b")
            self.assertEqual(result["prefixes"][ids[0]], [1])
            self.assertEqual(original.call_count, 1)
            rows[0]["messages_sha256"] = "wrong"
            path.write_bytes(
                gzip.compress("\n".join(json.dumps(r) for r in rows).encode())
            )
            manifest["files_sha256"]["gemma26/icl.jsonl.gz"] = runner.probe.digest(path)
            write(manifest_path, manifest)
            registration["inputs"] = {}
            with self.assertRaisesRegex(ValueError, "different frozen messages"):
                runner.prepare_inputs(registration, "icl")

    def test_useful_f1_and_http_timing_keep_failures_and_exclude_grading(self):
        report = runner.quality_metrics(
            {
                "displayed_useful_precision": 0.75,
                "useful_edit_recall": 129 / 213,
                "all_request_p50_ms": 9000,
                "all_request_p95_ms": 10000,
                "by_action": {
                    "no_action": {
                        "displayed_useful_precision": None,
                        "useful_edit_recall": 0,
                    },
                    "empty": {
                        "displayed_useful_precision": None,
                        "useful_edit_recall": None,
                    },
                },
            }
        )
        self.assertAlmostEqual(report["derived_useful_edit_f1"], 0.6701298701298701)
        self.assertEqual(report["by_action"]["no_action"]["derived_useful_edit_f1"], 0)
        self.assertIsNone(report["by_action"]["empty"]["derived_useful_edit_f1"])
        self.assertNotIn("all_request_p50_ms", report)
        self.assertEqual(
            report["legacy_grading_inclusive_latency"]["all_request_p50_ms"], 9000
        )
        metrics = runner.http_latency(
            [
                {"http_response_ms": 100, "complete_json_ms": 9000},
                {"http_response_ms": 200, "request_failed": True},
                {"http_response_ms": 1000, "reused_from": "old"},
                {"http_response_ms": float("nan")},
            ],
            5,
        )
        self.assertEqual(
            (metrics["p50_ms"], metrics["p95_ms"], metrics["unmeasured_requests"]),
            (100, 200, 3),
        )

    def test_failed_or_missing_runs_never_qualify_as_learning(self):
        base = {
            "displayed_useful_precision": 0.75,
            "displayed_useful_edits": 129,
            "correct_abstentions": 63,
            "invalid_results": 24,
            "missing_requests": 0,
        }
        better = {**base, "displayed_useful_edits": 130}
        self.assertEqual(
            runner.learning_signal(base, better)["classification"],
            "descriptive_learning_signal",
        )
        self.assertEqual(
            runner.learning_signal(base, {**better, "run_status": "failed"})[
                "classification"
            ],
            "incomplete",
        )
        self.assertEqual(
            runner.learning_signal(base, {**better, "missing_requests": 1})[
                "classification"
            ],
            "incomplete",
        )

    def test_export_pair_keeps_native_control_quantization_and_all_terminal_ids(self):
        root = self.root
        native = root / "native"
        token_config = write(native / "tokenizer_config.json", {"eos_token": "<eos>"})
        tokens = write(
            native / "tokenizer.json",
            {"added_tokens": [{"id": 1, "content": "<eos>", "special": True}]},
        )
        template = write(native / "chat_template.jinja", "template")
        anchor = {
            "id": "gemma26",
            "base_repository": runner.BASE_REPOSITORY,
            "base_revision": runner.BASE_REVISION,
            "tokenizer_path": str(native),
            "tokenizer_files_sha256": {
                "tokenizer_config.json": token_config["sha256"],
                "tokenizer.json": tokens["sha256"],
                "chat_template.jinja": template["sha256"],
            },
            "template_path": template["path"],
            "template_sha256": template["sha256"],
        }
        old_registry = {
            "runtime": {"revision": runner.RUNTIME_REVISION},
            "tokenizers": {},
            "candidates": [anchor],
        }
        old_path = Path(write(root / "old-registry.json", old_registry)["path"])
        registry = {**old_registry, "candidates": []}
        artifacts = {}
        for role, key in runner.ARTIFACT_KEYS.items():
            item = write(root / "export" / f"{key}.gguf", role)
            item["bytes"] = Path(item["path"]).stat().st_size
            artifacts[key] = item
            registry["candidates"].append(
                {
                    **anchor,
                    "id": role,
                    "tokenizer_id": "gemma26",
                    "model_path": item["path"],
                    "model_sha256": item["sha256"],
                }
            )
        new_registry = Path(write(root / "registry.json", registry)["path"])
        pin = write(root / "pin.json", {})
        train = write(root / "train.json", {"configuration_sha256": "config"})
        tensor = write(
            root / "export/tensor-verification.json",
            {
                "status": "passed",
                "template_sha256": template["sha256"],
                "eos_token_id": 1,
                "native_generation_stop_ids": [1, 106, 50],
            },
        )
        quantization = {
            "quantization": "Q4_K_M",
            "importance_matrix": None,
            "threads": 4,
            "artifacts": artifacts,
            "quantizer_fallbacks": {
                "base": ["shape fallback"],
                "merged": ["shape fallback"],
            },
            "commands": {
                key: {
                    suffix: write(root / "logs" / f"{key}.{suffix}", {})
                    for suffix in ("command.json", "result.json", "log")
                }
                for key in ("base", "merged")
            },
        }
        quant = write(root / "export/quantization.json", quantization)
        receipt = {
            "version": "passive-gemma-sft-export-v1",
            "status": "tensor_verified_paired_export_runtime_preflight_pending",
            "base_repository": runner.BASE_REPOSITORY,
            "base_revision": runner.BASE_REVISION,
            "runtime_revision": runner.RUNTIME_REVISION,
            "native_pin": pin,
            "training_record": train,
            "artifacts": artifacts,
            "native_generation_stop_ids": [1, 106, 50],
            "tensor_verification_sha256": tensor["sha256"],
            "quantization_sha256": quant["sha256"],
            "quantizer_fallbacks": quantization["quantizer_fallbacks"],
            "quantized_tensor_type_census": {
                key: {"Q4_K": 10, "Q5_0": 3} for key in artifacts
            },
        }
        receipt_path = Path(write(root / "export/receipt.json", receipt)["path"])
        registration = {
            "paths": {
                "new_registry": new_registry,
                "registry": old_path,
                "new_export_receipt": receipt_path,
            },
            "inputs": {},
            "execution": {
                "candidate_ids": {"native_base": "native_base", "sft": "sft"}
            },
        }
        config = {
            "generation": {"eos_token_id": [1, 106, 50]},
            "bindings": {"native_pin": pin},
        }
        with (
            patch.object(runner, "validate_training", return_value=config),
            patch.object(runner.probe, "inspect_runtime"),
        ):
            runner.validate_pair(registration)
            registry["candidates"][0]["model_path"] = str(
                root / "historical-vendor-qat.gguf"
            )
            write(new_registry, registry)
            with self.assertRaisesRegex(ValueError, "corresponding member"):
                runner.validate_pair(registration)
            registry["candidates"][0]["model_path"] = artifacts["base_quantized"][
                "path"
            ]
            write(new_registry, registry)
            receipt["native_generation_stop_ids"] = [1, 106]
            write(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "terminal IDs"):
                runner.validate_pair(registration)

    def test_server_exception_retains_write_once_failure_receipt(self):
        registration, _, _ = self.training_evidence()
        for key in ("execution_new", "new_export_receipt", "new_registry"):
            value = write(self.root / f"{key}.json", {})
            registration["paths"][key] = Path(value["path"])
            registration["inputs"][Path(value["path"])] = value["sha256"]
        registration.update(
            args=runner.anchor_args(
                {
                    "paths": {
                        **registration["paths"],
                        "registry": self.root / "unused-registry.json",
                        "protocol": self.root / "unused-protocol.json",
                        "execution": self.root / "unused-execution.json",
                        "prefix_manifest": self.root / "unused-prefix.json",
                    }
                },
                "full",
            ),
            prepared={"cases": [], "prefixes": {}},
            binding={"version": runner.VERSION},
            execution={"candidate_ids": {"native_base": "base", "sft": "adapted"}},
        )
        registration["args"].release = self.root / "release"
        output = self.root / "failed"

        def failure(args, prepared):
            args.output.mkdir()
            write(args.output / "manifest.json", {"status": "failed"})
            raise TimeoutError("synthetic startup timeout")

        with patch.object(runner.probe, "run", side_effect=failure) as called:
            with self.assertRaises(TimeoutError):
                runner.run_condition(registration, "sft-full", output, "screen", "test")
            receipt = runner.load_json(output / "gemma-receipt.json")
            self.assertEqual(receipt["status"], "failed")
            self.assertIn("source-execution.jsonl", receipt["files_sha256"])
            with self.assertRaisesRegex(ValueError, "write-once"):
                runner.run_condition(registration, "sft-full", output, "screen", "test")
            self.assertEqual(called.call_count, 1)

    def test_actual_protocol_load_is_read_only_and_hardware_amendment_is_explicit(self):
        path = (
            runner.ROOT
            / "artifacts/benchmarks/passive-t1-gemma26-v1/protocol-attempt-02.json"
        )
        if not path.is_file():
            self.skipTest("Requires the local Gemma benchmark protocol.")
        before = path.read_bytes()
        registration = runner.load_protocol(path)
        self.assertEqual(
            registration["inputs"][path],
            "51c4ec869b5094260918f2a14c6485af398d79bada6016993c0be9e536188417",
        )
        self.assertEqual(
            registration["protocol"]["engineering_smoke"]["native_gpu"],
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        )
        self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
