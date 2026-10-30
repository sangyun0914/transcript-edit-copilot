"""Synthetic loaded-adapter, raw HTTP and selection checks; no model or network."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_small_final_freeze_actions_v3 as builder
import evaluate_gemma_final as development
import gemma_final_deployment as deployed
import gemma_final_freeze as freeze
import gemma_final_test_runtime as gemma
import infer_deployed_task_test_actions_v3 as auxiliary
from test_infer_deployed_task_test_actions_v3 import capture


class GemmaDeployedTests(unittest.TestCase):
    def setUp(self):
        (gemma.ROOT / "artifacts").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="gemma-deployed-fixture-", dir=gemma.ROOT / "artifacts"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for module, name, filename in (
            (builder, "UI_AMENDMENT", "ui-plan.md"),
            (deployed, "PLAN", "deployment-plan.md"),
            (deployed.q8, "PLAN", "q8-plan.md"),
        ):
            path = self.root / filename
            path.write_text("Synthetic plan for deployment binding tests only.\n")
            patch = mock.patch.object(module, name, path)
            patch.start()
            self.addCleanup(patch.stop)
        for name in ("base.gguf", "adapter.gguf", "chat_template.jinja"):
            (self.root / name).write_text("Engineering fixture; never loaded: " + name)
        self.gateway = {
            "version": 2,
            "contract_version": "passive-nep-actions-v1",
            "endpoint": "http://127.0.0.1:1",
            "model_id": "synthetic-gemma",
            "model_path": str(self.root / "base.gguf"),
            "model_sha256": gemma.sha(self.root / "base.gguf"),
            "adapter_path": str(self.root / "adapter.gguf"),
            "adapter_sha256": gemma.sha(self.root / "adapter.gguf"),
            "adapter_scale": 1.0,
            "output_format": "json",
            "chat_template_path": str(self.root / "chat_template.jinja"),
            "chat_template_sha256": gemma.sha(self.root / "chat_template.jinja"),
            "runtime_build": "synthetic-build",
            "cache_prompt": True,
            "capture_runtime_evidence": False,
        }
        self.loaded = [{"id": 0, "path": self.gateway["adapter_path"], "scale": 1.0}]

    def test_real_loaded_adapter_validator_keeps_id_path_scale_and_ordinary_lora(self):
        deployed.verify_loaded_adapter(self.gateway, self.loaded)
        self.assertEqual(
            auxiliary.request_settings(self.gateway)["lora"], [{"id": 0, "scale": 1.0}]
        )
        self.assertEqual(
            auxiliary.request_settings({**self.gateway, "adapter_path": None})["lora"],
            [],
        )
        for loaded in (
            [],
            self.loaded * 2,
            [{**self.loaded[0], "id": 1}],
            [{**self.loaded[0], "scale": 0.5}],
            [{**self.loaded[0], "path": self.gateway["model_path"]}],
            [{**self.loaded[0], "alora_invocation_tokens": [2]}],
        ):
            with (
                self.subTest(loaded=loaded),
                self.assertRaisesRegex(ValueError, "ordinary LoRA"),
            ):
                deployed.verify_loaded_adapter(self.gateway, loaded)

    def test_generated_final_policy_binds_actual_separate_files_and_export_receipts(
        self,
    ):
        config = {
            "tokenizer": {"template_sha256": self.gateway["chat_template_sha256"]}
        }
        configuration = self.save("configuration.json", config)
        tokenizer = {"engineering_fixture": True}
        metadata = self.save("metadata.json", {"tokenizer": tokenizer})
        model = {
            "runtime_family": gemma.FAMILY,
            "role": "sft",
            "identity": {
                "kind": gemma.evidence.sft.VERSION,
                "adapter_sha256": "native-adapter",
            },
            "inference_configuration": configuration,
            "preparation_manifest": metadata,
        }
        model_reference = self.save("native-model.json", model)
        directory = self.root / "deployment"
        gemma.write(directory / "gateway.json", self.gateway)
        binary = self.root / "synthetic-server"
        binary.write_text("Engineering fixture; never executed.")
        command = [
            str(binary),
            "-m",
            self.gateway["model_path"],
            "--lora",
            self.gateway["adapter_path"],
            "-c",
            "4096",
        ]
        gemma.write(directory / "command.json", command)
        evidence = self.save("export-evidence.json", {"engineering_fixture": True})
        report = {
            "adapter_export_receipt": evidence,
            "base_export_receipt": evidence,
            "base_registry": evidence,
            "runtime_files_sha256": {
                str(binary.relative_to(gemma.ROOT)): gemma.sha(binary)
            },
            "numerical_regime": "native_derived_q4_base_fp32_lora",
        }
        gemma.write(directory / "deployment.json", report)
        context = {
            "registration": {
                "models": {"sft": model_reference},
                "deployment_evaluation_sources_sha256": auxiliary.sources(),
            },
            "ready": {
                "protocol_id": "synthetic-only",
                "selected_deployment_receipt": gemma.binding(
                    directory / "deployment.json"
                ),
            },
            "models": {"sft": model},
            "selected_model": "sft",
            "registration_binding": evidence,
            "conditions": {"conditions": builder.gate.conditions.condition_specs([])},
        }
        native_run = {
            "id": "sft-json",
            "role": "sft",
            "format": "json",
            "condition": context["conditions"]["conditions"]["canonical_full_context"],
        }
        with (
            mock.patch.object(
                deployed,
                "deployment_check",
                return_value=(self.gateway, command, report),
            ),
            mock.patch.object(deployed, "validate_deployment"),
            mock.patch.object(gemma, "tokenizer_identity", return_value=tokenizer),
        ):
            for regime in (deployed.q8.Q4_REGIME, deployed.q8.REGIME):
                report["numerical_regime"] = regime
                policy = builder.deployed_policy(context)
                self.assertEqual(policy["numerical_regime"], regime)
                self.assertEqual(
                    policy["plan"],
                    gemma.binding(
                        deployed.q8.PLAN
                        if regime == deployed.q8.REGIME
                        else deployed.PLAN
                    ),
                )
                self.assertEqual(
                    policy["adapter_file"], gemma.binding(self.gateway["adapter_path"])
                )
                self.assertEqual(
                    policy["request_settings"]["lora"], [{"id": 0, "scale": 1.0}]
                )
                self.assertEqual(
                    auxiliary.validate_policy(
                        policy, model, native_run, config, tokenizer
                    ),
                    self.gateway,
                )
            policy["adapter_file"] = policy["model_file"]
            with self.assertRaisesRegex(ValueError, "file identity changed"):
                auxiliary.validate_policy(policy, model, native_run, config, tokenizer)

    def transport(self, endpoint, route, body=None, timeout=30):
        self.assertEqual(endpoint, self.gateway["endpoint"])
        if route == "/health":
            return capture({"status": "ok"})
        if route == "/props":
            return capture(
                {
                    "model_alias": self.gateway["model_id"],
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
            return capture(self.loaded)
        if route == "/apply-template":
            return capture({"prompt": json.dumps(body["messages"])})
        if route == "/tokenize":
            return capture({"tokens": [2, 20, 106]})
        self.assertEqual(route, "/v1/chat/completions")
        self.assertEqual(body["lora"], [{"id": 0, "scale": 1.0}])
        return capture(
            {
                "model": self.gateway["model_id"],
                "usage": {"completion_tokens": 8},
                "choices": [
                    {
                        "message": {"content": '{"action":"no_action"}'},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    def test_dev_http_evidence_retains_separate_lora_and_all_startup_failures(self):
        evaluation = self.root / "evaluation"
        requests = [
            {
                "case_id": f"engineering-dev-{i}",
                "split": "dev",
                "format": "json",
                "input_sha256": "input",
                "messages_sha256": "messages",
                "prefix_ids_sha256": "prefix",
                "prefix_ids": [2, 20, 106],
                "messages": [{"role": "user", "content": "Engineering fixture"}],
            }
            for i in range(2)
        ]
        gemma.evidence.prep.write_rows(evaluation / "requests-json.jsonl", requests)
        gemma.write(evaluation / "manifest.json", {"engineering_fixture": True})
        gemma.write(self.root / "model.json", {"engineering_fixture": True})
        gemma.write(self.root / "deployment.json", {"engineering_fixture": True})
        model = {
            "identity": {
                "kind": gemma.evidence.sft.VERSION,
                "adapter_sha256": "native-sft",
            }
        }
        context = {
            "gateway": self.gateway,
            "policy": {
                "context_tokens": 4096,
                "request_settings": auxiliary.request_settings(self.gateway),
            },
            "model": model,
            "identity": deployed.product_identity(self.gateway, model),
            "requests": requests,
        }
        with (
            mock.patch.object(development, "deployed_context", return_value=context),
            mock.patch.object(auxiliary.http, "request", side_effect=self.transport),
        ):
            result = development.infer_deployed(
                evaluation,
                self.root / "model.json",
                self.root / "deployment.json",
                self.root / "success",
            )
            self.assertEqual(result["observed"], 2)
            rows = gemma.evidence.prep.rows(self.root / "success/observations.jsonl")
            self.assertTrue(all(r["status"] == "completed" for r in rows))
            rows[0]["http_evidence"]["completion"]["request"]["lora"] = []
            path = self.root / "success/observations.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            gemma.evidence.sft.manifest(
                path.parent, "inference_complete", development.VERSION
            )
            with self.assertRaisesRegex(ValueError, "different request or LoRA"):
                development.verify_deployed_run(evaluation, path.parent)
            self.loaded = []
            result = development.infer_deployed(
                evaluation,
                self.root / "model.json",
                self.root / "deployment.json",
                self.root / "startup-failed",
            )
            self.assertEqual(result["requested"], result["observed"])
            rows = gemma.evidence.prep.rows(
                self.root / "startup-failed/observations.jsonl"
            )
            self.assertTrue(
                all(
                    r["status"] == "request_failure" and not r["http_evidence"]
                    for r in rows
                )
            )

    def quality_fixture(self):
        config = {"engineering_fixture": True}
        gemma.write(self.root / "config.json", config)
        refs = [
            self.save(f"failed-{i}.json", {"engineering_fixture": True, "candidate": i})
            for i in range(3)
        ]
        gemma.write(
            self.root / "protocol.json",
            {
                "completed_small_failure_summaries": refs[:2],
                "completed_lr_recovery_failure_summary": refs[2],
            },
        )
        native = {
            "kind": gemma.evidence.sft.VERSION,
            "adapter_sha256": "native-adapter",
        }
        model = {
            "runtime_family": gemma.FAMILY,
            "identity": native,
            "inference_configuration": gemma.binding(self.root / "config.json"),
            "inference_policy": self.save(
                "native-policy.json", {"native_experts_overlay": None}
            ),
        }
        deployment = {"numerical_regime": "native_derived_q4_base_fp32_lora"}
        deployed_reference = self.save("deployment.json", deployment)
        report = {
            "schema_constrained": True,
            "numerical_regime": deployment["numerical_regime"],
            "inference_run": self.save(
                "product-run.json", {"deployment": deployed_reference}
            ),
        }
        identity = {
            key: self.gateway[key]
            for key in ("model_id", "model_sha256", "adapter_sha256", "adapter_scale")
        }
        stats = {
            "precision": {"value": 0.96, "numerator": 96, "denominator": 100},
            "recall": {"value": 0.90, "numerator": 90, "denominator": 100},
            "model_preservation": {"value": 0.95, "numerator": 68},
            "status_counts": {"completed": 471},
        }
        summary = {
            "model_identity": native,
            "overall": stats,
            "cohorts": {
                k: {"metrics": copy.deepcopy(stats)}
                for k in ("canonical_dev322", "supplemental_dev149")
            },
            "minimum_numeric_quality_gate": {"passed": True},
            "macro_action_f1": {"value": 0.90},
            "cohort_mapping": refs[0],
        }
        product = copy.deepcopy(summary)
        product["model_identity"] = deployed.product_identity(self.gateway, model)
        native_ref, product_ref = (
            self.save("native-summary.json", summary),
            self.save("product-summary.json", product),
        )
        review = self.save("review.json", {"engineering_fixture": True})
        quality = {
            "native_sft_summary": native_ref,
            "selected_native_summary": native_ref,
            "selected_deployed_summary": product_ref,
            "subgroup_review": {
                "status": "reviewed_pass",
                "rationale": "Engineering fixture only.",
                "evidence": [review],
            },
        }
        selection = {
            "evidence": [*refs, native_ref, product_ref, review],
            "quality_without_ui": quality,
        }
        values = {
            native_ref["sha256"]: (
                summary,
                {
                    "schema_constrained": False,
                    "numerical_regime": "native_bfloat16_with_optional_float32_lora",
                    "inference_run": self.save(
                        "native-run.json", {"runtime_overlay": None}
                    ),
                },
            ),
            product_ref["sha256"]: (product, report),
        }
        return selection, {"sft": model}, config, deployment, identity, values

    def save(self, name, value):
        gemma.write(self.root / name, value)
        return gemma.binding(self.root / name)

    def test_gemma_pending_selection_keeps_three_failures_quality_and_actual_product_identity(
        self,
    ):
        selection, models, config, deployment, identity, values = self.quality_fixture()
        with (
            mock.patch.object(gemma, "verify_fallback", return_value=(config, {})),
            mock.patch.object(
                development,
                "verify_summary",
                side_effect=lambda ref: values[ref["sha256"]],
            ),
        ):

            def call():
                return builder.quality_without_ui(
                    selection, models, "sft", config, deployment, identity
                )

            self.assertEqual(
                call(), "registered_fallback_after_three_completed_native_failures"
            )
            failure = selection["evidence"].pop(2)
            with self.assertRaisesRegex(ValueError, "LR recovery failure"):
                call()
            selection["evidence"].append(failure)
            product = values[
                selection["quality_without_ui"]["selected_deployed_summary"]["sha256"]
            ][0]
            product["model_identity"]["adapter_scale"] = 0.5
            with self.assertRaisesRegex(ValueError, "separate adapter, scale"):
                call()
            product["model_identity"]["adapter_scale"] = 1.0
            product["minimum_numeric_quality_gate"]["passed"] = False
            with self.assertRaisesRegex(ValueError, "unchanged full DEV gates"):
                call()
            product["minimum_numeric_quality_gate"]["passed"] = True
            product["cohorts"]["supplemental_dev149"]["metrics"]["recall"]["value"] = (
                0.87
            )
            product["cohorts"]["supplemental_dev149"]["metrics"]["recall"][
                "numerator"
            ] = 87
            with self.assertRaisesRegex(ValueError, "native drop allowance"):
                call()

    def test_gemma_pending_ui_remains_explicitly_incomplete_and_cannot_claim_success(
        self,
    ):
        selection, models, config, deployment, identity, values = self.quality_fixture()
        selection_reference = self.save("selection.json", selection)
        deployment_reference = gemma.binding(self.root / "deployment.json")
        ui = builder.pending_ui_template()
        ui.update(
            status="pending_product_validation",
            deployment=deployment_reference,
            model=identity,
            selection_receipt=selection_reference,
            reason="Engineering fixture: desktop unavailable.",
            evidence=[selection["evidence"][0]],
        )
        reference = self.save("pending-ui.json", ui)
        ready = {
            "selected_model_ui_receipt": reference,
            "selected_deployment_receipt": deployment_reference,
            "evidence_review": {
                "attestations": {
                    key: key not in builder.UI_ATTESTATIONS
                    for key in builder.legacy.inputs.ATTESTATIONS
                }
            },
        }
        with (
            mock.patch.object(gemma, "verify_fallback", return_value=(config, {})),
            mock.patch.object(
                development,
                "verify_summary",
                side_effect=lambda ref: values[ref["sha256"]],
            ),
        ):
            result = builder.validate_product_validation(
                ready,
                {"selection_receipt": selection_reference},
                models,
                "sft",
                config,
                identity,
            )
            self.assertEqual(result["status"], "pending_product_validation")
            self.assertFalse(result["product_completion_asserted"])
            ui["product_complete"] = True
            changed = self.save("false-ui-success.json", ui)
            with self.assertRaisesRegex(ValueError, "without UI success claims"):
                builder.validate_product_validation(
                    {**ready, "selected_model_ui_receipt": changed},
                    {"selection_receipt": selection_reference},
                    models,
                    "sft",
                    config,
                    identity,
                )

    def test_native_readiness_uses_actual_run_and_loading_receipts(self):
        training = self.root / "training"
        config = {
            "libraries": {"python": "synthetic"},
            "absent_libraries": ["torchao"],
            "generation": {"eos_token_id": [1, 106, 50]},
            "model": {
                "gpu_name_contains": "Engineering GPU",
                "minimum_gpu_bytes": 75_000_000_000,
                "target_modules": ["synthetic.q_proj"],
            },
        }
        environment = {
            "matched": True,
            "differences": {},
            "actual": {"python": "synthetic", "torchao": None},
        }
        gemma.write(training / "run.json", {"environment": environment})
        gemma.write(training / "manifest.json", {"engineering_fixture": True})
        runtime = {
            "gpu": "Engineering GPU",
            "gpu_total_bytes": 80_000_000_000,
            "parameter_dtypes": ["torch.bfloat16"],
            "numerical_regime": "native_bfloat16_lora_float32",
            "text_attn_implementation": "sdpa",
            "text_experts_implementation": "eager",
            "targets": {"synthetic.q_proj": {}},
            "effective_generation": config["generation"],
            "completion_label_eos_token_id": 1,
            "ignored_native_tensor_names": [],
            "loading_info": {},
        }
        gemma.write(training / "runtime.json", runtime)
        ready = {
            "native_environment_receipt": gemma.binding(training / "run.json"),
            "native_runtime_receipt": gemma.binding(training / "runtime.json"),
        }
        models = {
            "sft": {"artifact_manifest": gemma.binding(training / "manifest.json")},
            "native_base": {"artifact_manifest": ready["native_runtime_receipt"]},
        }
        freeze.validate_native_readiness(
            ready, models, config, {"environment": environment}
        )
        with self.assertRaisesRegex(ValueError, "verified training run"):
            freeze.validate_native_readiness(
                {
                    **ready,
                    "native_environment_receipt": ready["native_runtime_receipt"],
                },
                models,
                config,
                {"environment": environment},
            )
        runtime["loading_info"] = {"missing_keys": ["synthetic.q_proj"]}
        (training / "runtime.json").write_text(json.dumps(runtime))
        ready["native_runtime_receipt"] = models["native_base"]["artifact_manifest"] = (
            gemma.binding(training / "runtime.json")
        )
        with self.assertRaisesRegex(ValueError, "native loading differs"):
            freeze.validate_native_readiness(
                ready, models, config, {"environment": environment}
            )


if __name__ == "__main__":
    unittest.main()
