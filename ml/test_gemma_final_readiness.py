import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import gemma_final_readiness as final
import gemma_final_freeze as freeze
import prepare_gemma_final_dpo as mining
import train_gemma_actions_v3_dpo as dpo
from test_gemma_actions_v3 import cases, Tokenizer


class FinalGemmaTests(unittest.TestCase):
    def setUp(self):
        (final.ROOT / "artifacts").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="gemma-readiness-plans-", dir=final.ROOT / "artifacts"
        )
        self.addCleanup(temporary.cleanup)
        for module, name in (
            (final, "PLAN"),
            (freeze, "DESCRIPTIVE_PLAN"),
            (freeze, "PRODUCT_PLAN"),
        ):
            path = Path(temporary.name) / (name + ".md")
            path.write_text("Synthetic plan for metadata binding tests only.\n")
            patch = mock.patch.object(module, name, path)
            patch.start()
            self.addCleanup(patch.stop)

    def descriptive_fixture(self, role="sft"):
        temporary = tempfile.TemporaryDirectory(
            prefix="descriptive-selection-", dir=final.ROOT / "artifacts"
        )
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        def save(name, value):
            final.write(root / name, value)
            return final.binding(root / name)

        evidence = save("evidence.json", {"engineering_fixture_only": True})
        config = {"engineering_fixture_only": True}
        configuration = save("configuration.json", config)
        save(
            "protocol.json",
            {
                "completed_small_failure_summaries": [evidence, evidence],
                "completed_lr_recovery_failure_summary": evidence,
            },
        )
        parent_identity = {"kind": "gemma-actions-v3-sft", "adapter_sha256": "parent"}
        selected_identity = (
            parent_identity
            if role == "sft"
            else {"kind": "gemma-actions-v3-dpo", "adapter_sha256": "dpo"}
        )
        models = {
            "sft": {
                "identity": parent_identity,
                "inference_configuration": configuration,
            }
        }
        models[role] = {
            "identity": selected_identity,
            "inference_configuration": configuration,
            "runtime_family": freeze.gemma.FAMILY,
        }
        policy = save("inference-policy.json", {"engineering_fixture_only": True})
        for model in models.values():
            model["inference_policy"] = policy
        stats = {
            "precision": {"value": 0.95, "numerator": 95, "denominator": 100},
            "recall": {"value": 0.90, "numerator": 90, "denominator": 100},
            "model_preservation": {"value": 0.95, "numerator": 68},
            "status_counts": {"completed": 471},
        }
        parent = {
            "model_identity": parent_identity,
            "minimum_numeric_quality_gate": {"passed": True},
            "overall": copy.deepcopy(stats),
            "macro_action_f1": {"value": 0.90},
            "cohort_mapping": evidence,
            "cohorts": {
                name: {"metrics": copy.deepcopy(stats)}
                for name in ("canonical_dev322", "supplemental_dev149")
            },
        }
        native = copy.deepcopy(parent)
        native.update(
            model_identity=selected_identity,
            macro_action_f1={"value": 0.90 if role == "sft" else 0.91},
        )
        native_report = {
            "numerical_regime": "native_bfloat16_with_optional_float32_lora",
            "schema_constrained": False,
            "inference_run": save(
                "native-run.json", {"engineering_fixture_only": True}
            ),
        }
        deployment = {"numerical_regime": "native_derived_q4_base_fp32_lora"}
        deployment_ref = save("deployment.json", deployment)
        identity = {
            "model_id": "fixture",
            "model_sha256": "base",
            "adapter_sha256": "lora",
        }
        product = copy.deepcopy(native)
        product["model_identity"] = {
            **identity,
            "runtime_family": "gemma26-final-separate-lora-gguf",
            "training_adapter_sha256": selected_identity["adapter_sha256"],
            "native_identity_sha256": freeze.digest(selected_identity),
            "kind": selected_identity["kind"],
        }
        product_report = {
            **deployment,
            "schema_constrained": True,
            "inference_run": save("product-run.json", {"deployment": deployment_ref}),
        }
        parent_ref = save("parent-summary.json", parent)
        native_ref = (
            parent_ref if role == "sft" else save("native-summary.json", native)
        )
        product_ref = save("product-summary.json", product)
        summaries = {
            parent_ref["sha256"]: (parent, native_report),
            native_ref["sha256"]: (native, native_report),
            product_ref["sha256"]: (product, product_report),
        }
        audit = save(
            "audit.json",
            {
                "model_identity": selected_identity,
                "TEST_bodies_read": False,
                "bindings": [native_ref],
            },
        )
        review = {
            "status": "reviewed_fail",
            "rationale": "The registered unknown-speaker group fails 0/2.",
            "evidence": [audit],
            "registered_no_entire_new_subgroup_failure_passed": False,
            "reserved_TEST_bodies_read": False,
            "failed_registered_typed_groups": ["C_contract/speaker_unknown"],
            "registered_typed_group_results": [
                {
                    "registered_group": "C_contract/speaker_unknown",
                    "n": 2,
                    "outcome_matches": 0,
                    "all_failed": True,
                }
            ],
        }
        review_ref = save("review.json", review)
        contract = {
            "version": freeze.DESCRIPTIVE_VERSION,
            "status": "user_selected_descriptive_evaluation",
            "actor": "/root",
            "selected_model": role,
            "selected_identity_sha256": freeze.digest(selected_identity),
            "subgroup_review": review_ref,
            "subgroup_audit": audit,
            "user_selection_evidence": evidence,
            "plan": freeze.binding(freeze.DESCRIPTIVE_PLAN),
            "strict_subgroup_gate_passed": False,
            "failed_registered_typed_groups": review["failed_registered_typed_groups"],
            "limitations": [
                "Unknown-speaker preservation failed 0/2; original strict gate remains failed."
            ],
            "quality_success_claimed": False,
            "final_test_bodies_read": False,
            "body_access_authorized": False,
        }
        selection = {
            "quality_without_ui": {
                "native_sft_summary": parent_ref,
                "selected_native_summary": native_ref,
                "selected_deployed_summary": product_ref,
                "subgroup_review": review,
            },
            "selection_contract": contract,
            "evidence": [
                evidence,
                parent_ref,
                native_ref,
                product_ref,
                audit,
                review_ref,
                contract["plan"],
            ],
        }
        patch = mock.patch.object(
            freeze.gemma, "verify_fallback", return_value=(config, {})
        )
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(
            freeze.development,
            "verify_summary",
            side_effect=lambda ref: summaries[ref["sha256"]],
        )
        patch.start()
        self.addCleanup(patch.stop)
        return root, selection, models, config, deployment, identity, summaries

    def product_descriptive_fixture(self):
        root, selection, models, config, deployment, identity, values = (
            self.descriptive_fixture()
        )
        quality = selection["quality_without_ui"]
        native = values[quality["selected_native_summary"]["sha256"]][0]
        product, product_report = values[quality["selected_deployed_summary"]["sha256"]]
        product["cohorts"]["supplemental_dev149"]["metrics"]["recall"].update(
            value=0.85, numerator=85
        )
        final.write(root / "product-failed-summary.json", product)
        quality["selected_deployed_summary"] = final.binding(
            root / "product-failed-summary.json"
        )
        values[quality["selected_deployed_summary"]["sha256"]] = (
            product,
            product_report,
        )
        checks = freeze.product_drop_checks(native, product)
        final.write(
            root / "drop-audit.json", {"drop_checks": checks, "drop_gate_passed": False}
        )
        evidence = selection["selection_contract"]["user_selection_evidence"]
        decision = {
            "version": "gemma26-root-final-product-selection-1",
            "actor": "/root",
            "status": "descriptive_product_selected_registered_drop_gate_failed",
            "selected_model": "sft",
            "selected_numerical_regime": freeze.q8.Q4_REGIME,
            "adapter_sha256": models["sft"]["identity"]["adapter_sha256"],
            "plan": freeze.binding(freeze.PRODUCT_PLAN),
            "native_selection": evidence,
            "selected_native_summary": quality["selected_native_summary"],
            "selected_deployed_summary": quality["selected_deployed_summary"],
            "deployment": final.binding(root / "deployment.json"),
            "registered_drop_audit": final.binding(root / "drop-audit.json"),
            "Q8_audit": evidence,
            "Q8_summary": evidence,
            "user_selection_evidence": evidence,
            "limitations": [
                "Synthetic supplemental recall exceeds the registered drop; no quality-success claim."
            ],
            "registered_product_drop_gate_passed": False,
            "quality_success_claimed": False,
            "final_test_bodies_read": False,
            "body_access_authorized": False,
            "no_additional_training_model_or_quantization_experiments": True,
        }
        path = root / "root-product-decision.json"
        final.write(path, decision)
        for name, value in (
            ("PRODUCT_DECISION", path),
            ("PRODUCT_DECISION_SHA", final.sha(path)),
        ):
            patch = mock.patch.object(freeze, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        contract = freeze.product_contract(models["sft"])
        selection["product_selection_contract"] = contract
        for key in (
            "root_decision",
            "deployment",
            "plan",
            "user_selection_evidence",
            "selected_native_summary",
            "selected_deployed_summary",
            "registered_drop_audit",
        ):
            if contract[key] not in selection["evidence"]:
                selection["evidence"].append(contract[key])
        return root, selection, models, config, deployment, identity, values

    def test_product_descriptive_contract_retains_all_checks_and_default_still_rejects(
        self,
    ):
        _, selection, models, config, deployment, identity, _ = (
            self.product_descriptive_fixture()
        )

        def call(value):
            return freeze.quality_without_ui(
                value, models, "sft", config, deployment, identity
            )

        without = {
            key: value
            for key, value in selection.items()
            if key != "product_selection_contract"
        }
        with self.assertRaisesRegex(ValueError, "native drop allowance"):
            call(without)
        self.assertEqual(call(selection), freeze.DESCRIPTIVE_BASIS)
        contract = selection["product_selection_contract"]
        self.assertEqual(len(contract["drop_checks"]), 9)
        self.assertEqual(
            contract["failed_product_drop_checks"], ["supplemental_dev149:recall"]
        )
        for field, value in (
            ("selected_native_identity", {}),
            ("selected_identity_sha256", "other"),
            ("deployment", contract["selected_native_summary"]),
            ("failed_product_drop_checks", []),
            ("drop_checks", contract["drop_checks"][:-1]),
            ("registered_product_drop_gate_passed", True),
            ("quality_success_claimed", True),
            ("final_test_bodies_read", True),
            ("body_access_authorized", True),
            ("selected_deployed_summary", contract["selected_native_summary"]),
            ("user_selection_evidence", contract["registered_drop_audit"]),
        ):
            changed = copy.deepcopy(selection)
            changed["product_selection_contract"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                call(changed)
        missing = copy.deepcopy(selection)
        missing["evidence"].remove(contract["root_decision"])
        with self.assertRaisesRegex(ValueError, "descriptive product evidence"):
            call(missing)
        with self.assertRaisesRegex(ValueError, "selected model"):
            freeze.product_contract(models["sft"], "sft_dpo")

    def test_product_descriptive_contract_does_not_relax_minimum_or_native_subgroup_gates(
        self,
    ):
        root, selection, models, config, deployment, identity, values = (
            self.product_descriptive_fixture()
        )
        product = values[
            selection["quality_without_ui"]["selected_deployed_summary"]["sha256"]
        ][0]
        product["minimum_numeric_quality_gate"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "unchanged full DEV gates"):
            freeze.quality_without_ui(
                selection, models, "sft", config, deployment, identity
            )
        product["minimum_numeric_quality_gate"]["passed"] = True
        strict_native = {
            key: value
            for key, value in selection.items()
            if key != "selection_contract"
        }
        with self.assertRaisesRegex(ValueError, "subgroup selection gate failed"):
            freeze.quality_without_ui(
                strict_native, models, "sft", config, deployment, identity
            )
        decision = final.read(root / "root-product-decision.json")
        decision["quality_success_claimed"] = True
        (root / "root-product-decision.json").write_text(json.dumps(decision))
        with self.assertRaisesRegex(ValueError, "exact root product decision"):
            freeze.quality_without_ui(
                selection, models, "sft", config, deployment, identity
            )

    def test_product_descriptive_failed_goal_is_retained_with_actual_ui_readiness(self):
        import build_small_final_freeze_actions_v3 as builder

        root, selection, models, config, deployment, identity, _ = (
            self.product_descriptive_fixture()
        )
        final.write(root / "selection.json", selection)
        evidence = selection["selection_contract"]["user_selection_evidence"]
        ready = dict.fromkeys(builder.EVIDENCE, evidence)
        ready["selected_deployment_receipt"] = final.binding(root / "deployment.json")
        with (
            mock.patch.object(builder, "validate_deployment", return_value=identity),
            mock.patch.object(freeze, "validate_native_readiness"),
            mock.patch.object(
                builder,
                "validate_product_validation",
                return_value={
                    "status": "observed",
                    "selection_basis": "actual_ui_evidence_observed",
                },
            ),
        ):
            result = builder.validate_readiness_evidence(
                ready,
                {"selection_receipt": final.binding(root / "selection.json")},
                models,
                "sft",
                config,
                {},
            )
        self.assertFalse(result["registered_product_drop_gate_passed"])
        self.assertFalse(result["quality_success_claimed"])
        self.assertFalse(result["development_selection"]["strict_subgroup_gate_passed"])
        self.assertEqual(
            result["product_development_selection"],
            selection["product_selection_contract"],
        )

    def test_descriptive_selection_retains_failure_and_default_still_rejects_it(self):
        _, selection, models, config, deployment, identity, _ = (
            self.descriptive_fixture()
        )
        original = copy.deepcopy(selection)
        strict = {
            key: value
            for key, value in selection.items()
            if key != "selection_contract"
        }
        with self.assertRaisesRegex(ValueError, "subgroup selection gate failed"):
            freeze.quality_without_ui(
                strict, models, "sft", config, deployment, identity
            )
        self.assertEqual(
            freeze.quality_without_ui(
                selection, models, "sft", config, deployment, identity
            ),
            freeze.DESCRIPTIVE_BASIS,
        )
        self.assertEqual(selection, original)
        for key, value in (
            ("selected_identity_sha256", "other"),
            ("strict_subgroup_gate_passed", True),
            ("body_access_authorized", True),
            ("quality_success_claimed", True),
            ("limitations", []),
            ("failed_registered_typed_groups", []),
        ):
            changed = copy.deepcopy(selection)
            changed["selection_contract"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                freeze.quality_without_ui(
                    changed, models, "sft", config, deployment, identity
                )
        changed = copy.deepcopy(selection)
        changed["quality_without_ui"]["subgroup_review"]["status"] = "reviewed_pass"
        with self.assertRaisesRegex(ValueError, "exact failed subgroup review"):
            freeze.quality_without_ui(
                changed, models, "sft", config, deployment, identity
            )
        changed = copy.deepcopy(selection)
        changed["evidence"].remove(
            changed["selection_contract"]["user_selection_evidence"]
        )
        with self.assertRaisesRegex(ValueError, "descriptive decision evidence"):
            freeze.quality_without_ui(
                changed, models, "sft", config, deployment, identity
            )
        Path(selection["selection_contract"]["subgroup_review"]["path"]).write_text(
            "{}"
        )
        with self.assertRaises(ValueError):
            freeze.quality_without_ui(
                selection, models, "sft", config, deployment, identity
            )

    def test_final_selection_rejects_mixed_canonical_and_grouped_native_evidence(self):
        root, selection, models, _, _, _, values = self.descriptive_fixture("sft_dpo")
        parent_report = values[
            selection["quality_without_ui"]["native_sft_summary"]["sha256"]
        ][1]
        native_report = values[
            selection["quality_without_ui"]["selected_native_summary"]["sha256"]
        ][1]
        grouped_run = root / "grouped-run.json"
        final.write(
            grouped_run, {"runtime_overlay": freeze.gemma.native_overlay.OVERLAY}
        )
        grouped_report = {**native_report, "inference_run": final.binding(grouped_run)}
        with self.assertRaisesRegex(ValueError, "Canonical native selection"):
            freeze.verify_native_backend(
                models,
                "sft_dpo",
                selection["quality_without_ui"],
                parent_report,
                grouped_report,
            )
        with mock.patch.object(
            freeze.gemma,
            "native_policy_overlay",
            return_value={"settings": freeze.gemma.native_overlay.OVERLAY},
        ):
            with self.assertRaisesRegex(ValueError, "share the adopted native backend"):
                freeze.verify_native_backend(
                    models,
                    "sft_dpo",
                    selection["quality_without_ui"],
                    parent_report,
                    grouped_report,
                )

    def test_descriptive_selection_keeps_numeric_deployment_and_dpo_adoption_checks(
        self,
    ):
        _, selection, models, config, deployment, identity, values = (
            self.descriptive_fixture("sft_dpo")
        )
        quality = selection["quality_without_ui"]
        native = values[quality["selected_native_summary"]["sha256"]][0]
        product = values[quality["selected_deployed_summary"]["sha256"]][0]

        def call():
            return freeze.quality_without_ui(
                selection, models, "sft_dpo", config, deployment, identity
            )

        self.assertEqual(call(), freeze.DESCRIPTIVE_BASIS)
        product["minimum_numeric_quality_gate"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "unchanged full DEV gates"):
            call()
        product["minimum_numeric_quality_gate"]["passed"] = True
        product["model_identity"]["adapter_sha256"] = "wrong-adapter"
        with self.assertRaisesRegex(ValueError, "selected base, separate adapter"):
            call()
        product["model_identity"]["adapter_sha256"] = identity["adapter_sha256"]
        product["overall"]["precision"]["value"] = 0.92
        product["overall"]["precision"]["numerator"] = 92
        with self.assertRaisesRegex(ValueError, "native drop allowance"):
            call()
        product["overall"]["precision"]["value"] = 0.95
        product["overall"]["precision"]["numerator"] = 95
        native["macro_action_f1"]["value"] = 0.90
        with self.assertRaisesRegex(ValueError, "same-parent macro action F1"):
            call()

    def test_descriptive_failure_is_retained_in_readiness_with_actual_ui(self):
        import build_small_final_freeze_actions_v3 as builder

        root, selection, models, config, deployment, identity, _ = (
            self.descriptive_fixture()
        )
        final.write(root / "selection.json", selection)
        evidence = selection["selection_contract"]["user_selection_evidence"]
        ready = dict.fromkeys(builder.EVIDENCE, evidence)
        ready["selected_deployment_receipt"] = final.binding(root / "deployment.json")
        registration = {"selection_receipt": final.binding(root / "selection.json")}
        with (
            mock.patch.object(builder, "validate_deployment", return_value=identity),
            mock.patch.object(freeze, "validate_native_readiness"),
            mock.patch.object(
                builder,
                "validate_product_validation",
                return_value={
                    "status": "observed",
                    "selection_basis": "actual_ui_evidence_observed",
                },
            ),
        ):
            result = builder.validate_readiness_evidence(
                ready, registration, models, "sft", config, {}
            )
        self.assertEqual(result["status"], "observed")
        retained = result["development_selection"]
        self.assertFalse(retained["strict_subgroup_gate_passed"])
        self.assertFalse(retained["quality_success_claimed"])
        self.assertEqual(
            retained["failed_registered_typed_groups"], ["C_contract/speaker_unknown"]
        )
        self.assertEqual(
            retained["subgroup_review"],
            selection["selection_contract"]["subgroup_review"],
        )

    def test_all_final_rows_have_exactly_three_epochs_and_final_partial_update(self):
        rows = [{"id": f"engineering-{i}"} for i in range(1261)]
        order = [row["id"] for row in rows] * 3
        config = {
            "format": "json",
            "phase": "substantive",
            "training": {
                "sample_order": order,
                "draw_count": 3783,
                "unique_cases": 1261,
                "max_updates": 946,
                "gradient_accumulation": 4,
                "final_update_rows": 3,
                "learning_rate": 5e-5,
                "seed": 42,
            },
        }
        with mock.patch.object(final.sft, "validate_configuration"):
            final.verify_draws(config, rows)
            with self.assertRaisesRegex(ValueError, "1261"):
                final.verify_draws(config, rows[:870])
            config["training"]["sample_order"][0] = order[1]
            with self.assertRaisesRegex(ValueError, "once in each"):
                final.verify_draws(config, rows)

    def test_fallback_requires_both_actual_failed_native_candidates(self):
        import build_small_final_freeze_actions_v3 as builder
        import small_actions_final as small

        values = {
            name: (
                {
                    "model_identity": {"kind": "sft", "base": small.PROFILES[name]},
                    "minimum_numeric_quality_gate": {"passed": False},
                },
                {
                    "numerical_regime": "native_bf16_fp32_lora",
                    "schema_constrained": False,
                },
            )
            for name in final.fallback.MODELS
        }
        references = {name: name for name in values}
        with mock.patch.object(
            builder, "verified_dev_summary", side_effect=lambda ref: values[ref]
        ):
            self.assertEqual(final.verify_small_failures(references), references)
            with self.assertRaisesRegex(ValueError, "both completed"):
                final.verify_small_failures(
                    {next(iter(references)): next(iter(references))}
                )
            values["gemma4-e2b"][0]["minimum_numeric_quality_gate"]["passed"] = True
            with self.assertRaisesRegex(ValueError, "passing small"):
                final.verify_small_failures(references)
            values["gemma4-e2b"][0]["minimum_numeric_quality_gate"]["passed"] = False
            values["lfm25-1.2b"][1]["numerical_regime"] = "native_bf16_merged_q8_0"
            with self.assertRaisesRegex(ValueError, "quantized"):
                final.verify_small_failures(references)

    def test_new_sft_identity_and_complete_optimizer_exposure_are_required(self):
        import infer_task_test_actions_v3 as native
        import prepare_b003_final_gate_actions_v3 as recorded

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, reloaded = root / "training", root / "reload"
            config = {
                "sources_sha256": {"engineering.py": "source"},
                "engineering_only": True,
            }
            for path in (
                root / "prepared/config.json",
                training / "configuration.json",
            ):
                final.write(path, config)
            for directory in (training, reloaded):
                final.write(directory / "manifest.json", {"engineering_only": True})
            final.write(training / "runtime.json", {"engineering_only": True})
            exposure = {
                "phases": {"optimizer_committed": {"rows": 3783}},
                "completed_updates": 946,
                "optimizer_commit_uncertain": False,
                "pending_backward_rows_not_committed": 0,
            }
            final.write(training / "exposure.json", exposure)
            identity = {"kind": final.sft.VERSION, "adapter_sha256": "new-final-parent"}
            before = {
                "training_updates": 946,
                "optimizer_committed_rows": 3783,
                "environment": {"matched": True},
            }
            after = {"reloaded_training_updates": 946}
            with (
                mock.patch.object(
                    final, "verify_preparation", return_value=(config, {})
                ),
                mock.patch.object(
                    recorded,
                    "recorded_run",
                    return_value=(config, identity, {}, [before, after]),
                ),
                mock.patch.object(
                    native, "native_identity", return_value=identity
                ) as actual,
            ):
                result = final.verify_trained(
                    root / "prepared/config.json", training, reloaded
                )
                self.assertEqual(result["identity"], identity)
                before["optimizer_committed_rows"] = 3782
                with self.assertRaisesRegex(ValueError, "complete three-epoch"):
                    final.verify_trained(
                        root / "prepared/config.json", training, reloaded
                    )
                before["optimizer_committed_rows"] = 3783
                actual.return_value = {**identity, "adapter_sha256": "other-reload"}
                with self.assertRaisesRegex(ValueError, "payload identity differs"):
                    final.verify_trained(
                        root / "prepared/config.json", training, reloaded
                    )

    def test_model_boundary_receipt_cannot_claim_final_registration_or_choose_dpo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                final,
                "verify_trained",
                return_value={"identity": {"kind": "engineering-SFT"}},
            ):
                result = final.verify_model(
                    root / "prep/config.json",
                    root / "train",
                    root / "reload",
                    root / "receipt.json",
                )
                self.assertFalse(result["body_access_authorized"])
                self.assertFalse(result["model_selection_performed"])
                self.assertTrue(result["remaining_final_contracts"])
                self.assertEqual(
                    result["status"], "model_evidence_verified_not_final_registered"
                )
                with self.assertRaisesRegex(ValueError, "no verified training"):
                    final.verify_model(
                        root / "prep/config.json",
                        root / "train",
                        root / "reload",
                        root / "other.json",
                        deployment=root / "deploy",
                        deployment_role="sft_dpo",
                    )

    def test_dpo_omission_counts_remain_prospective_and_do_not_register_a_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final.write(root / "mined/run.json", {"engineering_only": True})
            report = {"requested": 1261, "pairs": 7, "preservation_pairs": 1}
            with (
                mock.patch.object(
                    final,
                    "verify_trained",
                    return_value={"identity": {"kind": "new-parent"}},
                ),
                mock.patch.object(mining, "verify_mining", return_value=report),
            ):
                result = mining.assess(
                    root / "config",
                    root / "train",
                    root / "reload",
                    root / "mined",
                    root / "omission.json",
                    plan=final.PLAN,
                )
                self.assertEqual(result["decision"], "eligible_omission")
                self.assertEqual(
                    result["reason_codes"], ["fewer_than_eight_strict_pairs"]
                )
                self.assertFalse(result["final_registration_performed"])
                report["pairs"] = 8
                result = mining.assess(
                    root / "config",
                    root / "train",
                    root / "reload",
                    root / "mined",
                    root / "required.json",
                    plan=final.PLAN,
                )
                self.assertEqual(result["decision"], "same_parent_DPO_required")
                self.assertEqual(result["reason_codes"], [])

    def test_unresolved_mining_cannot_be_reclassified_as_pair_scarcity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final.write(root / "configuration.json", {"engineering_only": True})
            final.write(root / "candidates/manifest.json", {"engineering_only": True})
            parent = {
                "configuration_sha256": final.sha(root / "configuration.json"),
                "adapter_sha256": "adapter",
                "training_manifest_sha256": "training",
                "reload_manifest_sha256": "reload",
            }
            candidate_sha = final.sha(root / "candidates/manifest.json")
            observations = [
                {
                    "case_id": f"fixture-{i}",
                    "candidate_id": f"fixture-{i}:greedy:0",
                    "input_sha256": f"input-{i}",
                    "text": '{"action":"no_action"}',
                }
                for i in range(1261)
            ]
            final.prep.write_rows(root / "candidates/candidates.jsonl", observations)
            decisions = [
                {key: row[key] for key in ("case_id", "candidate_id", "input_sha256")}
                | {"candidate_run_sha256": candidate_sha, "status": "tie"}
                for row in observations
            ]
            final.prep.write_rows(root / "mined/decisions.jsonl", decisions)
            final.prep.write_rows(root / "mined/pairs.jsonl", [])
            final.prep.write_rows(root / "mined/review-queue.jsonl", [])
            report = {
                "version": mining.VERSION,
                "status": "train_errors_mined",
                "parent": parent,
                "requested": 1261,
                "saved_projections_preserved": True,
                "final_test_bodies_read": False,
                "configuration": final.binding(root / "configuration.json"),
                "candidates": final.binding(root / "candidates/manifest.json"),
                "candidate_run_sha256": candidate_sha,
                "pairs": 0,
                "preservation_pairs": 0,
                "decisions": {"tie": 1261},
                "files_sha256": {
                    name: final.sha(root / "mined" / name)
                    for name in ("decisions.jsonl", "pairs.jsonl", "review-queue.jsonl")
                },
            }
            final.write(root / "mined/run.json", report)
            observed = {
                "train_only": True,
                "final_test_bodies_read": False,
                "candidates": 1261,
                **{"anchor_" + key: value for key, value in parent.items()},
            }
            with mock.patch.object(
                dpo.sft, "verified_run", return_value=({}, observed)
            ):
                mining.verify_mining(root / "mined", parent)
                decisions[0]["status"] = "needs_review"
                (root / "mined/decisions.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in decisions)
                )
                report["decisions"] = {"tie": 1260, "needs_review": 1}
                report["files_sha256"]["decisions.jsonl"] = final.sha(
                    root / "mined/decisions.jsonl"
                )
                (root / "mined/run.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(
                    ValueError, "no unresolved semantic review"
                ):
                    mining.verify_mining(root / "mined", parent)

    def test_context_audit_fails_whole_cohort_without_cropping(self):
        rows = cases()[:2]
        for row in rows:
            row["split"] = row["source"]["split"] = "dev"
        tokenizer = Tokenizer()
        examples = [cases()[1], cases()[0]]
        examples[1]["language"] = "ko"
        examples[1]["input"]["language"] = "ko"
        examples[1]["messages"] = final.prep.formats.messages(examples[1]["input"])
        for row in examples:
            row["accepted_predictions"] = row["acceptable"]
        demos = [final.conditions.demonstration(row, tokenizer, {}) for row in examples]
        actual = final.context_records(rows, demos, tokenizer)
        self.assertEqual(
            [row["case_id"] for row in actual], [row["id"] for row in rows]
        )
        with mock.patch.object(
            final.conditions, "native_prefix", side_effect=[[2] * 100, [2] * 7681]
        ):
            with self.assertRaisesRegex(ValueError, "entire common cohort"):
                final.context_records(rows[:1], demos, tokenizer)
        rows[0]["split"] = "final_test"
        with self.assertRaisesRegex(ValueError, "exposed DEV"):
            final.context_records(rows, demos, tokenizer)

    def test_saved_h0_h3_h8_never_rebuild_the_default_window(self):
        import prepare_history_length_actions_v3 as histories
        from test_prepare_history_length_actions_v3 import Tokenizer as HistoryTokenizer

        if not histories.FIXTURE.is_file():
            self.skipTest("History diagnostics require local experiment fixtures.")
        fixture = final.read(histories.FIXTURE)
        episode = fixture["episodes"][0]
        rows, _, _ = histories.build_episode(
            fixture, episode, HistoryTokenizer(), histories.digest(fixture)
        )
        observed_lengths = []
        with mock.patch.object(
            dpo.dataset,
            "runtime_bundle",
            side_effect=AssertionError("Default H3 rebuild is forbidden"),
        ):
            for row in rows:
                row["split"] = row["source"]["split"] = "train"
                row["lineage"] = {}
                candidate = {
                    "case_id": row["id"],
                    "split": "train",
                    "candidate_id": row["id"] + ":greedy:0",
                    "input_sha256": mining.conditions_hash(row),
                    "parsed_action": None,
                    "text": "invalid completed JSON",
                }
                pair = {
                    "pair_id": row["id"] + ":error",
                    "case_id": row["id"],
                    "input_sha256": candidate["input_sha256"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_run_sha256": "run",
                    "chosen": row["primary"],
                    "rejected_text": candidate["text"],
                    "judgment": {
                        "status": "preferred",
                        "actor": "ai_generated",
                        "rationale": "Engineering contract-error fixture.",
                    },
                }
                result = dpo.prepare_pair(
                    pair,
                    row,
                    candidate,
                    Tokenizer(),
                    {"format": "json"},
                    "run",
                    saved_projection=True,
                )
                self.assertEqual(result["messages"], row["messages"])
                self.assertEqual(
                    result["rejected"]["completion_text"], candidate["text"]
                )
                self.assertEqual(result["rejected"]["input_ids"][-1], 1)
                observed_lengths.append(row["history_length"]["length"])
        self.assertEqual(observed_lengths, [0, 3, 8])

    def test_mining_preserves_actual_text_and_does_not_turn_valid_alternatives_into_negatives(
        self,
    ):
        rows = cases()[:3]
        for row in rows:
            row.update(accepted_predictions=row["acceptable"], lineage={})
        texts = [
            "broken JSON",
            final.prep.formats.serialize(rows[1]["input"], rows[1]["primary"], "json"),
            '{"action":"correct_text","target":[1],"text":"you"}',
        ]
        observations = [
            {
                "case_id": row["id"],
                "split": "train",
                "format": "json",
                "candidate_id": row["id"] + ":greedy:0",
                "input_sha256": mining.conditions_hash(row),
                "text": text,
                "finish_reason": "stop",
                "parsed_action": None,
            }
            for row, text in zip(rows, texts, strict=True)
        ]
        decisions, pairs, queue = mining.decisions_for(rows, observations, "run")
        self.assertEqual(
            [row["status"] for row in decisions], ["preferred", "tie", "needs_review"]
        )
        self.assertEqual(pairs[0]["rejected_text"], observations[0]["text"])
        self.assertEqual(len(queue), 1)
        with self.assertRaisesRegex(ValueError, "full unique TRAIN"):
            mining.decisions_for(rows, observations[:-1], "run")

    def test_deployment_binds_separate_adapter_base_scale_and_new_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = {
                "repository": "native",
                "revision": "pin",
                "adapter_sha256": "new-sft",
                "training_manifest_sha256": "new-training",
                "reload_manifest_sha256": "reload",
                "native_pin_sha256": "pin-file",
            }
            export = {
                "version": "gemma-actions-v3-lora-export",
                "all_tensors_exact": True,
                "adapter_tensor_count": 410,
                "status": "adapter_conversion_verified_serving_pending",
                "contract_version": final.prep.formats.contract.VERSION,
                "format_version": final.prep.formats.VERSION,
                "output_format": "json",
                "runtime_revision": "427291b5b34cd914a31b3fd3b61a68f6184f4b9f",
                "gguf_sha256": "fp32-lora",
                "base_repository": "native",
                "base_revision": "pin",
                **selected,
            }
            final.write(root / "export.json", export)
            final.write(
                root / "base.json",
                {
                    "artifacts": {"base_quantized": {"sha256": "q4-base"}},
                    "base_repository": "native",
                    "base_revision": "pin",
                },
            )
            gateway = SimpleNamespace(
                capture_runtime_evidence=False,
                output_format="json",
                adapter_path="/adapter.gguf",
                adapter_scale=1.0,
                adapter_sha256="fp32-lora",
                model_path="/base.gguf",
                model_sha256="q4-base",
                model_id="engineering-only",
            )
            command = [
                "server",
                "-m",
                "/base.gguf",
                "--lora",
                "/adapter.gguf",
                "-c",
                "4096",
            ]
            receipt = {
                "version": "local-gemma-actions-v3-lora",
                "historical_tooling_only": False,
                "purpose": "ui_performance",
                "format": "json",
                "numerical_regime": "native_derived_q4_base_fp32_lora",
                "adapter_export_receipt": final.binding(root / "export.json"),
                "base_export_receipt": final.binding(root / "base.json"),
            }
            fake = SimpleNamespace(
                VERSION="local-gemma-actions-v3-lora",
                check=lambda _: (gateway, None, command, receipt),
            )
            with mock.patch.dict("sys.modules", {"run_local_copilot_actions_v3": fake}):
                identity = final.verify_deployment(root, selected)
                self.assertEqual(identity["adapter_sha256"], "fp32-lora")
                self.assertEqual(identity["adapter_scale"], 1.0)
                for key, value in (
                    ("adapter_scale", 0.5),
                    ("adapter_sha256", "other"),
                    ("model_sha256", "other"),
                ):
                    original = getattr(gateway, key)
                    setattr(gateway, key, value)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        final.verify_deployment(root, selected)
                    setattr(gateway, key, original)
                with self.assertRaisesRegex(ValueError, "another trained adapter"):
                    final.verify_deployment(
                        root, {**selected, "adapter_sha256": "B003"}
                    )


if __name__ == "__main__":
    unittest.main()
