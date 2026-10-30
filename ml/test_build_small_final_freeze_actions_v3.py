"""Metadata-only engineering fixtures; no reserved source or model is opened."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_small_final_freeze_actions_v3 as builder


class SmallFreezeBuilderTests(unittest.TestCase):
    def setUp(self):
        (builder.ROOT / "artifacts").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="small-freeze-engineering-", dir=builder.ROOT / "artifacts"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("DEPLOYMENT_PLAN", "UI_AMENDMENT"):
            path = self.root / (name + ".md")
            path.write_text("Synthetic plan for freeze metadata tests only.\n")
            patch = mock.patch.object(builder, name, path)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(
            builder, "DEPLOYMENT_PLAN_SHA", builder.sha(builder.DEPLOYMENT_PLAN)
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.evidence = builder.write(
            self.root / "engineering-evidence.json", {"engineering_fixture_only": True}
        )

    def test_derived_dev_wrapper_may_change_only_native_token_ids(self):
        original, derived = self.root / "original", self.root / "derived"
        request = {
            "case_id": "engineering-dev",
            "split": "dev",
            "messages": [{"role": "user", "content": "Engineering"}],
            "input": {"engineering_fixture": True},
            "prefix_ids": [2],
            "prefix_ids_sha256": "old",
        }
        references = []
        for root, ids in ((original, [2]), (derived, [2, 3])):
            root.mkdir()
            (root / "cases.jsonl").write_text('{"id":"engineering-dev"}\n')
            (root / "exclusions.jsonl").write_text("")
            (root / "requests-json.jsonl").write_text(
                json.dumps(
                    {**request, "prefix_ids": ids, "prefix_ids_sha256": str(ids)}
                )
                + "\n"
            )
            references.append(
                builder.write(
                    root / "manifest.json",
                    {
                        "dataset_manifest": self.evidence,
                        "final_test_bodies_read": False,
                        "excluded": 0,
                        "files_sha256": {
                            p.name: builder.sha(p) for p in root.iterdir()
                        },
                    },
                )
            )
        builder.verify_dev_cohort_inputs(references[1], references[0])
        changed = {
            **request,
            "messages": [{"role": "user", "content": "Changed prompt"}],
        }
        (derived / "requests-json.jsonl").write_text(json.dumps(changed) + "\n")
        manifest = builder.read(derived / "manifest.json")
        manifest["files_sha256"]["requests-json.jsonl"] = builder.sha(
            derived / "requests-json.jsonl"
        )
        (derived / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "only native token IDs"):
            builder.verify_dev_cohort_inputs(
                builder.binding(derived / "manifest.json"), references[0]
            )

    def test_exposure_extension_preserves_reservations_and_existing_splits(self):
        original = {
            "groups": [{"source_group": "reserved", "role": "primary"}],
            "exposed_source_group_splits": {"old-train": "train", "old-dev": "dev"},
            "exposed_inventory_sha256": {"old.json": "old"},
            "fallback_rule": "fixed-order",
        }
        actual = builder.reservations_for_release(
            original, {"old-train": "train", "new-dev": "dev"}, {"new.json": "new"}
        )
        self.assertEqual(actual["groups"], original["groups"])
        self.assertEqual(actual["fallback_rule"], original["fallback_rule"])
        self.assertEqual(
            actual["exposed_source_group_splits"],
            {"old-train": "train", "old-dev": "dev", "new-dev": "dev"},
        )
        self.assertNotIn("new-dev", original["exposed_source_group_splits"])
        for split_map in ({"old-train": "dev"}, {"reserved": "train"}):
            with self.assertRaises(ValueError):
                builder.reservations_for_release(original, split_map, {})

    def test_fixed_icl_reencodes_tokens_without_reselecting_or_changing_words(self):
        ids = builder.small.DEMONSTRATION_IDS
        cases = [{"id": identity} for identity in ids]
        originals = [
            {
                "case_id": identity,
                "input": {"text": identity},
                "release_binding": {"old": True},
                "selection_tokens": {"json": 100},
                "native_prefix_ids_sha256": {"json": "old"},
            }
            for identity in ids
        ]
        encoded = [
            {
                **value,
                "selection_tokens": {"json": 120},
                "native_prefix_ids_sha256": {"json": "new"},
                "release_binding": {"new": True},
            }
            for value in originals
        ]
        with mock.patch.object(
            builder.gate.conditions, "demonstration", side_effect=encoded
        ):
            self.assertEqual(
                builder.fixed_demonstrations(cases, originals, object(), {}), encoded
            )
        with self.assertRaisesRegex(ValueError, "reselect"):
            builder.fixed_demonstrations(cases[::-1], originals, object(), {})
        changed = copy.deepcopy(encoded)
        changed[0]["input"]["text"] = "different wording"
        with mock.patch.object(
            builder.gate.conditions, "demonstration", side_effect=changed
        ):
            with self.assertRaisesRegex(ValueError, "content or reference semantics"):
                builder.fixed_demonstrations(cases, originals, object(), {})

    def test_exposure_addendum_is_not_automatic_global_clearance(self):
        baseline = builder.write(
            self.root / "index-baseline.json", {"sha256": ["a" * 64]}
        )
        extended = builder.write(
            self.root / "index-extended.json", {"sha256": ["a" * 64, "b" * 64]}
        )
        prepared = {"exposed_text_hashes": baseline}
        review = {
            "version": "task-test-actions-v3-exposure-readiness-1",
            "status": "parent_reviewed",
            "actor": "/root",
            "selected_training_dataset_manifest_sha256": builder.FINAL_SHA,
            "reserved_bodies_read": False,
            "new_project_visible_inputs_unaccounted_for": False,
            "exposed_text_hashes": extended,
            "exposure_inventory_evidence": {"fixture": self.evidence},
            "review_evidence": [self.evidence],
        }
        builder.exposure_review(review, prepared)
        missing = builder.write(
            self.root / "index-missing.json", {"sha256": ["b" * 64]}
        )
        with self.assertRaisesRegex(ValueError, "retain every registered"):
            builder.exposure_review(
                {**review, "exposed_text_hashes": missing}, prepared
            )
        for update in (
            {"status": "draft_not_authorization"},
            {"new_project_visible_inputs_unaccounted_for": None},
            {"selected_training_dataset_manifest_sha256": "old-release"},
        ):
            with self.assertRaisesRegex(ValueError, "coordinator exposure review"):
                builder.exposure_review({**review, **update}, prepared)

    def test_draft_readiness_never_reaches_registration_or_body_gate(self):
        ready = builder.write(
            self.root / "draft.json",
            {"version": builder.READINESS_VERSION, "status": "draft_not_authorization"},
        )
        with mock.patch.object(builder.gate, "load_gate") as gate:
            with self.assertRaisesRegex(
                ValueError, "draft metadata grants no authority"
            ):
                builder.preflight(
                    self.root / "missing-registration.json",
                    "not-read",
                    ready["path"],
                    ready["sha256"],
                    self.root / "never-created",
                )
        gate.assert_not_called()
        self.assertFalse((self.root / "never-created").exists())

    def test_actual_ui_identity_and_all_planned_outcomes_are_required(self):
        identity = {
            "model_id": "engineering-merged",
            "model_sha256": "gguf",
            "adapter_sha256": None,
        }
        ui = builder.ui_template()
        ui.update(
            status="observed",
            deployment=self.evidence,
            model=identity,
            planned_case_ids=["en", "ko", "failure"],
            launch_method="npm start",
            **dict.fromkeys(builder.UI_MEASUREMENTS, self.evidence),
            outcomes=[
                {
                    "case_id": name,
                    "language": "ko" if name == "ko" else "en",
                    "status": "request_failure" if name == "failure" else "preview",
                    "automatic_trigger": True,
                    "evidence": [self.evidence],
                }
                for name in ("en", "ko", "failure")
            ],
            evidence={name: [self.evidence] for name in ui["evidence"]},
        )
        builder.validate_ui(ui, self.evidence, identity)
        changed = copy.deepcopy(ui)
        changed["outcomes"].pop()
        with self.assertRaisesRegex(ValueError, "Every planned UI outcome"):
            builder.validate_ui(changed, self.evidence, identity)
        for field, value in (
            ("model", {**identity, "adapter_sha256": "unmerged-adapter"}),
            ("manual_request_button_used", True),
        ):
            with self.assertRaisesRegex(ValueError, "another model/deployment"):
                builder.validate_ui({**ui, field: value}, self.evidence, identity)

    def test_materialization_keeps_explicit_omission_and_calls_existing_gate(self):
        if not builder.legacy.CORE.is_file():
            self.skipTest("Requires the archived experiment freeze builder.")
        core = builder.legacy.load_core()
        draft_dir = self.root / "drafts"
        drafts = {}
        for name in core.LEAF_ORDER:
            drafts[name] = {
                "status": "draft",
                "scope": "final_test",
                "body_access_authorized": False,
                "unresolved_bindings": {"parent_policy_approval": None},
            }
            if name == "runtime-policy":
                drafts[name]["local_deployment_comparison"] = {}
            builder.write(draft_dir / (name + ".json"), drafts[name])
        revision = builder.write(
            draft_dir / "revision.json", {"engineering_fixture_only": True}
        )
        model_bindings = {"native_base": self.evidence, "sft": self.evidence}
        roles = [role for role in builder.ROLES if role != "sft_dpo"]
        registration = {
            "basis_documents": {},
            "models": model_bindings,
            "dpo_omission": self.evidence,
            "draft_revision": revision,
            "selection_receipt": self.evidence,
            "training_dataset_manifest": self.evidence,
            "controller": self.evidence,
            "metadata": {},
            "numerical_targets": self.evidence,
            "actors": {},
            "actor_registry": self.evidence,
            "run_roles": roles,
        }
        drafts["admission-policy"]["exposure_inventory_evidence"] = {
            "fixture": self.evidence
        }
        ready = {
            "protocol_id": "engineering-only",
            **dict.fromkeys(builder.EVIDENCE, self.evidence),
        }
        context = {
            "registration": registration,
            "registration_binding": self.evidence,
            "ready": ready,
            "ready_binding": builder.write(self.root / "fixture-readiness.json", ready),
            "drafts": drafts,
            "models": {
                role: {
                    "identity": {"engineering_fixture_only": True},
                    "execution_runtime": self.evidence,
                }
                for role in model_bindings
            },
            "selected_model": "sft",
            "closure": {},
            "product_validation": {
                "status": "pending_product_validation",
                "receipt": self.evidence,
                "product_completion_asserted": False,
                "registered_product_drop_gate_passed": False,
                "quality_success_claimed": False,
                "product_development_selection": {
                    "registered_product_drop_gate_passed": False,
                    "failed_product_drop_checks": ["supplemental_dev149:recall"],
                    "quality_success_claimed": False,
                    "body_access_authorized": False,
                },
                "development_selection": {
                    "strict_subgroup_gate_passed": False,
                    "failed_registered_typed_groups": ["engineering-group"],
                },
            },
            "conditions": {
                "conditions": {"train_icl": {}, "canonical_full_context": {}}
            },
        }
        output = self.root / "synthetic-only-freeze"
        with (
            mock.patch.object(
                builder.gate,
                "load_gate",
                return_value={
                    "binding": self.evidence,
                    "roster": [None] * 104,
                    "windows": [None] * 16,
                },
            ) as gate,
            mock.patch.object(
                builder,
                "deployed_policy",
                return_value={
                    "engineering_fixture_only": True,
                    "role": "deployed_selected",
                },
            ),
        ):
            result = builder.materialize(context, output, builder.gate.now())
        gate.assert_called_once_with(output / "freeze.json")
        freeze = builder.read(output / "freeze.json")
        self.assertEqual(set(freeze["models"]), {"native_base", "sft"})
        self.assertEqual([run["role"] for run in freeze["runs"]], roles)
        self.assertEqual(freeze["dpo_omission"], self.evidence)
        self.assertEqual(
            builder.read_bound(freeze["deployment_evaluation_policy"])["role"],
            "deployed_selected",
        )
        self.assertFalse(result["reserved_bodies_read"])
        self.assertFalse(result["quality_success_asserted"])
        self.assertEqual(freeze["product_validation"], context["product_validation"])
        self.assertEqual(
            result["product_validation"]["status"], "pending_product_validation"
        )
        runtime = builder.read(output / "runtime-policy.json")
        self.assertFalse(
            runtime["local_deployment_comparison"]["product_validation"][
                "product_completion_asserted"
            ]
        )
        for value in (
            freeze["product_validation"],
            result["product_validation"],
            runtime["local_deployment_comparison"]["product_validation"],
        ):
            self.assertFalse(value["registered_product_drop_gate_passed"])
            self.assertFalse(value["quality_success_claimed"])
            self.assertEqual(
                value["product_development_selection"]["failed_product_drop_checks"],
                ["supplemental_dev149:recall"],
            )

    def quality_fixture(self, *, competitor_passes=False, competitor_score=0.85):
        chosen = "lfm25-1.2b"
        config = {"model": chosen, "base": builder.small.data.PROFILES[chosen]}
        identity = {
            "kind": "sft",
            "configuration_sha256": "config",
            "adapter_sha256": "adapter",
            "reload_run_sha256": "reload",
        }
        models = {"sft": {"identity": identity}}
        stats = {
            "precision": {"value": 0.96},
            "recall": {"value": 0.90},
            "model_preservation": {"value": 0.95, "numerator": 68},
            "status_counts": {"completed": 471},
        }
        summary = {
            "model_identity": {**identity, "base": config["base"]},
            "overall": stats,
            "cohorts": {
                name: {"metrics": copy.deepcopy(stats)}
                for name in ("canonical_dev322", "supplemental_dev149")
            },
            "minimum_numeric_quality_gate": {"passed": True},
            "macro_action_f1": {"value": 0.90},
            "cohort_mapping": self.evidence,
        }
        native_report = {
            "numerical_regime": "native_bf16_fp32_lora",
            "schema_constrained": False,
        }
        other = copy.deepcopy(summary)
        other["model_identity"].update(
            base=builder.small.data.PROFILES["gemma4-e2b"], adapter_sha256="other"
        )
        other["minimum_numeric_quality_gate"]["passed"] = competitor_passes
        other["macro_action_f1"]["value"] = competitor_score
        product = copy.deepcopy(summary)
        product["model_identity"].update(
            gguf_sha256="gguf", export_receipt_sha256="export"
        )
        product_report = {
            "numerical_regime": "native_bf16_merged_q8_0",
            "schema_constrained": True,
        }
        references = {
            name: builder.write(self.root / f"quality-{name}.json", value)
            for name, value in (
                (chosen, summary),
                ("gemma4-e2b", other),
                ("deployed", product),
            )
        }
        values = {
            references[name]["sha256"]: (
                value,
                product_report if name == "deployed" else native_report,
            )
            for name, value in (
                (chosen, summary),
                ("gemma4-e2b", other),
                ("deployed", product),
            )
        }
        selection = {
            "evidence": [*references.values(), self.evidence],
            "quality_without_ui": {
                "native_sft_summaries": {
                    name: references[name] for name in builder.small.data.PROFILES
                },
                "selected_native_summary": references[chosen],
                "selected_deployed_summary": references["deployed"],
                "subgroup_review": {
                    "status": "reviewed_pass",
                    "rationale": "Engineering fixture subgroup review.",
                    "evidence": [self.evidence],
                },
            },
        }
        deployment = {
            "export_receipt": {"sha256": "export"},
            "numerical_regime": "native_bf16_merged_q8_0",
        }
        gateway_identity = {
            "model_id": "engineering-only",
            "model_sha256": "gguf",
            "adapter_sha256": None,
        }
        return selection, models, config, deployment, gateway_identity, values

    def test_quality_only_choice_requires_both_actual_results_and_excludes_ui_ties(
        self,
    ):
        selection, models, config, deployment, identity, values = self.quality_fixture()
        with mock.patch.object(
            builder,
            "verified_dev_summary",
            side_effect=lambda reference: values[reference["sha256"]],
        ):

            def call():
                return builder.quality_without_ui(
                    selection, models, "sft", config, deployment, identity
                )

            self.assertEqual(call(), "only_numeric_quality_candidate")
            other = values[
                selection["quality_without_ui"]["native_sft_summaries"]["gemma4-e2b"][
                    "sha256"
                ]
            ][0]
            other["minimum_numeric_quality_gate"]["passed"] = True
            self.assertEqual(call(), "macro_f1_margin_over_two_percentage_points")
            for score in (0.88, 0.89, 0.90, 0.91):
                other["macro_action_f1"]["value"] = score
                with (
                    self.subTest(score=score),
                    self.assertRaisesRegex(ValueError, "Actual UI latency is required"),
                ):
                    call()
            with self.assertRaisesRegex(ValueError, "boolean"):
                builder.quality_without_ui(
                    {**selection, "quality_without_ui": True},
                    models,
                    "sft",
                    config,
                    deployment,
                    identity,
                )
            del selection["quality_without_ui"]["native_sft_summaries"]["gemma4-e2b"]
            with self.assertRaisesRegex(ValueError, "both registered"):
                call()

    def test_quality_only_choice_keeps_deployed_identity_and_full_dev_quality_gates(
        self,
    ):
        selection, models, config, deployment, identity, values = self.quality_fixture()
        product = values[
            selection["quality_without_ui"]["selected_deployed_summary"]["sha256"]
        ][0]
        with mock.patch.object(
            builder,
            "verified_dev_summary",
            side_effect=lambda reference: values[reference["sha256"]],
        ):

            def call():
                return builder.quality_without_ui(
                    selection, models, "sft", config, deployment, identity
                )

            product["model_identity"]["gguf_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "another GGUF"):
                call()
            product["model_identity"]["gguf_sha256"] = "gguf"
            product["minimum_numeric_quality_gate"]["passed"] = False
            with self.assertRaisesRegex(ValueError, "both pass"):
                call()
            product["minimum_numeric_quality_gate"]["passed"] = True
            product["cohorts"]["supplemental_dev149"]["metrics"]["recall"]["value"] = (
                0.87
            )
            with self.assertRaisesRegex(ValueError, "quality-drop allowance"):
                call()

    def test_quality_only_dpo_keeps_same_parent_adoption_rules(self):
        selection, models, config, deployment, identity, values = self.quality_fixture()
        quality = selection["quality_without_ui"]
        parent, report = values[quality["selected_native_summary"]["sha256"]]
        native = copy.deepcopy(parent)
        native["model_identity"].update(
            kind="dpo", adapter_sha256="dpo-adapter", reload_run_sha256="dpo-reload"
        )
        native["macro_action_f1"]["value"] = 0.91
        models["sft_dpo"] = {
            "identity": {
                key: value
                for key, value in native["model_identity"].items()
                if key != "base"
            }
        }
        reference = builder.write(self.root / "quality-dpo.json", native)
        quality["selected_native_summary"] = reference
        selection["evidence"].append(reference)
        values[reference["sha256"]] = native, report
        product = values[quality["selected_deployed_summary"]["sha256"]][0]
        product["model_identity"].update(kind="dpo", adapter_sha256="dpo-adapter")
        with mock.patch.object(
            builder,
            "verified_dev_summary",
            side_effect=lambda value: values[value["sha256"]],
        ):

            def call():
                return builder.quality_without_ui(
                    selection, models, "sft_dpo", config, deployment, identity
                )

            self.assertEqual(call(), "only_numeric_quality_candidate")
            native["macro_action_f1"]["value"] = 0.90
            with self.assertRaisesRegex(ValueError, "improve its same-parent"):
                call()
            native["macro_action_f1"]["value"] = 0.91
            cohort = native["cohorts"]["supplemental_dev149"]["metrics"]
            for change in (
                {"precision": {"value": 0.95}},
                {"recall": {"value": 0.88}},
                {"model_preservation": {"value": 0.94, "numerator": 67}},
                {"status_counts": {"completed": 470, "invalid": 1}},
            ):
                original = copy.deepcopy(cohort)
                cohort.update(change)
                with (
                    self.subTest(change=change),
                    self.assertRaisesRegex(ValueError, "parent-quality adoption rule"),
                ):
                    call()
                cohort.clear()
                cohort.update(original)

    def pending_fixture(self):
        identity = {
            "model_id": "engineering-only",
            "model_sha256": "gguf",
            "adapter_sha256": None,
        }
        config, meta = (
            {"base": {"engineering_only": True}},
            {"environment": {"engineering_only": True}},
        )
        runtime = builder.write(
            self.root / "native-runtime.json",
            {
                **meta,
                **config,
                "dtype": "bfloat16",
                "attention": "sdpa",
                "device": "cuda",
                "gpu": "Engineering fixture; no actual GPU accessed.",
                "loading": {},
            },
        )
        registration = {"selection_receipt": self.evidence}
        models = {
            "native_base": {"artifact_manifest": runtime},
            "sft": {"identity": {"engineering_only": True}},
        }
        ui = builder.pending_ui_template()
        ui.update(
            status="pending_product_validation",
            reason="Engineering blocked-screen fixture, not an actual observation.",
            evidence=[self.evidence],
            deployment=self.evidence,
            model=identity,
            selection_receipt=self.evidence,
        )
        ready = {
            "selected_deployment_receipt": self.evidence,
            "native_environment_receipt": runtime,
            "native_runtime_receipt": runtime,
            "selected_model_ui_receipt": builder.write(
                self.root / "pending-ui.json", ui
            ),
            "evidence_review": {
                "attestations": {
                    key: key not in builder.UI_ATTESTATIONS
                    for key in builder.legacy.inputs.ATTESTATIONS
                }
            },
        }
        return ready, registration, models, config, meta, identity, ui

    def test_pending_ui_keeps_native_runtime_and_deployment_verification(self):
        ready, registration, models, config, meta, identity, _ = self.pending_fixture()
        with (
            mock.patch.object(
                builder, "validate_deployment", return_value=identity
            ) as deployment,
            mock.patch.object(
                builder,
                "quality_without_ui",
                return_value="only_numeric_quality_candidate",
            ) as selection,
        ):
            result = builder.validate_readiness_evidence(
                ready, registration, models, "sft", config, meta
            )
            self.assertEqual(result["status"], "pending_product_validation")
            self.assertFalse(result["product_completion_asserted"])
            deployment.assert_called_once_with(self.evidence, models["sft"], config)
            selection.assert_called_once()
            wrong = builder.write(
                self.root / "wrong-runtime.json",
                {
                    **builder.read_bound(ready["native_runtime_receipt"]),
                    "dtype": "float16",
                },
            )
            with self.assertRaisesRegex(ValueError, "Actual native environment"):
                builder.validate_readiness_evidence(
                    {**ready, "native_runtime_receipt": wrong},
                    registration,
                    models,
                    "sft",
                    config,
                    meta,
                )
            deployment.side_effect = ValueError("Actual deployment identity changed")
            with self.assertRaisesRegex(ValueError, "deployment identity changed"):
                builder.validate_readiness_evidence(
                    ready, registration, models, "sft", config, meta
                )
            deployment.side_effect = None
            selection.side_effect = ValueError("Actual UI latency is required")
            with self.assertRaisesRegex(ValueError, "Actual UI latency is required"):
                builder.validate_readiness_evidence(
                    ready, registration, models, "sft", config, meta
                )

    def test_pending_ui_cannot_claim_observation_completion_or_skip_blocker_evidence(
        self,
    ):
        ready, registration, models, config, _, identity, ui = self.pending_fixture()
        changes = [
            {"status": "observed"},
            {"product_complete": True},
            {"reason": ""},
            {"evidence": []},
            {"outcomes": [{"status": "preview"}]},
            {"model": {**identity, "model_sha256": "other"}},
        ]
        with mock.patch.object(
            builder, "quality_without_ui", return_value="only_numeric_quality_candidate"
        ):
            for index, change in enumerate(changes):
                reference = builder.write(
                    self.root / f"bad-pending-{index}.json", {**ui, **change}
                )
                with (
                    self.subTest(change=change),
                    self.assertRaisesRegex(ValueError, "without UI success claims"),
                ):
                    builder.validate_product_validation(
                        {**ready, "selected_model_ui_receipt": reference},
                        registration,
                        models,
                        "sft",
                        config,
                        identity,
                    )
            ready["evidence_review"]["attestations"] = dict.fromkeys(
                builder.legacy.inputs.ATTESTATIONS, True
            )
            with self.assertRaisesRegex(ValueError, "pending UI checks as false"):
                builder.validate_product_validation(
                    ready, registration, models, "sft", config, identity
                )

    def test_quality_summary_is_recomputed_from_bound_full_dev_rows(self):
        import evaluate_small_actions_final as development

        release_dir = self.root / "fixture-release"
        release = builder.write(
            release_dir / "manifest.json",
            {"files_sha256": {"cases/train.jsonl": "fixture-train"}},
        )
        configuration = builder.write(
            self.root / "training-configuration.json",
            {
                "source_cases_sha256": "fixture-train",
                "training": {"epochs": 3, "sample_order": list(range(1261)) * 3},
            },
        )
        evaluation = self.root / "evaluation"
        evaluation.mkdir()
        requests = [
            {
                "case_id": f"engineering-only-{i}",
                "split": "dev",
                "messages": [],
                "input": {},
            }
            for i in range(471)
        ]
        for name, content in (
            ("requests-json.jsonl", requests),
            ("cases.jsonl", requests),
            ("exclusions.jsonl", []),
        ):
            development.data.write_rows(evaluation / name, content)
        original = builder.write(
            evaluation / "manifest.json",
            {
                "dataset_manifest": release,
                "final_test_bodies_read": False,
                "excluded": 0,
                "files_sha256": {p.name: builder.sha(p) for p in evaluation.iterdir()},
            },
        )
        prepared = builder.write(
            evaluation / "prepared-dev.json",
            {
                **builder.read_bound(original),
                "parent_evaluation": original,
                "model_preparation": configuration,
            },
        )
        identity = {
            "kind": "sft",
            "engineering_fixture_only": True,
            "configuration_sha256": configuration["sha256"],
        }
        numerical = {
            "numerical_regime": "native_bf16_fp32_lora",
            "schema_constrained": False,
        }
        run = builder.write(
            self.root / "dev-inference.json",
            {
                "status": "inference_complete",
                "requested": 471,
                "observed": 471,
                "final_test_bodies_read": False,
                "model_identity": identity,
                **numerical,
            },
        )
        records, cases = [], []
        actions = list(builder.gate.contract.ACTIONS)
        for index in range(471):
            action, language = (
                actions[index // 2 % len(actions)],
                "en" if index % 2 == 0 else "ko",
            )
            preserve = action == "no_action"
            case_id = f"engineering-only-{index}"
            cases.append(
                {
                    "case_id": case_id,
                    "cohort": "canonical_dev322"
                    if index < 322
                    else "supplemental_dev149",
                }
            )
            records.append(
                {
                    "case_id": case_id,
                    "reference_action": action,
                    "predicted_action": action,
                    "language": language,
                    "judgment": "preserve" if preserve else "edit_required",
                    "status": "completed",
                    "displayed": not preserve,
                    "useful": not preserve,
                    "preserved": preserve,
                    "model_abstained": preserve,
                    "valid": True,
                    "outcome_match": True,
                    "suppressed": False,
                    "source_integrity": "passed",
                    "generation_ms": None,
                }
            )
        records_path = self.root / "fixture-graded.jsonl"
        development.data.write_rows(records_path, records)
        report = builder.write(
            self.root / "fixture-report.json",
            {
                "status": "graded_development",
                "final_test": False,
                "format": "json",
                "condition": "canonical_full_context",
                "original_development_cases": 471,
                "excluded_before_inference": 0,
                "preparation": prepared,
                "inference_run": run,
                "inference_manifest": self.evidence,
                "observations": self.evidence,
                "graded": builder.binding(records_path),
                "overall": development.grading.summarize(records),
                "model_identity": identity,
                **numerical,
            },
        )
        mapping = builder.write(
            self.root / "fixture-cohorts.json",
            {
                "counts": {"canonical_dev322": 322, "supplemental_dev149": 149},
                "cases": cases,
                "release_manifest": release,
                "evaluation_manifest": original,
            },
        )
        summary_path = self.root / "fixture-summary.json"
        expected = development.summarize(report["path"], mapping["path"], summary_path)
        with (
            mock.patch.object(builder, "FINAL", release_dir),
            mock.patch.object(builder, "FINAL_SHA", release["sha256"]),
        ):
            actual, _ = builder.verified_dev_summary(builder.binding(summary_path))
            self.assertEqual(actual, expected)
            changed = copy.deepcopy(expected)
            changed["macro_action_f1"]["value"] = 0.99
            altered = builder.write(self.root / "authored-score.json", changed)
            with self.assertRaisesRegex(
                ValueError, "differs from its actual graded rows"
            ):
                builder.verified_dev_summary(altered)

    def test_generated_q8_policy_matches_the_auxiliary_metadata_validator(self):
        import infer_deployed_task_test_actions_v3 as auxiliary

        assets = self.root / "assets"
        assets.mkdir()
        profile = copy.deepcopy(builder.small.data.PROFILES["lfm25-1.2b"])
        for name in builder.small.data.ASSET_FILES:
            builder.write(
                assets / name,
                {"vocab_size": 65536}
                if name == "config.json"
                else {"engineering_fixture_only": True},
            )
        receipt = {
            "version": builder.small.FAMILY,
            "model": "lfm25-1.2b",
            "repository": profile["repository"],
            "revision": profile["revision"],
            "weights_downloaded": False,
            "files_sha256": {
                name: builder.sha(assets / name)
                for name in builder.small.data.ASSET_FILES
            },
        }
        builder.write(assets / "assets.json", receipt)
        config = {
            "version": builder.small.FAMILY,
            "model": "lfm25-1.2b",
            "base": profile,
            "assets_path": str(assets),
            "assets": receipt,
            "contract_version": builder.gate.contract.VERSION,
            "format": "json",
            "final_test_bodies_read": False,
            "context_tokens": 4096,
            "max_completion_tokens": 512,
            "generation": {
                "max_new_tokens": 512,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
                "eos_token_id": [7],
                "pad_token_id": 0,
                "bos_token_id": 1,
            },
        }
        config_path = self.root / "configuration.json"
        builder.write(config_path, config)
        tokenizer = builder.small.tokenizer_identity(config)
        configuration = builder.binding(config_path)
        identity = {
            "kind": "sft",
            "adapter_sha256": "a" * 64,
            "configuration_sha256": configuration["sha256"],
        }
        metadata = builder.write(
            self.root / "native-metadata.json", {"tokenizer": tokenizer}
        )
        model = {
            "runtime_family": builder.small.FAMILY,
            "role": "sft",
            "identity": identity,
            "inference_configuration": configuration,
            "preparation_manifest": metadata,
        }
        model_binding = builder.write(self.root / "selected-model.json", model)
        model_file, binary = (
            self.root / "fixture-Q8_0.gguf",
            self.root / "fixture-server",
        )
        model_file.write_bytes(b"Engineering fixture bytes, never loaded or executed.")
        binary.write_bytes(b"Engineering fixture binary, never executed.")
        gateway = {
            "version": 2,
            "model_id": "engineering-selected-q8",
            "model_path": str(model_file),
            "model_sha256": builder.sha(model_file),
            "chat_template_path": str(
                Path(config["assets_path"]) / "chat_template.jinja"
            ),
            "chat_template_sha256": config["assets"]["files_sha256"][
                "chat_template.jinja"
            ],
            "adapter_path": None,
            "adapter_sha256": None,
            "output_format": "json",
            "capture_runtime_evidence": False,
            "cache_prompt": True,
            "endpoint": "http://127.0.0.1:18091",
        }
        deployment_dir = self.root / "deployment"
        builder.write(deployment_dir / "gateway.json", gateway)
        command = [str(binary), "-c", "4096"]
        builder.write(deployment_dir / "command.json", command)
        export = builder.write(
            self.root / "export.json",
            {
                "status": "gguf_exported_runtime_validation_pending",
                "kind": "sft",
                "base": config["base"],
                "adapter_sha256": identity["adapter_sha256"],
                "training_configuration_sha256": configuration["sha256"],
                "gguf_sha256": gateway["model_sha256"],
                "quantization": "Q8_0",
            },
        )
        report = {
            "purpose": "ui_performance",
            "kind": "sft",
            "base": config["base"],
            "export_receipt": export,
            "native_preparation": configuration,
            "training_adapter_sha256": identity["adapter_sha256"],
            "runtime_registry": self.evidence,
            "runtime_files_sha256": {
                str(binary.relative_to(builder.ROOT)): builder.sha(binary)
            },
            "numerical_regime": "native_bf16_merged_q8_0",
        }
        deployment = builder.write(deployment_dir / "deployment.json", report)
        context = {
            "registration": {
                "models": {"sft": model_binding},
                "deployment_evaluation_plan": builder.binding(builder.DEPLOYMENT_PLAN),
                "deployment_evaluation_sources_sha256": auxiliary.sources(),
            },
            "registration_binding": self.evidence,
            "selected_model": "sft",
            "models": {"sft": model},
            "ready": {
                "protocol_id": "engineering-only",
                "selected_deployment_receipt": deployment,
            },
            "conditions": {"conditions": builder.gate.conditions.condition_specs([])},
        }
        with (
            mock.patch.object(
                auxiliary, "_deployment_check", return_value=(gateway, command, report)
            ),
            mock.patch.object(auxiliary, "PLAN_SHA256", builder.DEPLOYMENT_PLAN_SHA),
        ):
            policy = builder.deployed_policy(context)
        self.assertEqual(policy["numerical_regime"], "native_bf16_merged_q8_0")
        self.assertEqual(policy["selected_native_run_id"], "sft-json")
        self.assertEqual(
            policy["request_settings"], auxiliary.request_settings(gateway)
        )
        self.assertIsNone(policy["model_identity"]["adapter_sha256"])
        self.assertFalse(policy["body_access_authorized"])


if __name__ == "__main__":
    unittest.main()
