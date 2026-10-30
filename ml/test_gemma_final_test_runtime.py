"""Synthetic metadata and CPU controller fixtures; no model or reserved body access."""

import json
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import assemble_task_test_actions_v3 as assembly
import compare_task_test_actions_v3 as comparison
import gemma_final_test_runtime as gemma
import infer_task_test_actions_v3 as native
import test_prepare_task_test_actions_v3 as fixtures


class GemmaFinalRuntimeTests(unittest.TestCase):
    def setUp(self):
        (gemma.ROOT / "artifacts").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="gemma-final-fixture-", dir=gemma.ROOT / "artifacts"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prep = self.root / "prepared"
        self.prep.mkdir()
        assets = self.prep / "native"
        assets.mkdir()
        for name in gemma.evidence.prep.TOKENIZER_FILES:
            gemma.write(
                assets / name,
                {
                    "text_config": {
                        "vocab_size": 262144,
                        "max_position_embeddings": 262144,
                    }
                }
                if name == "config.json"
                else {"engineering_fixture": True},
            )
        files = {
            name: gemma.sha(assets / name)
            for name in gemma.evidence.prep.TOKENIZER_FILES
        }
        gemma.write(
            self.prep / "native-pin.json",
            {
                "repository": gemma.evidence.prep.gemma.REPOSITORY,
                "revision": gemma.evidence.prep.gemma.REVISION,
                "files": {k: {"sha256": v} for k, v in files.items()},
            },
        )
        self.config = {
            "format": "json",
            "phase": "substantive",
            "tokenizer": {"path": "native", "files_sha256": files},
            "base": {
                "repository": gemma.evidence.prep.gemma.REPOSITORY,
                "revision": gemma.evidence.prep.gemma.REVISION,
            },
            "native_pin_sha256": gemma.sha(self.prep / "native-pin.json"),
            "generation": {"eos_token_id": [1, 106, 50]},
            "training": {"unique_cases": 1261},
            "sources_sha256": {"frozen_trainer.py": "synthetic-source"},
        }
        gemma.write(self.prep / "configuration.json", self.config)
        gemma.write(self.prep / "manifest.json", {"engineering_fixture": True})
        self.config_path = self.prep / "configuration.json"
        self.audit = self.root / "context-audit.json"
        gemma.write(
            self.audit,
            {
                "scope": "exposed_development_only",
                "cases": 471,
                "over_budget": 0,
                "canonical_max": 3584,
                "icl_max": 4452,
                "fixed_demonstration_ids": gemma.DEMONSTRATION_IDS,
                "model_weights_loaded": False,
                "final_test_bodies_read": False,
                "native_pin": gemma.binding(self.prep / "native-pin.json"),
            },
        )
        self.sft, self.reload = self.make_run("sft")
        self.dpo, self.dpo_reload = self.make_run("sft_dpo")
        self.dpo_prepared = self.root / "dpo-prepared"
        gemma.write(
            self.dpo_prepared / "configuration.json", {"engineering_fixture": True}
        )
        self.patch(gemma, "validate_configuration")
        self.patch(gemma, "verify_fallback", return_value=(self.config, {}))
        self.patch(gemma, "CONTEXT_AUDIT", self.audit)
        self.patch(gemma, "CONTEXT_AUDIT_SHA256", gemma.sha(self.audit))
        plan = self.root / "context-plan.md"
        plan.write_text("Synthetic plan for runtime metadata tests only.\n")
        self.patch(gemma, "CONTEXT_PLAN", plan)
        self.patch(gemma.evidence, "PLAN", plan)
        release = self.root / "training-manifest.json"
        gemma.write(release, {"engineering_fixture": True})
        self.patch(gemma.evidence, "RELEASE", release)
        self.patch(gemma.evidence, "verify_trained", side_effect=self.verified)
        self.patch(assembly, "source_plan", side_effect=self.source_plan)

    def patch(self, owner, name, *args, **kwargs):
        patch = mock.patch.object(owner, name, *args, **kwargs)
        value = patch.start()
        self.addCleanup(patch.stop)
        return value

    def make_run(self, role):
        training, reloaded = (
            self.root / (role + "-training"),
            self.root / (role + "-reload"),
        )
        for directory in (training, reloaded):
            gemma.write(
                directory / "manifest.json",
                {"engineering_fixture": True, "role": role, "path": str(directory)},
            )
        gemma.write(
            training / "runtime.json", {"engineering_fixture": True, "role": role}
        )
        return training, reloaded

    def verified(
        self,
        config_path,
        training,
        reloaded,
        *,
        kind=gemma.evidence.sft.VERSION,
        parent=None,
        **kwargs,
    ):
        role = "sft" if kind == gemma.evidence.sft.VERSION else "sft_dpo"
        identity = {
            **gemma.native_identity(self.config),
            "kind": kind,
            "adapter_sha256": "engineering-" + role,
            "configuration_sha256": gemma.sha(config_path),
        }
        if role == "sft_dpo":
            self.assertEqual(
                parent, self.verified(config_path, self.sft, self.reload)["identity"]
            )
        return {
            "identity": identity,
            "training": gemma.binding(Path(training) / "manifest.json"),
            "reload": gemma.binding(Path(reloaded) / "manifest.json"),
            "environment": {"matched": True},
            "training_source_sha256": {"frozen_trainer.py": "synthetic-source"},
        }

    def source_plan(self, path):
        frozen, overlay = self.root / "frozen_trainer.py", self.root / "entry.py"
        frozen.write_text("ENGINEERING_FIXTURE = 'frozen training'\n")
        overlay.write_text("ENGINEERING_FIXTURE = 'final inference overlay'\n")
        self.config["sources_sha256"] = {frozen.name: gemma.sha(frozen)}
        return self.config, {
            "ml/" + frozen.name: {
                "path": frozen,
                "sha256": gemma.sha(frozen),
                "origin": "frozen_preparation",
            },
            assembly.ENTRY: {
                "path": overlay,
                "sha256": gemma.sha(overlay),
                "origin": "final_only_overlay",
            },
        }

    def register(self, *, omission=None, native_runtime_adoption=None):
        kwargs = (
            {"dpo_omission": omission}
            if omission
            else {
                "dpo_training": self.dpo,
                "dpo_reloaded": self.dpo_reload,
                "dpo_preparation": self.dpo_prepared,
            }
        )
        return gemma.register(
            self.config_path,
            self.root / "registered",
            sft_training=self.sft,
            sft_reloaded=self.reload,
            native_runtime_adoption=native_runtime_adoption,
            **kwargs,
        )

    def test_completed_family_preserves_frozen_sources_and_rejects_other_dpo_parent(
        self,
    ):
        manifest = self.register()
        self.assertFalse(manifest["body_access_authorized"])
        models = {
            role: gemma.read(gemma.bound(ref))
            for role, ref in manifest["models"].items()
        }
        self.assertEqual(set(models), {"native_base", "sft", "sft_dpo"})
        runtime = assembly.load_runtime(self.root / "registered/runtime")
        self.assertEqual(runtime["runtime_family"], gemma.FAMILY)
        gemma.write(
            self.root / "runtime-policy.json",
            {"inference_policy": manifest["inference_policy"]},
        )
        freeze = {
            "models": manifest["models"],
            "source_sha256": {
                "ml/gemma_final_test_runtime.py": gemma.sha(gemma.__file__)
            },
            "policies": {"runtime": gemma.binding(self.root / "runtime-policy.json")},
        }
        self.assertEqual(
            comparison.full_cohort_policy(freeze, gemma.bound)[
                "inference_prefix_limit"
            ],
            7680,
        )
        meta = gemma.read(gemma.bound(manifest["preparation"]))
        models["sft_dpo"]["parent_identity_sha256"] = "another-parent"
        with self.assertRaisesRegex(ValueError, "DPO parent changed"):
            gemma.verify_model_set(
                models, meta["tokenizer"], meta["training_dataset_manifest"]
            )
        (self.root / "registered/runtime/ml/frozen_trainer.py").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "runtime source changed"):
            assembly.load_runtime(self.root / "registered/runtime")

    def test_only_reviewed_same_parent_omission_keeps_five_roles(self):
        parent = self.verified(self.config_path, self.sft, self.reload)["identity"]
        mining = self.root / "mining/run.json"
        gemma.write(mining, {"engineering_fixture": True})
        decision = {
            "version": "gemma26-final-dpo-decision-1",
            "status": "registered_before_test",
            "decision": "eligible_omission",
            "parent": parent,
            "mining": gemma.binding(mining),
            "reason_codes": ["fewer_than_eight_strict_pairs"],
            "counts": {"requested": 1261, "pairs": 7, "preservation_pairs": 1},
            "plan": gemma.binding(gemma.evidence.PLAN),
            "body_access_authorized": False,
            "final_test_bodies_read": False,
            "final_registration_performed": False,
        }
        omission = self.root / "omission.json"
        gemma.write(omission, decision)
        with mock.patch.object(
            gemma.preferences, "verify_mining", return_value=decision["counts"]
        ):
            manifest = self.register(omission=omission)
            freeze = {
                "models": manifest["models"],
                "selected_model": "sft",
                "dpo_omission": manifest["dpo_omission"],
                "source_sha256": {
                    "ml/gemma_final_test_runtime.py": gemma.sha(gemma.__file__)
                },
            }
            canonical, roles, details = comparison.role_policy(freeze, gemma.bound)
            self.assertEqual(len(roles), 5)
            self.assertEqual(len(canonical), 4)
            self.assertEqual(details["counts"]["pairs"], 7)
            decision["reason_codes"] = []
            with self.assertRaisesRegex(ValueError, "full TRAIN pair scarcity"):
                gemma.validate_omission(decision, self.config, parent)
        with self.assertRaisesRegex(ValueError, "Supply same-parent"):
            gemma.register(
                self.config_path,
                self.root / "silent-skip",
                sft_training=self.sft,
                sft_reloaded=self.reload,
            )

    def test_terminal_and_whole_cohort_context_policy_cannot_drift(self):
        identity = gemma.tokenizer_identity(self.config_path)
        self.assertEqual(identity["terminal_ids"], [1, 106, 50])
        identity["terminal_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "tokenizer identity changed"):
            gemma.validate_tokenizer_identity(identity)
        policy = gemma.inference_policy(self.config_path)
        for key, value in (
            ("physical_context_tokens", 4096),
            ("inference_prefix_limit", 3584),
            ("fixed_demonstration_ids", list(reversed(gemma.DEMONSTRATION_IDS))),
        ):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "inference policy"),
            ):
                gemma.validate_inference_policy({**policy, key: value})

    def test_adopted_overlay_is_common_to_all_roles_and_rejects_changed_execution_policy(
        self,
    ):
        decision = self.root / "adoption.json"
        gemma.write(decision, {"engineering_fixture": True})
        with mock.patch.object(gemma, "validate_native_adoption") as verify:
            result = self.register(native_runtime_adoption=decision)
            self.assertTrue(
                any(
                    call.kwargs.get("parent")
                    == self.verified(self.config_path, self.sft, self.reload)[
                        "identity"
                    ]
                    for call in verify.call_args_list
                )
            )
        models = [gemma.read(gemma.bound(value)) for value in result["models"].values()]
        self.assertTrue(
            all(
                model["execution_runtime"] == models[0]["execution_runtime"]
                for model in models
            )
        )
        manifest = self.root / "registered/runtime/runtime-manifest.json"
        saved = gemma.read(manifest)
        spec = {
            "runtime_manifest_sha256": gemma.sha(manifest),
            "inference_policy_sha256": saved["inference_policy_sha256"],
            "configuration_sha256": saved["configuration_sha256"],
        }
        overlay = gemma.execution_overlay(spec, manifest)
        self.assertEqual(overlay["adoption_sha256"], gemma.sha(decision))
        self.assertEqual(overlay["settings"], gemma.native_overlay.OVERLAY)
        with self.assertRaisesRegex(ValueError, "another frozen policy"):
            gemma.execution_overlay(
                {**spec, "inference_policy_sha256": "another-policy"}, manifest
            )
        saved["native_inference_overlay"]["settings"]["compile"] = True
        manifest.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "frozen execution runtime manifest"):
            gemma.execution_overlay(spec, manifest)


class GroupedCompletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="grouped-completion-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reg = {
            "version": "gemma26-grouped-development-1",
            "model_identity": {"kind": gemma.evidence.sft.VERSION},
            "runtime_overlay": gemma.native_overlay.OVERLAY,
            "prospective_registration_sha256": gemma.NATIVE_PROSPECTIVE_SHA,
            "cohorts": {
                scope: {"case_ids": [f"synthetic-{scope}-{i}" for i in range(n)]}
                for scope, n in (("dev", 471), ("history", 24))
            },
        }
        gemma.write(self.root / "registration.json", self.reg)
        self.info = {
            "version": self.reg["version"],
            "status": "completed",
            "fatal_error": None,
            "request_failures": 0,
            "observed_by_scope": {"dev": 471, "history": 24},
            "runtime_fingerprint": "synthetic-assignment",
            "registration_sha256": gemma.sha(self.root / "registration.json"),
            "runtime_overlay": gemma.native_overlay.OVERLAY,
            "model_identity": self.reg["model_identity"],
        }
        gemma.write(self.root / "run.json", self.info)
        gemma.write(
            self.root / "adapter-parity.json",
            {
                "adapter_tensors_identical": True,
                "tensors": {f"tensor-{i}": {} for i in range(410)},
            },
        )
        for scope, count in (("dev", 471), ("history", 24)):
            folder = self.root / scope
            run = {
                **self.info,
                "status": "inference_complete",
                "requested": count,
                "observed": count,
                "completed_with_fatal_failure": False,
                "scope": scope,
            }
            gemma.write(folder / "run.json", run)
            gemma.write(folder / "runtime.json", self.info)
            trace = [
                {
                    "input_tokens": 3,
                    "experts": "grouped_mm",
                    "cache_type": "DynamicCache",
                    "cache_length_before": 0,
                }
            ]
            gemma.evidence.prep.write_rows(
                folder / "observations.jsonl",
                [
                    {
                        "case_id": key,
                        "status": "completed",
                        "output_ids": [1],
                        "backend_trace": trace,
                    }
                    for key in self.reg["cohorts"][scope]["case_ids"]
                ],
            )
            gemma.write(
                folder / "manifest.json",
                {
                    "status": "inference_complete",
                    "files_sha256": {
                        p.name: gemma.sha(p) for p in folder.iterdir() if p.is_file()
                    },
                },
            )
        self.publish()

    def publish(self):
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "version": self.reg["version"],
                    "status": "completed",
                    "files_sha256": {
                        str(p.relative_to(self.root)): gemma.sha(p)
                        for p in self.root.rglob("*")
                        if p.is_file() and p != self.root / "manifest.json"
                    },
                }
            )
        )

    def test_completed_children_do_not_hide_late_outer_failure_or_missing_adapter_parity(
        self,
    ):
        gemma.verify_native_development_run(self.root)
        self.info.update(
            status="completed_with_failures", fatal_error={"type": "cleanup-failure"}
        )
        (self.root / "run.json").write_text(json.dumps(self.info))
        self.publish()
        with self.assertRaisesRegex(ValueError, "failed after a child"):
            gemma.verify_native_development_run(self.root)
        self.info.update(status="completed", fatal_error=None)
        (self.root / "run.json").write_text(json.dumps(self.info))
        parity = gemma.read(self.root / "adapter-parity.json")
        parity["adapter_tensors_identical"] = False
        (self.root / "adapter-parity.json").write_text(json.dumps(parity))
        self.publish()
        with self.assertRaisesRegex(ValueError, "unchanged adapter"):
            gemma.verify_native_development_run(self.root)

    def test_history_census_and_backend_bytes_are_required_for_adoption(self):
        path = self.root / "history/observations.jsonl"
        original = gemma.evidence.prep.rows(path)
        changed = copy.deepcopy(original)
        changed[0]["backend_trace"][0]["experts"] = "eager"
        path.write_text("".join(json.dumps(row) + "\n" for row in changed))
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            gemma.verify_native_development_run(self.root)
        child = gemma.read(self.root / "history/manifest.json")
        child["files_sha256"]["observations.jsonl"] = gemma.sha(path)
        (self.root / "history/manifest.json").write_text(json.dumps(child))
        self.publish()
        with self.assertRaisesRegex(ValueError, "backend/cache"):
            gemma.verify_native_development_run(self.root)

    def test_adoption_rejects_wrong_parent_dpo_based_choice_and_failed_checks(self):
        source = self.root / "source.py"
        source.write_text("# synthetic metadata fixture\n")
        prospective = self.root / "prospective.json"
        original = gemma.binding(self.root / "registration.json")
        gemma.write(
            prospective,
            {
                "version": "gemma26-grouped-dev-prospective-1",
                "criteria": {"synthetic": True},
                "bindings": {"original_sft_dev": original},
            },
        )
        value = {
            "version": gemma.NATIVE_ADOPTION_VERSION,
            "status": "adopted",
            "backend": "grouped_mm",
            "all_passed": True,
            "checks": {
                key: True
                for key in (
                    "minimum_numeric_gates",
                    "cohort_quality",
                    "action_language_successes",
                    "no_new_entire_typed_subgroup_failure",
                    "relative_speed",
                )
            },
            "runtime_overlay": gemma.native_overlay.OVERLAY,
            "prospective_registration": gemma.binding(prospective),
            "criteria": {"synthetic": True},
            "model_identity": self.reg["model_identity"],
            "runtime_selection_uses_DPO_results": False,
            "DPO_thresholds_changed": False,
            "final_test_bodies_read": False,
            "model_execution": False,
            "inputs": {
                "original_SFT_summary": original,
                "optimized_outer_manifest": gemma.binding(self.root / "manifest.json"),
            },
            "implementation": gemma.binding(source),
        }
        with (
            mock.patch.object(gemma, "NATIVE_PROSPECTIVE_SHA", gemma.sha(prospective)),
            mock.patch.object(
                gemma,
                "verify_native_development_run",
                return_value=(self.info, self.reg),
            ),
        ):
            gemma.validate_native_adoption(value, parent=self.reg["model_identity"])
            with self.assertRaisesRegex(ValueError, "same-SFT runtime decision"):
                gemma.validate_native_adoption(
                    value,
                    parent={**self.reg["model_identity"], "adapter_sha256": "other"},
                )
            for key, replacement in (
                ("runtime_selection_uses_DPO_results", True),
                ("status", "rejected"),
            ):
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(ValueError, "same-SFT runtime decision"),
                ):
                    gemma.validate_native_adoption({**value, key: replacement})
            value["checks"]["relative_speed"] = False
            with self.assertRaisesRegex(ValueError, "same-SFT runtime decision"):
                gemma.validate_native_adoption(value)
            value["checks"]["relative_speed"] = True
            value["checks"]["minimum_numeric_gates"] = False
            with self.assertRaisesRegex(ValueError, "same-SFT runtime decision"):
                gemma.validate_native_adoption(value)


class GemmaFinalControllerTests(unittest.TestCase):
    def test_native_controller_records_overlay_dispatch_and_keeps_backend_failure_output(
        self,
    ):
        fixture = fixtures.SealedTaskTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        release = fixture.seal(fixture.prepare())
        portable = fixture.root / "grouped-portable"
        receipt = fixtures.sealed.export_execution(
            fixture.freeze_path, release, "native_base", portable
        )
        spec, requests = native.load_bundle(portable, receipt["sha256"])
        spec["runtime_family"] = gemma.FAMILY

        class Generation(SimpleNamespace):
            def to_dict(self):
                return vars(self)

        class DynamicCache:
            def get_seq_length(self):
                return 0

        class Output:
            def __getitem__(self, key):
                return SimpleNamespace(tolist=lambda: [1])

        class Model:
            def __init__(self):
                self.config = SimpleNamespace(
                    text_config=SimpleNamespace(_experts_implementation="grouped_mm")
                )
                self.calls, self.closed = [], False

            def register_forward_pre_hook(self, callback, **kwargs):
                self.callback = callback
                return SimpleNamespace(remove=lambda: setattr(self, "closed", True))

            def eval(self):
                pass

            def gradient_checkpointing_disable(self):
                pass

            def generate(self, *, input_ids, attention_mask, generation_config):
                self.calls.append(generation_config.to_dict())
                self.config.text_config._experts_implementation = (
                    "grouped_mm" if len(self.calls) == 1 else "eager"
                )
                self.callback(
                    self,
                    (),
                    {"input_ids": input_ids, "past_key_values": DynamicCache()},
                )
                return Output()

        model = Model()
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda: None, max_memory_allocated=lambda: 0
            ),
            tensor=lambda values, **kwargs: SimpleNamespace(shape=[1, len(values[0])]),
            long=object(),
            ones_like=lambda value: value,
            inference_mode=nullcontext,
            OutOfMemoryError=MemoryError,
        )
        runtime = {
            "native_inference_overlay": {
                "settings": gemma.native_overlay.OVERLAY,
                "adoption_sha256": "synthetic",
            }
        }
        loaded = (
            {"generation": {"eos_token_id": [1]}},
            spec["model_identity"],
            torch,
            model,
            SimpleNamespace(decode=lambda *args, **kwargs: "synthetic output"),
            "synthetic-device",
            runtime,
        )
        output = fixture.root / "grouped-observations"
        with (
            mock.patch.object(native, "load_bundle", return_value=(spec, requests)),
            mock.patch.object(gemma, "load_runtime", return_value=loaded),
            mock.patch.object(
                native.conditions,
                "native_prefix",
                side_effect=[r["prefix_ids"] for r in requests],
            ),
            mock.patch.dict(
                "sys.modules",
                {"transformers": SimpleNamespace(GenerationConfig=Generation)},
            ),
        ):
            result = native.run(
                portable, receipt["sha256"], fixture.inference_config["path"], output
            )
        records = gemma.evidence.prep.rows(output / "observations.jsonl")
        self.assertEqual(result["observed"], 2)
        self.assertEqual(
            [r["status"] for r in records], ["completed", "request_failure"]
        )
        self.assertEqual(records[1]["output_ids"], [1])
        self.assertEqual(records[1]["backend_trace"][0]["experts"], "eager")
        self.assertTrue(model.closed)
        self.assertTrue(
            all(
                c["cache_implementation"] == "dynamic" and c["disable_compile"] is True
                for c in model.calls
            )
        )

    def test_third_registered_lr_failure_is_recomputed_and_cannot_be_a_pass(self):
        import build_small_final_freeze_actions_v3 as builder

        (gemma.ROOT / "artifacts").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="gemma-fallback-fixture-", dir=gemma.ROOT / "artifacts"
        ) as temporary:
            root = Path(temporary)
            plan = root / "recovery-plan.md"
            plan.write_text("Synthetic plan for recovery metadata tests only.\n")
            for name, value in (
                ("LR_RECOVERY_PLAN", plan),
                ("LR_RECOVERY_PLAN_SHA", gemma.sha(plan)),
            ):
                patch = mock.patch.object(gemma.evidence.fallback, name, value)
                patch.start()
                self.addCleanup(patch.stop)
            original = {"model": "lfm25-1.2b", "training": {"learning_rate": 5e-5}}
            recovery = {"model": "lfm25-1.2b", "training": {"learning_rate": 2e-4}}
            for name, value in (
                ("original.json", original),
                ("recovery.json", recovery),
                ("failure.json", {"engineering_fixture": True}),
            ):
                gemma.write(root / name, value)
            gemma.write(
                root / "draft.json",
                {
                    "inputs": {
                        "small_configurations": [gemma.binding(root / "original.json")],
                        "lr_recovery_configuration": gemma.binding(
                            root / "recovery.json"
                        ),
                        "lr_recovery_plan": gemma.binding(
                            gemma.evidence.fallback.LR_RECOVERY_PLAN
                        ),
                    }
                },
            )
            gemma.write(
                root / "protocol.json",
                {
                    "readiness_manifest": gemma.binding(root / "draft.json"),
                    "completed_lr_recovery_failure_summary": gemma.binding(
                        root / "failure.json"
                    ),
                },
            )
            summary = {
                "model_identity": {
                    "kind": "sft",
                    "configuration_sha256": gemma.sha(root / "recovery.json"),
                },
                "minimum_numeric_quality_gate": {"passed": False},
            }
            report = {
                "numerical_regime": "native_bf16_fp32_lora",
                "schema_constrained": False,
            }
            with (
                mock.patch.object(
                    gemma.evidence, "verify_preparation", return_value=({}, {})
                ),
                mock.patch.object(
                    builder, "verified_dev_summary", return_value=(summary, report)
                ) as verify,
            ):
                gemma.verify_fallback(root / "config.json")
                verify.assert_called_once_with(gemma.binding(root / "failure.json"))
                summary["minimum_numeric_quality_gate"]["passed"] = True
                with self.assertRaisesRegex(ValueError, "LR recovery failure"):
                    gemma.verify_fallback(root / "config.json")

    def test_icl_above_old_cap_keeps_all_roles_and_overflow_fails_whole_cohort(self):
        for size in (4000, 7681):
            fixture = fixtures.SealedTaskTests()
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)

            class Tokenizer(fixtures.FixtureTokenizer):
                def apply_chat_template(self, messages, **kwargs):
                    return (
                        [2] * size
                        if len(messages) > 2
                        else super().apply_chat_template(messages, **kwargs)
                    )

            gate = fixtures.sealed.load_gate(fixture.freeze_path)
            gate["policies"]["runtime"]["tokenizer"]["runtime_family"] = gemma.FAMILY
            output = fixture.root / "gemma-full-cohort"
            with mock.patch.object(fixtures.sealed, "load_gate", return_value=gate):
                if size <= 7680:
                    fixtures.sealed.freeze_inputs(
                        fixture.freeze_path,
                        fixture.candidates(),
                        output,
                        tokenizer=Tokenizer(),
                    )
                    self.assertEqual(
                        fixtures.sealed.rows(output / "run-exclusions.jsonl"), []
                    )
                    ids = [
                        r["id"] for r in fixtures.sealed.rows(output / "inputs.jsonl")
                    ]
                    for role in fixtures.sealed.RUN_ROLES:
                        self.assertEqual(
                            [
                                r["case_id"]
                                for r in fixtures.sealed.rows(
                                    output / f"requests-{role}.jsonl"
                                )
                            ],
                            ids,
                        )
                else:
                    with self.assertRaisesRegex(
                        ValueError, "fail the whole preparation"
                    ):
                        fixtures.sealed.freeze_inputs(
                            fixture.freeze_path,
                            fixture.candidates(),
                            output,
                            tokenizer=Tokenizer(),
                        )
                    self.assertFalse(output.exists())

    def test_gemma_dispatch_failure_retains_every_request_without_model_execution(self):
        fixture = fixtures.SealedTaskTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        release = fixture.seal(fixture.prepare())
        portable = fixture.root / "gemma-portable"
        fixtures.sealed.export_execution(
            fixture.freeze_path, release, "native_base", portable
        )
        spec = gemma.read(portable / "execution.json")
        spec.update(
            runtime_family=gemma.FAMILY,
            vocab_size=262144,
            physical_context_tokens=8192,
            inference_prefix_limit=7680,
            output_tokens=512,
            inference_policy_sha256="synthetic-policy",
        )
        (portable / "execution.json").write_text(json.dumps(spec))
        manifest = gemma.read(portable / "manifest.json")
        manifest["files_sha256"]["execution.json"] = gemma.sha(
            portable / "execution.json"
        )
        (portable / "manifest.json").write_text(json.dumps(manifest))
        runtime = {
            "source_sha256": spec["source_sha256"],
            "runtime_manifest_sha256": None,
            "runtime_family": gemma.FAMILY,
            "inference_policy_sha256": spec["inference_policy_sha256"],
        }
        with (
            mock.patch.object(native, "execution_runtime", return_value=runtime),
            mock.patch.object(
                gemma,
                "load_runtime",
                side_effect=ValueError("synthetic loading failure"),
            ) as loader,
        ):
            result = native.run(
                portable,
                gemma.sha(portable / "manifest.json"),
                fixture.inference_config["path"],
                fixture.root / "failed-run",
            )
        loader.assert_called_once()
        self.assertEqual(result["requested"], result["observed"])
        rows = gemma.evidence.prep.rows(fixture.root / "failed-run/observations.jsonl")
        self.assertTrue(
            all(
                r["status"] == "request_failure" and not r["generation_attempted"]
                for r in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
