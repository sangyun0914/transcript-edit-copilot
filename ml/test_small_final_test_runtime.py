"""Engineering metadata fixtures only; no native model or held-out body is loaded."""

import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import assemble_task_test_actions_v3 as assembly
import compare_task_test_actions_v3 as comparison
import infer_task_test_actions_v3 as native
import small_final_test_runtime as small
import test_prepare_task_test_actions_v3 as sealed_fixtures


class SmallFinalRuntimeTests(unittest.TestCase):
    def setUp(self):
        (small.ROOT / "artifacts").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="small-final-runtime-engineering-", dir=small.ROOT / "artifacts"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        context_plan = self.root / "context-plan.md"
        self.omission_plan = self.root / "dpo-plan.md"
        for path in (context_plan, self.omission_plan):
            path.write_text("Synthetic plan for metadata binding tests only.\n")
        for name, value in (
            ("CONTEXT_PLAN", context_plan),
            ("CONTEXT_PLAN_SHA256", small.sha(context_plan)),
            ("OMISSION_PLAN_SHA256", small.sha(self.omission_plan)),
        ):
            patch = mock.patch.object(small, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        assets = self.root / "assets"
        assets.mkdir()
        profile = copy.deepcopy(small.data.PROFILES["lfm25-1.2b"])
        for name in small.data.ASSET_FILES:
            small.data.write(
                assets / name,
                {"vocab_size": 65536} if name == "config.json" else {"fixture": True},
            )
        receipt = {
            "version": small.FAMILY,
            "model": "lfm25-1.2b",
            "repository": profile["repository"],
            "revision": profile["revision"],
            "weights_downloaded": False,
            "files_sha256": {
                name: small.sha(assets / name) for name in small.data.ASSET_FILES
            },
        }
        small.data.write(assets / "assets.json", receipt)
        audit_path = self.root / "context-audit.json"
        small.data.write(
            audit_path,
            {
                "version": "small-final-full-cohort-icl-budget-audit-1",
                "scope": "exposed_development_only",
                "model_weights_loaded": False,
                "final_test_bodies_read": False,
                "fixed_demonstration_ids": small.DEMONSTRATION_IDS,
                "models": {
                    "lfm25-1.2b": {
                        "assets": {"sha256": small.sha(assets / "assets.json")},
                        "cases": 471,
                        "over_budget": 0,
                        "canonical_max": 3200,
                        "icl_max": 4100,
                        "native_config_max_position_embeddings": 128000,
                    }
                },
            },
        )
        for name, value in (
            ("CONTEXT_AUDIT", audit_path),
            ("CONTEXT_AUDIT_SHA256", small.sha(audit_path)),
        ):
            patch = mock.patch.object(small, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        preparation = self.root / "preparation"
        preparation.mkdir()
        small.data.write_rows(
            preparation / "train.jsonl.gz",
            [
                {
                    "case_id": "engineering-train",
                    "input_ids": [2, 40, 7],
                    "completion_start": 1,
                    "labels": [-100, 40, 7],
                    "attention_mask": [1, 1, 1],
                }
            ],
        )
        dataset = self.root / "dataset"
        (dataset / "cases").mkdir(parents=True)
        small.data.write_rows(
            dataset / "cases/train.jsonl",
            [{"id": "engineering-train", "split": "train"}],
        )
        small.data.write(
            dataset / "manifest.json",
            {
                "test_bodies_read": False,
                "counts": {"train": 1, "dev": 0},
                "files_sha256": {
                    "cases/train.jsonl": small.sha(dataset / "cases/train.jsonl")
                },
            },
        )
        self.config = {
            "version": small.FAMILY,
            "model": "lfm25-1.2b",
            "base": profile,
            "assets_path": str(assets),
            "assets": receipt,
            "source_cases_sha256": small.sha(dataset / "cases/train.jsonl"),
            "train_rows_sha256": small.sha(preparation / "train.jsonl.gz"),
            "contract_version": small.data.formats.contract.VERSION,
            "format": "json",
            "context_tokens": 4096,
            "max_completion_tokens": 512,
            "final_test_bodies_read": False,
            "training": {
                "sample_order": [0, 0, 0],
                "epochs": 3,
                "gradient_accumulation": 2,
            },
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
        small.data.write(preparation / "configuration.json", self.config)
        self.preparation, self.dataset = preparation, dataset / "manifest.json"
        self.sft_training, self.sft_reload = self.make_run("sft")
        self.dpo_training, self.dpo_reload = self.make_run("dpo")

    def make_run(self, kind):
        training, reloaded = (
            self.root / (kind + "-training"),
            self.root / (kind + "-reload"),
        )
        (training / "adapter").mkdir(parents=True)
        (training / "implementation").mkdir()
        reloaded.mkdir()
        small.data.write(training / "configuration.json", self.config)
        (training / "adapter/adapter_model.safetensors").write_bytes(
            b"engineering fixture bytes " + kind.encode()
        )
        (training / "implementation/trainer.py").write_text(
            "ENGINEERING_FIXTURE = True\n"
        )
        runtime = {
            "base": self.config["base"],
            "dtype": "bfloat16",
            "environment": {"engineering": "fixture"},
        }
        small.data.write(training / "runtime.json", runtime)
        info = {
            "version": small.FAMILY,
            "kind": kind,
            "status": "training_complete_reload_pending",
            "pid": 101,
            "configuration_sha256": small.sha(training / "configuration.json"),
            "rows": 1,
            "draws": 3,
            "draws_completed": 3,
            "updates_completed": 2,
            "adapter": {
                "sha256": small.sha(training / "adapter/adapter_model.safetensors")
            },
            "implementation_sha256": {
                "trainer.py": small.sha(training / "implementation/trainer.py")
            },
            "final_test_bodies_read": False,
        }
        reload = {
            "kind": kind,
            "status": "independent_reload_passed",
            "pid": 202,
            "training_configuration_sha256": info["configuration_sha256"],
            "adapter_sha256": info["adapter"]["sha256"],
            "runtime": runtime,
            "final_test_bodies_read": False,
        }
        if kind == "dpo":
            parent = small.sft.model_identity(self.sft_training, self.sft_reload)
            (training / "plan.md").write_text("Engineering fixture optimizer plan.\n")
            dpo = {
                "parent": parent,
                "pairs": 1,
                "plan_sha256": small.sha(training / "plan.md"),
                "training": {"sample_order": [0, 0, 0], "updates": 2},
            }
            small.data.write(training / "dpo_configuration.json", dpo)
            info.update(
                dpo_configuration_sha256=small.sha(training / "dpo_configuration.json"),
                parent_adapter_sha256=parent["adapter_sha256"],
            )
            reload.update(
                dpo_configuration_sha256=info["dpo_configuration_sha256"],
                parent_adapter_sha256=parent["adapter_sha256"],
                pair_diagnostics=1,
            )
        small.data.write(training / "run.json", info)
        small.data.write(reloaded / "run.json", reload)
        return training, reloaded

    def register(self, output):
        return small.register(
            self.preparation,
            self.dataset,
            self.sft_training,
            self.sft_reload,
            self.dpo_training,
            self.dpo_reload,
            output,
        )

    def test_native_pin_eos_and_budget_are_explicit(self):
        identity = small.tokenizer_identity(self.config)
        self.assertEqual(identity["eos_token_id"], 7)
        small.validate_tokenizer_identity(identity)
        identity["eos_token_id"] = 1
        with self.assertRaisesRegex(ValueError, "native identity"):
            small.validate_tokenizer_identity(identity)
        changed = copy.deepcopy(self.config)
        changed["base"]["revision"] = "unregistered"
        with self.assertRaisesRegex(ValueError, "base, contract"):
            small.validate_configuration(changed)
        changed = copy.deepcopy(self.config)
        changed["context_tokens"] = 8192
        with self.assertRaisesRegex(ValueError, "4096/512"):
            small.validate_configuration(changed)

    def test_completed_same_parent_models_register_without_body_authority(self):
        output = self.root / "registered"
        manifest = self.register(output)
        self.assertFalse(manifest["body_access_authorized"])
        self.assertFalse(manifest["final_test_bodies_read"])
        models = {
            role: small.read(small.bound(value))
            for role, value in manifest["models"].items()
        }
        meta = small.verify_model_set(
            models, small.tokenizer_identity(self.config), small.binding(self.dataset)
        )
        self.assertEqual(meta["train_rows"], 1)
        runtime = assembly.load_runtime(output / "runtime")
        self.assertEqual(runtime["runtime_family"], small.FAMILY)
        self.assertEqual(
            runtime["configuration_sha256"],
            small.sha(self.preparation / "configuration.json"),
        )
        self.assertFalse(any((output / "runtime").rglob("*.safetensors")))
        self.assertFalse(any((output / "runtime").rglob("*.jsonl")))
        (output / "runtime/ml/small_final_test_runtime.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "runtime source changed"):
            assembly.load_runtime(output / "runtime")

    def test_incomplete_sft_and_different_dpo_parent_cannot_register(self):
        path = self.sft_training / "run.json"
        original = small.read(path)
        changed = {**original, "updates_completed": math.ceil(3 / 2) - 1}
        small.data.write(path, changed)
        with self.assertRaisesRegex(ValueError, "optimizer budget"):
            self.register(self.root / "never-incomplete")
        small.data.write(path, original)
        dpo_path = self.dpo_training / "dpo_configuration.json"
        dpo = small.read(dpo_path)
        dpo["parent"]["reload_run_sha256"] = "another-parent-reload"
        small.data.write(dpo_path, dpo)
        for parent in (self.dpo_training, self.dpo_reload):
            info = small.read(parent / "run.json")
            info["dpo_configuration_sha256"] = small.sha(dpo_path)
            small.data.write(parent / "run.json", info)
        with self.assertRaisesRegex(ValueError, "exact selected SFT parent"):
            self.register(self.root / "never-wrong-parent")
        self.assertFalse((self.root / "never-incomplete").exists())
        self.assertFalse((self.root / "never-wrong-parent").exists())

    def test_small_preflight_failure_preserves_label_free_request_census(self):
        fixture = sealed_fixtures.SealedTaskTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        sealed = sealed_fixtures.sealed
        inputs = fixture.prepare()
        release = fixture.seal(inputs)
        portable = fixture.root / "small-portable"
        sealed.export_execution(fixture.freeze_path, release, "native_base", portable)
        spec = small.read(portable / "execution.json")
        spec.update(
            runtime_family=small.FAMILY,
            vocab_size=65536,
            physical_context_tokens=8192,
            inference_prefix_limit=7680,
            output_tokens=512,
            inference_policy_sha256="engineering-context-policy",
        )
        small.data.write(portable / "execution.json", spec)
        manifest = small.read(portable / "manifest.json")
        manifest["files_sha256"]["execution.json"] = small.sha(
            portable / "execution.json"
        )
        small.data.write(portable / "manifest.json", manifest)
        runtime = {
            "source_sha256": spec["source_sha256"],
            "runtime_manifest_sha256": None,
            "runtime_family": small.FAMILY,
            "inference_policy_sha256": spec["inference_policy_sha256"],
        }
        with (
            mock.patch.object(native, "execution_runtime", return_value=runtime),
            mock.patch.object(
                small,
                "load_runtime",
                side_effect=ValueError("engineering preflight failure"),
            ) as loader,
        ):
            result = native.run(
                portable,
                small.sha(portable / "manifest.json"),
                fixture.inference_config["path"],
                fixture.root / "failed-small-run",
            )
        loader.assert_called_once()
        self.assertEqual(result["requested"], result["observed"])
        observations = small.data.rows(
            fixture.root / "failed-small-run/observations.jsonl"
        )
        self.assertTrue(
            all(
                row["status"] == "request_failure"
                and row["generation_ms"] is None
                and not row["generation_attempted"]
                for row in observations
            )
        )

    def test_explicit_insufficient_pair_omission_keeps_controls_and_cannot_skip_unreviewed_candidates(
        self,
    ):
        mining, candidates = self.root / "mining", self.root / "candidates"
        mining.mkdir()
        candidates.mkdir()
        parent = small.sft.model_identity(self.sft_training, self.sft_reload)
        small.data.write_rows(
            candidates / "observations.jsonl",
            [{"case_id": "engineering-train", "status": "completed"}],
        )
        candidate = {
            "status": "inference_complete",
            "kind": "sft",
            "splits": ["train"],
            "model_identity": parent,
            "configuration_sha256": parent["configuration_sha256"],
            "requested": 1,
            "observations_sha256": small.sha(candidates / "observations.jsonl"),
            "final_test_bodies_read": False,
        }
        small.data.write(candidates / "run.json", candidate)
        small.data.write(mining / "configuration.json", self.config)
        small.data.write_rows(mining / "pairs.jsonl.gz", [])
        small.data.write_rows(
            mining / "decisions.jsonl",
            [{"case_id": "engineering-train", "status": "tie"}],
        )
        small.data.write_rows(mining / "review-queue.jsonl", [])
        report = {
            "version": "small-actions-final-dpo-v1",
            "status": "train_errors_mined",
            "parent": parent,
            "source_cases_sha256": self.config["source_cases_sha256"],
            "candidate_run_sha256": small.sha(candidates / "run.json"),
            "requested": 1,
            "pairs": 0,
            "preservation_pairs": 0,
            "decisions": {"tie": 1},
            "final_test_bodies_read": False,
            "files_sha256": {
                name: small.sha(mining / name)
                for name in (
                    "pairs.jsonl.gz",
                    "decisions.jsonl",
                    "review-queue.jsonl",
                    "configuration.json",
                )
            },
        }
        small.data.write(mining / "run.json", report)
        plan = self.omission_plan
        output = self.root / "registered-with-omission"
        manifest = small.register(
            self.preparation,
            self.dataset,
            self.sft_training,
            self.sft_reload,
            output=output,
            dpo_mining=mining,
            dpo_candidates=candidates,
            dpo_plan=plan,
        )
        self.assertEqual(set(manifest["models"]), {"native_base", "sft"})
        self.assertFalse((output / "sft_dpo.json").exists())
        freeze = {
            "dpo_omission": manifest["dpo_omission"],
            "selected_model": "sft",
            "models": manifest["models"],
            "source_sha256": {
                "ml/small_final_test_runtime.py": small.sha(small.__file__)
            },
        }
        canonical, roles, omitted = comparison.role_policy(freeze, small.bound)
        self.assertEqual(len(canonical), 4)
        self.assertEqual(
            set(roles),
            {
                "native_base",
                "base_icl",
                "sft",
                "always_no_action",
                "consistent_history_rule",
            },
        )
        self.assertEqual(omitted["counts"]["strict_pairs"], 0)
        with self.assertRaisesRegex(ValueError, "Supply completed DPO"):
            small.register(
                self.preparation,
                self.dataset,
                self.sft_training,
                self.sft_reload,
                output=self.root / "silent-skip",
            )
        fixture = sealed_fixtures.SealedTaskTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.freeze["dpo_omission"] = manifest["dpo_omission"]
        fixture.replace_freeze()
        with self.assertRaisesRegex(
            ValueError, "Only an explicitly documented same-parent"
        ):
            sealed_fixtures.sealed.freeze_inputs(
                fixture.freeze_path,
                sealed_fixtures.ForbiddenBodyPath(),
                fixture.root / "never-read",
            )
        small.data.write_rows(
            mining / "decisions.jsonl",
            [{"case_id": "engineering-train", "status": "needs_review"}],
        )
        small.data.write_rows(
            mining / "review-queue.jsonl", [{"case_id": "engineering-train"}]
        )
        report["decisions"] = {"needs_review": 1}
        report["files_sha256"] = {
            name: small.sha(mining / name) for name in report["files_sha256"]
        }
        small.data.write(mining / "run.json", report)
        with self.assertRaisesRegex(ValueError, "Unresolved semantic candidates"):
            small.register(
                self.preparation,
                self.dataset,
                self.sft_training,
                self.sft_reload,
                output=self.root / "never-unreviewed",
                dpo_mining=mining,
                dpo_candidates=candidates,
                dpo_plan=plan,
            )
        self.assertFalse((self.root / "never-unreviewed").exists())

    def test_small_icl_keeps_every_case_or_fails_whole_preparation(self):
        for prefix_length in (4000, 7681):
            fixture = sealed_fixtures.SealedTaskTests(methodName="runTest")
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            sealed = sealed_fixtures.sealed

            class LongICLTokenizer(sealed_fixtures.FixtureTokenizer):
                def apply_chat_template(self, messages, **kwargs):
                    return (
                        [2] * prefix_length
                        if len(messages) > 2
                        else super().apply_chat_template(messages, **kwargs)
                    )

            tokenizer = LongICLTokenizer()
            gate = sealed.load_gate(fixture.freeze_path)
            gate["policies"]["runtime"]["tokenizer"]["runtime_family"] = small.FAMILY
            output = fixture.root / "full-cohort"
            with mock.patch.object(sealed, "load_gate", return_value=gate):
                if prefix_length <= 7680:
                    sealed.freeze_inputs(
                        fixture.freeze_path,
                        fixture.candidates(),
                        output,
                        tokenizer=tokenizer,
                    )
                    self.assertEqual(sealed.rows(output / "run-exclusions.jsonl"), [])
                    ids = [row["id"] for row in sealed.rows(output / "inputs.jsonl")]
                    self.assertEqual(len(ids), 3)
                    for role in sealed.RUN_ROLES:
                        self.assertEqual(
                            [
                                row["case_id"]
                                for row in sealed.rows(
                                    output / f"requests-{role}.jsonl"
                                )
                            ],
                            ids,
                        )
                else:
                    with self.assertRaisesRegex(
                        ValueError, "fail the whole preparation"
                    ):
                        sealed.freeze_inputs(
                            fixture.freeze_path,
                            fixture.candidates(),
                            output,
                            tokenizer=tokenizer,
                        )
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
