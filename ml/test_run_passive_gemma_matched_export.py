import copy
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_passive_gemma_matched_export as driver
import test_train_passive_gemma_matched as training_tests


FAKE_EXPORTER = r"""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('stage');p.add_argument('--config');p.add_argument('--output')
a=p.parse_args();root=Path(a.output);c=json.loads(Path(a.config).read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f)
def artifacts(keys):
    result={}
    for key in keys:
        path=root/c['fake_names'][key]
        with path.open('xb') as f:f.write(bytes.fromhex(c['fake_payloads'][key]))
        result[key]={'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size}
    return result
if a.stage=='preflight':
    root.mkdir();write(root/'configuration.json',c);write(root/'preflight.json',{'status':'passed'})
elif a.stage=='convert':
    write(root/'conversion.json',{'artifacts':artifacts(('base','adapter','merged'))})
elif a.stage=='verify':
    records=json.loads((root/'conversion.json').read_text())['artifacts']
    assert all(digest(Path(v['path']))==v['sha256'] for v in records.values())
    write(root/'tensor-verification.json',{'status':'passed','artifacts':records})
elif a.stage=='quantize':
    assert (root/'tensor-verification.json').is_file()
    write(root/'quantization.json',{'artifacts':artifacts(('base_quantized','merged_quantized'))})
elif a.stage=='receipt':
    records={}
    for name in ('conversion.json','quantization.json'):
        records.update(json.loads((root/name).read_text())['artifacts'])
    assert all(digest(Path(v['path']))==v['sha256'] for v in records.values())
    write(root/'receipt.json',{
        'status':'tensor_verified_paired_export_runtime_preflight_pending',
        'configuration_sha256':digest(root/'configuration.json'),
        'training_record':c['training_record'],'artifacts':records})
print('tiny fake stage completed',a.stage)
"""


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fake_plan(root):
    (root / "artifacts/training").mkdir(parents=True)
    canonical = root / "artifacts/training/canonical"
    canonical.mkdir()
    payloads = {
        "base": b"a" * 12288,
        "adapter": b"unit-adapter",
        "merged": b"a" * 4096 + b"b" * 4096 + b"a" * 4096,
        "base_quantized": b"q" * 8192,
        "merged_quantized": b"q" * 4096 + b"r" * 4096,
    }
    templates = {}
    for key, path in driver.exporter.paths(canonical).items():
        path.write_bytes(payloads[key].replace(b"b", b"c").replace(b"r", b"s"))
        templates[key] = {**driver.binding(path), "bytes": path.stat().st_size}
    record = root / "run.json"
    put(record, {"unit_test": True})
    config = {
        "version": driver.exporter.VERSION,
        "base": {
            "repository": driver.exporter.BASE_REPOSITORY,
            "revision": driver.exporter.BASE_REVISION,
            "path": str(canonical),
        },
        "adapter": {"path": str(root)},
        "training_record": driver.binding(record),
        "native_pin": driver.binding(record),
        "runtime": {
            "revision": driver.exporter.RUNTIME_REVISION,
            "source_path": str(root),
            "python": sys.executable,
            "tools": {
                key: {"path": str(record)}
                for key in ("llama-export-lora", "llama-quantize")
            },
            "support_files_sha256": {},
        },
        "fake_names": driver.exporter.shared.GGUF_FILES,
        "fake_payloads": {key: value.hex() for key, value in payloads.items()},
    }
    config_path = root / "configuration.json"
    put(config_path, config)
    script = root / "fake_exporter.py"
    script.write_text(FAKE_EXPORTER)
    policy = root / "policy.json"
    put(policy, {"unit_test": "No model computation or actual export."})
    output = root / "artifacts/training/attempt"
    plan = {
        "version": driver.VERSION,
        "status": "plan_only_no_export_executed",
        "output": str(output),
        "configuration": driver.binding(config_path),
        "policy": driver.binding(policy),
        "templates": templates,
        "bindings": {},
        "space_required_before_stages": driver.stage_requirements(templates),
        "pins": {str(p): driver.digest(p) for p in (config_path, script, policy)},
        "commands": {
            stage: [
                sys.executable,
                "-B",
                str(script),
                stage,
                "--config",
                str(config_path),
                "--output",
                str(output / "export"),
            ]
            for stage in driver.STAGES
        },
    }
    return plan, payloads


def final_fixture(root):
    config, _ = training_tests.configuration()
    config["bindings"].update(
        {
            name: {"path": f"{name}.json", "sha256": "d" * 64}
            for name in driver.trainer.T1_BINDINGS
        }
    )
    rows, diagnostics = training_tests.select(config, training_tests.prepared_bundle())
    paths = [root / "artifacts/training" / name for name in ("train", "reload")]
    for path in paths:
        path.mkdir(parents=True)
        for name, key, source in driver.trainer.SOURCE_MODULES:
            (path / name).write_bytes(Path(source).read_bytes())
            config[key] = driver.digest(path / name)
    config_path = root / "artifacts/training/input/configuration.json"
    put(config_path, config)
    exposure = driver.trainer.curriculum.Exposure()
    for row in rows:
        for phase in driver.trainer.curriculum.PHASES:
            exposure.record(phase, row)
    exposure.updates = 201
    run = {
        "version": driver.trainer.VERSION,
        "status": driver.TRAIN_STATUS,
        "mode": "train",
        "phase": "substantive",
        "format": "json",
        "training_view": "json",
        "stage": "matched_format_sft",
        "configuration_sha256": driver.digest(config_path),
        "optimizer_state_resumed": False,
        "initialization": "original_native_base_with_fresh_lora",
        "training_updates": 201,
        "optimizer_committed_rows": 802,
        "consumed_rows": 802,
        "final_test_bodies_read": 0,
        "pid": os.getpid(),
        "started_unix": 1,
        "finished_unix": 2,
        **{key: config[key] for _, key, _ in driver.trainer.SOURCE_MODULES},
        "preparation_sha256": config["bindings"]["preparation"]["sha256"],
        "evaluation_protocol_sha256": config["bindings"]["evaluation_protocol"][
            "sha256"
        ],
        "native_pin_sha256": config["bindings"]["native_pin"]["sha256"],
        "t1_evidence_sha256": {
            key: config["bindings"][key]["sha256"] for key in driver.trainer.T1_BINDINGS
        },
    }
    runtime = {
        "targets": {
            key: {"class": "Linear"} for key in config["model"]["target_modules"]
        },
        "tokenizer_parity_rows": 802,
        "numerical_regime": "native_bfloat16_lora_float32",
        "parameter_dtypes": ["torch.bfloat16"],
        "ignored_native_tensor_names": [],
        "loading_info": {"missing_keys": []},
        "effective_generation": config["generation"],
        "completion_label_eos_token_id": 1,
    }
    generations = [
        {"case_id": row["case_id"], "output_ids": [1]} for row in diagnostics
    ]
    for path in paths:
        (path / "configuration.json").write_bytes(config_path.read_bytes())
        put(path / "runtime.json", runtime)
        put(path / "smoke-generation.json", generations)
    put(paths[0] / "run.json", run)
    put(paths[0] / "exposure.json", exposure.summary())
    put(
        paths[0] / "initialization.json",
        {
            "initialization": "original_native_base_with_fresh_lora",
            "format": "json",
            "smoke_adapter_loaded": False,
            "t1_adapter_loaded": False,
            "other_format_adapter_loaded": False,
            "trainable_tensors": {str(i): 1 for i in range(410)},
            "trainable_parameters": 18585600,
        },
    )
    for name in (
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter-tensors.json",
        "training-state.pt",
        "training-rows.jsonl",
        "training-events.jsonl",
    ):
        path = paths[0] / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"tiny synthetic placeholder, never loaded as a tensor")
    driver.trainer.manifest(paths[0], driver.TRAIN_STATUS)
    put(
        paths[1] / "run.json",
        {
            **run,
            "mode": "fresh_reload",
            "status": driver.RELOAD_STATUS,
            "initialization": "original_native_base_with_saved_lora",
            "pid": os.getpid() + 1,
            "started_unix": 3,
            "finished_unix": 4,
            "reloaded_training_updates": 201,
            "previous_manifest_sha256": driver.digest(paths[0] / "manifest.json"),
        },
    )
    put(
        paths[1] / "reload-parity.json",
        {
            "adapter_tensors_identical": True,
            "optimizer_state_resumed": False,
            "cases": [
                {"case_id": row["case_id"], "identical": True} for row in diagnostics
            ],
        },
    )
    driver.trainer.manifest(paths[1], driver.RELOAD_STATUS)
    return config, config_path, paths, rows, diagnostics


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "artifacts").mkdir()
        self.patch_root = patch.object(driver, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        limits = copy.deepcopy(driver.LIMITS)
        limits.update(minimum_free_bytes=0, stage_overhead_bytes=0, chunk_bytes=4096)
        limits["max_changed_bytes"].update(merged=4096, merged_quantized=4096)
        self.patch_limits = patch.object(driver, "LIMITS", limits)
        self.patch_limits.start()
        self.addCleanup(self.patch_limits.stop)

    def test_five_fake_stages_compact_exact_bytes_base_first_without_changing_inputs(
        self,
    ):
        plan, payloads = fake_plan(self.root)
        originals = {
            Path(value["path"]): driver.digest(Path(value["path"]))
            for value in plan["templates"].values()
        }
        originals.update({Path(path): sha for path, sha in plan["pins"].items()})
        result = driver.execute(plan)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual([r["stage"] for r in result["stages"]], list(driver.STAGES))
        self.assertEqual(
            [r["artifact"] for r in result["compactions"]],
            ["base", "merged", "base_quantized", "merged_quantized"],
        )
        output = Path(plan["output"])
        for key, path in driver.exporter.paths(output / "export").items():
            self.assertEqual(path.read_bytes(), payloads[key])
        for path, sha in originals.items():
            self.assertEqual(driver.digest(path), sha)
        self.assertFalse(list((output / "compaction").glob("*.scratch.gguf")))
        self.assertEqual(driver.read(output / "result.json")["exit_code"], 0)
        receipt_bytes = (output / "export/receipt.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "previous attempt"):
            driver.execute(plan)
        self.assertEqual((output / "export/receipt.json").read_bytes(), receipt_bytes)
        with self.assertRaisesRegex(ValueError, "after its final receipt"):
            driver.compact(
                result["owner"],
                "base",
                plan["templates"]["base"],
                driver.read(output / "export/receipt.json")["artifacts"]["base"],
            )

    def test_publication_failure_preserves_owned_target_and_canonical_inputs(self):
        plan, payloads = fake_plan(self.root)
        original_replace = os.replace

        def fail_merged(source, target):
            if Path(target).name == "merged-F16.gguf":
                raise OSError("synthetic publication failure")
            return original_replace(source, target)

        with patch.object(driver.os, "replace", side_effect=fail_merged):
            result = driver.execute(plan)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("publication failure", result["error"])
        output = Path(plan["output"])
        self.assertEqual(
            (output / "export/merged-F16.gguf").read_bytes(), payloads["merged"]
        )
        self.assertTrue((output / "compaction/merged.scratch.gguf").exists())
        self.assertFalse((output / "export/tensor-verification.json").exists())
        for value in plan["templates"].values():
            self.assertEqual(driver.digest(Path(value["path"])), value["sha256"])

    def test_changed_hash_or_private_write_budget_stops_before_verify(self):
        for fault in ("template_hash", "budget", "native_base_hash"):
            with (
                self.subTest(fault=fault),
                tempfile.TemporaryDirectory(dir=self.root) as tmp,
            ):
                plan, payloads = fake_plan(Path(tmp))
                if fault == "template_hash":
                    plan["templates"]["merged"]["sha256"] = "0" * 64
                elif fault == "native_base_hash":
                    plan["templates"]["base"]["sha256"] = "0" * 64
                with patch.dict(
                    driver.LIMITS["max_changed_bytes"],
                    {"merged": 0 if fault == "budget" else 4096},
                ):
                    result = driver.execute(plan)
                self.assertEqual(result["exit_code"], 1)
                output = Path(plan["output"])
                self.assertEqual(
                    (output / "export/merged-F16.gguf").read_bytes(), payloads["merged"]
                )
                self.assertFalse((output / "export/tensor-verification.json").exists())
                self.assertFalse((output / "compaction/merged.scratch.gguf").exists())

    def test_ownership_loss_records_failure_and_stops_before_convert(self):
        plan, _ = fake_plan(self.root)
        original = driver.run_stage

        def lose_ownership(*args):
            result = original(*args)
            path = Path(plan["output"]) / "ownership.json"
            put(path, {**driver.read(path), "token": "not-our-token"})
            return result

        with patch.object(driver, "run_stage", side_effect=lose_ownership):
            result = driver.execute(plan)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("ownership", result["error"])
        output = Path(plan["output"])
        self.assertEqual(driver.read(output / "result.json")["status"], "failed")
        self.assertFalse((output / "export/base-F16.gguf").exists())

    def test_changed_sources_busy_lock_symlinks_and_stage_space_fail_closed(self):
        plan, _ = fake_plan(self.root)
        script = self.root / "fake_exporter.py"
        original = script.read_bytes()
        script.write_bytes(original + b"# changed\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            driver.execute(plan)
        self.assertFalse(Path(plan["output"]).exists())
        script.write_bytes(original)
        with (self.root / "artifacts/nep-passive-b1.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                driver.execute(plan)
        self.assertFalse(Path(plan["output"]).exists())
        alias = self.root / "linked"
        alias.symlink_to(self.root / "canonical", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            driver.safe_path(alias / "base-F16.gguf")
        with self.assertRaisesRegex(ValueError, "traversal"):
            driver.safe_path(self.root / "canonical/../run.json")
        with patch.object(
            driver.shutil, "disk_usage", return_value=SimpleNamespace(free=1)
        ):
            result = driver.execute(plan)
        self.assertIn("space before convert", result["error"])
        self.assertEqual([row["stage"] for row in result["stages"]], ["preflight"])

    def test_final_training_requires_exact_exposure_independent_reload_and_no_t1_adapter(
        self,
    ):
        config, config_path, paths, rows, diagnostics = final_fixture(self.root)

        def check():
            return driver.validate_final(
                config,
                config_path,
                paths[0] / "manifest.json",
                paths[1] / "manifest.json",
                rows,
                diagnostics,
                {},
            )

        check()  # Historical remote PID deliberately equals this local PID.
        originals = {
            p: p.read_bytes()
            for directory in paths
            for p in directory.rglob("*")
            if p.is_file()
        }
        faults = (
            (0, "run.json", "training_updates", 2),
            (1, "run.json", "pid", os.getpid()),
            (1, "run.json", "format", "rewrite"),
            (0, "initialization.json", "t1_adapter_loaded", True),
            (1, "runtime.json", "tokenizer_parity_rows", 8),
            (1, "smoke-generation.json", None, []),
            (0, "exposure.json", "pending_backward_rows_not_committed", 2),
        )
        for index, name, key, value in faults:
            with self.subTest(name=name, key=key):
                path = paths[index] / name
                data = driver.read(path)
                if key is None:
                    data = value
                else:
                    data[key] = value
                put(path, data)
                driver.trainer.manifest(paths[0], driver.TRAIN_STATUS)
                reload_run = driver.read(paths[1] / "run.json")
                reload_run["previous_manifest_sha256"] = driver.digest(
                    paths[0] / "manifest.json"
                )
                put(paths[1] / "run.json", reload_run)
                driver.trainer.manifest(paths[1], driver.RELOAD_STATUS)
                with self.assertRaises(ValueError):
                    check()
                for p, payload in originals.items():
                    p.write_bytes(payload)
        (paths[0] / "unbound").write_text("not in manifest")
        with self.assertRaisesRegex(ValueError, "inventory"):
            check()

    def test_default_is_plan_only_and_policy_cannot_expand_caps(self):
        args = [
            "driver",
            "--config",
            "config.json",
            "--policy",
            "policy.json",
            "--policy-sha256",
            "0" * 64,
            "--output",
            "new",
        ]
        with (
            patch.object(sys, "argv", args),
            patch.object(driver, "load_plan", return_value={"planned": True}),
            patch.object(driver, "execute") as execute,
            patch("builtins.print"),
        ):
            self.assertEqual(driver.main(), 0)
            execute.assert_not_called()
        path = self.root / "policy.json"
        limits = copy.deepcopy(driver.LIMITS)
        limits["max_changed_bytes"]["merged"] += 1
        put(path, {"version": driver.VERSION, "limits": limits})
        with self.assertRaisesRegex(ValueError, "budgets"):
            driver.load_plan(
                self.root / "config.json",
                path,
                driver.digest(path),
                self.root / "output",
            )

    def test_metadata_plan_binds_exact_final_artifacts_and_creates_no_output(self):
        tiny, _ = fake_plan(self.root)
        config, config_path, paths, _, _ = final_fixture(self.root)
        canonical = Path(tiny["templates"]["base"]["path"]).parent
        export_path = self.root / "export-config.json"
        export = {
            "version": driver.exporter.VERSION,
            "base": {
                "path": str(canonical),
                "files_sha256": {},
                "repository": driver.exporter.BASE_REPOSITORY,
                "revision": driver.exporter.BASE_REVISION,
            },
            "adapter": {"path": str(paths[0] / "adapter"), "files_sha256": {}},
            "runtime": {
                "revision": driver.exporter.RUNTIME_REVISION,
                "source_path": str(self.root / "converter"),
                "python": sys.executable,
                "tools": {
                    key: {"path": str(self.root / "run.json")}
                    for key in ("llama-export-lora", "llama-quantize")
                },
                "support_files_sha256": {},
            },
        }
        config["base"]["files_sha256"] = {}
        for name in config["bindings"]:
            path = config_path.parent / f"binding-{name}.json"
            put(path, {"synthetic": name})
            config["bindings"][name] = driver.binding(path)
        prep = {"release_manifest_sha256": "a" * 64}
        put(Path(config["bindings"]["preparation"]["path"]), prep)
        config["bindings"]["preparation"] = driver.binding(
            Path(config["bindings"]["preparation"]["path"])
        )
        export["native_pin"] = config["bindings"]["native_pin"]
        canonical_config = canonical / "configuration.json"
        put(canonical_config, export)
        canonical_path = canonical / "receipt.json"
        put(
            canonical_path,
            {
                "version": driver.exporter.VERSION,
                "status": driver.EXPORT_STATUS,
                "configuration_sha256": driver.digest(canonical_config),
                "base_repository": driver.exporter.BASE_REPOSITORY,
                "base_revision": driver.exporter.BASE_REVISION,
                "runtime_revision": driver.exporter.RUNTIME_REVISION,
                "native_generation_stop_ids": [1, 106, 50],
                "artifacts": tiny["templates"],
            },
        )
        portable_copy = self.root / "portable-t1-export-receipt.json"
        portable_copy.write_bytes(canonical_path.read_bytes())
        config["bindings"]["t1_export_receipt"] = driver.binding(portable_copy)
        protocol_copy = self.root / "registered-protocol.json"
        protocol_copy.write_bytes(
            Path(config["bindings"]["evaluation_protocol"]["path"]).read_bytes()
        )
        put(config_path, config)
        for path in paths:
            (path / "configuration.json").write_bytes(config_path.read_bytes())
            run = driver.read(path / "run.json")
            run.update(
                configuration_sha256=driver.digest(config_path),
                preparation_sha256=config["bindings"]["preparation"]["sha256"],
                native_pin_sha256=config["bindings"]["native_pin"]["sha256"],
                evaluation_protocol_sha256=config["bindings"]["evaluation_protocol"][
                    "sha256"
                ],
                t1_evidence_sha256={
                    name: config["bindings"][name]["sha256"]
                    for name in driver.trainer.T1_BINDINGS
                },
            )
            if path == paths[1]:
                run["previous_manifest_sha256"] = driver.digest(
                    paths[0] / "manifest.json"
                )
            put(path / "run.json", run)
            driver.trainer.manifest(path, run["status"])
        export["training_record"] = driver.binding(paths[0] / "run.json")
        export["adapter"]["files_sha256"] = {
            name.removeprefix("adapter/"): sha
            for name, sha in driver.read(paths[0] / "manifest.json")[
                "files_sha256"
            ].items()
            if name.startswith("adapter/")
        }
        put(export_path, export)
        modules = tuple(
            (name, key, str(paths[0] / name))
            for name, key, _ in driver.trainer.SOURCE_MODULES
        )
        source_paths = [Path(value) for _, _, value in modules]
        policy = {
            "version": driver.VERSION,
            "limits": driver.LIMITS,
            "export_configuration": driver.binding(export_path),
            "training_configuration": driver.binding(config_path),
            "training_manifest": driver.binding(paths[0] / "manifest.json"),
            "fresh_reload": driver.binding(paths[1] / "manifest.json"),
            "protocol": driver.binding(protocol_copy),
            "canonical_export_receipt": driver.binding(canonical_path),
            "sources_sha256": {
                str(path.relative_to(self.root)): driver.digest(path)
                for path in source_paths
            },
        }
        policy_path = self.root / "storage-policy.json"
        put(policy_path, policy)
        bundle = training_tests.prepared_bundle()
        bundle.update(manifest=prep, verification={"dev_test_record_bodies_read": 0})
        with ExitStack() as stack:
            stack.enter_context(patch.object(driver, "FROZEN", {}))
            stack.enter_context(
                patch.object(driver, "source_paths", return_value=source_paths)
            )
            stack.enter_context(patch.object(driver.trainer, "SOURCE_MODULES", modules))
            stack.enter_context(
                patch.object(
                    driver.trainer.preparation, "verify_bundle", return_value=bundle
                )
            )
            for name in (
                "validate_configuration",
                "validate_protocol",
                "validate_t1_evidence",
                "validate_smoke_gate",
            ):
                stack.enter_context(patch.object(driver.trainer, name))
            plan = driver.load_plan(
                export_path,
                policy_path,
                driver.digest(policy_path),
                Path(tiny["output"]),
            )
            self.assertEqual(plan["training_updates"], 201)
            self.assertEqual(plan["committed_draws"], 802)
            self.assertFalse(Path(plan["output"]).exists())
            self.assertEqual(plan["bindings"]["fresh_reload"], policy["fresh_reload"])
            for directory in (*paths, canonical, config_path.parent):
                with (
                    self.subTest(protected_directory=str(directory)),
                    self.assertRaisesRegex(
                        ValueError, "outside every immutable input directory"
                    ),
                ):
                    driver.load_plan(
                        export_path,
                        policy_path,
                        driver.digest(policy_path),
                        directory / "new-owned-export",
                    )
                self.assertFalse((directory / "new-owned-export").exists())
            changed_receipt = self.root / "wrong-canonical-receipt.json"
            put(changed_receipt, {**driver.read(canonical_path), "status": "different"})
            policy["canonical_export_receipt"] = driver.binding(changed_receipt)
            put(policy_path, policy)
            with self.assertRaisesRegex(
                ValueError, "registered experiment/T1 templates"
            ):
                driver.load_plan(
                    export_path,
                    policy_path,
                    driver.digest(policy_path),
                    Path(tiny["output"]),
                )
            policy["canonical_export_receipt"] = driver.binding(canonical_path)
            export["adapter"]["path"] = str(paths[1])
            put(export_path, export)
            policy["export_configuration"] = driver.binding(export_path)
            put(policy_path, policy)
            with self.assertRaisesRegex(ValueError, "exact final adapter"):
                driver.load_plan(
                    export_path,
                    policy_path,
                    driver.digest(policy_path),
                    Path(tiny["output"]),
                )


if __name__ == "__main__":
    unittest.main()
