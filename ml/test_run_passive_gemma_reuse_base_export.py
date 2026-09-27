import copy
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_passive_gemma_reuse_base_export as reuse


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    canonical = root / "canonical"
    canonical.mkdir()
    for name in ("native", "adapter", "source"):
        (root / name).mkdir()
    record = root / "inputs/training-record.json"
    put(record, {"test_fixture": True})
    config = {
        "version": reuse.exporter.VERSION,
        "base": {
            "repository": reuse.exporter.BASE_REPOSITORY,
            "revision": reuse.exporter.BASE_REVISION,
            "path": str(root / "native"),
            "files_sha256": {},
        },
        "adapter": {"path": str(root / "adapter"), "files_sha256": {}},
        "native_pin": reuse.binding(record),
        "training_record": reuse.binding(record),
        "runtime": {
            "revision": reuse.exporter.RUNTIME_REVISION,
            "source_path": str(root / "source"),
            "source_files_sha256": {},
            "python": sys.executable,
            "tools": {
                name: reuse.binding(record)
                for name in ("llama-export-lora", "llama-quantize")
            },
            "support_files_sha256": {},
        },
    }
    config_path = root / "config.json"
    put(config_path, config)
    config = reuse.exporter.configuration(config_path)
    put(canonical / "configuration.json", config)
    payloads = {
        "base": b"base" * 3072,
        "adapter": b"tiny adapter",
        "merged": b"m" * 12288,
        "base_quantized": b"q" * 8192,
        "merged_quantized": b"z" * 8192,
    }
    paths = reuse.exporter.paths(canonical)
    for key, path in paths.items():
        path.write_bytes(payloads[key])
    artifacts = {key: reuse.exporter.artifact(path) for key, path in paths.items()}
    argv = {
        "convert-base": [
            str(v) for v in reuse.exporter.conversion_commands(config, canonical)[0][1]
        ],
        "quantize-base": [
            config["runtime"]["tools"]["llama-quantize"]["path"],
            str(paths["base"]),
            str(paths["base_quantized"]),
            "Q4_K_M",
            "4",
        ],
    }
    for name, command in argv.items():
        put(canonical / "logs" / f"{name}.command.json", {"argv": command})
        log = canonical / "logs" / f"{name}.log"
        log.write_text("fake successful canonical command; no model operations\n")
        put(
            canonical / "logs" / f"{name}.result.json",
            {"returncode": 0, "log_sha256": reuse.digest(log)},
        )
    old_commands = {
        suffix: reuse.exporter.artifact(canonical / "logs" / f"quantize-base.{suffix}")
        for suffix in ("command.json", "result.json", "log")
    }
    put(
        canonical / "tensor-verification.json",
        {
            "status": "passed",
            "adapter_tensors_exact": 410,
            "native_targets_exact": 205,
            "merged_targets": list(range(205)),
            "unadapted_tensors_exact": list(range(453)),
            "artifacts": {k: artifacts[k] for k in ("base", "adapter", "merged")},
        },
    )
    put(
        canonical / "quantization.json",
        {
            "quantization": "Q4_K_M",
            "importance_matrix": None,
            "threads": 4,
            "commands": {"base": old_commands},
            "quantizer_fallbacks": {"base": ["canonical fallback"]},
            "artifacts": {
                k: artifacts[k] for k in ("base_quantized", "merged_quantized")
            },
        },
    )
    put(
        canonical / "receipt.json",
        {
            "version": reuse.exporter.VERSION,
            "status": reuse.previous.EXPORT_STATUS,
            "base_repository": reuse.exporter.BASE_REPOSITORY,
            "base_revision": reuse.exporter.BASE_REVISION,
            "runtime_revision": reuse.exporter.RUNTIME_REVISION,
            "native_generation_stop_ids": [1, 106, 50],
            "configuration_sha256": reuse.digest(canonical / "configuration.json"),
            "tensor_verification_sha256": reuse.digest(
                canonical / "tensor-verification.json"
            ),
            "quantization_sha256": reuse.digest(canonical / "quantization.json"),
            "artifacts": artifacts,
        },
    )
    put(
        root / "independent.json",
        {
            "status": "passed",
            "checks": {"fixture_only": True},
            "bindings": {"export_receipt": reuse.binding(canonical / "receipt.json")},
        },
    )
    policy = {
        "version": reuse.VERSION,
        "method": reuse.METHOD,
        "mode": "engineering_reexport",
        "registration_status": "registered_before_execution",
        "limits": reuse.LIMITS,
        "sources_sha256": {
            str(p.relative_to(reuse.ROOT)): reuse.digest(p)
            for p in reuse.source_paths()
        },
        "export_configuration": reuse.binding(config_path),
        "canonical_export_receipt": reuse.binding(canonical / "receipt.json"),
        "canonical_independent_verification": reuse.binding(root / "independent.json"),
        "canonical_base_conversion_logs": {
            suffix: reuse.binding(canonical / "logs" / f"convert-base.{suffix}")
            for suffix in ("command.json", "result.json", "log")
        },
        "canonical_base_quantization_logs": {
            suffix: reuse.binding(canonical / "logs" / f"quantize-base.{suffix}")
            for suffix in ("command.json", "result.json", "log")
        },
    }
    policy_path = root / "policy.json"
    put(policy_path, policy)
    return config_path, policy_path, payloads


class ReuseBaseTests(unittest.TestCase):
    def setUp(self):
        (reuse.ROOT / "artifacts/training").mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(
            prefix="unit-reuse-", dir=reuse.ROOT / "artifacts/training"
        )
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "artifacts").mkdir()
        self.config, self.policy, self.payloads = fixture(self.root)
        self.plan = reuse.load_plan(
            self.config, self.policy, reuse.digest(self.policy), self.root / "attempt"
        )

    def fake_stage(self, plan, owner, stage, *, bad_merged=False):
        root, output = Path(plan["output"]), Path(plan["output"]) / "export"
        config = reuse.exporter.configuration(self.config)
        calls = []

        def command(_, name, argv):
            calls.append(name)
            self.assertNotIn(name, ("convert-base", "quantize-base"))
            key = {
                "convert-adapter": "adapter",
                "merge": "merged",
                "quantize-merged": "merged_quantized",
            }[name]
            payload = self.payloads[key]
            if bad_merged and key == "merged":
                payload = b"x" + payload[1:]
            reuse.exporter.paths(output)[key].write_bytes(payload)
            put(
                output / "logs" / f"{name}.command.json",
                {"argv": [str(v) for v in argv]},
            )
            log = output / "logs" / f"{name}.log"
            log.write_text("fake current command\n")
            put(
                output / "logs" / f"{name}.result.json",
                {"returncode": 0, "log_sha256": reuse.digest(log)},
            )

        with (
            patch.object(reuse.exporter, "bound_run", return_value=(config, {})),
            patch.object(reuse.exporter, "run_command", side_effect=command),
        ):
            if stage == "preflight":
                output.mkdir()
                put(output / "configuration.json", config)
                put(output / "preflight.json", {"status": "passed"})
            elif stage == "convert":
                reuse.convert_reusing_base(plan, owner)
                self.assertEqual(calls, ["convert-adapter", "merge"])
            elif stage == "verify":
                put(
                    output / "tensor-verification.json",
                    {
                        "status": "passed",
                        "artifacts": reuse.read(output / "conversion.json")[
                            "artifacts"
                        ],
                    },
                )
            elif stage == "quantize":
                reuse.quantize_reusing_base(plan, owner)
                self.assertEqual(calls, ["quantize-merged"])
            elif stage == "receipt":
                artifacts = {
                    **reuse.read(output / "conversion.json")["artifacts"],
                    **reuse.read(output / "quantization.json")["artifacts"],
                }
                put(
                    output / "receipt.json",
                    {
                        "status": reuse.previous.EXPORT_STATUS,
                        "configuration_sha256": reuse.digest(
                            output / "configuration.json"
                        ),
                        "training_record": config["training_record"],
                        "artifacts": artifacts,
                    },
                )
        log = root / "stages" / f"{stage}.log"
        log.write_text("fake unit-test stage; no model operations\n")
        put(
            root / "stages" / f"{stage}.command.json",
            {"stage": stage, "argv": plan["commands"][stage]},
        )
        result = {"stage": stage, "returncode": 0, "log_sha256": reuse.digest(log)}
        put(root / "stages" / f"{stage}.result.json", result)
        return result

    def execute(self, *, bad_merged=False, free=200 * 1024**3):
        with ExitStack() as stack:
            stack.enter_context(patch.object(reuse, "ROOT", self.root))
            stack.enter_context(
                patch.object(reuse.clone, "clone_file", side_effect=shutil.copyfile)
            )
            stack.enter_context(
                patch.object(
                    reuse.shutil, "disk_usage", return_value=SimpleNamespace(free=free)
                )
            )
            stack.enter_context(
                patch.object(
                    reuse.previous,
                    "run_stage",
                    side_effect=lambda p, o, s: self.fake_stage(
                        p, o, s, bad_merged=bad_merged
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    reuse,
                    "run_worker_stage",
                    side_effect=lambda p, o, s, *_: self.fake_stage(
                        p, o, s, bad_merged=bad_merged
                    ),
                )
            )
            return reuse.execute(self.plan)

    def test_gates_include_simultaneous_full_target_scratch_and_reserve(self):
        templates = {
            "adapter": {"bytes": 74370944},
            "merged": {"bytes": 50505137152},
            "merged_quantized": {"bytes": 16796017664},
        }
        self.assertEqual(
            reuse.stage_requirements(templates),
            {"convert": 95676664704, "quantize": 55450723328},
        )
        self.assertEqual(reuse.LIMITS, reuse.previous.LIMITS)

    def test_exact_reexport_reuses_real_historical_logs_and_preserves_inputs(self):
        originals = {
            p: p.read_bytes()
            for p in (self.root / "canonical").rglob("*")
            if p.is_file()
        }
        result = self.execute()
        self.assertEqual(result["status"], "engineering_reexport_exact_parity_passed")
        self.assertTrue(result["all_five_artifacts_match_canonical"])
        output = self.root / "attempt/export"
        quant = reuse.read(output / "quantization.json")
        self.assertEqual(
            quant["commands"]["base"],
            self.plan["canonical"]["executions"]["quantize-base"],
        )
        self.assertEqual(
            quant["execution_provenance"]["base"]["mode"],
            "reused_canonical_output_not_executed",
        )
        self.assertFalse((output / "logs/convert-base.command.json").exists())
        self.assertFalse((output / "logs/quantize-base.command.json").exists())
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)
        reuse.validate_engineering_result(
            self.root / "attempt/result.json", self.plan["canonical"], {}
        )

    def test_known_target_mismatch_preserves_failed_attempt_evidence(self):
        result = self.execute(bad_merged=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Known-adapter", result["error"])
        self.assertTrue((self.root / "attempt/export/receipt.json").is_file())
        self.assertEqual(reuse.read(self.root / "attempt/result.json"), result)
        self.assertEqual(
            (self.root / "canonical/merged-F16.gguf").read_bytes(),
            self.payloads["merged"],
        )

    def test_low_space_stops_before_any_conversion_and_preserves_result(self):
        result = self.execute(
            free=self.plan["space_required_before_stages"]["convert"] - 1
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual([r["stage"] for r in result["stages"]], ["preflight"])
        self.assertFalse((self.root / "attempt/export/base-F16.gguf").exists())
        self.assertFalse((self.root / "attempt/export/merged-F16.gguf").exists())

    def test_busy_shared_inference_export_lock_creates_no_attempt(self):
        with (self.root / "artifacts/nep-passive-b1.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute()
        self.assertFalse((self.root / "attempt").exists())

    def test_worker_requires_live_parent_stage_capability_and_inherited_lock(self):
        program = f"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, {str(Path(reuse.__file__).parent)!r})
import run_passive_gemma_reuse_base_export as module
module.ROOT = Path(sys.argv[1])
owner = module.read(Path(sys.argv[2]))
mode = sys.argv[3]
if mode == 'no_capability':
    os.environ.pop(module.WORKER_AUTH_ENV, None)
elif mode == 'wrong_parent':
    owner['pid'] += 1
elif mode == 'wrong_descriptor':
    value = json.loads(os.environ[module.WORKER_AUTH_ENV])
    value['lock_fd'] = 2
    os.environ[module.WORKER_AUTH_ENV] = json.dumps(value)
module.authorize_worker(owner, 'quantize' if mode == 'wrong_stage' else 'convert')
assert module.WORKER_AUTH_ENV not in os.environ
"""
        lock_path = self.root / "artifacts/nep-passive-b1.lock"
        with lock_path.open("a") as lock, patch.object(reuse, "ROOT", self.root):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for mode in (
                "valid",
                "no_capability",
                "wrong_parent",
                "wrong_stage",
                "wrong_descriptor",
            ):
                with self.subTest(mode=mode):
                    root = self.root / f"worker-{mode}"
                    (root / "stages").mkdir(parents=True)
                    secret = "unit-test-ephemeral-capability"
                    owner = {
                        "root": str(root),
                        "root_identity": list(reuse.clone.identity(root)[:2]),
                        "pid": os.getpid(),
                        "worker_authorization_sha256": hashlib.sha256(
                            secret.encode()
                        ).hexdigest(),
                    }
                    put(root / "ownership.json", owner)
                    plan = {
                        "commands": {
                            "convert": [
                                sys.executable,
                                "-c",
                                program,
                                str(self.root),
                                str(root / "ownership.json"),
                                mode,
                            ]
                        }
                    }
                    if mode == "valid":
                        result = reuse.run_worker_stage(
                            plan, owner, "convert", lock, secret
                        )
                        self.assertEqual(result["returncode"], 0)
                    else:
                        with self.assertRaisesRegex(ValueError, "worker failed"):
                            reuse.run_worker_stage(plan, owner, "convert", lock, secret)
                        self.assertNotEqual(
                            reuse.read(root / "stages/convert.result.json")[
                                "returncode"
                            ],
                            0,
                        )
                    with lock_path.open("a") as other:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_worker_cannot_replay_a_failed_space_gate(self):
        self.execute(free=self.plan["space_required_before_stages"]["convert"] - 1)
        with (
            patch.object(reuse, "authorize_worker"),
            patch.object(reuse, "convert_reusing_base") as convert,
            patch.object(
                reuse.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12)
            ),
            self.assertRaisesRegex(ValueError, "passed stage space gate"),
        ):
            reuse.worker(self.root / "attempt/plan.json", "convert")
        convert.assert_not_called()

    def test_canonical_native_and_tool_identity_and_log_bindings_are_mandatory(self):
        canonical = Path(self.plan["canonical"]["receipt"]["path"])
        config = reuse.exporter.configuration(self.config)
        pins = {Path(k): v for k, v in self.plan["pins"].items()}
        for section in ("base", "runtime"):
            with self.subTest(section=section):
                altered = copy.deepcopy(config)
                altered[section]["unexpected_change"] = True
                with self.assertRaisesRegex(ValueError, "converter sources"):
                    reuse.canonical_evidence(canonical, altered, dict(pins))
        missing = dict(pins)
        missing.pop(self.root / "canonical/logs/convert-base.log")
        with self.assertRaisesRegex(ValueError, "bind original"):
            reuse.canonical_evidence(canonical, config, missing)

    def test_wrong_base_bytes_cannot_be_published(self):
        path = self.root / "canonical/base-F16.gguf"
        path.write_bytes(b"wrong".ljust(len(self.payloads["base"]), b"!"))
        changed = path.read_bytes()
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Input artifact hash", result["error"])
        self.assertEqual(path.read_bytes(), changed)
        self.assertFalse((self.root / "attempt/export/conversion.json").exists())

    def test_immutable_policy_output_and_configuration_guards(self):
        self.root.joinpath("attempt").mkdir()
        with self.assertRaises(ValueError):
            reuse.load_plan(
                self.config,
                self.policy,
                reuse.digest(self.policy),
                self.root / "attempt",
            )
        with self.assertRaisesRegex(ValueError, "Wrong export configuration"):
            reuse.load_plan(
                self.root / "different.json",
                self.policy,
                reuse.digest(self.policy),
                self.root / "new",
            )
        with self.assertRaises(ValueError):
            reuse.load_plan(
                self.config,
                self.policy,
                reuse.digest(self.policy),
                self.root / "canonical/new",
            )
        draft = copy.deepcopy(self.plan)
        draft["registration_status"] = "draft_pending_review"
        with self.assertRaisesRegex(ValueError, "Draft policy"):
            reuse.execute(draft)

    def test_matched_mode_cannot_use_generic_passed_engineering_receipt(self):
        path = self.root / "generic.json"
        put(
            path,
            {
                "version": reuse.VERSION,
                "method": reuse.METHOD,
                "mode": "engineering_reexport",
                "status": "passed",
            },
        )
        with self.assertRaises((ValueError, KeyError)):
            reuse.validate_engineering_result(path, self.plan["canonical"], {})


if __name__ == "__main__":
    unittest.main()
