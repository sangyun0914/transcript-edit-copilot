"""Plan or run one fresh Gemma T2 export with bounded, byte-identical compaction.

Policy bindings (paths relative to the policy): export_configuration,
training_configuration (the original input), training_manifest, fresh_reload,
protocol, canonical_export_receipt. sources_sha256 binds source_paths() relative
to ROOT; limits must equal LIMITS. No future evidence or model outputs are made
by plan mode. Only --execute creates a new owned attempt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid

import clone_passive_export as clone
import export_passive_gemma as exporter
import train_passive_gemma_matched as trainer

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-gemma-matched-export-storage-v1"
STAGES = ("preflight", "convert", "verify", "quantize", "receipt")
GIB = 1024**3
LIMITS = {
    "minimum_free_bytes": 30 * GIB,
    "max_changed_bytes": {
        "base": 0,
        "merged": 10 * GIB,
        "base_quantized": 0,
        "merged_quantized": 4 * GIB,
    },
    "stage_overhead_bytes": 2 * GIB,
    "chunk_bytes": 1024**2,
    "stage_timeout_seconds": 21600,
}
FROZEN = {
    "ml/export_passive_gemma.py": "0b9eb72b69875a430cb8c98adceab1e1506a7922787ed2e09d8a34e7cf9abff2",
    "ml/export_passive_sft.py": "a5843d9991103daaa81b25d8b894acb03f644af02523428cb1147841e3c33dd5",
    "ml/clone_passive_export.py": "e0d7ed95fd93ca7b2d55d8cd8727eea59e10c913ed2df20c98119ea63d708df3",
}
TRAIN_STATUS = "training_complete_fresh_reload_pending"
RELOAD_STATUS = "fresh_reload_passed_local_export_pending"
EXPORT_STATUS = "tensor_verified_paired_export_runtime_preflight_pending"
require = exporter.require
digest = clone.digest


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_path(value, parent=None):
    path = Path(value)
    if not path.is_absolute():
        path = (parent or ROOT) / path
    require(".." not in path.parts, "Parent traversal is not supported.")
    path = Path(os.path.abspath(path))
    require(path.is_relative_to(ROOT), "Path escapes this project.")
    for part in (path, *path.parents):
        require(not part.is_symlink(), f"Symlink path is unsupported: {part}")
        if part == ROOT:
            break
    return path


def binding(path):
    return {"path": str(path), "sha256": digest(path)}


def checked(record, parent, pins):
    path = safe_path(record["path"], parent)
    expected = record["sha256"]
    require(re.fullmatch(r"[0-9a-f]{64}", expected), "Bind a complete SHA256.")
    require(path.is_file() and digest(path) == expected, f"Changed input: {path}")
    require(path not in pins or pins[path] == expected, "Conflicting file bindings.")
    pins[path] = expected
    return path


def source_paths():
    return sorted(
        {
            Path(__file__).resolve(),
            Path(clone.__file__).resolve(),
            Path(exporter.__file__).resolve(),
            Path(exporter.shared.__file__).resolve(),
            Path(exporter.export_qwen06.__file__).resolve(),
            *(Path(module).resolve() for _, _, module in trainer.SOURCE_MODULES),
        }
    )


def manifest_files(path, status, pins):
    manifest, directory = read(path), path.parent
    require(
        manifest["version"] == trainer.VERSION and manifest["status"] == status,
        "Require a completed matched Gemma stage.",
    )
    files = manifest["files_sha256"]
    actual = set()
    for child in directory.rglob("*"):
        safe_path(child)
        if child.is_file() and child != path:
            actual.add(str(child.relative_to(directory)))
    require(set(files) == actual, "Run inventory contains unbound or missing files.")
    for name, expected in files.items():
        child = checked({"path": name, "sha256": expected}, directory, pins)
        require(child.is_relative_to(directory), "Manifest file escaped its run.")
    return manifest


def validate_final(
    config, config_path, train_path, reload_path, rows, diagnostics, pins
):
    """Verify saved evidence without comparing a remote PID to this host's PID."""
    train = manifest_files(train_path, TRAIN_STATUS, pins)
    reload = manifest_files(reload_path, RELOAD_STATUS, pins)
    config_sha = digest(config_path)
    required = {
        "run.json",
        "configuration.json",
        "runtime.json",
        "smoke-generation.json",
        *(name for name, _, _ in trainer.SOURCE_MODULES),
    }
    require(
        required
        | {
            "exposure.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "adapter-tensors.json",
            "initialization.json",
            "training-state.pt",
            "training-rows.jsonl",
            "training-events.jsonl",
        }
        <= set(train["files_sha256"])
        and required | {"reload-parity.json"} <= set(reload["files_sha256"]),
        "Missing required final training/reload payloads.",
    )
    runs = [read(p.parent / "run.json") for p in (train_path, reload_path)]
    for manifest, run, mode, status in zip(
        (train, reload),
        runs,
        ("train", "fresh_reload"),
        (TRAIN_STATUS, RELOAD_STATUS),
        strict=True,
    ):
        require(
            manifest["files_sha256"]["configuration.json"] == config_sha
            and run["configuration_sha256"] == config_sha
            and run["version"] == trainer.VERSION
            and run["status"] == status
            and run["mode"] == mode
            and run["phase"] == "substantive"
            and run["format"] == config["format"]
            and run["training_view"] == config["format"]
            and run["stage"] == "matched_format_sft"
            and run["optimizer_state_resumed"] is False
            and run["final_test_bodies_read"] == 0,
            "Final run identity, phase, format or configuration differs.",
        )
        for name, key, _ in trainer.SOURCE_MODULES:
            require(
                manifest["files_sha256"][name] == run[key] == config[key],
                "Saved runtime source differs from the registered source.",
            )
        for name, key in (
            ("preparation", "preparation_sha256"),
            ("evaluation_protocol", "evaluation_protocol_sha256"),
            ("native_pin", "native_pin_sha256"),
        ):
            require(
                run[key] == config["bindings"][name]["sha256"],
                "Run input provenance changed.",
            )
        require(
            run["t1_evidence_sha256"]
            == {
                name: config["bindings"][name]["sha256"] for name in trainer.T1_BINDINGS
            },
            "Run T1 evidence changed.",
        )
    for path in (train_path, reload_path):
        runtime = read(path.parent / "runtime.json")
        require(
            set(runtime["targets"]) == set(config["model"]["target_modules"])
            and len(runtime["targets"]) == 205
            and all(value["class"] == "Linear" for value in runtime["targets"].values())
            and runtime["tokenizer_parity_rows"] == 802
            and runtime["numerical_regime"] == "native_bfloat16_lora_float32"
            and runtime["parameter_dtypes"] == ["torch.bfloat16"]
            and not runtime["ignored_native_tensor_names"]
            and not any(runtime["loading_info"].values())
            and runtime["effective_generation"] == config["generation"]
            and runtime["completion_label_eos_token_id"] == 1,
            "Actual native runtime/tokenizer/generation census differs.",
        )
    saved, restored = runs
    require(
        saved["training_updates"] == 201
        and saved["optimizer_committed_rows"] == saved["consumed_rows"] == 802
        and saved["initialization"] == "original_native_base_with_fresh_lora"
        and restored["initialization"] == "original_native_base_with_saved_lora"
        and restored["reloaded_training_updates"] == 201
        and restored["previous_manifest_sha256"] == digest(train_path)
        and type(saved["pid"]) is int
        and type(restored["pid"]) is int
        and 0 < saved["pid"] != restored["pid"] > 0
        and saved["started_unix"]
        < saved["finished_unix"]
        <= restored["started_unix"]
        < restored["finished_unix"]
        < time.time(),
        "Require final 201/802 and a subsequent independent fresh reload.",
    )
    expected = trainer.curriculum.Exposure()
    for row in rows:
        for phase in trainer.curriculum.PHASES:
            expected.record(phase, row)
    expected.updates = 201
    require(
        read(train_path.parent / "exposure.json") == expected.summary(),
        "Exact source/case/token exposure or final accumulation differs.",
    )
    init = read(train_path.parent / "initialization.json")
    require(
        init["initialization"] == "original_native_base_with_fresh_lora"
        and all(
            init[key] is False
            for key in (
                "smoke_adapter_loaded",
                "t1_adapter_loaded",
                "other_format_adapter_loaded",
            )
        )
        and init["format"] == config["format"]
        and len(init["trainable_tensors"]) == 410
        and init["trainable_parameters"] == 18585600,
        "Matched arms must each start from the native base with a fresh adapter.",
    )
    order = [row["case_id"] for row in diagnostics]
    original = read(train_path.parent / "smoke-generation.json")
    actual = read(reload_path.parent / "smoke-generation.json")
    parity = read(reload_path.parent / "reload-parity.json")
    require(
        len(order) == len(set(order)) == 8
        and original == actual
        and [row["case_id"] for row in actual] == order
        and parity
        == {
            "adapter_tensors_identical": True,
            "optimizer_state_resumed": False,
            "cases": [{"case_id": name, "identical": True} for name in order],
        },
        "Require all eight exact format-specific fresh-reload outputs/tensors.",
    )
    return train, reload


def load_plan(config_path, policy_path, policy_sha, output):
    pins = {}
    policy_path = checked({"path": str(policy_path), "sha256": policy_sha}, ROOT, pins)
    policy = read(policy_path)
    require(
        policy["version"] == VERSION and policy["limits"] == LIMITS,
        "Storage policy/version changed; budgets cannot be expanded implicitly.",
    )
    required = {str(path.relative_to(ROOT)) for path in source_paths()}
    require(set(policy["sources_sha256"]) == required, "Bind every driver dependency.")
    for name, sha in policy["sources_sha256"].items():
        checked({"path": name, "sha256": sha}, ROOT, pins)
    for name, sha in FROZEN.items():
        require(
            policy["sources_sha256"][name] == sha, "Frozen export/clone source changed."
        )
    bound = {
        name: checked(policy[name], policy_path.parent, pins)
        for name in (
            "export_configuration",
            "training_configuration",
            "training_manifest",
            "fresh_reload",
            "protocol",
            "canonical_export_receipt",
        )
    }
    config_path = safe_path(config_path)
    require(config_path == bound["export_configuration"], "Wrong export configuration.")
    training_path = bound["training_configuration"]
    training = read(training_path)
    require(
        training["phase"] == "substantive", "Smoke exports are not final T2 exports."
    )
    trainer.validate_configuration(training)
    for _, key, module in trainer.SOURCE_MODULES:
        require(
            pins[Path(module).resolve()] == training[key], "Trainer source pin changed."
        )
    inputs = {
        name: checked(value, training_path.parent, pins)
        for name, value in training["bindings"].items()
    }
    require(
        pins[inputs["evaluation_protocol"]] == pins[bound["protocol"]]
        and pins[inputs["t1_export_receipt"]]
        == pins[bound["canonical_export_receipt"]],
        "Storage policy differs from the registered experiment/T1 templates.",
    )
    bundle = trainer.preparation.verify_bundle(inputs["preparation"].parent)
    require(
        bundle["manifest"] == read(inputs["preparation"])
        and bundle["verification"]["dev_test_record_bodies_read"] == 0,
        "Train-only preparation changed.",
    )
    release_sha = bundle["manifest"]["release_manifest_sha256"]
    trainer.validate_protocol(read(bound["protocol"]), training, release_sha)
    trainer.validate_t1_evidence(training, training_path.parent, release_sha)
    rows, diagnostics = trainer.validate_prepared(training, bundle)
    trainer.validate_smoke_gate(
        training,
        training_path.parent,
        release_sha,
        [row["case_id"] for row in diagnostics],
    )
    train, _ = validate_final(
        training,
        training_path,
        bound["training_manifest"],
        bound["fresh_reload"],
        rows,
        diagnostics,
        pins,
    )
    raw_export = read(config_path)
    for key in ("base", "adapter", "training_record", "native_pin"):
        safe_path(raw_export[key]["path"], config_path.parent)
    safe_path(raw_export["runtime"]["source_path"], config_path.parent)
    for value in raw_export["runtime"]["tools"].values():
        safe_path(value["path"], config_path.parent)
    for name in raw_export["runtime"].get("support_files_sha256", {}):
        safe_path(name, config_path.parent)
    config = exporter.configuration(config_path)
    train_root = bound["training_manifest"].parent
    record = checked(config["training_record"], config_path.parent, pins)
    require(
        record in (train_root / "run.json", train_root / "manifest.json")
        and safe_path(config["adapter"]["path"]) == train_root / "adapter"
        and config["adapter"]["files_sha256"]
        == {
            name.removeprefix("adapter/"): sha
            for name, sha in train["files_sha256"].items()
            if name.startswith("adapter/")
        },
        "Export does not bind the exact final adapter/run inventory.",
    )
    for key in ("base", "adapter"):
        safe_path(config[key]["path"])
    require(
        config["base"]["files_sha256"] == training["base"]["files_sha256"]
        and config["native_pin"]["sha256"]
        == training["bindings"]["native_pin"]["sha256"],
        "Native export identity differs from native training.",
    )
    canonical_path = bound["canonical_export_receipt"]
    canonical = read(canonical_path)
    canonical_config_path = checked(
        {"path": "configuration.json", "sha256": canonical["configuration_sha256"]},
        canonical_path.parent,
        pins,
    )
    canonical_config = read(canonical_config_path)
    require(
        canonical["version"] == exporter.VERSION
        and canonical["status"] == EXPORT_STATUS
        and canonical["base_repository"] == exporter.BASE_REPOSITORY
        and canonical["base_revision"] == exporter.BASE_REVISION
        and canonical["runtime_revision"] == exporter.RUNTIME_REVISION
        and canonical["native_generation_stop_ids"] == [1, 106, 50]
        and all(
            canonical_config[key] == config[key]
            for key in ("base", "native_pin", "runtime")
        ),
        "Reuse the canonical native base/native-pin paths and identical conversion runtime.",
    )
    templates = canonical["artifacts"]
    require(
        set(templates) == set(exporter.paths(Path("unused"))),
        "Incomplete canonical pair.",
    )
    for key, value in templates.items():
        path = safe_path(value["path"])
        require(
            path == exporter.paths(canonical_path.parent)[key]
            and path.is_file()
            and path.stat().st_size == value["bytes"],
            "Missing or relocated canonical artifact; full hash proof occurs in compaction.",
        )
    output = safe_path(output)
    require(
        output.is_relative_to(ROOT / "artifacts" / "training")
        and output.parent.is_dir()
        and not output.exists(),
        "Use a brand-new attempt inside artifacts/training with an existing parent.",
    )
    protected = {
        training_path.parent,
        train_root,
        bound["fresh_reload"].parent,
        inputs["preparation"].parent,
        canonical_path.parent,
        safe_path(config["base"]["path"]),
        safe_path(config["adapter"]["path"]),
        safe_path(config["native_pin"]["path"]).parent,
        safe_path(config["runtime"]["source_path"]),
    }
    require(
        not any(output.is_relative_to(path) for path in protected),
        "New output must be outside every immutable input directory.",
    )
    require(
        not any(path.is_relative_to(output) for path in pins),
        "Output contains an input.",
    )
    return {
        "version": VERSION,
        "status": "plan_only_no_export_executed",
        "configuration": binding(config_path),
        "policy": binding(policy_path),
        "format": training["format"],
        "training_updates": 201,
        "committed_draws": 802,
        "bindings": {name: binding(path) for name, path in bound.items()},
        "output": str(output),
        "export_directory": str(output / "export"),
        "protected_directories": sorted(str(path) for path in protected),
        "templates": templates,
        "limits": LIMITS,
        "commands": {
            stage: [
                config["runtime"]["python"],
                "-u",
                "-B",
                str(Path(exporter.__file__).resolve()),
                stage,
                "--config",
                str(config_path),
                "--output",
                str(output / "export"),
            ]
            for stage in STAGES
        },
        "space_required_before_stages": stage_requirements(templates),
        "free_bytes_observed": shutil.disk_usage(output.parent).free,
        "full_canonical_hashes": "Verified by rebuild_clone before each atomic publication; plan checks metadata and sizes only.",
        "pins": {str(path): sha for path, sha in pins.items()},
    }


def stage_requirements(templates):
    return {
        stage: LIMITS["minimum_free_bytes"]
        + LIMITS["stage_overhead_bytes"]
        + sum(templates[key]["bytes"] for key in keys)
        for stage, keys in (
            ("convert", ("base", "adapter", "merged")),
            ("quantize", ("base_quantized", "merged_quantized")),
        )
    }


def guard_pins(plan):
    for name, expected in plan["pins"].items():
        path = safe_path(name)
        require(
            path.is_file() and digest(path) == expected, f"Bound input changed: {name}"
        )


def assert_owned(owner, *, before_receipt=False):
    root = safe_path(owner["root"])
    require(
        root.is_dir() and list(clone.identity(root)[:2]) == owner["root_identity"],
        "Attempt directory ownership changed.",
    )
    path = safe_path(root / "ownership.json")
    require(
        read(path) == {k: v for k, v in owner.items() if k != "export_identity"},
        "Attempt ownership marker changed.",
    )
    for child in root.rglob("*"):
        safe_path(child)
    if "export_identity" in owner:
        require(
            list(clone.identity(root / "export")[:2]) == owner["export_identity"],
            "Export directory ownership changed.",
        )
        require(
            read(root / "export-ownership.json")
            == {"identity": owner["export_identity"], "owner_token": owner["token"]},
            "Export ownership receipt changed.",
        )
    if before_receipt:
        require(
            not (root / "export" / "receipt.json").exists(),
            "Never compact an artifact after its final receipt.",
        )
    return root


def compact(owner, key, template, target_record):
    root = assert_owned(owner, before_receipt=True)
    target = safe_path(target_record["path"])
    require(
        target == exporter.paths(root / "export")[key],
        "Target is not this attempt's artifact.",
    )
    require(
        target.is_file() and target.stat().st_nlink == 1,
        "Target must be an owned regular file.",
    )
    base = safe_path(template["path"])
    require(
        not base.is_relative_to(root) and base.is_file(),
        "Template must be immutable external input.",
    )
    require(
        target_record["bytes"] == template["bytes"] == target.stat().st_size,
        "Artifact size differs from its canonical physical template.",
    )
    if key in ("base", "base_quantized"):
        require(
            target_record["sha256"] == template["sha256"],
            "Native baseline must be byte-identical.",
        )
    scratch = root / "compaction" / f"{key}.scratch.gguf"
    original_identity = clone.identity(target)
    base_identity = clone.identity(base)
    report = clone.rebuild_clone(
        base,
        target,
        scratch,
        base_sha256=template["sha256"],
        target_sha256=target_record["sha256"],
        max_changed_bytes=LIMITS["max_changed_bytes"][key],
        minimum_free_bytes=LIMITS["minimum_free_bytes"],
        chunk_bytes=LIMITS["chunk_bytes"],
    )
    write_new(root / "compaction" / f"{key}.reconstruction.json", report)
    assert_owned(owner, before_receipt=True)
    scratch_identity = clone.identity(scratch)
    require(
        clone.identity(target) == original_identity
        and digest(target) == target_record["sha256"]
        and digest(scratch) == target_record["sha256"],
        "Complete artifact proof changed immediately before publication.",
    )
    require(
        clone.identity(target) == original_identity
        and clone.identity(scratch) == scratch_identity
        and clone.identity(base) == base_identity,
        "An artifact identity changed while hashing.",
    )
    require(
        shutil.disk_usage(root).free >= LIMITS["minimum_free_bytes"],
        "Free-space reserve was crossed before publication.",
    )
    os.replace(scratch, target)
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    published = {
        "status": "byte_identical_owned_artifact_published",
        "artifact": key,
        "owner_token": owner["token"],
        "pid": os.getpid(),
        "target": target_record,
        "canonical_template": template,
        "reconstruction": binding(root / "compaction" / f"{key}.reconstruction.json"),
        "free_bytes_after_publication": shutil.disk_usage(root).free,
        "canonical_inputs_replaced": False,
    }
    write_new(root / "compaction" / f"{key}.publication.json", published)
    return published


def run_stage(plan, owner, stage):
    root = assert_owned(owner)
    command = plan["commands"][stage]
    record = {"stage": stage, "argv": command, "started_unix": time.time()}
    write_new(root / "stages" / f"{stage}.command.json", record)
    log, child, started = root / "stages" / f"{stage}.log", None, time.monotonic()
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["pid"] = child.pid
            child.wait(timeout=LIMITS["stage_timeout_seconds"])
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        record.update(
            returncode=child.returncode if child else None,
            seconds=time.monotonic() - started,
            log_sha256=digest(log) if log.exists() else None,
        )
        write_new(root / "stages" / f"{stage}.result.json", record)
    require(record["returncode"] == 0, f"Frozen exporter stage failed: {stage}")
    return record


def execute(plan):
    guard_pins(plan)
    output = safe_path(plan["output"])
    require(not output.exists(), "Preserve every previous attempt.")
    lock_path = safe_path(ROOT / "artifacts/nep-passive-b1.lock")
    with os.fdopen(
        os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard_pins(plan)
        output.mkdir(exist_ok=False)
        owner = {
            "root": str(output),
            "root_identity": list(clone.identity(output)[:2]),
            "pid": os.getpid(),
            "token": uuid.uuid4().hex,
            "created_unix": time.time(),
        }
        write_new(output / "ownership.json", owner)
        (output / "stages").mkdir()
        (output / "compaction").mkdir()
        write_new(output / "plan.json", plan)
        result = {
            "version": VERSION,
            "status": "running",
            "pid": os.getpid(),
            "owner": owner,
            "started_unix": time.time(),
            "lock": str(lock_path),
            "stages": [],
            "compactions": [],
            "space_checks": [],
            "configuration": plan["configuration"],
            "policy": plan["policy"],
            "bindings": plan["bindings"],
            "model_inference_requests": 0,
        }
        result_fd = os.open(
            output / "result.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            for stage in STAGES:
                guard_pins(plan)
                assert_owned(owner, before_receipt=True)
                if stage in plan["space_required_before_stages"]:
                    free = shutil.disk_usage(output).free
                    needed = plan["space_required_before_stages"][stage]
                    check = {
                        "stage": stage,
                        "free_bytes": free,
                        "required_bytes": needed,
                    }
                    result["space_checks"].append(check)
                    write_new(output / "stages" / f"{stage}.space.json", check)
                    require(
                        free >= needed,
                        f"Insufficient conservative space before {stage}.",
                    )
                print(f"Starting frozen exporter {stage}", flush=True)
                result["stages"].append(run_stage(plan, owner, stage))
                assert_owned(owner)
                if stage == "preflight":
                    directory = safe_path(output / "export")
                    require(
                        directory.is_dir(),
                        "Frozen preflight did not create its export directory.",
                    )
                    owner["export_identity"] = list(clone.identity(directory)[:2])
                    write_new(
                        output / "export-ownership.json",
                        {
                            "identity": owner["export_identity"],
                            "owner_token": owner["token"],
                        },
                    )
                if stage in ("convert", "quantize"):
                    report = read(
                        output
                        / "export"
                        / (
                            "conversion.json"
                            if stage == "convert"
                            else "quantization.json"
                        )
                    )
                    keys = (
                        ("base", "merged")
                        if stage == "convert"
                        else ("base_quantized", "merged_quantized")
                    )
                    for key in keys:
                        guard_pins(plan)
                        result["compactions"].append(
                            compact(
                                owner,
                                key,
                                plan["templates"][key],
                                report["artifacts"][key],
                            )
                        )
                print(f"Passed frozen exporter {stage}", flush=True)
            guard_pins(plan)
            receipt_path = output / "export/receipt.json"
            receipt = read(receipt_path)
            require(
                receipt["status"] == EXPORT_STATUS
                and receipt["configuration_sha256"]
                == digest(output / "export/configuration.json")
                and receipt["training_record"]
                == exporter.configuration(Path(plan["configuration"]["path"]))[
                    "training_record"
                ],
                "Final frozen receipt identity changed.",
            )
            result.update(
                status="paired_export_passed_local_evaluation_pending",
                exit_code=0,
                export_receipt=binding(receipt_path),
            )
        except BaseException as error:
            result.update(
                status="failed", exit_code=1, error=f"{type(error).__name__}: {error}"
            )
            print(result["error"], flush=True)
        finally:
            result["finished_utc"] = datetime.now(timezone.utc).isoformat()
            # Keep failure evidence on our original inode even if path ownership is lost.
            with os.fdopen(result_fd, "w") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.config, args.policy, args.policy_sha256, args.output)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    def interrupted(*_):
        raise KeyboardInterrupt("Export driver interrupted.")

    signal.signal(signal.SIGTERM, interrupted)
    return execute(plan)["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
