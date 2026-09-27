"""Export a new Gemma adapter with explicit reuse of verified canonical base files.

This separate operational policy never runs convert-base or quantize-base. It
clones their exact canonical outputs, executes the frozen adapter/merge commands
and merged quantization, then runs the unchanged tensor verifier and receipt.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import run_passive_gemma_matched_export as previous

ROOT = previous.ROOT
exporter = previous.exporter
clone = previous.clone
require = previous.require
read = previous.read
write_new = previous.write_new
safe_path = previous.safe_path
checked = previous.checked
binding = previous.binding
digest = previous.digest
VERSION = "passive-gemma-reuse-base-export-v1"
PREVIOUS_SHA256 = "c84bfd3c84bf46a07b7c3318b9d244da0f702b256fd1405adde1ad8b7c068f79"
LIMITS = dict(previous.LIMITS)
STAGES = previous.STAGES
METHOD = "verified_canonical_base_clones_fresh_adapter_merge_and_merged_quantization"
WORKER_AUTH_ENV = "PASSIVE_GEMMA_EXPORT_WORKER_AUTH"


def source_paths():
    return sorted({Path(__file__).resolve(), *previous.source_paths()})


def stage_requirements(templates):
    fixed = LIMITS["minimum_free_bytes"] + LIMITS["stage_overhead_bytes"]
    return {
        "convert": fixed
        + templates["adapter"]["bytes"]
        + templates["merged"]["bytes"]
        + LIMITS["max_changed_bytes"]["merged"],
        "quantize": fixed
        + templates["merged_quantized"]["bytes"]
        + LIMITS["max_changed_bytes"]["merged_quantized"],
    }


def canonical_evidence(path, config, pins):
    receipt = read(path)
    require(
        receipt["version"] == exporter.VERSION
        and receipt["status"] == previous.EXPORT_STATUS
        and receipt["base_repository"] == exporter.BASE_REPOSITORY
        and receipt["base_revision"] == exporter.BASE_REVISION
        and receipt["runtime_revision"] == exporter.RUNTIME_REVISION
        and receipt["native_generation_stop_ids"] == [1, 106, 50],
        "Require the completed canonical Gemma export identity.",
    )
    records = {}
    for name, field in (
        ("configuration.json", "configuration_sha256"),
        ("tensor-verification.json", "tensor_verification_sha256"),
        ("quantization.json", "quantization_sha256"),
    ):
        records[name] = checked(
            {"path": name, "sha256": receipt[field]}, path.parent, pins
        )
    original = read(records["configuration.json"])
    require(
        all(original[key] == config[key] for key in ("base", "native_pin", "runtime")),
        "Canonical native files, converter sources, tools and support hashes differ.",
    )
    tensor = read(records["tensor-verification.json"])
    quantization = read(records["quantization.json"])
    require(
        tensor["status"] == "passed"
        and tensor["adapter_tensors_exact"] == 410
        and tensor["native_targets_exact"] == 205
        and len(tensor["merged_targets"]) == 205
        and len(tensor["unadapted_tensors_exact"]) == 453
        and quantization["quantization"] == "Q4_K_M"
        and quantization["importance_matrix"] is None
        and quantization["threads"] == 4,
        "Canonical tensor/quantization proof changed.",
    )
    artifacts = {**tensor["artifacts"], **quantization["artifacts"]}
    require(artifacts == receipt["artifacts"], "Canonical artifact records differ.")
    require(set(artifacts) == set(exporter.paths(Path("unused"))), "Missing template.")
    for key, value in artifacts.items():
        artifact_path = safe_path(value["path"])
        require(
            artifact_path == exporter.paths(path.parent)[key]
            and artifact_path.is_file()
            and artifact_path.stat().st_size == value["bytes"],
            "Canonical artifact missing or relocated.",
        )
    executions = {}
    expected_commands = {
        "convert-base": [
            str(v) for v in exporter.conversion_commands(original, path.parent)[0][1]
        ],
        "quantize-base": [
            original["runtime"]["tools"]["llama-quantize"]["path"],
            str(exporter.paths(path.parent)["base"]),
            str(exporter.paths(path.parent)["base_quantized"]),
            "Q4_K_M",
            "4",
        ],
    }
    for name in ("convert-base", "quantize-base"):
        files = {}
        for suffix in ("command.json", "result.json", "log"):
            command_path = safe_path(path.parent / "logs" / f"{name}.{suffix}")
            if name == "quantize-base":
                value = quantization["commands"]["base"][suffix]
                require(
                    safe_path(value["path"]) == command_path,
                    "Wrong canonical command path.",
                )
                checked(value, path.parent, pins)
            else:
                # Original conversion.json does not bind logs. The independently
                # reviewed operational policy supplies their exact bindings.
                require(
                    command_path in pins,
                    "Policy must bind original base-conversion logs.",
                )
            files[suffix] = exporter.artifact(command_path)
        command = read(Path(files["command.json"]["path"]))
        result = read(Path(files["result.json"]["path"]))
        require(
            command["argv"] == expected_commands[name]
            and result["returncode"] == 0
            and result["log_sha256"] == files["log"]["sha256"],
            "Canonical command/result/log provenance differs.",
        )
        executions[name] = files
    return {
        "receipt": binding(path),
        "configuration": binding(records["configuration.json"]),
        "tensor_verification": binding(records["tensor-verification.json"]),
        "quantization": binding(records["quantization.json"]),
        "templates": artifacts,
        "executions": executions,
        "base_quantizer_fallbacks": quantization["quantizer_fallbacks"]["base"],
    }


def validate_engineering_result(path, canonical, pins):
    result = read(path)
    require(
        result["version"] == VERSION
        and result["method"] == METHOD
        and result["mode"] == "engineering_reexport"
        and result["status"] == "engineering_reexport_exact_parity_passed"
        and result["exit_code"] == 0
        and result["canonical_receipt"] == canonical["receipt"]
        and result["all_five_artifacts_match_canonical"] is True
        and result["implementation_sha256"] == digest(Path(__file__)),
        "Require a completed engineering reexport with this exact implementation.",
    )
    receipt_path = checked(result["export_receipt"], path.parent, pins)
    receipt = read(receipt_path)
    require(
        receipt["status"] == previous.EXPORT_STATUS
        and {k: (v["sha256"], v["bytes"]) for k, v in receipt["artifacts"].items()}
        == {k: (v["sha256"], v["bytes"]) for k, v in canonical["templates"].items()},
        "Engineering output hashes differ from the known complete artifacts.",
    )
    require(
        [stage["stage"] for stage in result["stages"]] == list(STAGES)
        and all(stage["returncode"] == 0 for stage in result["stages"]),
        "Engineering stages are incomplete.",
    )


def load_plan(config_path, policy_path, policy_sha, output):
    pins = {}
    policy_path = checked({"path": str(policy_path), "sha256": policy_sha}, ROOT, pins)
    policy = read(policy_path)
    require(
        policy["version"] == VERSION
        and policy["method"] == METHOD
        and policy["limits"] == LIMITS
        and policy["mode"] in ("engineering_reexport", "matched_final")
        and policy["registration_status"]
        in ("draft_pending_review", "registered_before_execution"),
        "Register the distinct reuse policy, mode and unchanged limits.",
    )
    expected_sources = {str(p.relative_to(ROOT)) for p in source_paths()}
    require(set(policy["sources_sha256"]) == expected_sources, "Bind every dependency.")
    for name, sha in policy["sources_sha256"].items():
        checked({"path": name, "sha256": sha}, ROOT, pins)
    require(
        digest(Path(previous.__file__)) == PREVIOUS_SHA256, "Frozen validator changed."
    )
    for name, sha in previous.FROZEN.items():
        require(pins[safe_path(name)] == sha, "Frozen export/clone helper changed.")
    requested_config = safe_path(config_path)
    config_path = checked(policy["export_configuration"], policy_path.parent, pins)
    require(requested_config == config_path, "Wrong export configuration argument.")
    config = exporter.configuration(config_path)
    for group in ("canonical_base_conversion_logs", "canonical_base_quantization_logs"):
        require(
            set(policy[group]) == {"command.json", "result.json", "log"},
            "Bind all three original base command records.",
        )
        for value in policy[group].values():
            checked(value, policy_path.parent, pins)
    canonical_path = checked(
        policy["canonical_export_receipt"], policy_path.parent, pins
    )
    canonical = canonical_evidence(canonical_path, config, pins)
    for group, command in (
        ("canonical_base_conversion_logs", "convert-base"),
        ("canonical_base_quantization_logs", "quantize-base"),
    ):
        require(
            all(
                checked(policy[group][suffix], policy_path.parent, pins)
                == Path(canonical["executions"][command][suffix]["path"])
                for suffix in ("command.json", "result.json", "log")
            ),
            "Explicit canonical log binding differs from genuine command history.",
        )
    if policy["mode"] == "matched_final":
        validation_path = checked(
            policy["scientific_validation_policy"], policy_path.parent, pins
        )
        validated = previous.load_plan(
            config_path, validation_path, digest(validation_path), output
        )
        require(
            validated["bindings"]["canonical_export_receipt"] == canonical["receipt"],
            "Scientific validation and storage templates differ.",
        )
        pins.update({Path(k): v for k, v in validated["pins"].items()})
        engineering = checked(policy["engineering_probe"], policy_path.parent, pins)
        validate_engineering_result(engineering, canonical, pins)
        protected = {Path(p) for p in validated["protected_directories"]}
        scientific = {
            "validation_policy": binding(validation_path),
            "engineering_probe": binding(engineering),
            "format": validated["format"],
            "training_updates": 201,
            "committed_draws": 802,
        }
    else:
        require(
            config == read(Path(canonical["configuration"]["path"])),
            "Engineering reexport must use the exact completed T1 export configuration.",
        )
        verified_path = checked(
            policy["canonical_independent_verification"], policy_path.parent, pins
        )
        verified = read(verified_path)
        require(
            verified["status"] == "passed"
            and verified["bindings"]["export_receipt"] == canonical["receipt"]
            and verified["checks"]
            and all(value is True for value in verified["checks"].values()),
            "Require the genuine independent canonical export verification.",
        )
        protected = {
            canonical_path.parent,
            safe_path(config["base"]["path"]),
            safe_path(config["adapter"]["path"]),
            safe_path(config["native_pin"]["path"]).parent,
            safe_path(config["runtime"]["source_path"]),
            safe_path(config["training_record"]["path"]).parent,
        }
        scientific = {
            "canonical_independent_verification": binding(verified_path),
            "new_training_updates": 0,
            "known_output_parity_required": True,
        }
    output = safe_path(output)
    require(
        output.is_relative_to(ROOT / "artifacts/training")
        and output.parent.is_dir()
        and not output.exists()
        and not any(output.is_relative_to(p) for p in protected)
        and not any(p.is_relative_to(output) for p in pins),
        "Use a new owned attempt outside every immutable input.",
    )
    commands = {
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
    }
    for stage in ("convert", "quantize"):
        commands[stage] = [
            config["runtime"]["python"],
            "-u",
            "-B",
            str(Path(__file__).resolve()),
            "worker",
            "--plan",
            str(output / "plan.json"),
            "--stage",
            stage,
        ]
    return {
        "version": VERSION,
        "method": METHOD,
        "mode": policy["mode"],
        "registration_status": policy["registration_status"],
        "status": "plan_only_no_export_executed",
        "configuration": binding(config_path),
        "policy": binding(policy_path),
        "output": str(output),
        "canonical": canonical,
        "templates": canonical["templates"],
        "scientific_validation": scientific,
        "limits": LIMITS,
        "commands": commands,
        "space_required_before_stages": stage_requirements(canonical["templates"]),
        "free_bytes_observed": shutil.disk_usage(output.parent).free,
        "pins": {str(path): sha for path, sha in pins.items()},
        "implementation_sha256": digest(Path(__file__)),
    }


def reuse_base(plan, owner, key):
    root = previous.assert_owned(owner, before_receipt=True)
    template = plan["templates"][key]
    source = safe_path(template["path"])
    target = exporter.paths(root / "export")[key]
    require(
        key in ("base", "base_quantized") and not target.exists(),
        "Only new base clones.",
    )
    report = clone.rebuild_clone(
        source,
        source,
        target,
        base_sha256=template["sha256"],
        target_sha256=template["sha256"],
        max_changed_bytes=0,
        minimum_free_bytes=LIMITS["minimum_free_bytes"],
        chunk_bytes=LIMITS["chunk_bytes"],
    )
    receipt = {
        "status": "canonical_base_exact_clone_reused",
        "artifact": key,
        "canonical_receipt": plan["canonical"]["receipt"],
        "canonical_artifact": template,
        "new_artifact": exporter.artifact(target),
        "executed_base_conversion_or_quantization": False,
        "clone": report,
    }
    write_new(root / "compaction" / f"{key}.reuse.json", receipt)
    return receipt


def convert_reusing_base(plan, owner):
    root = previous.assert_owned(owner, before_receipt=True)
    output = root / "export"
    config, _ = exporter.bound_run(Path(plan["configuration"]["path"]), output)
    target = exporter.paths(output)
    exporter.refuse_existing(
        output / "conversion.json", *(target[k] for k in ("base", "adapter", "merged"))
    )
    reused = reuse_base(plan, owner, "base")
    commands = exporter.conversion_commands(config, output)
    require(
        [name for name, _ in commands] == ["convert-base", "convert-adapter", "merge"],
        "Frozen conversion command structure changed.",
    )
    for name, command in commands[1:]:
        exporter.run_command(output, name, command)
    report = {
        "status": "converted_tensor_verification_pending",
        "artifacts": {
            key: exporter.artifact(target[key]) for key in ("base", "adapter", "merged")
        },
        "operation_method": METHOD,
        "execution_provenance": {
            "convert-base": {
                "mode": "reused_canonical_output_not_executed",
                "canonical_execution": plan["canonical"]["executions"]["convert-base"],
                "reuse": binding(root / "compaction/base.reuse.json"),
            },
            "convert-adapter": {"mode": "executed_current_attempt"},
            "merge": {"mode": "executed_current_attempt"},
        },
    }
    require(
        reused["new_artifact"] == report["artifacts"]["base"], "Base clone changed."
    )
    write_new(output / "conversion.json", report)
    return report


def quantize_reusing_base(plan, owner):
    root = previous.assert_owned(owner, before_receipt=True)
    output = root / "export"
    config, _ = exporter.bound_run(Path(plan["configuration"]["path"]), output)
    verification = read(output / "tensor-verification.json")
    require(verification["status"] == "passed", "Fresh tensor verification must pass.")
    target = exporter.paths(output)
    exporter.refuse_existing(
        output / "quantization.json",
        target["base_quantized"],
        target["merged_quantized"],
    )
    for key in ("base", "merged"):
        exporter.check_hash(target[key], verification["artifacts"][key]["sha256"])
    require(
        verification["artifacts"]["base"]["sha256"]
        == plan["templates"]["base"]["sha256"],
        "Quantized baseline must derive from the identical canonical F16 base.",
    )
    reuse_base(plan, owner, "base_quantized")
    exporter.run_command(
        output,
        "quantize-merged",
        [
            config["runtime"]["tools"]["llama-quantize"]["path"],
            target["merged"],
            target["merged_quantized"],
            "Q4_K_M",
            "4",
        ],
    )
    merged_records = {
        suffix: exporter.artifact(output / "logs" / f"quantize-merged.{suffix}")
        for suffix in ("command.json", "result.json", "log")
    }
    report = {
        "status": "quantized_pair_verification_pending",
        "quantization": "Q4_K_M",
        "importance_matrix": None,
        "threads": 4,
        "commands": {
            "base": plan["canonical"]["executions"]["quantize-base"],
            "merged": merged_records,
        },
        "quantizer_fallbacks": {
            "base": plan["canonical"]["base_quantizer_fallbacks"],
            "merged": [
                line
                for line in (output / "logs/quantize-merged.log")
                .read_text()
                .splitlines()
                if "not divisible by" in line or "fallback" in line.lower()
            ],
        },
        "artifacts": {
            key: exporter.artifact(target[key])
            for key in ("base_quantized", "merged_quantized")
        },
        "operation_method": METHOD,
        "execution_provenance": {
            "base": {
                "mode": "reused_canonical_output_not_executed",
                "canonical_receipt": plan["canonical"]["receipt"],
                "reuse": binding(root / "compaction/base_quantized.reuse.json"),
            },
            "merged": {"mode": "executed_current_attempt"},
        },
    }
    write_new(output / "quantization.json", report)
    return report


def authorize_worker(owner, stage):
    capability = json.loads(os.environ.pop(WORKER_AUTH_ENV, "null"))
    require(
        isinstance(capability, dict)
        and os.getppid() == owner["pid"]
        and capability.get("stage") == stage
        and hashlib.sha256(capability.get("secret", "").encode()).hexdigest()
        == owner["worker_authorization_sha256"],
        "Worker requires its live execute parent and inherited authorization.",
    )
    descriptor = capability["lock_fd"]
    require(type(descriptor) is int and descriptor >= 3, "Missing inherited lock.")
    stat = os.fstat(descriptor)
    lock_path = safe_path(ROOT / "artifacts/nep-passive-b1.lock")
    require(
        (stat.st_dev, stat.st_ino) == clone.identity(lock_path)[:2],
        "Inherited descriptor is not the shared evaluation/export lock.",
    )
    # The inherited open-file description retains the parent's lock even if
    # the parent exits; do not unlock or close it while this worker runs.
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def run_worker_stage(plan, owner, stage, lock, secret):
    root = previous.assert_owned(owner)
    command = plan["commands"][stage]
    record = {
        "stage": stage,
        "argv": command,
        "started_unix": time.time(),
        "parent_pid": os.getpid(),
        "parent_lock_inherited": True,
    }
    write_new(root / "stages" / f"{stage}.command.json", record)
    environment = dict(os.environ)
    environment[WORKER_AUTH_ENV] = json.dumps(
        {"stage": stage, "secret": secret, "lock_fd": lock.fileno()}
    )
    log, child, started = root / "stages" / f"{stage}.log", None, time.monotonic()
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
                pass_fds=(lock.fileno(),),
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
    require(record["returncode"] == 0, f"Base-reuse worker failed: {stage}")
    return record


def worker(plan_path, stage):
    plan_path = safe_path(plan_path)
    plan = read(plan_path)
    root = safe_path(plan["output"])
    require(plan_path == root / "plan.json", "Worker plan is not in its owned attempt.")
    owner = read(root / "ownership.json")
    authorize_worker(owner, stage)
    owner["export_identity"] = read(root / "export-ownership.json")["identity"]
    previous.assert_owned(owner, before_receipt=True)
    require(
        plan["version"] == VERSION
        and plan["method"] == METHOD
        and plan["registration_status"] == "registered_before_execution"
        and plan["implementation_sha256"] == digest(Path(__file__)),
        "Worker identity changed.",
    )
    previous.guard_pins(plan)
    require(digest(plan_path) == owner["plan_sha256"], "Owned plan changed.")
    space = read(root / "stages" / f"{stage}.space.json")
    required_bytes = stage_requirements(plan["templates"])[stage]
    require(
        space["stage"] == stage
        and space["required_bytes"] == required_bytes
        and space["free_bytes"] >= required_bytes
        and shutil.disk_usage(root).free >= required_bytes,
        "Worker requires the passed stage space gate and enough current space.",
    )
    require(
        read(root / "stages" / f"{stage}.command.json")["argv"]
        == plan["commands"][stage]
        and not (root / "stages" / f"{stage}.result.json").exists(),
        "Worker is not the current unfinished parent stage.",
    )
    (convert_reusing_base if stage == "convert" else quantize_reusing_base)(plan, owner)


def execute(plan):
    require(
        plan["registration_status"] == "registered_before_execution",
        "Draft policy is plan-only; register the reviewed operational method first.",
    )
    previous.guard_pins(plan)
    output = safe_path(plan["output"])
    require(not output.exists(), "Preserve every previous attempt.")
    lock_path = safe_path(ROOT / "artifacts/nep-passive-b1.lock")
    with os.fdopen(
        os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "a"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous.guard_pins(plan)
        output.mkdir(exist_ok=False)
        write_new(output / "plan.json", plan)
        worker_secret = uuid.uuid4().hex + uuid.uuid4().hex
        owner = {
            "root": str(output),
            "root_identity": list(clone.identity(output)[:2]),
            "pid": os.getpid(),
            "token": uuid.uuid4().hex,
            "created_unix": time.time(),
            "plan_sha256": digest(output / "plan.json"),
            "worker_authorization_sha256": hashlib.sha256(
                worker_secret.encode()
            ).hexdigest(),
        }
        write_new(output / "ownership.json", owner)
        (output / "stages").mkdir()
        (output / "compaction").mkdir()
        result = {
            "version": VERSION,
            "method": METHOD,
            "mode": plan["mode"],
            "status": "running",
            "pid": os.getpid(),
            "started_unix": time.time(),
            "policy": plan["policy"],
            "configuration": plan["configuration"],
            "canonical_receipt": plan["canonical"]["receipt"],
            "implementation_sha256": plan["implementation_sha256"],
            "scientific_validation": plan["scientific_validation"],
            "stages": [],
            "compactions": [],
            "space_checks": [],
            "model_inference_requests": 0,
            "base_commands_executed_this_attempt": False,
            "lock": str(lock_path),
        }
        fd = os.open(
            output / "result.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            for stage in STAGES:
                previous.guard_pins(plan)
                previous.assert_owned(owner, before_receipt=True)
                if stage in plan["space_required_before_stages"]:
                    check = {
                        "stage": stage,
                        "free_bytes": shutil.disk_usage(output).free,
                        "required_bytes": plan["space_required_before_stages"][stage],
                    }
                    result["space_checks"].append(check)
                    write_new(output / "stages" / f"{stage}.space.json", check)
                    require(
                        check["free_bytes"] >= check["required_bytes"],
                        "Insufficient space for full output and scratch clone.",
                    )
                print(f"Starting {stage} under explicit base-reuse policy", flush=True)
                stage_result = (
                    run_worker_stage(plan, owner, stage, lock, worker_secret)
                    if stage in ("convert", "quantize")
                    else previous.run_stage(plan, owner, stage)
                )
                result["stages"].append(stage_result)
                previous.assert_owned(owner)
                if stage == "preflight":
                    owner["export_identity"] = list(
                        clone.identity(output / "export")[:2]
                    )
                    write_new(
                        output / "export-ownership.json",
                        {
                            "identity": owner["export_identity"],
                            "owner_token": owner["token"],
                        },
                    )
                if stage in ("convert", "quantize"):
                    key = "merged" if stage == "convert" else "merged_quantized"
                    report = read(
                        output
                        / "export"
                        / (
                            "conversion.json"
                            if stage == "convert"
                            else "quantization.json"
                        )
                    )
                    result["compactions"].append(
                        previous.compact(
                            owner, key, plan["templates"][key], report["artifacts"][key]
                        )
                    )
                print(f"Passed {stage}", flush=True)
            receipt_path = output / "export/receipt.json"
            receipt = read(receipt_path)
            require(
                receipt["status"] == previous.EXPORT_STATUS
                and receipt["configuration_sha256"]
                == digest(output / "export/configuration.json")
                and receipt["training_record"]
                == exporter.configuration(Path(plan["configuration"]["path"]))[
                    "training_record"
                ]
                and set(receipt["artifacts"]) == set(plan["templates"])
                and all(
                    safe_path(value["path"]) == exporter.paths(output / "export")[key]
                    for key, value in receipt["artifacts"].items()
                ),
                "Final frozen receipt identity differs.",
            )
            for key in ("base", "base_quantized"):
                require(
                    receipt["artifacts"][key]["sha256"]
                    == plan["templates"][key]["sha256"],
                    "Reused base hash changed.",
                )
            exact = all(
                receipt["artifacts"][k]["sha256"] == v["sha256"]
                and receipt["artifacts"][k]["bytes"] == v["bytes"]
                for k, v in plan["templates"].items()
            )
            if plan["mode"] == "engineering_reexport":
                require(
                    exact,
                    "Known-adapter engineering reexport changed complete artifact bytes.",
                )
            previous.guard_pins(plan)
            result.update(
                status="engineering_reexport_exact_parity_passed"
                if plan["mode"] == "engineering_reexport"
                else "paired_export_passed_local_evaluation_pending",
                exit_code=0,
                export_receipt=binding(receipt_path),
                all_five_artifacts_match_canonical=exact,
                conversion_provenance=binding(output / "export/conversion.json"),
                quantization_provenance=binding(output / "export/quantization.json"),
            )
        except BaseException as error:
            result.update(
                status="failed", exit_code=1, error=f"{type(error).__name__}: {error}"
            )
            print(result["error"], flush=True)
        finally:
            result["finished_utc"] = datetime.now(timezone.utc).isoformat()
            with os.fdopen(fd, "w") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "execute"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--policy-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("worker")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--stage", choices=("convert", "quantize"), required=True)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.plan, args.stage)
        return
    plan = load_plan(args.config, args.policy, args.policy_sha256, args.output)
    value = plan if args.command == "plan" else execute(plan)
    print(json.dumps(value, indent=2))
    if value.get("exit_code", 0):
        sys.exit(value["exit_code"])


if __name__ == "__main__":
    main()
