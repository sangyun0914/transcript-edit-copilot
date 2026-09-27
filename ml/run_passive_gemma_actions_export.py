"""Register or export the independently verified 42-draw Gemma continuation.

The new planner validates continuation lineage. All model conversion, tensor
verification, cloning, quantization and worker authorization remain in the
unchanged, engineering-qualified reuse executor.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

import run_passive_gemma_reuse_base_export as reuse
import train_passive_gemma_actions as trainer


ROOT = reuse.ROOT
previous, exporter = reuse.previous, reuse.exporter
require, read, checked = reuse.require, reuse.read, reuse.checked
safe_path, digest, binding, write_new = (
    reuse.safe_path,
    reuse.digest,
    reuse.binding,
    reuse.write_new,
)
VERSION = "passive-gemma-actions-export-v1"
MODE = "actions_final_continuation"
EXECUTOR_SHA = "ac10a483ed9fc95ddc5be91651060d3a5180eddd3acae682b6244676f1bd8240"
TRAINER_SHA = "a6e8c5c3c57025c418a14bbd21839b5b57b49b1eb2789610c25d8c561ab4a61f"
VERIFIER_SHA = "3d7731cb8995cd15c3356b8b9991db5d17d735d1855f1cc6a6f3db0bb01dff0f"
CONFIG_SHA = "c6094d86d195f306bf1833fcdfccc53f61d8fa5a83b50d5e359b9ae7e7e6e309"
FINAL_PROOF_SHA = "e9a68a63b66ab9fb2a058af4bec180ae1457ea56dab7d0ff7c5a19b81e3273e1"
PROTOCOL_SHA = "8f65a1ead40e075dfe599915018076003b3f7214e5278bffe53d2508062c45bd"
CANONICAL_SHA = "481fa3e27cf31dd2422d1558dfe6f90e5e6db60589a1be84d0f2f2a3e8c8f4ea"
CANONICAL_CONFIG_SHA = (
    "07ca5586a1cb354099362b8fc4568d4a1b0451a527b04b624df9c3dc5511ed3d"
)
ENGINEERING_SHA = "9087f4dfc5e97d2e95c454d2335de29ad872ea642d016acce6ec4792d396db28"
ENGINEERING_PROOF_SHA = (
    "64be239c56a1a915515a0f53f012ff8eb8a8bd68c7a00e2958fb888610b9dbf4"
)
CAPACITY_SHA = "0c2f5ae70efd97a08d0d97f8f66c3199bb70990a7bc39c89998db31f13d455ee"
STORAGE = {
    "new_model_cap_bytes": 15112385536,
    "adapter_cap_bytes": 80000000,
    "stage_requirements": {"convert": 95676664704, "quantize": 55450723328},
    "minimum_free_bytes": 32212254720,
}
EVIDENCE = (
    "training_configuration",
    "training_manifest",
    "fresh_reload",
    "protocol",
    "final_verification",
    "final_verifier",
    "canonical_export_receipt",
    "canonical_independent_verification",
    "engineering_probe",
    "engineering_independent_verification",
    "capacity_registration",
    "capacity_reconciliation",
)
FINAL_CHECKS = {
    "complete_remote_local_inventory_and_all_manifest_hashes",
    "all_eight_snapshots_and_frozen_training_inputs",
    "ordered_draws_and_actual_partial_accumulation",
    "source_bound_fresh_empty_adamw_and_final_step_all410",
    "actual_initial_tensor_hashes_exact_genuine_t1",
    "native_bf16_fp32_lora_provenance",
    "actual_final_adapter_moments_finite_and_changed_records_exact",
    "distinct_later_reload_exact_adapter_and_eight_raw_outputs",
    "all_controller_stages_and_exact_bound_hardware",
}


def source_paths():
    pending = [
        Path(__file__).resolve(),
        *reuse.source_paths(),
        *(Path(path).resolve() for _, _, path in trainer.SOURCE_MODULES),
    ]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
            for name in names:
                candidate = ROOT / "ml" / (name.split(".")[0] + ".py")
                if candidate.is_file() and candidate not in found:
                    pending.append(candidate)
    return sorted(found)


def adapter_headers(config, training, pins):
    base, adapter = (
        safe_path(config["base"]["path"]),
        safe_path(config["adapter"]["path"]),
    )
    native_pin = checked(config["native_pin"], ROOT, pins)
    checked(
        {"path": "native-api.json", "sha256": read(native_pin)["api_sha256"]},
        native_pin.parent,
        pins,
    )
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "generation_config.json",
    ):
        checked(
            {"path": name, "sha256": config["base"]["files_sha256"][name]}, base, pins
        )
    exporter.validate_native_pin(config)
    native_config = read(base / "config.json")
    targets = exporter.target_names(native_config)
    require(
        targets == {name + ".weight" for name in training["model"]["target_modules"]},
        "Native architecture differs from the actual 205 training targets.",
    )
    files = [
        safe_path(base / name)
        for name in config["base"]["files_sha256"]
        if name.endswith(".safetensors")
    ]
    files.append(safe_path(adapter / "adapter_model.safetensors"))
    before = {path: path.stat() for path in files}
    inventory = exporter.native_inventory(base, config["base"]["files_sha256"])
    tensors = exporter.tensor_header(files[-1])
    adapter_config = read(
        checked(
            {
                "path": "adapter_config.json",
                "sha256": config["adapter"]["files_sha256"]["adapter_config.json"],
            },
            adapter,
            pins,
        )
    )
    pairs = exporter.validate_adapter(
        adapter_config,
        inventory,
        tensors,
        native_config["text_config"]["num_hidden_layers"],
        exporter.BASE_REPOSITORY,
        trained_modules=training["model"]["target_modules"],
    )
    parameters = sum(math.prod(value["shape"]) for value in tensors.values())
    require(
        len(pairs) == 205
        and len(tensors) == 410
        and parameters == 18585600
        and all(value["dtype"] == "F32" for value in tensors.values())
        and adapter_config["r"] == adapter_config["lora_alpha"] == 16
        and adapter_config["lora_dropout"] == 0,
        "Actual adapter header/rank/dtype census differs from the fixed continuation.",
    )
    for path, previous_stat in before.items():
        observed = path.stat()
        require(
            all(
                getattr(previous_stat, key) == getattr(observed, key)
                for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            ),
            "Native or adapter file changed while reading its header.",
        )
    return {
        "status": "passed_metadata_and_actual_safetensors_headers",
        "targets": len(pairs),
        "adapter_tensors": len(tensors),
        "fp32_parameters": parameters,
        "native_tensor_headers": len(inventory),
        "full_native_payload_hashes_read": False,
        "full_payload_hash_gate": "Unchanged exporter preflight and executor pin checks before conversion.",
    }


def validate_proof(proof, config_sha):
    require(
        proof["version"] == "passive-actions-gemma-independent-phase-verification-v1"
        and proof["status"] == "passed_complete_phase_and_independent_reload"
        and proof["phase"] == "substantive"
        and proof["configuration_sha256"] == config_sha == CONFIG_SHA
        and proof["verifier_sha256"] == VERIFIER_SHA
        and proof["budget"]
        == {
            "draws": 42,
            "updates": 11,
            "final_rows": 2,
            "sequence_tokens": 56276,
            "completion_tokens": 598,
        }
        and set(proof["checks"]) == FINAL_CHECKS
        and all(value is True for value in proof["checks"].values()),
        "Require this genuine final 42/11 continuation proof, never a smoke or T2 proof.",
    )
    require(
        proof["initial_t1_adapter_tensors_exact"]
        == proof["finite_final_fp32_adapter_tensors"]
        == proof["finite_final_fp32_adamw_states"]
        == 410
        and proof["trainable_parameters"] == 18585600
        and 0 < proof["changed_final_vs_t1_tensors"] <= 410
        and proof["identical_raw_diagnostic_outputs"] == 8
        and proof["train_pid"] != proof["reload_pid"]
        and proof["new_training_updates"] == 0
        and proof["model_forward_performed"] is False,
        "Final tensor, fresh optimizer or independent reload proof is incomplete.",
    )


def proven_file(path, expected, proof, pins):
    path = safe_path(path)
    record = proof["bindings"].get(str(path))
    require(
        record is not None and record["sha256"] == expected,
        "Consumed final payload is absent from the independent proof.",
    )
    observed = path.stat()
    require(
        path.is_file() and observed.st_size == record["bytes"],
        "Verified payload size changed.",
    )
    verified_ns = int(datetime.fromisoformat(proof["verified_at"]).timestamp() * 1e9)
    require(0 < verified_ns <= time.time_ns(), "Invalid final proof chronology.")
    if observed.st_size <= 4 * 1024**2:
        checked({"path": str(path), "sha256": expected}, ROOT, pins)
    else:
        require(
            observed.st_mtime_ns <= verified_ns and observed.st_ctime_ns <= verified_ns,
            "Large verified payload changed; obtain a new independent readback.",
        )
        require(
            path not in pins or pins[path] == expected,
            "Conflicting large-payload binding.",
        )
        pins[path] = expected
    return path


def final_manifest(path, status, required, proof, pins):
    value = read(path)
    require(
        value["version"] == trainer.VERSION
        and value["status"] == status
        and required <= set(value["files_sha256"]),
        "Incomplete continuation manifest.",
    )
    paths = list(path.parent.rglob("*"))
    require(
        not any(p.is_symlink() for p in paths), "Final outputs cannot contain aliases."
    )
    require(
        {str(p.relative_to(path.parent)) for p in paths if p.is_file()}
        == set(value["files_sha256"]) | {"manifest.json"},
        "Final run inventory differs.",
    )
    for name, expected in value["files_sha256"].items():
        payload = safe_path(name, path.parent)
        require(
            payload.is_relative_to(path.parent), "Manifest payload escaped its run."
        )
        proven_file(payload, expected, proof, pins)
    proven_file(path, digest(path), proof, pins)
    return value


def validate_final(paths, pins):
    config_path = paths["training_configuration"]
    config, proof = read(config_path), read(paths["final_verification"])
    validate_proof(proof, digest(config_path))
    proven_file(config_path, CONFIG_SHA, proof, pins)
    require(
        digest(paths["final_verifier"]) == VERIFIER_SHA
        and digest(paths["protocol"]) == PROTOCOL_SHA,
        "Frozen verifier/protocol changed.",
    )
    spec = importlib.util.spec_from_file_location(
        "bound_actions_phase_verifier", paths["final_verifier"]
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    trainer.validate_configuration(config)
    require(
        config["phase"] == "substantive" and config["trainer_sha256"] == TRAINER_SHA,
        "Only the final all-action continuation is exportable.",
    )
    for _, key, module in trainer.SOURCE_MODULES:
        require(
            pins[Path(module).resolve()] == config[key],
            "Frozen training helper changed.",
        )
    inputs = {
        name: checked(value, config_path.parent, pins)
        for name, value in config["bindings"].items()
    }
    require(
        inputs["evaluation_protocol"] == paths["protocol"],
        "Protocol path differs from training.",
    )
    bundle = trainer.preparation.verify_bundle(inputs["preparation"].parent)
    rows, diagnostics = trainer.validate_prepared(config, bundle)
    trainer.validate_protocol(read(paths["protocol"]), config, bundle["configuration"])
    trainer.validate_configuration(config, read(inputs["anchor_configuration"]))
    trainer.validate_smoke_gate(config, config_path.parent)
    shared = {
        "run.json",
        "configuration.json",
        "runtime.json",
        "smoke-generation.json",
        *(name for name, _, _ in trainer.SOURCE_MODULES),
    }
    train = final_manifest(
        paths["training_manifest"],
        verifier.TRAIN_STATUS,
        shared
        | {
            "exposure.json",
            "initialization.json",
            "adapter-tensors.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "training-state.pt",
            "training-rows.jsonl",
            "training-events.jsonl",
        },
        proof,
        pins,
    )
    final_manifest(
        paths["fresh_reload"],
        verifier.RELOAD_STATUS,
        shared | {"reload-parity.json"},
        proof,
        pins,
    )
    train_root, reload_root = (
        paths["training_manifest"].parent,
        paths["fresh_reload"].parent,
    )
    require(
        train_root.parent == reload_root.parent
        and train_root.name == "train"
        and reload_root.name == "reload",
        "Use the same canonical final phase's train/reload pair.",
    )
    runs, runtimes = {}, {}
    for label, directory in (("train", train_root), ("reload", reload_root)):
        require(
            digest(directory / "configuration.json") == CONFIG_SHA,
            "Saved configuration changed.",
        )
        for name, key, _ in trainer.SOURCE_MODULES:
            require(
                digest(directory / name) == config[key],
                "Saved training source changed.",
            )
        runs[label], runtimes[label] = (
            read(directory / "run.json"),
            read(directory / "runtime.json"),
        )
        verifier.verify_run(
            runs[label], config, CONFIG_SHA, trainer, trained=label == "train"
        )
    require(
        runs["train"]["pid"] == proof["train_pid"]
        and runs["reload"]["pid"] == proof["reload_pid"]
        and runs["reload"]["started_unix"] > runs["train"]["finished_unix"]
        and runs["reload"]["previous_manifest_sha256"]
        == digest(paths["training_manifest"]),
        "Fresh reload does not bind the actual final continuation.",
    )
    exposure = read(train_root / "exposure.json")
    events = [
        json.loads(line)
        for line in (train_root / "training-rows.jsonl").read_text().splitlines()
    ]
    updates = [
        json.loads(line)
        for line in (train_root / "training-events.jsonl").read_text().splitlines()
    ]
    verifier.verify_exposure_records(exposure, events, updates, rows, "substantive")
    require(
        exposure == proof["exact_three_phase_exposure"],
        "Exposure differs from the independent proof.",
    )
    initial, records = (
        read(train_root / "initialization.json"),
        read(train_root / "adapter-tensors.json"),
    )
    anchor_records = read(inputs["anchor_adapter_tensors"])
    verifier.verify_initialization(
        initial,
        anchor_records,
        records,
        config["bindings"]["anchor_training_manifest"]["sha256"],
    )
    require(
        runtimes["train"]["initial_adapter_tensors"] == anchor_records
        and runtimes["reload"]["initial_adapter_tensors"] == records,
        "Loaded train/reload adapters differ from their exact tensor records.",
    )
    output_records = read(train_root / "smoke-generation.json")
    require(
        output_records == read(reload_root / "smoke-generation.json")
        and [v["case_id"] for v in output_records]
        == [r["case_id"] for r in diagnostics]
        and read(reload_root / "reload-parity.json")
        == {
            "adapter_tensors_identical": True,
            "optimizer_state_resumed": False,
            "cases": [
                {"case_id": row["case_id"], "identical": True} for row in diagnostics
            ],
        },
        "Eight native training diagnostics did not survive fresh reload exactly.",
    )
    for runtime in runtimes.values():
        require(
            runtime["targets"] == read(inputs["anchor_runtime"])["targets"]
            and runtime["numerical_regime"] == "native_bfloat16_lora_float32"
            and runtime["tokenizer_parity_rows"] == 42
            and runtime["completion_label_eos_token_id"] == 1
            and runtime["effective_generation"] == config["generation"]
            and not any(runtime["loading_info"].values()),
            "Native runtime boundary changed.",
        )
    return config, train, inputs


def validate_capacity(original, current):
    require(
        original["status"] == "passed_registered_fixed_future_allocations"
        and current["status"]
        == "passed_completed_rewrite_future_reservation_retired_once"
        and current["previous_registration_sha256"] == CAPACITY_SHA,
        "Unregistered capacity lineage.",
    )
    expected = dict(original["future_caps_bytes"])
    retired = expected.pop("old_rewrite_export")
    require(
        current["retired_newly_completed_rewrite_future_bytes"] == retired
        and current["remaining_future_caps_bytes"] == expected
        and current["remaining_future_required_bytes"] == sum(expected.values())
        and expected["new_all_action_export"] == STORAGE["new_model_cap_bytes"]
        and expected["minimum_free_reserve"] == STORAGE["minimum_free_bytes"]
        and current["new_smoke_or_input_allocation_credits"]
        == current["previous_json_or_training_credits_repeated"]
        == 0
        and current["next_export_order"]
        == [
            "new_all_action_after_final_reload_and_own_lineage_registration",
            "ordinary_qwen",
        ]
        and current["minimum_export_stage_bytes"]
        == {
            "conversion": STORAGE["stage_requirements"]["convert"],
            "quantize": STORAGE["stage_requirements"]["quantize"],
        },
        "Reserved allocations, unchanged stage gates or required export order differ.",
    )
    return current["remaining_future_required_bytes"]


def load_context(request_path, request_sha):
    pins = {}
    request_path = checked(
        {"path": str(request_path), "sha256": request_sha}, ROOT, pins
    )
    request = read(request_path)
    require(
        request["version"] == VERSION
        and request["mode"] == MODE
        and request["method"] == reuse.METHOD
        and request["limits"] == reuse.LIMITS
        and request["storage"] == STORAGE
        and request["registration_status"]
        in ("draft_pending_review", "registered_before_execution"),
        "Register the new continuation boundary with unchanged execution/storage limits.",
    )
    require(
        set(request["sources_sha256"])
        == {str(p.relative_to(ROOT)) for p in source_paths()},
        "Bind the complete planner/executor source set.",
    )
    for name, expected in request["sources_sha256"].items():
        checked({"path": name, "sha256": expected}, ROOT, pins)
    require(
        digest(Path(reuse.__file__)) == EXECUTOR_SHA
        and digest(Path(previous.__file__)) == reuse.PREVIOUS_SHA256
        and digest(Path(trainer.__file__)) == TRAINER_SHA,
        "Frozen executor or trainer changed.",
    )
    for name, expected in previous.FROZEN.items():
        require(
            pins[safe_path(name)] == expected,
            "Frozen numerical export/clone helper changed.",
        )
    require(
        set(request["bindings"]) == set(EVIDENCE),
        "Require every actual scientific/storage proof.",
    )
    paths = {
        name: checked(record, request_path.parent, pins)
        for name, record in request["bindings"].items()
    }
    for key, expected in (
        ("final_verification", FINAL_PROOF_SHA),
        ("canonical_export_receipt", CANONICAL_SHA),
        ("engineering_probe", ENGINEERING_SHA),
        ("engineering_independent_verification", ENGINEERING_PROOF_SHA),
        ("capacity_registration", CAPACITY_SHA),
    ):
        require(
            pins[paths[key]] == expected, "Historical proof identity changed: " + key
        )
    training, train, inputs = validate_final(paths, pins)
    canonical_receipt = read(paths["canonical_export_receipt"])
    canonical_config_path = checked(
        {
            "path": "configuration.json",
            "sha256": canonical_receipt["configuration_sha256"],
        },
        paths["canonical_export_receipt"].parent,
        pins,
    )
    require(
        pins[canonical_config_path] == CANONICAL_CONFIG_SHA,
        "Use the receipted normalized canonical runtime.",
    )
    canonical_config = read(canonical_config_path)
    require(
        canonical_config["base"]["files_sha256"] == training["base"]["files_sha256"]
        and canonical_config["native_pin"]["sha256"]
        == training["bindings"]["native_pin"]["sha256"],
        "Training/export native assets differ.",
    )
    for group in ("canonical_base_conversion_logs", "canonical_base_quantization_logs"):
        require(
            set(request[group]) == {"command.json", "result.json", "log"},
            "Bind actual reused base commands/logs/results.",
        )
        for record in request[group].values():
            checked(record, request_path.parent, pins)
    canonical = reuse.canonical_evidence(
        paths["canonical_export_receipt"], canonical_config, pins
    )
    for group, command in (
        ("canonical_base_conversion_logs", "convert-base"),
        ("canonical_base_quantization_logs", "quantize-base"),
    ):
        require(
            all(
                safe_path(request[group][suffix]["path"], request_path.parent)
                == Path(canonical["executions"][command][suffix]["path"])
                for suffix in request[group]
            ),
            "Wrong canonical command history.",
        )
    independent = read(paths["canonical_independent_verification"])
    require(
        independent["status"] == "passed"
        and independent["bindings"]["export_receipt"] == canonical["receipt"]
        and independent["checks"]
        and all(value is True for value in independent["checks"].values()),
        "Canonical independent export proof failed.",
    )
    reuse.validate_engineering_result(paths["engineering_probe"], canonical, pins)
    engineering_policy_path = checked(
        read(paths["engineering_probe"])["policy"],
        paths["engineering_probe"].parent,
        pins,
    )
    for name, expected in read(engineering_policy_path)["sources_sha256"].items():
        require(
            pins.get(safe_path(name)) == expected,
            "An engineering-qualified execution dependency changed.",
        )
    engineering = read(paths["engineering_independent_verification"])
    require(
        engineering["status"] == "passed"
        and engineering["checks"]
        and all(v is True for v in engineering["checks"].values())
        and engineering["bindings_sha256"][
            str(paths["engineering_probe"].relative_to(ROOT))
        ]
        == ENGINEERING_SHA,
        "Engineering independent full-byte proof does not cover the original executor result.",
    )
    remaining = validate_capacity(
        read(paths["capacity_registration"]), read(paths["capacity_reconciliation"])
    )
    require(
        reuse.stage_requirements(canonical["templates"])
        == STORAGE["stage_requirements"],
        "Exact full-output-plus-scratch space gates changed.",
    )
    protected = {
        paths["training_configuration"].parent,
        paths["training_manifest"].parent,
        paths["fresh_reload"].parent,
        inputs["preparation"].parent,
        paths["canonical_export_receipt"].parent,
        safe_path(canonical_config["base"]["path"]),
        safe_path(canonical_config["runtime"]["source_path"]),
        safe_path(canonical_config["native_pin"]["path"]).parent,
    }
    return (
        request,
        paths,
        training,
        train,
        canonical_config,
        canonical,
        pins,
        protected,
        remaining,
    )


def build_configuration(canonical, training_manifest, train, final_proof):
    config = copy.deepcopy(canonical)
    directory = training_manifest.parent
    config["adapter"] = {
        "path": str(directory / "adapter"),
        "files_sha256": {
            name.removeprefix("adapter/"): expected
            for name, expected in train["files_sha256"].items()
            if name.startswith("adapter/")
        },
    }
    config["training_record"] = {
        "path": str(directory / "run.json"),
        "sha256": train["files_sha256"]["run.json"],
    }
    config["continuation"] = {
        "contract_version": trainer.CONTRACT_VERSION,
        "training_version": trainer.VERSION,
        "initialization": trainer.INITIALIZATION,
        "optimizer_state_resumed": False,
        "training_updates": 11,
        "committed_draws": 42,
        "final_accumulation": 2,
        "final_verification": final_proof,
    }
    return config


def fresh_output(path, protected, pins):
    path = safe_path(path)
    require(
        path.is_relative_to(ROOT / "artifacts/training")
        and path.parent.is_dir()
        and not path.exists()
        and not any(path.is_relative_to(p) for p in protected)
        and not any(p.is_relative_to(path) for p in pins),
        "Use a fresh owned output outside every immutable input.",
    )
    return path


def commands(config, config_path, output):
    result = {
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
        for stage in reuse.STAGES
    }
    for stage in ("convert", "quantize"):
        result[stage] = [
            config["runtime"]["python"],
            "-u",
            "-B",
            str(Path(reuse.__file__).resolve()),
            "worker",
            "--plan",
            str(output / "plan.json"),
            "--stage",
            stage,
        ]
    return result


def load_plan(config_path, policy_path, policy_sha, output):
    requested_config = safe_path(config_path)
    policy_pins = {}
    policy_path = checked(
        {"path": str(policy_path), "sha256": policy_sha}, ROOT, policy_pins
    )
    policy = read(policy_path)
    require(
        policy["version"] == VERSION and policy["mode"] == MODE,
        "Wrong continuation policy.",
    )
    request_path = checked(policy["request"], policy_path.parent, policy_pins)
    (
        request,
        paths,
        training,
        train,
        canonical_config,
        canonical,
        pins,
        protected,
        remaining,
    ) = load_context(request_path, digest(request_path))
    pins.update(policy_pins)
    config_path = checked(policy["export_configuration"], policy_path.parent, pins)
    require(
        requested_config == config_path,
        "Export configuration argument differs from its policy.",
    )
    config = exporter.configuration(config_path)
    require(
        config
        == build_configuration(
            canonical_config,
            paths["training_manifest"],
            train,
            binding(paths["final_verification"]),
        ),
        "Export configuration must bind only the verified final continuation and normalized canonical runtime.",
    )
    header_validation = adapter_headers(config, training, pins)
    require(
        policy["registration_status"] == request["registration_status"]
        and policy["output"] == str(safe_path(output)),
        "Policy output/registration changed.",
    )
    for tool in config["runtime"]["tools"].values():
        checked(tool, ROOT, pins)
    for name, expected in config["runtime"]["support_files_sha256"].items():
        checked({"path": name, "sha256": expected}, ROOT, pins)
    for name, expected in config["runtime"]["source_files_sha256"].items():
        checked(
            {"path": name, "sha256": expected},
            Path(config["runtime"]["source_path"]),
            pins,
        )
    output = fresh_output(
        output, protected | {policy_path.parent, config_path.parent}, pins
    )
    free = shutil.disk_usage(output.parent).free
    require(
        free >= max(remaining, STORAGE["stage_requirements"]["convert"]),
        "Current free space fails the retained-allocation and full conversion gates.",
    )
    return {
        "version": reuse.VERSION,
        "method": reuse.METHOD,
        "mode": MODE,
        "registration_status": request["registration_status"],
        "status": "plan_only_no_export_executed",
        "configuration": binding(config_path),
        "policy": binding(policy_path),
        "output": str(output),
        "canonical": canonical,
        "templates": canonical["templates"],
        "limits": reuse.LIMITS,
        "commands": commands(config, config_path, output),
        "space_required_before_stages": STORAGE["stage_requirements"],
        "free_bytes_observed": free,
        "remaining_future_required_bytes": remaining,
        "pins": {str(path): expected for path, expected in pins.items()},
        "implementation_sha256": EXECUTOR_SHA,
        "planner_version": VERSION,
        "planner_sha256": digest(Path(__file__)),
        "storage": STORAGE,
        "scientific_validation": {
            "planner": binding(Path(__file__).resolve()),
            "request": binding(request_path),
            "final_verification": binding(paths["final_verification"]),
            "final_verifier": binding(paths["final_verifier"]),
            "training_configuration": binding(paths["training_configuration"]),
            "training_manifest": binding(paths["training_manifest"]),
            "fresh_reload": binding(paths["fresh_reload"]),
            "protocol": binding(paths["protocol"]),
            "engineering_probe": binding(paths["engineering_probe"]),
            "engineering_independent_verification": binding(
                paths["engineering_independent_verification"]
            ),
            "capacity_registration": binding(paths["capacity_registration"]),
            "capacity_reconciliation": binding(paths["capacity_reconciliation"]),
            "contract_version": trainer.CONTRACT_VERSION,
            "initialization": trainer.INITIALIZATION,
            "optimizer_state_resumed": False,
            "training_updates": 11,
            "committed_draws": 42,
            "final_accumulation": 2,
            "anchor_adapter_sha256": training["anchor"]["files_sha256"][
                "adapter_model.safetensors"
            ],
            "adapter_header_validation": header_validation,
            "large_final_payload_hashes": "Independent proof plus unchanged size/timestamps at planning; original executor rehashes every pinned payload before stages.",
            "local_prefix_generation_required_after_export": True,
        },
    }


def assemble(request_path, request_sha, registration, output):
    context = load_context(request_path, request_sha)
    request, paths, _, train, canonical_config, _, pins, protected, _ = context
    registration = fresh_output(registration, protected, pins)
    output = fresh_output(output, protected | {registration}, pins)
    require(
        not registration.is_relative_to(output),
        "Registration cannot be inside model output.",
    )
    registration.mkdir(exist_ok=False)
    receipt = {
        "version": VERSION,
        "status": "assembly_started",
        "new_training_updates": 0,
        "model_operations": 0,
    }
    try:
        config_path = registration / "export-configuration.json"
        write_new(
            config_path,
            build_configuration(
                canonical_config,
                paths["training_manifest"],
                train,
                binding(paths["final_verification"]),
            ),
        )
        policy_path = registration / "policy.json"
        write_new(
            policy_path,
            {
                "version": VERSION,
                "mode": MODE,
                "registration_status": request["registration_status"],
                "request": binding(safe_path(request_path)),
                "export_configuration": binding(config_path),
                "output": str(output),
            },
        )
        plan = load_plan(config_path, policy_path, digest(policy_path), output)
        write_new(registration / "plan.json", plan)
        receipt.update(
            status="assembled_metadata_only_export_unrun",
            configuration=binding(config_path),
            policy=binding(policy_path),
            plan=binding(registration / "plan.json"),
            planner=binding(Path(__file__).resolve()),
        )
    except BaseException as error:
        receipt.update(
            status="failed_partial_assembly_preserved",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        write_new(registration / "assembly.json", receipt)
    return receipt


def retained_bytes(result, output):
    require(
        result["status"] == "paired_export_passed_local_evaluation_pending"
        and result["exit_code"] == 0
        and result["mode"] == MODE
        and result["implementation_sha256"] == EXECUTOR_SHA,
        "Original executor did not finish the registered continuation export.",
    )
    receipt_path = safe_path(result["export_receipt"]["path"])
    require(
        receipt_path == output / "export/receipt.json"
        and digest(receipt_path) == result["export_receipt"]["sha256"],
        "Final export receipt changed.",
    )
    receipt = read(receipt_path)
    total = receipt["artifacts"]["adapter"]["bytes"]
    require(
        total <= STORAGE["adapter_cap_bytes"],
        "Converted adapter exceeded the80MB allowance.",
    )
    require(
        [v["artifact"] for v in result["compactions"]]
        == ["merged", "merged_quantized"],
        "Missing owned compaction proofs.",
    )
    for item in result["compactions"]:
        record = item["reconstruction"]
        path = safe_path(record["path"])
        require(
            path == output / "compaction" / f"{item['artifact']}.reconstruction.json"
            and digest(path) == record["sha256"],
            "Compaction proof changed.",
        )
        value = read(path)
        cap = reuse.LIMITS["max_changed_bytes"][item["artifact"]]
        require(
            value["status"] == "exact_bytes_verified"
            and value["max_changed_bytes"] == cap
            and 0 <= value["changed_chunk_bytes"] <= cap
            and value["minimum_free_bytes"] == STORAGE["minimum_free_bytes"],
            "Changed-chunk or reserve cap failed.",
        )
        total += value["changed_chunk_bytes"]
    require(
        total <= STORAGE["new_model_cap_bytes"],
        "New-model retained write budget exceeded.",
    )
    return total, receipt_path


def execute(plan):
    require(
        plan["mode"] == MODE
        and plan["planner_sha256"] == digest(Path(__file__))
        and plan["storage"] == STORAGE
        and plan["limits"] == reuse.LIMITS
        and plan["space_required_before_stages"] == STORAGE["stage_requirements"]
        and plan["implementation_sha256"]
        == digest(Path(reuse.__file__))
        == EXECUTOR_SHA,
        "Registered continuation planner/executor identity changed.",
    )
    output = safe_path(plan["output"])
    require(
        shutil.disk_usage(output.parent).free
        >= max(
            plan["remaining_future_required_bytes"],
            STORAGE["stage_requirements"]["convert"],
        ),
        "Current shared-capacity gate failed before the original executor.",
    )
    result = reuse.execute(plan)
    final = {
        "version": VERSION,
        "status": "failed",
        "mode": MODE,
        "model_inference_requests": 0,
        "bindings": {
            "configuration": plan["configuration"],
            "policy": plan["policy"],
            "final_verification": plan["scientific_validation"]["final_verification"],
            "plan": binding(output / "plan.json"),
            "executor_result": binding(output / "result.json"),
            "planner": binding(Path(__file__).resolve()),
        },
        "scientific_validation": plan["scientific_validation"],
        "storage": STORAGE,
    }
    try:
        total, receipt_path = retained_bytes(result, output)
        final["bindings"]["receipt"] = binding(receipt_path)
        require(
            shutil.disk_usage(output).free >= STORAGE["minimum_free_bytes"],
            "Final free-space reserve failed.",
        )
        final.update(
            status="passed_actions_continuation_export_local_prefix_pending",
            retained_changed_chunk_plus_adapter_bytes=total,
            allocation_scope="Explicit changed-chunk writes plus fresh adapter; not inferred APFS physical allocation.",
        )
    except BaseException as error:
        final.update(error_type=type(error).__name__, error=str(error))
    final["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_new(output / "actions-export-result.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("assemble")
    command.add_argument("--request", type=Path, required=True)
    command.add_argument("--request-sha256", required=True)
    command.add_argument("--registration", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    for name in ("plan", "execute"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--policy-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "assemble":
        result = assemble(
            args.request, args.request_sha256, args.registration, args.output
        )
    else:
        result = load_plan(args.config, args.policy, args.policy_sha256, args.output)
        if args.command == "execute":
            result = execute(result)
    print(json.dumps(result, indent=2))
    return int(result.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
