"""Evaluate the fixed native-derived Gemma SFT pair on exposed B2 development.

Frozen B2 sources supply request construction, prefix parity, execution grading and
one-server lifecycle. Historical vendor QAT results remain external observations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import math
from pathlib import Path
from types import SimpleNamespace

import passive_b2_benchmark as b2
import probe_passive as probe
import probe_passive_b2 as original
import probe_passive_curriculum as shared
import train_passive_gemma as trainer
from export_passive_gemma import (
    BASE_REPOSITORY,
    BASE_REVISION,
    GENERATION_STOP_IDS,
    RUNTIME_REVISION,
)
from probe_passive_curriculum import (
    ARTIFACT_KEYS,
    B2_BINDINGS,
    CONDITIONS,
    IDENTITY_FIELDS,
    correct_preserves,
    learning_signal,
    load_json,
    native_eos,
    new_output,
    project_path,
    read_bound,
    require,
    source_execution,
    validate_results,
    validate_runtime_manifest,
    verify_inputs,
    write_new,
)
import json

ROOT = probe.ROOT
VERSION = "passive-gemma-evaluation-v1"
EXECUTION_VERSION = "passive-gemma-execution-v1"
SOURCE_MODULES = (
    ("trainer.py", "trainer_sha256"),
    ("train_passive_sft.py", "smoke_trainer_sha256"),
    ("train_passive_curriculum.py", "curriculum_trainer_sha256"),
    ("prepare_passive_gemma.py", "preparer_sha256"),
    ("prepare_passive_curriculum.py", "curriculum_preparer_sha256"),
    ("prepare_passive_sft.py", "smoke_preparer_sha256"),
)


def load_protocol(protocol_path):
    protocol_path = project_path(protocol_path)
    inputs = {protocol_path: probe.digest(protocol_path)}
    protocol = load_json(protocol_path)
    require(
        protocol.get("version") == VERSION
        and protocol.get("conditions") == list(CONDITIONS)
        and protocol.get("registered_requests") == 289
        and protocol.get("icl_eligible_requests") == 100
        and protocol.get("icl_excluded_requests") == 189
        and protocol.get("candidate_id") == "gemma26"
        and protocol.get("final_test_bodies_read") is False,
        "Register the unchanged full289 / common-ICL100 cohort and all four conditions.",
    )
    paths = {
        key: read_bound(protocol["b2"][key], protocol_path.parent, inputs)
        for key in B2_BINDINGS
    }
    paths["protocol_new"] = protocol_path
    old = load_json(paths["protocol"])
    require(
        old["release_manifest_sha256"] == inputs[paths["release_manifest"]],
        "The B2 release binding changed.",
    )
    entries = protocol.get("external_comparators", [])
    required = {
        ("qwen35-9b", "full"),
        ("qwen35-9b", "icl"),
        ("gemma26", "full"),
        ("qwen27", "full"),
        ("always_no_action", "full"),
        ("consistent_history_rule", "full"),
    }
    require(
        len({row["id"] for row in entries}) == len(entries)
        and not {row["id"] for row in entries} & set(CONDITIONS)
        and required <= {(row["candidate"], row["condition"]) for row in entries}
        and all(row["condition"] in ("full", "icl") for row in entries),
        "Bind distinct external B2 observations, including 9B full/ICL and both fixed baselines.",
    )
    for entry in entries:
        for key in ("manifest", "results"):
            read_bound(entry[key], protocol_path.parent, inputs)
    return {"protocol": protocol, "paths": paths, "inputs": inputs}


def load_registration(protocol_path, execution_path):
    registration = load_protocol(protocol_path)
    execution_path = project_path(execution_path)
    paths, inputs = registration["paths"], registration["inputs"]
    inputs[execution_path] = probe.digest(execution_path)
    execution = load_json(execution_path)
    require(
        execution.get("version") == EXECUTION_VERSION
        and execution.get("protocol_sha256") == inputs[paths["protocol_new"]],
        "Execution does not bind this Gemma evaluation protocol.",
    )
    paths["execution_new"] = execution_path
    for key in ("registry", "export_receipt", "training_configuration", "fresh_reload"):
        paths[f"new_{key}"] = read_bound(execution[key], execution_path.parent, inputs)
    registration["execution"] = execution
    return registration


def anchor_args(registration, condition):
    paths = registration["paths"]
    return SimpleNamespace(
        release=paths["release_manifest"].parent,
        registry=paths["registry"],
        protocol=paths["protocol"],
        execution=paths["execution"],
        prefix_manifest=paths["prefix_manifest"],
        candidate="qwen35-9b" if condition == "icl" else "gemma26",
        condition=condition,
        reuse_full_run=None,
        unconstrained=False,
        sampling_config=None,
        diagnostic_protocol=None,
        timeout=original.POLICY["request_timeout_seconds"],
    )


def prepare_inputs(registration, condition):
    args = anchor_args(registration, condition)
    # The frozen B2 ICL runner accepts only its original anchor. Keep that full
    # validation, then bind Gemma's independently registered prefixes below.
    prepared = original.prepare(args)
    manifest = load_json(args.prefix_manifest)
    inputs = registration["inputs"]
    eligibility_path = read_bound(
        {
            "path": manifest["common_icl_eligibility_file"],
            "sha256": manifest["common_icl_eligibility_sha256"],
        },
        args.prefix_manifest.parent,
        inputs,
    )
    eligibility = load_json(eligibility_path)
    ids = prepared["binding"]["registered_case_ids"]
    prefixes = {}
    gemma_eligible = []
    for view in {condition, "icl"}:
        name = f"gemma26/{view}.jsonl.gz"
        path = read_bound(
            {"path": name, "sha256": manifest["files_sha256"][name]},
            args.prefix_manifest.parent,
            inputs,
        )
        rows = original.unique_rows(
            original.read_rows(path), ids, "Gemma frozen prefixes"
        )
        counts = {identity: len(row["input_ids"]) for identity, row in rows.items()}
        require(
            counts == manifest["prefix_counts"]["gemma26"][view],
            "Gemma measured prefix counts differ from the preregistration.",
        )
        if view == "icl":
            gemma_eligible = [
                identity
                for identity in ids
                if counts[identity] <= original.POLICY["input_tokens"]
            ]
        if view == condition:
            require(
                all(
                    row["messages_sha256"]
                    == prepared["binding"]["messages_sha256"][identity]
                    for identity, row in rows.items()
                ),
                "Gemma prefix belongs to different frozen messages.",
            )
            prefixes = {identity: row["input_ids"] for identity, row in rows.items()}
    require(
        len(ids) == 289
        and len(gemma_eligible) == 100
        and gemma_eligible == eligibility["eligible_case_ids"]
        and len(eligibility["excluded"]) == 189,
        "Gemma's own ICL eligibility must equal the fixed common100 with189 exclusions.",
    )
    if condition == "icl":
        require(
            [case["id"] for case in prepared["cases"]] == gemma_eligible,
            "ICL cases differ from Gemma's independent fixed eligibility.",
        )
    prepared["prefixes"] = prefixes
    return args, prepared, eligibility


def validate_training(registration, training_path, training):
    paths, inputs = registration["paths"], registration["inputs"]
    config_path = paths["new_training_configuration"]
    config = load_json(config_path)
    protocol_sha = inputs[paths["protocol_new"]]
    trainer_path = Path(trainer.__file__).resolve()
    require(
        probe.digest(trainer_path) == config["trainer_sha256"],
        "The Gemma validator must match the trained source snapshot.",
    )
    inputs[trainer_path] = config["trainer_sha256"]
    trainer.validate_configuration(config)
    trainer.validate_protocol(
        registration["protocol"], config, inputs[paths["release_manifest"]]
    )
    require(
        training.get("version") == config.get("version") == "passive-gemma-sft-v1"
        and training.get("phase") == config.get("phase") == "substantive"
        and training.get("status") == "training_complete_fresh_reload_pending"
        and training.get("mode") == "train"
        and training.get("training_updates") == 203
        and training.get("optimizer_committed_rows") == 809
        and training.get("configuration_sha256") == inputs[config_path]
        and training.get("evaluation_protocol_sha256") == protocol_sha
        and training.get("initialization") == "original_native_base_with_fresh_lora"
        and training.get("optimizer_state_resumed") is False
        and training.get("final_test_bodies_read") == 0,
        "Evaluate only this protocol's final 203-update / 809-committed-draw curriculum run.",
    )
    plan, recipe = registration["protocol"]["training_plan"], config["training"]
    require(
        plan["candidate"] == config["candidate_id"] == "gemma26"
        and plan["view"] == "json-curriculum"
        and plan["updates"] == recipe["max_updates"] == 203
        and plan["draw_count"]
        == recipe["draw_count"]
        == len(recipe["sample_order"])
        == 809
        and plan["sampling_seed"] == recipe["sampling_seed"] == "3407"
        and plan["model_seed"] == recipe["seed"] == 3407
        and recipe["fresh_native_initialization"] is True
        and plan["checkpoint_selection"] == "final_update_203_only"
        and plan["sequence_tokens"] == recipe["max_total_sequence_tokens"]
        and plan["completion_tokens"] == recipe["max_total_completion_tokens"]
        and config["base"]["repository"] == BASE_REPOSITORY
        and config["base"]["revision"] == BASE_REVISION,
        "The exported training recipe is outside the preregistered final-only experiment.",
    )
    bound_protocol = read_bound(
        config["bindings"]["evaluation_protocol"], config_path.parent, inputs
    )
    require(
        inputs[bound_protocol] == protocol_sha,
        "Training used another evaluation protocol.",
    )
    preparation_path = read_bound(
        config["bindings"]["preparation"], config_path.parent, inputs
    )
    preparation = load_json(preparation_path)
    require(
        preparation.get("version") == "passive-sft-gemma-preparation-v1"
        and preparation.get("status") == "frozen_train_only_preparation"
        and preparation.get("release_manifest_sha256")
        == inputs[paths["release_manifest"]]
        and training.get("preparation_sha256") == inputs[preparation_path],
        "Training preparation differs from this frozen train-only release.",
    )
    for module, key in (
        (trainer.preparation, "preparer_sha256"),
        (trainer.curriculum_preparation, "curriculum_preparer_sha256"),
        (trainer.smoke_preparation, "smoke_preparer_sha256"),
    ):
        preparer_path = Path(module.__file__).resolve()
        require(
            probe.digest(preparer_path) == config[key],
            "The Gemma preparation verifier differs from the trained source snapshot.",
        )
        inputs[preparer_path] = config[key]
    bundle = trainer.preparation.verify_bundle(preparation_path.parent)
    readback = bundle["verification"]
    require(
        bundle["manifest"] == preparation
        and readback.get("status") == "passed_train_only_gemma_readback"
        and readback.get("train_rows") == readback.get("draws") == 809
        and readback.get("optimizer_updates") == 203
        and readback.get("sequence_tokens") == recipe["max_total_sequence_tokens"]
        and readback.get("completion_tokens") == recipe["max_total_completion_tokens"]
        and readback.get("dev_test_record_bodies_read") == 0
        and readback.get("training_updates_executed") == 0,
        "Gemma train-only preparation readback did not pass the registered scope and counts.",
    )
    trainer.validate_prepared(config, bundle)
    for name, expected in preparation["files_sha256"].items():
        read_bound({"path": name, "sha256": expected}, preparation_path.parent, inputs)
    selection_path = read_bound(
        config["bindings"]["selection"], config_path.parent, inputs
    )
    selection = load_json(selection_path)
    require(
        selection["selected_t1_candidate_ids"] == ["gemma26"]
        and selection["bindings"]["train_only_preparation"]["sha256"]
        == inputs[preparation_path]
        and selection["bindings"]["release_manifest_sha256"]
        == inputs[paths["release_manifest"]],
        "Training lost the explicit Gemma priority selection and preparation binding.",
    )
    for name in trainer.SMOKE_BINDINGS:
        read_bound(config["bindings"][name], config_path.parent, inputs)
    trainer.validate_smoke_gate(
        config, config_path.parent, inputs[paths["release_manifest"]]
    )
    native_pin = read_bound(
        config["bindings"]["native_pin"], config_path.parent, inputs
    )
    require(
        training["native_pin_sha256"] == inputs[native_pin],
        "Training run lost its native acquisition pin.",
    )
    pin = load_json(native_pin)
    require(
        pin["repository"] == BASE_REPOSITORY
        and pin["revision"] == BASE_REVISION
        and {name: value["sha256"] for name, value in pin["files"].items()}
        == config["base"]["files_sha256"],
        "Training native inventory differs from the pinned export origin.",
    )
    reload_manifest_path = paths["new_fresh_reload"]
    reload_manifest = load_json(reload_manifest_path)
    require(
        reload_manifest.get("version") == "passive-gemma-sft-v1"
        and reload_manifest.get("status") == "fresh_reload_passed_local_export_pending",
        "A successful fresh reload is required before development evaluation.",
    )

    def run_file(directory, manifest, name):
        return read_bound(
            {"path": name, "sha256": manifest["files_sha256"][name]}, directory, inputs
        )

    reload_run = load_json(
        run_file(reload_manifest_path.parent, reload_manifest, "run.json")
    )
    training_manifest_path = read_bound(
        {"path": "manifest.json", "sha256": reload_run["previous_manifest_sha256"]},
        training_path.parent,
        inputs,
    )
    training_manifest = load_json(training_manifest_path)
    require(
        training_manifest.get("version") == "passive-gemma-sft-v1"
        and training_manifest.get("status") == "training_complete_fresh_reload_pending"
        and run_file(training_path.parent, training_manifest, "run.json")
        == training_path
        and reload_run.get("status") == "fresh_reload_passed_local_export_pending"
        and reload_run.get("mode") == "fresh_reload"
        and reload_run.get("phase") == "substantive"
        and reload_run.get("configuration_sha256") == inputs[config_path]
        and reload_run.get("evaluation_protocol_sha256") == protocol_sha
        and reload_run.get("reloaded_training_updates") == 203
        and reload_run.get("native_pin_sha256") == inputs[native_pin]
        and reload_run.get("pid") != training.get("pid"),
        "Fresh reload does not refer to the same completed training/configuration in a separate process.",
    )
    require(
        all(
            type(record.get("peak_allocated_gpu_bytes")) is int
            and 0
            < record["peak_allocated_gpu_bytes"]
            <= record["peak_reserved_gpu_bytes"]
            for record in (training, reload_run)
        ),
        "Training and fresh reload must retain measured GPU memory observations.",
    )
    for directory, manifest in (
        (training_path.parent, training_manifest),
        (reload_manifest_path.parent, reload_manifest),
    ):
        require(
            load_json(run_file(directory, manifest, "configuration.json")) == config,
            "Saved training/reload configuration differs from the bound input.",
        )
        for name, key in SOURCE_MODULES:
            source = run_file(directory, manifest, name)
            require(
                inputs[source] == config[key],
                "Training/reload source snapshot changed.",
            )
        runtime = load_json(run_file(directory, manifest, "runtime.json"))
        base_counts = runtime["base_parameter_census"]["parameter_elements_by_dtype"]
        require(
            runtime["numerical_regime"] == "native_bfloat16_lora_float32"
            and runtime["parameter_dtypes"] == ["torch.bfloat16"]
            and runtime["ignored_native_tensor_names"] == []
            and sorted(runtime["targets"]) == config["model"]["target_modules"]
            and runtime["effective_generation"] == config["generation"]
            and runtime["native_generation_config"]["eos_token_id"]
            == GENERATION_STOP_IDS
            and runtime["completion_label_eos_token_id"] == 1
            and config["model"]["gpu_name_contains"] in runtime["gpu"]
            and runtime["gpu_total_bytes"] >= config["model"]["minimum_gpu_bytes"],
            "Runtime lost the exact Gemma target, GPU, precision or native termination census.",
        )
        require(
            set(base_counts) == {"torch.bfloat16"}
            and base_counts["torch.bfloat16"] > 0,
            "The frozen native base must remain BF16.",
        )
        if directory == training_path.parent:
            require(
                runtime["after_adapter_parameter_census"]["parameter_elements_by_dtype"]
                == {**base_counts, "torch.float32": 18585600},
                "The trained model must retain BF16 base and exactly 18,585,600 FP32 adapter elements.",
            )
    initialization = load_json(
        run_file(training_path.parent, training_manifest, "initialization.json")
    )
    require(
        initialization["initialization"] == "original_native_base_with_fresh_lora"
        and initialization["smoke_adapter_loaded"] is False
        and len(initialization["trainable_tensors"]) == 410
        and initialization["trainable_parameters"] == 18585600,
        "Substantive training must start from fresh LoRA, never the engineering adapter.",
    )
    exposure = load_json(
        run_file(training_path.parent, training_manifest, "exposure.json")
    )
    committed = exposure["phases"]["optimizer_committed"]
    require(
        exposure["completed_updates"] == 203
        and committed["rows"] == 809
        and exposure["optimizer_commit_uncertain"] is False
        and exposure["pending_backward_rows_not_committed"] == 0
        and committed["counts"]["case_id"] == dict(Counter(recipe["sample_order"]))
        and committed["sequence_tokens"] == recipe["max_total_sequence_tokens"]
        and committed["completion_tokens"] == recipe["max_total_completion_tokens"],
        "Observed optimizer-committed exposure differs from the fixed final draw stream.",
    )
    parity = load_json(
        run_file(reload_manifest_path.parent, reload_manifest, "reload-parity.json")
    )
    require(
        parity.get("adapter_tensors_identical") is True
        and len(parity["cases"])
        == len({row["case_id"] for row in parity["cases"]})
        == 8
        and all(row["identical"] is True for row in parity["cases"]),
        "Fresh reload lacks exact adapter and eight-output parity.",
    )
    before = run_file(training_path.parent, training_manifest, "smoke-generation.json")
    after = run_file(
        reload_manifest_path.parent, reload_manifest, "smoke-generation.json"
    )
    require(
        load_json(before) == load_json(after)
        and [row["case_id"] for row in load_json(before)]
        == [row["case_id"] for row in parity["cases"]],
        "Fresh reload generation evidence differs from the saved training outputs.",
    )
    require(
        all(
            isinstance(row["text_raw"], str)
            and isinstance(row["output_ids"], list)
            and row["terminal_token_id"]
            == (row["output_ids"][-1] if row["output_ids"] else None)
            for row in load_json(before)
        ),
        "Native reload diagnostics must retain raw token IDs and the terminal token.",
    )
    return config


def validate_pair(registration):
    paths, inputs = registration["paths"], registration["inputs"]
    registry = load_json(paths["new_registry"])
    previous = load_json(paths["registry"])
    anchor = next(row for row in previous["candidates"] if row["id"] == "gemma26")
    candidates = registration["execution"]["candidate_ids"]
    require(
        set(candidates) == set(ARTIFACT_KEYS)
        and len(set(candidates.values())) == 2
        and not set(candidates.values()) & set(probe.BASELINES),
        "Bind two distinct native_base / sft candidate IDs.",
    )
    models = {row["id"]: row for row in registry["candidates"]}
    require(
        len(registry["candidates"]) == 2 and set(models) == set(candidates.values()),
        "Use a new registry containing exactly the two exported candidates.",
    )
    require(
        registry["runtime"] == previous["runtime"]
        and registry["tokenizers"] == previous["tokenizers"]
        and probe.sampling_options(paths["new_registry"])
        == probe.sampling_options(paths["registry"]),
        "Preserve B2's exact runtime, tokenizer registry and greedy sampling.",
    )
    receipt_path = paths["new_export_receipt"]
    receipt = load_json(receipt_path)
    require(
        receipt.get("version") == "passive-gemma-sft-export-v1"
        and receipt.get("status")
        == "tensor_verified_paired_export_runtime_preflight_pending"
        and receipt.get("base_repository") == BASE_REPOSITORY
        and receipt.get("base_revision") == BASE_REVISION
        and receipt.get("runtime_revision") == RUNTIME_REVISION
        and registry["runtime"]["revision"] == RUNTIME_REVISION,
        "Use the exact native Gemma / b10816 verified paired export receipt.",
    )
    tensor_path = read_bound(
        {
            "path": "tensor-verification.json",
            "sha256": receipt["tensor_verification_sha256"],
        },
        receipt_path.parent,
        inputs,
    )
    tensor = load_json(tensor_path)
    require(
        tensor.get("status") == "passed"
        and tensor.get("template_sha256") == anchor["template_sha256"],
        "Export tensor/native-template verification differs.",
    )
    quantization_path = read_bound(
        {"path": "quantization.json", "sha256": receipt["quantization_sha256"]},
        receipt_path.parent,
        inputs,
    )
    quantization = load_json(quantization_path)
    require(
        quantization["quantization"] == "Q4_K_M"
        and quantization["importance_matrix"] is None
        and quantization["threads"] == 4
        and set(quantization["commands"]) == {"base", "merged"}
        and all(
            set(files) == {"command.json", "result.json", "log"}
            for files in quantization["commands"].values()
        )
        and quantization["quantizer_fallbacks"] == receipt["quantizer_fallbacks"]
        and all(
            quantization["artifacts"][key] == receipt["artifacts"][key]
            for key in ARTIFACT_KEYS.values()
        ),
        "Paired quantization recipe, fallback evidence or artifacts changed.",
    )
    for files in quantization["commands"].values():
        for binding in files.values():
            read_bound(binding, quantization_path.parent, inputs)
    training_path = read_bound(receipt["training_record"], receipt_path.parent, inputs)
    training = load_json(training_path)
    if "configuration_sha256" not in training:
        training_path = read_bound(
            {"path": "run.json", "sha256": training["files_sha256"]["run.json"]},
            training_path.parent,
            inputs,
        )
        training = load_json(training_path)
    training_config = validate_training(registration, training_path, training)
    require(
        tensor.get("eos_token_id") == native_eos(anchor, inputs) == 1
        and tensor.get("native_generation_stop_ids")
        == receipt.get("native_generation_stop_ids")
        == training_config["generation"]["eos_token_id"]
        == GENERATION_STOP_IDS
        and receipt["native_pin"]["sha256"]
        == training_config["bindings"]["native_pin"]["sha256"],
        "Export EOS, native terminal IDs or acquisition pin differs from training.",
    )
    read_bound(receipt["native_pin"], receipt_path.parent, inputs)
    for role, identity in candidates.items():
        model = models[identity]
        require(
            model.get("tokenizer_id") == "gemma26"
            and all(model.get(key) == anchor[key] for key in IDENTITY_FIELDS),
            "An exported candidate changed the native tokenizer/template identity.",
        )
        artifact = receipt["artifacts"][ARTIFACT_KEYS[role]]
        expected_path = project_path(artifact["path"], receipt_path.parent)
        types = receipt["quantized_tensor_type_census"][ARTIFACT_KEYS[role]]
        require(
            project_path(model["model_path"]) == expected_path
            and model["model_sha256"] == artifact["sha256"]
            and expected_path.stat().st_size == artifact["bytes"]
            and types
            and all(type(count) is int and count > 0 for count in types.values()),
            "Candidate weights are not the corresponding member of this export pair.",
        )
        # The original runtime inspector rehashes the model, server, libraries and native files.
        probe.inspect_runtime(paths["new_registry"], identity)
    registration.update(registry=registry, export=receipt, training=training)


def prepare(protocol_path, execution_path, condition):
    require(condition in CONDITIONS, "Unknown registered Gemma condition.")
    registration = load_registration(protocol_path, execution_path)
    role, original_condition = CONDITIONS[condition]
    args, prepared, eligibility = prepare_inputs(registration, original_condition)
    validate_pair(registration)
    require(
        len(prepared["cases"]) == (100 if original_condition == "icl" else 289),
        "Prepared condition has a different request denominator.",
    )
    for source in (
        Path(__file__).resolve(),
        Path(shared.__file__).resolve(),
        Path(trainer.__file__).resolve(),
    ):
        prepared["source_files"].append(source)
        prepared["binding"]["snapshot_sha256"][source.name] = probe.digest(source)
        registration["inputs"][source] = probe.digest(source)
    binding = {
        "version": VERSION,
        "condition": condition,
        "model_role": role,
        "protocol_sha256": probe.digest(registration["paths"]["protocol_new"]),
        "execution_sha256": probe.digest(registration["paths"]["execution_new"]),
        "registry_sha256": probe.digest(registration["paths"]["new_registry"]),
        "export_receipt_sha256": probe.digest(
            registration["paths"]["new_export_receipt"]
        ),
        "training_configuration_sha256": probe.digest(
            registration["paths"]["new_training_configuration"]
        ),
        "fresh_reload_manifest_sha256": probe.digest(
            registration["paths"]["new_fresh_reload"]
        ),
        "paired_baseline": "native-derived exact export pair; public B2 GGUF is external only",
        "registered_requests": 289,
        "requested_cases": len(prepared["cases"]),
        "icl_eligible_requests": 100,
        "icl_excluded_requests": 189,
        "native_generation_stop_ids": GENERATION_STOP_IDS,
        "icl_eligibility_basis": "Gemma's own frozen prefixes independently equal B2 common100.",
    }
    prepared["binding"]["gemma"] = binding
    registration.update(
        prepared=prepared, binding=binding, eligibility=eligibility, args=args
    )
    verify_inputs(registration["inputs"])
    return registration


def snapshot_registration(registration, output):
    implementation = output / "implementation"
    implementation.mkdir(exist_ok=True)
    for key, name in (
        ("protocol_new", "gemma-protocol.json"),
        ("execution_new", "gemma-execution.json"),
        ("new_export_receipt", "gemma-export-receipt.json"),
        ("new_training_configuration", "gemma-training-configuration.json"),
        ("new_fresh_reload", "gemma-reload-manifest.json"),
    ):
        path = registration["paths"][key]
        require(
            probe.digest(path) == registration["inputs"][path],
            "Registration changed before snapshot.",
        )
        with (implementation / name).open("xb") as stream:
            stream.write(path.read_bytes())


def run_condition(registration, condition, output, stage, editor_state):
    args, prepared = registration["args"], registration["prepared"]
    output = new_output(output, args.release)
    if stage == "prepare":
        output.mkdir(parents=True)
        snapshot_registration(registration, output)
        write_new(
            output / "gemma-receipt.json",
            {
                **registration["binding"],
                "status": "prepared_generation_unrun",
                "case_ids": [case["id"] for case in prepared["cases"]],
                "common_icl_eligibility": registration["eligibility"],
            },
        )
        return
    args.candidate = registration["execution"]["candidate_ids"][
        CONDITIONS[condition][0]
    ]
    args.registry = registration["paths"]["new_registry"]
    args.output, args.stage, args.editor_state = output, stage, editor_state
    error = None
    try:
        probe.run(args, prepared=prepared)
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    if output.exists():
        try:
            verify_inputs(registration["inputs"])
            snapshot_registration(registration, output)
            results_path = output / "results.jsonl"
            rows = original.read_rows(results_path) if results_path.exists() else []
            validate_results(prepared["cases"], rows, prepared["prefixes"])
            with (output / "source-execution.jsonl").open("x") as stream:
                for value in source_execution(prepared["cases"], rows):
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            status = load_json(output / "manifest.json")["status"]
        except (Exception, KeyboardInterrupt) as caught:
            error = error or caught
            status = "failed"
        files = {
            path.name: probe.digest(path) for path in output.iterdir() if path.is_file()
        }
        write_new(
            output / "gemma-receipt.json",
            {
                **registration["binding"],
                "status": "failed" if error else status,
                "error": f"{type(error).__name__}: {error}" if error else None,
                "files_sha256": files,
                "runtime_manifest_version": "passive-b2-runtime-v1 (unchanged shared runner)",
            },
        )
    if error:
        raise error


def quality_metrics(report):
    if isinstance(report, dict):
        for value in list(report.values()):
            quality_metrics(value)
        if "displayed_useful_precision" in report and "useful_edit_recall" in report:
            precision, recall = (
                report["displayed_useful_precision"],
                report["useful_edit_recall"],
            )
            report["derived_useful_edit_f1"] = (
                None
                if recall is None
                else 0.0
                if recall == 0
                else 2 * precision * recall / (precision + recall)
                if precision is not None
                else None
            )
            report["legacy_grading_inclusive_latency"] = {
                key: report.pop(key)
                for key in (
                    "all_request_p50_ms",
                    "all_request_p95_ms",
                    "latency_samples",
                    "missing_latency_samples",
                )
                if key in report
            }
    elif isinstance(report, list):
        for value in report:
            quality_metrics(value)
    return report


def http_latency(rows, expected_requests, *, historical=False):
    measured = [
        row["http_response_ms"]
        for row in rows
        if not row.get("reused_from")
        and type(row.get("http_response_ms")) in (int, float)
        and math.isfinite(row["http_response_ms"])
        and row["http_response_ms"] >= 0
    ]
    return {
        "p50_ms": b2.b1._percentile(measured, 0.5),
        "p95_ms": b2.b1._percentile(measured, 0.95),
        "measured_requests": len(measured),
        "expected_requests": expected_requests,
        "unmeasured_requests": expected_requests - len(measured),
        "scope": "Historical HTTP observations; not newly paired latency."
        if historical
        else "Fresh request elapsed through HTTP response or transport error, before grading; all measured failures included. No missing-time imputation.",
    }


def resource_observations(manifest):
    return {
        key: manifest.get(key)
        for key in (
            "cold_load_ms",
            "sampled_peak_server_rss_bytes",
            "rss_samples",
            "memory_bytes",
            "swap_before",
            "swap_after",
            "vm_stat_before",
            "vm_stat_after",
            "editor_state",
        )
    }


def historical_results(registration, full, icl):
    paths = registration["paths"]
    entries = registration["protocol"].get("external_comparators", [])
    require(
        len({row["id"] for row in entries}) == len(entries)
        and not {row["id"] for row in entries} & set(CONDITIONS),
        "External comparator IDs must be unique and separate from the paired runs.",
    )
    required = {
        ("qwen35-9b", "full"),
        ("qwen35-9b", "icl"),
        ("always_no_action", "full"),
        ("consistent_history_rule", "full"),
    }
    require(
        required <= {(row["candidate"], row["condition"]) for row in entries},
        "Bind original B2 9B full/ICL and both fixed deterministic baselines.",
    )
    reports, provenance = {}, {}
    for entry in entries:
        require(
            entry["condition"] in ("full", "icl"),
            "Only full/ICL external observations are registered.",
        )
        prepared = full if entry["condition"] == "full" else icl
        manifest_path = read_bound(
            entry["manifest"], paths["protocol_new"].parent, registration["inputs"]
        )
        results_path = read_bound(
            entry["results"], paths["protocol_new"].parent, registration["inputs"]
        )
        require(
            results_path.parent == manifest_path.parent,
            "External manifest and results must belong to one run.",
        )
        manifest = load_json(manifest_path)
        binding = manifest.get("benchmark", {})
        validate_runtime_manifest(
            manifest, registration, prepared, entry["candidate"], paths["registry"]
        )
        require(
            manifest.get("version") == "passive-b2-runtime-v1"
            and manifest.get("status") == "completed"
            and manifest.get("candidate") == entry["candidate"]
            and manifest.get("condition") == entry["condition"]
            and manifest.get("registry_sha256") == probe.digest(paths["registry"])
            and manifest.get("case_ids") == [case["id"] for case in prepared["cases"]]
            and all(
                binding.get(key) == prepared["binding"][key]
                for key in (
                    "protocol_sha256",
                    "execution_sha256",
                    "prefix_manifest_sha256",
                    "registry_sha256",
                    "release_manifest_sha256",
                    "case_registry_sha256",
                    "requests_sha256",
                    "registered_case_ids",
                    "messages_sha256",
                    "common_icl_eligibility_sha256",
                )
            ),
            "External results do not bind the same frozen B2 condition.",
        )
        for name, expected in binding["snapshot_sha256"].items():
            read_bound(
                {"path": name, "sha256": expected},
                manifest_path.parent / "implementation",
                registration["inputs"],
            )
        read_bound(
            {"path": "model-registry.json", "sha256": binding["registry_sha256"]},
            manifest_path.parent / "implementation",
            registration["inputs"],
        )
        rows = original.read_rows(results_path)
        original.unique_rows(
            rows, manifest["case_ids"], "Completed B2 external results"
        )
        require(
            manifest.get("completed_requests") == len(rows),
            "External completion count differs.",
        )
        deterministic = entry["candidate"] in probe.BASELINES
        # Other B2 models have different token counts; their saved native checks bind those prefixes.
        if deterministic or entry["candidate"] == "gemma26":
            prefixes = prepared["prefixes"]
        else:
            tokens = original.read_rows(
                paths["prefix_manifest"].parent
                / entry["candidate"]
                / f"{entry['condition']}.jsonl.gz"
            )
            prefixes = {row["case_id"]: row["input_ids"] for row in tokens}
        validate_results(prepared["cases"], rows, prefixes, deterministic=deterministic)
        reports[entry["id"]] = quality_metrics(
            prepared["summarize"](prepared["cases"], rows)
        )
        reports[entry["id"]]["historical_http_response_ms_p50_p95"] = http_latency(
            rows, len(prepared["cases"]), historical=True
        )
        reports[entry["id"]]["actual_memory_and_swap"] = resource_observations(manifest)
        provenance[entry["id"]] = {
            **entry,
            "comparison_role": "external B2 observation; not the native-derived paired control",
        }
    return reports, provenance


def report(protocol_path, execution_path, runs, output):
    registration = prepare(protocol_path, execution_path, "native-base-full")
    full = registration["prepared"]
    _, icl, _ = prepare_inputs(registration, "icl")
    runs = project_path(runs)
    output = new_output(output, registration["args"].release)
    summaries, results, provenance = {}, {}, {}
    for condition, (role, view) in CONDITIONS.items():
        prepared = full if view == "full" else icl
        directory = runs / condition
        receipt_path, manifest_path = (
            directory / "gemma-receipt.json",
            directory / "manifest.json",
        )
        rows = []
        manifest = {}
        status = "missing_run"
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            expected = {
                **registration["binding"],
                "condition": condition,
                "model_role": role,
                "requested_cases": len(prepared["cases"]),
            }
            require(
                all(receipt.get(key) == value for key, value in expected.items()),
                "Run receipt belongs to another protocol/export pair/condition.",
            )
            require(
                "files_sha256" in receipt,
                "A preparation-only receipt is not a completed evaluation run.",
            )
            for name, expected_hash in receipt["files_sha256"].items():
                read_bound(
                    {"path": name, "sha256": expected_hash},
                    directory,
                    registration["inputs"],
                )
            manifest = load_json(manifest_path)
            validate_runtime_manifest(
                manifest,
                registration,
                prepared,
                registration["execution"]["candidate_ids"][role],
                registration["paths"]["new_registry"],
            )
            require(
                manifest.get("benchmark", {}).get("gemma") == expected
                and manifest.get("candidate")
                == registration["execution"]["candidate_ids"][role]
                and manifest.get("case_ids")
                == [case["id"] for case in prepared["cases"]]
                and manifest.get("stage") == "screen",
                "Saved runtime condition differs from the registered evaluation.",
            )
            for name, expected_hash in manifest["benchmark"]["snapshot_sha256"].items():
                read_bound(
                    {"path": name, "sha256": expected_hash},
                    directory / "implementation",
                    registration["inputs"],
                )
            read_bound(
                {
                    "path": "model-registry.json",
                    "sha256": registration["binding"]["registry_sha256"],
                },
                directory / "implementation",
                registration["inputs"],
            )
            results_path = directory / "results.jsonl"
            if results_path.exists():
                require(
                    "results.jsonl" in receipt["files_sha256"], "Unbound result file."
                )
                rows = original.read_rows(results_path)
            require(
                manifest.get("completed_requests") == len(rows),
                "Recorded completion count differs.",
            )
            status = receipt["status"]
            provenance[condition] = {
                "receipt_sha256": probe.digest(receipt_path),
                "status": status,
            }
        elif directory.exists():
            raise ValueError(f"Incomplete run has no immutable receipt: {directory}")
        validate_results(prepared["cases"], rows, prepared["prefixes"])
        summaries[condition] = quality_metrics(
            prepared["summarize"](prepared["cases"], rows)
        )
        summaries[condition]["fresh_http_response_ms_p50_p95"] = http_latency(
            rows, len(prepared["cases"])
        )
        summaries[condition]["actual_memory_and_swap"] = resource_observations(manifest)
        summaries[condition]["run_status"] = status
        summaries[condition]["correct_preserves"] = correct_preserves(
            prepared["cases"], rows
        )
        results[condition] = rows
    external, external_provenance = historical_results(registration, full, icl)
    pairs = {}
    for view in ("full", "icl"):
        left, right = f"native-base-{view}", f"sft-{view}"
        eligibility = registration["eligibility"] if view == "icl" else None
        pairs[view] = {
            "learning_signal": learning_signal(summaries[left], summaries[right]),
            "source_paired_comparison": quality_metrics(
                b2.compare_runs(
                    full["cases"],
                    {left: results[left], right: results[right]},
                    eligibility_by_name={left: eligibility, right: eligibility},
                )
            ),
        }
    eligible = set(registration["eligibility"]["eligible_case_ids"])
    icl_effects = {}
    for role in ("native-base", "sft"):
        names = (f"{role}-full", f"{role}-icl")
        icl_effects[role] = quality_metrics(
            b2.compare_runs(
                full["cases"],
                {
                    name: [row for row in results[name] if row["case_id"] in eligible]
                    for name in names
                },
                eligibility_by_name={
                    name: registration["eligibility"] for name in names
                },
            )
        )
        icl_effects[role]["fresh_http_response_ms_p50_p95"] = {
            name: http_latency(
                [row for row in results[name] if row["case_id"] in eligible], 100
            )
            for name in names
        }
    verify_inputs(registration["inputs"])
    output.mkdir(parents=True)
    snapshot_registration(registration, output)
    write_new(
        output / "summary.json",
        {
            "version": VERSION,
            "protocol_sha256": registration["binding"]["protocol_sha256"],
            "execution_sha256": registration["binding"]["execution_sha256"],
            "registered_new_requests": 778,
            "registered_conditions": list(CONDITIONS),
            "runs": summaries,
            "paired": pairs,
            "icl_effect_on_common100": icl_effects,
            "icl_comparison_scope": "Existing full/ICL outputs compared only on the common100; both keep189 exclusions in the289 registration. No additional requests.",
            "external_comparators": external,
            "provenance": provenance,
            "external_provenance": external_provenance,
            "common_icl_eligibility": registration["eligibility"],
            "paired_export": {
                "artifacts": {
                    role: registration["export"]["artifacts"][key]
                    for role, key in ARTIFACT_KEYS.items()
                },
                "quantized_tensor_type_census": registration["export"][
                    "quantized_tensor_type_census"
                ],
                "quantizer_fallbacks": registration["export"]["quantizer_fallbacks"],
                "limitations": registration["export"]["limitations"],
            },
            "f1_definition": "Harmonic mean of reference-known displayed useful precision and useful-edit recall. Zero recall yields zero F1 even if no edits are shown; no edit-needed references yields undefined F1.",
            "timing_scope": "Fresh HTTP measurements exclude frozen source grading; sequential runs have uncontrolled thermal/resource state. Historical vendor QAT and B2 timing is external, not a new paired latency experiment.",
            "termination_scope": "Pinned b10816 recognizes native terminal IDs 1,106,50. Native training/reload retains raw token IDs; local OpenAI responses retain their raw payload and finish reason and do not expose terminal token IDs.",
            "scope": "Frozen exposed development; correlated synthetic/source states; no final-test access or deployment promotion.",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "report"), default="run")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "preflight", "screen"), default="screen"
    )
    parser.add_argument("--runs", type=Path)
    parser.add_argument(
        "--editor-state", default="Not controlled; no editor-preview latency claim"
    )
    args = parser.parse_args(argv)
    if args.command == "report":
        require(
            args.runs is not None and args.condition is None,
            "Report requires --runs, without --condition.",
        )
        report(args.protocol, args.execution, args.runs, args.output)
        return
    require(
        args.condition is not None and args.runs is None,
        "Run requires --condition, without --runs.",
    )
    lock_path = ROOT / "artifacts/nep-passive-b1.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another passive benchmark owns the local predictor."
            ) from error
        registration = prepare(args.protocol, args.execution, args.condition)
        run_condition(
            registration, args.condition, args.output, args.stage, args.editor_state
        )


if __name__ == "__main__":
    main()
