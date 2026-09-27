"""Evaluate final DPO on dev289, reusing the exact registered T1 SFT observations.

Only DPO generates new requests. Training/profile/reference/reload evidence and the
existing SFT result receipt must all bind the fixed T1 JSON anchor before serving.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace

import passive_b2_benchmark as b2
import prepare_passive_preferences as preferences
import probe_passive as probe
import probe_passive_b2 as original
import probe_passive_curriculum as curriculum
import train_passive_dpo as trainer
from export_passive_sft import BASE_REPOSITORY, BASE_REVISION, RUNTIME_REVISION
from probe_passive_curriculum import (
    load_json,
    project_path,
    read_bound,
    require,
    verify_inputs,
    write_new,
)

ROOT = probe.ROOT
VERSION = "passive-dpo-evaluation-v1"
EXECUTION_VERSION = "passive-dpo-execution-v1"
TRAIN_COMPLETE = "training_complete_fresh_reload_pending"
RELOAD_COMPLETE = "fresh_reload_passed_local_export_pending"
DPO_BINDINGS = (
    "configuration",
    "export_receipt",
    "reference_cache",
    "profile",
    "profile_reload",
    "fresh_reload",
)
SOURCES = (
    ("trainer.py", "trainer_sha256", trainer.__file__),
    ("prepare_passive_preferences.py", "preparer_sha256", preferences.__file__),
    (
        "train_passive_curriculum.py",
        "curriculum_trainer_sha256",
        trainer.curriculum.__file__,
    ),
    ("train_passive_sft.py", "smoke_trainer_sha256", trainer.smoke.__file__),
    (
        "prepare_passive_curriculum.py",
        "curriculum_preparer_sha256",
        trainer.curriculum.preparation.__file__,
    ),
    (
        "prepare_passive_sft.py",
        "smoke_preparer_sha256",
        trainer.curriculum.smoke_preparation.__file__,
    ),
)
TIMING_SCOPE = (
    "SFT timing/memory is reused from its original T1 run; DPO is observed later. "
    "These are descriptive runtime observations, not a fresh paired latency experiment."
)


def load_protocol(path):
    path = project_path(path)
    value = load_json(path)
    require(
        value.get("version") == VERSION
        and value["anchor"]["candidate"] == "qwen35-9b"
        and value["anchor"]["sft_updates"] == 203
        and value["anchor"]["sft_draws"] == 809
        and value["evaluation"]["conditions"] == ["sft-full", "dpo-full"]
        and value["evaluation"]["requests_per_condition"] == 289
        and value["evaluation"]["json_schema_constrained"] is True
        and value["evaluation"]["same_runtime_tokenizer_prompt_and_output_budget"]
        is True
        and value["learning_signal_policy"]
        == {
            "no_auto_extra_training": True,
            "deployment_promotion": False,
            "report_all_final_outcomes": True,
            "negative_result_completes_experiment": True,
        }
        and value["final_test_bodies_read"] is False,
        "Keep the fixed final T1 anchor, both full289 conditions and all final outcomes.",
    )
    return {
        "protocol_path": path,
        "protocol": value,
        "inputs": {path: probe.digest(path)},
    }


def bound_json(binding, parent, inputs):
    path = read_bound(binding, parent, inputs)
    return path, load_json(path)


def inventory(path, inputs):
    manifest = load_json(path)
    files = {}
    for name, expected in manifest["files_sha256"].items():
        child = project_path(name, path.parent)
        require(
            child.is_relative_to(path.parent) and child != path,
            "Manifest payload must stay inside its own evidence directory.",
        )
        files[name] = read_bound(
            {"path": name, "sha256": expected}, path.parent, inputs
        )
    return {"path": path, "manifest": manifest, "files": files}


def record(evidence, name):
    return load_json(evidence["files"][name])


def validate_results(cases, rows, prefixes):
    curriculum.validate_results(cases, rows, prefixes)
    for row in rows:
        if failed(row):
            continue
        tokens = row["response"]["usage"]["completion_tokens"]
        require(
            type(tokens) is int and 0 <= tokens <= 512,
            "A raw response exceeded the registered 512-token output budget.",
        )


def failed(row):
    return (
        row.get("request_failed")
        or row.get("cancelled")
        or row.get("error_type") in original.REQUEST_ERRORS
    )


def saved_results(registration, receipt_path, expected, prepared, candidate, registry):
    evidence = inventory(receipt_path, registration["inputs"])
    receipt = evidence["manifest"]
    require(
        all(receipt.get(key) == value for key, value in expected.items())
        and receipt.get("status") in ("completed", "failed"),
        "Use the exact registered screen receipt, including its failures; no reselection.",
    )
    manifest = record(evidence, "manifest.json")
    curriculum.validate_runtime_manifest(
        manifest, registration, prepared, candidate, registry
    )
    require(
        manifest["benchmark"] == prepared["binding"]
        and manifest["status"] in ("completed", "failed"),
        "Saved results changed the frozen prompt, source grader or condition.",
    )
    for name, expected_hash in manifest["benchmark"]["snapshot_sha256"].items():
        read_bound(
            {"path": name, "sha256": expected_hash},
            receipt_path.parent / "implementation",
            registration["inputs"],
        )
    read_bound(
        {"path": "model-registry.json", "sha256": probe.digest(registry)},
        receipt_path.parent / "implementation",
        registration["inputs"],
    )
    rows = (
        original.read_rows(evidence["files"]["results.jsonl"])
        if "results.jsonl" in evidence["files"]
        else []
    )
    require(
        manifest["completed_requests"] == len(rows)
        and (manifest["status"] != "completed" or len(rows) == len(prepared["cases"]))
        and (receipt["status"] != "completed" or manifest["status"] == "completed"),
        "Saved completion count/status differs from the registered run.",
    )
    require(
        not (receipt_path.parent / "results.jsonl").exists()
        or "results.jsonl" in evidence["files"],
        "Unbound results appeared beside the receipt.",
    )
    validate_results(prepared["cases"], rows, prepared["prefixes"])
    return {
        "rows": rows,
        "status": receipt["status"],
        "receipt_sha256": probe.digest(receipt_path),
        "sampled_peak_server_rss_bytes": manifest.get("sampled_peak_server_rss_bytes"),
        "timing_scope": TIMING_SCOPE,
    }


def load_stage(path, config, common, inputs, status):
    evidence = inventory(path, inputs)
    run = record(evidence, "run.json")
    require(
        evidence["manifest"].get("version") == run.get("version") == trainer.VERSION
        and evidence["manifest"].get("status") == run.get("status") == status
        and all(run.get(key) == value for key, value in common.items())
        and run.get("actual_tokenizer_parity") is True
        and run.get("final_test_bodies_read") == 0
        and record(evidence, "configuration.json") == config,
        "DPO stage evidence belongs to another configuration/anchor or is incomplete.",
    )
    for name, key, source in SOURCES:
        require(
            inputs[evidence["files"][name]]
            == config[key]
            == probe.digest(Path(source)),
            "DPO stage source snapshot differs from its bound verifier/trainer.",
        )
        inputs[Path(source).resolve()] = config[key]
    evidence["run"] = run
    return evidence


def validate_reference(evidence, pairs, anchor_tensors):
    run = evidence["run"]
    require(
        run.get("mode") == "reference_precompute"
        and run.get("training_updates") == run.get("reference_updates") == 0
        and run.get("reference_adapter_unchanged") is True
        and run.get("completed_pairs") == len(pairs)
        and run.get("reference_forward_attempts")
        == run.get("reference_forward_completions")
        == 2 * len(pairs)
        and run.get("reference_forward_sequence_tokens")
        == sum(len(pair[side]["input_ids"]) for pair in pairs for side in trainer.SIDES)
        and run.get("reference_forward_completion_tokens")
        == sum(
            len(pair[side]["input_ids"]) - pair[side]["completion_start"]
            for pair in pairs
            for side in trainer.SIDES
        )
        and run.get("reference_source") == "exact_final_T1_adapter"
        and run.get("log_probability") == "summed_completion_including_eos"
        and record(evidence, "runtime.json")["initial_adapter_tensors"]
        == anchor_tensors,
        "Reference must score the unchanged T1 adapter before any optimizer update.",
    )
    rows = original.read_rows(evidence["files"]["reference-logps.jsonl"])
    require(
        [row["pair_id"] for row in rows] == [pair["pair_id"] for pair in pairs],
        "Reference cache must preserve every qualified pair in its fixed order.",
    )
    for pair, row in zip(pairs, rows, strict=True):
        require(row["case_id"] == pair["case_id"], "Reference case identity changed.")
        for side in trainer.SIDES:
            value = row[side + "_logprob"]
            require(
                type(value) in (int, float)
                and math.isfinite(value)
                and value <= 0
                and row[side + "_completion_tokens"]
                == len(pair[side]["input_ids"]) - pair[side]["completion_start"],
                "Reference score is invalid or omits completion/EOS tokens.",
            )


def validate_final_run(run, pairs, phase):
    require(
        run.get("mode") == "train"
        and run.get("status") == TRAIN_COMPLETE
        and run.get("experiment_phase") == phase
        and run.get("training_updates") == math.ceil(len(pairs) / 4)
        and run.get("optimizer_committed_pairs") == run.get("pair_count") == len(pairs)
        and run.get("pair_order") == [pair["pair_id"] for pair in pairs]
        and run.get("initialization") == "exact_final_T1_adapter"
        and run.get("reference_source") == "exact_final_T1_adapter"
        and run.get("optimizer_state_resumed") is False
        and run.get("actual_tokenizer_parity") is True
        and bool(run.get("changed_adapter_tensors")),
        "Use only this phase's final exact pair stream, initialized from T1 without resume.",
    )


def validate_exposure(evidence, pairs):
    expected = trainer.Exposure()
    sides = {
        "chosen_scored": "chosen",
        "rejected_forward": "rejected",
        "rejected_backward": "rejected",
        "chosen_forward": "chosen",
        "chosen_backward": "chosen",
    }
    for pair in pairs:
        for phase in trainer.PHASES:
            if phase in sides:
                expected.attempted_compute_calls[phase] += 1
            expected.record(phase, pair, sides.get(phase))
    expected.updates = math.ceil(len(pairs) / 4)
    require(
        record(evidence, "exposure.json") == expected.summary(),
        "DPO committed exposure differs from one exact pair pass / three forwards / two backwards.",
    )
    events = original.read_rows(evidence["files"]["training-events.jsonl"])
    require(len(events) == expected.updates, "DPO update census differs.")
    for index, event in enumerate(events):
        group = pairs[4 * index : 4 * (index + 1)]
        require(
            event["update"] == index + 1
            and event["pair_ids"] == [pair["pair_id"] for pair in group]
            and event["microbatches"] == len(group)
            and event["committed_pairs"] == min(4 * (index + 1), len(pairs))
            and len(event["pair_losses"]) == len(group)
            and all(math.isfinite(value) for value in event["pair_losses"])
            and math.isfinite(event["mean_pair_loss"])
            and math.isfinite(event["gradient_norm_before_clip"])
            and event["gradient_norm_before_clip"] >= 0
            and event["learning_rate"] == 1e-5,
            "DPO update order, finite loss or final partial accumulation differs.",
        )


def validate_reload(training, reloaded, diagnostic_ids):
    before, after = training["run"], reloaded["run"]
    parity = record(reloaded, "reload-parity.json")
    generated = record(training, "smoke-generation.json")
    require(
        after.get("mode") == "fresh_reload"
        and after.get("experiment_phase") == before["experiment_phase"]
        and after.get("previous_manifest_sha256") == probe.digest(training["path"])
        and after.get("reference_cache_sha256") == before["reference_cache_sha256"]
        and after.get("reloaded_training_updates") == before["training_updates"]
        and after.get("optimizer_state_resumed") is False
        and type(after.get("pid")) is int
        and after["pid"] != before["pid"]
        and parity.get("adapter_tensors_identical") is True
        and parity.get("optimizer_state_resumed") is False
        and [row["case_id"] for row in parity["cases"]] == diagnostic_ids
        and all(row["identical"] is True for row in parity["cases"])
        and [row["case_id"] for row in generated] == diagnostic_ids
        and generated == record(reloaded, "smoke-generation.json")
        and record(reloaded, "runtime.json")["initial_adapter_tensors"]
        == record(training, "adapter-tensors.json"),
        "Fresh reload must bind exact final tensors and all eight unchanged diagnostics.",
    )


def validate_training(registration):
    inputs, paths, t1 = (
        registration["inputs"],
        registration["paths"],
        registration["t1"],
    )
    config_path = paths["dpo_configuration"]
    config = load_json(config_path)
    bound = {
        name: read_bound(value, config_path.parent, inputs)
        for name, value in config["bindings"].items()
    }
    anchor_config = load_json(t1["paths"]["new_training_configuration"])
    anchor_reload = inventory(bound["anchor_reload_manifest"], inputs)
    anchor = inventory(bound["anchor_training_manifest"], inputs)
    require(
        inputs[bound["evaluation_protocol"]] == inputs[registration["protocol_path"]]
        and inputs[bound["anchor_configuration"]]
        == t1["binding"]["training_configuration_sha256"]
        == registration["protocol"]["anchor"]["sft_configuration_sha256"]
        and load_json(bound["anchor_configuration"]) == anchor_config
        and inputs[bound["anchor_reload_manifest"]]
        == t1["binding"]["fresh_reload_manifest_sha256"]
        and record(anchor, "run.json") == t1["training"]
        and record(anchor_reload, "run.json")["previous_manifest_sha256"]
        == inputs[bound["anchor_training_manifest"]]
        and inputs[bound["anchor_preparation"]]
        == anchor_config["bindings"]["preparation"]["sha256"],
        "DPO must continue exactly the fixed T1 training/reload/configuration used by SFT dev.",
    )
    for name in ("preparation", "anchor_preparation", "candidate_generation"):
        inventory(bound[name], inputs)
    bundle = preferences.verify_bundle(
        bound["preparation"].parent,
        bound["anchor_preparation"].parent,
        bound["candidate_generation"].parent,
    )
    pairs, prepared = bundle["pairs"], bundle["configuration"]
    trainer.validate_configuration(config, anchor_config, prepared, pairs)
    trainer.validate_protocol(registration["protocol"], config, t1["training"])
    require(
        all(
            prepared[key] == inputs[bound[name]]
            for name, key in (
                ("anchor_configuration", "anchor_configuration_sha256"),
                ("anchor_training_manifest", "anchor_training_manifest_sha256"),
                ("anchor_reload_manifest", "anchor_reload_manifest_sha256"),
            )
        )
        and bundle["manifest"]["bindings"]["release_manifest_sha256"]
        == t1["prepared"]["binding"]["release_manifest_sha256"],
        "Qualified preference pairs belong to another anchor or frozen release.",
    )
    common = {
        "configuration_sha256": inputs[config_path],
        "trainer_sha256": config["trainer_sha256"],
        "preparation_sha256": inputs[bound["preparation"]],
        "evaluation_protocol_sha256": inputs[registration["protocol_path"]],
        "anchor_configuration_sha256": inputs[bound["anchor_configuration"]],
        "anchor_training_manifest_sha256": inputs[bound["anchor_training_manifest"]],
        "anchor_reload_manifest_sha256": inputs[bound["anchor_reload_manifest"]],
        "anchor_adapter_sha256": anchor["manifest"]["files_sha256"][
            "adapter/adapter_model.safetensors"
        ],
        "candidate_manifest_sha256": inputs[bound["candidate_generation"]],
        "release_manifest_sha256": t1["prepared"]["binding"]["release_manifest_sha256"],
        "preference_origin": "AI",
    }
    export = load_json(paths["dpo_export_receipt"])
    train_path, train_value = bound_json(
        export["training_record"], paths["dpo_export_receipt"].parent, inputs
    )
    if "files_sha256" not in train_value:
        reload_run_path = read_bound(
            {
                "path": "run.json",
                "sha256": load_json(paths["dpo_fresh_reload"])["files_sha256"][
                    "run.json"
                ],
            },
            paths["dpo_fresh_reload"].parent,
            inputs,
        )
        train_path = read_bound(
            {
                "path": "manifest.json",
                "sha256": load_json(reload_run_path)["previous_manifest_sha256"],
            },
            train_path.parent,
            inputs,
        )
    paths["dpo_training"] = train_path
    stages = {
        name: load_stage(paths["dpo_" + name], config, common, inputs, status)
        for name, status in (
            ("reference_cache", "reference_cache_complete"),
            ("profile", TRAIN_COMPLETE),
            ("profile_reload", RELOAD_COMPLETE),
            ("training", TRAIN_COMPLETE),
            ("fresh_reload", RELOAD_COMPLETE),
        )
    }
    require(
        train_value == stages["training"]["manifest"]
        or train_value == stages["training"]["run"],
        "Export training record is not the bound final substantive run.",
    )
    anchor_tensors = record(anchor, "adapter-tensors.json")
    diagnostic_ids = [row["case_id"] for row in record(anchor, "smoke-generation.json")]
    require(
        len(set(diagnostic_ids)) == len(diagnostic_ids) == 8,
        "Keep eight fixed train diagnostics.",
    )
    validate_reference(stages["reference_cache"], pairs, anchor_tensors)
    reference_sha = inputs[paths["dpo_reference_cache"]]
    for name, phase, selected, reload_name in (
        (
            "profile",
            "engineering_profile",
            trainer.profile_pairs(pairs),
            "profile_reload",
        ),
        ("training", "substantive", pairs, "fresh_reload"),
    ):
        evidence = stages[name]
        validate_final_run(evidence["run"], selected, phase)
        validate_exposure(evidence, selected)
        validate_reload(evidence, stages[reload_name], diagnostic_ids)
        require(
            evidence["run"]["reference_cache_sha256"] == reference_sha
            and record(evidence, "runtime.json")["initial_adapter_tensors"]
            == anchor_tensors
            and record(evidence, "initialization.json")["profile_adapter_loaded"]
            is False
            and record(evidence, "initialization.json")["initial_adapter_tensors"]
            == anchor_tensors
            and record(evidence, "runtime.json")["gpu"]
            == stages["reference_cache"]["run"]["runtime_gpu"],
            "Training must use the fixed reference and restart from T1, discarding profile weights.",
        )
    final = stages["training"]["run"]
    require(
        final["profile_manifest_sha256"] == inputs[paths["dpo_profile"]]
        and final["profile_reload_manifest_sha256"]
        == inputs[paths["dpo_profile_reload"]],
        "Substantive DPO lost its independent engineering-profile launch gate.",
    )
    for earlier, later in (
        ("reference_cache", "profile"),
        ("profile_reload", "training"),
    ):
        a, b = stages[earlier]["run"], stages[later]["run"]
        require(
            a["seconds"] >= 0 and a["started_unix"] + a["seconds"] <= b["started_unix"],
            "Reference/profile evidence must finish before dependent optimizer updates.",
        )
    registration.update(
        configuration=config, stages=stages, preferences=bundle, export=export
    )


def validate_registry(registration):
    paths, inputs, t1 = (
        registration["paths"],
        registration["inputs"],
        registration["t1"],
    )
    registry = load_json(paths["registry"])
    old = t1["registry"]
    sft_id = t1["execution"]["candidate_ids"]["sft"]
    dpo_id = registration["execution"]["dpo_candidate_id"]
    models = {row["id"]: row for row in registry["candidates"]}
    anchor = next(row for row in old["candidates"] if row["id"] == sft_id)
    require(
        dpo_id != sft_id
        and dpo_id not in probe.BASELINES
        and len(registry["candidates"]) == 2
        and set(models) == {sft_id, dpo_id}
        and models[sft_id] == anchor
        and registry["runtime"] == old["runtime"]
        and registry["tokenizers"] == old["tokenizers"]
        and probe.sampling_options(paths["registry"])
        == probe.sampling_options(t1["paths"]["new_registry"]),
        "Preserve the exact SFT control and B2 runtime/tokenizers/sampling in a two-model registry.",
    )
    export = registration["export"]
    require(
        export.get("version") == "passive-sft-export-v1"
        and export.get("status")
        == "tensor_verified_paired_export_runtime_preflight_pending"
        and export.get("base_repository") == BASE_REPOSITORY
        and export.get("base_revision") == BASE_REVISION
        and export.get("runtime_revision") == RUNTIME_REVISION,
        "DPO needs the exact native-base tensor-verified b10816 export.",
    )
    parent = paths["dpo_export_receipt"].parent
    _, tensor = bound_json(
        {
            "path": "tensor-verification.json",
            "sha256": export["tensor_verification_sha256"],
        },
        parent,
        inputs,
    )
    _, export_config = bound_json(
        {"path": "configuration.json", "sha256": export["configuration_sha256"]},
        parent,
        inputs,
    )
    _, anchor_export_config = bound_json(
        {"path": "configuration.json", "sha256": t1["export"]["configuration_sha256"]},
        t1["paths"]["new_export_receipt"].parent,
        inputs,
    )
    require(
        export_config["base"]["files_sha256"]
        == anchor_export_config["base"]["files_sha256"]
        and export_config["runtime"]["source_files_sha256"]
        == anchor_export_config["runtime"]["source_files_sha256"]
        and {
            name: value["sha256"]
            for name, value in export_config["runtime"]["tools"].items()
        }
        == {
            name: value["sha256"]
            for name, value in anchor_export_config["runtime"]["tools"].items()
        }
        and export_config["training_record"] == export["training_record"],
        "DPO export must preserve T1's exact native files, converter and quantizer.",
    )
    training = registration["stages"]["training"]
    require(
        tensor.get("status") == "passed"
        and tensor.get("template_sha256") == anchor["template_sha256"]
        and tensor.get("eos_token_id")
        == registration["configuration"]["generation"]["eos_token_id"]
        == curriculum.native_eos(anchor, inputs)
        and all(
            export_config["adapter"]["files_sha256"][name]
            == training["manifest"]["files_sha256"]["adapter/" + name]
            for name in ("adapter_model.safetensors", "adapter_config.json")
        ),
        "Export changed the final DPO adapter or native template/EOS.",
    )
    model, artifact = models[dpo_id], export["artifacts"]["merged_quantized"]
    require(
        model.get("tokenizer_id") == "qwen35-9b"
        and all(model.get(key) == anchor[key] for key in curriculum.IDENTITY_FIELDS)
        and project_path(model["model_path"]) == project_path(artifact["path"], parent)
        and model["model_sha256"] == artifact["sha256"],
        "DPO registry candidate is not this final native-template Q4_K_M export.",
    )
    probe.inspect_runtime(paths["registry"], dpo_id)
    registration["registry"] = registry


def prepare(protocol_path, execution_path):
    registration = load_protocol(protocol_path)
    inputs = registration["inputs"]
    execution_path = project_path(execution_path)
    inputs[execution_path] = probe.digest(execution_path)
    execution = load_json(execution_path)
    require(
        execution.get("version") == EXECUTION_VERSION
        and execution.get("protocol_sha256") == inputs[registration["protocol_path"]],
        "Execution must bind this DPO protocol.",
    )
    paths = {
        name: read_bound(execution[name], execution_path.parent, inputs)
        for name in ("t1_protocol", "t1_execution", "sft_full_receipt", "registry")
    }
    paths.update(
        {
            "dpo_" + name: read_bound(
                execution["dpo"][name], execution_path.parent, inputs
            )
            for name in DPO_BINDINGS
        }
    )
    require(
        inputs[paths["t1_protocol"]]
        == registration["protocol"]["anchor"]["sft_protocol_sha256"],
        "The T1 protocol is fixed before the DPO experiment.",
    )
    t1 = curriculum.prepare(paths["t1_protocol"], paths["t1_execution"], "sft-full")
    for path, expected in t1["inputs"].items():
        require(
            path not in inputs or inputs[path] == expected,
            "Conflicting T1 input identity.",
        )
        inputs[path] = expected
    require(
        t1["binding"]["training_configuration_sha256"]
        == registration["protocol"]["anchor"]["sft_configuration_sha256"],
        "Reuse only the preregistered final T1 configuration, independent of T2 results.",
    )
    registration.update(
        execution_path=execution_path, execution=execution, paths=paths, t1=t1
    )
    registration["sft"] = saved_results(
        registration,
        paths["sft_full_receipt"],
        t1["binding"],
        t1["prepared"],
        t1["execution"]["candidate_ids"]["sft"],
        t1["paths"]["new_registry"],
    )
    validate_training(registration)
    validate_registry(registration)
    binding = {
        "version": VERSION,
        "condition": "dpo-full",
        "protocol_sha256": inputs[registration["protocol_path"]],
        "execution_sha256": inputs[execution_path],
        "registry_sha256": inputs[paths["registry"]],
        "registered_requests": 289,
        "registered_new_requests": 289,
        "reused_sft_receipt_sha256": inputs[paths["sft_full_receipt"]],
        "t1_configuration_sha256": t1["binding"]["training_configuration_sha256"],
        "artifact_bindings": {
            name: inputs[paths["dpo_" + name]] for name in (*DPO_BINDINGS, "training")
        },
        "preference_pair_count": len(registration["preferences"]["pairs"]),
        "final_training_updates": registration["stages"]["training"]["run"][
            "training_updates"
        ],
        "sft_generation_repeated": False,
        "grading_callback": "unchanged passive_benchmark.evaluate_result; frozen source execution and repeat gate",
        "timing_scope": TIMING_SCOPE,
    }
    prepared = {**t1["prepared"], "binding": copy.deepcopy(t1["prepared"]["binding"])}
    prepared["binding"].pop("curriculum")
    prepared["binding"]["dpo"] = binding
    sources = [
        Path(__file__).resolve(),
        *(Path(source).resolve() for _, _, source in SOURCES),
    ]
    prepared["source_files"] = list(
        dict.fromkeys([*prepared["source_files"], *sources])
    )
    for source in sources:
        inputs[source] = probe.digest(source)
        prepared["binding"]["snapshot_sha256"][source.name] = inputs[source]
    prepared["summarize"] = summarize
    registration.update(prepared=prepared, binding=binding)
    verify_inputs(inputs)
    return registration


def outcome_reason(case, row):
    if row is None:
        return "missing_result"
    if failed(row):
        return "request_failure"
    if not row.get("raw_valid"):
        return row.get("error_type", "invalid_result")
    if row.get("suppression_reason"):
        return "repeated_dismissed_edit"
    prediction = row["prediction"]
    matched = (row.get("raw_grade") or {}).get("outcome_match") is True
    if matched:
        return (
            "correct_abstention"
            if prediction["action"] == "no_action"
            else "useful_displayed_edit"
        )
    references = b2.b1._references(case)
    if not references:
        return "reference_unknown"
    if prediction["action"] == "no_action":
        return "missed_required_edit"
    if all(value["action"] == "no_action" for value in references):
        return "unnecessary_edit_on_preserve_state"
    return "valid_unmatched_edit"


def summarize(cases, rows):
    indexed = {row["case_id"]: row for row in rows}
    reasons = defaultdict(list)
    for case in cases:
        reasons[outcome_reason(case, indexed.get(case["id"]))].append(case["id"])
    return {
        **b2.summarize_b2(cases, rows),
        "correct_preserves": curriculum.correct_preserves(cases, rows),
        "outcome_reasons": {
            name: {"count": len(ids), "case_ids": ids}
            for name, ids in sorted(reasons.items())
        },
        "reason_scope": "Development outcome descriptions from frozen references; training preference reasons are reported separately.",
        "timing_scope": TIMING_SCOPE,
    }


def snapshot(registration, output):
    target = output / "implementation"
    target.mkdir(exist_ok=True)
    paths = {
        "dpo-protocol.json": registration["protocol_path"],
        "dpo-execution.json": registration["execution_path"],
        **{
            f"dpo-binding-{name}.json": path
            for name, path in registration["paths"].items()
        },
    }
    for name, path in paths.items():
        require(
            probe.digest(path) == registration["inputs"][path],
            "DPO input changed before snapshot.",
        )
        with (target / name).open("xb") as stream:
            stream.write(path.read_bytes())


def run_condition(registration, output, stage, editor_state):
    release = registration["t1"]["args"].release
    output = curriculum.new_output(output, release)
    if stage == "prepare":
        output.mkdir(parents=True)
        snapshot(registration, output)
        write_new(
            output / "dpo-receipt.json",
            {**registration["binding"], "status": "prepared_generation_unrun"},
        )
        return
    args = SimpleNamespace(
        release=release,
        registry=registration["paths"]["registry"],
        candidate=registration["execution"]["dpo_candidate_id"],
        output=output,
        stage=stage,
        timeout=120,
        unconstrained=False,
        sampling_config=None,
        diagnostic_protocol=None,
        editor_state=editor_state,
    )
    error = None
    prepared = registration["prepared"]
    try:
        probe.run(args, prepared=prepared)
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    if output.exists():
        try:
            verify_inputs(registration["inputs"])
            snapshot(registration, output)
            path = output / "results.jsonl"
            rows = original.read_rows(path) if path.exists() else []
            validate_results(prepared["cases"], rows, prepared["prefixes"])
            with (output / "source-execution.jsonl").open("x") as stream:
                for value in curriculum.source_execution(prepared["cases"], rows):
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            status = load_json(output / "manifest.json")["status"]
        except (Exception, KeyboardInterrupt) as caught:
            error, status = error or caught, "failed"
        write_new(
            output / "dpo-receipt.json",
            {
                **registration["binding"],
                "status": "failed" if error else status,
                "error": f"{type(error).__name__}: {error}" if error else None,
                "files_sha256": {
                    path.name: probe.digest(path)
                    for path in output.iterdir()
                    if path.is_file()
                },
                "runtime_manifest_version": "passive-b2-runtime-v1 (unchanged shared runner)",
            },
        )
    if error:
        raise error


def report(protocol_path, execution_path, run, output):
    registration = prepare(protocol_path, execution_path)
    run = project_path(run)
    output = curriculum.new_output(output, registration["t1"]["args"].release)
    prepared = registration["prepared"]
    receipt_path = run / "dpo-receipt.json"
    if receipt_path.exists():
        registration["inputs"][receipt_path] = probe.digest(receipt_path)
        dpo = saved_results(
            registration,
            receipt_path,
            registration["binding"],
            prepared,
            registration["execution"]["dpo_candidate_id"],
            registration["paths"]["registry"],
        )
    else:
        require(
            not run.exists(),
            "Existing DPO run has no immutable completion/failure receipt.",
        )
        dpo = {"rows": [], "status": "missing_run"}
    observations = {"sft-full": registration["sft"], "dpo-full": dpo}
    summaries = {
        name: {
            **summarize(prepared["cases"], value["rows"]),
            "run_status": value["status"],
        }
        for name, value in observations.items()
    }
    complete = all(
        value["status"] == "completed" and len(value["rows"]) == 289
        for value in observations.values()
    )
    verify_inputs(registration["inputs"])
    output.mkdir(parents=True)
    snapshot(registration, output)
    write_new(
        output / "summary.json",
        {
            **registration["binding"],
            "experiment_status": "completed" if complete else "incomplete_engineering",
            "registered_conditions": ["sft-full", "dpo-full"],
            "registered_requests_per_condition": 289,
            "registered_reused_sft_requests": 289,
            "runs": summaries,
            "paired_quality": b2.compare_runs(
                prepared["cases"],
                {name: value["rows"] for name, value in observations.items()},
            ),
            "training_preference_reasons": dict(
                Counter(pair["reason"] for pair in registration["preferences"]["pairs"])
            ),
            "provenance": {
                name: {
                    key: value for key, value in observation.items() if key != "rows"
                }
                for name, observation in observations.items()
            },
            "negative_result_policy": "Worse DPO quality still completes the fixed experiment. No automatic extra epochs, checkpoint selection or deployment promotion.",
            "scope": registration["protocol"]["scope"],
            "final_test_bodies_read": 0,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "report"), default="run")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "preflight", "screen"), default="screen"
    )
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--editor-state", default="Not controlled; no editor-preview latency claim"
    )
    args = parser.parse_args(argv)
    if args.command == "report":
        require(
            args.run is not None, "Report requires the one --run directory for DPO."
        )
        report(args.protocol, args.execution, args.run, args.output)
        return
    require(
        args.run is None, "Run never accepts an SFT generation condition or prior run."
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
        registration = prepare(args.protocol, args.execution)
        run_condition(registration, args.output, args.stage, args.editor_state)


if __name__ == "__main__":
    main()
