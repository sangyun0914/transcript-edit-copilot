"""Evaluate separately registered Gemma matched formats without changing frozen runners."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import export_passive_gemma as exporter
import format_comparison
import passive_b2_benchmark as b2
import passive_formats_v2 as formats
import probe_passive as probe
import probe_passive_b2 as original
import probe_passive_curriculum as curriculum
import probe_passive_gemma as gemma
import probe_passive_matched as matched
from passive_contract import EDIT
from replay_session import session_hash

ROOT = probe.ROOT
VERSION = "passive-gemma-matched-evaluation-v1"
EXECUTION_VERSION = "passive-gemma-matched-execution-v1"
TRAINING_VERSION = "passive-gemma-matched-sft-v1"
CONDITIONS = matched.CONDITIONS
BASE_REPOSITORY = exporter.BASE_REPOSITORY
BASE_REVISION = exporter.BASE_REVISION
RUNTIME_REVISION = exporter.RUNTIME_REVISION
GENERATION_STOP_IDS = exporter.GENERATION_STOP_IDS
RESPONSE_EVIDENCE = {
    "verbose": True,
    "return_tokens": True,
    "metric_scope": "api_output_reference_agreement_with_separate_raw_format_diagnostics",
    "tool_handoff_token_id": 50,
    "tool_handoff_credited": False,
}
SOURCE_MODULES = (
    exporter,
    format_comparison,
    b2,
    formats,
    probe,
    original,
    curriculum,
    gemma,
    matched,
)
require = curriculum.require
project_path = curriculum.project_path
read_bound = curriculum.read_bound
load_json = curriculum.load_json
write_new = curriculum.write_new
bound_json = matched.bound_json
snapshot = matched.snapshot


def source_paths():
    return [
        Path(__file__).resolve(),
        *(Path(m.__file__).resolve() for m in SOURCE_MODULES),
    ]


def verify_source_pins(protocol, inputs):
    pins = protocol["evaluation_sources_sha256"]
    required = {str(path.relative_to(ROOT)) for path in source_paths()}
    require(
        set(pins) == required,
        "Bind the complete exact Gemma matched evaluator source set.",
    )
    for name, expected in pins.items():
        read_bound({"path": name, "sha256": expected}, ROOT, inputs)


def grade_response(case, response, format_name):
    choice, usage, raw = (
        response["choices"][0],
        response["usage"],
        response["__verbose"],
    )
    text, finish = choice["message"].get("content") or "", choice["finish_reason"]
    ids = raw["tokens"]
    require(
        isinstance(raw["content"], str)
        and isinstance(ids, list)
        and 0 < len(ids) <= 512
        and all(type(token) is int and 0 <= token < 262144 for token in ids)
        and type(usage["completion_tokens"]) is int
        and len(ids) == usage["completion_tokens"]
        and raw["stop_type"] == ("eos" if ids[-1] in GENERATION_STOP_IDS else "limit")
        and (
            (ids[-1] in (1, 106) and finish == "stop")
            or (ids[-1] == 50 and finish in ("stop", "tool_calls"))
            or (
                ids[-1] not in GENERATION_STOP_IDS
                and len(ids) == 512
                and finish == "length"
            )
        ),
        "Missing or inconsistent exact sampled content/token/terminal evidence.",
    )
    row = matched.grade(case, text, finish, format_name=format_name)
    raw_format_valid = False
    raw_parse_valid = False
    try:
        if format_name == "json":
            EDIT.validate_json(raw["content"], strict=True)
        else:
            format_comparison.Rewrite.model_validate_json(raw["content"], strict=True)
        raw_format_valid = True
        formats.parse(raw["content"], case["input"], format_name, finish_reason=finish)
        raw_parse_valid = ids[-1] != 50
    except (ValueError, TypeError):
        pass
    row.update(
        sampled_content=raw["content"],
        sampled_token_ids=ids,
        terminal_token_id=ids[-1],
        raw_stop_type=raw["stop_type"],
        sampled_content_equals_api_content=raw["content"] == text,
        sampled_format_valid=raw_format_valid,
        sampled_source_preserving_parse_valid=raw_parse_valid,
        token_evidence_valid=True,
    )
    if ids[-1] == 50:
        row.update(
            raw_valid=False,
            source_preserving_valid=False,
            displayed=False,
            raw_grade=None,
            displayed_grade=None,
            error_type="tool_handoff",
            error="Native terminal50 is outside the passive editing contract.",
        )
    return row


def isolated_runtime(format_name):
    runtime = matched.isolated_runtime(format_name)
    request = runtime.request_body
    runtime.request_body = lambda *a, **kw: {
        **request(*a, **kw),
        "verbose": True,
        "return_tokens": True,
    }
    captured = []

    class CaptureClient(probe.httpx.Client):
        def post(self, url, *args, **kwargs):
            if str(url) == "/v1/chat/completions":
                captured.clear()
            response = super().post(url, *args, **kwargs)
            if str(url) == "/v1/chat/completions":
                captured.append(response)
            return response

    def evaluate(case, text, finish_reason="stop"):
        require(
            len(captured) == 1, "No response accompanies this isolated grading call."
        )
        response = captured.pop().json()
        choice = response["choices"][0]
        require(
            text == (choice["message"].get("content") or "")
            and finish_reason == choice["finish_reason"],
            "Captured response differs from the current API output.",
        )
        return grade_response(case, response, format_name)

    runtime.httpx = SimpleNamespace(**{**vars(probe.httpx), "Client": CaptureClient})
    runtime.evaluate_result = evaluate
    return runtime


def summarize(cases, rows):
    accounting_rows = []
    for row in rows:
        response = row.get("response")
        usage = response.get("usage") if isinstance(response, dict) else None
        count = usage.get("completion_tokens") if isinstance(usage, dict) else None
        accounting_rows.append(
            {
                **row,
                "response": {"usage": {"completion_tokens": count}}
                if type(count) is int and 0 <= count <= 512
                else {},
            }
        )
    result = gemma.quality_metrics(matched.summarize(cases, accounting_rows))
    result["output_tokens"]["scope"] = (
        "API-reported integer completion counts in [0,512]; malformed counts remain unmeasured. "
        "Exact sampled-token evidence is counted separately; original responses remain unchanged."
    )
    result.update(
        fresh_http_response_ms_p50_p95=gemma.http_latency(rows, len(cases)),
        sampled_format_valid=sum(
            row.get("sampled_format_valid") is True for row in rows
        ),
        sampled_source_preserving_parse_valid=sum(
            row.get("sampled_source_preserving_parse_valid") is True for row in rows
        ),
        sampled_content_differs_from_api=sum(
            row.get("sampled_content_equals_api_content") is False for row in rows
        ),
        token_evidence_observations=sum(
            row.get("token_evidence_valid") is True for row in rows
        ),
        tool_handoffs=sum(row.get("terminal_token_id") == 50 for row in rows),
        raw_format_scope="Strict unchanged sampled content; separate from API-output outcome credit. No output repair.",
    )
    indexed = {case["id"]: case for case in cases}

    def action(case):
        return case.get("_b2", {}).get(
            "original_primary_action",
            (case.get("expectedOutput") or {}).get("action", "unknown"),
        )

    groups = {
        "action": action,
        "language": lambda c: c["metadata"]["language"],
        "action_language": lambda c: f"{action(c)}|{c['metadata']['language']}",
        "source": lambda c: c["metadata"]["source_group_id"],
        "episode": lambda c: c["metadata"]["episode_id"],
        "stratum": lambda c: c["metadata"].get("release_stratum", "unspecified"),
        "feedback_condition": lambda c: c.get("_b2", {}).get(
            "original_feedback_condition", b2.feedback_condition(c)
        ),
    }
    for label, group in groups.items():
        for value, summary in result[f"by_{label}"].items():
            selected = [row for row in rows if group(indexed[row["case_id"]]) == value]
            summary.update(
                fresh_http_response_ms_p50_p95=gemma.http_latency(
                    selected, summary["expected_requests"]
                ),
                sampled_format_valid=sum(
                    row.get("sampled_format_valid") is True for row in selected
                ),
                extra_change=sum(row.get("extra_change") is True for row in selected),
                truncated=sum(row.get("finish_reason") == "length" for row in selected),
                tool_handoffs=sum(
                    row.get("terminal_token_id") == 50 for row in selected
                ),
            )
    return result


def validate_results(cases, rows, prefixes, fmt):
    ids = [case["id"] for case in cases]
    observed = [row["case_id"] for row in rows]
    require(
        len(set(observed)) == len(observed)
        and observed == [x for x in ids if x in observed],
        "Results changed matched case order or contain duplicate/foreign states.",
    )
    indexed = {case["id"]: case for case in cases}
    for row in rows:
        if (
            row.get("request_failed")
            or row.get("cancelled")
            or row.get("error_type") in original.REQUEST_ERRORS
        ):
            require(
                row.get("raw_valid") is False
                and row.get("displayed") is False
                and row.get("raw_grade") is None
                and row.get("displayed_grade") is None
                and row.get("prediction") is None
                and all(
                    row.get(field) is not True
                    for field in (
                        "format_valid",
                        "source_preserving_valid",
                        "sampled_format_valid",
                        "sampled_source_preserving_parse_valid",
                        "token_evidence_valid",
                    )
                ),
                "Failed requests cannot receive editing or preservation credit.",
            )
            continue
        choice, usage = row["response"]["choices"][0], row["response"]["usage"]
        require(
            row["raw_text"] == (choice["message"].get("content") or "")
            and row["finish_reason"] == choice["finish_reason"]
            and usage["prompt_tokens"] == len(prefixes[row["case_id"]]),
            "API completion or exact frozen prefix changed.",
        )
        expected = grade_response(indexed[row["case_id"]], row["response"], fmt)
        require(
            all(row.get(key) == value for key, value in expected.items()),
            "Saved API/raw-format/source grading or token evidence changed.",
        )


def validate_run_identity(config, run, fmt, configuration_sha, protocol_sha):
    require(
        config.get("version") == run.get("version") == TRAINING_VERSION
        and config.get("phase") == run.get("phase") == "substantive"
        and config.get("format") == run.get("format") == fmt
        and run.get("status") == "training_complete_fresh_reload_pending"
        and run.get("mode") == "train"
        and run.get("training_updates") == 201
        and run.get("optimizer_committed_rows") == 802
        and run.get("configuration_sha256") == configuration_sha
        and run.get("evaluation_protocol_sha256") == protocol_sha
        and run.get("initialization") == "original_native_base_with_fresh_lora"
        and run.get("optimizer_state_resumed") is False
        and run.get("final_test_bodies_read") == 0,
        "Require each format's independent final201/802 substantive Gemma arm, never smoke or recovery.",
    )


def validate_initialization(initialization, fmt):
    require(
        initialization["initialization"] == "original_native_base_with_fresh_lora"
        and initialization["smoke_adapter_loaded"] is False
        and initialization["t1_adapter_loaded"] is False
        and initialization["other_format_adapter_loaded"] is False
        and initialization["format"] == fmt
        and len(initialization["trainable_tensors"]) == 410
        and initialization["trainable_parameters"] == 18585600,
        "Matched arms must independently initialize fresh205-target Gemma LoRA.",
    )


def validate_exposure(exposure, recipe, rows):
    committed = exposure["phases"]["optimizer_committed"]
    require(
        exposure["completed_updates"] == recipe["max_updates"] == 201
        and committed["rows"]
        == recipe["draw_count"]
        == len(recipe["sample_order"])
        == 802
        and exposure["optimizer_commit_uncertain"] is False
        and exposure["pending_backward_rows_not_committed"] == 0
        and committed["counts"]["case_id"] == dict(Counter(recipe["sample_order"]))
        and committed["sequence_tokens"] == recipe["max_total_sequence_tokens"]
        and committed["completion_tokens"] == recipe["max_total_completion_tokens"],
        "Final matched optimizer exposure differs from the fixed802-draw stream.",
    )
    expected = gemma.trainer.curriculum.Exposure()
    for row in rows:
        for phase in gemma.trainer.curriculum.PHASES:
            expected.record(phase, row)
    expected.updates = recipe["max_updates"]
    require(
        exposure == expected.summary(),
        "Every forward/backward/committed source, episode and token exposure must match the frozen rows.",
    )


def validate_runtime(runtime, config, *, trained):
    counts = runtime["base_parameter_census"]["parameter_elements_by_dtype"]
    require(
        runtime["numerical_regime"] == "native_bfloat16_lora_float32"
        and runtime["parameter_dtypes"] == ["torch.bfloat16"]
        and runtime["ignored_native_tensor_names"] == []
        and len(runtime["targets"]) == len(set(runtime["targets"])) == 205
        and sorted(runtime["targets"]) == config["model"]["target_modules"]
        and runtime["effective_generation"] == config["generation"]
        and runtime["native_generation_config"]["eos_token_id"] == GENERATION_STOP_IDS
        and runtime["completion_label_eos_token_id"] == 1
        and runtime["text_attn_implementation"]
        == config["model"]["attn_implementation"]
        == "sdpa"
        and runtime["text_experts_implementation"]
        == config["model"]["experts_implementation"]
        == "eager"
        and config["model"]["gpu_name_contains"] == runtime["gpu"]
        and runtime["gpu_total_bytes"] >= config["model"]["minimum_gpu_bytes"]
        and set(counts) == {"torch.bfloat16"}
        and counts["torch.bfloat16"] > 0,
        "Gemma205-target BF16/native-EOS/backend runtime census differs.",
    )
    if trained:
        require(
            runtime["after_adapter_parameter_census"]["parameter_elements_by_dtype"]
            == {**counts, "torch.float32": 18585600},
            "Native base changed dtype or FP32 LoRA parameter count changed.",
        )


def validate_training(
    registration, fmt, config_path, config, run_path, run, reload_path, reload_manifest
):
    import train_passive_gemma_matched as trainer

    inputs = registration["inputs"]
    protocol_sha = inputs[registration["protocol_path"]]
    release_sha = inputs[registration["release"] / "manifest.json"]
    validate_run_identity(config, run, fmt, inputs[config_path], protocol_sha)
    for _, key, source in trainer.SOURCE_MODULES:
        path = Path(source).resolve()
        require(
            probe.digest(path) == config[key],
            "Current trainer/preparer differs from the trained source pin.",
        )
        inputs[path] = config[key]
    trainer.validate_configuration(config)
    trainer.validate_protocol(registration["protocol"], config, release_sha)
    recipe = config["training"]
    require(
        config["candidate_id"] == "gemma26"
        and config["base"]["repository"] == BASE_REPOSITORY
        and config["base"]["revision"] == BASE_REVISION
        and recipe["max_updates"] == 201
        and recipe["draw_count"] == 802
        and recipe["gradient_accumulation"] == 4
        and recipe["micro_batch_size"] == 1
        and recipe["sampling_seed"] == "3407"
        and recipe["seed"] == 3407
        and recipe["fresh_native_initialization"] is True
        and recipe["final_partial_accumulation"] == "actual_microbatch_count",
        "Use the fixed native Gemma201/802 recipe with a final two-row mean.",
    )
    protocol_path = read_bound(
        config["bindings"]["evaluation_protocol"], config_path.parent, inputs
    )
    require(
        inputs[protocol_path] == protocol_sha,
        "Training used another matched evaluation protocol.",
    )
    prep_path, prep = bound_json(
        config["bindings"]["preparation"], config_path.parent, inputs
    )
    require(
        prep["version"] == "passive-sft-gemma-matched-preparation-v1"
        and prep["status"] == "frozen_matched_train_only_preparation"
        and prep["release_manifest_sha256"] == release_sha
        and run["preparation_sha256"] == inputs[prep_path],
        "Matched preparation differs from the frozen train-only Gemma release.",
    )
    bundle = trainer.preparation.verify_bundle(prep_path.parent)
    readback = bundle["verification"]
    require(
        bundle["manifest"] == prep
        and readback["status"] == "passed_matched_train_only_gemma_readback"
        and readback["train_rows"] == readback["shared_draws_per_format"] == 802
        and readback["optimizer_updates_per_format"] == 201
        and readback["final_update_rows"] == 2
        and readback["dev_test_record_bodies_read"] == 0
        and readback["training_updates_executed"] == 0,
        "Genuine Gemma matched train-only verifier did not pass the registered scope.",
    )
    selected_rows, diagnostic_rows = trainer.validate_prepared(config, bundle)
    for name, expected in prep["files_sha256"].items():
        read_bound({"path": name, "sha256": expected}, prep_path.parent, inputs)
    for name in (*trainer.T1_BINDINGS, *trainer.SMOKE_BINDINGS):
        read_bound(config["bindings"][name], config_path.parent, inputs)
    trainer.validate_t1_evidence(config, config_path.parent, release_sha)
    trainer.validate_smoke_gate(config, config_path.parent, release_sha)
    native_pin, pin = bound_json(
        config["bindings"]["native_pin"], config_path.parent, inputs
    )
    require(
        run["native_pin_sha256"] == inputs[native_pin]
        and pin["repository"] == BASE_REPOSITORY
        and pin["revision"] == BASE_REVISION
        and {name: row["sha256"] for name, row in pin["files"].items()}
        == config["base"]["files_sha256"],
        "Native acquisition inventory changed between matched training and export.",
    )

    def payload(directory, manifest, name):
        return read_bound(
            {"path": name, "sha256": manifest["files_sha256"][name]}, directory, inputs
        )

    require(
        reload_manifest["version"] == TRAINING_VERSION
        and reload_manifest["status"] == "fresh_reload_passed_local_export_pending",
        "Require this matched arm's completed independent reload.",
    )
    reload_run = load_json(payload(reload_path.parent, reload_manifest, "run.json"))
    train_manifest_path, train_manifest = bound_json(
        {"path": "manifest.json", "sha256": reload_run["previous_manifest_sha256"]},
        run_path.parent,
        inputs,
    )
    require(
        train_manifest["version"] == TRAINING_VERSION
        and train_manifest["status"] == "training_complete_fresh_reload_pending"
        and payload(run_path.parent, train_manifest, "run.json") == run_path
        and reload_run["version"] == TRAINING_VERSION
        and reload_run["format"] == fmt
        and reload_run["phase"] == "substantive"
        and reload_run["mode"] == "fresh_reload"
        and reload_run["status"] == "fresh_reload_passed_local_export_pending"
        and reload_run["initialization"] == "original_native_base_with_saved_lora"
        and reload_run["optimizer_state_resumed"] is False
        and reload_run["configuration_sha256"] == inputs[config_path]
        and reload_run["evaluation_protocol_sha256"] == protocol_sha
        and reload_run["native_pin_sha256"] == inputs[native_pin]
        and reload_run["preparation_sha256"] == inputs[prep_path]
        and reload_run["reloaded_training_updates"] == 201
        and type(run["pid"]) is int
        and run["pid"] > 0
        and type(reload_run["pid"]) is int
        and reload_run["pid"] > 0
        and reload_run["pid"] != run["pid"],
        "Fresh reload belongs to another arm/checkpoint/configuration or the same process.",
    )
    require(
        reload_run["started_unix"] > run["finished_unix"],
        "Independent reload must start after final training finishes.",
    )
    expected_t1 = {
        name: config["bindings"][name]["sha256"] for name in trainer.T1_BINDINGS
    }
    require(
        run["t1_evidence_sha256"] == reload_run["t1_evidence_sha256"] == expected_t1,
        "Matched train/reload changed the preregistered T1 engineering evidence.",
    )
    require(
        all(
            type(row.get("peak_allocated_gpu_bytes")) is int
            and 0 < row["peak_allocated_gpu_bytes"] <= row["peak_reserved_gpu_bytes"]
            for row in (run, reload_run)
        ),
        "Missing actual training/reload GPU memory observations.",
    )
    for directory, manifest in (
        (run_path.parent, train_manifest),
        (reload_path.parent, reload_manifest),
    ):
        require(
            inputs[payload(directory, manifest, "configuration.json")]
            == inputs[config_path],
            "Saved matched configuration bytes changed.",
        )
        for name, key, _ in trainer.SOURCE_MODULES:
            require(
                inputs[payload(directory, manifest, name)] == config[key],
                "Matched training source snapshot changed.",
            )
        validate_runtime(
            load_json(payload(directory, manifest, "runtime.json")),
            config,
            trained=directory == run_path.parent,
        )
    initialization = load_json(
        payload(run_path.parent, train_manifest, "initialization.json")
    )
    validate_initialization(initialization, fmt)
    validate_exposure(
        load_json(payload(run_path.parent, train_manifest, "exposure.json")),
        recipe,
        selected_rows,
    )
    parity = load_json(
        payload(reload_path.parent, reload_manifest, "reload-parity.json")
    )
    before = load_json(
        payload(run_path.parent, train_manifest, "smoke-generation.json")
    )
    after = load_json(
        payload(reload_path.parent, reload_manifest, "smoke-generation.json")
    )
    diagnostics = bundle["views"][fmt]["diagnostics"]
    diagnostic_ids = [case["id"] for case in diagnostics["cases"]]
    require(
        parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and before == after
        and len(diagnostic_ids) == len(set(diagnostic_ids)) == 8
        and [row["case_id"] for row in before] == diagnostic_ids
        and [row["case_id"] for row in parity["cases"]] == diagnostic_ids
        and all(row["identical"] is True for row in parity["cases"])
        and all(
            isinstance(row["text_raw"], str)
            and isinstance(row["output_ids"], list)
            and row["terminal_token_id"]
            == (row["output_ids"][-1] if row["output_ids"] else None)
            for row in before
        ),
        "Matched fresh reload lacks exact adapter/shared-eight raw-output parity.",
    )
    for saved, row in zip(before, diagnostic_rows, strict=True):
        prefix = row["input_ids"][: row["completion_start"]]
        ids = saved["output_ids"]
        require(
            saved["prompt_ids_sha256"]
            == hashlib.sha256(
                json.dumps(prefix, separators=(",", ":")).encode()
            ).hexdigest()
            and 0 < len(ids) <= 512
            and all(type(token) is int and 0 <= token < 262144 for token in ids)
            and saved["finish_reason"]
            == ("stop" if ids[-1] in GENERATION_STOP_IDS else "length")
            and (ids[-1] in GENERATION_STOP_IDS or len(ids) == 512),
            "Native reload diagnostics lost exact format prefixes or completion/terminal bounds.",
        )
    for name in (
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter-tensors.json",
    ):
        payload(run_path.parent, train_manifest, name)
    return train_manifest_path, train_manifest


def validate_export(
    registration, config, export_path, export, run_path, train_manifest
):
    inputs = registration["inputs"]
    require(
        export["version"] == "passive-gemma-sft-export-v1"
        and export["status"]
        == "tensor_verified_paired_export_runtime_preflight_pending"
        and export["base_repository"] == BASE_REPOSITORY
        and export["base_revision"] == BASE_REVISION
        and export["runtime_revision"] == RUNTIME_REVISION,
        "Require exact native Gemma/b10816 final paired export evidence.",
    )
    _, normalized = bound_json(
        {"path": "configuration.json", "sha256": export["configuration_sha256"]},
        export_path.parent,
        inputs,
    )
    require(
        normalized["base"]["repository"] == BASE_REPOSITORY
        and normalized["base"]["revision"] == BASE_REVISION
        and normalized["base"]["files_sha256"] == config["base"]["files_sha256"]
        and normalized["training_record"] == export["training_record"]
        and normalized["native_pin"]["sha256"]
        == config["bindings"]["native_pin"]["sha256"]
        and normalized["adapter"]["files_sha256"]
        == {
            name.removeprefix("adapter/"): expected
            for name, expected in train_manifest["files_sha256"].items()
            if name.startswith("adapter/")
        },
        "Export normalized configuration differs from this final adapter/native input.",
    )
    require(
        project_path(normalized["adapter"]["path"], export_path.parent)
        == run_path.parent / "adapter",
        "Export used another adapter directory.",
    )
    _, tensor = bound_json(
        {
            "path": "tensor-verification.json",
            "sha256": export["tensor_verification_sha256"],
        },
        export_path.parent,
        inputs,
    )
    require(
        tensor["status"] == "passed"
        and tensor["adapter_tensors_exact"] == 410
        and tensor["native_targets_exact"] == len(tensor["merged_targets"]) == 205
        and tensor["template_sha256"] == registration["anchor"]["template_sha256"]
        and tensor["eos_token_id"] == registration["native_eos"] == 1
        and tensor["native_generation_stop_ids"]
        == export["native_generation_stop_ids"]
        == config["generation"]["eos_token_id"]
        == GENERATION_STOP_IDS
        and export["native_pin"]["sha256"]
        == config["bindings"]["native_pin"]["sha256"],
        "Paired export native template/label EOS/stop-token evidence changed.",
    )
    read_bound(export["native_pin"], export_path.parent, inputs)
    quant_path, quant = bound_json(
        {"path": "quantization.json", "sha256": export["quantization_sha256"]},
        export_path.parent,
        inputs,
    )
    require(
        quant["quantization"] == "Q4_K_M"
        and quant["importance_matrix"] is None
        and quant["threads"] == 4
        and set(quant["commands"]) == {"base", "merged"}
        and all(
            set(v) == {"command.json", "result.json", "log"}
            for v in quant["commands"].values()
        )
        and quant["quantizer_fallbacks"] == export["quantizer_fallbacks"]
        and all(
            quant["artifacts"][k] == export["artifacts"][k]
            for k in curriculum.ARTIFACT_KEYS.values()
        ),
        "Both arms must retain the unchanged paired Q4_K_M quantization recipe/evidence.",
    )
    for command in quant["commands"].values():
        for value in command.values():
            read_bound(value, quant_path.parent, inputs)
    for artifact in export["artifacts"].values():
        path = project_path(artifact["path"], export_path.parent)
        require(
            path.is_file()
            and path.stat().st_size == artifact["bytes"]
            and artifact["bytes"] > 0,
            "Paired export artifact size changed.",
        )
    require(
        set(export["artifacts"])
        == {"base", "adapter", "merged", "base_quantized", "merged_quantized"},
        "Retain all five unchanged paired-export artifacts.",
    )


def validate_arm(registration, fmt, arm):
    inputs, parent = registration["inputs"], registration["execution_path"].parent
    config_path, config = bound_json(arm["training_configuration"], parent, inputs)
    export_path, export = bound_json(arm["export_receipt"], parent, inputs)
    reload_path, reload_manifest = bound_json(arm["fresh_reload"], parent, inputs)
    run_path, run = bound_json(export["training_record"], export_path.parent, inputs)
    if "configuration_sha256" not in run:
        run_path, run = bound_json(
            {"path": "run.json", "sha256": run["files_sha256"]["run.json"]},
            run_path.parent,
            inputs,
        )
    train_manifest_path, train_manifest = validate_training(
        registration,
        fmt,
        config_path,
        config,
        run_path,
        run,
        reload_path,
        reload_manifest,
    )
    validate_export(registration, config, export_path, export, run_path, train_manifest)
    return {
        "configuration": config,
        "export": export,
        "configuration_path": config_path,
        "export_path": export_path,
        "reload_path": reload_path,
        "training_manifest_path": train_manifest_path,
    }


def validate_matched_recipes(arms):
    matched.validate_matched_recipes(arms)
    left, right = (arms[fmt]["configuration"] for fmt in ("json", "rewrite"))
    require(
        left["phase"] == right["phase"] == "substantive"
        and left["candidate_id"] == right["candidate_id"] == "gemma26"
        and left["bindings"]["native_pin"]["sha256"]
        == right["bindings"]["native_pin"]["sha256"],
        "Both independently initialized matched arms must bind the same native Gemma origin.",
    )
    require(
        all(
            arms["json"]["export"]["artifacts"][name]["sha256"]
            == arms["rewrite"]["export"]["artifacts"][name]["sha256"]
            for name in ("base", "base_quantized")
        ),
        "Matched paired exports must preserve the identical untrained native/F16/Q4 baseline.",
    )


def load_protocol(path):
    path = project_path(path)
    protocol = load_json(path)
    inputs = {path: probe.digest(path)}
    require(
        protocol.get("version") == VERSION
        and protocol.get("conditions") == list(CONDITIONS)
        and protocol.get("formats") == ["json", "rewrite"]
        and protocol.get("registered_requests_per_condition") == 286
        and protocol.get("full_dev_registered_requests") == 289
        and protocol.get("excluded_from_matched_comparison") == 3
        and protocol.get("training_rows") == 802
        and protocol.get("training_exclusions") == 7
        and protocol.get("final_test_bodies_read") is False,
        "Use the fixed three-condition matched286 experiment with no final-test access.",
    )
    plan = protocol["training_plan"]
    require(
        plan["candidate"] == "gemma26"
        and plan["updates"] == 201
        and plan["draw_count"] == 802
        and plan["effective_batch"] == 4
        and plan["final_accumulation"] == 2
        and plan["checkpoint_selection"] == "final_update_201_only"
        and plan["independent_fresh_native_runs"] is True
        and protocol["same_case_sampling_order_required"] is True
        and protocol["generation"]["max_new_tokens"] == 512
        and protocol["generation"]["temperature"] == 0
        and protocol["generation"]["context_limit"] == 4096
        and protocol["generation"]["input_limit"] == 3584
        and protocol["generation"]["truncation"] is False,
        "802 draws / accumulation four require 201 updates and a final two-row mean.",
    )
    require(
        protocol.get("response_evidence") == RESPONSE_EVIDENCE,
        "Use equal verbose/token evidence and uncredited tool handoffs.",
    )
    verify_source_pins(protocol, inputs)
    t1_path = read_bound(protocol["t1_protocol"], path.parent, inputs)
    t1 = gemma.load_protocol(t1_path)
    inputs.update(t1["inputs"])
    release_path = read_bound(protocol["b2"]["release_manifest"], path.parent, inputs)
    require(
        inputs[release_path] == probe.digest(t1["paths"]["release_manifest"]),
        "T1 and T2 must use the same frozen release.",
    )
    return {
        "protocol": protocol,
        "protocol_path": path,
        "inputs": inputs,
        "release": release_path.parent,
        "previous_registry": t1["paths"]["registry"],
        "b2_protocol": t1["paths"]["protocol"],
    }


def load_views(registration, *, execution=True):
    release, inputs = registration["release"], registration["inputs"]
    manifest = load_json(release / "manifest.json")
    probe.code_bindings(release)

    def payload(name):
        return read_bound(
            {"path": name, "sha256": manifest["files_sha256"][name]}, release, inputs
        )

    full, _ = b2._load_split(release, "dev")
    full_ids = [case["id"] for case in full]
    require(
        len(full_ids) == 289
        and full_ids == load_json(registration["b2_protocol"])["case_ids"],
        "The full development roster differs from frozen B2.",
    )
    cases = original.read_rows(payload("matched/dev-cases.jsonl"))
    ids = [case["id"] for case in cases]
    require(
        len(ids) == len(set(ids)) == 286
        and ids == [identity for identity in full_ids if identity in ids],
        "Matched cases must retain the ordered 286-case subset of full289.",
    )
    full_by_id = {case["id"]: case for case in full}
    for case in cases:
        original_case = copy.deepcopy(full_by_id[case["id"]])
        original_case["metadata"]["sampling_weight"] = case["metadata"][
            "sampling_weight"
        ]
        require(
            case == original_case,
            "Matched evidence/reference/source changed beyond frozen sampling weights.",
        )
    eligibility = load_json(payload("matched/eligibility.json"))
    excluded = [row for row in eligibility["excluded"] if row["case_id"] in full_by_id]
    require(
        eligibility["qualified"] == {"train": 802, "dev": 286}
        and eligibility["input_limit"] == 3584
        and eligibility["output_limit_each_format"] == 512
        and len(excluded) == 3
        and {row["case_id"] for row in excluded} == set(full_ids) - set(ids),
        "Retain the registered three development exclusions without recropping.",
    )
    registry = load_json(registration["previous_registry"])
    anchor = next(row for row in registry["candidates"] if row["id"] == "gemma26")
    native = load_json(project_path(anchor["tokenizer_path"]) / "config.json")
    native = native.get("text_config", native)
    eos = curriculum.native_eos(anchor, inputs)
    require(
        eos == 1 and native["vocab_size"] == 262144,
        "Gemma native label EOS/vocabulary changed.",
    )
    views = {}
    full_prefixes = probe.frozen_prefixes(release, "gemma26")
    for fmt in ("json", "rewrite"):
        prompt = payload(f"prompt-{fmt}.txt").read_text().removesuffix("\n")
        tokens = original.read_rows(payload(f"tokenized/gemma26/{fmt}-dev.jsonl.gz"))
        require(
            [row["id"] for row in tokens] == ids,
            "Format token rows changed the matched case order.",
        )
        messages, prefixes = {}, {}
        for case, row in zip(cases, tokens, strict=True):
            encoded = row["primary"]
            values, start = encoded["input_ids"], encoded["completion_start"]
            require(
                row["split"] == "dev"
                and row["input_sha256"] == case["metadata"]["input_sha256"]
                and row["sampling_weight"] == case["metadata"]["sampling_weight"]
                and row["messages"] == formats.messages(case["input"], fmt)
                and row["messages"][0] == {"role": "system", "content": prompt}
                and type(start) is int
                and 0 < start <= 3584
                and 0 < len(values) - start <= 512
                and len(values) <= 4096
                and all(
                    type(value) is int and 0 <= value < native["vocab_size"]
                    for value in values
                )
                and values[-1] == eos
                and encoded["labels"] == [-100] * start + values[start:]
                and encoded["attention_mask"] == [1] * len(values),
                "Frozen format messages, token bounds, masks or EOS differ.",
            )
            references = [case["expectedOutput"], *case["acceptableOutputs"]]
            require(
                row["references"]
                == [
                    formats.serialize(case["input"], action, fmt)
                    for action in references
                ]
                and [
                    formats.parse(text, case["input"], fmt).model_dump()
                    for text in row["references"]
                ]
                == references,
                "Frozen format references do not reconstruct the same admitted actions.",
            )
            messages[case["id"]], prefixes[case["id"]] = row["messages"], values[:start]
            if fmt == "json":
                require(
                    prefixes[case["id"]] == full_prefixes[case["id"]],
                    "JSON matched prefix differs from B2 full input.",
                )
            else:
                require(
                    messages[case["id"]][1] == views["json"]["messages"][case["id"]][1],
                    "JSON and Rewrite changed the user evidence window.",
                )
        views[fmt] = {"messages": messages, "prefixes": prefixes}
    if execution:
        cases = [b2._bind_execution(case, manifest) for case in cases]
    registration.update(
        cases=cases,
        views=views,
        exclusions=excluded,
        full_dev_case_ids=full_ids,
        anchor=anchor,
        native_eos=eos,
    )
    return registration


def prepare(protocol_path, execution_path, condition):
    require(condition in CONDITIONS, "Unknown matched condition.")
    registration = load_views(load_protocol(protocol_path))
    execution_path = project_path(execution_path)
    execution = load_json(execution_path)
    registration["inputs"][execution_path] = probe.digest(execution_path)
    registration.update(execution_path=execution_path, execution=execution)
    require(
        execution["version"] == EXECUTION_VERSION
        and execution["protocol_sha256"] == probe.digest(registration["protocol_path"])
        and set(execution["arms"]) == {"json", "rewrite"},
        "Execution must bind both matched arms to this corrected protocol.",
    )
    registry_path, registry = bound_json(
        execution["registry"], execution_path.parent, registration["inputs"]
    )
    previous = load_json(registration["previous_registry"])
    require(
        registry["runtime"] == previous["runtime"]
        and registry["tokenizers"] == previous["tokenizers"]
        and probe.sampling_options(registry_path)
        == probe.sampling_options(registration["previous_registry"])
        and registry["runtime"]["revision"] == RUNTIME_REVISION,
        "Retain the exact B2 runtime, native tokenizer and greedy decoding settings.",
    )
    ids = [execution["arms"][fmt]["candidate_id"] for fmt in ("json", "rewrite")]
    require(
        len(set(ids)) == 2
        and not set(ids) & set(probe.BASELINES)
        and len(registry["candidates"]) == 2
        and {row["id"] for row in registry["candidates"]} == set(ids),
        "Use a two-candidate registry for the two matched final adapters.",
    )
    arms = {}
    for fmt, arm in execution["arms"].items():
        arms[fmt] = validate_arm(registration, fmt, arm)
        model = next(
            row for row in registry["candidates"] if row["id"] == arm["candidate_id"]
        )
        require(
            model["tokenizer_id"] == "gemma26"
            and all(
                model[key] == registration["anchor"][key]
                for key in curriculum.IDENTITY_FIELDS
            ),
            "Matched model changed the registered native template/tokenizer.",
        )
        artifact = arms[fmt]["export"]["artifacts"]["merged_quantized"]
        require(
            project_path(model["model_path"])
            == project_path(artifact["path"], arms[fmt]["export_path"].parent)
            and model["model_sha256"] == artifact["sha256"]
            and model["model_bytes"] == artifact["bytes"]
            and arms[fmt]["export"]["quantized_tensor_type_census"]["merged_quantized"]
            and all(
                type(count) is int and count > 0
                for count in arms[fmt]["export"]["quantized_tensor_type_census"][
                    "merged_quantized"
                ].values()
            ),
            "Registry does not contain this arm's final adapted Q4 artifact.",
        )
        probe.inspect_runtime(registry_path, arm["candidate_id"])
    validate_matched_recipes(arms)
    fmt, constrained = CONDITIONS[condition]
    sources = source_paths()
    binding = {
        "version": VERSION,
        "response_evidence": RESPONSE_EVIDENCE,
        "condition": condition,
        "format": fmt,
        "schema_constrained": constrained,
        "protocol_sha256": probe.digest(registration["protocol_path"]),
        "execution_sha256": probe.digest(execution_path),
        "registry_sha256": probe.digest(registry_path),
        "release_manifest_sha256": probe.digest(
            registration["release"] / "manifest.json"
        ),
        "registered_requests": 286,
        "full_dev_registered_requests": 289,
        "registered_coverage": {
            "eligible": 286,
            "registered": 289,
            "input_exclusions": 3,
        },
        "excluded_case_ids": [row["case_id"] for row in registration["exclusions"]],
        "messages_sha256": {
            identity: session_hash(messages)
            for identity, messages in registration["views"][fmt]["messages"].items()
        },
        "prefix_payload_sha256": registration["inputs"][
            registration["release"] / f"tokenized/gemma26/{fmt}-dev.jsonl.gz"
        ],
        "artifact_bindings": {
            name: {
                "export_receipt_sha256": probe.digest(value["export_path"]),
                "training_configuration_sha256": probe.digest(
                    value["configuration_path"]
                ),
                "fresh_reload_manifest_sha256": probe.digest(value["reload_path"]),
            }
            for name, value in arms.items()
        },
        "grading_callback": "probe_passive_gemma_matched.grade_response: unchanged API format parser/source execution; separate unmodified sampled-content format diagnostics",
        "snapshot_sha256": {path.name: probe.digest(path) for path in sources},
    }
    for path in sources:
        registration["inputs"][path] = probe.digest(path)
    prepared = {
        "cases": registration["cases"],
        **registration["views"][fmt],
        "source_files": sources,
        "reused_results": {},
        "binding": binding,
        "summarize": summarize,
    }
    registration.update(
        arms=arms, registry_path=registry_path, binding=binding, prepared=prepared
    )
    curriculum.verify_inputs(registration["inputs"])
    return registration


def run_condition(registration, condition, output, stage, editor_state):
    output = curriculum.new_output(output, registration["release"])
    if stage == "prepare":
        output.mkdir(parents=True)
        snapshot(registration, output)
        write_new(
            output / "matched-receipt.json",
            {**registration["binding"], "status": "prepared_generation_unrun"},
        )
        return
    fmt, constrained = CONDITIONS[condition]
    args = SimpleNamespace(
        release=registration["release"],
        registry=registration["registry_path"],
        candidate=registration["execution"]["arms"][fmt]["candidate_id"],
        output=output,
        stage=stage,
        timeout=120,
        unconstrained=not constrained,
        sampling_config=None,
        diagnostic_protocol=None,
        editor_state=editor_state,
    )
    error = None
    try:
        isolated_runtime(fmt).run(args, prepared=registration["prepared"])
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    if output.exists():
        try:
            curriculum.verify_inputs(registration["inputs"])
            snapshot(registration, output)
            results_path = output / "results.jsonl"
            rows = original.read_rows(results_path) if results_path.exists() else []
            validate_results(
                registration["cases"], rows, registration["views"][fmt]["prefixes"], fmt
            )
            with (output / "source-execution.jsonl").open("x") as stream:
                for value in curriculum.source_execution(registration["cases"], rows):
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            status = load_json(output / "manifest.json")["status"]
        except (Exception, KeyboardInterrupt) as caught:
            error, status = error or caught, "failed"
        write_new(
            output / "matched-receipt.json",
            {
                **registration["binding"],
                "status": "failed" if error else status,
                "error": f"{type(error).__name__}: {error}" if error else None,
                "files_sha256": {
                    path.name: probe.digest(path)
                    for path in output.iterdir()
                    if path.is_file()
                },
                "runtime_manifest_version": "passive-b2-runtime-v1 (isolated shared runner; explicit frozen format callback)",
            },
        )
    if error:
        raise error


def report(protocol_path, execution_path, runs, output):
    registration = prepare(protocol_path, execution_path, "matched-json-unconstrained")
    runs = project_path(runs)
    output = curriculum.new_output(output, registration["release"])
    summaries, results, provenance = {}, {}, {}
    for condition, (fmt, constrained) in CONDITIONS.items():
        path = runs / condition
        receipt_path = path / "matched-receipt.json"
        rows, status, manifest = [], "missing_run", {}
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            expected = {
                **registration["binding"],
                "condition": condition,
                "format": fmt,
                "schema_constrained": constrained,
                "messages_sha256": {
                    identity: session_hash(messages)
                    for identity, messages in registration["views"][fmt][
                        "messages"
                    ].items()
                },
                "prefix_payload_sha256": registration["inputs"][
                    registration["release"] / f"tokenized/gemma26/{fmt}-dev.jsonl.gz"
                ],
            }
            require(
                all(receipt.get(key) == value for key, value in expected.items()),
                "Run receipt belongs to another matched experiment/arm.",
            )
            for name, value in receipt["files_sha256"].items():
                read_bound(
                    {"path": name, "sha256": value}, path, registration["inputs"]
                )
            require(
                "manifest.json" in receipt["files_sha256"], "Unbound runtime manifest."
            )
            manifest = load_json(path / "manifest.json")
            candidate = registration["execution"]["arms"][fmt]["candidate_id"]
            require(
                manifest["benchmark"] == expected
                and manifest["candidate"] == candidate
                and manifest["stage"] == "screen"
                and manifest["schema_constrained"] == constrained
                and manifest["case_ids"]
                == [case["id"] for case in registration["cases"]]
                and manifest["registry_sha256"] == expected["registry_sha256"]
                and manifest["sampling"]
                == probe.sampling_options(registration["registry_path"]),
                "Run changed the matched cohort, grammar or sampling condition.",
            )
            for name, value in manifest["benchmark"]["snapshot_sha256"].items():
                read_bound(
                    {"path": name, "sha256": value},
                    path / "implementation",
                    registration["inputs"],
                )
            read_bound(
                {"path": "model-registry.json", "sha256": expected["registry_sha256"]},
                path / "implementation",
                registration["inputs"],
            )
            result_path = path / "results.jsonl"
            if result_path.exists():
                require(
                    "results.jsonl" in receipt["files_sha256"],
                    "Unbound raw results appeared.",
                )
                rows = original.read_rows(result_path)
            require(
                manifest["completed_requests"] == len(rows),
                "Run completion census differs.",
            )
            if receipt["status"] == "completed":
                require(
                    manifest["status"] == "completed"
                    and len(rows) == 286
                    and manifest["template_checks"] == {"requests": 286, "passed": 286},
                    "A completed matched run requires all286 exact native runtime prefixes and result rows.",
                )
            status = receipt["status"]
            provenance[condition] = {
                "receipt_sha256": probe.digest(receipt_path),
                "status": status,
                "sampled_peak_server_rss_bytes": manifest.get(
                    "sampled_peak_server_rss_bytes"
                ),
            }
        elif path.exists():
            raise ValueError(f"Run has no immutable completion/failure receipt: {path}")
        validate_results(
            registration["cases"], rows, registration["views"][fmt]["prefixes"], fmt
        )
        summaries[condition] = {
            **summarize(registration["cases"], rows),
            "run_status": status,
            "actual_memory_and_swap": gemma.resource_observations(manifest),
            "correct_preserves": curriculum.correct_preserves(
                registration["cases"], rows
            ),
        }
        results[condition] = rows
    curriculum.verify_inputs(registration["inputs"])
    output.mkdir(parents=True)
    snapshot(registration, output)
    primary = {name: results[name] for name in list(CONDITIONS)[:2]}
    write_new(
        output / "summary.json",
        {
            "version": VERSION,
            "protocol_sha256": probe.digest(registration["protocol_path"]),
            "execution_sha256": probe.digest(registration["execution_path"]),
            "registered_new_requests": 858,
            "registered_requests_per_condition": 286,
            "full_dev_registered_requests": 289,
            "matched_input_exclusions": registration["exclusions"],
            "runs": summaries,
            "primary_unconstrained_comparison": gemma.quality_metrics(
                b2.compare_runs(registration["cases"], primary)
            ),
            "metric_scope": RESPONSE_EVIDENCE["metric_scope"],
            "timing_scope": "Fresh HTTP measurements include all measured failures and exclude grading; sequential thermal/resource/editor state is uncontrolled.",
            "raw_format_scope": "Strict parse of unmodified __verbose.content, separately from API message content; no fence stripping or repair in any arm. Native stop50 is never credited.",
            "registered_conditions": list(CONDITIONS),
            "constrained_json_diagnostic": "Reported separately; never an advantage attributed to unconstrained format skill.",
            "provenance": provenance,
            "scope": "Fixed exposed development; all final arms and failures retained. No deployment promotion or T3 anchor selection.",
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
