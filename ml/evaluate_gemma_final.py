"""Prepare saved DEV inputs and verify native/separate-LoRA Gemma observations."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import evaluate_passive_actions_v3 as grading
import evaluate_small_actions_final as summary
import gemma_final_test_runtime as gemma
import gemma_final_deployment as deployed

VERSION = "gemma26-final-development-1"
MODEL_VERSION = "gemma26-final-development-model-1"
require, read, sha, write, binding, bound = (gemma.require, gemma.read, gemma.sha,
                                           gemma.write, gemma.binding, gemma.bound)


def register_model(config_path, sft_training, sft_reloaded, output, *, dpo_training=None, dpo_reloaded=None, dpo_preparation=None):
    output = Path(output).resolve()
    require(output.is_relative_to(gemma.ROOT) and not output.exists()
            and all(path is None or not output.is_relative_to(Path(path).resolve()) for path in
                    (Path(config_path).parent, sft_training, sft_reloaded, dpo_training, dpo_reloaded, dpo_preparation)),
            "Use a fresh DEV-only receipt outside immutable model inputs.")
    gemma.verify_fallback(config_path)
    require(all(v is None for v in (dpo_training, dpo_reloaded, dpo_preparation))
            or all(v is not None for v in (dpo_training, dpo_reloaded, dpo_preparation)),
            "Supply the complete same-parent DPO evidence or only SFT.")
    sft = gemma.evidence.verify_trained(config_path, sft_training, sft_reloaded)
    dpo = gemma.evidence.verify_trained(config_path, dpo_training, dpo_reloaded, kind="gemma-actions-v3-dpo",
        parent=sft["identity"], dpo_preparation=dpo_preparation) if dpo_training else None
    model = {"version": MODEL_VERSION, "runtime_family": gemma.FAMILY, "scope": "exposed_development_only",
        "role": "sft_dpo" if dpo else "sft", "identity": (dpo or sft)["identity"],
        "inference_configuration": binding(config_path), "native_preparation": binding(Path(config_path).parent / "manifest.json"),
        "tokenizer": gemma.tokenizer_identity(config_path), "sft_evidence": sft, "dpo_evidence": dpo,
        "dpo_preparation": binding(Path(dpo_preparation) / "configuration.json") if dpo else None,
        "model_selection_performed": False, "body_access_authorized": False, "final_test_bodies_read": False}
    write(output, model)
    return model


def verify_model(path):
    model = read(path)
    require(model.get("version") == MODEL_VERSION and model["runtime_family"] == gemma.FAMILY
            and model["scope"] == "exposed_development_only" and model["body_access_authorized"] is False
            and model["final_test_bodies_read"] is False and model["model_selection_performed"] is False,
            "Use an actual DEV-only Gemma training/reload receipt.")
    config_path = bound(model["inference_configuration"])
    gemma.verify_fallback(config_path)
    require(model["native_preparation"] == binding(config_path.parent / "manifest.json")
            and model["tokenizer"] == gemma.tokenizer_identity(config_path), "Gemma DEV assets changed.")
    parent = model["sft_evidence"]
    require(parent == gemma.evidence.verify_trained(config_path, bound(parent["training"]).parent, bound(parent["reload"]).parent),
            "Gemma DEV SFT/reload identity changed.")
    selected = parent
    if model["dpo_evidence"] is not None:
        selected = model["dpo_evidence"]
        require(model["role"] == "sft_dpo" and selected == gemma.evidence.verify_trained(config_path,
            bound(selected["training"]).parent, bound(selected["reload"]).parent, kind="gemma-actions-v3-dpo",
            parent=parent["identity"], dpo_preparation=bound(model["dpo_preparation"]).parent),
            "Gemma DEV DPO changed its completed parent or reload.")
    else:
        require(model["role"] == "sft" and model["dpo_preparation"] is None, "Unproven Gemma DEV model role.")
    require(model["identity"] == selected["identity"], "Gemma DEV selected identity changed.")
    return model


def prepare(evaluation, config_path, output):
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    original = read(evaluation / "manifest.json")
    require(original["version"] == grading.VERSION and original["final_test_bodies_read"] is False
            and original["excluded"] == 0, "Use an unchanged exposed DEV cohort without exclusions.")
    grading.verify_files(evaluation, original)
    for path, expected in original["sources_sha256"].items():
        require(sha(path) == expected, "Saved DEV preparation source changed.")
    config, _ = gemma.verify_fallback(config_path)
    from transformers import AutoTokenizer
    identity = gemma.tokenizer_identity(config_path)
    tokenizer = AutoTokenizer.from_pretrained(identity["path"], local_files_only=True, trust_remote_code=False)
    requests = gemma.evidence.prep.rows(evaluation / "requests-json.jsonl")
    for row in requests:
        require(row["split"] == "dev" and row["messages"] == gemma.evidence.prep.formats.messages(row["input"]),
                "Saved DEV input or H0/H3/H8 messages changed.")
        prefix = gemma.evidence.conditions.native_prefix(tokenizer, row["messages"])
        require(len(prefix) <= 3584, "Complete DEV must fit without cropping or excluding rows.")
        row.update(prefix_ids=prefix, prefix_ids_sha256=gemma.digest(prefix))
    require(not output.exists(), "Use a fresh Gemma DEV preparation.")
    output.mkdir(parents=True)
    for name in ("cases.jsonl", "exclusions.jsonl"):
        shutil.copyfile(evaluation / name, output / name)
    gemma.evidence.prep.write_rows(output / "requests-json.jsonl", requests)
    write(output / "manifest.json", {**original, "gemma_final_version": VERSION,
        "parent_evaluation": binding(evaluation / "manifest.json"), "model_preparation": binding(config_path),
        "tokenizer": identity, "native_model": config["base"],
        "files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "sources_sha256": {**original["sources_sha256"], str(Path(__file__).resolve()): sha(__file__)}})
    return {"requests": len(requests), "max_prefix_tokens": max(len(r["prefix_ids"]) for r in requests),
            "saved_cases_byte_identical": sha(output / "cases.jsonl") == original["files_sha256"]["cases.jsonl"]}


def verify_summary(reference):
    from build_small_final_freeze_actions_v3 import verify_dev_cohort_inputs
    result = read(bound(reference))
    report_path = bound(result["report"])
    report = read(report_path)
    prepared = read(bound(report["preparation"]))
    mapping = read(bound(result["cohort_mapping"]))
    config_path = bound(prepared["model_preparation"])
    gemma.verify_fallback(config_path)
    require(prepared.get("gemma_final_version") == VERSION and prepared["final_test_bodies_read"] is False
            and prepared["dataset_manifest"]["sha256"] == gemma.evidence.RELEASE_SHA
            and mapping["release_manifest"]["sha256"] == gemma.evidence.RELEASE_SHA
            and report["status"] == "graded_development" and report["final_test"] is False
            and report["format"] == "json" and report["condition"] == "canonical_full_context"
            and report["original_development_cases"] == 471 and report["excluded_before_inference"] == 0,
            "Gemma quality requires the unchanged canonical322 and supplemental149 DEV cohort.")
    require(mapping["counts"] == {"canonical_dev322": 322, "supplemental_dev149": 149}, "Gemma DEV cohort counts changed.")
    verify_dev_cohort_inputs(report["preparation"], mapping["evaluation_manifest"])
    run = read(bound(report["inference_run"]))
    require(run["requested"] == run["observed"] == 471 and run["status"] == "inference_complete"
            and run["final_test_bodies_read"] is False and run["model_identity"] == report["model_identity"]
            and run["numerical_regime"] == report["numerical_regime"]
            and run["schema_constrained"] is report["schema_constrained"], "Gemma DEV run identity or census changed.")
    grading.verify_observations(bound(report["inference_run"]).parent,
                                bound(report["preparation"]).parent / "requests-json.jsonl", "json")
    if not report["schema_constrained"]:
        require(report["numerical_regime"] == "native_bfloat16_with_optional_float32_lora"
                and run["configuration_sha256"] == sha(config_path), "Gemma native DEV preparation changed.")
    else:
        verify_deployed_run(bound(report["preparation"]).parent, bound(report["inference_run"]).parent)
    with tempfile.TemporaryDirectory(prefix="gemma-dev-summary-") as temporary:
        recomputed = summary.summarize(report_path, bound(result["cohort_mapping"]), Path(temporary) / "summary.json")
    require(recomputed == result, "Gemma DEV summary differs from its actual graded rows.")
    return result, report


def deployed_context(evaluation, model_metadata, deployment):
    evaluation = Path(evaluation)
    prepared = read(evaluation / "manifest.json")
    require(prepared.get("gemma_final_version") == VERSION and prepared["final_test_bodies_read"] is False,
            "Only saved exposed Gemma DEV requests are allowed.")
    grading.verify_files(evaluation, prepared)
    model = verify_model(model_metadata)
    require(model["inference_configuration"] == prepared["model_preparation"] and model["tokenizer"] == prepared["tokenizer"],
            "Product DEV uses another native preparation or tokenizer.")
    config = read(bound(model["inference_configuration"]))
    deployed.validate_deployment(binding(deployment), model, config)
    gateway, _, _ = deployed.deployment_check(Path(deployment).parent)
    import infer_deployed_task_test_actions_v3 as auxiliary
    requests = gemma.evidence.prep.rows(evaluation / "requests-json.jsonl")
    require(all(r["split"] == "dev" and len(r["prefix_ids"]) <= 3584
                and r["messages"] == gemma.evidence.prep.formats.messages(r["input"]) for r in requests),
            "Product DEV changed its saved canonical input.")
    return {"gateway": gateway, "policy": {"request_settings": auxiliary.request_settings(gateway), "context_tokens": 4096},
            "model": model, "identity": deployed.product_identity(gateway, model), "requests": requests,
            "tokenizer": model["tokenizer"]}


def infer_deployed(evaluation, model_metadata, deployment, output):
    import infer_deployed_task_test_actions_v3 as auxiliary
    context = deployed_context(evaluation, model_metadata, deployment)
    output = Path(output)
    require(not output.exists(), "Use a fresh product DEV run.")
    output.mkdir(parents=True)
    gateway, requests = context["gateway"], context["requests"]
    runtime = {"endpoint": gateway["endpoint"]}
    for key, route in (("health", "/health"), ("props", "/props"), ("lora_adapters", "/lora-adapters")):
        runtime[key] = auxiliary.http.request(gateway["endpoint"], route, timeout=30)
    write(output / "runtime.json", runtime)
    error = None
    try:
        auxiliary.verify_runtime(runtime, context)
    except Exception as caught:
        error = caught
    records = []
    for row in requests:
        observed = {key: row[key] for key in ("case_id", "format", "input_sha256", "messages_sha256", "prefix_ids_sha256")}
        observed.update(model_identity=context["identity"], generation_ms=None, http_evidence={})
        try:
            if error is not None:
                raise error
            apply, _, body = auxiliary.expected_calls(row, context)
            capture = observed["http_evidence"]
            prompt = auxiliary.response(auxiliary.capture_request(capture, "apply_template", gateway["endpoint"],
                "/apply-template", apply, 30))["prompt"]
            _, tokenize, _ = auxiliary.expected_calls(row, context, prompt)
            tokens = auxiliary.response(auxiliary.capture_request(capture, "tokenize", gateway["endpoint"],
                "/tokenize", tokenize, 30))["tokens"]
            require(auxiliary.same_prefix(tokens, row["prefix_ids"]), "Gemma product prefix differs from native IDs.")
            raw = auxiliary.capture_request(capture, "completion", gateway["endpoint"], "/v1/chat/completions", body, 30)
            text, finish = auxiliary.completion_value(raw, context)
            observed.update(status="completed", text=text, finish_reason=finish,
                            native_token_parity=True, generation_ms=raw["elapsed_ms"])
        except Exception as caught:
            observed.update(status="request_failure", error_type=type(caught).__name__, error=str(caught))
        records.append(observed)
        gemma.evidence.sft.curriculum.append(output / "observations.jsonl", observed)
    result = {"version": VERSION, "status": "inference_complete", "format": "json", "requested": len(requests),
        "observed": len(records), "requests_sha256": sha(Path(evaluation) / "requests-json.jsonl"),
        "evaluation": binding(Path(evaluation) / "manifest.json"), "model_metadata": binding(model_metadata),
        "deployment": binding(deployment), "model_identity": context["identity"],
        "numerical_regime": "native_derived_q4_base_fp32_lora", "schema_constrained": True,
        "final_test_bodies_read": False, "request_settings": context["policy"]["request_settings"],
        "measurement": "HTTP completion only; no UI latency claim", "implementation_sha256": sha(__file__)}
    write(output / "run.json", result)
    gemma.evidence.sft.manifest(output, "inference_complete", VERSION)
    verify_deployed_run(evaluation, output)
    return result


def verify_deployed_run(evaluation, run):
    import infer_deployed_task_test_actions_v3 as auxiliary
    run = Path(run)
    info = read(run / "run.json")
    context = deployed_context(evaluation, bound(info["model_metadata"]), bound(info["deployment"]))
    records, _ = grading.verify_observations(run, Path(evaluation) / "requests-json.jsonl", "json")
    require(info["implementation_sha256"] == sha(__file__) and info["model_identity"] == context["identity"]
            and info["request_settings"] == context["policy"]["request_settings"]
            and info["version"] == VERSION and info["evaluation"] == binding(Path(evaluation) / "manifest.json")
            and info["numerical_regime"] == "native_derived_q4_base_fp32_lora" and info["schema_constrained"] is True,
            "Product DEV implementation, model or evaluation changed.")
    runtime = read(run / "runtime.json")
    startup_ok = True
    try:
        auxiliary.verify_runtime(runtime, context)
    except ValueError:
        startup_ok = False
    for request, record in zip(context["requests"], records, strict=True):
        if record["status"] != "completed":
            require(record["status"] == "request_failure" and record.get("error"), "Unknown product DEV disposition.")
            for capture in record["http_evidence"].values():
                auxiliary.verify_http_capture(capture["response"])
            continue
        require(startup_ok and record["native_token_parity"] is True, "Completed product DEV lacks loaded-model validation.")
        capture = record["http_evidence"]
        apply, _, body = auxiliary.expected_calls(request, context)
        prompt = auxiliary.response(capture["apply_template"]["response"])["prompt"]
        _, tokenize, _ = auxiliary.expected_calls(request, context, prompt)
        for key, expected in (("apply_template", apply), ("tokenize", tokenize), ("completion", body)):
            require(capture[key]["request"] == expected, "Product DEV sent a different request or LoRA selection.")
        require(auxiliary.same_prefix(auxiliary.response(capture["tokenize"]["response"])["tokens"], request["prefix_ids"])
                and auxiliary.completion_value(capture["completion"]["response"], context)
                == (record["text"], record["finish_reason"]), "Product DEV raw response/prefix changed.")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("evaluation", "config-path", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    model = sub.add_parser("register-model")
    for name in ("config-path", "sft-training", "sft-reloaded", "output"):
        model.add_argument("--" + name, type=Path, required=True)
    for name in ("dpo-training", "dpo-reloaded", "dpo-preparation"):
        model.add_argument("--" + name, type=Path)
    infer = sub.add_parser("infer-deployed")
    for name in ("evaluation", "model-metadata", "deployment", "output"):
        infer.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command").replace("-", "_")
    print(json.dumps(globals()[command](**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
