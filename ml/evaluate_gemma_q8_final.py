"""One full DEV471 and separate H24 for the registered direct Q8 product."""

import argparse
import json
from pathlib import Path
import tempfile

import evaluate_gemma_final as development
import evaluate_passive_actions_v3 as grading
import evaluate_small_actions_final as summary
import gemma_final_test_runtime as gemma
import gemma_q8_product as q8

VERSION = "gemma26-final-q8-development-1"
require, read, sha, write, binding, bound = (gemma.require, gemma.read, gemma.sha,
                                           gemma.write, gemma.binding, gemma.bound)


def deployed_context(evaluation, model_metadata, deployment):
    context = development.deployed_context(evaluation, model_metadata, deployment)
    receipt = read(deployment)
    require(receipt["numerical_regime"] == q8.REGIME, "Q8 evaluator requires an explicit verified Q8 deployment.")
    registration = q8.verify_registration(bound(receipt["q8_registration"]), context["model"]["identity"])
    cohort = next((v for v in registration["cohorts"].values()
                   if v["preparation"] == binding(Path(evaluation) / "manifest.json")), None)
    require(cohort is not None and cohort["requests"] == binding(Path(evaluation) / "requests-json.jsonl")
            and [r["case_id"] for r in context["requests"]] == cohort["case_ids"]
            and len(context["requests"]) == cohort["count"], "Q8 evaluator changed the registered full cohort/order.")
    return context


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
        "numerical_regime": q8.REGIME, "schema_constrained": True,
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
            and info["numerical_regime"] == q8.REGIME and info["schema_constrained"] is True,
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


def verify_summary(reference):
    from build_small_final_freeze_actions_v3 import verify_dev_cohort_inputs
    result = read(bound(reference))
    report_path = bound(result["report"])
    report = read(report_path)
    prepared = read(bound(report["preparation"]))
    mapping = read(bound(result["cohort_mapping"]))
    config_path = bound(prepared["model_preparation"])
    gemma.verify_fallback(config_path)
    require(prepared.get("gemma_final_version") == development.VERSION and prepared["final_test_bodies_read"] is False
            and prepared["dataset_manifest"]["sha256"] == gemma.evidence.RELEASE_SHA
            and mapping["release_manifest"]["sha256"] == gemma.evidence.RELEASE_SHA
            and report["status"] == "graded_development" and report["final_test"] is False
            and report["format"] == "json" and report["condition"] == "canonical_full_context"
            and report["original_development_cases"] == 471 and report["excluded_before_inference"] == 0,
            "Gemma quality requires the unchanged canonical322 and supplemental149 DEV cohort.")
    require(mapping["counts"] == {"canonical_dev322": 322, "supplemental_dev149": 149}, "Gemma DEV cohort counts changed.")
    verify_dev_cohort_inputs(report["preparation"], mapping["evaluation_manifest"])
    run = read(bound(report["inference_run"]))
    require(report["schema_constrained"] is True and report["numerical_regime"] == q8.REGIME,
            "Q8 quality summary must retain its explicit product numerical regime.")
    verify_full_run(bound(report["inference_run"]).parent.parent)
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


def verify_full_run(directory):
    directory = Path(directory).resolve()
    manifest, result = read(directory / "manifest.json"), read(directory / "run.json")
    require(manifest["version"] == result["version"] == VERSION
            and manifest["status"] == result["status"] == "inference_complete"
            and result["final_test_bodies_read"] is False and set(result["scopes"]) == {"dev", "history"},
            "Q8 full DEV/history run is incomplete or failed after a child completed.")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*")
              if p.is_file() and p != directory / "manifest.json"}
    require(set(manifest["files_sha256"]) == actual and "failure.json" not in actual,
            "Q8 complete publication inventory changed.")
    grading.verify_files(directory, manifest)
    value = q8.verify_registration(bound(result["registration"]))
    require(read(directory / "registration.json") == value, "Q8 outer registration changed.")
    for scope, count in (("dev", 471), ("history", 24)):
        evaluation = bound(value["cohorts"][scope]["preparation"]).parent
        info = verify_deployed_run(evaluation, directory / scope)
        require(result["scopes"][scope] == {"requested": count, "observed": count,
                    "run": binding(directory / scope / "run.json")}
                and info["deployment"] == result["deployment"], "Q8 full cohort or deployment changed.")
    import infer_deployed_task_test_actions_v3 as auxiliary
    context = deployed_context(bound(value["cohorts"]["dev"]["preparation"]).parent,
                               bound(value["model_metadata"]), bound(result["deployment"]))
    auxiliary.verify_runtime(read(directory / "final-runtime.json"), context)
    return result


def infer_full(registration, deployment, output):
    value = q8.verify_registration(registration)
    output = Path(output).resolve()
    require(not output.exists() and read(deployment)["q8_registration"] == binding(registration),
            "Use one fresh registered Q8 DEV/history attempt.")
    output.mkdir(parents=True)
    write(output / "registration.json", value)
    result = {"version": VERSION, "status": "running", "registration": binding(registration),
              "deployment": binding(deployment), "scopes": {}, "final_test_bodies_read": False}
    try:
        for scope in ("dev", "history"):
            evaluation = bound(value["cohorts"][scope]["preparation"]).parent
            info = infer_deployed(evaluation, bound(value["model_metadata"]), deployment, output / scope)
            result["scopes"][scope] = {"requested": info["requested"], "observed": info["observed"],
                                      "run": binding(output / scope / "run.json")}
        import infer_deployed_task_test_actions_v3 as auxiliary
        context = deployed_context(bound(value["cohorts"]["dev"]["preparation"]).parent,
                                   bound(value["model_metadata"]), deployment)
        endpoint = context["gateway"]["endpoint"]
        runtime = {"endpoint": endpoint}
        for key, route in (("health", "/health"), ("props", "/props"), ("lora_adapters", "/lora-adapters")):
            runtime[key] = auxiliary.http.request(endpoint, route, timeout=30)
        write(output / "final-runtime.json", runtime)
        auxiliary.verify_runtime(runtime, context)
        result["status"] = "inference_complete"
        write(output / "run.json", result)
        write(output / "manifest.json", {"version": VERSION, "status": "inference_complete",
            "files_sha256": {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}})
        verify_full_run(output)
        return result
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write(output / "failure.json", result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "deployment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    print(json.dumps(infer_full(**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
