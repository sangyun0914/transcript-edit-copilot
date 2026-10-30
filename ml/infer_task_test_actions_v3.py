"""Portable native final-test inference from an explicitly authorized label-free bundle."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import assemble_task_test_actions_v3 as assembly
import infer_gemma_actions_v3 as native_helpers
import passive_actions as contract
import passive_formats_actions_v3 as formats
import prepare_actions_v3_conditions as conditions
import prepare_gemma_actions_v3 as preparation
import prepare_passive_actions_final_test as sealed_helpers
import train_gemma_actions_v3 as sft

VERSION = "task-test-actions-v3-native-inference-1"
BUNDLE_VERSION = "task-test-actions-v3-execution-bundle-1"
SCOPE = "final_test"
ROOT = Path(__file__).resolve().parents[1]
FULL_COHORT_FAMILIES = {"small-actions-final-v1", "gemma26-final-native-v1"}
require, file_sha, read = sft.require, sft.digest, sft.read
value_hash, write = preparation.value_hash, preparation.write


def sources():
    values = sealed_helpers.source_closure([Path(__file__)])
    values[str(contract.PROMPT_PATH.relative_to(ROOT))] = file_sha(contract.PROMPT_PATH)
    return dict(sorted(values.items()))


def execution_runtime():
    if (ROOT / assembly.MANIFEST).is_file():
        return assembly.load_runtime(ROOT)
    return {"runtime_manifest_sha256": None, "source_sha256": sources()}


def validate_requests(requests, specification):
    run = specification["run"]
    vocab_size = specification.get("vocab_size", 262144)
    require(type(vocab_size) is int and vocab_size > 0, "Invalid native vocabulary budget.")
    input_limit = formats.INPUT_LIMIT
    if specification.get("runtime_family") in FULL_COHORT_FAMILIES:
        require(specification["physical_context_tokens"] == 8192 and specification["inference_prefix_limit"] == 7680
                and specification["output_tokens"] == 512, "Small inference context policy changed.")
        input_limit = specification["inference_prefix_limit"]
    require([row["case_id"] for row in requests] == specification["case_ids"]
            and len({row["case_id"] for row in requests}) == len(requests), "Final-test request census changed.")
    keys = {"case_id", "split", "run_id", "format", "condition", "condition_sha256", "input", "input_sha256",
            "messages", "messages_sha256", "prefix_ids", "prefix_ids_sha256", "freeze_sha256"}
    for row in requests:
        require(set(row) == keys and row["split"] == SCOPE and row["run_id"] == run["id"]
                and row["format"] == run["format"] and row["freeze_sha256"] == specification["freeze_sha256"],
                "Inference requires label-free final_test requests for this run.")
        require(row["condition"] == run["condition"]
                and row["condition_sha256"] == value_hash(run["condition"])
                and row["input_sha256"] == value_hash(contract.wire_snapshot(row["input"]))
                and row["messages"] == conditions.expected_messages(row["input"], run["format"], run["condition"])
                and row["messages_sha256"] == value_hash(row["messages"])
                and row["prefix_ids_sha256"] == value_hash(row["prefix_ids"]), "Portable request binding changed.")
        require(0 < len(row["prefix_ids"]) <= input_limit
                and all(type(token) is int and 0 <= token < vocab_size for token in row["prefix_ids"]),
                "Invalid final-test native token budget.")


def load_bundle(directory, expected_sha256, *, expected_runtime=None):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    require(file_sha(manifest_path) == expected_sha256, "Execution bundle differs from the local authorization.")
    manifest = read(manifest_path)
    require(manifest["version"] == BUNDLE_VERSION and manifest["scope"] == SCOPE
            and manifest["status"] == "authorized_label_free_execution"
            and set(manifest["files_sha256"]) == {"execution.json", "requests.jsonl"}, "Invalid portable execution bundle.")
    for name, expected in manifest["files_sha256"].items():
        require(file_sha(directory / name) == expected, "Execution payload changed.")
    spec = read(directory / "execution.json")
    runtime = execution_runtime() if expected_runtime is None else expected_runtime
    require(spec["scope"] == SCOPE and spec["local_gate_passed"] is True
            and spec["source_sha256"] == runtime["source_sha256"]
            and spec["runtime_manifest_sha256"] == runtime["runtime_manifest_sha256"]
            and spec["run"]["role"] in {"native_base", "base_icl", "sft", "sft_dpo"}
            and spec["schema_constrained"] is False, "Execution authorization/runtime source changed.")
    if runtime["runtime_manifest_sha256"] is None:
        require(spec["engineering_fixture_only"] is True, "Real inference requires a frozen model-specific runtime.")
    else:
        require(all(spec[key] == runtime[key] for key in ("configuration_sha256", "preparation_manifest_sha256")),
                "Execution runtime belongs to another model preparation.")
    require(spec.get("runtime_family") == runtime.get("runtime_family"), "Execution model runtime family changed.")
    if spec.get("runtime_family") in FULL_COHORT_FAMILIES:
        require(spec["inference_policy_sha256"] == runtime["inference_policy_sha256"], "Execution inference policy changed.")
    require(spec["requests_sha256"] == manifest["files_sha256"]["requests.jsonl"], "Request authorization changed.")
    requests = preparation.rows(directory / "requests.jsonl")
    validate_requests(requests, spec)
    return spec, requests


def native_identity(config, training=None, reloaded=None, adapter_version=sft.VERSION):
    if config.get("version") == "small-actions-final-v1":
        import small_final_test_runtime as small
        return small.native_identity(config, training, reloaded)
    require(bool(training) == bool(reloaded), "Both trained and independently reloaded adapters are required.")
    identity = {"kind": "native_base", "repository": config["base"]["repository"],
                "revision": config["base"]["revision"], "native_pin_sha256": config["native_pin_sha256"]}
    if training:
        identity.update(native_helpers.adapter_identity(training, reloaded, adapter_version, config))
    return identity


def verify_loaded_configuration(spec, config, config_path, verified, identity):
    require(file_sha(config_path) == spec["configuration_sha256"]
            and verified["preparation_sha256"] == spec["preparation_manifest_sha256"]
            and verified["environment"]["matched"] is True and verified["native_assets_verified"] is True
            and config["format"] == spec["run"]["format"]
            and config["generation"] == spec["generation"]
            and config["libraries"] == spec["libraries"]
            and identity == spec["model_identity"], "Native model/configuration differs from the frozen execution.")


def observation_identity(request, spec):
    row = {key: request[key] for key in ("case_id", "input_sha256", "messages_sha256", "prefix_ids_sha256",
                                       "condition_sha256", "format", "run_id", "freeze_sha256")}
    row.update(split=SCOPE, model_identity=spec["model_identity"], request_sha256=value_hash(request))
    return row


def failure(request, spec, error, stage, *, identity_verified=False, attempted=False):
    return {**observation_identity(request, spec), "status": "request_failure", "error_type": type(error).__name__,
            "error": str(error), "failure_stage": stage, "model_identity_verified": identity_verified,
            "generation_attempted": attempted, "generation_ms": None, "timing_source": "not_measured"}


def append(stream, row):
    stream.write(formats.compact(row) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run(bundle_path, bundle_sha256, config_path, output, *, training=None, reloaded=None, assets=None):
    spec, requests = load_bundle(bundle_path, bundle_sha256)
    output = Path(output).resolve()
    protected = [Path(bundle_path).resolve(), Path(config_path).resolve().parent]
    protected.extend(Path(path).resolve() for path in (training, reloaded) if path is not None)
    require(not output.exists() and not any(output == path or output.is_relative_to(path) for path in protected),
            "Use a new inference directory outside frozen inputs and model runs.")
    output.mkdir(parents=True)
    info = {"version": VERSION, "scope": SCOPE, "status": "registered", "bundle_sha256": bundle_sha256,
            "freeze_sha256": spec["freeze_sha256"], "sealed_manifest_sha256": spec["sealed_manifest_sha256"],
            "requests_sha256": spec["requests_sha256"], "run_id": spec["run"]["id"],
            "configuration_sha256": spec["configuration_sha256"], "model_identity": spec["model_identity"],
            "source_sha256": spec["source_sha256"], "runtime_manifest_sha256": spec["runtime_manifest_sha256"],
            "controller_source_sha256": spec["controller_source_sha256"],
            "requested": len(requests), "started_at": datetime.now(timezone.utc).isoformat(),
            "numerical_regime": "native_bfloat16_with_optional_float32_lora", "schema_constrained": False,
            "final_test_bodies_read": True, "reference_labels_available_to_predictor": False,
            "actual_model_identity": None, "model_identity_verified": False}
    write(output / "run-registration.json", info)
    emitted, fatal, loaded, backend_trace = 0, None, False, None
    started = time.monotonic()
    with (output / "observations.jsonl").open("x") as stream:
        try:
            if requests:
                require(file_sha(config_path) == spec["configuration_sha256"], "Frozen inference configuration changed.")
                if spec.get("runtime_family") == "small-actions-final-v1":
                    import small_final_test_runtime as small
                    config, identity, torch, model, tokenizer, device, runtime = small.load_runtime(
                        spec, config_path, training, reloaded, assets)
                elif spec.get("runtime_family") == "gemma26-final-native-v1":
                    import gemma_final_test_runtime as gemma
                    config, identity, torch, model, tokenizer, device, runtime = gemma.load_runtime(
                        spec, config_path, training, reloaded, assets)
                else:
                    require(assets is None, "The historical native runtime does not accept a small-model asset override.")
                    config, root, bundle, _, _, verified = sft.preflight(config_path)
                    identity = native_identity(config, training, reloaded, spec["adapter_version"])
                    verify_loaded_configuration(spec, config, config_path, verified, identity)
                    expected_adapter = spec["run"]["role"] in {"sft", "sft_dpo"}
                    require(bool(training) == expected_adapter, "Native base/adapter run role changed.")
                    torch, model, tokenizer, device, runtime = sft.gemma.load_runtime(config, root, bundle["tokens"][config["format"]])
                    if training:
                        model, _, _ = sft.load_adapter(model, Path(training) / "adapter",
                                                      read(Path(training) / "adapter-tensors.json"), config, torch, False)
                for request in requests:
                    require(conditions.native_prefix(tokenizer, request["messages"]) == request["prefix_ids"],
                            "Actual native prompt differs from the registered final-test input.")
                loaded = True
                info.update(actual_model_identity=identity, model_identity_verified=True)
                from transformers import GenerationConfig
                generation = GenerationConfig(**config["generation"])
                if runtime.get("native_inference_overlay") is not None:
                    import gemma_native_inference_overlay as overlay
                    require(spec.get("runtime_family") == "gemma26-final-native-v1"
                            and runtime["native_inference_overlay"]["settings"] == overlay.OVERLAY,
                            "Unregistered native runtime overlay.")
                    generation.cache_implementation = "dynamic"
                    generation.disable_compile = True
                    backend_trace = overlay.BackendTrace(overlay.base_model(model))
                    info["native_inference_overlay"] = runtime["native_inference_overlay"]
                    runtime["effective_inference_generation"] = generation.to_dict()
                write(output / "runtime.json", runtime)
                model.eval()
                model.gradient_checkpointing_disable()
                for request in requests:
                    row = observation_identity(request, spec)
                    generation_started = None
                    request_started = time.perf_counter()
                    try:
                        if backend_trace is not None:
                            backend_trace.rows.clear()
                        ids = torch.tensor([request["prefix_ids"]], dtype=torch.long, device=device)
                        torch.cuda.synchronize()
                        generation_started = time.perf_counter()
                        with torch.inference_mode():
                            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation)
                        torch.cuda.synchronize()
                        generation_ms = (time.perf_counter() - generation_started) * 1000
                        tokens = result[0, len(request["prefix_ids"]):].tolist()
                        require(0 < len(tokens) <= formats.OUTPUT_LIMIT, "Native generation exceeded its frozen token budget.")
                        terminal = tokens[-1]
                        row.update(status="completed", output_ids=tokens, terminal_token_id=terminal,
                                   text=tokenizer.decode(tokens, skip_special_tokens=True),
                                   text_raw=tokenizer.decode(tokens, skip_special_tokens=False),
                                   finish_reason="stop" if terminal in generation.eos_token_id else "length",
                                   generation_ms=generation_ms, timing_source="cuda_synchronized_generate",
                                   model_identity_verified=True, generation_attempted=True)
                        if backend_trace is not None:
                            backend_trace.validate(len(request["prefix_ids"]), len(tokens))
                    except Exception as error:
                        generated = {key: row[key] for key in ("output_ids", "text", "text_raw", "terminal_token_id", "finish_reason", "generation_ms") if key in row}
                        row = failure(request, spec, error, "generation", identity_verified=True,
                                      attempted=generation_started is not None)
                        row.update(generated)
                        if isinstance(error, torch.OutOfMemoryError):
                            torch.cuda.empty_cache()
                    if backend_trace is not None:
                        row["backend_trace"] = list(backend_trace.rows)
                    row["controller_wall_ms"] = (time.perf_counter() - request_started) * 1000
                    append(stream, row)
                    emitted += 1
                    print(json.dumps({"event": "final_test_observation", "index": emitted,
                                      "case_id": request["case_id"], "status": row["status"]}), flush=True)
        except BaseException as error:
            fatal = error
            stage = "aborted_before_request" if loaded else "preflight_or_model_loading"
            for request in requests[emitted:]:
                append(stream, failure(request, spec, error, stage, identity_verified=loaded))
                emitted += 1
        finally:
            if backend_trace is not None:
                try:
                    backend_trace.close()
                except Exception as error:
                    fatal = fatal or error
    info.update(status="inference_complete", observed=emitted, seconds=time.monotonic() - started,
                completed_with_startup_or_interruption_failure=fatal is not None)
    if fatal is not None:
        info.update(error_type=type(fatal).__name__, error=str(fatal))
    if loaded:
        info["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    write(output / "run.json", info)
    write(output / "manifest.json", {"version": VERSION, "scope": SCOPE, "status": "inference_complete",
          "files_sha256": {path.name: file_sha(path) for path in sorted(output.iterdir()) if path.is_file()}})
    if isinstance(fatal, (KeyboardInterrupt, SystemExit)):
        raise fatal
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--reloaded", type=Path)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.bundle, args.bundle_sha256, args.config, args.output,
                         training=args.training, reloaded=args.reloaded, assets=args.assets), indent=2))


if __name__ == "__main__":
    main()
