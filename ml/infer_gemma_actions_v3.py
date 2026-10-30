"""Run frozen, label-free development requests with the pinned native Gemma runtime."""

import argparse
import json
from pathlib import Path
import time

import prepare_gemma_actions_v3 as prep
import prepare_actions_v3_conditions as conditions
import train_gemma_actions_v3 as sft

VERSION = "gemma-actions-v3-native-inference-1"
HISTORICAL_VERSION = "passive-gemma-actions-sft-v1"
require, digest, read, write = sft.require, sft.digest, sft.read, sft.write


def validate_requests(requests, format_name):
    require(
        requests and len({row["case_id"] for row in requests}) == len(requests),
        "Require nonempty unique development requests.",
    )
    for row in requests:
        if "condition" in row:
            conditions.validate_request(row)
        require(
            row["split"] == "dev" and row["format"] == format_name,
            "This runner accepts exposed development requests only.",
        )
        require(
            not {
                "primary",
                "acceptable",
                "accepted_predictions",
                "judgment",
                "rationale",
            }.intersection(row),
            "Evaluation references must not enter the inference bundle.",
        )
        require(
            row["input_sha256"]
            == prep.value_hash(prep.formats.contract.wire_snapshot(row["input"]))
            and row["messages_sha256"] == prep.value_hash(row["messages"])
            and row["prefix_ids_sha256"] == prep.value_hash(row["prefix_ids"]),
            "Request input, prompt or native-token binding changed.",
        )
        require(
            row["messages"]
            == (
                conditions.expected_messages(
                    row["input"], format_name, row["condition"]
                )
                if "condition" in row
                else prep.formats.messages(row["input"], format_name)
            ),
            "Unexpected prompt condition.",
        )
        require(
            0 < len(row["prefix_ids"]) <= prep.formats.INPUT_LIMIT
            and all(
                type(token) is int and 0 <= token < 262144
                for token in row["prefix_ids"]
            ),
            "Input exceeds its registered native budget.",
        )
    require(
        len({row.get("condition_sha256") for row in requests}) == 1,
        "A run must use one registered condition.",
    )


def adapter_identity(training, reloaded, version, config):
    receipt, info = sft.verify_reload(training, reloaded, version=version)
    saved = read(Path(training) / "configuration.json")
    saved_format = saved.get("format")
    if version == HISTORICAL_VERSION:
        require(
            saved["version"] == HISTORICAL_VERSION
            and "format" not in saved
            and saved["contract_version"] == prep.formats.contract.VERSION,
            "Only the documented historical JSON action adapter has an implicit format.",
        )
        saved_format = "json"
    require(
        saved_format == config["format"]
        and saved["contract_version"] == config["contract_version"],
        "Adapter format or action contract differs from inference.",
    )
    require(
        all(
            saved["base"][key] == config["base"][key]
            for key in ("repository", "revision", "files_sha256")
        ),
        "Adapter base revision differs.",
    )
    return {
        "kind": version,
        "configuration_sha256": info["configuration_sha256"],
        "training_manifest_sha256": digest(Path(training) / "manifest.json"),
        "reload_manifest_sha256": digest(Path(reloaded) / "manifest.json"),
        "adapter_sha256": receipt["files_sha256"]["adapter/adapter_model.safetensors"],
    }


def run(
    config_path,
    requests_path,
    requests_sha256,
    output,
    *,
    training=None,
    reloaded=None,
    adapter_version=sft.VERSION,
):
    require(
        bool(training) == bool(reloaded),
        "Supply both trained and independently reloaded adapter runs.",
    )
    config, root, bundle, _, _, verified = sft.preflight(config_path)
    require(verified["environment"]["matched"], "Pinned runtime differs.")
    require(
        digest(requests_path) == requests_sha256,
        "Inference requests changed after registration.",
    )
    requests = prep.rows(requests_path)
    validate_requests(requests, config["format"])
    identity = {
        "kind": "native_base",
        "repository": config["base"]["repository"],
        "revision": config["base"]["revision"],
        "native_pin_sha256": config["native_pin_sha256"],
    }
    if training:
        identity.update(adapter_identity(training, reloaded, adapter_version, config))
    output = sft.new_output(output, root, training)
    info = {
        "version": VERSION,
        "status": "loading",
        "configuration_sha256": digest(config_path),
        "requests_sha256": requests_sha256,
        "requested": len(requests),
        "model_identity": identity,
        "numerical_regime": "native_bfloat16_with_optional_float32_lora",
        "schema_constrained": False,
        "final_test_bodies_read": False,
        "implementation_sha256": digest(__file__),
        "condition": requests[0].get("condition", {"id": "canonical_full_context"})[
            "id"
        ],
        "condition_sha256": requests[0].get("condition_sha256"),
        "format": config["format"],
    }
    write(output / "run.json", info)
    started = time.monotonic()
    emitted = 0
    try:
        torch, model, tokenizer, device, runtime = sft.gemma.load_runtime(
            config, root, bundle["tokens"][config["format"]]
        )
        if training:
            model, _, _ = sft.load_adapter(
                model,
                Path(training) / "adapter",
                read(Path(training) / "adapter-tensors.json"),
                config,
                torch,
                False,
            )
        write(output / "runtime.json", runtime)
        from transformers import GenerationConfig

        generation = GenerationConfig(**config["generation"])
        model.eval()
        model.gradient_checkpointing_disable()
        for request in requests:
            actual = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            require(
                actual == request["prefix_ids"],
                "Actual native prompt differs from the registered request.",
            )
            row = {
                key: request[key]
                for key in ("case_id", "input_sha256", "messages_sha256", "format")
            }
            row.update(
                model_identity=identity, prefix_ids_sha256=request["prefix_ids_sha256"]
            )
            if "condition_sha256" in request:
                row["condition_sha256"] = request["condition_sha256"]
            request_started = time.perf_counter()
            try:
                ids = torch.tensor([actual], dtype=torch.long, device=device)
                torch.cuda.synchronize()
                request_started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        generation_config=generation,
                    )
                torch.cuda.synchronize()
                tokens = generated[0, len(actual) :].tolist()
                terminal = tokens[-1] if tokens else None
                row.update(
                    status="completed",
                    output_ids=tokens,
                    terminal_token_id=terminal,
                    text=tokenizer.decode(tokens, skip_special_tokens=True),
                    text_raw=tokenizer.decode(tokens, skip_special_tokens=False),
                    finish_reason="stop"
                    if terminal in generation.eos_token_id
                    else "length",
                )
            except Exception as error:
                row.update(
                    status="request_failure",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                if isinstance(error, torch.OutOfMemoryError):
                    torch.cuda.empty_cache()
            row["generation_ms"] = (time.perf_counter() - request_started) * 1000
            sft.curriculum.append(output / "observations.jsonl", row)
            emitted += 1
            print(
                json.dumps(
                    {
                        "event": "observation",
                        "index": emitted,
                        "case_id": row["case_id"],
                        "status": row["status"],
                        "generation_ms": row["generation_ms"],
                    }
                ),
                flush=True,
            )
        info.update(
            status="inference_complete",
            observed=emitted,
            seconds=time.monotonic() - started,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
        )
        write(output / "run.json", info)
        sft.manifest(output, info["status"], VERSION)
        return info
    except BaseException as error:
        info.update(
            status="failed_incomplete",
            observed=emitted,
            seconds=time.monotonic() - started,
            error_type=type(error).__name__,
            error=str(error),
        )
        write(output / "run.json", info)
        sft.manifest(output, info["status"], VERSION)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "requests", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--requests-sha256", required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--reloaded", type=Path)
    parser.add_argument(
        "--adapter-version",
        choices=(sft.VERSION, "gemma-actions-v3-dpo", HISTORICAL_VERSION),
        default=sft.VERSION,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.requests,
                args.requests_sha256,
                args.output,
                training=args.training,
                reloaded=args.reloaded,
                adapter_version=args.adapter_version,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
