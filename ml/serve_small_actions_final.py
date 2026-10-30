"""Export trained small models and measure an explicitly managed llama-server.

The infer command never starts or stops a server. Process scheduling remains with
the caller, and every request verifies native template/token parity before generation.
"""

import argparse
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

import small_actions_final as data
import train_small_actions_final as training


def merge(preparation, trained, reloaded, output, *, assets=None, cache_dir=None, device="cpu"):
    training.verified_reload(trained, reloaded)
    config, _, assets = data.prepared(preparation, assets)
    output = data.new_directory(output)
    torch, model, tokenizer, runtime = training.load_base(config, assets, device=device, cache_dir=cache_dir)
    model = training.load_adapter(model, trained, config, torch)
    model = model.merge_and_unload(safe_merge=True)
    data.require(not any("lora_" in name for name, _ in model.named_parameters()), "LoRA merge left adapter modules.")
    model.save_pretrained(output / "model", safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output / "model")
    shutil.copyfile(assets / "chat_template.jinja", output / "model/chat_template.jinja")
    data.write(output / "configuration.json", config)
    trained_info = data.read(Path(trained) / "run.json")
    kind = trained_info.get("kind", "sft")
    if kind == "dpo":
        for name in ("dpo_configuration.json", "plan.md"):
            shutil.copyfile(Path(trained) / name, output / name)
    result = {
        "version": data.VERSION, "status": "merged_native_bf16_local_validation_pending",
        "kind": kind,
        "base": config["base"], "adapter_sha256": data.sha(Path(trained) / "adapter/adapter_model.safetensors"),
        "training_configuration_sha256": data.sha(Path(trained) / "configuration.json"),
        "files_sha256": {str(p.relative_to(output)): data.sha(p) for p in sorted((output / "model").iterdir()) if p.is_file()},
        "runtime": runtime, "numerical_note": "BF16 merge can round weights; evaluate merged/quantized predictions separately.",
    }
    if kind == "dpo":
        result.update(parent_adapter_sha256=trained_info["parent_adapter_sha256"],
                      dpo_configuration_sha256=data.sha(output / "dpo_configuration.json"))
    data.write(output / "run.json", result)
    return result


def export(merged, llama_cpp, quantize_binary, output, *, quantization="Q4_K_M"):
    merged, llama_cpp = Path(merged).resolve(), Path(llama_cpp).resolve()
    report, config = data.read(merged / "run.json"), data.read(merged / "configuration.json")
    data.require(report["status"] == "merged_native_bf16_local_validation_pending", "Native merge is incomplete.")
    data.require(data.sha(merged / "configuration.json") == report["training_configuration_sha256"],
                 "Merged model training configuration changed.")
    for name, expected in report["files_sha256"].items():
        data.require(data.sha(merged / name) == expected, f"Merged source changed: {name}")
    if report.get("kind") == "dpo":
        frozen = data.read(merged / "dpo_configuration.json")
        data.require(data.sha(merged / "dpo_configuration.json") == report["dpo_configuration_sha256"]
                     and data.sha(merged / "plan.md") == frozen["plan_sha256"]
                     and data.sha(merged / "configuration.json") == frozen["parent"]["configuration_sha256"]
                     and frozen["parent"]["adapter_sha256"] == report["parent_adapter_sha256"],
                     "Merged DPO parent or actual training plan changed.")
    output = data.new_directory(output)
    commands = [
        [sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(merged / "model"),
         "--outtype", "bf16", "--outfile", str(output / "model-BF16.gguf")],
        [str(Path(quantize_binary).resolve()), str(output / "model-BF16.gguf"),
         str(output / f"model-{quantization}.gguf"), quantization],
    ]
    data.write(output / "commands.json", commands)
    for index, command in enumerate(commands):
        with (output / f"command-{index}.log").open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf

    model_path = output / f"model-{quantization}.gguf"
    reader = gguf.GGUFReader(str(model_path))
    data.require(reader.get_field("general.architecture").contents() == config["base"]["gguf_architecture"],
                 "Converted model architecture differs.")
    shutil.copyfile(merged / "model/chat_template.jinja", output / "chat_template.jinja")
    data.write(output / "configuration.json", config)
    result = {"version": data.VERSION, "status": "gguf_exported_runtime_validation_pending",
              "gguf_sha256": data.sha(model_path), "gguf_bytes": model_path.stat().st_size,
              "quantization": quantization, "base": config["base"], "adapter_sha256": report["adapter_sha256"],
              "training_configuration_sha256": report["training_configuration_sha256"],
              "converter_sha256": data.sha(llama_cpp / "convert_hf_to_gguf.py"),
              "quantizer_sha256": data.sha(quantize_binary), "merged_receipt_sha256": data.sha(merged / "run.json")}
    result["kind"] = report.get("kind", "sft")
    if result["kind"] == "dpo":
        for name in ("dpo_configuration.json", "plan.md"):
            shutil.copyfile(merged / name, output / name)
        result.update(parent_adapter_sha256=report["parent_adapter_sha256"],
                      dpo_configuration_sha256=report["dpo_configuration_sha256"])
    data.write(output / "run.json", result)
    return result


def server_command(binary, model, template, *, alias, port, context_tokens):
    return [str(Path(binary).resolve()), "-m", str(Path(model).resolve()),
            "--alias", alias, "--host", "127.0.0.1", "--port", str(port),
            "-c", str(context_tokens), "-np", "1", "-ngl", "99", "--jinja", "--no-ui",
            "--perf", "--metrics", "--cache-ram", "0", "--reasoning", "off", "--reasoning-budget", "0",
            "--chat-template-file", str(Path(template).resolve()),
            "--chat-template-kwargs", '{"enable_thinking":false}']


def runtime_schema(value):
    if isinstance(value, list):
        return [runtime_schema(item) for item in value]
    if isinstance(value, dict):
        return {key: runtime_schema(item) for key, item in value.items()
                if not (key in {"maxLength", "maxItems"} and item > 64)}
    return value


class LocalClient:
    def __init__(self, endpoint, timeout=120):
        url = urlparse(endpoint)
        data.require(url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1")
                     and url.path in ("", "/") and not url.username,
                     "Use an explicitly managed loopback llama-server endpoint.")
        self.connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)

    def request(self, method, path, body=None):
        self.connection.request(method, path, body=data.compact(body).encode() if body is not None else None,
                                headers={"Content-Type": "application/json"})
        response = self.connection.getresponse()
        raw = response.read()
        data.require(response.status == 200, f"Runtime HTTP {response.status}: {raw[:1000]!r}")
        return json.loads(raw)

    def close(self):
        self.connection.close()


def request_rows(path):
    result = data.rows(path)
    data.require(result and len({r.get("case_id", r.get("id")) for r in result}) == len(result), "Empty/duplicate request IDs.")
    for row in result:
        data.require(row["split"] in ("train", "dev"), "Held-out final-test bodies are not accepted by this preflight runner.")
        data.require(not {"primary", "acceptable", "accepted_predictions", "judgment", "rationale"}.intersection(row),
                     "Use label-free requests.")
        data.require(row["messages"] == data.formats.messages(row["input"]), "Request messages differ from saved canonical input.")
    return result


def infer(model_name, assets, requests_path, endpoint, alias, output, *, context_tokens=6144, cache_prompt=True):
    tokenizer = data.tokenizer_from(assets, model_name)
    requests = request_rows(requests_path)
    prefixes = [data.prompt_ids(tokenizer, row["messages"]) for row in requests]
    data.require(all(len(prefix) + 512 <= context_tokens for prefix in prefixes),
                 "Entire request cohort must fit the declared budget without truncation/exclusion.")
    output = data.new_directory(output)
    client = LocalClient(endpoint)
    failures = 0
    try:
        client.request("GET", "/health")
        props = client.request("GET", "/props")
        template = (Path(assets) / "chat_template.jinja").read_text()
        data.require(props.get("chat_template", "").rstrip("\n") == template.rstrip("\n"), "Runtime chat template differs from native pin.")
        settings = props.get("default_generation_settings", {})
        actual_context = settings.get("n_ctx", props.get("n_ctx"))
        data.require(actual_context == context_tokens, "Runtime context differs from request budget.")
        data.write(output / "runtime.json", {"endpoint": endpoint, "alias": alias, "props": props,
                                            "assets": data.verify_assets(assets, model_name),
                                            "request_sha256": data.sha(requests_path), "cache_prompt": cache_prompt})
        for request, prefix in zip(requests, prefixes, strict=True):
            row = {"case_id": request.get("case_id", request.get("id")), "format": "json",
                   "input_sha256": data.value_hash(data.formats.contract.wire_snapshot(request["input"])),
                   "messages_sha256": data.value_hash(request["messages"]), "prefix_tokens": len(prefix),
                   "prefix_ids_sha256": data.value_hash(prefix), "cache_prompt": cache_prompt}
            try:
                rendered = client.request("POST", "/apply-template", {"model": alias, "messages": request["messages"],
                                            "chat_template_kwargs": {"enable_thinking": False}})
                native = client.request("POST", "/tokenize", {"content": rendered["prompt"], "add_special": True,
                                                               "parse_special": True})["tokens"]
                data.require(native == prefix, "Runtime and native prompt token IDs differ.")
                schema = data.formats.contract.OUTPUT_SCHEMA
                body = {"model": alias, "messages": request["messages"], "temperature": 0,
                        "seed": 42, "max_tokens": 512, "stream": False, "cache_prompt": cache_prompt,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "response_format": {"type": "json_schema", "json_schema": {
                            "name": "next_edit", "strict": True, "schema": runtime_schema(schema)}}}
                started = time.perf_counter()
                response = client.request("POST", "/v1/chat/completions", body)
                elapsed = (time.perf_counter() - started) * 1000
                data.require(response.get("model") == alias and len(response.get("choices", [])) == 1,
                             "Response model or choices differ.")
                choice = response["choices"][0]
                row.update(status="completed", text=choice["message"]["content"], finish_reason=choice["finish_reason"],
                           http_response_ms=elapsed, raw_response=response, raw_request=body, native_token_parity=True)
                try:
                    prediction = data.formats.parse(row["text"], request["input"], "json", finish_reason=row["finish_reason"])
                    row.update(valid=True, prediction=prediction.model_dump())
                except (ValueError, TypeError) as error:
                    row.update(valid=False, validation_error=str(error))
            except Exception as error:
                row.update(status="request_failure", error_type=type(error).__name__, error=str(error))
                failures += 1
            training.append(output / "observations.jsonl", row)
            print(data.compact({"case_id": row["case_id"], "status": row["status"], "http_response_ms": row.get("http_response_ms")}), flush=True)
    finally:
        client.close()
    result = {"version": data.VERSION, "status": "runtime_preflight_complete", "model": model_name,
              "requested": len(requests), "failures": failures, "final_test_bodies_read": False,
              "observations_sha256": data.sha(output / "observations.jsonl"),
              "measurement": "HTTP completion only; excludes automatic debounce and editor preparation/rendering"}
    data.write(output / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge_parser = sub.add_parser("merge")
    for name in ("preparation", "training", "reloaded", "output"):
        merge_parser.add_argument("--" + name, type=Path, required=True)
    merge_parser.add_argument("--assets", type=Path)
    merge_parser.add_argument("--cache-dir", type=Path)
    merge_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    export_parser = sub.add_parser("export")
    for name in ("merged", "llama-cpp", "quantize-binary", "output"):
        export_parser.add_argument("--" + name, type=Path, required=True)
    export_parser.add_argument("--quantization", choices=("Q4_K_M", "Q8_0"), default="Q4_K_M")
    command_parser = sub.add_parser("command")
    for name in ("binary", "model", "template"):
        command_parser.add_argument("--" + name, type=Path, required=True)
    command_parser.add_argument("--alias", required=True)
    command_parser.add_argument("--port", type=int, default=18092)
    command_parser.add_argument("--context-tokens", type=int, default=6144)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--model", choices=data.PROFILES, required=True)
    for name in ("assets", "requests", "output"):
        infer_parser.add_argument("--" + name, type=Path, required=True)
    infer_parser.add_argument("--endpoint", required=True)
    infer_parser.add_argument("--alias", required=True)
    infer_parser.add_argument("--context-tokens", type=int, default=6144)
    infer_parser.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.command == "merge":
        result = merge(args.preparation, args.training, args.reloaded, args.output,
                       assets=args.assets, cache_dir=args.cache_dir, device=args.device)
    elif args.command == "export":
        result = export(args.merged, args.llama_cpp, args.quantize_binary, args.output, quantization=args.quantization)
    elif args.command == "command":
        result = server_command(args.binary, args.model, args.template, alias=args.alias,
                                port=args.port, context_tokens=args.context_tokens)
    else:
        result = infer(args.model, args.assets, args.requests, args.endpoint, args.alias, args.output,
                       context_tokens=args.context_tokens, cache_prompt=args.cache_prompt)
    print(data.compact(result), flush=True)


if __name__ == "__main__":
    main()
