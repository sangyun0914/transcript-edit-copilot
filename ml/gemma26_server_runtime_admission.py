"""Prepare an explicit server candidate without changing converter or model provenance."""

import argparse
import copy
import gzip
import json
from pathlib import Path

import evaluate_gemma_final as evaluation
import gemma_final_deployment as deployment

ROOT = deployment.ROOT
VERSION = "gemma26-final-server-runtime-admission-1"
PLAN = ROOT / "docs/plans/final-gemma26-llama-stable-comparison-2026-10-25.md"
PLAN_SHA = "f5a27a83e8e1478ba49309057497bc111c73b90fc31ebe6338e3f539c3624c3e"
OLD_INVENTORY = ROOT / "artifacts/local-runtime/gemma-actions-v3/b003-ui-deployment-attempt-003/deployment.json"
OLD_INVENTORY_SHA = "497103a421eb857e7f243f6c123e51190513bbb6d6ce45d0825fc6619654bc89"
STABLE = ROOT / "artifacts/nep-probe/llama-v0.7.0-b11669"
DOWNLOAD_SHA = "86ec96e14705f239c781d1c33f77dac0101679f6b92e3841d01cfad70ae475ba"
DISTRIBUTION_SHA = "3bfd1530d159e817d5976cf13b4ff20b0d07157c81ef17a4ad3a2c59fd3a9672"
ARCHIVE_SHA = "dbed55493d72bc1ba2babc8770d669c2289b5bc4ee9fb2a4dfc158d45b7d13fd"
SERVERS = {
    "old": {"tag": "b10816", "build": 10816, "revision": "427291b5b34cd914a31b3fd3b61a68f6184f4b9f",
            "server": "artifacts/nep-probe/llama-b10816/llama-server"},
    "stable": {"tag": "b11669", "build": 11669, "revision": "620d2225cc436ab0b001e3200e590e3b286ef9ab",
               "server": "artifacts/nep-probe/llama-v0.7.0-b11669/llama-b11669/llama-server"},
}
CONVERTER_REVISION = "427291b5b34cd914a31b3fd3b61a68f6184f4b9f"
require, read, sha, binding, bound = deployment.require, deployment.read, deployment.sha, deployment.binding, deployment.bound


def checked_inventory(files, links):
    require(isinstance(files, dict) and files, "Server inventory is empty.")
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(ROOT) and path.is_file()
                and sha(path) == expected, "Server inventory bytes differ: " + name)
    for name, target in links.items():
        path = ROOT / name
        require(name in files and target in files and path.is_symlink()
                and path.resolve() == (ROOT / target).resolve(), "Server library link differs: " + name)


def server_runtime(name):
    require(name in SERVERS and sha(PLAN) == PLAN_SHA, "Unregistered server or comparison plan.")
    identity = dict(SERVERS[name])
    if name == "old":
        require(sha(OLD_INVENTORY) == OLD_INVENTORY_SHA, "Pinned old server inventory changed.")
        source = read(OLD_INVENTORY)
        files, links = source["runtime_files_sha256"], source["runtime_symlink_targets"]
        proof = {"inventory": binding(OLD_INVENTORY), "scope": "binary_inventory_only_no_B003_model_admission"}
    else:
        require(sha(STABLE / "download.json") == DOWNLOAD_SHA
                and sha(STABLE / "distribution.json") == DISTRIBUTION_SHA,
                "Pinned official stable package metadata changed.")
        download, distribution = read(STABLE / "download.json"), read(STABLE / "distribution.json")
        require(download["stable_tag"] == "v0.7.0" and download["nightly_tag"] == identity["tag"]
                and download["revision"] == identity["revision"]
                and download["asset"]["url"] == "https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-macos-arm64.tar.gz"
                and download["asset"]["digest"] == "sha256:" + ARCHIVE_SHA
                and download["asset"]["size"] == 12191582,
                "Stable server is not the registered official release.")
        archive = STABLE / download["asset"]["name"]
        require(archive.name == "llama-b11669-bin-macos-arm64.tar.gz"
                and archive.stat().st_size == 12191582 and sha(archive) == ARCHIVE_SHA,
                "Official stable archive changed.")
        files, links = distribution["files_sha256"], distribution["symlink_targets"]
        proof = {"download": binding(STABLE / "download.json"), "inventory": binding(STABLE / "distribution.json"),
                 "archive": binding(archive), "stable_tag": "v0.7.0"}
    checked_inventory(files, links)
    require(identity["server"] in files, "Server executable is missing from its fixed inventory.")
    return {**identity, "runtime_build": identity["tag"] + "-" + identity["revision"][:9],
            "binary": binding(ROOT / identity["server"]), "files_sha256": files,
            "symlink_targets": links, "proof": proof, "plan": binding(PLAN)}


def canonical_command(gateway, binary, port):
    return [str(Path(binary).resolve()), "-m", gateway["model_path"], "--alias", gateway["model_id"],
        "--host", "127.0.0.1", "--port", str(port), "-c", "4096", "-np", "1", "-ngl", "99", "--jinja",
        "--no-ui", "--perf", "--metrics", "--cache-ram", "0", "--reasoning", "off", "--reasoning-budget", "0",
        "--chat-template-file", gateway["chat_template_path"], "--chat-template-kwargs", '{"enable_thinking":false}',
        "--lora", gateway["adapter_path"]]


def preparation_checks(config_path):
    checks = {}
    with gzip.open(Path(config_path).parent / "tokens-json.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            language = json.loads(record["messages"][1]["content"])["language"]
            if language in ("en", "ko") and language not in checks:
                checks[language] = {"case_id": record["id"], "messages": record["messages"],
                    "expected_native_token_ids": record["primary"]["input_ids"][:record["primary"]["completion_start"]]}
            if len(checks) == 2:
                break
    require(set(checks) == {"en", "ko"}, "Both native preparation languages must be present.")
    return list(checks.values())


def model_source(model_metadata, source_deployment):
    model_metadata, source_deployment = Path(model_metadata).resolve(), Path(source_deployment).resolve()
    require(source_deployment.name == "deployment.json", "Supply the actual source deployment receipt.")
    model = evaluation.verify_model(model_metadata)
    config = read(bound(model["inference_configuration"]))
    deployment.validate_deployment(binding(source_deployment), model, config)
    gateway, command, receipt = deployment.deployment_check(source_deployment.parent)
    require("server_runtime_admission" not in receipt, "Derive from the original checked deployment, not another candidate.")
    old = server_runtime("old")
    port = int(command[command.index("--port") + 1])
    require(gateway["runtime_build"] == old["runtime_build"]
            and gateway["endpoint"] == f"http://127.0.0.1:{port}"
            and gateway["chat_template_sha256"] == config["tokenizer"]["template_sha256"]
            and receipt["runtime_files_sha256"] == old["files_sha256"]
            and receipt["runtime_symlink_targets"] == old["symlink_targets"]
            and command == canonical_command(gateway, old["binary"]["path"], port),
            "Source server identity differs from the independently pinned old runtime.")
    converter = read(bound(receipt["adapter_export_receipt"]))
    require(converter["runtime_revision"] == CONVERTER_REVISION,
            "Adapter converter provenance changed; server admission cannot replace it.")
    parity_path = bound({"path": gateway["token_parity_path"], "sha256": gateway["token_parity_sha256"]})
    parity = read(parity_path)
    require(parity_path == source_deployment.parent / "native-token-parity.json"
            and parity["token_source"] == "native_tokenizer" and len(parity["checks"]) == 2
            and parity["checks"] == preparation_checks(bound(model["inference_configuration"]))
            and parity["training_preparation_sha256"] == model["native_preparation"]["sha256"]
            and all(parity[key] == gateway[key] for key in ("version", "contract_version", "model_sha256",
                    "adapter_sha256", "adapter_scale", "output_format", "chat_template_sha256", "runtime_build")),
            "Source native prefix proof differs from the selected preparation or gateway.")
    return {"model": model, "config": config, "gateway": gateway, "command": command, "receipt": receipt,
            "parity": parity, "model_metadata": binding(model_metadata), "source_deployment": binding(source_deployment)}


def payloads(source, runtime, output, port):
    require(type(port) is int and 1024 <= port <= 65535, "Use an explicit unprivileged loopback port.")
    gateway, parity = copy.deepcopy(source["gateway"]), copy.deepcopy(source["parity"])
    parity["runtime_build"] = runtime["runtime_build"]
    gateway.update(runtime_build=runtime["runtime_build"], endpoint=f"http://127.0.0.1:{port}",
                   token_parity_path=str(output / "native-token-parity.json"))
    command = canonical_command(gateway, runtime["binary"]["path"], port)
    admission = {"version": VERSION, "status": "candidate_prepared_runtime_validation_pending",
        "model_started": False, "app_default_changed": False, "model_selection_performed": False,
        "body_access_authorized": False, "final_test_bodies_read": False,
        "model_metadata": source["model_metadata"], "native_identity": source["model"]["identity"],
        "source_deployment": source["source_deployment"], "server_runtime": runtime,
        "converter_provenance": {"revision": CONVERTER_REVISION,
            **{key: source["receipt"][key] for key in ("base_registry", "base_export_receipt", "adapter_export_receipt")}},
        "port": port, "implementation": binding(Path(__file__))}
    return gateway, parity, command, admission


def receipt_for(source, runtime, output):
    receipt = copy.deepcopy(source["receipt"])
    receipt.update(status="prepared_not_runtime_verified", runtime_files_sha256=runtime["files_sha256"],
                   runtime_symlink_targets=runtime["symlink_targets"],
                   server_runtime_admission=binding(output / "server-runtime-admission.json"))
    receipt["implementation_sha256"][str(Path(__file__).resolve())] = sha(__file__)
    receipt["files_sha256"] = {name: sha(output / name) for name in
                              ("gateway.json", "native-token-parity.json", "command.json", "server-runtime-admission.json")}
    return receipt


def verify_candidate(directory, model_metadata):
    directory = Path(directory).resolve()
    receipt = read(directory / "deployment.json")
    require(bound(receipt["server_runtime_admission"]) == directory / "server-runtime-admission.json",
            "Runtime admission escaped its candidate deployment.")
    admission = read(directory / "server-runtime-admission.json")
    require(admission["model_metadata"] == binding(model_metadata), "Candidate selected a different native model receipt.")
    source = model_source(model_metadata, bound(admission["source_deployment"]))
    name = next((name for name, value in SERVERS.items() if value["revision"] == admission["server_runtime"]["revision"]), None)
    runtime = server_runtime(name)
    gateway, parity, command, expected = payloads(source, runtime, directory, admission["port"])
    gateway["token_parity_sha256"] = sha(directory / "native-token-parity.json")
    require(admission == expected and read(directory / "gateway.json") == gateway
            and read(directory / "native-token-parity.json") == parity and read(directory / "command.json") == command
            and receipt == receipt_for(source, runtime, directory), "Candidate runtime, gateway, parity or provenance changed.")
    deployment.validate_deployment(binding(directory / "deployment.json"), source["model"], source["config"])
    actual_gateway, actual_command, _ = deployment.deployment_check(directory)
    require(actual_gateway == gateway and actual_command == command, "External deployment checks disagree with admission.")
    return {"model": source["model"], "gateway": gateway, "command": command, "receipt": receipt,
            "runtime": runtime, "admission": admission, "parity": parity}


def protected_roots(source):
    protected = [bound(source["source_deployment"]).parent, bound(source["model"]["inference_configuration"]).parent]
    protected.extend(Path(source["gateway"][key]).parent for key in ("model_path", "adapter_path", "chat_template_path"))
    for evidence in (source["model"]["sft_evidence"], source["model"].get("dpo_evidence")):
        if evidence:
            protected.extend(bound(evidence[key]).parent for key in ("training", "reload"))
    return protected


def prepare_candidate(model_metadata, source_deployment, runtime_name, output, *, port=18094):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Use a fresh candidate directory in this workspace.")
    source = model_source(model_metadata, source_deployment)
    protected = protected_roots(source)
    require(not any(output.is_relative_to(path) for path in protected), "Candidate output overlaps immutable model inputs.")
    runtime = server_runtime(runtime_name)
    gateway, parity, command, admission = payloads(source, runtime, output, port)
    output.mkdir(parents=True)
    deployment.gemma.write(output / "native-token-parity.json", parity)
    gateway["token_parity_sha256"] = sha(output / "native-token-parity.json")
    for name, value in (("gateway.json", gateway), ("command.json", command), ("server-runtime-admission.json", admission)):
        deployment.gemma.write(output / name, value)
    deployment.gemma.write(output / "deployment.json", receipt_for(source, runtime, output))
    return verify_candidate(output, model_metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--source-deployment", type=Path, required=True)
    parser.add_argument("--runtime", choices=SERVERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18094)
    args = parser.parse_args(argv)
    result = prepare_candidate(args.model_metadata, args.source_deployment, args.runtime, args.output, port=args.port)
    print(json.dumps({"status": result["admission"]["status"], "model_started": False,
                      "deployment": str(args.output.resolve() / "deployment.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
