"""Prepare a native-derived small-model deployment for the automatic editor."""

import argparse
import json
from pathlib import Path

import run_local_copilot_actions_v3 as legacy
import serve_small_actions_final as serving
import small_actions_final as data

from app.copilot import LocalConfig, LocalLlamaAdapter

VERSION = "local-small-actions-final-v1"
ROOT = legacy.ROOT
require, digest, binding, bound = legacy.require, legacy.digest, legacy.binding, legacy.bound


def export_identity(exported, report):
    kind = report.get("kind", "sft")
    require(kind in {"sft", "dpo"}, "Unknown trained model kind.")
    identity = {"kind": kind}
    if kind == "dpo":
        frozen = data.read(exported / "dpo_configuration.json")
        require(
            digest(exported / "dpo_configuration.json") == report["dpo_configuration_sha256"]
            and digest(exported / "plan.md") == frozen["plan_sha256"]
            and digest(exported / "configuration.json") == frozen["parent"]["configuration_sha256"]
            and frozen["parent"]["adapter_sha256"] == report["parent_adapter_sha256"],
            "DPO export differs from its registered SFT parent or training plan.",
        )
        identity.update(
            parent_adapter_sha256=report["parent_adapter_sha256"],
            dpo_configuration_sha256=report["dpo_configuration_sha256"],
        )
    return identity


def prepare(
    preparation, exported, output, model_id, registry_path, *, port=18091, ui_performance=True
):
    preparation, exported = Path(preparation).resolve(), Path(exported).resolve()
    config, rows, _ = data.prepared(preparation)
    require(config["context_tokens"] == 4096, "The current editor contract uses context4096.")
    report = data.read(exported / "run.json")
    require(
        report["status"] == "gguf_exported_runtime_validation_pending"
        and report["base"] == config["base"]
        and data.read(exported / "configuration.json") == config,
        "Require a matching native-derived GGUF export.",
    )
    identity = export_identity(exported, report)
    model = exported / f"model-{report['quantization']}.gguf"
    template = exported / "chat_template.jinja"
    require(
        digest(model) == report["gguf_sha256"]
        and digest(template) == config["assets"]["files_sha256"]["chat_template.jinja"],
        "Exported model or native template changed.",
    )
    runtime = data.read(registry_path)["runtime"]
    require(
        runtime["revision"] == "427291b5b34cd914a31b3fd3b61a68f6184f4b9f",
        "Use the tested llama.cpp runtime revision.",
    )
    for name, expected in runtime["files_sha256"].items():
        require(digest(ROOT / name) == expected, "Runtime dependency changed.")
    build = f"{runtime['tag']}-{runtime['revision'][:9]}"
    checks = {}
    for row in rows:
        if row["language"] not in checks:
            checks[row["language"]] = {
                "case_id": row["case_id"],
                "messages": row["messages"],
                "expected_native_token_ids": row["input_ids"][: row["completion_start"]],
            }
    require(set(checks) == {"en", "ko"}, "Both languages require native-prefix checks.")
    require(1024 <= port <= 65535, "Use an explicit local port.")
    output = data.new_directory(output)
    parity = {
        "version": 2,
        "contract_version": data.formats.contract.VERSION,
        "model_sha256": report["gguf_sha256"],
        "adapter_sha256": None,
        "adapter_scale": None,
        "output_format": "json",
        "chat_template_sha256": digest(template),
        "runtime_build": build,
        "token_source": "native_tokenizer",
        "checks": list(checks.values()),
        "training_preparation_sha256": digest(preparation / "configuration.json"),
    }
    data.write(output / "native-token-parity.json", parity)
    gateway = LocalConfig(
        version=2,
        contract_version=data.formats.contract.VERSION,
        endpoint=f"http://127.0.0.1:{port}",
        model_id=model_id,
        model_path=str(model),
        model_sha256=report["gguf_sha256"],
        chat_template_path=str(template),
        chat_template_sha256=digest(template),
        runtime_build=build,
        token_parity_path=str(output / "native-token-parity.json"),
        token_parity_sha256=digest(output / "native-token-parity.json"),
        cache_prompt=True,
        capture_runtime_evidence=not ui_performance,
    )
    data.write(output / "gateway.json", gateway.model_dump())
    command = serving.server_command(
        ROOT / runtime["server"], model, template, alias=model_id, port=port, context_tokens=4096
    )
    data.write(output / "command.json", command)
    implementation = {
        Path(__file__).resolve(),
        ROOT / "server/scripts/run_local_copilot_actions_v3.py",
        ROOT / "server/scripts/run_local_copilot.py",
        ROOT / "server/app/copilot.py",
        ROOT / "server/app/copilot_runtime.py",
        ROOT / "server/app/copilot_store.py",
        *(
            ROOT / "ml" / name
            for name in legacy.preparation.source_closure(
                ["passive_formats_actions_v3.py", "passive_actions.py"]
            )
        ),
        *(ROOT / "ml").glob("prompt*.txt"),
    }
    receipt = {
        "version": VERSION,
        "status": "prepared_not_runtime_verified",
        "purpose": "ui_performance" if ui_performance else "runtime_evidence",
        "historical_tooling_only": False,
        "app_default_changed": False,
        "format": "json",
        "numerical_regime": "native_bf16_merged_" + report["quantization"].lower(),
        "model": config["model"],
        "base": config["base"],
        "training_adapter_sha256": report["adapter_sha256"],
        **identity,
        "export_receipt": binding(exported / "run.json"),
        "native_preparation": binding(preparation / "configuration.json"),
        "runtime_registry": binding(registry_path),
        "runtime_files_sha256": runtime["files_sha256"],
        "runtime_symlink_targets": runtime.get("symlink_targets", {}),
        "files_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        "implementation_sha256": {str(p): digest(p) for p in sorted(implementation)},
    }
    data.write(output / "deployment.json", receipt)
    return {"status": receipt["status"], "deployment": str(output), "model": config["model"]}


def check(deployment):
    deployment = Path(deployment).resolve()
    receipt = data.read(deployment / "deployment.json")
    require(receipt["version"] == VERSION, "Unknown small-model deployment version.")
    for name, expected in receipt["files_sha256"].items():
        require(
            digest(legacy.preparation.child(deployment, name)) == expected,
            "Prepared deployment file changed.",
        )
    for name, expected in receipt["implementation_sha256"].items():
        require(
            digest(name) == expected, "Deployment implementation changed; prepare its replacement."
        )
    for name, expected in receipt["runtime_files_sha256"].items():
        require(digest(ROOT / name) == expected, "Runtime dependency changed.")
    for name, target in receipt["runtime_symlink_targets"].items():
        require(
            (ROOT / name).resolve() == (ROOT / target).resolve(), "Runtime library link changed."
        )
    for name in ("export_receipt", "native_preparation", "runtime_registry"):
        bound(receipt[name])
    config = LocalConfig.model_validate(data.read(deployment / "gateway.json"))
    report = data.read(bound(receipt["export_receipt"]))
    identity = export_identity(bound(receipt["export_receipt"]).parent, report)
    require(
        all(receipt.get(key) == value for key, value in identity.items()),
        "Deployment lost the trained model's SFT/DPO identity.",
    )
    require(
        config.model_sha256 == report["gguf_sha256"]
        and receipt["base"] == report["base"]
        and receipt["training_adapter_sha256"] == report["adapter_sha256"]
        and config.adapter_path is None,
        "The merged deployment differs from its trained native export.",
    )
    require(
        receipt["purpose"] in {"ui_performance", "runtime_evidence"}
        and config.capture_runtime_evidence == (receipt["purpose"] == "runtime_evidence"),
        "Deployment purpose differs from runtime evidence settings.",
    )
    adapter = LocalLlamaAdapter(config)
    adapter._verify_files()
    return config, adapter, data.read(deployment / "command.json"), receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--registry", type=Path, default=legacy.DEFAULT_REGISTRY)
    parser.add_argument("--port", type=int, default=18091)
    parser.add_argument("--runtime-evidence", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.preparation,
                args.export,
                args.output,
                args.model_id,
                args.registry,
                port=args.port,
                ui_performance=not args.runtime_evidence,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
