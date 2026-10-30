"""Prepare, check and run an opt-in native-derived Q4 base plus verified Gemma LoRA."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import run_local_copilot as lifecycle

ROOT = lifecycle.ROOT
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "ml")]
import passive_formats_actions_v3 as formats  # noqa: E402
import prepare_gemma_actions_v3 as preparation  # noqa: E402

from app.copilot import LocalConfig, LocalLlamaAdapter, contract  # noqa: E402

VERSION = "local-gemma-actions-v3-lora"
REPOSITORY, REVISION = preparation.gemma.REPOSITORY, preparation.gemma.REVISION
DEFAULT_REGISTRY = (
    ROOT
    / "artifacts/benchmarks/passive-gemma-caption-prompt-v1/execution-attempt-01"
    / "runs/base-original/implementation/model-registry.json"
)
BASE_ID = "gemma26-native-base-q4-k-m"
require = lifecycle.require


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def bound(value):
    path = Path(value["path"])
    require(
        path.is_absolute() and path.is_file() and digest(path) == value["sha256"],
        f"Bound deployment file changed: {path}",
    )
    return path


def prepare(
    bundle,
    registry_path,
    adapter,
    adapter_receipt,
    output,
    model_id,
    *,
    format_name="json",
    port=18091,
    historical_tooling_only=False,
    ui_performance=False,
):
    output = Path(output).resolve()
    require(
        not output.exists() and 1024 <= port <= 65535,
        "Use a new deployment directory and explicit local port.",
    )
    require(
        not (ui_performance and historical_tooling_only),
        "Actual UI performance requires a current v3 deployment, not historical tooling.",
    )
    bundle = Path(bundle).resolve()
    prepared = preparation.verify_bundle(bundle)
    records = prepared["tokens"][format_name]
    registry = lifecycle.read(registry_path)
    base = next(value for value in registry["candidates"] if value["id"] == BASE_ID)
    require(
        (base["base_repository"], base["base_revision"]) == (REPOSITORY, REVISION),
        "Use the original native-derived Gemma base.",
    )
    runtime = registry["runtime"]
    require(
        runtime["revision"] == "427291b5b34cd914a31b3fd3b61a68f6184f4b9f",
        "Runtime revision differs.",
    )
    base_receipt = lifecycle.read(bound(base["export_receipt"]))
    require(
        base_receipt["base_repository"] == REPOSITORY
        and base_receipt["base_revision"] == REVISION
        and base_receipt["artifacts"]["base_quantized"]["sha256"] == base["model_sha256"],
        "Native-derived base receipt differs.",
    )
    model, template = (
        (ROOT / base["model_path"]).resolve(),
        (ROOT / base["template_path"]).resolve(),
    )
    require(
        digest(model) == base["model_sha256"] and digest(template) == base["template_sha256"],
        "Pinned base or native template changed.",
    )
    for name, expected in runtime["files_sha256"].items():
        require(digest(ROOT / name) == expected, "Pinned runtime dependency changed.")
    adapter_hash = None
    if adapter is not None:
        require(adapter_receipt is not None, "An adapter requires its conversion receipt.")
        receipt = lifecycle.read(adapter_receipt)
        require(
            receipt["base_repository"] == REPOSITORY
            and receipt["base_revision"] == REVISION
            and receipt["runtime_revision"] == runtime["revision"],
            "Adapter native/converter identity differs.",
        )
        if historical_tooling_only:
            require(
                receipt["version"] == "passive-gemma-sft-export-v1"
                and receipt["status"] == "tensor_verified_paired_export_runtime_preflight_pending",
                "Historical tooling fixture needs its original export receipt.",
            )
            adapter_hash = receipt["artifacts"]["adapter"]["sha256"]
        else:
            require(
                receipt["version"] == "gemma-actions-v3-lora-export"
                and receipt["status"] == "adapter_conversion_verified_serving_pending"
                and receipt["all_tensors_exact"] is True
                and receipt["adapter_tensor_count"] == 410
                and receipt["contract_version"] == formats.contract.VERSION
                and receipt["format_version"] == formats.VERSION
                and receipt["output_format"] == format_name,
                "Require exact v3 FP32 LoRA conversion for this contract and output format.",
            )
            adapter_hash = receipt["gguf_sha256"]
        adapter = Path(adapter).resolve()
        require(digest(adapter) == adapter_hash, "Adapter GGUF hash differs from conversion proof.")
    else:
        require(
            adapter_receipt is None and not historical_tooling_only,
            "No adapter receipt belongs to a plain base condition.",
        )
    checks = {}
    for record in records:
        snapshot = formats.contract.parse_snapshot(json.loads(record["messages"][1]["content"]))
        if snapshot.language in ("en", "ko") and snapshot.language not in checks:
            checks[snapshot.language] = {
                "case_id": record["id"],
                "messages": record["messages"],
                "expected_native_token_ids": record["primary"]["input_ids"][
                    : record["primary"]["completion_start"]
                ],
            }
    require(set(checks) == {"en", "ko"}, "Both languages require native-prefix checks.")
    build = f"{runtime['tag']}-{runtime['revision'][:9]}"
    parity = {
        "version": 2,
        "contract_version": formats.contract.VERSION,
        "model_sha256": base["model_sha256"],
        "adapter_sha256": adapter_hash,
        "adapter_scale": 1.0 if adapter else None,
        "output_format": format_name,
        "chat_template_sha256": base["template_sha256"],
        "runtime_build": build,
        "token_source": "native_tokenizer",
        "checks": list(checks.values()),
        "training_preparation_sha256": digest(bundle / "manifest.json"),
    }
    output.mkdir(parents=True)
    preparation.write(output / "native-token-parity.json", parity)
    config = LocalConfig(
        version=2,
        contract_version=formats.contract.VERSION,
        endpoint=f"http://127.0.0.1:{port}",
        model_id=model_id,
        model_path=str(model),
        model_sha256=base["model_sha256"],
        adapter_path=str(adapter) if adapter else None,
        adapter_sha256=adapter_hash,
        output_format=format_name,
        chat_template_path=str(template),
        chat_template_sha256=base["template_sha256"],
        runtime_build=build,
        token_parity_path=str(output / "native-token-parity.json"),
        token_parity_sha256=digest(output / "native-token-parity.json"),
        cache_prompt=True,
        capture_runtime_evidence=not ui_performance,
    )
    preparation.write(output / "gateway.json", config.model_dump())
    command = [
        str((ROOT / runtime["server"]).resolve()),
        "-m",
        str(model),
        "--alias",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "4096",
        "-np",
        "1",
        "-ngl",
        "99",
        "--jinja",
        "--no-ui",
        "--perf",
        "--metrics",
        "--cache-ram",
        "0",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--chat-template-file",
        str(template),
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    if adapter:
        command.extend(["--lora", str(adapter)])
    preparation.write(output / "command.json", command)
    implementation = {
        Path(__file__).resolve(),
        ROOT / "server/scripts/run_local_copilot.py",
        ROOT / "server/app/copilot.py",
        ROOT / "server/app/copilot_runtime.py",
        ROOT / "server/app/copilot_store.py",
        *(
            ROOT / "ml" / name
            for name in preparation.source_closure(
                ["passive_formats_actions_v3.py", "passive_actions.py"]
            )
        ),
        *(ROOT / "ml").glob("prompt*.txt"),
    }
    deployment = {
        "version": VERSION,
        "status": "prepared_not_runtime_verified",
        "app_default_changed": False,
        "purpose": "ui_performance" if ui_performance else "runtime_evidence",
        "historical_tooling_only": historical_tooling_only,
        "format": format_name,
        "numerical_regime": "native_derived_q4_base_fp32_lora"
        if adapter
        else "native_derived_q4_base",
        "base_registry": binding(registry_path),
        "base_export_receipt": base["export_receipt"],
        "adapter_export_receipt": binding(adapter_receipt) if adapter_receipt else None,
        "native_preparation": binding(bundle / "manifest.json"),
        "runtime_files_sha256": runtime["files_sha256"],
        "runtime_symlink_targets": runtime.get("symlink_targets", {}),
        "files_sha256": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
        "implementation_sha256": {str(path): digest(path) for path in sorted(implementation)},
    }
    preparation.write(output / "deployment.json", deployment)
    return {
        "status": deployment["status"],
        "deployment": str(output),
        "gateway_sha256": digest(output / "gateway.json"),
    }


def check(deployment):
    deployment = Path(deployment).resolve()
    receipt = lifecycle.read(deployment / "deployment.json")
    if receipt["version"] == "local-small-actions-final-v1":
        from run_local_copilot_small_final import check as check_small

        return check_small(deployment)
    require(receipt["version"] == VERSION, "Unknown local deployment version.")
    for name, expected in receipt["files_sha256"].items():
        require(
            digest(preparation.child(deployment, name)) == expected,
            "Prepared deployment file changed.",
        )
    for name, expected in receipt["implementation_sha256"].items():
        require(
            digest(name) == expected,
            "Deployment implementation changed; create a new deployment freeze.",
        )
    for name, expected in receipt["runtime_files_sha256"].items():
        require(digest(ROOT / name) == expected, "Runtime dependency changed.")
    for name, target in receipt["runtime_symlink_targets"].items():
        require(
            (ROOT / name).resolve() == (ROOT / target).resolve(), "Runtime library link changed."
        )
    for name in (
        "base_registry",
        "base_export_receipt",
        "native_preparation",
        "adapter_export_receipt",
    ):
        if receipt[name]:
            bound(receipt[name])
    config = LocalConfig.model_validate(lifecycle.read(deployment / "gateway.json"))
    if "purpose" in receipt:
        require(
            receipt["purpose"] in {"ui_performance", "runtime_evidence"}
            and config.capture_runtime_evidence == (receipt["purpose"] == "runtime_evidence"),
            "Deployment purpose differs from its frozen runtime evidence setting.",
        )
    adapter = LocalLlamaAdapter(config)
    adapter._verify_files()
    return config, adapter, lifecycle.read(deployment / "command.json"), receipt


def run(deployment, output, *, with_app=False, smoke_only=False):
    deployment, output = Path(deployment).resolve(), Path(output).resolve()
    config, adapter, command, receipt = check(deployment)
    require(not output.exists(), "Use a new runtime evidence directory.")
    require(
        not with_app or not receipt["historical_tooling_only"],
        "Historical fixture is tooling-only and cannot launch the editor.",
    )
    output.mkdir(parents=True)
    info = {
        "version": VERSION,
        "status": "starting",
        "deployment_sha256": digest(deployment / "deployment.json"),
        "numerical_regime": receipt["numerical_regime"],
        "historical_tooling_only": receipt["historical_tooling_only"],
        "app_default_changed": False,
        "model_started": False,
        "app_started": False,
    }
    model = app = None
    started = time.monotonic()
    try:
        with lifecycle.LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lifecycle.refuse_occupied(config.endpoint)
            with (output / "server.log").open("x") as log:
                try:
                    model = subprocess.Popen(
                        command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True
                    )
                    info.update(model_started=True, model_pid=model.pid)
                    lifecycle.wait_model(model, config.endpoint)
                    health = adapter.health()
                    require(
                        health["status"] == "ready",
                        "Native/runtime prefix or loaded-adapter verification failed.",
                    )
                    preparation.write(output / "health.json", health)
                    with adapter._client() as client:
                        preparation.write(
                            output / "loaded-adapters.json",
                            adapter._json(client, "GET", "/lora-adapters"),
                        )
                    if smoke_only:
                        checks = lifecycle.read(Path(config.token_parity_path))["checks"]
                        results = []
                        for value in checks:
                            generated = asyncio.run(
                                adapter.generate(
                                    value["messages"],
                                    contract(config.contract_version).OUTPUT_SCHEMA,
                                    120,
                                )
                            )
                            row = {
                                "case_id": value["case_id"],
                                "text": generated.text,
                                "finish_reason": generated.finish_reason,
                                "timings": generated.timings,
                                "runtime_evidence": generated.runtime_evidence,
                            }
                            try:
                                snapshot = json.loads(value["messages"][1]["content"])
                                row["prediction"] = formats.parse(
                                    generated.text,
                                    snapshot,
                                    config.output_format,
                                    finish_reason=generated.finish_reason,
                                ).model_dump()
                                row["valid"] = True
                            except ValueError as error:
                                row.update(valid=False, validation_error=str(error))
                            results.append(row)
                        preparation.write(output / "smoke-generation.json", results)
                        info.update(
                            status="runtime_smoke_complete",
                            requests=len(results),
                            valid_outputs=sum(value["valid"] for value in results),
                            quality_evaluated=False,
                        )
                    else:
                        if with_app:
                            env = {
                                **os.environ,
                                "AUDAPOLIS_NEP_CONFIG": str(deployment / "gateway.json"),
                                "MODE": "development",
                                "NODE_ENV": "development",
                                "UV_NO_SYNC": "1",
                                "UV_OFFLINE": "1",
                            }
                            app = subprocess.Popen(
                                ["npm", "start"], cwd=ROOT / "app", env=env, start_new_session=True
                            )
                            info["app_started"] = True
                        print(
                            "Verified local model is ready. Ctrl+C stops this owned session.",
                            flush=True,
                        )
                        while model.poll() is None and (app is None or app.poll() is None):
                            time.sleep(0.5)
                        require(
                            app is not None and app.returncode == 0,
                            "Owned model/app stopped unexpectedly.",
                        )
                        info["status"] = "session_stopped"
                finally:
                    lifecycle.stop_owned(app)
                    lifecycle.stop_owned(model)
    except KeyboardInterrupt:
        info["status"] = "session_stopped"
    except BaseException as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        info.update(
            seconds=time.monotonic() - started,
            model_process_stopped=model is None or model.poll() is not None,
            app_process_stopped=app is None or app.poll() is not None,
            model_returncode=model.returncode if model is not None else None,
            app_returncode=app.returncode if app is not None else None,
        )
        preparation.write(output / "run.json", info)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("bundle", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--adapter-receipt", type=Path)
    p.add_argument("--model-id", required=True)
    p.add_argument("--format", choices=("json", "rewrite"), default="json")
    p.add_argument("--port", type=int, default=18091)
    p.add_argument("--historical-tooling-only", action="store_true")
    p.add_argument(
        "--ui-performance", action="store_true",
        help="Prepare a new UI measurement deployment with runtime evidence capture disabled.",
    )
    for name in ("check", "smoke", "start"):
        p = commands.add_parser(name)
        p.add_argument("--deployment", type=Path, required=True)
        if name != "check":
            p.add_argument("--output", type=Path, required=True)
        if name == "start":
            p.add_argument("--with-app", action="store_true")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["registry_path"], args["format_name"] = args.pop("registry"), args.pop("format")
        result = prepare(**args)
    elif command == "check":
        config, _, _, receipt = check(args["deployment"])
        result = {
            "status": "deployment_files_verified",
            "model_id": config.model_id,
            "format": config.output_format,
            "numerical_regime": receipt["numerical_regime"],
            "model_started": False,
        }
    else:
        for name in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(name, lifecycle.interrupted)
        result = run(**args, smoke_only=command == "smoke")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
