"""Run frozen label-free requests through one owned, sequential local model session."""

import argparse
import asyncio
import fcntl
import json
import signal
import subprocess
import time
from pathlib import Path

import evaluate_local_actions_v3 as core

VERSION = "local-actions-v3-session-inference-1"
CACHE_MODE = "sequential_session"
CACHE_POLICY = {
    "name": "prestarted_sequential_session_observed_partial_cache",
    "processes": 1,
    "request_order": "exact_registered_order",
    "cache_prompt": True,
    "priming_requests": 0,
    "cache_reset_between_requests": False,
    "cache_claim": "observed_cached_tokens_only_no_cold_or_warm_classification",
}
require = core.require


def source_hashes():
    return {**core.source_hashes(), str(Path(__file__).resolve()): core.sha256(__file__)}


def specification(request_binding, split, condition, deployment_path, timeout):
    require(
        split in core.SPLITS, "Only TRAIN/DEV requests are allowed; held-out bodies stay sealed."
    )
    require(condition in core.MODEL_CONDITIONS, "Use a named base, base_icl, SFT or DPO model.")
    require(type(timeout) in (int, float) and 0 < timeout <= 600, "Use a finite timeout <=600s.")
    rows = core.load_requests(request_binding, split, condition)
    config, _, command, receipt = core.deployment.check(deployment_path)
    require(config.output_format == rows[0]["format"], "Request and adapter formats differ.")
    identity, bindings = core.model_identity(config, receipt, condition)
    config = config.model_copy(update={"cache_prompt": True, "capture_runtime_evidence": False})
    info = {
        "split": split,
        "model_condition": condition,
        "requests": request_binding,
        "requested": len(rows),
        "case_ids": [row["case_id"] for row in rows],
        "format": rows[0]["format"],
        "condition": rows[0].get("condition", {"id": "canonical_full_context"})["id"],
        "condition_sha256": rows[0].get("condition_sha256"),
        "cache_mode": CACHE_MODE,
        "cache_policy": CACHE_POLICY,
        "diagnostics": "app_default",
        "timeout_seconds": timeout,
        "measurement_scope": "backend_adapter_sequential_session_including_http_hooks",
        "excluded_from_generation_ms": ["process_startup", "health", "native_prefix_parity"],
        "electron_latency_measured": False,
        "final_test_bodies_read": False,
        "quality_references_read": False,
        "app_default_changed": False,
        "schema_constrained": True,
        "environment": core.environment(),
        "sources_sha256": source_hashes(),
        "deployment": core.deployment.binding(Path(deployment_path) / "deployment.json"),
        "identity_bindings": bindings,
        "model_identity": identity,
        "numerical_regime": receipt["numerical_regime"],
        "gateway": config.model_dump(),
        "command": command,
        "runtime_sha256": core.runtime_hash(command, receipt, identity, config),
    }
    return info, rows, core.FrozenRequestAdapter(config, rows)


def register(requests, requests_sha256, split, condition, deployment_path, output, timeout=120):
    binding = {"path": str(Path(requests).resolve()), "sha256": requests_sha256}
    info, _, _ = specification(binding, split, condition, deployment_path, timeout)
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh session registration directory.")
    output.mkdir(parents=True)
    core.write_new(
        output / "protocol.json",
        {
            "version": VERSION,
            "status": "registered_not_inferred",
            "created_at": core.now(),
            **info,
        },
    )
    return {
        "status": "registered_not_inferred",
        "protocol": core.deployment.binding(output / "protocol.json"),
    }


def check(protocol_path, protocol_sha256=None):
    if protocol_sha256 is not None:
        require(core.sha256(protocol_path) == protocol_sha256, "Frozen protocol hash differs.")
    info = core.read(Path(protocol_path))
    require(
        info["version"] == VERSION
        and info["status"] == "registered_not_inferred"
        and info["split"] in core.SPLITS
        and info["final_test_bodies_read"] is False,
        "Unknown or held-out protocol; no request bodies were opened.",
    )
    require(info["cache_policy"] == CACHE_POLICY, "Registered sequential cache policy differs.")
    path = core.deployment.bound(info["deployment"])
    expected, rows, adapter = specification(
        info["requests"],
        info["split"],
        info["model_condition"],
        path.parent,
        info["timeout_seconds"],
    )
    actual = {
        key: value for key, value in info.items() if key not in ("version", "status", "created_at")
    }
    require(
        actual == expected, "Registered source, request, runtime or generation settings changed."
    )
    return info, rows, adapter


def cache_evidence(timings, prefix_length):
    timings = timings or {}
    count = timings.get("runtime_cached_tokens")
    usage = timings.get("runtime_usage_cached_tokens", count)
    verified = (
        type(count) is int
        and 0 <= count <= prefix_length
        and type(usage) is int
        and count == usage
        and timings.get("runtime_timing_evidence") != "inconsistent"
    )
    return {
        "status": "observed_sequential_session" if verified else "missing_or_inconsistent",
        "cached_tokens": count,
        "usage_cached_tokens": usage,
        "verified": verified,
        "cold_or_warm_claimed": False,
    }


async def generate(adapter, request, info, record, path):
    try:
        return await core.generate(adapter, request, info, record, path)
    finally:
        record["cache_evidence"] = cache_evidence(record.get("timings"), len(request["prefix_ids"]))
        core.write_state(path, record)


def freeze_output(output, status):
    core.write_new(
        output / "manifest.json",
        {
            "version": VERSION,
            "status": status,
            "files_sha256": {
                str(path.relative_to(output)): core.sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )


def run(protocol_path, output, *, protocol_sha256):
    info, requests, adapter = check(protocol_path, protocol_sha256)
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh session inference output directory.")
    output.mkdir(parents=True)
    observed = [core.observation(row, info, CACHE_MODE) for row in requests]
    for index, record in enumerate(observed):
        record.update(
            session_index=index, previous_case_id=requests[index - 1]["case_id"] if index else None
        )
    startup = {
        "status": "not_started",
        "model_started": False,
        "process_stopped": True,
        "model_identity": info["model_identity"],
        "runtime_sha256": info["runtime_sha256"],
        "command": info["command"],
    }
    state = {
        **info,
        "protocol": core.deployment.binding(protocol_path),
        "status": "running",
        "started_at": core.now(),
        "requests_sha256": info["requests"]["sha256"],
        "planned_scored_requests": len(requests),
        "planned_priming_requests": 0,
        "startup": startup,
    }
    core.write_new(output / "protocol.json", info)
    core.write_state(output / "run.json", state)
    fatal = None
    try:
        with core.lifecycle.LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            process = None
            startup_started = time.perf_counter()
            try:
                core.lifecycle.refuse_occupied(adapter.config.endpoint)
                startup.update(status="starting", started_at=core.now())
                with (output / "server.log").open("x") as log:
                    process = subprocess.Popen(
                        info["command"],
                        cwd=core.ROOT,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    startup.update(model_started=True, pid=process.pid, process_stopped=False)
                    core.write_state(output / "startup.json", startup)
                    core.lifecycle.wait_model(process, adapter.config.endpoint)
                    startup["health"] = adapter.health()
                    require(startup["health"]["status"] == "ready", "Runtime health/parity failed.")
                    with adapter._client() as client:
                        startup["props"] = adapter._json(client, "GET", "/props")
                        startup["loaded_adapters"] = adapter._json(client, "GET", "/lora-adapters")
                        adapter._verify_loaded_adapter(startup["loaded_adapters"])
                    startup.update(
                        status="ready", ready_ms=(time.perf_counter() - startup_started) * 1000
                    )
                    core.write_state(output / "startup.json", startup)
                    for index, (request, record) in enumerate(zip(requests, observed, strict=True)):
                        folder = output / "cases" / f"{index:05d}"
                        folder.mkdir(parents=True)
                        parity_started = time.perf_counter()
                        try:
                            with adapter.request(request["messages"]):
                                actual = adapter.apply_chat_template(
                                    request["messages"],
                                    tokenize=True,
                                    return_dict=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False,
                                )
                            record["native_prefix"] = {
                                "expected_sha256": request["prefix_ids_sha256"],
                                "actual_sha256": core.value_hash(actual),
                                "tokens": len(actual),
                                "identical": actual == request["prefix_ids"],
                                "verification_ms": (time.perf_counter() - parity_started) * 1000,
                            }
                            require(
                                actual == request["prefix_ids"],
                                "Runtime/native request prefix differs.",
                            )
                        except BaseException as error:
                            record.update(
                                status="error", error_type=type(error).__name__, error=str(error)
                            )
                            core.write_state(folder / "observation.json", record)
                            raise
                        asyncio.run(
                            generate(adapter, request, info, record, folder / "observation.json")
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "observation",
                                    "index": index + 1,
                                    "case_id": request["case_id"],
                                    "status": record["status"],
                                }
                            ),
                            flush=True,
                        )
            except BaseException as error:
                startup.update(status="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                core.lifecycle.stop_owned(process)
                startup.update(
                    process_stopped=process is None or process.poll() is not None,
                    process_returncode=process.returncode if process is not None else None,
                    finished_at=core.now(),
                )
                core.write_state(output / "startup.json", startup)
        state["status"] = "inference_complete"
    except BaseException as error:
        fatal = error
        state.update(
            status="cancelled"
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    finally:
        for record in observed:
            if record["status"] == "not_run":
                record["error"] = "Run stopped before this scheduled request."
        state.update(
            finished_at=core.now(),
            observed=len(observed),
            generation_attempts=sum("started_at" in row for row in observed),
            model_requests=sum(
                event["method"] == "POST" and event["url"].endswith("/v1/chat/completions")
                for row in observed
                for event in row.get("http_records", [])
            ),
            owned_processes_stopped=startup["process_stopped"],
            summary=core.timings_summary(observed),
            quality_evaluated=False,
        )
        state["runtime_preflight_passed"] = (
            state["status"] == "inference_complete"
            and startup["process_stopped"]
            and all(
                row["status"] == "completed"
                and row["native_prefix"]["identical"]
                and row["cache_evidence"]["verified"]
                for row in observed
            )
        )
        state["format_outputs_all_valid"] = all(row["valid_output"] is True for row in observed)
        core.write_rows(output / "observations.jsonl", observed)
        core.write_state(output / "run.json", state)
        freeze_output(output, state["status"])
    if fatal:
        raise fatal
    return {
        key: state[key]
        for key in ("status", "runtime_preflight_passed", "format_outputs_all_valid", "summary")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("register")
    command.add_argument("--requests", type=Path, required=True)
    command.add_argument("--requests-sha256", required=True)
    command.add_argument("--split", choices=core.SPLITS, required=True)
    command.add_argument("--condition", choices=core.MODEL_CONDITIONS, required=True)
    command.add_argument("--deployment", dest="deployment_path", type=Path, required=True)
    command.add_argument("--timeout", type=float, default=120)
    command.add_argument("--output", type=Path, required=True)
    for name in ("check", "run"):
        command = commands.add_parser(name)
        command.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
        command.add_argument("--protocol-sha256", required=name == "run")
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    name = args.pop("command")
    if name == "register":
        result = register(**args)
    elif name == "check":
        info, _, _ = check(**args)
        result = {
            "status": "preflight_files_verified",
            "requests": info["requested"],
            "model_started": False,
        }
    else:
        for number in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, core.lifecycle.interrupted)
        result = run(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if name == "run" and not result["runtime_preflight_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
