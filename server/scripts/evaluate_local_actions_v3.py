"""Freeze and measure label-free train/dev requests through the local v3 adapter.

This measures the backend adapter, not Electron trigger-to-preview latency. Every
case starts an owned process. Warm measurements follow an explicit identical
priming request; actual cache token counts determine whether that state held.
"""

import argparse
import asyncio
import fcntl
import json
import math
import platform
import signal
import statistics
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "ml")]
import prepare_actions_v3_conditions as conditions  # noqa: E402
import run_local_copilot_actions_v3 as deployment  # noqa: E402

from app import copilot  # noqa: E402

VERSION = "local-actions-v3-cache-evaluation-1"
MODEL_CONDITIONS = ("base", "base_icl", "sft", "dpo")
BASELINES = ("always_no_action", "consistent_history_rule")
SPLITS = ("train", "dev")
REQUEST_KEYS = {
    "case_id",
    "split",
    "format",
    "input",
    "input_sha256",
    "messages",
    "messages_sha256",
    "prefix_ids",
    "prefix_ids_sha256",
}
OPTIONAL_REQUEST_KEYS = {"condition", "condition_sha256", "original_input_sha256"}
ROOT, lifecycle, formats = deployment.ROOT, deployment.lifecycle, deployment.formats
require, read, sha256 = deployment.require, lifecycle.read, deployment.digest
value_hash = deployment.preparation.value_hash
write_new, write_rows = deployment.preparation.write, deployment.preparation.write_rows


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: package_version(name) for name in ("httpx", "pydantic")},
    }


def source_hashes():
    files = {
        Path(__file__).resolve(),
        Path(deployment.__file__).resolve(),
        Path(lifecycle.__file__).resolve(),
        Path(copilot.__file__).resolve(),
        ROOT / "server/app/copilot_store.py",
        *(
            ROOT / "ml" / name
            for name in deployment.preparation.source_closure(
                ["passive_formats_actions_v3.py", "prepare_actions_v3_conditions.py"]
            )
        ),
        *(ROOT / "ml").glob("prompt*.txt"),
    }
    return {str(path): sha256(path) for path in sorted(files)}


def validate_requests(rows, split, condition):
    require(split in SPLITS, "Only TRAIN/DEV requests are allowed; held-out bodies stay sealed.")
    require(condition in (*MODEL_CONDITIONS, *BASELINES), "Unknown model or baseline condition.")
    require(
        rows and len({row["case_id"] for row in rows}) == len(rows), "Unique requests required."
    )
    for row in rows:
        require(
            REQUEST_KEYS <= row.keys()
            and not (row.keys() - REQUEST_KEYS - OPTIONAL_REQUEST_KEYS)
            and row["split"] == split,
            "Requests must be label-free and belong to the registered train/dev split.",
        )
        require(isinstance(row["case_id"], str) and row["case_id"], "Missing case ID.")
        require(row["format"] in formats.FORMATS, "Unknown output format.")
        require(
            ("condition" in row) == ("condition_sha256" in row),
            "Condition and hash must be supplied together.",
        )
        spec = row.get("condition")
        expected = (
            conditions.expected_messages(row["input"], row["format"], spec)
            if spec
            else formats.messages(row["input"], row["format"])
        )
        require(
            bool(spec and spec["id"] == "train_icl") == (condition == "base_icl"),
            "Only base_icl uses the frozen train-only demonstrations.",
        )
        if spec:
            require(row["condition_sha256"] == value_hash(spec), "Condition hash differs.")
            require(
                row["case_id"] not in {item["case_id"] for item in spec["demonstrations"]},
                "The current case cannot be its own ICL demonstration.",
            )
        require(
            row["input_sha256"] == conditions.input_hash(row["input"])
            and row["messages"] == expected
            and row["messages_sha256"] == value_hash(expected)
            and row["prefix_ids_sha256"] == value_hash(row["prefix_ids"]),
            "Visible input, prompt or native prefix binding differs.",
        )
        require(
            isinstance(row["prefix_ids"], list)
            and 0 < len(row["prefix_ids"]) <= formats.INPUT_LIMIT
            and all(type(token) is int and 0 <= token < 262144 for token in row["prefix_ids"]),
            "Invalid or oversized native prefix.",
        )
    require(len({row["format"] for row in rows}) == 1, "A run uses one output format.")
    require(
        len({row.get("condition_sha256") for row in rows}) == 1,
        "A run uses one frozen context condition.",
    )


def load_requests(binding, split, condition):
    require(split in SPLITS, "Only TRAIN/DEV metadata may open request bodies.")
    path = deployment.bound(binding)
    rows = deployment.preparation.rows(path)
    validate_requests(rows, split, condition)
    return rows


def model_identity(config, receipt, condition):
    require(config.version == 2, "Require the seven-action v3 local configuration.")
    require(
        not receipt["historical_tooling_only"], "Historical tooling is not a v3 model condition."
    )
    require(
        bool(config.adapter_path) == (condition in ("sft", "dpo")),
        "Base conditions have no adapter; SFT/DPO require the corresponding adapter.",
    )
    identity = {
        "kind": condition,
        "id": config.model_id,
        "base_repository": deployment.REPOSITORY,
        "base_revision": deployment.REVISION,
        "model_sha256": config.model_sha256,
        "adapter_sha256": config.adapter_sha256,
        "adapter_scale": config.adapter_scale if config.adapter_path else None,
        "chat_template_sha256": config.chat_template_sha256,
        "token_parity_sha256": config.token_parity_sha256,
        "runtime_build": config.runtime_build,
        "output_format": config.output_format,
        "contract_version": config.contract_version,
    }
    bindings = []
    if config.adapter_path:
        export_path = deployment.bound(receipt["adapter_export_receipt"])
        export = read(export_path)
        export_config = export_path.parent / "configuration.json"
        require(
            sha256(export_config) == export["configuration_sha256"], "Export configuration changed."
        )
        training_binding = read(export_config)["training_record"]
        training_path = deployment.bound(training_binding)
        require(
            training_binding["sha256"] == export["training_manifest_sha256"]
            and read(training_path)["version"] == f"gemma-actions-v3-{condition}"
            and export["gguf_sha256"] == config.adapter_sha256,
            "The named condition does not match its hash-bound training/export evidence.",
        )
        bindings = [deployment.binding(export_config), training_binding]
        identity.update(
            native_pin_sha256=export["native_pin_sha256"],
            adapter_export_sha256=sha256(export_path),
            training_manifest_sha256=export["training_manifest_sha256"],
            reload_manifest_sha256=export["reload_manifest_sha256"],
            training_phase=export["training_phase"],
        )
    return identity, bindings


def runtime_hash(command, receipt, identity, config):
    return value_hash(
        {
            "command": command,
            "files": receipt["runtime_files_sha256"],
            "symlinks": receipt["runtime_symlink_targets"],
            "model_identity": identity,
            "gateway": config.model_dump(),
        }
    )


def register(
    requests,
    requests_sha256,
    split,
    condition,
    output,
    *,
    deployment_path=None,
    cache_modes=("cold", "warm"),
    diagnostics="app_default",
    timeout=120,
):
    require(split in SPLITS, "Only TRAIN/DEV registration is implemented.")
    request_binding = {"path": str(Path(requests).resolve()), "sha256": requests_sha256}
    rows = load_requests(request_binding, split, condition)
    require(diagnostics in ("app_default", "full"), "Unknown diagnostic setting.")
    require(type(timeout) in (int, float) and 0 < timeout <= 600, "Use a finite timeout <=600s.")
    require(
        cache_modes
        and len(set(cache_modes)) == len(cache_modes)
        and set(cache_modes) <= {"cold", "warm"},
        "Choose distinct cold and/or warm blocks.",
    )
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh measurement registration directory.")
    is_model = condition in MODEL_CONDITIONS
    require(
        bool(deployment_path) == is_model, "Model conditions require a verified deployment only."
    )
    info = {
        "version": VERSION,
        "status": "registered_not_inferred",
        "created_at": now(),
        "split": split,
        "model_condition": condition,
        "requests": request_binding,
        "requested": len(rows),
        "case_ids": [row["case_id"] for row in rows],
        "format": rows[0]["format"],
        "condition": rows[0].get("condition", {"id": "canonical_full_context"})["id"],
        "condition_sha256": rows[0].get("condition_sha256"),
        "cache_modes": list(cache_modes) if is_model else ["not_applicable"],
        "cache_policy": {
            "cache_prompt": True,
            "cold": "first_generation_after_owned_process_restart",
            "warm": "explicit_identical_priming_then_identical_scored_request",
            "cold_cached_tokens": 0,
            "warm_minimum": "native_prefix_length_minus_one",
            "failed_priming": "do_not_issue_warm_scored_request",
        }
        if is_model
        else None,
        "diagnostics": diagnostics if is_model else "not_applicable",
        "timeout_seconds": timeout,
        "measurement_scope": "backend_adapter_and_output_validation_including_http_hooks",
        "excluded_from_latency": ["process_startup", "health", "native_prefix_parity", "priming"],
        "electron_latency_measured": False,
        "final_test_bodies_read": False,
        "quality_references_read": False,
        "app_default_changed": False,
        "environment": environment(),
        "sources_sha256": source_hashes(),
        "deployment": None,
        "identity_bindings": [],
        "runtime_sha256": None,
        "model_identity": {"kind": "deterministic_baseline", "name": condition},
        "numerical_regime": "deterministic_no_model",
        "schema_constrained": is_model,
    }
    if is_model:
        config, _, command, receipt = deployment.check(deployment_path)
        require(config.output_format == info["format"], "Request and adapter formats differ.")
        identity, bindings = model_identity(config, receipt, condition)
        measured_config = config.model_copy(
            update={
                "cache_prompt": True,
                "capture_runtime_evidence": diagnostics == "full",
            }
        )
        info.update(
            deployment=deployment.binding(Path(deployment_path) / "deployment.json"),
            identity_bindings=bindings,
            model_identity=identity,
            numerical_regime=receipt["numerical_regime"],
            gateway=measured_config.model_dump(),
            command=command,
            runtime_sha256=runtime_hash(command, receipt, identity, measured_config),
        )
    output.mkdir(parents=True)
    write_new(output / "protocol.json", info)
    return {"status": info["status"], "protocol": deployment.binding(output / "protocol.json")}


def check(protocol_path, protocol_sha256=None):
    if protocol_sha256 is not None:
        require(sha256(protocol_path) == protocol_sha256, "Frozen protocol hash differs.")
    info = read(Path(protocol_path))
    require(
        info["version"] == VERSION
        and info["status"] == "registered_not_inferred"
        and info["split"] in SPLITS
        and info["final_test_bodies_read"] is False,
        "Unknown or held-out protocol; no request bodies were opened.",
    )
    require(info["environment"] == environment(), "Registered Python environment changed.")
    require(info["sources_sha256"] == source_hashes(), "Registered runner/policy source changed.")
    is_model = info["model_condition"] in MODEL_CONDITIONS
    require(
        info["schema_constrained"] is is_model
        and info["diagnostics"] in (("app_default", "full") if is_model else ("not_applicable",))
        and type(info["timeout_seconds"]) in (int, float)
        and 0 < info["timeout_seconds"] <= 600
        and info["cache_modes"]
        and len(set(info["cache_modes"])) == len(info["cache_modes"])
        and set(info["cache_modes"]) <= ({"cold", "warm"} if is_model else {"not_applicable"}),
        "Registered generation/cache policy differs.",
    )
    rows = load_requests(info["requests"], info["split"], info["model_condition"])
    require(
        [row["case_id"] for row in rows] == info["case_ids"]
        and len(rows) == info["requested"]
        and rows[0]["format"] == info["format"]
        and rows[0].get("condition_sha256") == info["condition_sha256"]
        and rows[0].get("condition", {"id": "canonical_full_context"})["id"] == info["condition"],
        "Registered request census changed.",
    )
    adapter = None
    if info["deployment"]:
        path = deployment.bound(info["deployment"])
        config, _, command, receipt = deployment.check(path.parent)
        identity, bindings = model_identity(config, receipt, info["model_condition"])
        require(
            identity == info["model_identity"] and bindings == info["identity_bindings"],
            "Model identity changed.",
        )
        require(command == info["command"], "Runtime command changed.")
        expected = config.model_copy(
            update={
                "cache_prompt": True,
                "capture_runtime_evidence": info["diagnostics"] == "full",
            }
        )
        require(expected.model_dump() == info["gateway"], "Measurement gateway changed.")
        require(
            runtime_hash(command, receipt, identity, expected) == info["runtime_sha256"]
            and receipt["numerical_regime"] == info["numerical_regime"],
            "Registered runtime hash or numerical regime differs.",
        )
        adapter = FrozenRequestAdapter(expected, rows)
    else:
        require(info["model_condition"] in BASELINES, "A model condition lost its deployment.")
    return info, rows, adapter


class FrozenRequestAdapter(copilot.LocalLlamaAdapter):
    def __init__(self, config, rows):
        super().__init__(config)
        self.allowed = {row["messages_sha256"]: row["messages"] for row in rows}
        self.active_messages = None

    @contextmanager
    def request(self, messages):
        require(self.active_messages is None, "Local evaluation is strictly sequential.")
        require(
            self.allowed.get(value_hash(messages)) == messages, "Unregistered runtime messages."
        )
        self.active_messages = messages
        try:
            yield
        finally:
            self.active_messages = None

    def _model_messages(self, messages):
        if self.active_messages is not None:
            require(messages == self.active_messages, "Active request changed.")
            return messages
        return super()._model_messages(messages)


def http_body(raw):
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


@contextmanager
def capture_http(events):
    # The server adapter owns generation. A process-local proxy adds HTTP hooks
    # without changing its request body, diagnostic flags or response parsing.
    original = copilot.httpx
    pending = {}

    async def sent(request):
        item = {
            "method": request.method,
            "url": str(request.url),
            "started_at": now(),
            "request": http_body(await request.aread()),
            "response": None,
        }
        pending[id(request)] = item
        events.append(item)

    async def received(response):
        item = pending[id(response.request)]
        item.update(
            status_code=response.status_code,
            response=http_body(await response.aread()),
            completed_at=now(),
        )

    class RecordingHTTP:
        def __getattr__(self, name):
            return getattr(original, name)

        def AsyncClient(self, *args, **kwargs):
            require(
                "event_hooks" not in kwargs, "Adapter HTTP hooks changed; review instrumentation."
            )
            return original.AsyncClient(
                *args,
                **kwargs,
                event_hooks={"request": [sent], "response": [received]},
            )

    copilot.httpx = RecordingHTTP()
    try:
        yield
    finally:
        copilot.httpx = original


def cache_evidence(timings, mode, prefix_length, primed):
    timings = timings or {}
    count = timings.get("runtime_cached_tokens")
    usage = timings.get("runtime_usage_cached_tokens", count)
    if (
        type(count) is not int
        or not 0 <= count <= prefix_length
        or type(usage) is not int
        or count != usage
        or timings.get("runtime_timing_evidence") == "inconsistent"
    ):
        return {"status": "missing_or_inconsistent", "cached_tokens": count, "verified": False}
    if mode == "cold":
        passed = count == 0
        state = "verified_cold" if passed else "unexpected_cached_prefix"
    elif mode == "warm":
        passed = primed and count >= max(1, prefix_length - 1)
        state = "verified_warm" if passed else "unprimed_or_partial_cache"
    else:
        passed, state = False, "priming_observation_only"
    return {"status": state, "cached_tokens": count, "verified": passed}


def observation(request, info, mode):
    keys = ("case_id", "input_sha256", "messages_sha256", "prefix_ids_sha256", "format")
    return {
        **{key: request[key] for key in keys},
        **(
            {"condition_sha256": request["condition_sha256"]}
            if "condition_sha256" in request
            else {}
        ),
        "model_identity": info["model_identity"],
        "runtime_sha256": info["runtime_sha256"],
        "cache_mode": mode,
        "status": "not_run",
        "text": None,
        "finish_reason": None,
        "generation_ms": None,
        "valid_output": None,
        "cache_evidence": {"status": "not_observed", "verified": False},
    }


async def generate(adapter, request, info, record, path, primed=False):
    record.update(status="started", started_at=now(), http_records=[])
    write_state(path, record)
    started = time.perf_counter()
    try:
        with adapter.request(request["messages"]), capture_http(record["http_records"]):
            generated = await asyncio.wait_for(
                adapter.generate(
                    request["messages"], formats.contract.OUTPUT_SCHEMA, info["timeout_seconds"]
                ),
                timeout=info["timeout_seconds"],
            )
        record.update(
            status="completed",
            text=generated.text,
            finish_reason=generated.finish_reason,
            timings=generated.timings,
            runtime_evidence=generated.runtime_evidence,
            cache_evidence=cache_evidence(
                generated.timings,
                record["cache_mode"],
                len(request["prefix_ids"]),
                primed,
            ),
        )
        try:
            record["prediction"] = formats.parse(
                generated.text,
                request["input"],
                request["format"],
                finish_reason=generated.finish_reason,
            ).model_dump()
            record["valid_output"] = True
        except ValueError as error:
            record.update(valid_output=False, validation_error=str(error))
    except (asyncio.CancelledError, KeyboardInterrupt) as error:
        record.update(status="cancelled", error_type=type(error).__name__, error=str(error))
        raise
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__, error=str(error))
    finally:
        record.update(generation_ms=(time.perf_counter() - started) * 1000, finished_at=now())
        write_state(path, record)
    return record


def timings_summary(records):
    values = sorted(row["generation_ms"] for row in records if row["status"] == "completed")
    return {
        "requested": len(records),
        "status_counts": dict(Counter(row["status"] for row in records)),
        "valid_outputs": sum(row["valid_output"] is True for row in records),
        "invalid_outputs": sum(row["valid_output"] is False for row in records),
        "cache_verified": sum(row["cache_evidence"]["verified"] for row in records),
        "completed_latency_ms": {
            "count": len(values),
            "p50": statistics.median(values) if values else None,
            "p95": values[math.ceil(len(values) * 0.95) - 1] if values else None,
        },
        "latency_denominator": (
            "completed_requests_including_invalid_outputs; failures_counted_separately"
        ),
    }


def freeze_output(output, status):
    write_new(
        output / "manifest.json",
        {
            "version": VERSION,
            "status": status,
            "files_sha256": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )


def run(protocol_path, output, *, protocol_sha256):
    info, requests, adapter = check(protocol_path, protocol_sha256)
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh local measurement output directory.")
    output.mkdir(parents=True)
    modes = info["cache_modes"]
    observed = {mode: [observation(row, info, mode) for row in requests] for mode in modes}
    priming = [observation(row, info, "priming") for row in requests] if "warm" in modes else []
    state = {
        **info,
        "protocol": deployment.binding(protocol_path),
        "status": "running",
        "started_at": now(),
        "requests_sha256": info["requests"]["sha256"],
        "planned_scored_requests": len(requests) * len(modes),
        "planned_priming_requests": len(priming),
        "startup_attempts": [],
        "model_requests": 0,
        "owned_processes_stopped": True,
    }
    write_new(output / "protocol.json", info)
    write_state(output / "run.json", state)
    fatal = None
    try:
        if adapter is None:
            for request, record in zip(requests, observed["not_applicable"], strict=True):
                started = time.perf_counter()
                try:
                    prediction = conditions.baseline_prediction(
                        request["input"], info["model_condition"]
                    )
                    text = formats.serialize(request["input"], prediction, request["format"])
                    formats.parse(text, request["input"], request["format"])
                    record.update(
                        status="completed", text=text, finish_reason="stop", valid_output=True
                    )
                except Exception as error:
                    record.update(status="error", error_type=type(error).__name__, error=str(error))
                record["generation_ms"] = (time.perf_counter() - started) * 1000
        else:
            with lifecycle.LOCK.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for index, request in enumerate(requests):
                    folder = output / "cases" / f"{index:05d}"
                    folder.mkdir(parents=True)
                    startup = {
                        "case_id": request["case_id"],
                        "model_identity": info["model_identity"],
                        "runtime_sha256": info["runtime_sha256"],
                        "status": "starting",
                        "started_at": now(),
                        "model_started": False,
                        "process_stopped": True,
                        "command": info["command"],
                    }
                    state["startup_attempts"].append(startup)
                    process = None
                    startup_started = time.perf_counter()
                    try:
                        lifecycle.refuse_occupied(adapter.config.endpoint)
                        with (folder / "server.log").open("x") as log:
                            process = subprocess.Popen(
                                info["command"],
                                cwd=ROOT,
                                stdout=log,
                                stderr=log,
                                start_new_session=True,
                            )
                            startup.update(
                                model_started=True, pid=process.pid, process_stopped=False
                            )
                            write_state(folder / "startup.json", startup)
                            lifecycle.wait_model(process, adapter.config.endpoint)
                            adapter._verified_parity = None
                            startup["health"] = adapter.health()
                            require(
                                startup["health"]["status"] == "ready",
                                "Runtime health/parity failed.",
                            )
                            with adapter._client() as client:
                                startup["props"] = adapter._json(client, "GET", "/props")
                                startup["loaded_adapters"] = adapter._json(
                                    client, "GET", "/lora-adapters"
                                )
                                adapter._verify_loaded_adapter(startup["loaded_adapters"])
                            with adapter.request(request["messages"]):
                                actual = adapter.apply_chat_template(
                                    request["messages"],
                                    tokenize=True,
                                    return_dict=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False,
                                )
                            startup["actual_prefix_ids_sha256"] = value_hash(actual)
                            require(
                                actual == request["prefix_ids"],
                                "Runtime/native request prefix differs.",
                            )
                            startup.update(
                                status="ready",
                                ready_ms=(time.perf_counter() - startup_started) * 1000,
                            )
                            write_state(folder / "startup.json", startup)
                            if "cold" in modes:
                                asyncio.run(
                                    generate(
                                        adapter,
                                        request,
                                        info,
                                        observed["cold"][index],
                                        folder / "cold.json",
                                    )
                                )
                            if "warm" in modes:
                                prime = asyncio.run(
                                    generate(
                                        adapter,
                                        request,
                                        info,
                                        priming[index],
                                        folder / "priming.json",
                                    )
                                )
                                if prime["status"] == "completed":
                                    asyncio.run(
                                        generate(
                                            adapter,
                                            request,
                                            info,
                                            observed["warm"][index],
                                            folder / "warm.json",
                                            primed=True,
                                        )
                                    )
                                else:
                                    observed["warm"][index]["error"] = (
                                        "Priming failed; no warm generation issued."
                                    )
                                    write_state(folder / "warm.json", observed["warm"][index])
                    except BaseException as error:
                        startup.update(
                            status="failed", error_type=type(error).__name__, error=str(error)
                        )
                        raise
                    finally:
                        lifecycle.stop_owned(process)
                        startup.update(
                            process_stopped=process is None or process.poll() is not None,
                            process_returncode=process.returncode if process is not None else None,
                            finished_at=now(),
                        )
                        write_state(folder / "startup.json", startup)
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
        all_records = [row for rows in observed.values() for row in rows]
        for record in [*all_records, *priming]:
            if record["status"] == "not_run":
                record.setdefault("error", "Run stopped before this scheduled request.")
        state.update(
            finished_at=now(),
            observed=len(all_records),
            generation_attempts=sum("started_at" in row for row in [*all_records, *priming]),
            model_requests=sum(
                event["method"] == "POST" and event["url"].endswith("/v1/chat/completions")
                for row in [*all_records, *priming]
                for event in row.get("http_records", [])
            ),
            owned_processes_stopped=all(
                row["process_stopped"] for row in state["startup_attempts"]
            ),
            by_cache={mode: timings_summary(rows) for mode, rows in observed.items()},
            priming=timings_summary(priming),
        )
        state["runtime_preflight_passed"] = (
            state["status"] == "inference_complete"
            and state["owned_processes_stopped"]
            and all(row["status"] == "completed" for row in [*all_records, *priming])
            and (adapter is None or all(row["cache_evidence"]["verified"] for row in all_records))
        )
        state["format_outputs_all_valid"] = all(row["valid_output"] is True for row in all_records)
        state["quality_evaluated"] = False
        write_rows(output / "priming.jsonl", priming)
        for mode, rows in observed.items():
            directory = output / mode
            directory.mkdir()
            write_rows(directory / "observations.jsonl", rows)
            write_new(
                directory / "run.json",
                {
                    **state,
                    "cache_mode": mode,
                    "requested": len(requests),
                    "observed": len(rows),
                    "parent_run": str(output / "run.json"),
                },
            )
            freeze_output(directory, state["status"])
        write_state(output / "run.json", state)
        freeze_output(output, state["status"])
    if fatal:
        raise fatal
    return {
        "status": state["status"],
        "runtime_preflight_passed": state["runtime_preflight_passed"],
        "format_outputs_all_valid": state["format_outputs_all_valid"],
        "by_cache": state["by_cache"],
        "priming": state["priming"],
        "manifest": deployment.binding(output / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("register")
    command.add_argument("--requests", type=Path, required=True)
    command.add_argument("--requests-sha256", required=True)
    command.add_argument("--split", choices=SPLITS, required=True)
    command.add_argument("--condition", choices=(*MODEL_CONDITIONS, *BASELINES), required=True)
    command.add_argument("--deployment", dest="deployment_path", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--cache-modes", nargs="+", choices=("cold", "warm"), default=["cold", "warm"]
    )
    command.add_argument("--diagnostics", choices=("app_default", "full"), default="app_default")
    command.add_argument("--timeout", type=float, default=120)
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
            signal.signal(number, lifecycle.interrupted)
        result = run(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if name == "run" and not result["runtime_preflight_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
