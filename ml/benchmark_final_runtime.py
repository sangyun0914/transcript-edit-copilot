"""Measure native local HTTP inference on fixed TRAIN inputs; never load final TEST."""

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

import passive_actions as contract

CONDITIONS = ("cache_off", "sequential_cache", "identical_repeat")
SAMPLING = {
    "greedy": {
        "temperature": 0,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "repeat_penalty": 1.0,
    },
    "lfm_vendor": {
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.05,
    },
    "lfm_unsloth": {
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 0.1,
        "min_p": 0.0,
        "repeat_penalty": 1.05,
    },
    "gemma_vendor": {
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    },
}
SCOPE = "HTTP_complete_response_not_UI_latency"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def request(endpoint, route, body=None, timeout=120):
    data = None if body is None else encoded(body)
    req = urllib.request.Request(
        endpoint.rstrip("/") + route,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    record = {}
    try:
        try:
            response = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            record["http_status"] = response.status
            payload = response.read()
        # Stop the HTTP clock before JSON parsing and prediction validation.
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
    except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
        record.update(
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=str(error),
            error_type=type(error).__name__,
        )
        payload = getattr(error, "partial", None)
        if not isinstance(payload, bytes) or not payload:
            return record
    try:
        record["raw_text"] = payload.decode("utf-8")
    except UnicodeDecodeError:
        record["raw_text"] = payload.decode("utf-8", errors="replace")
        record["raw_body_base64"] = base64.b64encode(payload).decode()
    record["raw_body_sha256"] = hashlib.sha256(payload).hexdigest()
    try:
        record["raw"] = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as error:
        record.setdefault("error", str(error))
        record.setdefault("error_type", "invalid_response_json")
    if not 200 <= record["http_status"] < 300:
        record.update(error=f"HTTP {record['http_status']}", error_type="http_error")
    return record


def runtime_schema(value):
    if isinstance(value, list):
        return [runtime_schema(item) for item in value]
    if isinstance(value, dict):
        return {
            key: runtime_schema(item)
            for key, item in value.items()
            if not (key in {"maxLength", "maxItems"} and item > 64)
        }
    return value


def roster(train, live, per_cell=1):
    if type(per_cell) is not int or per_cell < 1:
        raise ValueError("per_cell must be a positive integer.")
    cells = {
        (language, action): []
        for language in ("en", "ko")
        for action in contract.ALLOWED_ACTIONS
    }
    identities = set()
    with Path(train).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "train":
                raise ValueError("Preflight accepts TRAIN only.")
            identity = row["id"]
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("TRAIN case identities must be nonempty and unique.")
            identities.add(identity)
            key = (row["language"], row["primary"]["action"])
            if key not in cells:
                raise ValueError(f"Unexpected TRAIN language/action cell: {key}")
            cells[key].append({"id": identity, "input": row["input"]})
            cells[key].sort(
                key=lambda value: (
                    hashlib.sha256(value["id"].encode()).hexdigest(),
                    value["id"],
                )
            )
            del cells[key][per_cell:]
    rows = []
    for (language, action), candidates in cells.items():
        if len(candidates) < per_cell:
            raise ValueError(
                f"TRAIN cell {language}/{action} needs {per_cell} cases; got {len(candidates)}."
            )
        rows.extend(
            {
                "id": row["id"],
                "language": language,
                "reference_action": action,
                "group": "train_action_language",
                "input": row["input"],
            }
            for row in candidates
        )
    live_id = "shuka-original-history3"
    if live_id in identities:
        raise ValueError("TRAIN identity collides with the live diagnostic.")
    rows.append(
        {
            "id": live_id,
            "language": "ko",
            "reference_action": None,
            "group": "live_diagnostic",
            "input": json.loads(Path(live).read_text())["input"],
        }
    )
    return rows


def settings(model, sampling_profile="greedy", seed=42):
    return {
        "model": model,
        **SAMPLING[sampling_profile],
        "seed": seed,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "next_edit",
                "strict": True,
                "schema": runtime_schema(contract.OUTPUT_SCHEMA),
            },
        },
    }


def lora_binding(gateway_path, model_path):
    if not gateway_path:
        return None
    path = Path(gateway_path).resolve()
    gateway = json.loads(path.read_text())
    adapter = Path(gateway.get("adapter_path") or "").resolve()
    if (gateway.get("version") != 2 or gateway.get("adapter_scale") != 1.0
            or not adapter.is_file() or sha(adapter) != gateway.get("adapter_sha256")
            or Path(gateway["model_path"]).resolve() != Path(model_path).resolve()
            or sha(model_path) != gateway["model_sha256"]):
        raise ValueError("Separate LoRA or base differs from the hash-bound gateway.")
    return {"gateway": str(path), "gateway_sha256": sha(path), "adapter_path": str(adapter),
            "adapter_sha256": gateway["adapter_sha256"], "id": 0, "scale": 1.0,
            "ordinary_lora": True}


def loaded_lora_matches(identity, response):
    value = response.get("raw")
    return ("error" not in response and isinstance(value, list) and len(value) == 1
            and isinstance(value[0], dict) and value[0].get("id") == identity["id"]
            and isinstance(value[0].get("path"), str)
            and Path(value[0].get("path", "")).resolve() == Path(identity["adapter_path"])
            and value[0].get("scale") == identity["scale"]
            and not value[0].get("alora_invocation_tokens"))


def host_context():
    value = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "gpu_utilization_measured": False,
        "power_and_thermal_state_controlled": False,
    }
    if sys.platform == "darwin":
        for key, command in (
            ("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
            ("memory_bytes", ["sysctl", "-n", "hw.memsize"]),
        ):
            try:
                value[key] = subprocess.check_output(
                    command, text=True, timeout=5
                ).strip()
            except (OSError, subprocess.SubprocessError) as error:
                value[key] = {"error": str(error)}
        try:
            displays = json.loads(
                subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    text=True,
                    timeout=15,
                )
            )
            fields = {
                "sppci_model",
                "sppci_vendor",
                "sppci_cores",
                "spdisplays_vram",
                "spdisplays_metalfamily",
            }
            value["gpus"] = [
                {key: item[key] for key in fields if key in item}
                for item in displays.get("SPDisplaysDataType", [])
            ]
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            value["gpus"] = {"error": str(error)}
    return value


def code_hashes():
    root = Path(__file__).resolve().parent
    paths = {Path(__file__).resolve(), contract.PROMPT_PATH.resolve()}
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if (
            source
            and Path(source).suffix == ".py"
            and Path(source).resolve().parent == root
        ):
            paths.add(Path(source).resolve())
    return {str(path): sha(path) for path in sorted(paths)}


def classify(record, snapshot):
    if "error" in record:
        record["outcome"] = "request_error"
        return
    try:
        choice = record["raw"]["choices"][0]
        envelope = contract.Envelope(
            document_id="probe", request_id="probe", revision=0, interaction_epoch=0
        )
        prediction = contract.validate_prediction(
            choice["message"]["content"],
            snapshot,
            envelope,
            envelope,
            choice["finish_reason"],
        )
        record["valid_prediction"] = prediction.model_dump()
        record["outcome"] = (
            "no_action" if prediction.action == "no_action" else "proposal"
        )
    except (ValueError, KeyError, IndexError, TypeError) as error:
        record.update(
            outcome="validation_error",
            validation_error=str(error),
            validation_error_type=type(error).__name__,
        )


def run_cases(endpoint, rows, configuration, conditions, repeats, output):
    results = []
    with (output / "results.jsonl").open("w") as stream:
        for condition in conditions:
            for repeat in range(repeats):
                for row in rows:
                    messages = contract.canonical_messages(row["input"])
                    body = {
                        **configuration,
                        "messages": messages,
                        "cache_prompt": condition != "cache_off",
                    }
                    warmup_ok = True
                    phases = (
                        ("warmup", "measured")
                        if condition == "identical_repeat"
                        else ("measured",)
                    )
                    for phase in phases:
                        record = {
                            key: row[key]
                            for key in ("id", "language", "reference_action", "group")
                        }
                        record.update(
                            condition=condition,
                            repeat=repeat + 1,
                            phase=phase,
                            request_sha256=hashlib.sha256(encoded(body)).hexdigest(),
                            condition_prerequisites_met=warmup_ok,
                        )
                        record.update(request(endpoint, "/v1/chat/completions", body))
                        classify(record, row["input"])
                        if phase == "warmup":
                            warmup_ok = "error" not in record
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        results.append(record)
                        print(
                            json.dumps(
                                {
                                    key: value
                                    for key, value in record.items()
                                    if key not in {"raw", "raw_text", "raw_body_base64"}
                                }
                            ),
                            flush=True,
                        )
    return results


def latency(values):
    times = sorted(row["elapsed_ms"] for row in values)
    return {
        "n": len(times),
        "p50_ms": statistics.median(times) if times else None,
        "p95_ms": times[math.ceil(len(times) * 0.95) - 1] if times else None,
        "min_ms": times[0] if times else None,
        "max_ms": times[-1] if times else None,
    }


def summarize_group(rows):
    http = [row for row in rows if "error" not in row]
    eligible = [row for row in rows if row["condition_prerequisites_met"]]
    return {
        "n": len(rows),
        "http_success": len(http),
        "valid_predictions": sum("valid_prediction" in row for row in rows),
        **{key: value for key, value in latency(http).items() if key != "n"},
        "condition_prerequisite_failures": len(rows) - len(eligible),
        "outcomes": {
            outcome: sum(row["outcome"] == outcome for row in rows)
            for outcome in (
                "proposal",
                "no_action",
                "validation_error",
                "request_error",
            )
        },
        "eligible_latency": {
            outcome: latency([row for row in eligible if row["outcome"] == outcome])
            for outcome in (
                "proposal",
                "no_action",
                "validation_error",
                "request_error",
            )
        },
    }


def summarize(results, conditions):
    summary = {}
    for condition in conditions:
        current = [row for row in results if row["condition"] == condition]
        measured = [row for row in current if row["phase"] == "measured"]
        summary[condition] = {
            **summarize_group(measured),
            "groups": {
                group: summarize_group(
                    [row for row in measured if row["group"] == group]
                )
                for group in ("train_action_language", "live_diagnostic")
            },
            "warmups": summarize_group(
                [row for row in current if row["phase"] == "warmup"]
            ),
        }
    return summary


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18093")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--lora-gateway", help="Optional hash-bound ordinary LoRA gateway.json.")
    parser.add_argument(
        "--train",
        default="artifacts/datasets/nep-passive-actions-v3/release-attempt-003/cases/train.jsonl",
    )
    parser.add_argument(
        "--live", default="artifacts/audits/shuka-live-20261025-001/after-bundle.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-cell", type=positive, default=1)
    parser.add_argument("--sampling-profile", choices=SAMPLING, default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions", choices=CONDITIONS, nargs="+", default=list(CONDITIONS)
    )
    parser.add_argument("--repeats", type=positive, default=1)
    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--runtime-command-file",
        help="Optional saved launch command; recorded as operator-supplied evidence.",
    )
    args = parser.parse_args(argv)
    if args.seed < 0 or len(set(args.conditions)) != len(args.conditions):
        parser.error("Use a nonnegative seed and distinct conditions.")
    return args


def main(argv=None):
    args = arguments(argv)
    rows = roster(args.train, args.live, args.per_cell)
    for row in rows:
        row["messages"] = contract.canonical_messages(row["input"])
        row["messages_sha256"] = hashlib.sha256(encoded(row["messages"])).hexdigest()
    configuration = settings(args.model, args.sampling_profile, args.seed)
    adapter = lora_binding(args.lora_gateway, args.gguf)
    if adapter:
        configuration["lora"] = [{"id": adapter["id"], "scale": adapter["scale"]}]
    protocol = {
        "version": 2,
        "model": args.model,
        "endpoint": args.endpoint,
        "gguf": str(Path(args.gguf).resolve()),
        "gguf_sha256": sha(args.gguf),
        "train": str(Path(args.train).resolve()),
        "train_sha256": sha(args.train),
        "live": str(Path(args.live).resolve()),
        "live_sha256": sha(args.live),
        "scope": SCOPE,
        "final_test_bodies_opened": False,
        "quality_scope": "TRAIN_diagnostic_and_contract_validation_not_independent_quality_or_executed_edit_correctness",
        "settings": configuration,
        "lora": adapter,
        "sampling_profile": args.sampling_profile,
        "order": args.conditions,
        "repeats": args.repeats,
        "per_cell": args.per_cell,
        "roster_selection": "language_en_ko_then_action_order_sha256_utf8_case_id_ascending_seed_independent_v1",
        "roster_unique_messages": len({row["messages_sha256"] for row in rows}),
        "run_label": args.run_label,
        "source_sha256": code_hashes(),
        "host_context": host_context(),
        "roster": rows,
        "timing": {
            "clock": "perf_counter",
            "ends_after": "HTTP_body_read_before_JSON_decode_and_validation",
            "cold_model_loading_included": False,
            "slot_state_reset_between_conditions": False,
            "sequential_order": "whole_roster_per_repeat",
            "identical_repeat": "immediate_logged_warmup_then_measured_same_request",
            "cache_hits_verified": False,
            "ui_latency_measured": False,
        },
    }
    if args.runtime_command_file:
        path = Path(args.runtime_command_file)
        protocol["runtime_launch"] = {
            "source": "operator_supplied_command_file",
            "path": str(path.resolve()),
            "sha256": sha(path),
            "text": path.read_text(),
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    protocol["status"] = "preflight"
    write(output / "protocol.json", protocol)
    protocol["runtime_before"] = {
        route: request(args.endpoint, "/" + route, timeout=10)
        for route in (("health", "props", "lora-adapters") if adapter else ("health", "props"))
    }
    problems = [
        route for route, value in protocol["runtime_before"].items() if "error" in value
    ]
    props = protocol["runtime_before"]["props"].get("raw", {})
    health = protocol["runtime_before"]["health"].get("raw", {})
    if not isinstance(health, dict) or health.get("status") not in {"ok", "ready"}:
        problems.append("health_not_ready")
    if isinstance(props, dict):
        if props.get("model_alias") is not None and props["model_alias"] != args.model:
            problems.append("model_alias_mismatch")
        if (
            props.get("model_path") is not None
            and Path(props["model_path"]).resolve() != Path(args.gguf).resolve()
        ):
            problems.append("model_path_mismatch")
        if props.get("total_slots") is not None and props["total_slots"] != 1:
            problems.append("runtime_requires_one_slot")
    else:
        problems.append("props_not_an_object")
    if adapter and not loaded_lora_matches(adapter, protocol["runtime_before"]["lora-adapters"]):
        problems.append("loaded_lora_mismatch")
    protocol["status"] = "preflight_failed" if problems else "running"
    protocol["preflight_errors"] = problems
    write(output / "protocol.json", protocol)
    if problems:
        raise SystemExit(
            "Runtime preflight failed; see protocol.json. No inference was requested."
        )
    try:
        results = run_cases(
            args.endpoint, rows, configuration, args.conditions, args.repeats, output
        )
    finally:
        results_path = output / "results.jsonl"
        saved = []
        if results_path.exists():
            with results_path.open() as stream:
                saved = [json.loads(line) for line in stream]
        write(output / "summary.json", summarize(saved, args.conditions))
        protocol["runtime_after"] = {
            route: request(args.endpoint, "/" + route, timeout=10)
            for route in (("health", "props", "lora-adapters") if adapter else ("health", "props"))
        }
        protocol["host_context_after"] = {
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None
        }
        protocol["status"] = (
            "completed"
            if len(saved)
            == len(rows)
            * args.repeats
            * sum(2 if c == "identical_repeat" else 1 for c in args.conditions)
            else "interrupted"
        )
        if adapter:
            try:
                unchanged = (loaded_lora_matches(adapter, protocol["runtime_after"]["lora-adapters"])
                             and lora_binding(args.lora_gateway, args.gguf) == adapter)
            except (OSError, ValueError, KeyError, TypeError):
                unchanged = False
            if not unchanged:
                protocol["status"] = "runtime_identity_changed"
        write(output / "protocol.json", protocol)
    if protocol["status"] == "runtime_identity_changed":
        raise SystemExit("Runtime adapter identity changed; all observations retained as inadmissible.")
    return results


if __name__ == "__main__":
    main()
