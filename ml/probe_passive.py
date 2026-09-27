"""Run the frozen passive B1 screen without modifying its evidence or dependencies."""

import argparse
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from passive_benchmark import (
    baseline_prediction,
    evaluate_result,
    load_screen,
    summarize,
)
from passive_contract import OUTPUT_SCHEMA, canonical_messages
from probe import runtime_schema, sample_rss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "artifacts/datasets/nep-passive-completion-v2/frozen"
DEFAULT_REGISTRY = Path(__file__).with_name("passive-models-v2.json")
BASELINES = ("always_no_action", "consistent_history_rule")
SAMPLING = {
    "temperature": 0,
    "seed": 42,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_tokens": 512,
}


def sampling_options(registry_path, overrides=None):
    settings = {**SAMPLING, **json.loads(Path(registry_path).read_text()).get("sampling", {})}
    if overrides is not None:
        settings.update(json.loads(Path(overrides).read_text()))
    if settings.pop("reasoning", "off") != "off" or settings.pop("reasoning_budget", 0) != 0:
        raise ValueError("This frozen-token runner requires native non-thinking mode")
    if set(settings) != set(SAMPLING):
        raise ValueError("Unknown sampling key; only explicit supported decoding controls are allowed")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in settings.values()):
        raise ValueError("Sampling values must be finite numbers")
    if (settings["max_tokens"] != 512 or type(settings["max_tokens"]) is not int
            or type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**32
            or type(settings["top_k"]) is not int or settings["top_k"] < 0
            or settings["temperature"] < 0 or not 0 < settings["top_p"] <= 1
            or not 0 <= settings["min_p"] <= 1 or settings["repeat_penalty"] <= 0):
        raise ValueError("Invalid sampling controls or changed frozen 512-token output reserve")
    return settings


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def checked_path(path, expected):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(ROOT) or digest(path) != expected:
        raise ValueError(f"Pinned file differs or escapes project: {path}")
    return path


def request_body(case, candidate, *, constrained=True, sampling=None, messages=None):
    body = {
        "model": candidate,
        "messages": canonical_messages(case["input"]) if messages is None else messages,
        **(SAMPLING if sampling is None else sampling),
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": False,
        "stream": False,
        "return_tokens": True,
    }
    if constrained:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "next_edit",
                "strict": True,
                "schema": runtime_schema(OUTPUT_SCHEMA),
            },
        }
    return body


def frozen_prefixes(release, candidate):
    path = release / "tokenized" / candidate / "json-curriculum-dev.jsonl.gz"
    manifest = json.loads((release / "manifest.json").read_text())
    if digest(path) != manifest["files_sha256"][str(path.relative_to(release))]:
        raise ValueError("Frozen tokenizer view changed.")
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {
        row["id"]: row["primary"]["input_ids"][:row["primary"]["completion_start"]]
        for row in rows
    }


def diagnostic_cases(cases, release, protocol_path, candidate, sampling_path):
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("kind") != "native_sampling_diagnostic"
            or protocol.get("release_manifest_sha256") != digest(release / "manifest.json")
            or protocol.get("screening_protocol_sha256") != digest(release / "screening-protocol.json")):
        raise ValueError("Diagnostic protocol does not bind the frozen screen")
    ids = protocol["case_ids"]
    selected = [case for case in cases if case["id"] in ids]
    if not ids or [case["id"] for case in selected] != ids:
        raise ValueError("Diagnostic IDs must be a unique ordered subset of the frozen screen")
    configs = protocol["configs"].get(candidate, [])
    if sampling_path is None or not any(digest(sampling_path) == row["sha256"] for row in configs):
        raise ValueError("Sampling configuration is not registered for this diagnostic candidate")
    return selected


def validate_diagnostic_run(protocol_path, registry_path, candidate, *, unconstrained, timeout):
    protocol = json.loads(protocol_path.read_text())
    fixed = protocol["fixed_conditions"]
    if unconstrained or not fixed["schema_constrained"] or timeout != fixed["request_timeout_seconds"]:
        raise ValueError("Run differs from the diagnostic schema or timeout conditions")
    registry = json.loads(registry_path.read_text())
    identities = protocol["stable_identity_bindings"]
    registered = next(row for row in identities["candidates"] if row["id"] == candidate)
    current = next(row for row in registry["candidates"] if row["id"] == candidate)
    if any(current.get(key) != value for key, value in registered.items()):
        raise ValueError("Diagnostic candidate identity differs from preregistration")
    if (registry["tokenizers"] != identities["tokenizer_registry"]
            or registry["runtime"]["revision"] != identities["runtime_revision"]
            or registry["runtime"]["sha256"] != identities["runtime_launcher_sha256"]):
        raise ValueError("Diagnostic tokenizer or runtime identity differs from preregistration")
    execution_path = protocol_path.with_name("execution.json")
    execution = json.loads(execution_path.read_text())
    if (execution["protocol_sha256"] != digest(protocol_path)
            or execution["registry_sha256"] != digest(registry_path)
            or candidate not in execution["selected_candidate_ids"]):
        raise ValueError("Diagnostic execution does not bind this protocol, registry and candidate")
    return execution_path


def code_bindings(release):
    manifest = json.loads((release / "manifest.json").read_text())
    bindings = {
        name: expected for name, expected in manifest["dependencies_sha256"].items()
        if name.startswith("ml/") and name.endswith(".py")
    }
    for name, expected in bindings.items():
        checked_path(name, expected)
    for name in ("probe_passive.py", "passive_benchmark.py", "probe.py", "context_feedback_report.py"):
        path = Path(__file__).with_name(name)
        bindings[str(path.relative_to(ROOT))] = digest(path)
    return bindings


def server_json(client, endpoint, body):
    response = client.post(endpoint, json=body)
    response.raise_for_status()
    return response.json()


def template_check(client, body, expected):
    rendered = server_json(client, "/apply-template", body)
    # b10816's chat endpoint adds model-specific special tokens after rendering.
    actual = server_json(client, "/tokenize", {
        "content": rendered["prompt"], "add_special": True, "parse_special": True,
    })["tokens"]
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
        min(len(actual), len(expected)) if len(actual) != len(expected) else None,
    )
    return {
        "passed": actual == expected,
        "frozen_tokens": len(expected),
        "runtime_tokens": len(actual),
        "within_budget": len(actual) <= 3584,
        "first_mismatch": mismatch,
        "runtime_token_ids": actual,
        "rendered_prompt": rendered["prompt"],
    }


def inspect_runtime(registry_path, candidate_id):
    registry = json.loads(registry_path.read_text())
    candidate = next(c for c in registry["candidates"] if c["id"] == candidate_id)
    if not candidate.get("model_path") or not candidate.get("model_sha256"):
        raise ValueError(f"{candidate_id}: {candidate['status']}; local weights unavailable")
    model = checked_path(candidate["model_path"], candidate["model_sha256"])
    runtime = registry["runtime"]
    server = checked_path(runtime["server"], runtime["sha256"])
    for name, expected in runtime["files_sha256"].items():
        checked_path(name, expected)
    tokenizer_registry = registry["tokenizers"]
    path = checked_path(
        tokenizer_registry["registry_path"], tokenizer_registry["registry_sha256"]
    )
    tokenizers = json.loads(path.read_text())
    tokenizer_id = candidate.get("tokenizer_id", candidate_id)
    tokenizer = next(c for c in tokenizers["candidates"] if c["id"] == tokenizer_id)
    for name, expected in tokenizer["files_sha256"].items():
        checked_path(str(Path(tokenizer["path"]) / name), expected)
    template = ROOT / tokenizer["path"] / "chat_template.jinja"
    return registry, candidate, tokenizer, model, server, template


def system_output(*command):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run(args, *, prepared=None):
    output = args.output.resolve()
    if output.is_relative_to(args.release.resolve()) or any(
        parent.name == "frozen" and parent.is_relative_to(ROOT / "artifacts/datasets")
        for parent in (output, *output.parents)
    ):
        raise ValueError("Benchmark output must be outside the immutable release")
    cases = prepared["cases"] if prepared else load_screen(args.release)
    summarize_run = prepared["summarize"] if prepared else summarize
    bindings = code_bindings(args.release)
    overrides = getattr(args, "sampling_config", None)
    sampling = sampling_options(args.registry, overrides)
    diagnostic = getattr(args, "diagnostic_protocol", None)
    if prepared and diagnostic:
        raise ValueError("B2 and the separate B1 diagnostic cannot share a run")
    diagnostic_execution = None
    if diagnostic:
        cases = diagnostic_cases(cases, args.release, diagnostic, args.candidate, overrides)
        diagnostic_execution = validate_diagnostic_run(
            diagnostic, args.registry, args.candidate,
            unconstrained=args.unconstrained, timeout=args.timeout,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    implementation = args.output / "implementation"
    implementation.mkdir()
    for name in ("probe_passive.py", "passive_benchmark.py", "probe.py", "context_feedback_report.py"):
        shutil.copy2(Path(__file__).with_name(name), implementation / name)
    shutil.copy2(args.registry, implementation / "model-registry.json")
    if diagnostic:
        shutil.copy2(diagnostic, implementation / "diagnostic-protocol.json")
        shutil.copy2(diagnostic_execution, implementation / "diagnostic-execution.json")
    if overrides:
        shutil.copy2(overrides, implementation / "sampling-config.json")
    if prepared:
        for path in prepared["source_files"]:
            shutil.copy2(path, implementation / path.name)
            bindings[str(path.relative_to(ROOT))] = digest(path)
    messages_by_id = prepared["messages"] if prepared else {}
    reused = prepared["reused_results"] if prepared else {}

    def body_for(case):
        return request_body(case, args.candidate, constrained=not args.unconstrained,
                            sampling=sampling, messages=messages_by_id.get(case["id"]))

    manifest = {
        "version": "passive-b1-runtime-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "candidate": args.candidate,
        "stage": args.stage,
        "release": str(args.release.resolve().relative_to(ROOT)),
        "release_manifest_sha256": digest(args.release / "manifest.json"),
        "screening_protocol_sha256": digest(args.release / "screening-protocol.json"),
        "case_ids": [c["id"] for c in cases],
        "code_sha256": bindings,
        "python_packages": {name: importlib.metadata.version(name) for name in ("httpx", "pydantic")},
        "sampling": sampling,
        "registry_sha256": digest(args.registry),
        "sampling_override_sha256": digest(overrides) if overrides else None,
        "diagnostic_protocol_sha256": digest(diagnostic) if diagnostic else None,
        "diagnostic_execution_sha256": digest(diagnostic_execution) if diagnostic_execution else None,
        "schema_constrained": not args.unconstrained,
        "reasoning": "off; pinned native template enable_thinking=false",
        "condition": (
            "Separate sampling diagnostic; unchanged frozen evidence/output budget; no final ranking."
            if overrides else "Controlled product-budget screen; no vendor-optimal or final-ranking claim."
        ),
        "cache_requested": False,
        "request_timeout_seconds": args.timeout,
        "context_tokens": 4096,
        "latency_scope": "Local HTTP request through validation, execution grading and repeat gate; no editor preview.",
        "http_latency_scope": "Request start to complete HTTP response or transport failure; post-response validation/reference grading is separate. Neither measures editor preview latency.",
        "scope": f"{len(cases)} fixed exposed-development states; passive text/caption only. No final-test access.",
        "platform": platform.platform(),
        "cpu": system_output("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": system_output("sysctl", "-n", "hw.memsize"),
        "swap_before": system_output("sysctl", "vm.swapusage"),
        "vm_stat_before": system_output("vm_stat"),
        "editor_state": args.editor_state,
    }
    if prepared:
        manifest.update(version="passive-b2-runtime-v1", benchmark=prepared["binding"],
                        scope=f"{len(cases)} registered exposed-development states in the B2 condition.",
                        condition=prepared["binding"]["condition"])
    write(args.output / "manifest.json", manifest)
    results, process, sampler, readings = [], None, None, []
    stop = threading.Event()
    try:
        if args.candidate in BASELINES:
            with (args.output / "results.jsonl").open("x") as stream:
                for case in cases:
                    started = time.perf_counter()
                    text = json.dumps(baseline_prediction(case, args.candidate), ensure_ascii=False)
                    row = evaluate_result(case, text)
                    row.update(raw_text=text, complete_json_ms=(time.perf_counter() - started) * 1000)
                    results.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.update(status="completed", backend="deterministic_rule")
            return
        registry, candidate, tokenizer, model, server, template = inspect_runtime(
            args.registry, args.candidate
        )
        manifest.update(
            registry_sha256=digest(args.registry), candidate_binding=candidate,
            tokenizer_binding=tokenizer, runtime_binding=registry["runtime"],
            runtime_version=system_output(str(server), "--version"),
            template_policy="Explicit frozen native template; embedded GGUF default is not assumed identical.",
        )
        expected = prepared["prefixes"] if prepared else frozen_prefixes(
            args.release, candidate.get("tokenizer_id", args.candidate)
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [
            str(server), "-m", str(model), "--alias", args.candidate,
            "--host", "127.0.0.1", "--port", str(port), "-c", "4096",
            "-np", "1", "-ngl", "99", "--jinja", "--no-ui", "--perf",
            "--metrics", "--cache-ram", "0", "--reasoning", "off",
            "--reasoning-budget", "0", "--chat-template-file", str(template),
            "--chat-template-kwargs", '{"enable_thinking":false}',
        ]
        manifest["command"] = command
        write(args.output / "manifest.json", manifest)
        with (args.output / "server.log").open("x") as log:
            start = time.perf_counter()
            process = subprocess.Popen(command, stdout=log, stderr=log)
            sampler = threading.Thread(target=sample_rss, args=(process.pid, stop, readings), daemon=True)
            sampler.start()
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=args.timeout,
                              trust_env=False) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("llama-server exited during startup; see server.log")
                    try:
                        if client.get("/health", timeout=1).status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    if time.perf_counter() - start > 180:
                        raise TimeoutError("Model loading/warmup exceeded 180 seconds")
                    time.sleep(0.25)
                manifest["load_and_warmup_seconds"] = time.perf_counter() - start
                response = client.get("/props")
                response.raise_for_status()
                write(args.output / "props.json", response.json())
                preflight = []
                with (args.output / "template-checks.jsonl").open("x") as stream:
                    for case in cases:
                        row = {"case_id": case["id"], **template_check(
                            client, body_for(case),
                            expected[case["id"]],
                        )}
                        preflight.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                manifest["template_checks"] = {
                    "requests": len(preflight),
                    "passed": sum(r["passed"] and r["within_budget"] for r in preflight),
                }
                if manifest["template_checks"]["passed"] != len(cases):
                    raise ValueError("Runtime template/token mismatch; benchmark generation is blocked")
                print(f"{args.candidate}: frozen template/token parity {len(cases)}/{len(cases)}", flush=True)
                if args.stage == "preflight":
                    manifest["status"] = "template_preflight_passed_generation_unrun"
                    return
                with (args.output / "results.jsonl").open("x") as stream:
                    for index, case in enumerate(cases):
                        if case["id"] in reused:
                            row = reused[case["id"]]
                            results.append(row)
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            stream.flush()
                            continue
                        body = body_for(case)
                        row = {"case_id": case["id"], "raw_valid": False, "displayed": False}
                        started = time.perf_counter()
                        abort = False
                        try:
                            response = client.post("/v1/chat/completions", json=body)
                            row["http_response_ms"] = (time.perf_counter() - started) * 1000
                            row["http_status"] = response.status_code
                            if not response.is_success:
                                row["http_error_body"] = response.text
                            response.raise_for_status()
                            data = response.json()
                            row["response"] = data
                            choice = data["choices"][0]
                            text = choice["message"].get("content") or ""
                            row.update(raw_text=text, finish_reason=choice["finish_reason"])
                            if data.get("usage", {}).get("prompt_tokens") != len(expected[case["id"]]):
                                abort = True
                                raise ValueError("Completion prompt token count differs from verified frozen prefix")
                            row.update(evaluate_result(case, text, choice["finish_reason"]))
                        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
                            row.update(error=f"{type(error).__name__}: {error}",
                                       request_failed=True, raw_valid=False, displayed=False)
                            row["error_type"] = (
                                "timeout" if isinstance(error, httpx.TimeoutException) else
                                "runtime_mismatch" if abort else "request_failure"
                            )
                            abort = abort or isinstance(error, httpx.TimeoutException)
                        except KeyboardInterrupt:
                            row.update(error="Cancelled by operator", request_failed=True,
                                       cancelled=True, error_type="cancelled")
                            abort = True
                        row["complete_json_ms"] = (time.perf_counter() - started) * 1000
                        if "http_response_ms" not in row:
                            row["http_response_ms"] = row["complete_json_ms"]
                        row["post_response_processing_ms"] = max(
                            0, row["complete_json_ms"] - row["http_response_ms"]
                        )
                        row["cache_requested"] = False
                        results.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        print(f"{args.candidate} {index + 1}/{len(cases)} {case['id']} "
                              f"{row['complete_json_ms']:.0f}ms valid={row['raw_valid']} "
                              f"shown={row['displayed']}", flush=True)
                        if abort:
                            raise RuntimeError("Stopped after cancellation/timeout or runtime mismatch; remaining requests are missing, not abstentions")
                response = client.get("/metrics")
                (args.output / "metrics.txt").write_text(response.text)
                manifest["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stop.set()
        if sampler is not None:
            sampler.join(timeout=2)
        manifest.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            sampled_peak_server_rss_bytes=max(readings, default=None),
            rss_samples=len(readings),
            swap_after=system_output("sysctl", "vm.swapusage"),
            vm_stat_after=system_output("vm_stat"),
            completed_requests=len(results),
            reused_results=sum(bool(row.get("reused_from")) for row in results),
        )
        write(args.output / "manifest.json", manifest)
        if args.stage == "screen" or args.candidate in BASELINES:
            write(args.output / "summary.json", summarize_run(cases, results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "screen"), default="screen")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--unconstrained", action="store_true")
    parser.add_argument("--sampling-config", type=Path,
                        help="JSON decoding overrides for a separate diagnostic; frozen output budget remains 512")
    parser.add_argument("--diagnostic-protocol", type=Path,
                        help="Hash-bound preregistered ordered subset and candidate sampling configurations")
    parser.add_argument("--editor-state", default="Not controlled; no editor-preview latency claim")
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error("Request timeout must be positive and at most 600 seconds")
    lock_path = ROOT / "artifacts/nep-passive-b1.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another passive benchmark owns the local predictor")
        run(args)


if __name__ == "__main__":
    main()
