"""Run synthetic cases against one locally owned llama-server process."""

import argparse
import hashlib
import importlib
import json
import math
import platform
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from grading import grade, load_executions
from model_input import messages

HERE = Path(__file__).resolve().parent


def runtime_schema(value):
    """Avoid llama.cpp's bounded-repetition expansion; validate full limits after decoding."""
    if isinstance(value, list):
        return [runtime_schema(item) for item in value]
    if isinstance(value, dict):
        return {
            key: runtime_schema(item)
            for key, item in value.items()
            if not (key in ("maxLength", "maxItems") and item > 64)
        }
    return value


def percentile(values, fraction):
    return (
        sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]
        if values
        else None
    )


def sample_rss(pid, stop, readings):
    while not stop.is_set():
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
        )
        if result.stdout.strip().isdigit():
            readings.append(int(result.stdout.strip()) * 1024)
        stop.wait(0.1)


def run(args):
    if args.execution_fixtures and args.contract_version < 2:
        raise ValueError(
            "Execution grading fixtures require contract version 2 or later."
        )
    output_format = getattr(args, "output_format", "json")
    if output_format == "rewrite" and args.contract_version != 4:
        raise ValueError("Scoped Rewrite requires contract version 4.")
    executions = (
        load_executions(args.execution_fixtures) if args.execution_fixtures else {}
    )
    contract = importlib.import_module(
        f"nep_v{args.contract_version}" if args.contract_version >= 2 else "nep"
    )
    EDIT, Envelope, Snapshot, validate_prediction = (
        contract.EDIT,
        contract.Envelope,
        contract.Snapshot,
        contract.validate_prediction,
    )
    cases = [
        json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()
    ]
    for case in cases:
        if case.get("metadata", {}).get("requires_regeneration"):
            raise ValueError(
                "Human review feedback must be regenerated before prediction evaluation."
            )
        context = case["input"]
        if args.history_mode == "omit":
            context["recent_edits"] = []
        if args.contract_version >= 3 and args.feedback_mode == "omit":
            context["recent_feedback"] = []
        if args.padding_items:
            background = [
                {
                    "id": 1000 + index,
                    "type": "text",
                    "text": "이 대화에서는 연구 과정을 차례대로 설명하고 실험의 조건과 관찰한 내용을 함께 검토합니다.",
                    "duration_ms": 4000,
                    "paragraph": 0,
                    "speaker": "s0",
                    "language": "ko",
                    "source_backed": True,
                    "source_run": 0,
                    "boundary_status": "unverified",
                    "audio_class": "speech",
                }
                for index in range(args.padding_items)
            ]
            for item in context["items"]:
                item["paragraph"] += 1
            context["items"] = background + context["items"]
            context["protected_ids"] += [item["id"] for item in background]
    snapshots = [Snapshot.model_validate(case["input"]) for case in cases]
    prompt_path = args.prompt or HERE / (
        "prompt-rewrite-v1.txt"
        if output_format == "rewrite"
        else f"prompt-v{args.contract_version}.txt"
        if args.contract_version >= 2
        else "prompt.txt"
    )
    prompt = prompt_path.read_text()
    schema = EDIT.json_schema()
    if output_format == "rewrite":
        from format_comparison import Rewrite, parse_rewrite

        schema = Rewrite.model_json_schema()
        validate_prediction = parse_rewrite
    if args.max_tokens is None:
        from format_comparison import BUDGETS

        args.max_tokens = (
            BUDGETS["output_tokens"][output_format]
            if args.contract_version == 4
            else 192
        )
    decoding_schema = runtime_schema(schema) if args.contract_version >= 2 else schema
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "seed-cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    )
    (args.output / "prompt.txt").write_text(prompt)
    (args.output / "edit.schema.json").write_text(json.dumps(schema, indent=2))
    (args.output / "decoding.schema.json").write_text(
        json.dumps(decoding_schema, indent=2)
    )
    with args.model.open("rb") as file:
        model_hash = hashlib.file_digest(file, "sha256").hexdigest()
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    command = [
        str(args.server.resolve()),
        "-m",
        str(args.model.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(args.context),
        "-np",
        "1",
        "-ngl",
        "99",
        "--jinja",
        "--no-ui",
        "--cors-origins",
        f"http://127.0.0.1:{port}",
        "--log-verbosity",
        "4",
        "--perf",
        "--metrics",
        "--cache-ram",
        "0",
        "--reasoning",
        args.reasoning,
        "--reasoning-budget",
        str(args.reasoning_budget),
    ]
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(
            Path(contract.__file__).read_bytes()
        ).hexdigest(),
        "contract_version": args.contract_version,
        "output_format": output_format,
        "output_format_version": 1
        if output_format == "rewrite"
        else args.contract_version,
        "history_mode": args.history_mode,
        "feedback_mode": getattr(args, "feedback_mode", "keep"),
        "model_file": args.model.name,
        "model_sha256": model_hash,
        "model_bytes": args.model.stat().st_size,
        "runtime": subprocess.check_output(
            [str(args.server.resolve()), "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip(),
        "command": command,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "physical_memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "swap_before": subprocess.check_output(
            ["sysctl", "vm.swapusage"], text=True
        ).strip(),
        "editor_state": args.editor_state,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "execution_fixtures_sha256": hashlib.sha256(
            args.execution_fixtures.read_bytes()
        ).hexdigest()
        if args.execution_fixtures
        else None,
        "effective_cases_sha256": hashlib.sha256(
            (args.output / "seed-cases.jsonl").read_bytes()
        ).hexdigest(),
        "padding_items": args.padding_items,
        "schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode()
        ).hexdigest(),
        "decoding_schema_sha256": hashlib.sha256(
            json.dumps(decoding_schema, sort_keys=True).encode()
        ).hexdigest(),
        "decoding_schema_policy": "V2 large length bounds enforced by post-decode validation",
        "reasoning": args.reasoning,
        "reasoning_budget": args.reasoning_budget,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 42,
        "repeats": args.repeats,
        "latency_scope": "HTTP request through complete JSON parsing and semantic validation; no editor preview",
        "quality_scope": "agreement with synthetic draft expectations; not human-labeled quality",
    }
    results = []
    readings = []
    stop = threading.Event()
    started = time.perf_counter()
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        sampler = threading.Thread(
            target=sample_rss, args=(process.pid, stop, readings), daemon=True
        )
        sampler.start()
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=args.timeout,
                trust_env=False,
            ) as client:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "llama-server stopped during startup; see server.log"
                        )
                    try:
                        if client.get("/health", timeout=1).status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("Local model startup exceeded 180 seconds.")
                    time.sleep(0.25)
                manifest["process_startup_with_warmup_seconds"] = (
                    time.perf_counter() - started
                )
                (args.output / "props.json").write_text(
                    json.dumps(client.get("/props").json(), indent=2)
                )
                with (args.output / "results.jsonl").open("w") as output:
                    for case, snapshot in zip(cases, snapshots, strict=True):
                        for repeat in range(args.repeats):
                            request = Envelope(
                                document_id=case["id"],
                                request_id=f"{case['id']}-{repeat}",
                                revision=0,
                                interaction_epoch=0,
                            )
                            body = {
                                "model": args.model.name,
                                "messages": messages(case["input"], prompt),
                                "response_format": {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "next_edit",
                                        "strict": True,
                                        "schema": decoding_schema,
                                    },
                                },
                                "temperature": 0,
                                "seed": 42,
                                "max_tokens": args.max_tokens,
                                "cache_prompt": repeat > 0,
                                "stream": False,
                            }
                            row = {
                                "case_id": case["id"],
                                "repeat": repeat,
                                "cache_requested": repeat > 0,
                                "valid": False,
                                "expected_match": False,
                                "reference_action": (
                                    case.get("expectedOutput") or {}
                                ).get("action"),
                                "reference_known": case.get("expectedOutput")
                                is not None,
                            }
                            begin = time.perf_counter()
                            try:
                                response = client.post(
                                    "/v1/chat/completions", json=body
                                )
                                response.raise_for_status()
                                data = response.json()
                                row["response"] = data
                                choice = data["choices"][0]
                                prediction = validate_prediction(
                                    choice["message"].get("content") or "",
                                    snapshot,
                                    request,
                                    request,
                                    choice["finish_reason"],
                                )
                                grading = grade(
                                    case,
                                    prediction.model_dump(),
                                    executions.get(case["id"]),
                                )
                                row.update(
                                    valid=True,
                                    prediction=prediction.model_dump(),
                                    expected_match=prediction.model_dump()
                                    == case["expectedOutput"],
                                    grading=grading,
                                )
                            except (
                                httpx.HTTPError,
                                ValueError,
                                KeyError,
                                IndexError,
                                TypeError,
                            ) as error:
                                row["error"] = f"{type(error).__name__}: {error}"
                                if isinstance(error, httpx.HTTPStatusError):
                                    row["http_error_body"] = error.response.text[:2000]
                            row["complete_json_ms"] = (
                                time.perf_counter() - begin
                            ) * 1000
                            results.append(row)
                            output.write(json.dumps(row, ensure_ascii=False) + "\n")
                            output.flush()
                            print(
                                case["id"],
                                repeat,
                                round(row["complete_json_ms"]),
                                "ms",
                                "valid" if row["valid"] else "invalid",
                                "draft match" if row["expected_match"] else "differs",
                                flush=True,
                            )
                (args.output / "metrics.txt").write_text(client.get("/metrics").text)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stop.set()
            sampler.join(timeout=2)
            manifest["sampled_peak_server_rss_bytes"] = max(readings, default=0)
            manifest["rss_samples"] = len(readings)
            manifest["swap_after"] = subprocess.check_output(
                ["sysctl", "vm.swapusage"], text=True
            ).strip()
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            summary = {}
            for cached in (False, True):
                rows = [row for row in results if row["cache_requested"] == cached]
                latencies = [row["complete_json_ms"] for row in rows if row["valid"]]
                summary["reuse_requested" if cached else "no_reuse_requested"] = {
                    "requests": len(rows),
                    "valid": sum(row["valid"] for row in rows),
                    "draft_matches": sum(row["expected_match"] for row in rows),
                    "outcome_matches": sum(
                        row.get("grading", {}).get("outcome_match") is True
                        for row in rows
                    ),
                    "reference_known": sum(row["reference_known"] for row in rows),
                    "shown": sum(
                        row["valid"] and row["prediction"]["action"] != "no_action"
                        for row in rows
                    ),
                    "shown_draft_matches": sum(
                        row["valid"]
                        and row["prediction"]["action"] != "no_action"
                        and row["expected_match"]
                        for row in rows
                    ),
                    "shown_unknown": sum(
                        row["valid"]
                        and row["prediction"]["action"] != "no_action"
                        and not row["reference_known"]
                        for row in rows
                    ),
                    "per_reference_action": {
                        action: {
                            "cases": sum(r["reference_action"] == action for r in rows),
                            "draft_matches": sum(
                                r["reference_action"] == action and r["expected_match"]
                                for r in rows
                            ),
                        }
                        for action in sorted(
                            {
                                r["reference_action"]
                                for r in rows
                                if r["reference_known"]
                            }
                        )
                    },
                    "all_response_p50_ms": percentile(
                        [r["complete_json_ms"] for r in rows], 0.5
                    ),
                    "all_response_p95_ms": percentile(
                        [r["complete_json_ms"] for r in rows], 0.95
                    ),
                    "valid_response_p50_ms": percentile(latencies, 0.5),
                    "valid_response_p95_ms": percentile(latencies, 0.95),
                }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=HERE / "seed-cases.jsonl")
    parser.add_argument(
        "--prompt", type=Path, help="Alternate prompt for controlled comparisons"
    )
    parser.add_argument(
        "--execution-fixtures",
        type=Path,
        help="Trusted pre-states/maps for equivalent-correction grading",
    )
    parser.add_argument("--contract-version", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--output-format", choices=["json", "rewrite"], default="json")
    parser.add_argument("--history-mode", choices=["keep", "omit"], default="keep")
    parser.add_argument("--feedback-mode", choices=["keep", "omit"], default="keep")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--reasoning", choices=["off", "on"], default="off")
    parser.add_argument("--reasoning-budget", type=int, default=0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Default: v4 JSON 256 / Rewrite 512; historical contracts 192",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--editor-state", default="not controlled")
    parser.add_argument(
        "--padding-items",
        type=int,
        default=0,
        help="Synthetic protected context for latency scaling; not real-session quality evidence",
    )
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or args.timeout <= 0:
        parser.error("Use 1–10 repeats and a positive request timeout.")
    if not 0 <= args.padding_items <= 50:
        parser.error("Use 0–50 padding items (total input must stay within 64 items).")
    if args.contract_version >= 2 and args.padding_items:
        parser.error(
            "V2 compares actual windows; synthetic padding is only supported by the V1 probe."
        )
    run(args)
