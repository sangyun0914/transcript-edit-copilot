"""Run a registered all-action pilot through the real local copilot gateway."""

import argparse
import asyncio
import fcntl
import gzip
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from app.copilot import (  # noqa: E402
    Completion,
    CopilotService,
    GatewayError,
    LocalConfig,
    LocalLlamaAdapter,
    PredictRequest,
    schema_for_runtime,
)
import passive_actions as contract  # noqa: E402
from grading import transcript_media  # noqa: E402

PAIRED_REGISTRY = (
    ROOT
    / "artifacts/benchmarks/passive-gemma-caption-prompt-v1/execution-attempt-01/runs/base-original/implementation/model-registry.json"
)
MODEL_IDS = ("gemma26-native-base-q4-k-m", "gemma26-sft203-q4-k-m")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def bound(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Pilot inputs must belong to this repository.")
    return {"path": str(path), "sha256": digest(path)}


def read_bound(value):
    path = Path(value["path"])
    if digest(path) != value["sha256"]:
        raise ValueError(f"Registered input changed: {path}")
    return path


def validate_cohort(cases, runtime, native):
    ids = [case["id"] for case in cases]
    expected_cells = Counter(
        (language, action)
        for language in ("en", "ko")
        for action in contract.ALLOWED_ACTIONS
    )
    if (
        len(ids) != 14
        or len(set(ids)) != 14
        or Counter((c["language"], c["primary"]["action"]) for c in cases)
        != expected_cells
    ):
        raise ValueError("Require one development state per action and language.")
    if len({row["id"] for row in runtime}) != len(runtime):
        raise ValueError("Runtime rows have duplicate identities.")
    if [row["id"] for row in native] != ids or [
        row["id"] for row in runtime if row["split"] == "dev"
    ] != ids:
        raise ValueError("Native/runtime development cohort differs.")
    for case, row in zip(cases, native, strict=True):
        if case["split"] != "dev" or row["split"] != "dev":
            raise ValueError("Only the fixed development cohort belongs in this probe.")
        snapshot = contract.parse_snapshot(case["input"])
        if row["messages"] != contract.canonical_messages(snapshot):
            raise ValueError(
                "Native messages differ from the fixed policy and snapshot."
            )


def verify_runtime_registry(path):
    registry = json.loads(Path(path).read_text())
    for relative, expected in registry["runtime"]["files_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Pinned local runtime dependency changed.")
    for relative, target in registry["runtime"].get("symlink_targets", {}).items():
        if (ROOT / relative).resolve() != (ROOT / target).resolve():
            raise ValueError("Pinned runtime library target changed.")
    return registry


def register(release, configs, protocol, pilot_review):
    if protocol.exists():
        raise ValueError("Use a fresh protocol path.")
    manifest = json.loads((release / "manifest.json").read_text())
    review = json.loads(pilot_review.read_text())
    if review.get("status") != "passed" or review.get("manifest_sha256") != digest(
        release / "manifest.json"
    ):
        raise ValueError("Independent pilot review does not bind this release.")
    files = manifest["files_sha256"]
    for name, expected in files.items():
        path = (release / name).resolve()
        if not path.is_relative_to(release.resolve()) or digest(path) != expected:
            raise ValueError("Pilot release manifest differs.")
    for name, expected in manifest["sources_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Live pilot dependency differs from prepared source.")
    cases = rows(release / "cases/dev.jsonl")
    runtime = rows(release / "runtime-inputs.jsonl")
    native = rows(release / "tokenized/gemma26/json-curriculum-dev.jsonl.gz")
    validate_cohort(cases, runtime, native)
    registry = verify_runtime_registry(PAIRED_REGISTRY)
    candidates = {candidate["id"]: candidate for candidate in registry["candidates"]}
    arms = []
    for condition, model_id, config_path in zip(
        ("base", "sft-t1"), MODEL_IDS, configs, strict=True
    ):
        config = LocalConfig.model_validate(json.loads(config_path.read_text()))
        if config.contract_version != contract.VERSION:
            raise ValueError("Each deployment must use the new all-action contract.")
        candidate = candidates[model_id]
        if (
            config.model_id != model_id
            or config.model_sha256 != candidate["model_sha256"]
            or Path(config.model_path).resolve()
            != (ROOT / candidate["model_path"]).resolve()
            or config.chat_template_sha256 != candidate["template_sha256"]
            or config.runtime_build
            != f"{registry['runtime']['tag']}-{registry['runtime']['revision'][:9]}"
        ):
            raise ValueError(
                "Model arm differs from the genuine paired base/T1 exports."
            )
        export = candidate["export_receipt"]
        read_bound(export)
        arms.append(
            {
                "condition": condition,
                "configuration": bound(config_path),
                "launcher": bound(config_path.with_name("start-model.sh")),
                "parity": bound(config.token_parity_path),
                "paired_export_receipt": export,
            }
        )
    registration = {
        "version": "passive-actions-pilot-probe-v1",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "contract_version": contract.VERSION,
        "release_manifest": bound(release / "manifest.json"),
        "pilot_independent_review": bound(pilot_review),
        "cases": bound(release / "cases/dev.jsonl"),
        "runtime_inputs": bound(release / "runtime-inputs.jsonl"),
        "native_prefixes": bound(
            release / "tokenized/gemma26/json-curriculum-dev.jsonl.gz"
        ),
        "case_ids": [case["id"] for case in cases],
        "arms": arms,
        "runtime_registry": bound(PAIRED_REGISTRY),
        "implementation": [
            bound(ROOT / name)
            for name in sorted(
                set(manifest["sources_sha256"])
                | {
                    "ml/probe_passive_actions.py",
                    "ml/passive_actions.py",
                    "ml/prompt-passive-actions-v1.txt",
                    "ml/grading.py",
                    "server/app/copilot.py",
                }
            )
        ],
        "requests": 28,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 512,
        "schema_constrained": True,
        "cache_prompt": False,
        "no_icl": True,
        "native_generation_stop_ids": [1, 106, 50],
        "raw_format_scope": "Unmodified sampled content and API content are separate; tool-handoff token 50 never receives credit.",
        "scope": "Small authored symbolic capability diagnostic, not natural-audio quality.",
        "reference_scope": "Authored references are not exhaustive intent. Report mechanically valid alternatives separately; exact-reference mismatch alone does not establish an unwanted or harmful edit.",
        "final_test_bodies_read": False,
        "decision": (
            "Report all action/language cells and failures. No promotion or checkpoint selection. "
            "Decide a separately fixed all-action SFT experiment from the complete paired result."
        ),
    }
    protocol.parent.mkdir(parents=True, exist_ok=True)
    with protocol.open("x") as stream:
        json.dump(registration, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


class CapturingAdapter(LocalLlamaAdapter):
    def __init__(self, config, expected):
        super().__init__(config)
        self.expected = expected
        self.observation = {}
        self.case_id = None

    def fail_integrity(self, message):
        self.observation["integrity_failure"] = message
        raise ValueError(message)

    def _json(self, client, method, endpoint, **kwargs):
        value = super()._json(client, method, endpoint, **kwargs)
        if endpoint == "/props":
            self.observed_props = value
        return value

    async def generate(self, messages, output_schema, timeout):
        expected = self.expected[self.case_id]
        if messages != expected["messages"]:
            self.fail_integrity(
                "Gateway messages differ from the registered native input."
            )
        prefix = expected["primary"]["input_ids"][
            : expected["primary"]["completion_start"]
        ]
        actual = await asyncio.to_thread(
            self.apply_chat_template,
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if actual != prefix:
            self.fail_integrity("Actual server prefix differs from native tokenizer.")
        body = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": 0,
            "seed": 42,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": False,
            "stream": False,
            "return_tokens": True,
            "verbose": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "next_edit",
                    "strict": True,
                    "schema": schema_for_runtime(output_schema),
                },
            },
        }
        self.observation = {
            "native_prefix": prefix,
            "server_prefix": actual,
            "request": body,
        }
        async with httpx.AsyncClient(
            base_url=self.config.endpoint,
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            started = time.perf_counter()
            try:
                response = await client.post("/v1/chat/completions", json=body)
            finally:
                self.observation["http_ms"] = (time.perf_counter() - started) * 1000
            self.observation["http_status"] = response.status_code
            self.observation["response_body"] = response.text
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError:
                self.fail_integrity(
                    "Completion did not contain JSON response evidence."
                )
        self.observation["response"] = data
        if not isinstance(data, dict) or (
            data.get("model") != self.config.model_id
            or not isinstance(data.get("choices"), list)
            or len(data["choices"]) != 1
        ):
            self.fail_integrity("Completion model identity or choice count differs.")
        if not isinstance(data.get("usage"), dict) or data["usage"].get(
            "prompt_tokens"
        ) != len(prefix):
            self.fail_integrity("Completion prefix count differs from native parity.")
        try:
            evidence = sampled_evidence(data)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.fail_integrity(f"Raw sampled evidence differs: {error}")
        self.observation.update(evidence)
        if evidence["terminal_token_id"] == 50:
            raise ValueError("Tool-response terminal is outside the edit contract.")
        choice = data["choices"][0]
        return Completion(choice["message"]["content"], choice["finish_reason"])


def sampled_evidence(data):
    raw, usage, choice = data["__verbose"], data["usage"], data["choices"][0]
    ids, text, finish = (
        raw["tokens"],
        choice["message"]["content"],
        choice["finish_reason"],
    )
    if not (
        isinstance(text, str)
        and isinstance(raw["content"], str)
        and isinstance(ids, list)
        and 0 < len(ids) <= 512
        and all(type(token) is int and 0 <= token < 262144 for token in ids)
        and type(usage["completion_tokens"]) is int
        and len(ids) == usage["completion_tokens"]
        and raw["stop_type"] == ("eos" if ids[-1] in (1, 106, 50) else "limit")
        and (
            (ids[-1] in (1, 106) and finish == "stop")
            or (ids[-1] == 50 and finish in ("stop", "tool_calls"))
            or (ids[-1] not in (1, 106, 50) and len(ids) == 512 and finish == "length")
        )
    ):
        raise ValueError(
            "Inconsistent sampled content, token count or terminal behavior."
        )
    formats = {}
    for name, content in (("raw", raw["content"]), ("api", text)):
        try:
            contract.EDIT.validate_json(content, strict=True)
            formats[f"{name}_format_valid"] = True
        except ValueError:
            formats[f"{name}_format_valid"] = False
    return {
        "sampled_content": raw["content"],
        "sampled_token_ids": ids,
        "terminal_token_id": ids[-1],
        "raw_stop_type": raw["stop_type"],
        "sampled_content_equals_api_content": raw["content"] == text,
        "token_evidence_valid": True,
        **formats,
    }


def outcome(case, document, predicted_document):
    return transcript_media(predicted_document) in [
        transcript_media(
            contract.apply_passive_edit(document, case["bundle"], prediction)
        )
        for prediction in case["accepted_predictions"]
    ]


def summarize(cases, observations):
    if [row["case_id"] for row in observations] != [case["id"] for case in cases]:
        raise ValueError("Incomplete or reordered pilot observations.")
    displayed = useful = preservation = invalid = required = preserve_cases = 0
    cells, times = {}, []
    for case, row in zip(cases, observations, strict=True):
        action = case["primary"]["action"]
        needs_edit = action != "no_action"
        shown = row.get("gateway", {}).get("status") == "proposal"
        matched = row.get("outcome_match") is True
        bad = row["status"] != "completed"
        required += needs_edit
        preserve_cases += not needs_edit
        displayed += shown
        useful += shown and matched and needs_edit
        preservation += not needs_edit and matched
        invalid += bad
        key = f"{case['language']}/{action}"
        if key in cells:
            raise ValueError("Duplicate action/language cell in the fixed pilot.")
        cells[key] = {
            "cases": 1,
            "outcome_match": int(matched),
            "invalid": int(bad),
            "displayed": int(shown),
        }
        elapsed = row.get("evidence", {}).get("http_ms")
        if isinstance(elapsed, (int, float)):
            times.append(elapsed)
    precision = useful / displayed if displayed else None
    recall = useful / required if required else None
    f1 = 2 * useful / (displayed + required) if displayed + required else None
    times.sort()
    return {
        "cases": len(cases),
        "displayed": displayed,
        "useful": useful,
        "edit_needed": required,
        "preserve_cases": preserve_cases,
        "preserved": preservation,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "invalid": invalid,
        "coverage": displayed / len(cases),
        "raw_format_valid": sum(
            row.get("evidence", {}).get("raw_format_valid") is True
            for row in observations
        ),
        "api_format_valid": sum(
            row.get("evidence", {}).get("api_format_valid") is True
            for row in observations
        ),
        "tool_handoffs": sum(
            row.get("evidence", {}).get("terminal_token_id") == 50
            for row in observations
        ),
        "cells": cells,
        "http_ms": {
            "count": len(times),
            "p50": statistics.median(times) if times else None,
            "p95": times[math.ceil(len(times) * 0.95) - 1] if times else None,
        },
        "scope": "One authored development state per action/language; no natural-audio accuracy claim.",
    }


async def run(protocol_path, output, protocol_sha256):
    if digest(protocol_path) != protocol_sha256:
        raise ValueError("Registered protocol changed before execution.")
    registration = json.loads(protocol_path.read_text())
    if registration.get("version") != "passive-actions-pilot-probe-v1":
        raise ValueError("Unknown pilot protocol.")
    for item in registration["implementation"]:
        read_bound(item)
    verify_runtime_registry(read_bound(registration["runtime_registry"]))
    cases = rows(read_bound(registration["cases"]))
    runtime_rows = rows(read_bound(registration["runtime_inputs"]))
    native_rows = rows(read_bound(registration["native_prefixes"]))
    validate_cohort(cases, runtime_rows, native_rows)
    inputs = {row["id"]: row for row in runtime_rows}
    expected = {row["id"]: row for row in native_rows}
    read_bound(registration["release_manifest"])
    read_bound(registration["pilot_independent_review"])
    if [case["id"] for case in cases] != registration["case_ids"]:
        raise ValueError("Pilot cohort order changed.")
    if shutil.disk_usage(ROOT).free < 30 * 1024**3 + 256 * 1024**2:
        raise ValueError(
            "Preserve the thirty-GiB reserve and bounded evidence allowance."
        )
    with (ROOT / "artifacts/nep-passive-b1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        state = {
            "version": registration["version"],
            "protocol": bound(protocol_path),
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "arms": [],
        }
        write(output / "pipeline.json", state)
        try:
            for arm in registration["arms"]:
                config_path = read_bound(arm["configuration"])
                launcher = read_bound(arm["launcher"])
                read_bound(arm["parity"])
                read_bound(arm["paired_export_receipt"])
                config = LocalConfig.model_validate(json.loads(config_path.read_text()))
                folder = output / arm["condition"]
                folder.mkdir()
                adapter = CapturingAdapter(config, expected)
                adapter._verify_files()
                with httpx.Client(trust_env=False, timeout=1) as client:
                    try:
                        client.get(config.endpoint + "/health")
                    except httpx.ConnectError:
                        pass
                    else:
                        raise ValueError(
                            "The dedicated pilot endpoint is already occupied."
                        )
                with (folder / "server.log").open("x") as log:
                    process = subprocess.Popen(
                        ["sh", str(launcher)], stdout=log, stderr=log
                    )
                    current = {
                        "condition": arm["condition"],
                        "pid": process.pid,
                        "status": "starting",
                    }
                    state["arms"].append(current)
                    write(output / "pipeline.json", state)
                    try:
                        deadline = time.monotonic() + 180
                        async with httpx.AsyncClient(
                            trust_env=False, timeout=1
                        ) as startup_client:
                            while True:
                                if process.poll() is not None:
                                    raise RuntimeError(
                                        "Owned model process exited during startup."
                                    )
                                try:
                                    response = await startup_client.get(
                                        config.endpoint + "/health"
                                    )
                                    if response.status_code == 200:
                                        break
                                except httpx.RequestError:
                                    pass
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("Owned model startup timed out.")
                                await asyncio.sleep(0.25)
                        try:
                            health = await asyncio.to_thread(adapter.health)
                        finally:
                            if hasattr(adapter, "observed_props"):
                                write(folder / "props.json", adapter.observed_props)
                        if health["status"] != "ready":
                            raise ValueError(
                                "Owned model failed its contract/token parity check."
                            )
                        write(folder / "health.json", health)
                        current["status"] = "running"
                        with (folder / "results.jsonl").open("x") as stream:
                            for case in cases:
                                runtime = inputs[case["id"]]
                                if runtime.get("feedback_journal") is not None:
                                    raise ValueError(
                                        "This initial pilot has no seeded feedback ledger."
                                    )
                                if (
                                    runtime.get("caption_profile")
                                    != contract.CaptionProfile().model_dump()
                                ):
                                    raise ValueError(
                                        "Pilot differs from actual default editor caption settings."
                                    )
                                service = CopilotService(adapter)
                                adapter.case_id, adapter.observation = case["id"], {}
                                body = PredictRequest(
                                    request={
                                        "document_id": case["id"],
                                        "request_id": "pilot-1",
                                        "revision": 0,
                                        "interaction_epoch": 0,
                                    },
                                    document=runtime["document"],
                                    focus_uuid=runtime["focus_uuid"],
                                    selection=runtime["selection"],
                                    sequence=runtime["sequence"],
                                    history=runtime["history"],
                                    trigger="explicit",
                                    deadline_ms=120000,
                                )
                                row = {
                                    "case_id": case["id"],
                                    "condition": arm["condition"],
                                }
                                try:
                                    result = await service.predict(body)
                                    row.update(status="completed", gateway=result)
                                    predicted = (
                                        result["result_document"] or runtime["document"]
                                    )
                                    bundle = contract.PassiveContextBuilder(
                                        adapter
                                    ).build(
                                        runtime["document"],
                                        runtime["focus_uuid"],
                                        as_of=runtime["sequence"],
                                        selection=runtime["selection"],
                                        history=runtime["history"],
                                    )
                                    row["outcome_match"] = outcome(
                                        {**case, "bundle": bundle},
                                        runtime["document"],
                                        predicted,
                                    )
                                except GatewayError as error:
                                    row.update(
                                        status="failed",
                                        error_code=error.code,
                                        error=str(error),
                                        outcome_match=False,
                                    )
                                except Exception as error:
                                    adapter.observation["integrity_failure"] = (
                                        f"{type(error).__name__}: {error}"
                                    )
                                    row.update(
                                        status="failed",
                                        error_code="evidence_integrity",
                                        error=str(error),
                                        outcome_match=False,
                                    )
                                row["evidence"] = adapter.observation
                                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                                stream.flush()
                                if "integrity_failure" in adapter.observation:
                                    raise ValueError(
                                        adapter.observation["integrity_failure"]
                                    )
                                current["requests_completed"] = (
                                    current.get("requests_completed", 0) + 1
                                )
                                write(output / "pipeline.json", state)
                                print(
                                    arm["condition"],
                                    current["requests_completed"],
                                    case["id"],
                                    row["status"],
                                    flush=True,
                                )
                        current["status"] = "completed"
                        write(
                            folder / "summary.json",
                            summarize(cases, rows(folder / "results.jsonl")),
                        )
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                        current["process_exit_code"] = process.returncode
                        write(output / "pipeline.json", state)
            state["status"] = "completed"
            write(
                output / "comparison.json",
                {
                    "protocol": bound(protocol_path),
                    "conditions": {
                        arm["condition"]: json.loads(
                            (output / arm["condition"] / "summary.json").read_text(),
                        )
                        for arm in registration["arms"]
                    },
                    "decision": registration["decision"],
                    "training_started": False,
                },
            )
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            write(output / "pipeline.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("register")
    preparation.add_argument("--release", type=Path, required=True)
    preparation.add_argument("--configs", type=Path, nargs=2, required=True)
    preparation.add_argument("--protocol", type=Path, required=True)
    preparation.add_argument("--pilot-review", type=Path, required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--protocol", type=Path, required=True)
    execution.add_argument("--protocol-sha256", required=True)
    execution.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        register(args.release, args.configs, args.protocol, args.pilot_review)
    else:
        asyncio.run(run(args.protocol, args.output, args.protocol_sha256))


if __name__ == "__main__":
    main()
