"""Isolated concurrent Electron bridge; scripted checks are not model-quality evidence."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "server/scripts"), str(Path(__file__).parent)]

import uvicorn  # noqa: E402
from copilot_p6_roster import RegisteredAdapter, RegisteredRoster, require  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.copilot import (  # noqa: E402
    Completion,
    CopilotService,
    GatewayError,
    contract,
    create_copilot_router,
)
from app.copilot_store import EvidenceError  # noqa: E402

TOKEN = "isolated-synthetic-ui-test"


class ScriptedAdapter:
    contract_version = "passive-nep-actions-v1"
    scripted_fixture = True
    model = {
        "id": "p6-scripted-known-action",
        "kind": "synthetic_fixture_not_a_language_model",
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer": "constant_length_stub_not_native",
    }

    def __init__(self):
        self.prediction = {"action": "no_action"}
        self.delay_ms = 0
        self.started = 0

    def health(self):
        return {"status": "ready", "model": self.model, "message": "Scripted product check."}

    def apply_chat_template(self, messages, **kwargs):
        return [1] * 100

    async def generate(self, messages, output_schema, timeout):
        self.started += 1
        prediction, delay = self.prediction, self.delay_ms
        contract(self.contract_version).parse_snapshot(json.loads(messages[-1]["content"]))
        await asyncio.sleep(delay / 1000)
        return Completion(json.dumps(prediction, ensure_ascii=False), "stop")


class NativeScriptedAdapter(ScriptedAdapter):
    def __init__(self, python):
        super().__init__()
        self.model = {**self.model, "tokenizer": "native_cpu_only_no_model_weights"}
        self.tokenizer = subprocess.Popen(
            [
                str(python),
                "-u",
                str(Path(__file__).with_name("copilot_p6_roster.py")),
                "--tokenizer-server",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=ROOT,
        )
        self.tokenizer_lock = threading.Lock()

    def apply_chat_template(self, messages, **kwargs):
        with self.tokenizer_lock:
            self.tokenizer.stdin.write(json.dumps(messages, ensure_ascii=False) + "\n")
            self.tokenizer.stdin.flush()
            row = json.loads(self.tokenizer.stdout.readline())
        require("ids" in row, row.get("error", "Native fixture tokenizer failed."))
        return row["ids"]

    def close(self):
        self.tokenizer.stdin.close()
        try:
            self.tokenizer.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.tokenizer.terminate()
            self.tokenizer.wait(timeout=10)
        self.tokenizer.stdout.close()


class FixtureService(CopilotService):
    """The stdin fixture boundary is separate from live prediction/feedback events."""

    @staticmethod
    def _journal_entry(event):
        # Fixture restoration needs provenance and ledger fields omitted by product journals.
        return copy.deepcopy(event)

    def __init__(self, adapter, roster, audit, **kwargs):
        super().__init__(adapter, **kwargs)
        self.roster, self.audit = roster, audit

    def _session(self, document_id):
        restore = document_id not in self.sessions
        session = super()._session(document_id)
        if restore:
            seeds = [r for r in self.journals[document_id] if r["kind"] == "fixture_context_seed"]
            if seeds:
                require(len(seeds) == 1, "A fixture session has more than one seed.")
                seed = seeds[0]
                require(
                    seed["roster_manifest_sha256"] == self.roster.manifest_sha256,
                    "Restored fixture belongs to another roster.",
                )
                fixture = self.roster.fixture(seed["case_id"], document_id)
                require(seed["fixture"] == fixture, "Persisted fixture projection changed.")
                ledger = copy.deepcopy(fixture["runtime"].get("feedback_journal")) or {
                    "contract_version": self._contract().VERSION,
                    "document_id": document_id,
                    "records": [],
                }
                failed = {
                    r["request"]["request_id"]
                    for r in self.journals[document_id]
                    if r["kind"] == "prediction_failed"
                }
                ledger["records"].extend(
                    r["ledger_record"]
                    for r in self.journals[document_id]
                    if r.get("ledger_record") is not None
                    and not (
                        r["kind"] == "prediction_completed" and r["request"]["request_id"] in failed
                    )
                )
                session.feedback = self._contract().PassiveFeedbackLedger.restore(ledger)
        return session

    def seed(self, case_id, document_id):
        require(self.active is None, "Cannot hydrate while a request is active.")
        fixture = self.roster.fixture(case_id, document_id)
        media = self.roster.media(case_id)
        self._session(document_id)
        require(not self.journals[document_id], "Fixture hydration requires a new empty session.")
        self._append(
            document_id,
            kind="fixture_context_seed",
            actor="synthetic",
            case_id=case_id,
            roster_manifest_sha256=self.roster.manifest_sha256,
            provenance=(
                "Registered evaluation fixture, not a new model prediction or human feedback."
            ),
            fixture=fixture,
            original_runtime=self.roster.cases[case_id]["runtime"],
            original_source=self.roster.cases[case_id]["source"],
            media=media,
            model_attempted=False,
        )
        self.sessions.pop(document_id)
        self._session(document_id)
        self.adapter.arm(case_id, self.roster.requests[case_id])
        return {"fixture": fixture, "media": media}

    async def predict(self, body, disconnected=None):
        try:
            self._session(body.request.document_id)
            records = self.journals[body.request.document_id]
            seeds = [r for r in records if r["kind"] == "fixture_context_seed"]
            require(len(seeds) == 1, "Actual roster requests require a registered fixture seed.")
            fixture = seeds[0]["fixture"]
            require(
                not any(r["kind"] == "prediction_requested" for r in records),
                "One measured model request per registered case; repeats require another protocol.",
            )
            rt = fixture["runtime"]
            require(
                body.document.model_dump() == fixture["session_document"]
                and body.history == rt["history"]
                and body.focus_uuid == rt["focus_uuid"]
                and body.selection == rt["selection"]
                and body.sequence == fixture["prediction_sequence"],
                "Renderer document/history/focus/sequence differs from the registered fixture.",
            )
            require(
                body.origin == "synthetic"
                and body.trigger == "explicit"
                and body.deadline_ms == 30000,
                "Actual roster requests require the synthetic actor and 30 second UI deadline.",
            )
            self.adapter.arm(fixture["case_id"], self.roster.requests[fixture["case_id"]])
        except ValueError as error:
            self.audit(
                {
                    "kind": "hydration_binding_failure",
                    "request": body.model_dump(),
                    "model_attempted": False,
                    "error": str(error),
                }
            )
            raise GatewayError(409, "roster_binding", str(error)) from error
        return await super().predict(body, disconnected)

    def _append(self, document_id, **fields):
        if fields["kind"] == "prediction_prepared":
            expected = self.adapter.expected
            if (
                fields["bundle"]["input"] != expected["input"]
                or fields["messages"] != expected["messages"]
            ):
                self.audit(
                    {
                        "kind": "input_binding_failure",
                        "case_id": self.adapter.case_id,
                        "model_attempted": False,
                        "observed_bundle": fields["bundle"],
                        "observed_messages": fields["messages"],
                    }
                )
                raise GatewayError(
                    409, "roster_binding", "Gateway input/messages differ from the frozen request."
                )
        return super()._append(document_id, **fields)


@contextmanager
def runtime(args):
    lock = model = model_log = None
    native_fixture = None
    metadata = {
        "pid": os.getpid(),
        "mode": "actual_model" if args.deployment else "scripted",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "deployment": str(args.deployment) if args.deployment else None,
    }
    receipt_path = args.output / f"runtime-{os.getpid()}.json"
    receipt_path.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        adapter = ScriptedAdapter()
        if getattr(args, "native_fixture_tokenizer", None):
            require(
                not args.deployment,
                "Scripted tokenizer controls cannot be used with a model deployment.",
            )
            native_fixture = adapter = NativeScriptedAdapter(args.native_fixture_tokenizer)
        if args.deployment:
            import run_local_copilot_actions_v3 as deployment

            config, adapter, command, receipt = deployment.check(args.deployment)
            deployment.require(
                not receipt["historical_tooling_only"], "A current v3 deployment is required."
            )
            deployment.require(config.version == 2, "Explicit v3 configuration is required.")
            deployment.require(
                config.capture_runtime_evidence is False,
                "UI performance requires capture_runtime_evidence=false.",
            )
            if getattr(args, "roster", None):
                registered = RegisteredRoster(
                    args.roster,
                    args.roster_sha256,
                    args.format_roster,
                    args.format_roster_sha256,
                )
                deployment.require(
                    config.output_format == registered.format_name,
                    "Selected deployment and registered request format differ.",
                )
            lock = deployment.lifecycle.LOCK.open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            deployment.lifecycle.refuse_occupied(config.endpoint)
            model_log = (args.output / f"model-{os.getpid()}.log").open("x")
            model = subprocess.Popen(
                command, cwd=ROOT, stdout=model_log, stderr=model_log, start_new_session=True
            )
            metadata["model_pid"] = model.pid
            deployment.lifecycle.wait_model(model, config.endpoint)
            health = adapter.health()
            deployment.require(health["status"] == "ready", "Runtime identity verification failed.")
            with adapter._client() as client:
                loaded = adapter._json(client, "GET", "/lora-adapters")
            metadata.update(
                deployment_sha256=deployment.digest(args.deployment / "deployment.json"),
                configuration_sha256=deployment.digest(args.deployment / "gateway.json"),
                numerical_regime=receipt["numerical_regime"],
                health=health,
                loaded_adapters=loaded,
                runtime_evidence_enabled=config.capture_runtime_evidence,
                cold_warm_classification=(
                    "Use actual per-completion cache counts; absent counts are unmeasured."
                ),
            )
        metadata.update(model=adapter.model, status="ready")
        receipt_path.write_text(json.dumps(metadata, indent=2) + "\n")
        yield adapter, metadata
        metadata["status"] = "session_closed"
    except BaseException as error:
        metadata.update(status="failed", error=str(error), error_type=type(error).__name__)
        raise
    finally:
        if native_fixture is not None:
            native_fixture.close()
        if model is not None:
            deployment.lifecycle.stop_owned(model)
        if model_log is not None:
            model_log.close()
        if lock is not None:
            lock.close()
        metadata.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            model_process_stopped=model is None or model.poll() is not None,
            model_returncode=model.returncode if model is not None else None,
        )
        receipt_path.write_text(json.dumps(metadata, indent=2) + "\n")


def serve(args, adapter, metadata):
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://127\.0\.0\.1:\d+",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Copilot-Evidence-Head", "Content-Disposition"],
    )
    output_lock = threading.Lock()
    binding_path = args.output / f"bindings-{os.getpid()}.jsonl"

    def binding_audit(row):
        with output_lock, binding_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        **row,
                        "at": datetime.now(timezone.utc).isoformat(),
                        "monotonic_ns": time.monotonic_ns(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    roster = None
    if args.roster:
        roster = RegisteredRoster(
            args.roster, args.roster_sha256, args.format_roster, args.format_roster_sha256
        )
        require(
            getattr(adapter, "output_format", "json") == roster.format_name,
            "Selected deployment and registered request format differ.",
        )
        adapter = RegisteredAdapter(adapter, binding_audit)
        service = FixtureService(
            adapter, roster, binding_audit, feedback_directory=args.feedback_directory
        )
        metadata.update(
            roster_manifest_sha256=roster.manifest_sha256,
            format_name=roster.format_name,
            format_manifest_sha256=roster.format_manifest_sha256,
            planned_cases=len(roster.entries),
            deadline_ms=30000,
        )
    else:
        service = CopilotService(adapter, feedback_directory=args.feedback_directory)
    app.state.copilot = service
    fail_feedback = False
    original_append = service.evidence.append

    def append(document_id, fields, expected_head):
        nonlocal fail_feedback
        if fail_feedback and fields.get("kind") == "feedback":
            fail_feedback = False
            raise EvidenceError("Injected durable feedback write failure.")
        return original_append(document_id, fields, expected_head)

    service.evidence.append = append

    async def auth(request: Request):
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            raise HTTPException(401, "Not authorized")
        if request.url.path.rstrip("/") in ("/copilot/predict", "/copilot/feedback"):
            if (await request.json()).get("origin") not in ("synthetic", "ai_computer_use"):
                raise HTTPException(422, "Automated UI evidence requires an automated actor.")

    app.include_router(create_copilot_router(auth))
    audit = (args.output / f"bridge-{os.getpid()}.jsonl").open("x")
    http_audit = (args.output / f"http-{os.getpid()}.jsonl").open("x")

    @app.middleware("http")
    async def record_http(request: Request, call_next):
        started = time.monotonic_ns()
        body = await request.body()
        response = await call_next(request)
        row = {
            "method": request.method,
            "url": str(request.url),
            "origin": request.headers.get("origin"),
            "body": body.decode("utf-8"),
            "status": response.status_code,
            "started_ns": started,
            "response_ready_ns": time.monotonic_ns(),
            "at": datetime.now(timezone.utc).isoformat(),
        }
        with output_lock:
            http_audit.write(json.dumps(row, ensure_ascii=False) + "\n")
            http_audit.flush()
        return response

    def emit(message, result, started):
        row = {
            "id": message["id"],
            **result,
            "started_ns": started,
            "finished_ns": time.monotonic_ns(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "bridge_pid": os.getpid(),
        }
        with output_lock:
            audit.write(
                json.dumps({"request": message, "response": row}, ensure_ascii=False) + "\n"
            )
            audit.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)

    http_server = http_thread = None
    try:
        if args.http_port:
            http_server = uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=args.http_port, log_level="warning")
            )
            http_thread = threading.Thread(target=http_server.run, daemon=True)
            http_thread.start()
            deadline = time.monotonic() + 10
            while not http_server.started:
                if not http_thread.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError("Owned loopback gateway did not start.")
                time.sleep(0.01)
            metadata["http_endpoint"] = f"http://127.0.0.1:{args.http_port}"
            metadata["transport"] = "renderer_fetch_to_loopback_uvicorn_http"
        else:
            metadata["transport"] = "stdio_to_fastapi_testclient"
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=4) as executor:

            def handle(message):
                nonlocal fail_feedback
                started = time.monotonic_ns()
                try:
                    command = message.get("command")
                    if command == "info":
                        result = {
                            "status": 200,
                            "body": {
                                **metadata,
                                "generation_started": getattr(adapter, "started", None),
                                "active_request": (
                                    service.active.identity.request_id if service.active else None
                                ),
                            },
                        }
                    elif command == "seed_case":
                        require(roster is not None, "Fixture seeding requires a registered roster.")
                        result = {
                            "status": 200,
                            "body": service.seed(message["case_id"], message["document_id"]),
                        }
                    elif command in ("configure", "fail_feedback"):
                        if args.deployment:
                            raise ValueError("Scripted controls are disabled for actual models.")
                        if command == "configure":
                            adapter.prediction = message["prediction"]
                            adapter.delay_ms = message.get("delay_ms", 0)
                        else:
                            fail_feedback = True
                        result = {"status": 200, "body": {"configured": True}}
                    elif command is not None:
                        raise ValueError("Unknown bridge command.")
                    else:
                        body = message.get("body")
                        if message["route"].rstrip("/") in (
                            "/copilot/predict",
                            "/copilot/feedback",
                        ):
                            if body.get("origin") not in ("synthetic", "ai_computer_use"):
                                raise ValueError(
                                    "Automated UI requests require an explicit automated actor."
                                )
                        response = client.request(
                            message["method"],
                            message["route"],
                            json=body,
                            headers=message.get("headers", {}),
                        )
                        result = {
                            "status": response.status_code,
                            "text": response.text,
                            "headers": dict(response.headers),
                        }
                        if response.headers.get("content-type", "").startswith("application/json"):
                            result["body"] = response.json()
                    emit(message, result, started)
                except Exception as error:
                    emit(message, {"status": 500, "body": {"detail": str(error)}}, started)

            for line in sys.stdin:
                executor.submit(handle, json.loads(line))
    finally:
        if http_server is not None:
            http_server.should_exit = True
            http_thread.join(timeout=10)
            metadata["http_server_stopped"] = not http_thread.is_alive()
        audit.close()
        http_audit.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--http-port", type=int)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--roster-sha256")
    parser.add_argument("--format-roster", type=Path)
    parser.add_argument("--format-roster-sha256")
    parser.add_argument("--native-fixture-tokenizer", type=Path)
    args = parser.parse_args()
    if args.deployment:
        require(
            args.http_port and args.roster and args.roster_sha256,
            "Actual-model UI runs require real HTTP and an explicitly pinned registered roster.",
        )
    require(bool(args.roster) == bool(args.roster_sha256), "Provide both roster path and hash.")
    require(
        bool(args.format_roster) == bool(args.format_roster_sha256),
        "Provide both format path and hash.",
    )
    args.output.mkdir(parents=True, exist_ok=True)

    def stop(signum, frame):
        raise KeyboardInterrupt("Owned bridge received a termination signal.")

    signal.signal(signal.SIGTERM, stop)
    with runtime(args) as (adapter, metadata):
        serve(args, adapter, metadata)


if __name__ == "__main__":
    main()
