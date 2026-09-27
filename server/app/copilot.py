"""Authenticated, opt-in local gateway for the shared passive NEP contract."""

import asyncio
import hashlib
import importlib
import ipaddress
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "passive-nep-v1"
CONTRACT_MODULES = {
    "passive-nep-v1": "passive_contract",
    "passive-nep-actions-v1": "passive_actions",
}
CONTRACT_ACTIONS = {
    "passive-nep-v1": (
        ["no_action", "correct_text", "split_paragraph", "merge_paragraph"],
        ["delete_span", "trim_silence", "remove_silence"],
    ),
    "passive-nep-actions-v1": (
        [
            "no_action",
            "correct_text",
            "delete_span",
            "trim_silence",
            "remove_silence",
            "split_paragraph",
            "merge_paragraph",
        ],
        [],
    ),
}
ROOT = Path(__file__).resolve().parents[2]


def contract(version=CONTRACT_VERSION):
    directory = str(ROOT / "ml")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    if version not in CONTRACT_MODULES:
        raise ValueError("Unknown passive contract version.")
    return importlib.import_module(CONTRACT_MODULES[version])


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RequestIdentity(StrictModel):
    request_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(min_length=1, max_length=256)
    revision: int = Field(ge=0)
    interaction_epoch: int = Field(ge=0)


class SessionDocument(StrictModel):
    content: list[dict]
    metadata: dict


class PredictRequest(StrictModel):
    request: RequestIdentity
    document: SessionDocument
    focus_uuid: str = Field(min_length=1)
    selection: list[str] = Field(default_factory=list)
    sequence: int = Field(ge=0)
    history: list[dict] = Field(default_factory=list, max_length=128)
    trigger: Literal["automatic", "explicit"]
    deadline_ms: int = Field(default=30000, ge=100, le=120000)


class FeedbackRequest(StrictModel):
    document_id: str = Field(min_length=1, max_length=256)
    proposal_id: str = Field(min_length=1, max_length=256)
    outcome: Literal[
        "shown", "accepted", "dismissed", "modified", "undone", "expired", "invalidated"
    ]
    sequence: int = Field(ge=0)
    origin: Literal["human", "ai_computer_use", "synthetic"] = "human"
    reason: Literal["unspecified", "timing", "target", "parameter", "preference"] = "unspecified"
    transition: dict | None = None


class LocalConfig(StrictModel):
    version: Literal[1] = 1
    contract_version: Literal["passive-nep-v1", "passive-nep-actions-v1"] = CONTRACT_VERSION
    endpoint: str
    model_id: str = Field(min_length=1)
    model_path: str
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chat_template_path: str
    chat_template_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    runtime_build: str = Field(min_length=1)
    token_parity_path: str | None = None
    token_parity_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    cache_prompt: bool = False
    capture_runtime_evidence: bool = False

    @model_validator(mode="after")
    def local_only(self):
        if (self.token_parity_path is None) != (self.token_parity_sha256 is None):
            raise ValueError("A native-token parity receipt requires both its path and hash.")
        url = urlsplit(self.endpoint)
        try:
            local = ipaddress.ip_address(url.hostname or "").is_loopback
        except ValueError:
            local = False
        if (
            url.scheme != "http"
            or not local
            or url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
            or url.port is None
        ):
            raise ValueError("NEP requires an explicit loopback HTTP endpoint with a port.")
        return self


class GatewayError(Exception):
    def __init__(self, status_code, code, message):
        self.status_code, self.code, self.message = status_code, code, message
        super().__init__(message)


@dataclass
class Completion:
    text: str
    finish_reason: str
    timings: dict | None = None
    runtime_evidence: dict | None = None


def inference_timings(result, elapsed_ms, cache_prompt):
    evidence = {
        "runtime_http_ms": elapsed_ms,
        "runtime_cache_requested": "on" if cache_prompt else "off",
    }
    timing = result.get("timings")
    if not isinstance(timing, dict):
        timing = {}
    fields = {
        "prompt_ms": "runtime_prompt_ms",
        "predicted_ms": "runtime_decode_ms",
        "prompt_n": "runtime_prompt_tokens",
        "predicted_n": "runtime_output_tokens",
        "cache_n": "runtime_cached_tokens",
    }
    for source, target in fields.items():
        value = timing.get(source)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            if source.endswith("_n") and type(value) is not int:
                continue
            evidence[target] = value
    usage = result.get("usage")
    detail = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    cached = detail.get("cached_tokens") if isinstance(detail, dict) else None
    if type(cached) is int and cached >= 0:
        evidence["runtime_usage_cached_tokens"] = cached
    evidence["runtime_timing_evidence"] = (
        "complete" if all(key in evidence for key in fields.values()) else "partial"
    )
    if "runtime_cached_tokens" in evidence and cached is not None:
        if type(cached) is not int or cached != evidence["runtime_cached_tokens"]:
            evidence["runtime_timing_evidence"] = "inconsistent"
    return evidence


def schema_for_runtime(value):
    if isinstance(value, list):
        return [schema_for_runtime(item) for item in value]
    if isinstance(value, dict):
        return {
            key: schema_for_runtime(item)
            for key, item in value.items()
            if not (key in {"maxLength", "maxItems"} and item > 64)
        }
    return value


class DisabledAdapter:
    model = None

    def __init__(
        self, message="Local copilot is disabled. Configure AUDAPOLIS_NEP_CONFIG to opt in."
    ):
        self.message = message

    def health(self):
        return {"status": "disabled", "model": None, "message": self.message}


class LocalLlamaAdapter:
    """Connect to an explicitly configured server; never own its process lifecycle."""

    def __init__(self, config):
        self.config = config
        self.contract_version = config.contract_version
        self.model = {
            "id": config.model_id,
            "sha256": config.model_sha256,
            "chat_template_sha256": config.chat_template_sha256,
            "runtime_build": config.runtime_build,
            "training_token_parity": "unverified",
            "contract_version": self.contract_version,
            "cache_prompt": config.cache_prompt,
        }
        self._verified_files = None
        self._verified_parity = None

    def _client(self):
        return httpx.Client(
            base_url=self.config.endpoint.rstrip("/"),
            timeout=5,
            trust_env=False,
            follow_redirects=False,
        )

    def _json(self, client, method, endpoint, **kwargs):
        response = client.request(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()

    def _verify_files(self):
        model, template = Path(self.config.model_path), Path(self.config.chat_template_path)
        signature = tuple(
            (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in (model, template)
        )
        if signature != self._verified_files:
            for path, expected in (
                (model, self.config.model_sha256),
                (template, self.config.chat_template_sha256),
            ):
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError("Configured local model or native template hash differs.")
            self._verified_files = signature

    def health(self):
        self._verify_files()
        with self._client() as client:
            self._json(client, "GET", "/health")
            props = self._json(client, "GET", "/props")
        settings = props.get("default_generation_settings", {})
        native_template = Path(self.config.chat_template_path).read_text()
        reported_template = props.get("chat_template")
        template_matches = reported_template == native_template or (
            native_template.endswith("\n") and reported_template == native_template[:-1]
        )
        if (
            props.get("model_alias") != self.config.model_id
            or Path(props.get("model_path", "")).resolve() != Path(self.config.model_path).resolve()
            or props.get("build_info") != self.config.runtime_build
            or props.get("total_slots") != 1
            or settings.get("n_ctx") != 4096
            or not template_matches
        ):
            raise ValueError(
                "Local runtime identity, native template, context or slot count differs."
            )
        params = settings.get("params", {})
        if params.get("reasoning_format") != "none" or params.get("reasoning_in_content"):
            raise ValueError("Local runtime must start with reasoning disabled.")
        if self.config.token_parity_path is None:
            return {
                "status": "unqualified",
                "model": self.model,
                "message": "Native/server token parity is required before local predictions.",
            }
        self._verify_parity()
        return {"status": "ready", "model": self.model, "message": "Experimental local copilot."}

    def _verify_parity(self):
        path = Path(self.config.token_parity_path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.config.token_parity_sha256:
            raise ValueError("Native-token parity receipt hash differs.")
        if self._verified_parity == (digest, self._verified_files):
            return
        receipt = json.loads(raw)
        if any(
            receipt.get(key) != value
            for key, value in {
                "version": 1,
                "model_sha256": self.config.model_sha256,
                "chat_template_sha256": self.config.chat_template_sha256,
                "runtime_build": self.config.runtime_build,
                "token_source": "native_tokenizer",
            }.items()
        ):
            raise ValueError("Native-token parity receipt belongs to another deployment.")
        if receipt.get("contract_version", CONTRACT_VERSION) != self.contract_version:
            raise ValueError("Native-token parity receipt belongs to another contract.")
        module = contract(self.contract_version)
        checks = receipt.get("checks", [])
        if not isinstance(checks, list) or not 2 <= len(checks) <= 16:
            raise ValueError("Use two to sixteen fixed native-token parity checks.")
        languages = set()
        for check in checks:
            messages = check["messages"]
            value = module.parse_snapshot(json.loads(messages[1]["content"]))
            if module.canonical_messages(value) != messages:
                raise ValueError("Parity messages differ from the shared passive prompt.")
            expected = check["expected_native_token_ids"]
            if (
                not isinstance(expected, list)
                or not 1 <= len(expected) <= 3584
                or any(type(token) is not int or token < 0 for token in expected)
                or self.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                != expected
            ):
                raise ValueError("Runtime prompt tokens differ from the native-tokenizer receipt.")
            languages.add(value.language)
        if not {"en", "ko"} <= languages:
            raise ValueError("Native-token parity must cover English and Korean.")
        self._verified_parity = (digest, self._verified_files)
        self.model["training_token_parity"] = "verified_fixed_bilingual_checks"
        self.model["token_parity_sha256"] = digest

    def apply_chat_template(self, messages, **kwargs):
        if kwargs != {
            "tokenize": True,
            "return_dict": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }:
            raise ValueError("Unsupported passive tokenization options.")
        with self._client() as client:
            rendered = self._json(
                client,
                "POST",
                "/apply-template",
                json={
                    "model": self.config.model_id,
                    "messages": messages,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            tokens = self._json(
                client,
                "POST",
                "/tokenize",
                json={
                    "content": rendered["prompt"],
                    "add_special": True,
                    "parse_special": True,
                },
            )["tokens"]
        if not isinstance(tokens, list) or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("Local tokenizer returned invalid token IDs.")
        return tokens

    async def generate(self, messages, output_schema, timeout):
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
            "cache_prompt": self.config.cache_prompt,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "next_edit",
                    "strict": True,
                    "schema": schema_for_runtime(output_schema),
                },
            },
        }
        if self.config.capture_runtime_evidence:
            body.update(verbose=True, return_tokens=True)
        started = time.perf_counter()
        async with httpx.AsyncClient(
            base_url=self.config.endpoint.rstrip("/"),
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            response = await client.post("/v1/chat/completions", json=body)
            response.raise_for_status()
            result = response.json()
        choices = result.get("choices")
        if result.get("model") != self.config.model_id:
            raise ValueError("Local completion belongs to a different model.")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Local inference must return exactly one completion.")
        choice = choices[0]
        text = choice.get("message", {}).get("content")
        if not isinstance(text, str):
            raise ValueError("Local inference returned no text completion.")
        return Completion(
            text,
            choice.get("finish_reason"),
            inference_timings(
                result, (time.perf_counter() - started) * 1000, self.config.cache_prompt
            ),
            {"request": body, "response": result} if self.config.capture_runtime_evidence else None,
        )


def adapter_from_environment():
    path = os.environ.get("AUDAPOLIS_NEP_CONFIG")
    if not path:
        return DisabledAdapter()
    try:
        return LocalLlamaAdapter(LocalConfig.model_validate(json.loads(Path(path).read_text())))
    except (ValueError, OSError):
        return DisabledAdapter(
            "Local copilot configuration is invalid; check AUDAPOLIS_NEP_CONFIG."
        )


@dataclass
class ActiveRequest:
    identity: RequestIdentity
    cancelled: asyncio.Event


class RequestTokenizer:
    def __init__(self, adapter, cancelled):
        self.adapter, self.cancelled = adapter, cancelled

    def apply_chat_template(self, *args, **kwargs):
        if self.cancelled.is_set():
            raise ValueError("Prediction preparation was cancelled.")
        result = self.adapter.apply_chat_template(*args, **kwargs)
        if self.cancelled.is_set():
            raise ValueError("Prediction preparation was cancelled.")
        return result


class CopilotService:
    def __init__(self, adapter=None):
        self.adapter = adapter if adapter is not None else adapter_from_environment()
        self.sessions = OrderedDict()
        self.latest = {}
        self.used_requests = set()
        self.active = None

    def _contract(self):
        return contract(getattr(self.adapter, "contract_version", CONTRACT_VERSION))

    async def health(self):
        try:
            result = await asyncio.to_thread(self.adapter.health)
        except (ValueError, OSError, httpx.HTTPError, KeyError, TypeError, ImportError):
            result = {
                "status": "unavailable",
                "model": self.adapter.model,
                "message": "Local model is unavailable or does not match its pinned configuration.",
            }
        if self.active is not None and result["status"] == "ready":
            result["status"] = "busy"
        version = getattr(self.adapter, "contract_version", CONTRACT_VERSION)
        allowed, gated = CONTRACT_ACTIONS[version]
        return {
            **result,
            "contract_version": version,
            "allowed_actions": list(allowed),
            "gated_actions": list(gated),
            "input_tokens": 3584,
            "output_tokens": 512,
            "context_tokens": 4096,
            "experimental": True,
            "process_lifecycle": "externally_managed",
        }

    def cancel(self, identity):
        active = self.active
        cancelled = active is not None and active.identity == identity
        if cancelled:
            active.cancelled.set()
        return {"request": identity.model_dump(), "cancelled": cancelled}

    def _session(self, document_id):
        if document_id not in self.sessions:
            if len(self.sessions) >= 8:
                removed, _ = self.sessions.popitem(last=False)
                self.latest.pop(removed, None)
                self.used_requests = {key for key in self.used_requests if key[0] != removed}
            module = self._contract()
            self.sessions[document_id] = module.PassivePredictionSession(
                document_id, module.PassiveContextBuilder(self.adapter)
            )
        self.sessions.move_to_end(document_id)
        return self.sessions[document_id]

    async def predict(self, body, disconnected=None):
        identity = body.request
        key = (identity.document_id, identity.request_id)
        if self.active is not None:
            raise GatewayError(409, "busy", "A local prediction is already active.")
        if key in self.used_requests:
            raise GatewayError(409, "duplicate", "This prediction request was already used.")
        previous = self.latest.get(identity.document_id)
        if previous is not None and (
            identity.revision < previous.revision
            or identity.interaction_epoch < previous.interaction_epoch
        ):
            raise GatewayError(409, "stale", "Document or interaction identity moved backwards.")
        active = ActiveRequest(identity, asyncio.Event())
        self.active = active
        self.used_requests.add(key)
        self.latest[identity.document_id] = identity
        started = time.perf_counter()
        worker = asyncio.create_task(self._predict(body, active, started))
        monitor = asyncio.create_task(self._wait_cancel(active, disconnected))
        try:
            done, _ = await asyncio.wait(
                {worker, monitor},
                timeout=body.deadline_ms / 1000,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                active.cancelled.set()
                raise GatewayError(504, "deadline", "Local prediction exceeded its deadline.")
            if active.cancelled.is_set() or monitor in done:
                raise GatewayError(409, "cancelled", "Local prediction was cancelled.")
            return await worker
        except asyncio.CancelledError:
            active.cancelled.set()
            raise
        finally:
            # Cancelling the HTTP task closes the upstream response connection. Preparation
            # checks the event between tokenizer calls and never starts late generation.
            active.cancelled.set()
            worker.cancel()
            monitor.cancel()
            await asyncio.gather(worker, monitor, return_exceptions=True)
            if self.active is active:
                self.active = None

    async def _wait_cancel(self, active, disconnected):
        while not active.cancelled.is_set():
            if disconnected is not None and await disconnected():
                active.cancelled.set()
                return
            try:
                await asyncio.wait_for(active.cancelled.wait(), timeout=0.05)
            except TimeoutError:
                pass

    async def _predict(self, body, active, started):
        completion = None
        prepared = generated = None

        def diagnostic(error):
            if completion is not None and completion.runtime_evidence is not None:
                error.runtime_evidence = completion.runtime_evidence
                error.timings = {
                    **(completion.timings or {}),
                    "preparation_ms": (prepared - started) * 1000,
                    "generation_ms": (generated - prepared) * 1000,
                    "validation_ms": (time.perf_counter() - generated) * 1000,
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "trigger": body.trigger,
                }
            return error

        try:
            ready = await asyncio.to_thread(self.adapter.health)
            if ready["status"] != "ready":
                raise GatewayError(
                    503, "unconfigured", ready.get("message", "Local model unavailable.")
                )
            if active.cancelled.is_set():
                raise GatewayError(409, "cancelled", "Local prediction was cancelled.")
            module = self._contract()
            session = self._session(body.request.document_id)
            events = session.feedback._ledger.events
            if events and body.sequence < events[-1]["sequence"]:
                raise GatewayError(409, "stale", "Prediction sequence predates recorded feedback.")
            if (
                len(session.feedback._records) >= 512
                or sum(
                    document_id == body.request.document_id for document_id, _ in self.used_requests
                )
                >= 512
            ):
                raise GatewayError(409, "session_full", "Start a fresh copilot session.")
            document = body.document.model_dump()
            session.builder = module.PassiveContextBuilder(
                RequestTokenizer(self.adapter, active.cancelled)
            )
            bundle = await asyncio.to_thread(
                session.prepare,
                document,
                body.focus_uuid,
                as_of=body.sequence,
                selection=body.selection,
                history=body.history,
            )
            prepared = time.perf_counter()
            remaining = body.deadline_ms / 1000 - (prepared - started)
            if active.cancelled.is_set() or remaining <= 0:
                raise GatewayError(409, "cancelled", "Local prediction was cancelled.")
            completion = await self.adapter.generate(
                module.canonical_messages(bundle["input"]),
                module.OUTPUT_SCHEMA,
                remaining,
            )
            generated = time.perf_counter()
            if active.cancelled.is_set():
                raise GatewayError(409, "cancelled", "Local prediction was cancelled.")
            request = module.Envelope(**body.request.model_dump())
            current = module.Envelope(**self.latest[body.request.document_id].model_dump())
            validated = session.validate(
                completion.text,
                document,
                bundle,
                request,
                current,
                completion.finish_reason,
                body.sequence,
            )
            prediction = validated["prediction"]
            proposal_id, result, result_hash, targets = None, None, None, []
            status = "suppressed" if validated["suppressed"] else "no_action"
            if validated["show"]:
                result = module.apply_passive_edit(document, bundle, prediction)
                result_hash = module.session_hash(result)
                targets = [bundle["mapping"][str(alias)]["uuid"] for alias in prediction["target"]]
                proposal_id = str(uuid4())
                if (time.perf_counter() - started) * 1000 >= body.deadline_ms:
                    raise GatewayError(504, "deadline", "Local prediction exceeded its deadline.")
                session.feedback.propose(proposal_id, body.sequence, document, bundle, prediction)
                if (time.perf_counter() - started) * 1000 >= body.deadline_ms:
                    # The shared ledger validates synchronously; an over-deadline draft
                    # must not become a proposal that could later receive feedback.
                    del session.feedback._ledger.proposals[proposal_id]
                    session.feedback._records.pop()
                    raise GatewayError(504, "deadline", "Local prediction exceeded its deadline.")
                status = "proposal"
            elif (time.perf_counter() - started) * 1000 >= body.deadline_ms:
                raise GatewayError(504, "deadline", "Local prediction exceeded its deadline.")
            return {
                **(
                    {"runtime_evidence": completion.runtime_evidence}
                    if completion.runtime_evidence
                    else {}
                ),
                "request": body.request.model_dump(),
                "status": status,
                "proposal_id": proposal_id,
                "prediction": prediction,
                "target_uuids": targets,
                "document_hash": bundle["document_hash"],
                "result_document": result,
                "result_hash": result_hash,
                "context": bundle["context"],
                "model": self.adapter.model,
                "timings": {
                    **(completion.timings or {}),
                    "preparation_ms": (prepared - started) * 1000,
                    "generation_ms": (generated - prepared) * 1000,
                    "validation_ms": (time.perf_counter() - generated) * 1000,
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "trigger": body.trigger,
                },
            }
        except GatewayError as error:
            raise diagnostic(error)
        except httpx.TimeoutException as error:
            raise GatewayError(504, "deadline", "Local model request timed out.") from error
        except (httpx.HTTPError, OSError) as error:
            raise GatewayError(503, "unavailable", "Local model request failed.") from error
        except ImportError as error:
            raise GatewayError(
                503, "unavailable", "Shared NEP contract requires the source checkout."
            ) from error
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise diagnostic(GatewayError(422, "invalid", str(error))) from error

    def feedback(self, body):
        session = self.sessions.get(body.document_id)
        if session is None:
            raise GatewayError(409, "unknown_session", "This copilot session has expired.")
        if self.active is not None and self.active.identity.document_id == body.document_id:
            raise GatewayError(
                409, "busy", "Wait for the active prediction before recording feedback."
            )
        try:
            session.feedback.record(
                body.proposal_id,
                body.outcome,
                body.sequence,
                origin=body.origin,
                reason=body.reason,
                transition=body.transition,
            )
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise GatewayError(422, "invalid_feedback", str(error)) from error
        return {"recorded": True, "proposal_id": body.proposal_id, "outcome": body.outcome}


def create_copilot_router(auth_dependency):
    router = APIRouter(prefix="/copilot", dependencies=[Depends(auth_dependency)])

    def translate(error):
        return HTTPException(
            error.status_code,
            detail={
                "code": error.code,
                "message": error.message,
                **(
                    {"runtime_evidence": error.runtime_evidence, "timings": error.timings}
                    if getattr(error, "runtime_evidence", None) is not None
                    else {}
                ),
            },
        )

    @router.get("/health")
    async def health(request: Request):
        return await request.app.state.copilot.health()

    @router.post("/predict")
    async def predict(body: PredictRequest, request: Request):
        try:
            return await request.app.state.copilot.predict(body, request.is_disconnected)
        except GatewayError as error:
            raise translate(error) from error

    @router.post("/feedback")
    async def feedback(body: FeedbackRequest, request: Request):
        try:
            return request.app.state.copilot.feedback(body)
        except GatewayError as error:
            raise translate(error) from error

    @router.post("/cancel")
    async def cancel(body: RequestIdentity, request: Request):
        return request.app.state.copilot.cancel(body)

    return router
