"""Authenticated, opt-in local gateway for the shared passive NEP contract."""

import asyncio
import copy
import hashlib
import importlib
import ipaddress
import json
import math
import os
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .copilot_store import CopilotEvidenceStore, EvidenceError, evidence_hash

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
_tokenization_client = ContextVar("copilot_tokenization_client", default=None)


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
    history: list[dict] = Field(default_factory=list, max_length=512)
    history_transport: Literal["incremental"] | None = None
    request_transport: Literal["sequenced-v1"] | None = None
    history_base: str | None = Field(default=None, min_length=1, max_length=256)
    trigger: Literal["automatic", "explicit"]
    deadline_ms: int = Field(default=30000, ge=100, le=120000)
    origin: Literal["human", "ai_computer_use", "synthetic"] | None = None
    observed_at: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def history_transport_limits(self):
        if self.history_transport is None and (
            self.history_base is not None or len(self.history) > 128
        ):
            raise ValueError("Legacy requests require full history with at most 128 transitions.")
        if self.request_transport is not None:
            identity, separator, sequence = self.request.request_id.rpartition(":")
            if (
                self.history_transport != "incremental"
                or not separator
                or sequence != str(self.sequence)
                or str(UUID(identity, version=4)) != identity
            ):
                raise ValueError(
                    "Sequenced requests require incremental history and a UUIDv4:sequence identity."
                )
        return self


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
    event_id: str | None = Field(default=None, min_length=1, max_length=256)
    request: RequestIdentity | None = None
    revision: int | None = Field(default=None, ge=0)
    interaction_epoch: int | None = Field(default=None, ge=0)
    observed_at: str | None = Field(default=None, max_length=64)


class LocalConfig(StrictModel):
    version: Literal[1, 2] = 1
    contract_version: Literal["passive-nep-v1", "passive-nep-actions-v1"] = CONTRACT_VERSION
    endpoint: str
    model_id: str = Field(min_length=1)
    model_path: str
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_path: str | None = None
    adapter_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    adapter_scale: Literal[1.0] = 1.0
    output_format: Literal["json", "rewrite"] = "json"
    chat_template_path: str
    chat_template_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    runtime_build: str = Field(min_length=1)
    token_parity_path: str | None = None
    token_parity_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    cache_prompt: bool = False
    capture_runtime_evidence: bool = False

    @model_validator(mode="after")
    def local_only(self):
        if (self.adapter_path is None) != (self.adapter_sha256 is None):
            raise ValueError("A local LoRA requires both its path and hash.")
        if self.version == 1 and (self.adapter_path is not None or self.output_format != "json"):
            raise ValueError("LoRA and Rewrite require an explicitly versioned v2 deployment.")
        if self.version == 2 and self.contract_version != "passive-nep-actions-v1":
            raise ValueError("The v2 deployment uses the shared seven-action contract.")
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
        self.output_format = config.output_format
        self.model = {
            "id": config.model_id,
            "sha256": config.model_sha256,
            "chat_template_sha256": config.chat_template_sha256,
            "runtime_build": config.runtime_build,
            "training_token_parity": "unverified",
            "contract_version": self.contract_version,
            "cache_prompt": config.cache_prompt,
        }
        if config.version == 2:
            self.model.update(
                adapter_sha256=config.adapter_sha256,
                adapter_scale=config.adapter_scale if config.adapter_path else None,
                output_format=config.output_format,
                model_layout="separate_gguf_base_and_lora" if config.adapter_path else "gguf_base",
            )
        self._verified_files = None
        self._verified_parity = None
        self._native_count_supported = None

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
        files = [(model, self.config.model_sha256), (template, self.config.chat_template_sha256)]
        if self.config.adapter_path:
            files.append((Path(self.config.adapter_path), self.config.adapter_sha256))
        signature = tuple(
            (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p, _ in files
        )
        if signature != self._verified_files:
            for path, expected in files:
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError(
                            "Configured local model, adapter or native template hash differs."
                        )
            self._verified_files = signature

    def _verify_loaded_adapter(self, value):
        if self.config.version != 2:
            return
        if self.config.adapter_path is None:
            if value != []:
                raise ValueError("The configured native base has an unexpected loaded LoRA.")
            return
        if (
            not isinstance(value, list)
            or len(value) != 1
            or value[0].get("id") != 0
            or Path(value[0].get("path", "")).resolve() != Path(self.config.adapter_path).resolve()
            or value[0].get("scale") != self.config.adapter_scale
            or value[0].get("alora_invocation_tokens")
        ):
            raise ValueError(
                "Runtime did not load exactly the hash-bound ordinary LoRA at scale 1."
            )

    def request_messages(self, snapshot):
        if self.config.version == 2:
            contract(self.contract_version)
            return importlib.import_module("passive_formats_actions_v3").messages(
                snapshot, self.output_format
            )
        return contract(self.contract_version).canonical_messages(snapshot)

    def _model_messages(self, messages):
        if self.config.version != 2:
            return messages
        module = contract(self.contract_version)
        snapshot = module.parse_snapshot(json.loads(messages[1]["content"]))
        expected = self.request_messages(snapshot)
        if messages not in (expected, module.canonical_messages(snapshot)):
            raise ValueError("Runtime messages differ from the configured visible input format.")
        return expected

    def health(self):
        self._verify_files()
        with self._client() as client:
            self._json(client, "GET", "/health")
            props = self._json(client, "GET", "/props")
            if self.config.version == 2:
                self._verify_loaded_adapter(self._json(client, "GET", "/lora-adapters"))
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
                "version": self.config.version,
                "model_sha256": self.config.model_sha256,
                "chat_template_sha256": self.config.chat_template_sha256,
                "runtime_build": self.config.runtime_build,
                "token_source": "native_tokenizer",
            }.items()
        ):
            raise ValueError("Native-token parity receipt belongs to another deployment.")
        if receipt.get("contract_version", CONTRACT_VERSION) != self.contract_version:
            raise ValueError("Native-token parity receipt belongs to another contract.")
        if self.config.version == 2 and any(
            receipt.get(key) != value
            for key, value in {
                "adapter_sha256": self.config.adapter_sha256,
                "adapter_scale": self.config.adapter_scale if self.config.adapter_path else None,
                "output_format": self.output_format,
            }.items()
        ):
            raise ValueError("Native-token parity receipt belongs to another adapter or format.")
        module = contract(self.contract_version)
        checks = receipt.get("checks", [])
        if not isinstance(checks, list) or not 2 <= len(checks) <= 16:
            raise ValueError("Use two to sixteen fixed native-token parity checks.")
        languages = set()
        for check in checks:
            messages = check["messages"]
            value = module.parse_snapshot(json.loads(messages[1]["content"]))
            if self.request_messages(value) != messages:
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

    @contextmanager
    def tokenization_session(self):
        current = _tokenization_client.get()
        if current is not None and current[0] is self:
            yield
            return
        state = [self, None]
        token = _tokenization_client.set(state)
        try:
            yield
        finally:
            if state[1] is not None:
                state[1].close()
            _tokenization_client.reset(token)

    def _tokenizer_client(self):
        current = _tokenization_client.get()
        if current is None or current[0] is not self:
            return self._client()
        if current[1] is None:
            current[1] = self._client()
        return nullcontext(current[1])

    def _tokenizer_messages(self, messages, kwargs):
        if kwargs != {
            "tokenize": True,
            "return_dict": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }:
            raise ValueError("Unsupported passive tokenization options.")
        return self._model_messages(messages)

    def apply_chat_template(self, messages, **kwargs):
        messages = self._tokenizer_messages(messages, kwargs)
        with self._tokenizer_client() as client:
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

    def count_chat_template_tokens(self, messages, **kwargs):
        model_messages = self._tokenizer_messages(messages, kwargs)
        if self._native_count_supported is False:
            return len(self.apply_chat_template(messages, **kwargs))
        with self._tokenizer_client() as client:
            try:
                result = self._json(
                    client,
                    "POST",
                    "/v1/chat/completions/input_tokens",
                    json={
                        "model": self.config.model_id,
                        "messages": model_messages,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in (404, 405):
                    raise
                self._native_count_supported = False
                return len(self.apply_chat_template(messages, **kwargs))
        count = result.get("input_tokens") if isinstance(result, dict) else None
        if type(count) is not int or count < 0:
            raise ValueError("Local tokenizer returned an invalid native token count.")
        self._native_count_supported = True
        return count

    async def generate(self, messages, output_schema, timeout):
        messages = self._model_messages(messages)
        if self.output_format == "rewrite":
            output_schema = importlib.import_module(
                "passive_formats_actions_v3"
            ).Rewrite.model_json_schema()
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
        if self.config.version == 2:
            body["lora"] = (
                [{"id": 0, "scale": self.config.adapter_scale}] if self.config.adapter_path else []
            )
        if self.config.capture_runtime_evidence:
            body.update(verbose=True, return_tokens=True)
        started = time.perf_counter()
        async with httpx.AsyncClient(
            base_url=self.config.endpoint.rstrip("/"),
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            if self.config.version == 2:
                loaded = await client.get("/lora-adapters")
                loaded.raise_for_status()
                self._verify_loaded_adapter(loaded.json())
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
        from .copilot_runtime import default_adapter

        return default_adapter()
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
    def __init__(self, adapter, cancelled, cache=None):
        self.adapter, self.cancelled = adapter, cancelled
        self.cache = cache

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise ValueError("Prediction preparation was cancelled.")

    def apply_chat_template(self, *args, **kwargs):
        self.check_cancelled()
        if self.cache is None:
            result = self.adapter.apply_chat_template(*args, **kwargs)
        else:
            values, lock = self.cache
            key = evidence_hash([self.adapter.model, args, kwargs])
            with lock:
                result = values.get(key)
                if result is not None:
                    values.move_to_end(key)
            if result is None:
                result = self.adapter.apply_chat_template(*args, **kwargs)
                self.check_cancelled()
                with lock:
                    values[key] = tuple(result)
                    values.move_to_end(key)
                    while len(values) > 128:
                        values.popitem(last=False)
            result = list(result)
        self.check_cancelled()
        return result

    def count_chat_template_tokens(self, *args, **kwargs):
        count = getattr(self.adapter, "count_chat_template_tokens", None)
        if count is None:
            return len(self.apply_chat_template(*args, **kwargs))
        self.check_cancelled()
        if self.cache is None:
            result = count(*args, **kwargs)
        else:
            values, lock = self.cache
            key = evidence_hash(["count", self.adapter.model, args, kwargs])
            with lock:
                result = values.get(key)
                if result is not None:
                    values.move_to_end(key)
            if result is None:
                result = count(*args, **kwargs)
                self.check_cancelled()
                if type(result) is not int or result < 0:
                    raise ValueError("Native tokenizer returned an invalid token count.")
                with lock:
                    values[key] = result
                    values.move_to_end(key)
                    while len(values) > 128:
                        values.popitem(last=False)
        self.check_cancelled()
        return result


class CopilotService:
    def __init__(self, adapter=None, feedback_directory=None):
        self.adapter = adapter if adapter is not None else adapter_from_environment()
        self.sessions = OrderedDict()
        self.latest = {}
        self.used_requests = set()
        self.request_sequences = {}
        self.active = None
        self.evidence = CopilotEvidenceStore(feedback_directory)
        self.journals = {}
        self.process_id = str(uuid4())
        self.storage_errors = {}
        self.tokenizer_cache = (OrderedDict(), RLock())

    @staticmethod
    def _journal_entry(event):
        value = {key: event[key] for key in ("kind", "event_hash", "seq", "document_id")}
        if event["kind"] == "prediction_completed":
            value.update(
                request=copy.deepcopy(event["request"]),
                prepared_event_hash=event["prepared_event_hash"],
                response={"proposal_id": event["response"]["proposal_id"]},
            )
        elif event["kind"] == "feedback":
            value.update(
                feedback={
                    key: event["feedback"].get(key)
                    for key in ("event_id", "proposal_id", "outcome")
                },
                feedback_hash=event.get("feedback_hash") or evidence_hash(event["feedback"]),
            )
        return value

    def _append(self, document_id, **fields):
        if document_id in self.storage_errors:
            raise GatewayError(503, "feedback_storage", self.storage_errors[document_id])
        records = self.journals[document_id]
        if document_id in self.request_sequences or fields.get("request_transport"):
            fields = {**fields, "capture_mode": "product-compact-v1"}
            if "response" in fields:
                fields["response"] = {
                    key: value
                    for key, value in fields["response"].items()
                    if key != "result_document"
                }
            if "feedback" in fields:
                fields["feedback_hash"] = evidence_hash(fields["feedback"])
                fields["feedback"] = {
                    **fields["feedback"],
                    "transition": fields["ledger_record"]["transition"],
                }
        try:
            record = self.evidence.append(
                document_id,
                copy.deepcopy({"process_id": self.process_id, **fields}),
                records[-1]["event_hash"] if records else None,
            )
        except EvidenceError as error:
            self.storage_errors[document_id] = str(error)
            raise GatewayError(503, "feedback_storage", str(error)) from error
        records.append(self._journal_entry(record))
        return record

    def _contract(self):
        return contract(getattr(self.adapter, "contract_version", CONTRACT_VERSION))

    def _model_identity(self):
        model = copy.deepcopy(self.adapter.model)
        if isinstance(model, dict):
            model.pop("training_token_parity", None)
            config = getattr(self.adapter, "config", None)
            if config is not None:
                model["token_parity_sha256"] = config.token_parity_sha256
        return model

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
            "process_lifecycle": getattr(self.adapter, "process_lifecycle", "externally_managed"),
            "feedback_storage": {
                "format": "copilot-feedback-v1",
                "directory": str(self.evidence.directory),
                "local_only": True,
                "sessions_endpoint": "/copilot/feedback/sessions",
                "export_endpoint": "/copilot/feedback/export",
            },
        }

    def cancel(self, identity):
        active = self.active
        cancelled = active is not None and active.identity == identity
        if cancelled:
            active.cancelled.set()
        return {"request": identity.model_dump(), "cancelled": cancelled}

    def _session(self, document_id):
        if document_id in self.storage_errors:
            raise GatewayError(503, "feedback_storage", self.storage_errors[document_id])
        if document_id not in self.sessions:
            if len(self.sessions) >= 8:
                removed = next(
                    key
                    for key in self.sessions
                    if self.active is None or self.active.identity.document_id != key
                )
                self.sessions.pop(removed)
                self.latest.pop(removed, None)
                self.request_sequences.pop(removed, None)
                self.journals.pop(removed, None)
                self.used_requests = {key for key in self.used_requests if key[0] != removed}
            module = self._contract()
            session = module.PassivePredictionSession(
                document_id, module.PassiveContextBuilder(self.adapter)
            )
            try:
                records, ledger_records, history_records = [], [], []
                failed = set()
                high_water = None

                def restore_event(event):
                    nonlocal high_water
                    records.append(self._journal_entry(event))
                    if event["kind"] == "prediction_requested":
                        if (
                            event["contract_version"] != module.VERSION
                            or event["model_identity"] != self._model_identity()
                        ):
                            raise GatewayError(
                                409,
                                "session_identity",
                                "Model or contract changed; start a new copilot session.",
                            )
                        identity = RequestIdentity.model_validate(event["request"])
                        self.latest[document_id] = identity
                        if event.get("request_transport") == "sequenced-v1":
                            if high_water is not None and event["sequence"] <= high_water:
                                raise ValueError("Durable request sequence moved backwards.")
                            high_water = event["sequence"]
                        elif high_water is not None:
                            raise ValueError("A sequenced session changed transport.")
                        else:
                            self.used_requests.add((document_id, identity.request_id))
                    if event["kind"] == "prediction_failed":
                        failed.add(event["request"]["request_id"])
                    completed = (
                        event["request"]["request_id"]
                        if event["kind"] == "prediction_completed"
                        else None
                    )
                    if event.get("ledger_record") is not None:
                        ledger_records.append((completed, event["ledger_record"]))
                    if event.get("history_record") is not None:
                        history_records.append((completed, event["history_record"]))

                self.evidence.scan(document_id, restore_event)
                if high_water is not None:
                    session.enable_compact_runtime()
                    for request_id, value in ledger_records:
                        if request_id not in failed:
                            session.feedback.restore_record(value)
                    for request_id, value in history_records:
                        if request_id not in failed:
                            session._history_state = session._history_state.restore_receipt(value)
                            session._history_persisted = True
                    self.request_sequences[document_id] = high_water
                else:
                    session.feedback = module.PassiveFeedbackLedger.restore(
                        {
                            "contract_version": module.VERSION,
                            "document_id": document_id,
                            "records": [
                                value
                                for request_id, value in ledger_records
                                if request_id not in failed
                            ],
                        }
                    )
            except (EvidenceError, ValueError, KeyError, TypeError) as error:
                raise GatewayError(503, "feedback_storage", str(error)) from error
            self.sessions[document_id] = session
            self.journals[document_id] = records
        self.sessions.move_to_end(document_id)
        return self.sessions[document_id]

    async def predict(self, body, disconnected=None):
        started = time.perf_counter()
        identity = body.request
        key = (identity.document_id, identity.request_id)
        if self.active is not None:
            raise GatewayError(409, "busy", "A local prediction is already active.")
        try:
            self._session(identity.document_id)
        except ImportError as error:
            raise GatewayError(
                503, "unavailable", "Shared NEP contract requires the source checkout."
            ) from error
        sequenced = body.request_transport == "sequenced-v1"
        if identity.document_id in self.request_sequences and not sequenced:
            raise GatewayError(
                409, "request_transport", "This session requires sequenced requests."
            )
        if sequenced and not hasattr(self.sessions[identity.document_id], "enable_compact_runtime"):
            raise GatewayError(422, "request_transport", "This contract has no sequenced runtime.")
        if sequenced and body.sequence <= self.request_sequences.get(identity.document_id, -1):
            raise GatewayError(409, "request_replay", "This request sequence was already consumed.")
        if key in self.used_requests:
            raise GatewayError(409, "duplicate", "This prediction request was already used.")
        previous = self.latest.get(identity.document_id)
        if previous is not None and (
            identity.revision < previous.revision
            or identity.interaction_epoch < previous.interaction_epoch
        ):
            raise GatewayError(409, "stale", "Document or interaction identity moved backwards.")
        self._append(
            identity.document_id,
            kind="prediction_requested",
            request=identity.model_dump(),
            sequence=body.sequence,
            actor=body.origin or "unclassified",
            contract_version=getattr(self.adapter, "contract_version", CONTRACT_VERSION),
            model=self.adapter.model,
            model_identity=self._model_identity(),
            **({"request_transport": body.request_transport} if sequenced else {}),
            input_request=body.model_dump(exclude={"document", "history"})
            if sequenced
            else body.model_dump(),
        )
        active = ActiveRequest(identity, asyncio.Event())
        self.active = active
        if sequenced:
            self.request_sequences[identity.document_id] = body.sequence
            self.sessions[identity.document_id].enable_compact_runtime()
        else:
            self.used_requests.add(key)
        self.latest[identity.document_id] = identity
        session = self.sessions[identity.document_id]
        previous_feedback = session.feedback
        previous_history = getattr(session, "_history_state", None)
        previous_persisted = getattr(session, "_history_persisted", False)

        def rollback():
            session.feedback = previous_feedback
            if previous_history is not None:
                session._history_state = previous_history
                session._history_persisted = previous_persisted

        worker = asyncio.create_task(self._predict(body, active, started))
        monitor = asyncio.create_task(self._wait_cancel(active, disconnected))
        try:
            done, _ = await asyncio.wait(
                {worker, monitor},
                timeout=max(0, body.deadline_ms / 1000 - (time.perf_counter() - started)),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                active.cancelled.set()
                raise GatewayError(504, "deadline", "Local prediction exceeded its deadline.")
            if active.cancelled.is_set() or monitor in done:
                raise GatewayError(409, "cancelled", "Local prediction was cancelled.")
            return await worker
        except GatewayError as error:
            rollback()
            if error.code != "feedback_storage":
                self._append(
                    identity.document_id,
                    kind="prediction_failed",
                    request=identity.model_dump(),
                    sequence=body.sequence,
                    actor=body.origin or "unclassified",
                    error={"code": error.code, "message": error.message},
                )
            raise
        except asyncio.CancelledError:
            active.cancelled.set()
            rollback()
            self._append(
                identity.document_id,
                kind="prediction_failed",
                request=identity.model_dump(),
                sequence=body.sequence,
                actor=body.origin or "unclassified",
                error={"code": "cancelled", "message": "Prediction request task was cancelled."},
            )
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
            if body.request_transport is None and (
                len(session.feedback._records) >= 512
                or sum(
                    document_id == body.request.document_id for document_id, _ in self.used_requests
                )
                >= 512
            ):
                raise GatewayError(409, "session_full", "Start a fresh copilot session.")
            document = body.document.model_dump()
            tokenizer = RequestTokenizer(self.adapter, active.cancelled, self.tokenizer_cache)
            session.builder = module.PassiveContextBuilder(tokenizer)

            def prepare():
                scope = (
                    session.validation_scope(tokenizer.check_cancelled)
                    if hasattr(session, "validation_scope")
                    else nullcontext()
                )
                native = (
                    self.adapter.tokenization_session()
                    if hasattr(self.adapter, "tokenization_session")
                    else nullcontext()
                )
                with scope, native:
                    prepare_context = (
                        session.prepare_incremental
                        if body.history_transport == "incremental"
                        and hasattr(session, "prepare_incremental")
                        else session.prepare
                    )
                    if body.history_transport is not None and not hasattr(
                        session, "prepare_incremental"
                    ):
                        raise GatewayError(
                            422, "invalid", "This contract does not support incremental history."
                        )
                    try:
                        prepared_context = prepare_context(
                            document,
                            body.focus_uuid,
                            as_of=body.sequence,
                            selection=body.selection,
                            history=body.history,
                            **(
                                {"history_base": body.history_base}
                                if body.history_transport
                                else {}
                            ),
                        )
                    except ValueError as error:
                        if isinstance(error, getattr(module, "HistoryResyncRequired", ())):
                            raise GatewayError(
                                409, "history_resync_required", str(error)
                            ) from error
                        if isinstance(error, getattr(module, "HistoryLimitError", ())):
                            raise GatewayError(409, "session_full", str(error)) from error
                        raise
                    return prepared_context if body.history_transport else (prepared_context, None)

            bundle, history_candidate = await asyncio.to_thread(prepare)
            messages = (
                self.adapter.request_messages(bundle["input"])
                if hasattr(self.adapter, "request_messages")
                else module.canonical_messages(bundle["input"])
            )
            output_schema = module.OUTPUT_SCHEMA
            if getattr(self.adapter, "output_format", "json") == "rewrite":
                output_schema = importlib.import_module(
                    "passive_formats_actions_v3"
                ).Rewrite.model_json_schema()
            prepared_record = self._append(
                body.request.document_id,
                kind="prediction_prepared",
                request=body.request.model_dump(),
                sequence=body.sequence,
                actor=body.origin or "unclassified",
                model=self.adapter.model,
                contract_version=module.VERSION,
                bundle=bundle,
                messages=messages,
                messages_hash=evidence_hash(messages),
                output_schema=output_schema,
                hash_format="sha256-canonical-json-utf8-v1",
            )
            prepared = time.perf_counter()
            remaining = body.deadline_ms / 1000 - (prepared - started)
            if active.cancelled.is_set() or remaining <= 0:
                raise GatewayError(409, "cancelled", "Local prediction was cancelled.")
            completion = await self.adapter.generate(
                messages,
                output_schema,
                remaining,
            )
            generated = time.perf_counter()
            if active.cancelled.is_set():
                raise GatewayError(409, "cancelled", "Local prediction was cancelled.")
            self._append(
                body.request.document_id,
                kind="prediction_generated",
                request=body.request.model_dump(),
                sequence=body.sequence,
                prepared_event_hash=prepared_record["event_hash"],
                completion={
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "timings": completion.timings,
                    "runtime_evidence": completion.runtime_evidence,
                },
            )

            def checkpoint():
                if active.cancelled.is_set():
                    raise GatewayError(409, "cancelled", "Local prediction was cancelled.")
                if (time.perf_counter() - started) * 1000 >= body.deadline_ms:
                    raise GatewayError(504, "deadline", "Local prediction exceeded its deadline.")

            def finish():
                scope = (
                    session.validation_scope(checkpoint)
                    if hasattr(session, "validation_scope")
                    else nullcontext()
                )
                with scope:
                    checkpoint()
                    request = module.Envelope(**body.request.model_dump())
                    current = module.Envelope(**self.latest[body.request.document_id].model_dump())
                    prediction_text = completion.text
                    if getattr(self.adapter, "output_format", "json") == "rewrite":
                        formats = importlib.import_module("passive_formats_actions_v3")
                        prediction_text = formats.compact(
                            formats.parse(
                                completion.text,
                                bundle["input"],
                                "rewrite",
                                envelope=request,
                                current=current,
                                finish_reason=completion.finish_reason,
                            ).model_dump()
                        )
                    validated = session.validate(
                        prediction_text,
                        document,
                        bundle,
                        request,
                        current,
                        completion.finish_reason,
                        body.sequence,
                    )
                    checkpoint()
                    prediction = validated["prediction"]
                    proposal_id, result, result_hash, targets = None, None, None, []
                    feedback = session.feedback
                    ledger_record = None
                    status = "suppressed" if validated["suppressed"] else "no_action"
                    if validated["show"]:
                        feedback = (
                            session.feedback.fork()
                            if hasattr(session.feedback, "fork")
                            else copy.deepcopy(session.feedback)
                        )
                        result = module.apply_passive_edit(document, bundle, prediction)
                        result_hash = module.session_hash(result)
                        targets = [
                            bundle["mapping"][str(alias)]["uuid"] for alias in prediction["target"]
                        ]
                        proposal_id = str(uuid4())
                        checkpoint()
                        feedback.propose(proposal_id, body.sequence, document, bundle, prediction)
                        checkpoint()
                        ledger_record = feedback._records[-1]
                        status = "proposal"
                    return (
                        prediction,
                        proposal_id,
                        result,
                        result_hash,
                        targets,
                        feedback,
                        ledger_record,
                        status,
                    )

            (
                prediction,
                proposal_id,
                result,
                result_hash,
                targets,
                feedback,
                ledger_record,
                status,
            ) = await asyncio.to_thread(finish)
            checkpoint()
            response = {
                **({"history_tip": history_candidate.tip} if history_candidate is not None else {}),
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
            saving = time.perf_counter()
            saved = self._append(
                body.request.document_id,
                kind="prediction_completed",
                request=body.request.model_dump(),
                sequence=body.sequence,
                actor=body.origin or "unclassified",
                prepared_event_hash=prepared_record["event_hash"],
                response=response,
                ledger_record=ledger_record,
                **(
                    {
                        "history_record": history_candidate.receipt(
                            full=not session._history_persisted
                        )
                    }
                    if body.request_transport == "sequenced-v1"
                    else {}
                ),
            )
            if (time.perf_counter() - started) * 1000 >= body.deadline_ms:
                raise GatewayError(504, "deadline", "Local prediction exceeded its deadline.")
            checkpoint()
            if history_candidate is not None:
                session.commit_history(history_candidate)
                if body.request_transport == "sequenced-v1":
                    session._history_persisted = True
            session.feedback = feedback
            return {
                **response,
                "timings": {
                    **response["timings"],
                    "evidence_write_ms": (time.perf_counter() - saving) * 1000,
                    "total_ms": (time.perf_counter() - started) * 1000,
                },
                "evidence": {"event_hash": saved["event_hash"], "seq": saved["seq"]},
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
        if self.active is not None and self.active.identity.document_id == body.document_id:
            raise GatewayError(
                409, "busy", "Wait for the active prediction before recording feedback."
            )
        session = self._session(body.document_id)
        records = self.journals[body.document_id]
        for event in records:
            if (
                body.event_id
                and event["kind"] == "feedback"
                and event["feedback"].get("event_id") == body.event_id
            ):
                if event["feedback_hash"] != evidence_hash(body.model_dump()):
                    raise GatewayError(
                        409,
                        "feedback_conflict",
                        "Feedback event identity was reused with different content.",
                    )
                try:
                    self.evidence.confirm(body.document_id, records[-1]["event_hash"])
                except EvidenceError as error:
                    self.storage_errors[body.document_id] = str(error)
                    raise GatewayError(503, "feedback_storage", str(error)) from error
                return self._feedback_receipt(event)
        proposal = next(
            (
                event
                for event in records
                if event["kind"] == "prediction_completed"
                and event["response"]["proposal_id"] == body.proposal_id
            ),
            None,
        )
        if proposal is None:
            raise GatewayError(
                409, "unknown_proposal", "This proposal has no saved prediction evidence."
            )
        if body.request is not None and body.request.model_dump() != proposal["request"]:
            raise GatewayError(
                422, "invalid_feedback", "Feedback refers to a different prediction request."
            )
        if (body.revision is not None and body.revision < proposal["request"]["revision"]) or (
            body.interaction_epoch is not None
            and body.interaction_epoch < proposal["request"]["interaction_epoch"]
        ):
            raise GatewayError(422, "invalid_feedback", "Feedback state predates its prediction.")
        try:
            candidate = (
                session.feedback.fork()
                if hasattr(session.feedback, "fork")
                else copy.deepcopy(session.feedback)
            )
            candidate.record(
                body.proposal_id,
                body.outcome,
                body.sequence,
                origin=body.origin,
                reason=body.reason,
                transition=body.transition,
            )
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise GatewayError(422, "invalid_feedback", str(error)) from error
        event = self._append(
            body.document_id,
            kind="feedback",
            request=proposal["request"],
            sequence=body.sequence,
            actor=body.origin,
            prediction_event_hash=proposal["event_hash"],
            prepared_event_hash=proposal["prepared_event_hash"],
            feedback=body.model_dump(),
            ledger_record=candidate._records[-1],
            preference_qualification="unqualified_observation",
        )
        session.feedback = candidate
        return self._feedback_receipt(event)

    @staticmethod
    def _feedback_receipt(event):
        feedback = event["feedback"]
        return {
            "recorded": True,
            "persisted": True,
            "document_id": event["document_id"],
            "event_id": feedback["event_id"],
            "proposal_id": feedback["proposal_id"],
            "outcome": feedback["outcome"],
            "event_hash": event["event_hash"],
            "seq": event["seq"],
        }


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

    @router.get("/feedback/sessions")
    async def feedback_sessions(request: Request):
        return {"sessions": request.app.state.copilot.evidence.sessions()}

    @router.get("/feedback/export")
    async def export_feedback(document_id: str, request: Request):
        store = request.app.state.copilot.evidence
        try:
            records, raw = store.read(document_id)
        except EvidenceError as error:
            raise HTTPException(
                503, detail={"code": "feedback_storage", "message": str(error)}
            ) from error
        if not records:
            raise HTTPException(404, detail="No saved copilot session.")
        return Response(
            raw,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": f'attachment; filename="{store.path(document_id).name}"',
                "Cache-Control": "no-store",
                "X-Copilot-Evidence-Head": records[-1]["event_hash"],
            },
        )

    @router.post("/cancel")
    async def cancel(body: RequestIdentity, request: Request):
        return request.app.state.copilot.cancel(body)

    return router
