"""Authorized small-family deployed GGUF inference and admission on the sealed cohort.

This auxiliary product row never changes the native roles, requests or references.
The caller owns the local server; there is no predictor callback or TEST-to-DEV path.
"""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import benchmark_final_runtime as http
import infer_task_test_actions_v3 as native
import prepare_task_test_actions_v3 as core
import serve_small_actions_final as serving
import small_final_test_runtime as small

ROOT = core.ROOT
VERSION = "small-final-deployed-observations-1"
POLICY_VERSION = "small-final-deployed-evaluation-policy-1"
PLAN_SHA256 = "81bbbbad1f43765642defafe4ca1845fec15a35b5146edddeb85f742d540faac"
ROLE = "deployed_selected"
require, read, sha, digest = core.require, core.read_json, core.sha256, core.digest


def sources():
    values = core.sealed_v1.source_closure(
        [
            Path(__file__),
            ROOT / "ml/compare_deployed_task_test_actions_v3.py",
        ]
    )
    for path in (
        ROOT / "server/app/copilot.py",
        ROOT / "server/scripts/run_local_copilot_small_final.py",
        ROOT / "server/scripts/run_local_copilot_actions_v3.py",
    ):
        values[str(path.relative_to(ROOT))] = sha(path)
    return dict(sorted(values.items()))


def request_settings(gateway):
    return {
        "model": gateway["model_id"],
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
        "cache_prompt": gateway["cache_prompt"],
        "stream": False,
        "lora": [{"id": 0, "scale": gateway["adapter_scale"]}] if gateway.get("adapter_path") else [],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "next_edit",
                "strict": True,
                "schema": serving.runtime_schema(core.contract.OUTPUT_SCHEMA),
            },
        },
    }


def _deployment_check(directory):
    for path in (ROOT / "server", ROOT / "server/scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_local_copilot_small_final as deployment

    gateway, _, command, report = deployment.check(directory)
    return gateway.model_dump(), command, report


def validate_policy(value, selected_model, native_run, config, tokenizer_identity):
    """Metadata-only validation. This function never authorizes or reads TEST bodies."""
    if selected_model.get("runtime_family") == "gemma26-final-native-v1":
        from gemma_final_deployment import validate_policy as validate_gemma
        return validate_gemma(value, selected_model, native_run, config, tokenizer_identity)
    require(
        value.get("version") == POLICY_VERSION
        and value.get("status") == "registered_before_test"
        and value.get("scope") == "final_test"
        and value.get("role") == ROLE
        and value.get("body_access_authorized") is False
        and value.get("final_test_used_for_tuning") is False,
        "Unknown prospective deployed-model policy.",
    )
    require(
        selected_model.get("runtime_family") == small.FAMILY
        and selected_model["role"] in {"sft", "sft_dpo"}
        and native_run["role"] == selected_model["role"]
        and native_run["format"] == "json"
        and native_run["condition"]
        == core.conditions.condition_specs([])["canonical_full_context"],
        "The deployed auxiliary requires the selected canonical small-family SFT/DPO run.",
    )
    small.validate_configuration(config)
    require(
        value["selected_native_run_id"] == native_run["id"]
        and read(core.bound(value["selected_native_model"])) == selected_model
        and value["selected_native_identity"] == selected_model["identity"]
        and value["selected_native_identity_sha256"]
        == digest(selected_model["identity"])
        and value["native_inference_configuration"]
        == selected_model["inference_configuration"]
        and read(core.bound(value["native_inference_configuration"])) == config
        and value["native_tokenizer"]
        == tokenizer_identity
        == small.tokenizer_identity(config),
        "Deployed policy changed its selected native model, configuration or tokenizer.",
    )
    require(
        value["plan"]["sha256"] == PLAN_SHA256 and value["source_sha256"] == sources(),
        "Prospective deployment plan or implementation changed.",
    )
    core.bound(value["plan"])
    core.bound(value["native_core_registration"])
    directory = core.bound(value["deployment"]).parent
    gateway, command, report = _deployment_check(directory)
    require(
        core.bound(value["gateway"]) == directory / "gateway.json"
        and read(core.bound(value["gateway"])) == gateway
        and core.bound(value["command"]) == directory / "command.json"
        and read(core.bound(value["command"])) == command
        and value["export_receipt"] == report["export_receipt"]
        and value["runtime_registry"] == report["runtime_registry"]
        and value["runtime_files_sha256"] == report["runtime_files_sha256"],
        "The auxiliary differs from the actual prepared deployment.",
    )
    export = read(core.bound(value["export_receipt"]))
    core.bound(value["runtime_registry"])
    require(
        core.bound(value["runtime_binary"]) == Path(command[0]).resolve()
        and core.bound(value["model_file"]) == Path(gateway["model_path"]).resolve()
        and core.bound(value["chat_template"])
        == Path(gateway["chat_template_path"]).resolve()
        and value["model_file"]["sha256"]
        == gateway["model_sha256"]
        == export["gguf_sha256"]
        and value["chat_template"]["sha256"]
        == gateway["chat_template_sha256"]
        == config["assets"]["files_sha256"]["chat_template.jinja"],
        "GGUF, native template or runtime binary binding changed.",
    )
    identity = selected_model["identity"]
    expected = {
        "runtime_family": "small-actions-final-gguf",
        "role": ROLE,
        "model": config["model"],
        "model_id": gateway["model_id"],
        "model_sha256": gateway["model_sha256"],
        "adapter_sha256": None,
        "training_adapter_sha256": identity["adapter_sha256"],
        "kind": identity["kind"],
        **{
            key: report[key]
            for key in ("parent_adapter_sha256", "dpo_configuration_sha256")
            if key in report
        },
    }
    require(
        value["model_identity"] == expected
        and export["status"] == "gguf_exported_runtime_validation_pending"
        and export["kind"] == report["kind"] == identity["kind"]
        and export["base"] == report["base"] == config["base"]
        and export["adapter_sha256"]
        == report["training_adapter_sha256"]
        == identity["adapter_sha256"]
        and export["training_configuration_sha256"] == identity["configuration_sha256"]
        and report["native_preparation"]["sha256"] == identity["configuration_sha256"]
        and value["numerical_regime"]
        == report["numerical_regime"]
        == "native_bf16_merged_" + export["quantization"].lower(),
        "Deployed GGUF is not the exact selected native-trained adapter export.",
    )
    if identity["kind"] == "dpo":
        trained = read(
            core.bound(selected_model["artifact_manifest"]).parent
            / "dpo_configuration.json"
        )
        require(
            expected["parent_adapter_sha256"]
            == export["parent_adapter_sha256"]
            == trained["parent"]["adapter_sha256"]
            and expected["dpo_configuration_sha256"]
            == export["dpo_configuration_sha256"]
            == sha(
                core.bound(selected_model["artifact_manifest"]).parent
                / "dpo_configuration.json"
            ),
            "Selected DPO GGUF changed its actual SFT parent or DPO configuration.",
        )
    require(
        gateway["version"] == 2
        and gateway["output_format"] == "json"
        and gateway["adapter_path"] is None
        and gateway["adapter_sha256"] is None
        and gateway["capture_runtime_evidence"] is False
        and report["purpose"] == "ui_performance"
        and command[command.index("-c") + 1] == "4096"
        and value["canonical_prefix_limit"] == 3584
        and value["context_tokens"] == 4096
        and value["output_tokens"] == 512
        and digest(value["request_settings"]) == digest(request_settings(gateway))
        and value["request_deadline_seconds"] == 30
        and value["max_attempts_per_request"] == 1
        and value["overflow_policy"] == "fail_whole_preparation"
        and all(
            value[key] is True
            for key in (
                "schema_constrained",
                "all_roles_same_case_ids",
                "native_prefix_parity_required",
                "all_request_failures_in_denominator",
            )
        ),
        "Frozen deployed decoding, cache, schema or full-cohort policy changed.",
    )
    url = urlsplit(gateway["endpoint"])
    require(
        url.scheme == "http"
        and url.hostname in {"127.0.0.1", "localhost", "::1"}
        and url.port is not None
        and not url.username
        and not url.password
        and not url.query
        and not url.fragment
        and url.path in {"", "/"},
        "Use the frozen local endpoint.",
    )
    return gateway


def load_authorized(freeze, sealed):
    gate = core.load_gate(freeze)
    leaf = gate["receipt"].get("deployment_evaluation_policy")
    require(
        leaf is not None,
        "The core freeze did not authorize a deployed-selected auxiliary.",
    )
    policy = read(core.bound(leaf))
    selected = gate["receipt"]["selected_model"]
    model = gate["models"][selected]
    run = next(
        row
        for row in gate["receipt"]["runs"]
        if row["id"] == policy["selected_native_run_id"]
    )
    require(
        policy["native_core_registration"] == gate["receipt"]["registration"]
        and policy["protocol_id"] == gate["receipt"]["protocol_id"]
        and policy["selected_native_model"] == gate["receipt"]["models"][selected],
        "Auxiliary policy belongs to another core freeze or selection.",
    )
    gateway = validate_policy(
        policy,
        model,
        run,
        read(core.bound(model["inference_configuration"])),
        gate["policies"]["runtime"]["tokenizer"],
    )
    release = core.load_stage(gate, sealed, "sealed_before_inference")
    ids = release["case_ids"]
    require(
        ids
        and len(ids) == len(set(ids)) == release["admitted_cases"]
        and all(value == ids for value in release["case_ids_by_run"].values()),
        "The product auxiliary requires the unchanged full cohort for every native role.",
    )
    path = Path(sealed) / f"requests-{run['id']}.jsonl"
    requests = core.rows(path)
    require(
        [row["case_id"] for row in requests] == ids,
        "Frozen deployed request census changed.",
    )
    for row in requests:
        core.validate_request(row, run, gate)
        require(
            len(row["prefix_ids"]) <= 3584,
            "Entire deployed cohort must fit 3584; no truncation or exclusions.",
        )
    return {
        "gate": gate,
        "policy": policy,
        "gateway": gateway,
        "source_run": run,
        "release": release,
        "requests": requests,
        "requests_binding": core.binding(path),
        "sealed_binding": core.binding(Path(sealed) / "manifest.json"),
    }


def common(context):
    policy = context["policy"]
    return {
        "scope": "final_test",
        "auxiliary_role": ROLE,
        "freeze": context["gate"]["binding"],
        "deployment_evaluation_policy": context["gate"]["receipt"][
            "deployment_evaluation_policy"
        ],
        "sealed_manifest": context["sealed_binding"],
        "requests": context["requests_binding"],
        "source_run_id": context["source_run"]["id"],
        "model_identity": policy["model_identity"],
        "selected_native_identity": policy["selected_native_identity"],
        "source_sha256": policy["source_sha256"],
        "numerical_regime": policy["numerical_regime"],
        "schema_constrained": True,
        "final_test_bodies_read": True,
        "reference_labels_available_to_predictor": False,
        "engineering_fixture_only": context["gate"]["receipt"].get(
            "engineering_fixture_only", False
        ),
    }


def observation_identity(row, context):
    value = {
        key: row[key]
        for key in (
            "case_id",
            "split",
            "run_id",
            "format",
            "input_sha256",
            "messages_sha256",
            "prefix_ids_sha256",
            "condition_sha256",
            "freeze_sha256",
        )
    }
    return {
        **value,
        "auxiliary_role": ROLE,
        "request_sha256": digest(row),
        "model_identity": context["policy"]["model_identity"],
        "deployment_policy_sha256": context["gate"]["receipt"][
            "deployment_evaluation_policy"
        ]["sha256"],
    }


def finite_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def same_prefix(actual, expected):
    return (
        isinstance(actual, list)
        and all(type(token) is int for token in actual)
        and actual == expected
    )


def verify_http_capture(value):
    require(finite_time(value["elapsed_ms"]), "Invalid actual HTTP duration.")
    if "raw_text" in value:
        payload = (
            base64.b64decode(value["raw_body_base64"], validate=True)
            if "raw_body_base64" in value
            else value["raw_text"].encode()
        )
        require(
            hashlib.sha256(payload).hexdigest() == value["raw_body_sha256"]
            and payload.decode("utf-8", errors="replace") == value["raw_text"],
            "Raw HTTP bytes changed.",
        )
        if "raw" in value:
            require(
                digest(json.loads(payload)) == digest(value["raw"]),
                "Parsed HTTP response differs from captured bytes.",
            )
    else:
        require(
            "error" in value and "raw" not in value,
            "HTTP capture has neither a response nor a failure.",
        )
    if "error" not in value:
        require(
            value.get("http_status") == 200 and "raw" in value,
            "Successful HTTP capture is incomplete.",
        )
    else:
        require(
            bool(value["error_type"]) and isinstance(value["error"], str),
            "Missing actual HTTP failure.",
        )
    return value.get("raw")


def response(value):
    raw = verify_http_capture(value)
    require("error" not in value, value.get("error", "Runtime request failed."))
    return raw


def verify_runtime(runtime, context):
    gateway, policy = context["gateway"], context["policy"]
    require(
        response(runtime["health"]).get("status") in {"ok", "ready"},
        "Frozen runtime is not ready.",
    )
    props = response(runtime["props"])
    settings = props.get("default_generation_settings", {})
    reasoning = settings.get("params", {})
    require(
        runtime["endpoint"] == gateway["endpoint"]
        and props.get("model_alias") == gateway["model_id"]
        and Path(props.get("model_path", "")).resolve()
        == Path(gateway["model_path"]).resolve()
        and props.get("build_info") == gateway["runtime_build"]
        and props.get("total_slots") == 1
        and settings.get("n_ctx", props.get("n_ctx")) == policy["context_tokens"]
        and reasoning.get("reasoning_format") == "none"
        and not reasoning.get("reasoning_in_content")
        and props.get("chat_template", "").rstrip("\n")
        == Path(gateway["chat_template_path"]).read_text().rstrip("\n"),
        "Actual native runtime differs from the frozen endpoint, GGUF, build, template, reasoning, slot or context.",
    )
    if gateway.get("adapter_path"):
        from gemma_final_deployment import verify_loaded_adapter
        verify_loaded_adapter(gateway, response(runtime["lora_adapters"]))
    else:
        require(response(runtime["lora_adapters"]) == [], "A merged selected GGUF must not load an extra LoRA.")


def expected_calls(row, context, prompt=None):
    settings = context["policy"]["request_settings"]
    apply = {
        "model": settings["model"],
        "messages": row["messages"],
        "chat_template_kwargs": settings["chat_template_kwargs"],
    }
    tokenize = {"content": prompt, "add_special": True, "parse_special": True}
    return apply, tokenize, {**settings, "messages": row["messages"]}


def completion_value(capture, context):
    raw = response(capture)
    require(
        raw.get("model") == context["gateway"]["model_id"]
        and len(raw.get("choices", [])) == 1,
        "Actual HTTP response model/choice differs.",
    )
    choice = raw["choices"][0]
    require(
        isinstance(choice["message"]["content"], str)
        and choice["finish_reason"] in {"stop", "length"},
        "Malformed actual completion.",
    )
    tokens = raw.get("usage", {}).get("completion_tokens")
    require(
        type(tokens) is int and 0 <= tokens <= 512,
        "Actual completion exceeded or omitted its token budget.",
    )
    return choice["message"]["content"], choice["finish_reason"]


def failed(row, context, error, stage, verified, attempted=False, **extra):
    return {
        **observation_identity(row, context),
        "status": "request_failure",
        "error": str(error),
        "error_type": type(error).__name__,
        "failure_stage": stage,
        "model_identity_verified": verified,
        "generation_attempted": attempted,
        "native_token_parity": stage == "generation" and verified,
        "generation_ms": None,
        "timing_source": "not_measured",
        **extra,
    }


def capture_request(captures, name, endpoint, route, body, timeout):
    capture = captures[name] = {"request": body}
    started = time.perf_counter()
    try:
        capture["response"] = http.request(endpoint, route, body, timeout)
    except BaseException as error:
        capture["response"] = {
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "http_status": None,
            "error": str(error),
            "error_type": type(error).__name__,
        }
        raise
    return capture["response"]


def run(freeze, sealed, output):
    context = load_authorized(freeze, sealed)
    output = Path(output).resolve()
    require(
        output.is_relative_to(ROOT) and not output.exists(),
        "Use a fresh workspace inference directory.",
    )
    output.mkdir(parents=True, mode=0o700)
    policy, gateway, requests = (
        context["policy"],
        context["gateway"],
        context["requests"],
    )
    info = {
        **common(context),
        "version": VERSION,
        "status": "registered",
        "started_at": core.now(),
        "requested": len(requests),
        "observed": 0,
        "model_identity_verified": False,
        "actual_model_identity": None,
        "max_attempts_per_request": 1,
        "measurement": "HTTP completion only; excludes parity calls, model load and editor latency",
    }
    core.write(output / "run-registration.json", info)
    emitted, verified, fatal = 0, False, None
    endpoint, timeout = gateway["endpoint"], policy["request_deadline_seconds"]
    with (output / "observations.jsonl").open("x") as stream:
        try:
            runtime = {
                "endpoint": endpoint,
                "health": http.request(endpoint, "/health", timeout=timeout),
                "props": http.request(endpoint, "/props", timeout=timeout),
                "lora_adapters": http.request(
                    endpoint, "/lora-adapters", timeout=timeout
                ),
            }
            core.write(output / "runtime.json", runtime)
            verify_runtime(runtime, context)
            verified = True
            info.update(
                model_identity_verified=True,
                actual_model_identity=policy["model_identity"],
            )
            for row in requests:
                started, stage, attempted = (
                    time.perf_counter(),
                    "native_template",
                    False,
                )
                captures = {}
                interruption = None
                record = observation_identity(row, context)
                try:
                    apply, _, body = expected_calls(row, context)
                    prompt = response(
                        capture_request(
                            captures,
                            "apply_template",
                            endpoint,
                            "/apply-template",
                            apply,
                            timeout,
                        )
                    )["prompt"]
                    require(
                        isinstance(prompt, str),
                        "Native chat template did not produce text.",
                    )
                    stage = "native_token_parity"
                    _, tokenize, _ = expected_calls(row, context, prompt)
                    tokens = response(
                        capture_request(
                            captures,
                            "tokenize",
                            endpoint,
                            "/tokenize",
                            tokenize,
                            timeout,
                        )
                    )["tokens"]
                    require(
                        same_prefix(tokens, row["prefix_ids"]),
                        "Actual GGUF template/token prefix differs from frozen native IDs.",
                    )
                    stage, attempted = "generation", True
                    capture_request(
                        captures,
                        "completion",
                        endpoint,
                        "/v1/chat/completions",
                        body,
                        timeout,
                    )
                    text, finish_reason = completion_value(
                        captures["completion"]["response"], context
                    )
                    record.update(
                        status="completed",
                        text=text,
                        finish_reason=finish_reason,
                        model_identity_verified=True,
                        native_token_parity=True,
                        generation_attempted=True,
                        http_response_ms=captures["completion"]["response"][
                            "elapsed_ms"
                        ],
                        generation_ms=None,
                        timing_source="http_complete_response_not_native_generation",
                    )
                except BaseException as error:
                    record = failed(row, context, error, stage, verified, attempted)
                    if not isinstance(error, Exception):
                        interruption = error
                record.update(
                    http_evidence=captures,
                    controller_wall_ms=(time.perf_counter() - started) * 1000,
                )
                native.append(stream, record)
                emitted += 1
                if interruption is not None:
                    raise interruption
                print(
                    json.dumps(
                        {
                            "case_id": row["case_id"],
                            "status": record["status"],
                            "index": emitted,
                        }
                    ),
                    flush=True,
                )
        except BaseException as error:
            fatal = error
            for row in requests[emitted:]:
                native.append(
                    stream,
                    failed(
                        row,
                        context,
                        error,
                        "aborted_before_request" if verified else "runtime_preflight",
                        verified,
                    ),
                )
                emitted += 1
    info.update(
        status="inference_complete",
        observed=emitted,
        completed_with_startup_or_interruption_failure=fatal is not None,
    )
    if fatal is not None:
        info.update(error_type=type(fatal).__name__, error=str(fatal))
    core.write(output / "run.json", info)
    core.finish_output(
        output, {"version": VERSION, "status": "inference_complete", **common(context)}
    )
    if isinstance(fatal, (KeyboardInterrupt, SystemExit)):
        raise fatal
    return info


def load_output(context, directory, status):
    directory = Path(directory).resolve()
    manifest = read(directory / "manifest.json")
    require(
        manifest["version"] == VERSION
        and manifest["status"] == status
        and all(manifest.get(key) == value for key, value in common(context).items()),
        "Deployed result belongs to another authorized input, model or implementation.",
    )
    core.evaluation.verify_files(directory, manifest)
    return manifest


def verify_failure(observation, info, captures, row, context):
    stage = observation["failure_stage"]
    keys = {
        "runtime_preflight": set(),
        "aborted_before_request": set(),
        "native_template": {"apply_template"},
        "native_token_parity": {"apply_template", "tokenize"},
        "generation": {"apply_template", "tokenize", "completion"},
    }
    require(
        stage in keys
        and set(captures) == keys[stage]
        and observation["model_identity_verified"] is info["model_identity_verified"]
        and observation["generation_attempted"] is (stage == "generation")
        and observation["native_token_parity"] is (stage == "generation"),
        "Failed request stage, generation attempt or parity evidence changed.",
    )
    if stage in {"runtime_preflight", "aborted_before_request"}:
        require(
            info["completed_with_startup_or_interruption_failure"] is True
            and info["model_identity_verified"] is (stage == "aborted_before_request"),
            "Unattempted request lacks an actual startup or interruption failure.",
        )
        return
    require(
        info["model_identity_verified"] is True,
        "Request preceded runtime verification.",
    )
    try:
        if stage == "native_template":
            require(
                isinstance(
                    response(captures["apply_template"]["response"])["prompt"], str
                ),
                "Native chat template did not produce text.",
            )
        elif stage == "native_token_parity":
            require(
                same_prefix(
                    response(captures["tokenize"]["response"])["tokens"],
                    row["prefix_ids"],
                ),
                "Native prefix differs.",
            )
        else:
            completion_value(captures["completion"]["response"], context)
    except Exception:
        return
    raise ValueError(
        "Failed request has successful evidence for its claimed failure stage."
    )


def verify_observations(context, directory, *, tokenizer=None):
    directory = Path(directory)
    manifest = load_output(context, directory, "inference_complete")
    info = read(directory / "run.json")
    require(
        info["version"] == VERSION
        and info["status"] == "inference_complete"
        and all(info.get(key) == value for key, value in common(context).items())
        and info["max_attempts_per_request"] == 1,
        "Deployed inference receipt changed.",
    )
    registration = read(directory / "run-registration.json")
    require(
        registration["status"] == "registered"
        and registration["requested"] == info["requested"]
        and registration["started_at"] == info["started_at"]
        and all(
            registration.get(key) == value for key, value in common(context).items()
        ),
        "Deployed request registration changed.",
    )
    requests = context["requests"]
    observations = core.rows(directory / "observations.jsonl")
    require(
        info["observed"] == info["requested"] == len(requests)
        and [row["case_id"] for row in observations]
        == [row["case_id"] for row in requests],
        "Missing, extra or reordered deployed observations.",
    )
    if info["model_identity_verified"]:
        require(
            info["actual_model_identity"] == context["policy"]["model_identity"],
            "Actual model identity changed.",
        )
        verify_runtime(read(directory / "runtime.json"), context)
    else:
        require(
            info["model_identity_verified"] is False
            and info["actual_model_identity"] is None,
            "Unverified runtime asserted an actual model identity.",
        )
    tokenizer = tokenizer or core.load_tokenizer(context["gate"])
    for row, observation in zip(requests, observations, strict=True):
        require(
            all(
                observation.get(key) == value
                for key, value in observation_identity(row, context).items()
            )
            and observation["generation_ms"] is None,
            "Deployed observation request/identity or timing regime changed.",
        )
        require(
            same_prefix(
                core.conditions.native_prefix(tokenizer, row["messages"]),
                row["prefix_ids"],
            ),
            "Frozen native prefix readback changed.",
        )
        captures = observation.get("http_evidence", {})
        require(
            set(captures) <= {"apply_template", "tokenize", "completion"},
            "Unknown runtime evidence.",
        )
        apply, _, body = expected_calls(row, context)
        for capture in captures.values():
            verify_http_capture(capture["response"])
        if "apply_template" in captures:
            require(
                digest(captures["apply_template"]["request"]) == digest(apply),
                "Native template request changed.",
            )
        if "tokenize" in captures:
            prompt = response(captures["apply_template"]["response"])["prompt"]
            require(
                digest(captures["tokenize"]["request"])
                == digest(expected_calls(row, context, prompt)[1]),
                "Native tokenization request changed.",
            )
        if "completion" in captures:
            require(
                digest(captures["completion"]["request"]) == digest(body)
                and same_prefix(
                    response(captures["tokenize"]["response"])["tokens"],
                    row["prefix_ids"],
                ),
                "Actual generation bypassed native parity or changed frozen request settings.",
            )
        if observation["status"] == "completed":
            require(
                info["model_identity_verified"] is True
                and observation["model_identity_verified"] is True
                and observation["generation_attempted"] is True
                and observation["native_token_parity"] is True
                and set(captures) == {"apply_template", "tokenize", "completion"}
                and finite_time(observation["controller_wall_ms"]),
                "Completed product output lacks runtime/parity evidence.",
            )
            text, finish = completion_value(captures["completion"]["response"], context)
            require(
                observation["text"] == text
                and observation["finish_reason"] == finish
                and observation["http_response_ms"]
                == captures["completion"]["response"]["elapsed_ms"]
                and observation["timing_source"]
                == "http_complete_response_not_native_generation",
                "Actual completed HTTP output or duration changed.",
            )
        else:
            require(
                observation["status"] == "request_failure"
                and bool(observation["error_type"])
                and isinstance(observation["error"], str)
                and observation["timing_source"] == "not_measured"
                and "text" not in observation
                and "http_response_ms" not in observation
                and type(observation["generation_attempted"]) is bool,
                "Failed deployed request was disguised as an answer or successful latency.",
            )
            verify_failure(observation, info, captures, row, context)
    return info, observations, manifest


def admit(freeze, sealed, observed, output, *, tokenizer=None):
    context = load_authorized(freeze, sealed)
    info, observations, _ = verify_observations(context, observed, tokenizer=tokenizer)
    return core.publish(
        output,
        {"observations.jsonl": observations, "run.json": info},
        {
            "version": VERSION,
            "status": "admitted",
            **common(context),
            "observation_manifest": core.binding(Path(observed) / "manifest.json"),
            "requested": len(context["requests"]),
            "observed": len(observations),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "admit"):
        command = sub.add_parser(name)
        for field in ("freeze", "sealed", "output"):
            command.add_argument("--" + field, required=True, type=Path)
        if name == "admit":
            command.add_argument("--observed", required=True, type=Path)
    args = parser.parse_args()
    value = (
        run(args.freeze, args.sealed, args.output)
        if args.command == "run"
        else admit(args.freeze, args.sealed, args.observed, args.output)
    )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
