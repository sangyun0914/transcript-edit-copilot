"""Five sequential local serving conditions on the exact eight T1 training inputs."""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import time

import httpx

from export_passive_sft import (
    BASE_REPOSITORY,
    BASE_REVISION,
    RUNTIME_REVISION,
    check_hash,
    require,
    resolve,
    sha256,
    write_new,
)
from passive_benchmark import _execution, evaluate_result
from passive_contract import EDIT, Envelope, canonical_messages, validate_prediction
from probe_passive import request_body, server_json, template_check
import train_passive_sft as trainer

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-sft-runtime-smoke-v1"
CONDITIONS = (
    ("native-base-f16", "base", False),
    ("adapted-f16", "merged", False),
    ("native-base-q4-k-m", "base_quantized", False),
    ("adapted-q4-k-m", "merged_quantized", False),
    ("adapted-q4-k-m-schema", "merged_quantized", True),
)


def read_bound(binding, parent):
    path = resolve(binding["path"], parent)
    check_hash(path, binding["sha256"])
    return path, json.loads(path.read_text())


def prompt_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def serving_rows(cases, tokens):
    result = []
    for case, token in zip(cases, tokens, strict=True):
        messages = token["messages"]
        require(
            messages == canonical_messages(case["input"])
            and [message["role"] for message in messages] == ["system", "user"],
            "Serving messages differ from the frozen predictor evidence.",
        )
        values = token["primary"]
        prefix = values["input_ids"][: values["completion_start"]]
        result.append({"case": case, "messages": messages, "prefix": prefix})
    return result


def load_inputs(config_path):
    config = json.loads(config_path.read_text())
    require(
        config["version"] == trainer.VERSION and config["candidate_id"] == "qwen35-9b",
        "Use the selected T1 smoke input configuration.",
    )
    check_hash(Path(trainer.__file__), config["trainer_sha256"])
    root = config_path.parent
    prep_path, receipt = read_bound(config["bindings"]["preparation"], root)
    _, selection = read_bound(config["bindings"]["selection"], root)
    require(
        selection["selected_t1_candidate_ids"] == ["qwen35-9b"]
        and selection["bindings"]["train_only_preparation"]["sha256"]
        == sha256(prep_path)
        and receipt["native_revision"] == BASE_REVISION
        and config["base"]["repository"] == BASE_REPOSITORY
        and config["base"]["revision"] == BASE_REVISION,
        "Selection, preparation and native base bindings differ.",
    )
    names = (
        "configuration.json",
        "smoke-cases.jsonl",
        "smoke-messages.jsonl",
        "smoke-tokens.jsonl",
        "smoke-model-inputs.jsonl",
    )
    for name in names:
        check_hash(prep_path.parent / name, receipt["files_sha256"][name])
    prepared = json.loads((prep_path.parent / "configuration.json").read_text())
    cases, messages, tokens, inputs = [
        [
            json.loads(line)
            for line in (prep_path.parent / name).read_text().splitlines()
        ]
        for name in names[1:]
    ]
    trainer.validate_smoke(config, prepared, cases, messages, tokens, inputs)
    generation = config["generation"]
    require(
        generation["max_new_tokens"] == 512
        and generation["do_sample"] is False
        and generation["num_beams"] == 1,
        "Keep the registered greedy 512-token decoding budget.",
    )
    tokenizer = config["tokenizer"]
    template = resolve(tokenizer["path"], root) / "chat_template.jinja"
    check_hash(template, tokenizer["template_sha256"])
    return config, receipt, serving_rows(cases, tokens), template


def attach_execution(rows, binding, parent, expected_release):
    if binding is None:
        return {
            row["case"]["id"]: "No frozen release manifest was bound." for row in rows
        }, {}
    path, manifest = read_bound(binding, parent)
    require(
        sha256(path) == expected_release,
        "Execution evidence belongs to a different release.",
    )
    dependencies = manifest["dependencies_sha256"]
    for name, expected in dependencies.items():
        if name.startswith("ml/") and name.endswith(".py"):
            check_hash(ROOT / name, expected)
    groups_path = path.parent / "source-groups.json"
    check_hash(groups_path, manifest["files_sha256"]["source-groups.json"])
    groups = json.loads(groups_path.read_text())
    accessed = {}

    def read_source(source_path):
        source_path = source_path.resolve()
        require(
            source_path.is_relative_to(ROOT), "Execution evidence escapes the project."
        )
        name = str(source_path.relative_to(ROOT))
        require(name in dependencies, f"Unbound execution dependency: {name}")
        check_hash(source_path, dependencies[name])
        accessed[name] = dependencies[name]
        return json.loads(source_path.read_text())

    unavailable = {}
    for row in rows:
        case = row["case"]
        require(
            groups.get(case["metadata"]["source_group_id"]) == "train",
            "A non-train source cannot enter serving smoke.",
        )
        try:
            case["_execution"] = _execution(case, read_source)
        except FileNotFoundError as error:
            unavailable[case["id"]] = str(error)
    return unavailable, accessed


def validate_native_smoke(values, rows, eos):
    require(
        isinstance(values, list) and len(values) == 8,
        "Native smoke must contain all eight generation results.",
    )
    require(
        [value["case_id"] for value in values] == [row["case"]["id"] for row in rows],
        "Native smoke row identities/order differ.",
    )
    for value, row in zip(values, rows, strict=True):
        ids = value["output_ids"]
        require(
            value["prompt_ids_sha256"] == prompt_digest(row["prefix"])
            and isinstance(ids, list)
            and 0 < len(ids) <= 512
            and all(type(token) is int and token >= 0 for token in ids)
            and isinstance(value["text"], str)
            and value["finish_reason"] == ("stop" if ids[-1] == eos else "length"),
            "Native smoke prompt, budget or EOS/stopping differs.",
        )


def prepare(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    require(config.get("version") == VERSION, "Unknown serving smoke configuration.")
    parent = config_path.parent
    input_path, _ = read_bound(config["training_input"], parent)
    training, preparation, rows, template = load_inputs(input_path)
    export_path, export = read_bound(config["export_receipt"], parent)
    require(
        export["version"] == "passive-sft-export-v1"
        and export["status"]
        == "tensor_verified_paired_export_runtime_preflight_pending"
        and export["base_repository"] == BASE_REPOSITORY
        and export["base_revision"] == BASE_REVISION
        and export["runtime_revision"] == RUNTIME_REVISION,
        "Use the completed, verified export pair from the selected native base.",
    )
    training_record_path, training_record = read_bound(
        export["training_record"], export_path.parent
    )
    if "configuration_sha256" not in training_record:
        training_record_path, training_record = read_bound(
            {"path": "run.json", "sha256": training_record["files_sha256"]["run.json"]},
            training_record_path.parent,
        )
    require(
        training_record["configuration_sha256"] == sha256(input_path)
        and training_record["training_updates"] == 2,
        "Export does not bind this completed two-update training run.",
    )
    native_path, native = read_bound(config["native_smoke"], parent)
    require(
        native_path.parent == training_record_path.parent
        and native_path.name == "smoke-generation.json",
        "Use generation results from the bound training record's directory.",
    )
    eos = training["generation"]["eos_token_id"]
    validate_native_smoke(native, rows, eos)
    tensor_path = export_path.parent / "tensor-verification.json"
    check_hash(tensor_path, export["tensor_verification_sha256"])
    tensor_receipt = json.loads(tensor_path.read_text())
    require(
        tensor_receipt["eos_token_id"] == eos
        and tensor_receipt["template_sha256"] == sha256(template),
        "Export EOS/template differs from the frozen training input.",
    )
    models = {}
    for _, key, _ in CONDITIONS:
        if key in models:
            continue
        model = export["artifacts"][key]
        path = resolve(model["path"], export_path.parent)
        check_hash(path, model["sha256"])
        models[key] = str(path)
    runtime = config["runtime"]
    require(
        runtime["revision"] == RUNTIME_REVISION, "Use the bound b10816 server runtime."
    )
    server = resolve(runtime["server"], parent)
    check_hash(server, runtime["sha256"])
    require(bool(runtime["files_sha256"]), "Bind the runtime support libraries.")
    for name, expected in runtime["files_sha256"].items():
        check_hash(resolve(name, parent), expected)
    timeout = config.get("request_timeout_seconds", 120)
    startup_timeout = config.get("startup_timeout_seconds", 180)
    require(
        all(
            type(value) in (float, int) and math.isfinite(value) and 0 < value <= 300
            for value in (timeout, startup_timeout)
        ),
        "Use a bounded positive request/startup timeout of at most 300 seconds.",
    )
    unavailable, execution_files = attach_execution(
        rows,
        config.get("release_manifest"),
        parent,
        preparation["release_manifest_sha256"],
    )
    sampling = {
        "temperature": 0,
        "seed": training["training"]["seed"],
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "max_tokens": 512,
    }
    return {
        "config": config,
        "rows": rows,
        "models": models,
        "server": str(server),
        "template": str(template),
        "eos": eos,
        "native": native,
        "sampling": sampling,
        "timeout": timeout,
        "startup_timeout": startup_timeout,
        "execution_unavailable": unavailable,
        "execution_files_sha256": execution_files,
        "protected_directories": [
            input_path.parent,
            export_path.parent,
            native_path.parent,
        ],
    }


def grade(case, text, finish_reason):
    if "_execution" in case:
        result = evaluate_result(case, text, finish_reason)
        result.update(execution_available=True, contract_valid=result["raw_valid"])
        return result
    result = {
        "case_id": case["id"],
        "format_valid": False,
        "contract_valid": False,
        "execution_available": False,
        "raw_valid": None,
        "raw_grade": None,
        "prediction": None,
        "displayed": None,
    }
    envelope = Envelope(
        document_id=case["id"], request_id="t1-smoke", revision=0, interaction_epoch=0
    )
    try:
        EDIT.validate_json(text, strict=True)
        result["format_valid"] = True
        result["prediction"] = validate_prediction(
            text, case["input"], envelope, envelope, finish_reason
        ).model_dump()
        result["contract_valid"] = True
    except (ValueError, TypeError) as error:
        result["error"] = str(error)
    return result


def make_body(row, identity, prepared, constrained):
    body = request_body(
        row["case"],
        identity,
        constrained=constrained,
        sampling=prepared["sampling"],
        messages=row["messages"],
    )
    body["ignore_eos"] = False
    return body


def stop_server(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_condition(prepared, condition, output):
    identity, model_key, constrained = condition
    output.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        prepared["server"],
        "-m",
        prepared["models"][model_key],
        "--alias",
        identity,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "4096",
        "-np",
        "1",
        "-ngl",
        "99",
        "--jinja",
        "--no-ui",
        "--perf",
        "--cache-ram",
        "0",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--chat-template-file",
        prepared["template"],
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    write_new(
        output / "command.json",
        {"argv": command, "condition": identity, "schema_constrained": constrained},
    )
    state = {
        "condition": identity,
        "status": "failed",
        "expected_requests": len(prepared["rows"]),
        "schema_constrained": constrained,
    }
    process, results = None, []
    try:
        with (output / "server.log").open("x") as log:
            try:
                started = time.perf_counter()
                process = subprocess.Popen(command, stdout=log, stderr=log)
                with httpx.Client(
                    base_url=f"http://127.0.0.1:{port}",
                    timeout=prepared["timeout"],
                    trust_env=False,
                ) as client:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError("llama-server exited during startup.")
                        try:
                            if client.get("/health", timeout=1).status_code == 200:
                                break
                        except httpx.RequestError:
                            pass
                        if time.perf_counter() - started > prepared["startup_timeout"]:
                            raise TimeoutError(
                                "Model startup exceeded the registered timeout."
                            )
                        time.sleep(0.25)
                    state["load_and_warmup_seconds"] = time.perf_counter() - started
                    response = client.get("/props")
                    response.raise_for_status()
                    props = response.json()
                    write_new(output / "props.json", props)
                    require(
                        props["model_alias"] == identity
                        and RUNTIME_REVISION[:9] in props["build_info"],
                        "Serving alias/build identity differs.",
                    )
                    eos_ids = server_json(
                        client,
                        "/tokenize",
                        {
                            "content": props["eos_token"],
                            "add_special": False,
                            "parse_special": True,
                        },
                    )["tokens"]
                    require(
                        eos_ids == [prepared["eos"]],
                        "Runtime EOS differs from native training generation EOS.",
                    )
                    checks = []
                    with (output / "template-checks.jsonl").open("x") as stream:
                        for row in prepared["rows"]:
                            check = {
                                "case_id": row["case"]["id"],
                                **template_check(
                                    client,
                                    make_body(row, identity, prepared, constrained),
                                    row["prefix"],
                                ),
                            }
                            checks.append(check)
                            stream.write(json.dumps(check, ensure_ascii=False) + "\n")
                            stream.flush()
                    state["template_checks_passed"] = sum(
                        check["passed"] and check["within_budget"] for check in checks
                    )
                    require(
                        state["template_checks_passed"] == len(prepared["rows"]),
                        "Full frozen prefix parity failed; generation is blocked.",
                    )
                    with (output / "results.jsonl").open("x") as stream:
                        for row in prepared["rows"]:
                            result = {
                                "case_id": row["case"]["id"],
                                "request_failed": False,
                            }
                            started = time.perf_counter()
                            try:
                                body = make_body(row, identity, prepared, constrained)
                                result["request"] = body
                                response = client.post(
                                    "/v1/chat/completions", json=body
                                )
                                result.update(
                                    http_response_ms=(time.perf_counter() - started)
                                    * 1000,
                                    http_status=response.status_code,
                                    raw_http_body=response.text,
                                )
                                response.raise_for_status()
                                data = response.json()
                                result["response"] = data
                                choice = data["choices"][0]
                                text = choice["message"].get("content") or ""
                                result.update(
                                    raw_text=text, finish_reason=choice["finish_reason"]
                                )
                                require(
                                    data["usage"]["prompt_tokens"]
                                    == len(row["prefix"]),
                                    "Completion prompt length differs from the verified prefix.",
                                )
                                require(
                                    data["usage"]["completion_tokens"] <= 512,
                                    "Generation exceeded the registered output budget.",
                                )
                                result.update(
                                    grade(row["case"], text, choice["finish_reason"])
                                )
                            except (Exception, KeyboardInterrupt) as error:
                                result.update(
                                    request_failed=True,
                                    error=f"{type(error).__name__}: {error}",
                                )
                                raise
                            finally:
                                result["request_and_grading_ms"] = (
                                    time.perf_counter() - started
                                ) * 1000
                                results.append(result)
                                stream.write(
                                    json.dumps(result, ensure_ascii=False) + "\n"
                                )
                                stream.flush()
                            print(
                                f"{identity}: {len(results)}/{len(prepared['rows'])} complete",
                                flush=True,
                            )
                    state["status"] = "completed"
            finally:
                stop_server(process)
    except (Exception, KeyboardInterrupt) as error:
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state.update(
            observed_requests=len(results),
            missing_requests=len(prepared["rows"]) - len(results),
            server_stopped=process is None or process.poll() is not None,
        )
        write_new(output / "receipt.json", state)
    return results


def comparison(rows, native, conditions):
    def results_by_id(values):
        return {value["case_id"]: value for value in values}

    native_results = [
        {
            **grade(row["case"], value["text"], value["finish_reason"]),
            "raw_text": value["text"],
            "finish_reason": value["finish_reason"],
            "output_ids": value["output_ids"],
        }
        for row, value in zip(rows, native, strict=True)
    ]
    all_results = {
        "colab-adapted-nf4-bf16": native_results,
        **{identity: conditions.get(identity, []) for identity, _, _ in CONDITIONS},
    }
    counts = {}
    for name, values in all_results.items():
        counts[name] = {
            "observed": len(values),
            "missing": len(rows) - len(values),
            "request_failures": sum(
                value.get("request_failed", False) for value in values
            ),
            "format_valid": sum(value.get("format_valid") is True for value in values),
            "contract_valid": sum(
                value.get("contract_valid") is True for value in values
            ),
            "execution_valid": sum(value.get("raw_valid") is True for value in values),
            "execution_unavailable": sum(
                value.get("execution_available") is False for value in values
            ),
            "matched_outcomes": sum(
                (value.get("raw_grade") or {}).get("outcome_match") is True
                for value in values
            ),
        }
    pairs = []
    for left, right in (
        ("colab-adapted-nf4-bf16", "adapted-f16"),
        ("adapted-f16", "adapted-q4-k-m"),
        ("native-base-f16", "native-base-q4-k-m"),
        ("native-base-f16", "adapted-f16"),
        ("native-base-q4-k-m", "adapted-q4-k-m"),
        ("adapted-q4-k-m", "adapted-q4-k-m-schema"),
    ):
        a, b = (
            results_by_id(all_results.get(left, [])),
            results_by_id(all_results.get(right, [])),
        )
        observations = []
        for row in rows:
            identity = row["case"]["id"]
            if (
                identity not in a
                or identity not in b
                or a[identity].get("request_failed")
                or b[identity].get("request_failed")
            ):
                continue
            before, after = a[identity], b[identity]
            observations.append(
                {
                    "case_id": identity,
                    "same_text": before["raw_text"] == after["raw_text"],
                    "same_valid_prediction": before["prediction"] == after["prediction"]
                    if before.get("contract_valid") and after.get("contract_valid")
                    else None,
                    "before_outcome_match": (before.get("raw_grade") or {}).get(
                        "outcome_match"
                    ),
                    "after_outcome_match": (after.get("raw_grade") or {}).get(
                        "outcome_match"
                    ),
                }
            )
        pairs.append(
            {"before": left, "after": right, "paired_observations": observations}
        )
    return {
        "counts": counts,
        "pairs": pairs,
        "native_results": native_results,
        "scope": "Eight training states; engineering round trip and in-sample diagnostics only. Different quantization/runtime outputs need not be identical. Chat endpoint output-token identities are not inferred from text.",
    }


def run(config_path, output, prepare_only=False):
    prepared = prepare(config_path)
    output = output.resolve()
    require(
        not any(
            output.is_relative_to(parent)
            for parent in prepared["protected_directories"]
        ),
        "Serving output must be outside bound input/model/training directories.",
    )
    require(
        not any(
            parent.name == "frozen"
            and parent.is_relative_to(ROOT / "artifacts/datasets")
            for parent in (output, *output.parents)
        ),
        "Serving output cannot modify a frozen dataset.",
    )
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "version": VERSION,
        "config_sha256": sha256(config_path),
        "configuration": prepared["config"],
        "conditions": [
            {"id": identity, "artifact": model, "schema_constrained": constrained}
            for identity, model, constrained in CONDITIONS
        ],
        "case_ids": [row["case"]["id"] for row in prepared["rows"]],
        "expected_generation_requests": 40,
        "sampling": prepared["sampling"],
        "eos_token_id": prepared["eos"],
        "execution_unavailable": prepared["execution_unavailable"],
        "execution_files_sha256": prepared["execution_files_sha256"],
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "probe_passive_sft.py",
                "probe_passive.py",
                "passive_benchmark.py",
                "probe.py",
                "train_passive_sft.py",
                "export_passive_sft.py",
                "export_qwen06.py",
            )
        },
        "request_timeout_seconds": prepared["timeout"],
        "startup_timeout_seconds": prepared["startup_timeout"],
        "latency_scope": "Offline HTTP response; source/reference grading measured separately. No editor trigger-to-preview or controlled performance claim.",
        "test_or_dev_bodies_read": 0,
    }
    write_new(output / "protocol.json", protocol)
    status, conditions = "prepared_generation_unrun", {}
    try:
        if prepare_only:
            return
        for condition in CONDITIONS:
            conditions[condition[0]] = run_condition(
                prepared, condition, output / condition[0]
            )
        status = "completed"
    except (Exception, KeyboardInterrupt) as error:
        status = "failed"
        write_new(
            output / "failure.json", {"error": f"{type(error).__name__}: {error}"}
        )
        raise
    finally:
        for identity, _, _ in CONDITIONS:
            path = output / identity / "results.jsonl"
            if identity not in conditions and path.exists():
                conditions[identity] = [
                    json.loads(line) for line in path.read_text().splitlines()
                ]
        summary = comparison(prepared["rows"], prepared["native"], conditions)
        write_new(output / "summary.json", {"status": status, **summary})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    lock_path = ROOT / "artifacts/nep-passive-b1.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(
                "Another passive benchmark or predictor owns the inference lock."
            )
        run(args.config.resolve(), args.output, args.prepare_only)


if __name__ == "__main__":
    main()
