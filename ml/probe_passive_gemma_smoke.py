"""Three Q4 serving conditions on the exact eight Gemma engineering train rows."""

import argparse
import fcntl
import importlib.util
import json
import math
from pathlib import Path

import probe_passive_sft as shared
import train_passive_gemma as trainer
import prepare_passive_gemma as preparation_module
from export_passive_gemma import (
    BASE_REPOSITORY,
    BASE_REVISION,
    RUNTIME_REVISION,
    GENERATION_STOP_IDS,
)
from export_passive_sft import check_hash, require, resolve, sha256, write_new
from probe_passive_sft import (
    read_bound,
    prompt_digest,
    serving_rows,
    attach_execution,
    grade,
)

ROOT = shared.ROOT
VERSION = "passive-gemma-runtime-smoke-v1"
CONDITIONS = (
    ("native-base-q4-k-m", "base_quantized", False),
    ("adapted-q4-k-m", "merged_quantized", False),
    ("adapted-q4-k-m-schema", "merged_quantized", True),
)


def load_inputs(config_path, native_directory=None):
    config = json.loads(config_path.read_text())
    trainer.validate_configuration(config)
    require(
        config["phase"] == "smoke",
        "This runner accepts only the eight-row engineering phase.",
    )
    for _, key, source in trainer.SOURCE_MODULES:
        check_hash(Path(source), config[key])
    root = config_path.parent
    prep_path, receipt = read_bound(config["bindings"]["preparation"], root)
    bundle = preparation_module.verify_bundle(prep_path.parent)
    require(bundle["manifest"] == receipt, "Preparation receipt changed.")
    trainer.validate_prepared(config, bundle)
    _, selection = read_bound(config["bindings"]["selection"], root)
    _, protocol = read_bound(config["bindings"]["evaluation_protocol"], root)
    _, pin = read_bound(config["bindings"]["native_pin"], root)
    trainer.validate_protocol(protocol, config, receipt["release_manifest_sha256"])
    require(
        selection["selected_t1_candidate_ids"] == ["gemma26"]
        and selection["bindings"]["train_only_preparation"]["sha256"]
        == sha256(prep_path)
        and pin["repository"] == BASE_REPOSITORY
        and pin["revision"] == BASE_REVISION
        and {name: value["sha256"] for name, value in pin["files"].items()}
        == config["base"]["files_sha256"],
        "Gemma preparation, selection and exact native acquisition must agree.",
    )
    tokenizer = native_directory or resolve(config["tokenizer"]["path"], root)
    for name, expected in config["tokenizer"]["files_sha256"].items():
        check_hash(tokenizer / name, expected)
    template = tokenizer / "chat_template.jinja"
    check_hash(template, config["tokenizer"]["template_sha256"])
    rows = serving_rows(bundle["smoke_cases"], bundle["smoke_tokens"])
    require(
        [row["case"]["id"] for row in rows] == config["training"]["sample_order"],
        "Use the exact eight trained inputs in their registered order.",
    )
    return config, receipt, rows, template


def validate_native_smoke(values, rows):
    require(
        len(values) == len(rows) == 8
        and [value["case_id"] for value in values]
        == [row["case"]["id"] for row in rows],
        "Native smoke must retain the exact eight trained cases.",
    )
    for value, row in zip(values, rows, strict=True):
        ids = value["output_ids"]
        require(
            value["prompt_ids_sha256"] == prompt_digest(row["prefix"])
            and isinstance(ids, list)
            and 0 < len(ids) <= 512
            and all(type(token) is int and token >= 0 for token in ids)
            and isinstance(value["text"], str)
            and isinstance(value["text_raw"], str)
            and value["terminal_token_id"] == ids[-1]
            and value["finish_reason"]
            == ("stop" if ids[-1] in GENERATION_STOP_IDS else "length"),
            "Native smoke lost its raw tokens, registered prefix or terminal behavior.",
        )


def validate_reload(config, parent, input_path, train_path, training, native):
    reload_path, manifest = read_bound(config["fresh_reload"], parent)
    root = reload_path.parent

    def bound(directory, evidence, name):
        return read_bound(
            {"path": name, "sha256": evidence["files_sha256"][name]}, directory
        )

    _, reloaded = bound(root, manifest, "run.json")
    _, previous = read_bound(
        {"path": "manifest.json", "sha256": reloaded["previous_manifest_sha256"]},
        train_path.parent,
    )
    require(
        previous["version"]
        == manifest["version"]
        == training["version"]
        == reloaded["version"]
        == trainer.VERSION
        and previous["status"]
        == training["status"]
        == "training_complete_fresh_reload_pending"
        and manifest["status"]
        == reloaded["status"]
        == "fresh_reload_passed_local_export_pending"
        and previous["files_sha256"]["run.json"] == sha256(train_path)
        and training["mode"] == "train"
        and training["optimizer_state_resumed"] is False
        and reloaded["mode"] == "fresh_reload"
        and reloaded["optimizer_state_resumed"] is False
        and reloaded["phase"] == "smoke"
        and reloaded["reloaded_training_updates"] == 2
        and reloaded["configuration_sha256"]
        == training["configuration_sha256"]
        == sha256(input_path)
        and reloaded["pid"] != training["pid"],
        "A separate fresh reload of this exact two-update adapter is required.",
    )
    expected = json.loads(input_path.read_text())
    for directory, evidence in ((train_path.parent, previous), (root, manifest)):
        require(
            bound(directory, evidence, "configuration.json")[1] == expected,
            "Saved run/reload configuration differs.",
        )
        for name, key, _ in trainer.SOURCE_MODULES:
            source = resolve(name, directory)
            check_hash(source, evidence["files_sha256"][name])
            check_hash(source, expected[key])
    require(
        bound(train_path.parent, previous, "smoke-generation.json")[1] == native
        and bound(root, manifest, "smoke-generation.json")[1] == native,
        "The saved native output must equal fresh reload output exactly.",
    )
    _, parity = bound(root, manifest, "reload-parity.json")
    require(
        parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and [value["case_id"] for value in parity["cases"]]
        == [value["case_id"] for value in native]
        and all(value["identical"] is True for value in parity["cases"]),
        "Fresh reload lacks exact adapter and eight-output parity.",
    )
    return reload_path


def diagnostic_body(row, identity, prepared, constrained):
    return {
        **shared.make_body(row, identity, prepared, constrained),
        "verbose": True,
        "return_tokens": True,
    }


def isolated_runner():
    spec = importlib.util.spec_from_file_location(
        "_gemma_smoke_runtime", shared.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.make_body = diagnostic_body
    return module


def qualify_local(value):
    result = dict(value)
    if result.get("request_failed"):
        result.update(
            raw_valid=False,
            displayed=False,
            contract_valid=False,
            raw_grade=None,
            displayed_grade=None,
        )
        return result
    try:
        response = result["response"]
        raw = response["__verbose"]
        ids = raw["tokens"]
        require(
            isinstance(ids, list)
            and 0 < len(ids) <= 512
            and all(type(token) is int and token >= 0 for token in ids)
            and len(ids) == response["usage"]["completion_tokens"]
            and raw["stop_type"]
            == ("eos" if ids[-1] in GENERATION_STOP_IDS else "limit"),
            "Local smoke must retain exact sampled token IDs and the native terminal/length stop.",
        )
        result.update(
            output_ids=ids, terminal_token_id=ids[-1], generated_text_raw=raw["content"]
        )
    except (KeyError, TypeError, ValueError) as error:
        result.update(
            request_failed=True,
            raw_valid=False,
            displayed=False,
            contract_valid=False,
            raw_grade=None,
            displayed_grade=None,
            token_evidence_error=str(error),
            error_type="token_evidence_invalid",
        )
        return result
    if ids[-1] == 50:
        result.update(
            raw_valid=False,
            displayed=False,
            contract_valid=False,
            raw_grade=None,
            displayed_grade=None,
            error_type="tool_handoff",
            error="Tool-response terminal is outside the passive edit contract.",
        )
    return result


def native_result(row, value):
    result = {
        **grade(row["case"], value["text"], value["finish_reason"]),
        "case_id": value["case_id"],
        "raw_text": value["text"],
        "text_raw": value["text_raw"],
        "finish_reason": value["finish_reason"],
        "output_ids": value["output_ids"],
        "terminal_token_id": value["terminal_token_id"],
    }
    if value["terminal_token_id"] == 50:
        result.update(
            raw_valid=False,
            displayed=False,
            contract_valid=False,
            raw_grade=None,
            displayed_grade=None,
            error_type="tool_handoff",
            error="Tool-response terminal is outside the passive edit contract.",
        )
    return result


def run_condition(prepared, condition, output):
    rows = isolated_runner().run_condition(prepared, condition, output)
    qualified = [qualify_local(row) for row in rows]
    with (output / "token-evidence.jsonl").open("x") as stream:
        for row in qualified:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_new(
        output / "token-evidence-receipt.json",
        {
            "status": "failed"
            if any(row.get("token_evidence_error") for row in qualified)
            else "passed",
            "results_sha256": sha256(output / "results.jsonl"),
            "token_evidence_sha256": sha256(output / "token-evidence.jsonl"),
            "observed_requests": len(qualified),
        },
    )
    require(
        not any(row.get("token_evidence_error") for row in qualified),
        "Exact local token evidence failed; retain raw outputs and preserve this failed attempt.",
    )
    return qualified


def prepare(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    require(config.get("version") == VERSION, "Unknown serving smoke configuration.")
    parent = config_path.parent
    input_path, _ = read_bound(config["training_input"], parent)
    training, preparation, rows, template = load_inputs(
        input_path, resolve(config["native_directory"], parent)
    )
    export_path, export = read_bound(config["export_receipt"], parent)
    require(
        export["version"] == "passive-gemma-sft-export-v1"
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
        and training_record["training_updates"] == 2
        and training_record["optimizer_committed_rows"] == 8
        and training_record["phase"] == "smoke"
        and training_record["status"] == "training_complete_fresh_reload_pending",
        "Export does not bind this completed two-update training run.",
    )
    native_path, native = read_bound(config["native_smoke"], parent)
    require(
        native_path.parent == training_record_path.parent
        and native_path.name == "smoke-generation.json",
        "Use generation results from the bound training record's directory.",
    )
    eos = 1
    validate_native_smoke(native, rows)
    reload_path = validate_reload(
        config, parent, input_path, training_record_path, training_record, native
    )
    tensor_path = export_path.parent / "tensor-verification.json"
    check_hash(tensor_path, export["tensor_verification_sha256"])
    tensor_receipt = json.loads(tensor_path.read_text())
    require(
        tensor_receipt["status"] == "passed"
        and tensor_receipt["eos_token_id"] == eos
        and tensor_receipt["native_generation_stop_ids"]
        == export["native_generation_stop_ids"]
        == GENERATION_STOP_IDS
        and export["native_pin"]["sha256"]
        == training["bindings"]["native_pin"]["sha256"]
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
        config["release_manifest"],
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
            reload_path.parent,
        ],
    }


def comparison(rows, native, conditions):
    def results_by_id(values):
        return {value["case_id"]: value for value in values}

    native_results = [
        native_result(row, value) for row, value in zip(rows, native, strict=True)
    ]
    all_results = {
        "colab-adapted-native-bf16": native_results,
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
        ("colab-adapted-native-bf16", "adapted-q4-k-m"),
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
        "scope": "Eight training states; engineering round trip and in-sample diagnostics only. Different quantization/runtime outputs need not be identical. Native and local token identities are retained from actual generation, not inferred from text. Tool-handoff terminals are uncredited.",
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
        "expected_generation_requests": 24,
        "sampling": prepared["sampling"],
        "eos_token_id": prepared["eos"],
        "native_generation_stop_ids": GENERATION_STOP_IDS,
        "execution_unavailable": prepared["execution_unavailable"],
        "execution_files_sha256": prepared["execution_files_sha256"],
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "probe_passive_gemma_smoke.py",
                "probe_passive_sft.py",
                "train_passive_gemma.py",
                "prepare_passive_gemma.py",
                "export_passive_gemma.py",
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
    with (output / "source-execution.jsonl").open("x") as stream:
        for row in prepared["rows"]:
            stream.write(
                json.dumps(
                    {
                        "case_id": row["case"]["id"],
                        "execution": row["case"].get("_execution"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
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
                    qualify_local(json.loads(line))
                    for line in path.read_text().splitlines()
                ]
        summary = comparison(prepared["rows"], prepared["native"], conditions)
        write_new(output / "summary.json", {"status": status, **summary})
        write_new(
            output / "manifest.json",
            {
                "version": VERSION,
                "status": status,
                "config_sha256": sha256(config_path),
                "files_sha256": {
                    str(path.relative_to(output)): sha256(path)
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                },
            },
        )


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
