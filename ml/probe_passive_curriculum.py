"""Evaluate one native-derived SFT pair against the unchanged exposed B2 cohort.

This bridge reuses B2's complete input/prefix validation and single-server runner.
It never changes a B2 registration, retrieves test bodies, or selects a checkpoint.
The new protocol binds historical comparators; the execution binds one export pair.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import passive_b2_benchmark as b2
import probe_passive as probe
import probe_passive_b2 as original
from export_passive_sft import BASE_REPOSITORY, BASE_REVISION, RUNTIME_REVISION
from passive_benchmark import evaluate_result
from passive_contract import apply_passive_edit

ROOT = probe.ROOT
VERSION = "passive-curriculum-evaluation-v1"
EXECUTION_VERSION = "passive-curriculum-execution-v1"
CONDITIONS = {
    "native-base-full": ("native_base", "full"),
    "native-base-icl": ("native_base", "icl"),
    "sft-full": ("sft", "full"),
    "sft-icl": ("sft", "icl"),
}
ARTIFACT_KEYS = {"native_base": "base_quantized", "sft": "merged_quantized"}
B2_BINDINGS = (
    "release_manifest",
    "protocol",
    "execution",
    "registry",
    "prefix_manifest",
)
IDENTITY_FIELDS = (
    "base_repository",
    "base_revision",
    "tokenizer_path",
    "tokenizer_files_sha256",
    "template_path",
    "template_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def project_path(value, parent=ROOT):
    path = (parent / value).resolve()
    require(
        path.is_relative_to(ROOT),
        f"Evaluation input/output escapes the project: {path}",
    )
    return path


def read_bound(binding, parent, inputs):
    path = project_path(binding["path"], parent)
    expected = binding["sha256"]
    require(probe.digest(path) == expected, f"Changed bound input: {path}")
    require(
        path not in inputs or inputs[path] == expected, f"Conflicting hashes: {path}"
    )
    inputs[path] = expected
    return path


def load_json(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_inputs(inputs):
    for path, expected in inputs.items():
        require(
            probe.digest(path) == expected,
            f"Bound input changed during evaluation: {path}",
        )


def native_eos(anchor, inputs):
    directory = project_path(anchor["tokenizer_path"])
    files = {
        name: load_json(
            read_bound(
                {"path": name, "sha256": anchor["tokenizer_files_sha256"][name]},
                directory,
                inputs,
            )
        )
        for name in ("tokenizer_config.json", "tokenizer.json")
    }
    token = files["tokenizer_config.json"]["eos_token"]
    content = token["content"] if isinstance(token, dict) else token
    matches = [
        row["id"]
        for row in files["tokenizer.json"]["added_tokens"]
        if row["content"] == content and row.get("special") is True
    ]
    require(
        len(matches) == 1 and type(matches[0]) is int,
        "The pinned tokenizer must identify exactly one EOS token.",
    )
    return matches[0]


def load_protocol(protocol_path):
    protocol_path = project_path(protocol_path)
    inputs = {protocol_path: probe.digest(protocol_path)}
    protocol = load_json(protocol_path)
    require(
        protocol.get("version") == VERSION
        and protocol.get("conditions") == list(CONDITIONS)
        and protocol.get("registered_requests") == 289
        and protocol.get("icl_eligible_requests") == 100
        and protocol.get("icl_excluded_requests") == 189
        and protocol.get("final_test_bodies_read") is False,
        "Register the unchanged full289 / common-ICL100 cohort and all four conditions.",
    )
    paths = {
        key: read_bound(protocol["b2"][key], protocol_path.parent, inputs)
        for key in B2_BINDINGS
    }
    paths["protocol_new"] = protocol_path
    old = load_json(paths["protocol"])
    require(
        old["release_manifest_sha256"] == inputs[paths["release_manifest"]],
        "The B2 release binding changed.",
    )
    entries = protocol.get("external_comparators", [])
    required = {
        ("qwen35-9b", "full"),
        ("qwen35-9b", "icl"),
        ("always_no_action", "full"),
        ("consistent_history_rule", "full"),
    }
    require(
        len({row["id"] for row in entries}) == len(entries)
        and not {row["id"] for row in entries} & set(CONDITIONS)
        and required <= {(row["candidate"], row["condition"]) for row in entries}
        and all(row["condition"] in ("full", "icl") for row in entries),
        "Bind distinct external B2 observations, including 9B full/ICL and both fixed baselines.",
    )
    for entry in entries:
        for key in ("manifest", "results"):
            read_bound(entry[key], protocol_path.parent, inputs)
    return {"protocol": protocol, "paths": paths, "inputs": inputs}


def load_registration(protocol_path, execution_path):
    registration = load_protocol(protocol_path)
    execution_path = project_path(execution_path)
    paths, inputs = registration["paths"], registration["inputs"]
    inputs[execution_path] = probe.digest(execution_path)
    execution = load_json(execution_path)
    require(
        execution.get("version") == EXECUTION_VERSION
        and execution.get("protocol_sha256") == inputs[paths["protocol_new"]],
        "Execution does not bind this curriculum evaluation protocol.",
    )
    paths["execution_new"] = execution_path
    for key in ("registry", "export_receipt", "training_configuration", "fresh_reload"):
        paths[f"new_{key}"] = read_bound(execution[key], execution_path.parent, inputs)
    registration["execution"] = execution
    return registration


def anchor_args(registration, condition):
    paths = registration["paths"]
    return SimpleNamespace(
        release=paths["release_manifest"].parent,
        registry=paths["registry"],
        protocol=paths["protocol"],
        execution=paths["execution"],
        prefix_manifest=paths["prefix_manifest"],
        candidate="qwen35-9b",
        condition=condition,
        reuse_full_run=None,
        unconstrained=False,
        sampling_config=None,
        diagnostic_protocol=None,
        timeout=original.POLICY["request_timeout_seconds"],
    )


def validate_training(registration, training_path, training):
    paths, inputs = registration["paths"], registration["inputs"]
    config_path = paths["new_training_configuration"]
    config = load_json(config_path)
    protocol_sha = inputs[paths["protocol_new"]]
    require(
        training.get("version") == config.get("version") == "passive-sft-curriculum-v1"
        and training.get("status") == "training_complete_fresh_reload_pending"
        and training.get("mode") == "train"
        and training.get("training_updates") == 203
        and training.get("optimizer_committed_rows") == 809
        and training.get("configuration_sha256") == inputs[config_path]
        and training.get("evaluation_protocol_sha256") == protocol_sha
        and training.get("final_test_bodies_read") == 0,
        "Evaluate only this protocol's final 203-update / 809-committed-draw curriculum run.",
    )
    plan, recipe = registration["protocol"]["training_plan"], config["training"]
    require(
        plan["candidate"] == config["candidate_id"] == "qwen35-9b"
        and plan["view"] == "json-curriculum"
        and plan["updates"] == recipe["max_updates"] == 203
        and plan["draw_count"]
        == recipe["draw_count"]
        == len(recipe["sample_order"])
        == 809
        and plan["sampling_seed"] == recipe["sampling_seed"] == "3407"
        and plan["model_seed"] == recipe["seed"] == 3407
        and recipe["fresh_native_initialization"] is True
        and config["base"]["repository"] == BASE_REPOSITORY
        and config["base"]["revision"] == BASE_REVISION,
        "The exported training recipe is outside the preregistered final-only experiment.",
    )
    bound_protocol = read_bound(
        config["bindings"]["evaluation_protocol"], config_path.parent, inputs
    )
    require(
        inputs[bound_protocol] == protocol_sha,
        "Training used another evaluation protocol.",
    )
    preparation_path = read_bound(
        config["bindings"]["preparation"], config_path.parent, inputs
    )
    preparation = load_json(preparation_path)
    require(
        preparation.get("version") == "passive-sft-curriculum-preparation-v1"
        and preparation.get("status") == "frozen_train_only_preparation"
        and preparation.get("dev_test_record_bodies_copied") == 0
        and preparation.get("release_manifest_sha256")
        == inputs[paths["release_manifest"]]
        and training.get("preparation_sha256") == inputs[preparation_path],
        "Training preparation differs from this frozen train-only release.",
    )
    for name, expected in preparation["files_sha256"].items():
        read_bound({"path": name, "sha256": expected}, preparation_path.parent, inputs)
    reload_manifest_path = paths["new_fresh_reload"]
    reload_manifest = load_json(reload_manifest_path)
    require(
        reload_manifest.get("version") == "passive-sft-curriculum-v1"
        and reload_manifest.get("status") == "fresh_reload_passed_local_export_pending",
        "A successful fresh reload is required before development evaluation.",
    )

    def run_file(directory, manifest, name):
        return read_bound(
            {"path": name, "sha256": manifest["files_sha256"][name]}, directory, inputs
        )

    reload_run = load_json(
        run_file(reload_manifest_path.parent, reload_manifest, "run.json")
    )
    training_manifest_path = read_bound(
        {"path": "manifest.json", "sha256": reload_run["previous_manifest_sha256"]},
        training_path.parent,
        inputs,
    )
    training_manifest = load_json(training_manifest_path)
    require(
        training_manifest.get("version") == "passive-sft-curriculum-v1"
        and training_manifest.get("status") == "training_complete_fresh_reload_pending"
        and run_file(training_path.parent, training_manifest, "run.json")
        == training_path
        and reload_run.get("status") == "fresh_reload_passed_local_export_pending"
        and reload_run.get("mode") == "fresh_reload"
        and reload_run.get("configuration_sha256") == inputs[config_path]
        and reload_run.get("evaluation_protocol_sha256") == protocol_sha
        and reload_run.get("reloaded_training_updates") == 203
        and reload_run.get("pid") != training.get("pid"),
        "Fresh reload does not refer to the same completed training/configuration in a separate process.",
    )
    for directory, manifest in (
        (training_path.parent, training_manifest),
        (reload_manifest_path.parent, reload_manifest),
    ):
        require(
            load_json(run_file(directory, manifest, "configuration.json")) == config,
            "Saved training/reload configuration differs from the bound input.",
        )
        for name, key in (
            ("trainer.py", "trainer_sha256"),
            ("train_passive_sft.py", "smoke_trainer_sha256"),
            ("prepare_passive_curriculum.py", "preparer_sha256"),
            ("prepare_passive_sft.py", "smoke_preparer_sha256"),
        ):
            source = run_file(directory, manifest, name)
            require(
                inputs[source] == config[key],
                "Training/reload source snapshot changed.",
            )
    exposure = load_json(
        run_file(training_path.parent, training_manifest, "exposure.json")
    )
    committed = exposure["phases"]["optimizer_committed"]
    require(
        exposure["completed_updates"] == 203
        and committed["rows"] == 809
        and exposure["optimizer_commit_uncertain"] is False
        and exposure["pending_backward_rows_not_committed"] == 0
        and committed["counts"]["case_id"] == dict(Counter(recipe["sample_order"]))
        and committed["sequence_tokens"] == recipe["max_total_sequence_tokens"]
        and committed["completion_tokens"] == recipe["max_total_completion_tokens"],
        "Observed optimizer-committed exposure differs from the fixed final draw stream.",
    )
    parity = load_json(
        run_file(reload_manifest_path.parent, reload_manifest, "reload-parity.json")
    )
    require(
        parity.get("adapter_tensors_identical") is True
        and len(parity["cases"])
        == len({row["case_id"] for row in parity["cases"]})
        == 8
        and all(row["identical"] is True for row in parity["cases"]),
        "Fresh reload lacks exact adapter and eight-output parity.",
    )
    before = run_file(training_path.parent, training_manifest, "smoke-generation.json")
    after = run_file(
        reload_manifest_path.parent, reload_manifest, "smoke-generation.json"
    )
    require(
        load_json(before) == load_json(after)
        and [row["case_id"] for row in load_json(before)]
        == [row["case_id"] for row in parity["cases"]],
        "Fresh reload generation evidence differs from the saved training outputs.",
    )
    return config


def validate_pair(registration):
    paths, inputs = registration["paths"], registration["inputs"]
    registry = load_json(paths["new_registry"])
    previous = load_json(paths["registry"])
    anchor = next(row for row in previous["candidates"] if row["id"] == "qwen35-9b")
    candidates = registration["execution"]["candidate_ids"]
    require(
        set(candidates) == set(ARTIFACT_KEYS)
        and len(set(candidates.values())) == 2
        and not set(candidates.values()) & set(probe.BASELINES),
        "Bind two distinct native_base / sft candidate IDs.",
    )
    models = {row["id"]: row for row in registry["candidates"]}
    require(
        len(registry["candidates"]) == 2 and set(models) == set(candidates.values()),
        "Use a new registry containing exactly the two exported candidates.",
    )
    require(
        registry["runtime"] == previous["runtime"]
        and registry["tokenizers"] == previous["tokenizers"]
        and probe.sampling_options(paths["new_registry"])
        == probe.sampling_options(paths["registry"]),
        "Preserve B2's exact runtime, tokenizer registry and greedy sampling.",
    )
    receipt_path = paths["new_export_receipt"]
    receipt = load_json(receipt_path)
    require(
        receipt.get("version") == "passive-sft-export-v1"
        and receipt.get("status")
        == "tensor_verified_paired_export_runtime_preflight_pending"
        and receipt.get("base_repository") == BASE_REPOSITORY
        and receipt.get("base_revision") == BASE_REVISION
        and receipt.get("runtime_revision") == RUNTIME_REVISION
        and registry["runtime"]["revision"] == RUNTIME_REVISION,
        "Use the exact native 9B / b10816 verified paired export receipt.",
    )
    tensor_path = read_bound(
        {
            "path": "tensor-verification.json",
            "sha256": receipt["tensor_verification_sha256"],
        },
        receipt_path.parent,
        inputs,
    )
    tensor = load_json(tensor_path)
    require(
        tensor.get("status") == "passed"
        and tensor.get("template_sha256") == anchor["template_sha256"],
        "Export tensor/native-template verification differs.",
    )
    training_path = read_bound(receipt["training_record"], receipt_path.parent, inputs)
    training = load_json(training_path)
    if "configuration_sha256" not in training:
        training_path = read_bound(
            {"path": "run.json", "sha256": training["files_sha256"]["run.json"]},
            training_path.parent,
            inputs,
        )
        training = load_json(training_path)
    training_config = validate_training(registration, training_path, training)
    require(
        tensor.get("eos_token_id")
        == training_config["generation"]["eos_token_id"]
        == native_eos(anchor, inputs),
        "Export EOS differs from the registered native tokenizer.",
    )
    for role, identity in candidates.items():
        model = models[identity]
        require(
            model.get("tokenizer_id") == "qwen35-9b"
            and all(model.get(key) == anchor[key] for key in IDENTITY_FIELDS),
            "An exported candidate changed the native tokenizer/template identity.",
        )
        artifact = receipt["artifacts"][ARTIFACT_KEYS[role]]
        expected_path = project_path(artifact["path"], receipt_path.parent)
        require(
            project_path(model["model_path"]) == expected_path
            and model["model_sha256"] == artifact["sha256"],
            "Candidate weights are not the corresponding member of this export pair.",
        )
        # The original runtime inspector rehashes the model, server, libraries and native files.
        probe.inspect_runtime(paths["new_registry"], identity)
    registration.update(registry=registry, export=receipt, training=training)


def prepare(protocol_path, execution_path, condition):
    require(condition in CONDITIONS, "Unknown registered curriculum condition.")
    registration = load_registration(protocol_path, execution_path)
    role, original_condition = CONDITIONS[condition]
    args = anchor_args(registration, original_condition)
    prepared = original.prepare(args)
    manifest = load_json(args.prefix_manifest)
    eligibility_path = read_bound(
        {
            "path": manifest["common_icl_eligibility_file"],
            "sha256": manifest["common_icl_eligibility_sha256"],
        },
        args.prefix_manifest.parent,
        registration["inputs"],
    )
    eligibility = load_json(eligibility_path)
    require(
        len(eligibility["eligible_case_ids"]) == 100
        and len(eligibility["excluded"]) == 189,
        "Keep the original all-five-tokenizer ICL100 / 189 exclusions.",
    )
    validate_pair(registration)
    require(
        len(prepared["cases"]) == (100 if original_condition == "icl" else 289),
        "Prepared condition has a different request denominator.",
    )
    source = Path(__file__).resolve()
    prepared["source_files"].append(source)
    prepared["binding"]["snapshot_sha256"][source.name] = probe.digest(source)
    binding = {
        "version": VERSION,
        "condition": condition,
        "model_role": role,
        "protocol_sha256": probe.digest(registration["paths"]["protocol_new"]),
        "execution_sha256": probe.digest(registration["paths"]["execution_new"]),
        "registry_sha256": probe.digest(registration["paths"]["new_registry"]),
        "export_receipt_sha256": probe.digest(
            registration["paths"]["new_export_receipt"]
        ),
        "training_configuration_sha256": probe.digest(
            registration["paths"]["new_training_configuration"]
        ),
        "fresh_reload_manifest_sha256": probe.digest(
            registration["paths"]["new_fresh_reload"]
        ),
        "paired_baseline": "native-derived exact export pair; public B2 GGUF is external only",
        "registered_requests": 289,
        "requested_cases": len(prepared["cases"]),
        "icl_eligible_requests": 100,
        "icl_excluded_requests": 189,
    }
    prepared["binding"]["curriculum"] = binding
    registration.update(
        prepared=prepared, binding=binding, eligibility=eligibility, args=args
    )
    registration["inputs"][source] = probe.digest(source)
    verify_inputs(registration["inputs"])
    return registration


def validate_results(cases, rows, prefixes, *, deterministic=False):
    ids = [case["id"] for case in cases]
    observed = [row["case_id"] for row in rows]
    require(
        len(set(observed)) == len(observed)
        and observed == [identity for identity in ids if identity in observed],
        "Results must preserve unique registered case order, including failures.",
    )
    indexed = {case["id"]: case for case in cases}
    for row in rows:
        failed = (
            row.get("request_failed")
            or row.get("cancelled")
            or row.get("error_type") in original.REQUEST_ERRORS
        )
        if failed:
            require(
                row.get("raw_valid") is False and row.get("displayed") is False,
                "A failed request cannot receive validity/display credit.",
            )
            continue
        text = row.get("raw_text")
        finish = row.get("finish_reason", "stop" if deterministic else None)
        require(
            isinstance(text, str) and isinstance(finish, str),
            "A result lost its raw completion.",
        )
        if not deterministic:
            response = row["response"]
            choice = response["choices"][0]
            require(
                text == (choice["message"].get("content") or "")
                and finish == choice["finish_reason"]
                and response["usage"]["prompt_tokens"] == len(prefixes[row["case_id"]]),
                "Saved completion differs from its response or frozen token prefix.",
            )
        graded = evaluate_result(indexed[row["case_id"]], text, finish)
        require(
            all(row.get(key) == graded.get(key) for key in original.GRADE_FIELDS),
            "Saved quality differs from frozen source execution/repeat gating.",
        )


def validate_runtime_manifest(
    manifest, registration, prepared, candidate, registry_path
):
    registry = load_json(registry_path)
    expected = {
        "version": "passive-b2-runtime-v1",
        "stage": "screen",
        "candidate": candidate,
        "case_ids": [case["id"] for case in prepared["cases"]],
        "registry_sha256": probe.digest(registry_path),
        "release_manifest_sha256": prepared["binding"]["release_manifest_sha256"],
        "schema_constrained": True,
        "context_tokens": 4096,
        "cache_requested": False,
        "request_timeout_seconds": 120,
        "sampling": probe.sampling_options(registry_path),
        "reused_results": 0,
    }
    require(
        all(manifest.get(key) == value for key, value in expected.items()),
        "Saved runtime changed the registered schema, sampling, cohort or artifact bindings.",
    )
    if candidate in probe.BASELINES:
        require(
            manifest.get("backend") == "deterministic_rule",
            "Invalid deterministic baseline backend.",
        )
        return
    model = next(row for row in registry["candidates"] if row["id"] == candidate)
    tokenizer_registry = project_path(registry["tokenizers"]["registry_path"])
    tokenizer = next(
        row
        for row in load_json(tokenizer_registry)["candidates"]
        if row["id"] == model.get("tokenizer_id", candidate)
    )
    if "runtime_binding" in manifest or manifest["status"] == "completed":
        require(
            manifest.get("candidate_binding") == model
            and manifest.get("runtime_binding") == registry["runtime"]
            and manifest.get("tokenizer_binding") == tokenizer,
            "Saved runtime model, server or native tokenizer identity differs.",
        )
    if manifest["status"] == "completed":
        require(
            manifest.get("template_checks")
            == {"requests": len(prepared["cases"]), "passed": len(prepared["cases"])},
            "Completed run lacks complete frozen-prefix parity.",
        )


def source_execution(cases, rows):
    indexed = {row["case_id"]: row for row in rows}
    for case in cases:
        row, evidence = indexed.get(case["id"], {}), case["_execution"]
        value = {
            "case_id": case["id"],
            "execution": evidence,
            "raw_grade": row.get("raw_grade"),
            "displayed_grade": row.get("displayed_grade"),
        }
        if row.get("raw_valid"):
            value["predicted_after"] = apply_passive_edit(
                evidence["before"], evidence["bundle"], row["prediction"]
            )
        yield value


def new_output(path, release):
    path = project_path(path)
    require(
        not path.is_relative_to(release)
        and not any(
            parent.name == "frozen"
            and parent.is_relative_to(ROOT / "artifacts/datasets")
            for parent in (path, *path.parents)
        ),
        "Output must be outside frozen releases.",
    )
    require(not path.exists(), "Output is write-once; choose a new directory.")
    return path


def snapshot_registration(registration, output):
    implementation = output / "implementation"
    implementation.mkdir(exist_ok=True)
    for key, name in (
        ("protocol_new", "curriculum-protocol.json"),
        ("execution_new", "curriculum-execution.json"),
        ("new_export_receipt", "curriculum-export-receipt.json"),
        ("new_training_configuration", "curriculum-training-configuration.json"),
        ("new_fresh_reload", "curriculum-reload-manifest.json"),
    ):
        path = registration["paths"][key]
        require(
            probe.digest(path) == registration["inputs"][path],
            "Registration changed before snapshot.",
        )
        with (implementation / name).open("xb") as stream:
            stream.write(path.read_bytes())


def run_condition(registration, condition, output, stage, editor_state):
    args, prepared = registration["args"], registration["prepared"]
    output = new_output(output, args.release)
    if stage == "prepare":
        output.mkdir(parents=True)
        snapshot_registration(registration, output)
        write_new(
            output / "curriculum-receipt.json",
            {
                **registration["binding"],
                "status": "prepared_generation_unrun",
                "case_ids": [case["id"] for case in prepared["cases"]],
                "common_icl_eligibility": registration["eligibility"],
            },
        )
        return
    args.candidate = registration["execution"]["candidate_ids"][
        CONDITIONS[condition][0]
    ]
    args.registry = registration["paths"]["new_registry"]
    args.output, args.stage, args.editor_state = output, stage, editor_state
    error = None
    try:
        probe.run(args, prepared=prepared)
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    if output.exists():
        try:
            verify_inputs(registration["inputs"])
            snapshot_registration(registration, output)
            results_path = output / "results.jsonl"
            rows = original.read_rows(results_path) if results_path.exists() else []
            validate_results(prepared["cases"], rows, prepared["prefixes"])
            with (output / "source-execution.jsonl").open("x") as stream:
                for value in source_execution(prepared["cases"], rows):
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            status = load_json(output / "manifest.json")["status"]
        except (Exception, KeyboardInterrupt) as caught:
            error = error or caught
            status = "failed"
        files = {
            path.name: probe.digest(path) for path in output.iterdir() if path.is_file()
        }
        write_new(
            output / "curriculum-receipt.json",
            {
                **registration["binding"],
                "status": "failed" if error else status,
                "error": f"{type(error).__name__}: {error}" if error else None,
                "files_sha256": files,
                "runtime_manifest_version": "passive-b2-runtime-v1 (unchanged shared runner)",
            },
        )
    if error:
        raise error


def historical_results(registration, full, icl):
    paths = registration["paths"]
    entries = registration["protocol"].get("external_comparators", [])
    require(
        len({row["id"] for row in entries}) == len(entries)
        and not {row["id"] for row in entries} & set(CONDITIONS),
        "External comparator IDs must be unique and separate from the paired runs.",
    )
    required = {
        ("qwen35-9b", "full"),
        ("qwen35-9b", "icl"),
        ("always_no_action", "full"),
        ("consistent_history_rule", "full"),
    }
    require(
        required <= {(row["candidate"], row["condition"]) for row in entries},
        "Bind original B2 9B full/ICL and both fixed deterministic baselines.",
    )
    reports, provenance = {}, {}
    for entry in entries:
        require(
            entry["condition"] in ("full", "icl"),
            "Only full/ICL external observations are registered.",
        )
        prepared = full if entry["condition"] == "full" else icl
        manifest_path = read_bound(
            entry["manifest"], paths["protocol_new"].parent, registration["inputs"]
        )
        results_path = read_bound(
            entry["results"], paths["protocol_new"].parent, registration["inputs"]
        )
        require(
            results_path.parent == manifest_path.parent,
            "External manifest and results must belong to one run.",
        )
        manifest = load_json(manifest_path)
        binding = manifest.get("benchmark", {})
        validate_runtime_manifest(
            manifest, registration, prepared, entry["candidate"], paths["registry"]
        )
        require(
            manifest.get("version") == "passive-b2-runtime-v1"
            and manifest.get("status") == "completed"
            and manifest.get("candidate") == entry["candidate"]
            and manifest.get("condition") == entry["condition"]
            and manifest.get("registry_sha256") == probe.digest(paths["registry"])
            and manifest.get("case_ids") == [case["id"] for case in prepared["cases"]]
            and all(
                binding.get(key) == prepared["binding"][key]
                for key in (
                    "protocol_sha256",
                    "execution_sha256",
                    "prefix_manifest_sha256",
                    "registry_sha256",
                    "release_manifest_sha256",
                    "case_registry_sha256",
                    "requests_sha256",
                    "registered_case_ids",
                    "messages_sha256",
                    "common_icl_eligibility_sha256",
                )
            ),
            "External results do not bind the same frozen B2 condition.",
        )
        for name, expected in binding["snapshot_sha256"].items():
            read_bound(
                {"path": name, "sha256": expected},
                manifest_path.parent / "implementation",
                registration["inputs"],
            )
        read_bound(
            {"path": "model-registry.json", "sha256": binding["registry_sha256"]},
            manifest_path.parent / "implementation",
            registration["inputs"],
        )
        rows = original.read_rows(results_path)
        original.unique_rows(
            rows, manifest["case_ids"], "Completed B2 external results"
        )
        require(
            manifest.get("completed_requests") == len(rows),
            "External completion count differs.",
        )
        deterministic = entry["candidate"] in probe.BASELINES
        # Other B2 models have different token counts; their saved native checks bind those prefixes.
        if deterministic or entry["candidate"] == "qwen35-9b":
            prefixes = prepared["prefixes"]
        else:
            tokens = original.read_rows(
                paths["prefix_manifest"].parent
                / entry["candidate"]
                / f"{entry['condition']}.jsonl.gz"
            )
            prefixes = {row["case_id"]: row["input_ids"] for row in tokens}
        validate_results(prepared["cases"], rows, prefixes, deterministic=deterministic)
        reports[entry["id"]] = prepared["summarize"](prepared["cases"], rows)
        provenance[entry["id"]] = {
            **entry,
            "comparison_role": "external B2 observation; not the native-derived paired control",
        }
    return reports, provenance


def learning_signal(base, adapted):
    precision = (
        adapted["displayed_useful_precision"] >= base["displayed_useful_precision"]
        if adapted["displayed_useful_precision"] is not None
        and base["displayed_useful_precision"] is not None
        else None
    )
    checks = {
        "useful_edits_increased": adapted["displayed_useful_edits"]
        > base["displayed_useful_edits"],
        "shown_precision_not_lower": precision,
        "correct_abstentions_not_lower": adapted["correct_abstentions"]
        >= base["correct_abstentions"],
        "invalid_results_not_higher": adapted["invalid_results"]
        <= base["invalid_results"],
    }
    complete = not (base["missing_requests"] or adapted["missing_requests"]) and all(
        summary.get("run_status", "completed") == "completed"
        for summary in (base, adapted)
    )
    return {
        "classification": "incomplete"
        if not complete
        else "descriptive_learning_signal"
        if all(value is True for value in checks.values())
        else "criteria_not_met",
        "checks": checks,
        "complete_cohort": complete,
        "scope": "Exposed development description only; no deployment promotion or automatic extra training.",
    }


def correct_preserves(cases, rows, eligibility=None):
    qualified = b2.input_qualified_cases(cases, eligibility)
    indexed = {row["case_id"]: row for row in rows}
    return sum(
        bool(b2.b1._references(case))
        and all(value["action"] == "no_action" for value in b2.b1._references(case))
        and indexed.get(case["id"], {}).get("raw_valid") is True
        and (indexed.get(case["id"], {}).get("prediction") or {}).get("action")
        == "no_action"
        for case in qualified
    )


def report(protocol_path, execution_path, runs, output):
    registration = prepare(protocol_path, execution_path, "native-base-full")
    full = registration["prepared"]
    icl = original.prepare(anchor_args(registration, "icl"))
    runs = project_path(runs)
    output = new_output(output, registration["args"].release)
    summaries, results, provenance = {}, {}, {}
    for condition, (role, view) in CONDITIONS.items():
        prepared = full if view == "full" else icl
        directory = runs / condition
        receipt_path, manifest_path = (
            directory / "curriculum-receipt.json",
            directory / "manifest.json",
        )
        rows = []
        status = "missing_run"
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            expected = {
                **registration["binding"],
                "condition": condition,
                "model_role": role,
                "requested_cases": len(prepared["cases"]),
            }
            require(
                all(receipt.get(key) == value for key, value in expected.items()),
                "Run receipt belongs to another protocol/export pair/condition.",
            )
            require(
                "files_sha256" in receipt,
                "A preparation-only receipt is not a completed evaluation run.",
            )
            for name, expected_hash in receipt["files_sha256"].items():
                read_bound(
                    {"path": name, "sha256": expected_hash},
                    directory,
                    registration["inputs"],
                )
            manifest = load_json(manifest_path)
            validate_runtime_manifest(
                manifest,
                registration,
                prepared,
                registration["execution"]["candidate_ids"][role],
                registration["paths"]["new_registry"],
            )
            require(
                manifest.get("benchmark", {}).get("curriculum") == expected
                and manifest.get("candidate")
                == registration["execution"]["candidate_ids"][role]
                and manifest.get("case_ids")
                == [case["id"] for case in prepared["cases"]]
                and manifest.get("stage") == "screen",
                "Saved runtime condition differs from the registered evaluation.",
            )
            for name, expected_hash in manifest["benchmark"]["snapshot_sha256"].items():
                read_bound(
                    {"path": name, "sha256": expected_hash},
                    directory / "implementation",
                    registration["inputs"],
                )
            read_bound(
                {
                    "path": "model-registry.json",
                    "sha256": registration["binding"]["registry_sha256"],
                },
                directory / "implementation",
                registration["inputs"],
            )
            results_path = directory / "results.jsonl"
            if results_path.exists():
                require(
                    "results.jsonl" in receipt["files_sha256"], "Unbound result file."
                )
                rows = original.read_rows(results_path)
            require(
                manifest.get("completed_requests") == len(rows),
                "Recorded completion count differs.",
            )
            status = receipt["status"]
            provenance[condition] = {
                "receipt_sha256": probe.digest(receipt_path),
                "status": status,
            }
        elif directory.exists():
            raise ValueError(f"Incomplete run has no immutable receipt: {directory}")
        validate_results(prepared["cases"], rows, prepared["prefixes"])
        summaries[condition] = prepared["summarize"](prepared["cases"], rows)
        summaries[condition]["run_status"] = status
        summaries[condition]["correct_preserves"] = correct_preserves(
            prepared["cases"], rows
        )
        results[condition] = rows
    external, external_provenance = historical_results(registration, full, icl)
    pairs = {}
    for view in ("full", "icl"):
        left, right = f"native-base-{view}", f"sft-{view}"
        eligibility = registration["eligibility"] if view == "icl" else None
        pairs[view] = {
            "learning_signal": learning_signal(summaries[left], summaries[right]),
            "source_paired_comparison": b2.compare_runs(
                full["cases"],
                {left: results[left], right: results[right]},
                eligibility_by_name={left: eligibility, right: eligibility},
            ),
        }
    eligible = set(registration["eligibility"]["eligible_case_ids"])
    icl_effects = {}
    for role in ("native-base", "sft"):
        names = (f"{role}-full", f"{role}-icl")
        icl_effects[role] = b2.compare_runs(
            full["cases"],
            {
                name: [row for row in results[name] if row["case_id"] in eligible]
                for name in names
            },
            eligibility_by_name={name: registration["eligibility"] for name in names},
        )
    verify_inputs(registration["inputs"])
    output.mkdir(parents=True)
    snapshot_registration(registration, output)
    write_new(
        output / "summary.json",
        {
            "version": VERSION,
            "protocol_sha256": registration["binding"]["protocol_sha256"],
            "execution_sha256": registration["binding"]["execution_sha256"],
            "registered_new_requests": 778,
            "registered_conditions": list(CONDITIONS),
            "runs": summaries,
            "paired": pairs,
            "icl_effect_on_common100": icl_effects,
            "icl_comparison_scope": "Existing full/ICL outputs compared only on the common100; both keep189 exclusions in the289 registration. No additional requests.",
            "external_comparators": external,
            "provenance": provenance,
            "external_provenance": external_provenance,
            "common_icl_eligibility": registration["eligibility"],
            "scope": "Frozen exposed development; correlated synthetic/source states; no final-test access or deployment promotion.",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run", "report"), default="run")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "preflight", "screen"), default="screen"
    )
    parser.add_argument("--runs", type=Path)
    parser.add_argument(
        "--editor-state", default="Not controlled; no editor-preview latency claim"
    )
    args = parser.parse_args(argv)
    if args.command == "report":
        require(
            args.runs is not None and args.condition is None,
            "Report requires --runs, without --condition.",
        )
        report(args.protocol, args.execution, args.runs, args.output)
        return
    require(
        args.condition is not None and args.runs is None,
        "Run requires --condition, without --runs.",
    )
    lock_path = ROOT / "artifacts/nep-passive-b1.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another passive benchmark owns the local predictor."
            ) from error
        registration = prepare(args.protocol, args.execution, args.condition)
        run_condition(
            registration, args.condition, args.output, args.stage, args.editor_state
        )


if __name__ == "__main__":
    main()
