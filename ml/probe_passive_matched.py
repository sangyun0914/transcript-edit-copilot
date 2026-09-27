"""Run the fixed matched JSON/Rewrite comparison on exposed development only."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import format_comparison
import passive_b2_benchmark as b2
import passive_formats_v2 as formats
import probe_passive as probe
import probe_passive_b2 as original
import probe_passive_curriculum as curriculum
from export_passive_sft import BASE_REPOSITORY, BASE_REVISION, RUNTIME_REVISION
from passive_benchmark import evaluate_result
from replay_session import session_hash

ROOT = probe.ROOT
VERSION = "passive-matched-evaluation-v1"
EXECUTION_VERSION = "passive-matched-execution-v1"
CONDITIONS = {
    "matched-json-unconstrained": ("json", False),
    "matched-rewrite-unconstrained": ("rewrite", False),
    "matched-json-constrained": ("json", True),
}
SOURCES = {
    "trainer.py": "trainer_sha256",
    "train_passive_sft.py": "smoke_trainer_sha256",
    "train_passive_curriculum.py": "curriculum_trainer_sha256",
    "prepare_passive_matched.py": "preparer_sha256",
    "prepare_passive_curriculum.py": "curriculum_preparer_sha256",
    "prepare_passive_sft.py": "smoke_preparer_sha256",
}
require = curriculum.require
project_path = curriculum.project_path
read_bound = curriculum.read_bound
load_json = curriculum.load_json
write_new = curriculum.write_new


def bound_json(binding, parent, inputs):
    path = read_bound(binding, parent, inputs)
    return path, load_json(path)


def load_protocol(path):
    path = project_path(path)
    protocol = load_json(path)
    inputs = {path: probe.digest(path)}
    require(
        protocol.get("version") == VERSION
        and protocol.get("conditions") == list(CONDITIONS)
        and protocol.get("formats") == ["json", "rewrite"]
        and protocol.get("registered_requests_per_condition") == 286
        and protocol.get("full_dev_registered_requests") == 289
        and protocol.get("excluded_from_matched_comparison") == 3
        and protocol.get("training_rows") == 802
        and protocol.get("training_exclusions") == 7
        and protocol.get("final_test_bodies_read") is False,
        "Use the fixed three-condition matched286 experiment with no final-test access.",
    )
    plan = protocol["training_plan"]
    require(
        plan["updates"] == 201
        and plan["draw_count"] == 802
        and plan["effective_batch"] == 4
        and plan["final_accumulation"] == 2
        and plan["checkpoint_selection"] == "final_update_201_only"
        and plan["independent_fresh_native_runs"] is True
        and protocol["same_case_sampling_order_required"] is True
        and protocol["generation"]["max_new_tokens"] == 512
        and protocol["generation"]["temperature"] == 0
        and protocol["generation"]["context_limit"] == 4096
        and protocol["generation"]["input_limit"] == 3584
        and protocol["generation"]["truncation"] is False,
        "802 draws / accumulation four require 201 updates and a final two-row mean.",
    )
    t1_path = read_bound(protocol["t1_protocol"], path.parent, inputs)
    t1 = curriculum.load_protocol(t1_path)
    inputs.update(t1["inputs"])
    release_path = read_bound(protocol["b2"]["release_manifest"], path.parent, inputs)
    require(
        inputs[release_path] == probe.digest(t1["paths"]["release_manifest"]),
        "T1 and T2 must use the same frozen release.",
    )
    return {
        "protocol": protocol,
        "protocol_path": path,
        "inputs": inputs,
        "release": release_path.parent,
        "previous_registry": t1["paths"]["registry"],
        "b2_protocol": t1["paths"]["protocol"],
    }


def load_views(registration, *, execution=True):
    release, inputs = registration["release"], registration["inputs"]
    manifest = load_json(release / "manifest.json")
    probe.code_bindings(release)

    def payload(name):
        return read_bound(
            {"path": name, "sha256": manifest["files_sha256"][name]}, release, inputs
        )

    full, _ = b2._load_split(release, "dev")
    full_ids = [case["id"] for case in full]
    require(
        len(full_ids) == 289
        and full_ids == load_json(registration["b2_protocol"])["case_ids"],
        "The full development roster differs from frozen B2.",
    )
    cases = original.read_rows(payload("matched/dev-cases.jsonl"))
    ids = [case["id"] for case in cases]
    require(
        len(ids) == len(set(ids)) == 286
        and ids == [identity for identity in full_ids if identity in ids],
        "Matched cases must retain the ordered 286-case subset of full289.",
    )
    full_by_id = {case["id"]: case for case in full}
    for case in cases:
        original_case = copy.deepcopy(full_by_id[case["id"]])
        original_case["metadata"]["sampling_weight"] = case["metadata"][
            "sampling_weight"
        ]
        require(
            case == original_case,
            "Matched evidence/reference/source changed beyond frozen sampling weights.",
        )
    eligibility = load_json(payload("matched/eligibility.json"))
    excluded = [row for row in eligibility["excluded"] if row["case_id"] in full_by_id]
    require(
        eligibility["qualified"] == {"train": 802, "dev": 286}
        and eligibility["input_limit"] == 3584
        and eligibility["output_limit_each_format"] == 512
        and len(excluded) == 3
        and {row["case_id"] for row in excluded} == set(full_ids) - set(ids),
        "Retain the registered three development exclusions without recropping.",
    )
    registry = load_json(registration["previous_registry"])
    anchor = next(row for row in registry["candidates"] if row["id"] == "qwen35-9b")
    native = load_json(project_path(anchor["tokenizer_path"]) / "config.json")
    native = native.get("text_config", native)
    eos = curriculum.native_eos(anchor, inputs)
    views = {}
    full_prefixes = probe.frozen_prefixes(release, "qwen35-9b")
    for fmt in ("json", "rewrite"):
        prompt = payload(f"prompt-{fmt}.txt").read_text().removesuffix("\n")
        tokens = original.read_rows(payload(f"tokenized/qwen35-9b/{fmt}-dev.jsonl.gz"))
        require(
            [row["id"] for row in tokens] == ids,
            "Format token rows changed the matched case order.",
        )
        messages, prefixes = {}, {}
        for case, row in zip(cases, tokens, strict=True):
            encoded = row["primary"]
            values, start = encoded["input_ids"], encoded["completion_start"]
            require(
                row["split"] == "dev"
                and row["input_sha256"] == case["metadata"]["input_sha256"]
                and row["sampling_weight"] == case["metadata"]["sampling_weight"]
                and row["messages"] == formats.messages(case["input"], fmt)
                and row["messages"][0] == {"role": "system", "content": prompt}
                and type(start) is int
                and 0 < start <= 3584
                and 0 < len(values) - start <= 512
                and len(values) <= 4096
                and all(
                    type(value) is int and 0 <= value < native["vocab_size"]
                    for value in values
                )
                and values[-1] == eos
                and encoded["labels"] == [-100] * start + values[start:]
                and encoded["attention_mask"] == [1] * len(values),
                "Frozen format messages, token bounds, masks or EOS differ.",
            )
            references = [case["expectedOutput"], *case["acceptableOutputs"]]
            require(
                row["references"]
                == [
                    formats.serialize(case["input"], action, fmt)
                    for action in references
                ]
                and [
                    formats.parse(text, case["input"], fmt).model_dump()
                    for text in row["references"]
                ]
                == references,
                "Frozen format references do not reconstruct the same admitted actions.",
            )
            messages[case["id"]], prefixes[case["id"]] = row["messages"], values[:start]
            if fmt == "json":
                require(
                    prefixes[case["id"]] == full_prefixes[case["id"]],
                    "JSON matched prefix differs from B2 full input.",
                )
            else:
                require(
                    messages[case["id"]][1] == views["json"]["messages"][case["id"]][1],
                    "JSON and Rewrite changed the user evidence window.",
                )
        views[fmt] = {"messages": messages, "prefixes": prefixes}
    if execution:
        cases = [b2._bind_execution(case, manifest) for case in cases]
    registration.update(
        cases=cases,
        views=views,
        exclusions=excluded,
        full_dev_case_ids=full_ids,
        anchor=anchor,
        native_eos=eos,
    )
    return registration


def validate_arm(registration, fmt, arm):
    inputs, parent = registration["inputs"], registration["execution_path"].parent
    config_path, config = bound_json(arm["training_configuration"], parent, inputs)
    export_path, export = bound_json(arm["export_receipt"], parent, inputs)
    reload_path, reload_manifest = bound_json(arm["fresh_reload"], parent, inputs)
    run_path, run = bound_json(export["training_record"], export_path.parent, inputs)
    if "configuration_sha256" not in run:
        run_path, run = bound_json(
            {"path": "run.json", "sha256": run["files_sha256"]["run.json"]},
            run_path.parent,
            inputs,
        )
    require(
        config["version"]
        == run["version"]
        == reload_manifest["version"]
        == "passive-sft-matched-v1"
        and config["format"] == run["format"] == fmt
        and run["status"] == "training_complete_fresh_reload_pending"
        and run["mode"] == "train"
        and run["training_updates"] == 201
        and run["optimizer_committed_rows"] == 802
        and run["final_test_bodies_read"] == 0
        and run["configuration_sha256"] == inputs[config_path]
        and run["evaluation_protocol_sha256"] == inputs[registration["protocol_path"]]
        and reload_manifest["status"] == "fresh_reload_passed_local_export_pending",
        "Both matched arms require their final 201-update / 802-draw run and successful fresh reload.",
    )
    recipe = config["training"]
    require(
        recipe["max_updates"] == 201
        and recipe["draw_count"] == len(recipe["sample_order"]) == 802
        and recipe["sampling_seed"] == "3407"
        and recipe["seed"] == 3407
        and recipe["micro_batch_size"] == 1
        and recipe["gradient_accumulation"] == 4
        and recipe["fresh_native_initialization"] is True
        and recipe["final_partial_accumulation"] == "actual_microbatch_count"
        and config["base"]["repository"] == BASE_REPOSITORY
        and config["base"]["revision"] == BASE_REVISION,
        "Matched training differs from the fixed fresh-native recipe.",
    )
    protocol_path = read_bound(
        config["bindings"]["evaluation_protocol"], config_path.parent, inputs
    )
    require(
        inputs[protocol_path] == inputs[registration["protocol_path"]],
        "Training used a different T2 protocol.",
    )
    prep_path, prep = bound_json(
        config["bindings"]["preparation"], config_path.parent, inputs
    )
    require(
        prep["version"] == "passive-sft-matched-preparation-v1"
        and prep["status"] == "frozen_matched_train_only_preparation"
        and prep["dev_test_record_bodies_copied"] == 0
        and prep["release_manifest_sha256"]
        == probe.digest(registration["release"] / "manifest.json")
        and run["preparation_sha256"] == inputs[prep_path],
        "Matched preparation changed or includes held-out bodies.",
    )
    for name, expected in prep["files_sha256"].items():
        read_bound({"path": name, "sha256": expected}, prep_path.parent, inputs)
    for name, expected in run["t1_launch_evidence_sha256"].items():
        path = read_bound(config["bindings"][name], config_path.parent, inputs)
        require(inputs[path] == expected, "Matched run changed its T1 launch evidence.")
    local_path, local = bound_json(
        config["bindings"]["t1_local_preflight"], config_path.parent, inputs
    )
    require(
        local["version"] == curriculum.VERSION
        and local["condition"] == "sft-full"
        and local["status"] == "template_preflight_passed_generation_unrun"
        and local["error"] is None
        and local["requested_cases"] == 289
        and local["protocol_sha256"]
        == registration["protocol"]["t1_protocol"]["sha256"]
        and run["t1_local_gate_status"] == local["status"],
        "Matched training lacks the preregistered T1 engineering preflight.",
    )

    def file(directory, manifest, name):
        return read_bound(
            {"path": name, "sha256": manifest["files_sha256"][name]}, directory, inputs
        )

    reload_run = load_json(file(reload_path.parent, reload_manifest, "run.json"))
    train_manifest_path, train_manifest = bound_json(
        {"path": "manifest.json", "sha256": reload_run["previous_manifest_sha256"]},
        run_path.parent,
        inputs,
    )
    require(
        train_manifest["status"] == "training_complete_fresh_reload_pending"
        and file(run_path.parent, train_manifest, "run.json") == run_path
        and reload_run["status"] == "fresh_reload_passed_local_export_pending"
        and reload_run["format"] == fmt
        and reload_run["mode"] == "fresh_reload"
        and reload_run["configuration_sha256"] == inputs[config_path]
        and reload_run["evaluation_protocol_sha256"]
        == inputs[registration["protocol_path"]]
        and reload_run["reloaded_training_updates"] == 201
        and reload_run["pid"] != run["pid"],
        "Fresh reload belongs to a different matched arm, checkpoint or process.",
    )
    for directory, manifest in (
        (run_path.parent, train_manifest),
        (reload_path.parent, reload_manifest),
    ):
        require(
            load_json(file(directory, manifest, "configuration.json")) == config,
            "Saved matched configuration differs.",
        )
        for name, key in SOURCES.items():
            require(
                inputs[file(directory, manifest, name)] == config[key],
                "Matched source snapshot changed.",
            )
    exposure = load_json(file(run_path.parent, train_manifest, "exposure.json"))
    committed = exposure["phases"]["optimizer_committed"]
    require(
        exposure["completed_updates"] == 201
        and committed["rows"] == 802
        and exposure["optimizer_commit_uncertain"] is False
        and exposure["pending_backward_rows_not_committed"] == 0
        and committed["counts"]["case_id"] == dict(Counter(recipe["sample_order"]))
        and committed["sequence_tokens"] == recipe["max_total_sequence_tokens"]
        and committed["completion_tokens"] == recipe["max_total_completion_tokens"],
        "Committed matched exposure differs from the registered draw stream.",
    )
    parity = load_json(file(reload_path.parent, reload_manifest, "reload-parity.json"))
    before = load_json(file(run_path.parent, train_manifest, "smoke-generation.json"))
    after = load_json(
        file(reload_path.parent, reload_manifest, "smoke-generation.json")
    )
    require(
        parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and before == after
        and len(parity["cases"])
        == len({row["case_id"] for row in parity["cases"]})
        == 8
        and [row["case_id"] for row in before]
        == [row["case_id"] for row in parity["cases"]]
        and all(row["identical"] is True for row in parity["cases"]),
        "Matched reload lacks exact adapter and eight-output parity.",
    )
    require(
        export["version"] == "passive-sft-export-v1"
        and export["status"]
        == "tensor_verified_paired_export_runtime_preflight_pending"
        and export["base_repository"] == BASE_REPOSITORY
        and export["base_revision"] == BASE_REVISION
        and export["runtime_revision"] == RUNTIME_REVISION,
        "Matched arm export changed its exact native/runtime identity.",
    )
    _, tensor = bound_json(
        {
            "path": "tensor-verification.json",
            "sha256": export["tensor_verification_sha256"],
        },
        export_path.parent,
        inputs,
    )
    require(
        tensor["status"] == "passed"
        and tensor["template_sha256"] == registration["anchor"]["template_sha256"]
        and tensor["eos_token_id"]
        == config["generation"]["eos_token_id"]
        == registration["native_eos"],
        "Matched export changed native template or EOS.",
    )
    return {
        "configuration": config,
        "export": export,
        "configuration_path": config_path,
        "export_path": export_path,
        "reload_path": reload_path,
        "training_manifest_path": train_manifest_path,
        "t1_preflight_path": local_path,
    }


def validate_matched_recipes(arms):
    left, right = (arms[fmt]["configuration"] for fmt in ("json", "rewrite"))
    omitted = {"max_total_sequence_tokens", "max_total_completion_tokens"}
    require(
        {key: value for key, value in left["training"].items() if key not in omitted}
        == {
            key: value for key, value in right["training"].items() if key not in omitted
        }
        and left["bindings"]["preparation"]["sha256"]
        == right["bindings"]["preparation"]["sha256"]
        and all(
            left[key] == right[key]
            for key in ("candidate_id", "model", "libraries", "generation")
        )
        and all(
            left[key]["files_sha256"] == right[key]["files_sha256"]
            for key in ("base", "tokenizer")
        ),
        "JSON/Rewrite must share sampling order, native identity and recipe; only format token totals differ.",
    )


def prepare(protocol_path, execution_path, condition):
    require(condition in CONDITIONS, "Unknown matched condition.")
    registration = load_views(load_protocol(protocol_path))
    execution_path = project_path(execution_path)
    execution = load_json(execution_path)
    registration["inputs"][execution_path] = probe.digest(execution_path)
    registration.update(execution_path=execution_path, execution=execution)
    require(
        execution["version"] == EXECUTION_VERSION
        and execution["protocol_sha256"] == probe.digest(registration["protocol_path"])
        and set(execution["arms"]) == {"json", "rewrite"},
        "Execution must bind both matched arms to this corrected protocol.",
    )
    registry_path, registry = bound_json(
        execution["registry"], execution_path.parent, registration["inputs"]
    )
    previous = load_json(registration["previous_registry"])
    require(
        registry["runtime"] == previous["runtime"]
        and registry["tokenizers"] == previous["tokenizers"]
        and probe.sampling_options(registry_path)
        == probe.sampling_options(registration["previous_registry"])
        and registry["runtime"]["revision"] == RUNTIME_REVISION,
        "Retain the exact B2 runtime, native tokenizer and greedy decoding settings.",
    )
    ids = [execution["arms"][fmt]["candidate_id"] for fmt in ("json", "rewrite")]
    require(
        len(set(ids)) == 2
        and len(registry["candidates"]) == 2
        and {row["id"] for row in registry["candidates"]} == set(ids),
        "Use a two-candidate registry for the two matched final adapters.",
    )
    arms = {}
    for fmt, arm in execution["arms"].items():
        arms[fmt] = validate_arm(registration, fmt, arm)
        model = next(
            row for row in registry["candidates"] if row["id"] == arm["candidate_id"]
        )
        require(
            model["tokenizer_id"] == "qwen35-9b"
            and all(
                model[key] == registration["anchor"][key]
                for key in curriculum.IDENTITY_FIELDS
            ),
            "Matched model changed the registered native template/tokenizer.",
        )
        artifact = arms[fmt]["export"]["artifacts"]["merged_quantized"]
        require(
            project_path(model["model_path"])
            == project_path(artifact["path"], arms[fmt]["export_path"].parent)
            and model["model_sha256"] == artifact["sha256"],
            "Registry does not contain this arm's final adapted Q4 artifact.",
        )
        probe.inspect_runtime(registry_path, arm["candidate_id"])
    validate_matched_recipes(arms)
    fmt, constrained = CONDITIONS[condition]
    sources = [
        Path(module.__file__).resolve()
        for module in (curriculum, formats, format_comparison, b2, original)
    ]
    sources.append(Path(__file__).resolve())
    binding = {
        "version": VERSION,
        "condition": condition,
        "format": fmt,
        "schema_constrained": constrained,
        "protocol_sha256": probe.digest(registration["protocol_path"]),
        "execution_sha256": probe.digest(execution_path),
        "registry_sha256": probe.digest(registry_path),
        "release_manifest_sha256": probe.digest(
            registration["release"] / "manifest.json"
        ),
        "registered_requests": 286,
        "full_dev_registered_requests": 289,
        "excluded_case_ids": [row["case_id"] for row in registration["exclusions"]],
        "messages_sha256": {
            identity: session_hash(messages)
            for identity, messages in registration["views"][fmt]["messages"].items()
        },
        "prefix_payload_sha256": registration["inputs"][
            registration["release"] / f"tokenized/qwen35-9b/{fmt}-dev.jsonl.gz"
        ],
        "artifact_bindings": {
            name: {
                "export_receipt_sha256": probe.digest(value["export_path"]),
                "training_configuration_sha256": probe.digest(
                    value["configuration_path"]
                ),
                "fresh_reload_manifest_sha256": probe.digest(value["reload_path"]),
            }
            for name, value in arms.items()
        },
        "grading_callback": "probe_passive_matched.grade: frozen format parse, then shared passive source execution/repeat gate",
        "snapshot_sha256": {path.name: probe.digest(path) for path in sources},
    }
    for path in sources:
        registration["inputs"][path] = probe.digest(path)
    prepared = {
        "cases": registration["cases"],
        **registration["views"][fmt],
        "source_files": sources,
        "reused_results": {},
        "binding": binding,
        "summarize": summarize,
    }
    registration.update(
        arms=arms, registry_path=registry_path, binding=binding, prepared=prepared
    )
    curriculum.verify_inputs(registration["inputs"])
    return registration


def grade(case, text, finish_reason="stop", *, format_name):
    if format_name == "json":
        row = evaluate_result(case, text, finish_reason)
        row.update(
            format_name="json",
            source_preserving_valid=row["raw_valid"],
            extra_change=False,
        )
        return row
    require(format_name == "rewrite", "Unknown output format.")
    row = {
        "case_id": case["id"],
        "format_name": "rewrite",
        "format_valid": False,
        "raw_valid": False,
        "source_preserving_valid": False,
        "prediction": None,
        "raw_grade": None,
        "displayed": False,
        "displayed_grade": None,
        "suppression_reason": None,
        "extra_change": False,
    }
    try:
        format_comparison.Rewrite.model_validate_json(text, strict=True)
        row["format_valid"] = True
        action = formats.parse(
            text, case["input"], "rewrite", finish_reason=finish_reason
        ).model_dump()
        row.update(
            evaluate_result(case, json.dumps(action, ensure_ascii=False), finish_reason)
        )
        row["source_preserving_valid"] = row["raw_valid"]
    except (ValueError, TypeError) as error:
        row.update(
            error=str(error),
            error_type="truncated"
            if finish_reason != "stop"
            else "semantic_invalid"
            if row["format_valid"]
            else "format_invalid",
        )
        row["extra_change"] = finish_reason == "stop" and any(
            phrase in str(error)
            for phrase in (
                "empty scope cannot create",
                "combines a caption change",
                "trim cannot change",
                "changes immutable source state",
                "exactly one supported transaction",
                "extra source changes",
            )
        )
    return row


def isolated_runtime(format_name):
    # Each comparison gets its own globals; the frozen shared runner remains untouched.
    spec = importlib.util.spec_from_file_location(
        "passive_matched_runtime", probe.__file__
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.evaluate_result = lambda case, text, finish_reason="stop": grade(
        case, text, finish_reason, format_name=format_name
    )
    return runtime


def summarize(cases, rows):
    result = b2.summarize_b2(cases, rows)
    tokens = [
        row["response"]["usage"]["completion_tokens"]
        for row in rows
        if isinstance(
            row.get("response", {}).get("usage", {}).get("completion_tokens"), int
        )
    ]
    result.update(
        source_preserving_valid=sum(
            row.get("source_preserving_valid") is True for row in rows
        ),
        extra_change=sum(row.get("extra_change") is True for row in rows),
        extra_change_scope="Explicit immutable-state/multiple-operation errors from the frozen Rewrite parser; other semantic errors remain separately invalid.",
        truncated=sum(row.get("finish_reason") == "length" for row in rows),
        output_tokens={
            "observations": len(tokens),
            "missing": len(cases) - len(tokens),
            "total": sum(tokens),
            "maximum": max(tokens, default=None),
            "p50": b2.b1._percentile(tokens, 0.5),
            "p95": b2.b1._percentile(tokens, 0.95),
        },
    )
    return result


def validate_results(cases, rows, prefixes, fmt):
    ids = [case["id"] for case in cases]
    observed = [row["case_id"] for row in rows]
    require(
        len(set(observed)) == len(observed)
        and observed == [identity for identity in ids if identity in observed],
        "Results changed matched case order or include duplicate/foreign cases.",
    )
    indexed = {case["id"]: case for case in cases}
    for row in rows:
        if (
            row.get("request_failed")
            or row.get("cancelled")
            or row.get("error_type") in original.REQUEST_ERRORS
        ):
            require(
                row.get("raw_valid") is False and row.get("displayed") is False,
                "Failed requests cannot receive edit or abstention credit.",
            )
            continue
        choice, usage = row["response"]["choices"][0], row["response"]["usage"]
        require(
            row["raw_text"] == (choice["message"].get("content") or "")
            and row["finish_reason"] == choice["finish_reason"]
            and usage["prompt_tokens"] == len(prefixes[row["case_id"]])
            and type(usage["completion_tokens"]) is int
            and 0 <= usage["completion_tokens"] <= 512,
            "Saved response differs from exact prefix/512-output budget or raw completion.",
        )
        expected = grade(
            indexed[row["case_id"]],
            row["raw_text"],
            row["finish_reason"],
            format_name=fmt,
        )
        require(
            all(row.get(key) == value for key, value in expected.items()),
            "Saved format/source grade differs from frozen execution.",
        )


def snapshot(registration, output):
    target = output / "implementation"
    target.mkdir(exist_ok=True)
    paths = {
        "matched-protocol.json": registration["protocol_path"],
        "matched-execution.json": registration["execution_path"],
    }
    for fmt, arm in registration["arms"].items():
        for name, key in (
            ("export-receipt", "export_path"),
            ("training-configuration", "configuration_path"),
            ("reload-manifest", "reload_path"),
        ):
            paths[f"matched-{fmt}-{name}.json"] = arm[key]
    for name, path in paths.items():
        require(
            probe.digest(path) == registration["inputs"][path],
            "Matched evidence changed before snapshot.",
        )
        with (target / name).open("xb") as stream:
            stream.write(path.read_bytes())


def run_condition(registration, condition, output, stage, editor_state):
    output = curriculum.new_output(output, registration["release"])
    if stage == "prepare":
        output.mkdir(parents=True)
        snapshot(registration, output)
        write_new(
            output / "matched-receipt.json",
            {**registration["binding"], "status": "prepared_generation_unrun"},
        )
        return
    fmt, constrained = CONDITIONS[condition]
    args = SimpleNamespace(
        release=registration["release"],
        registry=registration["registry_path"],
        candidate=registration["execution"]["arms"][fmt]["candidate_id"],
        output=output,
        stage=stage,
        timeout=120,
        unconstrained=not constrained,
        sampling_config=None,
        diagnostic_protocol=None,
        editor_state=editor_state,
    )
    error = None
    try:
        isolated_runtime(fmt).run(args, prepared=registration["prepared"])
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
    if output.exists():
        try:
            curriculum.verify_inputs(registration["inputs"])
            snapshot(registration, output)
            results_path = output / "results.jsonl"
            rows = original.read_rows(results_path) if results_path.exists() else []
            validate_results(
                registration["cases"], rows, registration["views"][fmt]["prefixes"], fmt
            )
            with (output / "source-execution.jsonl").open("x") as stream:
                for value in curriculum.source_execution(registration["cases"], rows):
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            status = load_json(output / "manifest.json")["status"]
        except (Exception, KeyboardInterrupt) as caught:
            error, status = error or caught, "failed"
        write_new(
            output / "matched-receipt.json",
            {
                **registration["binding"],
                "status": "failed" if error else status,
                "error": f"{type(error).__name__}: {error}" if error else None,
                "files_sha256": {
                    path.name: probe.digest(path)
                    for path in output.iterdir()
                    if path.is_file()
                },
                "runtime_manifest_version": "passive-b2-runtime-v1 (isolated shared runner; explicit frozen format callback)",
            },
        )
    if error:
        raise error


def report(protocol_path, execution_path, runs, output):
    registration = prepare(protocol_path, execution_path, "matched-json-unconstrained")
    runs = project_path(runs)
    output = curriculum.new_output(output, registration["release"])
    summaries, results, provenance = {}, {}, {}
    for condition, (fmt, constrained) in CONDITIONS.items():
        path = runs / condition
        receipt_path = path / "matched-receipt.json"
        rows, status = [], "missing_run"
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            expected = {
                **registration["binding"],
                "condition": condition,
                "format": fmt,
                "schema_constrained": constrained,
                "messages_sha256": {
                    identity: session_hash(messages)
                    for identity, messages in registration["views"][fmt][
                        "messages"
                    ].items()
                },
                "prefix_payload_sha256": registration["inputs"][
                    registration["release"] / f"tokenized/qwen35-9b/{fmt}-dev.jsonl.gz"
                ],
            }
            require(
                all(receipt.get(key) == value for key, value in expected.items()),
                "Run receipt belongs to another matched experiment/arm.",
            )
            for name, value in receipt["files_sha256"].items():
                read_bound(
                    {"path": name, "sha256": value}, path, registration["inputs"]
                )
            manifest = load_json(path / "manifest.json")
            candidate = registration["execution"]["arms"][fmt]["candidate_id"]
            require(
                manifest["benchmark"] == expected
                and manifest["candidate"] == candidate
                and manifest["stage"] == "screen"
                and manifest["schema_constrained"] == constrained
                and manifest["case_ids"]
                == [case["id"] for case in registration["cases"]]
                and manifest["registry_sha256"] == expected["registry_sha256"]
                and manifest["sampling"]
                == probe.sampling_options(registration["registry_path"]),
                "Run changed the matched cohort, grammar or sampling condition.",
            )
            for name, value in manifest["benchmark"]["snapshot_sha256"].items():
                read_bound(
                    {"path": name, "sha256": value},
                    path / "implementation",
                    registration["inputs"],
                )
            read_bound(
                {"path": "model-registry.json", "sha256": expected["registry_sha256"]},
                path / "implementation",
                registration["inputs"],
            )
            result_path = path / "results.jsonl"
            if result_path.exists():
                require(
                    "results.jsonl" in receipt["files_sha256"],
                    "Unbound raw results appeared.",
                )
                rows = original.read_rows(result_path)
            require(
                manifest["completed_requests"] == len(rows),
                "Run completion census differs.",
            )
            status = receipt["status"]
            provenance[condition] = {
                "receipt_sha256": probe.digest(receipt_path),
                "status": status,
                "sampled_peak_server_rss_bytes": manifest.get(
                    "sampled_peak_server_rss_bytes"
                ),
            }
        elif path.exists():
            raise ValueError(f"Run has no immutable completion/failure receipt: {path}")
        validate_results(
            registration["cases"], rows, registration["views"][fmt]["prefixes"], fmt
        )
        summaries[condition] = {
            **summarize(registration["cases"], rows),
            "run_status": status,
        }
        results[condition] = rows
    curriculum.verify_inputs(registration["inputs"])
    output.mkdir(parents=True)
    snapshot(registration, output)
    primary = {name: results[name] for name in list(CONDITIONS)[:2]}
    write_new(
        output / "summary.json",
        {
            "version": VERSION,
            "protocol_sha256": probe.digest(registration["protocol_path"]),
            "execution_sha256": probe.digest(registration["execution_path"]),
            "registered_new_requests": 858,
            "registered_requests_per_condition": 286,
            "full_dev_registered_requests": 289,
            "matched_input_exclusions": registration["exclusions"],
            "runs": summaries,
            "primary_unconstrained_comparison": b2.compare_runs(
                registration["cases"], primary
            ),
            "constrained_json_diagnostic": "Reported separately; never an advantage attributed to unconstrained format skill.",
            "provenance": provenance,
            "scope": "Fixed exposed development; all final arms and failures retained. No deployment promotion or T3 anchor selection.",
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
