"""Run a separately frozen T1 caption-prompt diagnostic on local paired exports."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import passive_b2_benchmark as b2
import probe_passive as probe
import probe_passive_b2 as original
import probe_passive_curriculum as shared
import probe_passive_gemma as gemma
import probe_passive_gemma_matched as evidence
from passive_contract import canonical_messages

ROOT = probe.ROOT
VERSION = "passive-gemma-caption-prompt-evaluation-v1"
PREPARATION_VERSION = "passive-gemma-caption-prompt-preparation-v1"
CONDITIONS = {
    "base-original": ("native_base", "original"),
    "base-candidate": ("native_base", "candidate"),
    "sft-original": ("sft", "original"),
    "sft-candidate": ("sft", "candidate"),
}
POLICY = {
    "input_tokens": 3584,
    "output_tokens": 512,
    "context_tokens": 4096,
    "request_timeout_seconds": 120,
    "temperature": 0,
    "seed": 42,
    "schema_constrained": True,
    "cache_requested": False,
    "icl": False,
    "parallel_slots": 1,
    "native_generation_stop_ids": [1, 106, 50],
    **evidence.RESPONSE_EVIDENCE,
}
SUCCESS_CRITERIA = {
    "candidate": "sft-candidate",
    "preservation_floor": "max(base-original,base-candidate)",
    "non_regression_reference": "sft-original",
    "correct_abstentions": "at_least_preservation_floor",
    "displayed_useful_edits": "not_lower",
    "displayed_useful_precision": "not_lower",
    "invalid_results": "not_higher",
    "complete_requests_per_condition": 289,
    "required_conditions": list(CONDITIONS),
    "scope": "Exposed development diagnostic; never a retroactive T1 pass or final-test result.",
}
T1_PINS = {
    "t1_protocol": "51c4ec869b5094260918f2a14c6485af398d79bada6016993c0be9e536188417",
    "t1_execution": "2e001c324a641c205f635e1a14a915474d5cd06f8d04a3a5e80d8b9aa398dfb2",
}
require = shared.require
load_json = shared.load_json
read_bound = shared.read_bound
write_new = shared.write_new
project_path = shared.project_path


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def source_paths():
    return sorted(
        {
            Path(__file__).resolve(),
            *evidence.source_paths(),
            *(ROOT / name for name in probe.code_bindings(probe.DEFAULT_RELEASE)),
        }
    )


def read_preparation(path, inputs):
    manifest = load_json(path)
    directory = path.parent
    require(
        manifest.get("version") == PREPARATION_VERSION
        and manifest.get("status") == "frozen_rule_only_prompt_preparation",
        "Require the frozen single-candidate rule-only preparation.",
    )
    inventory = manifest["files_sha256"]
    require(
        {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
        == {"manifest.json", *inventory},
        "Preparation inventory changed after freezing.",
    )
    require(
        manifest["preparation_code_sha256"] == inventory["source/prepare.py"],
        "Preparation source snapshot differs from its manifest.",
    )
    for name, expected in inventory.items():
        bound = read_bound({"path": name, "sha256": expected}, directory, inputs)
        require(
            bound.is_relative_to(directory), "Preparation payload escaped its bundle."
        )
    for name, expected in manifest["inputs_sha256"].items():
        require(
            not any(
                part.startswith("test-") or "-test." in part
                for part in Path(name).parts
            ),
            "Final-test bodies are outside the diagnostic scope.",
        )
        read_bound({"path": name, "sha256": expected}, ROOT, inputs)
    config = load_json(directory / "configuration.json")
    require(
        config["candidate_id"] == "gemma26"
        and config["native_base"]
        == {"repository": gemma.BASE_REPOSITORY, "revision": gemma.BASE_REVISION}
        and config["template_arguments"]
        == {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "return_dict": False,
        }
        and config["input_limit"] == POLICY["input_tokens"]
        and config["output_limit"] == POLICY["output_tokens"]
        and set(config["prompt_variants"]) == {"original", "candidate"}
        and set(config["cohorts"]) == {"dev", "diagnostic"},
        "Preparation changed the model, prompt variants or fixed token budgets.",
    )
    return {"manifest": manifest, "configuration": config, "directory": directory}


def validate_views(bundle, cases_by_cohort, original_prefixes):
    config, directory = bundle["configuration"], bundle["directory"]
    prompts = {
        variant: (directory / name).read_text().strip()
        for variant, name in config["prompt_variants"].items()
    }
    require(
        prompts["candidate"] != prompts["original"],
        "Candidate prompt must be a distinct frozen rule.",
    )
    views = {variant: {} for variant in prompts}
    for cohort, cases in cases_by_cohort.items():
        ids = [case["id"] for case in cases]
        setting = config["cohorts"][cohort]
        roster = load_json(directory / setting["roster"])
        require(
            setting["case_count"] == len(ids)
            and setting["case_ids"] == roster["case_ids"] == ids
            and [row["case_id"] for row in roster["cases"]] == ids
            and roster["split"] == ("train" if cohort == "diagnostic" else "dev"),
            "Preparation changed the ordered diagnostic/development roster.",
        )
        for case, row in zip(cases, roster["cases"], strict=True):
            expected = canonical_messages(case["input"])
            require(
                row["user_sha256"] == text_hash(expected[1]["content"])
                and all(
                    row[key] == case["metadata"][key]
                    for key in ("split", "language", "source_group_id", "episode_id")
                ),
                "Roster metadata or observable user input differs from the frozen case.",
            )
        for variant in prompts:
            payload = config["payloads"][variant][cohort]
            message_rows = original.read_rows(directory / payload["messages"])
            prefix_rows = original.read_rows(directory / payload["prefixes"])
            require(
                [row["case_id"] for row in message_rows] == ids
                and [row["case_id"] for row in prefix_rows] == ids,
                "Messages/prefixes must keep every frozen case exactly once in order.",
            )
            messages, prefixes, rendered = {}, {}, {}
            for case, message, prefix in zip(
                cases, message_rows, prefix_rows, strict=True
            ):
                identity = case["id"]
                expected = canonical_messages(case["input"])
                require(
                    expected[0]["content"] == prompts["original"],
                    "Original prompt drifted.",
                )
                expected[0]["content"] = prompts[variant]
                ids_for_case = prefix["input_ids"]
                require(
                    message["messages"] == expected
                    and message["messages_sha256"]
                    == prefix["messages_sha256"]
                    == json_hash(expected)
                    and message["user_sha256"] == text_hash(expected[1]["content"])
                    and isinstance(ids_for_case, list)
                    and 0 < len(ids_for_case) <= POLICY["input_tokens"]
                    and all(
                        type(token) is int and 0 <= token < 262144
                        for token in ids_for_case
                    )
                    and prefix["input_tokens"] == len(ids_for_case)
                    and prefix["prefix_sha256"] == json_hash(ids_for_case)
                    and isinstance(prefix["rendered_prompt"], str)
                    and bool(prefix["rendered_prompt"]),
                    "Only the system prompt may change; exact bounded token/message evidence is required.",
                )
                if variant == "original":
                    require(
                        ids_for_case == original_prefixes[cohort][identity],
                        "Original prefix differs from frozen T1 input.",
                    )
                messages[identity], prefixes[identity] = expected, ids_for_case
                rendered[identity] = prefix["rendered_prompt"]
            views[variant][cohort] = {
                "messages": messages,
                "prefixes": prefixes,
                "rendered": rendered,
            }
    return views


def prepare(protocol_path, condition, cohort="dev"):
    require(
        condition in CONDITIONS and cohort in ("dev", "diagnostic"),
        "Unknown diagnostic condition/cohort.",
    )
    protocol_path = project_path(protocol_path)
    inputs = {protocol_path: probe.digest(protocol_path)}
    protocol = load_json(protocol_path)
    require(
        protocol.get("version") == VERSION
        and protocol.get("conditions") == list(CONDITIONS)
        and protocol.get("registered_requests") == 289
        and protocol.get("diagnostic_requests") == 8
        and protocol.get("final_test_bodies_read") is False
        and protocol.get("runtime") == POLICY
        and protocol.get("success_criteria") == SUCCESS_CRITERIA,
        "Freeze the four conditions, full289/eight-train roster, runtime and success criteria before generation.",
    )
    pins = protocol["evaluation_sources_sha256"]
    require(
        set(pins) == {str(p.relative_to(ROOT)) for p in source_paths()},
        "Bind the complete exact evaluator source set.",
    )
    for name, expected in pins.items():
        read_bound({"path": name, "sha256": expected}, ROOT, inputs)
    paths = {}
    for key, expected in T1_PINS.items():
        require(
            protocol[key]["sha256"] == expected,
            "Use the unchanged completed T1 protocol/export registration.",
        )
        paths[key] = read_bound(protocol[key], protocol_path.parent, inputs)
    preparation_path = read_bound(protocol["preparation"], protocol_path.parent, inputs)
    bundle = read_preparation(preparation_path, inputs)
    t1 = gemma.load_registration(paths["t1_protocol"], paths["t1_execution"])
    args, prepared, _ = gemma.prepare_inputs(t1, "full")
    gemma.validate_pair(t1)
    inputs.update(t1["inputs"])
    release = args.release
    require(
        bundle["manifest"]["release_manifest_sha256"]
        == probe.digest(release / "manifest.json"),
        "Preparation uses another frozen release.",
    )
    train, manifest = b2._load_split(release, "train")
    require(
        bundle["manifest"]["dev_case_snapshot_sha256"]
        == manifest["files_sha256"]["dev-cases.jsonl"],
        "Frozen development grading snapshot differs from preparation.",
    )
    tokenizer = bundle["configuration"]["tokenizer"]
    model = t1["registry"]["candidates"][0]
    require(
        tokenizer["id"] == "gemma26"
        and tokenizer["path"] == model["tokenizer_path"]
        and tokenizer["files_sha256"] == model["tokenizer_files_sha256"]
        and tokenizer["eos_token_id"] == 1
        and tokenizer["vocab_size"] == 262144,
        "Prepared native tokenizer identity differs from the verified export pair.",
    )
    train_by_id = {case["id"]: case for case in train}
    diagnostic_ids = bundle["configuration"]["cohorts"]["diagnostic"]["case_ids"]
    require(
        len(diagnostic_ids) == len(set(diagnostic_ids)) == 8
        and set(diagnostic_ids) <= set(train_by_id),
        "Diagnostics must be exactly eight frozen training states.",
    )
    diagnostic = [
        b2._bind_execution(train_by_id[identity], manifest)
        for identity in diagnostic_ids
    ]
    cases_by_cohort = {"dev": prepared["cases"], "diagnostic": diagnostic}
    train_prefix_name = "tokenized/gemma26/json-curriculum-train.jsonl.gz"
    train_prefix_path = read_bound(
        {
            "path": train_prefix_name,
            "sha256": manifest["files_sha256"][train_prefix_name],
        },
        release,
        inputs,
    )
    train_prefixes = {
        row["id"]: row["primary"]["input_ids"][: row["primary"]["completion_start"]]
        for row in original.read_rows(train_prefix_path)
    }
    views = validate_views(
        bundle,
        cases_by_cohort,
        {"dev": prepared["prefixes"], "diagnostic": train_prefixes},
    )
    role, variant = CONDITIONS[condition]
    selected = views[variant][cohort]
    cases = cases_by_cohort[cohort]
    source_files = sorted({*prepared["source_files"], *source_paths()})
    binding = {
        "version": VERSION,
        "protocol_sha256": inputs[protocol_path],
        "preparation_sha256": inputs[preparation_path],
        "configuration_sha256": inputs[bundle["directory"] / "configuration.json"],
        "t1_protocol_sha256": T1_PINS["t1_protocol"],
        "t1_execution_sha256": T1_PINS["t1_execution"],
        "registry_sha256": inputs[t1["paths"]["new_registry"]],
        "release_manifest_sha256": probe.digest(release / "manifest.json"),
        "condition": condition,
        "model_role": role,
        "prompt_variant": variant,
        "cohort": cohort,
        "registered_requests": len(cases),
        "registered_case_ids": [case["id"] for case in cases],
        "messages_sha256": {
            identity: json_hash(messages)
            for identity, messages in selected["messages"].items()
        },
        "prefixes_sha256": {
            identity: json_hash(ids) for identity, ids in selected["prefixes"].items()
        },
        "rendered_prompts_sha256": {
            identity: text_hash(text) for identity, text in selected["rendered"].items()
        },
        "snapshot_sha256": {path.name: probe.digest(path) for path in source_files},
        "runtime": POLICY,
        "reused_results": 0,
        "input_exclusions": [],
    }
    for source in source_files:
        inputs[source] = probe.digest(source)
    shared.verify_inputs(inputs)
    return {
        "protocol_path": protocol_path,
        "protocol": protocol,
        "preparation_path": preparation_path,
        "bundle": bundle,
        "inputs": inputs,
        "t1": t1,
        "release": release,
        "registry_path": t1["paths"]["new_registry"],
        "cases": cases,
        "views": views,
        "binding": binding,
        "cohort": cohort,
        "prepared": {
            "cases": cases,
            **selected,
            "binding": binding,
            "source_files": source_files,
            "reused_results": {},
            "summarize": summarize,
        },
    }


def isolated_runtime(prepared):
    runtime = evidence.isolated_runtime("json")
    template_check = runtime.template_check
    by_messages = {
        json_hash(messages): identity
        for identity, messages in prepared["messages"].items()
    }

    def check(client, body, expected):
        identity = by_messages[json_hash(body["messages"])]
        require(
            expected == prepared["prefixes"][identity],
            "Runtime received another prefix.",
        )
        row = template_check(client, body, expected)
        row["rendered_prompt_matches_frozen"] = (
            row["rendered_prompt"] == prepared["rendered"][identity]
        )
        row["passed"] = row["passed"] and row["rendered_prompt_matches_frozen"]
        return row

    runtime.template_check = check
    return runtime


def server_timings(rows, expected_requests):
    result = {}
    for label, key in (("prefill", "prompt_ms"), ("decode", "predicted_ms")):
        measured = []
        for row in rows:
            response = row.get("response")
            timing = response.get("timings") if isinstance(response, dict) else None
            value = timing.get(key) if isinstance(timing, dict) else None
            if (
                not row.get("reused_from")
                and type(value) in (int, float)
                and math.isfinite(value)
                and value >= 0
            ):
                measured.append(value)
        result[label] = {
            "p50_ms": b2.b1._percentile(measured, 0.5),
            "p95_ms": b2.b1._percentile(measured, 0.95),
            "measured_requests": len(measured),
            "expected_requests": expected_requests,
            "unmeasured_requests": expected_requests - len(measured),
        }
    result["scope"] = (
        "Raw server timings only, including failed responses when available; missing values are not inferred from HTTP latency."
    )
    return result


def summarize(cases, rows):
    result = evidence.summarize(cases, rows)
    result["server_prefill_decode_ms_p50_p95"] = server_timings(rows, len(cases))
    result["semantic_invalid_results"] = sum(
        row.get("error_type") == "semantic_invalid" for row in rows
    )
    result["scope"] = (
        "Frozen reference/source-execution agreement; diagnostic train and exposed development cohorts are always reported separately."
    )
    return result


def validate_results(cases, rows, prefixes):
    require(
        not any(row.get("reused_from") for row in rows),
        "Every diagnostic request must be fresh.",
    )
    evidence.validate_results(cases, rows, prefixes, "json")


def validate_template_checks(cases, rows, prepared, *, completed):
    ids = [case["id"] for case in cases]
    observed = [row["case_id"] for row in rows]
    require(
        observed == ids[: len(observed)],
        "Template checks changed case order or repeat a case.",
    )
    if completed:
        require(
            observed == ids, "Completed runs require prefix parity for every request."
        )
    for row in rows:
        identity = row["case_id"]
        tokens = prepared["prefixes"][identity]
        matches = (
            row["runtime_token_ids"] == tokens
            and row["rendered_prompt"] == prepared["rendered"][identity]
        )
        require(
            row["passed"] == matches
            and row["rendered_prompt_matches_frozen"]
            == (row["rendered_prompt"] == prepared["rendered"][identity]),
            "Saved template/prefix parity evidence changed.",
        )
        if completed:
            require(
                matches
                and row["within_budget"] is True
                and row["frozen_tokens"] == row["runtime_tokens"] == len(tokens),
                "Completed run contains mismatched rendering or prefix tokens.",
            )


def new_output(path, registration):
    path = shared.new_output(path, registration["release"])
    require(
        not path.is_relative_to(registration["bundle"]["directory"])
        and not any(parent.name == "frozen" for parent in (path, *path.parents)),
        "Diagnostic outputs cannot modify any frozen bundle.",
    )
    return path


def snapshot(registration, output):
    directory = output / "registration"
    directory.mkdir()
    paths = {
        "protocol.json": registration["protocol_path"],
        "preparation-manifest.json": registration["preparation_path"],
        "preparation-configuration.json": registration["bundle"]["directory"]
        / "configuration.json",
    }
    for name, path in paths.items():
        require(
            probe.digest(path) == registration["inputs"][path],
            "Registration changed before snapshot.",
        )
        with (directory / name).open("xb") as stream:
            stream.write(path.read_bytes())


@contextmanager
def predictor_lock():
    path = ROOT / "artifacts/nep-passive-b1.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another passive benchmark owns the local predictor."
            ) from error
        yield


def run_condition(registration, condition, output, stage, editor_state):
    require(
        registration["binding"]["condition"] == condition,
        "Prepared condition differs from run.",
    )
    require(stage in ("prepare", "preflight", "screen"), "Unknown diagnostic stage.")
    output = new_output(output, registration)
    if stage == "prepare":
        output.mkdir(parents=True)
        snapshot(registration, output)
        write_new(
            output / "caption-prompt-receipt.json",
            {**registration["binding"], "status": "prepared_generation_unrun"},
        )
        return
    args = SimpleNamespace(
        release=registration["release"],
        registry=registration["registry_path"],
        candidate=registration["t1"]["execution"]["candidate_ids"][
            CONDITIONS[condition][0]
        ],
        output=output,
        stage=stage,
        timeout=POLICY["request_timeout_seconds"],
        unconstrained=False,
        sampling_config=None,
        diagnostic_protocol=None,
        editor_state=editor_state,
    )
    with predictor_lock():
        error = None
        try:
            shared.verify_inputs(registration["inputs"])
            isolated_runtime(registration["prepared"]).run(
                args, prepared=registration["prepared"]
            )
        except (Exception, KeyboardInterrupt) as caught:
            error = caught
        if output.exists():
            try:
                shared.verify_inputs(registration["inputs"])
                snapshot(registration, output)
                path = output / "results.jsonl"
                rows = original.read_rows(path) if path.exists() else []
                validate_results(
                    registration["cases"], rows, registration["prepared"]["prefixes"]
                )
                with (output / "source-execution.jsonl").open("x") as stream:
                    for row in shared.source_execution(registration["cases"], rows):
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                status = load_json(output / "manifest.json")["status"]
            except (Exception, KeyboardInterrupt) as caught:
                error, status = error or caught, "failed"
            write_new(
                output / "caption-prompt-receipt.json",
                {
                    **registration["binding"],
                    "status": "failed" if error else status,
                    "error": f"{type(error).__name__}: {error}" if error else None,
                    "files_sha256": {
                        str(path.relative_to(output)): probe.digest(path)
                        for path in output.rglob("*")
                        if path.is_file()
                    },
                },
            )
        if error:
            raise error


def decision(summaries, cohort):
    complete = cohort == "dev" and all(
        summaries[name]["run_status"] == "completed"
        and summaries[name]["observed_results"] == 289
        and summaries[name]["request_errors"] == 0
        and summaries[name]["token_evidence_observations"] == 289
        for name in CONDITIONS
    )
    candidate, control = summaries["sft-candidate"], summaries["sft-original"]
    native_floor = max(
        summaries[name]["correct_abstentions"]
        for name in ("base-original", "base-candidate")
    )
    checks = {
        "complete_full_development_evidence": complete,
        "preservation_recovers_both_native_controls": candidate["correct_abstentions"]
        >= native_floor,
        "useful_count_not_lower": candidate["displayed_useful_edits"]
        >= control["displayed_useful_edits"],
        "precision_not_lower": (
            candidate["displayed_useful_precision"] is not None
            and control["displayed_useful_precision"] is not None
            and candidate["displayed_useful_precision"]
            >= control["displayed_useful_precision"]
        ),
        "invalid_not_higher": candidate["invalid_results"]
        <= control["invalid_results"],
    }
    return {
        "criteria": SUCCESS_CRITERIA,
        "checks": checks,
        "native_preservation_floor": native_floor,
        "status": "not_assessable"
        if not complete
        else "passed"
        if all(checks.values())
        else "failed",
    }


def report(protocol_path, runs, output, cohort="dev"):
    registration = prepare(protocol_path, "base-original", cohort)
    runs = project_path(runs)
    output = new_output(output, registration)
    summaries, provenance = {}, {}
    previous_finish = None
    for condition, (role, variant) in CONDITIONS.items():
        directory = runs / condition
        receipt_path = directory / "caption-prompt-receipt.json"
        rows, manifest, status = [], {}, "missing_run"
        selected = registration["views"][variant][cohort]
        expected = {
            **registration["binding"],
            "condition": condition,
            "model_role": role,
            "prompt_variant": variant,
            "messages_sha256": {
                identity: json_hash(value)
                for identity, value in selected["messages"].items()
            },
            "prefixes_sha256": {
                identity: json_hash(value)
                for identity, value in selected["prefixes"].items()
            },
            "rendered_prompts_sha256": {
                identity: text_hash(value)
                for identity, value in selected["rendered"].items()
            },
        }
        if receipt_path.exists():
            receipt = load_json(receipt_path)
            require(
                all(receipt.get(key) == value for key, value in expected.items()),
                "Run receipt belongs to another protocol/cohort/prompt.",
            )
            files = receipt.get("files_sha256", {})
            require(
                "manifest.json" in files,
                "Reporting needs a bound runtime completion/failure manifest.",
            )
            require(
                {
                    str(path.relative_to(directory))
                    for path in directory.rglob("*")
                    if path.is_file()
                }
                == {*files, "caption-prompt-receipt.json"},
                "Run files changed after immutable receipt creation.",
            )
            for name, expected_hash in files.items():
                bound = read_bound(
                    {"path": name, "sha256": expected_hash},
                    directory,
                    registration["inputs"],
                )
                require(
                    bound.is_relative_to(directory),
                    "Run file escaped its immutable output.",
                )
            manifest = load_json(directory / "manifest.json")
            require(
                manifest["benchmark"] == expected
                and manifest["stage"] == "screen"
                and manifest["candidate"]
                == registration["t1"]["execution"]["candidate_ids"][role]
                and manifest["registry_sha256"] == expected["registry_sha256"]
                and manifest["case_ids"] == expected["registered_case_ids"]
                and manifest["sampling"]
                == probe.sampling_options(registration["registry_path"])
                and manifest["schema_constrained"] is True
                and manifest["cache_requested"] is False
                and manifest["context_tokens"] == 4096
                and manifest["reused_results"] == 0,
                "Run changed the model, cohort, schema, sampling, cache or context.",
            )
            registry = registration["t1"]["registry"]
            if "candidate_binding" in manifest or receipt["status"] == "completed":
                model = next(
                    value
                    for value in registry["candidates"]
                    if value["id"] == manifest["candidate"]
                )
                require(
                    manifest.get("candidate_binding") == model
                    and manifest.get("runtime_binding") == registry["runtime"]
                    and manifest.get("tokenizer_binding")
                    == {
                        key: value
                        for key, value in registration["bundle"]["configuration"][
                            "tokenizer"
                        ].items()
                        if key not in ("eos_token_id", "vocab_size")
                    },
                    "Saved model, runtime or native tokenizer binding differs.",
                )
            for name, expected_hash in expected["snapshot_sha256"].items():
                require(
                    files.get(f"implementation/{name}") == expected_hash,
                    "Saved evaluator source differs from its binding.",
                )
            require(
                files.get("implementation/model-registry.json")
                == expected["registry_sha256"],
                "Saved model registry differs.",
            )
            rows = (
                original.read_rows(directory / "results.jsonl")
                if "results.jsonl" in files
                else []
            )
            require(
                manifest["completed_requests"] == len(rows),
                "Completion census differs from raw results.",
            )
            status = receipt["status"]
            if status == "completed":
                require(
                    manifest["status"] == "completed"
                    and len(rows) == len(registration["cases"]),
                    "Completed run must retain the entire registered cohort.",
                )
            checks = (
                original.read_rows(directory / "template-checks.jsonl")
                if "template-checks.jsonl" in files
                else []
            )
            validate_template_checks(
                registration["cases"], checks, selected, completed=status == "completed"
            )
            if rows:
                require(
                    len(checks) == len(registration["cases"])
                    and all(row["passed"] and row["within_budget"] for row in checks),
                    "Generation occurred without all-request exact prefix preflight.",
                )
            require(
                previous_finish is None or manifest["started_at"] >= previous_finish,
                "Fresh conditions must run sequentially in the registered order.",
            )
            previous_finish = manifest["finished_at"]
            if "source-execution.jsonl" in files:
                require(
                    original.read_rows(directory / "source-execution.jsonl")
                    == list(shared.source_execution(registration["cases"], rows)),
                    "Saved source-execution grading changed.",
                )
            elif status == "completed":
                raise ValueError(
                    "Completed run lacks frozen source-execution evidence."
                )
            provenance[condition] = {
                "receipt_sha256": probe.digest(receipt_path),
                "status": status,
            }
        elif directory.exists():
            raise ValueError(
                f"Run has no immutable completion/failure receipt: {directory}"
            )
        validate_results(registration["cases"], rows, selected["prefixes"])
        summaries[condition] = {
            **summarize(registration["cases"], rows),
            "run_status": status,
            "actual_memory_and_swap": gemma.resource_observations(manifest),
        }
    shared.verify_inputs(registration["inputs"])
    output.mkdir(parents=True)
    snapshot(registration, output)
    write_new(
        output / "summary.json",
        {
            "version": VERSION,
            "protocol_sha256": probe.digest(registration["protocol_path"]),
            "cohort": cohort,
            "registered_requests_per_condition": len(registration["cases"]),
            "registered_conditions": list(CONDITIONS),
            "input_exclusions": [],
            "runs": summaries,
            "decision": decision(summaries, cohort),
            "provenance": provenance,
            "timing_scope": "Fresh HTTP timing includes measured failures and excludes grading; server prefill/decode are separate. Sequential desktop load/thermal state remains uncontrolled.",
            "raw_format_scope": "Unmodified sampled content/tokens and API content are separate evidence. Native terminal50 and missing sampled-token evidence never receive credit.",
            "scope": "Train-only engineering diagnostics or already exposed development, never pooled. No final-test access, model training or deployment promotion.",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--cohort", choices=("diagnostic", "dev"), default="dev")
    parser.add_argument(
        "--stage", choices=("prepare", "preflight", "screen"), default="prepare"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument(
        "--editor-state", default="Not controlled; no editor-preview latency claim"
    )
    args = parser.parse_args(argv)
    if args.command == "report":
        require(
            args.runs is not None and args.condition is None,
            "Report requires --runs without --condition.",
        )
        report(args.protocol, args.runs, args.output, args.cohort)
    else:
        require(
            args.condition is not None and args.runs is None,
            "Run requires --condition without --runs.",
        )
        registration = prepare(args.protocol, args.condition, args.cohort)
        run_condition(
            registration, args.condition, args.output, args.stage, args.editor_state
        )


if __name__ == "__main__":
    main()
