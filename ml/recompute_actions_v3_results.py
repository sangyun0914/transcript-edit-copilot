"""Recompute registered v3 evidence using only the Python standard library."""

import argparse
from collections import Counter, defaultdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as source:
        for line in source:
            yield json.loads(line)


def resolve_path(root, recorded_path, recorded_workspace=None):
    root = Path(root).resolve()
    workspace = Path(recorded_workspace) if recorded_workspace is not None else root
    require(workspace.is_absolute(), "Recorded workspace must be absolute.")
    path = Path(recorded_path)
    require(".." not in path.parts, f"Escaping evidence path: {path}")
    if path.is_absolute():
        require(
            path.is_relative_to(workspace),
            f"Evidence is outside the recorded workspace: {path}",
        )
        path = path.relative_to(workspace)
    path = (root / path).resolve()
    require(path.is_relative_to(root), f"Evidence escapes the supplied root: {path}")
    return path


def bound(root, binding, recorded_workspace=None):
    path = resolve_path(root, binding["path"], recorded_workspace)
    require(sha256(path) == binding["sha256"], f"Changed bound evidence: {path}")
    return path


def verify_manifest(path):
    manifest = read(path)
    for name, expected in manifest["files_sha256"].items():
        require(not Path(name).is_absolute(), "Manifest payload must be relative.")
        payload = resolve_path(path.parent, name)
        require(sha256(payload) == expected, f"Changed payload: {name}")
    return manifest


def dataset_summary(manifest_path):
    manifest = verify_manifest(manifest_path)
    result, split_groups, all_ids = {}, {}, set()
    for split in ("train", "dev"):
        counts, actions, languages, groups = Counter(), Counter(), Counter(), set()
        for case in rows(manifest_path.parent / f"cases/{split}.jsonl"):
            require(
                case["id"] not in all_ids, "A case repeats within or across splits."
            )
            all_ids.add(case["id"])
            require(case["split"] == split, "Wrong case split.")
            counts["cases"] += 1
            counts["accepted_references"] += len(case["accepted_predictions"])
            counts["edit_required"] += case["judgment"]["status"] == "edit_required"
            counts["preserve"] += case["judgment"]["status"] == "preserve"
            actions[case["primary"]["action"]] += 1
            languages[case["language"]] += 1
            groups.add(case["source"]["source_group"])
        require(
            counts["cases"] == manifest["counts"][split],
            "Declared dataset count differs.",
        )
        require(
            counts["edit_required"] + counts["preserve"] == counts["cases"],
            "Unknowns entered the admitted corpus.",
        )
        split_groups[split] = groups
        result[split] = {
            **counts,
            "primary_actions": dict(actions),
            "languages": dict(languages),
            "source_groups": len(groups),
        }
    require(
        not split_groups["train"] & split_groups["dev"], "Source-group split leakage."
    )
    return result


def committed_counts(exposure, events, is_dpo):
    require(
        not exposure["optimizer_commit_uncertain"], "Optimizer commitment is uncertain."
    )
    committed = exposure["phases"]["optimizer_committed"]
    require(
        exposure["completed_updates"] == len(events),
        "Update count differs from actual events.",
    )
    identities = Counter()
    key = "pair_ids" if is_dpo else "case_ids"
    total_key = "committed_pairs" if is_dpo else "committed_rows"
    for index, event in enumerate(events, 1):
        require(event["update"] == index, "Missing or repeated optimizer update.")
        identities.update(event[key])
        require(
            event[total_key] == sum(identities.values()),
            "Cumulative committed exposure differs.",
        )
    expected = committed["counts"]["pair_id" if is_dpo else "case_id"]
    require(
        dict(identities) == expected,
        "Committed identities differ from the exposure ledger.",
    )
    require(
        sum(identities.values()) == committed["rows"], "Committed row count differs."
    )
    cases = committed["counts"]["case_id"]
    require(
        sum(cases.values()) == committed["rows"]
        and len(cases) == committed["unique_cases"],
        "Unique-state exposure differs.",
    )
    return {
        "updates": len(events),
        "committed_draws_or_pairs": committed["rows"],
        "unique_states": len(cases),
        "unique_pairs": len(identities) if is_dpo else None,
        "sequence_tokens": committed["sequence_tokens"],
        "completion_tokens": committed["completion_tokens"],
        "actions": committed["counts"]["chosen_action" if is_dpo else "action"],
        "languages": committed["counts"]["language"],
        "source_groups": len(committed["counts"]["source_group_id"]),
    }


def training_summary(manifest_path):
    manifest = verify_manifest(manifest_path)
    directory = manifest_path.parent
    info = read(directory / "run.json")
    require(
        manifest["status"]
        == info["status"]
        == "training_complete_fresh_reload_pending",
        "Run did not finish training.",
    )
    exposure = read(directory / "exposure.json")
    result = committed_counts(
        exposure,
        list(rows(directory / "training-events.jsonl")),
        info["version"] == "gemma-actions-v3-dpo",
    )
    require(
        info["training_updates"] == result["updates"], "Saved run update count differs."
    )
    require(
        info["optimizer_committed_rows"] == result["committed_draws_or_pairs"],
        "Saved run committed count differs.",
    )
    configuration = read(directory / "configuration.json")
    require(
        sha256(directory / "configuration.json") == info["configuration_sha256"],
        "Training configuration changed.",
    )
    return {
        **result,
        "phase": configuration["phase"],
        "format": configuration["format"],
        "kind": info["version"],
        "initialization": info["initialization"],
        "seconds": info["seconds"],
        "peak_allocated_gpu_bytes": info["peak_allocated_gpu_bytes"],
        "adapter_sha256": manifest["files_sha256"]["adapter/adapter_model.safetensors"],
        "declared_train_rows": info.get("train_rows"),
        "declared_draws": info.get("draws"),
        "declared_count_caveat": "Legacy engineering DPO inherited SFT train_rows/draws. Actual pair exposure above comes from optimizer events and is never replaced by those declarations.",
        "reload_verified_by_this_summary": False,
    }


def metric_counts(records):
    needed = sum(row["judgment"] == "edit_required" for row in records)
    preserve = sum(row["judgment"] == "preserve" for row in records)
    shown = sum(row["displayed"] for row in records)
    useful = sum(row["useful"] for row in records)
    pairs = {
        "precision": (useful, shown),
        "recall": (useful, needed),
        "preservation": (sum(row["preserved"] for row in records), preserve),
        "model_preservation": (
            sum(row["preserved"] and row["model_abstained"] for row in records),
            preserve,
        ),
        "coverage": (shown, len(records)),
        "validity": (sum(row["valid"] for row in records), len(records)),
        "outcome_accuracy": (
            sum(row["outcome_match"] for row in records),
            len(records),
        ),
    }
    return {
        name: {"numerator": a, "denominator": b, "value": a / b if b else None}
        for name, (a, b) in pairs.items()
    }


def verify_evaluation_rows(records, observations, requests, info):
    identities = [row["case_id"] for row in requests]
    require(
        identities
        == [row["case_id"] for row in observations]
        == [row["case_id"] for row in records]
        and len(set(identities)) == len(identities)
        and info["requested"] == info["observed"] == len(identities),
        "Every registered request must retain one ordered observation and grade.",
    )
    for observation, request in zip(observations, requests, strict=True):
        require(
            observation["model_identity"] == info["model_identity"]
            and all(
                observation[key] == request[key]
                for key in (
                    "input_sha256",
                    "messages_sha256",
                    "prefix_ids_sha256",
                    "format",
                )
            ),
            "Observed model, input or native prefix changed.",
        )


def evaluation_summary(root, report_path, recorded_workspace=None):
    evidence_path = partial(bound, root, recorded_workspace=recorded_workspace)
    report = read(report_path)
    require(
        report["final_test"] is False,
        "Independent test needs its own registered inventory type.",
    )
    for name in (
        "preparation",
        "observations",
        "inference_manifest",
        "inference_run",
        "graded",
    ):
        evidence_path(report[name])
    records = list(rows(evidence_path(report["graded"])))
    preparation = evidence_path(report["preparation"])
    verify_manifest(preparation)
    run_manifest = verify_manifest(evidence_path(report["inference_manifest"]))
    info = read(evidence_path(report["inference_run"]))
    requests_path = preparation.parent / f"requests-{report['format']}.jsonl"
    require(
        run_manifest["status"] == info["status"] == "inference_complete"
        and sha256(requests_path) == info["requests_sha256"],
        "Incomplete inference or changed request roster.",
    )
    verify_evaluation_rows(
        records,
        list(rows(evidence_path(report["observations"]))),
        list(rows(requests_path)),
        info,
    )
    require(
        len({row["case_id"] for row in records})
        == len(records)
        == report["overall"]["requests"],
        "Scored request count differs.",
    )
    require(
        [row["id"] for row in rows(preparation.parent / "cases.jsonl")]
        == [row["case_id"] for row in records],
        "Reference case roster differs from scored requests.",
    )
    metrics = metric_counts(records)
    for name, value in metrics.items():
        require(
            all(report["overall"][name][key] == item for key, item in value.items()),
            f"Saved metric differs: {name}",
        )
    return {
        "requests": len(records),
        "metrics": metrics,
        "status_counts": dict(Counter(row["status"] for row in records)),
        "model_identity": report["model_identity"],
        "numerical_regime": report["numerical_regime"],
        "condition": report["condition"],
        "format": report["format"],
        "excluded_before_inference": report["excluded_before_inference"],
        "final_test": False,
    }


def final_census(dispositions, requested, excluded):
    indexed = {row["id"]: row for row in dispositions}
    require(len(indexed) == len(dispositions), "Repeated final-test slot.")
    admitted = {
        identity for identity, row in indexed.items() if row["status"] == "admitted"
    }
    require(
        len(set(requested)) == len(requested) and len(set(excluded)) == len(excluded),
        "Repeated final-test request or exclusion.",
    )
    require(
        not set(requested) & set(excluded)
        and set(requested) | set(excluded) == admitted,
        "Every admitted final-test case needs a request or an explicit run exclusion.",
    )
    result = {}
    for dimension in ("source_group", "registered_source_group", "stratum", "language"):
        groups = defaultdict(list)
        for row in dispositions:
            groups[row.get(dimension, row["source_group"])].append(row)
        result[dimension] = {
            key: {
                "registered": len(values),
                "admitted": sum(row["status"] == "admitted" for row in values),
                "requested": sum(row["id"] in requested for row in values),
                "dispositions": dict(Counter(row["status"] for row in values)),
                "natural_windows": len(
                    {row["window_id"] for row in values if row.get("window_id")}
                ),
            }
            for key, values in sorted(groups.items())
        }
    return result


def verify_metrics(records, saved):
    calculated = metric_counts(records)
    require(
        saved["requests"] == len(records)
        and saved["status_counts"] == dict(Counter(row["status"] for row in records)),
        "Saved scored population differs.",
    )
    for name, value in calculated.items():
        require(
            all(saved[name][key] == item for key, item in value.items()),
            f"Saved metric differs: {name}",
        )
    return calculated


def verify_final_execution(root, manifest_path, freeze, sealed_path, run, requests, observations, recorded_workspace=None):
    evidence_path = partial(bound, root, recorded_workspace=recorded_workspace)
    execution = verify_manifest(manifest_path)
    require("observations.jsonl" in execution["files_sha256"], "Execution observations are not manifest-bound.")
    kind = execution.get("execution_kind")
    baseline = run["role"] in {"always_no_action", "consistent_history_rule"}
    require(kind == "deterministic_baseline" if baseline else kind == "remote_native"
            or (kind == "engineering_callback" and freeze.get("engineering_fixture_only") is True),
            "Unadmitted final-test model execution kind.")
    require(execution["requested"] == execution["observed"] == len(requests) == len(observations)
            and [row["case_id"] for row in observations] == [row["case_id"] for row in requests]
            and len({row["case_id"] for row in requests}) == len(requests)
            and execution["reference_labels_available_to_predictor"] is False,
            "Every request needs an ordered, label-free execution outcome.")
    role = "native_base" if run["role"] == "base_icl" else run["role"]
    expected_identity = ({"baseline": role, "code_sha256": freeze["source_sha256"]["ml/prepare_actions_v3_conditions.py"]}
                         if baseline else read(evidence_path(freeze["models"][role]))["identity"])
    require(execution["model_identity"] == expected_identity, "Execution model identity differs from the freeze.")
    for request, row in zip(requests, observations, strict=True):
        request_hash = hashlib.sha256(json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(row["split"] == "final_test" and row["model_identity"] == expected_identity
                and row["request_sha256"] == request_hash
                and all(row[key] == request[key] for key in ("case_id", "input_sha256", "messages_sha256", "prefix_ids_sha256",
                                                           "condition_sha256", "format", "run_id", "freeze_sha256")),
                "Execution observation is not bound to its exact native request.")
    result = {"execution_kind": kind, "requested": len(requests), "observed": len(observations),
              "engineering_fixture_only": freeze.get("engineering_fixture_only", False)}
    if kind != "remote_native":
        require("run-registration.json" in execution["files_sha256"], "Execution registration is not manifest-bound.")
        registration = read(manifest_path.parent / "run-registration.json")
        require(all(execution.get(key) == value for key, value in registration.items())
                and registration["execution_kind"] == kind,
                "Registered controller execution changed.")
        require(all(row["generation_ms"] is None and row["timing_source"] == "controller_only_not_gpu_generation"
                    for row in observations), "Controller execution claimed native generation timing.")
        return {**result, "registration_sha256": sha256(manifest_path.parent / "run-registration.json"),
                "native_model_execution_verified": False}

    require("remote-run.json" in execution["files_sha256"] and {"remote_manifest", "execution_bundle"} <= set(execution),
            "Admitted native receipt is not manifest-bound.")
    remote_path = evidence_path(execution["remote_manifest"])
    portable_path = evidence_path(execution["execution_bundle"])
    remote, portable = verify_manifest(remote_path), verify_manifest(portable_path)
    require(remote["version"] == "task-test-actions-v3-native-inference-1"
            and remote["scope"] == "final_test" and remote["status"] == "inference_complete"
            and {"run.json", "run-registration.json", "observations.jsonl"} <= set(remote["files_sha256"]),
            "Remote native execution evidence is incomplete.")
    require(portable["version"] == "task-test-actions-v3-execution-bundle-1"
            and portable["scope"] == "final_test" and portable["status"] == "authorized_label_free_execution"
            and set(portable["files_sha256"]) == {"execution.json", "requests.jsonl"},
            "Native execution authorization is incomplete.")
    spec = read(portable_path.parent / "execution.json")
    model = read(evidence_path(freeze["models"][role]))
    config = read(evidence_path(model["inference_configuration"]))
    preparation_path = evidence_path(model["preparation_manifest"])
    if model.get("runtime_family") in {"small-actions-final-v1", "gemma26-final-native-v1"}:
        import small_final_test_runtime as small
        import gemma_final_test_runtime as gemma
        backend = gemma if model["runtime_family"] == gemma.FAMILY else small
        metadata = read(preparation_path)
        require(config.get("version") == (gemma.evidence.sft.VERSION if backend is gemma else small.FAMILY)
                and metadata.get("version") == backend.VERSION
                and metadata["status"] == "verified_not_body_authorized"
                and metadata["body_access_authorized"] is False and metadata["final_test_bodies_read"] is False
                and metadata["configuration"] == model["inference_configuration"]
                and metadata["training_dataset_manifest"] == freeze["training_dataset_manifest"],
                "Native authorization lacks its bound pre-test metadata.")
        libraries = config["libraries"] if backend is gemma else metadata["environment"]
        if backend is gemma:
            evidence_path(metadata["native_preparation"])
            require(spec["native_preparation_manifest_sha256"] == metadata["native_preparation"]["sha256"],
                    "Gemma native training preparation changed.")
        require(spec.get("runtime_family") == model["runtime_family"]
                and spec["vocab_size"] == metadata["tokenizer"]["vocab_size"]
                and model["inference_policy"] == metadata["inference_policy"], "Native family/vocabulary changed.")
        policy = backend.validate_inference_policy(read(evidence_path(model["inference_policy"])), config, resolve=evidence_path)
        require(spec["inference_policy_sha256"] == model["inference_policy"]["sha256"]
                and all(spec[key] == policy[key] for key in ("physical_context_tokens", "inference_prefix_limit", "output_tokens")),
                "Native inference policy differs from its prospective registration.")
    else:
        require("runtime_family" not in model and "runtime_family" not in spec, "Unknown native execution family.")
        libraries = config["libraries"]
    require(spec["scope"] == "final_test" and spec["local_gate_passed"] is True and spec["run"] == run
            and spec["freeze_sha256"] == execution["freeze"]["sha256"]
            and spec["sealed_manifest_sha256"] == sha256(sealed_path)
            and spec["controller_source_sha256"] == freeze["source_sha256"]
            and spec["model_identity"] == model["identity"] == execution["model_identity"]
            and spec["configuration_sha256"] == model["inference_configuration"]["sha256"]
            and spec["preparation_manifest_sha256"] == model["preparation_manifest"]["sha256"]
            and spec["generation"] == config["generation"] and spec["libraries"] == libraries
            and spec["schema_constrained"] is False
            and spec["engineering_fixture_only"] == freeze.get("engineering_fixture_only", False),
            "Native authorization differs from the frozen model/configuration.")
    if "execution_runtime" in model:
        runtime_path = evidence_path(model["execution_runtime"])
        runtime = read(runtime_path)
        require(runtime["status"] == "assembled_model_specific_runtime" and runtime["scope"] == "final_test"
                and runtime["final_test_bodies_read"] is False and runtime["model_weights_read"] is False
                and spec["runtime_manifest_sha256"] == model["execution_runtime"]["sha256"]
                and spec["source_sha256"] == runtime["source_sha256"]
                and runtime["configuration_sha256"] == spec["configuration_sha256"]
                and runtime["preparation_manifest_sha256"] == spec["preparation_manifest_sha256"],
                "Native execution runtime differs from the model-specific assembly.")
        for name, expected in runtime["source_sha256"].items():
            require(sha256(resolve_path(runtime_path.parent, name)) == expected, "Native runtime source changed.")
    else:
        require(freeze.get("engineering_fixture_only") is True and spec["runtime_manifest_sha256"] is None
                and spec["source_sha256"]
                and all(freeze["source_sha256"].get(name) == value for name, value in spec["source_sha256"].items()),
                "Real native execution lacks its assembled runtime.")
    request_path = evidence_path(execution["requests"])
    require(spec["requests_sha256"] == portable["files_sha256"]["requests.jsonl"] == sha256(request_path)
            and list(rows(portable_path.parent / "requests.jsonl")) == requests
            and spec["case_ids"] == [row["case_id"] for row in requests]
            and list(rows(remote_path.parent / "observations.jsonl")) == observations,
            "Admitted native requests/observations differ from the original execution.")
    info = read(remote_path.parent / "run.json")
    require(info == read(manifest_path.parent / "remote-run.json")
            and info["version"] == remote["version"] and info["scope"] == "final_test"
            and info["status"] == "inference_complete"
            and info["bundle_sha256"] == execution["execution_bundle"]["sha256"]
            and info["run_id"] == run["id"]
            and all(info[key] == spec[key] for key in (
                "freeze_sha256", "sealed_manifest_sha256", "requests_sha256", "configuration_sha256",
                "model_identity", "source_sha256", "runtime_manifest_sha256", "controller_source_sha256"))
            and info["schema_constrained"] is False and info["final_test_bodies_read"] is True
            and info["reference_labels_available_to_predictor"] is False
            and info["requested"] == info["observed"] == len(requests)
            and type(info["model_identity_verified"]) is bool,
            "Admitted remote execution receipt changed.")
    registration = read(remote_path.parent / "run-registration.json")
    require(registration["status"] == "registered" and registration["model_identity_verified"] is False
            and registration["actual_model_identity"] is None
            and all(info[key] == value for key, value in registration.items()
                    if key not in {"status", "model_identity_verified", "actual_model_identity"}),
            "Remote execution registration changed.")
    verified = info["model_identity_verified"]
    require((info["actual_model_identity"] == spec["model_identity"] and "runtime.json" in remote["files_sha256"])
            if verified else info["actual_model_identity"] is None, "Native loading identity evidence is missing.")
    for request, row in zip(requests, observations, strict=True):
        require(row["model_identity_verified"] is verified and type(row["generation_attempted"]) is bool,
                "Native observation identity verification changed.")
        if row["status"] == "completed":
            tokens = row["output_ids"]
            require(verified and row["generation_attempted"] is True
                    and row["timing_source"] == "cuda_synchronized_generate"
                    and type(row["generation_ms"]) in (int, float)
                    and math.isfinite(row["generation_ms"]) and row["generation_ms"] >= 0
                    and 0 < len(tokens) <= 512
                    and all(type(token) is int and 0 <= token < spec.get("vocab_size", 262144) for token in tokens)
                    and row["terminal_token_id"] == tokens[-1]
                    and row["finish_reason"] == ("stop" if tokens[-1] in spec["generation"]["eos_token_id"] else "length")
                    and isinstance(row["text"], str) and isinstance(row["text_raw"], str),
                    "Completed native output lacks verified execution/timing/token evidence.")
        else:
            require(row["status"] == "request_failure" and row["generation_ms"] is None
                    and row["timing_source"] == "not_measured" and bool(row["error_type"])
                    and not {"text", "text_raw", "output_ids", "terminal_token_id", "finish_reason"}.intersection(row)
                    and row["failure_stage"] in {"generation", "preflight_or_model_loading", "aborted_before_request"}
                    and (verified or (row["failure_stage"] == "preflight_or_model_loading"
                                      and row["generation_attempted"] is False)),
                    "Native failure was disguised as a verified generated answer.")
    return {**result, "remote_manifest": execution["remote_manifest"], "execution_bundle": execution["execution_bundle"],
            "remote_run_sha256": sha256(remote_path.parent / "run.json"),
            "native_model_execution_verified": verified,
            "actual_model_identity": info["actual_model_identity"],
            "failure_stages": dict(Counter(row["failure_stage"] for row in observations if row["status"] == "request_failure"))}


def final_evaluation_summary(root, manifest_path, recorded_workspace=None):
    evidence_path = partial(bound, root, recorded_workspace=recorded_workspace)
    manifest = verify_manifest(manifest_path)
    require(
        manifest["scope"] == "final_test" and manifest["status"] == "graded",
        "Require a graded, independently frozen final-test artifact.",
    )
    freeze = read(evidence_path(manifest["freeze"]))
    require(
        freeze["scope"] == "final_test"
        and freeze["status"] == "frozen"
        and freeze["body_access_authorized"] is True
        and freeze["reserved_bodies_read_before_freeze"] is False,
        "Final-test freeze attestation is missing.",
    )
    sealed_path = evidence_path(manifest["sealed_manifest"])
    execution_path = evidence_path(manifest["observation_manifest"])
    sealed, execution = verify_manifest(sealed_path), verify_manifest(execution_path)
    require(
        sealed["scope"] == execution["scope"] == "final_test"
        and sealed["freeze"] == execution["freeze"] == manifest["freeze"]
        and sealed["status"] == "sealed_before_inference"
        and sealed["predictions_seen"] is False
        and execution["status"] == "inference_complete"
        and execution["run_id"] == manifest["run_id"]
        and execution["sealed_manifest"] == manifest["sealed_manifest"]
        and execution["final_test_bodies_read"] is True,
        "Final-test stages do not share a complete frozen execution.",
    )
    run = next(row for row in freeze["runs"] if row["id"] == manifest["run_id"])
    request_path = evidence_path(execution["requests"])
    require(
        request_path.resolve()
        == (sealed_path.parent / f"requests-{run['id']}.jsonl").resolve(),
        "Execution used another sealed request file.",
    )
    requests = list(rows(request_path))
    observations = list(rows(execution_path.parent / "observations.jsonl"))
    records = list(rows(manifest_path.parent / "graded.jsonl"))
    verify_evaluation_rows(records, observations, requests, execution)
    execution_evidence = verify_final_execution(root, execution_path, freeze, sealed_path, run, requests, observations, recorded_workspace)
    require(manifest.get("execution_evidence") == execution_evidence, "Grade did not retain verified execution evidence.")
    require(
        [row["case_id"] for row in requests] == sealed["case_ids_by_run"][run["id"]]
        and all(
            row["split"] == "final_test"
            and row["run_id"] == run["id"]
            and row["format"] == run["format"]
            and row["freeze_sha256"] == manifest["freeze"]["sha256"]
            for row in requests
        ),
        "Final-test request scope changed.",
    )
    case_rows = list(rows(sealed_path.parent / "cases.jsonl"))
    cases = {row["id"]: row for row in case_rows}
    require(
        len(cases) == len(case_rows) == sealed["admitted_cases"]
        and list(cases) == sealed["case_ids"]
        and sum(len(row["accepted_predictions"]) for row in case_rows)
        == sealed["accepted_references"],
        "Canonical final-test case count differs.",
    )
    for record in records:
        case = cases[record["case_id"]]
        require(
            case["split"] == case["source"]["split"] == "final_test"
            and record["judgment"] == case["judgment"]["status"]
            and record["judgment"] in {"edit_required", "preserve"}
            and record["reference_action"] == case["primary"]["action"]
            and record["source_group"] == case["source"]["source_group"]
            and record["language"] == case["language"],
            "Final-test grade provenance differs.",
        )
    dispositions = list(rows(sealed_path.parent / "dispositions.jsonl"))
    require(
        [row["id"] for row in dispositions]
        == [row["slot_id"] for row in rows(evidence_path(freeze["roster"]))],
        "Final-test disposition roster changed.",
    )
    require(
        {row["id"] for row in dispositions if row["status"] == "admitted"}
        == set(cases),
        "Unknown or unavailable final-test slot became a scored case.",
    )
    exclusions = [
        row
        for row in rows(sealed_path.parent / "run-exclusions.jsonl")
        if row["run_id"] == run["id"]
    ]
    census = final_census(
        dispositions,
        [row["case_id"] for row in requests],
        [row["id"] for row in exclusions],
    )
    report = read(manifest_path.parent / "report.json")
    require(
        report["scope"] == "final_test"
        and report["run_id"] == run["id"]
        and report["model_selection_changed"] is False
        and report["engineering_fixture_only"]
        == freeze.get("engineering_fixture_only", False)
        and report["registered_slots"]
        == sealed["registered_slots"]
        == len(dispositions)
        and report["canonical_admitted_cases"] == len(cases)
        and report["disposition_counts"]
        == dict(Counter(row["status"] for row in dispositions))
        and report["run_eligibility_exclusions"] == exclusions
        and report["source_stratum_denominators"] == census,
        "Final-test report changed its cohort, selection or denominator.",
    )
    require(
        report["source_groups"] == len({row["source_group"] for row in dispositions})
        and report["registered_source_groups"]
        == len(
            {
                row.get("registered_source_group", row["source_group"])
                for row in dispositions
            }
        )
        and report["natural_windows"]
        == len({row["window_id"] for row in dispositions if row.get("window_id")}),
        "Final-test source or independent window count differs.",
    )
    metrics = verify_metrics(records, report["summary"])
    for dimension, groups in report["by"].items():
        grouped = defaultdict(list)
        for record in records:
            grouped[
                record[dimension] if record[dimension] is not None else "unavailable"
            ].append(record)
        require(
            set(grouped) == set(groups), "Final-test subgroup was omitted or added."
        )
        for key, values in grouped.items():
            verify_metrics(values, groups[key])
    return {
        "scope": "final_test",
        "requests": len(records),
        "registered_slots": len(dispositions),
        "canonical_admitted_cases": len(cases),
        "accepted_references": sealed["accepted_references"],
        "run_eligibility_exclusions": len(exclusions),
        "metrics": metrics,
        "source_stratum_denominators": census,
        "model_identity": execution["model_identity"],
        "execution_evidence": execution_evidence,
        "run": run,
        "engineering_fixture_only": report["engineering_fixture_only"],
        "semantic_judgments_and_execution_not_rerun": True,
    }


def recompute(registry_path, root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), "Use a fresh summary output.")
    registry = read(registry_path)
    recorded_workspace = registry.get("recorded_workspace")
    entries = registry["entries"]
    require(
        len({item["id"] for item in entries}) == len(entries),
        "Duplicate registry entries.",
    )
    results = []
    for item in entries:
        if item["state"] != "recorded":
            require(
                item["state"] in ("planned", "running", "pending"),
                "Unknown work state.",
            )
            results.append({**item, "verified": False})
            continue
        path = bound(root, item["evidence"], recorded_workspace)
        kind = item["kind"]
        if kind == "dataset":
            summary = dataset_summary(path)
        elif kind == "training":
            summary = training_summary(path)
        elif kind == "development":
            summary = evaluation_summary(root, path, recorded_workspace)
        elif kind == "final_test":
            summary = final_evaluation_summary(root, path, recorded_workspace)
        else:
            require(kind == "receipt", "Unknown evidence kind.")
            summary = {
                "status": "bound receipt hash verified; contents not independently rerun"
            }
        results.append({**item, "verified": True, "recomputed": summary})
    result = {
        "version": "passive-actions-v3-evidence-recomputation-1",
        "registry_sha256": sha256(registry_path),
        "implementation_sha256": sha256(__file__),
        "entries": results,
        "unrecorded_work": [
            item["id"] for item in entries if item["state"] != "recorded"
        ],
        "quality_goal_achievement": "Determined by separately registered selection and independent evaluation, never by this count recomputation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {
        "verified_entries": sum(item["verified"] for item in results),
        "unrecorded_work": result["unrecorded_work"],
        "output": str(output),
        "sha256": sha256(output),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recompute(args.registry, args.root, args.output), indent=2))
