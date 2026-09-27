"""Run hash-bound B2 conditions through the existing single-predictor runtime."""

import argparse
import copy
import fcntl
import gzip
import json
from pathlib import Path

import passive_b2_benchmark as b2
import probe_passive as probe
from passive_benchmark import evaluate_result
from replay_session import session_hash

ROOT = probe.ROOT
DEFAULT_DIRECTORY = ROOT / "artifacts/benchmarks/passive-b2-v1"
POLICY = {
    "context_tokens": 4096, "input_tokens": 3584, "output_tokens": 512,
    "temperature": 0, "seed": 42, "reasoning": "off", "schema_constrained": True,
    "cache_requested": False, "request_timeout_seconds": 120, "automatic_retries": 0,
}
GRADE_FIELDS = (
    "format_valid", "raw_valid", "prediction", "raw_grade", "displayed",
    "displayed_grade", "suppression_reason", "error", "error_type",
)
REQUEST_ERRORS = {"request_failure", "timeout", "runtime_mismatch", "cancelled"}


def checked(path, expected):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or probe.digest(path) != expected:
        raise ValueError(f"Changed or unbound B2 project file: {path}")
    return path


def read_rows(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def unique_rows(rows, ids, label):
    if [row["case_id"] for row in rows] != ids:
        raise ValueError(f"{label} must preserve the complete ordered B2 roster")
    return {row["case_id"]: row for row in rows}


def validate_execution(args, protocol, prefixes):
    registry = json.loads(args.registry.read_text())
    execution = json.loads(args.execution.read_text())
    for key, path in (("protocol_sha256", args.protocol),
                      ("prefix_manifest_sha256", args.prefix_manifest),
                      ("registry_sha256", args.registry)):
        if execution.get(key) != probe.digest(path):
            raise ValueError(f"B2 execution does not bind {key}")
    selected = execution.get("selected_candidate_ids")
    anchor = execution.get("anchor_candidate_id")
    candidates = {row["id"]: row for row in registry["candidates"]}
    if (not isinstance(selected, list) or len(selected) not in (2, 3)
            or any(not isinstance(name, str) for name in selected)
            or len(set(selected)) != len(selected) or anchor not in selected
            or not set(selected) <= set(candidates)
            or set(selected) & set(probe.BASELINES)
            or len(candidates) != len(registry["candidates"])):
        raise ValueError("Bind two or three unique model candidates and one selected anchor")
    if args.candidate in probe.BASELINES:
        if args.condition != "full" or args.candidate not in protocol["baseline_candidates"]:
            raise ValueError("Deterministic B2 baselines only use the full condition")
    elif (args.candidate not in selected
          or args.condition in ("history_only", "icl") and args.candidate != anchor):
        raise ValueError("B2 candidate or condition is outside the selected anchor policy")
    if ((args.condition in ("context_only", "history_only"))
            != (args.reuse_full_run is not None)):
        raise ValueError("Context/history conditions require --reuse-full-run; other conditions forbid it")
    if any(protocol["runtime"].get(key) != value for key, value in POLICY.items()):
        raise ValueError("B2 protocol differs from the fixed product-budget policy")
    if (args.timeout != POLICY["request_timeout_seconds"] or args.unconstrained
            or args.sampling_config is not None):
        raise ValueError("B2 schema, timeout and greedy sampling cannot be overridden")
    sampling = probe.sampling_options(args.registry)
    if (sampling["temperature"] != POLICY["temperature"]
            or sampling["seed"] != POLICY["seed"]
            or sampling["max_tokens"] != POLICY["output_tokens"]):
        raise ValueError("B2 registry sampling differs from its registered policy")
    tokenizer_binding = protocol["tokenizer_registry"]
    if (registry["tokenizers"]["registry_path"] != tokenizer_binding["path"]
            or registry["tokenizers"]["registry_sha256"] != tokenizer_binding["sha256"]
            or prefixes.get("tokenizer_registry") != tokenizer_binding):
        raise ValueError("B2 tokenizer registry identity differs between bindings")
    tokenizer_path = checked(ROOT / tokenizer_binding["path"], tokenizer_binding["sha256"])
    tokenizers = json.loads(tokenizer_path.read_text())["candidates"]
    if (prefixes["candidates"] != tokenizers
            or prefixes.get("candidate_ids") != [row["id"] for row in tokenizers]
            or len(tokenizers) != 5 or len({row["id"] for row in tokenizers}) != 5):
        raise ValueError("B2 prefixes must bind all five frozen tokenizer identities")
    for name in selected:
        candidate = candidates[name]
        tokenizer = next((row for row in tokenizers
                          if row["id"] == candidate.get("tokenizer_id", name)), None)
        if tokenizer is None or any(candidate.get(key) != value for key, value in {
            "base_repository": tokenizer["repository"], "base_revision": tokenizer["revision"],
            "tokenizer_path": tokenizer["path"], "tokenizer_files_sha256": tokenizer["files_sha256"],
            "template_path": str(Path(tokenizer["path"]) / "chat_template.jinja"),
            "template_sha256": tokenizer["files_sha256"]["chat_template.jinja"],
        }.items()):
            raise ValueError("B2 model identity differs from its frozen native tokenizer")
    if (registry["runtime"]["context_tokens"] != POLICY["context_tokens"]
            or registry["runtime"]["input_tokens"] != POLICY["input_tokens"]
            or registry["runtime"]["chat_template_kwargs"] != {"enable_thinking": False}
            or registry["runtime"]["parallel_slots"] != 1):
        raise ValueError("B2 runtime differs from its context or non-thinking policy")
    return registry, sampling


def load_requests(args, protocol):
    cases = b2.load_dev(args.release)
    ids = [case["id"] for case in cases]
    if (len(ids) != b2.DEV_REQUESTS or len(set(ids)) != len(ids)
            or protocol["registered_requests"] != b2.DEV_REQUESTS
            or protocol["case_ids"] != ids or set(protocol["conditions"]) != set(b2.CONDITIONS)):
        raise ValueError("B2 requires the full original 289-case roster and all conditions")
    registry_path = checked(args.protocol.parent / "case-registry.jsonl",
                            protocol["case_registry_sha256"])
    request_path = checked(args.protocol.parent / "requests.jsonl.gz", protocol["requests_sha256"])
    registration = unique_rows(read_rows(registry_path), ids, "B2 case registry")
    requests = {condition: {} for condition in b2.CONDITIONS}
    for row in read_rows(request_path):
        condition, identity = row["condition"], row["case_id"]
        if (condition not in requests or identity not in registration
                or identity in requests[condition]
                or row["messages_sha256"] != session_hash(row["messages"])):
            raise ValueError("Duplicate, unregistered or changed B2 request messages")
        requests[condition][identity] = row
    if any(set(rows) != set(ids) for rows in requests.values()):
        raise ValueError("Every B2 message condition must retain all 289 cases")
    demonstrations = b2.select_demonstrations(args.release)
    if {language: [case["id"] for case in values]
            for language, values in demonstrations.items()} != protocol["icl"]["demo_case_ids"]:
        raise ValueError("B2 demonstrations differ from the train-only preregistration")
    views = []
    for case in cases:
        identity, metadata = case["id"], case["metadata"]
        registered = registration[identity]
        if any(registered[key] != metadata[key] for key in ("source_group_id", "episode_id", "language")):
            raise ValueError("B2 case metadata differs from the frozen source")
        if set(registered["conditions"]) != set(b2.CONDITIONS):
            raise ValueError("B2 registry must describe all four conditions")
        for condition in b2.CONDITIONS:
            view = b2.condition_case(case, condition)
            messages = b2.condition_messages(
                view, condition, demonstrations[metadata["language"]] if condition == "icl" else ()
            )
            row = requests[condition][identity]
            expected = {
                **view["_b2"], "messages_sha256": session_hash(messages),
                "identical_to_full_request": messages == requests["full"][identity]["messages"],
            }
            if row["messages"] != messages or registered["conditions"][condition] != expected:
                raise ValueError("B2 messages or evidence masks differ from the registered condition")
            if condition == args.condition:
                views.append(view)
    return cases, views, requests


def load_prefixes(args, protocol, manifest, requests, tokenizer_id):
    ids, directory = protocol["case_ids"], args.prefix_manifest.parent
    expected_files = {f"{candidate['id']}/{condition}.jsonl.gz"
                      for candidate in manifest["candidates"] for condition in b2.CONDITIONS}
    eligibility_name = manifest["common_icl_eligibility_file"]
    if (eligibility_name != "common-icl-eligibility.json"
            or set(manifest["files_sha256"]) != expected_files | {eligibility_name}
            or set(manifest["prefix_counts"]) != set(manifest["candidate_ids"])
            or set(manifest["full_frozen_parity"]) != set(manifest["candidate_ids"])):
        raise ValueError("B2 prefix manifest has missing or unexpected files/candidates")
    selected, full_selected, counts = {}, {}, {}
    for candidate in manifest["candidates"]:
        name = candidate["id"]
        for filename, expected in candidate["files_sha256"].items():
            checked(ROOT / candidate["path"] / filename, expected)
        config = json.loads((ROOT / candidate["path"] / "config.json").read_text())
        vocabulary_size = config.get("text_config", config)["vocab_size"]
        frozen = probe.frozen_prefixes(args.release, name)
        if set(frozen) != set(ids) or manifest["full_frozen_parity"][name] != {
            "passed": len(ids), "expected": len(ids),
        } or set(manifest["prefix_counts"][name]) != set(b2.CONDITIONS):
            raise ValueError("B2 full prefixes do not bind the complete frozen token roster")
        counts[name] = {}
        for condition in b2.CONDITIONS:
            filename = f"{name}/{condition}.jsonl.gz"
            path = checked(directory / filename, manifest["files_sha256"][filename])
            rows = unique_rows(read_rows(path), ids, "B2 prefix file")
            measured = {}
            for identity, row in rows.items():
                tokens = row["input_ids"]
                if (not isinstance(tokens, list) or not tokens
                        or any(type(token) is not int or not 0 <= token < vocabulary_size for token in tokens)
                        or row["messages_sha256"] != requests[condition][identity]["messages_sha256"]
                        or condition != "icl" and len(tokens) > POLICY["input_tokens"]
                        or condition == "full" and tokens != frozen[identity]):
                    raise ValueError("Invalid, changed or message-mismatched B2 prefix tokens")
                measured[identity] = len(tokens)
            if measured != manifest["prefix_counts"][name][condition]:
                raise ValueError("B2 prefix token counts differ from the manifest")
            if condition in ("full", "icl"):
                counts[name][condition] = measured
            if name == tokenizer_id:
                if condition == args.condition:
                    selected = {identity: row["input_ids"] for identity, row in rows.items()}
                if condition == "full":
                    full_selected = {identity: row["input_ids"] for identity, row in rows.items()}
    eligibility_path = checked(directory / eligibility_name, manifest["common_icl_eligibility_sha256"])
    if manifest["files_sha256"][eligibility_name] != manifest["common_icl_eligibility_sha256"]:
        raise ValueError("B2 eligibility file has conflicting hashes")
    eligibility = json.loads(eligibility_path.read_text())
    expected_eligibility = {
        **b2.common_input_eligibility(ids, counts), "candidate_ids": manifest["candidate_ids"],
        "candidate_basis": "All five preregistered tokenizers, fixed before B2 candidate choice.",
    }
    if (eligibility != expected_eligibility
            or manifest["common_icl_eligible_requests"] != len(eligibility["eligible_case_ids"])
            or manifest["common_icl_excluded_requests"] != len(eligibility["excluded"])):
        raise ValueError("B2 ICL eligibility differs from all five frozen tokenizer budgets")
    return selected, full_selected, eligibility


def reused_results(args, views, requests, full_prefixes, binding, registry, sampling):
    if args.reuse_full_run is None:
        return {}
    source = args.reuse_full_run.resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError("Reuse source must be a saved B2 project run")
    manifest_path, results_path = source / "manifest.json", source / "results.jsonl"
    manifest = json.loads(manifest_path.read_text())
    ids = [case["id"] for case in views]
    model = next(row for row in registry["candidates"] if row["id"] == args.candidate)
    tokenizer_registry = json.loads((ROOT / registry["tokenizers"]["registry_path"]).read_text())
    tokenizer = next(row for row in tokenizer_registry["candidates"]
                     if row["id"] == model.get("tokenizer_id", args.candidate))
    expected = {
        "version": "passive-b2-runtime-v1", "status": "completed", "stage": "screen",
        "candidate": args.candidate, "condition": "full", "case_ids": ids,
        "completed_requests": len(ids), "reused_results": 0,
        "registry_sha256": binding["registry_sha256"], "sampling": sampling,
        "candidate_binding": model, "tokenizer_binding": tokenizer,
        "runtime_binding": registry["runtime"], "schema_constrained": True,
        "release_manifest_sha256": binding["release_manifest_sha256"],
        "screening_protocol_sha256": probe.digest(args.release / "screening-protocol.json"),
        "reasoning": "off; pinned native template enable_thinking=false",
        "context_tokens": POLICY["context_tokens"], "cache_requested": False,
        "request_timeout_seconds": POLICY["request_timeout_seconds"],
        "template_checks": {"requests": len(ids), "passed": len(ids)},
    }
    benchmark = manifest.get("benchmark", {})
    if (any(manifest.get(key) != value for key, value in expected.items())
            or benchmark.get("condition") != "full"
            or any(benchmark.get(key) != binding[key] for key in (
                "protocol_sha256", "prefix_manifest_sha256", "execution_sha256", "registry_sha256",
                "release_manifest_sha256", "registered_case_ids", "case_registry_sha256", "requests_sha256",
            )) or benchmark.get("messages_sha256") != {
                identity: row["messages_sha256"] for identity, row in requests["full"].items()
            }):
        raise ValueError("Reuse requires a completed matching full289 B2 run and bindings")
    implementation = source / "implementation"
    for name, expected_hash in benchmark["snapshot_sha256"].items():
        checked(implementation / name, expected_hash)
    checked(implementation / "model-registry.json", binding["registry_sha256"])
    rows = unique_rows(read_rows(results_path), ids, "Reused full run results")
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    if any(summary.get(key) != value for key, value in {
        "registered_requests": len(ids), "observed_results": len(ids),
        "missing_requests": 0, "input_budget_excluded": 0, "reused_quality_rows": 0,
    }.items()):
        raise ValueError("Reuse source summary does not contain all full289 observations")
    provenance = {
        "run": str(source.relative_to(ROOT)), "manifest_sha256": probe.digest(manifest_path),
        "results_sha256": probe.digest(results_path), "summary_sha256": probe.digest(summary_path),
        "condition": "full",
    }
    reused = {}
    for view in views:
        identity = view["id"]
        original = rows[identity]
        failed = (original.get("request_failed") or original.get("cancelled")
                  or original.get("error_type") in REQUEST_ERRORS
                  or isinstance(original.get("http_status"), int)
                  and not 200 <= original["http_status"] < 300)
        if not failed:
            choice = original["response"]["choices"][0]
            if (not isinstance(original.get("raw_text"), str)
                    or original["raw_text"] != (choice["message"].get("content") or "")
                    or original.get("finish_reason") != choice["finish_reason"]
                    or original["response"]["usage"]["prompt_tokens"] != len(full_prefixes[identity])):
                raise ValueError("Reuse source response differs from its saved text or frozen prompt")
        if requests[args.condition][identity]["messages"] != requests["full"][identity]["messages"]:
            continue
        row = copy.deepcopy(original)
        if failed:
            row.update(raw_valid=False, displayed=False, prediction=None,
                       raw_grade=None, displayed_grade=None, suppression_reason=None)
        else:
            for field in GRADE_FIELDS:
                row.pop(field, None)
            row.update(evaluate_result(view, original["raw_text"], original["finish_reason"]))
        row.update(complete_json_ms=None, inherited_complete_json_ms=original.get("complete_json_ms"),
                   http_response_ms=None, inherited_http_response_ms=original.get("http_response_ms"),
                   post_response_processing_ms=None,
                   inherited_post_response_processing_ms=original.get("post_response_processing_ms"),
                   reused_from={**provenance, "messages_sha256": requests["full"][identity]["messages_sha256"]})
        reused[identity] = row
    return reused


def prepare(args):
    for key in ("release", "registry", "protocol", "execution", "prefix_manifest"):
        path = getattr(args, key).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("B2 inputs must remain inside the project for source snapshots")
        setattr(args, key, path)
    input_hashes = {key: probe.digest(getattr(args, key))
                    for key in ("registry", "protocol", "execution", "prefix_manifest")}
    protocol = json.loads(args.protocol.read_text())
    prefixes = json.loads(args.prefix_manifest.read_text())
    if (protocol.get("version") != "passive-b2-preregistration-v1"
            or protocol.get("b2_predictions_consulted") is not False
            or protocol.get("final_test_bodies_read") is not False
            or prefixes.get("version") != "passive-b2-prefixes-v1"
            or prefixes.get("status") != "completed"
            or prefixes.get("protocol_sha256") != probe.digest(args.protocol)
            or prefixes.get("case_ids") != protocol["case_ids"]
            or prefixes.get("conditions") != list(b2.CONDITIONS)
            or any(prefixes.get(key) != protocol[key] for key in (
                "release_manifest_sha256", "case_registry_sha256", "requests_sha256",
            ))):
        raise ValueError("B2 protocol or complete prefix manifest binding differs")
    checked(args.release / "manifest.json", protocol["release_manifest_sha256"])
    registry, sampling = validate_execution(args, protocol, prefixes)
    _, views, requests = load_requests(args, protocol)
    model = next((row for row in registry["candidates"] if row["id"] == args.candidate), None)
    tokenizer_id = model.get("tokenizer_id", args.candidate) if model else None
    tokens, full_tokens, eligibility = load_prefixes(args, protocol, prefixes, requests, tokenizer_id)
    source_files = [Path(__file__).resolve(), Path(b2.__file__).resolve(),
                    args.protocol, args.execution, args.prefix_manifest]
    names = [path.name for path in source_files]
    if len(names) != len(set(names)) or set(names) & {
        "probe_passive.py", "passive_benchmark.py", "probe.py", "context_feedback_report.py",
        "model-registry.json", "sampling-config.json",
    }:
        raise ValueError("B2 source snapshots require unique non-conflicting basenames")
    binding = {
        "condition": args.condition, "protocol_sha256": probe.digest(args.protocol),
        "prefix_manifest_sha256": probe.digest(args.prefix_manifest),
        "execution_sha256": probe.digest(args.execution), "registry_sha256": probe.digest(args.registry),
        "release_manifest_sha256": protocol["release_manifest_sha256"],
        "case_registry_sha256": protocol["case_registry_sha256"], "requests_sha256": protocol["requests_sha256"],
        "registered_requests": b2.DEV_REQUESTS, "registered_case_ids": protocol["case_ids"],
        "messages_sha256": {identity: row["messages_sha256"]
                            for identity, row in requests[args.condition].items()},
        "snapshot_sha256": {path.name: probe.digest(path) for path in source_files},
        "common_icl_eligibility_sha256": prefixes["common_icl_eligibility_sha256"],
        "icl_budget_basis": "All five preregistered tokenizer candidates; never narrowed after model selection",
    }
    reused = reused_results(args, views, requests, full_tokens, binding, registry, sampling)
    selected_ids = set(eligibility["eligible_case_ids"]) if args.condition == "icl" else set(protocol["case_ids"])
    selected = [case for case in views if case["id"] in selected_ids]
    if any(probe.digest(getattr(args, key)) != expected for key, expected in input_hashes.items()):
        raise ValueError("B2 execution inputs changed during preparation")

    def summarize(_requested_cases, results):
        return b2.summarize_b2(views, results, eligibility=eligibility if args.condition == "icl" else None)

    return {
        "cases": selected, "messages": {case["id"]: requests[args.condition][case["id"]]["messages"]
                                          for case in selected},
        "prefixes": {case["id"]: tokens[case["id"]] for case in selected} if model else {},
        "summarize": summarize, "source_files": source_files, "reused_results": reused,
        "binding": binding,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=probe.DEFAULT_RELEASE)
    parser.add_argument("--registry", type=Path, default=probe.DEFAULT_REGISTRY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_DIRECTORY / "protocol.json")
    parser.add_argument("--prefix-manifest", type=Path, default=DEFAULT_DIRECTORY / "prefixes/manifest.json")
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--condition", choices=b2.CONDITIONS, required=True)
    parser.add_argument("--reuse-full-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "screen"), default="screen")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--editor-state", default="Not controlled; no editor-preview latency claim")
    args = parser.parse_args(argv)
    args.unconstrained, args.sampling_config, args.diagnostic_protocol = False, None, None
    lock_path = ROOT / "artifacts/nep-passive-b1.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another passive benchmark owns the local predictor")
        probe.run(args, prepared=prepare(args))


if __name__ == "__main__":
    main()
