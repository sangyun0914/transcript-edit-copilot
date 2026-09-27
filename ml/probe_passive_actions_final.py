"""Evaluate the final deployed checkpoint on a prereviewed exposed large-dev roster.

Runs exactly one local model under the existing shared lock. No training, reference
changes, held-out inputs, or automatic retries are performed. Pilot14 evidence is
reused only as a separately identified, disjoint same-checkpoint stratum.
"""

import argparse
import asyncio
import copy
import fcntl
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

import passive_actions as actions
import prepare_passive_actions_final as prep
import probe_passive_actions as pilot
from replay_session import session_hash
from verify_release_export import verify_operation

ROOT = prep.ROOT
BENCHMARK = ROOT / "artifacts/benchmarks/passive-actions-final-v1"
FINAL14 = ROOT / "artifacts/benchmarks/passive-actions-v1/final11-evaluation-attempt-01"
FINAL_PROTOCOL = FINAL14 / "registration-attempt-01/protocol.json"
FINAL_REPORT = FINAL14 / "comparison-attempt-01.json"
FINAL_REPORT_SHA = "1271378d1910cd4ea9ed4b5d21bf7ee7eead966916061cc0766e558c44bc4225"
FINAL_RUNNER_SHA = "5228b9683ac0b51199e5eb518734a5b38d63c624d90d453fe1018e77576345f5"
FINAL_MODEL_SHA = "5baa80d78bd683142a200b48ce32012baa25db8d5f963b9b66845f3b631feb7e"
VERSION = "passive-actions-final-development-evaluation-v1"
CONDITION = "final11-seven-action-large-development"
CUTS = prep.CUTS
require, sha, digest, record, rows = prep.require, prep.sha, prep.digest, prep.record, prep.rows


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def bound(value):
    path = Path(value["path"])
    require(path.resolve() == path and path.is_relative_to(ROOT) and path.is_file(), "Bound metadata escaped project or uses an alias.")
    require(path.stat().st_size < 128 * 1024**2 and sha(path) == value["sha256"], "Bound small input changed.")
    return path


def read(value):
    return json.loads(bound(value).read_text())


def final_export_gate(original):
    path = FINAL14 / "evaluate_final11.py"
    require(sha(path) == FINAL_RUNNER_SHA, "Verified final11 metadata validator changed.")
    spec = importlib.util.spec_from_file_location("frozen_final11_evidence_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lineage = original["continuation"]
    module.export_gate(lineage["training_verification"], lineage["export_result"], lineage["independent_export_verification"])


def verify_preparation(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["version"] == prep.VERSION and manifest["contract_version"] == actions.VERSION
            and manifest["census"]["model_requests"] == 0 and manifest["census"]["final_test_bodies_read"] == 0,
            "Preparation scope differs.")
    require({p.name for p in directory.iterdir()} == {"manifest.json", *manifest["files_sha256"]}, "Unexpected preparation payload.")
    for name, value in manifest["files_sha256"].items():
        require(Path(name).name == name and sha(directory / name) == value, "Prepared payload changed.")
    for name, value in manifest["sources_sha256"].items():
        require(sha(ROOT / name) == value, "Prepared source changed.")
    bound(json.loads((directory / "tokenizer.json").read_text())["registry"])
    cases, runtime, native = (rows(directory / name) for name in
                              ("cases.jsonl.gz", "runtime-inputs.jsonl.gz", "native-prefixes.jsonl.gz"))
    ids = [c["id"] for c in cases]
    require(len(ids) == len(set(ids)) == 289 and ids == manifest["case_ids"]
            and [r["id"] for r in runtime] == ids and [r["id"] for r in native] == ids,
            "Incomplete, duplicate or reordered development roster.")
    require(sum(c["provenance"]["prior_matched286"] for c in cases) == 286, "Prior286 membership changed.")
    for case, item, token in zip(cases, runtime, native, strict=True):
        require(case["split"] == item["split"] == token["split"] == "dev"
                and case["input"] == case["bundle"]["input"]
                and token["messages"] == actions.canonical_messages(case["input"])
                and session_hash(item["document"]) == case["bundle"]["document_hash"]
                and item["caption_profile"] == prep.PROFILE
                and len(token["primary"]["input_ids"]) == token["primary"]["completion_start"] <= 3584,
                "Prepared case/runtime/prefix linkage differs.")
        require(token["prefix_ids_sha256"] == digest(token["primary"]["input_ids"]), "Prefix binding changed.")
        require(case["applicability"]["reference_altered"] is False
                and case["accepted_predictions"] == [r["prediction"] for r in case["reference_outcomes"]],
                "A frozen reference was relabelled.")
    return manifest, cases, runtime, native


def register(preparation, review_record, output):
    require(output.parent == BENCHMARK and not output.exists(), "Use a new owned registration directory.")
    manifest, cases, _, _ = verify_preparation(preparation)
    review = read(review_record)
    require(review.get("status") == "passed" and review.get("manifest_sha256") == sha(preparation / "manifest.json")
            and review.get("model_predictions_consulted") is False
            and review.get("cases_reviewed") == 289, "Require independent, pre-prediction all289 applicability review.")
    original = json.loads(FINAL_PROTOCOL.read_text())
    final_export_gate(original)
    report = read({"path": str(FINAL_REPORT), "sha256": FINAL_REPORT_SHA})
    require(report["status"] == "complete_comparison", "Final14 comparison was not completed.")
    arm = copy.deepcopy(original["arms"][0])
    config = pilot.LocalConfig.model_validate(read(arm["configuration"]))
    require(config.model_sha256 == FINAL_MODEL_SHA and config.contract_version == actions.VERSION
            and config.cache_prompt is False, "Require unchanged final11 native runtime and cache-off policy.")
    for name in ("launcher", "parity", "paired_export_receipt"):
        bound(arm[name])
    old_cases = rows(bound(original["cases"]))
    require(len(old_cases) == 14 and not {c["id"] for c in cases} & {c["id"] for c in old_cases}, "Pilot14 overlaps large development.")
    final_results = FINAL14 / "run-attempt-01/verified_final_update_11_Q4_K_M/results.jsonl"
    require(report["protocol"] == record(FINAL_PROTOCOL)
            and report["result_files_sha256"][str(final_results.relative_to(ROOT))] == sha(final_results),
            "Independent final14 report does not bind the reused protocol/results.")
    final_rows = rows(final_results)
    require([r["case_id"] for r in final_rows] == [c["id"] for c in old_cases]
            and all(r.get("gateway", {}).get("model", {}).get("sha256") == FINAL_MODEL_SHA for r in final_rows),
            "Reused pilot predictions belong to another checkpoint or cohort.")
    sources = prep.source_closure([Path(__file__), ROOT / "ml/probe_passive_actions.py", actions.PROMPT_PATH,
                                   old_contract_prompt(), ROOT / "server/app/copilot.py"])
    sources[str((FINAL14 / "evaluate_final11.py").relative_to(ROOT))] = FINAL_RUNNER_SHA
    protocol = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                "condition": CONDITION, "contract_version": actions.VERSION,
                "preparation": record(preparation / "manifest.json"), "independent_review": review_record,
                "final14_protocol": record(FINAL_PROTOCOL), "final14_report": record(FINAL_REPORT),
                "final14_results": record(final_results), "final14_cases": original["cases"],
                "arm": arm, "runtime_registry": original["runtime_registry"],
                "case_ids": manifest["case_ids"], "requests": 289, "new_pilot_requests": 0,
                "prior_matched_case_ids": [c["id"] for c in cases if c["provenance"]["prior_matched286"]],
                "sources_sha256": sources,
                "policy": {"max_input_tokens": 3584, "context_tokens": 4096, "max_output_tokens": 512,
                           "temperature": 0, "seed": 42, "schema_constrained": True, "cache_prompt": False,
                           "icl": False, "one_local_model_at_a_time": True,
                           "runtime_windows": "Exact deployed PassiveContextBuilder; no reference-dependent cropping.",
                           "reference_labels": "Unchanged;20 conversational preserve labels initially flagged for scope review; final exact statuses/counts bind preparation.",
                           "metrics": "Known-reference precision/recall/F1, conditional legacy preservation and separately re-admitted full-action preservation; all request failures retained.",
                           "aggregation": "Same final checkpoint/settings and disjoint states only. Large289, historical286 subset and authored14 reported separately; descriptive303 accepted-reference micro counts with ambiguity explicit.",
                           "timing": "Raw HTTP median and nearest-rank p95 over every measured request; gateway timing separate from actual editor presentation.",
                           "natural_audio_cut_quality_claim": False, "final_test": False},
                "census": manifest["census"]}
    output.mkdir()
    prep.write(output / "protocol.json", protocol)
    return record(output / "protocol.json")


def old_contract_prompt():
    return ROOT / "ml/prompt-passive-v1.txt"


def validate_registration(path, expected):
    require(sha(path) == expected, "Registered protocol changed.")
    protocol = json.loads(path.read_text())
    require(protocol["version"] == VERSION and protocol["requests"] == 289, "Unknown evaluation protocol.")
    for name, value in protocol["sources_sha256"].items():
        require(sha(ROOT / name) == value, "Evaluation implementation changed.")
    manifest_path = bound(protocol["preparation"])
    manifest, cases, runtime, native = verify_preparation(manifest_path.parent)
    review = read(protocol["independent_review"])
    require(review["status"] == "passed" and review["manifest_sha256"] == protocol["preparation"]["sha256"], "Applicability review changed.")
    require([c["id"] for c in cases] == protocol["case_ids"] and manifest["census"] == protocol["census"], "Registered cohort changed.")
    original = read(protocol["final14_protocol"])
    final_export_gate(original)
    read(protocol["final14_report"])
    bound(protocol["final14_results"])
    bound(protocol["final14_cases"])
    for item in protocol["arm"].values():
        if isinstance(item, dict) and "sha256" in item:
            bound(item)
    config = pilot.LocalConfig.model_validate(read(protocol["arm"]["configuration"]))
    require(config.model_sha256 == FINAL_MODEL_SHA and config.contract_version == actions.VERSION and config.cache_prompt is False,
            "Deployment settings differ from fixed final11.")
    pilot.verify_runtime_registry(bound(protocol["runtime_registry"]))
    return protocol, cases, runtime, native, config


class CapturingAdapter(pilot.CapturingAdapter):
    def _json(self, client, method, endpoint, **kwargs):
        value = super()._json(client, method, endpoint, **kwargs)
        if endpoint == "/apply-template" and self.case_id is not None:
            expected = self.expected[self.case_id]
            if kwargs.get("json", {}).get("messages") == expected["messages"]:
                self.last_rendered = value["prompt"]
                if self.last_rendered != expected["rendered_prompt"]:
                    self.fail_integrity("Server-rendered prompt differs from exact native rendering.")
        return value

    async def generate(self, messages, output_schema, timeout):
        try:
            return await super().generate(messages, output_schema, timeout)
        finally:
            self.observation["server_rendered_prompt"] = getattr(self, "last_rendered", None)


def source_result(case, runtime, gateway):
    before, bundle = runtime["document"], case["bundle"]
    prediction = gateway["prediction"]
    require(gateway["status"] in {"proposal", "no_action", "suppressed"}, "Unknown gateway outcome.")
    identity = actions.Envelope(**gateway["request"])
    validated = actions.validate_prediction(prep.packed(prediction), bundle["input"], identity, identity, "stop")
    require(validated.model_dump() == prediction and gateway["context"] == bundle["context"],
            "Gateway prediction or issued context differs.")
    saved = runtime.get("feedback_journal")
    ledger = actions.PassiveFeedbackLedger.restore(saved) if saved else actions.PassiveFeedbackLedger(identity.document_id)
    require(identity.document_id == ledger.document_id, "Gateway feedback document identity differs.")
    suppressed = ledger.suppressed(before, bundle, prediction, runtime["sequence"])
    expected_status = "suppressed" if suppressed else "no_action" if prediction["action"] == "no_action" else "proposal"
    require(gateway["status"] == expected_status, "Gateway suppression/display policy differs.")
    shown = gateway["status"] == "proposal"
    effective = prediction if shown else {"action": "no_action"}
    actual = gateway["result_document"] or before
    targets = [bundle["mapping"][str(i)]["uuid"] for i in prediction.get("target", [])] if shown else []
    require(gateway["target_uuids"] == targets and (shown or gateway["result_document"] is None),
            "Gateway target mapping or hidden result differs.")
    expected = actions.apply_passive_edit(before, bundle, effective)
    require(actual == expected, "Gateway result differs from independent exact source execution.")
    verify_operation(before, actual, bundle["mapping"], effective)
    require(gateway["document_hash"] == session_hash(before)
            and (gateway["result_hash"] == session_hash(actual) if shown else gateway["result_hash"] is None),
            "Gateway document/result hash differs.")
    matched = prep.outcome_digest(actual) in {r["outcome_sha256"] for r in case["reference_outcomes"] if r["available"]}
    return {"outcome_match": matched, "source_integrity": "passed",
            "source_execution_sha256": prep.outcome_digest(actual), "result_document_sha256": session_hash(actual)}


def quantiles(values):
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values), "Invalid measured number.")
    values = sorted(values)
    return {"count": len(values), "p50": statistics.median(values) if values else None,
            "p95": values[math.ceil(len(values) * .95) - 1] if values else None}


def metric_rows(cases, observations):
    require([r["case_id"] for r in observations] == [c["id"] for c in cases], "Incomplete or reordered observations.")
    output = []
    for case, row in zip(cases, observations, strict=True):
        gateway, evidence = row.get("gateway", {}), row.get("evidence", {})
        completed = row.get("status") == "completed"
        shown = completed and gateway.get("status") == "proposal"
        action = case["primary"]["action"]
        predicted = gateway.get("prediction", {}).get("action")
        matched = completed and row.get("outcome_match") is True
        admitted = case.get("applicability", {}).get("full_action_preservation_admitted", True)
        output.append({"id": case["id"], "language": case["language"], "reference_action": action,
                       "predicted_action": predicted, "displayed": shown,
                       "useful": shown and matched and action != "no_action",
                       "preserved": action == "no_action" and completed and matched and predicted == "no_action",
                       "full_action_preservation_admitted": action == "no_action" and admitted,
                       "scope_ambiguous_preservation": action == "no_action" and not admitted,
                       "invalid": not completed, "outcome_match": matched,
                       "unmatched_cut": shown and predicted in CUTS and not matched,
                       "cut_on_legacy_preserve": shown and predicted in CUTS and action == "no_action",
                       "cut_on_readmitted_preserve": shown and predicted in CUTS and action == "no_action" and admitted,
                       "reference_unavailable": not any(r["available"] for r in case.get("reference_outcomes", [{"available": True}])),
                       "http_ms": evidence.get("http_ms"), "raw_format_valid": evidence.get("raw_format_valid") is True,
                       "api_format_valid": evidence.get("api_format_valid") is True,
                       "tool_handoff": evidence.get("terminal_token_id") == 50,
                       "truncated": evidence.get("raw_stop_type") == "limit",
                       "output_tokens": len(evidence["sampled_token_ids"]) if isinstance(evidence.get("sampled_token_ids"), list) else None,
                       "source_integrity_failure": row.get("error_code") == "source_integrity"})
    return output


def counts(values):
    require(bool(values), "Cannot summarize an empty cohort.")
    useful, displayed = sum(r["useful"] for r in values), sum(r["displayed"] for r in values)
    needed = sum(r["reference_action"] != "no_action" for r in values)
    preserve = len(values) - needed
    admitted = [r for r in values if r["full_action_preservation_admitted"]]
    return {"cases": len(values), "useful": useful, "displayed": displayed, "reference_edit_needed": needed,
            "known_reference_precision": useful / displayed if displayed else None,
            "known_reference_recall": useful / needed if needed else None,
            "known_reference_f1": 2 * useful / (displayed + needed) if displayed + needed else None,
            "legacy_preservation": {"numerator": sum(r["preserved"] for r in values), "denominator": preserve},
            "readmitted_full_action_preservation": {"numerator": sum(r["preserved"] for r in admitted), "denominator": len(admitted)},
            "display_coverage": {"numerator": displayed, "denominator": len(values)},
            **{k: sum(r[k] for r in values) for k in ("invalid", "scope_ambiguous_preservation", "reference_unavailable", "unmatched_cut", "cut_on_legacy_preserve", "cut_on_readmitted_preserve", "raw_format_valid", "api_format_valid", "tool_handoff", "truncated", "source_integrity_failure")},
            "http_ms": quantiles([r["http_ms"] for r in values if type(r["http_ms"]) in (int, float)]),
            "output_tokens": {**quantiles([r["output_tokens"] for r in values if r["output_tokens"] is not None]),
                              "sum": sum(r["output_tokens"] or 0 for r in values)}}


def summarize(cases, observations):
    values = metric_rows(cases, observations)
    strata = {}
    for field in ("reference_action", "language"):
        grouped = defaultdict(list)
        for value in values:
            grouped[value[field]].append(value)
        strata[field] = {name: counts(group) for name, group in grouped.items()}
    return {**counts(values), "strata": strata, "rows": values}


def validate_saved_observation(case, item, token, row, decoded_text, config):
    evidence = row.get("evidence", {})
    if row["status"] != "completed":
        require(row["status"] == "failed" and not row.get("gateway") and row.get("outcome_match") is False
                and "source_integrity" not in row and "source_execution_sha256" not in row,
                "A failed request received credit.")
        return
    prefix = token["primary"]["input_ids"]
    actual = pilot.sampled_evidence(evidence["response"])
    request = {"model": config.model_id, "messages": token["messages"], "temperature": 0, "seed": 42,
               "top_k": 40, "top_p": .95, "min_p": .05, "repeat_penalty": 1.,
               "presence_penalty": 0., "frequency_penalty": 0., "max_tokens": 512,
               "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False,
               "stream": False, "return_tokens": True, "verbose": True,
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "next_edit", "strict": True, "schema": pilot.schema_for_runtime(actions.OUTPUT_SCHEMA)}}}
    require(all(evidence.get(k) == v for k, v in actual.items())
            and evidence["native_prefix"] == evidence["server_prefix"] == prefix
            and evidence["server_rendered_prompt"] == token["rendered_prompt"]
            and evidence["request"] == request and evidence["response"]["model"] == config.model_id
            and evidence["response"]["usage"]["prompt_tokens"] == len(prefix)
            and evidence["http_status"] == 200 and actual["terminal_token_id"] in (1, 106),
            "Saved successful raw/API/token/request evidence differs.")
    quantiles([evidence["http_ms"]])
    require(decoded_text == actual["sampled_content"], "Actual sampled token decoding differs.")
    parsed = actions.EDIT.validate_json(evidence["response"]["choices"][0]["message"]["content"], strict=True).model_dump()
    require(parsed == row["gateway"]["prediction"], "Gateway prediction differs from actual API content.")
    require(all(row.get(k) == v for k, v in source_result(case, item, row["gateway"]).items()), "Saved source/outcome evidence differs.")
    require(row["gateway"]["model"]["sha256"] == config.model_sha256, "Saved prediction belongs to another model.")


async def run(protocol_path, protocol_sha, output):
    require(output.parent == BENCHMARK and not output.exists(), "Use a new owned run directory.")
    protocol, cases, runtime, native, config = validate_registration(protocol_path, protocol_sha)
    require(shutil.disk_usage(ROOT).free >= 30 * 1024**3 + 512 * 1024**2, "Insufficient reserve for bounded evidence.")
    expected = {r["id"]: r for r in native}
    inputs = {r["id"]: r for r in runtime}
    with (ROOT / "artifacts/nep-passive-b1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir()
        state = {"version": VERSION, "protocol": record(protocol_path), "status": "running", "condition": CONDITION,
                 "started_at": datetime.now(timezone.utc).isoformat(), "requests_completed": 0}
        write(output / "pipeline.json", state)
        process = None
        try:
            adapter = CapturingAdapter(config, expected)
            adapter._verify_files()
            with httpx.Client(trust_env=False, timeout=1) as client:
                try:
                    client.get(config.endpoint + "/health")
                except httpx.ConnectError:
                    pass
                else:
                    raise ValueError("Dedicated model endpoint already occupied.")
            with (output / "server.log").open("x") as log:
                process = subprocess.Popen(["sh", str(bound(protocol["arm"]["launcher"]))], stdout=log, stderr=log)
                state["owned_model_pid"] = process.pid
                write(output / "pipeline.json", state)
                deadline = time.monotonic() + 180
                async with httpx.AsyncClient(trust_env=False, timeout=1) as startup:
                    while True:
                        require(process.poll() is None, "Owned model exited during startup.")
                        try:
                            response = await startup.get(config.endpoint + "/health")
                            if response.status_code == 200:
                                break
                        except httpx.RequestError:
                            pass
                        require(time.monotonic() < deadline, "Owned model startup timed out.")
                        await asyncio.sleep(.25)
                try:
                    health = await asyncio.to_thread(adapter.health)
                finally:
                    if hasattr(adapter, "observed_props"):
                        write(output / "props.json", adapter.observed_props)
                require(health["status"] == "ready", "Runtime native-prefix qualification failed.")
                write(output / "health.json", health)
                with (output / "results.jsonl").open("x") as stream:
                    for case in cases:
                        for name, value in protocol["sources_sha256"].items():
                            require(sha(ROOT / name) == value, "Source changed before next model request.")
                        item = inputs[case["id"]]
                        service = pilot.CopilotService(adapter)
                        document_id = case["id"]
                        if item["feedback_journal"] is not None:
                            ledger = actions.PassiveFeedbackLedger.restore(item["feedback_journal"])
                            document_id = ledger.document_id
                            service._session(document_id).feedback = ledger
                        adapter.case_id, adapter.observation = case["id"], {}
                        body = pilot.PredictRequest(request={"document_id": document_id, "request_id": "large-dev-1", "revision": 0, "interaction_epoch": 0},
                                                    document=item["document"], focus_uuid=item["focus_uuid"], selection=item["selection"],
                                                    sequence=item["sequence"], history=item["history"], trigger="explicit", deadline_ms=120000)
                        row = {"case_id": case["id"], "condition": CONDITION}
                        try:
                            gateway = await service.predict(body)
                            row.update(status="completed", gateway=gateway)
                            row.update(source_result(case, item, gateway))
                        except pilot.GatewayError as error:
                            row.update(status="failed", error_code=error.code, error=str(error), outcome_match=False)
                            row.pop("gateway", None)
                        except Exception as error:
                            row.update(status="failed", error_code="source_integrity", error=f"{type(error).__name__}: {error}", outcome_match=False)
                            row.pop("gateway", None)
                            adapter.observation["integrity_failure"] = row["error"]
                        row["evidence"] = adapter.observation
                        stream.write(prep.packed(row) + "\n")
                        stream.flush()
                        state["requests_completed"] += 1
                        write(output / "pipeline.json", state)
                        require("integrity_failure" not in adapter.observation, adapter.observation.get("integrity_failure", ""))
                        print(f"{state['requests_completed']}/289 {case['id']} {row['status']}", flush=True)
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                state["owned_model_exit_code"] = process.returncode
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            write(output / "pipeline.json", state)
            write(output / "manifest.json", {"version": VERSION, "protocol": record(protocol_path),
                  "status": state["status"], "files_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})


def report(protocol_path, protocol_sha, run_directory, output):
    require(output.parent == BENCHMARK and not output.exists(), "Use a fresh report file.")
    protocol, cases, runtime, native, config = validate_registration(protocol_path, protocol_sha)
    manifest = json.loads((run_directory / "manifest.json").read_text())
    require(manifest["status"] == "completed" and manifest["protocol"] == record(protocol_path), "Wait for the complete registered run.")
    for name, expected in manifest["files_sha256"].items():
        require(Path(name).name == name and sha(run_directory / name) == expected, "Completed result payload changed.")
    pipeline = json.loads((run_directory / "pipeline.json").read_text())
    require(pipeline["status"] == "completed" and pipeline["requests_completed"] == 289
            and pipeline["owned_model_exit_code"] == 0, "Run or owned model shutdown incomplete.")
    observations = rows(run_directory / "results.jsonl")
    require([r["case_id"] for r in observations] == protocol["case_ids"], "Saved request roster changed.")
    payload = {"source_sha256": sha(Path(prep.__file__)), "manifest": protocol["preparation"],
               "sampled": [{"id": row["case_id"], "ids": row.get("evidence", {}).get("sampled_token_ids")}
                           for row in observations]}
    process = subprocess.run([str(ROOT / "artifacts/model-preparation/gemma26-v1/.venv/bin/python"), "-B",
                              str(Path(prep.__file__)), "--native-readback"],
                             input=prep.packed(payload), text=True, capture_output=True, timeout=180)
    require(process.returncode == 0, "Native tokenizer-only readback failed: " + process.stderr[-2000:])
    actual_native = json.loads(process.stdout)
    require(actual_native["status"] == "native_tokenizer_readback_complete" and actual_native["model_loaded"] is False
            and [r["id"] for r in actual_native["cases"]] == protocol["case_ids"], "Native readback roster differs.")
    for case, item, token, row, actual in zip(cases, runtime, native, observations, actual_native["cases"], strict=True):
        prefix = token["primary"]["input_ids"]
        require(actual["prefix"] == prefix and actual["rendered_prompt"] == token["rendered_prompt"], "Native prefix readback differs.")
        validate_saved_observation(case, item, token, row, actual["decoded"], config)
    full = summarize(cases, observations)
    by_id = {r["case_id"]: r for r in observations}
    matched = [c for c in cases if c["provenance"]["prior_matched286"]]
    matched_summary = summarize(matched, [by_id[c["id"]] for c in matched])
    pilot_cases = rows(bound(protocol["final14_cases"]))
    pilot_rows = rows(bound(protocol["final14_results"]))
    pilot_summary = summarize(pilot_cases, pilot_rows)
    combined_cases, combined_rows = [*cases, *pilot_cases], [*observations, *pilot_rows]
    require(len({c["id"] for c in combined_cases}) == 303, "Combined descriptive cohort is not disjoint.")
    value = {"version": VERSION, "status": "complete_final_checkpoint_development_report",
             "protocol": record(protocol_path), "run_manifest": record(run_directory / "manifest.json"),
             "final_model_sha256": FINAL_MODEL_SHA, "large289": full, "prior_matched286": matched_summary,
             "reused_authored14": pilot_summary, "descriptive_disjoint303": summarize(combined_cases, combined_rows),
             "ambiguity": "Original references are unchanged. The legacy-reference metric is conditional and non-exhaustive for newly enabled cuts. Scope-ambiguous no_action states stay in all289/286 coverage, invalidity, timing and reference-adherence denominators; only separately named re-admitted preservation has a smaller disclosed denominator.",
             "timing_scope": "Raw HTTP, not editor input-to-preview; sequential runs and two different source strata.",
             "new_requests": 289, "reused_pilot_requests": 14, "training_updates": 0, "final_test_bodies_read": 0}
    value["native_tokenizer_readback"] = {"prefixes_verified": 289, "subprocess_exit_code": process.returncode,
                                         "model_loaded": False, "tokenizer": actual_native["tokenizer"]}
    prep.write(output, value)
    return {"status": value["status"], "report": record(output), "large289": counts(full["rows"]),
            "prior_matched286": counts(matched_summary["rows"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    registration = sub.add_parser("register")
    registration.add_argument("--preparation", type=Path, required=True)
    registration.add_argument("--review", type=Path, required=True)
    registration.add_argument("--review-sha256", required=True)
    registration.add_argument("--output", type=Path, required=True)
    for name in ("validate", "run", "report"):
        part = sub.add_parser(name)
        part.add_argument("--protocol", type=Path, required=True)
        part.add_argument("--protocol-sha256", required=True)
        if name in ("run", "report"):
            part.add_argument("--output", type=Path, required=True)
        if name == "report":
            part.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.preparation.resolve(), {"path": str(args.review.resolve()), "sha256": args.review_sha256}, args.output.resolve())
    elif args.command == "validate":
        protocol, cases, _, _, _ = validate_registration(args.protocol.resolve(), args.protocol_sha256)
        result = {"status": "registered_metadata_ready", "cases": len(cases), "census": protocol["census"], "model_started": False}
    elif args.command == "run":
        asyncio.run(run(args.protocol.resolve(), args.protocol_sha256, args.output.resolve()))
        result = {"status": "run_completed"}
    else:
        result = report(args.protocol.resolve(), args.protocol_sha256, args.run.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
