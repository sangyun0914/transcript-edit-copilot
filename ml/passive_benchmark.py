"""Frozen B1 development loading, source-preserving grading and display accounting.

Only canonical_messages(case['input']) belongs in a model request. load_screen adds
private _execution evidence for the evaluator; it must never enter predictor input.
Results contain one quality observation per registered case. Repeats require a
separate latency experiment, rather than inflating the quality denominator.
"""

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from context_feedback_report import history_rule
from grading import transcript_media
from passive_contract import (
    EDIT,
    Envelope,
    PassiveFeedbackLedger,
    apply_passive_edit,
    canonical_messages,
    parse_snapshot,
    validate_prediction,
)
from passive_dataset import RESERVED, apply_corruptions
from replay_session import replay, session_hash
from session_cases import document_records

ROOT = Path(__file__).resolve().parents[1]


def _checked_bytes(path, expected):
    data = path.read_bytes()
    if not expected or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Frozen payload is missing or changed: {path}")
    return data


def _execution(case, read_bound):
    reproduction = case["reproduction"]
    directory = (ROOT / reproduction["candidate_directory"]).resolve()
    path = (directory / reproduction["episode"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Episode escapes its registered candidate directory.")
    saved = read_bound(path)
    source = read_bound(ROOT / saved["source_document"])
    ledger = None
    if "checkpoints" in saved:
        checkpoint = next(c for c in saved["checkpoints"] if c["case_id"] == case["id"])
        before, bundle = checkpoint["before"], checkpoint["bundle"]
        if replay(saved["journal"])[0] != saved["final_document"]:
            raise ValueError("Session replay differs from its saved final state.")
        ledger = saved["feedback_ledger"]
    elif "prepared_context" in saved:
        prepared = read_bound(ROOT / saved["prepared_context"])
        before, bundle = source, prepared["bundle"]
        if prepared["messages"] != canonical_messages(case["input"]):
            raise ValueError("Checkpoint serving messages changed.")
    else:
        journal = copy.deepcopy(saved["journal"])
        journal[0]["document"] = apply_corruptions(source, saved["spec"].get("corruptions", []))
        replay(journal)
        wanted = case["metadata"]["document_before_hash"]
        prefix = next((n + 1 for n, event in enumerate(journal)
                       if event.get("after_hash", event.get("document_hash")) == wanted), None)
        if prefix is None:
            raise ValueError("Case before-state is absent from its replay journal.")
        before = replay(journal[:prefix])[0]
        _, mapping = document_records(before)
        bundle = {
            "input": case["input"],
            "mapping": {str(i["id"]): mapping[str(i["id"])] for i in case["input"]["items"]},
            "document_hash": session_hash(before),
            "context": case["metadata"]["context"],
            "caption_authority": {"version": 1, "protected_uuids": [],
                                  "profile": case["input"]["caption_profile"]},
        }
    if (session_hash(before) != case["metadata"]["document_before_hash"]
            or bundle["input"] != case["input"]
            or bundle["context"] != case["metadata"]["context"]):
        raise ValueError("Reconstructed case differs from its frozen input/state.")
    as_of = bundle["context"]["as_of"]
    feedback = (PassiveFeedbackLedger.restore(ledger).context(
        before, bundle["mapping"], as_of, bundle["context"]["policy"]["feedback_limit"]
    ) if ledger else [])
    if feedback != case["input"]["recent_feedback"]:
        raise ValueError("Frozen feedback differs from its past-only ledger.")
    apply_passive_edit(before, bundle, {"action": "no_action"})
    return {"before": before, "bundle": bundle, "feedback_ledger": ledger, "as_of": as_of}


def load_screen(release: Path):
    """Verify registered dev payloads and return 36 cases in immutable screen order.

    Reads only dev cases/messages, the screening protocol and their bound execution
    dependencies. No training rows, tokenizer downloads or final-test bodies are read.
    """
    release = Path(release).resolve()
    manifest = json.loads((release / "manifest.json").read_text())
    if (manifest.get("version") != "passive-dataset-completion-v2"
            or manifest.get("status") != "frozen_supported_scope"):
        raise ValueError("Use the frozen completion-v2 supported-scope release.")
    files = manifest["files_sha256"]
    payloads = {name: _checked_bytes(release / name, files.get(name)) for name in
                ("dev-cases.jsonl", "dev-messages.jsonl", "screening-protocol.json")}
    protocol = json.loads(payloads["screening-protocol.json"])
    ids = protocol["selected_case_ids"]
    if (protocol.get("version") != "passive-b1-screen-v2"
            or protocol.get("model_predictions_consulted") is not False
            or protocol.get("request_budget") != 36 or len(ids) != 36 or len(set(ids)) != 36):
        raise ValueError("Expected the frozen, prediction-independent 36-request screen.")
    rows = [json.loads(line) for line in payloads["dev-cases.jsonl"].splitlines() if line.strip()]
    messages = [json.loads(line) for line in payloads["dev-messages.jsonl"].splitlines() if line.strip()]
    cases, encoded = {c["id"]: c for c in rows}, {r["id"]: r for r in messages}
    if len(cases) != len(rows) or len(encoded) != len(messages) or set(encoded) != set(cases):
        raise ValueError("Duplicate or missing development cases/messages.")
    for case in rows:
        meta = case["metadata"]
        if meta["split"] != "dev" or meta["source_group_id"] in RESERVED:
            raise ValueError("Only exposed development may enter B1.")
        canonical = canonical_messages(case["input"])
        if (encoded[case["id"]]["messages"] != canonical
                or session_hash(canonical) != meta["input_sha256"]):
            raise ValueError("Frozen messages differ from exact passive serving input.")

    def read_bound(path):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Execution evidence escapes the project.")
        name = str(path.relative_to(ROOT))
        return json.loads(_checked_bytes(path, manifest["dependencies_sha256"].get(name)))

    selected = []
    for identity in ids:
        if identity not in cases:
            raise ValueError("Screen case is absent from the frozen development set.")
        case = copy.deepcopy(cases[identity])
        case["_execution"] = _execution(case, read_bound)
        selected.append(case)
    return selected


def _references(case):
    expected = case.get("expectedOutput")
    return [] if expected is None else [expected, *case.get("acceptableOutputs", [])]


def grade_prediction(case, prediction):
    """Compare valid execution outcomes, preserving paragraphs, source audio and metadata."""
    execution = case["_execution"]
    predicted = apply_passive_edit(execution["before"], execution["bundle"], prediction)
    references = _references(case)
    if not references:
        return {"reference_known": False, "exact_match": None, "outcome_match": None,
                "matched_reference_index": None, "method": "unknown_reference"}
    match, method = None, "different_outcome"
    for index, reference in enumerate(references):
        after = apply_passive_edit(execution["before"], execution["bundle"], reference)
        if transcript_media(predicted) == transcript_media(after):
            match = index
            method = "exact_edit" if prediction == reference else "transcript_media_equivalence"
            break
    return {"reference_known": True, "exact_match": prediction == references[0],
            "outcome_match": match is not None, "matched_reference_index": match, "method": method}


def baseline_prediction(case, name):
    """Use observable input only; the display evaluator applies the actual feedback gate."""
    snapshot = parse_snapshot(case["input"]).model_dump()
    if name == "always_no_action":
        return {"action": "no_action"}
    if name != "consistent_history_rule":
        raise ValueError("Unknown passive baseline.")
    snapshot["items"] = [{**i, "audio_class": "unknown", "boundary_status": "unverified"}
                         for i in snapshot["items"]]
    return history_rule(snapshot)


def evaluate_result(case, text, finish_reason="stop"):
    """Separate raw validity/agreement from actual shown edits; blocked edits earn no credit.

    Returns case_id, format_valid, raw_valid, prediction, raw_grade, displayed,
    displayed_grade, suppression_reason, and optional error/error_type. A no_action
    can be a correct raw abstention but is never a displayed edit.
    """
    row = {"case_id": case["id"], "format_valid": False, "raw_valid": False,
           "prediction": None, "raw_grade": None, "displayed": False,
           "displayed_grade": None, "suppression_reason": None}
    envelope = Envelope(document_id=case["id"], request_id="b1", revision=0, interaction_epoch=0)
    try:
        EDIT.validate_json(text, strict=True)
        row["format_valid"] = True
        prediction = validate_prediction(text, case["input"], envelope, envelope, finish_reason).model_dump()
        execution = case["_execution"]
        apply_passive_edit(execution["before"], execution["bundle"], prediction)
    except (ValueError, TypeError) as error:
        row.update(error=str(error), error_type="truncated" if finish_reason != "stop" else
                   "semantic_invalid" if row["format_valid"] else "format_invalid")
        return row
    row.update(raw_valid=True, prediction=prediction, raw_grade=grade_prediction(case, prediction))
    ledger = execution["feedback_ledger"]
    suppressed = bool(ledger and PassiveFeedbackLedger.restore(ledger).suppressed(
        execution["before"], execution["bundle"], prediction, execution["as_of"]))
    row["suppression_reason"] = "equivalent_rejected_edit_unchanged_context" if suppressed else None
    row["displayed"] = prediction["action"] != "no_action" and not suppressed
    if row["displayed"]:
        row["displayed_grade"] = row["raw_grade"]
    return row


def _percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def _metrics(cases, results):
    counts = Counter(expected_requests=len(cases), observed_results=0, missing_requests=0,
                     raw_valid=0, format_valid=0, raw_outcome_matches=0, raw_proposed=0,
                     raw_unnecessary_edits=0, displayed=0, displayed_useful_edits=0,
                     displayed_unnecessary_edits=0, correct_abstentions=0, suppressed=0,
                     edit_needed_cases=0, preserve_cases=0, known_references=0,
                     unknown_references=0, displayed_unknown=0, matched_required_edits=0,
                     missed_useful_edits=0, invalid_results=0, request_errors=0)
    latencies, errors = [], Counter()
    for case, row in zip(cases, results, strict=True):
        references = _references(case)
        known = bool(references)
        abstention_ok = any(r["action"] == "no_action" for r in references)
        edit_needed = known and not abstention_ok
        preserve = known and all(r["action"] == "no_action" for r in references)
        valid = row.get("raw_valid") is True
        proposed = valid and row["prediction"]["action"] != "no_action"
        matched = valid and (row.get("raw_grade") or {}).get("outcome_match") is True
        shown = proposed and row.get("displayed") is True
        useful = shown and matched
        missing = row.get("error_type") == "missing_result"
        counts.update(known_references=known, unknown_references=not known,
                      displayed_unknown=shown and not known, matched_required_edits=edit_needed and useful,
                      edit_needed_cases=edit_needed, preserve_cases=preserve,
                      observed_results=not missing, missing_requests=missing, raw_valid=valid,
                      format_valid=row.get("format_valid") is True, raw_outcome_matches=matched,
                      raw_proposed=proposed, raw_unnecessary_edits=proposed and preserve,
                      displayed=shown, displayed_useful_edits=useful,
                      displayed_unnecessary_edits=shown and preserve,
                      correct_abstentions=valid and not proposed and matched and abstention_ok,
                      suppressed=bool(row.get("suppression_reason")),
                      missed_useful_edits=edit_needed and not useful,
                      invalid_results=not missing and not valid)
        if row.get("error") or row.get("error_type"):
            errors[row.get("error_type", "request_failure")] += 1
            counts["request_errors"] += row.get("error_type", "request_failure") in {
                "request_failure", "timeout", "cancelled", "missing_result"}
        latency = row.get("complete_json_ms", row.get("latency_ms"))
        if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
            latencies.append(latency)
    return {**counts, "errors_by_type": dict(errors), "latency_samples": len(latencies),
            "missing_latency_samples": len(cases) - len(latencies),
            "all_request_p50_ms": _percentile(latencies, 0.5),
            "all_request_p95_ms": _percentile(latencies, 0.95),
            "displayed_useful_precision": counts["displayed_useful_edits"] / counts["displayed"]
            if counts["displayed"] else None,
            "useful_edit_recall": counts["matched_required_edits"] / counts["edit_needed_cases"]
            if counts["edit_needed_cases"] else None}


def summarize(cases, results):
    """Fixed case denominators, one result per case, failed-request-inclusive latency.

    Transport failures may be supplied as {case_id, error, error_type, latency_ms}.
    Missing rows count as failures; their unmeasured latency is reported, not imputed.
    """
    known = {c["id"]: c for c in cases}
    received = {r["case_id"]: r for r in results}
    if len(known) != len(cases) or len(received) != len(results) or set(received) - set(known):
        raise ValueError("Require unique registered cases and at most one result per case.")
    ordered = [received.get(c["id"], {"case_id": c["id"], "error_type": "missing_result",
                                     "error": "Registered request has no result."}) for c in cases]
    report = {"scope": "Purposive exposed-development agreement; synthetic feedback; audio actions gated.",
              **_metrics(cases, ordered),
              "missing_case_ids": [c["id"] for c in cases if c["id"] not in received]}
    for label, field in (("language", "language"), ("source", "source_group_id"),
                         ("episode", "episode_id"), ("stratum", "release_stratum"),
                         ("action", None)):
        groups = defaultdict(list)
        for case, row in zip(cases, ordered, strict=True):
            value = case["metadata"].get(field, "unspecified") if field else (
                case.get("expectedOutput") or {}).get("action", "unknown")
            groups[value].append((case, row))
        report[f"by_{label}"] = {name: _metrics([c for c, _ in batch], [r for _, r in batch])
                                  for name, batch in sorted(groups.items())}
    return report
