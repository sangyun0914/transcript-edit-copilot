"""Describe the admitted release002/release003 DEV322 without rerunning the grader.

Only the standard library is used. Cases are streamed and reduced to metadata;
documents, prompts, tokenizers and models are neither retained nor loaded.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


VERSION = "passive-actions-v3-diagnostics-1"
ADMITTED_DEV_SHA256 = "b87bea4dd5fe7033be4d46fd6fbcea1cafa24c37ddc07a1509798ea53af07e7a"
BOOLEAN_FIELDS = (
    "valid", "displayed", "suppressed", "useful", "preserved",
    "model_abstained", "outcome_match",
)
GROUP_FIELDS = (
    "judgment", "language", "stratum", "source_group", "source_provenance",
    "natural_audio", "reference_action", "action_language",
    "cursor_selection_before_judgment", "cursor_at_first_accepted_target",
)
EDIT_ACTIONS = {
    "correct_text", "delete_span", "trim_silence", "remove_silence",
    "split_paragraph", "merge_paragraph",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def read_json(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return json.loads(data, parse_constant=reject_constant), {
        "path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def stream_jsonl(path, consume):
    path = Path(path).resolve()
    digest, size, count = hashlib.sha256(), 0, 0
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            digest.update(raw)
            size += len(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw, parse_constant=reject_constant)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid JSONL: {path}:{line_number}") from error
            require(type(row) is dict, f"Expected an object: {path}:{line_number}")
            consume(row)
            count += 1
    return {
        "path": str(path), "sha256": digest.hexdigest(), "bytes": size,
        "rows": count,
    }


def target_span(value):
    return (
        value if type(value) is list and value
        and all(type(alias) is int for alias in value) else None
    )


def boolean_group(value):
    return str(value).lower() if type(value) is bool else "unknown"


def history_rule_predicates(snapshot):
    """Census the raw predicates in context_feedback_report.history_rule.

    This does not validate, execute or grade a proposed edit. In particular,
    whole-item text equality is retained, rather than changed to a substring test.
    """
    missing = [
        key for key in ("recent_edits", "items", "protected_ids", "allowed_actions")
        if type(snapshot.get(key)) is not list
    ]
    edits, items = snapshot.get("recent_edits", []), snapshot.get("items", [])
    if not missing:
        for edit in edits:
            keys = {"action"}
            if isinstance(edit, dict) and edit.get("action") == "correct_text":
                keys.update(("before", "after"))
            if isinstance(edit, dict) and edit.get("action") == "trim_silence":
                keys.add("after_ms")
            if not isinstance(edit, dict) or not keys <= edit.keys():
                missing.append("recent_edits.fields")
        for item in items:
            if not isinstance(item, dict) or not {
                "id", "text", "audio_class", "boundary_status", "duration_ms"
            } <= item.keys():
                missing.append("items.fields")
    if missing:
        return {"schema_compatible": False, "missing_fields": sorted(set(missing))}
    corrections, pauses = defaultdict(set), set()
    correction_entries = trim_entries = 0
    for edit in edits:
        if edit["action"] == "correct_text" and edit["before"] and edit["after"]:
            corrections[edit["before"]].add(edit["after"])
            correction_entries += 1
        if edit["action"] == "trim_silence" and edit["after_ms"]:
            pauses.add(edit["after_ms"])
            trim_entries += 1
    counts = {
        "history_entries": len(edits), "correction_history_entries": correction_entries,
        "trim_history_entries": trim_entries, "correction_before_values": len(corrections),
        "pause_after_ms_values": len(pauses), "unprotected_items": 0,
        "whole_item_text_matches": 0, "unambiguous_text_matches": 0,
        "verified_silence_items": 0, "verified_boundary_silence_items": 0,
        "duration_eligible_silence_items": 0,
        "correction_candidate_items": 0, "trim_candidate_items": 0,
    }
    first_action = "no_action"
    for item in items:
        if item["id"] in snapshot["protected_ids"]:
            continue
        counts["unprotected_items"] += 1
        values = corrections.get(item["text"], set())
        counts["whole_item_text_matches"] += bool(values)
        counts["unambiguous_text_matches"] += len(values) == 1
        correction = len(values) == 1 and "correct_text" in snapshot["allowed_actions"]
        silence = item["audio_class"] == "verified_silence"
        boundary = silence and item["boundary_status"] == "verified"
        duration = boundary and 3000 <= item["duration_ms"] <= 5000
        trim = duration and len(pauses) == 1 and "trim_silence" in snapshot["allowed_actions"]
        counts["verified_silence_items"] += silence
        counts["verified_boundary_silence_items"] += boundary
        counts["duration_eligible_silence_items"] += duration
        counts["correction_candidate_items"] += correction
        counts["trim_candidate_items"] += trim
        if first_action == "no_action" and (correction or trim):
            first_action = "correct_text" if correction else "trim_silence"
    return {"schema_compatible": True, **counts, "first_raw_rule_action": first_action}


def case_metadata(case):
    identity = case.get("id")
    require(isinstance(identity, str) and identity, "A case needs a nonempty ID.")
    require(case.get("split") == "dev", f"Only exposed dev cases are accepted: {identity}")
    status = case["judgment"]["status"]
    require(status in ("edit_required", "preserve"), f"Unknown judgment: {identity}")
    refs = case["accepted_predictions"]
    require(type(refs) is list and refs, f"Missing accepted predictions: {identity}")
    require(refs[0] == case["primary"], f"Primary reference changed: {identity}")
    for ref in refs:
        require(type(ref) is dict, f"Invalid reference: {identity}")
        if status == "preserve":
            require(ref == {"action": "no_action"}, f"Mixed reference judgment: {identity}")
        else:
            require(ref.get("action") in EDIT_ACTIONS and target_span(ref.get("target")),
                    f"An edit reference needs an action and target span: {identity}")
    source, lineage, snapshot = case["source"], case.get("lineage", {}), case["input"]
    cursor = snapshot.get("cursor_before")
    cursor_group = "not_applicable_preserve"
    if status == "edit_required":
        cursor_group = (
            "unknown" if type(cursor) is not int else
            "at_first_accepted_target" if any(cursor == ref["target"][0] for ref in refs)
            else "elsewhere"
        )
    return {
        "case_id": identity, "judgment": status, "language": case["language"],
        "reference_action": case["primary"]["action"],
        "action_language": f"{case['primary']['action']}:{case['language']}",
        "source_group": source["source_group"],
        "source_provenance": source.get("provenance", source.get("kind", "unspecified")),
        "stratum": lineage.get("stratum", source.get("provenance", source.get("kind", "unspecified"))),
        "natural_audio": boolean_group(source.get("natural_audio")),
        "acoustically_verified": boolean_group(source.get("acoustically_verified")),
        "cursor_selection_before_judgment": boolean_group(lineage.get("cursor_selection_before_judgment")),
        "cursor_at_first_accepted_target": cursor_group,
        "accepted_predictions": refs,
        "history_rule_predicates": history_rule_predicates(snapshot),
    }


def load_release(release):
    release = Path(release).resolve()
    manifest, binding = read_json(release / "manifest.json")
    cases = {}

    def consume(row):
        case = case_metadata(row)
        require(case["case_id"] not in cases, f"Duplicate case: {case['case_id']}")
        cases[case["case_id"]] = case

    case_binding = stream_jsonl(release / "cases/dev.jsonl", consume)
    require(cases, "The dev roster is empty.")
    require(case_binding["sha256"] == manifest["files_sha256"]["cases/dev.jsonl"],
            "Dev cases differ from the immutable release manifest.")
    require(case_binding["sha256"] == ADMITTED_DEV_SHA256,
            "Dev cases differ from the pinned admitted release002/release003 DEV322.")
    require(len(cases) == manifest["counts"]["dev"], "Dev roster count differs from its manifest.")
    return cases, {"release_manifest": binding, "dev_cases": case_binding}


def parse_observation(observation, format_name):
    result = {
        "present": observation is not None, "format": format_name,
        "status": observation.get("status") if observation else None,
        "finish_reason": observation.get("finish_reason") if observation else None,
        "diagnostic_format_supported": format_name == "json",
        "syntactically_parseable_json": None if format_name != "json" else False,
        "json_object": False, "duplicate_keys": False, "raw_action": None,
        "raw_target": None,
    }
    if format_name != "json":
        result["exclusion"] = "noncanonical_format_not_decoded"
        return result, None
    if observation is None or not isinstance(observation.get("text"), str):
        result["parse_failure"] = "missing_observation" if observation is None else "missing_text"
        return result, None
    duplicates = []

    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                duplicates.append(key)
            obj[key] = value
        return obj

    try:
        value = json.loads(observation["text"], parse_constant=reject_constant,
                           object_pairs_hook=pairs)
    except ValueError:
        result["parse_failure"] = "invalid_json_syntax"
        return result, None
    result.update(syntactically_parseable_json=True, json_object=type(value) is dict,
                  duplicate_keys=bool(duplicates))
    if type(value) is not dict or duplicates:
        return result, None
    result["raw_action"] = value.get("action") if isinstance(value.get("action"), str) else None
    result["raw_target"] = target_span(value.get("target"))
    return result, value if observation.get("status") == "completed" else None


def diagnose_case(case, grade, observation, format_name):
    output, prediction = parse_observation(observation, format_name)
    edit = case["judgment"] == "edit_required"
    eligible = edit and format_name == "json"
    matches = {key: None if not eligible else False for key in (
        "action_type_match", "full_target_span_match", "same_alternative_joint_match",
        "full_prediction_match",
    )}
    if eligible and prediction is not None:
        action, target = prediction.get("action"), target_span(prediction.get("target"))
        positioned = target is not None and action != "no_action"
        refs = case["accepted_predictions"]
        matches.update(
            action_type_match=any(action == ref["action"] for ref in refs),
            full_target_span_match=positioned and any(target == ref["target"] for ref in refs),
            same_alternative_joint_match=positioned and any(
                action == ref["action"] and target == ref["target"] for ref in refs
            ),
            full_prediction_match=any(
                json.dumps(prediction, sort_keys=True, ensure_ascii=False)
                == json.dumps(ref, sort_keys=True, ensure_ascii=False) for ref in refs
            ),
        )
    return {
        **{key: value for key, value in case.items()
           if key not in ("accepted_predictions", "history_rule_predicates")},
        "grade": grade, "observation": output,
        "edit_diagnostic_eligible": eligible, "exact_diagnostics": matches,
    }


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def summarize(records):
    grades = [row["grade"] for row in records]
    n = len(records)
    needed = sum(row["judgment"] == "edit_required" for row in records)
    preserve = n - needed
    counts = {key: sum(row[key] for row in grades) for key in BOOLEAN_FIELDS}
    model_preserved = sum(row["preserved"] and row["model_abstained"] for row in grades)
    official = {
        "precision": ratio(counts["useful"], counts["displayed"]),
        "recall": ratio(counts["useful"], needed),
        "preservation": ratio(counts["preserved"], preserve),
        "model_preservation": ratio(model_preserved, preserve),
        "coverage": ratio(counts["displayed"], n),
        "validity": ratio(counts["valid"], n),
        "outcome_accuracy": ratio(counts["outcome_match"], n),
    }
    eligible = [row for row in records if row["edit_diagnostic_eligible"]]
    json_rows = [row for row in records if row["observation"]["diagnostic_format_supported"]]
    parseable = sum(row["observation"]["syntactically_parseable_json"] for row in json_rows)
    execution_completed = sum(
        row["status"] == "completed" and row["source_integrity"] == "passed"
        for row in grades
    )
    return {
        "case_ids": sorted(row["case_id"] for row in records),
        "counts": {"requests": n, "edit_required": needed, "preserve": preserve,
                   **counts, "model_preserved": model_preserved},
        "status_counts": dict(sorted(Counter(row["status"] for row in grades).items())),
        "canonical_metrics": official,
        "edit_diagnostics": {
            "eligible_cases": len(eligible), "excluded_noncanonical_format": needed - len(eligible),
            **{name: ratio(sum(row["exact_diagnostics"][name] for row in eligible), len(eligible))
               for name in ("action_type_match", "full_target_span_match",
                            "same_alternative_joint_match", "full_prediction_match")},
        },
        "output_diagnostics": {
            "canonical_json_cases": len(json_rows),
            "excluded_noncanonical_format": n - len(json_rows),
            "missing_observation_case_ids": sorted(row["case_id"] for row in records
                                                   if not row["observation"]["present"]),
            "json_syntax_parseability": ratio(parseable, len(json_rows)),
            "parseable_but_canonically_invalid_case_ids": sorted(
                row["case_id"] for row in json_rows
                if row["observation"]["syntactically_parseable_json"] and not row["grade"]["valid"]
            ),
            "execution_completed": ratio(execution_completed, n),
            "preserve_no_display": ratio(sum(
                row["judgment"] == "preserve" and not row["grade"]["displayed"]
                for row in records
            ), preserve),
            "raw_exact_vs_useful_disagreement_case_ids": sorted(
                row["case_id"] for row in eligible
                if row["exact_diagnostics"]["full_prediction_match"] != row["grade"]["useful"]
            ),
        },
    }


def preservation_disposition(grade):
    if grade["status"] in ("request_failure", "invalid_output", "execution_failure"):
        return grade["status"]
    suffix = "preserved" if grade["preserved"] else "not_preserved"
    if grade["model_abstained"]:
        return "model_no_action_" + suffix
    if grade["suppressed"]:
        return "feedback_suppressed_" + suffix
    if grade["displayed"]:
        return "displayed_unnecessary"
    return "other_not_displayed_" + suffix


def grouped(records, field):
    groups = defaultdict(list)
    for row in records:
        groups[row[field]].append(row)
    return {key: summarize(values) for key, values in sorted(groups.items())}


def load_run(name, grade_directory, observation_directory, cases):
    grade_directory, observation_directory = Path(grade_directory), Path(observation_directory)
    saved, report_binding = read_json(grade_directory / "report.json")
    require(saved.get("final_test") is False and saved.get("status") == "graded_development",
            f"Only saved development grades are accepted: {name}")
    require(saved.get("condition") == "canonical_full_context",
            f"A changed context needs its own references; only canonical inputs are supported: {name}")
    format_name, grades, observations = saved["format"], {}, {}

    def consume_grade(row):
        identity = row["case_id"]
        require(identity in cases and identity not in grades, f"Foreign/duplicate graded case: {identity}")
        case = cases[identity]
        for key in ("judgment", "reference_action", "language", "source_group", "stratum"):
            require(row[key] == case[key], f"Graded metadata differs from release: {identity}:{key}")
        require(all(type(row.get(key)) is bool for key in BOOLEAN_FIELDS),
                f"Non-boolean canonical grade: {identity}")
        require(row.get("status") in ("completed", "invalid_output", "request_failure", "execution_failure"),
                f"Unknown canonical status: {identity}")
        require(not row["useful"] or row["displayed"] and case["judgment"] == "edit_required",
                f"Inconsistent useful grade: {identity}")
        require(not row["preserved"] or case["judgment"] == "preserve",
                f"Preserved edit-required case: {identity}")
        grades[identity] = {key: row[key] for key in (
            *BOOLEAN_FIELDS, "status", "predicted_action", "source_integrity",
        )}

    def consume_observation(row):
        identity = row["case_id"]
        require(identity in grades and identity not in observations,
                f"Foreign/duplicate observation: {identity}")
        require(row.get("format") == format_name, f"Observation format differs: {identity}")
        observations[identity] = {key: row.get(key) for key in ("text", "status", "finish_reason")}

    graded_binding = stream_jsonl(grade_directory / "graded.jsonl", consume_grade)
    observation_binding = stream_jsonl(observation_directory / "observations.jsonl", consume_observation)
    for key, binding in (("graded", graded_binding), ("observations", observation_binding)):
        require(binding["sha256"] == saved[key]["sha256"], f"Saved {key} hash differs: {name}")
    require(len(grades) == saved["overall"]["requests"], f"Missing graded rows: {name}")
    records = [diagnose_case(cases[i], grades[i], observations.get(i), format_name) for i in sorted(grades)]
    overall = summarize(records)
    require(overall["status_counts"] == saved["overall"]["status_counts"], f"Saved status counts differ: {name}")
    for metric, value in overall["canonical_metrics"].items():
        require(all(saved["overall"][metric][key] == number for key, number in value.items()),
                f"Saved metric differs: {name}:{metric}")
    dispositions = defaultdict(list)
    for row in records:
        if row["judgment"] == "preserve":
            dispositions[preservation_disposition(row["grade"])].append(row)
    return {
        "format": format_name,
        "input_bindings": {"report": report_binding, "graded": graded_binding,
                           "observations": observation_binding},
        "cohort": {
            "definition": "All graded rows; missing observations and invalid outputs remain in their denominators.",
            "not_in_graded_cohort_case_ids": sorted(set(cases) - set(grades)),
            "excluded_before_inference_as_reported": saved.get("excluded_before_inference"),
        },
        "saved_overall": saved["overall"], "overall": overall,
        "by": {field: grouped(records, field) for field in GROUP_FIELDS},
        "preservation_dispositions": {
            key: {"case_ids": [row["case_id"] for row in values], "count": len(values),
                  "by_language": dict(sorted(Counter(row["language"] for row in values).items())),
                  "raw_action_counts": dict(sorted(Counter(
                      row["observation"]["raw_action"] or "unavailable" for row in values
                  ).items()))}
            for key, values in sorted(dispositions.items())
        },
        "cases": records,
    }


def dataset_census(cases):
    rows = list(cases.values())
    baseline = [row["history_rule_predicates"] for row in rows]
    compatible = [row for row in rows if row["history_rule_predicates"]["schema_compatible"]]
    numeric_keys = sorted({key for row in baseline for key, value in row.items() if type(value) is int})
    return {
        "case_ids": sorted(cases), "requests": len(cases),
        "by": {field: {key: {"count": len(ids), "case_ids": ids} for key, ids in sorted(
            ((value, sorted(row["case_id"] for row in rows if row[field] == value))
             for value in {row[field] for row in rows})
        )} for field in GROUP_FIELDS},
        "history_rule_applicability": {
            "source": "Predicates transcribed from ml/context_feedback_report.py:history_rule; no import, validation or execution.",
            "schema_compatible_cases": len(compatible),
            "schema_incompatible_case_ids": sorted(row["case_id"] for row in rows
                                                    if not row["history_rule_predicates"]["schema_compatible"]),
            "predicate_totals": {key: sum(row.get(key, 0) for row in baseline) for key in numeric_keys},
            "first_raw_rule_action": dict(sorted(Counter(
                row["history_rule_predicates"]["first_raw_rule_action"] for row in compatible
            ).items())),
            "cases": {row["case_id"]: row["history_rule_predicates"] for row in rows},
        },
    }


DEFINITIONS = {
    "scope": "Pinned admitted release002/release003 DEV322 only, verified against the code's fixed SHA256 as well as the supplied manifest. Descriptive diagnostics only; no regrading, inference, causal claims or final-test access.",
    "cohort": "Per-run graded roster, verified against saved report.requests. Release cases outside that roster are listed, not silently treated as failures. Missing observations stay in graded denominators.",
    "cursor_at_first_accepted_target": "Edit-required only: integer input.cursor_before equals the first target alias of any accepted prediction. Preserve is not applicable; missing/noninteger cursor is unknown.",
    "cursor_selection_before_judgment": "The exact boolean at case.lineage.cursor_selection_before_judgment; missing/nonboolean is unknown, never inferred from cursor alignment or source type.",
    "exact_diagnostics": "All edit-required canonical-json graded cases are the denominator, including invalid, failed and missing outputs. Compare strict parsed object fields from completed observations; no contract-validity filter. Target is the entire ordered integer array, independent of action, except no_action cannot match a position. Joint requires action and target on the same accepted alternative. Text and keep_ms are checked only by full_prediction_match.",
    "format_exclusions": "Only observation format json is decoded. Rewrite and every other format retain all canonical graded metrics, with explicit exclusion from JSON syntax and exact action/target diagnostics.",
    "parseability": "JSON syntax only (non-JSON NaN/Infinity rejected). Arrays/scalars are syntactically parseable but are not action objects. Duplicate object keys are flagged and cannot match references. Neither schema nor runtime validity is inferred from parsing.",
    "canonical_validity": "Unmodified graded.valid. This includes contract validation, not just JSON syntax; it can remain true on execution_failure.",
    "execution_completed": "Unmodified grade has status completed and source_integrity passed; no edit is executed here.",
    "preservation": "Unmodified graded.preserved / preserve cases; model preservation requires both preserved and model_abstained. Invalid output is not reclassified as preserved.",
    "preserve_no_display": "Derived display avoidance only: not displayed / preserve cases. Includes failures and invalid outputs; this is not model or system preservation.",
    "history_rule_applicability": "Source-only census of the existing rule's exact predicates, including whole-item text equality, protection, allowed_actions and its first-match order. Raw applicability does not certify an executable or useful edit.",
}


def markdown(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    def fraction(value):
        return f"{value['numerator']}/{value['denominator']}"

    lines = ["# Actions v3 development diagnostics", "", DEFINITIONS["scope"], "",
             "Exact output diagnostics do not replace the frozen execution-based grades.", "",
             "| Run | N | Useful/displayed | Useful/edit | Model preserve | System preserve | Displayed/N | JSON parseable | Valid/N |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, run in report["runs"].items():
        overall = run["overall"]
        values = overall["canonical_metrics"]
        lines.append("| " + " | ".join([
            cell(name), str(overall["counts"]["requests"]),
            *(fraction(values[key]) for key in ("precision", "recall", "model_preservation", "preservation", "coverage")),
            fraction(overall["output_diagnostics"]["json_syntax_parseability"]), fraction(values["validity"]),
        ]) + " |")
    for name, run in report["runs"].items():
        lines.extend(["", f"## {cell(name)}", "", f"Format: `{cell(run['format'])}`. "
                      f"Outside graded cohort: {len(run['cohort']['not_in_graded_cohort_case_ids'])}. "
                      f"Missing observations: {len(run['overall']['output_diagnostics']['missing_observation_case_ids'])}.", "",
                      "| Cohort | N | Useful/displayed | Useful/edit | Action/edit | Target/edit | Joint/edit |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        cohorts = [("overall", run["overall"])]
        for field in ("cursor_at_first_accepted_target", "cursor_selection_before_judgment", "stratum", "language"):
            cohorts.extend((f"{field}={key}", value) for key, value in run["by"][field].items())
        for label, value in cohorts:
            lines.append("| " + " | ".join([
                cell(label), str(value["counts"]["requests"]),
                fraction(value["canonical_metrics"]["precision"]), fraction(value["canonical_metrics"]["recall"]),
                *(fraction(value["edit_diagnostics"][key]) for key in (
                    "action_type_match", "full_target_span_match", "same_alternative_joint_match")),
            ]) + " |")
        lines.extend(["", "Preserve-case dispositions (canonical flags remain unchanged):", ""])
        for key, value in run["preservation_dispositions"].items():
            lines.append(f"- `{key}`: {value['count']}; language {value['by_language']}; raw action {value['raw_action_counts']}.")
    baseline = report["dataset"]["history_rule_applicability"]
    lines.extend(["", "## Existing history-rule applicability", "",
                  f"Schema-compatible cases: {baseline['schema_compatible_cases']}/{report['dataset']['requests']}.",
                  f"First raw rule action counts: {baseline['first_raw_rule_action']}.",
                  f"Predicate totals: {baseline['predicate_totals']}.", "", "## Definitions and input hashes", ""])
    lines.extend(f"- **{key}**: {value}" for key, value in DEFINITIONS.items())
    bindings = {**report["input_bindings"], "implementation": report["implementation"]}
    for name, run in report["runs"].items():
        bindings.update({f"{name}:{key}": value for key, value in run["input_bindings"].items()})
    lines.extend(["", "All case IDs, cohort exclusions, per-case flags and denominators are in `diagnostics.json`.", ""])
    lines.extend(f"- `{cell(key)}`: `{value['sha256']}` (`{cell(value['path'])}`)" for key, value in bindings.items())
    return "\n".join(lines) + "\n"


def report(release, runs, output):
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh diagnostic output directory; nothing is overwritten.")
    inputs = [Path(release).resolve(), *(Path(path).resolve() for _, grade, observed in runs for path in (grade, observed))]
    require(all(not output.is_relative_to(path) for path in inputs), "Output must be outside immutable input directories.")
    require(runs and len({name for name, _, _ in runs}) == len(runs), "Use nonempty, unique run names.")
    cases, bindings = load_release(release)
    source = Path(__file__).resolve()
    implementation = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    result = {
        "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
        "admitted_dev_sha256": ADMITTED_DEV_SHA256,
        "implementation": implementation, "input_bindings": bindings,
        "definitions": DEFINITIONS, "dataset": dataset_census(cases),
        "runs": {name: load_run(name, grade, observed, cases) for name, grade, observed in runs},
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "diagnostics.json").open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    with (output / "diagnostics.md").open("x", encoding="utf-8") as target:
        target.write(markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path, help="Immutable release containing manifest.json and cases/dev.jsonl")
    parser.add_argument("--run", required=True, action="append", nargs=3,
                        metavar=("NAME", "GRADE_DIR", "OBSERVATION_DIR"),
                        help="Repeat for each saved report.json/graded.jsonl and observations.jsonl pair")
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory outside all input directories")
    args = parser.parse_args()
    result = report(args.release, args.run, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "runs": list(result["runs"]),
                      "release_dev_cases": result["dataset"]["requests"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
