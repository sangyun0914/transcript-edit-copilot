"""Prepare and grade variable-count seven-action development experiments.

References stay on the grading side. Inference requests contain only model-visible
input. Every requested case needs an observation, including failed requests.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

import passive_actions as contract
import passive_formats_actions_v3 as formats
import prepare_passive_actions_v3 as dataset
import prepare_actions_v3_conditions as conditions
import grading
import verify_release_export
from grading import transcript_media
from replay_session import session_hash
from verify_release_export import verify_operation

VERSION = "passive-actions-v3-evaluation-1"
require, packed, digest, sha256 = (
    dataset.require,
    dataset.packed,
    dataset.digest,
    dataset.sha256,
)
rows, write, write_rows = dataset.rows, dataset.write, dataset.write_rows


def file_binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def verify_files(directory, manifest):
    directory = Path(directory).resolve()
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        require(
            path.is_relative_to(directory) and path.is_file(),
            "Payload escaped its release.",
        )
        require(sha256(path) == expected, f"Frozen payload changed: {name}")


def semantic_input_hash(snapshot):
    return digest(contract.wire_snapshot(snapshot))


def result_hash(document):
    return digest(transcript_media(document))


def references(case):
    values = case["accepted_predictions"]
    require(
        values and values[0] == case["primary"] and values == case["acceptable"],
        "Primary/alternative linkage changed.",
    )
    return values


def prepare(release, review, output, eligibility_format="common"):
    release, output = Path(release).resolve(), Path(output).resolve()
    require(not output.exists(), "Use a fresh evaluation preparation directory.")
    require(
        eligibility_format in ("common", *formats.FORMATS),
        "Unknown format eligibility cohort.",
    )
    selected_formats = (
        formats.FORMATS if eligibility_format == "common" else (eligibility_format,)
    )
    manifest = json.loads((release / "manifest.json").read_text())
    verify_files(release, manifest)
    evidence = json.loads(Path(review).read_text())
    require(
        evidence["status"] == "passed"
        and evidence["manifest_sha256"] == sha256(release / "manifest.json")
        and evidence["model_predictions_consulted"] is False,
        "Require review before model outputs.",
    )
    cases = rows(release / "cases/dev.jsonl")
    require(
        cases and len({case["id"] for case in cases}) == len(cases),
        "Development roster must be nonempty and unique.",
    )
    tokenizer, tokenizer_identity = dataset.legacy.tokenizer()
    builder = contract.PassiveContextBuilder(tokenizer)
    proof_verification = conditions.verify_reference_proofs(
        release, manifest, cases, builder
    )
    admitted, exclusions = [], []
    requests = {name: [] for name in selected_formats}
    for case in cases:
        require(case["split"] == "dev", "Evaluation accepts development cases only.")
        bundle = dataset.runtime_bundle(case, builder)
        require(bundle["input"] == case["input"], "Development runtime window changed.")
        encoded = {
            name: formats.prefix_ids(tokenizer, case["input"], name)
            for name in formats.FORMATS
        }
        if any(len(encoded[name]) > formats.INPUT_LIMIT for name in selected_formats):
            exclusions.append(
                {
                    "case_id": case["id"],
                    "reason": "format_input_budget",
                    "eligibility_format": eligibility_format,
                    "input_tokens": {
                        name: len(value) for name, value in encoded.items()
                    },
                }
            )
            continue
        accepted = [
            result_hash(contract.apply_passive_edit(case["document"], bundle, value))
            for value in references(case)
        ]
        admitted.append({**case, "bundle": bundle, "accepted_result_sha256": accepted})
        for name in selected_formats:
            messages = formats.messages(case["input"], name)
            requests[name].append(
                {
                    "case_id": case["id"],
                    "split": "dev",
                    "format": name,
                    "input": case["input"],
                    "messages": messages,
                    "input_sha256": semantic_input_hash(case["input"]),
                    "messages_sha256": digest(messages),
                    "prefix_ids": encoded[name],
                    "prefix_ids_sha256": digest(encoded[name]),
                }
            )
    require(admitted, "No development cases satisfy the selected format budget.")
    output.mkdir(parents=True)
    write_rows(output / "cases.jsonl", admitted)
    for name, records in requests.items():
        write_rows(output / f"requests-{name}.jsonl", records)
    write_rows(output / "exclusions.jsonl", exclusions)
    write(
        output / "manifest.json",
        {
            "version": VERSION,
            "status": "prepared_not_inferred",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_manifest": file_binding(release / "manifest.json"),
            "semantic_review": file_binding(review),
            "tokenizer": tokenizer_identity,
            "reference_proof_verification": proof_verification,
            "requests": len(admitted),
            "eligible_formats": list(selected_formats),
            "eligibility_format": eligibility_format,
            "original_development_cases": len(cases),
            "excluded": len(exclusions),
            "case_ids": [case["id"] for case in admitted],
            "final_test_bodies_read": False,
            "files_sha256": {
                path.name: sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
            "sources_sha256": {
                str(Path(path).resolve()): sha256(path)
                for path in [
                    __file__,
                    formats.__file__,
                    contract.__file__,
                    contract.PROMPT_PATH,
                    grading.__file__,
                    verify_release_export.__file__,
                    dataset.__file__,
                    conditions.__file__,
                ]
            },
        },
    )
    return {
        "requests": len(admitted),
        "excluded": len(exclusions),
        "manifest": file_binding(output / "manifest.json"),
    }


def evaluate_one(case, observation, format_name="json"):
    require(
        observation["case_id"] == case["id"], "Observation belongs to another case."
    )
    require(
        observation["input_sha256"] == semantic_input_hash(case["input"]),
        "Observation used another input.",
    )
    require(observation["format"] == format_name, "Observation format differs.")
    require(
        observation["messages_sha256"]
        == digest(
            conditions.expected_messages(case["input"], format_name, case["condition"])
            if "condition" in case
            else formats.messages(case["input"], format_name)
        ),
        "Observation used a different prompt or condition.",
    )
    if "condition" in case:
        require(
            case["condition_sha256"] == digest(case["condition"])
            and observation.get("condition_sha256") == case["condition_sha256"],
            "Observation belongs to a different registered condition.",
        )
    judgment = case["judgment"]["status"]
    require(
        judgment in ("edit_required", "preserve"),
        "Unknown references cannot be scored as abstention.",
    )
    require(
        (case["primary"]["action"] == "no_action") == (judgment == "preserve"),
        "Reference judgment contradicts primary.",
    )
    require(
        all(
            (value["action"] == "no_action") == (judgment == "preserve")
            for value in references(case)
        ),
        "Accepted references disagree on whether this state needs an edit.",
    )
    before, bundle = case["document"], case["bundle"]
    require(
        session_hash(before) == bundle["document_hash"]
        and bundle["input"] == case["input"],
        "Case runtime binding changed.",
    )
    accepted = {
        result_hash(contract.apply_passive_edit(before, bundle, value))
        for value in references(case)
    }
    require(
        accepted == set(case["accepted_result_sha256"]), "Reference execution changed."
    )
    lineage = case.get("lineage", {})
    row = {
        "case_id": case["id"],
        "reference_action": case["primary"]["action"],
        "language": case["language"],
        "action_language": f"{case['primary']['action']}:{case['language']}",
        "source_group": case["source"]["source_group"],
        "stratum": lineage.get(
            "stratum",
            case["source"].get("provenance", case["source"].get("kind", "unspecified")),
        ),
        "scenario_family": lineage.get("scenario_family", "legacy_unspecified"),
        "selection_method": lineage.get("selection_method", "legacy_unspecified"),
        "judgment": judgment,
        "acoustic_evidence": case["judgment"].get("acoustic_evidence", "none"),
        "status": "request_failure",
        "valid": False,
        "displayed": False,
        "suppressed": False,
        "useful": False,
        "preserved": False,
        "model_abstained": False,
        "outcome_match": False,
        "predicted_action": None,
        "predicted_action_language": None,
        "source_integrity": None,
    }
    elapsed = observation.get("generation_ms")
    require(
        elapsed is None
        or type(elapsed) in (int, float)
        and math.isfinite(elapsed)
        and elapsed >= 0,
        "Elapsed time is not a measured nonnegative number.",
    )
    row["generation_ms"] = elapsed
    if observation.get("status") != "completed":
        row["error"] = observation.get("error", "Request did not complete.")
        return row
    try:
        prediction = formats.parse(
            observation["text"],
            case["input"],
            format_name,
            finish_reason=observation["finish_reason"],
        ).model_dump()
    except (ValueError, TypeError, KeyError) as error:
        row.update(status="invalid_output", error=str(error))
        return row
    row.update(
        valid=True,
        predicted_action=prediction["action"],
        predicted_action_language=f"{prediction['action']}:{case['language']}",
        model_abstained=prediction["action"] == "no_action",
    )
    ledger = case["runtime"].get("feedback_journal")
    suppressed = bool(
        ledger
        and contract.PassiveFeedbackLedger.restore(ledger).suppressed(
            before, bundle, prediction, case["runtime"]["sequence"]
        )
    )
    effective = {"action": "no_action"} if suppressed else prediction
    try:
        after = contract.apply_passive_edit(before, bundle, effective)
        verify_operation(before, after, bundle["mapping"], effective)
    except (ValueError, AssertionError, KeyError) as error:
        row.update(
            status="execution_failure", source_integrity="failed", error=str(error)
        )
        return row
    matched = result_hash(after) in accepted
    displayed = not suppressed and prediction["action"] != "no_action"
    row.update(
        status="completed",
        source_integrity="passed",
        suppressed=suppressed,
        displayed=displayed,
        outcome_match=matched,
        result_sha256=result_hash(after),
        useful=displayed and matched and judgment == "edit_required",
        preserved=judgment == "preserve" and matched and after == before,
    )
    return row


def ratio(numerator, denominator):
    value = numerator / denominator if denominator else None
    interval = None
    if denominator:
        z = 1.959963984540054
        center = (value + z * z / (2 * denominator)) / (1 + z * z / denominator)
        radius = (
            z
            * math.sqrt(
                value * (1 - value) / denominator + z * z / (4 * denominator**2)
            )
            / (1 + z * z / denominator)
        )
        interval = [max(0, center - radius), min(1, center + radius)]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "wilson95_descriptive": interval,
    }


def summarize(records):
    count = len(records)
    needed = sum(row["judgment"] == "edit_required" for row in records)
    preserve = count - needed
    displayed = sum(row["displayed"] for row in records)
    useful = sum(row["useful"] for row in records)
    times = sorted(
        row["generation_ms"] for row in records if row["generation_ms"] is not None
    )
    return {
        "requests": count,
        "status_counts": dict(Counter(row["status"] for row in records)),
        "precision": ratio(useful, displayed),
        "recall": ratio(useful, needed),
        "f1": 2 * useful / (displayed + needed) if displayed + needed else None,
        "preservation": ratio(sum(row["preserved"] for row in records), preserve),
        "model_preservation": ratio(
            sum(row["preserved"] and row["model_abstained"] for row in records),
            preserve,
        ),
        "coverage": ratio(displayed, count),
        "validity": ratio(sum(row["valid"] for row in records), count),
        "outcome_accuracy": ratio(sum(row["outcome_match"] for row in records), count),
        "suppressed": sum(row["suppressed"] for row in records),
        "source_integrity_failures": sum(
            row["source_integrity"] == "failed" for row in records
        ),
        "successful_model_actions": dict(
            sorted(
                Counter(
                    row["predicted_action"]
                    for row in records
                    if row["useful"] or (row["preserved"] and row["model_abstained"])
                ).items()
            )
        ),
        "latency_ms": {
            "count": len(times),
            "p50": statistics.median(times) if times else None,
            "p95": times[math.ceil(len(times) * 0.95) - 1] if times else None,
        },
    }


def verify_observations(run_directory, requests_path, format_name):
    run_directory = Path(run_directory)
    manifest = json.loads((run_directory / "manifest.json").read_text())
    require(
        manifest["status"] == "inference_complete",
        "Incomplete runs are not complete evaluations.",
    )
    require(
        {"run.json", "observations.jsonl"} <= set(manifest["files_sha256"]),
        "Run manifest omits its evidence.",
    )
    verify_files(run_directory, manifest)
    info = json.loads((run_directory / "run.json").read_text())
    require(
        info["status"] == manifest["status"]
        and info["version"] == manifest["version"]
        and info["requests_sha256"] == sha256(requests_path)
        and info["format"] == format_name
        and info["final_test_bodies_read"] is False,
        "Run differs from the registered request condition.",
    )
    identity = info["model_identity"]
    require(
        isinstance(identity, dict)
        and bool(identity.get("kind"))
        and info["numerical_regime"]
        and type(info["schema_constrained"]) is bool,
        "Model identity and numerical regime must be explicit.",
    )
    requests, observed = rows(requests_path), rows(run_directory / "observations.jsonl")
    require(
        info["requested"] == info["observed"] == len(requests) == len(observed),
        "Run census is incomplete.",
    )
    for request, observation in zip(requests, observed, strict=True):
        if "condition" in request:
            conditions.validate_request(request)
            require(
                observation.get("condition_sha256")
                == info.get("condition_sha256")
                == request["condition_sha256"],
                "Run mixes registered conditions.",
            )
        require(
            observation["model_identity"] == identity,
            "A run mixes or omits model identities.",
        )
        for key in (
            "case_id",
            "format",
            "input_sha256",
            "messages_sha256",
            "prefix_ids_sha256",
        ):
            require(
                observation[key] == request[key],
                f"Observation request binding differs: {key}",
            )
    return observed, info


def grade(preparation, run_directory, output, format_name):
    preparation, output = Path(preparation), Path(output)
    require(not output.exists(), "Use a fresh graded-result directory.")
    manifest = json.loads((preparation / "manifest.json").read_text())
    require(
        manifest["version"] in (VERSION, conditions.VERSION)
        and manifest["final_test_bodies_read"] is False,
        "Unknown evaluation scope.",
    )
    verify_files(preparation, manifest)
    for name, expected in manifest["sources_sha256"].items():
        require(
            sha256(name) == expected, "Evaluation source changed after registration."
        )
    observed, run_info = verify_observations(
        run_directory, preparation / f"requests-{format_name}.jsonl", format_name
    )
    cases = rows(preparation / "cases.jsonl")
    require(
        [row["case_id"] for row in observed] == [case["id"] for case in cases],
        "Every request requires one ordered observation, including failures.",
    )
    records = [
        evaluate_one(case, row, format_name)
        for case, row in zip(cases, observed, strict=True)
    ]
    grouped = {}
    for dimension in (
        "reference_action",
        "language",
        "action_language",
        "predicted_action",
        "predicted_action_language",
        "source_group",
        "stratum",
        "scenario_family",
        "selection_method",
    ):
        groups = defaultdict(list)
        for record in records:
            groups[str(record[dimension])].append(record)
        grouped[dimension] = {
            key: summarize(values) for key, values in sorted(groups.items())
        }
    output.mkdir(parents=True)
    write_rows(output / "graded.jsonl", records)
    report = {
        "version": VERSION,
        "status": "graded_development",
        "format": format_name,
        "preparation": file_binding(preparation / "manifest.json"),
        "observations": file_binding(Path(run_directory) / "observations.jsonl"),
        "inference_manifest": file_binding(Path(run_directory) / "manifest.json"),
        "inference_run": file_binding(Path(run_directory) / "run.json"),
        "model_identity": run_info["model_identity"],
        "numerical_regime": run_info["numerical_regime"],
        "schema_constrained": run_info["schema_constrained"],
        "condition": run_info.get("condition", "canonical_full_context"),
        "condition_sha256": run_info.get("condition_sha256"),
        "graded": file_binding(output / "graded.jsonl"),
        "overall": summarize(records),
        "by": grouped,
        "excluded_before_inference": manifest["excluded"],
        "eligibility_format": manifest.get("eligibility_format", "matched_common"),
        "original_development_cases": manifest.get("original_development_cases"),
        "final_test": False,
        "limitations": "Known AI-authored references, exposed development. Micro counts and Wilson intervals are descriptive; related cases are not independent sessions. Source integrity does not certify acoustic usefulness.",
    }
    write(output / "report.json", report)
    return report["overall"]


def baseline(preparation, output, name="always_no_action"):
    output = Path(output)
    require(not output.exists(), "Use a fresh baseline run directory.")
    requests_path = Path(preparation) / "requests-json.jsonl"
    requests = rows(requests_path)
    output.mkdir(parents=True)
    identity = {"kind": "deterministic_baseline", "name": name}
    write_rows(
        output / "observations.jsonl",
        [
            {
                "case_id": item["case_id"],
                "input_sha256": item["input_sha256"],
                "messages_sha256": item["messages_sha256"],
                "prefix_ids_sha256": item["prefix_ids_sha256"],
                "format": "json",
                "status": "completed",
                "text": packed(conditions.baseline_prediction(item["input"], name)),
                "finish_reason": "stop",
                "model_identity": identity,
                "generation_ms": None,
                **(
                    {"condition_sha256": item["condition_sha256"]}
                    if "condition" in item
                    else {}
                ),
            }
            for item in requests
        ],
    )
    write(
        output / "run.json",
        {
            "version": VERSION,
            "status": "inference_complete",
            "format": "json",
            "requests_sha256": sha256(requests_path),
            "requested": len(requests),
            "observed": len(requests),
            "model_identity": identity,
            "numerical_regime": "deterministic_no_model",
            "schema_constrained": False,
            "condition": requests[0].get("condition", {"id": "canonical_full_context"})[
                "id"
            ],
            "condition_sha256": requests[0].get("condition_sha256"),
            "model_requests": 0,
            "final_test_bodies_read": False,
        },
    )
    write(
        output / "manifest.json",
        {
            "version": VERSION,
            "status": "inference_complete",
            "files_sha256": {path.name: sha256(path) for path in output.iterdir()},
        },
    )
    return {"observations": len(requests), "path": str(output), "model_requests": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    for name in ("release", "review", "output"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument(
        "--eligibility-format", choices=("common", *formats.FORMATS), default="common"
    )
    command = commands.add_parser("grade")
    for name in ("preparation", "run", "output"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--format", choices=formats.FORMATS, default="json")
    command = commands.add_parser("baseline")
    for name in ("preparation", "output"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument(
        "--name",
        choices=("always_no_action", "consistent_history_rule"),
        default="always_no_action",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.release, args.review, args.output, args.eligibility_format
        )
    elif args.command == "grade":
        result = grade(args.preparation, args.run, args.output, args.format)
    else:
        result = baseline(args.preparation, args.output, args.name)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
