"""Bind small-model DEV runs to the existing seven-action effect grader."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import evaluate_passive_actions_v3 as grading
import small_actions_final as data
import train_small_actions_final as training_runtime

VERSION = "small-actions-final-evaluation-v1"


def prepare(evaluation, model_preparation, output, *, assets=None):
    evaluation = Path(evaluation).resolve()
    original = data.read(evaluation / "manifest.json")
    data.require(
        original["version"] == grading.VERSION
        and original["final_test_bodies_read"] is False,
        "Only exposed DEV is allowed.",
    )
    grading.verify_files(evaluation, original)
    for path, expected in original["sources_sha256"].items():
        data.require(
            data.sha(path) == expected, "Development preparation source changed."
        )
    config, _, assets = data.prepared(model_preparation, assets)
    tokenizer = data.tokenizer_from(assets, config["model"])
    requests = data.rows(evaluation / "requests-json.jsonl")
    for row in requests:
        data.require(
            row["split"] == "dev"
            and row["messages"] == data.formats.messages(row["input"]),
            "Saved canonical DEV request changed.",
        )
        prefix = data.prompt_ids(tokenizer, row["messages"])
        data.require(
            len(prefix) <= config["context_tokens"] - config["max_completion_tokens"],
            "The complete DEV cohort must fit; no truncation or exclusions.",
        )
        row.update(prefix_ids=prefix, prefix_ids_sha256=data.value_hash(prefix))
    output = data.new_directory(output)
    for name in ("cases.jsonl", "exclusions.jsonl"):
        shutil.copyfile(evaluation / name, output / name)
    data.write_rows(output / "requests-json.jsonl", requests)
    manifest = {
        **original,
        "small_model_version": VERSION,
        "parent_evaluation": grading.file_binding(evaluation / "manifest.json"),
        "model_preparation": grading.file_binding(
            Path(model_preparation) / "configuration.json"
        ),
        "tokenizer": config["assets"],
        "native_model": config["base"],
        "files_sha256": {p.name: data.sha(p) for p in output.iterdir() if p.is_file()},
        "sources_sha256": {
            **original["sources_sha256"],
            str(Path(__file__).resolve()): data.sha(__file__),
        },
    }
    data.write(output / "manifest.json", manifest)
    return {
        "requests": len(requests),
        "model": config["model"],
        "max_prompt_tokens": max(len(row["prefix_ids"]) for row in requests),
        "cases_byte_identical": data.sha(output / "cases.jsonl")
        == original["files_sha256"]["cases.jsonl"],
    }


def validate_observations(requests, observations):
    data.require(
        len(requests) == len(observations),
        "Every DEV request needs an observation, including failures.",
    )
    for request, row in zip(requests, observations, strict=True):
        for key in (
            "case_id",
            "format",
            "input_sha256",
            "messages_sha256",
            "prefix_ids_sha256",
        ):
            data.require(
                row.get(key) == request[key], f"Native request binding differs: {key}"
            )


def admit(evaluation, run, output, *, exported=None, training=None, reloaded=None):
    evaluation, run = Path(evaluation).resolve(), Path(run).resolve()
    prepared = data.read(evaluation / "manifest.json")
    data.require(
        prepared.get("small_model_version") == VERSION
        and prepared["final_test_bodies_read"] is False,
        "Require a small-model DEV preparation.",
    )
    grading.verify_files(evaluation, prepared)
    requests_path = evaluation / "requests-json.jsonl"
    requests, observations = (
        data.rows(requests_path),
        data.rows(run / "observations.jsonl"),
    )
    validate_observations(requests, observations)
    info, runtime = data.read(run / "run.json"), data.read(run / "runtime.json")
    data.require(
        info["requested"] == len(requests) and info["final_test_bodies_read"] is False,
        "Run census or scope differs from DEV.",
    )
    if exported is None:
        data.require(
            info["status"] == "inference_complete"
            and info["request_sha256"] == data.sha(requests_path)
            and info["observations_sha256"] == data.sha(run / "observations.jsonl")
            and info["configuration_sha256"] == prepared["model_preparation"]["sha256"]
            and runtime["base"] == prepared["native_model"],
            "Native run identity differs.",
        )
        if info["model_identity"] is not None:
            data.require(
                training is not None and reloaded is not None,
                "Trained native admission requires its training and independent reload.",
            )
            actual_identity = training_runtime.model_identity(training, reloaded)
            data.require(
                actual_identity == info["model_identity"]
                and actual_identity["configuration_sha256"]
                == prepared["model_preparation"]["sha256"]
                and info["kind"] == actual_identity["kind"],
                "Native trained identity differs from the verified adapter and reload.",
            )
        else:
            data.require(
                training is None and reloaded is None and info["kind"] == "base",
                "Base admission must not supply a trained model identity.",
            )
        identity = {
            **(info["model_identity"] or {"kind": "base"}),
            "base": runtime["base"],
        }
        numerical_regime = (
            "native_bf16_fp32_lora" if info["model_identity"] else "native_bf16_base"
        )
        constrained = False
    else:
        data.require(
            training is None and reloaded is None,
            "Use the export identity for GGUF admission.",
        )
        exported = Path(exported).resolve()
        export = data.read(exported / "run.json")
        props = runtime["props"]
        model = exported / f"model-{export['quantization']}.gguf"
        data.require(
            info["status"] == "runtime_preflight_complete"
            and runtime["request_sha256"] == data.sha(requests_path)
            and Path(props["model_path"]).resolve() == model
            and props["model_alias"] == runtime["alias"]
            and data.sha(model) == export["gguf_sha256"]
            and export["base"] == prepared["native_model"]
            and data.sha(exported / "configuration.json")
            == export["training_configuration_sha256"]
            == prepared["model_preparation"]["sha256"]
            and info["observations_sha256"] == data.sha(run / "observations.jsonl"),
            "GGUF run identity differs.",
        )
        data.require(
            all(
                row.get("native_token_parity") is True
                for row in observations
                if row["status"] == "completed"
            ),
            "GGUF prompt token parity was not verified.",
        )
        identity = {
            "kind": export.get("kind", "sft"),
            "base": export["base"],
            "adapter_sha256": export["adapter_sha256"],
            "gguf_sha256": export["gguf_sha256"],
            "export_receipt_sha256": data.sha(exported / "run.json"),
        }
        for key in ("parent_adapter_sha256", "dpo_configuration_sha256"):
            if key in export:
                identity[key] = export[key]
        numerical_regime = "native_bf16_merged_" + export["quantization"].lower()
        constrained = True
    output = data.new_directory(output)
    normalized = [
        {
            **row,
            "model_identity": identity,
            "generation_ms": row.get("generation_ms", row.get("http_response_ms")),
        }
        for row in observations
    ]
    data.write_rows(output / "observations.jsonl", normalized)
    result = {
        "version": VERSION,
        "status": "inference_complete",
        "format": "json",
        "requested": len(requests),
        "observed": len(observations),
        "requests_sha256": data.sha(requests_path),
        "model_identity": identity,
        "numerical_regime": numerical_regime,
        "schema_constrained": constrained,
        "raw_run": grading.file_binding(run / "run.json"),
        "raw_observations": grading.file_binding(run / "observations.jsonl"),
        "raw_runtime": grading.file_binding(run / "runtime.json"),
        "final_test_bodies_read": False,
    }
    data.write(output / "run.json", result)
    data.write(
        output / "manifest.json",
        {
            "version": VERSION,
            "status": "inference_complete",
            "files_sha256": {
                p.name: data.sha(p) for p in output.iterdir() if p.is_file()
            },
        },
    )
    grading.verify_observations(output, requests_path, "json")
    return {"observed": len(observations), "numerical_regime": numerical_regime}


def semantic_macro_f1(records):
    actions = data.formats.contract.ACTIONS
    true, false, needed = Counter(), Counter(), Counter()
    for row in records:
        reference = row["reference_action"]
        needed[reference] += 1
        correct = row["useful"] or (row["preserved"] and row["model_abstained"])
        if correct:
            true[reference] += 1
        elif row["predicted_action"] in actions:
            false[row["predicted_action"]] += 1
    by = {}
    for action in actions:
        tp, fp, fn = true[action], false[action], needed[action] - true[action]
        denominator = 2 * tp + fp + fn
        by[action] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": 2 * tp / denominator if denominator else None,
        }
    values = [row["f1"] for row in by.values() if row["f1"] is not None]
    return {
        "value": sum(values) / len(values) if values else None,
        "actions": by,
        "definition": "Useful effect or correct model abstention is TP in the fixed primary reference stratum, including accepted alternatives. Incorrect valid predictions add FP to their predicted action and FN to the reference. Invalid/missing predictions add FN. Average the seven per-action F1s.",
    }


def summarize(report, cohort_mapping, output):
    report_path = Path(report)
    report = data.read(report_path)
    records = data.rows(report["graded"]["path"])
    data.require(
        data.sha(report["graded"]["path"]) == report["graded"]["sha256"]
        and grading.summarize(records) == report["overall"],
        "Graded totals changed.",
    )
    mapping = data.read(cohort_mapping)
    by_id = {row["case_id"]: row for row in mapping["cases"]}
    data.require(
        set(by_id) == {row["case_id"] for row in records}
        and len(by_id) == len(records),
        "Cohort mapping differs from the complete DEV roster.",
    )
    cohorts = {}
    for name, expected in mapping["counts"].items():
        selected = [row for row in records if by_id[row["case_id"]]["cohort"] == name]
        data.require(len(selected) == expected, "Cohort denominator changed.")
        success = Counter(
            f"{row['predicted_action']}:{row['language']}"
            for row in selected
            if row["useful"] or (row["preserved"] and row["model_abstained"])
        )
        cohorts[name] = {
            "metrics": grading.summarize(selected),
            "macro_action_f1": semantic_macro_f1(selected),
            "successful_action_language": dict(sorted(success.items())),
        }
    failures = []
    for name, limits in (
        ("canonical_dev322", (0.88, 0.83, 58 / 69, 17)),
        ("supplemental_dev149", (0.85, None, 0.90, None)),
    ):
        stats = cohorts[name]["metrics"]
        for metric, threshold in zip(
            ("precision", "recall", "model_preservation"), limits[:3], strict=True
        ):
            if threshold is not None and (
                stats[metric]["value"] is None or stats[metric]["value"] < threshold
            ):
                failures.append(f"{name}:{metric}")
        invalid = sum(
            value for key, value in stats["status_counts"].items() if key != "completed"
        )
        if limits[3] is not None and invalid > limits[3]:
            failures.append(f"{name}:invalid_or_failed")
        required = [
            (action, language)
            for action in data.formats.contract.ACTIONS
            if name == "canonical_dev322" or action != "no_action"
            for language in ("en", "ko")
        ]
        for action, language in required:
            if not cohorts[name]["successful_action_language"].get(
                f"{action}:{language}"
            ):
                failures.append(f"{name}:no_success:{action}:{language}")
    result = {
        "model_identity": report["model_identity"],
        "report": grading.file_binding(report_path),
        "cohort_mapping": grading.file_binding(cohort_mapping),
        "overall": report["overall"],
        "macro_action_f1": semantic_macro_f1(records),
        "cohorts": cohorts,
        "minimum_numeric_quality_gate": {"passed": not failures, "failures": failures},
        "additional_selection_review": "Inspect preservation/unknown-speaker/latest-correction subgroup failures, actual app quality/latency and quantized parity before selection.",
    }
    data.write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("evaluation", "model-preparation", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--assets", type=Path)
    observe = sub.add_parser("admit")
    for name in ("evaluation", "run", "output"):
        observe.add_argument("--" + name, type=Path, required=True)
    observe.add_argument("--exported", type=Path)
    observe.add_argument("--training", type=Path)
    observe.add_argument("--reloaded", type=Path)
    summary = sub.add_parser("summarize")
    for name in ("report", "cohort-mapping", "output"):
        summary.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.evaluation, args.model_preparation, args.output, assets=args.assets
        )
    elif args.command == "admit":
        result = admit(
            args.evaluation,
            args.run,
            args.output,
            exported=args.exported,
            training=args.training,
            reloaded=args.reloaded,
        )
    else:
        result = summarize(args.report, args.cohort_mapping, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
