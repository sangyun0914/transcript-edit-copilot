"""Summarize the frozen 8 x H0/H3/H8 DEV diagnostic without model execution."""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import compare_actions_v3_results as paired
import evaluate_passive_actions_v3 as evaluation
import prepare_final_unified_dataset as final
import prepare_history_length_actions_v3 as history
from compare_task_test_actions_v3 import outcome_pairs

VERSION = "final-history-actions-v3-summary-1"
LENGTHS = history.LENGTHS
require = evaluation.require


def read(path):
    return json.loads(Path(path).read_text())


def checked(binding):
    path = Path(binding["path"])
    require(evaluation.sha256(path) == binding["sha256"], "Changed evidence binding.")
    return path


def group_anchors(cases, manifest):
    require(len(cases) == 24 and len({case["id"] for case in cases}) == 24,
            "Require 24 unique history cases, including failed observations.")
    groups = defaultdict(dict)
    for case in cases:
        meta = case["history_length"]
        length, episode = meta["length"], meta["episode_id"]
        require(case["split"] == "dev" and type(length) is int and length in LENGTHS
                and meta["version"] == history.VERSION and meta["anchor_length"] == 8
                and meta["window_policy"] == history.WINDOW_POLICY
                and meta["arm"] == history.arm_name(length)
                and case["id"] == f"{episode}-h{length}", "Unregistered history arm.")
        require(length not in groups[episode], "Duplicate history arm for an anchor.")
        groups[episode][length] = case
    require(len(groups) == 8 and all(set(arms) == set(LENGTHS) for arms in groups.values()),
            "Require all eight anchors with H0/H3/H8 exactly once.")
    require(Counter((arms[0]["language"], arms[0]["lineage"]["scenario_family"])
                    for arms in groups.values())
            == Counter((language, mechanism) for language in ("en", "ko") for mechanism in final.MECHANISMS),
            "Require the registered language/mechanism roster.")
    bindings, fixtures, sources = {}, set(), set()
    for episode, arms in groups.items():
        anchor = arms[8]
        common = {k: v for k, v in anchor["history_length"].items() if k not in ("length", "arm")}
        fixtures.add(common["fixture_sha256"])
        sources.add(anchor["source"]["source_group"])
        require(set(common["projection_input_sha256"]) == {str(n) for n in LENGTHS}
                and common["anchor_input_sha256"] == common["projection_input_sha256"]["8"],
                "History anchor/projection binding differs.")
        for length, case in arms.items():
            require({k: v for k, v in case["history_length"].items() if k not in ("length", "arm")} == common
                    and case["source"] == anchor["source"] and case["language"] == anchor["language"]
                    and all(case["lineage"][key] == anchor["lineage"][key]
                            for key in ("stratum", "scenario_family", "selection_method")),
                    "An arm changed its anchor or provenance.")
            require(evaluation.semantic_input_hash(case["input"]) == common["projection_input_sha256"][str(length)]
                    and case["input"]["recent_feedback"] == [], "History projection or feedback changed.")
            final.bound_evaluation_bundle(case)
        bindings[episode] = [anchor["bundle"]["document_hash"], common["non_history_sha256"], common["history_chain_sha256"]]
    require(len(fixtures) == 1 and len(sources) == 8
            and bindings == manifest["history_anchor_bindings"], "Frozen history anchor census differs.")
    return dict(sorted(groups.items()))


def load_history(report_path):
    report, scored, inputs = paired.load_report(report_path)
    require(report["status"] == "graded_development" and report["format"] == "json"
            and report["condition"] == "canonical_full_context" and report["condition_sha256"] is None
            and report["excluded_before_inference"] == 0 and report["original_development_cases"] == 24,
            "Require the complete canonical final history DEV diagnostic.")
    preparation = Path(report["preparation"]["path"]).parent
    manifest = read(preparation / "manifest.json")
    require(manifest["version"] == evaluation.VERSION and manifest["final_test_bodies_read"] is False
            and manifest["final_builder_version"] == final.VERSION and manifest["cohort"] == "history"
            and manifest["excluded"] == 0 and manifest["requests"] == manifest["history_condition_cases"] == 24
            and manifest["history_episodes"] == manifest["unique_editor_anchors"] == 8,
            "Require the frozen final history preparation without exclusions.")
    evaluation.verify_files(preparation, manifest)
    for path, expected in manifest["sources_sha256"].items():
        require(evaluation.sha256(path) == expected, "Registered evaluation source changed.")
    checked(manifest["dataset_manifest"])
    config_path = checked(manifest["model_preparation"])
    config = read(config_path)
    require(config["base"] == manifest["native_model"], "Prepared model/config identity differs.")
    run_directory = Path(report["inference_run"]["path"]).parent
    for key, name in (("inference_run", "run.json"), ("inference_manifest", "manifest.json"), ("observations", "observations.jsonl")):
        require(report[key] == evaluation.file_binding(run_directory / name), "Report mixes inference runs.")
    observed, info = evaluation.verify_observations(run_directory, preparation / "requests-json.jsonl", "json")
    paired.validate_history_regime(report, info)
    require(info.get("condition", "canonical_full_context") == report["condition"]
            and info.get("condition_sha256") == report["condition_sha256"], "Report and run prompt conditions differ.")
    native_info = info
    if "raw_run" in info:
        native_info = read(checked(info["raw_run"]))
        raw_observations = evaluation.rows(checked(info["raw_observations"]))
        runtime = read(checked(info["raw_runtime"]))
        require(native_info["status"] == "inference_complete" and native_info["final_test_bodies_read"] is False
                and native_info["request_sha256"] == info["requests_sha256"]
                and native_info["observations_sha256"] == info["raw_observations"]["sha256"]
                and native_info["requested"] == len(observed) == len(raw_observations)
                and runtime["base"] == config["base"], "Raw native run/config binding differs.")
        require(info["model_identity"] == {**(native_info["model_identity"] or {"kind": "base"}), "base": runtime["base"]},
                "Admitted and raw model identities differ.")
        for raw, normalized in zip(raw_observations, observed, strict=True):
            require(normalized == {**raw, "model_identity": info["model_identity"],
                                  "generation_ms": raw.get("generation_ms", raw.get("http_response_ms"))},
                    "Admitted observation changed its raw result.")
    require(native_info["configuration_sha256"] == manifest["model_preparation"]["sha256"]
            and info["schema_constrained"] is False, "Native inference used a different configuration or constraint.")
    identity = info["model_identity"]
    if "base" not in identity:
        require(all(identity[key] == config["base"][key] for key in ("repository", "revision"))
                and identity.get("native_pin_sha256") == config.get("native_pin_sha256"),
                "Native model identity differs from its prepared base.")
    cases = evaluation.rows(preparation / "cases.jsonl")
    require([case["id"] for case in cases] == manifest["case_ids"] == [row["case_id"] for row in observed],
            "History case/request/observation order differs.")
    groups = group_anchors(cases, manifest)
    for case, observation in zip(cases, observed, strict=True):
        request = inputs[case["id"]]
        require(request["split"] == "dev" and request["input"] == case["input"]
                and request["messages"] == case["messages"] == evaluation.formats.messages(case["input"], "json")
                and request["messages_sha256"] == evaluation.digest(request["messages"])
                and request["input_sha256"] == evaluation.semantic_input_hash(request["input"])
                and request["prefix_ids"] and all(type(token) is int and token >= 0 for token in request["prefix_ids"])
                and request["prefix_ids_sha256"] == evaluation.digest(request["prefix_ids"]),
                "Saved request input, prompt or token binding differs.")
        require(evaluation.evaluate_one(case, observation, "json") == scored[case["id"]],
                "Saved grade differs from its actual observation and executable references.")
    return report, manifest, groups, scored, inputs


def reference(case):
    return {"judgment": case["judgment"]["status"], "primary": case["primary"],
            "accepted_predictions": evaluation.references(case),
            "accepted_result_sha256": sorted(set(case["accepted_result_sha256"]))}


def pair_summary(anchors, left, right):
    indexed = {length: {key: value["arms"][str(length)]["grade"] for key, value in anchors.items()}
               for length in (left, right)}
    outcomes = outcome_pairs(indexed[left], indexed[right], sorted(anchors))
    names = {"both_correct": "tie_both_correct", "left_only": "left_wins",
             "right_only": "right_wins", "neither_correct": "tie_both_incorrect"}
    changed, annotation_changed = [], []
    for key, value in anchors.items():
        a, b = (value["arms"][str(length)]["reference"] for length in (left, right))
        if (a["judgment"], a["accepted_result_sha256"]) != (b["judgment"], b["accepted_result_sha256"]):
            changed.append(key)
        if a != b:
            annotation_changed.append(key)
    return {"left": f"H{left}", "right": f"H{right}", "anchors": len(anchors),
            "outcomes": {names[key]: {"count": value["count"], "anchor_ids": value["case_ids"]}
                         for key, value in outcomes.items()},
            "changed_reference_effect_or_judgment": changed,
            "changed_reference_annotation": annotation_changed}


def describe(anchors):
    return {"anchors": len(anchors),
            "per_arm": {f"H{length}": evaluation.summarize([a["arms"][str(length)]["grade"] for a in anchors.values()])
                        for length in LENGTHS},
            "paired": {f"H{left}_vs_H{right}": pair_summary(anchors, left, right)
                       for left, right in combinations(LENGTHS, 2)}}


def summarize(report_path, output):
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh history-summary directory.")
    report, manifest, groups, scored, inputs = load_history(report_path)
    require(all(not output.is_relative_to(Path(report[key]["path"]).resolve().parent)
                for key in ("preparation", "inference_run", "graded")),
            "Write the summary outside frozen preparation, inference and grade directories.")
    anchors = {}
    for episode, cases in groups.items():
        row = scored[cases[0]["id"]]
        anchors[episode] = {key: row[key] for key in ("language", "source_group", "stratum", "scenario_family")}
        anchors[episode]["arms"] = {str(length): {
            "case_id": case["id"], "reference": reference(case), "grade": scored[case["id"]],
            **{key: inputs[case["id"]][key] for key in ("input_sha256", "messages_sha256", "prefix_ids_sha256")},
        } for length, case in sorted(cases.items())}
    result = {"version": VERSION, "status": "summarized_development", "final_test": False,
              "report": evaluation.file_binding(report_path), "preparation": report["preparation"],
              "dataset_manifest": manifest["dataset_manifest"], "model_preparation": manifest["model_preparation"],
              **{key: report[key] for key in ("model_identity", "numerical_regime", "schema_constrained")},
              "overall": report["overall"], **describe(anchors), "anchor_results": anchors,
              "by": {dimension: {value: describe({key: row for key, row in anchors.items() if row[dimension] == value})
                                 for value in sorted({row[dimension] for row in anchors.values()})}
                     for dimension in ("language", "source_group", "scenario_family")},
              "verification": "Existing bound-report/observation checks, actual saved history replay and existing grader recomputation; no inference or reference changes.",
              "implementation": evaluation.file_binding(__file__),
              "limitations": "Eight authored anchors with three correlated projections each, not 24 independent cases or an independent causal study. References can change with the visible history; paired wins score each arm against its own qualified reference. Language/mechanism/source concentration is descriptive, not natural-session or personalization evidence. Native generation time is not visible UI latency. This summary does not admit or select a final model."}
    output.mkdir(parents=True)
    evaluation.write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.report, args.output)
    print(json.dumps({"requests": result["overall"]["requests"], "anchors": result["anchors"],
                      "per_arm_correct": {key: value["outcome_accuracy"] for key, value in result["per_arm"].items()}}, indent=2))


if __name__ == "__main__":
    main()
