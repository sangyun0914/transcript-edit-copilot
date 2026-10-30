"""Compare bound development reports on matched cases and source-group resamples."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

import evaluate_passive_actions_v3 as evaluation
import prepare_history_length_actions_v3 as history_length

METRICS = (
    "precision",
    "recall",
    "model_preservation",
    "outcome_accuracy",
    "coverage",
    "validity",
)


def load_report(path):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    evaluation.require(
        report["final_test"] is False, "This comparison is development-only."
    )
    for key in (
        "preparation",
        "observations",
        "inference_manifest",
        "inference_run",
        "graded",
    ):
        item = report[key]
        evaluation.require(
            evaluation.sha256(item["path"]) == item["sha256"],
            f"Changed report binding: {key}",
        )
    records = evaluation.rows(report["graded"]["path"])
    evaluation.require(
        evaluation.summarize(records) == report["overall"],
        "Saved totals differ from graded rows.",
    )
    preparation = Path(report["preparation"]["path"]).parent
    manifest = json.loads((preparation / "manifest.json").read_text())
    requests = preparation / f"requests-{report['format']}.jsonl"
    evaluation.require(
        evaluation.sha256(requests) == manifest["files_sha256"][requests.name],
        "Changed registered requests.",
    )
    inputs = {row["case_id"]: row for row in evaluation.rows(requests)}
    cases_path = preparation / "cases.jsonl"
    evaluation.require(
        evaluation.sha256(cases_path) == manifest["files_sha256"][cases_path.name],
        "Changed registered references.",
    )
    cases = evaluation.rows(cases_path)
    evaluation.require(
        len(inputs) == len(cases) and set(inputs) == {case["id"] for case in cases},
        "Duplicate or missing reference cases.",
    )
    for case in cases:
        inputs[case["id"]]["accepted_result_sha256"] = sorted(
            set(case["accepted_result_sha256"])
        )
    indexed = {row["case_id"]: row for row in records}
    evaluation.require(
        len(indexed) == len(records) and set(inputs) == set(indexed),
        "Duplicate or missing scored cases.",
    )
    return report, indexed, inputs


def matched(left, right, left_inputs, right_inputs, mode):
    identities = sorted(set(left) & set(right))
    evaluation.require(identities, "No common scored cases.")
    if mode == "history-length":
        evaluation.require(
            set(left) == set(right) == set(left_inputs) == set(right_inputs)
            and len(identities) == 8
            and len({left[identity]["source_group"] for identity in identities}) == 8
            and Counter((left[identity]["language"], left[identity]["scenario_family"]) for identity in identities)
            == Counter((language, mechanism) for language in ("en", "ko") for mechanism in history_length.MECHANISMS)
            and all(
                len({rows[identity]["history_length"]["length"] for identity in identities}) == 1
                and len({rows[identity]["history_length"]["fixture_sha256"] for identity in identities}) == 1
                for rows in (left_inputs, right_inputs)
            ),
            "History-length comparison requires both complete eight-episode arms.",
        )
    changed_inputs, changed_judgments = [], []
    for identity in identities:
        a, b = left[identity], right[identity]
        evaluation.require(
            all(a[key] == b[key] for key in ("source_group", "language", "stratum")),
            "Matched case provenance differs.",
        )
        x, y = left_inputs[identity]["input"], right_inputs[identity]["input"]
        references_match = (
            left_inputs[identity]["accepted_result_sha256"]
            == right_inputs[identity]["accepted_result_sha256"]
        )
        if mode == "models":
            evaluation.require(
                x == y
                and references_match
                and a["judgment"] == b["judgment"]
                and a["reference_action"] == b["reference_action"],
                "Model comparison changed visible input or judgment.",
            )
        elif mode == "history-length":
            history_length.validate_pair(left_inputs[identity], right_inputs[identity])
        else:
            evaluation.require(mode == "context", "Unknown comparison mode.")

            def clean(snapshot):
                return {
                    k: v
                    for k, v in snapshot.items()
                    if k not in ("recent_edits", "recent_feedback")
                }

            evaluation.require(
                clean(x) == clean(y),
                "Context comparison changed transcript or non-history state.",
            )
            for key in ("recent_edits", "recent_feedback"):
                evaluation.require(
                    x[key] == y[key] or not x[key] or not y[key],
                    "Context comparison replaced history contents.",
                )
        if x != y:
            changed_inputs.append(identity)
        if not references_match or (a["judgment"], a["reference_action"]) != (
            b["judgment"],
            b["reference_action"],
        ):
            changed_judgments.append(identity)
    return identities, changed_inputs, changed_judgments


def tallies(records):
    return {
        "requests": len(records),
        "needed": sum(row["judgment"] == "edit_required" for row in records),
        "preserve": sum(row["judgment"] == "preserve" for row in records),
        "shown": sum(row["displayed"] for row in records),
        "useful": sum(row["useful"] for row in records),
        "model_preserved": sum(
            row["preserved"] and row["model_abstained"] for row in records
        ),
        "correct": sum(row["outcome_match"] for row in records),
        "valid": sum(row["valid"] for row in records),
    }


def values(counts):
    pairs = {
        "precision": ("useful", "shown"),
        "recall": ("useful", "needed"),
        "model_preservation": ("model_preserved", "preserve"),
        "outcome_accuracy": ("correct", "requests"),
        "coverage": ("shown", "requests"),
        "validity": ("valid", "requests"),
    }
    return {
        name: counts[a] / counts[b] if counts[b] else None
        for name, (a, b) in pairs.items()
    }


def quantile(sorted_values, fraction):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (
        position - low
    )


def source_resampling(left, right, identities, *, resamples=2000, seed=3407):
    groups = defaultdict(list)
    for identity in identities:
        groups[left[identity]["source_group"]].append(identity)
    names = sorted(groups)
    grouped = {
        name: (
            tallies([left[i] for i in groups[name]]),
            tallies([right[i] for i in groups[name]]),
        )
        for name in names
    }
    distributions = {name: [] for name in METRICS}
    randomizer = random.Random(seed)
    if len(names) >= 2:
        for _ in range(resamples):
            counts = (Counter(), Counter())
            for name in randomizer.choices(names, k=len(names)):
                for target, source in zip(counts, grouped[name], strict=True):
                    target.update(source)
            a, b = map(values, counts)
            for name in METRICS:
                if a[name] is not None and b[name] is not None:
                    distributions[name].append(b[name] - a[name])
    return {
        "source_groups": len(names),
        "resamples": resamples,
        "seed": seed,
        "unit": "registered source group, sampled with replacement; identical group draw in both arms",
        "difference": "right minus left",
        "percentile95_descriptive": {
            name: {
                "low": quantile(sorted(sample), 0.025),
                "high": quantile(sorted(sample), 0.975),
                "defined_resamples": len(sample),
                "undefined_resamples": resamples - len(sample),
            }
            for name, sample in distributions.items()
        },
        "limitation": "Exploratory resampling of this designed, exposed dataset. Source groups may still share templates or editing mechanisms; this is not a population confidence guarantee, a seed-replication result or a human study. Fewer than two groups produces no interval; zero metric denominators are retained as undefined.",
    }


def compare(left_path, right_path, output, mode="models"):
    output = Path(output)
    evaluation.require(not output.exists(), "Use a new comparison directory.")
    left_report, left, left_inputs = load_report(left_path)
    right_report, right, right_inputs = load_report(right_path)
    if mode == "history-length":
        validate_history_regime(left_report, right_report)
    identities, changed, relabeled = matched(
        left, right, left_inputs, right_inputs, mode
    )
    groups = defaultdict(list)
    for identity in identities:
        groups[left[identity]["source_group"]].append(identity)

    def pair_summary(ids):
        return {
            "cases": len(ids),
            "left": evaluation.summarize([left[i] for i in ids]),
            "right": evaluation.summarize([right[i] for i in ids]),
            "paired_outcome_counts": dict(
                Counter(
                    (
                        "both_correct"
                        if left[i]["outcome_match"] and right[i]["outcome_match"]
                        else "left_only"
                        if left[i]["outcome_match"]
                        else "right_only"
                        if right[i]["outcome_match"]
                        else "neither_correct"
                    )
                    for i in ids
                )
            ),
        }

    result = {
        "version": "passive-actions-v3-paired-comparison-1",
        "mode": mode,
        "left_report": evaluation.file_binding(left_path),
        "right_report": evaluation.file_binding(right_path),
        "implementation": evaluation.file_binding(__file__),
        "left_model": left_report["model_identity"],
        "right_model": right_report["model_identity"],
        "numerical_regimes": [
            left_report["numerical_regime"],
            right_report["numerical_regime"],
        ],
        "left_only_cases": sorted(set(left) - set(right)),
        "right_only_cases": sorted(set(right) - set(left)),
        "matched_case_ids": identities,
        "matched": pair_summary(identities),
        "by_source_group": {
            name: pair_summary(ids) for name, ids in sorted(groups.items())
        },
        "changed_inputs": {"case_ids": changed, **pair_summary(changed)},
        "changed_reference_judgments": {
            "case_ids": relabeled,
            **pair_summary(relabeled),
        },
        "source_resampling": source_resampling(left, right, identities),
        "method_reference": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html",
        "method_scope": "Paired replacement sampling and percentile calculation follow the documented definitions. Choosing source groups as units is this project's attempt to retain within-source dependence, not a claim that the source documents endorse this benchmark design.",
        "final_test": False,
    }
    if mode == "history-length":
        result["history_length"] = {
            "version": history_length.VERSION,
            "left_arm": left_inputs[identities[0]]["history_length"]["arm"],
            "right_arm": right_inputs[identities[0]]["history_length"]["arm"],
            "episodes": 8,
            "paired_views": 16,
            "claim_scope": "Authored symbolic mechanism controls; not natural history benefit or causal training-source evidence.",
            "per_episode": [
                {
                    "case_id": identity,
                    "language": left[identity]["language"],
                    "mechanism": left[identity]["scenario_family"],
                    "left_reference_action": left[identity]["reference_action"],
                    "right_reference_action": right[identity]["reference_action"],
                    "left_outcome_match": left[identity]["outcome_match"],
                    "right_outcome_match": right[identity]["outcome_match"],
                    "left_valid": left[identity]["valid"],
                    "right_valid": right[identity]["valid"],
                }
                for identity in identities
            ],
        }
    output.mkdir(parents=True)
    evaluation.write(output / "comparison.json", result)
    return {
        "matched": len(identities),
        "changed_inputs": len(changed),
        "changed_judgments": len(relabeled),
        "output": str(output),
    }


def validate_history_regime(left, right):
    evaluation.require(
        all(
            left[key] == right[key]
            for key in ("model_identity", "numerical_regime", "format", "schema_constrained")
        ),
        "History-length comparison must keep model, numerical regime, format and output constraint fixed.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("models", "context", "history-length"), default="models")
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right, args.output, args.mode), indent=2))
