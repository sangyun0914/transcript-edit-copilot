"""Grade and append one authorized deployed GGUF row without changing native comparisons."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import compare_task_test_actions_v3 as native_comparison
import infer_deployed_task_test_actions_v3 as deployed
import prepare_task_test_actions_v3 as core

VERSION = deployed.VERSION
require, read, digest = core.require, core.read_json, core.digest


def admitted(context, directory, *, tokenizer=None):
    directory = Path(directory)
    manifest = deployed.load_output(context, directory, "admitted")
    raw = core.bound(manifest["observation_manifest"]).parent
    info, observations, _ = deployed.verify_observations(
        context, raw, tokenizer=tokenizer
    )
    require(
        core.rows(directory / "observations.jsonl") == observations
        and read(directory / "run.json") == info
        and manifest["requested"] == manifest["observed"] == len(observations),
        "Admitted product observations differ from actual frozen HTTP evidence.",
    )
    return observations


def grade_rows(context, sealed, observations):
    cases = core.rows(Path(sealed) / "cases.jsonl")
    proofs = core.rows(Path(sealed) / "reference-executions.jsonl")
    require(
        [row["id"] for row in cases] == context["release"]["case_ids"]
        and len({row["id"] for row in proofs}) == len(proofs),
        "Sealed case/reference census changed.",
    )
    cases = {row["id"]: row for row in cases}
    proofs = {row["id"]: row for row in proofs}
    require(
        set(cases) == set(proofs), "Missing or extra sealed reference execution proof."
    )
    records = []
    for request, observation in zip(context["requests"], observations, strict=True):
        case = cases[request["case_id"]]
        require(
            case["split"] == case["source"]["split"] == "final_test"
            and case["input"] == request["input"]
            and case["reference_proof_sha256"] == digest(proofs[case["id"]]),
            "Product grading changed an independently sealed input/reference.",
        )
        conditioned = {
            **case,
            "condition": request["condition"],
            "condition_sha256": request["condition_sha256"],
        }
        record = core.evaluation.evaluate_one(conditioned, observation, "json")
        records.append(
            {
                **record,
                "auxiliary_role": deployed.ROLE,
                "http_response_ms": observation.get("http_response_ms"),
                "numerical_regime": context["policy"]["numerical_regime"],
            }
        )
    return records


def http_latencies(records):
    def values(rows):
        return deployed.http.latency(
            [
                {"elapsed_ms": row["http_response_ms"]}
                for row in rows
                if row["http_response_ms"] is not None
            ]
        )

    return {
        "scope": "HTTP_complete_response_not_UI_or_native_CUDA_generation",
        "all_completed_http": values(records),
        "valid_displayable_proposal": values(
            [
                row
                for row in records
                if row["status"] == "completed" and row["displayed"]
            ]
        ),
        "model_no_action": values(
            [
                row
                for row in records
                if row["status"] == "completed" and row["model_abstained"]
            ]
        ),
        "invalid_or_failed_edit": values(
            [row for row in records if row["status"] != "completed"]
        ),
    }


def report(context, sealed, records, observations):
    dispositions = core.rows(Path(sealed) / "dispositions.jsonl")
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
    ):
        groups = defaultdict(list)
        for row in records:
            groups[
                row[dimension] if row[dimension] is not None else "unavailable"
            ].append(row)
        grouped[dimension] = {
            key: core.evaluation.summarize(values)
            for key, values in sorted(groups.items())
        }
    arm = {
        "observations": {row["case_id"]: row for row in observations},
        "records": {row["case_id"]: row for row in records},
    }
    return {
        **deployed.common(context),
        "run_id": deployed.ROLE,
        "summary": native_comparison.summarize(records),
        "by": grouped,
        "http_latency": http_latencies(records),
        "registered_slots": len(dispositions),
        "disposition_counts": dict(Counter(row["status"] for row in dispositions)),
        "canonical_admitted_cases": context["release"]["admitted_cases"],
        "run_eligibility_exclusions": [],
        "source_stratum_denominators": core.census_summary(
            dispositions, (row["case_id"] for row in records)
        ),
        "failure_attribution": native_comparison.failure_attribution(arm),
        "model_selection_changed": False,
        "native_core_roles_changed": False,
        "interpretation": "Auxiliary deployed product GGUF with schema/cache/runtime policy; not a native precision causal training-effect arm.",
    }


def grade(freeze, sealed, observed, output, *, tokenizer=None):
    context = deployed.load_authorized(freeze, sealed)
    observations = admitted(context, observed, tokenizer=tokenizer)
    records = grade_rows(context, sealed, observations)
    return core.publish(
        output,
        {
            "graded.jsonl": records,
            "report.json": report(context, sealed, records, observations),
        },
        {
            "version": VERSION,
            "status": "graded",
            **deployed.common(context),
            "observation_manifest": core.binding(Path(observed) / "manifest.json"),
        },
    )


def load_grade(freeze, sealed, directory, *, tokenizer=None):
    context = deployed.load_authorized(freeze, sealed)
    directory = Path(directory)
    manifest = deployed.load_output(context, directory, "graded")
    observed = core.bound(manifest["observation_manifest"]).parent
    observations = admitted(context, observed, tokenizer=tokenizer)
    records = grade_rows(context, sealed, observations)
    expected = report(context, sealed, records, observations)
    require(
        core.rows(directory / "graded.jsonl") == records
        and read(directory / "report.json") == expected,
        "Saved product grade differs from deterministic execution of actual admitted output.",
    )
    return context, expected, records


def selected_native_pair(native_records, product_records, identities, native_role):
    def outcomes(ids):
        values = native_comparison.outcome_pairs(native_records, product_records, ids)
        return {
            name: values[key]
            for name, key in (
                ("both_correct", "both_correct"),
                ("native_only", "left_only"),
                ("product_only", "right_only"),
                ("both_wrong", "neither_correct"),
            )
        }

    counts = {
        role: native_comparison.paired.tallies(
            [records[identity] for identity in identities]
        )
        for role, records in (("native", native_records), ("product", product_records))
    }
    native_values, product_values = (
        native_comparison.paired.values(counts[role]) for role in ("native", "product")
    )
    return {
        "native_role": native_role,
        "product_role": deployed.ROLE,
        "cases": len(identities),
        "case_ids": identities,
        "paired_outcomes": outcomes(identities),
        "by_source_group": {
            group: outcomes(
                [
                    identity
                    for identity in identities
                    if native_records[identity]["source_group"] == group
                ]
            )
            for group in sorted(
                {native_records[identity]["source_group"] for identity in identities}
            )
        },
        "counts": counts,
        "metric_deltas": {
            name: {
                "native": native_values[name],
                "product": product_values[name],
                "product_minus_native": product_values[name] - native_values[name]
                if native_values[name] is not None and product_values[name] is not None
                else None,
            }
            for name in native_comparison.paired.METRICS
        },
        "interpretation": "Descriptive quality differences on the complete identical cohort, retaining every failure. Undefined denominators stay undefined; no hypothesis test or causal training-effect claim.",
    }


def aggregate(
    freeze, sealed, deployed_grade, native_grades, output, *, resamples=2000, seed=3407
):
    context, product, records = load_grade(freeze, sealed, deployed_grade)
    comparison, arms = native_comparison.build_comparison(
        native_grades, resamples=resamples, seed=seed
    )
    require(
        comparison["freeze"] == context["gate"]["binding"]
        and comparison["sealed_manifest"] == context["sealed_binding"]
        and comparison.get("cohort_keys", {}).get("canonical")
        == "full_cohort_all_roles",
        "Append the product result only to the same frozen small-family full-cohort native comparison.",
    )
    ids = context["release"]["case_ids"]
    require(
        all(set(arm["records"]) == set(ids) for arm in arms.values())
        and [row["case_id"] for row in records] == ids,
        "Product/native result denominators differ.",
    )
    selected = context["source_run"]["role"]
    require(
        arms[selected]["run"] == context["source_run"]
        and list(arms[selected]["requests"].values()) == context["requests"],
        "Product inference did not use the selected native row's exact frozen requests.",
    )
    table = [
        {
            "role": role,
            "numerical_regime": "deterministic_baseline"
            if role in {"always_no_action", "consistent_history_rule"}
            else "native_bfloat16_with_optional_float32_lora",
            "schema_constrained": False,
            "summary": comparison["runs"][role]["full_issued"],
        }
        for role in arms
    ]
    table.append(
        {
            "role": deployed.ROLE,
            "selected_native_role": selected,
            "numerical_regime": context["policy"]["numerical_regime"],
            "schema_constrained": True,
            "summary": product["summary"],
            "http_latency": product["http_latency"],
        }
    )
    value = {
        "version": "small-final-native-and-deployed-comparison-1",
        "scope": "final_test",
        "status": "compared",
        "freeze": context["gate"]["binding"],
        "sealed_manifest": context["sealed_binding"],
        "deployed_grade": core.binding(Path(deployed_grade) / "manifest.json"),
        "native_core_comparison": comparison,
        "selected_native_vs_deployed": selected_native_pair(
            arms[selected]["records"],
            {row["case_id"]: row for row in records},
            ids,
            selected,
        ),
        "rows": table,
        "cases": len(ids),
        "native_core_roles_changed": False,
        "model_selection_changed": False,
        "causal_native_comparisons_include_deployed_gguf": False,
        "limitations": [
            "The auxiliary uses the selected merged/quantized GGUF and actual schema/cache policy. Its difference from native output is not solely a training effect.",
            "Product HTTP latency excludes editor preparation, debounce/rendering and model startup; it is not the native CUDA generation clock.",
            "Every registered request and failure remains in the same full-cohort denominator.",
        ],
        "source_sha256": deployed.sources(),
    }
    return core.publish(
        output,
        {"comparison.json": value},
        {"version": VERSION, "status": "compared", **deployed.common(context)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("grade", "aggregate"):
        command = commands.add_parser(name)
        for key in ("freeze", "sealed", "output"):
            command.add_argument("--" + key, required=True, type=Path)
        if name == "grade":
            command.add_argument("--observed", required=True, type=Path)
        else:
            command.add_argument("--deployed-grade", required=True, type=Path)
            command.add_argument("--native-grade", required=True, nargs="+", type=Path)
            command.add_argument("--resamples", type=int, default=2000)
            command.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    result = (
        grade(args.freeze, args.sealed, args.observed, args.output)
        if args.command == "grade"
        else aggregate(
            args.freeze,
            args.sealed,
            args.deployed_grade,
            args.native_grade,
            args.output,
            resamples=args.resamples,
            seed=args.seed,
        )
    )
    print(core.packed(result))


if __name__ == "__main__":
    main()
