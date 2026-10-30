"""Compare the six frozen final-test roles without selecting or relabeling models."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import compare_actions_v3_results as paired
import evaluate_passive_actions_v3 as evaluation
import prepare_actions_v3_conditions as conditions
import recompute_actions_v3_results as evidence

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    "native_base",
    "sft",
    "sft_dpo",
    "always_no_action",
    "consistent_history_rule",
)
ROLES = (*CANONICAL, "base_icl")
require = evidence.require


def role_policy(freeze, bound):
    omission = freeze.get("dpo_omission")
    if omission is None:
        return CANONICAL, ROLES, None
    import small_final_test_runtime as small
    import gemma_final_test_runtime as gemma
    require(freeze["selected_model"] == "sft" and set(freeze["models"]) == {"native_base", "sft"},
            "DPO omission must be registered before test for the frozen model family.")
    models = {role: evidence.read(bound(value)) for role, value in freeze["models"].items()}
    backend = gemma if models["sft"].get("runtime_family") == gemma.FAMILY else small
    source = str(Path(backend.__file__).resolve().relative_to(ROOT))
    require(all(model.get("runtime_family") == backend.FAMILY for model in models.values())
            and freeze["source_sha256"][source] == evidence.sha256(backend.__file__),
            "Historical model comparisons may not omit their DPO arm.")
    model = models["sft"]
    metadata = evidence.read(bound(model["preparation_manifest"]))
    require(metadata["dpo_omission"] == omission, "Reported omission differs from its model metadata.")
    parent = model["identity"] if backend is gemma else small.adapter_parent(model["identity"])
    value = backend.validate_omission(evidence.read(bound(omission)), evidence.read(bound(model["inference_configuration"])),
                                     parent, resolve=bound)
    details = {"binding": omission, "reason": value.get("reason", value.get("decision")),
               **{key: value[key] for key in ("reason_codes", "counts", "plan", "parent")}}
    return tuple(role for role in CANONICAL if role != "sft_dpo"), tuple(role for role in ROLES if role != "sft_dpo"), details


def full_cohort_policy(freeze, bound):
    base = evidence.read(bound(freeze["models"]["native_base"]))
    if base.get("runtime_family") not in {"small-actions-final-v1", "gemma26-final-native-v1"}:
        return None
    import small_final_test_runtime as small
    import gemma_final_test_runtime as gemma
    backend = gemma if base["runtime_family"] == gemma.FAMILY else small
    source = str(Path(backend.__file__).resolve().relative_to(ROOT))
    require(freeze["source_sha256"][source] == evidence.sha256(backend.__file__),
            "Full-cohort policy implementation changed.")
    config = evidence.read(bound(base["inference_configuration"]))
    policy = backend.validate_inference_policy(evidence.read(bound(base["inference_policy"])), config, resolve=bound)
    runtime = evidence.read(bound(freeze["policies"]["runtime"]))
    require(runtime["inference_policy"] == base["inference_policy"]
            and all(evidence.read(bound(value)).get("inference_policy") == base["inference_policy"]
                    for value in freeze["models"].values()), "Final roles do not share one inference context policy.")
    return {"binding": base["inference_policy"], **policy}


def load_grade(path, root, recorded_workspace=None):
    path = evidence.resolve_path(root, path, recorded_workspace)
    verified = evidence.final_evaluation_summary(root, path, recorded_workspace)
    manifest = evidence.read(path)

    def bound(binding):
        return evidence.bound(root, binding, recorded_workspace)

    freeze = evidence.read(bound(manifest["freeze"]))
    canonical_roles, comparison_roles, omission = role_policy(freeze, bound)
    context_policy = full_cohort_policy(freeze, bound)
    sealed_path = bound(manifest["sealed_manifest"])
    execution_path = bound(manifest["observation_manifest"])
    execution = evidence.read(execution_path)
    run = verified["run"]
    role = run["role"]
    expected_condition = "train_icl" if role == "base_icl" else "canonical_full_context"
    require(
        role in ROLES
        and run["condition"]
        == conditions.condition_specs(run["condition"]["demonstrations"])[
            expected_condition
        ],
        "Unregistered final-test condition.",
    )
    for module in (conditions, evaluation, evaluation.formats, evaluation.contract):
        source = Path(module.__file__).resolve()
        require(
            freeze["source_sha256"][str(source.relative_to(ROOT))]
            == evidence.sha256(source),
            "Shared final-test input contract implementation changed.",
        )
    model_role = "native_base" if role == "base_icl" else role
    expected_identity = (
        evidence.read(bound(freeze["models"][model_role]))["identity"]
        if model_role in {"native_base", "sft", "sft_dpo"}
        else {"baseline": role, "code_sha256": evidence.sha256(conditions.__file__)}
    )
    require(
        execution["model_identity"] == expected_identity,
        "Frozen model identity differs.",
    )
    if role == "base_icl":
        reservations = evidence.read(bound(freeze["source_reservations"]))
        for demo in run["condition"]["demonstrations"]:
            require(
                demo["split"] == "train"
                and reservations["exposed_source_group_splits"].get(
                    demo["source_group"]
                )
                == "train"
                and demo["release_binding"]["manifest"]["sha256"]
                == freeze["training_dataset_manifest"]["sha256"],
                "ICL demonstration is not bound to the frozen TRAIN allocation.",
            )
    cases = {
        row["id"]: row for row in evidence.rows(sealed_path.parent / "cases.jsonl")
    }
    requests = {
        row["case_id"]: row for row in evidence.rows(bound(execution["requests"]))
    }
    observations = {
        row["case_id"]: row
        for row in evidence.rows(execution_path.parent / "observations.jsonl")
    }
    records = {
        row["case_id"]: row for row in evidence.rows(path.parent / "graded.jsonl")
    }
    report = evidence.read(path.parent / "report.json")
    require(
        evaluation.summarize(list(records.values())) == report["summary"],
        "Saved final-test action counts or totals differ.",
    )
    for identity, request in requests.items():
        case, record, observation = (
            cases[identity],
            records[identity],
            observations[identity],
        )
        require(
            request["input"] == case["input"]
            and request["input_sha256"] == evaluation.semantic_input_hash(case["input"])
            and request["condition"] == run["condition"]
            and request["condition_sha256"] == evaluation.digest(run["condition"])
            and request["messages"]
            == conditions.expected_messages(
                case["input"], run["format"], run["condition"]
            )
            and request["messages_sha256"] == evaluation.digest(request["messages"])
            and request["prefix_ids_sha256"]
            == evaluation.digest(request["prefix_ids"]),
            "Final-test input, condition, messages or native prefix binding differs.",
        )
        require(
            observation["request_sha256"] == evaluation.digest(request)
            and all(
                observation[key] == request[key]
                for key in ("split", "run_id", "freeze_sha256", "condition_sha256")
            ),
            "Observation is not bound to the complete final-test request.",
        )
        require(
            record["stratum"] == case["lineage"]["stratum"]
            and record["scenario_family"]
            == case["lineage"].get("scenario_family", "legacy_unspecified")
            and record["action_language"]
            == f"{case['primary']['action']}:{case['language']}",
            "Final-test scored provenance differs.",
        )
        conditioned = {
            **case,
            "condition": run["condition"],
            "condition_sha256": request["condition_sha256"],
        }
        require(
            evaluation.evaluate_one(conditioned, observation, run["format"]) == record,
            "Saved final-test grade differs from deterministic execution of its observation.",
        )
        if observation["status"] != "completed":
            require(
                record["status"] == "request_failure"
                and record["predicted_action"] is None
                and not any(
                    record[key]
                    for key in (
                        "valid",
                        "displayed",
                        "useful",
                        "preserved",
                        "model_abstained",
                        "outcome_match",
                    )
                ),
                "A failed request acquired a scored success or abstention.",
            )
        if record["status"] == "completed":
            require(
                record["outcome_match"]
                == (record["result_sha256"] in case["accepted_result_sha256"]),
                "Graded outcome is not bound to the accepted final-test results.",
            )
    return {
        "manifest_path": path,
        "manifest": manifest,
        "verified": verified,
        "frozen_runs": freeze["runs"],
        "canonical_roles": canonical_roles,
        "comparison_roles": comparison_roles,
        "dpo_omission": omission,
        "full_cohort_policy": context_policy,
        "run": run,
        "records": records,
        "requests": requests,
        "cases": cases,
        "observations": observations,
        "report": report,
        "dispositions": list(evidence.rows(sealed_path.parent / "dispositions.jsonl")),
    }


def summarize(records):
    result = evaluation.summarize(records)
    result["successful_predicted_action_language"] = {
        f"{action}:{language}": sum(
            row["language"] == language
            and row["predicted_action"] == action
            and (row["useful"] or row["preserved"] and row["model_abstained"])
            for row in records
        )
        for language in ("en", "ko")
        for action in evaluation.contract.ACTIONS
    }
    return result


def failure_attribution(arm):
    observations, records = arm["observations"], arm["records"]
    failed = [
        row for row in observations.values() if row["status"] == "request_failure"
    ]
    return {
        "request_failures": len(failed),
        "request_failure_stages": dict(
            sorted(
                Counter(
                    row.get("failure_stage", "unspecified") for row in failed
                ).items()
            )
        ),
        "generation_attempted_request_failures": sum(
            row.get("generation_attempted") is True for row in failed
        ),
        "pre_generation_request_failures": sum(
            row.get("generation_attempted") is False for row in failed
        ),
        "unspecified_generation_attempt_request_failures": sum(
            type(row.get("generation_attempted")) is not bool for row in failed
        ),
        "completed_response_grade_status_counts": dict(
            sorted(
                Counter(
                    records[identity]["status"]
                    for identity, row in observations.items()
                    if row["status"] == "completed"
                ).items()
            )
        ),
        "denominator_policy": "All registered requests remain scored. Attribution does not replace, retry or exclude a failed request.",
    }


def outcome_pairs(left, right, identities):
    groups = {
        name: []
        for name in ("both_correct", "left_only", "right_only", "neither_correct")
    }
    for identity in identities:
        a, b = left[identity]["outcome_match"], right[identity]["outcome_match"]
        groups[
            "both_correct"
            if a and b
            else "left_only"
            if a
            else "right_only"
            if b
            else "neither_correct"
        ].append(identity)
    return {
        key: {"count": len(value), "case_ids": value} for key, value in groups.items()
    }


def cohort(arms, roles, identities, *, resamples, seed):
    first = arms[roles[0]]
    census = evidence.final_census(
        first["dispositions"], identities, sorted(set(first["cases"]) - set(identities))
    )
    for groups in census.values():
        for value in groups.values():
            value["compared"] = value.pop("requested")
    by = {}
    for dimension in ("source_group", "language", "stratum"):
        grouped = defaultdict(list)
        for identity in identities:
            grouped[first["records"][identity][dimension]].append(identity)
        by[dimension] = {
            group: {
                role: summarize([arms[role]["records"][i] for i in ids])
                for role in roles
            }
            for group, ids in sorted(grouped.items())
        }
    comparisons = {}
    for a, b in combinations(roles, 2):
        left, right = arms[a]["records"], arms[b]["records"]
        sampled = paired.source_resampling(
            left, right, identities, resamples=resamples, seed=seed
        )
        sampled["limitation"] = (
            "Descriptive source-group resampling of a designed holdout. Few natural sources, "
            "dependent C0/C1/C2 states and authored siblings do not establish population reliability. "
            "Fewer than two groups gives no interval; zero denominators stay undefined."
        )
        comparisons[f"{a}__{b}"] = {
            "left_role": a,
            "right_role": b,
            "left_only_case_ids": sorted(set(left) - set(right)),
            "right_only_case_ids": sorted(set(right) - set(left)),
            "paired_outcomes": outcome_pairs(left, right, identities),
            "by_source_group": {
                group: outcome_pairs(
                    left,
                    right,
                    [i for i in identities if left[i]["source_group"] == group],
                )
                for group in sorted({left[i]["source_group"] for i in identities})
            },
            "source_resampling": sampled,
        }
    return {
        "roles": list(roles),
        "cases": len(identities),
        "case_ids": identities,
        "registered_admitted_compared_denominators": census,
        "arms": {
            role: summarize([arms[role]["records"][i] for i in identities])
            for role in roles
        },
        "by": by,
        "pairs": comparisons,
    }


def build_comparison(
    paths, *, root=ROOT, recorded_workspace=None, resamples=2000, seed=3407
):
    require(type(resamples) is int and resamples > 0, "Resamples must be positive.")
    loaded = [load_grade(path, root, recorded_workspace) for path in paths]
    arms = {arm["run"]["role"]: arm for arm in loaded}
    require("native_base" in arms, "The frozen native-base control is required.")
    first = arms["native_base"]
    canonical_roles, comparison_roles = first["canonical_roles"], first["comparison_roles"]
    full_cohort = first["full_cohort_policy"] is not None
    require(
        len(loaded) == len(arms) == len(comparison_roles) and set(arms) == set(comparison_roles),
        "Require exactly the frozen roles; DPO needs an explicit prospective omission receipt.",
    )
    require(
        len(first["frozen_runs"]) == len(loaded)
        and {run["id"] for run in first["frozen_runs"]}
        == {arm["run"]["id"] for arm in loaded},
        "Every frozen final-test run must be supplied exactly once.",
    )
    canonical_ids = sorted(first["cases"])
    for role, arm in arms.items():
        require(
            all(
                arm["manifest"][key] == first["manifest"][key]
                for key in ("freeze", "sealed_manifest")
            ),
            "Comparison mixes final-test freezes or sealed releases.",
        )
        require(
            arm["run"]["format"] == first["run"]["format"],
            "Final comparison formats differ.",
        )
        ids = sorted(arm["records"])
        require(
            set(ids) <= set(canonical_ids) if role == "base_icl" and not full_cohort else ids == canonical_ids,
            "Registered roles must retain the full admitted cohort; only legacy ICL may use its registered subset.",
        )
        exclusions = arm["report"]["run_eligibility_exclusions"]
        require(
            not exclusions
            if role != "base_icl" or full_cohort
            else all(
                row["reason"] == "icl_prefix_budget"
                and row["input_tokens"] > evaluation.formats.INPUT_LIMIT
                and row["input_sha256"]
                == evaluation.semantic_input_hash(arm["cases"][row["id"]]["input"])
                for row in exclusions
            ),
            "Unregistered run eligibility exclusion.",
        )
        if ids:
            left_inputs = {
                i: {
                    **first["requests"][i],
                    "accepted_result_sha256": first["cases"][i][
                        "accepted_result_sha256"
                    ],
                }
                for i in ids
            }
            right_inputs = {
                i: {
                    **arm["requests"][i],
                    "accepted_result_sha256": arm["cases"][i]["accepted_result_sha256"],
                }
                for i in ids
            }
            paired.matched(
                first["records"], arm["records"], left_inputs, right_inputs, "models"
            )
        if role != "base_icl":
            require(
                all(
                    all(
                        first["requests"][i][key] == arm["requests"][i][key]
                        for key in (
                            "condition_sha256",
                            "messages_sha256",
                            "prefix_ids_sha256",
                        )
                    )
                    for i in ids
                ),
                "Canonical run messages or native prefixes differ.",
            )
    common_ids = sorted(
        set.intersection(*(set(arm["records"]) for arm in arms.values()))
    )
    result = {
        "version": "task-test-actions-v3-comparison-1",
        "scope": "final_test",
        "status": "compared",
        "model_selection_changed": False,
        "thresholds_selected": False,
        "engineering_fixture_only": first["verified"]["engineering_fixture_only"],
        "freeze": first["manifest"]["freeze"],
        "sealed_manifest": first["manifest"]["sealed_manifest"],
        "implementation_sha256": {
            str(Path(module.__file__).resolve().relative_to(ROOT)): evidence.sha256(
                module.__file__
            )
            for module in (
                paired,
                evidence,
                evaluation,
                conditions,
                evaluation.formats,
                evaluation.contract,
            )
        },
        "runs": {
            role: {
                "grade_manifest": evaluation.file_binding(arm["manifest_path"]),
                "run": arm["run"],
                "verified_census": arm["verified"],
                "full_issued": summarize(list(arm["records"].values())),
                "run_eligibility_exclusions": arm["report"][
                    "run_eligibility_exclusions"
                ],
                "outside_common_case_ids": sorted(
                    set(arm["records"]) - set(common_ids)
                ),
                "failure_attribution": failure_attribution(arm),
                "failed_outcomes": [
                    {
                        **row,
                        "observation_status": arm["observations"][i]["status"],
                        "failure_stage": arm["observations"][i].get("failure_stage"),
                        "generation_attempted": arm["observations"][i].get(
                            "generation_attempted"
                        ),
                        "model_identity_verified": arm["observations"][i].get(
                            "model_identity_verified"
                        ),
                    }
                    for i, row in arm["records"].items()
                    if row["status"] != "completed"
                ],
            }
            for role, arm in arms.items()
        },
        "limitations": [
            ("Every available frozen role shares the full cohort. The registered 8192-token inference cap accommodates only the fixed ICL examples; canonical visible inputs, 4096-token SFT training and app budgets stay unchanged."
             if full_cohort else "Canonical five-role totals are not reduced to ICL eligibility. Common six-role results use the same cases and fixed outcomes, not success-selected requests."),
            "The only allowed message/prefix difference is the frozen base_icl train demonstrations. Visible case input, runtime provenance and accepted outcomes remain shared.",
            "Failed requests stay in all applicable denominators. Invalid or suppressed outputs are not silently converted to model no_action. Zero denominators are undefined, not perfect scores.",
            "Successful action-language counts use the actual predicted action, not the primary reference. Small cells establish observed functionality only.",
            "Natural windows, source groups and dependent editing states are distinct units. Synthetic challenges and few correlated sources cannot establish natural acoustic or population performance.",
            "Saved grades are independently rederived by deterministic action execution against the bound cases and observations. This does not rerun semantic judgment, caption export, native tokenization or models, choose a checkpoint, or change promotion thresholds.",
            "Startup, interruption and generation failures remain in the registered denominators and are attributed separately. A failure before generation is not observed model abstention or evidence about its intrinsic editing capability.",
            "The unchanged evaluator source-integrity failure category conservatively includes rejected action application as well as failed postconditions; that count alone does not establish actual source corruption.",
        ],
    }
    if full_cohort:
        import small_final_test_runtime as small
        import gemma_final_test_runtime as gemma
        result.update(full_cohort_all_roles=cohort(arms, comparison_roles, canonical_ids, resamples=resamples, seed=seed),
                      inference_policy=first["full_cohort_policy"],
                      cohort_keys={"canonical": "full_cohort_all_roles", "icl_common": "full_cohort_all_roles"})
        backend = gemma if first["full_cohort_policy"]["version"] == gemma.INFERENCE_POLICY_VERSION else small
        result["implementation_sha256"][str(Path(backend.__file__).resolve().relative_to(ROOT))] = evidence.sha256(backend.__file__)
    else:
        result["canonical_full_five_roles"] = cohort(arms, canonical_roles, canonical_ids, resamples=resamples, seed=seed)
        result["icl_common_six_roles"] = cohort(arms, comparison_roles, common_ids, resamples=resamples, seed=seed)
    if first["dpo_omission"] is not None:
        require(full_cohort, "DPO omission requires the prospective full-cohort policy.")
        result["dpo_omission"] = first["dpo_omission"]
    result["implementation_sha256"][
        str(evaluation.contract.PROMPT_PATH.relative_to(ROOT))
    ] = evidence.sha256(evaluation.contract.PROMPT_PATH)
    result["implementation_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = (
        evidence.sha256(__file__)
    )
    return result, arms


def compare(
    paths, output, *, root=ROOT, recorded_workspace=None, resamples=2000, seed=3407
):
    output = Path(output)
    require(not output.exists(), "Use a fresh final-test comparison directory.")
    result, _ = build_comparison(
        paths,
        root=root,
        recorded_workspace=recorded_workspace,
        resamples=resamples,
        seed=seed,
    )
    output.mkdir(parents=True, mode=0o700)
    evaluation.write(output / "comparison.json", result)
    evaluation.write(
        output / "manifest.json",
        {
            "version": result["version"],
            "scope": "final_test",
            "status": "compared",
            "files_sha256": {
                "comparison.json": evidence.sha256(output / "comparison.json")
            },
            "implementation_sha256": result["implementation_sha256"],
        },
    )
    keys = result.get("cohort_keys", {"canonical": "canonical_full_five_roles", "icl_common": "icl_common_six_roles"})
    return {
        "canonical_cases": result[keys["canonical"]]["cases"],
        "common_cases": result[keys["icl_common"]]["cases"],
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grades", nargs="+", required=True, help="Every frozen role's final grade manifest.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--recorded-workspace")
    args = parser.parse_args()
    print(
        json.dumps(
            compare(
                args.grades,
                args.output,
                root=args.root,
                recorded_workspace=args.recorded_workspace,
            )
        )
    )


if __name__ == "__main__":
    main()
