"""Final Gemma evidence checks for the existing metadata freeze builder."""

from fractions import Fraction

import evaluate_gemma_final as development
import gemma_final_test_runtime as gemma
import evaluate_gemma_q8_final as q8_development
import gemma_q8_product as q8

require, read, bound, binding, digest = gemma.require, gemma.read, gemma.bound, gemma.binding, gemma.digest
DESCRIPTIVE_VERSION = "gemma26-descriptive-development-selection-1"
DESCRIPTIVE_BASIS = "user_selected_descriptive_evaluation_registered_subgroup_failure_retained"
DESCRIPTIVE_PLAN = gemma.ROOT / "docs/plans/final-gemma26-descriptive-selection-2026-10-25.md"
PRODUCT_VERSION = "gemma26-descriptive-product-selection-1"
PRODUCT_PLAN = gemma.ROOT / "docs/plans/final-gemma26-product-descriptive-evaluation-2026-10-25.md"
PRODUCT_DECISION = gemma.ROOT / "artifacts/final-completion/20261025-selected26-001/root-final-product-decision-001.json"
PRODUCT_DECISION_SHA = "5a35ff829c3805d7ab7cb8443f34c527687f841a1bf8b765a9bea21823ecff3a"


def product_drop_checks(native, product):
    checks = []
    for cohort in ("overall", "canonical_dev322", "supplemental_dev149"):
        before = native["overall"] if cohort == "overall" else native["cohorts"][cohort]["metrics"]
        after = product["overall"] if cohort == "overall" else product["cohorts"][cohort]["metrics"]
        for metric in ("precision", "recall"):
            rates = [before[metric], after[metric]]
            require(all(type(r["numerator"]) is int and type(r["denominator"]) is int
                        and 0 <= r["numerator"] <= r["denominator"] and r["denominator"] > 0
                        and r["value"] == r["numerator"] / r["denominator"] for r in rates),
                    "Gemma native drop allowance requires actual consistent integer metric denominators.")
            passed = (Fraction(rates[0]["numerator"], rates[0]["denominator"])
                      - Fraction(rates[1]["numerator"], rates[1]["denominator"])) <= Fraction(1, 50)
            checks.append({"cohort": cohort, "metric": metric, "native": rates[0], "product": rates[1],
                           "delta_pp": 100 * (rates[1]["value"] - rates[0]["value"]),
                           "allowed_drop_pp": 2, "passed": passed})
        counts = [before["model_preservation"]["numerator"], after["model_preservation"]["numerator"]]
        require(all(type(v) is int and v >= 0 for v in counts), "Gemma preservation requires actual integer counts.")
        checks.append({"cohort": cohort, "metric": "model_preservation_count", "native": counts[0],
                       "product": counts[1], "allowed_drop_count": 2, "passed": counts[0] - counts[1] <= 2})
    return checks


def failed_product_checks(checks):
    return [row["cohort"] + ":" + row["metric"] for row in checks if not row["passed"]]


def product_contract(model, role="sft"):
    decision_reference = binding(PRODUCT_DECISION)
    require(decision_reference["sha256"] == PRODUCT_DECISION_SHA, "The exact root product decision changed.")
    decision = read(bound(decision_reference))
    require(decision["version"] == "gemma26-root-final-product-selection-1" and decision["actor"] == "/root"
            and decision["status"] == "descriptive_product_selected_registered_drop_gate_failed"
            and decision["selected_model"] == role == "sft"
            and decision["selected_numerical_regime"] == q8.Q4_REGIME
            and decision["adapter_sha256"] == model["identity"]["adapter_sha256"]
            and decision["plan"] == binding(PRODUCT_PLAN)
            and all(decision[k] is False for k in ("registered_product_drop_gate_passed", "quality_success_claimed",
                                                  "final_test_bodies_read", "body_access_authorized"))
            and decision["no_additional_training_model_or_quantization_experiments"] is True,
            "Descriptive product decision changed its selected model, failed goal or scope.")
    for key in ("native_selection", "selected_native_summary", "selected_deployed_summary", "deployment",
                "registered_drop_audit", "Q8_audit", "Q8_summary", "user_selection_evidence", "plan"):
        bound(decision[key])
    native, product = (read(bound(decision[key])) for key in ("selected_native_summary", "selected_deployed_summary"))
    require(native["model_identity"] == model["identity"]
            and product["model_identity"]["native_identity_sha256"] == digest(model["identity"])
            and product["model_identity"]["training_adapter_sha256"] == model["identity"]["adapter_sha256"],
            "Descriptive product summaries belong to another native adapter.")
    checks = product_drop_checks(native, product)
    failures = failed_product_checks(checks)
    audit = read(bound(decision["registered_drop_audit"]))
    require(failures and audit["drop_checks"] == checks and audit["drop_gate_passed"] is False,
            "Actual product drop audit differs from all recomputed checks.")
    require(isinstance(decision["limitations"], list) and decision["limitations"]
            and all(isinstance(v, str) and v.strip() for v in decision["limitations"]),
            "Descriptive product selection requires the actual disclosed limitations.")
    return {"version": PRODUCT_VERSION, "status": "descriptive_evaluation_registered_product_goal_failed",
        "actor": "/root", "selected_model": role, "selected_native_identity": model["identity"],
        "selected_identity_sha256": digest(model["identity"]), "root_decision": decision_reference,
        **{k: decision[k] for k in ("deployment", "plan", "user_selection_evidence", "selected_native_summary",
                                    "selected_deployed_summary", "registered_drop_audit", "limitations")},
        "drop_checks": checks, "failed_product_drop_checks": failures,
        "registered_product_drop_gate_passed": False, "quality_success_claimed": False,
        "final_test_bodies_read": False, "body_access_authorized": False}


def product_selection(selection, models, role, quality, native, product, product_run, deployment):
    checks = product_drop_checks(native, product)
    contract = selection.get("product_selection_contract")
    if contract is None:
        require(not failed_product_checks(checks), "Gemma deployed quality exceeds the registered native drop allowance.")
        return
    expected = product_contract(models[role], role)
    require(contract == expected and contract["drop_checks"] == checks
            and contract["selected_native_summary"] == quality["selected_native_summary"]
            and contract["selected_deployed_summary"] == quality["selected_deployed_summary"]
            and contract["deployment"] == product_run["deployment"]
            and read(bound(contract["deployment"])) == deployment and deployment["numerical_regime"] == q8.Q4_REGIME,
            "Descriptive product contract omitted or changed a failed check, identity, deployment or summary.")
    for key in ("root_decision", "deployment", "plan", "user_selection_evidence", "selected_native_summary",
                "selected_deployed_summary", "registered_drop_audit"):
        require(contract[key] in selection["evidence"], "Selection omitted its descriptive product evidence.")
        bound(contract[key])


def verify_native_backend(models, role, quality, parent_report, native_report):
    policy = read(bound(models[role]["inference_policy"]))
    overlay = gemma.native_policy_overlay(policy, parent=models["sft"]["identity"])
    for report in (parent_report, native_report):
        path = bound(report["inference_run"])
        run = read(path)
        if overlay is None:
            require(run.get("runtime_overlay") is None, "Canonical native selection may not use grouped DEV evidence.")
        else:
            require(run.get("runtime_overlay") == overlay["settings"], "Selected SFT/DPO DEV must share the adopted native backend.")
            info, _ = gemma.verify_native_development_run(path.parent.parent)
            require(info["model_identity"] == report["model_identity"], "Selected DEV differs from its completed grouped run.")
    if overlay is not None:
        decision = read(bound(overlay["adoption"]))
        require(decision["inputs"]["optimized_SFT_summary"] == quality["native_sft_summary"],
                "Final native selection replaced the SFT evidence used to adopt its backend.")


def subgroup_selection(selection, models, role, quality):
    review = quality["subgroup_review"]
    require(isinstance(review, dict) and isinstance(review.get("rationale"), str)
            and review["rationale"].strip() and review.get("evidence"),
            "Actual preservation, unknown-speaker and latest-correction review is missing.")
    contract = selection.get("selection_contract")
    if contract is None:
        require(review.get("status") == "reviewed_pass", "Registered subgroup selection gate failed.")
        return "registered_fallback_after_three_completed_native_failures"
    require(isinstance(contract, dict) and set(contract) == {
        "version", "status", "actor", "selected_model", "selected_identity_sha256", "subgroup_review",
        "subgroup_audit", "user_selection_evidence", "plan", "strict_subgroup_gate_passed",
        "failed_registered_typed_groups", "limitations", "quality_success_claimed",
        "final_test_bodies_read", "body_access_authorized"}, "Invalid descriptive selection contract.")
    require(contract["version"] == DESCRIPTIVE_VERSION
            and contract["status"] == "user_selected_descriptive_evaluation" and contract["actor"] == "/root"
            and contract["selected_model"] == role
            and contract["selected_identity_sha256"] == digest(models[role]["identity"])
            and all(contract[key] is False for key in ("strict_subgroup_gate_passed", "quality_success_claimed",
                                                       "final_test_bodies_read", "body_access_authorized"))
            and contract["plan"] == binding(DESCRIPTIVE_PLAN),
            "Descriptive selection changed its identity, failed-gate status, scope or plan.")
    require(review.get("status") == "reviewed_fail"
            and review.get("registered_no_entire_new_subgroup_failure_passed") is False
            and review.get("reserved_TEST_bodies_read") is False
            and read(bound(contract["subgroup_review"])) == review,
            "Descriptive selection must retain the exact failed subgroup review.")
    failed = review.get("failed_registered_typed_groups")
    groups = review.get("registered_typed_group_results", [])
    require(isinstance(failed, list) and failed and len(failed) == len(set(failed))
            and all(isinstance(name, str) and name.strip() for name in failed)
            and failed == contract["failed_registered_typed_groups"]
            and sorted(failed) == sorted(row["registered_group"] for row in groups
                                        if row["all_failed"] is True and row["n"] > 0 and row["outcome_matches"] == 0),
            "Descriptive selection omitted or changed the registered failed subgroups.")
    audit = read(bound(contract["subgroup_audit"]))
    require(contract["subgroup_audit"] in review["evidence"]
            and audit.get("model_identity") == models[role]["identity"] and audit.get("TEST_bodies_read") is False
            and quality["selected_native_summary"] in audit.get("bindings", []),
            "Failed subgroup evidence belongs to another native model or DEV summary.")
    require(isinstance(contract["limitations"], list) and contract["limitations"]
            and all(isinstance(text, str) and text.strip() for text in contract["limitations"]),
            "Descriptive selection requires explicit report limitations.")
    for key in ("subgroup_review", "subgroup_audit", "user_selection_evidence", "plan"):
        require(contract[key] in selection["evidence"], "Selection omitted its descriptive decision evidence.")
        bound(contract[key])
    return DESCRIPTIVE_BASIS


def quality_without_ui(selection, models, role, config, deployment, model_identity):
    from build_small_final_freeze_actions_v3 import within_drop

    config_path = bound(models[role]["inference_configuration"])
    checked, _ = gemma.verify_fallback(config_path)
    require(config == checked, "Gemma selection uses another unified training preparation.")
    protocol = read(config_path.parent / "protocol.json")
    failures = [*protocol["completed_small_failure_summaries"], protocol["completed_lr_recovery_failure_summary"]]
    quality = selection.get("quality_without_ui")
    require(isinstance(quality, dict) and set(quality) == {
        "native_sft_summary", "selected_native_summary", "selected_deployed_summary", "subgroup_review"},
        "Gemma selection requires actual SFT, selected-native and deployed DEV summaries.")
    parent, parent_report = development.verify_summary(quality["native_sft_summary"])
    native, native_report = development.verify_summary(quality["selected_native_summary"])
    evaluator = q8_development if deployment["numerical_regime"] == q8.REGIME else development
    product, product_report = evaluator.verify_summary(quality["selected_deployed_summary"])
    require(parent["model_identity"] == models["sft"]["identity"]
            and native["model_identity"] == models[role]["identity"]
            and all(r["numerical_regime"] == "native_bfloat16_with_optional_float32_lora"
                    and r["schema_constrained"] is False for r in (parent_report, native_report)),
            "Gemma native quality differs from the actual completed SFT or same-parent DPO.")
    verify_native_backend(models, role, quality, parent_report, native_report)
    require(all(s["minimum_numeric_quality_gate"]["passed"] for s in (parent, native, product)),
            "Gemma SFT, selected native and actual deployed model must pass unchanged full DEV gates.")
    require(role != "sft" or quality["selected_native_summary"] == quality["native_sft_summary"],
            "Selected SFT must use the same parent quality evidence.")
    product_run = read(bound(product_report["inference_run"]))
    actual = product["model_identity"]
    require(product_report["schema_constrained"] is True
            and product_report["numerical_regime"] == deployment["numerical_regime"]
            and product_report["numerical_regime"] in {q8.Q4_REGIME, q8.REGIME}
            and read(bound(product_run["deployment"])) == deployment
            and actual["runtime_family"] == "gemma26-final-separate-lora-gguf"
            and all(actual.get(key) == value for key, value in model_identity.items())
            and actual["training_adapter_sha256"] == models[role]["identity"]["adapter_sha256"]
            and actual["native_identity_sha256"] == digest(models[role]["identity"])
            and actual["kind"] == models[role]["identity"]["kind"],
            "Gemma product DEV differs from the selected base, separate adapter, scale or native parent.")
    if deployment["numerical_regime"] == q8.REGIME:
        registration = q8.verify_registration(bound(deployment["q8_registration"]), models[role]["identity"])
        require(registration["native_summary"] == quality["selected_native_summary"],
                "Q8 final selection changed its prospectively registered native comparator.")
    require(parent["cohort_mapping"] == native["cohort_mapping"] == product["cohort_mapping"],
            "Gemma DEV comparisons must retain the same canonical322 and supplemental149 cohorts.")
    product_selection(selection, models, role, quality, native, product, product_run, deployment)
    for cohort in (None, "canonical_dev322", "supplemental_dev149"):
        before = native["overall"] if cohort is None else native["cohorts"][cohort]["metrics"]
        if role == "sft_dpo" and cohort is not None:
            previous = parent["cohorts"][cohort]["metrics"]
            require(all(within_drop(previous[key]["value"], before[key]["value"], 0)
                        for key in ("precision", "model_preservation"))
                    and within_drop(previous["recall"]["value"], before["recall"]["value"], .01)
                    and sum(v for k, v in before["status_counts"].items() if k != "completed")
                    <= sum(v for k, v in previous["status_counts"].items() if k != "completed"),
                    "Selected Gemma DPO violates the existing same-parent adoption rule.")
    require(role != "sft_dpo" or native["macro_action_f1"]["value"] > parent["macro_action_f1"]["value"],
            "Selected Gemma DPO must improve its same-parent macro action F1.")
    reason = subgroup_selection(selection, models, role, quality)
    review = quality["subgroup_review"]
    references = [*failures, quality["native_sft_summary"], quality["selected_native_summary"],
                  quality["selected_deployed_summary"], *review["evidence"]]
    require(all(reference in selection["evidence"] for reference in references),
            "Selection must bind both original small failures, the LR recovery failure and actual Gemma quality evidence.")
    for reference in review["evidence"]:
        bound(reference)
    return reason


def validate_native_readiness(ready, models, config, meta):
    training = bound(models["sft"]["artifact_manifest"]).parent
    require(ready["native_environment_receipt"] == binding(training / "run.json")
            and ready["native_runtime_receipt"] == models["native_base"]["artifact_manifest"],
            "Gemma readiness requires the verified training run and actual native loading receipt.")
    run = read(bound(ready["native_environment_receipt"]))
    environment = run["environment"]
    require(environment == meta["environment"] and environment["matched"] is True
            and not environment["differences"] and all(environment["actual"].get(k) == v for k, v in config["libraries"].items())
            and all(environment["actual"].get(k) is None for k in config["absent_libraries"]),
            "Actual Gemma native environment differs from its exact preparation.")
    runtime = read(bound(ready["native_runtime_receipt"]))
    require(runtime["gpu"] == config["model"]["gpu_name_contains"]
            and runtime["gpu_total_bytes"] >= config["model"]["minimum_gpu_bytes"]
            and runtime["parameter_dtypes"] == ["torch.bfloat16"]
            and runtime["numerical_regime"] == "native_bfloat16_lora_float32"
            and runtime["text_attn_implementation"] == "sdpa" and runtime["text_experts_implementation"] == "eager"
            and set(runtime["targets"]) == set(config["model"]["target_modules"])
            and runtime["effective_generation"] == config["generation"]
            and runtime["completion_label_eos_token_id"] == 1 and runtime["ignored_native_tensor_names"] == []
            and not any(runtime["loading_info"].get(key) for key in
                        ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "Actual Gemma GPU, BF16 regime, attention/experts, targets or native loading differs.")
