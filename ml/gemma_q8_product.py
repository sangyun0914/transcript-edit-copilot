"""Final-only registration and deployment for one same-adapter direct Q8 comparison."""

import argparse
import copy
import json
from pathlib import Path

import gemma_q8_export as base
import gemma_final_test_runtime as gemma

ROOT, PLAN, REGIME = base.ROOT, base.PLAN, base.REGIME
VERSION = "gemma26-final-q8-product-comparison-1"
Q4_REGIME = "native_derived_q4_base_fp32_lora"
DEV = ROOT / "artifacts/model-preparation/gemma26-final-v1/evaluation-combined-001"
HISTORY = ROOT / "artifacts/model-preparation/gemma26-final-v1/evaluation-history-001"
Q4_FAILURE = ROOT / "artifacts/final-completion/20261025-selected26-001/sft002-product-dev471-summary-001.json"
Q4_FAILURE_SHA = "202faae5b00460abd25e52df12f7566f408468c2f51902fc4db2b10df4038635"
require, read, sha, write, binding, bound = (gemma.require, gemma.read, gemma.sha, gemma.write,
                                           gemma.binding, gemma.bound)


def implementation():
    return {str(ROOT / "ml" / name): sha(ROOT / "ml" / name) for name in (
        "gemma_q8_export.py", "gemma_q8_product.py", "evaluate_gemma_q8_final.py")}


def validate_adapter(export, selected):
    require(export["version"] == "gemma-actions-v3-lora-export"
            and export["status"] == "adapter_conversion_verified_serving_pending"
            and export["all_tensors_exact"] is True and export["adapter_tensor_count"] == 410
            and export["output_format"] == "json" and export["runtime_revision"] == base.REVISION
            and export["base_repository"] == selected["repository"] == base.REPOSITORY
            and export["base_revision"] == selected["revision"] == base.NATIVE_REVISION
            and all(export[k] == selected[k] for k in
                    ("adapter_sha256", "training_manifest_sha256", "reload_manifest_sha256", "native_pin_sha256")),
            "Q8 requires the selected adapter's exact410 FP32 tensors and independent native reload.")


def matched_q4_identity(summary, selected):
    identity = summary["model_identity"]
    require(identity["training_adapter_sha256"] == selected["adapter_sha256"]
            and identity["native_identity_sha256"] == gemma.digest(selected)
            and identity["kind"] == selected["kind"],
            "Historical SFT Q4 cannot be presented as a same-adapter comparison to DPO Q8.")


def validate_selection(decision, model, native_summary):
    require(decision["version"] == "gemma26-root-final-native-selection-1" and decision["actor"] == "/root"
            and decision["status"] == "native_model_selected_product_and_TEST_pending"
            and decision["selected_model"] == model["role"]
            and decision["adapter_sha256"] == model["identity"]["adapter_sha256"]
            and decision["selected_summary"] == native_summary and decision["native_backend"] == "grouped_mm"
            and decision["Mac_Q4_drop_failure_retained"] is True and decision["one_Q8_comparison_pending"] is True
            and all(decision[k] is False for k in ("final_test_bodies_read", "body_access_authorized", "new_models_or_SFT_authorized")),
            "Q8 must bind the actual root-selected adapter and unchanged grouped native comparator.")
    for key in ("backend_decision", "DPO_review", "selected_subgroup_review", "selected_subgroup_audit", "user_authorization"):
        bound(decision[key])


def cohort_binding(directory, count):
    meta = read(directory / "manifest.json")
    require(meta["gemma_final_version"] == "gemma26-final-development-1"
            and meta["requests"] == meta["original_development_cases"] == count
            and meta["excluded"] == 0 and meta["final_test_bodies_read"] is False,
            "Q8 requires the complete saved DEV471 and separate H24.")
    requests = directory / "requests-json.jsonl"
    require(sha(requests) == meta["files_sha256"][requests.name], "Registered Q8 requests changed.")
    return {"preparation": binding(directory / "manifest.json"), "requests": binding(requests),
            "count": count, "case_ids": meta["case_ids"], "native_preparation": meta["model_preparation"]}


def fixed_command(command):
    expected = {"-c": "4096", "-np": "1", "-ngl": "99", "--cache-ram": "0", "--reasoning": "off",
                "--reasoning-budget": "0", "-b": "2048", "-ub": "512", "-fa": "auto", "-ctk": "f16", "-ctv": "f16"}
    require(all(command.count(flag) == 1 and command[command.index(flag) + 1] == value for flag, value in expected.items())
            and all(command.count(flag) == 1 for flag in ("--swa-full", "--jinja", "--no-ui", "--lora", "-m")),
            "Q8 comparison changed the selected b10816/SWA-full product settings.")


def register(model_metadata, native_summary, selection_evidence, source_deployment, base_receipt, output,
             *, matched_q4_summary=None):
    import evaluate_gemma_final as development
    import gemma_final_deployment as deployed

    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh prospective Q8 product registration.")
    model = development.verify_model(model_metadata)
    native, report = development.verify_summary(binding(native_summary))
    require(native["model_identity"] == model["identity"] and report["schema_constrained"] is False
            and native["minimum_numeric_quality_gate"]["passed"] is True,
            "Q8 must compare to the selected adapter's actual eligible native DEV.")
    validate_selection(read(selection_evidence), model, binding(native_summary))
    run, _ = gemma.verify_native_development_run(bound(report["inference_run"]).parent.parent)
    require(run["model_identity"] == model["identity"], "Q8 native comparator differs from its completed grouped DEV/history run.")
    source = Path(source_deployment).resolve()
    deployed.validate_deployment(binding(source), model, read(bound(model["inference_configuration"])))
    gateway, command, original = deployed.deployment_check(source.parent)
    require(original["numerical_regime"] == Q4_REGIME and gateway["runtime_build"].startswith("b10816-")
            and gateway["cache_prompt"] is True and gateway["capture_runtime_evidence"] is False,
            "Use the same selected adapter's fixed Q4 serving configuration as the Q8 command source.")
    fixed_command(command)
    validate_adapter(read(bound(original["adapter_export_receipt"])), model["identity"])
    export, _ = base.verify_export(base_receipt)
    require(export["native_pin_sha256"] == model["identity"]["native_pin_sha256"], "Q8 base uses another native pin.")
    matched = None
    if matched_q4_summary is not None:
        prior, prior_report = development.verify_summary(binding(matched_q4_summary))
        matched_q4_identity(prior, model["identity"])
        require(prior_report["numerical_regime"] == Q4_REGIME, "Matched baseline is not the retained Q4 product.")
        matched = binding(matched_q4_summary)
    require(sha(Q4_FAILURE) == Q4_FAILURE_SHA, "Preserve the original failed Q4 product summary.")
    result = {"version": VERSION, "status": "registered_before_q8_product_inference", "created_at": base.utc(),
        "plan": binding(PLAN), "implementation_sha256": implementation(), "model_metadata": binding(model_metadata),
        "selected_identity": model["identity"], "selected_role": model["role"],
        "selection_evidence": binding(selection_evidence), "native_summary": binding(native_summary),
        "source_deployment": binding(source), "source_gateway": binding(source.parent / "gateway.json"),
        "source_command": binding(source.parent / "command.json"), "base_export_receipt": binding(base_receipt),
        "adapter_export_receipt": original["adapter_export_receipt"], "matched_q4_summary": matched,
        "historical_sft_q4_failed_summary": binding(Q4_FAILURE),
        "same_adapter_quantization_comparison_available": matched is not None,
        "cohorts": {"dev": cohort_binding(DEV, 471), "history": cohort_binding(HISTORY, 24)},
        "numerical_regime": REGIME, "minimum_numeric_gates_unchanged": True,
        "quality_allowance": {"precision_drop": .02, "recall_drop": .02, "preservation_count_drop": 2,
                              "cohorts": ["overall", "canonical_dev322", "supplemental_dev149"]},
        "max_attempts_per_request": 1, "one_full_dev_then_history": True, "app_default_changed": False,
        "model_execution": False, "final_test_bodies_read": False, "body_access_authorized": False}
    require(all(v["native_preparation"] == model["inference_configuration"] for v in result["cohorts"].values()),
            "Q8 DEV/history native preparation differs from its selected adapter.")
    write(output, result)
    return result


def verify_registration(path, selected=None):
    value = read(path)
    require(value["version"] == VERSION and value["status"] == "registered_before_q8_product_inference"
            and value["plan"] == binding(PLAN) and value["implementation_sha256"] == implementation()
            and value["numerical_regime"] == REGIME and value["minimum_numeric_gates_unchanged"] is True
            and value["max_attempts_per_request"] == 1 and value["one_full_dev_then_history"] is True
            and value["quality_allowance"] == {"precision_drop": .02, "recall_drop": .02, "preservation_count_drop": 2,
                                              "cohorts": ["overall", "canonical_dev322", "supplemental_dev149"]}
            and all(value[k] is False for k in ("model_execution", "app_default_changed", "final_test_bodies_read", "body_access_authorized"))
            and (selected is None or value["selected_identity"] == selected), "Q8 prospective product identity or conditions changed.")
    for key in ("model_metadata", "selection_evidence", "native_summary", "source_deployment", "source_gateway",
                "source_command", "base_export_receipt", "adapter_export_receipt", "historical_sft_q4_failed_summary"):
        bound(value[key])
    require(value["historical_sft_q4_failed_summary"]["sha256"] == Q4_FAILURE_SHA,
            "Q8 registration omitted the original Q4 failure.")
    validate_adapter(read(bound(value["adapter_export_receipt"])), value["selected_identity"])
    model = read(bound(value["model_metadata"]))
    require(model["identity"] == value["selected_identity"] and model["role"] == value["selected_role"],
            "Q8 selected model metadata changed.")
    validate_selection(read(bound(value["selection_evidence"])), model, value["native_summary"])
    if value["matched_q4_summary"] is not None:
        matched_q4_identity(read(bound(value["matched_q4_summary"])), value["selected_identity"])
    require(value["same_adapter_quantization_comparison_available"] is (value["matched_q4_summary"] is not None),
            "Q8 matched-adapter comparison label changed.")
    require(value["cohorts"] == {"dev": cohort_binding(DEV, 471), "history": cohort_binding(HISTORY, 24)},
            "Q8 full DEV/history inputs changed.")
    return value


def prepare_deployment(registration, output):
    import gemma_final_deployment as deployed

    value = verify_registration(registration)
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh Q8 deployment directory.")
    original_root = bound(value["source_deployment"]).parent
    gateway, command, original = deployed.deployment_check(original_root)
    require(original["numerical_regime"] == Q4_REGIME, "Only the registered Q4 command is the Q8 source.")
    fixed_command(command)
    export, model_path = base.verify_export(bound(value["base_export_receipt"]))
    parity = read(original_root / "native-token-parity.json")
    gateway, command, receipt = copy.deepcopy(gateway), list(command), copy.deepcopy(original)
    model_sha = export["artifacts"]["base_quantized"]["sha256"]
    gateway.update(model_path=str(model_path), model_sha256=model_sha,
                   token_parity_path=str(output / "native-token-parity.json"))
    parity["model_sha256"] = model_sha
    command[command.index("-m") + 1] = str(model_path)
    output.mkdir(parents=True)
    write(output / "native-token-parity.json", parity)
    gateway["token_parity_sha256"] = sha(output / "native-token-parity.json")
    write(output / "gateway.json", gateway)
    write(output / "command.json", command)
    registry = {"version": VERSION, "numerical_regime": REGIME, "base_export_receipt": value["base_export_receipt"],
        "model_path": str(model_path), "model_sha256": model_sha, "base_repository": base.REPOSITORY,
        "base_revision": base.NATIVE_REVISION, "source_runtime_registry": original["base_registry"]}
    write(output / "q8-base-registry.json", registry)
    receipt.update(numerical_regime=REGIME, q8_registration=binding(registration),
        base_registry=binding(output / "q8-base-registry.json"), base_export_receipt=value["base_export_receipt"],
        files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        implementation_sha256={**original["implementation_sha256"], **implementation()})
    write(output / "deployment.json", receipt)
    verify_deployment(output, value["selected_identity"], preparation=bound(value["cohorts"]["dev"]["native_preparation"]).parent)
    return {"status": "prepared_not_runtime_verified", "numerical_regime": REGIME,
            "deployment": binding(output / "deployment.json"), "model_execution": False}


def verify_deployment(directory, selected, *, preparation=None):
    import gemma_final_deployment as deployed

    directory = Path(directory).resolve()
    gateway, command, receipt = deployed.deployment_check(directory)
    value = verify_registration(bound(receipt["q8_registration"]), selected)
    require(receipt["numerical_regime"] == REGIME and receipt["historical_tooling_only"] is False
            and receipt["purpose"] == "ui_performance" and receipt["format"] == "json"
            and gateway["adapter_scale"] == 1.0 and gateway["capture_runtime_evidence"] is False,
            "Q8 deployment is not the registered separate FP32 LoRA product.")
    original = read(bound(value["source_deployment"]))
    old_gateway, old_command = read(bound(value["source_gateway"])), read(bound(value["source_command"]))
    export, model_path = base.verify_export(bound(receipt["base_export_receipt"]))
    artifact = export["artifacts"]["base_quantized"]
    require(receipt["base_export_receipt"] == value["base_export_receipt"]
            and receipt["adapter_export_receipt"] == value["adapter_export_receipt"] == original["adapter_export_receipt"]
            and export["native_pin_sha256"] == selected["native_pin_sha256"]
            and Path(gateway["model_path"]) == model_path and gateway["model_sha256"] == artifact["sha256"],
            "Q8 deployment uses another base export or adapter.")
    validate_adapter(read(bound(receipt["adapter_export_receipt"])), selected)
    require(gateway["adapter_sha256"] == read(bound(receipt["adapter_export_receipt"]))["gguf_sha256"],
            "Q8 adapter GGUF differs from its exact410 receipt.")
    expected = copy.deepcopy(old_gateway)
    expected.update(model_path=str(model_path), model_sha256=artifact["sha256"],
                    token_parity_path=str(directory / "native-token-parity.json"),
                    token_parity_sha256=sha(directory / "native-token-parity.json"))
    command_expected = list(old_command)
    command_expected[command_expected.index("-m") + 1] = str(model_path)
    require(gateway == expected and command == command_expected
            and receipt["runtime_files_sha256"] == original["runtime_files_sha256"]
            and receipt["runtime_symlink_targets"] == original["runtime_symlink_targets"]
            and receipt.get("runtime_settings") == original.get("runtime_settings"),
            "Q8 changed settings other than the registered base artifact.")
    parity = read(bound(value["source_deployment"]).parent / "native-token-parity.json")
    parity["model_sha256"] = artifact["sha256"]
    require(read(directory / "native-token-parity.json") == parity, "Q8 native prefix checks changed.")
    fixed_command(command)
    if preparation is not None:
        require(receipt["native_preparation"] == binding(Path(preparation) / "manifest.json"),
                "Q8 native preparation differs from its selected adapter.")
    return {key: gateway[key] for key in ("model_id", "model_sha256", "adapter_sha256", "adapter_scale")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    for name in ("model-metadata", "native-summary", "selection-evidence", "source-deployment", "base-receipt", "output"):
        reg.add_argument("--" + name, type=Path, required=True)
    reg.add_argument("--matched-q4-summary", type=Path)
    prep = sub.add_parser("prepare-deployment")
    for name in ("registration", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command").replace("-", "_")
    print(json.dumps(globals()[command](**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
