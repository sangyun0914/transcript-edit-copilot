"""Conditional Gemma26 evidence boundaries; never activates a model or TEST gate."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import prepare_gemma_actions_v3 as prep
import prepare_gemma_final_fallback as fallback
import train_gemma_actions_v3 as sft
import prepare_actions_v3_conditions as conditions

ROOT = Path(__file__).resolve().parents[1]
VERSION = "gemma26-final-readiness-1"
CONTEXT_VERSION = "gemma26-final-full-cohort-context-audit-1"
PLAN = ROOT / "docs/plans/final-gemma26-bridge-2026-10-25.md"
RELEASE = ROOT / "artifacts/datasets/nep-passive-actions-final-v1/release-attempt-001/manifest.json"
RELEASE_SHA = "86a997f28826597e63d55fd6bf4ab8496c3bdaa0950da5bda92713acbdaaaf58"
DEMONSTRATION_IDS = ["v3-pilot-actions-v1-en-field-recorder-01", "v3-pilot-actions-v1-ko-library-recording-00"]
require, read, sha, write = prep.require, prep.read, prep.digest, prep.write
binding, bound = fallback.binding, fallback.bound


def release_cases(release=RELEASE):
    release = Path(release).resolve()
    value = read(release)
    require(sha(release) == RELEASE_SHA and value["counts"]["train"] == 1261 and value["counts"]["dev"] == 471
            and value["counts"]["canonical_dev"] == 322 and value["counts"]["supplemental_dev"] == 149
            and value["test_bodies_read"] is False, "Use the exact final1261/471 release.")
    cases = {}
    for split in ("train", "dev"):
        path = release.parent / f"cases/{split}.jsonl"
        require(sha(path) == value["files_sha256"][f"cases/{split}.jsonl"], "Final release cases changed.")
        cases[split] = prep.rows(path)
        require(len(cases[split]) == value["counts"][split]
                and len({row["id"] for row in cases[split]}) == len(cases[split])
                and all(row["split"] == row["source"]["split"] == split for row in cases[split]),
                "Final release row identity or split changed.")
    return value, cases


def verify_draws(config, cases):
    sft.validate_configuration(config)
    require(config["format"] == "json" and config["phase"] == "substantive",
            "Final Gemma requires a new substantive JSON SFT.")
    ids = [row["id"] for row in cases]
    order = config["training"]["sample_order"]
    require(len(ids) == len(set(ids)) == 1261 and len(order) == 3783
            and all(Counter(order[start:start + 1261]) == Counter(ids) for start in range(0, 3783, 1261)),
            "Every final1261 row must occur once in each of the three epochs; B003 is not this parent.")
    expected = {"draw_count": 3783, "unique_cases": 1261, "max_updates": 946,
                "gradient_accumulation": 4, "final_update_rows": 3, "learning_rate": 5e-5, "seed": 42}
    require(all(config["training"].get(key) == value for key, value in expected.items()),
            "Final Gemma optimizer budget differs from the registered fallback.")


def verify_preparation(config_path, release=RELEASE):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, read(config_path)
    dataset, cases = release_cases(release)
    bundle = prep.verify_bundle(root)
    require(config == bundle["configs"]["json"] and bundle["cases"] == cases["train"],
            "Gemma preparation changed the saved final TRAIN messages or input order.")
    verify_draws(config, cases["train"])
    source, protocol = read(root / "source-binding.json"), read(root / "protocol.json")
    require(source["train_cases_sha256"] == dataset["files_sha256"]["cases/train.jsonl"]
            and source["original_train_rows"] == source["admitted_rows"] == 1261
            and source["budget_exclusions"] == 0 and sha(root / "protocol.json") == config["protocol_sha256"]
            and protocol["version"] == fallback.VERSION and protocol["final_test_bodies_read"] is False
            and protocol["dataset"]["release"] == binding(release)
            and protocol["dataset"]["train_cases_sha256"] == source["train_cases_sha256"]
            and all(config["training"][key] == value for key, value in protocol["training_plan"].items()),
            "Fallback preparation lacks the exact unchanged full TRAIN and conditional protocol.")
    failures = protocol.get("completed_small_failure_summaries")
    require(isinstance(failures, list) and len(failures) == len(fallback.MODELS),
            "The activated Gemma preparation must retain both actual small-family failure results.")
    verify_small_failures(dict(zip(fallback.MODELS, failures, strict=True)))
    return config, bundle


def verify_small_failures(references):
    from build_small_final_freeze_actions_v3 import verified_dev_summary
    import small_actions_final as small

    require(isinstance(references, dict) and set(references) == set(fallback.MODELS),
            "Fallback requires both completed native small-family DEV results.")
    for name, reference in references.items():
        summary, report = verified_dev_summary(reference)
        require(summary["model_identity"]["kind"] == "sft"
                and summary["model_identity"]["base"] == small.PROFILES[name]
                and report["numerical_regime"] == "native_bf16_fp32_lora"
                and report["schema_constrained"] is False
                and summary["minimum_numeric_quality_gate"]["passed"] is False,
                "Fallback cannot activate on an unfinished, quantized, or passing small candidate.")
    return references


def verify_trained(config_path, training, reloaded, *, release=RELEASE, kind=sft.VERSION, parent=None,
                   dpo_preparation=None):
    import infer_task_test_actions_v3 as native
    import prepare_b003_final_gate_actions_v3 as recorded

    config, _ = verify_preparation(config_path, release)
    training, reloaded = Path(training), Path(reloaded)
    saved = read(training / "configuration.json")
    if kind == sft.VERSION:
        require(saved == config and parent is None, "New SFT configuration differs from the full final preparation.")
    else:
        require(kind == "gemma-actions-v3-dpo" and parent is not None and dpo_preparation is not None,
                "Only the same-parent final Gemma DPO is supported.")
        import train_gemma_actions_v3_dpo as dpo
        import prepare_gemma_final_dpo as preferences
        dpo_config, dpo_root, _, _, pairs, _, _ = dpo.preflight(Path(dpo_preparation) / "configuration.json", assets=False)
        protocol = read(dpo_root / "protocol.json")
        mined = preferences.verify_mining(bound(protocol["mining"]).parent, parent)
        require(dpo_config == saved and protocol["version"] == preferences.VERSION
                and protocol["parent"] == parent and protocol["epochs"] == 1
                and protocol["saved_projection"] is True and mined["pairs"] >= 8
                and mined["preservation_pairs"] >= 1 and len(pairs) == mined["pairs"]
                and saved["training"]["learning_rate"] == 1e-6 and saved["training"]["beta"] == 0.1
                and saved["training"]["gradient_accumulation"] == 4,
                "DPO must retain the reviewed same-parent pairs and one-pass prospective budget.")
        require(saved["anchor"] == {"anchor_" + key: parent[key] for key in
                                    ("training_manifest_sha256", "reload_manifest_sha256", "configuration_sha256", "adapter_sha256")}
                and saved["sft_preparation_sha256"] == sha(Path(config_path).parent / "manifest.json")
                and saved.get("saved_projection") is True,
                "DPO must start from this new final SFT and preserve saved H0/H3/H8 projections.")
    _, identity, _, runs = recorded.recorded_run(training / "configuration.json", training, reloaded, kind)
    actual = native.native_identity(config, training, reloaded, adapter_version=kind)
    require(actual == identity, "Native adapter/reload payload identity differs from recorded completion.")
    before, after = runs
    if kind == sft.VERSION:
        exposure = read(training / "exposure.json")
        require(before["training_updates"] == after["reloaded_training_updates"] == 946
                and before["optimizer_committed_rows"] == 3783
                and exposure["phases"]["optimizer_committed"]["rows"] == 3783
                and exposure["completed_updates"] == 946 and exposure["optimizer_commit_uncertain"] is False
                and exposure["pending_backward_rows_not_committed"] == 0,
                "Final SFT did not commit its complete three-epoch optimizer budget.")
    else:
        require(before["initialization"] == "frozen_sft_adapter_fresh_optimizer"
                and before["optimizer_committed_rows"] == saved["training"]["pair_count"]
                and before["reference_manifest_sha256"] == after["reference_manifest_sha256"],
                "DPO completion differs from its frozen parent reference or full pair count.")
    return {"identity": identity, "training": binding(training / "manifest.json"),
            "reload": binding(reloaded / "manifest.json"), "configuration": binding(training / "configuration.json"),
            "runtime": binding(training / "runtime.json"), "environment": before["environment"],
            "training_source_sha256": saved.get("preparation_sources_sha256", saved["sources_sha256"])}


def context_records(cases, demonstrations, tokenizer):
    spec = conditions.condition_specs(demonstrations)["train_icl"]
    result = []
    for case in cases:
        require(case["split"] == case["source"]["split"] == "dev"
                and case["messages"] == prep.formats.messages(case["input"]),
                "Context preaudit only accepts unchanged exposed DEV inputs.")
        canonical = conditions.native_prefix(tokenizer, case["messages"])
        icl = conditions.native_prefix(tokenizer, conditions.expected_messages(case["input"], "json", spec))
        result.append({"case_id": case["id"], "canonical_tokens": len(canonical), "icl_tokens": len(icl),
                       "canonical_prefix_sha256": prep.value_hash(canonical), "icl_prefix_sha256": prep.value_hash(icl),
                       "input_sha256": conditions.input_hash(case["input"]),
                       "history_length": case.get("history_length", {}).get("length", 3)})
    require(result and max(row["canonical_tokens"] for row in result) <= 3584
            and max(row["icl_tokens"] for row in result) <= 7680,
            "The entire common cohort must fit 3584 canonical / 7680 ICL; never crop or drop rows.")
    return result


def audit_context(readiness, output, *, release=RELEASE):
    from transformers import AutoTokenizer

    readiness, output = Path(readiness), Path(output).resolve()
    require(not output.exists(), "Use a fresh exposed context audit directory.")
    draft = read(readiness)
    require(draft["version"] == fallback.VERSION and draft["training_started"] is False
            and draft["final_test_bodies_read"] is False and draft["inputs"]["release"] == binding(release),
            "Audit the existing conditional preparation without activating it.")
    dataset, cases = release_cases(release)
    pin, native = read(bound(draft["inputs"]["native_pin"])), Path(draft["native_metadata_path"])
    require(pin["repository"] == prep.gemma.REPOSITORY and pin["revision"] == prep.gemma.REVISION,
            "The Gemma26 native tokenizer pin changed.")
    files = {name: sha(native / name) for name in prep.TOKENIZER_FILES}
    require(all(files[name] == pin["files"][name]["sha256"] for name in files), "Native tokenizer metadata changed.")
    config = read(native / "config.json")
    require(config.get("text_config", config)["max_position_embeddings"] >= 8192, "Native context is below8192.")
    tokenizer = AutoTokenizer.from_pretrained(native, local_files_only=True, trust_remote_code=False)
    by_id = {row["id"]: row for row in cases["train"]}
    release_binding = {"manifest": binding(release), "train_cases_sha256": dataset["files_sha256"]["cases/train.jsonl"],
                       "dev_cases_sha256": dataset["files_sha256"]["cases/dev.jsonl"]}
    demos = [conditions.demonstration(by_id[name], tokenizer, release_binding) for name in DEMONSTRATION_IDS]
    records = context_records(cases["dev"], demos, tokenizer)
    output.mkdir(parents=True)
    prep.write_rows(output / "lengths.jsonl", records)
    write(output / "demonstrations.json", demos)
    result = {"version": CONTEXT_VERSION, "scope": "exposed_development_only", "release": binding(release),
              "plan": binding(PLAN), "conditional_readiness": binding(readiness),
              "model_weights_loaded": False, "training_started": False, "final_test_bodies_read": False,
              "tokenizer": {"path": str(native), "files_sha256": files}, "native_pin": draft["inputs"]["native_pin"],
              "fixed_demonstration_ids": DEMONSTRATION_IDS, "cases": len(records), "over_budget": 0,
              "canonical_max": max(row["canonical_tokens"] for row in records),
              "icl_max": max(row["icl_tokens"] for row in records),
              "physical_context_tokens": 8192, "inference_prefix_limit": 7680, "output_tokens": 512,
              "all_roles_same_case_ids": True, "overflow_policy": "fail_whole_preparation",
              "demonstrations": binding(output / "demonstrations.json"), "lengths": binding(output / "lengths.jsonl")}
    write(output / "audit.json", result)
    return result


def verify_deployment(deployment, selected, *, preparation=None):
    for path in (ROOT / "server", ROOT / "server/scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_local_copilot_actions_v3 as local

    directory = Path(deployment).resolve()
    gateway, _, command, receipt = local.check(directory)
    import gemma_q8_product as q8
    if receipt.get("numerical_regime") == q8.REGIME:
        return q8.verify_deployment(directory, selected, preparation=preparation)
    export = read(bound(receipt["adapter_export_receipt"]))
    require(receipt["version"] == local.VERSION and receipt["historical_tooling_only"] is False
            and receipt["purpose"] == "ui_performance" and receipt["format"] == "json"
            and receipt["numerical_regime"] == "native_derived_q4_base_fp32_lora"
            and gateway.capture_runtime_evidence is False and gateway.output_format == "json"
            and gateway.adapter_path is not None and gateway.adapter_scale == 1.0
            and gateway.adapter_sha256 == export["gguf_sha256"]
            and command[command.index("--lora") + 1] == str(gateway.adapter_path)
            and command[command.index("-m") + 1] == str(gateway.model_path)
            and command[command.index("-c") + 1] == "4096",
            "Gemma deployment must retain the actual Q4 base, separate FP32 LoRA and scale1.")
    require(export["version"] == "gemma-actions-v3-lora-export" and export["all_tensors_exact"] is True
            and export["status"] == "adapter_conversion_verified_serving_pending" and export["adapter_tensor_count"] == 410
            and export["contract_version"] == prep.formats.contract.VERSION and export["format_version"] == prep.formats.VERSION
            and export["output_format"] == "json" and export["runtime_revision"] == "427291b5b34cd914a31b3fd3b61a68f6184f4b9f"
            and all(export[key] == selected[key] for key in
                    ("adapter_sha256", "training_manifest_sha256", "reload_manifest_sha256", "native_pin_sha256"))
            and export["base_repository"] == selected["repository"] and export["base_revision"] == selected["revision"],
            "Gemma deployment belongs to another trained adapter, reload or native pin.")
    base = read(bound(receipt["base_export_receipt"]))
    require(base["artifacts"]["base_quantized"]["sha256"] == gateway.model_sha256
            and base["base_repository"] == selected["repository"] and base["base_revision"] == selected["revision"],
            "GGUF base is not the selected native-derived family.")
    if preparation is not None:
        require(receipt["native_preparation"] == binding(Path(preparation) / "manifest.json"),
                "Deployment does not bind the new unified preparation.")
    return {"model_id": gateway.model_id, "model_sha256": gateway.model_sha256,
            "adapter_sha256": gateway.adapter_sha256, "adapter_scale": gateway.adapter_scale}


def verify_model(config_path, training, reloaded, output, *, dpo_training=None, dpo_reloaded=None,
                 dpo_preparation=None, deployment=None, deployment_role="sft"):
    output = Path(output).resolve()
    require(not output.exists() and bool(dpo_training) == bool(dpo_reloaded) == bool(dpo_preparation)
            and deployment_role in ("sft", "sft_dpo"),
            "Use a fresh evidence receipt and supply both DPO training/reload or neither.")
    for source in (Path(config_path).parent, training, reloaded, dpo_training, dpo_reloaded, dpo_preparation):
        require(source is None or not output.is_relative_to(Path(source).resolve()),
                "Keep the new readiness receipt outside immutable model inputs.")
    evidence = {"sft": verify_trained(config_path, training, reloaded)}
    if dpo_training is not None:
        evidence["sft_dpo"] = verify_trained(config_path, dpo_training, dpo_reloaded,
                                             kind="gemma-actions-v3-dpo", parent=evidence["sft"]["identity"],
                                             dpo_preparation=dpo_preparation)
    require(deployment is None or deployment_role in evidence, "The requested deployment role has no verified training/reload.")
    result = {"version": VERSION, "status": "model_evidence_verified_not_final_registered",
              "scope": "conditional_evidence_boundaries", "plan": binding(PLAN), "models": evidence,
              "deployment": binding(Path(deployment) / "deployment.json") if deployment is not None else None,
              "deployment_model_role": deployment_role if deployment is not None else None,
              "deployment_identity": verify_deployment(deployment, evidence[deployment_role]["identity"], preparation=Path(config_path).parent)
              if deployment is not None else None,
              "model_selection_performed": False, "body_access_authorized": False, "final_test_bodies_read": False,
              "model_loaded": False, "training_started": False,
              "remaining_final_contracts": [
                  "full_DEV_native_and_separate_LoRA_admission_and_quality_selection",
                  "native_5_or_6_role_metadata_and_8192_common_cohort_controller_dispatch",
                  "same_parent_DPO_or_fully_reviewed_omission_final_registration",
                  "deployed_auxiliary_policy_and_actual_loaded_base_LoRA_scale_verification",
                  "Gemma_quality_only_UI_pending_selection_guard_and_product_completion",
                  "fresh_source_closure_freeze_and_independent_root_readback"]}
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-context")
    for name in ("readiness", "output"):
        audit.add_argument("--" + name, type=Path, required=True)
    model = sub.add_parser("verify-model")
    for name in ("config-path", "training", "reloaded", "output"):
        model.add_argument("--" + name, type=Path, required=True)
    for name in ("dpo-training", "dpo-reloaded", "dpo-preparation", "deployment"):
        model.add_argument("--" + name, type=Path)
    model.add_argument("--deployment-role", choices=("sft", "sft_dpo"), default="sft")
    args = vars(parser.parse_args())
    command = args.pop("command").replace("-", "_")
    print(json.dumps(globals()[command](**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
