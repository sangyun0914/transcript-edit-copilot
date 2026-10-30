"""Bind the new unified Gemma26 SFT to the full-cohort final comparison."""

import argparse
from pathlib import Path
import shutil

import gemma_final_readiness as evidence
import gemma_native_inference_overlay as native_overlay
import prepare_gemma_final_dpo as preferences

FAMILY = "gemma26-final-native-v1"
VERSION = "gemma26-final-test-model-metadata-1"
INFERENCE_POLICY_VERSION = "gemma26-final-full-cohort-inference-policy-1"
ROOT = evidence.ROOT
CONTEXT_PLAN = evidence.PLAN
CONTEXT_AUDIT = ROOT / "artifacts/model-preparation/gemma26-final-v1/final-context-audit-001/audit.json"
CONTEXT_AUDIT_SHA256 = "b3a166d721fe150b8c67eba6f694c32fdd4a53008eebec11f4762384489edbb8"
DEMONSTRATION_IDS = evidence.DEMONSTRATION_IDS
require, read, sha, write = evidence.require, evidence.read, evidence.sha, evidence.write
binding, bound, digest = evidence.binding, evidence.bound, evidence.prep.value_hash
NATIVE_ADOPTION_VERSION = "gemma26-native-runtime-adoption-1"
NATIVE_PROSPECTIVE_SHA = "0ab9f084fc91a69473460c4718b1a544080a0076df5d45d5ab299977aa3f3240"


def verify_native_development_run(directory):
    directory = Path(directory).resolve()
    manifest, info, registration = (read(directory / name) for name in ("manifest.json", "run.json", "registration.json"))
    require(manifest["version"] == info["version"] == registration["version"] == "gemma26-grouped-development-1"
            and manifest["status"] == info["status"] == "completed" and info["fatal_error"] is None
            and info["request_failures"] == 0 and info["observed_by_scope"] == {"dev": 471, "history": 24}
            and info["registration_sha256"] == sha(directory / "registration.json")
            and info["runtime_overlay"] == registration["runtime_overlay"] == native_overlay.OVERLAY
            and info["model_identity"] == registration["model_identity"]
            and registration["prospective_registration_sha256"] == NATIVE_PROSPECTIVE_SHA,
            "Grouped DEV/history outer run is incomplete or failed after a child scope completed.")
    files = manifest["files_sha256"]
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p != directory / "manifest.json"}
    require(set(files) == actual, "Grouped outer artifact inventory changed.")
    for name, expected in files.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory) and sha(path) == expected, "Grouped outer artifact bytes changed.")
    parity = read(directory / "adapter-parity.json")
    require(parity["adapter_tensors_identical"] is True and len(parity["tensors"]) == 410,
            "Grouped inference lacks the complete unchanged adapter tensor receipt.")
    for scope, count in (("dev", 471), ("history", 24)):
        run, runtime, child = (read(directory / scope / name) for name in ("run.json", "runtime.json", "manifest.json"))
        records = evidence.prep.rows(directory / scope / "observations.jsonl")
        require(run["status"] == child["status"] == "inference_complete" and run["requested"] == run["observed"] == count
                and run["request_failures"] == 0 and run["completed_with_fatal_failure"] is False
                and run["scope"] == scope and run["model_identity"] == info["model_identity"]
                and run["runtime_overlay"] == runtime["runtime_overlay"] == native_overlay.OVERLAY
                and run["registration_sha256"] == runtime["registration_sha256"] == info["registration_sha256"]
                and run["runtime_fingerprint"] == runtime["runtime_fingerprint"] == info["runtime_fingerprint"]
                and [r["case_id"] for r in records] == registration["cohorts"][scope]["case_ids"] and len(records) == count,
                "Grouped DEV/history scope changed its completed census, order or runtime identity.")
        for name, expected in child["files_sha256"].items():
            require(files.get(scope + "/" + name) == expected, "Child scope differs from its completed outer publication.")
        for row in records:
            require(row["status"] == "completed", "Grouped model evaluation contains an unsuccessful request.")
            trace = object.__new__(native_overlay.BackendTrace)
            trace.rows = row["backend_trace"]
            trace.validate(trace.rows[0]["input_tokens"], len(row["output_ids"]))
    return info, registration


def validate_native_adoption(value, *, parent=None, resolve=bound):
    prospective = read(resolve(value["prospective_registration"]))
    required_checks = {"minimum_numeric_gates", "cohort_quality", "action_language_successes",
                       "no_new_entire_typed_subgroup_failure", "relative_speed"}
    require(value.get("version") == NATIVE_ADOPTION_VERSION and value["status"] == "adopted"
            and value["backend"] == "grouped_mm" and value["all_passed"] is True
            and set(value["checks"]) == required_checks and all(v is True for v in value["checks"].values())
            and value["runtime_overlay"] == native_overlay.OVERLAY
            and value["prospective_registration"]["sha256"] == NATIVE_PROSPECTIVE_SHA
            and prospective["version"] == "gemma26-grouped-dev-prospective-1"
            and value["criteria"] == prospective["criteria"]
            and value["model_identity"]["kind"] == evidence.sft.VERSION
            and (parent is None or value["model_identity"] == parent)
            and all(value[key] is False for key in ("runtime_selection_uses_DPO_results", "DPO_thresholds_changed",
                                                    "final_test_bodies_read", "model_execution")),
            "Native inference requires the prospectively accepted same-SFT runtime decision.")
    for reference in value["inputs"].values():
        resolve(reference)
    resolve(value["implementation"])
    require(value["inputs"]["original_SFT_summary"]["sha256"] == prospective["bindings"]["original_sft_dev"]["sha256"],
            "Native runtime choice replaced its original SFT evidence.")
    outer, _ = verify_native_development_run(resolve(value["inputs"]["optimized_outer_manifest"]).parent)
    require(outer["model_identity"] == value["model_identity"], "Runtime adoption belongs to another completed model.")
    return value


def native_policy_overlay(policy, *, parent=None, resolve=bound):
    overlay = policy.get("native_experts_overlay")
    if overlay is None:
        return None
    require(set(overlay) == {"settings", "adoption"} and overlay["settings"] == native_overlay.OVERLAY,
            "Unregistered native inference overlay settings.")
    validate_native_adoption(read(resolve(overlay["adoption"])), parent=parent, resolve=resolve)
    return overlay


def validate_configuration(config):
    evidence.sft.validate_configuration(config)
    require(config["format"] == "json" and config["phase"] == "substantive",
            "Final Gemma uses the new substantive JSON configuration.")


def tokenizer_identity(config_path):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    validate_configuration(config)
    native = config_path.parent / config["tokenizer"]["path"]
    pin = read(config_path.parent / "native-pin.json")
    identity = {"runtime_family": FAMILY, "model": "gemma26", "path": str(native.resolve()),
                "eos_token_id": 1, "terminal_ids": config["generation"]["eos_token_id"],
                "vocab_size": read(native / "config.json")["text_config"]["vocab_size"],
                "files_sha256": config["tokenizer"]["files_sha256"], "assets": pin,
                "native_pin": binding(config_path.parent / "native-pin.json")}
    validate_tokenizer_identity(identity)
    return identity


def validate_tokenizer_identity(identity):
    require(identity.get("runtime_family") == FAMILY and identity["model"] == "gemma26"
            and identity["eos_token_id"] == 1 and identity["terminal_ids"] == [1, 106, 50]
            and identity["assets"] == read(bound(identity["native_pin"])), "Gemma tokenizer identity changed.")
    pin = identity["assets"]
    require(pin["repository"] == evidence.prep.gemma.REPOSITORY
            and pin["revision"] == evidence.prep.gemma.REVISION
            and set(identity["files_sha256"]) == set(evidence.prep.TOKENIZER_FILES),
            "Gemma tokenizer is not the pinned native family.")
    root = Path(identity["path"])
    for name, expected in identity["files_sha256"].items():
        require(sha(root / name) == expected == pin["files"][name]["sha256"], "Gemma tokenizer bytes changed.")
    native = read(root / "config.json")["text_config"]
    require(native["vocab_size"] == identity["vocab_size"] and native["max_position_embeddings"] >= 8192,
            "Gemma native vocabulary or physical context changed.")


def validate_inference_policy(policy, config=None, *, resolve=bound):
    require(policy.get("version") == INFERENCE_POLICY_VERSION and policy["status"] == "registered_before_test"
            and policy["model"] == "gemma26" and policy["canonical_window_input_tokens"] == 3584
            and policy["physical_context_tokens"] == 8192 and policy["inference_prefix_limit"] == 7680
            and policy["output_tokens"] == 512 and policy["training_context_tokens"] == 4096
            and policy["application_context_tokens"] == 4096 and policy["all_roles_same_case_ids"] is True
            and policy["overflow_policy"] == "fail_whole_preparation"
            and policy["fixed_demonstration_ids"] == DEMONSTRATION_IDS
            and policy["prospective_plan"] == binding(CONTEXT_PLAN)
            and policy["exposed_development_audit"]["sha256"] == CONTEXT_AUDIT_SHA256
            and policy["body_access_authorized"] is False and policy["final_test_bodies_read"] is False,
            "Unregistered Gemma full-cohort inference policy.")
    audit = read(resolve(policy["exposed_development_audit"]))
    pin = read(resolve(policy["assets_receipt"]))
    require(audit["scope"] == "exposed_development_only" and audit["cases"] == 471
            and audit["over_budget"] == 0 and audit["canonical_max"] <= 3584 and audit["icl_max"] <= 7680
            and audit["fixed_demonstration_ids"] == DEMONSTRATION_IDS and audit["model_weights_loaded"] is False
            and audit["final_test_bodies_read"] is False and read(resolve(audit["native_pin"])) == pin,
            "Gemma full-cohort context audit changed.")
    if config is not None:
        validate_configuration(config)
        require(pin["repository"] == config["base"]["repository"] and pin["revision"] == config["base"]["revision"],
                "Gemma context policy belongs to another native model.")
    native_policy_overlay(policy, resolve=resolve)
    return policy


def inference_policy(config_path, native_runtime_adoption=None):
    config_path = Path(config_path)
    policy = {"version": INFERENCE_POLICY_VERSION, "status": "registered_before_test",
        "model": "gemma26", "canonical_window_input_tokens": 3584, "physical_context_tokens": 8192,
        "inference_prefix_limit": 7680, "output_tokens": 512, "training_context_tokens": 4096,
        "application_context_tokens": 4096, "all_roles_same_case_ids": True,
        "overflow_policy": "fail_whole_preparation", "fixed_demonstration_ids": DEMONSTRATION_IDS,
        "prospective_plan": binding(CONTEXT_PLAN), "exposed_development_audit": binding(CONTEXT_AUDIT),
        "assets_receipt": binding(config_path.parent / "native-pin.json"),
        "body_access_authorized": False, "final_test_bodies_read": False}
    if native_runtime_adoption is not None:
        policy["native_experts_overlay"] = {"settings": native_overlay.OVERLAY,
                                             "adoption": binding(native_runtime_adoption)}
    return validate_inference_policy(policy, read(config_path))


def verify_fallback(config_path):
    config, bundle = evidence.verify_preparation(config_path)
    protocol = read(Path(config_path).parent / "protocol.json")
    draft = read(bound(protocol["readiness_manifest"]))
    evidence.fallback.validate_lr_recovery(read(bound(draft["inputs"]["small_configurations"][0])),
        read(bound(draft["inputs"]["lr_recovery_configuration"])))
    reference = protocol["completed_lr_recovery_failure_summary"]
    from build_small_final_freeze_actions_v3 import verified_dev_summary
    summary, report = verified_dev_summary(reference)
    require(summary["model_identity"]["kind"] == "sft" and summary["model_identity"]["configuration_sha256"]
            == draft["inputs"]["lr_recovery_configuration"]["sha256"]
            and read(bound(draft["inputs"]["lr_recovery_configuration"]))["training"]["learning_rate"] == 2e-4
            and summary["minimum_numeric_quality_gate"]["passed"] is False
            and report["numerical_regime"] == "native_bf16_fp32_lora" and report["schema_constrained"] is False,
            "Final Gemma requires the completed registered LR recovery failure too.")
    require(draft["inputs"]["lr_recovery_plan"] == binding(evidence.fallback.LR_RECOVERY_PLAN)
            and sha(evidence.fallback.LR_RECOVERY_PLAN) == evidence.fallback.LR_RECOVERY_PLAN_SHA,
            "The prospective LR recovery plan changed.")
    return config, bundle


def native_identity(config, training=None, reloaded=None, kind=evidence.sft.VERSION):
    from infer_gemma_actions_v3 import adapter_identity
    validate_configuration(config)
    require(bool(training) == bool(reloaded), "Gemma adapter requires its independent reload.")
    identity = {"kind": "native_base", "repository": config["base"]["repository"],
                "revision": config["base"]["revision"], "native_pin_sha256": config["native_pin_sha256"]}
    if training:
        identity.update(adapter_identity(training, reloaded, kind, config))
    return identity


def validate_omission(value, config, parent, *, resolve=bound):
    require(value.get("version") == "gemma26-final-dpo-decision-1"
            and value["status"] == "registered_before_test" and value["decision"] == "eligible_omission"
            and value["parent"] == parent and value["body_access_authorized"] is False
            and value["final_test_bodies_read"] is False and value["final_registration_performed"] is False
            and value["plan"] == binding(evidence.PLAN), "Unregistered same-parent Gemma DPO omission.")
    report = preferences.verify_mining(resolve(value["mining"]).parent, parent)
    counts = {key: report[key] for key in ("requested", "pairs", "preservation_pairs")}
    reasons = (["fewer_than_eight_strict_pairs"] if counts["pairs"] < 8 else [])
    reasons += ["no_actual_preservation_error_pair"] if counts["preservation_pairs"] == 0 else []
    require(reasons and value["reason_codes"] == reasons and value["counts"] == counts
            and counts["requested"] == config["training"]["unique_cases"] == 1261,
            "Gemma omission must retain reviewed full TRAIN pair scarcity.")
    return value


def validate_preparation(path, config_path):
    meta = read(path)
    require(meta.get("version") == VERSION and meta["status"] == "verified_not_body_authorized"
            and meta["body_access_authorized"] is False and meta["final_test_bodies_read"] is False
            and bound(meta["configuration"]) == Path(config_path).resolve(), "Gemma model metadata changed.")
    config, _ = verify_fallback(config_path)
    require(meta["training_dataset_manifest"] == binding(evidence.RELEASE) and meta["train_rows"] == 1261
            and meta["native_preparation"] == binding(Path(config_path).parent / "manifest.json")
            and meta["tokenizer"] == tokenizer_identity(config_path), "Gemma dataset, preparation or tokenizer changed.")
    validate_inference_policy(read(bound(meta["inference_policy"])), config)
    return meta, config


def verify_model_set(models, tokenizer, dataset_manifest, omission=None, policy_binding=None):
    roles = {"native_base", "sft"} if omission is not None else {"native_base", "sft", "sft_dpo"}
    require(set(models) == roles and all(m.get("runtime_family") == FAMILY for m in models.values()),
            "Gemma comparison requires one coherent new SFT family and explicit DPO disposition.")
    base = models["native_base"]
    common = {k: base[k] for k in ("inference_configuration", "preparation_manifest", "execution_runtime", "inference_policy")}
    require(all(all(m[k] == v for k, v in common.items()) for m in models.values()), "Gemma common runtime differs.")
    meta, config = validate_preparation(bound(common["preparation_manifest"]), bound(common["inference_configuration"]))
    require(meta["training_dataset_manifest"] == dataset_manifest and meta["tokenizer"] == tokenizer
            and meta["dpo_omission"] == omission and meta["inference_policy"] == common["inference_policy"]
            and (policy_binding is None or policy_binding == common["inference_policy"]), "Gemma model set policy changed.")
    parent = None
    for role in ("sft", "sft_dpo"):
        if role not in models:
            continue
        model = models[role]
        checked = evidence.verify_trained(bound(common["inference_configuration"]),
            bound(model["artifact_manifest"]).parent, bound(model["reload_receipt"]).parent,
            kind=evidence.sft.VERSION if role == "sft" else "gemma-actions-v3-dpo", parent=parent,
            dpo_preparation=bound(meta["dpo_preparation"]).parent if role == "sft_dpo" else None)
        require(checked["identity"] == model["identity"] and checked["environment"] == meta["environment"]
                and checked["training_source_sha256"] == meta["training_implementation_sha256"][role],
                "Gemma completed training/reload identity changed.")
        if role == "sft":
            parent = checked["identity"]
        else:
            require(model["parent_identity_sha256"] == digest(parent), "Gemma DPO parent changed.")
    if omission is not None:
        validate_omission(read(bound(omission)), config, parent)
    native_policy_overlay(read(bound(common["inference_policy"])), parent=parent)
    require(base["identity"] == native_identity(config)
            and bound(base["artifact_manifest"]) == bound(models["sft"]["artifact_manifest"]).parent / "runtime.json"
            and base["reload_receipt"] == models["sft"]["reload_receipt"], "Gemma native-base evidence changed.")
    return meta


def assemble(config_path, metadata_path, output):
    import assemble_task_test_actions_v3 as assembly
    meta, _ = validate_preparation(metadata_path, config_path)
    config, plan = assembly.source_plan(config_path)
    output = Path(output)
    require(not output.exists(), "Use a fresh final runtime directory.")
    output.mkdir(parents=True)
    for name, source in plan.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["path"], target)
    manifest = {"version": assembly.VERSION, "runtime_family": FAMILY,
        "status": "assembled_model_specific_runtime", "scope": "final_test", "entrypoint": assembly.ENTRY,
        "configuration_sha256": sha(config_path), "preparation_manifest_sha256": sha(metadata_path),
        "inference_policy_sha256": meta["inference_policy"]["sha256"],
        "configuration_sources_sha256": config["sources_sha256"],
        "source_sha256": {name: value["sha256"] for name, value in plan.items()},
        "source_origin": {name: value["origin"] for name, value in plan.items()},
        "source_policy": "frozen_training_sources_with_final_inference_overlay",
        "training_implementation_sha256": meta["training_implementation_sha256"],
        "dpo_omission_sha256": meta["dpo_omission"]["sha256"] if meta["dpo_omission"] else None,
        "final_test_bodies_read": False, "model_weights_read": False}
    overlay = native_policy_overlay(read(bound(meta["inference_policy"])))
    if overlay is not None:
        manifest["native_inference_overlay"] = {"settings": overlay["settings"],
            "adoption_sha256": overlay["adoption"]["sha256"], "prospective_registration_sha256": NATIVE_PROSPECTIVE_SHA}
    write(output / assembly.MANIFEST, manifest)
    return assembly.load_runtime(output)


def register(config_path, output, *, sft_training, sft_reloaded, dpo_training=None, dpo_reloaded=None,
             dpo_preparation=None, dpo_omission=None, native_runtime_adoption=None):
    config, _ = verify_fallback(config_path)
    has_dpo = all(v is not None for v in (dpo_training, dpo_reloaded, dpo_preparation))
    require(has_dpo != (dpo_omission is not None)
            and (has_dpo or all(v is None for v in (dpo_training, dpo_reloaded, dpo_preparation))),
            "Supply same-parent completed DPO or its reviewed omission decision.")
    proven = {"sft": evidence.verify_trained(config_path, sft_training, sft_reloaded)}
    if has_dpo:
        proven["sft_dpo"] = evidence.verify_trained(config_path, dpo_training, dpo_reloaded,
            kind="gemma-actions-v3-dpo", parent=proven["sft"]["identity"], dpo_preparation=dpo_preparation)
    else:
        validate_omission(read(dpo_omission), config, proven["sft"]["identity"])
    if native_runtime_adoption is not None:
        validate_native_adoption(read(native_runtime_adoption), parent=proven["sft"]["identity"])
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Use a fresh workspace metadata directory.")
    output.mkdir(parents=True)
    write(output / "inference-policy.json", inference_policy(config_path, native_runtime_adoption))
    meta = {"version": VERSION, "status": "verified_not_body_authorized", "body_access_authorized": False,
        "configuration": binding(config_path), "native_preparation": binding(Path(config_path).parent / "manifest.json"),
        "training_dataset_manifest": binding(evidence.RELEASE), "train_rows": 1261,
        "environment": proven["sft"]["environment"], "tokenizer": tokenizer_identity(config_path),
        "inference_policy": binding(output / "inference-policy.json"),
        "dpo_omission": binding(dpo_omission) if dpo_omission else None,
        "dpo_preparation": binding(Path(dpo_preparation) / "configuration.json") if has_dpo else None,
        "training_implementation_sha256": {role: item["training_source_sha256"] for role, item in proven.items()},
        "final_test_bodies_read": False}
    write(output / "preparation-metadata.json", meta)
    assemble(config_path, output / "preparation-metadata.json", output / "runtime")
    common = {"status": "verified", "runtime_family": FAMILY, "inference_configuration": binding(config_path),
        "preparation_manifest": binding(output / "preparation-metadata.json"),
        "execution_runtime": binding(output / "runtime/runtime-manifest.json"), "inference_policy": meta["inference_policy"]}
    models = {"native_base": {**common, "role": "native_base", "identity": native_identity(config),
        "artifact_manifest": binding(Path(sft_training) / "runtime.json"),
        "reload_receipt": proven["sft"]["reload"]}}
    for role, item in proven.items():
        models[role] = {**common, "role": role, "identity": item["identity"],
                       "artifact_manifest": item["training"], "reload_receipt": item["reload"]}
    if has_dpo:
        models["sft_dpo"]["parent_identity_sha256"] = digest(models["sft"]["identity"])
    verify_model_set(models, meta["tokenizer"], meta["training_dataset_manifest"], meta["dpo_omission"])
    for role, model in models.items():
        write(output / (role + ".json"), model)
    result = {"version": VERSION, "status": "model_receipts_ready_not_body_authorized", "body_access_authorized": False,
        "models": {role: binding(output / (role + ".json")) for role in models}, "dpo_omission": meta["dpo_omission"],
        "inference_policy": meta["inference_policy"], "preparation": binding(output / "preparation-metadata.json"),
        "final_test_bodies_read": False}
    write(output / "manifest.json", result)
    return result


def execution_overlay(spec, manifest_path=None):
    path = Path(manifest_path) if manifest_path else Path(__file__).resolve().parents[1] / "runtime-manifest.json"
    require(spec["runtime_manifest_sha256"] is not None and sha(path) == spec["runtime_manifest_sha256"],
            "Native expert settings require the frozen execution runtime manifest.")
    manifest = read(path)
    require(manifest["inference_policy_sha256"] == spec["inference_policy_sha256"]
            and manifest["configuration_sha256"] == spec["configuration_sha256"]
            and manifest["runtime_family"] == FAMILY, "Native expert runtime belongs to another frozen policy.")
    overlay = manifest.get("native_inference_overlay")
    if overlay is not None:
        require(set(overlay) == {"settings", "adoption_sha256", "prospective_registration_sha256"}
                and overlay["settings"] == native_overlay.OVERLAY
                and overlay["prospective_registration_sha256"] == NATIVE_PROSPECTIVE_SHA
                and isinstance(overlay["adoption_sha256"], str) and len(overlay["adoption_sha256"]) == 64,
                "Native execution overlay changed after adoption.")
    return overlay


def load_runtime(spec, config_path, training=None, reloaded=None, assets=None):
    require(assets is None, "Gemma uses the immutable preparation's native assets.")
    overlay = execution_overlay(spec)
    config, root, bundle, _, _, verified = evidence.sft.preflight(config_path)
    require(sha(config_path) == spec["configuration_sha256"] and config["generation"] == spec["generation"]
            and config["libraries"] == spec["libraries"] and verified["environment"]["matched"]
            and verified["preparation_sha256"] == spec["native_preparation_manifest_sha256"]
            and (spec["physical_context_tokens"], spec["inference_prefix_limit"], spec["output_tokens"]) == (8192, 7680, 512),
            "Gemma final native configuration, preparation or context changed.")
    identity = native_identity(config, training, reloaded, spec["adapter_version"])
    require(identity == spec["model_identity"] and bool(training) == (spec["run"]["role"] in {"sft", "sft_dpo"}),
            "Gemma final native model role changed.")
    torch, model, tokenizer, device, runtime = evidence.sft.gemma.load_runtime(config, root, bundle["tokens"]["json"])
    if training:
        model, _, _ = evidence.sft.load_adapter(model, Path(training) / "adapter",
            read(Path(training) / "adapter-tensors.json"), config, torch, False)
    if overlay is not None:
        native_overlay.apply(model, torch)
        runtime.update(native_inference_overlay=overlay, loading_text_experts_implementation=runtime["text_experts_implementation"],
                       text_experts_implementation="grouped_mm")
    return config, identity, torch, model, tokenizer, device, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config-path", "output", "sft-training", "sft-reloaded"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("dpo-training", "dpo-reloaded", "dpo-preparation", "dpo-omission", "native-runtime-adoption"):
        parser.add_argument("--" + name, type=Path)
    import json
    print(json.dumps(register(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
