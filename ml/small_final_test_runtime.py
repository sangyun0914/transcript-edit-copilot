"""Metadata and native runtime bridge for the existing sealed final-test gate.

Registration never freezes a selection or authorizes held-out body access.
The existing controller owns source allocation, independent labels and grading.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import shutil

import small_actions_final as data
import train_small_actions_final as sft

ROOT = Path(__file__).resolve().parents[1]
FAMILY = data.VERSION
VERSION = "small-final-test-model-metadata-1"
OMISSION_VERSION = "small-final-test-dpo-omission-1"
OMISSION_PLAN_SHA256 = "b012ba894078490c8283150eef6233f58ef3830e3405d671b28a3cd54c3930b6"
INFERENCE_POLICY_VERSION = "small-final-full-cohort-inference-policy-1"
CONTEXT_PLAN = ROOT / "docs/plans/final-small-model-test-context-2026-10-25.md"
CONTEXT_PLAN_SHA256 = "6a6895b6bd88070ebc4f77b99d7041510e4eb3d586db08a36005f814f208c402"
CONTEXT_AUDIT = ROOT / "artifacts/datasets/nep-passive-actions-final-v1/final-icl-budget-audit-001/audit.json"
CONTEXT_AUDIT_SHA256 = "42cf188b08e4765dbc1b5e8c5915b7538d5e15cd1b3e58cb7566563cfd835cd3"
DEMONSTRATION_IDS = ["v3-pilot-actions-v1-en-field-recorder-01", "v3-pilot-actions-v1-ko-library-recording-00"]
require, read, sha, digest = data.require, data.read, data.sha, data.value_hash


def binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "Small model metadata escaped the workspace.")
    return {"path": str(path), "sha256": sha(path)}


def bound(value):
    path = Path(value["path"]).resolve()
    require(binding(path) == value, "Bound small model artifact changed.")
    return path


def validate_configuration(config):
    require(config.get("version") == FAMILY and config.get("model") in data.PROFILES,
            "Unknown small-model inference family.")
    require(config["base"] == data.PROFILES[config["model"]]
            and config["contract_version"] == data.formats.contract.VERSION
            and config["format"] == "json" and config["final_test_bodies_read"] is False,
            "Small-model base, contract or preparation scope changed.")
    generation = config["generation"]
    require(config["context_tokens"] == 4096 and config["max_completion_tokens"] == 512
            and generation["max_new_tokens"] == 512 and generation["do_sample"] is False
            and generation["num_beams"] == 1 and generation["use_cache"] is True
            and generation["eos_token_id"] == config["base"]["terminal_ids"],
            "Small final-test bridge requires the original 4096/512 greedy SFT configuration.")


def tokenizer_identity(config):
    validate_configuration(config)
    assets = Path(config["assets_path"])
    require(data.verify_assets(assets, config["model"]) == config["assets"], "Small tokenizer receipt changed.")
    native = read(assets / "config.json")
    vocab_size = native.get("text_config", native)["vocab_size"]
    require(type(vocab_size) is int and vocab_size > max(config["generation"]["eos_token_id"]),
            "Invalid native vocabulary metadata.")
    return {"runtime_family": FAMILY, "model": config["model"], "path": str(assets.resolve()),
            "eos_token_id": config["base"]["terminal_ids"][0],
            "terminal_ids": config["generation"]["eos_token_id"], "vocab_size": vocab_size,
            "files_sha256": config["assets"]["files_sha256"], "assets": config["assets"]}


def validate_tokenizer_identity(identity):
    require(identity.get("runtime_family") == FAMILY and identity.get("model") in data.PROFILES,
            "Unknown small tokenizer family.")
    receipt = data.verify_assets(identity["path"], identity["model"])
    native = read(Path(identity["path"]) / "config.json")
    require(identity["assets"] == receipt and identity["files_sha256"] == receipt["files_sha256"]
            and identity["eos_token_id"] == data.PROFILES[identity["model"]]["terminal_ids"][0]
            and identity["terminal_ids"] == data.PROFILES[identity["model"]]["terminal_ids"]
            and identity["vocab_size"] == native.get("text_config", native)["vocab_size"],
            "Small tokenizer native identity differs from its pin.")


def validate_inference_policy(policy, config=None, *, resolve=bound):
    require(policy.get("version") == INFERENCE_POLICY_VERSION and policy.get("status") == "registered_before_test"
            and policy["model"] in data.PROFILES and policy["canonical_window_input_tokens"] == 3584
            and policy["physical_context_tokens"] == 8192 and policy["inference_prefix_limit"] == 7680
            and policy["output_tokens"] == 512 and policy["training_context_tokens"] == 4096
            and policy["application_context_tokens"] == 4096 and policy["all_roles_same_case_ids"] is True
            and policy["overflow_policy"] == "fail_whole_preparation"
            and policy["fixed_demonstration_ids"] == DEMONSTRATION_IDS
            and policy["prospective_plan"]["sha256"] == CONTEXT_PLAN_SHA256
            and policy["exposed_development_audit"]["sha256"] == CONTEXT_AUDIT_SHA256
            and policy["body_access_authorized"] is False and policy["final_test_bodies_read"] is False,
            "Unregistered small full-cohort inference context policy.")
    resolve(policy["prospective_plan"])
    audit = read(resolve(policy["exposed_development_audit"]))
    assets = read(resolve(policy["assets_receipt"]))
    observed = audit["models"][policy["model"]]
    require(audit["version"] == "small-final-full-cohort-icl-budget-audit-1"
            and audit["scope"] == "exposed_development_only" and audit["model_weights_loaded"] is False
            and audit["final_test_bodies_read"] is False and audit["fixed_demonstration_ids"] == DEMONSTRATION_IDS
            and observed["assets"]["sha256"] == policy["assets_receipt"]["sha256"]
            and observed["cases"] == 471 and observed["over_budget"] == 0
            and observed["canonical_max"] <= 3584 and observed["icl_max"] <= 7680
            and observed["native_config_max_position_embeddings"] >= 8192,
            "Small ICL context budget lacks its intact exposed native-tokenizer audit.")
    if config is not None:
        validate_configuration(config)
        require(policy["model"] == config["model"] and assets == config["assets"],
                "Inference policy tokenizer differs from the trained small model.")
    return policy


def inference_policy(config):
    policy = {"version": INFERENCE_POLICY_VERSION, "status": "registered_before_test", "model": config["model"],
              "canonical_window_input_tokens": 3584, "physical_context_tokens": 8192, "inference_prefix_limit": 7680,
              "output_tokens": 512, "training_context_tokens": 4096, "application_context_tokens": 4096,
              "all_roles_same_case_ids": True, "overflow_policy": "fail_whole_preparation",
              "fixed_demonstration_ids": DEMONSTRATION_IDS, "prospective_plan": binding(CONTEXT_PLAN),
              "exposed_development_audit": binding(CONTEXT_AUDIT),
              "assets_receipt": binding(Path(config["assets_path"]) / "assets.json"),
              "body_access_authorized": False, "final_test_bodies_read": False}
    return validate_inference_policy(policy, config)


def native_identity(config, training=None, reloaded=None):
    validate_configuration(config)
    require(bool(training) == bool(reloaded), "Small adapters require training and independent reload.")
    identity = {"runtime_family": FAMILY, "kind": "native_base", "model": config["model"],
                "repository": config["base"]["repository"], "revision": config["base"]["revision"],
                "assets_sha256": digest(config["assets"])}
    if training:
        require(read(Path(training) / "configuration.json") == config, "Adapter configuration differs from frozen inference.")
        identity.update(sft.model_identity(training, reloaded))
    return identity


def verify_training_evidence(training, reloaded, config, kind):
    training, reloaded = Path(training), Path(reloaded)
    info, reload_info = read(training / "run.json"), sft.verified_reload(training, reloaded)
    require(read(training / "configuration.json") == config and info.get("kind", "sft") == kind
            and info["status"] == "training_complete_reload_pending" and info["final_test_bodies_read"] is False
            and reload_info["final_test_bodies_read"] is False and reload_info["pid"] != info["pid"],
            "Small model lacks a completed run and independent pre-test reload.")
    implementation = info["implementation_sha256"]
    require(implementation and all(Path(name).name == name and sha(training / "implementation" / name) == expected
                                   for name, expected in implementation.items()), "Training implementation evidence changed.")
    if kind == "sft":
        draws = len(config["training"]["sample_order"])
        require(info["draws"] == info["draws_completed"] == draws
                and info["updates_completed"] == math.ceil(draws / config["training"]["gradient_accumulation"]),
                "SFT did not complete its declared optimizer budget.")
    else:
        from train_small_actions_final_dpo import verify_training
        verify_training(training)
    runtime = read(training / "runtime.json")
    require(runtime["base"] == config["base"] and runtime["dtype"] == "bfloat16"
            and runtime["environment"] == reload_info["runtime"]["environment"],
            "Independent small-model native reload environment changed.")
    return {"identity": native_identity(config, training, reloaded), "environment": runtime["environment"],
            "implementation_sha256": implementation}


def adapter_parent(identity):
    require(identity["kind"] == "sft", "DPO omission requires the actual selected SFT parent.")
    return {key: identity[key] for key in ("kind", "configuration_sha256", "adapter_sha256", "reload_run_sha256")}


def validate_omission(value, config, parent, *, resolve=bound):
    require(value.get("version") == OMISSION_VERSION and value.get("status") == "registered_before_test"
            and value["body_access_authorized"] is False and value["final_test_bodies_read"] is False
            and value["policy"] == {"minimum_pairs": 8, "minimum_preservation_pairs": 1}
            and value["parent"] == parent and bool(value["reason"].strip())
            and value["plan"]["sha256"] == OMISSION_PLAN_SHA256,
            "DPO omission is not bound to the prospective policy and exact SFT parent.")
    resolve(value["plan"])
    mining_path, candidate_path = resolve(value["mining"]), resolve(value["candidate_run"])
    report, candidate = read(mining_path), read(candidate_path)
    require(report["version"] == "small-actions-final-dpo-v1" and report["status"] == "train_errors_mined"
            and report["parent"] == parent and report["final_test_bodies_read"] is False
            and report["source_cases_sha256"] == config["source_cases_sha256"]
            and report["candidate_run_sha256"] == value["candidate_run"]["sha256"],
            "DPO omission mining belongs to another model, dataset or candidate run.")
    require(set(report["files_sha256"]) == {"pairs.jsonl.gz", "decisions.jsonl", "review-queue.jsonl", "configuration.json"},
            "Incomplete DPO omission mining inventory.")
    for name, expected in report["files_sha256"].items():
        require(sha(mining_path.parent / name) == expected, "DPO omission mining payload changed.")
    require(read(mining_path.parent / "configuration.json") == config
            and sha(mining_path.parent / "configuration.json") == parent["configuration_sha256"]
            and candidate["status"] == "inference_complete" and candidate["kind"] == "sft"
            and candidate["splits"] == ["train"] and candidate["model_identity"] == parent
            and candidate["final_test_bodies_read"] is False
            and candidate["configuration_sha256"] == parent["configuration_sha256"],
            "DPO omission requires an intact full-TRAIN run of the selected SFT.")
    observations_path = resolve(value["candidate_observations"])
    require(observations_path == candidate_path.parent / "observations.jsonl"
            and candidate["observations_sha256"] == value["candidate_observations"]["sha256"],
            "DPO omission candidate observations changed.")
    decisions = data.rows(mining_path.parent / "decisions.jsonl")
    pairs = data.rows(mining_path.parent / "pairs.jsonl.gz")
    observations = data.rows(observations_path)
    requested = len(set(config["training"]["sample_order"]))
    ids = {row["case_id"] for row in decisions}
    require(len(decisions) == len(ids) == len(observations) == candidate["requested"] == report["requested"] == requested
            and ids == {row["case_id"] for row in observations}
            and dict(Counter(row["status"] for row in decisions)) == report["decisions"]
            and len(pairs) == report["pairs"] == sum(row["status"] == "preferred" for row in decisions)
            and len({row["pair_id"] for row in pairs}) == len(pairs)
            and {row["case_id"] for row in pairs} == {row["case_id"] for row in decisions if row["status"] == "preferred"}
            and all(row["split"] == row["source"]["split"] == "train"
                    and row["judgment"]["status"] == "preferred"
                    and row["chosen"] in row["accepted_predictions"]
                    and row["chosen_action"] == row["chosen"]["action"] for row in pairs),
            "DPO omission lost the actual TRAIN/pair census.")
    counts = {"requested": requested, "strict_pairs": len(pairs),
              "preservation_pairs": sum(row["chosen_action"] == "no_action" for row in pairs),
              "unresolved_semantic_reviews": report["decisions"].get("needs_review", 0)}
    require(counts["unresolved_semantic_reviews"] == 0 and not data.rows(mining_path.parent / "review-queue.jsonl"),
            "Unresolved semantic candidates cannot be treated as confirmed DPO pair scarcity.")
    reasons = (["fewer_than_eight_strict_pairs"] if counts["strict_pairs"] < 8 else [])
    reasons += ["no_actual_preservation_error_pair"] if counts["preservation_pairs"] == 0 else []
    require(reasons and value["reason_codes"] == reasons and value["counts"] == counts
            and report["preservation_pairs"] == counts["preservation_pairs"],
            "DPO omission reason/counts do not satisfy the registered skip rule.")
    return value


def omission_payload(mining, candidates, plan, config, parent):
    report = read(Path(mining) / "run.json")
    counts = {"requested": report["requested"], "strict_pairs": report["pairs"],
              "preservation_pairs": report["preservation_pairs"],
              "unresolved_semantic_reviews": report["decisions"].get("needs_review", 0)}
    reasons = (["fewer_than_eight_strict_pairs"] if counts["strict_pairs"] < 8 else [])
    reasons += ["no_actual_preservation_error_pair"] if counts["preservation_pairs"] == 0 else []
    value = {"version": OMISSION_VERSION, "status": "registered_before_test", "parent": parent,
             "policy": {"minimum_pairs": 8, "minimum_preservation_pairs": 1}, "counts": counts,
             "reason_codes": reasons,
             "reason": "The prospective DPO execution minimum was not met by actual selected-SFT TRAIN errors. No DPO adapter is substituted or claimed.",
             "plan": binding(plan), "mining": binding(Path(mining) / "run.json"),
             "candidate_run": binding(Path(candidates) / "run.json"),
             "candidate_observations": binding(Path(candidates) / "observations.jsonl"),
             "body_access_authorized": False, "final_test_bodies_read": False}
    return validate_omission(value, config, parent)


def validate_preparation(path, config_path):
    value = read(path)
    require(value.get("version") == VERSION and value.get("status") == "verified_not_body_authorized"
            and value["final_test_bodies_read"] is False and value["body_access_authorized"] is False,
            "Unknown small-model metadata preparation.")
    require(bound(value["configuration"]) == Path(config_path).resolve(), "Small preparation belongs to another configuration.")
    config = read(config_path)
    validate_configuration(config)
    dataset = read(bound(value["training_dataset_manifest"]))
    require(config["source_cases_sha256"] == dataset["files_sha256"]["cases/train.jsonl"]
            and value["train_rows"] == dataset["counts"]["train"]
            and value["tokenizer"] == tokenizer_identity(config), "Small model training release/tokenizer changed.")
    validate_inference_policy(read(bound(value["inference_policy"])), config)
    return value, config


def verify_model_set(models, tokenizer, dataset_manifest, omission=None, policy_binding=None):
    roles = {"native_base", "sft"} if omission is not None else {"native_base", "sft", "sft_dpo"}
    require(set(models) == roles
            and all(model.get("runtime_family") == FAMILY for model in models.values()),
            "Small final comparison must freeze one coherent base/SFT/same-parent DPO family.")
    base = models["native_base"]
    common = {key: base[key] for key in ("inference_configuration", "preparation_manifest", "execution_runtime", "inference_policy")}
    require(all(all(model[key] == value for key, value in common.items()) for model in models.values()),
            "Small comparison changed its common preparation or runtime.")
    meta, config = validate_preparation(bound(common["preparation_manifest"]), bound(common["inference_configuration"]))
    require(meta["training_dataset_manifest"] == dataset_manifest and meta["tokenizer"] == tokenizer,
            "Small comparison differs from the frozen dataset/tokenizer policy.")
    require(meta.get("dpo_omission") == omission, "Small model metadata and final-freeze DPO omission differ.")
    require(meta["inference_policy"] == common["inference_policy"]
            and (policy_binding is None or policy_binding == common["inference_policy"]), "Frozen inference policy differs.")
    evidence = {}
    for role, kind in (("sft", "sft"), ("sft_dpo", "dpo")):
        if role not in models:
            continue
        training, reloaded = bound(models[role]["artifact_manifest"]).parent, bound(models[role]["reload_receipt"]).parent
        evidence[role] = verify_training_evidence(training, reloaded, config, kind)
        require(evidence[role]["identity"] == models[role]["identity"]
                and evidence[role]["environment"] == meta["environment"]
                and evidence[role]["implementation_sha256"] == meta["training_implementation_sha256"][role],
                "Frozen small model receipt changed.")
    parent = sft.model_identity(bound(models["sft"]["artifact_manifest"]).parent,
                                bound(models["sft"]["reload_receipt"]).parent)
    if omission is None:
        dpo = read(bound(models["sft_dpo"]["artifact_manifest"]).parent / "dpo_configuration.json")
        require(dpo["parent"] == parent and models["sft_dpo"]["parent_identity_sha256"] == digest(models["sft"]["identity"]),
                "Small DPO is not initialized from the exact frozen SFT parent.")
    else:
        validate_omission(read(bound(omission)), config, parent)
    require(base["identity"] == native_identity(config), "Frozen small native base identity changed.")
    require(bound(base["artifact_manifest"]) == bound(models["sft"]["artifact_manifest"]).parent / "runtime.json"
            and base["reload_receipt"] == models["sft"]["reload_receipt"],
            "Small native base must retain its actual SFT loading and independent reload evidence.")
    return meta


def assemble(config_path, metadata_path, output):
    import assemble_task_test_actions_v3 as assembly
    import prepare_passive_actions_final_test as sealed

    metadata, _ = validate_preparation(metadata_path, config_path)
    output = data.new_directory(output)
    sources = sealed.source_closure([ROOT / assembly.ENTRY, Path(assembly.__file__), Path(__file__)])
    sources[str(data.formats.contract.PROMPT_PATH.relative_to(ROOT))] = sha(data.formats.contract.PROMPT_PATH)
    for name in sorted(sources):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    manifest = {"version": assembly.VERSION, "runtime_family": FAMILY,
                "status": "assembled_model_specific_runtime", "scope": "final_test", "entrypoint": assembly.ENTRY,
                "configuration_sha256": sha(config_path), "preparation_manifest_sha256": sha(metadata_path),
                "inference_policy_sha256": metadata["inference_policy"]["sha256"],
                "configuration_sources_sha256": {}, "source_sha256": dict(sorted(sources.items())),
                "source_origin": {name: "final_only_overlay" for name in sources},
                "source_policy": "explicit_frozen_inference_runtime_with_separate_training_source_provenance",
                "training_implementation_sha256": metadata["training_implementation_sha256"],
                "dpo_omission_sha256": metadata["dpo_omission"]["sha256"] if metadata.get("dpo_omission") else None,
                "final_test_bodies_read": False, "model_weights_read": False}
    data.write(output / assembly.MANIFEST, manifest)
    return assembly.load_runtime(output)


def register(preparation, dataset_manifest, sft_training, sft_reloaded, dpo_training=None, dpo_reloaded=None, output=None,
             *, dpo_mining=None, dpo_candidates=None, dpo_plan=None):
    config, train_rows, _ = data.prepared(preparation)
    validate_configuration(config)
    dataset = read(dataset_manifest)
    require(dataset.get("test_bodies_read") is False and config["source_cases_sha256"] == dataset["files_sha256"]["cases/train.jsonl"]
            and len(train_rows) == dataset["counts"]["train"], "Use the selected complete pre-test TRAIN release.")
    has_dpo = dpo_training is not None and dpo_reloaded is not None
    has_omission = all(value is not None for value in (dpo_mining, dpo_candidates, dpo_plan))
    require(output is not None and bool(dpo_training) == bool(dpo_reloaded) and has_dpo != has_omission
            and (not has_dpo or all(value is None for value in (dpo_mining, dpo_candidates, dpo_plan))),
            "Supply completed DPO and reload, or explicit mining/candidates/plan evidence for its omission.")
    proven = {"sft": verify_training_evidence(sft_training, sft_reloaded, config, "sft")}
    omission = None
    if has_dpo:
        proven["sft_dpo"] = verify_training_evidence(dpo_training, dpo_reloaded, config, "dpo")
        require(proven["sft"]["environment"] == proven["sft_dpo"]["environment"], "SFT/DPO environments differ.")
        require(read(Path(dpo_training) / "dpo_configuration.json")["parent"] == sft.model_identity(sft_training, sft_reloaded),
                "Small DPO must use the exact selected SFT parent.")
    else:
        omission = omission_payload(dpo_mining, dpo_candidates, dpo_plan, config, sft.model_identity(sft_training, sft_reloaded))
    output = data.new_directory(output)
    config_path = Path(preparation) / "configuration.json"
    meta_path = output / "preparation-metadata.json"
    data.write(output / "inference-policy.json", inference_policy(config))
    if omission is not None:
        data.write(output / "dpo-omission.json", omission)
    metadata = {"version": VERSION, "status": "verified_not_body_authorized", "body_access_authorized": False,
                "configuration": binding(config_path), "training_dataset_manifest": binding(dataset_manifest),
                "train_rows": len(train_rows), "environment": proven["sft"]["environment"],
                "tokenizer": tokenizer_identity(config), "final_test_bodies_read": False,
                "inference_policy": binding(output / "inference-policy.json"),
                "dpo_omission": binding(output / "dpo-omission.json") if omission is not None else None,
                "training_implementation_sha256": {role: value["implementation_sha256"] for role, value in proven.items()}}
    data.write(meta_path, metadata)
    assemble(config_path, meta_path, output / "runtime")
    common = {"status": "verified", "runtime_family": FAMILY, "inference_configuration": binding(config_path),
              "preparation_manifest": binding(meta_path), "execution_runtime": binding(output / "runtime/runtime-manifest.json"),
              "inference_policy": metadata["inference_policy"]}
    models = {"native_base": {**common, "role": "native_base", "identity": native_identity(config),
                              "artifact_manifest": binding(Path(sft_training) / "runtime.json"),
                              "reload_receipt": binding(Path(sft_reloaded) / "run.json")}}
    for role, training, reloaded in (("sft", sft_training, sft_reloaded), ("sft_dpo", dpo_training, dpo_reloaded)):
        if role not in proven:
            continue
        models[role] = {**common, "role": role, "identity": proven[role]["identity"],
                        "artifact_manifest": binding(Path(training) / "run.json"), "reload_receipt": binding(Path(reloaded) / "run.json")}
    if has_dpo:
        models["sft_dpo"]["parent_identity_sha256"] = digest(models["sft"]["identity"])
    verify_model_set(models, metadata["tokenizer"], metadata["training_dataset_manifest"], metadata["dpo_omission"])
    for role, model in models.items():
        data.write(output / (role + ".json"), model)
    manifest = {"version": VERSION, "status": "model_receipts_ready_not_body_authorized", "body_access_authorized": False,
                "models": {role: binding(output / (role + ".json")) for role in models},
                "dpo_omission": metadata["dpo_omission"],
                "preparation": binding(meta_path), "final_test_bodies_read": False}
    manifest["inference_policy"] = metadata["inference_policy"]
    data.write(output / "manifest.json", manifest)
    return manifest


def load_runtime(spec, config_path, training=None, reloaded=None, assets=None):
    config, _, assets = data.prepared(Path(config_path).parent, assets)
    validate_configuration(config)
    require(sha(config_path) == spec["configuration_sha256"] and config["generation"] == spec["generation"]
            and data.environment() == spec["libraries"] and spec["physical_context_tokens"] == 8192
            and spec["inference_prefix_limit"] == 7680 and spec["output_tokens"] == 512,
            "Small final runtime configuration, context policy or library environment changed.")
    identity = native_identity(config, training, reloaded)
    require(identity == spec["model_identity"] and bool(training) == (spec["run"]["role"] in {"sft", "sft_dpo"}),
            "Small final runtime loaded a different model/adapter role.")
    torch, model, tokenizer, runtime = sft.load_base(config, assets)
    if training:
        model = sft.load_adapter(model, training, config, torch)
    require(runtime["environment"] == spec["libraries"], "Actual native environment differs from frozen small model.")
    return config, identity, torch, model, tokenizer, "cuda", runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "dataset-manifest", "sft-training", "sft-reloaded", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("dpo-training", "dpo-reloaded", "dpo-mining", "dpo-candidates", "dpo-plan"):
        parser.add_argument("--" + name, type=Path)
    args = vars(parser.parse_args())
    print(data.compact(register(**args)))


if __name__ == "__main__":
    main()
