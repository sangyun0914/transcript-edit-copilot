"""Prepare small B003 final-gate inputs, never a freeze or body-access receipt.

Only explicit JSON metadata (at most 2 MiB per file) is read. Model tensors,
tokenizers, TRAIN payloads, reserved bodies and recursive bindings are not read.
The output deliberately remains not_ready, including when every input is present.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "b003-final-gate-inputs-1"
LIMIT = 2 * 1024 * 1024
TRAINING = ROOT / "artifacts/training/gemma-actions-v3"
PREPARATION = TRAINING / "substantive-preparation-attempt-003-json-targeted"
SFT = TRAINING / "substantive-json-targeted-attempt-001"
NATIVE = ROOT / "artifacts/evaluations/passive-actions-v3/native-targeted-json-attempt-001"
RUNTIME = ROOT / "artifacts/evaluation/passive-actions-v3/final-runtime-targeted-json-attempt-001"
CONTINGENT = ROOT / "artifacts/evaluation/passive-actions-v3/holdout-metadata-release003-contingent-attempt-001"
EXPERIMENT = ROOT / "artifacts/experiments/passive-actions-v3"
ROSTER = ROOT / "artifacts/ui/passive-actions-v3/registered-roster-001/manifest.json"
ROSTER_SHA = "55b4e22f2c27c12ce59baa7f3e21e973b5b3e4cf6f26124f225eacf1c9da9663"
SFT_SHA = {
    "configuration_sha256": "9b73580bde6d7a57a8b6985af8584e9b29f0e07b60e979e44d0f54ff3f58b468",
    "training_manifest_sha256": "df8087947afe4ae1192c5f84aaba8824ae485fda593a6b0baae7639d22ca1b7a",
    "reload_manifest_sha256": "a2a72cd68263b6352d4c22fb03f6b6388b9e69c11d86998735d3937e38290c98",
    "adapter_sha256": "6d982ef7cab6a56b7acf85032ed6916681c3245891c3346b440bfd565ea42fcb",
}
ROLES = ("native_base", "base_icl", "sft", "sft_dpo", "always_no_action", "consistent_history_rule")
ATTESTATIONS = (
    "exact_reload_and_runtime_evidence_reviewed", "actual_selected_model_ui_reviewed",
    "local_apply_undo_export_and_durability_reviewed", "all_UI_planned_and_issued_outcomes_retained",
    "scripted_checks_separate_from_actual_model_quality", "quality_failures_retained_without_waiver",
    "reserved_bodies_unopened", "actor_access_limits_reviewed",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata(path, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.suffix == ".json", "Use workspace JSON metadata only.")
    require(path.is_file() and path.stat().st_size <= LIMIT, f"Missing or oversized metadata: {path}")
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    require(len(raw) <= LIMIT, f"Metadata exceeded the 2 MiB limit: {path}")
    sha = hashlib.sha256(raw).hexdigest()
    require(expected is None or sha == expected, f"Metadata binding changed: {path}")
    value = json.loads(raw)
    require(isinstance(value, dict), f"Expected a metadata object: {path}")
    return value, {"path": str(path), "sha256": sha}


def identity_digest(value):
    # Match prepare_passive_actions_v3.digest, including insertion order.
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def recorded_run(config_path, training, reloaded, version):
    config, config_binding = metadata(config_path)
    require(config["version"] == version and config["format"] == "json", "Run configuration version/format differs.")
    train, train_binding = metadata(Path(training) / "manifest.json")
    reload, reload_binding = metadata(Path(reloaded) / "manifest.json")
    statuses = ("training_complete_fresh_reload_pending", "fresh_reload_passed_local_export_pending")
    records = []
    for directory, manifest, status in zip((training, reloaded), (train, reload), statuses):
        require(manifest["version"] == version and manifest["status"] == status, "Unexpected run version/status.")
        require(manifest["files_sha256"]["configuration.json"] == config_binding["sha256"], "Manifest configuration differs.")
        info, _ = metadata(Path(directory) / "run.json", manifest["files_sha256"]["run.json"])
        metadata(Path(directory) / "configuration.json", config_binding["sha256"])
        require(info["version"] == version and info["status"] == status
                and info["configuration_sha256"] == config_binding["sha256"]
                and info["final_test_bodies_read"] is False
                and info["native_assets_verified"] is True
                and info["environment"]["matched"] is True
                and info["environment"]["differences"] == {}, "Run metadata is not a matched, completed native run.")
        records.append(info)
    before, after = records
    parity, _ = metadata(Path(reloaded) / "reload-parity.json", reload["files_sha256"]["reload-parity.json"])
    require(before["mode"] == "train" and after["mode"] == "fresh_reload"
            and before["pid"] != after["pid"]
            and after["previous_manifest_sha256"] == train_binding["sha256"]
            and parity["adapter_tensors_identical"] is True
            and parity["cases"] and all(row["identical"] is True for row in parity["cases"]),
            "Independent fresh-reload metadata is missing or inconsistent.")
    require(before["training_updates"] == after["reloaded_training_updates"] == config["training"]["max_updates"],
            "Training stopped before its registered update budget.")
    identity = {
        "kind": version, "repository": config["base"]["repository"], "revision": config["base"]["revision"],
        "native_pin_sha256": config["native_pin_sha256"], "configuration_sha256": config_binding["sha256"],
        "training_manifest_sha256": train_binding["sha256"], "reload_manifest_sha256": reload_binding["sha256"],
        "adapter_sha256": train["files_sha256"]["adapter/adapter_model.safetensors"],
    }
    return config, identity, {
        "version": "task-test-actions-v3-model-receipt-draft-1", "status": "draft_metadata_only_not_verified",
        "identity": identity, "training_configuration": config_binding,
        "artifact_manifest": train_binding, "reload_receipt": reload_binding,
        "payload_hashes_recomputed": False,
    }, records


def export_binding(path, identity):
    if path is None:
        return None
    value, reference = metadata(path)
    require(value["version"] == "gemma-actions-v3-lora-export"
            and value["status"] == "adapter_conversion_verified_serving_pending"
            and value["all_tensors_exact"] is True and value["adapter_tensor_count"] == 410
            and value["output_format"] == "json"
            and value["contract_version"] == "passive-nep-actions-v1"
            and value["format_version"] == "passive-formats-actions-v3"
            and value["runtime_revision"] == "427291b5b34cd914a31b3fd3b61a68f6184f4b9f",
            "Require the exact JSON v3 export receipt.")
    for key in ("adapter_sha256", "training_manifest_sha256", "reload_manifest_sha256", "native_pin_sha256"):
        require(value[key] == identity[key], f"Export belongs to another native model: {key}")
    require(value["base_repository"] == identity["repository"] and value["base_revision"] == identity["revision"],
            "Export base differs from the native model.")
    return reference


def local_evidence(deployment_path, ui_path, selected, selected_export):
    require(ui_path is None or deployment_path is not None, "UI metadata needs its explicit deployment.")
    if deployment_path is None:
        return None, None
    require(selected is not None and selected_export is not None, "Bind the selected native model/export before UI evidence.")
    deployment, deployment_binding = metadata(Path(deployment_path) / "deployment.json")
    require(deployment["version"] == "local-gemma-actions-v3-lora"
            and deployment["historical_tooling_only"] is False, "Historical tooling is not a selected v3 deployment.")
    require(deployment["adapter_export_receipt"]["sha256"] == selected_export["sha256"], "Deployment export differs.")
    exported, _ = metadata(selected_export["path"], selected_export["sha256"])
    gateway, gateway_binding = metadata(Path(deployment_path) / "gateway.json", deployment["files_sha256"]["gateway.json"])
    require(gateway["version"] == 2 and gateway["output_format"] == "json"
            and gateway["capture_runtime_evidence"] is False and gateway["adapter_scale"] == 1.0
            and gateway["adapter_sha256"] == exported["gguf_sha256"], "Selected serving settings differ.")
    if ui_path is None:
        return deployment_binding, None
    ui, ui_binding = metadata(ui_path)
    identity = ui["identity"]
    require(ui["kind"] == "actual_model_registered_ui_roster" and ui["model_quality_evaluated"] is False
            and ui["roster_manifest_sha256"] == ROSTER_SHA and ui["planned"] == 23
            and ui["finished_cases"] == len(ui["results"]) == 23 and ui["unfinished_cases"] == 0,
            "Require all 23 actual-model roster outcomes, including failures.")
    require(identity["status"] == 200 and identity["body"]["deployment_sha256"] == deployment_binding["sha256"]
            and identity["body"]["configuration_sha256"] == gateway_binding["sha256"]
            and identity["body"]["runtime_evidence_enabled"] is False, "UI ran a different model deployment.")
    return deployment_binding, ui_binding


def prepare(output, selected_role="sft", dpo_config=None, dpo_training=None, dpo_reloaded=None,
            dpo_export_receipt=None, selected_deployment=None, selected_ui_receipt=None):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Use a fresh workspace output directory.")
    require(selected_role in {"sft", "sft_dpo"}, "Selected role must be sft or sft_dpo.")
    supplied = [value is not None for value in (dpo_config, dpo_training, dpo_reloaded)]
    require(all(supplied) or not any(supplied), "Supply DPO config, training and reloaded together, or omit all three.")
    require(dpo_export_receipt is None or all(supplied), "A DPO export requires its actual DPO config/train/reload.")
    config, identity, sft, _ = recorded_run(PREPARATION / "config-json.json", SFT / "train", SFT / "reload", "gemma-actions-v3-sft")
    require(all(identity[key] == value for key, value in SFT_SHA.items()), "The SFT parent is not B003.")
    native, native_binding = metadata(NATIVE / "run.json")
    require(native["status"] == "inference_complete" and native["model_identity"] == identity,
            "B003 native DEV identity differs.")
    _, prep_binding = metadata(PREPARATION / "manifest.json", "edd651d7d9c3997da869cccd4c4aab265e9692dc12dd467c92395c2db8e344c1")
    runtime, runtime_binding = metadata(RUNTIME / "runtime-manifest.json")
    require(runtime["configuration_sha256"] == identity["configuration_sha256"]
            and runtime["preparation_manifest_sha256"] == prep_binding["sha256"]
            and runtime["final_test_bodies_read"] is False, "B003 inference assembly metadata differs.")
    inference = {"inference_configuration": sft["training_configuration"],
                 "preparation_manifest": prep_binding, "execution_runtime": runtime_binding}
    sft.update(role="sft", **inference, native_development_run=native_binding)
    sft_export = export_binding(TRAINING / "substantive-json-targeted-export-attempt-001/adapter-export/adapter-receipt.json", identity)
    dpo, dpo_export = None, None
    if all(supplied):
        dpo_config_value, dpo_identity, dpo, records = recorded_run(dpo_config, dpo_training, dpo_reloaded, "gemma-actions-v3-dpo")
        require(dpo_config_value["phase"] == "substantive"
                and dpo_config_value["anchor"] == {"anchor_" + key: value for key, value in SFT_SHA.items()}
                and dpo_config_value["sft_preparation_sha256"] == prep_binding["sha256"],
                "Require a substantive standard DPO run anchored to B003, not the historical B002 pair.")
        metadata(Path(dpo_config).parent / dpo_config_value["sft_config_path"], identity["configuration_sha256"])
        for field in ("base", "model", "tokenizer", "native_pin_sha256", "format", "contract_version", "format_version", "generation", "libraries", "absent_libraries"):
            require(dpo_config_value[field] == config[field], "DPO/native inference contract differs: " + field)
        before, after = records
        require(before["initialization"] == "frozen_sft_adapter_fresh_optimizer"
                and before["optimizer_committed_rows"] == dpo_config_value["training"]["pair_count"]
                and before["reference_manifest_sha256"] == after["reference_manifest_sha256"],
                "DPO did not commit every pair against the same reference cache.")
        dpo.update(role="sft_dpo", parent_identity_sha256=identity_digest(identity), **inference)
        dpo_export = export_binding(dpo_export_receipt, dpo_identity)
    selected = sft if selected_role == "sft" else dpo
    selected_export = sft_export if selected_role == "sft" else dpo_export
    deployment, ui = local_evidence(selected_deployment, selected_ui_receipt, selected, selected_export)
    contingent, contingent_binding = metadata(CONTINGENT / "manifest.json")
    require(contingent["status"] == "contingent_not_selected_not_body_authorized"
            and contingent["body_access_authorized"] is False
            and contingent["reserved_final_test_bodies_read"] is False, "Contingent metadata authority changed.")
    _, roster_binding = metadata(ROSTER, ROSTER_SHA)
    missing = [key for key, value in {
        "B003_DPO_completed_native_run": dpo, "selected_export_receipt": selected_export,
        "actual_selected_model_deployment": deployment, "actual_selected_model_23_case_UI": ui,
    }.items() if value is None]
    missing += ["versioned_B003_release003_policy_and_controller_registration",
                "verified_model_receipts_and_payload_runtime_checks", "frozen_DEV_selection_receipt",
                "actual_actor_registry_and_access_evidence", "exact_parent_readiness_attestations",
                "independent_parent_gate_readback_then_separate_body_access_instruction"]
    result = {
        "version": VERSION, "status": "not_ready", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "small_metadata_draft_only", "body_access_authorized": False,
        "reserved_final_test_bodies_read": False, "app_activation_performed": False,
        "selected_role_proposal": selected_role,
        "selected_identity_sha256_proposal": identity_digest(selected["identity"]) if selected else None,
        "model_receipt_drafts": {"sft": sft, "sft_dpo": dpo},
        "existing_native_base_receipt": metadata(EXPERIMENT / "final-model-receipts-001/native-base.json")[1],
        "exports": {"sft": sft_export, "sft_dpo": dpo_export},
        "selected_deployment_receipt": deployment, "selected_model_ui_receipt": ui,
        "existing_B003_development_deployment_not_UI_ready": metadata(
            ROOT / "artifacts/local-runtime/gemma-actions-v3/substantive-json-targeted-deployment-attempt-001/deployment.json")[1],
        "roster": roster_binding, "release003_contingent_bundle": contingent_binding,
        "contingent_inputs_not_promoted": {key: contingent[key] for key in (
            "metadata_candidate", "source_resolution_candidate", "icl_candidate", "verification", "exposed_text_index")},
        "run_layout_proposal": [{"id": role + "-json", "role": role, "format": "json"} for role in ROLES],
        "historical_comparison_only": "B002 and B002-DPO remain historical DEV evidence.",
        "readiness_placeholders": {
            "selection_receipt": None, "native_environment_receipt": None, "native_runtime_receipt": None,
            "new_builder": None, "new_draft_revision": None, "new_controller": None,
            "actors": {key: None for key in ("preparer", "author", "reviewer", "grader")},
            "author_reviewer_grader_must_be_pairwise_distinct": True,
            "evidence_review": {"attestations": dict.fromkeys(ATTESTATIONS, False)},
            "assemble_final_metadata": False, "acquire_only_after_independent_parent_readback": False,
        },
        "not_ready_errors": missing,
        "limits": ["Declared tensor hashes were not recomputed; all model receipts remain drafts.",
                   "Execution-runtime source closure and nested artifact bindings were not recursively verified.",
                   "UI metadata binding does not certify UI quality, latency, durability or acoustic correctness.",
                   "Training configuration identifies DPO optimization; inference always uses the B003 SFT assembly.",
                   "This is not an input accepted by the historical B002 freeze builder or load_gate."],
    }
    output.mkdir(parents=True)
    with (output / "inputs.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-role", choices=("sft", "sft_dpo"), default="sft")
    for name in ("dpo-config", "dpo-training", "dpo-reloaded", "dpo-export-receipt", "selected-deployment", "selected-ui-receipt"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--require-ready", action="store_true", help="Exit 2 after writing the draft; this helper never grants readiness.")
    args = vars(parser.parse_args())
    require_ready = args.pop("require_ready")
    result = prepare(**args)
    print(json.dumps({"status": result["status"], "not_ready_errors": result["not_ready_errors"]}, indent=2))
    if require_ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
