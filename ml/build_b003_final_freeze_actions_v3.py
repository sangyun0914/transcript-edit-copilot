"""Versioned B003/release003 adapter for the immutable final-freeze builder.

Registration and templates are not authorization. Assembly delegates every
original authority/runtime gate and final load_gate readback to the pinned core.
No command acquires reserved bodies, loads a model, or activates the application.
Registration/preflight/assembly verify large existing payloads: run remotely or
only after the local hashing restriction is lifted.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import prepare_b003_final_gate_actions_v3 as inputs

ROOT = inputs.ROOT
VERSION = "b003-final-freeze-registration-1"
CORE = ROOT / "artifacts/audits/project-completion-2026-10-11/final-freeze-builder-001/build_freeze.py"
CORE_SHA = "c23919f2c477158a816f3daf187ef974c40689a7d211a375283b63044c351e0a"
OLD_DRAFT = inputs.EXPERIMENT / "final-policy-draft-005"
OLD_DRAFT_SHA = "08f921bc28476891b8b4f782290e91124146a1e66609e0492a73f1e4d16b16b6"
RELEASE_SHA = "f91b3b49856d918d32958b653bc48b7170422f867504bb802995ffd421ca4a3d"
CONTINGENT_SHA = "8a8aa534182ad68dd2a4ea13ee13271e327abbefc8dec438771e830134dbc51a"
DEMO_IDS = ["v3-pilot-actions-v1-en-field-recorder-01", "v3-pilot-actions-v1-ko-library-recording-00"]
require = inputs.require


def source_binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file() and path.stat().st_size <= inputs.LIMIT,
            "Missing or oversized builder source.")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def implementations():
    return {"b003_builder": source_binding(__file__), "metadata_helper": source_binding(inputs.__file__),
            "core_builder": source_binding(CORE)}


def load_core():
    require(source_binding(CORE)["sha256"] == CORE_SHA, "Immutable core builder changed.")
    spec = importlib.util.spec_from_file_location("b003_pinned_freeze_core", CORE)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def actors_from(registry, read_bound):
    require(registry.get("version") == "task-test-actions-v3-actor-registry-1"
            and registry.get("status") == "parent_reviewed_roles" and registry.get("actor") == "/root"
            and registry.get("reserved_bodies_read") is False, "Missing actual parent-reviewed actor registry.")
    actors = registry.get("actors", {})
    require(set(actors) == {"preparer", "author", "reviewer", "grader"}
            and all(isinstance(value, str) and value.strip() and value.lower() not in {"unknown", "pending", "todo"}
                    for value in actors.values())
            and len({actors[key] for key in ("author", "reviewer", "grader")}) == 3,
            "Author, reviewer and grader must be distinct actual actors.")
    evidence = registry.get("access_evidence", {})
    require(set(evidence) == set(actors) and all(isinstance(value, list) and value for value in evidence.values()),
            "Every actor requires preserved access evidence.")
    for references in evidence.values():
        for reference in references:
            read_bound(reference)
    return actors


def exposure_from(review, read_bound):
    require(review.get("version") == "task-test-actions-v3-exposure-readiness-1"
            and review.get("status") == "parent_reviewed" and review.get("actor") == "/root"
            and review.get("selected_training_dataset_manifest_sha256") == RELEASE_SHA
            and review.get("reserved_bodies_read") is False
            and review.get("new_project_visible_inputs_unaccounted_for") is False,
            "An actual release003 exposure review is required; unaccounted new inputs block registration.")
    read_bound(review["exposed_text_hashes"])
    evidence = review.get("exposure_inventory_evidence", {})
    require(evidence and isinstance(evidence, dict), "Exposure inventory evidence is missing.")
    for reference in evidence.values():
        read_bound(reference)
    require(isinstance(review.get("review_evidence"), list) and review["review_evidence"], "Exposure review evidence is missing.")
    for reference in review["review_evidence"]:
        read_bound(reference)
    return review


def contingent_context(core):
    bundle = core.read_bound({"path": str(inputs.CONTINGENT / "manifest.json"), "sha256": CONTINGENT_SHA})
    require(bundle["status"] == "contingent_not_selected_not_body_authorized"
            and bundle["body_access_authorized"] is False and bundle["reserved_final_test_bodies_read"] is False,
            "Contingent metadata authority changed.")
    for name, sha in bundle["files_sha256"].items():
        core.gate.bound({"path": str(inputs.CONTINGENT / name), "sha256": sha})
    registry = core.read_bound(bundle["icl_candidate"])
    original = core.read_bound(registry["original_registry"])
    require(original["selection_used_development_outputs"] is False
            and registry["release_binding"]["manifest"]["sha256"] == RELEASE_SHA
            and registry["fixed_demonstration_ids"] == DEMO_IDS, "Fixed TRAIN-only ICL registration changed.")

    def without_release(value):
        if isinstance(value, dict):
            return {key: without_release(child) for key, child in value.items() if key != "release_binding"}
        if isinstance(value, list):
            return [without_release(child) for child in value]
        return value

    require(without_release(registry["conditions"]) == without_release(original["conditions"]),
            "ICL content, order or canonical condition changed beyond its exposure release binding.")
    verification = core.read_bound(bundle["verification"])
    require(verification["unchanged_roster_slots"] == 104 and verification["unchanged_natural_windows"] == 16
            and all(verification[key] is True for key in (
                "unchanged_source_reservations", "unchanged_fallback_publisher_order", "effective_mapping_identical",
                "assessment_cohorts_records_timestamps_locators_evidence_unchanged",
                "index002_retained_and_covers_release003_train_dev", "original_icl_pair_native_semantics_unchanged")),
            "Contingent metadata continuity evidence differs.")
    metadata = {"metadata_manifest": bundle["metadata_candidate"], "source_resolution": bundle["source_resolution_candidate"]}
    for key, name in (("source_reservations", "source-reservations-proposed.json"),
                      ("roster", "request-slots-proposed.jsonl"), ("natural_windows", "natural-window-proposal.json")):
        metadata[key] = core.binding(inputs.CONTINGENT / "metadata" / name)
    allocation = core.gate.metadata_allocation(core.gate.bound(metadata["metadata_manifest"]))
    require(len(allocation["windows"]) == 16 and len(allocation["roster"]) == 104, "Original allocation changed.")
    core.gate.load_source_resolution(metadata, datetime.now(timezone.utc))
    return bundle, registry, metadata, allocation


def verify_native_model(core, model, role):
    import infer_task_test_actions_v3 as native

    require(model["role"] == role, "Model receipt role differs.")
    config = core.read_bound(model["inference_configuration"])
    training, reloaded = (core.gate.bound(model[key]).parent for key in ("artifact_manifest", "reload_receipt"))
    actual = native.native_identity(config, training, reloaded, adapter_version=model["identity"]["kind"])
    require(actual == model["identity"], "Native artifact/reload identity differs from its receipt.")
    require(model["inference_configuration"]["sha256"] == inputs.SFT_SHA["configuration_sha256"],
            "Learned roles must use the B003 inference configuration.")
    saved, recorded_identity, _, records = inputs.recorded_run(
        training / "configuration.json", training, reloaded, model["identity"]["kind"])
    require(recorded_identity == actual, "Recorded training configuration differs from native identity.")
    if role == "sft":
        require(actual["kind"] == "gemma-actions-v3-sft"
                and all(actual[key] == value for key, value in inputs.SFT_SHA.items()), "SFT parent is not B003.")
    else:
        require(actual["kind"] == "gemma-actions-v3-dpo" and saved["phase"] == "substantive"
                and saved["anchor"] == {"anchor_" + key: value for key, value in inputs.SFT_SHA.items()}
                and saved["sft_preparation_sha256"] == model["preparation_manifest"]["sha256"],
                "DPO training provenance is not the registered B003 parent.")
        before, after = records
        require(before["initialization"] == "frozen_sft_adapter_fresh_optimizer"
                and before["optimizer_committed_rows"] == saved["training"]["pair_count"]
                and before["reference_manifest_sha256"] == after["reference_manifest_sha256"],
                "DPO pair/reference completion differs from the registered training run.")
    core.gate.model_runtime(model)
    return actual


def register(output, actor_registry, exposure_review, dpo_config, dpo_training, dpo_reloaded,
             selected_role="sft", dpo_export_receipt=None):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Use a fresh workspace registration directory.")
    core = load_core()
    actors_binding, exposure_binding = core.binding(actor_registry), core.binding(exposure_review)
    actors = actors_from(core.read_bound(actors_binding), core.read_bound)
    exposure = exposure_from(core.read_bound(exposure_binding), core.read_bound)
    bundle, registry, metadata, _ = contingent_context(core)
    revision = core.read_bound({"path": str(OLD_DRAFT / "revision.json"), "sha256": OLD_DRAFT_SHA})
    drafts = {name.removesuffix(".json"): core.read_bound({"path": str(OLD_DRAFT / name), "sha256": sha})
              for name, sha in revision["files_sha256"].items()}
    require(set(drafts) == set(core.LEAF_ORDER), "Expected the original eight policy leaves.")
    output.mkdir(parents=True)
    proposed = inputs.prepare(output / "model-inputs", selected_role, dpo_config, dpo_training, dpo_reloaded,
                              dpo_export_receipt)
    require(proposed["existing_native_base_receipt"] == core.pinned("native_base"), "Original native base receipt changed.")
    models = {}
    for role in ("sft", "sft_dpo"):
        model = copy.deepcopy(proposed["model_receipt_drafts"][role])
        require(model is not None, "A completed B003-DPO run is required; a B002 substitute is forbidden.")
        verify_native_model(core, model, role)
        model.update(version="task-test-actions-v3-model-receipt-1", status="verified", payload_hashes_recomputed=True,
                     verification_scope="Native train/reload payloads and model-specific inference runtime verified; UI and parent authorization remain separate.",
                     model_input_inventory=core.binding(output / "model-inputs/inputs.json"))
        models[role] = core.write_new(output / "models" / (role + ".json"), model)
    controller = core.write_new(output / "controller-source-closure.json", core.gate.code_sources())
    old_metadata = drafts["admission-policy"]["current_metadata_bindings"]
    replacements = {str(Path(old_metadata[key]["path"]).resolve()): value for key, value in metadata.items()}
    written = {}

    def rebind(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = Path(value["path"]).resolve()
                if str(path) in replacements:
                    return replacements[str(path)]
                if path.parent == OLD_DRAFT and path.stem in written:
                    return written[path.stem]
                if path.is_relative_to(ROOT / "docs"):
                    return core.binding(path)
            return {key: rebind(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rebind(child) for child in value]
        return value

    amendment = {
        "version": VERSION, "status": "draft_not_authorization", "body_access_authorized": False,
        "sft": "B003", "sft_dpo": "Standard DPO initialized from exactly B003", "run_roles": list(inputs.ROLES),
        "metadata": metadata, "conditions": bundle["icl_candidate"], "actor_registry": actors_binding,
        "exposure_review": exposure_binding, "historical_DEV_only": ["B002", "B002-DPO"],
        "handoff": core.binding(ROOT / "docs/plans/b003-prelocal-handoff-2026-10-24.md"),
    }
    amendment_binding = core.write_new(output / "amendment.json", amendment)
    for name in core.LEAF_ORDER:
        leaf = rebind(copy.deepcopy(drafts[name]))
        require(leaf["status"] == "draft" and leaf["body_access_authorized"] is False, "Original draft authority changed.")
        leaf.update(version=leaf["version"] + "-b003-1", created_at=core.now(),
                    b003_amendment=amendment_binding, b003_basis_document=amendment["handoff"],
                    original_draft=core.binding(OLD_DRAFT / (name + ".json")))
        if name == "generator-recipe":
            leaf["pre_body_gate"]["metadata_exposure_binding"] = "This draft binds the preserved release003 contingent metadata; only complete parent-authorized final assembly selects it for body admission."
        if name == "admission-policy":
            leaf.update(current_metadata_bindings=metadata, actors=actors, actor_registry_status="actual_roles_bound_not_body_authorization",
                        actual_actor_registry_evidence=actors_binding, exposed_text_hashes=exposure["exposed_text_hashes"],
                        exposure_inventory_evidence=exposure["exposure_inventory_evidence"], final_exposure_review=exposure_binding,
                        metadata_relation="Release003 contingent metadata is explicitly bound; reservation, roster, windows and effective allocation are unchanged. Parent authority remains pending.")
        written[name] = core.write_new(output / "policies" / (name + ".json"), leaf)
    revision_binding = core.write_new(output / "policies/revision.json", {
        "version": VERSION, "status": "draft", "body_access_authorized": False, "reserved_test_bodies_read": False,
        "files_sha256": {name + ".json": reference["sha256"] for name, reference in written.items()},
        "original_revision": {"path": str(OLD_DRAFT / "revision.json"), "sha256": OLD_DRAFT_SHA},
        "amendment": amendment_binding,
    })
    pins = {"draft_revision": revision_binding, "controller": controller, "conditions": bundle["icl_candidate"],
            "native_base": proposed["existing_native_base_receipt"], "sft": models["sft"],
            "numeric_targets": core.pinned("numeric_targets")}
    registration = {
        "version": VERSION, "status": "draft_not_authorization", "body_access_authorized": False,
        "reserved_bodies_read": False, "implementations": implementations(), "pins": pins,
        "dpo_model_receipt": models["sft_dpo"], "proposed_selected_role": selected_role,
        "actor_registry": actors_binding, "actors": actors, "exposure_review": exposure_binding,
        "contingent_bundle": core.binding(inputs.CONTINGENT / "manifest.json"),
        "training_dataset_manifest": registry["release_binding"]["manifest"], "metadata": metadata,
        "amendment": amendment_binding,
    }
    return core.write_new(output / "registration.json", registration)


def validate_registration(core, registration):
    revision = core.read_bound(core.pinned("draft_revision"))
    drafts, documents = {}, {}
    for filename, sha in revision["files_sha256"].items():
        leaf = core.read_bound({"path": str(core.DRAFT / filename), "sha256": sha})
        require(leaf["status"] == "draft" and leaf["scope"] == "final_test"
                and leaf["body_access_authorized"] is False, "Draft authority changed.")
        drafts[filename.removesuffix(".json")] = leaf
        for reference in core.bindings_in(leaf):
            path = Path(reference["path"]).resolve()
            require(path.is_relative_to(ROOT), "Draft binding escaped the workspace.")
            if path.is_relative_to(ROOT / "docs"):
                core.gate.bound(reference)
                documents[str(path.relative_to(ROOT))] = core.gate.sha256(path)
            else:
                core.gate.bound(reference)
    require(set(drafts) == set(core.LEAF_ORDER), "Expected eight unchanged registered policy leaves.")
    closure = core.read_bound(core.pinned("controller"))
    require(closure == core.gate.code_sources(), "Controller changed after registration; create a new registration.")
    for path, sha in drafts["metrics-policy"]["reporting_implementation_sha256"].items():
        core.gate.bound({"path": str(ROOT / path), "sha256": sha})
    core.read_bound(core.pinned("numeric_targets"))
    bundle, registry, metadata, allocation = contingent_context(core)
    require(core.pinned("conditions") == bundle["icl_candidate"] and registration["metadata"] == metadata
            and drafts["admission-policy"]["current_metadata_bindings"] == metadata
            and registration["training_dataset_manifest"] == registry["release_binding"]["manifest"],
            "Registered release003 metadata/ICL bindings differ.")
    exposure = exposure_from(core.read_bound(registration["exposure_review"]), core.read_bound)
    admission = drafts["admission-policy"]
    require(admission["actors"] == registration["actors"]
            and admission["exposed_text_hashes"] == exposure["exposed_text_hashes"]
            and admission["exposure_inventory_evidence"] == exposure["exposure_inventory_evidence"],
            "Policy actor/exposure bindings differ from their actual reviews.")
    models = {role: core.read_bound(core.pinned(role)) for role in ("native_base", "sft")}
    require(all(models["sft"]["identity"][key] == value for key, value in inputs.SFT_SHA.items()), "SFT parent is not B003.")
    runtimes = {role: core.validate_model(model, role) for role, model in models.items()}
    return {"drafts": drafts, "closure": closure, "metadata": metadata, "registry": registry,
            "models": models, "runtimes": runtimes, "basis_documents": documents,
            "window_ids": [window["window_id"] for window in allocation["windows"]]}


def protect_preflight(core, registration, registration_binding):
    original = core.preflight

    def checked(readiness_path, readiness_sha256, output):
        ready = core.read_bound({"path": str(Path(readiness_path).resolve()), "sha256": readiness_sha256})
        require(ready.get("b003_implementations") == registration["implementations"]
                and ready.get("b003_registration") == registration_binding
                and ready.get("actor_registry") == registration["actor_registry"]
                and ready.get("exposure_review") == registration["exposure_review"],
                "Parent readiness must approve the exact B003 builder, registration, actors and exposure review.")
        require(ready.get("dpo_model_receipt") == registration["dpo_model_receipt"], "Registered DPO receipt differs.")
        context = original(readiness_path, readiness_sha256, output)
        require(context["selected_model"] == registration["proposed_selected_role"], "Selection changed; register the new selected role.")
        model = context["models"][context["selected_model"]]
        deployment = core.read_bound(ready["selected_deployment_receipt"])
        selected_export = inputs.export_binding(deployment["adapter_export_receipt"]["path"], model["identity"])
        inputs.local_evidence(Path(ready["selected_deployment_receipt"]["path"]).parent,
                              ready["selected_model_ui_receipt"]["path"], model, selected_export)
        return context

    core.preflight = checked


def configure(registration_path, registration_sha256):
    registration, reference = inputs.metadata(registration_path, registration_sha256)
    require(registration.get("version") == VERSION and registration.get("status") == "draft_not_authorization"
            and registration.get("body_access_authorized") is False and registration.get("reserved_bodies_read") is False,
            "A complete unprivileged B003 registration is required.")
    require(registration["implementations"] == implementations(), "Registered builder/helper/core source changed.")
    core = load_core()
    require(set(registration["pins"]) == set(core.PINNED), "Registration pins are incomplete.")
    require(all(registration["pins"][name] == core.pinned(name) for name in ("native_base", "numeric_targets")),
            "Original native base or numerical targets changed.")
    require(registration["proposed_selected_role"] in {"sft", "sft_dpo"}, "Invalid selected-role proposal.")
    core.PINNED = {name: (core.gate.bound(value), value["sha256"]) for name, value in registration["pins"].items()}
    core.DRAFT = core.PINNED["draft_revision"][0].parent
    core.ACTORS = actors_from(core.read_bound(registration["actor_registry"]), core.read_bound)
    require(core.ACTORS == registration["actors"], "Registered actors changed.")
    exposure_from(core.read_bound(registration["exposure_review"]), core.read_bound)
    require(registration["contingent_bundle"] == core.binding(inputs.CONTINGENT / "manifest.json"), "Contingent bundle changed.")
    dpo = core.read_bound(registration["dpo_model_receipt"])
    require(dpo["status"] == "verified" and dpo["identity"]["kind"] == "gemma-actions-v3-dpo", "A verified standard B003 DPO receipt is required.")
    for role, model in (("sft", core.read_bound(core.pinned("sft"))), ("sft_dpo", dpo)):
        verify_native_model(core, model, role)
    core.validate_drafts = lambda: validate_registration(core, registration)
    core.HERE = Path(registration_path).resolve().parent
    protect_preflight(core, registration, reference)
    return core, registration, reference


def readiness_template(core, registration, reference):
    result = core.readiness_template(core.validate_drafts())
    result.update(b003_implementations=registration["implementations"], b003_registration=reference,
                  actor_registry=registration["actor_registry"], exposure_review=registration["exposure_review"],
                  dpo_model_receipt=registration["dpo_model_receipt"], selected_model_proposal=registration["proposed_selected_role"])
    return result


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("register")
    for name in ("output", "actor-registry", "exposure-review", "dpo-config", "dpo-training", "dpo-reloaded"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--dpo-export-receipt", type=Path)
    command.add_argument("--selected-role", choices=("sft", "sft_dpo"), default="sft")
    for name in ("readiness-template", "preflight", "assemble"):
        command = commands.add_parser(name)
        command.add_argument("--registration", type=Path, required=True)
        command.add_argument("--registration-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
        if name != "readiness-template":
            command.add_argument("--readiness", type=Path, required=True)
            command.add_argument("--readiness-sha256", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "register":
        result = register(**args)
    else:
        core, registration, reference = configure(args.pop("registration"), args.pop("registration_sha256"))
        if command == "readiness-template":
            result = core.write_new(args["output"], readiness_template(core, registration, reference))
        elif command == "preflight":
            context = core.preflight(args["readiness"], args["readiness_sha256"], args["output"])
            result = {"status": "parent_readiness_preflight_passed_no_freeze_written",
                      "selected_model": context["selected_model"], "body_access_authorized_by_this_command": False}
        else:
            result = core.assemble(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
