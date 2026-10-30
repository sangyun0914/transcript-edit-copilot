"""Build small-family final metadata without acquiring or reading TEST bodies.

Preparation and registration grant no body access. Assembly requires a separate,
exact coordinator readiness receipt and repeats the existing final metadata gate.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile

import build_b003_final_freeze_actions_v3 as legacy
import prepare_task_test_actions_v3 as gate
import small_final_test_runtime as small
import gemma_final_test_runtime as gemma

ROOT = gate.ROOT
VERSION = "small-final-freeze-builder-1"
READINESS_VERSION = "small-final-parent-freeze-readiness-1"
UI_VERSION = "small-final-selected-ui-evidence-1"
PENDING_UI_VERSION = "small-final-pending-product-validation-1"
UI_AMENDMENT = ROOT / "docs/plans/final-scientific-freeze-ui-amendment-2026-10-25.md"
SELECTION_PLAN = ROOT / "docs/plans/final-sft-comparison-2026-10-25.md"
DPO_PLAN = ROOT / "docs/plans/final-small-model-dpo-2026-10-25.md"
UI_ATTESTATIONS = {"actual_selected_model_ui_reviewed", "local_apply_undo_export_and_durability_reviewed",
                   "all_UI_planned_and_issued_outcomes_retained"}
PRODUCT_CHECKS = ["normal_npm_start_automatic_en_ko_previews", "shuka_split_delete_trim",
                  "full_visible_latency", "long_session"]
UI_MEASUREMENTS = ("measurement_plan", "measurement_roster", "latency_summary")
DEPLOYMENT_POLICY_VERSION = "small-final-deployed-evaluation-policy-1"
DEPLOYMENT_PLAN = ROOT / "docs/plans/final-deployed-model-test-2026-10-25.md"
DEPLOYMENT_PLAN_SHA = "81bbbbad1f43765642defafe4ca1845fec15a35b5146edddeb85f742d540faac"
FINAL = ROOT / "artifacts/datasets/nep-passive-actions-final-v1/release-attempt-001"
FINAL_SHA = "86a997f28826597e63d55fd6bf4ab8496c3bdaa0950da5bda92713acbdaaaf58"
EXPOSURE_SHA = "50732ae251e574daf6949f67bf0c2c4ee4dbd7dd59f285e21c5a0dfec0ac36de"
ROLES = ("native_base", "base_icl", "sft", "sft_dpo", "always_no_action", "consistent_history_rule")
EVIDENCE = ("selected_deployment_receipt", "selected_model_ui_receipt", "native_environment_receipt", "native_runtime_receipt")
require, read, sha, digest = gate.require, gate.read_json, gate.sha256, gate.digest


def binding(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "Use an existing workspace artifact.")
    return gate.binding(path)


def read_bound(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "Missing exact artifact binding.")
    return read(gate.bound(value))


def normalized(value):
    path = Path(value["path"])
    path = path if path.is_absolute() else ROOT / path
    result = binding(path)
    require(result["sha256"] == value["sha256"], "Existing artifact binding changed.")
    return result


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return binding(path)


def fresh(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and not path.exists(), "Use a fresh workspace output directory.")
    return path


def implementation():
    return {name: binding(path) for name, path in {
        "builder": Path(__file__), "legacy_registration": Path(legacy.__file__),
        "immutable_core": legacy.CORE, "model_bridge": Path(small.__file__),
        "scientific_ui_amendment": UI_AMENDMENT,
        "development_summary": ROOT / "ml/evaluate_small_actions_final.py",
        "development_grader": ROOT / "ml/evaluate_passive_actions_v3.py",
        "gemma_model_bridge": Path(gemma.__file__),
        "gemma_deployment": ROOT / "ml/gemma_final_deployment.py",
        "gemma_development": ROOT / "ml/evaluate_gemma_final.py",
        "gemma_selection": ROOT / "ml/gemma_final_freeze.py",
        "gemma_q8_export": ROOT / "ml/gemma_q8_export.py",
        "gemma_q8_product": ROOT / "ml/gemma_q8_product.py",
        "gemma_q8_development": ROOT / "ml/evaluate_gemma_q8_final.py",
        "gemma_q8_plan": ROOT / "docs/plans/final-gemma26-q8-product-comparison-2026-10-25.md",
        "gemma_descriptive_product_plan": ROOT / "docs/plans/final-gemma26-product-descriptive-evaluation-2026-10-25.md",
    }.items()}


def model_backend(manifest):
    require(manifest.get("version") in {small.VERSION, gemma.VERSION}, "Unknown final model metadata family.")
    return gemma if manifest["version"] == gemma.VERSION else small


def deployment_plan(backend):
    if backend is gemma:
        from gemma_final_deployment import PLAN
        return PLAN
    require(sha(DEPLOYMENT_PLAN) == DEPLOYMENT_PLAN_SHA, "The prospective product evaluation policy changed.")
    return DEPLOYMENT_PLAN


def deployment_sources():
    paths = [ROOT / "ml/infer_deployed_task_test_actions_v3.py", ROOT / "ml/compare_deployed_task_test_actions_v3.py"]
    require(all(path.is_file() for path in paths), "Finish the registered product evaluation implementation before model registration.")
    from infer_deployed_task_test_actions_v3 import sources

    return sources()


def reservations_for_release(original, splits, inventory):
    result = copy.deepcopy(original)
    old = result["exposed_source_group_splits"]
    require(all(name not in splits or splits[name] == split for name, split in old.items()),
            "An existing source changed its TRAIN/DEV split.")
    require(set(splits.values()) <= {"train", "dev"}, "Unexpected source split.")
    require(not {row["source_group"] for row in result["groups"]}.intersection(splits),
            "A reserved source has TRAIN/DEV exposure.")
    old.update(splits)
    result["exposed_inventory_sha256"].update(inventory)
    return result


def semantic_demo(value):
    return {key: child for key, child in value.items()
            if key not in {"release_binding", "native_prefix_ids_sha256", "selection_tokens"}}


def fixed_demonstrations(cases, original, tokenizer, release_binding):
    require([row["case_id"] for row in original] == small.DEMONSTRATION_IDS,
            "The original fixed ICL pair or order changed.")
    require([row["id"] for row in cases] == small.DEMONSTRATION_IDS,
            "Do not reselect ICL examples for another tokenizer or model result.")
    demos = [gate.conditions.demonstration(case, tokenizer, release_binding) for case in cases]
    require([semantic_demo(value) for value in demos] == [semantic_demo(value) for value in original],
            "Fixed TRAIN ICL content or reference semantics changed.")
    return demos


def prepare(output):
    output = fresh(output)
    core = legacy.load_core()
    bundle, registry, old_metadata, allocation = legacy.contingent_context(core)
    release_binding = {"path": str(FINAL / "manifest.json"), "sha256": FINAL_SHA}
    release = read_bound(release_binding)
    verification_path = FINAL.parent / "release-attempt-001-verification.json"
    verification = read(verification_path)
    require(verification["status"] == "passed" and normalized(verification["release_manifest"]) == release_binding
            and release["counts"]["train"] == 1261 and release["counts"]["dev"] == 471
            and release["test_bodies_read"] is False
            and verification["source_group_cross_split_collisions"] == verification["reserved_source_groups"] == 0,
            "Use the independently admitted, source-separated final release.")
    for name in ("source-splits.json", "cases/train.jsonl"):
        require(sha(FINAL / name) == release["files_sha256"][name], "Final release payload changed.")
    index_path = FINAL / "exposure-audit/index.json"
    require(sha(index_path) == EXPOSURE_SHA, "The registered 2940-text exposure addendum changed.")
    index = read(index_path)
    previous_index = read_bound(bundle["exposed_text_index"])
    require(len(index["sha256"]) == len(set(index["sha256"])) == 2940
            and set(previous_index["sha256"]) <= set(index["sha256"]), "Exposure history was dropped.")
    selected = {}
    with (FINAL / "cases/train.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in small.DEMONSTRATION_IDS:
                require(row["id"] not in selected and row["split"] == row["source"]["split"] == "train",
                        "Fixed demonstration is duplicate or not TRAIN.")
                selected[row["id"]] = row
    require(set(selected) == set(small.DEMONSTRATION_IDS), "A fixed TRAIN demonstration is absent.")
    originals = registry["conditions"]["train_icl"]["demonstrations"]
    require([row["case_id"] for row in originals] == small.DEMONSTRATION_IDS
            and all(digest(selected[row["case_id"]]) == row["case_sha256"] for row in originals),
            "The original fixed TRAIN case bytes changed semantically.")

    output.mkdir(parents=True)
    old_manifest_path = gate.bound(old_metadata["metadata_manifest"])
    old_manifest = read(old_manifest_path)
    for name, expected in old_manifest["files_sha256"].items():
        source = old_manifest_path.parent / name
        require(source.resolve().is_relative_to(old_manifest_path.parent) and sha(source) == expected,
                "Historical metadata payload changed.")
        target = output / "metadata" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name != "source-reservations-proposed.json":
            shutil.copyfile(source, target)
    inventories = {str(path.relative_to(ROOT)): sha(path) for path in
                   (FINAL / "source-splits.json", FINAL / "exposure-audit/manifest.json", index_path)}
    reservations = reservations_for_release(allocation["reservations"], read(FINAL / "source-splits.json"), inventories)
    write(output / "metadata/source-reservations-proposed.json", reservations)
    metadata = copy.deepcopy(old_manifest)
    metadata.update(status="contingent_final_unified_not_model_selected_not_body_authorized", created_at=gate.now(),
                    selected_dataset_manifest_sha256=FINAL_SHA, dataset_selected=False, model_selected=False,
                    body_access_authorized=False, original_metadata_manifest=old_metadata["metadata_manifest"],
                    implementation=binding(__file__), training_release_verification=binding(verification_path),
                    exposed_text_hashes=binding(index_path))
    metadata["previous_exposure_amendment"] = old_manifest["exposure_amendment"]
    metadata["exposure_amendment"] = binding(FINAL / "exposure-audit/manifest.json")
    metadata["files_sha256"]["source-reservations-proposed.json"] = sha(output / "metadata/source-reservations-proposed.json")
    write(output / "metadata/manifest.json", metadata)
    old_resolution = read_bound(old_metadata["source_resolution"])
    assessments = read_bound(old_resolution["assessments"])
    assessments["metadata_manifest_sha256"] = sha(output / "metadata/manifest.json")
    assessments["exposure_rebinding"] = {
        "original_assessment": old_resolution["assessments"], "original_resolution": old_metadata["source_resolution"],
        "records_observations_timestamps_locators_unchanged": True, "new_availability_claim": False,
        "network_requests": 0, "reserved_bodies_read": False, "body_access_authorized": False,
    }
    write(output / "resolution/source-assessments.json", assessments)
    gate.resolve_sources(output / "metadata/manifest.json", output / "resolution/source-assessments.json",
                         output / "resolution/source-resolution.json")
    metadata_bindings = {"metadata_manifest": binding(output / "metadata/manifest.json"),
                         "source_resolution": binding(output / "resolution/source-resolution.json")}
    for key in ("source_reservations", "roster", "natural_windows"):
        metadata_bindings[key] = binding(output / "metadata" / Path(old_metadata[key]["path"]).name)
    resolved = gate.load_source_resolution(metadata_bindings, datetime.now(timezone.utc))["receipt"]
    require(all(resolved[key] == old_resolution[key] for key in
                ("windows", "cohort_order", "fallback_source_group", "reserved_bodies_read", "quality_based_substitution")),
            "The preregistered effective source allocation changed.")
    require(reservations["groups"] == allocation["reservations"]["groups"]
            and (output / "metadata/request-slots-proposed.jsonl").read_bytes() == allocation["paths"]["roster"].read_bytes()
            and (output / "metadata/natural-window-proposal.json").read_bytes() == allocation["paths"]["windows"].read_bytes(),
            "Source reservations, roster slots or natural windows changed.")
    write(output / "fixed-training-cases.json", [selected[name] for name in small.DEMONSTRATION_IDS])
    result = {
        "version": VERSION, "status": "prepared_not_selected_not_body_authorized", "created_at": gate.now(),
        "body_access_authorized": False, "reserved_test_bodies_read": False, "model_selected": False,
        "implementation": implementation(), "metadata": metadata_bindings, "original_metadata": old_metadata,
        "original_contingent_bundle": binding(legacy.inputs.CONTINGENT / "manifest.json"),
        "original_icl_registry": bundle["icl_candidate"], "fixed_train_cases": binding(output / "fixed-training-cases.json"),
        "fixed_demonstration_ids": small.DEMONSTRATION_IDS, "training_dataset_manifest": release_binding,
        "training_release_verification": binding(verification_path), "exposed_text_hashes": binding(index_path),
        "exposure_addendum": binding(FINAL / "exposure-audit/manifest.json"),
        "global_exposure_census_complete": False,
        "counts": {"roster_slots": 104, "natural_windows": 16, "fixed_train_demonstrations": 2,
                   "train": 1261, "dev": 471, "exposed_wording_hashes": 2940},
        "continuity": {"roster_and_window_bytes_identical": True, "reserved_groups_unchanged": True,
                       "effective_allocation_and_fallback_order_unchanged": True,
                       "assessment_cohorts_records_timestamps_locators_unchanged": True},
    }
    write(output / "manifest.json", result)
    write(output / "actor-registry-template.json", {
        "version": "task-test-actions-v3-actor-registry-1", "status": "draft_not_authorization", "actor": "/root",
        "reserved_bodies_read": False, "actors": dict.fromkeys(("preparer", "author", "reviewer", "grader")),
        "access_evidence": {role: [] for role in ("preparer", "author", "reviewer", "grader")},
    })
    write(output / "exposure-review-template.json", {
        "version": "task-test-actions-v3-exposure-readiness-1", "status": "draft_not_authorization", "actor": "/root",
        "selected_training_dataset_manifest_sha256": FINAL_SHA, "reserved_bodies_read": False,
        "new_project_visible_inputs_unaccounted_for": None, "exposed_text_hashes": result["exposed_text_hashes"],
        "exposure_inventory_evidence": {"final_addendum": result["exposure_addendum"],
                                        "final_release_verification": result["training_release_verification"]},
        "review_evidence": [],
    })
    write(output / "ui-evidence-template.json", ui_template())
    write(output / "pending-product-validation-template.json", pending_ui_template())
    return binding(output / "manifest.json")


def exposure_review(value, prepared):
    require(value.get("version") == "task-test-actions-v3-exposure-readiness-1"
            and value.get("status") == "parent_reviewed" and value.get("actor") == "/root"
            and value.get("selected_training_dataset_manifest_sha256") == FINAL_SHA
            and value.get("reserved_bodies_read") is False
            and value.get("new_project_visible_inputs_unaccounted_for") is False,
            "Require an actual coordinator exposure review; the 2940-text addendum alone is not global clearance.")
    baseline = read_bound(prepared["exposed_text_hashes"])["sha256"]
    current = read_bound(value["exposed_text_hashes"])["sha256"]
    require(current and len(current) == len(set(current)) and set(baseline) <= set(current)
            and all(isinstance(item, str) and len(item) == 64 and set(item) <= set("0123456789abcdef") for item in current),
            "A final exposure index must retain every registered earlier wording hash.")
    require(value.get("exposure_inventory_evidence") and value.get("review_evidence"), "Exposure review evidence is missing.")
    for reference in [*value["exposure_inventory_evidence"].values(), *value["review_evidence"]]:
        gate.bound(reference)


def selected_model(selection, models):
    role = selection.get("selected_model")
    require(selection.get("status") == "frozen" and selection.get("selection_scope") == "development"
            and selection.get("test_bodies_read") is False and role in {"sft", "sft_dpo"} and role in models
            and selection.get("selected_identity_sha256") == digest(models[role]["identity"]),
            "Require an actual frozen DEV selection of the exact available small model.")
    require(selection.get("evidence"), "Bind the actual DEV selection evidence.")
    for reference in selection["evidence"]:
        gate.bound(reference)
    return role


def verified_dev_summary(reference):
    import evaluate_small_actions_final as development

    summary = read_bound(reference)
    report, mapping = read_bound(summary["report"]), read_bound(summary["cohort_mapping"])
    release = {"path": str(FINAL / "manifest.json"), "sha256": FINAL_SHA}
    require(mapping["counts"] == {"canonical_dev322": 322, "supplemental_dev149": 149}
            and normalized(mapping["release_manifest"]) == release
            and report["status"] == "graded_development" and report["final_test"] is False
            and report["format"] == "json" and report["condition"] == "canonical_full_context"
            and report["original_development_cases"] == 471 and report["excluded_before_inference"] == 0,
            "Quality-only selection requires the unchanged complete final DEV471 cohort.")
    prepared = read_bound(report["preparation"])
    require(normalized(prepared["dataset_manifest"]) == release and prepared["final_test_bodies_read"] is False,
            "DEV evidence belongs to a different final release or evaluation.")
    verify_dev_cohort_inputs(report["preparation"], mapping["evaluation_manifest"])
    configuration = read_bound(prepared["model_preparation"])
    require(configuration["source_cases_sha256"] == read_bound(release)["files_sha256"]["cases/train.jsonl"]
            and configuration["training"]["epochs"] == 3
            and Counter(configuration["training"]["sample_order"]) == Counter({index: 3 for index in range(1261)}),
            "Compare models trained on all final1261 rows with the registered three-epoch budget.")
    require(report["schema_constrained"] is True
            or report["model_identity"].get("configuration_sha256") == prepared["model_preparation"]["sha256"],
            "Native DEV results belong to another training preparation.")
    run = read_bound(report["inference_run"])
    require(run["status"] == "inference_complete" and run["requested"] == run["observed"] == 471
            and run["final_test_bodies_read"] is False and run["model_identity"] == report["model_identity"]
            and run["numerical_regime"] == report["numerical_regime"]
            and run["schema_constrained"] is report["schema_constrained"],
            "DEV inference identity or complete request denominator differs.")
    for key in ("observations", "inference_manifest", "graded"):
        gate.bound(report[key])
    with tempfile.TemporaryDirectory(prefix="small-final-dev-summary-") as temporary:
        recomputed = development.summarize(gate.bound(summary["report"]), gate.bound(summary["cohort_mapping"]),
                                           Path(temporary) / "summary.json")
    require(recomputed == summary, "DEV selection summary differs from its actual graded rows.")
    return summary, report


def verify_dev_cohort_inputs(prepared_reference, original_reference):
    prepared, original = read_bound(prepared_reference), read_bound(original_reference)
    require(normalized(prepared["dataset_manifest"]) == normalized(original["dataset_manifest"])
            and prepared["final_test_bodies_read"] is original["final_test_bodies_read"] is False
            and prepared["excluded"] == original["excluded"] == 0,
            "DEV wrappers must preserve the same release and no exclusions.")
    roots = [gate.bound(ref).parent for ref in (prepared_reference, original_reference)]
    for name in ("cases.jsonl", "exclusions.jsonl", "requests-json.jsonl"):
        require(all(sha(root / name) == value["files_sha256"][name]
                    for root, value in zip(roots, (prepared, original), strict=True)), "DEV wrapper payload changed.")
        if name != "requests-json.jsonl":
            require(prepared["files_sha256"][name] == original["files_sha256"][name],
                    "DEV wrapper changed canonical cases, references or exclusions.")
    requests = [gate.rows(root / "requests-json.jsonl") for root in roots]
    def without_tokens(rows):
        return [{key: value for key, value in row.items()
                 if key not in {"prefix_ids", "prefix_ids_sha256"}} for row in rows]
    require(without_tokens(requests[0]) == without_tokens(requests[1])
            and all(row["split"] == "dev" for row in requests[0]),
            "DEV wrapper changed saved inputs, messages or ordered cohort; only native token IDs may differ.")


def within_drop(before, after, tolerance):
    return before is not None and after is not None and before - after <= tolerance + 1e-12


def quality_without_ui(selection, models, role, config, deployment, model_identity):
    if models[role].get("runtime_family") == gemma.FAMILY:
        from gemma_final_freeze import quality_without_ui as verify_gemma_quality
        return verify_gemma_quality(selection, models, role, config, deployment, model_identity)
    quality = selection.get("quality_without_ui")
    require(isinstance(quality, dict) and set(quality) == {
        "native_sft_summaries", "selected_native_summary", "selected_deployed_summary", "subgroup_review"},
        "Bind actual full-DEV quality evidence; a UI-independent boolean is not a selection decision.")
    candidates = quality["native_sft_summaries"]
    require(isinstance(candidates, dict) and set(candidates) == set(small.data.PROFILES),
            "Quality-only selection must include both registered small SFT candidates.")
    summaries = {}
    for name, reference in candidates.items():
        summary, report = verified_dev_summary(reference)
        require(report["numerical_regime"] == "native_bf16_fp32_lora" and report["schema_constrained"] is False
                and summary["model_identity"]["base"] == small.data.PROFILES[name]
                and summary["model_identity"]["kind"] == "sft",
                "Compare the registered native SFT families before a UI-independent choice.")
        summaries[name] = summary
    chosen = summaries[config["model"]]
    require(chosen["minimum_numeric_quality_gate"]["passed"], "The selected SFT failed its unchanged DEV quality gate.")
    score = chosen["macro_action_f1"]["value"]
    require(isinstance(score, (int, float)) and math.isfinite(score), "Missing finite selection score.")
    competitors = [summary for name, summary in summaries.items()
                   if name != config["model"] and summary["minimum_numeric_quality_gate"]["passed"]]
    for other in competitors:
        difference = score - other["macro_action_f1"]["value"]
        require(difference > 0.02 and not math.isclose(difference, 0.02, rel_tol=0, abs_tol=1e-12),
                "Actual UI latency is required when a passing competitor is within 2 percentage points or better.")
    native, native_report = verified_dev_summary(quality["selected_native_summary"])
    product, product_report = verified_dev_summary(quality["selected_deployed_summary"])
    identity = models[role]["identity"]
    require(all(native["model_identity"].get(key) == value for key, value in identity.items())
            and native["model_identity"]["base"] == config["base"]
            and native_report["numerical_regime"] == "native_bf16_fp32_lora"
            and native_report["schema_constrained"] is False,
            "Selected native DEV evidence belongs to another model or numerical regime.")
    require(all(chosen["model_identity"].get(key) == value for key, value in models["sft"]["identity"].items()),
            "The selected family's SFT evidence belongs to another trained parent.")
    require(role != "sft" or quality["selected_native_summary"] == candidates[config["model"]],
            "Selected SFT must use the same candidate quality evidence.")
    require(native["minimum_numeric_quality_gate"]["passed"] and product["minimum_numeric_quality_gate"]["passed"],
            "Selected native and deployed models must both pass the complete DEV quality gate.")
    product_identity = product["model_identity"]
    require(product_identity["base"] == config["base"] and product_identity["kind"] == identity["kind"]
            and product_identity["adapter_sha256"] == identity["adapter_sha256"]
            and product_identity["gguf_sha256"] == model_identity["model_sha256"]
            and product_identity["export_receipt_sha256"] == deployment["export_receipt"]["sha256"]
            and product_report["numerical_regime"] == deployment["numerical_regime"]
            and product_report["schema_constrained"] is True,
            "Full deployed DEV evidence belongs to another GGUF, adapter, export or runtime regime.")
    require(all(summary["cohort_mapping"] == chosen["cohort_mapping"]
                for summary in [*summaries.values(), native, product]), "Quality evidence uses different DEV cohorts.")
    for name in (None, "canonical_dev322", "supplemental_dev149"):
        before = native["overall"] if name is None else native["cohorts"][name]["metrics"]
        after = product["overall"] if name is None else product["cohorts"][name]["metrics"]
        require(all(within_drop(before[key]["value"], after[key]["value"], 0.02) for key in ("precision", "recall"))
                and before["model_preservation"]["numerator"] - after["model_preservation"]["numerator"] <= 2,
                "The deployed model exceeds the registered native quality-drop allowance.")
        if role == "sft_dpo" and name is not None:
            parent = chosen["cohorts"][name]["metrics"]
            require(all(within_drop(parent[key]["value"], before[key]["value"], 0) for key in ("precision", "model_preservation"))
                    and within_drop(parent["recall"]["value"], before["recall"]["value"], 0.01)
                    and sum(v for k, v in before["status_counts"].items() if k != "completed")
                    <= sum(v for k, v in parent["status_counts"].items() if k != "completed"),
                    "Selected DPO violates the existing parent-quality adoption rule.")
    require(role != "sft_dpo" or native["macro_action_f1"]["value"] > score,
            "Selected DPO must improve its same-parent macro action F1.")
    review = quality["subgroup_review"]
    require(isinstance(review, dict) and review.get("status") == "reviewed_pass"
            and isinstance(review.get("rationale"), str) and review["rationale"].strip()
            and review.get("evidence"), "Actual preservation, unknown-speaker and latest-correction review is missing.")
    references = [*candidates.values(), quality["selected_native_summary"], quality["selected_deployed_summary"], *review["evidence"]]
    require(all(reference in selection["evidence"] for reference in references), "Selection omitted its actual quality review evidence.")
    for reference in review["evidence"]:
        gate.bound(reference)
    return "only_numeric_quality_candidate" if not competitors else "macro_f1_margin_over_two_percentage_points"


def register(prepared, model_metadata, selection, actor_registry, exposure, output):
    output = fresh(output)
    prepared_binding, model_binding = binding(prepared), binding(model_metadata)
    prepared, manifest = read_bound(prepared_binding), read_bound(model_binding)
    require(prepared.get("version") == VERSION and prepared["status"] == "prepared_not_selected_not_body_authorized"
            and prepared["implementation"] == implementation() and prepared["body_access_authorized"] is False,
            "The unprivileged metadata preparation changed.")
    backend = model_backend(manifest)
    require(manifest["status"] == "model_receipts_ready_not_body_authorized"
            and manifest["body_access_authorized"] is False and manifest["final_test_bodies_read"] is False,
            "Use actual completed model metadata receipts.")
    models = {role: read_bound(value) for role, value in manifest["models"].items()}
    meta = read_bound(manifest["preparation"])
    config = read_bound(meta["configuration"])
    backend.verify_model_set(models, meta["tokenizer"], prepared["training_dataset_manifest"],
                           manifest["dpo_omission"], manifest["inference_policy"])
    for model in models.values():
        gate.model_runtime(model)
    selection_binding, actors_binding, exposure_binding = binding(selection), binding(actor_registry), binding(exposure)
    role = selected_model(read_bound(selection_binding), models)
    product_plan = deployment_plan(backend)
    auxiliary_sources = deployment_sources()
    actors = legacy.actors_from(read_bound(actors_binding), read_bound)
    reviewed_exposure = read_bound(exposure_binding)
    exposure_review(reviewed_exposure, prepared)
    registry = read_bound(prepared["original_icl_registry"])
    release = read_bound(prepared["training_dataset_manifest"])
    release_binding = {**registry["release_binding"], "manifest": prepared["training_dataset_manifest"],
                       "train_cases_sha256": release["files_sha256"]["cases/train.jsonl"],
                       "dev_cases_sha256": release["files_sha256"]["cases/dev.jsonl"]}
    if backend is gemma:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer"]["path"], local_files_only=True, trust_remote_code=False)
    else:
        tokenizer = small.data.tokenizer_from(config["assets_path"], config["model"])
    demos = fixed_demonstrations(read_bound(prepared["fixed_train_cases"]),
                                registry["conditions"]["train_icl"]["demonstrations"], tokenizer, release_binding)
    output.mkdir(parents=True)
    conditions = write(output / "conditions.json", {
        "version": VERSION, "status": "registered_not_body_authorized", "body_access_authorized": False,
        "selection_used_development_outputs": False, "selection": "preserve_original_fixed_train_pair_no_reselection",
        "fixed_demonstration_ids": small.DEMONSTRATION_IDS, "original_registry": prepared["original_icl_registry"],
        "release_binding": release_binding, "tokenizer": meta["tokenizer"],
        "conditions": gate.conditions.condition_specs(demos), "inference_policy": manifest["inference_policy"],
    })
    core = legacy.load_core()
    old_revision = read_bound({"path": str(legacy.OLD_DRAFT / "revision.json"), "sha256": legacy.OLD_DRAFT_SHA})
    originals = {name.removesuffix(".json"): read_bound({"path": str(legacy.OLD_DRAFT / name), "sha256": expected})
                 for name, expected in old_revision["files_sha256"].items()}
    require(set(originals) == set(core.LEAF_ORDER), "The original eight preregistered policy leaves changed.")
    replacements = {}
    for source in (originals["admission-policy"]["current_metadata_bindings"], prepared["original_metadata"]):
        replacements.update({str(Path(reference["path"]).resolve()): prepared["metadata"][key]
                             for key, reference in source.items()})
    written, basis = {}, {}

    def rebind(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                path = Path(value["path"])
                path = (path if path.is_absolute() else ROOT / path).resolve()
                if str(path) in replacements:
                    return replacements[str(path)]
                if path.parent == legacy.OLD_DRAFT and path.stem in written:
                    return written[path.stem]
                if any(path.is_relative_to(ROOT / name) for name in ("docs", "ml", "server", "app")):
                    result = binding(path)
                    if path.is_relative_to(ROOT / "docs"):
                        basis[str(path.relative_to(ROOT))] = result["sha256"]
                    return result
            return {key: rebind(child) for key, child in value.items()}
        return [rebind(child) for child in value] if isinstance(value, list) else value

    run_roles = [value for value in ROLES if value != "sft_dpo" or value in models]
    for name in core.LEAF_ORDER:
        leaf = rebind(copy.deepcopy(originals[name]))
        require(leaf["status"] == "draft" and leaf["body_access_authorized"] is False, "Original draft authority changed.")
        leaf.update(version=leaf["version"] + "-small-final-1", created_at=gate.now(),
                    original_draft=binding(legacy.OLD_DRAFT / (name + ".json")), model_metadata=model_binding)
        if name == "generator-recipe":
            leaf["pre_body_gate"].update(
                metadata_exposure_binding="Final unified TRAIN1261/DEV471 exposure metadata only; complete exact coordinator readiness and freeze remain required.",
                models_and_runs="Freeze all available registered family roles, fixed TRAIN ICL examples and full-cohort8192 policy. DPO omission requires its actual fully reviewed mining receipt.")
        elif name == "exclusion-policy":
            leaf["budget_policy"].update(native_prefix_limit=7680, native_total_limit=8192, canonical_visible_input_limit=3584,
                common="Any canonical input over3584 or any role prefix over7680 fails the whole preparation; no cropping or per-role case omissions.",
                ICL="The original two TRAIN examples are the only added content. All available roles share the same ordered full cohort.")
        elif name == "runtime-policy":
            runtime_reference = models["native_base"]["artifact_manifest"]
            runtime = read_bound(runtime_reference)
            libraries = config["libraries"] if backend is gemma else {key: value for key, value in meta["environment"].items() if value is not None}
            absent = config["absent_libraries"] if backend is gemma else [key for key, value in meta["environment"].items() if value is None]
            native_pin = meta["tokenizer"]["native_pin"] if backend is gemma else binding(Path(config["assets_path"]) / "assets.json")
            architecture = config["model"]["class"] if backend is gemma else config["base"]["architecture"]
            leaf.update(tokenizer=meta["tokenizer"], generation={**config["generation"], "enable_thinking": False},
                        prompt_sha256=sha(gate.contract.PROMPT_PATH), contract_sha256=sha(gate.contract.__file__),
                        formats_sha256=sha(gate.formats.__file__), inference_policy=manifest["inference_policy"],
                        libraries=libraries, absent_libraries=absent, native_base_pin=native_pin,
                        registered_environment_reference={"actual_loading_runtime": runtime_reference, "gpu": runtime.get("gpu"),
                                                          "scope": "Actual selected-family native loading evidence; execution must retain the frozen libraries and BF16 regime."},
                        native_load_options={"class": architecture, "dtype": "bfloat16", "adapter_dtype": "float32",
                                             "quantization": "none", "attn_implementation": "sdpa", "trust_remote_code": False,
                                             "seed": config["training"]["seed"], "cuda_matmul_allow_tf32": False,
                                             "cudnn_allow_tf32": False, "cudnn_benchmark": False, "model_eval": True})
            leaf["local_deployment_comparison"]["registered_numerical_regime"] = "Selected native BF16 adapter merged into its pinned base, then separately evaluated GGUF; app context4096, final native context8192."
            if backend is gemma:
                leaf["native_load_options"]["experts_implementation"] = "eager"
                leaf["local_deployment_comparison"]["registered_numerical_regime"] = "Native-derived Q4 or prospectively registered direct Q8_0 base, as explicitly bound by the selected deployment policy, with separate FP32 LoRA at id0/scale1; app context4096, final native context8192."
        elif name == "import-policy":
            leaf.update(normalizer_sha256=sha(ROOT / "server/app/asr/scribe.py"),
                        document_importer_sha256=sha(ROOT / "app/src/core/transcription.ts"))
        elif name == "metrics-policy":
            leaf["evaluator_sha256"] = sha(gate.evaluation.__file__)
            leaf["reporting_implementation_sha256"] = {
                path: sha(ROOT / path) for path in [*leaf["reporting_implementation_sha256"], str(Path(backend.__file__).resolve().relative_to(ROOT))]}
            leaf["paired_comparisons"].update(canonical_roles=run_roles, primary_table="full_cohort_all_roles",
                rule="Every available role uses the exact same full cohort. No ICL subset table or per-role omissions; retain all failed requests.")
        elif name == "admission-policy":
            leaf.update(current_metadata_bindings=prepared["metadata"], actors=actors,
                        actor_registry_status="actual_roles_bound_not_body_authorization", actual_actor_registry_evidence=actors_binding,
                        exposed_text_hashes=reviewed_exposure["exposed_text_hashes"], final_exposure_review=exposure_binding,
                        exposure_inventory_evidence=reviewed_exposure["exposure_inventory_evidence"],
                        metadata_relation="Final1261 exposure wrapper preserves the original104slots,16windows and fallback order; no body authority is granted.")
        written[name] = write(output / "policies" / (name + ".json"), leaf)
    for path in (backend.CONTEXT_PLAN, ROOT / "docs/plans/final-runtime-experiments-2026-10-25.md", product_plan,
                 UI_AMENDMENT, SELECTION_PLAN, DPO_PLAN):
        basis[str(path.relative_to(ROOT))] = sha(path)
    if backend is gemma:
        from gemma_q8_product import PLAN as q8_plan
        basis[str(q8_plan.relative_to(ROOT))] = sha(q8_plan)
    if backend is gemma and read_bound(selection_binding).get("selection_contract") is not None:
        from gemma_final_freeze import DESCRIPTIVE_PLAN
        basis[str(DESCRIPTIVE_PLAN.relative_to(ROOT))] = sha(DESCRIPTIVE_PLAN)
    if backend is gemma and read_bound(selection_binding).get("product_selection_contract") is not None:
        from gemma_final_freeze import PRODUCT_PLAN
        basis[str(PRODUCT_PLAN.relative_to(ROOT))] = sha(PRODUCT_PLAN)
    revision = write(output / "policies/revision.json", {
        "version": VERSION, "status": "draft", "body_access_authorized": False,
        "files_sha256": {name + ".json": reference["sha256"] for name, reference in written.items()},
        "original_revision": {"path": str(legacy.OLD_DRAFT / "revision.json"), "sha256": legacy.OLD_DRAFT_SHA},
    })
    registration = {
        "version": VERSION, "status": "registered_not_body_authorized", "body_access_authorized": False,
        "reserved_test_bodies_read": False, "implementations": implementation(), "prepared": prepared_binding,
        "model_metadata": model_binding, "models": manifest["models"], "dpo_omission": manifest["dpo_omission"],
        "inference_policy": manifest["inference_policy"], "selection_receipt": selection_binding, "selected_model": role,
        "actor_registry": actors_binding, "actors": actors, "exposure_review": exposure_binding,
        "conditions": conditions, "metadata": prepared["metadata"], "training_dataset_manifest": prepared["training_dataset_manifest"],
        "draft_revision": revision, "controller": write(output / "controller.json", gate.code_sources()),
        "basis_documents": basis, "numerical_targets": core.pinned("numeric_targets"), "run_roles": run_roles,
        "deployment_evaluation_plan": binding(product_plan), "deployment_evaluation_sources_sha256": auxiliary_sources,
    }
    reference = write(output / "registration.json", registration)
    write(output / "readiness-template.json", readiness_template(registration, reference))
    return reference


def authorization(window_ids):
    return {"basis": "existing_user_project_authorization", "assemble_final_metadata": False,
            "acquire_only_after_independent_parent_readback": False, "window_ids": window_ids,
            "window_seconds": 90, "max_asr_attempts_per_window": 1, "encoded_byte_budget": 512 * 1024 * 1024,
            "pcm_byte_budget": 64 * 1024 * 1024, "automatic_billable_retry": False,
            "byte_budget_scope": "one cumulative ledger for all 16 fixed windows", "late_source_reallocation": False}


def readiness_template(registration, reference):
    windows = read_bound(registration["metadata"]["natural_windows"])
    return {"version": READINESS_VERSION, "status": "draft_not_authorization", "actor": "/root",
            "created_at": None, "protocol_id": None, "output": None, "builder": binding(__file__),
            "registration": reference, "selection_receipt": registration["selection_receipt"],
            "actor_registry": registration["actor_registry"], "exposure_review": registration["exposure_review"],
            **dict.fromkeys(EVIDENCE), "evidence_review": {"attestations": dict.fromkeys(legacy.inputs.ATTESTATIONS, False)},
            "authorization": authorization([row["window_id"] for row in windows])}


def ui_template():
    return {"version": UI_VERSION, "status": "draft_not_observed", "scope": "exposed_development_ui",
            "final_test_bodies_read": False, "deployment": None,
            "model": {"model_id": None, "model_sha256": None, "adapter_sha256": None},
            "launch_method": None, "automatic_only": True,
            "runtime_condition": "cold_and_warm_separate", **dict.fromkeys(UI_MEASUREMENTS),
            "manual_request_button_used": False, "public_copilot_toggle_present": False,
            "planned_case_ids": [], "outcomes": [],
            "evidence": {"visible_timing": [], "long_session": [], "shuka_split": [], "shuka_delete": [], "shuka_trim": []},
            "outcome_schema": {"case_id": "planned ID", "language": "en or ko", "status": "preview/no_action/suppressed/invalid/request_failure/not_issued",
                               "automatic_trigger": "actual boolean", "evidence": "one or more exact artifact bindings"},
            "interpretation": "Retain every planned outcome and failure. EN/KO automatic visible preview evidence demonstrates the product path; fourteen action/language quality cells remain a full-DEV claim. Timing evidence separates non-previews and HTTP-only measurements."}


def pending_ui_template():
    return {"version": PENDING_UI_VERSION, "status": "draft_not_observed", "actor": "/root",
            "scope": "exposed_development_ui", "final_test_bodies_read": False, "product_complete": False,
            "deployment": None, "model": None, "selection_receipt": None,
            "amendment": binding(UI_AMENDMENT), "reason": None, "evidence": [],
            "required_product_checks": PRODUCT_CHECKS}


def validate_ui(ui, deployment_binding, model_identity):
    require(ui.get("version") == UI_VERSION and ui.get("status") == "observed"
            and ui.get("scope") == "exposed_development_ui" and ui.get("final_test_bodies_read") is False
            and ui.get("deployment") == deployment_binding and ui.get("model") == model_identity
            and ui.get("launch_method") == "npm start" and ui.get("automatic_only") is True
            and ui.get("runtime_condition") == "cold_and_warm_separate"
            and ui.get("manual_request_button_used") is False and ui.get("public_copilot_toggle_present") is False,
            "Actual automatic UI evidence belongs to another model/deployment or interaction path.")
    planned, outcomes = ui["planned_case_ids"], ui["outcomes"]
    require(planned and len(planned) == len(set(planned)) and [row["case_id"] for row in outcomes] == planned,
            "Every planned UI outcome, including failure/not-issued, must be retained in order.")
    require(all(row["language"] in {"en", "ko"}
                and row["status"] in {"preview", "no_action", "suppressed", "invalid", "request_failure", "not_issued"}
                and type(row["automatic_trigger"]) is bool and row["evidence"] for row in outcomes),
            "UI outcomes need actual status, language, automatic trigger and evidence.")
    require({row["language"] for row in outcomes if row["status"] == "preview" and row["automatic_trigger"]} == {"en", "ko"},
            "Observe the automatic visible preview path in both languages.")
    require(set(ui["evidence"]) == {"visible_timing", "long_session", "shuka_split", "shuka_delete", "shuka_trim"}
            and all(ui["evidence"].values()), "Actual timing, long-session and Shuka operation evidence is missing.")
    for references in [*(row["evidence"] for row in outcomes), *ui["evidence"].values()]:
        for reference in references:
            gate.bound(reference)
    for key in UI_MEASUREMENTS:
        gate.bound(ui[key])


def validate_product_validation(ready, registration, models, role, config, identity):
    reference = ready["selected_model_ui_receipt"]
    ui = read_bound(reference)
    pending = ui.get("version") == PENDING_UI_VERSION
    if pending:
        require(set(ui) == set(pending_ui_template()) and ui["status"] == "pending_product_validation"
                and ui["actor"] == "/root" and ui["scope"] == "exposed_development_ui"
                and ui["final_test_bodies_read"] is False and ui["product_complete"] is False
                and ui["deployment"] == ready["selected_deployment_receipt"] and ui["model"] == identity
                and ui["selection_receipt"] == registration["selection_receipt"]
                and ui["amendment"] == binding(UI_AMENDMENT) and ui["required_product_checks"] == PRODUCT_CHECKS
                and isinstance(ui["reason"], str) and ui["reason"].strip() and ui["evidence"],
                "Pending product validation must bind the real blocker, exact deployment and selection without UI success claims.")
        for evidence in ui["evidence"]:
            gate.bound(evidence)
        reason = quality_without_ui(read_bound(registration["selection_receipt"]), models, role, config,
                                    read_bound(ready["selected_deployment_receipt"]), identity)
    else:
        validate_ui(ui, ready["selected_deployment_receipt"], identity)
        reason = "actual_ui_evidence_observed"
    attestations = {key: not (pending and key in UI_ATTESTATIONS) for key in legacy.inputs.ATTESTATIONS}
    require(ready["evidence_review"]["attestations"] == attestations,
            "Coordinator attestations must retain pending UI checks as false and completed scientific checks as true.")
    return {"status": ui["status"], "receipt": reference, "selection_basis": reason,
            "product_completion_asserted": False, "amendment": binding(UI_AMENDMENT)}


def validate_readiness_evidence(ready, registration, models, role, config, meta):
    for field in EVIDENCE:
        gate.bound(ready[field])
    identity = validate_deployment(ready["selected_deployment_receipt"], models[role], config)
    if models[role].get("runtime_family") == gemma.FAMILY:
        from gemma_final_freeze import DESCRIPTIVE_BASIS, validate_native_readiness
        validate_native_readiness(ready, models, config, meta)
        product = validate_product_validation(ready, registration, models, role, config, identity)
        selection = read_bound(registration["selection_receipt"])
        basis = product["selection_basis"]
        if product["status"] != "pending_product_validation":
            basis = quality_without_ui(selection, models, role, config,
                                       read_bound(ready["selected_deployment_receipt"]), identity)
        if basis == DESCRIPTIVE_BASIS:
            contract = selection["selection_contract"]
            product["development_selection"] = {"basis": basis, **{key: contract[key] for key in (
                "strict_subgroup_gate_passed", "failed_registered_typed_groups", "subgroup_review",
                "limitations", "quality_success_claimed")}}
        if selection.get("product_selection_contract") is not None:
            product.update(registered_product_drop_gate_passed=False, quality_success_claimed=False,
                           product_development_selection=copy.deepcopy(selection["product_selection_contract"]))
        return product
    product = validate_product_validation(ready, registration, models, role, config, identity)
    for field in ("native_environment_receipt", "native_runtime_receipt"):
        observed = read_bound(ready[field])
        require(observed["environment"] == meta["environment"] and observed["base"] == config["base"]
                and observed["dtype"] == "bfloat16" and observed["attention"] == "sdpa"
                and observed["device"] == "cuda" and bool(observed.get("gpu"))
                and not any(observed["loading"].get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                "Actual native environment or loading regime differs from the selected small family.")
        require(normalized(ready[field]) == models["native_base"]["artifact_manifest"],
                "Use the already verified actual native runtime.json, not an authored runtime assertion.")
    return product


def validate_deployment(value, model, config):
    if model.get("runtime_family") == gemma.FAMILY:
        from gemma_final_deployment import validate_deployment as validate_gemma_deployment
        return validate_gemma_deployment(value, model, config)
    import sys

    for path in (ROOT / "server", ROOT / "server/scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_local_copilot_small_final as deployment

    location = gate.bound(value).parent
    gateway, _, command, report = deployment.check(location)
    identity = model["identity"]
    require(report["version"] == "local-small-actions-final-v1" and report["purpose"] == "ui_performance"
            and report["training_adapter_sha256"] == identity["adapter_sha256"] and report["kind"] == identity["kind"]
            and report["native_preparation"]["sha256"] == identity["configuration_sha256"]
            and report["base"] == config["base"] and gateway.adapter_sha256 is None
            and gateway.output_format == "json" and gateway.capture_runtime_evidence is False
            and command[command.index("-c") + 1] == "4096", "UI deployment differs from the selected native-trained model or app budget.")
    if identity["kind"] == "dpo":
        trained = read(gate.bound(model["artifact_manifest"]).parent / "dpo_configuration.json")
        require(report["parent_adapter_sha256"] == trained["parent"]["adapter_sha256"]
                and report["dpo_configuration_sha256"] == sha(gate.bound(model["artifact_manifest"]).parent / "dpo_configuration.json"),
                "UI deployment changed the selected DPO parent.")
    return {"model_id": gateway.model_id, "model_sha256": gateway.model_sha256, "adapter_sha256": None}


def deployed_policy(context):
    registration, ready = context["registration"], context["ready"]
    role = context["selected_model"]
    model = context["models"][role]
    if model.get("runtime_family") == gemma.FAMILY:
        from gemma_final_deployment import policy
        return policy(context)
    deployment = read_bound(ready["selected_deployment_receipt"])
    directory = gate.bound(ready["selected_deployment_receipt"]).parent
    gateway = read(directory / "gateway.json")
    command = read(directory / "command.json")
    config = read_bound(model["inference_configuration"])
    meta = read_bound(model["preparation_manifest"])
    from serve_small_actions_final import runtime_schema

    model_identity = {
        "runtime_family": "small-actions-final-gguf", "role": "deployed_selected", "model": config["model"],
        "model_id": gateway["model_id"], "model_sha256": gateway["model_sha256"], "adapter_sha256": None,
        "training_adapter_sha256": deployment["training_adapter_sha256"], "kind": deployment["kind"],
        **{key: deployment[key] for key in ("parent_adapter_sha256", "dpo_configuration_sha256") if key in deployment},
    }
    value = {
        "version": DEPLOYMENT_POLICY_VERSION, "status": "registered_before_test", "scope": "final_test",
        "role": "deployed_selected", "protocol_id": ready["protocol_id"], "plan": registration["deployment_evaluation_plan"],
        "native_core_registration": context["registration_binding"], "selected_native_run_id": role + "-json",
        "selected_native_model": registration["models"][role], "selected_native_identity": model["identity"],
        "selected_native_identity_sha256": digest(model["identity"]), "deployment": ready["selected_deployment_receipt"],
        "gateway": binding(directory / "gateway.json"), "export_receipt": deployment["export_receipt"],
        "native_inference_configuration": model["inference_configuration"], "native_tokenizer": meta["tokenizer"],
        "model_identity": model_identity, "model_file": binding(gateway["model_path"]),
        "chat_template": binding(gateway["chat_template_path"]), "runtime_binary": binding(command[0]),
        "command": binding(directory / "command.json"), "runtime_registry": deployment["runtime_registry"],
        "runtime_files_sha256": deployment["runtime_files_sha256"], "numerical_regime": deployment["numerical_regime"],
        "canonical_prefix_limit": 3584, "context_tokens": 4096, "output_tokens": 512,
        "request_settings": {
            "model": gateway["model_id"], "temperature": 0, "seed": 42, "top_k": 40, "top_p": 0.95,
            "min_p": 0.05, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False},
            "cache_prompt": gateway["cache_prompt"], "stream": False, "lora": [],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "next_edit", "strict": True, "schema": runtime_schema(gate.contract.OUTPUT_SCHEMA)}},
        },
        "request_deadline_seconds": 30, "schema_constrained": True, "all_roles_same_case_ids": True,
        "overflow_policy": "fail_whole_preparation", "max_attempts_per_request": 1,
        "native_prefix_parity_required": True, "all_request_failures_in_denominator": True,
        "final_test_used_for_tuning": False, "source_sha256": registration["deployment_evaluation_sources_sha256"],
        "body_access_authorized": False,
    }
    from infer_deployed_task_test_actions_v3 import validate_policy

    native_run = {"id": role + "-json", "role": role, "format": "json",
                  "condition": context["conditions"]["conditions"]["canonical_full_context"]}
    validate_policy(value, model, native_run, config, meta["tokenizer"])
    return value


def preflight(registration, registration_sha256, readiness, readiness_sha256, output):
    output = fresh(output)
    reference = {"path": str(Path(registration).resolve()), "sha256": registration_sha256}
    readiness_binding = {"path": str(Path(readiness).resolve()), "sha256": readiness_sha256}
    ready = read_bound(readiness_binding)
    require(ready.get("version") == READINESS_VERSION and ready.get("status") == "authorized_for_final_metadata_assembly"
            and ready.get("actor") == "/root" and ready.get("builder") == binding(__file__)
            and ready.get("registration") == reference, "Exact coordinator readiness is required; draft metadata grants no authority.")
    approved_at = datetime.fromisoformat(ready["created_at"])
    require(approved_at.tzinfo is not None and approved_at <= datetime.now(timezone.utc)
            and ready["output"] == str(output) and isinstance(ready.get("protocol_id"), str) and ready["protocol_id"].strip(),
            "Readiness timestamp, output or protocol differs.")
    registration = read_bound(reference)
    require(registration.get("version") == VERSION and registration["status"] == "registered_not_body_authorized"
            and registration["body_access_authorized"] is False and registration["implementations"] == implementation(),
            "Registered builder or authority changed.")
    for key in ("selection_receipt", "actor_registry", "exposure_review"):
        require(ready[key] == registration[key], "Readiness binds another " + key)
    prepared = read_bound(registration["prepared"])
    require(registration["metadata"] == prepared["metadata"]
            and registration["training_dataset_manifest"] == prepared["training_dataset_manifest"], "Metadata allocation changed.")
    actors = legacy.actors_from(read_bound(registration["actor_registry"]), read_bound)
    require(actors == registration["actors"], "Actual actors changed.")
    reviewed_exposure = read_bound(registration["exposure_review"])
    exposure_review(reviewed_exposure, prepared)
    models = {role: read_bound(value) for role, value in registration["models"].items()}
    model_metadata = read_bound(registration["model_metadata"])
    require(all(registration[key] == model_metadata[key] for key in ("models", "dpo_omission", "inference_policy")),
            "Actual model metadata changed after registration.")
    meta = read_bound(model_metadata["preparation"])
    config = read_bound(meta["configuration"])
    backend = model_backend(model_metadata)
    backend.verify_model_set(models, meta["tokenizer"], registration["training_dataset_manifest"],
                           registration["dpo_omission"], registration["inference_policy"])
    for model in models.values():
        gate.model_runtime(model)
    role = selected_model(read_bound(registration["selection_receipt"]), models)
    require(role == registration["selected_model"], "Selected role changed.")
    require(read_bound(registration["controller"]) == gate.code_sources(), "Controller source changed; register a replacement.")
    require(registration["deployment_evaluation_plan"] == binding(deployment_plan(backend))
            and registration["deployment_evaluation_sources_sha256"] == deployment_sources(),
            "Prospective product evaluation policy or implementation changed.")
    for path, expected in registration["basis_documents"].items():
        require(sha(ROOT / path) == expected, "Basis documentation changed after registration.")
    revision = read_bound(registration["draft_revision"])
    drafts = {name.removesuffix(".json"): read_bound({"path": str(gate.bound(registration["draft_revision"]).parent / name), "sha256": expected})
              for name, expected in revision["files_sha256"].items()}
    core = legacy.load_core()
    require(set(drafts) == set(core.LEAF_ORDER) and all(row["status"] == "draft" and row["body_access_authorized"] is False for row in drafts.values()),
            "Registered policy leaves changed authority or membership.")
    admission = drafts["admission-policy"]
    require(admission["actors"] == actors and admission["current_metadata_bindings"] == registration["metadata"]
            and admission["exposed_text_hashes"] == reviewed_exposure["exposed_text_hashes"]
            and admission["exposure_inventory_evidence"] == reviewed_exposure["exposure_inventory_evidence"],
            "Actual actor or final exposure review differs from the admission policy.")
    for leaf in drafts.values():
        for value in core.bindings_in(leaf):
            gate.bound(normalized(value))
    for path, expected in drafts["metrics-policy"]["reporting_implementation_sha256"].items():
        require(sha(ROOT / path) == expected, "Reporting implementation changed.")
    targets = read_bound(registration["numerical_targets"])
    require(drafts["metrics-policy"]["decision_thresholds"]["registered_targets"] == registration["numerical_targets"]
            or normalized(drafts["metrics-policy"]["decision_thresholds"]["registered_targets"]) == registration["numerical_targets"],
            "Registered numerical target binding changed.")
    require(drafts["metrics-policy"]["decision_thresholds"]["numerical_targets"] == targets["numerical_targets"]
            and drafts["metrics-policy"]["decision_thresholds"]["exact_arithmetic"] == targets["exact_arithmetic"],
            "Preregistered numerical criteria changed.")
    allocation = gate.metadata_allocation(gate.bound(registration["metadata"]["metadata_manifest"]))
    require(len(allocation["roster"]) == 104 and len(allocation["windows"]) == 16, "Preregistered allocation changed.")
    gate.load_source_resolution(registration["metadata"], datetime.now(timezone.utc))
    expected_auth = authorization([row["window_id"] for row in allocation["windows"]])
    expected_auth.update(assemble_final_metadata=True, acquire_only_after_independent_parent_readback=True)
    require(ready["authorization"] == expected_auth, "Exact acquisition limits are missing.")
    product = validate_readiness_evidence(ready, registration, models, role, config, meta)
    return {"registration": registration, "registration_binding": reference, "ready": ready,
            "ready_binding": readiness_binding, "drafts": drafts, "models": models, "selected_model": role,
            "conditions": read_bound(registration["conditions"]), "closure": read_bound(registration["controller"]),
            "product_validation": product}


def materialize(context, output, frozen_at):
    output.mkdir(parents=True, mode=0o700)
    registration, ready = context["registration"], context["ready"]
    shutil.copyfile(gate.bound(context["ready_binding"]), output / "readiness.json")
    readiness = binding(output / "readiness.json")
    snapshots = {}
    for name, expected in registration["basis_documents"].items():
        source, target = ROOT / name, output / "basis" / name
        require(sha(source) == expected, "Basis document changed during assembly.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        snapshots[str(source.resolve())] = binding(target)
    model_bindings = registration["models"]
    runtimes = {role: model["execution_runtime"] for role, model in context["models"].items()}
    resolved = {
        "root_body_access_freeze": {"protocol_id": ready["protocol_id"], "ownership": "root freeze.json binds these leaf hashes; no reverse hash"},
        "parent_policy_approval": readiness, "selected_model_identity": context["models"][context["selected_model"]]["identity"],
        "selection_receipt": registration["selection_receipt"], "development_selection_receipt": registration["selection_receipt"],
        "training_dataset_manifest": registration["training_dataset_manifest"], "selected_training_dataset_manifest": registration["training_dataset_manifest"],
        "model_receipts": model_bindings, "native_base_sft_sft_dpo_model_reload_receipts": model_bindings,
        "selected_execution_runtime_receipts": runtimes, "exact_per_model_remote_runtime_assemblies": runtimes,
        "final_controller_source_closure": registration["controller"], "final_effective_metadata_and_source_resolution": registration["metadata"],
        "development_derived_numeric_acceptance": registration["numerical_targets"], "actual_actor_registry": registration["actors"],
        "actual_actor_registry_and_access_evidence": {"actors": registration["actors"], "registry": registration["actor_registry"], "parent_readiness": readiness},
        "explicit_acquisition_authorization_receipt": readiness,
        "final_training_exposure_receipt": context["drafts"]["admission-policy"]["exposure_inventory_evidence"],
    }
    leaves = {}
    draft_dir = gate.bound(registration["draft_revision"]).parent

    def rebind(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                path = Path(value["path"])
                path = (path if path.is_absolute() else ROOT / path).resolve()
                if str(path) in snapshots:
                    return snapshots[str(path)]
                if path.parent == draft_dir and path.stem in leaves:
                    return leaves[path.stem]
            return {key: rebind(child) for key, child in value.items()}
        return [rebind(child) for child in value] if isinstance(value, list) else value

    for name in legacy.load_core().LEAF_ORDER:
        leaf = rebind(copy.deepcopy(context["drafts"][name]))
        unresolved = leaf.pop("unresolved_bindings", {})
        if "frozen_runtime_asr_import_metrics_policies" in unresolved:
            resolved["frozen_runtime_asr_import_metrics_policies"] = {key: leaves[key + "-policy"] for key in ("runtime", "asr", "import", "metrics")}
        require(set(unresolved) <= set(resolved), "Unhandled original policy binding.")
        leaf["resolved_bindings"] = {key: resolved[key] for key in unresolved}
        leaf.update(status="frozen", frozen_at=frozen_at, body_access_authorized=True, protocol_id=ready["protocol_id"],
                    parent_readiness=readiness, original_draft=binding(draft_dir / (name + ".json")))
        leaf.pop("draft_limit", None)
        if name == "runtime-policy":
            leaf.update(environment_receipt=ready["native_environment_receipt"], actual_native_runtime_receipt=ready["native_runtime_receipt"])
            leaf["local_deployment_comparison"].update(selected_deployment_receipt=ready["selected_deployment_receipt"],
                                                       selected_model_ui_receipt=ready["selected_model_ui_receipt"],
                                                       product_validation=context["product_validation"])
        if name == "admission-policy":
            leaf.update(actor_registry_status="frozen_actual_parent_assessed_roles", dpo_omission=registration["dpo_omission"])
        leaves[name] = write(output / (name + ".json"), leaf)
    runs = [{"id": role + "-json", "role": role, "format": "json", "condition": context["conditions"]["conditions"][
        "train_icl" if role == "base_icl" else "canonical_full_context"]} for role in registration["run_roles"]]
    auxiliary = write(output / "deployment-evaluation-policy.json", deployed_policy(context))
    freeze = {
        "version": gate.FREEZE_VERSION, "scope": "final_test", "status": "frozen", "protocol_id": ready["protocol_id"],
        "frozen_at": frozen_at, "body_access_authorized": True, "reserved_bodies_read_before_freeze": False,
        "parent_independent_readback_before_body_instruction": True, "source_sha256": context["closure"], **registration["metadata"],
        "training_dataset_manifest": registration["training_dataset_manifest"], "selection_receipt": registration["selection_receipt"],
        "selected_model": context["selected_model"], "models": model_bindings,
        "policies": {name: leaves[name + "-policy"] for name in gate.POLICIES}, "runs": runs,
        "parent_readiness": readiness, "original_parent_readiness": context["ready_binding"], "builder": binding(__file__),
        "registration": context["registration_binding"], "quality_success_asserted_by_builder": False,
        "deployment_evaluation_policy": auxiliary,
        "product_validation": context["product_validation"],
    }
    if registration["dpo_omission"] is not None:
        freeze["dpo_omission"] = registration["dpo_omission"]
    write(output / "freeze.json", freeze)
    checked = gate.load_gate(output / "freeze.json")
    readback = {
        "status": "passed_metadata_gate_requires_parent_independent_readback", "freeze": checked["binding"],
        "registered_slots": len(checked["roster"]), "natural_windows": len(checked["windows"]),
        "run_roles": registration["run_roles"], "selected_model": context["selected_model"],
        "demonstration_case_ids": small.DEMONSTRATION_IDS, "reserved_bodies_read": False, "model_calls": 0,
        "quality_success_asserted": False, "dpo_omission": registration["dpo_omission"],
        "product_validation": context["product_validation"],
    }
    write(output / "gate-readback.json", readback)
    write(output / "assembly-manifest.json", {
        "version": VERSION, "status": "metadata_assembled_not_parent_body_instruction", "created_at": gate.now(),
        "registration": context["registration_binding"], "original_parent_readiness": context["ready_binding"],
        "basis_document_snapshots": snapshots, "reserved_bodies_read": False, "model_calls": 0,
        "files_sha256": {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()},
    })
    return readback


def assemble(**kwargs):
    context = preflight(**kwargs)
    output = Path(kwargs["output"]).resolve()
    frozen_at = gate.now()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="small-freeze-preflight-", dir=output.parent) as temporary:
        materialize(context, Path(temporary) / "assembled", frozen_at)
    try:
        result = materialize(context, output, frozen_at)
    except Exception as error:
        if output.exists():
            write(output / "assembly-failure.json", {"status": "failed_not_ready_for_body_access", "error": str(error)})
        raise
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    register_parser = commands.add_parser("register")
    for name in ("prepared", "model-metadata", "selection", "actor-registry", "exposure", "output"):
        register_parser.add_argument("--" + name, type=Path, required=True)
    for name in ("preflight", "assemble"):
        command = commands.add_parser(name)
        for field in ("registration", "readiness", "output"):
            command.add_argument("--" + field, type=Path, required=True)
        for field in ("registration-sha256", "readiness-sha256"):
            command.add_argument("--" + field, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = globals()[command](**args)
    if command == "preflight":
        result = {"status": "parent_readiness_preflight_passed_no_freeze_written", "body_access_authorized_by_this_command": False,
                  "selected_model": result["selected_model"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
