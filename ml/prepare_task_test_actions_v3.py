"""Sealed final-test preparation and grading, separate from exposed development.

Every body-reading entry point verifies a metadata-only freeze receipt first.
There is deliberately no downloader, ASR request, or model-selection command here.
"""

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import NAMESPACE_URL, uuid5
import wave

import evaluate_passive_actions_v3 as evaluation
import passive_actions as contract
import passive_formats_actions_v3 as formats
import prepare_actions_v3_conditions as conditions
import prepare_passive_actions_final_test as sealed_v1
import prepare_passive_actions_v3 as dataset
from context_policy import ContextBudgetError
from verify_release_export import Exporter, verify_export_delta, verify_operation

ROOT = Path(__file__).resolve().parents[1]
VERSION = "task-test-actions-v3-1"
FREEZE_VERSION = "task-test-actions-v3-freeze-1"
SCOPE = "final_test"
POLICIES = ("runtime", "asr", "import", "metrics", "admission")
MODEL_ROLES = {"native_base", "sft", "sft_dpo"}
RUN_ROLES = MODEL_ROLES | {"base_icl", "always_no_action", "consistent_history_rule"}
FULL_COHORT_FAMILIES = {"small-actions-final-v1", "gemma26-final-native-v1"}
UNKNOWN = {"unknown", "source_unavailable", "context_ineligible", "structural_state_unavailable"}
KNOWN = {"edit_required", "preserve"}
STRATA = {"natural_source_first", "natural_source_simulated_session", "authored_challenge"}
KO_COHORTS = ("kocw-tukorea-1445095-leechungseok", "kocw-konkuk-1113169-leesangyun")
KO_FALLBACK = "kocw-handong-1532697-kiminjung"
SOURCE_RESOLUTION_VERSION = "task-test-actions-v3-source-resolution-1"
TECHNICAL_FAILURES = {"publisher_metadata_unavailable", "media_locator_unresolved", "media_endpoint_unavailable",
                      "unsupported_media_transport", "verified_recording_too_short"}
require, packed, digest, sha256 = dataset.require, dataset.packed, dataset.digest, dataset.sha256
rows, write, write_rows = dataset.rows, dataset.write, dataset.write_rows
binding = evaluation.file_binding


def runtime_backend(family):
    if family == "small-actions-final-v1":
        import small_final_test_runtime
        return small_final_test_runtime
    if family == "gemma26-final-native-v1":
        import gemma_final_test_runtime
        return gemma_final_test_runtime
    raise ValueError("Unknown final native runtime family.")


def now():
    return datetime.now(timezone.utc).isoformat()


def opaque_source_id(window_id):
    require(isinstance(window_id, str) and bool(window_id.strip()), "Missing logical source window.")
    return str(uuid5(NAMESPACE_URL, "transcript-edit-copilot:task-test-actions-v3:source:" + window_id))


def read_json(path):
    return json.loads(Path(path).read_text())


def bound(value):
    path = Path(value["path"]).resolve()
    require(path.is_relative_to(ROOT), "Bound artifact escaped the workspace.")
    require(path.is_file() and sha256(path) == value["sha256"], "Bound artifact changed.")
    return path


def code_sources():
    sources = sealed_v1.source_closure([Path(__file__)])
    sources.update(sealed_v1.source_closure([ROOT / "ml/infer_task_test_actions_v3.py"]))
    acquisition = ROOT / "ml/acquire_task_test_actions_v3.py"
    if acquisition.is_file():
        sources.update(sealed_v1.source_closure([acquisition]))
    for name in (
        "server/app/asr/scribe.py", "ml/import_task_test_actions_v3.cjs",
        "ml/export_release_states.cjs", "app/src/core/transcription.ts",
        "app/src/core/document.ts", "app/src/core/webvtt.ts",
        "app/src/state/editor/selectors.ts", "app/package-lock.json",
    ):
        path = ROOT / name
        if path.is_file():
            sources[str(path.relative_to(ROOT))] = sha256(path)
    sources[str(contract.PROMPT_PATH.relative_to(ROOT))] = sha256(contract.PROMPT_PATH)
    for suffix in ("*.ts", "*.tsx"):
        for path in (ROOT / "app/src").rglob(suffix):
            sources[str(path.relative_to(ROOT))] = sha256(path)
    return dict(sorted(sources.items()))


def model_runtime(model, *, engineering_fixture_only=False):
    import assemble_task_test_actions_v3 as assembly
    import infer_task_test_actions_v3 as native
    if "execution_runtime" not in model:
        require(engineering_fixture_only is True, "Real final-test models require a frozen execution runtime.")
        return {"runtime_manifest_sha256": None, "source_sha256": native.sources()}
    path = bound(model["execution_runtime"])
    require(path.name == assembly.MANIFEST, "Unexpected execution runtime manifest.")
    runtime = assembly.load_runtime(path.parent, model["execution_runtime"]["sha256"])
    manifest = read_json(path)
    require(runtime.get("runtime_family") == model.get("runtime_family"), "Model receipt/runtime family differs.")
    if model.get("runtime_family") in FULL_COHORT_FAMILIES:
        require(runtime["inference_policy_sha256"] == model["inference_policy"]["sha256"],
                "Model receipt/runtime inference policy differs.")
    require(runtime["configuration_sha256"] == model["inference_configuration"]["sha256"]
            and runtime["preparation_manifest_sha256"] == model["preparation_manifest"]["sha256"],
            "Execution runtime differs from its frozen model preparation.")
    for name, origin in manifest["source_origin"].items():
        if origin == "final_only_overlay" or name in {"ml/" + item for item in assembly.SHARED}:
            require(runtime["source_sha256"][name] == sha256(ROOT / name),
                    "Remote overlay or shared predictor contract differs from the controller.")
    require(runtime["source_sha256"][assembly.ENTRY] == sha256(ROOT / assembly.ENTRY)
            and runtime["source_sha256"]["ml/assemble_task_test_actions_v3.py"] == sha256(assembly.__file__),
            "Final-only execution/assembly code changed.")
    return runtime


def metadata_allocation(metadata_manifest):
    path = bound(binding(metadata_manifest))
    metadata = read_json(path)
    require(metadata["reserved_final_test_bodies_read"] is False and metadata["generated_test_cases"] == 0,
            "Source resolution requires the original metadata-only registration.")
    names = {"reservations": "source-reservations-proposed.json", "roster": "request-slots-proposed.jsonl",
             "windows": "natural-window-proposal.json", "fallback": "kocw-1532697-recording-order.json"}
    paths = {key: path.parent / name for key, name in names.items()}
    for key, target in paths.items():
        require(sha256(target) == metadata["files_sha256"][names[key]], "Registered source metadata changed.")
    reservations, windows, fallback = (read_json(paths[key]) for key in ("reservations", "windows", "fallback"))
    groups = {row["source_group"]: row for row in reservations["groups"]}
    require(all(groups[name]["role"] == "primary" and groups[name]["language"] == "ko" for name in KO_COHORTS)
            and groups[KO_FALLBACK]["role"] == "metadata_failure_fallback_only"
            and KO_FALLBACK not in reservations["exposed_source_group_splits"]
            and fallback["source_group"] == KO_FALLBACK and fallback["role"] == "metadata_failure_fallback_only",
            "Korean fallback allocation differs from the original reservation.")
    candidates = fallback["first_eight_metadata_candidates"][:4]
    require(len(candidates) == 4 and len({r["publisher_recording_id"] for r in candidates}) == 4
            and all(r["body_sha256"] is None and r["media_and_caption_bodies_requested"] is False for r in candidates),
            "Fallback publisher recordings were selected after body access.")
    selected = {name: [row for row in windows if row["source_group"] == name] for name in KO_COHORTS}
    require(all(len(value) == 4 and all(row["language"] == "ko" and row["window_seconds"] == 90
                and row["body_opened"] is False and row["start_seconds"] is None for row in value)
                for value in selected.values()), "Korean logical window allocation changed.")
    return {"metadata_manifest": binding(path), "paths": paths, "reservations": reservations,
            "roster": rows(paths["roster"]), "windows": windows, "cohorts": selected, "fallback": candidates}


def validate_source_assessments(value, allocation):
    require(value["version"] == "task-test-actions-v3-source-assessments-1"
            and value["metadata_manifest_sha256"] == allocation["metadata_manifest"]["sha256"],
            "Technical assessments belong to another metadata registration.")
    require([row["source_group"] for row in value["cohorts"]] == [*KO_COHORTS, KO_FALLBACK],
            "Assess both primary cohorts in Java-then-biology order before fallback allocation.")
    for cohort in value["cohorts"]:
        name = cohort["source_group"]
        expected = (allocation["fallback"] if name == KO_FALLBACK else allocation["cohorts"][name])
        require([row["publisher_recording_id"] for row in cohort["recordings"]]
                == [row["publisher_recording_id"] for row in expected], "Technical assessment changed the fixed publisher recordings.")
        for record in cohort["recordings"]:
            require(record["status"] in {"metadata_available", "technical_failure"}
                    and record["before_words"] is True and record["media_body_read"] is False
                    and record["captions_or_transcript_read"] is False and record["asr_requested"] is False
                    and record["model_predictions_seen"] is False,
                    "Fallback decisions require metadata/header-only evidence before words, ASR or predictions.")
            recorded_at = datetime.fromisoformat(record["recorded_at"])
            require(recorded_at.tzinfo is not None and recorded_at <= datetime.now(timezone.utc),
                    "Invalid technical assessment timestamp.")
            require(record["evidence"], "Technical availability/failure requires preserved publisher evidence.")
            for evidence in record["evidence"]:
                bound(evidence)
            if record["status"] == "metadata_available":
                require(record.get("reason_code") is None and isinstance(record.get("media_url"), str)
                        and record["media_url"].startswith(("https://", "http://")),
                        "A usable publisher media locator is required; missing catalog duration alone is not failure.")
            else:
                require(record["reason_code"] in TECHNICAL_FAILURES, "Content/ASR/label quality cannot trigger a fallback.")
                if record["reason_code"] == "verified_recording_too_short":
                    require(type(record.get("duration_seconds")) in (int, float)
                            and math.isfinite(record["duration_seconds"]) and 0 <= record["duration_seconds"] < 180,
                            "Missing catalog duration cannot become a too-short source failure.")
    return {row["source_group"]: row for row in value["cohorts"]}


def resolved_windows(allocation, assessments):
    failed = {name for name, cohort in assessments.items()
              if any(record["status"] == "technical_failure" for record in cohort["recordings"])}
    first = next((name for name in KO_COHORTS if name in failed), None)
    result = []
    for name in KO_COHORTS:
        for index, window in enumerate(allocation["cohorts"][name]):
            replacement = name == first and KO_FALLBACK not in failed
            status = "primary" if name not in failed else "fallback" if replacement else "source_unavailable"
            target = allocation["fallback"][index] if status == "fallback" else window
            group = KO_FALLBACK if status == "fallback" else name
            record = assessments[group]["recordings"][index]
            result.append({"window_id": window["window_id"], "registered_source_group": name,
                           "registered_publisher_recording_id": window["publisher_recording_id"],
                           "source_group": group, "publisher_recording_id": target["publisher_recording_id"],
                           "source_allocation_status": status, "media_url": record.get("media_url"),
                           "slot_ids": [row["slot_id"] for row in allocation["roster"] if row.get("window_id") == window["window_id"]]})
    return result


def resolve_sources(metadata_manifest, assessments, output):
    allocation = metadata_allocation(metadata_manifest)
    assessment_path = bound(binding(assessments))
    values = validate_source_assessments(read_json(assessment_path), allocation)
    receipt = {"version": SOURCE_RESOLUTION_VERSION, "scope": SCOPE, "status": "resolved_before_body_freeze",
               "created_at": now(), "metadata_manifest": allocation["metadata_manifest"],
               "source_reservations": binding(allocation["paths"]["reservations"]),
               "roster": binding(allocation["paths"]["roster"]), "natural_windows": binding(allocation["paths"]["windows"]),
               "fallback_publisher_metadata": binding(allocation["paths"]["fallback"]),
               "assessments": binding(assessment_path), "cohort_order": list(KO_COHORTS),
               "fallback_source_group": KO_FALLBACK, "reserved_bodies_read": False,
               "quality_based_substitution": False, "windows": resolved_windows(allocation, values)}
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Use a fresh immutable source resolution receipt.")
    write(output, receipt)
    return binding(output)


def load_source_resolution(receipt, frozen_at):
    binding_value = receipt.get("source_resolution")
    reservations = read_json(bound(receipt["source_reservations"]))
    if not any(row["source_group"] == KO_FALLBACK for row in reservations["groups"]):
        require(binding_value is None, "Unexpected fallback receipt for this source reservation.")
        return None
    require(binding_value is not None, "Resolve both Korean cohorts before freezing body access.")
    value = read_json(bound(binding_value))
    require(value["version"] == SOURCE_RESOLUTION_VERSION and value["scope"] == SCOPE
            and value["status"] == "resolved_before_body_freeze" and value["reserved_bodies_read"] is False
            and value["quality_based_substitution"] is False and value["cohort_order"] == list(KO_COHORTS)
            and value["fallback_source_group"] == KO_FALLBACK,
            "Invalid pre-body technical source resolution.")
    created_at = datetime.fromisoformat(value["created_at"])
    require(created_at.tzinfo is not None and created_at <= frozen_at, "Source replacement was decided after the body freeze.")
    for key in ("metadata_manifest", "source_reservations", "roster", "natural_windows"):
        require(value[key] == receipt[key], "Source resolution differs from the frozen logical roster.")
    allocation = metadata_allocation(bound(receipt["metadata_manifest"]))
    require(value["fallback_publisher_metadata"] == binding(allocation["paths"]["fallback"]), "Fallback publisher metadata changed.")
    assessments = validate_source_assessments(read_json(bound(value["assessments"])), allocation)
    require(all(datetime.fromisoformat(row["recorded_at"]) <= created_at
                for cohort in assessments.values() for row in cohort["recordings"]), "Technical evidence postdates its resolution.")
    require(value["windows"] == resolved_windows(allocation, assessments),
            "Fallback identity, cohort order, timing allocation or logical slots changed.")
    return {"binding": binding_value, "receipt": value}


def effective_window(gate, window_id):
    window = copy.deepcopy(next(row for row in gate["windows"] if row["window_id"] == window_id))
    window.update(registered_source_group=window["source_group"],
                  registered_publisher_recording_id=window["publisher_recording_id"],
                  source_allocation_status="primary", source_resolution=None)
    resolution = gate.get("source_resolution")
    if window["source_group"] in KO_COHORTS and resolution is not None:
        mapped = next(row for row in resolution["receipt"]["windows"] if row["window_id"] == window_id)
        window.update(copy.deepcopy(mapped), source_resolution=resolution["binding"])
    return window


def verify_window_identity(gate, payload):
    window = effective_window(gate, payload["window_id"])
    require(window["source_allocation_status"] != "source_unavailable", "Registered source allocation remains unavailable.")
    require(payload["source_group"] == window["source_group"] and payload["recording_id"] == window["publisher_recording_id"]
            and payload.get("language", window["language"]) == window["language"]
            and payload.get("source_allocation_status", window["source_allocation_status"]) == window["source_allocation_status"],
            "Effective window source identity changed.")
    if window["source_resolution"] is not None:
        require(payload.get("source_resolution") == window["source_resolution"]
                and payload.get("registered_source_group") == window["registered_source_group"]
                and payload.get("registered_recording_id") == window["registered_publisher_recording_id"],
                "Logical source or immutable resolution binding changed.")
        if "media_url" in payload:
            require(payload["media_url"] == window["media_url"], "Media locator changed after metadata resolution.")
    return window


def verify_policy(name, value):
    require(value.get("status") == "frozen" and value.get("scope") == SCOPE,
            f"{name} policy is not frozen for final_test.")
    required = {
        "runtime": {"prompt_sha256", "contract_sha256", "formats_sha256", "tokenizer",
                    "generation", "cache_policy", "display_policy", "environment_receipt"},
        "asr": {"provider", "model_id", "no_verbatim", "timestamps_granularity", "diarize",
                "num_speakers", "tag_audio_events", "language_codes", "pcm", "retry_policy",
                "hosted_alias_limitation"},
        "import": {"normalizer_sha256", "document_importer_sha256", "paragraph_policy",
                   "source_clock", "cursor_policy", "window_policy", "subtitle_policy",
                   "text_or_timing_repair", "checkpoint_policy", "fallback_policy"},
        "metrics": {"evaluator_sha256", "all_request_failures_in_denominator",
                    "source_stratum_denominators", "unknown_is_no_action", "cluster_unit",
                    "decision_thresholds", "selection_after_test"},
        "admission": {"input_first", "all_seven_actions_reviewed", "independent_review",
                      "unknown_is_no_action", "audio_evidence_separate", "source_integrity_required",
                      "replay_undo_export_required", "duplicate_policy", "generator_recipe",
                      "label_rubric", "exclusions", "actors", "exposed_text_hashes"},
    }[name]
    require(required <= set(value), f"Incomplete frozen {name} policy.")
    if name == "runtime":
        require(value["prompt_sha256"] == sha256(contract.PROMPT_PATH)
                and value["contract_sha256"] == sha256(contract.__file__)
                and value["formats_sha256"] == sha256(formats.__file__), "Runtime contract changed.")
        generation = value["generation"]
        require(generation.get("do_sample") is False and generation.get("max_new_tokens") == 512
                and generation.get("enable_thinking") is False, "Unregistered generation policy.")
        require(bool(value["cache_policy"]) and bool(value["display_policy"]), "Missing runtime policy.")
        bound(value["environment_receipt"])
        tokenizer = value["tokenizer"]
        root = Path(tokenizer["path"])
        root = root if root.is_absolute() else ROOT / root
        require(root.resolve().is_relative_to(ROOT), "Invalid native tokenizer path.")
        if tokenizer.get("runtime_family") in FULL_COHORT_FAMILIES:
            backend = runtime_backend(tokenizer["runtime_family"])
            backend.validate_tokenizer_identity(tokenizer)
            require(generation["eos_token_id"] == tokenizer["terminal_ids"], "Small generation terminal policy changed.")
            context = backend.validate_inference_policy(read_json(bound(value["inference_policy"])))
            require(context["model"] == tokenizer["model"]
                    and read_json(bound(context["assets_receipt"])) == tokenizer["assets"],
                    "Small runtime policy uses another tokenizer.")
        else:
            require("runtime_family" not in tokenizer and tokenizer.get("eos_token_id") == 1,
                    "Invalid native tokenizer identity.")
        require({"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
                <= set(tokenizer["files_sha256"]), "Incomplete native tokenizer pin.")
        for filename, expected in tokenizer["files_sha256"].items():
            path = (root / filename).resolve()
            require(path.is_relative_to(root.resolve()) and sha256(path) == expected,
                    "Native tokenizer changed.")
    elif name == "asr":
        require(value["provider"] == "elevenlabs" and value["model_id"] == "scribe_v2"
                and value["no_verbatim"] is False and value["timestamps_granularity"] == "word"
                and value["diarize"] is True and value["num_speakers"] is None
                and value["tag_audio_events"] is True and value["language_codes"] == ["en", "ko"]
                and value["pcm"] == {"sample_rate": 16000, "channels": 1, "sample_width": 2}
                and value["retry_policy"] == "no_automatic_billable_retry"
                and bool(value["hosted_alias_limitation"]), "ASR policy differs from registered raw import.")
    elif name == "import":
        require(value["normalizer_sha256"] == sha256(ROOT / "server/app/asr/scribe.py")
                and value["document_importer_sha256"] == sha256(ROOT / "app/src/core/transcription.ts")
                and value["text_or_timing_repair"] is False
                and value["paragraph_policy"] == "existing_importTranscript_speaker_groups"
                and value["source_clock"] == "excerpt_relative_with_original_offset_provenance"
                and value["subtitle_policy"] == "judge_evidence_only"
                and value["cursor_policy"] == "nearest_timed_item_to_window_midpoint_ties_earlier",
                "Natural input import policy changed.")
    elif name == "metrics":
        require(value["evaluator_sha256"] == sha256(evaluation.__file__)
                and value["all_request_failures_in_denominator"] is True
                and value["source_stratum_denominators"] is True
                and value["unknown_is_no_action"] is False
                and value["cluster_unit"] == "source_group"
                and value["selection_after_test"] is False, "Invalid final-test metric policy.")
    else:
        require(all(value[key] is True for key in (
            "input_first", "all_seven_actions_reviewed", "independent_review",
            "audio_evidence_separate", "source_integrity_required", "replay_undo_export_required"))
            and value["unknown_is_no_action"] is False, "Invalid final-test admission policy.")
        actors = value["actors"]
        require(set(actors) == {"preparer", "author", "reviewer", "grader"}
                and all(isinstance(actor, str) and actor.strip() for actor in actors.values())
                and len({actors[key] for key in ("author", "reviewer", "grader")}) == 3,
                "Reference actors are not separated.")
        for field in ("generator_recipe", "label_rubric", "exclusions", "exposed_text_hashes"):
            bound(value[field])


def load_gate(path):
    """Verify freeze metadata and exposed TRAIN demos; never read reserved final bodies."""
    receipt = read_json(path)
    require(receipt.get("version") == FREEZE_VERSION and receipt.get("scope") == SCOPE
            and receipt.get("status") == "frozen" and receipt.get("body_access_authorized") is True
            and receipt.get("reserved_bodies_read_before_freeze") is False,
            "Final-test body access is not authorized by a complete freeze.")
    frozen_at = datetime.fromisoformat(receipt["frozen_at"])
    require(frozen_at.tzinfo is not None and frozen_at <= datetime.now(timezone.utc), "Invalid freeze time.")
    require(receipt["source_sha256"] == code_sources(), "Frozen preparation/runtime code changed.")
    metadata_path = bound(receipt["metadata_manifest"])
    metadata = read_json(metadata_path)
    require(metadata.get("reserved_final_test_bodies_read") is False
            and metadata.get("generated_test_cases") == 0, "Metadata registration was already exposed.")
    reservations_path = bound(receipt["source_reservations"])
    roster_path = bound(receipt["roster"])
    windows_path = bound(receipt["natural_windows"])
    for payload in (reservations_path, roster_path, windows_path):
        relative = str(payload.relative_to(metadata_path.parent))
        require(metadata["files_sha256"].get(relative) == sha256(payload), "Roster is not metadata-bound.")
    reservations, roster = read_json(reservations_path), rows(roster_path)
    groups = {group["source_group"]: group for group in reservations["groups"]}
    require(len(groups) == len(reservations["groups"]) and roster, "Empty or duplicate source groups.")
    require(not set(groups).intersection(reservations["exposed_source_group_splits"]),
            "Reserved group already has training/development exposure.")
    require(dataset.RESERVED <= set(groups), "Historical final-test reservations were lost.")
    require(len({row["slot_id"] for row in roster}) == len(roster), "Duplicate planned slot.")
    for row in roster:
        group = groups[row["source_group"]]
        require(group["role"] == "primary" and row["language"] == group["language"]
                and row["stratum"] in STRATA and row["admission"] == "not_materialized",
                "Invalid pre-body slot allocation.")
        require(not {"input", "document", "primary", "accepted_predictions"}.intersection(row),
                "The metadata roster contains test bodies or answers.")
    for filename, expected in reservations.get("exposed_inventory_sha256", {}).items():
        bound({"path": str(ROOT / filename), "sha256": expected})
    training_path = bound(receipt["training_dataset_manifest"])
    training_manifest = read_json(training_path)
    require(receipt["training_dataset_manifest"]["sha256"] == metadata["selected_dataset_manifest_sha256"],
            "Training exposure differs from metadata registration.")
    if training_manifest.get("version") == "passive-actions-final-unified-1":
        counts = training_manifest["counts"]
        require(set(counts) == {"train", "dev", "canonical_dev", "supplemental_train", "supplemental_dev"}
                and all(type(value) is int and value >= 0 for value in counts.values())
                and counts["dev"] == counts["canonical_dev"] + counts["supplemental_dev"]
                and training_manifest["test_bodies_read"] is False,
                "Final unified training manifest contains invalid scope/counts.")
    else:
        require(set(training_manifest["counts"]) <= {"train", "dev"}, "Training manifest contains test cases.")
    policies = {name: read_json(bound(receipt["policies"][name])) for name in POLICIES}
    for name, policy in policies.items():
        verify_policy(name, policy)
    selection = read_json(bound(receipt["selection_receipt"]))
    require(selection.get("status") == "frozen" and selection.get("test_bodies_read") is False
            and selection.get("selection_scope") == "development"
            and selection.get("selected_model") == receipt["selected_model"], "Selection was not frozen on development.")
    models = {role: read_json(bound(value)) for role, value in receipt["models"].items()}
    omission = receipt.get("dpo_omission")
    expected_models = MODEL_ROLES - {"sft_dpo"} if omission is not None else MODEL_ROLES
    expected_runs = RUN_ROLES - {"sft_dpo"} if omission is not None else RUN_ROLES
    if omission is not None:
        require(receipt["selected_model"] == "sft"
                and len({model.get("runtime_family") for model in models.values()}) == 1
                and all(model.get("runtime_family") in FULL_COHORT_FAMILIES for model in models.values()),
                "Only an explicitly documented same-parent DPO omission may omit a learned arm.")
        bound(omission)
    require(set(models) == expected_models and receipt["selected_model"] in expected_models - {"native_base"},
            "Base, SFT and SFT+DPO must all be frozen before test admission.")
    for role, model in models.items():
        require(model.get("role") == role and model.get("status") == "verified"
                and bool(model.get("identity")), "Unverified model identity.")
        bound(model["artifact_manifest"])
        bound(model["reload_receipt"])
        bound(model["inference_configuration"])
        bound(model["preparation_manifest"])
        model_runtime(model, engineering_fixture_only=receipt.get("engineering_fixture_only", False))
    if omission is None:
        require(models["sft_dpo"]["parent_identity_sha256"] == digest(models["sft"]["identity"]),
                "DPO is not bound to the frozen SFT parent.")
    if any("runtime_family" in model for model in models.values()):
        runtime_backend(policies["runtime"]["tokenizer"].get("runtime_family")).verify_model_set(
            models, policies["runtime"]["tokenizer"], receipt["training_dataset_manifest"], omission,
            policies["runtime"]["inference_policy"])
    runs = receipt["runs"]
    require(len({run["id"] for run in runs}) == len(runs)
            and {run["role"] for run in runs} == expected_runs, "Incomplete or duplicate final-test comparison roster.")
    demo_ids = {demo["case_id"] for run in runs for demo in run["condition"]["demonstrations"]}
    train_path = training_path.parent / "cases/train.jsonl"
    require(training_manifest.get("files_sha256", {}).get("cases/train.jsonl") == sha256(train_path),
            "Actual ICL TRAIN payload is not bound to the selected release.")
    demo_cases, training_ids = {}, set()
    with train_path.open() as stream:
        for line in stream:
            case = json.loads(line)
            require(case["id"] not in training_ids and case["split"] == case["source"]["split"] == "train",
                    "Duplicate or non-TRAIN case in the training release.")
            training_ids.add(case["id"])
            if case["id"] in demo_ids:
                demo_cases[case["id"]] = case
    require(set(demo_cases) == demo_ids and len(training_ids) == training_manifest["counts"]["train"],
            "ICL case is absent from the actual TRAIN release.")
    exposed = set(read_json(bound(policies["admission"]["exposed_text_hashes"]))["sha256"])
    for run in runs:
        require(Path(run["id"]).name == run["id"] and run["id"] not in {".", ".."}
                and run["format"] in formats.FORMATS, "Unsafe run ID or format.")
        spec = run["condition"]
        if policies["runtime"]["tokenizer"].get("runtime_family") in FULL_COHORT_FAMILIES:
            backend = runtime_backend(policies["runtime"]["tokenizer"]["runtime_family"])
            require(run["format"] == "json" and (run["role"] != "base_icl"
                    or [demo["case_id"] for demo in spec["demonstrations"]] == backend.DEMONSTRATION_IDS),
                    "Small final comparison must preserve the registered JSON ICL examples and their order.")
        expected = "train_icl" if run["role"] == "base_icl" else "canonical_full_context"
        require(spec["id"] == expected and spec == conditions.condition_specs(spec["demonstrations"])[expected],
                "Test run uses an unregistered input condition.")
        for demo in spec["demonstrations"]:
            require(demo["split"] == "train" and demo["source_group"] not in groups,
                    "Reserved or non-training demonstration.")
            require(reservations["exposed_source_group_splits"].get(demo["source_group"]) == "train",
                    "ICL source is not in the frozen training allocation.")
            require(demo["release_binding"]["manifest"]["sha256"]
                    == receipt["training_dataset_manifest"]["sha256"], "ICL training release changed.")
            case = demo_cases[demo["case_id"]]
            require(bound(demo["release_binding"]["manifest"]) == training_path
                    and demo["case_sha256"] == digest(case) and demo["input"] == case["input"]
                    and demo["input_sha256"] == conditions.input_hash(case["input"])
                    and demo["action"] == case["primary"] and demo["language"] == case["language"]
                    and demo["source_group"] == case["source"]["source_group"]
                    and demo["source_sha256"] == digest(case["source"])
                    and demo["accepted_predictions_sha256"] == digest(case["accepted_predictions"])
                    and demo["outputs_sha256"] == {name: digest(formats.serialize(case["input"], case["primary"], name))
                                                   for name in formats.FORMATS},
                    "ICL demonstration differs from its actual TRAIN case.")
            require(digest(sealed_v1.corpus_text(demo["input"])) in exposed,
                    "ICL demonstration is absent from the frozen exposed-text index.")
    windows = read_json(windows_path)
    windows = windows["windows"] if isinstance(windows, dict) else windows
    require({row["window_id"] for row in windows} == {row["window_id"] for row in roster if row.get("window_id")},
            "Natural window plan differs from the roster.")
    source_resolution = load_source_resolution(receipt, frozen_at)
    return {"receipt": receipt, "binding": binding(path), "roster": roster, "groups": groups, "windows": windows,
            "policies": policies, "models": models, "frozen_at": frozen_at, "source_resolution": source_resolution,
            "demonstration_cases": demo_cases}


def load_tokenizer(gate):
    from transformers import AutoTokenizer
    identity = gate["policies"]["runtime"]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / identity["path"]), local_files_only=True, trust_remote_code=False)
    require(tokenizer.eos_token_id == identity["eos_token_id"], "Native EOS changed.")
    return tokenizer


def finish_output(output, manifest):
    manifest = {"version": VERSION, "scope": SCOPE, "created_at": now(), **manifest,
                "files_sha256": {str(path.relative_to(output)): sha256(path)
                                 for path in sorted(output.rglob("*")) if path.is_file()}}
    write(output / "manifest.json", manifest)
    (output / "manifest.json").chmod(0o600)
    return binding(output / "manifest.json")


def publish(output, payloads, manifest):
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Use a fresh workspace output directory.")
    output.mkdir(parents=True, mode=0o700)
    for filename, value in payloads.items():
        target = output / filename
        (write_rows if filename.endswith(".jsonl") else write)(target, value)
        target.chmod(0o600)
    return finish_output(output, manifest)


def load_stage(gate, directory, status):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    require(manifest["version"] == VERSION and manifest["scope"] == SCOPE
            and manifest["status"] == status and manifest["freeze"] == gate["binding"],
            "Stage belongs to another freeze or scope.")
    evaluation.verify_files(directory, manifest)
    return manifest


def candidate_identity(candidate, slot, gate):
    source_group = slot["source_group"]
    if slot.get("window_id"):
        lineage = candidate["lineage"]
        window = verify_window_identity(gate, {
            "window_id": slot["window_id"], "source_group": candidate["source"]["source_group"],
            "recording_id": candidate["source"]["recording_id"], "language": candidate["language"],
            "source_resolution": lineage.get("source_resolution"),
            "registered_source_group": lineage.get("registered_source_group"),
            "registered_recording_id": lineage.get("registered_recording_id"),
        })
        require(lineage["window_id"] == slot["window_id"], "Candidate lost its logical window lineage.")
        source_group = window["source_group"]
    require(candidate["id"] == slot["slot_id"] and candidate["language"] == slot["language"]
            and candidate["split"] == candidate["source"]["split"] == SCOPE
            and candidate["source"]["source_group"] == source_group
            and candidate["lineage"]["stratum"] == slot["stratum"], "Candidate allocation changed.")
    require(not {"primary", "acceptable", "accepted_predictions", "judgment", "rationale"}.intersection(candidate),
            "Input-freeze candidates must not contain reference labels.")
    require(candidate["provenance"]["selection_before_words"] is True
            and candidate["provenance"]["model_predictions_seen"] is False,
            "Candidate was selected using words, labels or predictions.")
    created_at = datetime.fromisoformat(candidate["provenance"]["created_at"])
    require(created_at.tzinfo is not None and created_at >= gate["frozen_at"]
            and candidate["provenance"]["freeze_sha256"] == gate["binding"]["sha256"],
            "Candidate predates or belongs to another freeze.")


def synthetic_runtime(runtime):
    require(type(runtime["sequence"]) is int and runtime["sequence"] >= 0, "Invalid runtime sequence.")
    require(all(item["sequence"] <= runtime["sequence"] and item["origin"] == "synthetic"
                for item in runtime["history"]), "Future or falsely human history.")
    if runtime.get("feedback_journal"):
        records = contract.PassiveFeedbackLedger.restore(runtime["feedback_journal"]).dump()["records"]
        require(all(event["sequence"] <= runtime["sequence"]
                    and (event["kind"] != "record" or event["origin"] == "synthetic") for event in records),
                "Future or falsely human feedback entered the input.")


def authored_structure(candidate, slot, gate):
    structure_binding = candidate["lineage"]["preword_structure"]
    structure = read_json(bound(structure_binding))
    created_at = datetime.fromisoformat(structure["created_at"])
    require(set(structure) == {"freeze_sha256", "source_group", "scene_plan", "created_at", "wording_assigned",
                              "timed_slots", "anchor"}
            and structure["freeze_sha256"] == gate["binding"]["sha256"]
            and structure["source_group"] == slot["source_group"]
            and structure["scene_plan"] == slot["scene_plan"] and structure["wording_assigned"] is False
            and created_at.tzinfo is not None and gate["frozen_at"] <= created_at
            <= datetime.fromisoformat(candidate["provenance"]["created_at"]),
            "Authored structure was not fixed before wording under this scene/freeze.")
    slots = structure["timed_slots"]
    require(slots and len({row["uuid"] for row in slots}) == len(slots)
            and all(set(row) == {"uuid", "source", "sourceStart", "length"}
                    and isinstance(row["uuid"], str) and bool(row["uuid"])
                    and isinstance(row["source"], str) and bool(row["source"])
                    and all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in ("sourceStart", "length"))
                    and row["sourceStart"] >= 0 and row["length"] > 0 for row in slots),
            "Authored structure contains words or invalid source-clock slots.")
    runtime, document = candidate["runtime"], candidate["document"]
    initial = runtime["history"][0]["before"] if runtime["history"] else document
    require([{key: item[key] for key in ("uuid", "source", "sourceStart", "length")}
             for item in sealed_v1.timed(initial)] == slots, "Authored source clock changed after structure selection.")
    anchor, plan = structure["anchor"], slot["scene_plan"]
    require(set(anchor) == {"kind", "source", "seconds"} and type(anchor["seconds"]) in (int, float)
            and math.isfinite(anchor["seconds"]), "Invalid authored source-clock anchor.")
    source_slots = [row for row in slots if row["source"] == anchor["source"]]
    require(source_slots, "Authored anchor has no source-clock slots.")
    declared = plan.get("anchor") if isinstance(plan, dict) else None
    if declared is not None:
        require(anchor == {"kind": "inherited_declared", **declared}, "Registered authored anchor changed.")
    else:
        midpoint = (min(row["sourceStart"] for row in source_slots)
                    + max(row["sourceStart"] + row["length"] for row in source_slots)) / 2
        require(anchor["kind"] == "midpoint" and anchor["seconds"] == midpoint,
                "Authored cursor does not use the pre-word source-clock midpoint.")
    selection = plan.get("selection", []) if isinstance(plan, dict) else []
    require(runtime["selection"] == selection, "Authored selection was not preregistered.")
    current = [row for row in sealed_v1.timed(document) if row["source"] == anchor["source"]]
    require(current, "Authored anchor source has no surviving timed item.")
    focus = min(enumerate(current), key=lambda pair: (abs(pair[1]["sourceStart"] + pair[1]["length"] / 2
                                                       - anchor["seconds"]), pair[0]))[1]["uuid"]
    require(runtime["focus_uuid"] == focus, "Authored cursor was selected using the target.")
    return (slot["source_group"], packed(slot["scene_plan"])), structure_binding


def natural_checkpoint(candidate, slot, gate, previous):
    runtime, document = candidate["runtime"], candidate["document"]
    synthetic_runtime(runtime)
    if slot["stratum"] == "authored_challenge":
        return
    checkpoint = slot["checkpoint"]
    require(checkpoint in (0, 1, 2) and not runtime["selection"], "Invalid natural checkpoint.")
    timed = sealed_v1.timed(document)
    require(timed, "Natural state has no addressable items.")
    require(all(item["source"] == opaque_source_id(slot["window_id"]) for item in timed),
            "Natural model-visible source identity is not opaque or differs from its window.")
    if checkpoint == 0:
        imported = bound(candidate["provenance"]["initial_import_manifest"])
        load_stage(gate, imported.parent, "imported_unlabeled")
        imported_source = read_json(imported.parent / "provenance.json")
        require(imported_source["window_id"] == slot["window_id"], "Initial import belongs to another logical window.")
        verify_window_identity(gate, imported_source)
        require(document == read_json(imported.parent / "document.json")
                and runtime["sequence"] == 0 and not runtime["history"]
                and runtime.get("feedback_journal") is None, "Initial natural state was edited before its input freeze.")
        focus = min(enumerate(timed), key=lambda pair: (abs(pair[1]["sourceStart"] + pair[1]["length"] / 2 - 45), pair[0]))[1]["uuid"]
    else:
        require(previous is not None, "A dependent checkpoint lost its preceding state.")
        old = previous["runtime"]
        focus = old["focus_uuid"]
        if not any(item["uuid"] == focus for item in timed):
            old_item = next(item for item in sealed_v1.timed(previous["document"]) if item["uuid"] == focus)
            same = [item for item in timed if item.get("source") == old_item.get("source")]
            before = [item for item in same if item["sourceStart"] < old_item["sourceStart"]]
            after = [item for item in same if item["sourceStart"] >= old_item["sourceStart"]]
            require(before or after, "Removed cursor has no surviving source neighbor.")
            focus = (max(before, key=lambda item: item["sourceStart"]) if before
                     else min(after, key=lambda item: item["sourceStart"]))["uuid"]
        require(runtime["sequence"] > old["sequence"], "Checkpoint sequence did not advance.")
        if checkpoint == 1:
            require(len(runtime["history"]) == 1 and runtime["history"][0]["before"] == previous["document"]
                    and runtime["history"][0]["after"] == document
                    and runtime["history"][0]["sequence"] == runtime["sequence"]
                    and runtime.get("feedback_journal") == old.get("feedback_journal"), "C1 is not an actual intentional transaction.")
        else:
            require(document == previous["document"] and runtime["history"] == old["history"],
                    "C2 dismissal unexpectedly changed text, media or committed history.")
            journal = runtime.get("feedback_journal")
            require(journal is not None, "C2 lacks actual proposal feedback.")
            ledger = contract.PassiveFeedbackLedger.restore(journal)
            records = ledger.dump()["records"]
            events = [event for event in records if event["kind"] == "record"
                      and old["sequence"] < event["sequence"] <= runtime["sequence"]]
            require([event["outcome"] for event in events] == ["shown", "dismissed"]
                    and len({event["proposal_id"] for event in events}) == 1
                    and all(event["origin"] == "synthetic" for event in events), "C2 did not execute simulated display/dismissal.")
    require(runtime["focus_uuid"] == focus, "Natural cursor was selected using the target.")


def make_requests(case, gate, tokenizer):
    requests = {}
    for run in gate["receipt"]["runs"]:
        messages = conditions.expected_messages(case["input"], run["format"], run["condition"])
        prefix = conditions.native_prefix(tokenizer, messages)
        requests[run["id"]] = {
            "case_id": case["id"], "split": SCOPE, "run_id": run["id"], "format": run["format"],
            "condition": run["condition"], "condition_sha256": digest(run["condition"]),
            "input": case["input"], "input_sha256": evaluation.semantic_input_hash(case["input"]),
            "messages": messages, "messages_sha256": digest(messages),
            "prefix_ids": prefix, "prefix_ids_sha256": digest(prefix),
            "freeze_sha256": gate["binding"]["sha256"],
        }
    return requests


def input_record(candidate, bundle):
    require(not {"bundle", "messages", "input_sha256", "input_record_sha256"}.intersection(candidate),
            "Candidate already contains a prepared input record.")
    require("input" not in candidate or candidate["input"] == bundle["input"],
            "Saved candidate window differs from runtime.")
    case = copy.deepcopy({**candidate, "input": bundle["input"],
                          "messages": contract.canonical_messages(bundle["input"]), "bundle": bundle})
    case["input_sha256"] = evaluation.semantic_input_hash(case["input"])
    case["input_record_sha256"] = digest(case)
    return case


def freeze_inputs(freeze, candidates, output, *, tokenizer=None):
    gate = load_gate(freeze)
    candidates_path = Path(candidates)
    records = rows(candidates_path)
    require([row["id"] for row in records] == [row["slot_id"] for row in gate["roster"]],
            "Every registered slot needs an ordered candidate/disposition.")
    tokenizer = tokenizer or load_tokenizer(gate)
    builder = contract.PassiveContextBuilder(tokenizer)
    for run in gate["receipt"]["runs"]:
        for demo in run["condition"]["demonstrations"]:
            require(demo == conditions.demonstration(gate["demonstration_cases"][demo["case_id"]],
                                                     tokenizer, demo["release_binding"]),
                    "ICL native prefix or completion budget differs from the actual TRAIN demonstration.")
    exposed = set(read_json(bound(gate["policies"]["admission"]["exposed_text_hashes"]))["sha256"])
    seen, cases, census, previous, exclusions, structures = set(), [], [], {}, [], {}
    request_rows = {run["id"]: [] for run in gate["receipt"]["runs"]}
    for candidate, slot in zip(records, gate["roster"], strict=True):
        entry = {key: slot.get(key) for key in ("slot_id", "source_group", "language", "stratum", "window_id", "checkpoint")}
        entry["id"] = entry.pop("slot_id")
        entry["registered_source_group"] = slot["source_group"]
        if slot.get("window_id"):
            window = effective_window(gate, slot["window_id"])
            entry.update(source_group=window["source_group"], recording_id=window["publisher_recording_id"],
                         registered_recording_id=window["registered_publisher_recording_id"],
                         source_resolution=window["source_resolution"], source_allocation_status=window["source_allocation_status"])
            require(window["source_allocation_status"] != "source_unavailable" or candidate["status"] == "source_unavailable",
                    "Failed source cohort cannot be promoted to a candidate or preservation label.")
        if candidate["status"] in UNKNOWN:
            require(not {"document", "input", "primary", "accepted_predictions"}.intersection(candidate)
                    and bool(candidate["reason"]), "Unavailable slot must remain unscored.")
            census.append({**entry, "status": candidate["status"], "reason": candidate["reason"]})
            previous.pop(slot.get("window_id"), None)
            continue
        require(candidate["status"] == "candidate", "Unknown candidate status.")
        candidate_identity(candidate, slot, gate)
        if slot["stratum"] == "authored_challenge":
            scene, structure = authored_structure(candidate, slot, gate)
            require(scene not in structures or structures[scene] == structure, "Authored scene variants changed their shared structure/anchor.")
            structures[scene] = structure
        elif slot["checkpoint"] > 0 and slot["window_id"] not in previous:
            census.append({**entry, "status": "structural_state_unavailable", "reason": "preceding_input_not_eligible"})
            continue
        natural_checkpoint(candidate, slot, gate, previous.get(slot.get("window_id")))
        try:
            bundle = dataset.runtime_bundle(candidate, builder)
        except ContextBudgetError as error:
            census.append({**entry, "status": "context_ineligible", "reason": "frozen_context_builder_budget",
                           "error": str(error)})
            previous.pop(slot.get("window_id"), None)
            continue
        case = input_record(candidate, bundle)
        text_hash = digest(sealed_v1.corpus_text(case["input"]))
        input_hash = evaluation.semantic_input_hash(case["input"])
        require(text_hash not in exposed and input_hash not in seen, "Duplicate or exposed test wording/input.")
        seen.add(input_hash)
        prepared = make_requests(case, gate, tokenizer)
        sizes = {key: len(row["prefix_ids"]) for key, row in prepared.items()}
        canonical_runs = {run["id"] for run in gate["receipt"]["runs"] if run["role"] != "base_icl"}
        full_cohort = gate["policies"]["runtime"]["tokenizer"].get("runtime_family") in FULL_COHORT_FAMILIES
        if full_cohort:
            require(all(sizes[key] <= formats.INPUT_LIMIT for key in canonical_runs),
                    "Small canonical visible window exceeded its frozen 3584-token budget.")
            require(all(size <= 7680 for size in sizes.values()),
                    "A small final role exceeds 7680 prefix tokens; fail the whole preparation without role-specific exclusions.")
        if any(sizes[key] > formats.INPUT_LIMIT for key in canonical_runs):
            census.append({**entry, "status": "context_ineligible", "reason": "frozen_common_run_prefix_budget",
                           "input_tokens_by_run": sizes})
            previous.pop(slot.get("window_id"), None)
            continue
        if slot.get("window_id"):
            previous[slot["window_id"]] = case
        cases.append(case)
        census.append({**entry, "status": "input_frozen", "input_sha256": case["input_sha256"],
                       "input_record_sha256": case["input_record_sha256"], "text_sha256": text_hash})
        for run_id, request in prepared.items():
            if full_cohort or sizes[run_id] <= formats.INPUT_LIMIT:
                request_rows[run_id].append(request)
            else:
                exclusions.append({"id": case["id"], "run_id": run_id, "reason": "icl_prefix_budget",
                                   "input_tokens": sizes[run_id], "input_sha256": case["input_sha256"]})
    return publish(output, {"inputs.jsonl": cases, "census.jsonl": census, "run-exclusions.jsonl": exclusions,
                           **{f"requests-{key}.jsonl": value for key, value in request_rows.items()}},
                   {"status": "inputs_frozen_unlabeled", "freeze": gate["binding"],
                    "candidates": binding(candidates_path), "registered_slots": len(records),
                    "frozen_inputs": len(cases), "predictions_seen": False})


def audit_input(reference, case, gate, tokenizer):
    audit, actors = reference["audit"], gate["policies"]["admission"]["actors"]
    prelabel = bound(audit["prelabel_input_record"])
    require(prelabel.name == "manifest.json", "Prelabel input must bind its immutable manifest.")
    receipt = load_stage(gate, prelabel.parent, "prelabel_input_frozen")
    recorded = read_json(prelabel.parent / "input-record.json")
    require(set(receipt["files_sha256"]) == {"input-record.json", "requests.json"}
            and receipt["predictions_seen"] is False and recorded == case
            and recorded["input_record_sha256"] == digest({key: value for key, value in recorded.items()
                                                          if key != "input_record_sha256"})
            and read_json(prelabel.parent / "requests.json") == make_requests(case, gate, tokenizer),
            "Prelabel input record or registered requests changed after review.")
    times = [datetime.fromisoformat(value) for value in (
        case["provenance"]["created_at"], receipt["created_at"], audit["author_decided_at"], audit["reviewed_at"], now())]
    require(all(value.tzinfo is not None for value in times) and all(a <= b for a, b in zip(times, times[1:])),
            "Author/reviewer decisions must follow the immutable prelabel input record.")
    require(audit["author"] == actors["author"] and audit["reviewer"] == actors["reviewer"]
            and audit["input_first"] is True and audit["predictions_seen"] is False
            and audit["reviewed_actions"] == list(contract.ACTIONS)
            and bool(audit["rationale"]) and audit["input_record_sha256"] == case["input_record_sha256"],
            "Reference lacks a distinct, input-first whole-window review.")


def reference_audit(reference, case, gate, tokenizer):
    audit_input(reference, case, gate, tokenizer)
    refs = reference["accepted_predictions"]
    require(refs and refs[0] == reference["primary"] and len({packed(r) for r in refs}) == len(refs),
            "Primary must be first in unique accepted predictions.")
    require(all((r["action"] == "no_action") == (reference["status"] == "preserve") for r in refs),
            "Unknown or mixed necessity is not a supervised no_action.")
    return refs


def reference_proof(case, prediction, exporter):
    document, bundle = case["document"], case["bundle"]
    ledger = case["runtime"].get("feedback_journal")
    if ledger and prediction["action"] != "no_action":
        require(not contract.PassiveFeedbackLedger.restore(ledger).suppressed(
            document, bundle, prediction, case["runtime"]["sequence"]), "Reference repeats a suppressed proposal.")
    proof = dataset.execution(document, bundle, prediction, case["source"])
    after = contract.apply_passive_edit(document, bundle, prediction)
    operation = verify_operation(document, after, bundle["mapping"], prediction)
    line_length = bundle["input"]["caption_profile"]["export_line_length"]
    proof["caption_export"] = verify_export_delta(exporter.export(document, line_length),
                                                  exporter.export(after, line_length), operation, prediction["action"])
    proof["outcome_sha256"] = evaluation.result_hash(after)
    proof["acoustic_quality_verified"] = False
    return proof


def reference_dependency(reference, case, gate, all_references, all_cases):
    slots = {row["slot_id"]: row for row in gate["roster"]}
    slot = slots[case["id"]]
    if slot["stratum"] == "authored_challenge" or slot["checkpoint"] == 0:
        return
    prior = next((row for row in gate["roster"] if row.get("window_id") == slot["window_id"]
                  and row.get("checkpoint") == slot["checkpoint"] - 1), None)
    require(prior is not None and prior["slot_id"] in all_references and prior["slot_id"] in all_cases,
            "Dependent state has no eligible preceding input/reference.")
    previous_reference = all_references[prior["slot_id"]]
    require(previous_reference["status"] == "edit_required", "Dependent state has no qualified preceding edit.")
    require(datetime.fromisoformat(previous_reference["audit"]["reviewed_at"])
            <= datetime.fromisoformat(case["provenance"]["created_at"]),
            "Dependent checkpoint was constructed before its preceding independent review.")
    if slot["checkpoint"] == 1:
        prediction = case["runtime"]["history"][0]["prediction"]
    else:
        ledger = contract.PassiveFeedbackLedger.restore(case["runtime"]["feedback_journal"])
        records = ledger.dump()["records"]
        dismissed = next((event for event in reversed(records) if event.get("outcome") == "dismissed"), None)
        require(dismissed is not None, "Dependent state has no actual dismissal.")
        proposal = next(event for event in records if event["kind"] == "propose" and event["proposal_id"] == dismissed["proposal_id"])
        prediction = proposal["prediction"]
        previous_case = all_cases[prior["slot_id"]]
        require(proposal["bundle"] == previous_case["bundle"] and proposal["document"] == previous_case["document"],
                "Dismissal was not shown against the qualified preceding state.")
    require(prediction in previous_reference["accepted_predictions"], "Simulated history/feedback used an unqualified edit.")


def seal_references(freeze, inputs, references, output, *, tokenizer=None):
    gate = load_gate(freeze)
    load_stage(gate, inputs, "inputs_frozen_unlabeled")
    inputs = Path(inputs)
    census, reference_rows = rows(inputs / "census.jsonl"), rows(references)
    require([r["id"] for r in reference_rows] == [r["id"] for r in census], "Reference census changed.")
    indexed = {r["id"]: r for r in rows(inputs / "inputs.jsonl")}
    reference_index = {r["id"]: r for r in reference_rows}
    tokenizer = tokenizer or load_tokenizer(gate)
    builder = contract.PassiveContextBuilder(tokenizer)
    cases, proofs, disposition = [], [], []
    exporter = Exporter()
    try:
        for reference, entry in zip(reference_rows, census, strict=True):
            status = reference["status"]
            require(status in KNOWN | UNKNOWN, "Unregistered reference disposition.")
            if entry["status"] != "input_frozen":
                require(status == entry["status"] and not reference.get("accepted_predictions"),
                        "Unavailable slot was promoted into a gold reference.")
                disposition.append(entry)
                continue
            case = indexed[reference["id"]]
            require(reference["input_record_sha256"] == case["input_record_sha256"]
                    == entry["input_record_sha256"] == digest({key: value for key, value in case.items() if key != "input_record_sha256"}),
                    "Reference input binding changed.")
            if status in UNKNOWN:
                audit_input(reference, case, gate, tokenizer)
                require(not reference.get("accepted_predictions") and reference.get("primary") is None
                        and bool(reference["reason"]),
                        "Unknown cannot carry no_action or another accepted label.")
                disposition.append({**entry, "status": status, "reason": reference["reason"]})
                continue
            refs = reference_audit(reference, case, gate, tokenizer)
            reference_dependency(reference, case, gate, reference_index, indexed)
            require(dataset.runtime_bundle(case, builder) == case["bundle"], "Frozen input runtime changed.")
            case_proofs = []
            for prediction in refs:
                for run in gate["receipt"]["runs"]:
                    message = conditions.expected_messages(case["input"], run["format"], run["condition"])
                    prefix = conditions.native_prefix(tokenizer, message)
                    completion = tokenizer.encode(formats.serialize(case["input"], prediction, run["format"]), add_special_tokens=False)
                    require(len(completion) + 1 <= formats.OUTPUT_LIMIT
                            and (len(prefix) > formats.INPUT_LIMIT and run["role"] == "base_icl"
                                 or len(prefix) + len(completion) + 1 <= formats.CONTEXT_LIMIT),
                            "Reference exceeds the frozen native completion budget.")
                case_proofs.append(reference_proof(case, prediction, exporter))
            proof = {"id": case["id"], "input_record_sha256": case["input_record_sha256"], "references": case_proofs}
            proofs.append(proof)
            cases.append({**case, "schema_version": VERSION, "primary": refs[0], "acceptable": refs,
                          "accepted_predictions": refs, "accepted_result_sha256": [r["outcome_sha256"] for r in case_proofs],
                          "judgment": {"status": status, "actor": "ai_authored_independently_reviewed",
                                       "acoustic_evidence": reference["acoustic_evidence"]},
                          "rationale": reference["audit"]["rationale"], "review": reference["audit"],
                          "reference_proof_sha256": digest(proof)})
            disposition.append({**entry, "status": "admitted"})
    finally:
        exporter.close()
    admitted = {case["id"] for case in cases}
    payloads = {"cases.jsonl": cases, "reference-executions.jsonl": proofs, "dispositions.jsonl": disposition,
                "run-exclusions.jsonl": [row for row in rows(inputs / "run-exclusions.jsonl") if row["id"] in admitted]}
    case_ids_by_run = {}
    for run in gate["receipt"]["runs"]:
        filename = f"requests-{run['id']}.jsonl"
        payloads[filename] = [row for row in rows(inputs / filename) if row["case_id"] in admitted]
        case_ids_by_run[run["id"]] = [row["case_id"] for row in payloads[filename]]
    if gate["policies"]["runtime"]["tokenizer"].get("runtime_family") in FULL_COHORT_FAMILIES:
        require(all(ids == [case["id"] for case in cases] for ids in case_ids_by_run.values()),
                "Small final-test roles must share the full sealed case order.")
    return publish(output, payloads, {"status": "sealed_before_inference", "freeze": gate["binding"],
                                     "input_manifest": binding(inputs / "manifest.json"),
                                     "reference_source": binding(references), "registered_slots": len(census),
                                     "admitted_cases": len(cases), "accepted_references": sum(len(c["acceptable"]) for c in cases),
                                     "case_ids": [c["id"] for c in cases], "case_ids_by_run": case_ids_by_run,
                                     "predictions_seen": False})


def validate_request(request, run, gate):
    require(request["split"] == SCOPE and request["run_id"] == run["id"]
            and request["format"] == run["format"] and request["freeze_sha256"] == gate["binding"]["sha256"],
            "Request is not bound to this final-test run.")
    require(not {"primary", "accepted_predictions", "acceptable", "document", "judgment", "rationale"}.intersection(request),
            "Predictor request leaked reference-side fields.")
    require(request["condition"] == run["condition"]
            and request["condition_sha256"] == digest(run["condition"])
            and request["input_sha256"] == evaluation.semantic_input_hash(request["input"])
            and request["messages"] == conditions.expected_messages(request["input"], run["format"], run["condition"])
            and request["messages_sha256"] == digest(request["messages"])
            and request["prefix_ids_sha256"] == digest(request["prefix_ids"]), "Request binding changed.")
    vocab_size = gate["policies"]["runtime"]["tokenizer"].get("vocab_size", 262144)
    input_limit = (7680 if gate["policies"]["runtime"]["tokenizer"].get("runtime_family") in FULL_COHORT_FAMILIES
                   else formats.INPUT_LIMIT)
    require(0 < len(request["prefix_ids"]) <= input_limit
            and all(type(t) is int and 0 <= t < vocab_size for t in request["prefix_ids"]), "Invalid native request tokens.")


def run_identity(gate, run):
    role = "native_base" if run["role"] == "base_icl" else run["role"]
    return gate["models"][role]["identity"] if role in MODEL_ROLES else {"baseline": role, "code_sha256": sha256(conditions.__file__)}


def record_run(freeze, sealed, run_id, output, predictor=None, *, tokenizer=None, predictor_identity=None):
    """The predictor receives only frozen requests; one failure record is saved per call."""
    gate = load_gate(freeze)
    manifest = load_stage(gate, sealed, "sealed_before_inference")
    run = next(r for r in gate["receipt"]["runs"] if r["id"] == run_id)
    requests_path = Path(sealed) / f"requests-{run_id}.jsonl"
    requests = rows(requests_path)
    require([r["case_id"] for r in requests] == manifest["case_ids_by_run"][run_id], "Request roster differs from sealed cases.")
    for request in requests:
        validate_request(request, run, gate)
    tokenizer = tokenizer or load_tokenizer(gate)
    if predictor is None:
        require(run["role"] in {"always_no_action", "consistent_history_rule"}, "A frozen model predictor is required.")
        def predictor(request):
            action = conditions.baseline_prediction(request["input"], run["role"])
            return {"text": formats.serialize(request["input"], action, run["format"]), "finish_reason": "stop"}
    else:
        require(gate["receipt"].get("engineering_fixture_only") is True,
                "Real final-test models must use the portable native execution/admission path.")
        require(predictor_identity == run_identity(gate, run), "Model loader must report the actual frozen identity.")
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Use a fresh inference directory.")
    output.mkdir(parents=True, mode=0o700)
    registration = {"freeze": gate["binding"], "sealed_manifest": binding(Path(sealed) / "manifest.json"),
                    "requests": binding(requests_path), "run_id": run_id, "model_identity": run_identity(gate, run),
                    "requested": len(requests), "started_at": now(), "final_test_bodies_read": True,
                    "reference_labels_available_to_predictor": False,
                    "execution_kind": "engineering_callback" if predictor_identity is not None else "deterministic_baseline"}
    write(output / "run-registration.json", registration)
    (output / "run-registration.json").chmod(0o600)
    observed = 0
    with (output / "observations.jsonl").open("x") as stream:
        (output / "observations.jsonl").chmod(0o600)
        for request in requests:
            observation = {key: request[key] for key in ("case_id", "input_sha256", "messages_sha256", "prefix_ids_sha256",
                                                         "condition_sha256", "format", "run_id", "freeze_sha256")}
            observation.update(split=SCOPE, model_identity=run_identity(gate, run), request_sha256=digest(request))
            started = time.perf_counter()
            try:
                require(conditions.native_prefix(tokenizer, request["messages"]) == request["prefix_ids"], "Actual native prefix differs.")
                value = predictor(copy.deepcopy(request))
                require(isinstance(value, dict) and isinstance(value.get("text"), str)
                        and value.get("finish_reason") in {"stop", "length"}, "Malformed predictor result.")
                observation.update(status="completed", text=value["text"], finish_reason=value["finish_reason"])
                for key in ("output_ids", "text_raw", "terminal_token_id"):
                    if key in value:
                        observation[key] = value[key]
            except Exception as error:
                observation.update(status="request_failure", error_type=type(error).__name__, error=str(error))
            observation["controller_wall_ms"] = (time.perf_counter() - started) * 1000
            observation["generation_ms"] = None
            observation["timing_source"] = "controller_only_not_gpu_generation"
            stream.write(packed(observation) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            observed += 1
    return finish_output(output, {**registration, "status": "inference_complete", "observed": observed})


def export_execution(freeze, sealed, run_id, output):
    import infer_task_test_actions_v3 as native
    gate = load_gate(freeze)
    manifest = load_stage(gate, sealed, "sealed_before_inference")
    run = next(row for row in gate["receipt"]["runs"] if row["id"] == run_id)
    require(run["role"] in {"native_base", "base_icl", "sft", "sft_dpo"}, "Export only registered model runs.")
    role = "native_base" if run["role"] == "base_icl" else run["role"]
    model = gate["models"][role]
    runtime = model_runtime(model, engineering_fixture_only=gate["receipt"].get("engineering_fixture_only", False))
    config = read_json(bound(model["inference_configuration"]))
    require(config["format"] == run["format"]
            and config["generation"] == {key: value for key, value in gate["policies"]["runtime"]["generation"].items()
                                          if key != "enable_thinking"}, "Frozen native inference configuration changed.")
    request_path = Path(sealed) / f"requests-{run_id}.jsonl"
    requests = rows(request_path)
    require([row["case_id"] for row in requests] == manifest["case_ids_by_run"][run_id], "Portable request cohort changed.")
    for request in requests:
        validate_request(request, run, gate)
    specification = {
        "scope": SCOPE, "local_gate_passed": True, "gate_checked_at": now(), "freeze_sha256": gate["binding"]["sha256"],
        "sealed_manifest_sha256": sha256(Path(sealed) / "manifest.json"), "run": run,
        "case_ids": manifest["case_ids_by_run"][run_id], "requests_sha256": sha256(request_path),
        "model_identity": model["identity"], "configuration_sha256": model["inference_configuration"]["sha256"],
        "preparation_manifest_sha256": model["preparation_manifest"]["sha256"],
        "adapter_version": "gemma-actions-v3-dpo" if role == "sft_dpo" else "gemma-actions-v3-sft",
        "generation": config["generation"],
        "libraries": (read_json(bound(model["preparation_manifest"]))["environment"]
                      if model.get("runtime_family") == "small-actions-final-v1" else config["libraries"]),
        "source_sha256": runtime["source_sha256"], "runtime_manifest_sha256": runtime["runtime_manifest_sha256"],
        "controller_source_sha256": gate["receipt"]["source_sha256"],
        "schema_constrained": False, "engineering_fixture_only": gate["receipt"].get("engineering_fixture_only", False),
    }
    if model.get("runtime_family") in FULL_COHORT_FAMILIES:
        specification.update(runtime_family=model["runtime_family"], adapter_version=model["identity"]["kind"],
                             vocab_size=gate["policies"]["runtime"]["tokenizer"]["vocab_size"],
                             physical_context_tokens=8192, inference_prefix_limit=7680, output_tokens=512,
                             inference_policy_sha256=model["inference_policy"]["sha256"])
        if model["runtime_family"] == "gemma26-final-native-v1":
            specification["native_preparation_manifest_sha256"] = read_json(
                bound(model["preparation_manifest"]))["native_preparation"]["sha256"]
    native.validate_requests(requests, specification)
    return publish(output, {"execution.json": specification, "requests.jsonl": requests},
                   {"version": native.BUNDLE_VERSION, "status": "authorized_label_free_execution"})


def admit_observations(freeze, sealed, execution_bundle, remote_output, output, *, tokenizer=None):
    import infer_task_test_actions_v3 as native
    gate = load_gate(freeze)
    release = load_stage(gate, sealed, "sealed_before_inference")
    portable_manifest = Path(execution_bundle) / "manifest.json"
    exported_run = read_json(Path(execution_bundle) / "execution.json")["run"]
    run = next(row for row in gate["receipt"]["runs"] if row["id"] == exported_run["id"])
    role = "native_base" if run["role"] == "base_icl" else run["role"]
    runtime = model_runtime(gate["models"][role], engineering_fixture_only=gate["receipt"].get("engineering_fixture_only", False))
    spec, requests = native.load_bundle(execution_bundle, sha256(portable_manifest), expected_runtime=runtime)
    require(spec["freeze_sha256"] == gate["binding"]["sha256"]
            and spec["sealed_manifest_sha256"] == sha256(Path(sealed) / "manifest.json"), "Remote authorization belongs to another freeze.")
    require(spec["run"] == run and spec["model_identity"] == run_identity(gate, run), "Remote run/model identity changed.")
    local_requests = Path(sealed) / f"requests-{run['id']}.jsonl"
    role = "native_base" if run["role"] == "base_icl" else run["role"]
    require(spec["configuration_sha256"] == gate["models"][role]["inference_configuration"]["sha256"]
            and spec["preparation_manifest_sha256"] == gate["models"][role]["preparation_manifest"]["sha256"]
            and spec["requests_sha256"] == sha256(local_requests)
            and spec["controller_source_sha256"] == gate["receipt"]["source_sha256"]
            and spec["case_ids"] == release["case_ids_by_run"][run["id"]], "Remote configuration or request cohort changed.")
    remote_output = Path(remote_output)
    remote_manifest = read_json(remote_output / "manifest.json")
    require(remote_manifest["version"] == native.VERSION and remote_manifest["scope"] == SCOPE
            and remote_manifest["status"] == "inference_complete", "Remote run did not finish its request census.")
    evaluation.verify_files(remote_output, remote_manifest)
    info = read_json(remote_output / "run.json")
    require(info["version"] == native.VERSION and info["scope"] == SCOPE and info["status"] == "inference_complete"
            and info["bundle_sha256"] == sha256(portable_manifest)
            and info["freeze_sha256"] == spec["freeze_sha256"]
            and info["sealed_manifest_sha256"] == spec["sealed_manifest_sha256"]
            and info["run_id"] == run["id"] and info["requests_sha256"] == spec["requests_sha256"]
            and info["configuration_sha256"] == spec["configuration_sha256"]
            and info["model_identity"] == spec["model_identity"]
            and info["source_sha256"] == spec["source_sha256"]
            and info["runtime_manifest_sha256"] == spec["runtime_manifest_sha256"]
            and info["controller_source_sha256"] == spec["controller_source_sha256"]
            and info["schema_constrained"] is False and info["final_test_bodies_read"] is True
            and info["reference_labels_available_to_predictor"] is False, "Remote execution receipt changed.")
    observations = rows(remote_output / "observations.jsonl")
    require([row["case_id"] for row in observations] == [row["case_id"] for row in requests]
            and info["observed"] == info["requested"] == len(requests), "Missing or extra remote observations.")
    if info["model_identity_verified"]:
        require(info["actual_model_identity"] == spec["model_identity"] and "runtime.json" in remote_manifest["files_sha256"],
                "Native loading proof is missing.")
    else:
        require(info["actual_model_identity"] is None, "Unverified model has an asserted actual identity.")
    tokenizer = tokenizer or load_tokenizer(gate)
    for request, observation in zip(requests, observations, strict=True):
        require(all(observation.get(key) == value for key, value in native.observation_identity(request, spec).items()),
                "Remote observation input/model binding changed.")
        require(conditions.native_prefix(tokenizer, request["messages"]) == request["prefix_ids"], "Local native prefix readback changed.")
        if observation["status"] == "completed":
            require(info["model_identity_verified"] is True and observation["model_identity_verified"] is True
                    and observation["generation_attempted"] is True
                    and observation["timing_source"] == "cuda_synchronized_generate"
                    and type(observation["generation_ms"]) in (int, float)
                    and math.isfinite(observation["generation_ms"]) and observation["generation_ms"] >= 0,
                    "Completed output lacks actual native execution/timing.")
            tokens = observation["output_ids"]
            require(0 < len(tokens) <= formats.OUTPUT_LIMIT
                    and all(type(token) is int and 0 <= token < spec.get("vocab_size", 262144) for token in tokens)
                    and observation["terminal_token_id"] == tokens[-1]
                    and observation["text"] == tokenizer.decode(tokens, skip_special_tokens=True)
                    and observation["text_raw"] == tokenizer.decode(tokens, skip_special_tokens=False)
                    and observation["finish_reason"] == ("stop" if tokens[-1] in spec["generation"]["eos_token_id"] else "length"),
                    "Native output token/text/terminal binding changed.")
        else:
            require(observation["status"] == "request_failure" and observation["generation_ms"] is None
                    and bool(observation["error_type"]) and observation["timing_source"] == "not_measured",
                    "Failed native request was disguised as an answer or measured latency.")
            require(info["model_identity_verified"] or (observation["model_identity_verified"] is False
                                                       and observation["generation_attempted"] is False),
                    "Generation was claimed before model identity verification.")
    return publish(output, {"observations.jsonl": observations, "remote-run.json": info},
                   {"status": "inference_complete", "freeze": gate["binding"], "execution_kind": "remote_native",
                    "sealed_manifest": binding(Path(sealed) / "manifest.json"), "requests": binding(local_requests),
                    "remote_manifest": binding(remote_output / "manifest.json"), "execution_bundle": binding(portable_manifest),
                    "run_id": run["id"], "model_identity": spec["model_identity"], "requested": len(requests), "observed": len(observations),
                    "final_test_bodies_read": True, "reference_labels_available_to_predictor": False})


def census_summary(dispositions, requested=()):
    requested = set(requested)
    result = {}
    for dimension in ("source_group", "registered_source_group", "stratum", "language"):
        groups = defaultdict(list)
        for row in dispositions:
            groups[row.get(dimension, row["source_group"])].append(row)
        result[dimension] = {key: {"registered": len(values), "admitted": sum(r["status"] == "admitted" for r in values),
                                  "requested": sum(r["id"] in requested for r in values),
                                  "dispositions": dict(Counter(r["status"] for r in values)),
                                  "natural_windows": len({r["window_id"] for r in values if r.get("window_id")})}
                             for key, values in sorted(groups.items())}
    return result


def grade(freeze, sealed, observed, output):
    import recompute_actions_v3_results as evidence
    gate = load_gate(freeze)
    release = load_stage(gate, sealed, "sealed_before_inference")
    execution = load_stage(gate, observed, "inference_complete")
    require(execution["sealed_manifest"] == binding(Path(sealed) / "manifest.json"), "Run belongs to another sealed test.")
    run = next(r for r in gate["receipt"]["runs"] if r["id"] == execution["run_id"])
    request_path = Path(sealed) / f"requests-{run['id']}.jsonl"
    require(execution["requests"] == binding(request_path)
            and execution["model_identity"] == run_identity(gate, run)
            and execution["final_test_bodies_read"] is True, "Inference binding changed.")
    requests, observations = rows(request_path), rows(Path(observed) / "observations.jsonl")
    require([r["case_id"] for r in observations] == [r["case_id"] for r in requests] == release["case_ids_by_run"][run["id"]],
            "Every request needs exactly one ordered observation, including failures.")
    require(execution["observed"] == len(observations) == execution["requested"], "Run completion count changed.")
    execution_evidence = evidence.verify_final_execution(ROOT, Path(observed) / "manifest.json", gate["receipt"],
                                                        Path(sealed) / "manifest.json", run, requests, observations)
    cases = {case["id"]: case for case in rows(Path(sealed) / "cases.jsonl")}
    proofs = {proof["id"]: proof for proof in rows(Path(sealed) / "reference-executions.jsonl")}
    records = []
    for request, observation in zip(requests, observations, strict=True):
        validate_request(request, run, gate)
        require(observation["split"] == SCOPE and observation["run_id"] == run["id"]
                and observation["freeze_sha256"] == gate["binding"]["sha256"]
                and observation["model_identity"] == run_identity(gate, run)
                and observation["request_sha256"] == digest(request)
                and observation["prefix_ids_sha256"] == request["prefix_ids_sha256"], "Observation identity changed.")
        case = cases[request["case_id"]]
        require(case["split"] == case["source"]["split"] == SCOPE
                and case["input"] == request["input"]
                and case["reference_proof_sha256"] == digest(proofs[case["id"]]), "Sealed case/reference proof changed.")
        conditioned = {**case, "condition": run["condition"], "condition_sha256": request["condition_sha256"]}
        records.append(evaluation.evaluate_one(conditioned, observation, run["format"]))
    dispositions = rows(Path(sealed) / "dispositions.jsonl")
    grouped = {}
    for dimension in ("reference_action", "language", "action_language", "predicted_action", "predicted_action_language",
                      "source_group", "stratum", "scenario_family"):
        groups = defaultdict(list)
        for row in records:
            groups[row[dimension] if row[dimension] is not None else "unavailable"].append(row)
        grouped[dimension] = {key: evaluation.summarize(value) for key, value in sorted(groups.items())}
    report = {"scope": SCOPE, "run_id": run["id"], "summary": evaluation.summarize(records), "by": grouped,
              "registered_slots": len(dispositions), "disposition_counts": dict(Counter(r["status"] for r in dispositions)),
              "canonical_admitted_cases": release["admitted_cases"],
              "run_eligibility_exclusions": [r for r in rows(Path(sealed) / "run-exclusions.jsonl") if r["run_id"] == run["id"]],
              "source_stratum_denominators": census_summary(dispositions, (r["case_id"] for r in requests)),
              "source_groups": len({row["source_group"] for row in dispositions}),
              "registered_source_groups": len({row.get("registered_source_group", row["source_group"]) for row in dispositions}),
              "technical_source_resolution": gate.get("source_resolution"),
              "natural_windows": len({row["window_id"] for row in dispositions if row.get("window_id")}),
              "uncertainty": "Within-source states and authored siblings are correlated. Wilson intervals are descriptive; they do not treat slots as independent sources.",
              "model_selection_changed": False}
    report["engineering_fixture_only"] = gate["receipt"].get("engineering_fixture_only", False)
    return publish(output, {"graded.jsonl": records, "report.json": report},
                   {"status": "graded", "freeze": gate["binding"], "sealed_manifest": binding(Path(sealed) / "manifest.json"),
                    "observation_manifest": binding(Path(observed) / "manifest.json"), "run_id": run["id"],
                    "execution_evidence": execution_evidence})


def normalize_scribe_response(raw, provenance):
    """Pure fixture-testable adapter, using the actual server normalizer and app importer."""
    sys.path.insert(0, str(ROOT / "server"))
    from app.asr.scribe import normalize_transcript
    duration, language = provenance["duration_seconds"], provenance["language"]
    require(provenance["source_id"] == opaque_source_id(provenance["window_id"]), "Natural source identity must be opaque and window-bound.")
    require(language in {"en", "ko"} and math.isfinite(duration) and duration > 0, "Invalid ASR excerpt.")
    require(raw["language_code"] in {language, {"en": "eng", "ko": "kor"}[language]}, "ASR language differs from source registration.")
    options = {"model_id": "scribe_v2", "no_verbatim": "false", "timestamps_granularity": "word",
               "diarize": "true", "tag_audio_events": "true", "language_code": language}
    content, annotations = normalize_transcript(raw, duration, provenance["source_id"], True)
    metadata = {"version": 1, "provider": "elevenlabs", "model": "scribe_v2", "language_code": raw["language_code"],
                "duration": duration, "request": options, "raw_response": raw, "annotations": annotations}
    result = subprocess.run(["node", str(ROOT / "ml/import_task_test_actions_v3.cjs")],
                            input=packed({"content": content, "source": provenance["source_id"],
                                          "language": language, "transcription": metadata}),
                            text=True, capture_output=True, check=True)
    document = json.loads(result.stdout)
    for index, item in enumerate(document["content"]):
        item["uuid"] = str(uuid5(NAMESPACE_URL, f"{provenance['source_id']}:{index}"))
    document["metadata"]["transcriptions"][0]["id"] = str(uuid5(NAMESPACE_URL, provenance["source_id"] + ":asr"))
    document["metadata"]["test_source_provenance"] = copy.deepcopy(provenance)
    return document


def import_scribe(freeze, raw_response, provenance_path, output):
    gate = load_gate(freeze)
    provenance = read_json(provenance_path)
    window = verify_window_identity(gate, provenance)
    require(provenance["split"] == SCOPE and provenance["raw_response_sha256"] == sha256(raw_response)
            and datetime.fromisoformat(provenance["request_started_at"]) >= gate["frozen_at"],
            "ASR response was not acquired under the frozen source allocation.")
    require(provenance["duration_seconds"] == 90 and provenance["source_start_seconds"] >= 0
            and provenance["request_options"] == {"model_id": "scribe_v2", "no_verbatim": "false",
                "timestamps_granularity": "word", "diarize": "true", "tag_audio_events": "true",
                "language_code": provenance["language"]}, "ASR request or fixed window changed.")
    if window.get("start_seconds") is not None:
        require(provenance["source_start_seconds"] == window["start_seconds"], "ASR excerpt moved from its metadata window.")
    else:
        resolution = read_json(bound(provenance["duration_resolution"]))
        require(resolution["before_words"] is True and resolution["recording_id"] == window["publisher_recording_id"]
                and resolution["duration_seconds"] >= 180
                and provenance["source_start_seconds"] == resolution["duration_seconds"] / 2 - 45,
                "Unresolved metadata timing was replaced using transcript content.")
    with wave.open(str(bound(provenance["pcm_audio"])), "rb") as audio:
        require(audio.getframerate() == 16000 and audio.getnchannels() == 1 and audio.getsampwidth() == 2
                and audio.getnframes() == 90 * 16000, "ASR audio differs from the frozen PCM window.")
    document = normalize_scribe_response(read_json(raw_response), provenance)
    return publish(output, {"document.json": document, "provenance.json": provenance},
                   {"status": "imported_unlabeled", "freeze": gate["binding"], "raw_response": binding(raw_response),
                    "initial_input_edited": False, "acoustic_certification": "none"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    code = commands.add_parser("code-sources")
    code.add_argument("--output", required=True, type=Path)
    resolution = commands.add_parser("resolve-sources")
    for name in ("metadata-manifest", "assessments", "output"):
        resolution.add_argument("--" + name, required=True, type=Path)
    for name in ("check-freeze", "freeze-inputs", "seal-references", "baseline", "grade", "import-scribe", "export-execution", "admit-observations"):
        command = commands.add_parser(name)
        command.add_argument("--freeze", required=True, type=Path)
        if name != "check-freeze":
            command.add_argument("--output", required=True, type=Path)
        for field in {"freeze-inputs": ("candidates",), "seal-references": ("inputs", "references"),
                      "baseline": ("sealed", "run-id"), "grade": ("sealed", "observed"),
                      "export-execution": ("sealed", "run-id"),
                      "admit-observations": ("sealed", "execution-bundle", "remote-output"),
                      "import-scribe": ("raw-response", "provenance")}.get(name, ()):
            command.add_argument("--" + field, required=True)
    args = parser.parse_args()
    if args.command == "code-sources":
        write(args.output, code_sources())
        result = {"source_manifest": binding(args.output), "test_bodies_read": False}
    elif args.command == "resolve-sources":
        result = {"source_resolution": resolve_sources(args.metadata_manifest, args.assessments, args.output), "test_bodies_read": False}
    elif args.command == "check-freeze":
        gate = load_gate(args.freeze)
        result = {"status": "passed", "freeze": gate["binding"], "registered_slots": len(gate["roster"]), "test_bodies_read": False}
    elif args.command == "freeze-inputs":
        result = freeze_inputs(args.freeze, args.candidates, args.output)
    elif args.command == "seal-references":
        result = seal_references(args.freeze, args.inputs, args.references, args.output)
    elif args.command == "baseline":
        result = record_run(args.freeze, args.sealed, args.run_id, args.output)
    elif args.command == "grade":
        result = grade(args.freeze, args.sealed, args.observed, args.output)
    elif args.command == "export-execution":
        result = export_execution(args.freeze, args.sealed, args.run_id, args.output)
    elif args.command == "admit-observations":
        result = admit_observations(args.freeze, args.sealed, args.execution_bundle, args.remote_output, args.output)
    else:
        result = import_scribe(args.freeze, args.raw_response, args.provenance, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
