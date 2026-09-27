"""Seal E1 inputs before references, behind a completed model/runtime freeze.

This module contains no final-test scenes and never generates model predictions.
The isolated author supplies new source documents only after the parent freeze.
Only aggregate censuses and artifact hashes are printed by the CLI.
"""

import argparse
import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import unicodedata

import passive_actions as contract
from context_policy import ContextBudgetError
from grading import transcript_media
from passive_actions_final_sources import cues_from_srt, verify_initial_document
from replay_session import replay, session_hash
from session_cases import replayable_session
from verify_release_export import Exporter, verify_export_delta, verify_operation

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-actions-e1-sealed-preparation-v1"
LABELS = {
    "known_edit", "known_preserve", "ambiguous", "unsupported",
    "insufficient_audio", "out_of_window", "source_unavailable",
    "context_ineligible", "execution_invalid", "structural_state_unavailable",
}
KNOWN = {"known_edit", "known_preserve"}
UUID = re.compile(r"(?:[a-z_]+:)?[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


def require(value, code):
    if not value:
        raise ValueError(code)


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def bound(binding):
    path = (ROOT / binding["path"]).resolve()
    require(path.is_relative_to(ROOT), "bound_path_outside_repository")
    require(file_sha(path) == binding["sha256"], "bound_file_changed")
    return path


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, values):
    with path.open("x") as stream:
        for value in values:
            stream.write(packed(value) + "\n")


def recording_windows(duration):
    require(type(duration) in (int, float) and duration >= 450, "recording_too_short")
    return [{"start": duration * fraction - 45, "end": duration * fraction + 45}
            for fraction in (0.2, 0.4, 0.6, 0.8)]


def source_closure(entries):
    found, pending = {}, list(entries)
    while pending:
        path = Path(pending.pop()).resolve()
        if path in found:
            continue
        require(path.is_relative_to(ROOT) and path.is_file(), "source_path_escaped_project")
        found[path] = file_sha(path)
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in modules:
                for directory in (ROOT / "ml", ROOT / "server"):
                    target = directory / Path(*name.split(".")).with_suffix(".py")
                    if target.is_file():
                        pending.append(target)
    return {str(path.relative_to(ROOT)): value for path, value in sorted(found.items())}


def validate_freeze(protocol, binding):
    require(binding.get("status") == "frozen_for_sealed_generation", "runtime_freeze_pending")
    require(binding["protocol_sha256"] == digest(protocol), "protocol_value_changed")
    require(binding["models"] == protocol["models"], "frozen_models_changed")
    require(set(binding["models"]) == {"native_base", "final11"}, "wrong_model_conditions")
    policy = binding["effective_policy"]
    require(
        policy["contract_version"] == contract.VERSION
        and policy["allowed_actions"] == list(contract.ALLOWED_ACTIONS)
        and policy["gated_actions"] == []
        and policy["input_tokens"] == 3584 and policy["max_tokens"] == 512
        and policy["context_tokens"] == 4096
        and policy["history_limit"] == policy["feedback_limit"] == 3
        and policy["temperature"] == 0 and policy["seed"] == 42
        and policy["native_stop_ids"] == [1, 106, 50]
        and policy["tool_handoff_credited"] is False
        and policy["schema_constrained"] is True
        and type(policy["cache_prompt"]) is bool
        and policy["cache_reset_rule"] == ("reset_each_arm_reuse_within_arm" if policy["cache_prompt"] else "reset_each_request")
        and policy["timeout_seconds"] > 0 and policy["retry_policy"] == "none"
        and policy["icl"] is False and policy["truncation"] is False,
        "invalid_or_unresolved_effective_policy",
    )
    require(
        bool(binding["parent_authorization"])
        and binding["test_bodies_generated_before_freeze"] == 0
        and binding["test_bodies_inspected_before_freeze"] == 0
        and binding["author_actor"] != binding["reviewer_actor"],
        "missing_independent_prefreeze_authorization",
    )
    require(set(binding["sources"]) == set(protocol["source_plan"]), "source_allocation_changed")
    for group, source in binding["sources"].items():
        plan = protocol["source_plan"][group]
        if plan["kind"] == "natural_recording":
            require(source["status"] in {"available_uninspected", "source_unavailable"},
                    "natural_source_not_resolved")
            if source["status"] == "available_uninspected":
                require(
                    source["uri"].startswith("https://")
                    and source["windows"] == recording_windows(source["duration_seconds"])
                    and all(re.fullmatch(r"[0-9a-f]{64}", source[k])
                            for k in ("media_sha256", "captions_sha256"))
                    and source["selected_before_content_inspection"] is True,
                    "natural_metadata_or_windows_unfrozen",
                )
            else:
                require(bool(source["reason_code"]), "missing_source_exclusion_reason")
        else:
            require(source["status"] == "generation_authorized"
                    and source["parent_uri"] == plan["parent_uri"]
                    and bool(source["generator_identity"]) and bool(source["seed"]),
                    "authored_parent_generation_unbound")


def load_gate(protocol_path, binding_path):
    protocol, binding = read(protocol_path), read(binding_path)
    validate_freeze(protocol, binding)
    require(binding["protocol_file_sha256"] == file_sha(protocol_path), "protocol_bytes_changed")
    require(binding["preparer_sha256"] == file_sha(__file__), "preparer_changed")
    closure = source_closure([__file__, contract.PROMPT_PATH, ROOT / "ml/export_release_states.cjs",
                              ROOT / "app/src/core/webvtt.ts", ROOT / "app/src/state/editor/selectors.ts",
                              ROOT / "app/src/core/document.ts", ROOT / "app/package.json", ROOT / "app/package-lock.json"])
    require(all(binding["sources_sha256"].get(name) == value for name, value in closure.items()),
            "unbound_preparation_execution_source")
    for name, expected in binding["sources_sha256"].items():
        bound({"path": name, "sha256": expected})
    for value in protocol["bindings"].values():
        bound(value)
    for value in binding["inputs"].values():
        bound(value)
    for key in ("l2_decision", "parent_authorization_receipt", "source_selection_receipt"):
        require(key in binding["inputs"], "missing_prefreeze_receipt")
    tokenizer = ROOT / binding["tokenizer_path"]
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        require(bound(binding["inputs"]["tokenizer:" + name]) == (tokenizer / name).resolve(),
                "unbound_native_tokenizer_file")
    require(bound(binding["inputs"]["contract"]) == Path(contract.__file__).resolve()
            and bound(binding["inputs"]["prompt"]) == contract.PROMPT_PATH.resolve(),
            "unbound_contract_or_prompt")
    common_runtime = None
    for condition in ("native_base", "final11"):
        config = read(bound(binding["inputs"][condition + "_gateway"]))
        require(config["contract_version"] == contract.VERSION
                and config["model_sha256"] == protocol["models"][condition]["sha256"]
                and Path(config["model_path"]).resolve() == Path(protocol["models"][condition]["path"]).resolve()
                and config.get("cache_prompt", False) == binding["effective_policy"]["cache_prompt"],
                "gateway_model_or_contract_changed")
        settings = {key: config.get(key) for key in
                    ("contract_version", "chat_template_path", "chat_template_sha256", "runtime_build")}
        require(common_runtime is None or settings == common_runtime, "model_conditions_use_different_runtime")
        common_runtime = settings
    return protocol, binding


def corpus_text(value):
    value = value.get("input", value)
    items = value.get("items", [])
    fields = value.get("item_fields")
    if fields and items and isinstance(items[0], list):
        items = [dict(zip(fields, item, strict=True)) for item in items]
    text = " ".join(item.get("text", "") for item in items if isinstance(item, dict))
    return " ".join(unicodedata.normalize("NFC", text).split())


def exposed_text_hashes(metadata, additional):
    result = set()
    collections = {**metadata["historical_exposed_case_collections_sha256"], **additional}
    for path, expected in collections.items():
        for row in rows(bound({"path": path, "sha256": expected})):
            text = corpus_text(row)
            if text:
                result.add(digest(text))
    return result


def timed(document):
    return [i for i in document["content"] if i["type"] in {"text", "non_text"}]


def scheduled_focus(document, previous=None):
    current = timed(document)
    require(bool(current), "no_source_timed_items")
    if previous is None:
        return current[(len(current) - 1) // 2]["uuid"]
    old = next(i for i in timed(previous["document"]) if i["uuid"] == previous["focus_uuid"])
    if any(i["uuid"] == old["uuid"] for i in current):
        return old["uuid"]
    same = [i for i in current if i.get("source") == old.get("source")]
    require(bool(same), "focus_source_disappeared")
    return min(same, key=lambda i: (abs(i["sourceStart"] - old["sourceStart"]),
                                   i["sourceStart"], i["uuid"]))["uuid"]


def validate_checkpoint(runtime, checkpoint, previous):
    require(runtime["split"] == "test" and not runtime["selection"], "wrong_split_or_selection")
    require(bool(runtime["document_id"]) and (previous is None or runtime["document_id"] == previous["document_id"]),
            "checkpoint_document_identity_changed")
    require(runtime["feedback_journal"] is None
            or runtime["feedback_journal"]["document_id"] == runtime["document_id"],
            "feedback_document_identity_changed")
    require(runtime["focus_uuid"] == scheduled_focus(runtime["document"], previous),
            "reference_independent_focus_changed")
    history = runtime["history"]
    require(all(h["sequence"] <= runtime["sequence"] for h in history), "future_history_in_runtime")
    if runtime["feedback_journal"]:
        require(all(e["sequence"] <= runtime["sequence"]
                    for e in runtime["feedback_journal"]["records"]), "future_feedback_in_runtime")
    if checkpoint == 0:
        require(runtime["sequence"] == 0 and history == []
                and runtime["feedback_journal"] is None, "initial_state_has_future_history")
    elif checkpoint == 1:
        require(previous is not None and len(history) == 1
                and history[0]["before"] == previous["document"]
                and history[0]["after"] == runtime["document"]
                and history[0]["sequence"] == runtime["sequence"], "missing_actual_first_transaction")
    else:
        require(previous is not None and runtime["sequence"] > previous["sequence"]
                and history[:len(previous["history"])] == previous["history"],
                "checkpoint_history_not_consecutive")
        journal = runtime["feedback_journal"]
        require(journal is not None and any(
            event["kind"] == "record" and previous["sequence"] < event["sequence"] <= runtime["sequence"]
            for event in journal["records"]), "missing_actual_feedback_event")


def prepare_input_records(candidates, slots, protocol, binding, tokenizer, exposure):
    require([r["id"] for r in candidates] == [s["id"] for s in slots], "slot_census_changed")
    builder = contract.PassiveContextBuilder(tokenizer)
    public, inputs, native, runtime_rows = [], [], [], []
    previous = {}
    for candidate, slot in zip(candidates, slots, strict=True):
        require(set(candidate) <= {"id", "status", "reason_code", "runtime", "source", "source_binding"},
                "references_or_unregistered_fields_in_input_stage")
        source = binding["sources"][slot["source_group_id"]]
        status = candidate["status"]
        if source["status"] == "source_unavailable":
            require(status == "source_unavailable", "unavailable_source_replaced")
        if status != "available":
            require(status in LABELS - KNOWN and candidate.get("reason_code"), "invalid_input_exclusion")
            public.append({"id": slot["id"], "status": status, "reason_code": candidate["reason_code"]})
            previous.pop(slot["episode_id"], None)
            continue
        provenance, runtime = candidate["source"], copy.deepcopy(candidate["runtime"])
        require(provenance["source_group_id"] == slot["source_group_id"]
                and provenance["kind"] == slot["source_kind"]
                and provenance["language"] == slot["language"]
                and provenance["acoustically_verified"] is False
                and provenance["boundary_claim"] == "source_timing_only",
                "source_or_acoustic_provenance_changed")
        require(runtime["id"] == slot["id"], "runtime_identity_changed")
        require(runtime["caption_profile"] == contract.CaptionProfile().model_dump(),
                "caption_profile_differs_from_actual_gateway_default")
        require(all(UUID.fullmatch(item["source"]) and item["source"] == provenance["source_id"]
                    for item in timed(runtime["document"])),
                "source_identifiers_must_be_opaque")
        source_path = bound(candidate["source_binding"])
        if slot["source_kind"] == "natural_recording":
            window = source["windows"][int(slot["episode_id"].rsplit("s", 1)[1]) - 1]
            require(candidate["source_binding"]["sha256"] == source["captions_sha256"]
                    and provenance["timing_basis"] == "publisher_caption_cue"
                    and all(window["start"] <= item["sourceStart"]
                            and item["sourceStart"] + item["length"] <= window["end"]
                            for item in timed(runtime["document"])),
                    "natural_caption_hash_or_original_window_changed")
            require(provenance["editing_provenance"] == (
                "unmodified_publisher_caption" if slot["checkpoint"] == 0 else "source_derived_authored_edit"),
                "natural_recording_does_not_imply_natural_edit_labels")
            if slot["checkpoint"] == 0:
                verify_initial_document(runtime["document"], cues_from_srt(source_path.read_text(encoding="utf-8-sig")),
                                        window, provenance["source_id"], slot["language"])
        else:
            require(provenance["timing_basis"] == "authored_symbolic_not_audio"
                    and provenance["editing_provenance"] == "authored_symbolic",
                    "symbolic_timing_mislabeled")
        validate_checkpoint(runtime, slot["checkpoint"], previous.get(slot["episode_id"]))
        feedback = (contract.PassiveFeedbackLedger.restore(runtime["feedback_journal"])
                    if runtime["feedback_journal"] is not None else None)
        try:
            bundle = builder.build(runtime["document"], runtime["focus_uuid"],
                                   as_of=runtime["sequence"], selection=runtime["selection"],
                                   history=runtime["history"], feedback=feedback,
                                   caption_profile=runtime["caption_profile"])
        except ContextBudgetError:
            public.append({"id": slot["id"], "status": "context_ineligible",
                           "reason_code": "mandatory_native_context_overflow"})
            previous[slot["episode_id"]] = runtime
            continue
        require(bundle["input"]["language"] == slot["language"], "runtime_language_changed")
        require(digest(corpus_text(bundle)) not in exposure, "visible_text_duplicate_of_exposed_corpus")
        messages = contract.canonical_messages(bundle["input"])
        prefix = contract.token_ids(tokenizer, bundle["input"])
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        bos = getattr(tokenizer, "bos_token", None)
        if bos and rendered.startswith(bos):
            rendered = rendered[len(bos):]
        record = {"id": slot["id"], "split": "test", "language": slot["language"],
                  "source": provenance, "provenance": provenance, "source_binding": candidate["source_binding"],
                  "input": bundle["input"], "bundle": bundle, "messages": messages,
                  "runtime_sha256": digest(runtime), "input_sha256": digest(bundle["input"])}
        inputs.append(record)
        native.append({"id": slot["id"], "split": "test", "messages": messages,
                       "rendered_prompt": rendered, "prefix_sha256": digest(prefix),
                       "prefix_ids_sha256": hashlib.sha256(json.dumps(prefix, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                       "primary": {"input_ids": prefix, "completion_start": len(prefix)}})
        runtime_rows.append(runtime)
        public.append({"id": slot["id"], "status": "unlabeled_input_frozen",
                       "input_sha256": record["input_sha256"], "input_tokens": len(prefix),
                       "budget_reduced": bundle["context"]["budget_reduced"]})
        previous[slot["episode_id"]] = runtime
    return public, inputs, runtime_rows, native


def execute_reference(case, runtime, prediction, exporter):
    document, bundle = runtime["document"], case["bundle"]
    if runtime["feedback_journal"] is not None:
        feedback = contract.PassiveFeedbackLedger.restore(runtime["feedback_journal"])
        require(prediction["action"] == "no_action" or not feedback.suppressed(
            document, bundle, prediction, runtime["sequence"]), "reference_repeats_suppressed_edit")
    after = contract.apply_passive_edit(document, bundle, prediction)
    operation = verify_operation(document, after, bundle["mapping"], prediction)
    journal = replayable_session(document, after, prediction["action"], case["source"])
    replayed, summary = replay(journal)
    require(replayed == after and summary["complete"], "reference_replay_undo_failed")
    line_length = bundle["input"]["caption_profile"]["export_line_length"]
    exported = verify_export_delta(exporter.export(document, line_length), exporter.export(after, line_length),
                                   operation, prediction["action"])
    return {"prediction": prediction, "after_sha256": session_hash(after),
            "outcome_sha256": digest(transcript_media(after)), "operation": operation,
            "caption_export": exported, "journal": journal, "replay": summary,
            "acoustic_quality_verified": False}


def seal_reference_records(references, census, inputs, runtime, binding, exporter):
    require([r["id"] for r in references] == [c["id"] for c in census], "reference_census_changed")
    indexed, states = ({r["id"]: r for r in values} for values in (inputs, runtime))
    cases, executions, dispositions = [], [], []
    for reference, entry in zip(references, census, strict=True):
        label = reference["label_state"]
        require(label in LABELS, "unknown_reference_disposition")
        audit = reference["audit"]
        require(audit["reviewer_actor"] == binding["reviewer_actor"]
                and audit["author_actor"] == binding["author_actor"]
                and audit["input_first"] is True and audit["predictions_seen"] is False
                and audit["all_visible_opportunities_reviewed"] is True
                and bool(audit["novelty_rationale"]), "independent_reference_audit_missing")
        if entry["status"] != "unlabeled_input_frozen":
            require(label == entry["status"] and not reference.get("accepted_predictions"),
                    "excluded_slot_promoted_to_reference")
        if label not in KNOWN:
            require(not reference.get("accepted_predictions"), "unknown_reference_became_no_action")
            dispositions.append({"id": entry["id"], "status": label,
                                 "reason_code": reference["reason_code"]})
            continue
        case = indexed[entry["id"]]
        require(reference["input_sha256"] == case["input_sha256"], "reference_input_changed")
        predictions = reference["accepted_predictions"]
        require(bool(predictions) and reference["primary"] in predictions
                and len({packed(p) for p in predictions}) == len(predictions), "invalid_reference_alternatives")
        actions = {p["action"] for p in predictions}
        require((actions == {"no_action"}) if label == "known_preserve" else "no_action" not in actions,
                "edit_and_preserve_opportunities_mixed")
        results = [execute_reference(case, states[entry["id"]], prediction, exporter)
                   for prediction in predictions]
        cases.append({**case, "primary": reference["primary"], "accepted_predictions": predictions,
                      "reference_outcomes": [{"prediction": r["prediction"], "available": True,
                                              "outcome_sha256": r["outcome_sha256"]} for r in results],
                      "applicability": {"label_state": label, "full_action_preservation_admitted": label == "known_preserve",
                                        "reference_altered": False},
                      "audit": audit, "rationale": reference["rationale"]})
        executions.append({"id": case["id"], "references": results})
        dispositions.append({"id": case["id"], "status": label})
    return cases, executions, dispositions


def publish(output, phase, protocol_path, binding_path, payloads, census, counts, input_manifest=None):
    output = output.resolve()
    require(output.is_relative_to(ROOT / "artifacts/datasets/nep-passive-actions-final-test-v1"),
            "sealed_output_outside_e1_root")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sealed = output / "sealed"
    sealed.mkdir(mode=0o700)
    for name, values in payloads.items():
        write_rows(sealed / name, values)
    public_census = [{**row, "status": "admitted"} if row["status"] in KNOWN else row for row in census]
    manifest = {"version": VERSION, "status": phase, "protocol_sha256": file_sha(protocol_path),
                "freeze_binding_sha256": file_sha(binding_path), "preparer_sha256": file_sha(__file__),
                "files_sha256": {str(p.relative_to(output)): file_sha(p) for p in sealed.iterdir()},
                "registered_slots": len(census), "counts": counts, "slot_census": public_census,
                "outcome_hash_format": "sha256_canonical_JSON_sorted_keys_UTF8",
                "new_training_updates": 0, "model_requests": 0,
                "scope": "Held-out preparation; public manifest contains no body, rationale or per-case gold action."}
    if input_manifest is not None:
        manifest["input_manifest"] = {"path": str(input_manifest.resolve()), "sha256": file_sha(input_manifest)}
    write(output / "manifest.json", manifest)
    print(json.dumps({"status": phase, "manifest_sha256": file_sha(output / "manifest.json"),
                      "registered_slots": len(census), "counts": counts}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check-freeze", "freeze-inputs", "seal-references"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--input-bundle", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol, binding = load_gate(args.protocol, args.binding)
    if args.stage == "check-freeze":
        print(json.dumps({"status": "freeze_validated_no_test_body_read", "test_bodies_read": 0}))
        return
    if args.stage == "freeze-inputs":
        require(args.candidates is not None and args.output is not None, "missing_input_stage_arguments")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / binding["tokenizer_path"]), local_files_only=True)
        require(tokenizer.eos_token_id == 1, "wrong_native_tokenizer")
        metadata = read(bound(protocol["bindings"]["exposure_metadata"]))
        exposure = exposed_text_hashes(metadata, binding["additional_exposure_collections_sha256"])
        slots = read(bound(protocol["bindings"]["slots"]))["slots"]
        census, inputs, runtime, native = prepare_input_records(
            rows(args.candidates), slots, protocol, binding, tokenizer, exposure)
        publish(args.output, "unlabeled_inputs_frozen_before_reference_authoring", args.protocol, args.binding,
                {"inputs.jsonl": inputs, "runtime-inputs.jsonl": runtime, "native-prefixes.jsonl": native},
                census, dict(Counter(r["status"] for r in census)))
        return
    require(args.input_bundle is not None and args.references is not None and args.output is not None,
            "missing_reference_stage_arguments")
    manifest = read(args.input_bundle / "manifest.json")
    require(manifest["status"] == "unlabeled_inputs_frozen_before_reference_authoring"
            and manifest["protocol_sha256"] == file_sha(args.protocol)
            and manifest["freeze_binding_sha256"] == file_sha(args.binding), "input_bundle_changed")
    for name, expected in manifest["files_sha256"].items():
        require(file_sha(args.input_bundle / name) == expected, "sealed_input_bytes_changed")
    inputs, runtime = (rows(args.input_bundle / "sealed" / name)
                       for name in ("inputs.jsonl", "runtime-inputs.jsonl"))
    exporter = Exporter()
    try:
        cases, executions, dispositions = seal_reference_records(
            rows(args.references), manifest["slot_census"], inputs, runtime, binding, exporter)
    finally:
        exporter.close()
    counts = {"admitted": len(cases), "dispositions": dict(Counter(r["status"] for r in dispositions)),
              "by_action_language": dict(Counter(f'{c["primary"]["action"]}|{c["language"]}' for c in cases)),
              "by_provenance": dict(Counter(c["source"]["kind"] for c in cases))}
    publish(args.output, "references_independently_audited_and_sealed_before_predictions", args.protocol,
            args.binding, {"cases.jsonl": cases, "executions.jsonl": executions,
                           "review-records.jsonl": rows(args.references),
                           "dispositions.jsonl": dispositions}, dispositions, counts,
            input_manifest=args.input_bundle / "manifest.json")


if __name__ == "__main__":
    main()
