"""Build a source-backed, non-trainable draft using the shared NEP execution contract."""

import argparse
import copy
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from context_policy import (
    ContextBuilder,
    ContextBudgetError,
    ContextPolicy,
    FeedbackLedger,
)
from ami_release import ami_sources
from dataset_release_inventory import (
    ROOT,
    WORK_ORDER,
    build_inventory,
    digest,
    read,
    rows,
    write,
)
from grading import grade
from iteration import jsonl
from model_input import messages, token_ids
from replay_session import replay, session_hash
from reviewed_audio import (
    CUTS,
    attach_qualifications,
    load_qualifications,
)
from release_media import verify_media_edit
from release_sources import calculus_sources
from session_cases import apply_edit, replayable_session
from source_qualification import (
    EVIDENCE,
    TOKENIZER,
    focus_at,
    sources_for,
)
from train_adapter import MODEL, REVISION, training_row


def timed(document):
    return [i for i in document["content"] if "sourceStart" in i]


def item_at(document, start):
    found = [i for i in timed(document) if abs(i["sourceStart"] - start) < 1e-6]
    if len(found) != 1:
        raise ValueError(f"Expected one original timed item at {start}.")
    return found[0]


def action_for(spec, document, mapping):
    action = {"action": spec["action"]}
    if spec["action"] == "no_action":
        return action
    first = item_at(document, spec["target_start"])
    selected = [first]
    if "target_end" in spec:
        selected = [
            i
            for i in timed(document)
            if spec["target_start"] - 1e-6
            <= i["sourceStart"]
            < spec["target_end"] - 1e-6
        ]
        if (
            abs(
                selected[-1]["sourceStart"]
                + selected[-1]["length"]
                - spec["target_end"]
            )
            > 1e-6
        ):
            raise ValueError("Reference endpoint does not match an original item.")
    aliases = {v["uuid"]: int(k) for k, v in mapping.items()}
    if not {i["uuid"] for i in selected} <= aliases.keys():
        raise ContextBudgetError(
            "Reference is outside the independently selected cursor context."
        )
    action["target"] = [aliases[i["uuid"]] for i in selected]
    for parameter in ("text", "keep_ms"):
        if parameter in spec:
            action[parameter] = spec[parameter]
    return action


def case_specs(recipe, work):
    old = read(ROOT / recipe["source_recipe"])
    assignments = {
        s["case_id"]: s["related_decision_ids"]
        for s in work["existing_reference_candidates"]
    }
    retained = []
    if recipe["retain_existing_references"]:
        for spec in old["cases"]:
            retained.append(
                {
                    **spec,
                    "scenario_ids": assignments[spec["id"]],
                    "stratum": "open_next_edit"
                    if spec["id"]
                    in {
                        "source-en-distinct-teaching-questions",
                        "source-ko-rhetorical-emphasis",
                    }
                    else "reference_assisted",
                    "label_status": "known_preserve"
                    if spec["action"] == "no_action"
                    else "known_edit",
                    "parent_episode_id": f"{spec['source_id']}-window-{spec['window']}",
                    "retained_from": "artifacts/iterations/source-qualification-v1",
                    "preservation": [
                        "All original source intervals",
                        "Unrelated words and language",
                    ],
                }
            )
    result = [
        {**spec, "recipe_version": recipe["version"]}
        for spec in retained + recipe["cases"]
    ]
    for spec in result:
        spec.update(recipe.get("case_overrides", {}).get(spec["id"], {}))
        replacement = recipe.get("superseded_static_cases", {}).get(spec["id"])
        if replacement:
            spec.update(
                exclude_status="duplicate_episode_prestate",
                exclude_reason=f"Same input/reference is retained once as episode step {replacement}.",
            )
    return result


def source_evidence(source, document):
    return {
        item["uuid"]: source["boundary_evidence"][item["uuid"]]
        for item in document["content"]
        if item["uuid"] in source.get("boundary_evidence", {})
        and source["boundary_evidence"][item["uuid"]]["item_hash"] == session_hash(item)
    }


def restore_bundle(builder, document, focus, brief, settings):
    settings = copy.deepcopy(settings)
    saved = settings.pop("feedback", None)
    if saved:
        ledger = FeedbackLedger(saved["document_id"])
        ledger.proposals, ledger.events = saved["proposals"], saved["events"]
        settings["feedback"] = ledger
    return builder.build(document, focus, brief, **settings)


def comparison_tokenizers(recipe):
    from transformers import AutoTokenizer

    path = recipe.get("comparison_tokenizer_registry")
    if not path:
        return {}, None
    registry = read(ROOT / path)
    tokenizers = {}
    for candidate in registry["candidates"]:
        if candidate["status"] != "downloaded_not_runtime_verified":
            raise ValueError("The requested comparison tokenizer is unavailable.")
        folder = ROOT / candidate["path"]
        for name, expected in candidate["files_sha256"].items():
            if digest(folder / name) != expected:
                raise ValueError("Comparison tokenizer file changed.")
        tokenizers[candidate["id"]] = AutoTokenizer.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False
        )
    return tokenizers, registry


def comparison_contexts(cases, prompt, tokenizers, output_tokens):
    result = {}
    for identity, tokenizer in tokenizers.items():
        records = []
        for case in cases:
            prefix = token_ids(tokenizer, case["input"], prompt)
            completion_count = max(
                len(
                    tokenizer.encode(
                        json.dumps(a, ensure_ascii=False, separators=(",", ":")),
                        add_special_tokens=False,
                    )
                )
                + 1
                for a in [case["expectedOutput"], *case["acceptableOutputs"]]
            )
            if completion_count > output_tokens:
                raise ValueError(
                    "An accepted output exceeds a comparison tokenizer's reserve."
                )
            records.append(
                {
                    "case_id": case["id"],
                    "messages": messages(case["input"], prompt),
                    "input_ids": prefix,
                    "token_count": len(prefix),
                    "max_reference_tokens_including_eos": completion_count,
                    "effective_input_sha256": case["metadata"][
                        "effective_input_sha256"
                    ],
                }
            )
        result[identity] = records
    return result


def draft_case(spec, source, windows, builder, output, prompt, prepared=None):
    document = copy.deepcopy(prepared["document"] if prepared else source["document"])
    corruption = spec.get("corruption")
    if corruption:
        item = item_at(document, corruption["start"])
        if item.get("text") != corruption["original"]:
            raise ValueError("Corruption does not match the recorded source text.")
        item["text"] = corruption["replacement"]
    window = next(
        w
        for w in windows
        if w["source_id"] == source["id"] and w["id"].endswith(f"-{spec['window']:02d}")
    )
    if not window["start_seconds"] <= spec["focus_seconds"] < window["end_seconds"]:
        raise ValueError(f"Cursor outside registered window: {spec['id']}")
    focus = focus_at(timed(document), spec["focus_seconds"])
    settings = copy.deepcopy(prepared["settings"] if prepared else {"as_of": 0})
    settings["evidence"] = source_evidence(source, document)
    settings["protected"] = [
        item_at(document, t)["uuid"] for t in spec.get("protected_starts", [])
    ]
    if spec.get("caption_profile"):
        settings["caption_profile"] = spec["caption_profile"]
    bundle = restore_bundle(builder, document, focus["uuid"], spec["brief"], settings)
    visible_text = " ".join(i["text"] for i in bundle["input"]["items"] if i["text"])
    if any(
        phrase not in visible_text for phrase in spec.get("required_visible_text", [])
    ):
        raise ContextBudgetError(
            "Scenario evidence is absent from the independently selected context."
        )
    expected = action_for(spec, document, bundle["mapping"])
    alternatives = [
        action_for(
            {
                "action": "correct_text",
                "target_start": a["start_seconds"],
                "text": a["text"],
            },
            document,
            bundle["mapping"],
        )
        for a in spec.get("alternative_targets", [])
    ]
    alternatives.extend(
        action_for(a, document, bundle["mapping"])
        for a in spec.get("acceptable_actions", [])
    )
    alternatives = list(
        {session_hash(a): a for a in alternatives if a != expected}.values()
    )
    after = apply_edit(document, bundle, expected)
    qualifications = source.get("audio_qualifications", [])
    cut_proofs = []
    for action in [expected, *alternatives]:
        proof = verify_media_edit(
            document,
            apply_edit(document, bundle, action),
            bundle["mapping"],
            action,
            qualifications,
            source.get("media_kind", "recording"),
        )
        if proof:
            cut_proofs.append(proof["id"])
    source_info = {
        source["id"]: {
            "video_sha256": source["video_sha256"],
            "transcript_sha256": source["transcript_sha256"],
        }
    }
    events = replayable_session(document, after, expected["action"], source_info)
    events[0]["generator"] = spec["recipe_version"]
    replayed, _ = replay(events)
    if replayed != after:
        raise ValueError("Reference replay differs from execution.")
    path = f"sessions/{spec['id']}.jsonl"
    jsonl(output / path, events)
    evidence = []
    for note in spec.get("publisher_annotation_evidence", []):
        annotation = next(
            (a for a in source.get("native_annotations", []) if a["id"] == note["id"]),
            None,
        )
        if annotation is None or annotation["attributes"].get("type") != note["type"]:
            raise ValueError(
                "Publisher event evidence does not match the native annotation."
            )
        evidence.append(
            {
                "origin": "publisher_event_annotation",
                "annotation": annotation,
                "predictor_visibility": "event_observation_explicit_in_task_brief",
                "source_sha256": source["transcript_sha256"],
            }
        )
    for note in spec.get("transcription_evidence", []):
        if not any(
            a["type"] == note["type"]
            and a["text"] == note["text"]
            and abs(a["start"] + source["time_offset"] - note["source_seconds"]) < 1e-6
            for a in source.get("annotations", [])
        ):
            raise ValueError(
                "The stated transcription note is absent from the saved source."
            )
        evidence.append(
            {
                **note,
                "origin": "retained_provider_annotation",
                "predictor_visibility": "explicit_transcription_note_in_task_brief",
                "source_sha256": source["transcript_sha256"],
            }
        )
    for frame in spec.get("evidence", []):
        filename = f"evidence/{frame['file']}"
        dest = output / filename
        if not dest.exists():
            shutil.copyfile(EVIDENCE / frame["file"], dest)
        evidence.append(
            {
                **frame,
                "file": filename,
                "sha256": digest(dest),
                "predictor_visibility": "observation_explicitly_supplied_in_brief",
            }
        )
    for artifact in spec.get("artifact_evidence", []):
        artifact_path = ROOT / artifact["path"]
        if digest(artifact_path) != artifact["sha256"]:
            raise ValueError("Semantic evidence artifact changed.")
        if artifact["input_observation"] not in spec["brief"]:
            raise ValueError("Visual observation is hidden from the predictor.")
        evidence.append(
            {**artifact, "predictor_visibility": "factual_observation_in_task_brief"}
        )
    source_items = timed(source["document"])
    visible = bundle["input"]["items"]
    unit_counts = {
        "source_timed_items": len(source_items),
        "visible_timed_items": len(visible),
        "source_whitespace_units": sum(
            len(i.get("text", "").split()) for i in source_items
        ),
        "visible_whitespace_units": sum(len(i["text"].split()) for i in visible),
        "visible_text_items": sum(i["type"] == "text" for i in visible),
        "visible_sentence_end_markers": len(
            re.findall(r"[.!?。！？](?:\s|$)", " ".join(i["text"] for i in visible))
        ),
        "sentence_count_method": "Punctuation heuristic, not linguistic sentence segmentation; decimals and fragments are not word timing.",
        "visible_paragraphs": len({i["paragraph"] for i in visible}),
        "complete_paragraphs": len(bundle["input"]["complete_paragraphs"]),
        "input_token_count": bundle["context"]["input_tokens"],
        "itemization": source["itemization"],
    }
    case = {
        "id": spec["id"],
        "input": bundle["input"],
        "expectedOutput": expected,
        "acceptableOutputs": alternatives,
        "metadata": {
            "split": source["reserved_split"],
            "source_group_id": source["source_group_id"],
            "parent_source_id": source["id"],
            "language": source["language"],
            "parent_episode_id": spec["parent_episode_id"],
            "scenario_ids": spec["scenario_ids"],
            "scenario_family": spec["scenario_ids"][0],
            "scenario_variant": spec["id"],
            "decision_factor": spec["decision_factor"],
            "adds_coverage": spec["adds_coverage"],
            "label_status": spec["label_status"],
            "stratum": spec["stratum"],
            "preservation_obligations": spec["preservation"],
            "actor": "synthetic",
            "target_origin": "ai_generated_recipe",
            "label_origin": "agent_authored_from_source_evidence",
            "content_origin": source["transcript_origin"],
            "corruption_origin": "constructed_asr_like_damage"
            if corruption
            else "executed_source_edit_state"
            if document != source["document"]
            else "unchanged_source_transcript",
            "corruption": corruption,
            "timing_origin": source["itemization"],
            "media_kind": source.get("media_kind", "recording"),
            "source_identity_sha256": source.get(
                "source_identity_sha256", source["video_sha256"]
            ),
            "boundary_status": "reviewed_operation"
            if cut_proofs
            else "per_item_evidence",
            "audio_qualification_ids": cut_proofs,
            "semantic_contrast_id": spec.get("semantic_contrast_id"),
            "difficulty_mechanism": spec.get("difficulty_mechanism"),
            "history_origin": "executed_synthetic_episode" if prepared else "none",
            "episode_step": spec.get("episode_step"),
            "feedback_ids": spec.get("feedback_ids", []),
            "episode_protocol": spec.get("episode_protocol"),
            "training_eligible": False,
            "candidate_training_split": source["reserved_split"] == "train",
            "release_status": "draft_not_frozen",
            "requires_regeneration": False,
            "review_status": "pending_input_only_semantic_audit",
            "retained_from": spec.get("retained_from"),
            "generator": "ml/dataset_release.py",
            "recipe_version": spec["recipe_version"],
            "effective_input_sha256": session_hash(messages(bundle["input"], prompt)),
            "context": bundle["context"],
            "unit_counts": unit_counts,
            "focus_uuid": focus["uuid"],
            "source_window": window["id"],
            "source_context_seconds": [
                min(
                    i["sourceStart"]
                    for i in [
                        document["content"][v["index"]]
                        for v in bundle["mapping"].values()
                    ]
                ),
                max(
                    i["sourceStart"] + i["length"]
                    for i in [
                        document["content"][v["index"]]
                        for v in bundle["mapping"].values()
                    ]
                ),
            ],
            "source_video": str(source["video"].resolve().relative_to(ROOT))
            if source.get("video")
            else None,
            "source_document": str(source["source_path"].relative_to(ROOT))
            if source.get("source_path")
            else None,
            "video_sha256": source["video_sha256"],
            "transcript_sha256": source["transcript_sha256"],
            "source_page": source.get("source_page"),
            "license": source["license"],
            "project_use_authorization": "data/manifests/project-source-use-2026-09-25.json",
            "evidence": evidence,
            "session": path,
        },
    }
    prefix = token_ids(builder.tokenizer, bundle["input"], prompt)
    sequence = training_row(builder.tokenizer, case, prompt)
    if (
        sequence["input_ids"][: len(prefix)] != prefix
        or len(sequence["input_ids"]) - len(prefix) > builder.policy.output_tokens
    ):
        raise ValueError(
            "Reference exceeds output reserve or changes inference prefix."
        )
    context = {
        "case_id": spec["id"],
        "messages": messages(bundle["input"], prompt),
        "input_ids": prefix,
        "token_count": len(prefix),
        "completion_token_count": len(sequence["input_ids"]) - len(prefix),
        "training_sequence_diagnostic_only": sequence,
        "tokenizer_id": MODEL,
        "tokenizer_revision": REVISION,
    }
    execution = {
        "id": spec["id"],
        "before": document,
        "mapping": bundle["mapping"],
        "after_hash": session_hash(after),
        "reconstruction": settings,
        "audio_qualifications": qualifications,
        "media_kind": source.get("media_kind", "recording"),
        **(
            {"caption_authority": bundle["caption_authority"]}
            if "caption_authority" in bundle
            else {}
        ),
    }
    candidate_audit = []
    for candidate in spec.get("counterfactuals", []):
        prediction = action_for(candidate, document, bundle["mapping"])
        alternative = apply_edit(document, bundle, prediction)
        if grade(case, prediction, execution)["outcome_match"]:
            raise ValueError("A semantic counterfactual is accepted as a reference.")
        proof = verify_media_edit(
            document,
            alternative,
            bundle["mapping"],
            prediction,
            qualifications,
            source.get("media_kind", "recording"),
        )
        candidate_audit.append(
            {
                "prediction": prediction,
                "contract_valid": True,
                "reference_match": False,
                "reason": candidate["reason"],
                "audio_qualification_id": proof["id"] if proof else None,
            }
        )
    case["metadata"]["counterfactual_audit"] = candidate_audit
    rejected = []
    for candidate in spec.get("ineligible_actions", []):
        prediction = None
        try:
            prediction = action_for(candidate, document, bundle["mapping"])
            apply_edit(document, bundle, prediction)
        except ValueError as error:
            if candidate["expected_error"] not in str(error):
                raise ValueError(
                    "Eligibility probe failed for an unexpected reason."
                ) from error
            rejected.append(
                {
                    "specification": candidate,
                    "prediction": prediction,
                    "error": str(error),
                    "label": None,
                }
            )
        else:
            raise ValueError("An ineligible candidate unexpectedly executed.")
    case["metadata"]["eligibility_exclusions"] = rejected
    if not all(
        grade(case, action, execution)["outcome_match"]
        for action in [expected, *alternatives]
    ):
        raise ValueError("An acceptable reference fails the grader.")
    return case, context, execution


def audit_cases(cases, reserved, frozen=False):
    ids, pairs, inputs, groups = set(), set(), {}, {}
    for case in cases:
        meta, inp = case["metadata"], case["input"]
        group, split = meta["source_group_id"], meta["split"]
        if group in reserved or split not in {"train", "dev"}:
            raise ValueError("Reserved-test or unassigned material in release.")
        if groups.setdefault(group, split) != split:
            raise ValueError("Source group crosses splits.")
        if case["id"] in ids:
            raise ValueError("Duplicate case ID.")
        ids.add(case["id"])
        pair = session_hash([inp, case["expectedOutput"]])
        key = session_hash(inp)
        if pair in pairs:
            raise ValueError("Duplicate input/reference pair.")
        if key in inputs:
            raise ValueError(
                "Same effective input has conflicting references or splits."
            )
        inputs[key] = (split, case["expectedOutput"])
        pairs.add(pair)
        if meta["label_status"] not in {"known_edit", "known_preserve"} or meta[
            "training_eligible"
        ] is not (frozen and split == "train"):
            raise ValueError(
                "Unknown labels or trainable rows are forbidden in drafts."
            )
        if (meta["label_status"] == "known_preserve") != (
            case["expectedOutput"]["action"] == "no_action"
        ):
            raise ValueError("Reference contradicts its known edit/preserve decision.")
    spans = []
    for index, left in enumerate(cases):
        lm = left["metadata"]
        for right in cases[index + 1 :]:
            rm = right["metadata"]
            if lm.get("source_identity_sha256", lm["video_sha256"]) != rm.get(
                "source_identity_sha256", rm["video_sha256"]
            ):
                continue
            if (
                lm["source_group_id"] != rm["source_group_id"]
                or lm["split"] != rm["split"]
            ):
                raise ValueError(
                    "Identical media was assigned to different lineage groups."
                )
            a, b = lm["source_context_seconds"], rm["source_context_seconds"]
            if max(a[0], b[0]) < min(a[1], b[1]):
                spans.append([left["id"], right["id"]])
    return {
        "unique_pairs": len(pairs),
        "same_source_overlapping_context_pairs": spans,
        "overlap_interpretation": "Expected within-group contexts and deliberate contrasts; not independent recordings.",
        "split_conflicts": 0,
        "duplicate_inputs": 0,
    }


def release_builder(recipe, tokenizer, prompt, candidates):
    if recipe.get("schema_version", 3) == 3:
        return ContextBuilder(tokenizer, prompt, budget_tokenizers=candidates)
    from format_comparison import BUDGETS

    return ContextBuilder(
        tokenizer,
        prompt,
        ContextPolicy(
            schema_version=4,
            version="cursor-caption-v2",
            input_tokens=BUDGETS["common_input_tokens"],
            output_tokens=max(BUDGETS["output_tokens"].values()),
            context_tokens=BUDGETS["context_tokens"],
        ),
        budget_tokenizers=candidates,
        budget_prompts={"rewrite": (ROOT / "ml/prompt-rewrite-v1.txt").read_text()},
    )


def build(recipe_path, tokenizer, output):
    recipe = read(recipe_path)
    inventory_summary = build_inventory(ROOT, output, recipe_path)
    work = read(ROOT / WORK_ORDER)
    shutil.copyfile(ROOT / WORK_ORDER, output / "work-order-at-start.json")
    shutil.copyfile(recipe_path, output / "recipe.json")
    for name in ("sessions", "evidence", "contexts"):
        (output / name).mkdir()
    prompt = (ROOT / f"ml/prompt-v{recipe.get('schema_version', 3)}.txt").read_text()
    (output / "prompt.txt").write_text(prompt)
    sources, windows = sources_for(read(ROOT / recipe["source_recipe"]))
    if recipe.get("include_calculus"):
        additional_sources, additional_windows = calculus_sources()
        sources.update(additional_sources)
        windows.extend(additional_windows)
    if recipe.get("include_ami"):
        additional_sources, additional_windows = ami_sources()
        sources.update(additional_sources)
        windows.extend(additional_windows)
        ami = additional_sources["ami-es2002a"]
        write(
            output / "evidence/ami-import.json",
            {
                "source_group_id": ami["source_group_id"],
                "native_annotations": ami["native_annotations"],
                "item_annotations": ami["item_annotations"],
                "boundary_evidence": ami["boundary_evidence"],
                "representation": "Each mixed-audio interval appears once; simultaneous speakers share one tagged text island. Zero-duration annotations are retained, never assigned invented durations.",
            },
        )
    if recipe.get("include_ami_dev"):
        from ami_dev_release import ami_dev_sources

        additional_sources, additional_windows = ami_dev_sources()
        sources.update(additional_sources)
        windows.extend(additional_windows)
        source = additional_sources["ami-is1008b"]
        write(
            output / "evidence/ami-dev-import.json",
            {
                "source_group_id": source["source_group_id"],
                "native_annotations": source["native_annotations"],
                "item_annotations": source["item_annotations"],
                "boundary_evidence": source["boundary_evidence"],
            },
        )
    if recipe.get("scripted_sources"):
        from scripted_release import scripted_sources

        additional_sources, additional_windows = scripted_sources(
            recipe["scripted_sources"]
        )
        sources.update(additional_sources)
        windows.extend(additional_windows)
    candidates, registry = comparison_tokenizers(recipe)
    qualifications = []
    if recipe.get("reviewed_audio"):
        qualifications = load_qualifications(ROOT / recipe["reviewed_audio"])
        attach_qualifications(sources, qualifications)
        write(output / "evidence/reviewed-audio.json", qualifications)
    builder = release_builder(recipe, tokenizer, prompt, candidates)
    cases, contexts, executions, exclusions = [], [], [], []
    for spec in case_specs(recipe, work):
        if spec.get("exclude_reason"):
            exclusions.append(
                {
                    "id": spec["id"],
                    "scenario_ids": spec["scenario_ids"],
                    "status": spec["exclude_status"],
                    "reason": spec["exclude_reason"],
                    "expectedOutput": None,
                }
            )
            continue
        try:
            case, context, execution = draft_case(
                spec, sources[spec["source_id"]], windows, builder, output, prompt
            )
        except ContextBudgetError as error:
            exclusions.append(
                {
                    "id": spec["id"],
                    "scenario_ids": spec["scenario_ids"],
                    "status": "out_of_window",
                    "reason": str(error),
                    "expectedOutput": None,
                }
            )
            continue
        cases.append(case)
        contexts.append(context)
        executions.append(execution)
    from release_episodes import build_episodes

    episode_summary = build_episodes(
        recipe,
        sources,
        windows,
        builder,
        output,
        prompt,
        draft_case,
        action_for,
    )
    cases.extend(episode_summary.pop("cases"))
    contexts.extend(episode_summary.pop("contexts"))
    executions.extend(episode_summary.pop("executions"))
    write(output / "episode-summary.json", episode_summary)
    audit = audit_cases(cases, work["reserved_test_groups"])
    natural_cuts = sum(
        c["expectedOutput"]["action"] in CUTS
        and c["metadata"]["media_kind"] == "recording"
        for c in cases
    )
    symbolic_cuts = sum(
        c["expectedOutput"]["action"] in CUTS
        and c["metadata"]["media_kind"] == "authored_symbolic_timeline"
        for c in cases
    )
    for split in ("train", "dev"):
        jsonl(
            output / f"{split}-cases.jsonl",
            [c for c in cases if c["metadata"]["split"] == split],
        )
    jsonl(output / "contexts/qwen3.5-2b.jsonl", contexts)
    for identity, records in comparison_contexts(
        cases, prompt, candidates, builder.policy.output_tokens
    ).items():
        jsonl(output / f"contexts/{identity}.jsonl", records)
    if registry:
        write(output / "comparison-tokenizers.json", registry)
    if recipe.get("schema_version", 3) == 4:
        from format_comparison import build_format_comparison

        rewrite_prompt = (ROOT / "ml/prompt-rewrite-v1.txt").read_text()
        (output / "prompt-rewrite.txt").write_text(rewrite_prompt)
        comparison = build_format_comparison(cases, candidates, prompt, rewrite_prompt)
        write(output / "format-comparison.json", comparison["spec"])
        write(output / "format-targets.json", comparison["targets"])
        for format_name, models in comparison["contexts"].items():
            for identity, records in models.items():
                jsonl(output / f"contexts/{format_name}-{identity}.jsonl", records)
    jsonl(output / "executions.jsonl", executions)
    jsonl(output / "exclusions.jsonl", exclusions)
    from release_audit import audit_release

    write(output / "diversity-audit.json", audit_release(cases, contexts, executions))
    subprocess.run(
        ["node", "ml/verify_source_caption_export.cjs", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    coverage = []
    for decision in work["content_decisions"]:
        selected = [c for c in cases if decision["id"] in c["metadata"]["scenario_ids"]]
        coverage.append(
            {
                "id": decision["id"],
                "decision": decision["decision"],
                "status": "draft_candidates_validated_execution" if selected else "gap",
                "draft_case_ids": [c["id"] for c in selected],
                "admitted_case_ids": [],
                "language_outcomes": dict(
                    Counter(
                        f"{c['metadata']['language']}:{c['metadata']['label_status']}"
                        for c in selected
                    )
                ),
                "remaining": recipe["gaps"].get(
                    decision["id"],
                    "Input-only semantic audit and broader independent coverage remain before freeze.",
                ),
            }
        )
    write(
        output / "scenario-coverage.json",
        {
            "decisions": coverage,
            "covered_specifications_partial": sum(
                bool(r["draft_case_ids"]) for r in coverage
            ),
            "qualified_natural_audio_cuts": natural_cuts,
            "unique_reviewed_operations": len(
                {q for c in cases for q in c["metadata"]["audio_qualification_ids"]}
            ),
            "symbolic_cut_rows": symbolic_cuts,
            "coverage_complete": False,
        },
    )
    write(
        output / "audit.json",
        {
            **audit,
            "execution_replay_undo": "passed",
            "caption_export": "passed_actual_app_exporter",
            "semantic_audit": "pending",
            "cross_tokenizer_evidence_parity": "passed_shared_snapshot_for_registered_five_candidates"
            if registry
            else "pending",
            "natural_audio_qualification": "exact_reviewed_operations_only"
            if natural_cuts
            else "not_claimed",
            "inventory": inventory_summary,
            "test_bodies_opened": False,
        },
    )
    build_inputs = [
        recipe_path,
        ROOT / recipe["source_recipe"],
        *[
            ROOT / "ml" / name
            for name in (
                "dataset_release.py",
                "dataset_release_inventory.py",
                "source_qualification.py",
                "session_cases.py",
                "context_policy.py",
                "model_input.py",
                "nep.py",
                "nep_v2.py",
                "nep_v3.py",
                "prompt-v3.txt",
                "train_adapter.py",
                "grading.py",
                "replay_session.py",
                "release_sources.py",
                "ami_release.py",
                "release_episodes.py",
                "release_audit.py",
                "reviewed_audio.py",
                "release_media.py",
                "verify_source_caption_export.cjs",
            )
        ],
        *[p for p in TOKENIZER.iterdir() if p.is_file()],
    ]
    if recipe.get("schema_version", 3) == 4:
        build_inputs.extend(
            ROOT / "ml" / name
            for name in (
                "nep_v4.py",
                "caption_facts.py",
                "prompt-v4.txt",
                "format_comparison.py",
                "prompt-rewrite-v1.txt",
                "input-v4.schema.json",
                "rewrite-v1.schema.json",
                "verify_caption_metrics.cjs",
            )
        )
    build_inputs.extend(
        ROOT / name
        for name in (
            "app/src/core/webvtt.ts",
            "app/src/state/editor/selectors.ts",
            "app/package.json",
        )
    )
    if recipe.get("include_ami_dev"):
        build_inputs.append(ROOT / "ml/ami_dev_release.py")
    if recipe.get("scripted_sources"):
        build_inputs.append(ROOT / "ml/scripted_release.py")
    if recipe.get("batch_design"):
        build_inputs.append(ROOT / recipe["batch_design"])
    if recipe.get("regeneration_feedback"):
        build_inputs.append(ROOT / recipe["regeneration_feedback"])
    if recipe.get("reviewed_audio"):
        build_inputs.append(ROOT / recipe["reviewed_audio"])
    build_inputs.extend(ROOT / p for p in recipe.get("additional_evidence", []))
    if registry:
        build_inputs.append(ROOT / recipe["comparison_tokenizer_registry"])
        build_inputs.extend(
            ROOT / c["path"] / name
            for c in registry["candidates"]
            for name in c["files_sha256"]
        )
    manifest = {
        "release_id": "nep-release-v1",
        "draft_id": output.name,
        "status": "draft_not_frozen_not_trainable",
        "cases": len(cases),
        "splits": dict(Counter(c["metadata"]["split"] for c in cases)),
        "actions": dict(Counter(c["expectedOutput"]["action"] for c in cases)),
        "languages": dict(Counter(c["metadata"]["language"] for c in cases)),
        "strata": dict(Counter(c["metadata"]["stratum"] for c in cases)),
        "content_provenance": dict(
            Counter(c["metadata"]["corruption_origin"] for c in cases)
        ),
        "retained_references": sum(bool(c["metadata"]["retained_from"]) for c in cases),
        "new_references": sum(not c["metadata"]["retained_from"] for c in cases),
        "exclusions": len(exclusions),
        "episodes": episode_summary,
        "natural_audio_cuts": natural_cuts,
        "unique_reviewed_operations": len(
            {q for c in cases for q in c["metadata"]["audio_qualification_ids"]}
        ),
        "eligible_semantic_counterfactuals": sum(
            len(c["metadata"]["counterfactual_audit"]) for c in cases
        ),
        "symbolic_cuts": symbolic_cuts,
        "schema_version": recipe.get("schema_version", 3),
        "context_policy": builder.policy.__dict__,
        "source_episode_rows": dict(
            Counter(c["metadata"]["parent_episode_id"] for c in cases)
        ),
        "source_loss_tokens": {
            group: sum(
                ctx["completion_token_count"]
                for c, ctx in zip(cases, contexts)
                if c["metadata"]["source_group_id"] == group
            )
            for group in sorted({c["metadata"]["source_group_id"] for c in cases})
        },
        "tokenizer": {
            "model": MODEL,
            "revision": REVISION,
            "path": str(TOKENIZER.relative_to(ROOT)),
            "comparison_scope": "five_candidate_common_evidence_draft"
            if registry
            else "single_cached_tokenizer_diagnostic_not_multi_model_freeze",
        },
        "comparison_tokenizer_ids": list(candidates),
        "remaining_gates": [
            "Input-only semantic audit",
            "Natural-audio evidence and missing scenario decisions",
            "D1 long-paragraph policy",
            "Remaining episode semantics and feedback decisions, including conflicting pacing history",
            "JSON/Rewrite format reserve comparison"
            if registry
            else "Common evidence across all candidate tokenizers",
            "Final test protocol allocation and freeze",
        ],
        "code_and_recipe_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in build_inputs
        },
        "files_sha256": {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write(output / "manifest.json", manifest)
    return {
        k: v
        for k, v in manifest.items()
        if k
        not in {
            "files_sha256",
            "code_and_recipe_sha256",
            "source_loss_tokens",
            "source_episode_rows",
        }
    }


def verify(directory, tokenizer):
    manifest = read(directory / "manifest.json")
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Draft artifact changed: {name}")
    for name, expected in manifest["code_and_recipe_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Generator/tokenizer changed: {name}")
    inventory = read(directory / "source-inventory.json")
    for name, record in inventory["input_files"].items():
        path = (
            directory / "work-order-at-start.json"
            if name == WORK_ORDER
            else ROOT / name
        )
        if digest(path) != record["sha256"]:
            raise ValueError(f"Source/fixture changed: {name}")
    cases = rows(directory / "train-cases.jsonl") + rows(directory / "dev-cases.jsonl")
    executions = {r["id"]: r for r in rows(directory / "executions.jsonl")}
    contexts = {r["case_id"]: r for r in rows(directory / "contexts/qwen3.5-2b.jsonl")}
    recipe = read(directory / "recipe.json")
    candidates, _ = comparison_tokenizers(recipe)
    if recipe.get("reviewed_audio"):
        qualifications = load_qualifications(ROOT / recipe["reviewed_audio"])
        if qualifications != read(directory / "evidence/reviewed-audio.json"):
            raise ValueError("Reviewed-operation provenance changed.")
    builder = release_builder(
        recipe, tokenizer, (directory / "prompt.txt").read_text(), candidates
    )
    for case in cases:
        execution, context = executions[case["id"]], contexts[case["id"]]
        bundle = restore_bundle(
            builder,
            execution["before"],
            case["metadata"]["focus_uuid"],
            case["input"]["task_brief"],
            execution.get("reconstruction", {"as_of": 0}),
        )
        if (
            bundle["input"] != case["input"]
            or bundle["mapping"] != execution["mapping"]
        ):
            raise ValueError("Runtime reconstruction differs from stored input.")
        if (
            token_ids(tokenizer, case["input"], builder.prompt) != context["input_ids"]
            or messages(case["input"], builder.prompt) != context["messages"]
        ):
            raise ValueError("Saved tokenizer input differs from inference.")
        if (
            training_row(tokenizer, case, builder.prompt)
            != context["training_sequence_diagnostic_only"]
        ):
            raise ValueError("Completion mask or training prefix mismatch.")
        after = apply_edit(execution["before"], bundle, case["expectedOutput"])
        replayed, _ = replay(rows(directory / case["metadata"]["session"]))
        if after != replayed or session_hash(after) != execution["after_hash"]:
            raise ValueError("Execution/replay/undo hash mismatch.")
        for action in [case["expectedOutput"], *case["acceptableOutputs"]]:
            if not grade(case, action, execution)["outcome_match"]:
                raise ValueError("Reference fails grading.")
            verify_media_edit(
                execution["before"],
                apply_edit(execution["before"], bundle, action),
                bundle["mapping"],
                action,
                execution.get("audio_qualifications", []),
                execution.get("media_kind", "recording"),
            )
        for candidate in case["metadata"].get("counterfactual_audit", []):
            action = candidate["prediction"]
            alternate = apply_edit(execution["before"], bundle, action)
            verify_media_edit(
                execution["before"],
                alternate,
                bundle["mapping"],
                action,
                execution.get("audio_qualifications", []),
                execution.get("media_kind", "recording"),
            )
            if grade(case, action, execution)["outcome_match"]:
                raise ValueError(
                    "Semantic counterfactual unexpectedly matches the reference."
                )
        for candidate in case["metadata"].get("eligibility_exclusions", []):
            try:
                prediction = action_for(
                    candidate["specification"], execution["before"], bundle["mapping"]
                )
                apply_edit(execution["before"], bundle, prediction)
            except ValueError as error:
                if str(error) != candidate["error"]:
                    raise ValueError(
                        "Eligibility exclusion changed on reconstruction."
                    ) from error
            else:
                raise ValueError("Excluded candidate became eligible.")
    result = audit_cases(
        cases,
        read(directory / "split-manifest.json")["reserved_test_groups"],
        frozen=manifest["status"] == "frozen_supported_scope",
    )
    for identity, expected in comparison_contexts(
        cases, builder.prompt, candidates, builder.policy.output_tokens
    ).items():
        actual = {
            c["case_id"]: c for c in rows(directory / f"contexts/{identity}.jsonl")
        }
        if actual != {c["case_id"]: c for c in expected}:
            raise ValueError(
                "Comparison tokenizer messages/IDs differ on reconstruction."
            )
    if recipe.get("schema_version", 3) == 4:
        from format_comparison import build_format_comparison

        comparison = build_format_comparison(
            cases,
            candidates,
            builder.prompt,
            (directory / "prompt-rewrite.txt").read_text(),
        )
        if comparison["spec"] != read(directory / "format-comparison.json"):
            raise ValueError("Format budgets or roundtrip measurements changed.")
        targets = read(directory / "format-targets.json")
        if sorted(comparison["targets"], key=lambda r: r["case_id"]) != sorted(
            targets, key=lambda r: r["case_id"]
        ):
            raise ValueError("Format targets differ on reconstruction.")
        for format_name, models in comparison["contexts"].items():
            for identity, records in models.items():
                actual = rows(directory / f"contexts/{format_name}-{identity}.jsonl")
                if sorted(actual, key=lambda r: r["case_id"]) != sorted(
                    records, key=lambda r: r["case_id"]
                ):
                    raise ValueError(
                        "Format context messages, token IDs or masks changed."
                    )
    from release_audit import audit_release

    if audit_release(cases, list(contexts.values()), list(executions.values())) != read(
        directory / "diversity-audit.json"
    ):
        raise ValueError(
            "Diversity, context or loss-weight audit differs on reconstruction."
        )
    for episode in read(directory / "episode-summary.json")["episodes"]:
        journal = rows(directory / episode["session"])
        replay(journal)
        feedback = read(directory / episode["feedback"])
        ledger = FeedbackLedger(feedback["document_id"])
        ledger.proposals = feedback["proposals"]
        for event in feedback["events"]:
            saved = dict(event)
            if saved["outcome"] == "undone":
                saved["applied"] = None
            ledger.record(**saved)
        for step in feedback["steps"]:
            if step["recorded_case"]:
                if (
                    session_hash(executions[step["id"]]["before"])
                    != step["before_hash"]
                ):
                    raise ValueError(
                        "Episode state does not match the prediction pre-state."
                    )
    result.update(
        status="passed_deterministic_checks_only",
        cases=len(cases),
        trainable=manifest["status"] == "frozen_supported_scope",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "ml/fixtures/dataset-release-v1.json"
    )
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    result = (
        build(args.recipe.resolve(), tokenizer, args.directory.resolve())
        if args.command == "build"
        else verify(args.directory.resolve(), tokenizer)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
