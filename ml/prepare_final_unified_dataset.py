"""Prepare split-locked supplements and admit independently reviewed v3 rows.

No model weights, predictions, or reserved test bodies are read by this builder.
Candidate requests, semantic review decisions, and admitted examples stay separate.
"""

import argparse
import copy
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import prepare_passive_actions_v3 as data
from passive_checkpoint_v2 import structural_items

ROOT = data.ROOT
BASE = ROOT / "artifacts/datasets/nep-passive-actions-v3/release-attempt-003"
PLAN = ROOT / "artifacts/audits/third-review-response-20261024-001/data-refinement-plan.json"
DEFAULT = ROOT / "artifacts/datasets/nep-passive-actions-final-v1"
VERSION = "passive-actions-final-unified-1"
MECHANISMS = (
    "evidence_in_last_three", "evidence_only_in_older_five",
    "unrelated_role_history", "latest_correction_supersedes_older",
)


def read(path):
    return json.loads(Path(path).read_text())


def iter_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": data.sha256(path)}


def checked_file(binding):
    path = ROOT / binding["path"]
    data.require(path.is_relative_to(ROOT), "Source escaped repository.")
    data.require(data.sha256(path) == binding["sha256"], "Source bytes changed: " + str(path))
    return path


def runtime(focus):
    return {"focus_uuid": focus, "selection": [], "sequence": 0, "history": [],
            "feedback_journal": None, "caption_profile": data.PROFILE}


def register(output):
    data.require(not output.exists(), "Use a new candidate directory.")
    plan = read(PLAN)
    themes = plan["source_pools"]["authored_theme_pool"]
    splits = read(BASE / "source-splits.json")
    sources, slots = {}, []
    for source in plan["source_pools"]["ready_ami"]:
        data.require(source["source_group"] not in data.RESERVED, "Reserved source.")
        splits.setdefault(source["source_group"], source["split"])
        data.require(splits[source["source_group"]] == source["split"], "AMI split changed.")
        document = ROOT / source["document"]["path"]
        data.require(data.session_hash(read(document)) == source["document"]["sha256"],
                     "AMI typed-json-f64-v1 document identity changed.")
        windows = read(checked_file(source["candidates"]))["windows"]
        data.require(len(windows) == source["candidate_count"], "AMI window count changed.")
        item = {**source, "language": "en", "document": bind(document),
                "document_typed_hash": source["document"]["sha256"],
                "provenance": "ami_original_word_annotations", "natural_audio": True,
                "generated_audio": False, "acoustically_verified": False}
        sources[item["id"]] = item
        for window in windows:
            slots.append({"id": "b004-raw-" + window["id"], "track": "A_raw",
                          "source_id": source["id"], "source_group": source["source_group"],
                          "split": source["split"], "language": "en", "focus_uuid": window["focus_uuid"],
                          "original_window_id": window["id"], "moment": window["start_seconds"]})
        for third in range(3):
            pool = windows[len(windows) * third // 3:len(windows) * (third + 1) // 3]
            chosen = min(pool, key=lambda w: data.digest([
                plan["candidate_identity_and_group_assignment"]["terminal_slot_seed"], source["id"], third, w["id"]]))
            slots.append({"id": "b004-raw-" + chosen["id"] + "-terminal", "track": "A_terminal",
                          "parent_id": "b004-raw-" + chosen["id"], "source_id": source["id"],
                          "source_group": source["source_group"], "split": source["split"],
                          "language": "en", "third": third, "max_prior_operations": 3})
    registry = read(ROOT / "data/manifests/passive-existing-sources-v1.json")
    for source in registry["sources"]:
        group = source["source_group_id"]
        data.require(group not in data.RESERVED and splits[group] == source["split"], "Existing source split changed.")
        document = checked_file({"path": source["document_path"], "sha256": source["document_sha256"]})
        sources[source["id"]] = {**source, "source_group": group, "document": bind(document),
                                  "provenance": source["transcript_origin"], "natural_audio": True,
                                  "generated_audio": False, "acoustically_verified": False}
        structure = structural_items(read(document))
        for third in range(3):
            hashed = data.digest([plan["candidate_identity_and_group_assignment"]["new_existing_source_cursor_seed"], source["id"], third])
            draw = (int(hashed[:16], 16) >> 11) / 2**53
            moment = structure[-1]["timeline_end"] * (third + draw) / 3
            focus = next(item for item in structure if item["timeline_end"] > moment)
            identity = f"b004-raw-{source['id']}-third-{third}"
            slots.append({"id": identity, "track": "A_raw", "source_id": source["id"],
                          "source_group": group, "split": source["split"], "language": source["language"],
                          "third": third, "focus_uuid": focus["uuid"], "moment": moment, "hash_draw": hashed})
            slots.append({"id": identity + "-terminal", "track": "A_terminal", "parent_id": identity,
                          "source_id": source["id"], "source_group": group, "split": source["split"],
                          "language": source["language"], "third": third, "max_prior_operations": 3})
    for parent in plan["source_pools"]["natural_en_preserve_train_parents"]:
        for variant in ("fresh_history", "unknown_speaker", "unknown_speaker_typo"):
            slots.append({"id": "b004-bridge-" + parent["id"] + "-" + variant, "track": "B_preservation_bridge",
                          "parent_id": parent["id"], "split": "train", "language": "en",
                          "source_group": parent["source_group"], "variant": variant})
    for track, prefix in (("C_contract", "contract"), ("R_rare_necessity", "rare")):
        definition = next(item for item in plan["candidate_composition"] if item["id"] == track)
        cells = ([(condition["id"], None) for condition in definition["conditions"]] if track == "C_contract"
                 else [(family, action) for action, families in definition["families"].items() for family in families])
        for language in ("en", "ko"):
            for split, count in (("train", 2), ("dev", 1)):
                pool = sorted(item["id"] for item in themes if item["language"] == language and item["split"] == split)
                for j, (family, action) in enumerate(cells):
                    for realization in range(1, count + 1):
                        group = pool[(2 * j + realization - 1) % len(pool)] if split == "train" else pool[j % len(pool)]
                        for necessity in (("useful", "retain") if action else (None,)):
                            suffix = f"{action}-{family}-{necessity}" if action else family
                            slots.append({"id": f"b004-{prefix}-{language}-{suffix}-{split}-{realization}",
                                          "track": track, "source_group": group, "split": split, "language": language,
                                          "family": family, "action": action, "necessity": necessity,
                                          "realization": realization})
    for language in ("en", "ko"):
        for split, count in (("train", 2), ("dev", 1)):
            for mechanism in MECHANISMS:
                for realization in range(1, count + 1):
                    episode = f"b004-history-{language}-{mechanism}-{split}-{realization}"
                    group = f"authored-history-length-b004-{split}-{language}-{mechanism}-{realization}"
                    data.require(group not in splits, "History group already exposed.")
                    splits[group] = split
                    for length in (0, 3, 8):
                        slots.append({"id": episode + f"-h{length}", "track": "H_history", "episode_id": episode,
                                      "source_group": group, "split": split, "language": language,
                                      "mechanism": mechanism, "realization": realization, "history_length": length})
    data.require(Counter(slot["split"] for slot in slots) == {"train": 433, "dev": 161}, "Candidate arithmetic changed.")
    data.require(len({slot["id"] for slot in slots}) == 594, "Candidate IDs collide.")
    data.write(output / "registration.json", {"version": VERSION, "status": "registered_before_inputs_and_references",
               "plan": bind(PLAN), "base_manifest": bind(BASE / "manifest.json"),
               "base_train": bind(BASE / "cases/train.jsonl"), "base_dev": bind(BASE / "cases/dev.jsonl"),
               "sources": sources, "source_splits": splits, "slots": slots,
               "source_groups_are_exposed_development_material": True, "reserved_test_bodies_read": False,
               "model_predictions_consulted": False, "label_actor": "ai_authored_not_human",
               "counts": dict(Counter(slot["split"] for slot in slots)), "builder": bind(Path(__file__))})
    return {"registered": len(slots), "counts": dict(Counter(s["track"] for s in slots))}


def source_row(source):
    keys = ("id", "source_group", "split", "language", "provenance", "natural_audio", "generated_audio", "acoustically_verified", "media_path", "media_sha256")
    return {key: source[key] for key in keys if key in source}


def save_state(output, state, directory="states"):
    state["input_sha256"] = data.digest(state["bundle"]["input"])
    data.write(output / directory / (state["id"] + ".json"), state)
    return {"id": state["id"], "input_sha256": state["input_sha256"], "split": state["split"],
            "language": state["language"], "source": state["source"], "input": state["bundle"]["input"]}


def natural(output):
    registration = read(output / "registration.json")
    tokenizer, tokenizer_identity = data.legacy.tokenizer()
    builder = data.contract.PassiveContextBuilder(tokenizer)
    blind, dispositions, documents = [], [], {}
    for slot in registration["slots"]:
        if slot["track"] != "A_raw":
            continue
        source = registration["sources"][slot["source_id"]]
        if source["id"] not in documents:
            documents[source["id"]] = read(checked_file(source["document"]))
        document = documents[source["id"]]
        try:
            bundle = builder.build(document, slot["focus_uuid"], as_of=0, caption_profile=data.PROFILE)
            state = {**slot, "source": source_row(source), "document_ref": source["document"],
                     "runtime": runtime(slot["focus_uuid"]), "bundle": bundle}
            blind.append(save_state(output, state))
        except (ValueError, KeyError) as error:
            dispositions.append({**slot, "status": "ineligible", "reason": str(error)})
        if len(blind) % 20 == 0:
            print(json.dumps({"natural_prepared": len(blind), "ineligible": len(dispositions)}), flush=True)
    data.write_rows(output / "blind" / "natural.jsonl", blind)
    data.write_rows(output / "preparation-ineligible-natural.jsonl", dispositions)
    data.write(output / "natural-inputs.json", {"version": VERSION, "registration": bind(output / "registration.json"),
               "prepared": len(blind), "ineligible": len(dispositions), "tokenizer": tokenizer_identity,
               "blind_roster": bind(output / "blind/natural.jsonl"), "references_authored": False})
    return {"prepared": len(blind), "ineligible": len(dispositions)}


def document_for(state):
    return state["document"] if "document" in state else read(checked_file(state["document_ref"]))


def draft_label(state, primary, acceptable, reason):
    document, bundle = document_for(state), state["bundle"]
    proofs = []
    for prediction in [primary, *acceptable]:
        proofs.append(data.execution(document, bundle, prediction, state["source"]))
    return {"id": state["id"], "input_sha256": state["input_sha256"],
            "status": "preserve" if primary["action"] == "no_action" else "edit_required",
            "primary": primary, "acceptable": acceptable, "reason": reason,
            "reference_execution": proofs, "independent_review": "pending"}


def prepare_spec(slot, spec, builder):
    document, keys, source_id = data.authored_document(spec)
    history, sequence = [], 0
    for prior in spec.get("history", []):
        before = copy.deepcopy(document)
        bundle = builder.build(before, keys[prior["targets"][0]], as_of=sequence,
                               history=history, caption_profile=data.PROFILE)
        prediction = data.resolve(prior, keys, bundle)
        data.execution(before, bundle, prediction, {"id": source_id})
        document = data.contract.apply_passive_edit(before, bundle, prediction)
        sequence += 1
        history.append(data.contract.committed_transition(before, document, prediction, sequence))
    bundle = builder.build(document, keys[spec["focus"]], as_of=sequence, history=history, caption_profile=data.PROFILE)
    source = {"id": source_id, "source_group": slot["source_group"], "split": slot["split"],
              "language": slot["language"], "provenance": "authored_symbolic_timing_not_audio",
              "natural_audio": False, "generated_audio": False, "acoustically_verified": False}
    rt = {**runtime(keys[spec["focus"]]), "sequence": sequence, "history": history}
    state = {**slot, "source": source, "document": document, "runtime": rt, "bundle": bundle,
             "input_sha256": data.digest(bundle["input"]), "recipe": spec,
             "provenance": "fresh_authored_mechanism_control_not_human_or_recorded_audio"}
    primary = data.resolve(spec["primary"], keys, bundle)
    acceptable = [data.resolve(p, keys, bundle) for p in spec.get("acceptable", [])]
    label = draft_label(state, primary, acceptable, spec["rationale"])
    return state, label


def authored(output, revision=1):
    import final_dataset_recipes as recipes

    registration = read(output / "registration.json")
    suffix = "" if revision == 1 else f"-revision-{revision:03}"
    tokenizer, identity = data.legacy.tokenizer()
    builder = data.contract.PassiveContextBuilder(tokenizer)
    blind, labels, dispositions, specs = [], [], [], []
    for slot in registration["slots"]:
        if slot["track"] not in ("C_contract", "R_rare_necessity"):
            continue
        spec = recipes.contract(slot) if slot["track"] == "C_contract" else recipes.rare(slot)
        specs.append(spec)
        try:
            state, label = prepare_spec(slot, spec, builder)
            blind.append(save_state(output, state, "states" + suffix))
            labels.append(label)
        except (ValueError, KeyError) as error:
            dispositions.append({**slot, "status": "ineligible", "reason": str(error)})
    data.write(output / f"authored-recipes{suffix}.json", {"version": VERSION, "generator": bind(Path(recipes.__file__)), "specs": specs})
    data.write_rows(output / f"blind/authored{suffix}.jsonl", blind)
    data.write_rows(output / f"draft-labels-authored{suffix}.jsonl", labels)
    data.write_rows(output / f"preparation-ineligible-authored{suffix}.jsonl", dispositions)
    data.write(output / f"authored-inputs{suffix}.json", {"version": VERSION, "prepared": len(blind),
               "ineligible": len(dispositions), "tokenizer": identity,
               "blind_roster": bind(output / f"blind/authored{suffix}.jsonl"), "draft_references": bind(output / f"draft-labels-authored{suffix}.jsonl"),
               "admitted": 0, "model_predictions_consulted": False})
    return {"prepared": len(blind), "ineligible": dispositions}


def authored_repair(output):
    return authored(output, revision=2)


def bridge(output):
    registration = read(output / "registration.json")
    wanted = {slot["parent_id"] for slot in registration["slots"] if slot["track"] == "B_preservation_bridge"}
    parents = {row["id"]: row for row in iter_rows(BASE / "cases/train.jsonl") if row["id"] in wanted}
    tokenizer, identity = data.legacy.tokenizer()
    builder = data.contract.PassiveContextBuilder(tokenizer)
    blind, labels, dispositions = [], [], []
    typo_pairs = (("problem", "problme"), ("correct", "corerct"), ("computer", "comptuer"),
                  ("question", "qusetion"), ("thinking", "thikning"), ("sometimes", "sometiems"),
                  ("students", "studnets"), ("memory", "memroy"), ("function", "fucntion"))
    for slot in registration["slots"]:
        if slot["track"] != "B_preservation_bridge":
            continue
        parent = parents[slot["parent_id"]]
        data.require(parent["source"]["source_group"] == slot["source_group"] and parent["primary"]["action"] == "no_action",
                     "Registered bridge parent differs.")
        document = copy.deepcopy(parent["document"])
        if slot["variant"] != "fresh_history":
            for item in document["content"]:
                if item["type"] == "paragraph_start":
                    item["speaker"] = None
        rt = runtime(parent["runtime"]["focus_uuid"])
        try:
            bundle = builder.build(document, rt["focus_uuid"], as_of=0, caption_profile=data.PROFILE)
            primary = {"action": "no_action"}
            reason = "The prior TRAIN parent was independently preserved after real simulated edits. This new whole native window is re-reviewed without available history; no_action is a draft claim, not inherited admission. Preserve meaningful questions, uncertainty, repetitions and all numbers."
            if slot["variant"] == "unknown_speaker_typo":
                replacement = None
                for correct, wrong in typo_pairs:
                    for item in bundle["input"]["items"]:
                        if item["type"] == "text" and re.search(r"\b" + correct + r"\b", item["text"]):
                            target_uuid = bundle["mapping"][str(item["id"])]["uuid"]
                            target = next(value for value in document["content"] if value["uuid"] == target_uuid)
                            replacement = target["text"]
                            target["text"] = re.sub(r"\b" + correct + r"\b", wrong, replacement, count=1)
                            break
                    if replacement:
                        break
                data.require(replacement is not None, "No unambiguous ordinary-word typo available.")
                bundle = builder.build(document, rt["focus_uuid"], as_of=0, caption_profile=data.PROFILE)
                aliases = {value["uuid"]: int(alias) for alias, value in bundle["mapping"].items()}
                data.require(target_uuid in aliases, "Synthetic typo left the native window.")
                primary = {"action": "correct_text", "target": [aliases[target_uuid]], "text": replacement}
                reason = "One explicit ordinary-word transposition was inserted into this TRAIN-source derived control. Correct only that malformed spelling, preserving the full item audio, names, numbers and intended speech. Unknown speaker metadata does not prevent a clear display correction."
            state = {**slot, "document": document, "runtime": rt, "bundle": bundle,
                     "source": copy.deepcopy(parent["source"]), "input_sha256": data.digest(bundle["input"]),
                     "parent_sha256": data.digest(parent), "provenance": "source_derived_synthetic_metadata_and_history_control"}
            label = draft_label(state, primary, [], reason)
            blind.append(save_state(output, state))
            labels.append(label)
        except (ValueError, KeyError) as error:
            dispositions.append({**slot, "status": "ineligible", "reason": str(error)})
    data.write_rows(output / "blind/bridge.jsonl", blind)
    data.write_rows(output / "draft-labels-bridge.jsonl", labels)
    data.write_rows(output / "preparation-ineligible-bridge.jsonl", dispositions)
    data.write(output / "bridge-inputs.json", {"version": VERSION, "prepared": len(blind), "ineligible": len(dispositions),
               "blind_roster": bind(output / "blind/bridge.jsonl"), "tokenizer": identity, "admitted": 0})
    return {"prepared": len(blind), "ineligible": dispositions}


def history(output, revision=1):
    import final_dataset_recipes as recipes
    import prepare_history_length_actions_v3 as histories

    registration = read(output / "registration.json")
    suffix = "" if revision == 1 else f"-revision-{revision:03}"
    episodes = [slot for slot in registration["slots"] if slot["track"] == "H_history" and slot["history_length"] == 8]
    fixture = recipes.history_fixture(episodes)
    data.write(output / f"history-recipes{suffix}.json", fixture)
    tokenizer, identity = data.legacy.tokenizer()
    blind, labels, dispositions, proofs = [], [], [], []
    for slot, episode in zip(episodes, fixture["episodes"], strict=True):
        try:
            cases, requests, proof = histories.build_episode(fixture, episode, tokenizer, data.sha256(output / f"history-recipes{suffix}.json"))
            proofs.append(proof)
            for case, request in zip(cases, requests, strict=True):
                length = case["history_length"]["length"]
                source = {**case["source"], "split": slot["split"]}
                state = {**slot, "id": episode["id"] + f"-h{length}", "history_length": length,
                         "source": source, "document": case["document"], "runtime": case["runtime"],
                         "bundle": case["bundle"], "history_projection": case["history_length"],
                         "input_sha256": data.digest(case["input"]),
                         "provenance": "authored_fixed_window_actual_transition_history_projection"}
                blind.append(save_state(output, state, "states" + suffix))
                labels.append(draft_label(state, case["primary"], case["acceptable"][1:], case["rationale"]))
        except (ValueError, KeyError) as error:
            for length in (0, 3, 8):
                dispositions.append({**slot, "id": episode["id"] + f"-h{length}", "history_length": length,
                                     "status": "ineligible", "reason": str(error)})
    data.write_rows(output / f"blind/history{suffix}.jsonl", blind)
    data.write_rows(output / f"draft-labels-history{suffix}.jsonl", labels)
    data.write_rows(output / f"history-execution-proofs{suffix}.jsonl", proofs)
    data.write_rows(output / f"preparation-ineligible-history{suffix}.jsonl", dispositions)
    data.write(output / f"history-inputs{suffix}.json", {"version": VERSION, "prepared": len(blind), "ineligible": len(dispositions),
               "blind_roster": bind(output / f"blind/history{suffix}.jsonl"), "tokenizer": identity, "admitted": 0,
               "window_policy": "fixed_at_eight_history", "lengths": [0, 3, 8]})
    return {"prepared": len(blind), "ineligible": dispositions}


def history_revision(output):
    return history(output, revision=3)


def history_repair(output):
    import final_dataset_recipes as recipes
    import prepare_history_length_actions_v3 as histories

    registration = read(output / "registration.json")
    failed = data.rows(output / "preparation-ineligible-history.jsonl")
    episode_ids = {slot["episode_id"] for slot in failed}
    episodes = [slot for slot in registration["slots"] if slot["track"] == "H_history"
                and slot["history_length"] == 8 and slot["episode_id"] in episode_ids]
    fixture = recipes.history_fixture(episodes)
    data.write(output / "history-recipes-revision-002.json", fixture)
    tokenizer, identity = data.legacy.tokenizer()
    blind, labels, proofs = [], [], []
    for slot, episode in zip(episodes, fixture["episodes"], strict=True):
        cases, requests, proof = histories.build_episode(fixture, episode, tokenizer, data.sha256(output / "history-recipes-revision-002.json"))
        proofs.append(proof)
        for case in cases:
            length = case["history_length"]["length"]
            state = {**slot, "id": episode["id"] + f"-h{length}", "history_length": length,
                     "source": {**case["source"], "split": slot["split"]}, "document": case["document"],
                     "runtime": case["runtime"], "bundle": case["bundle"], "history_projection": case["history_length"],
                     "input_sha256": data.digest(case["input"]), "repair_of": bind(output / "preparation-ineligible-history.jsonl"),
                     "provenance": "authored_fixed_window_actual_transition_history_projection"}
            blind.append(save_state(output, state))
            labels.append(draft_label(state, case["primary"], [], case["rationale"]))
    data.write_rows(output / "blind/history-repaired.jsonl", blind)
    data.write_rows(output / "draft-labels-history-repaired.jsonl", labels)
    data.write_rows(output / "history-execution-proofs-repaired.jsonl", proofs)
    data.write(output / "history-repair.json", {"version": VERSION, "prepared": len(blind), "tokenizer": identity,
               "reason": "Replace substring-overlapping Korean name variants in one failed episode before semantic review.",
               "blind_roster": bind(output / "blind/history-repaired.jsonl"), "previous_failed_attempt_preserved": True})
    return {"repaired": len(blind)}


def state_paths(output):
    found = {}
    for directory in sorted(output.glob("states*")):
        if directory.is_dir():
            found.update({path.stem: path for path in directory.glob("*.json")})
    return found


def read_reviews(paths):
    decisions, receipts = {}, []
    for path in paths:
        review = read(path)
        data.require(review.get("reviewer_id") and review.get("actor") in ("ai_generated", "synthetic"),
                     "Independent AI reviewer identity is required; do not invent human labels.")
        data.require(review["reviewer_id"] != "final_dataset_agent", "The candidate author cannot independently admit its own references.")
        data.require(review.get("model_predictions_consulted") is False, "Admission review must be prediction-blind.")
        for decision in review["decisions"]:
            data.require(decision["id"] not in decisions, "Two current reviews for the same candidate.")
            data.require(decision["status"] in ("preserve", "edit_required", "unknown"), "Unknown review necessity.")
            data.require(decision.get("reason", "").strip(), "Review needs a concrete rationale.")
            if decision["status"] != "unknown":
                data.require(decision.get("whole_visible_window_reviewed") is True, "Admitted reference needs whole-window review.")
                data.require(decision.get("primary"), "Reviewed reference is missing.")
                data.require((decision["primary"]["action"] == "no_action") == (decision["status"] == "preserve"),
                             "Preservation and action necessity disagree.")
                actions = {prediction["action"] for prediction in [decision["primary"], *decision.get("acceptable", [])]}
                data.require(actions == {"no_action"} if decision["status"] == "preserve" else "no_action" not in actions,
                             "Accepted alternatives cannot mix preservation and edit necessity.")
            else:
                data.require(not decision.get("primary"), "Unknown must not become no_action.")
            decisions[decision["id"]] = {**decision, "reviewer_id": review["reviewer_id"], "review_file": bind(path)}
        receipts.append(bind(path))
    return decisions, receipts


def terminal(output, review_paths):
    decisions, receipts = read_reviews(review_paths)
    registration, paths = read(output / "registration.json"), state_paths(output)
    tokenizer, identity = data.legacy.tokenizer()
    builder = data.contract.PassiveContextBuilder(tokenizer)
    blind, dispositions = [], []
    for slot in registration["slots"]:
        if slot["track"] != "A_terminal":
            continue
        decision = decisions.get(slot["parent_id"])
        if decision is None or decision["status"] == "unknown":
            dispositions.append({**slot, "status": "unknown", "reason": "No independently justified prior edit is available in the registered raw parent; no synthetic terminal no_action is manufactured."})
            continue
        parent = read(paths[slot["parent_id"]])
        data.require(parent["input_sha256"] == decision["input_sha256"], "Terminal parent review input changed.")
        if decision["status"] == "preserve":
            dispositions.append({**slot, "status": "duplicate", "kept_id": slot["parent_id"],
                                 "reason": "The independently preserved raw state needs zero edits and is not duplicated as a terminal example."})
            continue
        try:
            before, bundle = document_for(parent), parent["bundle"]
            prior = decision["primary"]
            proof = data.execution(before, bundle, prior, parent["source"])
            after = data.contract.apply_passive_edit(before, bundle, prior)
            focus = parent["runtime"]["focus_uuid"]
            data.require(any(item["uuid"] == focus for item in after["content"]), "The prior edit removes the registered focus; do not retarget it.")
            rt = {**runtime(focus), "sequence": 1,
                  "history": [data.contract.committed_transition(before, after, prior, 1)]}
            final_bundle = builder.build(after, focus, as_of=1, history=rt["history"], caption_profile=data.PROFILE)
            state = {**slot, "source": parent["source"], "document": after, "runtime": rt,
                     "bundle": final_bundle, "prior_edits": [{"review": decision, "proof": proof}],
                     "actual_prior_edit_count": 1, "terminal_status": "after_one_reviewed_edit_not_assumed_clean",
                     "provenance": "natural_source_after_synthetic_acceptance_of_independently_reviewed_edit"}
            blind.append(save_state(output, state))
        except (ValueError, KeyError) as error:
            dispositions.append({**slot, "status": "ineligible", "reason": str(error)})
    data.write_rows(output / "blind/terminal.jsonl", blind)
    data.write_rows(output / "terminal-dispositions.jsonl", dispositions)
    data.write(output / "terminal-inputs.json", {"version": VERSION, "prepared": len(blind),
               "dispositions": len(dispositions), "independent_parent_reviews": receipts, "tokenizer": identity,
               "prior_operation_limit_per_slot": 3, "actual_prior_operations": len(blind),
               "blind_roster": bind(output / "blind/terminal.jsonl"), "admitted": 0})
    return {"prepared": len(blind), "dispositions": dict(Counter(row["status"] for row in dispositions))}


def semantic_input_hash(snapshot):
    value = copy.deepcopy(snapshot)
    sources = {}
    for item in value["items"]:
        if item["source_id"] is not None:
            original = item["source_id"]
            item["source_id"] = sources.setdefault(original, "source-" + str(len(sources)))
    return data.digest(value)


def census(cases):
    counts = Counter((r["split"], r["language"], r["primary"]["action"]) for r in cases)
    return {"rows": len(cases), "split": dict(Counter(r["split"] for r in cases)),
            "action_language": [{"split": split, "language": language, "action": action, "count": count}
                                for (split, language, action), count in sorted(counts.items())],
            "source_groups": {split: len({r["source"]["source_group"] for r in cases if r["split"] == split}) for split in ("train", "dev")},
            "necessity": dict(Counter(r["judgment"]["status"] for r in cases)),
            "accepted_references": sum(len(r["accepted_predictions"]) for r in cases),
            "strata": dict(Counter(r["lineage"].get("stratum", "legacy") for r in cases))}


def bound_evaluation_bundle(case):
    """Bind saved input to the current document without rebuilding its window."""
    snapshot, document = case["input"], case["document"]
    data.require(case["messages"] == data.contract.canonical_messages(snapshot), "Saved canonical messages changed.")
    _, mapping = data.contract._records(document)
    aliases = {str(item["id"]) for item in snapshot["items"]}
    data.require(aliases <= mapping.keys(), "Saved aliases no longer exist in the document.")
    bundle = {"input": snapshot, "mapping": {alias: mapping[alias] for alias in aliases},
              "document_hash": data.session_hash(document),
              "caption_authority": {"version": 1, "protected_uuids": [], "profile": snapshot["caption_profile"]},
              "context": case.get("bundle", {}).get("context", case["lineage"].get("context", {}))}
    if "bundle" in case:
        for key in ("input", "mapping", "document_hash", "caption_authority"):
            data.require(case["bundle"][key] == bundle[key], "Saved bundle changed its document or visible items.")
    data.contract._legacy_bundle(document, bundle)
    meta = case.get("history_length")
    if meta is not None:
        _, summaries = data.contract._history(document, case["runtime"]["history"], case["runtime"]["sequence"])
        length = meta["length"]
        data.require(length in (0, 3, 8) and len(summaries) == 8, "Unregistered history projection.")
        expected = summaries[-length:] if length else []
        data.require(snapshot["recent_edits"] == expected, "History projection differs from the actual chronological suffix.")
        data.require(data.digest({k: v for k, v in snapshot.items() if k != "recent_edits"}) == meta["non_history_sha256"],
                     "A history arm changed non-history input.")
        anchor = {**snapshot, "recent_edits": summaries}
        data.require(data.digest(data.contract.wire_snapshot(anchor)) == meta["anchor_input_sha256"], "History-8 anchor differs.")
        data.require(data.digest(case["runtime"]["history"]) == meta["history_chain_sha256"], "Actual history chain differs.")
    return bundle


def evaluation(release, output, cohort="combined"):
    import evaluate_passive_actions_v3 as grading
    import passive_formats_actions_v3 as formats

    data.require(not output.exists(), "Use a fresh development preparation.")
    manifest = read(release / "manifest.json")
    data.require(manifest["version"] == VERSION and manifest["test_bodies_read"] is False, "Unknown dataset scope.")
    grading.verify_files(release, manifest)
    filename = {"combined": "dev.jsonl", "canonical": "canonical-dev322.jsonl", "supplemental": "supplemental-dev.jsonl",
                "history": "history-diagnostic-dev.jsonl"}[cohort]
    cases_path = release / "cases" / filename
    tokenizer, identity = data.legacy.tokenizer()
    output.mkdir(parents=True)
    case_ids, requests, anchors, documents, groups, history_episodes, anchor_bindings = [], [], set(), set(), set(), set(), {}
    history_condition_cases = 0
    with (output / "cases.jsonl").open("x") as stream:
        for case in iter_rows(cases_path):
            data.require(case["split"] == "dev", "Development preparation cannot consume TRAIN or TEST.")
            bundle = bound_evaluation_bundle(case)
            accepted = [grading.result_hash(data.contract.apply_passive_edit(case["document"], bundle, prediction))
                        for prediction in grading.references(case)]
            prefix = formats.prefix_ids(tokenizer, case["input"], "json")
            data.require(len(prefix) <= 3584, "Complete saved evaluation input exceeds native reference budget; no truncation or silent exclusion.")
            prepared = {**case, "bundle": bundle, "accepted_result_sha256": sorted(set(accepted))}
            stream.write(data.packed(prepared) + "\n")
            messages = case["messages"]
            requests.append({"case_id": case["id"], "split": "dev", "format": "json", "input": case["input"],
                             "messages": messages, "input_sha256": grading.semantic_input_hash(case["input"]),
                             "messages_sha256": data.digest(messages), "prefix_ids": prefix, "prefix_ids_sha256": data.digest(prefix)})
            case_ids.append(case["id"])
            documents.add(bundle["document_hash"])
            groups.add(case["source"]["source_group"])
            anchors.add((bundle["document_hash"], case["runtime"]["focus_uuid"]))
            if "history_length" in case:
                history_condition_cases += 1
                meta = case["history_length"]
                episode = meta["episode_id"]
                binding = (bundle["document_hash"], meta["non_history_sha256"], meta["history_chain_sha256"])
                data.require(anchor_bindings.setdefault(episode, binding) == binding, "History arms changed their common document or anchor.")
                history_episodes.add(episode)
    data.require(case_ids and len(set(case_ids)) == len(case_ids), "Empty or duplicate evaluation roster.")
    data.write_rows(output / "requests-json.jsonl", requests)
    data.write_rows(output / "exclusions.jsonl", [])
    prepared_manifest = {"version": grading.VERSION, "final_builder_version": VERSION,
                         "status": "prepared_not_inferred", "dataset_manifest": grading.file_binding(release / "manifest.json"),
                         "semantic_reviews": manifest["independent_reviews"], "tokenizer": identity,
                         "requests": len(case_ids), "eligible_formats": ["json"], "eligibility_format": "json",
                         "original_development_cases": len(case_ids), "excluded": 0, "case_ids": case_ids,
                         "cohort": cohort, "unique_editor_anchors": len(anchors), "unique_current_documents": len(documents),
                         "source_groups": len(groups), "history_condition_cases": history_condition_cases,
                         "history_episodes": len(history_episodes), "history_anchor_bindings": anchor_bindings,
                         "history_interpretation": "0/3/8 conditions share an editor anchor; conditions are not independent states. Keep this development ablation separate from default-history-3 final TEST.",
                         "reference_proof_verification": "Every accepted effect executed against saved canonical input bound to its exact current document; H projections verify actual eight-transition history and common non-history anchor.",
                         "final_test_bodies_read": False,
                         "files_sha256": {path.name: data.sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
                         "sources_sha256": {str(path.resolve()): data.sha256(path) for path in [
                             Path(__file__), Path(grading.__file__), Path(formats.__file__), Path(data.__file__),
                             Path(data.contract.__file__), data.contract.PROMPT_PATH,
                             ROOT / "ml/caption_facts.py", ROOT / "ml/grading.py", ROOT / "ml/verify_release_export.py"]}}
    data.write(output / "manifest.json", prepared_manifest)
    return {key: prepared_manifest[key] for key in ("requests", "unique_editor_anchors", "source_groups", "history_condition_cases", "history_episodes")}


def render(output, names=("natural", "authored-revision-002", "bridge", "history", "history-repaired"), packet_name="review-packets"):
    packet = output / packet_name
    data.require(not packet.exists(), "Use a fresh review packet.")
    packet.mkdir()
    manifest = {"version": VERSION, "author_references_included": False, "model_outputs_included": False,
                "faithfulness": "All snapshot keys are represented. Identical contract/task/mode/actions/profile are printed once per chunk. Items use the complete ordered field schema; caption facts and history/feedback are unabridged JSON.",
                "chunks": []}
    item_fields = ["id", "type", "text", "duration_ms", "paragraph", "speaker", "language", "source_backed", "source_run", "boundary_status", "audio_class", "source_id", "source_start_ms", "source_end_ms"]
    shared_fields = ["contract_version", "task", "mode", "allowed_actions", "caption_profile"]
    for name in names:
        path = output / "blind" / f"{name}.jsonl"
        rows = data.rows(path)
        for start in range(0, len(rows), 24):
            chunk = rows[start:start + 24]
            lines = ["PREDICTION-BLIND COMPLETE INPUT PACKET", "Source: " + str(path.relative_to(ROOT)),
                     "Shared input: " + data.packed({key: chunk[0]["input"][key] for key in shared_fields}),
                     "Item field order: " + data.packed(item_fields),
                     "Every item below is one JSON array in this order. No visible text, interval, caption fact, history, or feedback is omitted.", ""]
            for row in chunk:
                snapshot = row["input"]
                lines += ["CASE " + row["id"], "input_sha256=" + row["input_sha256"],
                          "Source metadata: " + data.packed(row["source"]),
                          "State: " + data.packed({key: value for key, value in snapshot.items()
                              if key not in [*shared_fields, "items", "caption_facts", "recent_edits", "recent_feedback"]})]
                differences = {key: snapshot[key] for key in shared_fields if snapshot[key] != chunk[0]["input"][key]}
                if differences:
                    lines.append("Shared input overrides: " + data.packed(differences))
                lines.append("Items:")
                for item in snapshot["items"]:
                    data.require(set(item) == set(item_fields), "An item field would be omitted from review rendering.")
                    lines.append(data.packed([item[key] for key in item_fields]))
                lines.append("Caption facts: " + data.packed(snapshot["caption_facts"]))
                lines.append("Recent edits: " + data.packed(snapshot["recent_edits"]))
                lines.append("Recent feedback: " + data.packed(snapshot["recent_feedback"]))
                lines += ["END CASE", ""]
            target = packet / f"{name}-{start // 24 + 1:02}.txt"
            target.write_text("\n".join(lines) + "\n")
            manifest["chunks"].append({**bind(target), "input_roster": bind(path), "cases": len(chunk),
                                      "first_id": chunk[0]["id"], "last_id": chunk[-1]["id"]})
    data.write(packet / "manifest.json", manifest)
    return {"chunks": len(manifest["chunks"]), "cases": sum(row["cases"] for row in manifest["chunks"]), "path": str(packet)}


def render_revision(output):
    return render(output, names=("terminal", "history-revision-003"), packet_name="review-packets-revision-002")


def admit(output, review_paths, release):
    data.require(not release.exists(), "Use a new immutable release directory.")
    registration = read(output / "registration.json")
    checked_file(registration["base_train"])
    checked_file(registration["base_dev"])
    metadata_keys = ("id", "input", "split", "language", "source", "primary", "judgment", "lineage", "accepted_predictions")
    def base_metadata(split):
        result = []
        for row in iter_rows(BASE / f"cases/{split}.jsonl"):
            result.append({**{key: row[key] for key in metadata_keys},
                           "editor_anchor": (data.session_hash(row["document"]), row["runtime"]["focus_uuid"])})
        return result
    base_train, base_dev = base_metadata("train"), base_metadata("dev")
    data.require(len(base_train) == 870 and len(base_dev) == 322, "Original denominators changed.")
    decisions, receipts = read_reviews(review_paths)
    paths, slots = state_paths(output), {s["id"]: s for s in registration["slots"]}
    data.require(set(decisions) <= set(slots), "Unregistered review candidate.")
    tokenizer, tokenizer_identity = data.legacy.tokenizer()
    terminal_dispositions = {r["id"]: r for r in data.rows(output / "terminal-dispositions.jsonl")} if (output / "terminal-dispositions.jsonl").exists() else {}
    supplements, ledger, proofs, token_rows, history_diagnostic = [], [], [], [], []
    exact = {data.digest(row["input"]): row for row in [*base_train, *base_dev]}
    semantic = {semantic_input_hash(row["input"]): row for row in [*base_train, *base_dev]}
    for identity, slot in slots.items():
        if identity not in paths:
            ledger.append(terminal_dispositions.get(identity, {**slot, "status": "unavailable", "reason": "No valid prepared state exists; this is not an abstention label."}))
            continue
        state = read(paths[identity])
        decision = decisions.get(identity)
        if decision is None:
            ledger.append({**slot, "status": "unreviewed_excluded", "input_sha256": state["input_sha256"],
                           "reason": "No independent semantic admission decision was supplied."})
            continue
        data.require(decision["input_sha256"] == state["input_sha256"] == data.digest(state["bundle"]["input"]),
                     "Reviewed input changed: " + identity)
        if decision["status"] == "unknown":
            ledger.append({**slot, **decision, "status": "unknown_excluded"})
            continue
        source, document, bundle = state["source"], document_for(state), state["bundle"]
        data.require(registration["source_splits"][source["source_group"]] == slot["split"] == source["split"],
                     "Supplement crosses source split.")
        primary, acceptable = decision["primary"], decision.get("acceptable", [])
        row = data.case_row(identity, slot["split"], slot["language"], source, document, bundle, primary, acceptable,
                            state["runtime"], decision["reason"],
                            {"old_id": None, "selection_method": "source_first" if slot["track"] == "A_raw" else "registered_supplement",
                             "scenario_family": slot.get("family", slot.get("mechanism", slot["track"])), "supplement_track": slot["track"],
                             "stratum": "natural_recorded" if slot["track"] == "A_raw" else
                                 "natural_after_synthetic_edit" if slot["track"] == "A_terminal" else
                                 "source_derived_synthetic_control" if slot["track"] == "B_preservation_bridge" else
                                 "authored_history_projection" if slot["track"] == "H_history" else "authored_symbolic",
                             "candidate_state": bind(paths[identity]), "registration": bind(output / "registration.json"),
                             "independent_review": decision["review_file"], "reviewer_id": decision["reviewer_id"],
                             "context": bundle["context"], "parent_id": slot.get("parent_id"),
                             "episode_id": slot.get("episode_id"), "label_actor": "ai_generated_not_human"})
        row["judgment"]["independent_review"] = "input_blind_review_admitted"
        if "history_projection" in state:
            row.update(bundle=bundle, history_length=state["history_projection"])
            data.require(row["input"]["recent_edits"] == (state["bundle"]["input"]["recent_edits"]), "History projection changed.")
        try:
            evidence = [data.execution(document, bundle, prediction, source) for prediction in row["accepted_predictions"]]
            input_tokens = len(data.contract.token_ids(tokenizer, row["input"]))
            output_tokens = [len(tokenizer.encode(data.packed(prediction), add_special_tokens=False)) + 1 for prediction in row["accepted_predictions"]]
            data.require(input_tokens <= 3584 and max(output_tokens) <= 512, "Untruncated native token budget exceeded.")
        except (ValueError, KeyError) as error:
            ledger.append({**slot, "status": "reference_ineligible", "input_sha256": state["input_sha256"],
                           "reason": str(error), "review": decision})
            continue
        row["execution_sha256"] = data.digest(evidence[0])
        if "history_length" in row and row["split"] == "dev":
            history_diagnostic.append(row)
        key, content_key = state["input_sha256"], semantic_input_hash(row["input"])
        previous = exact.get(key) or semantic.get(content_key)
        if previous:
            same = previous["split"] == row["split"]
            reference_overlap = bool({data.packed(p) for p in previous["accepted_predictions"]}
                                     & {data.packed(p) for p in row["accepted_predictions"]})
            status = ("duplicate_excluded" if reference_overlap else "duplicate_reference_difference_excluded") if same else "cross_split_duplicate_excluded"
            ledger.append({**slot, "status": status, "accepted_reference_overlap": reference_overlap,
                           "kept_id": previous["id"], "input_sha256": key, "semantic_input_sha256": content_key,
                           "reason": "Equivalent predictor input after opaque source-identity normalization; never inflate requests using a new symbolic source ID."})
            continue
        supplements.append(row)
        exact[key], semantic[content_key] = row, row
        proofs.append({"id": identity, "accepted": evidence})
        token_rows.append({"id": identity, "input_tokens": input_tokens, "accepted_completion_tokens": output_tokens})
        ledger.append({**slot, "status": "admitted", "input_sha256": key,
                       "judgment_status": row["judgment"]["status"], "primary_action": primary["action"],
                       "review": decision["review_file"]})
    train_new = [row for row in supplements if row["split"] == "train"]
    dev_new = [row for row in supplements if row["split"] == "dev"]
    prior_spans, overlap_audit = {}, []
    def span(row):
        items = [item for item in row["input"]["items"] if item["source_start_ms"] is not None]
        return (min(item["source_start_ms"] for item in items), max(item["source_end_ms"] for item in items)) if items else None
    for row in [*base_train, *base_dev]:
        interval = span(row)
        if row["source"].get("natural_audio") and interval:
            key = (row["source"]["source_group"], row["source"]["id"])
            prior_spans.setdefault(key, []).append((row["id"], row["split"], *interval))
    for row in supplements:
        interval = span(row)
        if not row["source"].get("natural_audio") or not interval:
            continue
        key = (row["source"]["source_group"], row["source"]["id"])
        matches = []
        for old_id, old_split, lo, hi in prior_spans.get(key, []):
            overlap = max(0, min(interval[1], hi) - max(interval[0], lo))
            fraction = overlap / min(interval[1] - interval[0], hi - lo)
            if fraction >= 0.8:
                data.require(old_split == row["split"], "Overlapping source material crosses splits.")
                matches.append({"id": old_id, "smaller_window_overlap_fraction": fraction})
        overlap_audit.append({"id": row["id"], "split": row["split"], "source_group": key[0], "source_id": key[1],
                              "source_interval_ms": list(interval), "prior_windows_at_80pct_overlap": matches})
    release.mkdir(parents=True)
    (release / "cases").mkdir()
    for split, original, additions in (("train", base_train, train_new), ("dev", base_dev, dev_new)):
        parent_path = BASE / f"cases/{split}.jsonl"
        with parent_path.open("rb") as stream:
            stream.seek(-1, 2)
            data.require(stream.read() == b"\n", "Original JSONL needs a terminal newline.")
        with (release / f"cases/{split}.jsonl").open("xb") as stream:
            with parent_path.open("rb") as parent_stream:
                shutil.copyfileobj(parent_stream, stream)
            for row in additions:
                stream.write((data.packed(row) + "\n").encode())
        data.write_rows(release / f"cases/supplemental-{split}.jsonl", additions)
    shutil.copyfile(BASE / "cases/dev.jsonl", release / "cases/canonical-dev322.jsonl")
    data.write_rows(release / "cases/history-diagnostic-dev.jsonl", history_diagnostic)
    data.write_rows(release / "admission-ledger.jsonl", ledger)
    data.write_rows(release / "reference-executions-supplement.jsonl", proofs)
    data.write_rows(release / "native-token-counts-supplement.jsonl", token_rows)
    data.write(release / "natural-source-overlap-audit.json", {
        "scope": "Exact source-group/source-ID matches against original TRAIN870 and DEV322; reused source intervals are correlated development material.",
        "windows": overlap_audit, "new_natural_rows_with_prior_80pct_overlap": sum(bool(r["prior_windows_at_80pct_overlap"]) for r in overlap_audit),
        "cross_split_overlap_matches": 0, "new_source_independence_claimed": False})
    data.write(release / "source-splits.json", registration["source_splits"])
    data.write(release / "coverage.json", {"all": census([*base_train, *base_dev, *supplements]),
               "original": census([*base_train, *base_dev]), "supplement": census(supplements),
               "slot_dispositions": dict(Counter(row["status"] for row in ledger)),
               "supplement_by_track_split": [{"track": track, "split": split, "count": count}
                   for (track, split), count in sorted(Counter((r["lineage"]["supplement_track"], r["split"]) for r in supplements).items())]})
    protocol = read(ROOT / "artifacts/experiments/passive-actions-v3/substantive-protocol-003-json-targeted.json")
    original_draws = protocol["training_plan"]["sample_order"]
    data.require(len(original_draws) == 1530 and set(original_draws) == {r["id"] for r in base_train}, "Original B003 draw census changed.")
    decorated = [(data.digest([VERSION, "draw-interleave-001", i, name]), name) for i, name in enumerate([*original_draws, *[r["id"] for r in train_new]])]
    draw_order = [name for _, name in sorted(decorated)]
    data.write(release / "historical-proposed-draw-order.json", {"version": VERSION,
               "status": "historical_proposal_superseded_not_executed_for_final_small_models",
               "policy": "Historical B004 proposal: retain original B003 1530 multiplicities; each newly admitted TRAIN state once; seeded SHA256 interleaving. Final small-model training instead follows three epochs over all unique final TRAIN rows.",
               "original_unique": 870, "original_draws": 1530, "new_unique": len(train_new),
               "total_draws": len(draw_order), "sample_order": draw_order,
               "original_protocol": bind(ROOT / "artifacts/experiments/passive-actions-v3/substantive-protocol-003-json-targeted.json")})
    shutil.copyfile(ROOT / "docs/plans/final-sft-comparison-2026-10-25.md", release / "training-plan-at-admission.md")
    (release / "source").mkdir()
    for path in (Path(__file__), ROOT / "ml/final_dataset_recipes.py"):
        shutil.copyfile(path, release / "source" / path.name)
    shutil.copytree(output / "exposure-audit", release / "exposure-audit")
    manifest = {"version": VERSION, "status": "independently_reviewed_unified_train_development_release",
                "counts": {"train": 870 + len(train_new), "dev": 322 + len(dev_new), "canonical_dev": 322,
                           "supplemental_train": len(train_new), "supplemental_dev": len(dev_new)},
                "unique_editor_anchors": len({row["editor_anchor"] for row in [*base_train, *base_dev]}
                                             | {(data.session_hash(row["document"]), row["runtime"]["focus_uuid"]) for row in supplements}),
                "history_condition_cases": sum("history_length" in row for row in supplements),
                "history_episodes": len({row["history_length"]["episode_id"] for row in supplements if "history_length" in row}),
                "history_diagnostic_dev": {"requests": len(history_diagnostic),
                                           "episodes": len({row["history_length"]["episode_id"] for row in history_diagnostic}),
                                           "policy": "Independently reviewed paired 0/3/8 view, including within-split semantically duplicate zero-history controls. This is not an additional TRAIN/DEV admission denominator or independent-state count."},
                "original_train_prefix_byte_identical": True, "original_dev_prefix_byte_identical": True,
                "canonical_dev_sha256": data.sha256(release / "cases/canonical-dev322.jsonl"),
                "parent": bind(BASE / "manifest.json"), "registration": bind(output / "registration.json"),
                "independent_reviews": receipts, "tokenizer": tokenizer_identity,
                "builder": bind(Path(__file__)),
                "runtime_sources": [bind(ROOT / name) for name in ("ml/passive_actions.py", "ml/caption_facts.py", "ml/prompt-passive-actions-v1.txt", "ml/prepare_passive_actions_v3.py", "ml/replay_session.py")],
                "source_groups_are_exposed_development_material": True,
                "test_bodies_read": False, "label_actor": "AI-authored and independently AI-reviewed, never observed human behavior",
                "accepted_references_policy": "Primary first plus concrete independently accepted alternatives; no global whitespace, punctuation or number normalization.",
                "exposed_visible_wording_audit": bind(release / "exposure-audit/manifest.json"),
                "history_policy": "H arms retain the native history-8 anchor window and project only recent_edits to 0/3/8; other rows use the ordinary runtime builder. Default runtime remains history 3.",
                "acoustic_scope": "No new acoustic certification. Symbolic timing is not recorded audio; natural unknown remains unknown and only reversible source-valid previews are admitted.",
                "historical_proposed_draw_order": bind(release / "historical-proposed-draw-order.json"),
                "current_training_plan": bind(release / "training-plan-at-admission.md"),
                "current_training_plan_status": "Three epochs over all unique final TRAIN rows for each small model; dataset assembly does not execute training.",
                "files_sha256": {str(path.relative_to(release)): data.sha256(path) for path in sorted(release.rglob("*")) if path.is_file()}}
    data.write(release / "manifest.json", manifest)
    return manifest["counts"]


def verify(release, output):
    import hashlib
    import evaluate_passive_actions_v3 as grading

    manifest = read(release / "manifest.json")
    data.require(manifest["version"] == VERSION, "Unknown final dataset.")
    grading.verify_files(release, manifest)
    registration = read(checked_file(manifest["registration"]))
    ledger = data.rows(release / "admission-ledger.jsonl")
    data.require(len(ledger) == 594 and {r["id"] for r in ledger} == {r["id"] for r in registration["slots"]},
                 "Candidate disposition ledger is incomplete.")
    metadata, identities, split_groups, anchors = [], set(), {}, set()
    h_cases, h_episodes = [], set()
    input_seen = {}
    prefix_verified = {}
    for split, old_count in (("train", 870), ("dev", 322)):
        original_path, path = BASE / f"cases/{split}.jsonl", release / f"cases/{split}.jsonl"
        remaining, digest = original_path.stat().st_size, hashlib.sha256()
        with path.open("rb") as stream:
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                data.require(block, "Final release lost original bytes.")
                digest.update(block)
                remaining -= len(block)
        data.require(digest.hexdigest() == data.sha256(original_path), "Original case prefix changed.")
        prefix_verified[split] = {"rows": old_count, "bytes": original_path.stat().st_size, "sha256": digest.hexdigest()}
        count = 0
        for index, row in enumerate(iter_rows(path)):
            data.require(row["id"] not in identities and row["split"] == row["source"]["split"] == split, "Duplicate identity or changed split.")
            identities.add(row["id"])
            group = row["source"]["source_group"]
            data.require(group not in data.RESERVED and registration["source_splits"][group] == split,
                         "Source group violates the registered split.")
            data.require(split_groups.setdefault(group, split) == split, "A source group crosses TRAIN and DEV.")
            data.require(row["accepted_predictions"] == row["acceptable"] and row["primary"] == row["acceptable"][0], "Reference linkage differs.")
            actions = {p["action"] for p in row["acceptable"]}
            data.require(actions == {"no_action"} if row["judgment"]["status"] == "preserve" else "no_action" not in actions,
                         "Preserve and edit references are mixed.")
            key = semantic_input_hash(row["input"])
            if index >= old_count:
                data.require(key not in input_seen, "New admitted semantic duplicate remains.")
            input_seen.setdefault(key, (row["id"], split))
            anchor = (data.session_hash(row["document"]), row["runtime"]["focus_uuid"])
            anchors.add(anchor)
            if "history_length" in row:
                h_cases.append(row["id"])
                h_episodes.add(row["history_length"]["episode_id"])
                bound_evaluation_bundle(row)
            metadata.append({key: row[key] for key in ("id", "split", "language", "source", "primary", "judgment", "accepted_predictions", "lineage")})
            count += 1
        data.require(count == manifest["counts"][split], "Manifest row count differs.")
    data.require(data.sha256(release / "cases/canonical-dev322.jsonl") == data.sha256(BASE / "cases/dev.jsonl"), "Canonical DEV322 bytes differ.")
    actual_supplement = {row["id"] for split in ("train", "dev") for row in iter_rows(release / f"cases/supplemental-{split}.jsonl")}
    data.require(actual_supplement == {r["id"] for r in ledger if r["status"] == "admitted"}, "Admitted ledger and supplemental rows differ.")
    coverage = census(metadata)
    data.require(coverage == read(release / "coverage.json")["all"], "Independent census differs from saved coverage.")
    cells = {(r["split"], r["language"], r["primary"]["action"]) for r in metadata}
    data.require(cells == {(split, language, action) for split in ("train", "dev") for language in ("en", "ko")
                           for action in data.contract.ALLOWED_ACTIONS}, "The unified corpus lost an action-language cell.")
    result = {"version": VERSION, "status": "passed", "release_manifest": bind(release / "manifest.json"),
              "counts": manifest["counts"], "coverage": coverage, "original_prefixes": prefix_verified,
              "source_group_cross_split_collisions": 0, "reserved_source_groups": 0,
              "candidate_dispositions": dict(Counter(r["status"] for r in ledger)),
              "unique_editor_anchors": len(anchors), "history_condition_cases": len(h_cases),
              "history_episodes": len(h_episodes), "history_condition_case_ids": h_cases,
              "model_predictions_consulted": False, "models_loaded": False, "final_test_bodies_read": False}
    data.require(all(result[key] == manifest[key] for key in ("unique_editor_anchors", "history_condition_cases", "history_episodes")),
                 "History conditions and distinct anchor census differ.")
    data.write(output, result)
    return {key: result[key] for key in ("status", "counts", "candidate_dispositions", "unique_editor_anchors", "history_condition_cases", "history_episodes")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "natural", "authored", "authored_repair", "bridge", "history", "history_repair", "history_revision", "terminal", "render", "render_revision", "admit", "evaluation", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT / "candidate-attempt-001")
    parser.add_argument("--reviews", type=Path, nargs="+", default=[])
    parser.add_argument("--release", type=Path, default=DEFAULT / "release-attempt-001")
    parser.add_argument("--cohort", choices=("combined", "canonical", "supplemental", "history"), default="combined")
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.release.resolve(), args.output.resolve())
    elif args.command == "evaluation":
        result = evaluation(args.release.resolve(), args.output.resolve(), args.cohort)
    elif args.command == "admit":
        result = admit(args.output.resolve(), args.reviews, args.release.resolve())
    elif args.command == "terminal":
        result = terminal(args.output.resolve(), args.reviews)
    else:
        result = globals()[args.command](args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
