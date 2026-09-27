"""Re-admit the exposed 289 development states under the fixed seven-action policy.

References stay private and unchanged. A valid legacy reference is not an exhaustive
list of useful seven-action outcomes; conversational no-action ambiguity is recorded
before predictions. This module reads no held-out test body and loads no model.
"""

import argparse
import ast
import copy
import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import passive_actions as actions
import passive_b2_benchmark as old_data
import passive_contract as old_contract
from grading import transcript_media
from passive_dataset import apply_corruptions, episode_steps
from replay_session import replay, session_hash
from session_cases import document_records

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/datasets/nep-passive-completion-v2/frozen"
REGISTRY = ROOT / "ml/passive-models-v2.json"
VERSION = "passive-actions-final-development-preparation-v1"
CUTS = {"delete_span", "trim_silence", "remove_silence"}
PROFILE = actions.CaptionProfile().model_dump()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def outcome_digest(document):
    raw = json.dumps(transcript_media(document), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, values):
    raw = "".join(packed(row) + "\n" for row in values).encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    with path.open("xb") as stream:
        stream.write(raw)


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_closure(entries):
    found, pending = {}, list(entries)
    while pending:
        path = Path(pending.pop()).resolve()
        if path in found:
            continue
        require(path.is_relative_to(ROOT) and path.is_file(), "Local source escaped project.")
        found[path] = sha(path)
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            for name in names:
                relative = Path(*name.split(".")).with_suffix(".py")
                for base in (ROOT / "ml", ROOT / "server"):
                    target = base / relative
                    if target.is_file():
                        pending.append(target)
    return {str(path.relative_to(ROOT)): value for path, value in sorted(found.items())}


class Reader:
    def __init__(self, manifest):
        self.manifest, self.cache, self.bindings = manifest, {}, {}

    def __call__(self, path):
        path = path.resolve()
        require(path.is_relative_to(ROOT), "Execution source escaped project.")
        name = str(path.relative_to(ROOT))
        expected = self.manifest["dependencies_sha256"].get(name)
        require(expected and not any(part in {"sealed", "test-cases.jsonl"} for part in path.parts),
                "Only bound exposed execution evidence is readable.")
        if path not in self.cache:
            require(sha(path) == expected, f"Frozen source changed: {name}")
            self.cache[path] = json.loads(path.read_text())
            self.bindings[name] = expected
        return self.cache[path]


def source_map(documents):
    identities = {item["source"] for doc in documents for item in doc["content"] if "source" in item}
    result = {value: "src-" + hashlib.sha256((VERSION + "\0" + value).encode()).hexdigest()[:24]
              for value in sorted(identities)}
    require(len(set(result.values())) == len(result), "Opaque source identities collide.")
    return result


def project(value, mapping, generated_uuids=None):
    generated_uuids = generated_uuids or {}
    if isinstance(value, list):
        return [project(item, mapping, generated_uuids) for item in value]
    if isinstance(value, dict):
        return {key: mapping.get(item, item) if key in {"source", "source_id"} and isinstance(item, str)
                else generated_uuids.get(item, item) if key == "uuid" and isinstance(item, str)
                else project(item, mapping, generated_uuids) for key, item in value.items()}
    return value


def project_document(document, mapping, generated_uuids=None):
    generated_uuids = generated_uuids or {}
    result = project(document, mapping, generated_uuids)
    inverse = {value: key for key, value in mapping.items()}
    inverse_uuids = {value: key for key, value in generated_uuids.items()}
    require(project(result, inverse, inverse_uuids) == document, "Source projection is not exactly reversible.")
    require([i.get("text") for i in result["content"]] == [i.get("text") for i in document["content"]],
            "Source projection changed transcript text.")
    return result


def projected_history_uuids(history, mapping):
    generated = {}
    for transition in history:
        if transition.get("undo_of"):
            continue
        before, after, prediction = (transition[k] for k in ("before", "after", "prediction"))
        old_ids = {item["uuid"] for item in before["content"]}
        added = [item for item in after["content"] if item["uuid"] not in old_ids]
        if not added:
            continue
        projected_before = project_document(before, mapping, generated)
        key = session_hash({"before": session_hash(projected_before), "edit": prediction})
        if prediction["action"] == "split_paragraph":
            require(len(added) == 2 and {item["type"] for item in added} == {"paragraph_start", "paragraph_end"},
                    "Executed caption split has unexpected generated identities.")
            suffixes = {"paragraph_start": "start", "paragraph_end": "end"}
        elif prediction["action"] == "trim_silence":
            require(len(added) == 1 and added[0]["type"] == "non_text", "Executed trim has unexpected generated identities.")
            suffixes = {"non_text": "right"}
        else:
            raise ValueError("Unexpected fresh UUID in executed history.")
        for item in added:
            replacement = str(uuid5(NAMESPACE_URL, key + suffixes[item["type"]]))
            require(item["uuid"] not in generated or generated[item["uuid"]] == replacement,
                    "An executed UUID has inconsistent projected identity.")
            generated[item["uuid"]] = replacement
        require(len(set(generated.values())) == len(generated) and not set(generated.values()) & (old_ids - set(generated)),
                "Projected executed UUIDs collide.")
    return generated


def upgrade_bundle(old, document):
    records, mapping = actions._records(document)
    ids = [item["id"] for item in old["input"]["items"]]
    snapshot = copy.deepcopy(old["input"])
    snapshot.update(contract_version=actions.VERSION, allowed_actions=list(actions.ALLOWED_ACTIONS),
                    items=[{key: records[index][key] for key in actions.ITEM_FIELDS} for index in ids])
    result = {"input": actions.parse_snapshot(snapshot).model_dump(),
              "mapping": {str(index): mapping[str(index)] for index in ids},
              "document_hash": session_hash(document), "context": copy.deepcopy(old["context"]),
              "caption_authority": copy.deepcopy(old["caption_authority"])}
    result["context"]["policy"]["version"] = actions.VERSION
    result["context"].update(prompt_hash=session_hash(actions.canonical_messages(snapshot)),
                            input_tokens=None, comparison_token_counts={},
                            projection="Historical proposal replay only; no new generation or native-prefix token claim.")
    actions._legacy_bundle(document, result)
    return result


def migrate_feedback(saved, mapping, as_of, generated_uuids=None):
    if saved is None:
        return None
    require(saved["contract_version"] == old_contract.VERSION, "Unknown old feedback ledger.")
    ledger = actions.PassiveFeedbackLedger(saved["document_id"])
    for original in saved["records"]:
        if original["sequence"] > as_of:
            continue
        event = project(copy.deepcopy(original), mapping, generated_uuids)
        kind = event.pop("kind")
        if kind == "propose":
            event["bundle"] = upgrade_bundle(original["bundle"], event["document"])
            ledger.propose(**event)
        elif kind == "record":
            ledger.record(**event)
        else:
            raise ValueError("Unknown committed feedback event.")
    return ledger


def reconstruct(case, reader, episode_cache):
    reproduction = case["reproduction"]
    path = ROOT / reproduction["candidate_directory"] / reproduction["episode"]
    episode = reader(path)
    source = reader(ROOT / episode["source_document"])
    require(sha(ROOT / episode["source_document"]) == episode["source_document_sha256"], "Episode source binding differs.")
    as_of = case["metadata"]["context"]["as_of"]
    if "checkpoints" in episode:
        if path not in episode_cache:
            require(replay(episode["journal"])[0] == episode["final_document"], "Executed feedback journal differs.")
            episode_cache[path] = True
        checkpoint = next(c for c in episode["checkpoints"] if c["case_id"] == case["id"])
        before, old_bundle = checkpoint["before"], checkpoint["bundle"]
        history = [t for t in episode["committed_transitions"] if t["sequence"] <= as_of]
        feedback = episode["feedback_ledger"]
    elif "prepared_context" in episode:
        prepared = reader(ROOT / episode["prepared_context"])
        before, old_bundle, history, feedback = source, prepared["bundle"], [], None
        require(prepared["messages"] == old_contract.canonical_messages(case["input"]), "Prepared original messages differ.")
    else:
        if path not in episode_cache:
            journal = copy.deepcopy(episode["journal"])
            journal[0]["document"] = apply_corruptions(source, episode["spec"].get("corruptions", []))
            replay(journal)
            states, history = {}, []
            before = journal[0]["document"]
            states[session_hash(before)] = (before, [])
            steps = episode_steps(episode["spec"])
            step_index = 0
            for n, event in enumerate(journal):
                if event["kind"] != "transaction":
                    continue
                while steps[step_index]["action"] == "no_action":
                    step_index += 1
                step = steps[step_index]
                step_index += 1
                after = replay(journal[:n + 1])[0]
                _, mapping = document_records(before)
                by_uuid = {item["uuid"]: int(alias) for alias, item in mapping.items()}
                prediction = {k: step[k] for k in ("action", "text", "keep_ms") if k in step}
                prediction["target"] = [by_uuid[u] for u in step["target_uuids"]]
                history.append(old_contract.committed_transition(before, after, prediction, step_index))
                states[session_hash(after)] = (after, copy.deepcopy(history))
                before = after
            episode_cache[path] = states
        before, history = episode_cache[path][case["metadata"]["document_before_hash"]]
        _, mapping = document_records(before)
        old_bundle = {"input": case["input"], "mapping": {str(i["id"]): mapping[str(i["id"])] for i in case["input"]["items"]},
                      "document_hash": session_hash(before), "context": case["metadata"]["context"],
                      "caption_authority": {"version": 1, "protected_uuids": [], "profile": case["input"]["caption_profile"]}}
        feedback = None
    require(session_hash(before) == case["metadata"]["document_before_hash"]
            and old_bundle["input"] == case["input"]
            and old_bundle["context"] == case["metadata"]["context"], "Reconstructed original state changed.")
    _, summaries = old_contract._history(before, history, as_of)
    require(summaries[-3:] == case["input"]["recent_edits"], "Actual committed history does not reproduce frozen summaries.")
    old_ledger = old_contract.PassiveFeedbackLedger.restore(feedback) if feedback else None
    require((old_ledger.context(before, old_bundle["mapping"], as_of, 3) if old_ledger else [])
            == case["input"]["recent_feedback"], "Original past-only feedback differs.")
    old_contract.apply_passive_edit(before, old_bundle, {"action": "no_action"})
    return before, old_bundle, history, feedback


def cut_evidence(snapshot):
    actions.parse_snapshot(snapshot)
    evidence = []
    for item in snapshot["items"]:
        proposal = {"action": "delete_span" if item["type"] == "text" else "remove_silence", "target": [item["id"]]}
        reasons = []
        if not item["source_backed"]:
            reasons.append("no_source_interval")
        if not item["speaker"] or item["language"] in {"", "und"}:
            reasons.append("speaker_or_language_unknown")
        if item["boundary_status"] in {"needs_alignment", "overlap"}:
            reasons.append("conflicting_alignment")
        if item["audio_class"] == "audio_event":
            reasons.append("known_audio_event")
        if item["type"] == "non_text" and item["audio_class"] not in {"unknown", "verified_silence"}:
            reasons.append("gap_contains_known_speech")
        eligible, reason = not reasons, ",".join(reasons) or None
        evidence.append({"alias": item["id"], "action": proposal["action"], "structurally_eligible": eligible,
                         "reason": reason, "text": item["text"], "duration_ms": item["duration_ms"],
                         "boundary_status": item["boundary_status"], "audio_class": item["audio_class"],
                         "speaker": item["speaker"]})
    return evidence


def applicability(case, snapshot, reference_records, evidence, preservation_review):
    positive = case["expectedOutput"]["action"] != "no_action"
    if positive:
        status = "positive_reference_replayed" if any(r["available"] for r in reference_records) else "reference_not_executable_in_current_window"
        rationale = "Existing independently admitted repair remains an accepted outcome where executable; newly possible cut alternatives are not exhaustively labelled."
    else:
        require(preservation_review is not None and preservation_review["case_id"] == case["id"],
                "Every preserve state needs a concrete pre-prediction semantic review.")
        conversational = preservation_review["review"] == "scope_ambiguous"
        eligible = [r for r in evidence if r["structurally_eligible"]]
        if conversational:
            status = "legacy_no_action_cut_scope_ambiguous"
            rationale = ("This actual visible conversational state contains structurally eligible speech/gap intervals. "
                         "Its old review expressly preserved hesitations/unknown gaps under cut-disabled scope; no exhaustive seven-action cut judgment exists. "
                         "Keep its no_action reference and original denominator, but do not certify it as a full-action preservation negative.")
        else:
            status = "no_action_readmitted"
            if not eligible:
                rationale = "No visible target permits a cut under actual speaker/timing/audio constraints; the unchanged four-action preserve reference remains applicable."
            elif len(snapshot["items"]) == 1 and snapshot["items"][0]["type"] == "text":
                rationale = "Only the complete meaningful utterance is addressable; deleting it would remove its factual proposition. No gap or dispensable isolated subspan is exposed."
            else:
                rationale = ("Visible authored wording preserves narrative uncertainty, deliberate repetition, quoted text, response meaning or executed feedback. "
                             "The separate complete utterances and ordinary pauses show no clear pacing benefit; duration alone does not justify removing them. "
                             "Reference remains no_action; this is a textual policy assessment, not acoustic verification.")
            old_items = [{k: i[k] for k in old_contract.ITEM_FIELDS} for i in case["input"]["items"]]
            new_items = [{k: i[k] for k in old_contract.ITEM_FIELDS} for i in snapshot["items"]]
            if preservation_review["review"] == "preservation_applicable_if_current_cut_constraints_unchanged":
                require(not eligible, "Unknown-speaker preservation premise changed.")
            else:
                require(old_items == new_items, "Reviewed preserve window changed and needs a new semantic review.")
                require(snapshot["recent_feedback"] == case["input"]["recent_feedback"],
                        "Reviewed preserve feedback changed after projection.")
    return {"status": status, "original_action": case["expectedOutput"]["action"], "rationale": rationale,
            "reference_altered": False, "reference_exhaustive_for_new_cuts": False,
            "full_action_preservation_admitted": status == "no_action_readmitted",
            "structurally_eligible_cut_targets": sum(row["structurally_eligible"] for row in evidence),
            "cut_evidence": evidence, "original_review": case["metadata"].get("review_reason"),
            "pre_prediction_semantic_review": preservation_review,
            "current_visible_aliases": [item["id"] for item in snapshot["items"]]}


def tokenizer():
    from transformers import AutoTokenizer

    registry = json.loads(REGISTRY.read_text())
    candidate = next(c for c in registry["candidates"] if c["id"] == "gemma26")
    path = ROOT / candidate["tokenizer_path"]
    for name, value in candidate["tokenizer_files_sha256"].items():
        require(sha(path / name) == value, "Pinned native tokenizer changed.")
    value = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    require(value.eos_token_id == 1, "Gemma native tokenizer identity changed.")
    return value, {"registry": record(REGISTRY), "path": candidate["tokenizer_path"],
                   "files_sha256": candidate["tokenizer_files_sha256"], "eos_token_id": 1,
                   "weights_loaded": False}


def native_readback(payload):
    require(payload["source_sha256"] == sha(Path(__file__)), "Native readback source changed.")
    path = Path(payload["manifest"]["path"])
    require(path.resolve() == path and path.is_relative_to(ROOT) and sha(path) == payload["manifest"]["sha256"],
            "Native readback preparation binding differs.")
    manifest = json.loads(path.read_text())
    native_path = path.parent / "native-prefixes.jsonl.gz"
    require(manifest["version"] == VERSION and sha(native_path) == manifest["files_sha256"][native_path.name],
            "Native readback payload changed.")
    native = rows(native_path)
    require([r["id"] for r in native] == manifest["case_ids"] == [r["id"] for r in payload["sampled"]]
            and len(native) == 289, "Native readback needs every registered state in order.")
    value, identity = tokenizer()
    results = []
    for row, sampled in zip(native, payload["sampled"], strict=True):
        ids = sampled["ids"]
        require(ids is None or (isinstance(ids, list) and 0 < len(ids) <= 512
                               and all(type(i) is int and 0 <= i < 262144 for i in ids)), "Invalid sampled token IDs.")
        content_ids = ids[:-1] if ids and ids[-1] in (1, 106, 50) else ids
        rendered = value.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        require(rendered.startswith("<bos>"), "Native BOS representation differs.")
        results.append({"id": row["id"], "prefix": value.apply_chat_template(row["messages"], tokenize=True, return_dict=False,
                                                                            add_generation_prompt=True, enable_thinking=False),
                        "rendered_prompt": rendered[len("<bos>"):],
                        "decoded": value.decode(content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        if content_ids is not None else None})
    return {"status": "native_tokenizer_readback_complete", "model_loaded": False, "tokenizer": identity, "cases": results}


def prepare(output):
    require(not output.exists(), "Use a fresh preparation directory.")
    source_files = source_closure([Path(__file__), actions.PROMPT_PATH, ROOT / "ml/prompt-passive-v1.txt"])
    cases, manifest = old_data._load_split(RELEASE, "dev")
    require(len(cases) == 289, "Original full cohort differs.")
    matched_path = RELEASE / "matched/dev-cases.jsonl"
    require(sha(matched_path) == manifest["files_sha256"]["matched/dev-cases.jsonl"], "Matched subset changed.")
    matched = {c["id"] for c in rows(matched_path)}
    require(len(matched) == 286 and matched <= {c["id"] for c in cases}, "Prior matched286 membership differs.")
    review_path = ROOT / "artifacts/benchmarks/passive-actions-final-v1/root-review/original-preservation-semantics-attempt-01.json"
    require(sha(review_path) == "865ced5f2e584a7987ecce4718e206b10435a7ad5b5472d621549eb658bed1c7", "Original preserve review changed.")
    review = json.loads(review_path.read_text())
    require(review["source_sha256"] == sha(RELEASE / "dev-cases.jsonl")
            and review["no_model_predictions_for_new_study"] is True, "Semantic review binds different inputs or observed predictions.")
    preservation = {r["case_id"]: r for r in review["cases"]}
    require(set(preservation) == {c["id"] for c in cases if c["expectedOutput"]["action"] == "no_action"}, "Preservation review does not cover every original state.")
    native, native_identity = tokenizer()
    builder = actions.PassiveContextBuilder(native)
    reader, cache = Reader(manifest), {}
    prepared, runtime, encoded, audit, source_maps = [], [], [], [], []
    output.mkdir(parents=True)
    for index, case in enumerate(cases):
        before, old_bundle, history, feedback = reconstruct(case, reader, cache)
        documents = [before, *[doc for t in history for doc in (t["before"], t["after"])]]
        if feedback:
            documents += [r["document"] for r in feedback["records"] if r["kind"] == "propose"]
        mapping = source_map(documents)
        generated_uuids = projected_history_uuids(history, mapping)
        document = project_document(before, mapping, generated_uuids)
        new_history = project(history, mapping, generated_uuids)
        sequence = old_bundle["context"]["as_of"]
        ledger = migrate_feedback(feedback, mapping, sequence, generated_uuids)
        _, actual_mapping = actions._records(document)
        focus = actual_mapping[str(case["input"]["cursor_before"])]["uuid"]
        selection = [actual_mapping[str(i)]["uuid"] for i in case["input"]["selection"]]
        require(case["input"]["caption_profile"] == PROFILE, "A case differs from deployed default caption profile.")
        bundle = builder.build(document, focus, as_of=sequence, selection=selection, history=new_history, feedback=ledger)
        require(bundle["input"]["recent_edits"] == case["input"]["recent_edits"], "Projected history changed its observable content.")
        refs = [case["expectedOutput"], *case.get("acceptableOutputs", [])]
        reference_records = []
        for prediction in refs:
            old_after = old_contract.apply_passive_edit(before, old_bundle, prediction)
            projected_after = project_document(old_after, mapping, generated_uuids)
            entry = {"prediction": prediction, "outcome_sha256": outcome_digest(projected_after)}
            try:
                new_after = actions.apply_passive_edit(document, bundle, prediction)
            except ValueError as error:
                entry.update(available=False, reason=str(error))
            else:
                require(transcript_media(new_after) == transcript_media(projected_after), "Reference source/caption outcome changed.")
                entry.update(available=True, reason=None)
            reference_records.append(entry)
        assessment = applicability(case, bundle["input"], reference_records, cut_evidence(bundle["input"]), preservation.get(case["id"]))
        messages = actions.canonical_messages(bundle["input"])
        prefix = actions.token_ids(native, bundle["input"])
        require(len(prefix) <= 3584, "Fixed native input budget exceeded.")
        rendered = native.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        require(native.bos_token == "<bos>" and native.bos_token_id == 2 and rendered.startswith("<bos>"), "Native BOS representation changed.")
        current = {"id": case["id"], "split": "dev", "language": case["metadata"]["language"],
                   "primary": copy.deepcopy(case["expectedOutput"]), "accepted_predictions": copy.deepcopy(refs),
                   "reference_outcomes": reference_records, "input": bundle["input"], "bundle": bundle,
                   "applicability": assessment,
                   "provenance": {**{k: case["metadata"][k] for k in ("source_group_id", "source_id", "episode_id", "kind", "media_kind")},
                                  "original_case_sha256": digest(case), "original_document_sha256": session_hash(before),
                                  "projected_document_sha256": session_hash(document), "prior_matched286": case["id"] in matched}}
        prepared.append(current)
        runtime.append({"id": case["id"], "split": "dev", "document": document, "focus_uuid": focus,
                        "selection": selection, "sequence": sequence, "history": new_history,
                        "feedback_journal": ledger.dump() if ledger else None, "caption_profile": PROFILE})
        encoded.append({"id": case["id"], "split": "dev", "messages": messages,
                        "rendered_prompt": rendered[len("<bos>"):],
                        "prefix_ids_sha256": digest(prefix), "primary": {"input_ids": prefix, "completion_start": len(prefix)}})
        audit.append({"id": case["id"], "prior_matched286": case["id"] in matched, **assessment,
                      "references": reference_records, "old_visible_aliases": [i["id"] for i in case["input"]["items"]],
                      "new_visible_aliases": [i["id"] for i in bundle["input"]["items"]], "prefix_tokens": len(prefix),
                      "original_feedback": case["input"]["recent_feedback"], "new_feedback": bundle["input"]["recent_feedback"]})
        source_maps.append({"id": case["id"], "mapping": mapping, "executed_generated_uuid_mapping": generated_uuids})
        if (index + 1) % 25 == 0:
            print(f"Prepared {index + 1}/289 reference-audited states; no model predictions", flush=True)
    for path, value in source_files.items():
        require(sha(ROOT / path) == value, "Preparation dependency changed during construction.")
    write_rows(output / "cases.jsonl.gz", prepared)
    write_rows(output / "runtime-inputs.jsonl.gz", runtime)
    write_rows(output / "native-prefixes.jsonl.gz", encoded)
    write_rows(output / "applicability.jsonl", audit)
    write_rows(output / "private-source-mapping.jsonl", source_maps)
    write(output / "tokenizer.json", native_identity)
    write(output / "source-bindings.json", reader.bindings)
    census = {"registered_original": 289, "prior_matched_subset": 286, "prepared_runtime_states": len(prepared),
              "reference_actions": dict(Counter(c["primary"]["action"] for c in prepared)),
              "languages": dict(Counter(c["language"] for c in prepared)),
              "applicability": dict(Counter(c["applicability"]["status"] for c in prepared)),
              "primary_reference_unavailable": sum(not c["reference_outcomes"][0]["available"] for c in prepared),
              "changed_visible_windows": sum(r["old_visible_aliases"] != r["new_visible_aliases"] for r in audit),
              "max_prefix_tokens": max(len(r["primary"]["input_ids"]) for r in encoded),
              "model_requests": 0, "training_updates": 0, "final_test_bodies_read": 0}
    write(output / "census.json", census)
    write(output / "manifest.json", {"version": VERSION, "status": "prepared_pending_independent_reference_review",
          "created_at": datetime.now(timezone.utc).isoformat(), "contract_version": actions.VERSION,
          "original_release": record(RELEASE / "manifest.json"), "original_dev": record(RELEASE / "dev-cases.jsonl"),
          "original_matched_dev": record(matched_path), "case_ids": [c["id"] for c in prepared],
          "original_preservation_review": record(review_path),
          "files_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
          "sources_sha256": source_files, "census": census,
          "scope": "All289 exposed original states, current seven-action deployed context budget; no ICL, no labels changed, no future/test inputs. Source IDs are bijectively pseudonymized to avoid scenario-label leakage. Executed generated marker UUIDs are reversibly mapped using their unchanged UUID5 rule over the projected document hash; text/timing/aliases and transaction identities are retained.",
          "rendering": "Native prefix IDs retain BOS2; server /apply-template rendering omits exactly one leading <bos> literal, as verified in the completed caption study. No other normalization.",
          "grading": "Accepted executed source/caption outcome equivalence; legacy-reference adherence reported on original roster; expanded-policy ambiguous preservation stays explicit and is not declared a certified negative."})
    return census


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--native-readback", action="store_true")
    args = parser.parse_args()
    result = native_readback(json.load(sys.stdin)) if args.native_readback else prepare(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
