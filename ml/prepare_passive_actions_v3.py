"""Prepare replayable seven-action cases without touching historical releases.

The admission ledger covers every exposed legacy row, including unknown outcomes.
Only current-input-qualified references enter cases/. Natural acoustic evidence and
authored symbolic timing remain separate dimensions throughout the release.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import passive_actions as contract
import prepare_passive_actions_final as legacy
import validation_cache_actions_v3 as validation
from passive_dataset import RESERVED
from replay_session import replay, session_hash
from session_cases import replayable_session
from verify_release_export import verify_operation

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-actions-v3-preparation-1"
FIXTURES = ROOT / "ml/fixtures/passive-actions-v3"
OLD = ROOT / "artifacts/datasets/nep-passive-completion-v2/frozen"
PILOT = ROOT / "artifacts/datasets/nep-passive-actions-v1/pilot-draft-attempt-02"
PROFILE = contract.CaptionProfile().model_dump()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for value in values:
            stream.write(packed(value) + "\n")


def authored_recipes():
    latest = FIXTURES / "revision-005"
    return sorted((latest if latest.is_dir() else FIXTURES).glob("authored-*.json"))


def source_splits():
    result = json.loads((OLD / "source-groups.json").read_text())
    for split in ("train", "dev"):
        for case in rows(PILOT / "cases" / f"{split}.jsonl"):
            result[case["source"]["source_group"]] = split
    for path in authored_recipes():
        for group in json.loads(path.read_text())["source_groups"]:
            previous = result.setdefault(group["id"], group["split"])
            require(previous == group["split"], "A source group crosses splits.")
    require(result["shuka-local-recording"] == "dev", "Shuka must remain development.")
    require(not RESERVED.intersection(result), "Reserved final-test groups entered preparation.")
    return result


def runtime_bundle(case, builder):
    runtime = case["runtime"]
    ledger = runtime.get("feedback_journal")
    return builder.build(
        case["document"], runtime["focus_uuid"], as_of=runtime["sequence"],
        selection=runtime["selection"], history=runtime["history"],
        feedback=contract.PassiveFeedbackLedger.restore(ledger) if ledger else None,
        caption_profile=runtime["caption_profile"],
    )


def execution(document, bundle, prediction, source):
    after = contract.apply_passive_edit(document, bundle, prediction)
    proof = {
        "prediction": prediction, "before_sha256": session_hash(document),
        "after_sha256": session_hash(after), "changed": document != after,
        "source_integrity": "verified_exact_operation_and_outside_intervals",
    }
    if prediction["action"] == "no_action":
        require(document == after, "Abstention changed the document.")
        proof.update(journal=[], replay=None)
    else:
        require(document != after, "Reference edit did not change the document.")
        verify_operation(document, after, bundle["mapping"], prediction)
        journal = replayable_session(document, after, prediction["action"], source)
        restored, summary = replay(journal)
        require(restored == after and summary["transactions"] == 3, "Apply/undo/redo failed.")
        require(journal[2]["after_hash"] == session_hash(document), "Undo did not restore before state.")
        # The row already stores the document; hydration avoids another full copy.
        journal[0].pop("document")
        proof.update(journal=journal, replay=summary, hydrate_start_document_from="case.document")
    return proof


def validate_case(case, builder, *, bundle=None):
    require(case["split"] in ("train", "dev"), "Only exposed train/development cases are allowed.")
    require(case["source"]["split"] == case["split"], "Source split changed.")
    require(case["source"]["source_group"] not in RESERVED, "Reserved source group.")
    require(case["judgment"]["status"] in ("edit_required", "preserve"), "Unknown is not no_action.")
    bundle = bundle if bundle is not None else runtime_bundle(case, builder)
    require(bundle["input"] == case["input"], "Saved input differs from actual runtime preparation.")
    require(contract.canonical_messages(bundle["input"]) == case["messages"], "Prompt changed.")
    references = case["accepted_predictions"]
    actions = {prediction["action"] for prediction in references}
    require((actions == {"no_action"}) if case["judgment"]["status"] == "preserve" else "no_action" not in actions,
            "Mixed necessity cannot be a supervised reference.")
    require(references == case["acceptable"] and references[0] == case["primary"], "Reference list must include primary first.")
    require(len({packed(r) for r in references}) == len(references), "Duplicate accepted prediction.")
    ledger = case["runtime"].get("feedback_journal")
    ledger = contract.PassiveFeedbackLedger.restore(ledger) if ledger else None
    for prediction in references:
        if ledger and prediction["action"] != "no_action":
            require(not ledger.suppressed(case["document"], bundle, prediction, case["runtime"]["sequence"]), "Reference repeats dismissed/undone proposal.")
        contract.apply_passive_edit(case["document"], bundle, prediction)
    proof = execution(case["document"], bundle, case["primary"], case["source"])
    if "execution_sha256" in case:
        require(case["execution_sha256"] == digest(proof), "Execution proof changed.")
    return proof


def validated_proof(case, builder, proof_cache=None, *, bundle=None):
    def validator(row):
        return validate_case(row, builder, bundle=bundle)
    return validator(case) if proof_cache is None else proof_cache.validate(case, validator)


def case_row(identity, split, language, source, document, bundle, primary, acceptable,
             runtime, rationale, lineage, acoustic=None):
    refs = []
    for prediction in [primary, *acceptable]:
        if prediction not in refs:
            refs.append(prediction)
    result = {
        "id": identity, "schema_version": VERSION, "split": split, "language": language,
        "source": source, "document": document, "input": bundle["input"],
        "primary": primary, "acceptable": refs, "accepted_predictions": refs,
        "messages": contract.canonical_messages(bundle["input"]), "runtime": runtime,
        "judgment": {"status": "preserve" if primary["action"] == "no_action" else "edit_required",
                     "show_now": primary["action"] != "no_action", "actor": "synthetic",
                     "observed_action": False, "acoustic_evidence": acoustic or "none",
                     "reference_origin": "ai_authored_input_qualification",
                     "independent_review": "pending_parent_review"},
        "rationale": rationale, "lineage": lineage,
    }
    return result


def resolve(prediction, keys, bundle):
    result = copy.deepcopy(prediction)
    if "targets" in result:
        by_uuid = {item["uuid"]: int(alias) for alias, item in bundle["mapping"].items()}
        result["target"] = [by_uuid[keys[key]] for key in result.pop("targets")]
    return result


def authored_document(spec):
    source_id = "symbolic:" + str(uuid5(NAMESPACE_URL, VERSION + "/" + spec["id"]))
    content, keys, intervals, elapsed = [], {}, {}, 0
    def identity(key):
        return str(uuid5(NAMESPACE_URL, source_id + "/" + key))
    for index, paragraph in enumerate(spec["paragraphs"]):
        items = paragraph["items"] if isinstance(paragraph, dict) else paragraph
        language = paragraph.get("language", spec["language"]) if isinstance(paragraph, dict) else spec["language"]
        speaker = paragraph.get("speaker", "narrator") if isinstance(paragraph, dict) else "narrator"
        content.append({"type": "paragraph_start", "uuid": identity(f"p{index}"), "speaker": speaker, "language": language})
        for item in items:
            key, duration = item["key"], item["ms"]
            require(key not in keys and type(duration) is int and duration > 0, "Invalid authored item.")
            keys[key] = identity(key)
            intervals[key] = (elapsed / 1000, (elapsed + duration) / 1000)
            value = {"type": "text" if item.get("text") is not None else "non_text", "uuid": keys[key],
                     "source": source_id, "sourceStart": elapsed / 1000, "length": duration / 1000}
            if item.get("text") is not None:
                value.update(text=item["text"], conf=1)
            content.append(value)
            elapsed += duration
        content.append({"type": "paragraph_end", "uuid": identity(f"e{index}")})
    metadata = {"provenance": "authored_symbolic_timing_not_audio"}
    if spec.get("annotations"):
        metadata["transcriptions"] = [{"source": source_id, "provenance": "synthetic_control_annotation_not_audio_observation",
            "annotations": [{"type": kind, "start": intervals[key][0], "end": intervals[key][1]}
                            for key, kind in spec["annotations"]]}]
    return {"content": content, "metadata": metadata}, keys, source_id


def build_authored_cases(builder, proof_cache=None):
    cases, excluded = [], []
    for path in authored_recipes():
        recipe = json.loads(path.read_text())
        for spec in recipe["scenarios"]:
            document, keys, source_id = authored_document(spec)
            history, sequence = [], 0
            for prior in spec.get("history", []):
                before = copy.deepcopy(document)
                bundle = builder.build(before, keys[prior["targets"][0]], as_of=sequence, history=history, caption_profile=PROFILE)
                prediction = resolve(prior, keys, bundle)
                document = contract.apply_passive_edit(before, bundle, prediction)
                sequence += 1
                history.append(contract.committed_transition(before, document, prediction, sequence))
            ledger = contract.PassiveFeedbackLedger(spec["id"])
            for index, feedback in enumerate(spec.get("feedback", [])):
                before_bundle = builder.build(document, keys[spec["focus"]], as_of=sequence, history=history, feedback=ledger, caption_profile=PROFILE)
                prediction = resolve(feedback["prediction"], keys, before_bundle)
                proposal_id = f"{spec['id']}-proposal-{index}"
                ledger.propose(proposal_id, sequence, document, before_bundle, prediction)
                sequence += 1
                ledger.record(proposal_id, "shown", sequence, origin="synthetic")
                sequence += 1
                ledger.record(proposal_id, feedback["outcome"], sequence, origin="synthetic", reason=feedback.get("reason", "unspecified"))
            bundle = builder.build(document, keys[spec["focus"]], as_of=sequence,
                                   history=history, feedback=ledger, caption_profile=PROFILE)
            source = {"id": source_id, "source_group": spec["source_group"], "split": spec["split"],
                      "language": spec["language"], "provenance": "authored_symbolic_timing_not_audio",
                      "natural_audio": False, "generated_audio": False, "acoustically_verified": False}
            if spec.get("judgment_status") == "unknown":
                excluded.append({"id": spec["id"], "split": spec["split"], "source": source,
                                 "status": "deferred", "judgment_status": "unknown", "reason": spec["rationale"],
                                 "input": bundle["input"], "recipe_sha256": sha256(path)})
                continue
            primary = resolve(spec["primary"], keys, bundle)
            acceptable = [resolve(p, keys, bundle) for p in spec.get("acceptable", [])]
            runtime = {"focus_uuid": keys[spec["focus"]], "selection": [], "sequence": sequence,
                       "history": history, "feedback_journal": ledger.dump() if spec.get("feedback") else None,
                       "caption_profile": PROFILE}
            row = case_row(spec["id"], spec["split"], spec["language"], source, document, bundle,
                           primary, acceptable, runtime, spec["rationale"],
                           {"recipe": str(path.relative_to(ROOT)), "recipe_sha256": sha256(path),
                            "scenario_family": spec["family"], "selection_method": "authored_challenge",
                            "cursor_selection_before_judgment": False, "old_id": None,
                            "generation_provenance": recipe["provenance"],
                            "contrast_group": spec.get("contrast_group"), "context": bundle["context"]})
            proof = validated_proof(row, builder, proof_cache, bundle=bundle)
            row["execution_sha256"] = digest(proof)
            cases.append(row)
    return cases, excluded


def preservation_status(snapshot, old):
    eligible = [r for r in legacy.cut_evidence(snapshot) if r["structurally_eligible"]]
    if not eligible:
        return "adopted", "Current visible timing/speaker/event constraints allow no cut; legacy text/caption preservation remains applicable."
    if len(snapshot["items"]) == 1 and snapshot["items"][0]["type"] == "text":
        return "adopted", "Only a complete meaningful utterance is addressable; deleting the whole item would lose its proposition."
    if old["metadata"]["media_kind"] != "recording" and old["metadata"]["kind"] in {
        "authored_semantic_challenge", "authored_clean_history_control", "authored_feedback_session"
    }:
        return "adopted", "Authored preservation is supported by the observable complete wording/history; no new natural acoustic claim. " + old["metadata"].get("review_reason", "")
    return "deferred", "Legacy no_action is not exhaustive under seven actions; no current-input semantic/acoustic review establishes whole-window preservation."


def migrate_historical(builder, proof_cache=None):
    manifest = json.loads((OLD / "manifest.json").read_text())
    reader, cache, cases, migration, excluded = legacy.Reader(manifest), {}, [], [], []
    splits = source_splits()
    for split, expected in (("train", 809), ("dev", 289)):
        path = OLD / f"{split}-cases.jsonl"
        require(sha256(path) == manifest["files_sha256"][path.name], "Historical release bytes changed.")
        original = rows(path)
        require(len(original) == expected, "Historical count changed.")
        for old in original:
            identity = "v3-legacy-" + old["id"]
            meta = old["metadata"]
            record = {"old_release": "completion-v2", "old_id": old["id"], "old_split": split,
                      "source_group": meta["source_group_id"], "old_action": old["expectedOutput"]["action"],
                      "old_row_sha256": digest(old), "new_id": None, "status": None}
            require(splits[meta["source_group_id"]] == split == meta["split"], "Legacy source split changed.")
            before, old_bundle, history, feedback = legacy.reconstruct(old, reader, cache)
            documents = [before, *[d for t in history for d in (t["before"], t["after"])]]
            if feedback:
                documents += [r["document"] for r in feedback["records"] if r["kind"] == "propose"]
            mapping = legacy.source_map(documents)
            uuids = legacy.projected_history_uuids(history, mapping)
            document = legacy.project_document(before, mapping, uuids)
            new_history = legacy.project(history, mapping, uuids)
            sequence = old_bundle["context"]["as_of"]
            ledger = legacy.migrate_feedback(feedback, mapping, sequence, uuids)
            _, actual_mapping = contract._records(document)
            focus = actual_mapping[str(old["input"]["cursor_before"] )]["uuid"]
            selection = [actual_mapping[str(alias)]["uuid"] for alias in old["input"]["selection"]]
            bundle = builder.build(document, focus, as_of=sequence, selection=selection,
                                   history=new_history, feedback=ledger, caption_profile=PROFILE)
            # The current window is fixed before old reference availability is inspected.
            record["current_input_sha256"] = digest(bundle["input"])
            valid, invalid = [], []
            for prediction in [old["expectedOutput"], *old.get("acceptableOutputs", [])]:
                try:
                    contract.apply_passive_edit(document, bundle, prediction)
                    if ledger and prediction["action"] != "no_action":
                        require(not ledger.suppressed(document, bundle, prediction, sequence), "Equivalent past dismissal/undo.")
                except (ValueError, KeyError) as error:
                    invalid.append({"prediction": prediction, "reason": str(error)})
                else:
                    if prediction not in valid:
                        valid.append(prediction)
            record["unavailable_references"] = invalid
            if not valid:
                record.update(status="excluded", judgment_status="inapplicable", reason="No old reference is executable in the independently prepared current runtime window.")
            elif old["expectedOutput"]["action"] == "no_action":
                status, reason = preservation_status(bundle["input"], old)
                record.update(status=status, judgment_status="preserve" if status == "adopted" else "unknown", reason=reason)
            else:
                record.update(status="revised" if invalid or valid[0] != old["expectedOutput"] else "adopted",
                              judgment_status="edit_required", reason="Existing input-reviewed useful repair is executable under the seven-action window; only available alternatives are retained.")
            if record["status"] in ("adopted", "revised"):
                source = {"id": meta["source_id"], "source_group": meta["source_group_id"], "split": split,
                          "language": meta["language"], "provenance": meta["kind"],
                          "natural_audio": meta["media_kind"] == "recording", "generated_audio": False,
                          "acoustically_verified": False, "source_region": meta.get("source_region")}
                runtime = {"focus_uuid": focus, "selection": selection, "sequence": sequence, "history": new_history,
                           "feedback_journal": ledger.dump() if ledger else None, "caption_profile": PROFILE}
                lineage = {"old_id": old["id"], "old_release": "completion-v2", "old_row_sha256": digest(old),
                           "old_reproduction": old["reproduction"], "source_identity_projection": mapping,
                           "generated_uuid_projection": uuids, "scenario_family": meta["kind"],
                           "selection_method": "source_first" if meta["kind"] == "reference_independent_source_checkpoint" else "legacy_authored_cursor",
                           "cursor_selection_before_judgment": meta["kind"] == "reference_independent_source_checkpoint",
                           "context": bundle["context"], "original_review": meta.get("review_file"),
                           "review_reason": meta.get("review_reason")}
                row = case_row(identity, split, meta["language"], source, document, bundle, valid[0], valid[1:], runtime,
                               meta.get("review_reason", meta.get("decision_rationale", record["reason"])), lineage)
                proof = validated_proof(row, builder, proof_cache, bundle=bundle)
                row["execution_sha256"] = digest(proof)
                record["new_id"] = identity
                cases.append(row)
            else:
                excluded.append({"id": identity, "split": split, **record})
            migration.append(record)
            if len(migration) % 100 == 0:
                print(f"Migrated {len(migration)}/1154 candidates; admitted {len(cases)}, excluded/deferred {len(excluded)}", flush=True)
    # This pilot is already an explicit seven-action distribution, not legacy no_action relabeling.
    runtimes = {r["id"]: r for r in rows(PILOT / "runtime-inputs.jsonl")}
    for split, expected in (("train", 42), ("dev", 14)):
        old_cases = rows(PILOT / "cases" / f"{split}.jsonl")
        require(len(old_cases) == expected, "Historical seven-action pilot count changed.")
        for old in old_cases:
            runtime = copy.deepcopy(runtimes[old["id"]])
            document = runtime.pop("document")
            runtime.pop("id")
            runtime.pop("split")
            row = {"document": document, "runtime": runtime}
            bundle = runtime_bundle(row, builder)
            primary = old["primary"]
            row = case_row("v3-pilot-" + old["id"], split, old["language"], old["source"], document, bundle,
                           primary, old["accepted_predictions"], runtime, old["rationale"],
                           {"old_id": old["id"], "old_release": "actions-v1-pilot-attempt-02", "old_row_sha256": digest(old),
                            "scenario_family": "inherited_symbolic_seven_action_pilot", "selection_method": "legacy_authored_cursor",
                            "cursor_selection_before_judgment": False, "context": bundle["context"]})
            proof = validated_proof(row, builder, proof_cache, bundle=bundle)
            row["execution_sha256"] = digest(proof)
            cases.append(row)
            migration.append({"old_release": "actions-v1-pilot-attempt-02", "old_id": old["id"], "old_split": split,
                              "source_group": old["source"]["source_group"], "old_action": primary["action"],
                              "old_row_sha256": digest(old), "new_id": row["id"], "status": "adopted",
                              "judgment_status": row["judgment"]["status"],
                              "reason": "Explicit symbolic seven-action reference re-executed under the same preview policy; no natural acoustic claim."})
    return cases, migration, excluded, reader.bindings


def coverage(cases, migration, excluded):
    cells, groups = defaultdict(Counter), defaultdict(set)
    for case in cases:
        source = case["source"]
        kind = "natural" if source.get("natural_audio") else "generated_audio" if source.get("generated_audio") else "symbolic"
        cells[(case["split"], case["language"], kind)][case["primary"]["action"]] += 1
        groups[case["split"]].add(source["source_group"])
    require(not groups["train"].intersection(groups["dev"]), "Train/dev source leakage.")
    return {"counts": dict(Counter(c["split"] for c in cases)),
            "cells": [{"split": s, "language": lang, "source_kind": k, "actions": dict(count)} for (s, lang, k), count in sorted(cells.items())],
            "source_groups": {s: sorted(g) for s, g in groups.items()},
            "migration": dict(Counter(r["status"] for r in migration)),
            "exclusions": dict(Counter(r.get("judgment_status", "unknown") for r in excluded)),
            "source_first": dict(Counter(c["split"] for c in cases if c["lineage"]["selection_method"] == "source_first")),
            "long_context_24plus_visible_items": dict(Counter(c["split"] for c in cases if len(c["input"]["items"]) >= 24)),
            "history_visible": sum(bool(c["input"]["recent_edits"]) for c in cases),
            "feedback_visible": sum(bool(c["input"]["recent_feedback"]) for c in cases),
            "natural_acoustic_gap": "No qualified natural remove_silence reference, no natural KO deletion, and no EN development cut certificate is available in the prior evidence.",
            "final_test_bodies_read": 0, "model_predictions_consulted": False}


def prepare(output, *, authored_only=False):
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh attempt directory; previous revisions are immutable.")
    tokenizer, tokenizer_identity = legacy.tokenizer()
    builder = contract.PassiveContextBuilder(tokenizer)
    implementation = {"preparation_sha256": sha256(Path(__file__)), "contract_sha256": sha256(Path(contract.__file__)),
                      "cache_sha256": sha256(Path(validation.__file__)), "prompt_sha256": sha256(contract.PROMPT_PATH),
                      "tokenizer": tokenizer_identity}
    write(output / "preparation-registration.json", {"implementation": implementation,
          "authored_recipes_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in authored_recipes()},
          "historical_candidates": 0 if authored_only else 1154, "stage": "registered_before_materialization"})
    proof_cache = validation.ReferenceProofCache(output / "validation-cache", implementation)
    cases, excluded = build_authored_cases(builder, proof_cache)
    print(f"Prepared {len(cases)} authored cases and {len(excluded)} explicit unknowns", flush=True)
    migration, bindings = [], {}
    if not authored_only:
        inherited, migration, old_exclusions, bindings = migrate_historical(builder, proof_cache)
        cases = inherited + cases
        excluded = old_exclusions + excluded
    require(len({c["id"] for c in cases}) == len(cases), "Duplicate case ID.")
    splits = source_splits()
    for case in cases:
        require(splits[case["source"]["source_group"]] == case["split"], "Case differs from registered source split.")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        write_rows(output / "cases" / f"{split}.jsonl", [c for c in cases if c["split"] == split])
    write_rows(output / "runtime-inputs.jsonl", [{"id": c["id"], "split": c["split"], "document_sha256": session_hash(c["document"]),
                                                "hydrate_document_from": "case.document", **c["runtime"]} for c in cases])
    write_rows(output / "reference-executions.jsonl", ({"id": c["id"], **validated_proof(c, builder, proof_cache)} for c in cases))
    write_rows(output / "migration.jsonl", migration)
    write_rows(output / "exclusions.jsonl", excluded)
    write(output / "source-splits.json", splits)
    write(output / "source-bindings.json", bindings)
    write(output / "tokenizer.json", tokenizer_identity)
    report = coverage(cases, migration, excluded)
    write(output / "coverage.json", report)
    files = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = {"version": VERSION, "status": "prepared_pending_parent_semantic_review", "contract_version": contract.VERSION,
                "case_schema": "self-contained document/input/primary/acceptable/runtime; acceptable and accepted_predictions include primary first",
                "counts": report["counts"], "files_sha256": files,
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), Path(validation.__file__), contract.PROMPT_PATH, Path(contract.__file__), *authored_recipes()]},
                "validation_cache": {"hits": proof_cache.hits, "misses": proof_cache.misses,
                                     "binding": "Actual input, document, runtime, source, judgment, references, code/prompt and native tokenizer"},
                "historical_releases_unchanged": {str(p.relative_to(ROOT)): sha256(p) for p in [OLD / "manifest.json", PILOT / "manifest.json"]},
                "scope": "Exposed train/development only. AI next-action references are not observed human preferences. Unknown is excluded, never no_action.",
                "coverage": report}
    write(output / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authored-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, authored_only=args.authored_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
