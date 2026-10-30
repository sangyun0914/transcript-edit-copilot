"""Requalify exact pre-existing natural-audio operations for the passive policy."""

import argparse
import json
from pathlib import Path

import prepare_passive_actions_v3 as dataset
from reviewed_audio import bind_observation, verify_media_edit

QUALIFICATIONS = dataset.ROOT / "artifacts/datasets/nep-release-v1/reviewed-audio-v4/qualifications.json"


def prepare(output):
    output = Path(output).resolve()
    dataset.require(not output.exists(), "Use a fresh natural-evidence attempt directory.")
    output.mkdir(parents=True)
    registry = json.loads((dataset.ROOT / "data/manifests/passive-existing-sources-v1.json").read_text())
    sources = {source["id"]: source for source in registry["sources"]}
    qualified = json.loads(QUALIFICATIONS.read_text())
    for name, expected in qualified["input_sha256"].items():
        dataset.require(dataset.sha256(dataset.ROOT / name) == expected, "Acoustic qualification input changed.")
    tokenizer, tokenizer_identity = dataset.legacy.tokenizer()
    builder = dataset.contract.PassiveContextBuilder(tokenizer)
    cases, dispositions = [], []
    for proof in qualified["qualifications"]:
        path = dataset.ROOT / proof["evidence"]
        evidence = json.loads(path.read_text())
        bind_observation(proof["review_observation"], evidence, dataset.sha256(path))
        dataset.require(dataset.sha256(path) == proof["evidence_sha256"], "Exact reviewed operation changed.")
        record = {"qualification_id": proof["id"], "source_group": proof["source_group_id"], "split": proof["split"],
                  "qualified_acoustic_operation": {"action": proof["action"], "keep_ms": proof["keep_ms"], "items": proof["items"]},
                  "evidence_path": proof["evidence"], "evidence_sha256": proof["evidence_sha256"],
                  "human_listened_to_historical_render": True, "new_human_action_observed": False}
        if proof["action"] != "delete_span":
            record.update(status="deferred", judgment_status="unknown",
                          reason="The historical acoustic review supports this exact 900ms operation, but the fixed passive input contains neither its former editorial brief nor an executed pacing preference. Acoustic feasibility alone does not establish that target duration is useful now.")
            dispositions.append(record)
            continue
        source = sources[proof["source_id"]]
        document_path = dataset.ROOT / source["document_path"]
        dataset.require(dataset.sha256(document_path) == source["document_sha256"], "Natural source document changed.")
        document = json.loads(document_path.read_text())
        records, mapping = dataset.contract._records(document)
        # The first actual spoken item supplies a cursor independently of the reviewed target.
        focus = mapping[str(next(item["id"] for item in records if item["type"] == "text"))]["uuid"]
        bundle = builder.build(document, focus, as_of=0, caption_profile=dataset.PROFILE)
        target = next(item["id"] for item in bundle["input"]["items"]
                      if mapping[str(item["id"])]["uuid"] == proof["items"][0]["uuid"])
        current = next(item for item in bundle["input"]["items"] if item["id"] == target)
        dataset.require(current["text"] == "Um", "Visible reference is no longer the isolated opening planning filler.")
        primary = {"action": "delete_span", "target": [target]}
        after = dataset.contract.apply_passive_edit(document, bundle, primary)
        verify_media_edit(document, after, bundle["mapping"], primary, [proof])
        source_row = {"id": source["id"], "source_group": source["source_group_id"], "split": source["split"],
                      "language": source["language"], "natural_audio": True, "generated_audio": False,
                      "provenance": "native_words_with_exact_historical_audio_review",
                      "acoustically_verified": True, "acoustic_scope": proof["scope"],
                      "media_path": source["media_path"], "media_sha256": source["media_sha256"],
                      "qualification_id": proof["id"], "qualification_sha256": dataset.sha256(QUALIFICATIONS)}
        runtime = {"focus_uuid": focus, "selection": [], "sequence": 0, "history": [],
                   "feedback_journal": None, "caption_profile": dataset.PROFILE}
        row = dataset.case_row("actions-v3-natural-ami-opening-filler", source["split"], source["language"], source_row,
                               document, bundle, primary, [], runtime,
                               "The isolated planning filler precedes 'well this is the kick-off meeting'; its removal preserves well and the complete introduction. This fresh passive-policy reference is AI-authored from visible wording, and the exact source operation matches the earlier independently retained listening observation. Other disfluencies are not implicitly certified.",
                               {"old_id": None, "scenario_family": "natural_opening_planning_filler",
                                "selection_method": "reviewed_source_challenge", "cursor_selection_before_judgment": True,
                                "context": bundle["context"], "source_document": source["document_path"],
                                "source_document_sha256": source["document_sha256"], "audio_review": record,
                                "label_origin": "regenerated_ai_reference_not_copied_human_label"},
                               acoustic="exact_historical_human_listening_observation_and_source_interval_verification")
        result = dataset.validate_case(row, builder)
        row["execution_sha256"] = dataset.digest(result)
        cases.append(row)
        record.update(status="admitted", judgment_status="edit_required", new_id=row["id"],
                      reason="Visible isolated planning filler supports the passive cleanup; the exact operation and outside media intervals match its historical acoustic evidence.")
        dispositions.append(record)
    for withheld in qualified["withheld"]:
        dispositions.append({"qualification_id": withheld["id"], "status": "excluded", "judgment_status": "inapplicable",
                             "reason": withheld["reason"], "evidence_sha256": withheld["evidence_sha256"]})
    dataset.write_rows(output / "cases.jsonl", cases)
    dataset.write_rows(output / "acoustic-dispositions.jsonl", dispositions)
    dataset.write(output / "manifest.json", {"version": "actions-v3-natural-operation-requalification-1",
                  "case_count": len(cases), "status": "prepared_pending_parent_semantic_review",
                  "files_sha256": {str(p.relative_to(output)): dataset.sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
                  "tokenizer": tokenizer_identity, "code_sha256": dataset.sha256(Path(__file__)),
                  "qualification_path": str(QUALIFICATIONS.relative_to(dataset.ROOT)), "qualification_sha256": dataset.sha256(QUALIFICATIONS)})
    return {"natural_cut_cases": len(cases), "qualified_operations_reconsidered": 4, "deferred_missing_passive_pacing_evidence": 3,
            "excluded_event_conflict": 1, "new_listening_observations": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
