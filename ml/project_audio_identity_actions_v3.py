"""Remove generation-kind hints from generated-audio predictor source identities."""

import argparse
import copy
import json
from pathlib import Path

import prepare_passive_actions_v3 as dataset
from prepare_audio_actions_v3 import read_pcm, render


def prepare(original, output):
    original, output = Path(original).resolve(), Path(output).resolve()
    dataset.require(not output.exists(), "Use a fresh projection attempt.")
    previous = json.loads((original / "manifest.json").read_text())
    for name, expected in previous["files_sha256"].items():
        dataset.require(dataset.sha256(original / name) == expected, "Original generated-audio evidence changed.")
    tokenizer, identity = dataset.legacy.tokenizer()
    builder = dataset.contract.PassiveContextBuilder(tokenizer)
    cases, proofs, audio_proofs = [], [], []
    for old in dataset.rows(original / "cases.jsonl"):
        runtime = copy.deepcopy(old["runtime"])
        documents = [old["document"], *[doc for transition in runtime["history"] for doc in (transition["before"], transition["after"])]]
        mapping = dataset.legacy.source_map(documents)
        generated = dataset.legacy.projected_history_uuids(runtime["history"], mapping)
        document = dataset.legacy.project_document(old["document"], mapping, generated)
        runtime["history"] = dataset.legacy.project(runtime["history"], mapping, generated)
        runtime["focus_uuid"] = generated.get(runtime["focus_uuid"], runtime["focus_uuid"])
        bundle = dataset.runtime_bundle({"document": document, "runtime": runtime}, builder)
        row = copy.deepcopy(old)
        row.update(document=document, runtime=runtime, input=bundle["input"], messages=dataset.contract.canonical_messages(bundle["input"]))
        row.pop("execution_sha256")
        row["lineage"].update(context=bundle["context"], runtime_source_projection=mapping,
                              parent_component=str(original.relative_to(dataset.ROOT)), parent_input_sha256=dataset.digest(old["input"]),
                              projection_reason="Opaque serving identity removes the derived prefix; reference, split and PCM are unchanged.")
        proof = dataset.validate_case(row, builder)
        row["execution_sha256"] = dataset.digest(proof)
        after = dataset.contract.apply_passive_edit(document, bundle, row["primary"])
        media = dataset.ROOT / row["source"]["media_path"]
        pcm = read_pcm(media)
        reference = read_pcm(media.parent / "uncorrupted-reference.wav")
        dataset.require(render(after, pcm, mapping[row["source"]["id"]]) == reference, "Projection changed the exact original PCM restoration.")
        dataset.require(all(item["source_id"] is None or item["source_id"].startswith("src-") for item in row["input"]["items"]), "Generation-kind hint remains in predictor source identity.")
        cases.append(row)
        proofs.append({"id": row["id"], **proof})
        audio_proofs.append({"id": row["id"], "actual_projected_executor_render_matches_original_pcm": True,
                             "media_sha256": dataset.sha256(media), "reference_sha256": dataset.sha256(media.parent / "uncorrupted-reference.wav"),
                             "reference_unchanged": row["primary"] == old["primary"], "audio_copied": False})
    output.mkdir(parents=True)
    dataset.write_rows(output / "cases.jsonl", cases)
    dataset.write_rows(output / "reference-executions.jsonl", proofs)
    dataset.write_rows(output / "audio-proofs.jsonl", audio_proofs)
    dataset.write(output / "manifest.json", {"version": "actions-v3-audio-opaque-identity-projection-1", "status": "verified_generated_audio",
                  "case_count": len(cases), "counts": previous["counts"], "tokenizer": identity, "code_sha256": dataset.sha256(Path(__file__)),
                  "parent_manifest": str((original / "manifest.json").relative_to(dataset.ROOT)), "parent_manifest_sha256": dataset.sha256(original / "manifest.json"),
                  "files_sha256": {str(path.relative_to(output)): dataset.sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
                  "claims": "Same four controlled generated-audio restorations; source identity projection is reversible and supplies no acoustic evidence for natural gaps."})
    return {"cases": len(cases), "exact_pcm_restorations": len(audio_proofs), "media_copied": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.original, args.output), indent=2))
