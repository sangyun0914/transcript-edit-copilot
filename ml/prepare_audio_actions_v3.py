"""Materialize controlled transport gaps in exposed audio and prove exact restoration.

These are generated corruptions of existing recordings, never naturally occurring
silence examples. The serving contract continues to expose unverified/unknown
acoustics; the separate reference proof binds only the constructed operation.
"""

import argparse
import json
import subprocess
import wave
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import prepare_passive_actions_v3 as dataset
from passive_dataset import RESERVED

RATE = 16000
VERSION = "actions-v3-constructed-transport-audio-1"
REGISTRY = dataset.ROOT / "data/manifests/passive-existing-sources-v1.json"
SOURCE_IDS = ("ami-es2002a", "ami-is1008b", "calculus-0600-1500", "shuka")


def write_pcm(path, pcm):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(RATE)
        audio.writeframes(pcm)


def read_pcm(path):
    with wave.open(str(path), "rb") as audio:
        dataset.require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, RATE), "Unexpected PCM format.")
        return audio.readframes(audio.getnframes())


def candidate_boundaries(document):
    paragraphs, current = [], None
    for item in document["content"]:
        if item["type"] == "paragraph_start":
            current = {"paragraph": item, "items": []}
        elif item["type"] == "paragraph_end":
            if current:
                paragraphs.append(current)
            current = None
        elif current:
            current["items"].append(item)
    candidates = []
    for paragraph in paragraphs:
        if not paragraph["paragraph"].get("speaker"):
            continue
        items = paragraph["items"]
        for boundary in range(1, len(items)):
            left, right = items[boundary - 1:boundary + 1]
            if left["type"] != "text" or right["type"] != "text":
                continue
            if abs(left["sourceStart"] + left["length"] - right["sourceStart"]) > 1e-6:
                continue
            if right["sourceStart"] < 30:
                continue
            text_indices = [i for i, item in enumerate(items) if item["type"] == "text"]
            at = text_indices.index(boundary)
            low, high = text_indices[max(0, at - 4)], text_indices[min(len(text_indices) - 1, at + 3)] + 1
            excerpt = items[low:high]
            if sum(item["type"] == "text" for item in excerpt) < 5:
                continue
            if not all(a.get("source") == b.get("source") and abs(a["sourceStart"] + a["length"] - b["sourceStart"]) < 1e-6 for a, b in zip(excerpt, excerpt[1:])):
                continue
            if excerpt[-1]["sourceStart"] + excerpt[-1]["length"] - excerpt[0]["sourceStart"] > 12:
                continue
            candidates.append({"paragraph": paragraph["paragraph"], "items": excerpt,
                               "boundary_index": boundary - low, "focus_uuid_before_corruption": excerpt[0]["uuid"]})
    return candidates


def decode_excerpt(media, start_sample, end_sample, output):
    dataset.require(end_sample > start_sample >= 0, "Invalid original sample range.")
    # Accurate output seeking operates after decoding; no media leaves the machine.
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(media),
                    "-ss", str(start_sample / RATE), "-t", str((end_sample - start_sample) / RATE),
                    "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(output)], check=True)
    pcm = read_pcm(output)
    dataset.require(len(pcm) // 2 == end_sample - start_sample, "Decoded excerpt sample count differs.")
    return pcm


def render(document, pcm, source_id):
    result = bytearray()
    for item in document["content"]:
        if "sourceStart" not in item:
            continue
        dataset.require(item["source"] == source_id, "Derived render references another source.")
        left = round(item["sourceStart"] * RATE)
        right = round((item["sourceStart"] + item["length"]) * RATE)
        dataset.require(0 <= left < right <= len(pcm) // 2, "Derived interval exceeds PCM.")
        result.extend(pcm[left * 2:right * 2])
    return bytes(result)


def prepare(output):
    output = Path(output).resolve()
    dataset.require(not output.exists(), "Use a fresh audio attempt directory.")
    output.mkdir(parents=True)
    registry = json.loads(REGISTRY.read_text())
    sources = {source["id"]: source for source in registry["sources"]}
    registration = {"version": VERSION, "sources": list(SOURCE_IDS), "source_registry_sha256": dataset.sha256(REGISTRY),
                    "selection": "Before assigning references, choose the first and middle structural adjacent-word boundaries after 30s within a single known-speaker paragraph; retain up to four original words on each side. Original cursor is the excerpt first item before a gap is inserted.",
                    "corruption": "Insert exact zero PCM between contiguous original words. Remove one earlier inserted gap through the editor before asking about the second. Expected current edit follows observable executed history and phrase interruption.",
                    "claims": "Generated transport artifacts, not natural silence prevalence. Removal restores the decoded original samples. No new listening observation is fabricated.",
                    "gap_ms": [650, 950], "actor": "synthetic", "natural_audio": False}
    dataset.write(output / "selection-before-construction.json", registration)
    tokenizer, identity = dataset.legacy.tokenizer()
    builder = dataset.contract.PassiveContextBuilder(tokenizer)
    cases, proofs = [], []
    for source_id in SOURCE_IDS:
        source = sources[source_id]
        dataset.require(source["split"] in ("train", "dev") and source["source_group_id"] not in RESERVED, "Reserved source.")
        path = dataset.ROOT / source["document_path"]
        dataset.require(dataset.sha256(path) == source["document_sha256"], "Original transcript changed.")
        document = json.loads(path.read_text())
        candidates = candidate_boundaries(document)
        dataset.require(len(candidates) >= 2, "Not enough structural boundaries in exposed source.")
        selected = [candidates[0], candidates[len(candidates) // 2]]
        folder = output / source_id
        folder.mkdir()
        dataset.write(folder / "source-selection.json", {"source_id": source_id, "source_group": source["source_group_id"],
                       "split": source["split"], "candidate_count": len(candidates), "selected_ordinals": [0, len(candidates) // 2],
                       "selected": selected, "source_document_sha256": source["document_sha256"]})
        source_identity = "derived:" + str(uuid5(NAMESPACE_URL, VERSION + source_id))
        def uid(value):
            return str(uuid5(NAMESPACE_URL, source_identity + "/" + value))
        pcm, uncorrupted, content, targets, source_regions = bytearray(), bytearray(), [], [], []
        for index, excerpt in enumerate(selected):
            original = excerpt["items"]
            first_sample = round(original[0]["sourceStart"] * RATE)
            last_sample = round((original[-1]["sourceStart"] + original[-1]["length"]) * RATE)
            media = Path(source["media_path"])
            if not media.is_absolute():
                media = dataset.ROOT / media
            decoded = decode_excerpt(media, first_sample, last_sample, folder / f"original-{index}.wav")
            insertion = round(original[excerpt["boundary_index"]]["sourceStart"] * RATE) - first_sample
            gap_samples = registration["gap_ms"][index] * RATE // 1000
            zero = b"\0\0" * gap_samples
            offset = len(pcm) // 2
            pcm.extend(decoded[:insertion * 2] + zero + decoded[insertion * 2:])
            uncorrupted.extend(decoded)
            language = source["language"]
            content.append({"type": "paragraph_start", "uuid": uid(f"p{index}"), "speaker": excerpt["paragraph"]["speaker"], "language": language})
            for n, item in enumerate(original):
                if n == excerpt["boundary_index"]:
                    target = uid(f"gap{index}")
                    targets.append(target)
                    content.append({"type": "non_text", "uuid": target, "source": source_identity,
                                    "sourceStart": (offset + insertion) / RATE, "length": gap_samples / RATE})
                start = round(item["sourceStart"] * RATE) - first_sample
                end = round((item["sourceStart"] + item["length"]) * RATE) - first_sample
                shift = gap_samples if n >= excerpt["boundary_index"] else 0
                transformed = {k: v for k, v in item.items() if k not in ("asr", "boundary_status", "audio_class")}
                transformed.update(uuid=uid(f"{index}-{n}"), source=source_identity,
                                   sourceStart=(offset + start + shift) / RATE, length=(end - start) / RATE)
                content.append(transformed)
            content.append({"type": "paragraph_end", "uuid": uid(f"e{index}")})
            source_regions.append({"source": source["media_sha256"], "start_sample": first_sample, "end_sample": last_sample,
                                   "original_excerpt_sha256": dataset.sha256(folder / f"original-{index}.wav"),
                                   "insert_sample_in_excerpt": insertion, "inserted_zero_samples": gap_samples,
                                   "parent_item_rounding_max_seconds": 1 / RATE})
        pcm = bytes(pcm)
        write_pcm(folder / "constructed.wav", pcm)
        write_pcm(folder / "uncorrupted-reference.wav", bytes(uncorrupted))
        before = {"content": content, "metadata": {"provenance": "constructed_transport_gap_from_exposed_recording"}}
        first_bundle = builder.build(before, targets[0], as_of=0, caption_profile=dataset.PROFILE)
        aliases = {value["uuid"]: int(key) for key, value in first_bundle["mapping"].items()}
        previous = {"action": "remove_silence", "target": [aliases[targets[0]]]}
        current = dataset.contract.apply_passive_edit(before, first_bundle, previous)
        history = [dataset.contract.committed_transition(before, current, previous, 1)]
        focus = next(item["uuid"] for item in current["content"] if "sourceStart" in item)
        bundle = builder.build(current, focus, as_of=1, history=history, caption_profile=dataset.PROFILE)
        aliases = {value["uuid"]: int(key) for key, value in bundle["mapping"].items()}
        primary = {"action": "remove_silence", "target": [aliases[targets[1]]]}
        after = dataset.contract.apply_passive_edit(current, bundle, primary)
        dataset.require(render(before, pcm, source_identity) == pcm, "Constructed editor input lost samples.")
        dataset.require(render(after, pcm, source_identity) == bytes(uncorrupted), "Full removal does not restore original samples.")
        write_pcm(folder / "editor-before.wav", render(current, pcm, source_identity))
        write_pcm(folder / "editor-after.wav", render(after, pcm, source_identity))
        source_row = {"id": source_identity, "source_group": source["source_group_id"], "split": source["split"], "language": source["language"],
                      "natural_audio": False, "generated_audio": True, "parent_natural_audio": True,
                      "provenance": "constructed_transport_gap_from_exposed_recording", "acoustically_verified": True,
                      "acoustic_scope": "Only the constructed zero-PCM gap and sample-exact restoration of original decoded excerpts; no natural boundary or global pacing certificate.",
                      "parent_source_id": source_id, "media_path": str((folder / "constructed.wav").relative_to(dataset.ROOT)),
                      "media_sha256": dataset.sha256(folder / "constructed.wav"), "parent_source_sha256": source["media_sha256"]}
        runtime = {"focus_uuid": focus, "selection": [], "sequence": 1, "history": history,
                   "feedback_journal": None, "caption_profile": dataset.PROFILE}
        row = dataset.case_row("actions-v3-audio-" + source_id, source["split"], source["language"], source_row, current, bundle,
                               primary, [], runtime, "A prior committed removal supplies local pacing evidence. The current inserted interruption separates adjacent original words. Full removal restores the original decoded speech sample for sample. This is a generated corruption challenge, not a claim about natural silence.",
                               {"old_id": None, "recipe": str((output / "selection-before-construction.json").relative_to(dataset.ROOT)),
                                "selection_method": "source_derived_controlled_corruption", "cursor_selection_before_judgment": True,
                                "scenario_family": "constructed_transport_gap", "source_selection": str((folder / "source-selection.json").relative_to(dataset.ROOT)),
                                "context": bundle["context"]}, acoustic="constructed_zero_pcm_sample_exact_restoration")
        proof = dataset.validate_case(row, builder)
        row["execution_sha256"] = dataset.digest(proof)
        cases.append(row)
        proofs.append({"id": row["id"], "source_regions": source_regions, "zero_pcm_gap_proven": True,
                       "actual_executor_render_matches_uncorrupted_pcm": True,
                       "render_sha256": dataset.sha256(folder / "editor-after.wav"), "reference_sha256": dataset.sha256(folder / "uncorrupted-reference.wav"),
                       "serving_audio_class_remains_unknown": True, "human_listened": False, "agent_listened": False})
    dataset.write_rows(output / "cases.jsonl", cases)
    dataset.write_rows(output / "audio-proofs.jsonl", proofs)
    dataset.write(output / "manifest.json", {"version": VERSION, "status": "verified_generated_audio",
                  "case_count": len(cases), "counts": {split: sum(c["split"] == split for c in cases) for split in ("train", "dev")},
                  "files_sha256": {str(p.relative_to(output)): dataset.sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
                  "tokenizer": identity, "code_sha256": dataset.sha256(Path(__file__))})
    return {"cases": len(cases), "natural_audio_cases": 0, "generated_audio_cases": len(cases), "exact_pcm_restorations": len(proofs)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
