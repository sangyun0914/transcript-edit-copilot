"""Prepare a calibration batch from a saved project and its cached PCM audio."""

import argparse
import copy
import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app.review import Edit, validate_edit  # noqa: E402


def spread(values, count):
    if len(values) < count:
        raise ValueError(f"Need {count} suitable examples; found {len(values)}.")
    return [
        values[round(i * (len(values) - 1) / max(1, count - 1))] for i in range(count)
    ]


def prepare(project, audio, output):
    if output.exists():
        raise ValueError(
            "Choose a new output directory; existing reviews are never overwritten."
        )
    with zipfile.ZipFile(project) as archive:
        document = json.loads(archive.read("document.json"))
    source_keys = {i["source"] for i in document["content"] if "source" in i}
    if len(source_keys) != 1:
        raise ValueError("This calibration builder expects one continuous source.")
    source_key = source_keys.pop()
    annotations = [
        region
        for record in document["metadata"].get("transcriptions", [])
        for region in record.get("regions", [])
    ]
    items, source_items = [], {}
    paragraph, speaker, language = -1, None, None
    previous_end, source_run = None, 0
    for item in document["content"]:
        if item["type"] == "paragraph_start":
            paragraph += 1
            speaker, language = item["speaker"], item.get("language") or "ko"
        elif item["type"] in ("text", "non_text"):
            start, end = item["sourceStart"], item["sourceStart"] + item["length"]
            if previous_end is not None and abs(start - previous_end) > 1e-6:
                source_run += 1
            previous_end = end
            item_id = len(items)
            status = "unverified"
            for region in annotations:
                if (
                    region["sourceStart"] < end
                    and start < region["sourceStart"] + region["length"]
                ):
                    status = region.get("boundary_status", "unverified")
            items.append(
                {
                    "id": item_id,
                    "type": item["type"],
                    "text": item.get("text", ""),
                    "duration_ms": max(1, round(item["length"] * 1000)),
                    "paragraph": paragraph,
                    "speaker": speaker,
                    "language": language,
                    "source_backed": True,
                    "source_run": source_run,
                    "boundary_status": status,
                    "audio_class": "speech" if item["type"] == "text" else "unknown",
                }
            )
            source_items[str(item_id)] = {
                "uuid": item["uuid"],
                "source": source_key,
                "start_s": start,
                "end_s": end,
            }
    if source_run:
        raise ValueError("Use the unedited continuous import for this builder.")
    candidates = []

    def add(action, indices, count, note, **parameters):
        for index in spread(indices, count):
            candidates.append(
                (index, {"action": action, "target": [index], **parameters}, note, None)
            )

    add(
        "delete_span",
        [
            i
            for i, x in enumerate(items)
            if x["type"] == "text"
            and x["text"].strip(".,?!") in ("어", "음", "이제", "그러니까")
        ],
        4,
        "Candidate discourse-marker removal. It may carry meaning; judge in context.",
    )
    gaps = [
        i
        for i, x in enumerate(items)
        if x["type"] == "non_text" and 500 < x["duration_ms"] < 2500
    ]
    add(
        "trim_silence",
        gaps,
        4,
        "Candidate gap shortening. The interval has not been verified as silence.",
        keep_ms=300,
    )
    add(
        "remove_silence",
        [
            i
            for i, x in enumerate(items)
            if x["type"] == "non_text" and 450 < x["duration_ms"] < 800
        ],
        2,
        "Candidate pause removal. Check breaths, speech and the resulting join.",
    )
    splits = [
        i
        for i in range(1, len(items) - 1)
        if items[i]["type"] == "text"
        and items[i - 1]["paragraph"] == items[i]["paragraph"]
        and any(x["text"].endswith((".", "?", "!")) for x in items[max(0, i - 2) : i])
    ]
    add(
        "split_paragraph",
        splits,
        4,
        "Candidate caption split at a sentence transition. Audio remains unchanged.",
    )
    merges = [
        i
        for i in range(1, len(items))
        if items[i - 1]["paragraph"] != items[i]["paragraph"]
        and items[i - 1]["speaker"] == items[i]["speaker"]
        and items[i - 1]["language"] == items[i]["language"]
    ]
    if len(merges) >= 3:
        add(
            "merge_paragraph",
            merges,
            3,
            "Candidate caption merge. Check readability and cue duration.",
        )
    else:
        add(
            "merge_paragraph",
            splits,
            3,
            "A caption boundary was inserted for calibration. Review whether merging it "
            "improves readability; this is not an observed caption edit.",
        )
    text_indices = [
        i for i, x in enumerate(items) if x["type"] == "text" and "습니다" in x["text"]
    ]
    for index in spread(text_indices, 3):
        original = items[index]["text"]
        candidates.append(
            (
                index,
                {"action": "correct_text", "target": [index], "text": original},
                "Controlled spelling corruption for reviewer calibration; the target restores "
                "the cached ASR text, which is not human ground truth.",
                original.replace("습니다", "슴니다"),
            )
        )
    negative = [
        i
        for i, x in enumerate(items)
        if x["type"] == "text"
        and any(term in x["text"] for term in ("아니", "않", "무시", "중요"))
    ]
    for index in spread(negative, 4):
        candidates.append(
            (
                index,
                {"action": "no_action"},
                "Candidate keep decision around negation or meaningful content. "
                "Review other possible edits too.",
                None,
            )
        )
    cases = []
    for index, proposal, note, corrupted in candidates:
        window = copy.deepcopy(items[max(0, index - 18) : min(len(items), index + 19)])
        mapping = {str(x["id"]): source_items[str(x["id"])] for x in window}
        start = mapping[str(window[0]["id"])]["start_s"]
        end = mapping[str(window[-1]["id"])]["end_s"]
        if end - start > 60:
            raise ValueError("An excerpt exceeds the 60-second review limit.")
        origin = "constructed_source_window"
        if corrupted:
            next(x for x in window if x["id"] == index)["text"] = corrupted
            origin = "constructed_corruption"
        if proposal["action"] == "merge_paragraph" and index not in merges:
            old_paragraph = items[index]["paragraph"]
            for item in window:
                if item["id"] >= index and item["paragraph"] == old_paragraph:
                    item["paragraph"] += 10000
            origin = "constructed_caption_boundary"
        case = {
            "id": f"calibration-{len(cases) + 1:02d}",
            "input": {
                "task": "predict_next_transcript_edit",
                "language": "ko",
                "mode": "transcript_cleanup",
                "cursor_before": index,
                "selection": [],
                "recent_edits": [],
                "items": window,
                "allowed_actions": list(
                    Edit.model_fields["action"].annotation.__args__
                ),
                "protected_ids": [],
            },
            "expectedOutput": None,
            "proposal": proposal,
            "metadata": {
                "split": "calibration",
                "source_group_id": source_key,
                "context_origin": origin,
                "history_origin": "unobserved",
                "target_origin": "rule_candidate",
                "timing_origin": "cached_asr_unverified",
                "observed_next_action": None,
                "review_status": "unreviewed",
                "proposal_note": note,
                "position_s": source_items[str(index)]["start_s"],
                "audio": {
                    "media_id": "source",
                    "source": source_key,
                    "start_s": start,
                    "end_s": end,
                },
                "source_items": mapping,
            },
        }
        validate_edit(case, Edit.model_validate(proposal))
        cases.append(case)
    output.mkdir(parents=True)
    (output / "media").mkdir()
    shutil.copyfile(audio, output / "media/source.wav")
    with audio.open("rb") as file:
        audio_hash = hashlib.file_digest(file, "sha256").hexdigest()
    with project.open("rb") as file:
        project_hash = hashlib.file_digest(file, "sha256").hexdigest()
    manifest = {
        "schema_version": 1,
        "title": "Korean transcript · first review",
        "note": (
            "24 calibration drafts from the cached recording. These are constructed contexts "
            "and rule candidates, not observed next actions or a held-out quality benchmark. "
            "No training permission is inferred."
        ),
        "source_project_sha256": project_hash,
        "source_group_id": source_key,
        "media": {"source": {"path": "media/source.wav", "sha256": audio_hash}},
        "counts": dict(Counter(c["proposal"]["action"] for c in cases)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    )
    print(
        json.dumps(
            {"cases": len(cases), "counts": manifest["counts"], "output": str(output)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project, args.audio, args.output)
