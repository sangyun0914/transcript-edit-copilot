"""Assess a new project's fixture suitability without inventing natural correction labels."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from session_cases import load_project, snapshot_for


def audit(project, output, audio=None, candidates=(), video=None):
    if candidates and audio is None:
        raise ValueError("Review candidates need an audio excerpt source.")
    if video and (not candidates or video.suffix.lower() not in (".mp4", ".webm")):
        raise ValueError(
            "Video context requires review candidates and an MP4 or WebM source."
        )
    document = load_project(project)
    content = document["content"]
    words = [i for i in content if i["type"] == "text"]
    gaps = [i for i in content if i["type"] == "non_text"]
    records = document["metadata"].get("transcriptions", [])
    paragraphs, current = [], []
    for item in content:
        if item["type"] == "paragraph_start":
            current = []
        elif item["type"] == "text":
            current.append(item["text"])
        elif item["type"] == "paragraph_end":
            paragraphs.append(" ".join(current))
    report = {
        "project": str(project),
        "project_sha256": hashlib.sha256(project.read_bytes()).hexdigest(),
        "items": dict(Counter(i["type"] for i in content)),
        "duration_seconds": sum(i.get("length", 0) for i in content),
        "empty_text_items": sum(not i["text"].strip() for i in words),
        "gaps_over_one_second": sum(i["length"] > 1 for i in gaps),
        "paragraphs_over_80_characters": sum(len(p) > 80 for p in paragraphs),
        "boundary_statuses": dict(
            Counter(
                r["boundary_status"]
                for record in records
                for r in record.get("regions", [])
            )
        ),
        "annotations": dict(
            Counter(
                a["type"] for record in records for a in record.get("annotations", [])
            )
        ),
        "correction_accuracy": "not_established_without_audio_or_reference_review",
        "confirmed_natural_correction_targets": None,
        "review_candidates": len(candidates),
        "recommendation": (
            "Do not assume a complete transcript is error-free. If inspection finds few useful corrections, "
            "keep it for abstention/overcorrection checks; use separately labeled constructed corruptions "
            "or another permitted recording for positive correction coverage. Caption length and gap "
            "counts are inspection opportunities, not edit labels."
        ),
        "sample_paragraphs": [
            {"paragraph": i, "text": paragraphs[i]}
            for i in sorted({round(j * (len(paragraphs) - 1) / 11) for j in range(12)})
        ]
        if paragraphs
        else [],
    }
    output.mkdir(parents=True, exist_ok=False)
    if candidates:
        root = output / "review"
        root.mkdir()
        shutil.copyfile(audio, root / "source.wav")
        videos = {}
        if video:
            video_name = "source" + video.suffix.lower()
            shutil.copyfile(video, root / video_name)
            with video.open("rb") as file:
                videos["source"] = {
                    "path": video_name,
                    "sha256": hashlib.file_digest(file, "sha256").hexdigest(),
                }
        cases = []
        for index, candidate in enumerate(candidates):
            bundle = snapshot_for(
                document,
                candidate["target_uuids"][0],
                "의미와 말투를 유지하고 명백한 인식 오류만 수정하세요.",
            )
            aliases = {v["uuid"]: int(k) for k, v in bundle["mapping"].items()}
            proposal = {
                "action": "correct_text",
                "target": [aliases[u] for u in candidate["target_uuids"]],
                "text": candidate["text"],
            }
            sources = {}
            for key, value in bundle["mapping"].items():
                item = content[value["index"]]
                sources[key] = {
                    "uuid": item["uuid"],
                    "source": item["source"],
                    "start_s": item["sourceStart"],
                    "end_s": item["sourceStart"] + item["length"],
                }
            starts, ends = (
                [v["start_s"] for v in sources.values()],
                [v["end_s"] for v in sources.values()],
            )
            if (
                len({v["source"] for v in sources.values()}) != 1
                or max(ends) - min(starts) > 60
            ):
                raise ValueError(
                    "Choose a single-source review window of at most 60 seconds."
                )
            cases.append(
                {
                    "id": f"natural-review-{index + 1}",
                    "input": bundle["input"],
                    "expectedOutput": None,
                    "proposal": proposal,
                    "metadata": {
                        "split": "calibration",
                        "source_group_id": next(iter(sources.values()))["source"],
                        "context_origin": "unaltered_source",
                        "history_origin": "unobserved",
                        "target_origin": "ai_candidate_unconfirmed",
                        "review_status": "needs_review",
                        "training_eligible": False,
                        "position_s": sources[str(proposal["target"][0])]["start_s"],
                        "proposal_note": candidate["reason"],
                        "timing_origin": "cached_asr_unverified",
                        "source_items": sources,
                        "audio": {
                            "media_id": "source",
                            "source": next(iter(sources.values()))["source"],
                            "start_s": min(starts),
                            "end_s": max(ends),
                        },
                    },
                }
            )
        (root / "cases.jsonl").write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "title": "Natural transcript — uncertain correction",
                    "note": "Unconfirmed AI candidates from unchanged source text. Listen before judging. "
                    "Feedback improves generation; exported human labels are not dataset targets.",
                    "media": {
                        "source": {
                            "path": "source.wav",
                            "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                        }
                    },
                    "label_policy": "feedback_then_regenerate",
                    "videos": videos,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional original MP4/WebM sharing the audio source timeline",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        help="AI candidate notes; unknown targets remain null",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.project,
        args.output,
        args.audio,
        json.loads(args.candidates.read_text()) if args.candidates else [],
        args.video,
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "sample_paragraphs"}, indent=2
        )
    )
