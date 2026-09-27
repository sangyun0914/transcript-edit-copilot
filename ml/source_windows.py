"""Prepare deterministic, unlabeled real-media windows with local audio previews."""

import argparse
import html
import json
import re
import subprocess
from pathlib import Path

from download_sources import digest
from iteration import jsonl
from session_cases import load_project


def caption_cues(text):
    pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})[^\n]*\n(.+?)(?=\n\s*\n|\Z)",
        re.S,
    )

    def seconds(value):
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    return [
        {
            "start": seconds(start),
            "end": seconds(end),
            "text": html.unescape(re.sub(r"<[^>]+>", "", content))
            .replace("\n", " ")
            .strip(),
        }
        for start, end, content in pattern.findall(text)
    ]


def build(catalog_path, raw, output, local_catalog=None):
    catalog = json.loads(catalog_path.read_text())
    sources = []
    for source in catalog["sources"]:
        root = raw / source["id"]
        record = json.loads((root / "source.json").read_text())
        for name in ("video.mp4", "captions.vtt"):
            if digest(root / name) != record["files"][name]["sha256"]:
                raise ValueError("Downloaded source changed since acquisition.")
        sources.append(
            {
                **source,
                "video": root / "video.mp4",
                "duration": float(record["media"]["format"]["duration"]),
                "cues": caption_cues((root / "captions.vtt").read_text()),
                "transcript_origin": "publisher_captions_not_raw_asr",
                "transcript_sha256": record["files"]["captions.vtt"]["sha256"],
                "video_sha256": record["files"]["video.mp4"]["sha256"],
                "license": catalog["license"],
                "license_url": catalog["license_url"],
            }
        )
    if local_catalog:
        for source in json.loads(local_catalog.read_text())["sources"]:
            project, video = Path(source["project"]), Path(source["video"])
            document = load_project(project)
            items = [
                i for i in document["content"] if i["type"] == "text" and "source" in i
            ]
            if len({i["source"] for i in items}) != 1:
                raise ValueError(
                    "Local preview requires exactly one original media source."
                )
            duration = float(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "csv=p=0",
                        str(video),
                    ],
                    text=True,
                )
            )
            sources.append(
                {
                    **source,
                    "video": video,
                    "duration": duration,
                    "cues": [
                        {
                            "start": i["sourceStart"],
                            "end": i["sourceStart"] + i["length"],
                            "text": i["text"],
                        }
                        for i in items
                    ],
                    "transcript_origin": "cached_scribe_project_unverified_alignment",
                    "transcript_sha256": digest(project),
                    "video_sha256": digest(video),
                }
            )
    output.mkdir(parents=True, exist_ok=False)
    (output / "audio").mkdir()
    windows = []
    for source in sources:
        if not source["cues"]:
            raise ValueError("Source contains no timed transcript cues.")
        for index, fraction in enumerate((0.05, 0.18, 0.32, 0.47, 0.64, 0.82)):
            start = round((source["duration"] - 60) * fraction, 3)
            end = start + 60
            cues = [c for c in source["cues"] if c["end"] > start and c["start"] < end]
            if not cues:
                raise ValueError("A scheduled source window has no transcript.")
            identifier = f"{source['id']}-window-{index + 1:02}"
            audio = output / "audio" / f"{identifier}.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-ss",
                    str(start),
                    "-i",
                    str(source["video"]),
                    "-t",
                    "60",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ],
                check=True,
            )
            windows.append(
                {
                    "id": identifier,
                    "source_id": source["id"],
                    "source_group_id": source["source_group_id"],
                    "reserved_split": source["split"],
                    "language": source["language"],
                    "source_video": str(source["video"]),
                    "start_seconds": start,
                    "end_seconds": end,
                    "audio_preview": audio.relative_to(output).as_posix(),
                    "audio_sha256": digest(audio),
                    "text": " ".join(c["text"] for c in cues),
                    "cues": cues,
                    "transcript_origin": source["transcript_origin"],
                    "transcript_sha256": source["transcript_sha256"],
                    "video_sha256": source["video_sha256"],
                    "source_page": source.get("page_url"),
                    "creator": source.get("creator"),
                    "license": source["license"],
                    "license_url": source.get("license_url"),
                    "expectedOutput": None,
                    "training_eligible": False,
                    "boundary_status": "unverified",
                    "review_status": "source_context_only_no_ai_reference_yet",
                }
            )
    jsonl(output / "windows.jsonl", windows)
    report = {
        "windows": len(windows),
        "sources": len(sources),
        "preview_seconds_each": 60,
        "sampling": "Six fixed relative positions per recording, before model evaluation",
        "references": 0,
        "training_eligible": 0,
        "scope": "Source material for AI scenario authoring and evidence inspection; no natural-STT, word-alignment or audio-cut qualification claim.",
        "catalog_sha256": digest(catalog_path),
        "source_groups": {s["source_group_id"]: s["split"] for s in sources},
        "windows_sha256": digest(output / "windows.jsonl"),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--local-catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.catalog, args.raw, args.output, args.local_catalog), indent=2
        )
    )
