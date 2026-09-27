"""Prepare fixed AMI evidence excerpts without flattening speaker overlap into edits."""

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from download_sources import digest


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    acquisition = json.loads((source / "acquisition.json").read_text())
    catalog = acquisition["catalog"]
    for name, record in acquisition["files"].items():
        if digest(source / name) != record["sha256"]:
            raise ValueError(f"Changed AMI source: {name}")
    records = []
    for path in sorted((source / "annotations/words").glob("ES2002a.*.words.xml")):
        speaker = path.name.split(".")[1]
        for element in ET.parse(path).getroot():
            if "starttime" not in element.attrib or "endtime" not in element.attrib:
                continue
            records.append(
                {
                    "speaker": speaker,
                    "start": float(element.attrib["starttime"]),
                    "end": float(element.attrib["endtime"]),
                    "text": "".join(element.itertext()),
                    "native_tag": element.tag,
                    "attributes": element.attrib,
                }
            )
    if {r["speaker"] for r in records} != set("ABCD"):
        raise ValueError("Expected the four separate speaker annotations.")
    records.sort(key=lambda r: (r["start"], r["end"], r["speaker"]))
    (output / "source-records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    windows = []
    for index, start in enumerate(catalog["window_start_seconds"]):
        end = start + catalog["window_duration_seconds"]
        selected = [r for r in records if r["start"] < end and r["end"] > start]
        words = [
            r
            for r in selected
            if r["native_tag"] == "w"
            and r["attributes"].get("punc") != "true"
            and r["end"] > r["start"]
        ]
        endpoints = sorted(
            {
                start,
                end,
                *[max(start, r["start"]) for r in words],
                *[min(end, r["end"]) for r in words],
            }
        )
        overlap = sum(
            right - left
            for left, right in zip(endpoints, endpoints[1:])
            if len(
                {r["speaker"] for r in words if r["start"] < right and r["end"] > left}
            )
            > 1
        )
        preview = output / f"window-{index + 1:02}.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                str(start),
                "-i",
                str(source / "ES2002a.Mix-Headset.wav"),
                "-t",
                str(end - start),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(preview),
            ],
            check=True,
        )
        window = {
            "id": f"ami-es2002a-{index + 1:02}",
            "split": "train",
            "source_group_id": catalog["source_group_id"],
            "start": start,
            "end": end,
            "records": selected,
            "spoken_word_annotations": len(words),
            "speakers": sorted({r["speaker"] for r in words}),
            "annotation_implied_overlap_seconds": round(overlap, 3),
            "preview": preview.name,
            "preview_sha256": digest(preview),
            "training_eligible": False,
            "expectedOutput": None,
            "status": "source_only",
            "timing_status": "publisher_annotation_not_independently_verified_cut_alignment",
        }
        windows.append(window)
    (output / "windows.jsonl").write_text(
        "".join(json.dumps(w, ensure_ascii=False) + "\n" for w in windows)
    )
    summary = {
        "source": str(source),
        "source_acquisition_sha256": digest(source / "acquisition.json"),
        "script_sha256": digest(Path(__file__)),
        "record_count": len(records),
        "native_tags": dict(Counter(r["native_tag"] for r in records)),
        "zero_duration_records": sum(r["start"] == r["end"] for r in records),
        "windows": [{k: v for k, v in w.items() if k != "records"} for w in windows],
        "admitted_cases": 0,
        "limitations": [
            "This is source evidence, not a model input, edit reference, or replayable Audapolis document.",
            "Original simultaneous speaker intervals and event/punctuation records are retained; no fabricated sequential word timeline.",
            "Overlap duration is derived from publisher timestamps and is not an independent listening verification.",
            "Fixed excerpts may cut utterances; context must later be selected from the full source through the runtime policy.",
            "The headset letter-to-channel mapping and destructive-cut boundaries still require verification.",
        ],
    }
    summary["files"] = {
        p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("files", "limitations")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
