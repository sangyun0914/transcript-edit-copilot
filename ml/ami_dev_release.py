"""Load the independent, preregistered AMI IS1008 development conversation."""

import copy
import json
import math
import wave
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import pairwise
from uuid import NAMESPACE_URL, uuid5

from dataset_release_inventory import ROOT, digest, read
from replay_session import session_hash
from session_cases import document_records

FOLDER = ROOT / "data/raw/ami/ami-dev-v1"
CATALOG = ROOT / "data/manifests/ami-dev-v1.json"
NITE_ID = "{http://nite.sourceforge.net/}id"


def annotation_records(folder=FOLDER):
    records = []
    for path in sorted((folder / "annotations/words").glob("IS1008b.*.words.xml")):
        previous_word = None
        for position, element in enumerate(ET.parse(path).getroot()):
            if not {NITE_ID, "starttime", "endtime"} <= element.attrib.keys():
                raise ValueError("A native annotation has no ID or original timing.")
            record = {
                "id": element.attrib[NITE_ID],
                "speaker": path.name.split(".")[1],
                "position": position,
                "start": float(element.attrib["starttime"]),
                "end": float(element.attrib["endtime"]),
                "text": "".join(element.itertext()),
                "tag": element.tag,
                "attributes": dict(element.attrib),
            }
            records.append(record)
            if element.attrib.get("punc") == "true":
                if previous_word is not None:
                    previous_word["display"] += record["text"]
                    previous_word.setdefault("punctuation_ids", []).append(record["id"])
            elif element.tag == "disfmarker" and previous_word is not None:
                previous_word["display"] += "-"
                previous_word.setdefault("disfluency_ids", []).append(record["id"])
            elif element.tag == "w":
                record["display"] = record["text"]
                previous_word = record
    return sorted(records, key=lambda r: (r["start"], r["speaker"], r["position"]))


def dev_document(annotations, source_hash, duration):
    if any(
        not math.isfinite(r["start"])
        or not math.isfinite(r["end"])
        or min(r["start"], r["end"]) < 0
        or max(r["start"], r["end"]) > duration
        for r in annotations
    ):
        raise ValueError("Annotation falls outside the mixed recording.")
    positive = [r for r in annotations if r["end"] > r["start"]]
    if any(
        r["tag"] == "w"
        and r["attributes"].get("punc") != "true"
        and r["end"] <= r["start"]
        for r in annotations
    ):
        raise ValueError("A spoken word has no usable interval; do not invent timing.")
    clusters = []
    for record in sorted(positive, key=lambda r: (r["start"], r["end"])):
        if clusters and record["start"] < clusters[-1]["end"] - 1e-8:
            clusters[-1]["end"] = max(clusters[-1]["end"], record["end"])
            clusters[-1]["records"].append(record)
        else:
            clusters.append(
                {"start": record["start"], "end": record["end"], "records": [record]}
            )
    timeline, cursor = [], 0.0
    for cluster in clusters:
        if cluster["start"] > cursor:
            timeline.append({"start": cursor, "end": cluster["start"], "records": []})
        timeline.append(cluster)
        cursor = cluster["end"]
    if cursor < duration:
        timeline.append({"start": cursor, "end": duration, "records": []})
    content, evidence, lineage = [], {}, {}
    previous_speakers, paragraph = None, -1

    def uid(name):
        return str(uuid5(NAMESPACE_URL, f"ami-is1008b-mixed-v1/{name}"))

    for index, cluster in enumerate(timeline):
        members = cluster["records"]
        speakers = tuple(sorted({r["speaker"] for r in members}))
        if not speakers:
            speakers = previous_speakers or ()
        if paragraph < 0 or speakers != previous_speakers:
            if paragraph >= 0:
                content.append({"type": "paragraph_end", "uuid": uid(f"e{paragraph}")})
            paragraph += 1
            content.append(
                {
                    "type": "paragraph_start",
                    "uuid": uid(f"p{paragraph}"),
                    "speaker": speakers[0] if len(speakers) == 1 else None,
                    "language": "en",
                }
            )
        words = [
            r
            for r in members
            if r["tag"] == "w" and r["attributes"].get("punc") != "true"
        ]
        text = "\n".join(
            (f"{speaker}: " if len(speakers) > 1 else "")
            + " ".join(
                r.get("display", r["text"])
                for r in sorted(words, key=lambda r: r["position"])
                if r["speaker"] == speaker
            )
            for speaker in speakers
            if any(r["speaker"] == speaker for r in words)
        )
        item = {
            "uuid": uid(f"i{index}"),
            "type": "text" if words else "non_text",
            "source": source_hash,
            "sourceStart": cluster["start"],
            "length": cluster["end"] - cluster["start"],
        }
        if words:
            item.update(text=text, conf=1)
        content.append(item)
        overlapping = len({r["speaker"] for r in members}) > 1
        evidence[item["uuid"]] = {
            "item_hash": session_hash(item),
            "boundary_status": "overlap" if overlapping else "unverified",
            "audio_class": "speech"
            if words
            else "audio_event"
            if members
            else "unknown",
            "origin": "publisher_annotations_not_perceptual_cut_verification",
        }
        lineage[item["uuid"]] = {
            "start": cluster["start"],
            "end": cluster["end"],
            "annotation_ids": [r["id"] for r in members],
            "speakers": sorted({r["speaker"] for r in members}),
            "events": [r for r in members if r["tag"] != "w"],
        }
        previous_speakers = speakers
    content.append({"type": "paragraph_end", "uuid": uid(f"e{paragraph}")})
    return {"content": content, "metadata": {}}, evidence, lineage


def source_audit(source, duration):
    annotations = source["native_annotations"]
    counts = Counter(r["id"] for r in annotations)
    represented = Counter(
        identity
        for island in source["item_annotations"].values()
        for identity in island["annotation_ids"]
    )
    expected = Counter(r["id"] for r in annotations if r["end"] > r["start"])
    items = [i for i in source["document"]["content"] if "sourceStart" in i]
    if set(counts.values()) != {1} or represented != expected:
        raise ValueError("Native annotation IDs were duplicated or lost.")
    if (
        any(
            abs(left["sourceStart"] + left["length"] - right["sourceStart"]) > 1e-7
            for left, right in pairwise(items)
        )
        or abs(sum(i["length"] for i in items) - duration) > 1e-7
    ):
        raise ValueError("The source audio timeline is not retained exactly once.")
    point_ids = {r["id"] for r in annotations if r["end"] == r["start"]}
    reversed_ids = {r["id"] for r in annotations if r["end"] < r["start"]}
    if set(represented) & (point_ids | reversed_ids):
        raise ValueError("Non-positive annotations were assigned invented durations.")
    return {
        "native_records": len(annotations),
        "native_unique_ids": len(counts),
        "native_tags": dict(Counter(r["tag"] for r in annotations)),
        "positive_duration_annotations_retained_once": sum(represented.values()),
        "zero_duration_annotations_retained_in_native_records": len(point_ids),
        "reversed_annotations_retained_in_native_records": len(reversed_ids),
        "nonpositive_spoken_words": 0,
        "overlap_islands": sum(
            e["boundary_status"] == "overlap"
            for e in source["boundary_evidence"].values()
        ),
        "timeline_items": len(items),
        "source_duration_seconds": duration,
        "timeline_duration_seconds": sum(i["length"] for i in items),
        "audio_interval_multiplicity": 1,
        "zero_duration_event_policy": "Retain original point annotations; no inferred event durations and no silence certification at their boundaries.",
        "qualified_destructive_cuts": 0,
        "listening_performed": False,
        "all_invariants_passed": True,
    }


def ami_dev_sources():
    acquisition = read(FOLDER / "acquisition.json")
    catalog = acquisition["catalog"]
    if (
        catalog["split"] != "dev"
        or catalog["source_group_id"] != "ami-is1008-series"
        or catalog["meeting_id"] != "IS1008b"
        or digest(CATALOG) != acquisition["catalog_sha256"]
        or read(CATALOG) != catalog
    ):
        raise ValueError("The preregistered AMI development assignment changed.")
    for name, expected in acquisition["extracted_annotation_sha256"].items():
        if digest(FOLDER / "annotations" / name) != expected:
            raise ValueError("AMI development annotation changed.")
    for path, expected in (
        (ROOT / acquisition["archive_path"], acquisition["archive_sha256"]),
        (FOLDER / "meetings.xml", acquisition["metadata_sha256"]),
        (ROOT / acquisition["license_path"], acquisition["license_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError("AMI acquisition provenance changed.")
    media = FOLDER / "IS1008b.Mix-Headset.wav"
    media_hash = acquisition["files"][media.name]["sha256"]
    if digest(media) != media_hash:
        raise ValueError("AMI development mixed audio changed.")
    with wave.open(str(media)) as stream:
        duration = stream.getnframes() / stream.getframerate()
    annotations = annotation_records()
    document, evidence, lineage = dev_document(annotations, media_hash, duration)
    records, mapping = document_records(document, evidence)
    source = {
        "id": catalog["source_id"],
        "source_group_id": catalog["source_group_id"],
        "reserved_split": "dev",
        "language": "en",
        "video": media,
        "video_sha256": media_hash,
        "transcript_sha256": session_hash(annotations),
        "transcript_origin": "ami_publisher_verbatim_transcript_with_overlap_islands",
        "itemization": "publisher_word_alignment_overlap_components_no_retiming",
        "license": catalog["license"],
        "source_page": catalog["page_url"],
        "document": document,
        "records": records,
        "mapping": mapping,
        "boundary_evidence": evidence,
        "item_annotations": lineage,
        "native_annotations": copy.deepcopy(annotations),
        "acquisition_path": "data/raw/ami/ami-dev-v1/acquisition.json",
        "catalog_path": "data/manifests/ami-dev-v1.json",
    }
    source["import_audit"] = source_audit(source, duration)
    windows = [
        {
            "id": f"ami-is1008b-window-{index:02d}",
            "source_id": source["id"],
            "start_seconds": start,
            "end_seconds": start + catalog["window_duration_seconds"],
        }
        for index, start in enumerate(catalog["window_start_seconds"], 1)
    ]
    return {source["id"]: source}, windows


if __name__ == "__main__":
    sources, _ = ami_dev_sources()
    print(json.dumps(sources["ami-is1008b"]["import_audit"], indent=2))
