"""Represent AMI overlap once on the mixed-audio timeline, retaining annotations."""

import copy
import wave
import xml.etree.ElementTree as ET
from uuid import NAMESPACE_URL, uuid5

from dataset_release_inventory import ROOT, digest, read
from replay_session import session_hash
from session_cases import document_records

FOLDER = ROOT / "data/raw/ami/ami-pilot-v1"
NITE_ID = "{http://nite.sourceforge.net/}id"


def annotation_records(folder):
    records = []
    for path in sorted((folder / "annotations/words").glob("ES2002a.*.words.xml")):
        speaker = path.name.split(".")[1]
        previous_word = None
        for position, element in enumerate(ET.parse(path).getroot()):
            if not {"starttime", "endtime"} <= element.attrib.keys():
                continue
            record = {
                "id": element.attrib[NITE_ID],
                "speaker": speaker,
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


def ami_document(records, source_hash, duration):
    positive = [r for r in records if r["end"] > r["start"]]
    if any(r["start"] < 0 or r["end"] > duration for r in records):
        raise ValueError("Annotation falls outside the mixed recording.")
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
    previous_speakers = None
    paragraph = -1

    def uid(name):
        return str(uuid5(NAMESPACE_URL, f"ami-es2002a-mixed-v1/{name}"))

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


def ami_sources():
    acquisition = read(FOLDER / "acquisition.json")
    catalog = acquisition["catalog"]
    if catalog["split"] != "train" or catalog["source_group_id"] != "ami-es2002-series":
        raise ValueError("AMI's original meeting-series split changed.")
    for name, expected in acquisition["extracted_annotation_sha256"].items():
        if digest(FOLDER / "annotations" / name) != expected:
            raise ValueError("AMI annotation changed.")
    media = FOLDER / "ES2002a.Mix-Headset.wav"
    media_hash = acquisition["files"][media.name]["sha256"]
    if digest(media) != media_hash:
        raise ValueError("AMI mixed audio changed.")
    with wave.open(str(media)) as stream:
        duration = stream.getnframes() / stream.getframerate()
    annotations = annotation_records(FOLDER)
    document, evidence, lineage = ami_document(annotations, media_hash, duration)
    records, mapping = document_records(document, evidence)
    source = {
        "id": catalog["source_id"],
        "source_group_id": catalog["source_group_id"],
        "reserved_split": "train",
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
    }
    windows = [
        {
            "id": f"ami-es2002a-window-{index:02d}",
            "source_id": source["id"],
            "start_seconds": start,
            "end_seconds": start + catalog["window_duration_seconds"],
        }
        for index, start in enumerate(catalog["window_start_seconds"], 1)
    ]
    return {source["id"]: source}, windows
