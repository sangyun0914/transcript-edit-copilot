"""Acquire preregistered AMI sources without generating edit labels or briefs."""

import argparse
import hashlib
import json
import math
import wave
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, uuid5

import httpx

from ami_dev_release import source_audit
from download_sources import download
from replay_session import session_hash
from session_cases import document_records


ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data/manifests/passive-expansion-sources-v1.json"
CATALOG_SHA256 = "7bccdaa300b00133862847451d7a556c17c220df67cb7dce5017bf95fcdb7ddc"
OUTPUT = ROOT / "artifacts/datasets/nep-passive-v1"
NITE_ID = "{http://nite.sourceforge.net/}id"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text() != body:
            raise ValueError(f"Refusing to overwrite different acquisition evidence: {path}")
        return
    with path.open("x") as stream:
        stream.write(body)


def copy_once(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"Original publisher bytes changed: {path}")
        return
    with path.open("xb") as stream:
        stream.write(body)


def load_registration():
    if digest(CATALOG) != CATALOG_SHA256:
        raise ValueError("The acquisition preregistration changed.")
    catalog = read(CATALOG)
    for field in ("annotation_archive", "metadata", "license"):
        if digest(ROOT / catalog[field + "_path"]) != catalog[field + "_sha256"]:
            raise ValueError(f"Preregistered {field} changed.")
    selected = catalog["sources"]
    for index, source in enumerate(selected):
        if source["split"] not in {"train", "dev"}:
            raise ValueError("This acquisition never reads a reserved test source.")
        for other in selected[index + 1:] + catalog["existing_ami_assignments"]:
            if set(source["series_global_speaker_ids"]) & set(other["series_global_speaker_ids"]):
                raise ValueError("Selected source parents share a known AMI speaker.")
    return catalog


def annotation_records(folder, meeting_id):
    records = []
    for path in sorted((folder / "annotations/words").glob(f"{meeting_id}.*.words.xml")):
        previous_word = None
        for position, element in enumerate(ET.parse(path).getroot()):
            if not {NITE_ID, "starttime", "endtime"} <= element.attrib.keys():
                raise ValueError("Native annotation lacks an ID or original timing.")
            record = {
                "id": element.attrib[NITE_ID],
                "speaker": path.name.split(".")[1],
                "position": position,
                "start": float(element.attrib["starttime"]),
                "end": float(element.attrib["endtime"]),
                "text": "".join(element.itertext()),
                "tag": element.tag,
                "attributes": dict(element.attrib),
                "original_file": str(path.relative_to(ROOT)),
            }
            records.append(record)
            if element.attrib.get("punc") == "true" and previous_word is not None:
                previous_word["display"] += record["text"]
                previous_word.setdefault("punctuation_ids", []).append(record["id"])
            elif element.tag == "disfmarker" and previous_word is not None:
                previous_word["display"] += "-"
                previous_word.setdefault("disfluency_ids", []).append(record["id"])
            elif element.tag == "w":
                record["display"] = record["text"]
                previous_word = record
    if not records:
        raise ValueError("No selected native word annotations found.")
    return sorted(records, key=lambda r: (r["start"], r["speaker"], r["position"]))


def make_document(source, annotations, media_hash, duration):
    for record in annotations:
        if not all(math.isfinite(record[k]) and 0 <= record[k] <= duration for k in ("start", "end")):
            raise ValueError(f"Native annotation is outside the recording: {record['id']}")
        if record["tag"] == "w" and record["attributes"].get("punc") != "true" and record["end"] <= record["start"]:
            raise ValueError(f"Native lexical annotation has nonpositive timing: {record['id']}")
    clusters = []
    for record in sorted((r for r in annotations if r["end"] > r["start"]), key=lambda r: (r["start"], r["end"])):
        if clusters and record["start"] < clusters[-1]["end"] - 1e-8:
            clusters[-1]["end"] = max(clusters[-1]["end"], record["end"])
            clusters[-1]["records"].append(record)
        else:
            clusters.append({"start": record["start"], "end": record["end"], "records": [record]})
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
        return str(uuid5(NAMESPACE_URL, f"nep-passive-source-v1/{source['id']}/{name}"))

    for index, cluster in enumerate(timeline):
        members = cluster["records"]
        speakers = tuple(sorted({r["speaker"] for r in members})) or previous_speakers or ()
        if paragraph < 0 or speakers != previous_speakers:
            if paragraph >= 0:
                content.append({"type": "paragraph_end", "uuid": uid(f"e{paragraph}")})
            paragraph += 1
            content.append({"type": "paragraph_start", "uuid": uid(f"p{paragraph}"), "speaker": speakers[0] if len(speakers) == 1 else None, "language": "en"})
        words = [r for r in members if r["tag"] == "w" and r["attributes"].get("punc") != "true"]
        text = "\n".join(
            (f"{speaker}: " if len(speakers) > 1 else "")
            + " ".join(r.get("display", r["text"]) for r in sorted(words, key=lambda r: r["position"]) if r["speaker"] == speaker)
            for speaker in speakers if any(r["speaker"] == speaker for r in words)
        )
        item = {"uuid": uid(f"i{index}"), "type": "text" if words else "non_text", "source": media_hash, "sourceStart": cluster["start"], "length": cluster["end"] - cluster["start"]}
        if words:
            item.update(text=text, conf=1)
        content.append(item)
        evidence[item["uuid"]] = {"item_hash": session_hash(item), "boundary_status": "overlap" if len({r["speaker"] for r in members}) > 1 else "unverified", "audio_class": "speech" if words else "audio_event" if members else "unknown", "origin": "publisher_annotations_not_perceptual_cut_verification"}
        lineage[item["uuid"]] = {"start": cluster["start"], "end": cluster["end"], "annotation_ids": [r["id"] for r in members], "speakers": sorted({r["speaker"] for r in members}), "events": [r for r in members if r["tag"] != "w"]}
        previous_speakers = speakers
    content.append({"type": "paragraph_end", "uuid": uid(f"e{paragraph}")})
    return {"content": content, "metadata": {}}, evidence, lineage


def candidate_windows(source, document, lineage, duration, rule):
    items = [item for item in document["content"] if "sourceStart" in item]
    windows, start = [], rule["window_start_seconds"]
    while duration - start >= rule["minimum_remaining_seconds"]:
        end = min(duration, start + rule["window_duration_seconds"])
        visible = [item for item in items if item["sourceStart"] < end and item["sourceStart"] + item["length"] > start]
        focus = min(visible, key=lambda item: abs(item["sourceStart"] - (start + end) / 2))
        windows.append({"id": f"{source['id']}-window-{len(windows) + 1:03d}", "start_seconds": start, "end_seconds": end, "focus_uuid": focus["uuid"], "item_uuids": [item["uuid"] for item in visible], "native_annotation_ids": [identity for item in visible for identity in lineage[item["uuid"]]["annotation_ids"]], "text_items": sum(item["type"] == "text" for item in visible), "status": "unlabeled_source_candidate", "boundary_intersecting_items_are_not_clipped": True})
        start += rule["window_stride_seconds"]
    return {"source_id": source["id"], "source_group_id": source["source_group_id"], "split": source["split"], "method": rule, "document_sha256": session_hash(document), "candidate_count": len(windows), "windows": windows, "accepted_cases": 0, "perceptually_qualified_cuts": 0}


def acquire_one(catalog, source, download_bytes):
    folder = (ROOT / source["document_path"]).parent
    acquisition_path = ROOT / source["acquisition_path"]
    if acquisition_path.exists():
        previous = read(acquisition_path)
        if previous["registration_sha256"] != CATALOG_SHA256:
            raise ValueError("Acquisition uses a different registration.")
        for relative, expected in previous["file_sha256"].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f"Acquired source changed: {relative}")
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    start_path = folder / "acquisition-start.json"
    if not start_path.exists():
        write_once(start_path, {"started_at": datetime.now(timezone.utc).isoformat(), "registration_path": str(CATALOG.relative_to(ROOT)), "registration_sha256": CATALOG_SHA256, "source": source, "pre_extraction_probe": "A native annotation timing-health probe was performed after preregistration; no pre-registration transcript selection or semantic filtering."})
    original_files = []
    with zipfile.ZipFile(ROOT / catalog["annotation_archive_path"]) as archive:
        names = [name for name in archive.namelist() if PurePosixPath(name).parts[0] in {"words", "segments"} and PurePosixPath(name).name.startswith(source["meeting_id"] + ".") and name.endswith(".xml")]
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive member.")
            target = folder / "annotations" / relative
            copy_once(target, archive.read(name))
            original_files.append(target)
    for field, name in (("metadata_path", "meetings.xml"), ("license_path", "CCBY4.0.txt")):
        target = folder / name
        copy_once(target, (ROOT / catalog[field]).read_bytes())
        original_files.append(target)
    media = ROOT / source["media_path"]
    with httpx.Client(follow_redirects=True, timeout=120, headers={"Accept-Encoding": "identity"}) as client:
        media_record = download(client, source["media_url"], media, max_bytes=download_bytes)
    if media_record["bytes"] != download_bytes:
        raise ValueError("Downloaded media size differs from its publisher HEAD metadata.")
    with wave.open(str(media)) as stream:
        duration = stream.getnframes() / stream.getframerate()
        media_info = {"sample_rate": stream.getframerate(), "channels": stream.getnchannels(), "frames": stream.getnframes(), "sample_width_bytes": stream.getsampwidth(), "duration_seconds": duration}
    annotations = annotation_records(folder, source["meeting_id"])
    native_path = folder / "native-annotations.json"
    write_once(native_path, annotations)
    files = [*original_files, media, media.with_suffix(media.suffix + ".json"), native_path]
    result = {**source, "registration_path": str(CATALOG.relative_to(ROOT)), "registration_sha256": CATALOG_SHA256, "acquired_at": datetime.now(timezone.utc).isoformat(), "media": media_info, "media_download": media_record, "native_annotation_count": len(annotations), "native_tags": dict(Counter(r["tag"] for r in annotations)), "lexical_word_count": sum(r["tag"] == "w" and r["attributes"].get("punc") != "true" for r in annotations), "admitted_training_rows": 0, "listening_performed": False, "qualified_destructive_cuts": 0, "adapter_sha256": digest(Path(__file__)), "dependency_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (ROOT / "ml/ami_dev_release.py", ROOT / "ml/session_cases.py", ROOT / "ml/replay_session.py")}}
    try:
        document, evidence, lineage = make_document(source, annotations, media_record["sha256"], duration)
        records, _ = document_records(document, evidence)
        audit = source_audit({"native_annotations": annotations, "document": document, "boundary_evidence": evidence, "item_annotations": lineage}, duration)
        write_once(ROOT / source["document_path"], document)
        write_once(ROOT / source["evidence_path"], evidence)
        write_once(ROOT / source["lineage_path"], {"source_id": source["id"], "source_group_id": source["source_group_id"], "media_sha256": media_record["sha256"], "native_annotations_path": str(native_path.relative_to(ROOT)), "native_annotations_sha256": digest(native_path), "items": lineage, "adapter_audit": audit, "representation_limits": ["Overlap islands retain the union audio interval and per-speaker text; they are not retimed sequentially.", "Punctuation and disfluency markers are attached to the preceding same-channel word for display; original XML IDs and bytes remain intact.", "Native point/reversed event annotations remain in native-annotations.json without invented duration.", "No publisher-empty interval or manually transcribed token certifies a safe audio cut."]})
        candidates = candidate_windows(source, document, lineage, duration, catalog["candidate_generation"])
        write_once(ROOT / source["candidates_path"], candidates)
        files.extend(ROOT / source[field] for field in ("document_path", "evidence_path", "lineage_path", "candidates_path"))
        result.update(status="acquired_source_document_ready_unlabeled", document_hash=session_hash(document), document_record_count=len(records), candidate_count=candidates["candidate_count"], adapter_audit=audit)
    except ValueError as error:
        result.update(status="raw_acquired_document_withheld", adapter_failure=str(error), candidate_count=0)
    result["file_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in files}
    write_once(acquisition_path, result)
    print(json.dumps({"id": source["id"], "status": result["status"], "duration": duration, "words": result["lexical_word_count"], "candidates": result["candidate_count"]}), flush=True)
    return result


def publish_registry(catalog):
    acquired = [read(ROOT / source["acquisition_path"]) for source in catalog["sources"] if (ROOT / source["acquisition_path"]).exists()]
    ready = [source for source in acquired if source["status"] == "acquired_source_document_ready_unlabeled"]
    value = {"version": "passive-expansion-acquired-v1", "registration_path": str(CATALOG.relative_to(ROOT)), "registration_sha256": CATALOG_SHA256, "source_count": len(ready), "sources": ready, "withheld_sources": [source for source in acquired if source not in ready], "pending_source_ids": [source["id"] for source in catalog["sources"] if not (ROOT / source["acquisition_path"]).exists()], "admitted_cases": 0, "total_ready_duration_seconds": sum(source["media"]["duration_seconds"] for source in ready), "total_ready_native_words": sum(source["lexical_word_count"] for source in ready), "candidate_count": sum(source["candidate_count"] for source in ready)}
    target = OUTPUT / "sources/acquired-registry-v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    catalog = load_registration()
    selected = [source for source in catalog["sources"] if not args.source_id or source["id"] in args.source_id]
    if args.source_id and len(selected) != len(set(args.source_id)):
        raise ValueError("Unknown or duplicate selected source ID.")
    sizes = {}
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        for source in catalog["sources"]:
            sidecar = (ROOT / source["media_path"]).with_suffix(".wav.json")
            if sidecar.exists():
                sizes[source["id"]] = read(sidecar)["bytes"]
            else:
                response = client.head(source["media_url"])
                response.raise_for_status()
                sizes[source["id"]] = int(response.headers["content-length"])
            if not 0 < sizes[source["id"]] <= source["maximum_download_bytes"]:
                raise ValueError("Publisher file size exceeds the source download bound.")
    if sum(sizes.values()) > catalog["maximum_total_download_bytes"]:
        raise ValueError("Publisher files exceed the preregistered total download bound.")
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire_one, catalog, source, sizes[source["id"]]): source["id"] for source in selected}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append({"source_id": futures[future], "error": repr(error)})
                print(json.dumps(errors[-1]), flush=True)
            publish_registry(catalog)
    summary = publish_registry(catalog)
    print(json.dumps({"ready_sources": summary["source_count"], "withheld_sources": len(summary["withheld_sources"]), "pending_sources": summary["pending_source_ids"], "duration_hours": summary["total_ready_duration_seconds"] / 3600, "native_words": summary["total_ready_native_words"], "candidates": summary["candidate_count"], "errors": errors}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
