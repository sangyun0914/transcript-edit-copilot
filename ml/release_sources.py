"""Import registered Scribe excerpts without fabricating word times or silence evidence."""

from uuid import NAMESPACE_URL, uuid5

from dataset_release_inventory import CALCULUS, ROOT, digest, read, resolve_local
from session_cases import document_records


def scribe_document(content, source_hash, offset, identity, language):
    result = []

    def uid(name):
        return str(uuid5(NAMESPACE_URL, f"{identity}/{name}"))

    for index, paragraph in enumerate(content):
        result.append(
            {
                "type": "paragraph_start",
                "uuid": uid(f"p{index}"),
                "speaker": paragraph["speaker"] or None,
                "language": language,
            }
        )
        for position, item in enumerate(paragraph["content"]):
            if item["length"] <= 0 or item["type"] not in {"word", "silence"}:
                raise ValueError(
                    "Only positive normalized timed items may be imported."
                )
            value = {
                "uuid": uid(f"p{index}/i{position}"),
                "source": source_hash,
                "sourceStart": offset + item["sourceStart"],
                "length": item["length"],
                "type": "text" if item["type"] == "word" else "non_text",
            }
            if item["type"] == "word":
                value.update(text=item["word"], conf=item.get("conf"))
            result.append(value)
        result.append({"type": "paragraph_end", "uuid": uid(f"e{index}")})
    return {"content": result, "metadata": {}}


def calculus_sources():
    folder = ROOT / CALCULUS
    acquisition = read(folder / "acquisition.json")
    registration = read(folder / "registration.json")
    if (
        registration["split"] != "train"
        or not registration["selection_before_transcript_inspection"]
    ):
        raise ValueError("Calculus source allocation must precede inspection.")
    if (
        digest(ROOT / "data/manifests/calculus-30min-v1.json")
        != acquisition["registration_sha256"]
    ):
        raise ValueError("Transcription allocation changed.")
    record = next(
        r
        for r in read(ROOT / registration["source_manifest"])["files"]
        if r["sha256"] == registration["source_sha256"]
    )
    video = resolve_local(ROOT, "data/" + record["path"])
    if digest(video) != registration["source_sha256"]:
        raise ValueError("Calculus video changed.")
    sources, windows = {}, []
    for clip in acquisition["clips"]:
        path = folder / clip["result"]
        if digest(path) != clip["result_sha256"]:
            raise ValueError("Scribe result changed.")
        transcript = read(path)
        document = scribe_document(
            transcript["content"],
            clip["media_sha256"],
            clip["start_seconds"],
            clip["id"],
            "ko",
        )
        records, mapping = document_records(document)
        source = {
            "id": clip["id"],
            "source_id": clip["id"],
            "parent_recording_id": registration["source_id"],
            "source_group_id": registration["source_group_id"],
            "reserved_split": "train",
            "language": "ko",
            "video": video,
            "video_sha256": clip["media_sha256"],
            "transcript_path": path,
            "transcript_sha256": clip["result_sha256"],
            "transcript_origin": "elevenlabs_scribe_v2_normalized_with_retained_raw_response",
            "itemization": "original_scribe_items_with_registered_source_offset_no_retiming",
            "speaker_origin": "scribe_diarization_unverified",
            "license": registration["license"],
            "document": document,
            "records": records,
            "mapping": mapping,
            "annotations": transcript["transcription"]["annotations"],
            "time_offset": clip["start_seconds"],
        }
        sources[clip["id"]] = source
        windows.append(
            {
                "id": clip["id"] + "-window-01",
                "source_id": clip["id"],
                "start_seconds": clip["start_seconds"],
                "end_seconds": clip["start_seconds"] + clip["duration_seconds"],
            }
        )
    return sources, windows
