"""Acquire the preregistered independent AMI development meeting only."""

import json
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import PurePosixPath

import httpx
from dataset_release_inventory import ROOT, digest, read, write
from download_sources import download


def acquire():
    catalog_path = ROOT / "data/manifests/ami-dev-v1.json"
    catalog = read(catalog_path)
    if (
        catalog["meeting_id"] != "IS1008b"
        or catalog["split"] != "dev"
        or catalog["source_group_id"] != "ami-is1008-series"
        or not catalog["selection_before_transcript_inspection"]
    ):
        raise ValueError("Unexpected preregistered development assignment.")
    archive_path = ROOT / catalog["local_annotation_archive"]
    if digest(archive_path) != catalog["annotation_archive_sha256"]:
        raise ValueError("The preregistered publisher archive changed.")
    folder = ROOT / "data/raw/ami/ami-dev-v1"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "acquisition.json").exists():
        raise ValueError("Existing acquisition must not be overwritten.")
    inspected_at = datetime.now(timezone.utc).isoformat()
    extracted = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).name.startswith("IS1008b.")
            and name.endswith(".xml")
            and PurePosixPath(name).parts[0] in {"words", "segments"}
        ]
        for name in members:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive member.")
            target = folder / "annotations" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            extracted[name] = digest(target)
        metadata = archive.read(catalog["metadata_member"])
        (folder / "meetings.xml").write_bytes(metadata)
    media_name = "IS1008b.Mix-Headset.wav"
    url = (
        "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/"
        f"/amicorpus/IS1008b/audio/{media_name}"
    )
    with httpx.Client(
        follow_redirects=True, timeout=90, headers={"Accept-Encoding": "identity"}
    ) as client:
        size_response = client.head(url)
        size_response.raise_for_status()
        size = int(size_response.headers["content-length"])
        if size > catalog["maximum_download_bytes"]:
            raise ValueError("The preregistered acquisition limit was exceeded.")
        media_record = download(client, url, folder / media_name, max_bytes=size)
    if media_record["bytes"] != size:
        raise ValueError("Media size changed during acquisition.")
    with wave.open(str(folder / media_name)) as stream:
        signal = {
            "sample_rate": stream.getframerate(),
            "channels": stream.getnchannels(),
            "frames": stream.getnframes(),
            "duration_seconds": stream.getnframes() / stream.getframerate(),
        }
    license_path = ROOT / "data/raw/ami/ami-pilot-v1/CCBY4.0.txt"
    result = {
        "catalog": catalog,
        "catalog_sha256": digest(catalog_path),
        "selected_annotation_extraction_completed_at": inspected_at,
        "archive_path": str(archive_path.relative_to(ROOT)),
        "archive_sha256": digest(archive_path),
        "metadata_sha256": digest(folder / "meetings.xml"),
        "files": {media_name: media_record},
        "signals": {media_name: signal},
        "license_path": str(license_path.relative_to(ROOT)),
        "license_sha256": digest(license_path),
        "extracted_annotation_sha256": extracted,
        "other_meeting_annotation_bodies_opened": False,
        "new_scribe_requests": 0,
        "listening_performed": False,
        "admitted_cases": 0,
    }
    write(folder / "acquisition.json", result)
    print(json.dumps({"signal": signal, "annotation_files": len(extracted)}))


if __name__ == "__main__":
    acquire()
