"""Acquire one preassigned AMI meeting; extract no other meeting transcripts."""

import argparse
import json
import re
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import httpx

from download_sources import digest, download


def acquire(catalog_path, output):
    catalog = json.loads(catalog_path.read_text())
    if catalog["meeting_id"] != "ES2002a" or catalog["split"] != "train":
        raise ValueError("This bounded acquisition is assigned to ES2002a training.")
    selection = output / "selection-response.html"
    urls = sorted(
        set(
            re.findall(
                r'https://groups\.inf\.ed\.ac\.uk/ami/AMICorpusMirror/[^\s\\<>"\']+\.wav',
                selection.read_text(),
            )
        )
    )
    if {url.rsplit("/", 1)[1] for url in urls} != {
        "ES2002a.Mix-Headset.wav",
        *[f"ES2002a.Headset-{i}.wav" for i in range(4)],
    }:
        raise ValueError(
            "Unexpected signal selection; do not execute downloaded scripts."
        )
    urls += [
        catalog["annotation_url"],
        "https://groups.inf.ed.ac.uk/ami/download/CCBY4.0.txt",
    ]
    sizes = {}
    headers = {"Accept-Encoding": "identity"}
    with httpx.Client(follow_redirects=True, timeout=90, headers=headers) as client:
        for url in urls:
            response = client.head(url)
            response.raise_for_status()
            sizes[url] = int(response.headers["content-length"])
    if sum(sizes.values()) > catalog["maximum_download_bytes"]:
        raise ValueError("The selected sources exceed the predeclared download bound.")

    def fetch(url):
        path = output / url.rsplit("/", 1)[1]
        with httpx.Client(follow_redirects=True, timeout=90, headers=headers) as client:
            record = download(client, url, path, max_bytes=sizes[url])
        if record["bytes"] != sizes[url]:
            raise ValueError(f"Source size changed during acquisition: {path}")
        return path.name, record

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = dict(pool.map(fetch, urls))
    annotation = output / catalog["annotation_url"].rsplit("/", 1)[1]
    extracted = {}
    with zipfile.ZipFile(annotation) as archive:
        members = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).name.startswith("ES2002a.") and name.endswith(".xml")
        ]
        for name in members:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe publisher archive member path.")
            target = output / "annotations" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            extracted[name] = digest(target)
    if not any("words" in name for name in extracted):
        raise ValueError("Selected meeting has no word annotation.")
    signals = {}
    for name in files:
        if not name.endswith(".wav"):
            continue
        with wave.open(str(output / name)) as source:
            signals[name] = {
                "sample_rate": source.getframerate(),
                "channels": source.getnchannels(),
                "frames": source.getnframes(),
                "duration_seconds": source.getnframes() / source.getframerate(),
            }
    result = {
        "catalog": catalog,
        "catalog_sha256": digest(catalog_path),
        "selection_sha256": digest(selection),
        "files": files,
        "signals": signals,
        "extracted_annotation_sha256": extracted,
        "other_meeting_annotation_bodies_opened": False,
        "admitted_cases": 0,
        "new_scribe_requests": 0,
    }
    (output / "acquisition.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "bytes": sum(r["bytes"] for r in files.values()),
                "signals": signals,
                "selected_annotation_files": len(extracted),
                "admitted_cases": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.catalog, args.output)
