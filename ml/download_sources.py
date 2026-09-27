"""Download explicitly catalogued public videos/captions and verify local provenance."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import httpx


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(client, url, path, *, max_bytes=1024**3):
    record = path.with_suffix(path.suffix + ".json")
    if path.exists():
        previous = json.loads(record.read_text())
        if previous["url"] != url or previous["sha256"] != digest(path):
            raise ValueError(
                f"Existing source differs from its acquisition record: {path}"
            )
        return previous
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {path.name} from {url}", flush=True)
    with client.stream("GET", url) as response:
        response.raise_for_status()
        length = 0
        with temporary.open("wb") as stream:
            for chunk in response.iter_bytes(1024 * 1024):
                length += len(chunk)
                if length > max_bytes:
                    raise ValueError(
                        f"Source exceeds its {max_bytes}-byte acquisition limit."
                    )
                stream.write(chunk)
        content_type = response.headers.get("content-type")
        final_url = str(response.url)
    if not length:
        raise ValueError("Empty source download.")
    temporary.rename(path)
    result = {
        "url": url,
        "resolved_url": final_url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "bytes": length,
        "sha256": digest(path),
        "content_type": content_type,
    }
    record.write_text(json.dumps(result, indent=2) + "\n")
    return result


def acquire(catalog_path, output):
    catalog = json.loads(catalog_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    groups, ids = {}, set()
    for source in catalog["sources"]:
        if not re.fullmatch(r"[a-z0-9-]+", source["id"]) or source["id"] in ids:
            raise ValueError("Source IDs must be unique path-safe names.")
        ids.add(source["id"])
        group, split = source["source_group_id"], source["split"]
        if split not in ("train", "dev", "test") or groups.get(group, split) != split:
            raise ValueError("Each source group must stay in one split.")
        if group in catalog.get("reserved_test_groups", []) and split != "test":
            raise ValueError(
                "Reserved test source cannot enter development or training."
            )
        groups[group] = split
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        license_record = download(
            client, catalog["license_url"], output / "license.html"
        )
        records = []
        for source in catalog["sources"]:
            root = output / source["id"]
            root.mkdir(exist_ok=True)
            files = {}
            for name, field in (
                ("page.html", "page_url"),
                ("captions.vtt", "captions_url"),
                ("video.mp4", "video_url"),
            ):
                files[name] = download(client, source[field], root / name)
            if (
                not (root / "captions.vtt")
                .read_text()
                .lstrip("\ufeff")
                .startswith("WEBVTT")
            ):
                raise ValueError("Expected publisher WebVTT captions.")
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration,size:stream=codec_type,codec_name,width,height,sample_rate",
                        "-of",
                        "json",
                        str(root / "video.mp4"),
                    ],
                    text=True,
                )
            )
            if not {"video", "audio"} <= {s["codec_type"] for s in probe["streams"]}:
                raise ValueError("Downloaded lecture must contain audio and video.")
            record = {
                **source,
                "license": catalog["license"],
                "files": files,
                "media": probe,
            }
            (root / "source.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(
                f"Verified {source['id']}: {probe['format']['duration']} seconds",
                flush=True,
            )
    report = {
        "catalog_sha256": digest(catalog_path),
        "license": license_record,
        "sources": records,
        "scope": catalog["admission"],
    }
    (output / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.catalog, args.output)
