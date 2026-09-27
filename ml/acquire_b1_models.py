"""Acquire the preregistered public B1 GGUFs, checking identity before publication."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "artifacts/model-preparation/qwen35-v1"
PINS = {
    "4B": {
        "repository": "unsloth/Qwen3.5-4B-GGUF",
        "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
        "filename": "Qwen3.5-4B-Q4_K_M.gguf",
        "bytes": 2740937888,
        "sha256": "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
    },
    "9B": {
        "repository": "unsloth/Qwen3.5-9B-GGUF",
        "revision": "3885219b6810b007914f3a7950a8d1b469d598a5",
        "filename": "Qwen3.5-9B-Q4_K_M.gguf",
        "bytes": 5680522464,
        "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
    },
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire(client, url, destination, expected_bytes, expected_sha256):
    if destination.exists():
        if destination.stat().st_size != expected_bytes or digest(destination) != expected_sha256:
            raise ValueError(f"Existing model differs; refusing to overwrite {destination}")
        return "verified_existing"
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise ValueError("Partial download exceeds pinned size")
    if offset < expected_bytes:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            if offset and response.status_code == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", ""))
                if not match or int(match[1]) != offset or int(match[3]) != expected_bytes:
                    raise ValueError("Resume response does not match the pinned byte range")
            elif response.status_code == 200:
                offset = 0
            else:
                raise ValueError(f"Unexpected download status {response.status_code}")
            with partial.open("ab" if offset else "wb") as stream:
                next_report = offset + 256 * 1024 * 1024
                for chunk in response.iter_bytes(1024 * 1024):
                    offset += len(chunk)
                    if offset > expected_bytes:
                        raise ValueError("Download exceeds pinned size")
                    stream.write(chunk)
                    if offset >= next_report:
                        print(f"{destination.name}: {offset}/{expected_bytes} bytes", flush=True)
                        next_report = offset + 256 * 1024 * 1024
    if partial.stat().st_size != expected_bytes or digest(partial) != expected_sha256:
        raise ValueError("Downloaded bytes do not match the pinned size and SHA256; partial retained")
    partial.rename(destination)
    return "downloaded_and_verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", choices=PINS)
    args = parser.parse_args()
    pin = PINS[args.size]
    api = PREPARATION / f"{args.size}-api.json"
    metadata = json.loads(api.read_text())
    entry = next(row for row in metadata["siblings"] if row["rfilename"] == pin["filename"])
    if (metadata["id"] != pin["repository"] or metadata["sha"] != pin["revision"]
            or entry["lfs"]["sha256"] != pin["sha256"] or entry["size"] != pin["bytes"]):
        raise ValueError("Saved Hugging Face metadata differs from the preregistered pin")
    destination = ROOT / "ml/models" / pin["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{pin['repository']}/resolve/{pin['revision']}/{pin['filename']}"
    with httpx.Client(follow_redirects=True, timeout=90, trust_env=False) as client:
        status = acquire(client, url, destination, pin["bytes"], pin["sha256"])
        for filename in ("README.md",):
            target = PREPARATION / f"{args.size}-{filename}"
            if not target.exists():
                response = client.get(f"https://huggingface.co/{pin['repository']}/raw/{pin['revision']}/{filename}")
                response.raise_for_status()
                target.write_bytes(response.content)
    receipt = {
        **pin, "status": status, "url": url,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(destination.relative_to(ROOT)),
        "metadata_path": str(api.relative_to(ROOT)), "metadata_sha256": digest(api),
        "model_card_path": str(target.relative_to(ROOT)), "model_card_sha256": digest(target),
        "acquisition_code_sha256": digest(Path(__file__)),
        "declared_native_base": metadata.get("cardData", {}).get("base_model"),
        "license": metadata.get("cardData", {}).get("license"),
        "lineage_limit": "Publisher metadata declares the native base repository, not the exact native weight commit or quantization recipe. This receipt pins the quantized artifact, not a verified conversion from the frozen native revision.",
        "selection": "Q4_K_M selected before inference for both newly acquired 4B/9B candidates; existing 26B/27B quantizations are retained. This is a deployment-artifact comparison.",
    }
    path = PREPARATION / f"{args.size}-receipt.json"
    if path.exists():
        raise ValueError(f"Receipt already exists; keep immutable acquisition evidence: {path}")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"size": args.size, "status": status, "sha256": pin["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
