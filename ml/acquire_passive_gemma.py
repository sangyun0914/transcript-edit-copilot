"""Pin the native Gemma candidate without changing the earlier Qwen selection."""

import argparse
import hashlib
import json
from pathlib import Path

import httpx

from acquire_passive_sft import download, write_new
from acquire_b1_models import digest


REPOSITORY = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def pin(registry_path, output):
    registry = json.loads(registry_path.read_text())
    identity = next(c for c in registry["candidates"] if c["id"] == "gemma26")
    if (identity["base_repository"], identity["base_revision"]) != (
        REPOSITORY,
        REVISION,
    ):
        raise ValueError("Gemma identity differs from the frozen comparison registry.")
    output.mkdir(parents=True, exist_ok=False)
    native = output / "native"
    native.mkdir()
    with httpx.Client(follow_redirects=True, timeout=90, trust_env=False) as client:
        response = client.get(
            f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}",
            params={"blobs": "true"},
        )
        response.raise_for_status()
        metadata = response.json()
        if metadata["id"] != REPOSITORY or metadata["sha"] != REVISION:
            raise ValueError("Native API identity mismatch.")
        required = {
            *identity["tokenizer_files_sha256"],
            "README.md",
            "generation_config.json",
            "model.safetensors.index.json",
        }
        if not required <= {entry["rfilename"] for entry in metadata["siblings"]}:
            raise ValueError("Native revision lacks required metadata.")
        write_new(output / "native-api.json", metadata)
        files = {}
        for entry in metadata["siblings"]:
            name = entry["rfilename"]
            if name.startswith("."):
                continue
            if Path(name).name != name:
                raise ValueError("Unexpected nested native file.")
            if name.endswith(".safetensors"):
                if entry["size"] != entry["lfs"]["size"]:
                    raise ValueError("Native shard size mismatch.")
                files[name] = {"bytes": entry["size"], "sha256": entry["lfs"]["sha256"]}
                continue
            if entry["size"] > 40_000_000:
                raise ValueError("Unexpected large metadata file.")
            response = client.get(
                f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
            )
            response.raise_for_status()
            content = response.content
            actual = hashlib.sha256(content).hexdigest()
            if len(content) != entry["size"]:
                raise ValueError(f"Metadata size mismatch: {name}")
            if entry.get("lfs"):
                if actual != entry["lfs"]["sha256"]:
                    raise ValueError(f"Metadata SHA256 mismatch: {name}")
            elif (
                hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                != entry["blobId"]
            ):
                raise ValueError(f"Metadata Git blob mismatch: {name}")
            if (
                name in identity["tokenizer_files_sha256"]
                and actual != identity["tokenizer_files_sha256"][name]
            ):
                raise ValueError(f"Frozen tokenizer identity mismatch: {name}")
            (native / name).write_bytes(content)
            files[name] = {"bytes": len(content), "sha256": actual}
    index = json.loads((native / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != {
        name for name in files if name.endswith(".safetensors")
    }:
        raise ValueError("Native index and shard inventory differ.")
    if metadata.get("cardData", {}).get("license") != "apache-2.0":
        raise ValueError("Pinned license metadata changed.")
    receipt = {
        "version": "passive-native-pin-v1",
        "candidate_id": "gemma26",
        "repository": REPOSITORY,
        "revision": REVISION,
        "registry_sha256": digest(registry_path),
        "api_sha256": digest(output / "native-api.json"),
        "files": files,
        "native_weight_bytes": sum(
            v["bytes"] for k, v in files.items() if k.endswith(".safetensors")
        ),
        "generation_config_present": True,
        "acquired_weight_files": 0,
        "license": {
            "identifier": "apache-2.0",
            "snapshot": "native/README.md",
            "standalone_license_present": "LICENSE" in files,
        },
        "implementation_sha256": digest(Path(__file__)),
    }
    write_new(output / "native-pin.json", receipt)
    print(
        json.dumps(
            {
                "status": "metadata_pinned",
                "files": len(files),
                "weight_bytes": receipt["native_weight_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    metadata = commands.add_parser("pin")
    metadata.add_argument("--registry", required=True, type=Path)
    metadata.add_argument("--output", required=True, type=Path)
    weights = commands.add_parser("download")
    weights.add_argument("--pin", required=True, type=Path)
    weights.add_argument("--pin-sha256", required=True)
    weights.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "pin":
        pin(args.registry, args.output)
    else:
        download(args.pin, args.output, args.pin_sha256)
