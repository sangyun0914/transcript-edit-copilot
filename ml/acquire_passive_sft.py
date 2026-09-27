"""Pin and acquire native weights for the selected passive SFT feasibility run."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import httpx

from acquire_b1_models import acquire, digest


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def pin(selection, output):
    selected = json.loads(selection.read_text())
    if selected["selected_t1_candidate_ids"] != ["qwen35-9b"]:
        raise ValueError(
            "This acquisition requires the bound single-candidate T1 selection."
        )
    identity = selected["selected_native_identity"]
    repository, revision = identity["base_repository"], identity["base_revision"]
    if repository != "Qwen/Qwen3.5-9B" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Unexpected native identity.")
    output.mkdir(parents=True, exist_ok=False)
    native = output / "native"
    native.mkdir()
    with httpx.Client(follow_redirects=True, timeout=90, trust_env=False) as client:
        response = client.get(
            f"https://huggingface.co/api/models/{repository}/revision/{revision}",
            params={"blobs": "true"},
        )
        response.raise_for_status()
        metadata = response.json()
        if metadata["id"] != repository or metadata["sha"] != revision:
            raise ValueError("Native metadata identity mismatch.")
        required = {
            *identity["tokenizer_files_sha256"],
            "LICENSE",
            "model.safetensors.index.json",
        }
        if not required <= {entry["rfilename"] for entry in metadata["siblings"]}:
            raise ValueError(
                "Native revision lacks required model/tokenizer/license files."
            )
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
            if entry["size"] > 20_000_000:
                raise ValueError("Unexpected large metadata file.")
            response = client.get(
                f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
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
    receipt = {
        "version": "passive-native-pin-v1",
        "repository": repository,
        "revision": revision,
        "selection_sha256": digest(selection),
        "api_sha256": digest(output / "native-api.json"),
        "files": files,
        "native_weight_bytes": sum(
            v["bytes"] for k, v in files.items() if k.endswith(".safetensors")
        ),
        "generation_config_present": "generation_config.json" in files,
        "acquired_weight_files": 0,
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


def download(pin_path, output, expected_sha256):
    if digest(pin_path) != expected_sha256:
        raise ValueError("Native pin receipt differs from its execution binding.")
    receipt = json.loads(pin_path.read_text())
    if receipt["version"] != "passive-native-pin-v1":
        raise ValueError("Unknown native pin format.")
    output.mkdir(parents=True, exist_ok=True)
    repository, revision = receipt["repository"], receipt["revision"]
    with httpx.Client(follow_redirects=True, timeout=180, trust_env=False) as client:
        for name, item in receipt["files"].items():
            if Path(name).name != name:
                raise ValueError("Unsafe native path.")
            status = acquire(
                client,
                f"https://huggingface.co/{repository}/resolve/{revision}/{name}",
                output / name,
                item["bytes"],
                item["sha256"],
            )
            print(json.dumps({"file": name, "status": status}), flush=True)
    print(
        json.dumps({"status": "native_files_verified", "files": len(receipt["files"])}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    metadata = sub.add_parser("pin")
    metadata.add_argument("--selection", required=True, type=Path)
    metadata.add_argument("--output", required=True, type=Path)
    weights = sub.add_parser("download")
    weights.add_argument("--pin", required=True, type=Path)
    weights.add_argument("--pin-sha256", required=True)
    weights.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "pin":
        pin(args.selection, args.output)
    else:
        download(args.pin, args.output, args.pin_sha256)
