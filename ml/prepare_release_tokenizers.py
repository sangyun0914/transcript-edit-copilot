"""Pin public tokenizer/template files for the planned comparison without weights."""

import argparse
import json
import urllib.request
from pathlib import Path

from dataset_release_inventory import ROOT, digest, write
from train_adapter import MODEL, REVISION

CANDIDATES = {
    "qwen06": MODEL,
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen35-9b": "Qwen/Qwen3.5-9B",
    "gemma26": "google/gemma-4-26B-A4B-it",
    "qwen27": "Qwen/Qwen3.8-27B",
}
ALLOWED = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "config.json",
    "tokenizer.model",
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read(100_000_001)
    if len(content) > 100_000_000:
        raise ValueError("Tokenizer file exceeds the registered 100 MB limit.")
    return content


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    registry = {
        "version": "nep-release-comparison-tokenizers-v1",
        "weights_downloaded": False,
        "selection_basis": "Five candidates already specified in the September 25 execution plan; no performance-based selection.",
        "candidates": [],
    }
    write(directory / "selection.json", {**registry, "requested": CANDIDATES})
    for identity, repository in CANDIDATES.items():
        entry = {"id": identity, "repository": repository}
        try:
            info = json.loads(fetch(f"https://huggingface.co/api/models/{repository}"))
            revision = REVISION if identity == "qwen06" else info["sha"]
            entry["revision"] = revision
            target = directory / identity
            target.mkdir()
            files = sorted({s["rfilename"] for s in info["siblings"]} & ALLOWED)
            for name in files:
                (target / name).write_bytes(
                    fetch(
                        f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
                    )
                )
            if (
                not (target / "tokenizer_config.json").exists()
                or not (target / "tokenizer.json").exists()
            ):
                raise ValueError("Required tokenizer files are absent.")
            entry.update(
                status="downloaded_not_runtime_verified",
                path=str(target.relative_to(ROOT)),
                files_sha256={
                    p.name: digest(p) for p in target.iterdir() if p.is_file()
                },
            )
        except Exception as error:
            entry.update(status="unavailable", error=f"{type(error).__name__}: {error}")
        registry["candidates"].append(entry)
        print(
            json.dumps({k: v for k, v in entry.items() if k != "files_sha256"}),
            flush=True,
        )
    write(directory / "registry.json", registry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.directory.resolve())
