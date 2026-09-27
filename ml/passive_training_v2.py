"""Consume a frozen passive bundle with weighted sampling and completion-only labels."""

import argparse
import gzip
import json
from pathlib import Path

from passive_bundle_v2 import weighted_indices
from passive_dataset import digest


def training_epoch(release, candidate, format_name, *, seed, samples=None):
    release = Path(release)
    manifest = json.loads((release / "manifest.json").read_text())
    if manifest.get("version") != "passive-dataset-completion-v2":
        raise ValueError("Use a frozen completion-v2 bundle.")
    if format_name not in {"json-curriculum", "json", "rewrite"}:
        raise ValueError("Unknown training view.")
    if not candidate or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in candidate):
        raise ValueError("Use a pinned candidate ID.")
    case_name = "train-cases.jsonl" if format_name == "json-curriculum" else "matched/train-cases.jsonl"
    token_name = f"tokenized/{candidate}/{format_name}-train.jsonl.gz"
    for name in (case_name, token_name):
        expected = manifest["files_sha256"].get(name)
        if not expected or digest(release / name) != expected:
            raise ValueError("A frozen training payload is missing or changed.")
    cases = [json.loads(line) for line in (release / case_name).read_text().splitlines()]
    with gzip.open(release / token_name, "rt") as stream:
        encoded = [json.loads(line) for line in stream]
    if len(cases) != len(encoded):
        raise ValueError("Tokenized and semantic training sets differ.")
    for case, record in zip(cases, encoded, strict=True):
        if record["id"] != case["id"] or record["split"] != "train" or record["sampling_weight"] != case["metadata"]["sampling_weight"]:
            raise ValueError("Training rows/weights no longer match their semantic cases.")
        values = record["primary"]
        prefix = values["completion_start"]
        if values["labels"][:prefix] != [-100] * prefix or values["labels"][prefix:] != values["input_ids"][prefix:]:
            raise ValueError("Completion-only training mask is invalid.")
    indices = weighted_indices(cases, len(cases) if samples is None else samples, seed)
    # Sampling already applies the weight; multiplying loss by it again would square it.
    return [{"case_id": encoded[i]["id"], **encoded[i]["primary"]} for i in indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--format", choices=("json-curriculum", "json", "rewrite"), default="json-curriculum")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = training_epoch(args.release, args.candidate, args.format, seed=args.seed, samples=args.samples)
    with args.output.open("x") as stream:
        for row in result:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": len(result), "training_run": False, "output": str(args.output)}))
