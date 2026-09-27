"""Package only text cases and executable grading code for a Colab experiment."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from evaluate import read_rows
from grading import load_executions

HERE = Path(__file__).resolve().parent


def build(iteration, regression, output):
    source_manifest = json.loads((iteration / "manifest.json").read_text())
    for name, expected in source_manifest["files"].items():
        path = (iteration / name).resolve()
        if not path.is_relative_to(iteration.resolve()):
            raise ValueError("Iteration path escapes its directory.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Iteration hash mismatch: {name}")
    payload = {}
    for name in (
        "train-cases.jsonl",
        "dev-cases.jsonl",
        "groups.json",
        "prompt.txt",
        "recipe.json",
        "manifest.json",
    ):
        payload[name] = (iteration / name).read_bytes()
    payload["regression-cases.jsonl"] = (regression / "pilot-cases.jsonl").read_bytes()
    ids = {
        c["id"]
        for split in ("train", "dev")
        for c in read_rows(iteration / f"{split}-cases.jsonl")
    }
    executions = [
        r
        for k, r in load_executions(iteration / "executions.jsonl").items()
        if k in ids
    ]
    executions += list(load_executions(regression / "replay-fixtures.json").values())
    # Replay journals and original media stay local; grading needs only these fields.
    payload["eval-executions.jsonl"] = "".join(
        json.dumps(
            {
                k: r[k]
                for k in ("id", "before", "mapping", "caption_authority")
                if k in r
            },
            ensure_ascii=False,
        )
        + "\n"
        for r in executions
    ).encode()
    for name in (
        "train_adapter.py",
        "model_input.py",
        "iteration.py",
        "evaluate.py",
        "preference_diagnostic.py",
        "grading.py",
        "nep.py",
        "nep_v2.py",
        "nep_v3.py",
        "nep_v4.py",
        "caption_facts.py",
        "session_cases.py",
        "replay_session.py",
        "synthetic_pilot.py",
    ):
        payload[name] = (HERE / name).read_bytes()
    manifest = {
        "purpose": "Bounded text-only training/development loop with train-fit diagnostics",
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in payload.items()
        },
        "excluded": [
            "audio",
            "video",
            "full source project",
            "human review answers",
            "training replay journals",
        ],
    }
    payload["bundle.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.iteration, args.regression, args.output)
    print(f"Created {args.output} ({args.output.stat().st_size:,} bytes)")
