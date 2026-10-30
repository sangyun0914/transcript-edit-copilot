"""Run the October24 stored-evidence checks in a fresh Colab CPU workspace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not Path("/content").is_dir() or not root.is_relative_to("/content"):
        raise ValueError("Run this audit on Colab, not on the shared local Mac.")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "version": "review-response-m0-20261024-1",
        "status": "running",
        "started_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "workspace": str(root),
        "local_inference": False,
        "model_calls": 0,
        "training_updates": 0,
        "reserved_final_test_bodies_read": False,
        "source_sha256": {
            f"ml/{name}": sha256(root / "ml" / name)
            for name in (
                "run_review_response_m0.py",
                "report_actions_v3_diagnostics.py",
                "test_report_actions_v3_diagnostics.py",
                "prepare_dpo_supplement_actions_v3.py",
                "test_prepare_dpo_supplement_actions_v3.py",
            )
        },
        "commands": [],
    }
    (output / "source").mkdir()
    for name in record["source_sha256"]:
        shutil.copyfile(root / name, output / "source" / Path(name).name)

    def save():
        (output / "execution.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )

    def execute(name, arguments):
        command = [sys.executable, *arguments]
        started = time.time()
        print(f"Starting {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.run(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                check=False,
            )
        record["commands"].append({
            "name": name,
            "argv": command,
            "exit_code": process.returncode,
            "elapsed_seconds": time.time() - started,
            "log": f"{name}.log",
        })
        save()
        if process.returncode:
            raise RuntimeError(f"{name} failed; preserve the log and use a fresh attempt.")
        print(f"Completed {name}", flush=True)

    save()
    try:
        for name in ("report_actions_v3_diagnostics", "prepare_dpo_supplement_actions_v3"):
            execute(f"test-{name}", [
                "-m", "unittest", "discover", "-s", "ml", "-p", f"test_{name}.py", "-v"
            ])
        base = "artifacts/evaluations/passive-actions-v3"
        diagnostic_args = [
            "ml/report_actions_v3_diagnostics.py",
            "--release", "artifacts/datasets/nep-passive-actions-v3/release-attempt-003",
            "--output", str(output / "diagnostics"),
        ]
        for regime, versions in (
            ("native", ("base", "historical", "matched", "full", "targeted")),
            ("local", ("base", "matched", "full", "targeted")),
        ):
            for version in versions:
                name = f"{regime}-{version}"
                prefix = f"{base}/{regime}-{version}-json"
                observation = f"{prefix}-attempt-001"
                if regime == "local":
                    observation += "/run"
                diagnostic_args.extend([
                    "--run", name, f"{prefix}-grade-attempt-001", observation
                ])
        execute("diagnostics", diagnostic_args)
        execute("dpo-supplement", [
            "ml/prepare_dpo_supplement_actions_v3.py",
            "--input-root", str(root), "--output", str(output / "dpo-supplement")
        ])
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        record["finished_unix"] = time.time()
        save()
        files = {
            str(path.relative_to(output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.rglob("*")) if path.is_file() and path != output / "manifest.json"
        }
        (output / "manifest.json").write_text(json.dumps({
            "version": record["version"], "status": record["status"], "files": files,
            "scope": "Remote CPU tests and stored-evidence analysis; no new model inference."
        }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.root, args.output)
