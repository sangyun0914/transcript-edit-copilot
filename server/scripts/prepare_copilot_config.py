"""Prepare a local gateway configuration from pinned assets, without starting a model."""

import argparse
import gzip
import hashlib
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from app.copilot import CONTRACT_MODULES, CONTRACT_VERSION, LocalConfig, contract  # noqa: E402


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(
    root,
    registry_path,
    release,
    candidate_id,
    output,
    port,
    contract_version=CONTRACT_VERSION,
):
    module = contract(contract_version)
    registry = json.loads(registry_path.read_text())
    candidates = [row for row in registry["candidates"] if row["id"] == candidate_id]
    if len(candidates) != 1:
        raise ValueError("Choose exactly one candidate in the local registry.")
    candidate, runtime = candidates[0], registry["runtime"]
    model = (root / candidate["model_path"]).resolve()
    template = (root / candidate["template_path"]).resolve()
    server = (root / runtime["server"]).resolve()
    if not model.is_file() or not server.is_file():
        raise ValueError("Pinned local model and llama-server must already exist.")
    if digest(template) != candidate["template_sha256"] or digest(server) != runtime["sha256"]:
        raise ValueError("Pinned native template or runtime launcher differs.")
    manifest = json.loads((release / "manifest.json").read_text())
    tokenizer_id = candidate.get("tokenizer_id", candidate_id)
    relative = f"tokenized/{tokenizer_id}/json-curriculum-train.jsonl.gz"
    path = release / relative
    if digest(path) != manifest["files_sha256"][relative]:
        raise ValueError("Frozen native-tokenizer training view differs from its manifest.")
    checks = {}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != "train":
                raise ValueError("Local parity setup must use train-only states.")
            snapshot = module.parse_snapshot(json.loads(row["messages"][1]["content"]))
            if snapshot.language not in {"en", "ko"} or snapshot.language in checks:
                continue
            if row["messages"] != module.canonical_messages(snapshot):
                raise ValueError("Frozen messages differ from the shared passive contract.")
            prefix = row["primary"]["input_ids"][: row["primary"]["completion_start"]]
            if not 1 <= len(prefix) <= 3584:
                raise ValueError("Frozen native prefix exceeds the passive input budget.")
            checks[snapshot.language] = {
                "case_id": row["id"],
                "messages": row["messages"],
                "expected_native_token_ids": prefix,
            }
            if set(checks) == {"en", "ko"}:
                break
    if set(checks) != {"en", "ko"}:
        raise ValueError("Frozen training view must supply both English and Korean parity states.")
    build = f"{runtime['tag']}-{runtime['revision'][:9]}"
    receipt = {
        "version": 1,
        "contract_version": contract_version,
        "model_sha256": candidate["model_sha256"],
        "chat_template_sha256": candidate["template_sha256"],
        "runtime_build": build,
        "token_source": "native_tokenizer",
        "checks": list(checks.values()),
        "source_registry_sha256": digest(registry_path),
        "source_manifest_sha256": digest(release / "manifest.json"),
        "source_tokenized_train_sha256": digest(path),
    }
    receipt_raw = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    output = output.resolve()
    config = LocalConfig(
        contract_version=contract_version,
        endpoint=f"http://127.0.0.1:{port}",
        model_id=candidate_id,
        model_path=str(model),
        model_sha256=candidate["model_sha256"],
        chat_template_path=str(template),
        chat_template_sha256=candidate["template_sha256"],
        runtime_build=build,
        token_parity_path=str(output / "native-token-parity.json"),
        token_parity_sha256=hashlib.sha256(receipt_raw.encode()).hexdigest(),
    )
    command = [
        str(server),
        "-m",
        str(model),
        "--alias",
        candidate_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "4096",
        "-np",
        "1",
        "-ngl",
        "99",
        "--jinja",
        "--no-ui",
        "--perf",
        "--metrics",
        "--cache-ram",
        "0",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--chat-template-file",
        str(template),
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    output.mkdir(parents=True, exist_ok=False)
    (output / "native-token-parity.json").write_text(receipt_raw)
    (output / "gateway.json").write_text(json.dumps(config.model_dump(), indent=2) + "\n")
    (output / "start-model.sh").write_text("#!/bin/sh\nset -eu\nexec " + shlex.join(command) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="qwen35-9b")
    parser.add_argument("--registry", type=Path, default=ROOT / "ml/passive-models-v2.json")
    parser.add_argument(
        "--release", type=Path, default=ROOT / "artifacts/datasets/nep-passive-completion-v2/frozen"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--contract-version", choices=tuple(CONTRACT_MODULES), default=CONTRACT_VERSION
    )
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local port between 1024 and 65535.")
    try:
        output = prepare(
            ROOT,
            args.registry,
            args.release,
            args.candidate,
            args.output,
            args.port,
            args.contract_version,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Prepared {output / 'gateway.json'}; no model process was started.")
    print(f"Review and run: sh {shlex.quote(str(output / 'start-model.sh'))}")
    print(f"Launch the app with AUDAPOLIS_NEP_CONFIG={shlex.quote(str(output / 'gateway.json'))}")


if __name__ == "__main__":
    main()
