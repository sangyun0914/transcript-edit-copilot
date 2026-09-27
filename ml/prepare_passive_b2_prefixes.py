"""Tokenize preregistered B2 messages with pinned local files; never run inference."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path

from replay_session import session_hash

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("full", "context_only", "history_only", "icl")


def common_input_eligibility(case_ids, token_counts, *, input_limit=3584):
    """Intersect full/ICL budgets across candidates without silently dropping slots."""
    if not case_ids or len(set(case_ids)) != len(case_ids) or not token_counts:
        raise ValueError("Require a unique nonempty roster and tokenizer measurements.")
    reasons = defaultdict(list)
    for candidate, conditions in token_counts.items():
        if set(conditions) != {"full", "icl"}:
            raise ValueError("Measure both full and ICL for each compared candidate.")
        for condition, counts in conditions.items():
            if set(counts) != set(case_ids):
                raise ValueError(
                    "Input-budget measurements must cover every registered case."
                )
            for identity, count in counts.items():
                if type(count) is not int or count < 0:
                    raise ValueError(
                        "Prefix token counts must be nonnegative integers."
                    )
                if count > input_limit:
                    reasons[identity].append(
                        {
                            "candidate": candidate,
                            "condition": condition,
                            "input_tokens": count,
                        }
                    )
    return {
        "registered_requests": len(case_ids),
        "input_limit": input_limit,
        "eligible_case_ids": [
            identity for identity in case_ids if identity not in reasons
        ],
        "excluded": [
            {
                "case_id": identity,
                "reason": "common_input_budget_exceeded",
                "measurements": reasons[identity],
            }
            for identity in case_ids
            if identity in reasons
        ],
        "policy": "Symmetric paired subset; no recropping, truncation or failure-based exclusion.",
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or digest(path) != expected:
        raise ValueError(f"Changed or unbound project file: {path}")
    return path


def rows(path):
    content = (
        gzip.decompress(Path(path).read_bytes())
        if str(path).endswith(".gz")
        else Path(path).read_bytes()
    )
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, values):
    Path(path).write_bytes(
        gzip.compress(
            "".join(
                json.dumps(value, ensure_ascii=False) + "\n" for value in values
            ).encode(),
            mtime=0,
        )
    )


def prepare_prefixes(protocol_path, release, output):
    from transformers import AutoTokenizer

    protocol_path, release, output = (
        Path(protocol_path).resolve(),
        Path(release).resolve(),
        Path(output).resolve(),
    )
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("version") != "passive-b2-preregistration-v1"
        or protocol.get("b2_predictions_consulted") is not False
    ):
        raise ValueError("Use the prediction-independent B2 preregistration.")
    if output.is_relative_to(release) or any(
        parent.name == "frozen" and parent.is_relative_to(ROOT / "artifacts/datasets")
        for parent in (output, *output.parents)
    ):
        raise ValueError("Prefix preparation belongs outside frozen releases.")
    checked(release / "manifest.json", protocol["release_manifest_sha256"])
    release_manifest = json.loads((release / "manifest.json").read_text())
    checked(
        protocol_path.parent / "case-registry.jsonl", protocol["case_registry_sha256"]
    )
    checked(protocol_path.parent / "requests.jsonl.gz", protocol["requests_sha256"])
    registration = {
        row["case_id"]: row
        for row in rows(protocol_path.parent / "case-registry.jsonl")
    }
    requests = defaultdict(dict)
    for row in rows(protocol_path.parent / "requests.jsonl.gz"):
        identity, condition = row["case_id"], row["condition"]
        if (
            condition not in CONDITIONS
            or identity not in registration
            or identity in requests[condition]
            or row["messages_sha256"] != session_hash(row["messages"])
            or registration[identity]["conditions"][condition]["messages_sha256"]
            != row["messages_sha256"]
        ):
            raise ValueError("Unregistered, duplicate or changed B2 request messages.")
        requests[condition][identity] = row
    ids = protocol["case_ids"]
    if (
        len(ids) != 289
        or len(set(ids)) != len(ids)
        or set(registration) != set(ids)
        or any(set(requests[condition]) != set(ids) for condition in CONDITIONS)
    ):
        raise ValueError("Every condition must retain all 289 registered states.")
    tokenizer_registry = checked(
        ROOT / protocol["tokenizer_registry"]["path"],
        protocol["tokenizer_registry"]["sha256"],
    )
    candidates = json.loads(tokenizer_registry.read_text())["candidates"]
    if len(candidates) != 5 or len({candidate["id"] for candidate in candidates}) != 5:
        raise ValueError("Use all five preregistered tokenizer candidates.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": "passive-b2-prefixes-v1",
        "status": "started",
        "protocol_sha256": digest(protocol_path),
        "case_registry_sha256": protocol["case_registry_sha256"],
        "requests_sha256": protocol["requests_sha256"],
        "release_manifest_sha256": protocol["release_manifest_sha256"],
        "preparation_code_sha256": digest(Path(__file__)),
        "case_ids": ids,
        "conditions": list(CONDITIONS),
        "candidates": candidates,
        "candidate_ids": [candidate["id"] for candidate in candidates],
        "tokenizer_registry": protocol["tokenizer_registry"],
        "python_packages": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers")
        },
        "local_files_only": True,
        "inference_requests": 0,
        "files_sha256": {},
        "full_frozen_parity": {},
        "prefix_counts": {},
        "common_icl_eligibility_file": "common-icl-eligibility.json",
    }
    write(output / "manifest.json", manifest)
    try:
        comparison_counts = {}
        for candidate in candidates:
            name = candidate["id"]
            path = (ROOT / candidate["path"]).resolve()
            for filename, expected in candidate["files_sha256"].items():
                checked(path / filename, expected)
            tokenizer = AutoTokenizer.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            frozen_name = f"tokenized/{name}/json-curriculum-dev.jsonl.gz"
            frozen_path = checked(
                release / frozen_name, release_manifest["files_sha256"][frozen_name]
            )
            frozen = {
                row["id"]: row["primary"]["input_ids"][
                    : row["primary"]["completion_start"]
                ]
                for row in rows(frozen_path)
            }
            if set(frozen) != set(ids):
                raise ValueError(
                    "Frozen full prefixes differ from the complete B2 roster."
                )
            (output / name).mkdir()
            comparison_counts[name], manifest["prefix_counts"][name] = {}, {}
            for condition in CONDITIONS:
                encoded = []
                for identity in ids:
                    request = requests[condition][identity]
                    tokens = tokenizer.apply_chat_template(
                        request["messages"],
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    if condition == "full" and tokens != frozen[identity]:
                        raise ValueError(
                            f"Full B2 prefix differs from frozen tokens: {name}/{identity}"
                        )
                    if condition != "icl" and len(tokens) > 3584:
                        raise ValueError(
                            f"Non-ICL condition exceeds the fixed input budget: {name}/{identity}"
                        )
                    encoded.append(
                        {
                            "case_id": identity,
                            "input_ids": tokens,
                            "messages_sha256": request["messages_sha256"],
                        }
                    )
                filename = f"{name}/{condition}.jsonl.gz"
                write_rows(output / filename, encoded)
                manifest["files_sha256"][filename] = digest(output / filename)
                counts = {row["case_id"]: len(row["input_ids"]) for row in encoded}
                manifest["prefix_counts"][name][condition] = counts
                if condition in {"full", "icl"}:
                    comparison_counts[name][condition] = counts
            manifest["full_frozen_parity"][name] = {
                "passed": len(ids),
                "expected": len(ids),
            }
            write(output / "manifest.json", manifest)
            print(
                f"{name}: all four B2 conditions tokenized; full frozen parity {len(ids)}/{len(ids)}",
                flush=True,
            )
        eligibility = common_input_eligibility(ids, comparison_counts)
        eligibility["candidate_ids"] = manifest["candidate_ids"]
        eligibility["candidate_basis"] = (
            "All five preregistered tokenizers, fixed before B2 candidate choice."
        )
        filename = manifest["common_icl_eligibility_file"]
        write(output / filename, eligibility)
        manifest["files_sha256"][filename] = digest(output / filename)
        manifest["common_icl_eligibility_sha256"] = digest(output / filename)
        manifest["common_icl_eligible_requests"] = len(eligibility["eligible_case_ids"])
        manifest["common_icl_excluded_requests"] = len(eligibility["excluded"])
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--release",
        type=Path,
        default=ROOT / "artifacts/datasets/nep-passive-completion-v2/frozen",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_prefixes(args.protocol, args.release, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "common_icl_eligible_requests",
                    "common_icl_excluded_requests",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
