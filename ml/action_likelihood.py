"""Training-only, frozen-adapter action and argument likelihood diagnosis."""

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def token_parts(text, offsets):
    spans = {}
    for name in ("action", "target", "text"):
        pattern = rf'"{name}":("(?:[^"\\]|\\.)*"|\[[^\]]*\])'
        match = re.search(pattern, text)
        if match:
            spans[name] = match.span(1)
    return [
        next(
            (name for name, (lo, hi) in spans.items() if start < hi and end > lo),
            "syntax",
        )
        for start, end in offsets
    ]


def prepare(archive, output):
    if output.exists():
        raise ValueError("Use a new diagnostic bundle; never overwrite frozen inputs.")
    root = archive / "input-bundle"
    manifest = json.loads((root / "bundle.json").read_text())
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Archived input bundle changed: {name}")
    sys.path.insert(0, str(root.resolve()))
    from train_adapter import MODEL, REVISION, training_row
    from model_input import token_ids
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        archive / "adapter", local_files_only=True
    )
    prompt = (root / "prompt.txt").read_text()
    cases = [
        case
        for case in read_rows(root / "train-cases.jsonl")
        if case["expectedOutput"]["action"] in ("correct_text", "no_action")
    ]
    if Counter(c["expectedOutput"]["action"] for c in cases) != {
        "correct_text": 64,
        "no_action": 64,
    }:
        raise ValueError(
            "This diagnostic is frozen to all 128 Qwen05 training targets."
        )
    previous = {r["case_id"]: r for r in read_rows(archive / "adapter-train.jsonl")}
    branches = {
        action: tokenizer.encode('{"action":"' + action, add_special_tokens=False)
        for action in ("correct_text", "no_action")
    }
    shared = 0
    for left, right in zip(*branches.values()):
        if left != right:
            break
        shared += 1
    rows = []
    for case in cases:
        metadata = {**case["metadata"], "language": case["input"]["language"]}
        if metadata["split"] != "train" or not metadata["training_eligible"]:
            raise ValueError("Only archived eligible training inputs are permitted.")
        prefix = token_ids(tokenizer, case["input"], prompt)
        reference = json.dumps(
            case["expectedOutput"], ensure_ascii=False, separators=(",", ":")
        )
        encoded = tokenizer(
            reference, add_special_tokens=False, return_offsets_mapping=True
        )
        completion = encoded["input_ids"] + [tokenizer.eos_token_id]
        original = training_row(tokenizer, case, prompt)
        if (
            original["input_ids"] != prefix + completion
            or original["labels"] != [-100] * len(prefix) + completion
        ):
            raise ValueError(
                "Diagnostic differs from the archived training serializer."
            )
        action = case["expectedOutput"]["action"]
        if completion[: len(branches[action])] != branches[action]:
            raise ValueError(
                "Action probe tokenization is not a prefix of the reference."
            )
        alternative_text = (
            '{"action":"no_action"}'
            if action == "correct_text"
            else '{"action":"correct_text'
        )
        alternative = tokenizer.encode(alternative_text, add_special_tokens=False)
        if action == "correct_text":
            alternative += [tokenizer.eos_token_id]
        other = "no_action" if action == "correct_text" else "correct_text"
        if alternative[: len(branches[other])] != branches[other]:
            raise ValueError("Counterfactual action tokenization differs.")
        rows.append(
            {
                "case_id": case["id"],
                "metadata": metadata,
                "input_reference_sha256": hashlib.sha256(
                    json.dumps(
                        [case["input"], case["expectedOutput"]],
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "reference": case["expectedOutput"],
                "prefix_ids": prefix,
                "reference_ids": completion,
                "reference_parts": token_parts(reference, encoded["offset_mapping"])
                + ["eos"],
                "alternative_ids": alternative,
                "alternative_action": other,
                "alternative_is_complete_response": action == "correct_text",
                "saved_greedy_text": previous[case["id"]]["text"],
                "saved_greedy_prediction": previous[case["id"]].get("prediction"),
            }
        )
    output.mkdir(parents=True)
    (output / "rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    shutil.copytree(archive / "adapter", output / "adapter")
    shutil.copy2(__file__, output / "action_likelihood.py")
    save(
        output / "protocol.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question": "Is Qwen05 excessive abstention associated with weak action likelihood, argument fit, or decoder/runtime disagreement?",
            "model": MODEL,
            "revision": REVISION,
            "max_length": 4096,
            "archive": str(archive),
            "archive_bundle_sha256": digest(root / "bundle.json"),
            "archive_predictions_sha256": digest(archive / "adapter-train.jsonl"),
            "counts": {
                "train": 128,
                "dev": 0,
                "test": 0,
                "unique_input_reference_pairs": len(
                    {r["input_reference_sha256"] for r in rows}
                ),
            },
            "branch_ids": branches,
            "shared_action_prefix_tokens": shared,
            "rules": [
                "Frozen saved adapter; no optimizer, updates, truncation, new targets or changed prompt.",
                "All 64 corrections and 64 abstentions; retain historical row weights and also report unique pairs.",
                "Teacher-forced token NLL, reference versus complete no_action on corrections; length-sensitive sums and means are separate.",
                "Action comparison sums only each action value token branch after their identical structural prefix; no target/text tokens.",
                "The two-branch normalized probability is not calibrated correctness confidence or a deployment threshold.",
                "Target/text NLL is conditioned on the reference action and preceding reference tokens; it does not prove free generation succeeds.",
                "Token categories use overlap with serialized JSON values, including quotes and any shared boundary token; syntax/EOS separately.",
                "Greedy predictions rerun on all 128 inputs to audit equality to archived outputs and runtime drift.",
                "Select one follow-up intervention only after the complete diagnosis; no held-out score or population-quality claim.",
            ],
        },
    )
    save(
        output / "manifest.json",
        {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {
                "bundle": str(output),
                "rows": len(rows),
                "unique_pairs": len({r["input_reference_sha256"] for r in rows}),
            }
        )
    )


def nll_summary(values):
    return {
        "tokens": len(values),
        "sum_nll": sum(values),
        "mean_nll": statistics.mean(values) if values else None,
    }


def summarize(rows):
    groups = {}
    for key in (
        "all",
        "reference_action",
        "language",
        "scenario_variant",
        "source_group_id",
        "saved_action",
    ):
        buckets = {}
        for row in rows:
            if key == "all":
                value = "all"
            elif key in ("reference_action", "saved_action"):
                value = row[key]
            else:
                value = row["metadata"].get(key, "unspecified")
            buckets.setdefault(value, []).append(row)
        groups[key] = {
            value: {
                "count": len(bucket),
                "mean_correction_minus_abstention_logp": statistics.mean(
                    r["correction_minus_abstention_logp"] for r in bucket
                ),
                "correction_branch_preferred": sum(
                    r["correction_minus_abstention_logp"] > 0 for r in bucket
                ),
                "greedy_matches_archive": sum(
                    r["greedy_matches_archive"] for r in bucket
                ),
                "reference_parts": {
                    part: nll_summary(
                        [
                            v
                            for r in bucket
                            for v, p in zip(
                                r["reference_token_nll"], r["reference_parts"]
                            )
                            if p == part
                        ]
                    )
                    for part in ("action", "target", "text", "syntax", "eos")
                },
            }
            for value, bucket in buckets.items()
        }
    return groups


def run(root, output):
    import os
    import subprocess
    import time

    for name, expected in json.loads((root / "manifest.json").read_text()).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Diagnostic input hash mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    rows = read_rows(root / "rows.jsonl")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    from unsloth import FastLanguageModel
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use the approved BF16 L4 or equivalent GPU.")
    started = time.time()
    info = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "gpu": torch.cuda.get_device_name(),
        "training_steps": 0,
    }
    save(output / "run.json", info)
    (output / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    model, _ = FastLanguageModel.from_pretrained(
        model_name=protocol["model"],
        revision=protocol["revision"],
        max_seq_length=4096,
        load_in_4bit=False,
        load_in_16bit=True,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, str(root / "adapter"), is_trainable=False)
    tokenizer = AutoTokenizer.from_pretrained(root / "adapter", local_files_only=True)
    FastLanguageModel.for_inference(model)
    model.eval()

    def score(prefix, completion):
        ids = torch.tensor([prefix + completion], device="cuda")
        if ids.shape[1] > 4096:
            raise ValueError("No silent truncation.")
        with torch.inference_mode():
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=len(completion) + 1,
            ).logits
            # Each preceding position predicts the next completion token.
            selected = logits[0, -(len(completion) + 1) : -1].float()
            if selected.shape[0] != len(completion):
                raise ValueError("Runtime returned an incompatible logit window.")
            values = (
                torch.nn.functional.cross_entropy(
                    selected, torch.tensor(completion, device="cuda"), reduction="none"
                )
                .cpu()
                .tolist()
            )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite token likelihood.")
        return values

    results = []
    for index, row in enumerate(rows):
        reference_nll = score(row["prefix_ids"], row["reference_ids"])
        alternative_nll = score(row["prefix_ids"], row["alternative_ids"])
        action = row["reference"]["action"]
        branch_logp = {}
        for branch, values in (
            (action, reference_nll),
            (row["alternative_action"], alternative_nll),
        ):
            branch_logp[branch] = -sum(
                values[
                    protocol["shared_action_prefix_tokens"] : len(
                        protocol["branch_ids"][branch]
                    )
                ]
            )
        difference = branch_logp["correct_text"] - branch_logp["no_action"]
        ids = torch.tensor([row["prefix_ids"]], device="cuda")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[0, ids.shape[1] :].tolist()
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        result = {
            key: row[key]
            for key in (
                "case_id",
                "metadata",
                "input_reference_sha256",
                "reference_parts",
            )
        }
        result.update(
            {
                "reference_action": action,
                "saved_action": (row["saved_greedy_prediction"] or {}).get(
                    "action", "invalid"
                ),
                "reference_token_nll": reference_nll,
                "reference_nll": nll_summary(reference_nll),
                "alternative_token_nll": alternative_nll,
                "alternative_nll": nll_summary(alternative_nll),
                "alternative_is_complete_response": row[
                    "alternative_is_complete_response"
                ],
                "action_branch_logp": branch_logp,
                "correction_minus_abstention_logp": difference,
                "two_branch_correction_probability": 1
                / (1 + math.exp(-max(-700, min(700, difference)))),
                "greedy_text": raw,
                "greedy_token_ids": generated,
                "greedy_matches_archive": raw == row["saved_greedy_text"],
            }
        )
        with (output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)
        print(f"{index + 1}/{len(rows)} {row['case_id']} done", flush=True)
    unique = list({r["input_reference_sha256"]: r for r in results}.values())
    save(
        output / "summary.json",
        {"row_weighted": summarize(results), "unique_pairs": summarize(unique)},
    )
    info.update(
        status="complete",
        seconds=time.time() - started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        model_class=type(model.base_model.model).__name__,
    )
    save(output / "run.json", info)
    save(
        output / "manifest.json",
        {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "seconds": info["seconds"],
                "rows": len(results),
                "greedy_matches_archive": sum(
                    r["greedy_matches_archive"] for r in results
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.archive, args.output)
    else:
        run(args.bundle, args.output)
