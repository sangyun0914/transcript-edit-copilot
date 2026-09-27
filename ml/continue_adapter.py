"""One frozen 44-step second stage, initialized from the archived Qwen05 adapter."""

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def verify(root):
    for name, expected in json.loads((root / "manifest.json").read_text()).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"Frozen continuation input changed: {name}")


def prepare(archive, source, output):
    if output.exists():
        raise ValueError("Use a fresh bundle; frozen inputs cannot be overwritten.")
    sys.path.insert(0, str((archive / "input-bundle").resolve()))
    from train_adapter import MODEL, REVISION, load_bundle, prompt_ids, training_row
    from transformers import AutoTokenizer

    train, dev, regression = load_bundle(archive / "input-bundle")
    if (len(train), len(dev), len(regression)) != (176, 88, 17):
        raise ValueError("The protocol is fixed to the complete Qwen05 bundle.")
    source_cases = read_rows(source / "dev-cases.jsonl")
    if len(source_cases) != 4 or any(
        c["metadata"]["split"] != "dev" for c in source_cases
    ):
        raise ValueError("Expected the four separate source development cases.")
    source_manifest = json.loads((source / "manifest.json").read_text())
    for name in ("dev-cases.jsonl", "prompt.txt", "executions.jsonl"):
        if sha(source / name) != source_manifest["files_sha256"][name]:
            raise ValueError(f"Source inputs changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        archive / "adapter", local_files_only=True
    )
    prompt = (archive / "input-bundle/prompt.txt").read_text()
    rows = [training_row(tokenizer, c, prompt) for c in train]
    parity = {
        c["id"]: {
            "prefix_sha256": fingerprint(prompt_ids(tokenizer, c, prompt)),
            "training_row_sha256": fingerprint(row),
            "input_tokens": len(row["input_ids"]),
        }
        for c, row in zip(train, rows)
    }
    output.mkdir(parents=True)
    shutil.copytree(archive / "input-bundle", output / "archive")
    shutil.copytree(archive / "adapter", output / "initial-adapter")
    (output / "control").mkdir()
    for split in ("train", "dev", "regression"):
        for suffix in (".jsonl", "-report.json"):
            name = f"adapter-{split}{suffix}"
            shutil.copy2(archive / name, output / "control" / name)
    (output / "source").mkdir()
    for name in ("dev-cases.jsonl", "prompt.txt"):
        shutil.copy2(source / name, output / "source" / name)
    ids = {c["id"] for c in source_cases}
    source_executions = [
        r for r in read_rows(source / "executions.jsonl") if r["id"] in ids
    ]
    if len(source_executions) != 4:
        raise ValueError("Missing source execution evidence.")
    (output / "source/executions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in source_executions)
    )
    save(output / "training-parity.json", parity)
    # Choose checks from the pre-existing order and references, before new outputs.
    reload_ids = []
    seen = set()
    for case in dev:
        key = (case["input"]["language"], case["expectedOutput"]["action"])
        if key not in seen:
            seen.add(key)
            reload_ids.append(case["id"])
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "Does one additional exposure improve useful corrections without losing restraint?",
        "model": MODEL,
        "revision": REVISION,
        "initial_adapter_sha256": sha(archive / "adapter/adapter_model.safetensors"),
        "archived_bundle_sha256": sha(archive / "input-bundle/bundle.json"),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "steps": 44,
        "previous_steps": 44,
        "optimizer": "Fresh AdamW8bit; not optimizer-state resumption",
        "rank": 16,
        "alpha": 16,
        "dropout": 0,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 2,
        "weight_decay": 0.01,
        "schedule": "linear",
        "seed": 3407,
        "max_length": 4096,
        "output_reserve": 256,
        "training_rows": 176,
        "unique_training_pairs": len(
            {fingerprint([c["input"], c["expectedOutput"]]) for c in train}
        ),
        "reload_case_ids": reload_ids,
        "criteria": {
            "correct_text_greater_than": 8,
            "no_action_at_least": 31,
            "valid_at_least": 85,
            "trim_silence_at_least": 8,
            "remove_silence_at_least": 4,
        },
        "scope": "One diagnostic; unchanged training weights, inputs and labels. No early selection, test or promotion.",
    }
    save(output / "protocol.json", protocol)
    shutil.copy2(__file__, output / "continue_adapter.py")
    save(
        output / "manifest.json",
        {
            str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    verify(output)
    print(
        json.dumps(
            {
                "bundle": str(output),
                "training_rows_verified": len(rows),
                "protocol": protocol,
            }
        )
    )


def run(root, output, reload_only=False):
    verify(root)
    sys.path.insert(0, str((root / "archive").resolve()))
    from train_adapter import (
        MODEL,
        REVISION,
        MAX_LENGTH,
        MAX_NEW_TOKENS,
        SEED,
        load_bundle,
        prompt_ids,
        training_row,
        validate,
    )
    from evaluate import evaluate
    from grading import load_executions

    # Unsloth must patch the training stack before importing Torch/Transformers.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    started = time.time()
    protocol = json.loads((root / "protocol.json").read_text())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA GPU is required.")
    set_seed(SEED)
    train, dev, regression = load_bundle(root / "archive")
    prompt = (root / "archive/prompt.txt").read_text()
    source_prompt = (root / "source/prompt.txt").read_text()
    source_cases = read_rows(root / "source/dev-cases.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(root / "initial-adapter")
    executions = load_executions(root / "archive/eval-executions.jsonl")
    source_executions = load_executions(root / "source/executions.jsonl")
    rows = [training_row(tokenizer, c, prompt) for c in train]
    parity = json.loads((root / "training-parity.json").read_text())
    for case, row in zip(train, rows):
        expected = parity[case["id"]]
        if (
            fingerprint(row) != expected["training_row_sha256"]
            or fingerprint(prompt_ids(tokenizer, case, prompt))
            != expected["prefix_sha256"]
        ):
            raise ValueError("Archived training prefix/label parity failed.")
    for cases, text in (
        (train + dev + regression, prompt),
        (source_cases, source_prompt),
    ):
        for case in cases:
            if len(prompt_ids(tokenizer, case, text)) + MAX_NEW_TOKENS > MAX_LENGTH:
                raise ValueError(f"Input exceeds budget: {case['id']}")
    if not reload_only:
        output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(root / "protocol.json", output / "protocol.json")
        (output / "environment.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        )
    model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        revision=REVISION,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        dtype=torch.bfloat16,
    )

    def predict(cases, text):
        FastLanguageModel.for_inference(model)
        for case in cases:
            ids = torch.tensor([prompt_ids(tokenizer, case, text)], device="cuda")
            before = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0, ids.shape[1] :].tolist()
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            finish = (
                "stop"
                if generated and generated[-1] == tokenizer.eos_token_id
                else "length"
            )
            result = {
                "case_id": case["id"],
                "text": raw,
                "generated_ids": generated,
                "finish_reason": finish,
                "seconds": time.time() - before,
                "valid": False,
            }
            try:
                result.update(
                    prediction=validate(case, raw, finish).model_dump(), valid=True
                )
            except ValueError as error:
                result["error"] = str(error)
            yield result

    def measure(label, cases, text=prompt, evidence=executions):
        results = []
        with (output / f"{label}.jsonl").open("x") as stream:
            for i, row in enumerate(predict(cases, text), 1):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                results.append(row)
                print(label, i, len(cases), row["case_id"], flush=True)
        report = evaluate(cases, results, evidence)
        save(output / f"{label}-report.json", report)
        print(
            label,
            "outcomes",
            report["outcome_matches"],
            "valid",
            report["valid"],
            flush=True,
        )
        return results, report

    chosen = {c["id"]: c for c in dev}
    checks = [chosen[key] for key in protocol["reload_case_ids"]]
    if reload_only:
        weights = load_file(output / "adapter/adapter_model.safetensors")
        if not all(torch.isfinite(v).all().item() for v in weights.values()):
            raise ValueError("Saved adapter has non-finite weights.")
        model = PeftModel.from_pretrained(
            model, str(output / "adapter"), is_trainable=False
        )
        previous = {r["case_id"]: r for r in read_rows(output / "adapter-dev.jsonl")}
        results = list(predict(checks, prompt))
        comparison = [
            {
                "case_id": r["case_id"],
                "matches_saved": r["generated_ids"]
                == previous[r["case_id"]]["generated_ids"],
                "text": r["text"],
            }
            for r in results
        ]
        save(
            output / "reload-check.json",
            {
                "cases": comparison,
                "all_weights_finite": True,
                "seconds": time.time() - started,
            },
        )
        if not all(r["matches_saved"] for r in comparison):
            raise ValueError("Saved adapter generation differs after fresh reload.")
        print("Fresh reload verified", len(comparison), flush=True)
        return

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    archived_config = json.loads(
        (root / "initial-adapter/adapter_config.json").read_text()
    )
    if model.peft_config["default"].target_modules != archived_config["target_modules"]:
        raise ValueError("Adapter module selection differs from Qwen05.")
    weights = load_file(root / "initial-adapter/adapter_model.safetensors")
    set_peft_model_state_dict(model, weights)
    loaded = get_peft_model_state_dict(model)
    if set(loaded) != set(weights) or any(
        not torch.equal(loaded[k].detach().cpu(), v) for k, v in weights.items()
    ):
        raise ValueError(
            "Initial adapter tensors differ from the saved Qwen05 weights."
        )
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if (
        not trainable
        or any("lora_" not in n for n in trainable)
        or sum(trainable.values()) != sum(v.numel() for v in weights.values())
    ):
        raise ValueError("Unexpected trainable parameters.")
    save(
        output / "initialization.json",
        {
            "all_saved_adapter_tensors_equal": True,
            "trainable_parameters": sum(trainable.values()),
            "trainable_tensors": trainable,
            "training_rows_with_exact_prefix_and_labels": len(rows),
        },
    )
    control = {r["case_id"]: r for r in read_rows(root / "control/adapter-dev.jsonl")}
    initial = list(predict(checks, prompt))
    save(output / "control-reload.json", initial)
    if any(r["text"] != control[r["case_id"]]["text"] for r in initial):
        raise ValueError(
            "Qwen05 control reload does not reproduce archived predictions."
        )
    del loaded, weights
    gc.collect()
    torch.cuda.empty_cache()
    FastLanguageModel.for_training(model)
    set_seed(SEED)

    class FiniteGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                with (output / "training-events.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": state.global_step, **logs}) + "\n")
                for key in ("loss", "grad_norm"):
                    if key in logs and not math.isfinite(float(logs[key])):
                        raise RuntimeError(f"Non-finite {key}; reject this attempt.")

    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
        callbacks=[FiniteGuard()],
        args=TrainingArguments(
            output_dir=str(output / "checkpoints"),
            max_steps=44,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=2,
            logging_steps=1,
            logging_nan_inf_filter=False,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            fp16=False,
            bf16=True,
            seed=SEED,
            data_seed=SEED,
            report_to="none",
            save_strategy="no",
        ),
    )
    info = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "protocol_sha256": sha(root / "protocol.json"),
        "optimizer_state_resumed": False,
    }
    save(output / "run.json", info)
    training = trainer.train()
    save(
        output / "training.json",
        {"metrics": training.metrics, "log": trainer.state.log_history},
    )
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    reports = {}
    for split, cases in (("train", train), ("dev", dev), ("regression", regression)):
        _, reports[split] = measure(f"adapter-{split}", cases)
    _, reports["source-dev"] = measure(
        "adapter-source-dev", source_cases, source_prompt, source_executions
    )
    dev_report = reports["dev"]
    outcomes = {a: v["outcome_matches"] for a, v in dev_report["by_action"].items()}
    criteria = {
        "corrections_improve": outcomes["correct_text"] > 8,
        "preservation_retained": outcomes["no_action"] >= 31,
        "no_additional_invalid": dev_report["valid"] >= 85,
        "pause_trims_retained": outcomes["trim_silence"] >= 8,
        "pause_removals_retained": outcomes["remove_silence"] >= 4,
    }
    save(
        output / "decision.json",
        {
            "criteria": criteria,
            "diagnostic_pass": all(criteria.values()),
            "promoted": False,
            "next": "Report this fixed stage; no dev-driven extra steps.",
        },
    )
    info.update(
        status="training_and_evaluation_complete_reload_pending",
        finished_at=datetime.now(timezone.utc).isoformat(),
        seconds=time.time() - started,
        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
    )
    save(output / "run.json", info)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("run", "reload"):
        command = sub.add_parser(name)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.archive, args.source, args.output)
    else:
        run(args.bundle, args.output, args.command == "reload")
