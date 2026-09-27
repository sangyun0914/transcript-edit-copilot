"""Bounded Colab LoRA experiment; no editor application or automatic model promotion."""

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from evaluate import evaluate, read_rows
from grading import load_executions
from iteration import messages
from model_input import token_ids
from nep_v2 import Envelope, Snapshot, validate_prediction

MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MAX_LENGTH = 4096
MAX_NEW_TOKENS = 256
SEED = 3407


def load_bundle(root):
    manifest = json.loads((root / "bundle.json").read_text())
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Bundle path escapes its directory.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    train = read_rows(root / "train-cases.jsonl")
    dev = read_rows(root / "dev-cases.jsonl")
    regression = read_rows(root / "regression-cases.jsonl")
    groups = json.loads((root / "groups.json").read_text())
    seen = set()
    for split, cases in (("train", train), ("dev", dev)):
        if not cases:
            raise ValueError(f"Empty {split} split.")
        for case in cases:
            metadata = case["metadata"]
            if case["id"] in seen:
                raise ValueError("Duplicate case ID.")
            seen.add(case["id"])
            if (
                metadata["split"] != split
                or groups[metadata["source_group_id"]] != split
            ):
                raise ValueError("Source group crossed a split.")
            if (
                metadata["actor"] not in ("synthetic", "ai_computer_use")
                or metadata["target_origin"] != "ai_generated_recipe"
                or metadata.get("requires_regeneration") is not False
                or metadata["review_status"] != "agent_validated_observable_rule"
                or metadata.get("training_eligible") is not (split == "train")
            ):
                raise ValueError("Unqualified or feedback-only case.")
            validate(case, json.dumps(case["expectedOutput"]), "stop")
    return train, dev, regression


def validate(case, text, finish):
    if case["input"].get("schema_version") in {3, 4}:
        from model_input import parse_snapshot, validate_prediction as validate_case

        snapshot = parse_snapshot(case["input"])
    else:
        validate_case = validate_prediction
        snapshot = Snapshot.model_validate(case["input"])
    request = Envelope(
        document_id=case["id"], request_id="adapter", revision=0, interaction_epoch=0
    )
    return validate_case(text, snapshot, request, request, finish)


def prompt_ids(tokenizer, case, prompt):
    return token_ids(tokenizer, case["input"], prompt)


def training_row(tokenizer, case, prompt):
    prefix = prompt_ids(tokenizer, case, prompt)
    answer = messages(case, prompt)["messages"][-1]["content"]
    completion = tokenizer.encode(answer, add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    if len(prefix) + len(completion) > MAX_LENGTH:
        raise ValueError(f"{case['id']}: exceeds context; do not silently truncate.")
    return {
        "input_ids": prefix + completion,
        "attention_mask": [1] * (len(prefix) + len(completion)),
        "labels": [-100] * len(prefix) + completion,
    }


def run(root, output, steps):
    if not 1 <= steps <= 50:
        raise ValueError("Use 1–50 steps for a bounded experiment.")
    train, dev, regression = load_bundle(root)
    output.mkdir(parents=True, exist_ok=False)
    # Unsloth must patch Transformers before other training imports.
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before training.")
    set_seed(SEED)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This Qwen3.5 experiment requires a BF16 GPU (e.g. L4 or A100)."
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        revision=REVISION,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        dtype=torch.bfloat16,
    )
    # Qwen3.5 loaders can return a multimodal processor even for text-only use.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer.eos_token_id is None:
        raise ValueError("The checkpoint must declare an EOS token.")
    prompt = (root / "prompt.txt").read_text()
    rows = [training_row(tokenizer, case, prompt) for case in train]

    for case in train + dev + regression:
        if len(prompt_ids(tokenizer, case, prompt)) + MAX_NEW_TOKENS > MAX_LENGTH:
            raise ValueError(f"{case['id']}: inference context exceeds the limit.")
    run_info = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "question": json.loads((root / "recipe.json").read_text())["question"],
        "case_counts": {
            "train": len(train),
            "dev": len(dev),
            "regression": len(regression),
        },
        "model": MODEL,
        "revision": REVISION,
        "seed": SEED,
        "steps": steps,
        "rank": 16,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_length": MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "thinking": False,
        "do_sample": False,
        "quantization": "none",
        "dtype": "bfloat16",
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "gpu": torch.cuda.get_device_name(),
        "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
        "bundle_sha256": hashlib.sha256(
            (root / "bundle.json").read_bytes()
        ).hexdigest(),
        "max_training_tokens": max(len(r["input_ids"]) for r in rows),
        "scope": "Tiny development experiment; no held-out quality or local latency claim",
    }

    def save(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )

    save("run.json", run_info)
    (output / "environment.txt").write_text(
        subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
    )
    tokenizer.save_pretrained(output / "tokenizer")
    executions = load_executions(root / "eval-executions.jsonl")

    def measure(label, cases):
        FastLanguageModel.for_inference(model)
        results = []
        with (output / f"{label}.jsonl").open("x") as stream:
            for case in cases:
                ids = torch.tensor([prompt_ids(tokenizer, case, prompt)], device="cuda")
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
                text = tokenizer.decode(generated, skip_special_tokens=True)
                finish = (
                    "stop"
                    if generated and generated[-1] == tokenizer.eos_token_id
                    else "length"
                )
                result = {
                    "case_id": case["id"],
                    "valid": False,
                    "text": text,
                    "finish_reason": finish,
                }
                try:
                    result.update(
                        prediction=validate(case, text, finish).model_dump(), valid=True
                    )
                except ValueError as error:
                    result["error"] = str(error)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                results.append(result)
        report = evaluate(cases, results, executions)
        save(f"{label}-report.json", report)
        print(label, {k: v for k, v in report.items() if k != "rows"})
        return report

    base_train = measure("base-train", train)
    base_dev = measure("base-dev", dev)
    base_regression = measure("base-regression", regression)
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
    FastLanguageModel.for_training(model)

    class FiniteGradientGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if (
                logs
                and "grad_norm" in logs
                and not math.isfinite(float(logs["grad_norm"]))
            ):
                raise RuntimeError(
                    "Non-finite gradients; reject this training attempt."
                )

    trainer = Trainer(
        model=model,
        train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
        callbacks=[FiniteGradientGuard()],
        args=TrainingArguments(
            output_dir=str(output / "checkpoints"),
            max_steps=steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=min(2, steps),
            logging_steps=1,
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
    training = trainer.train()
    run_info["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    save(
        "training.json", {"metrics": training.metrics, "log": trainer.state.log_history}
    )
    tuned_train = measure("adapter-train", train)
    tuned_dev = measure("adapter-dev", dev)
    tuned_regression = measure("adapter-regression", regression)
    save(
        "comparison.json",
        {
            "status": "inspect_failures_before_next_generation",
            "training_outcome_delta": tuned_train["outcome_matches"]
            - base_train["outcome_matches"],
            "training_scope": "In-sample fit diagnostic only; never generalization evidence",
            "development_outcome_delta": tuned_dev["outcome_matches"]
            - base_dev["outcome_matches"],
            "regression_outcome_delta": tuned_regression["outcome_matches"]
            - base_regression["outcome_matches"],
            "next": "Inspect false edits, missed edits and invalid outputs by scenario; revise the AI recipe, regenerate a small next batch, train/evaluate again. No automatic promotion.",
        },
    )
    run_info["status"] = (
        "training_and_development_evaluation_complete_local_export_pending"
    )
    run_info["finished_at"] = datetime.now(timezone.utc).isoformat()
    save("run.json", run_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    run(args.bundle, args.output, args.steps)
