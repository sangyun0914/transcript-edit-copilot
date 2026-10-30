"""Prepare complete seven-action cohorts for pinned small local models.

This command downloads tokenizer/configuration assets only. Model loading and GPU
work live in train_small_actions_final.py; serving/export in serve_small_actions_final.py.
"""

import argparse
from collections import Counter
import gzip
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import urllib.request

import passive_formats_actions_v3 as formats

VERSION = "small-actions-final-v1"
PROFILES = {
    "lfm25-1.2b": {
        "repository": "LiquidAI/LFM2.5-1.2B-Instruct",
        "revision": "0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
        "architecture": "Lfm2ForCausalLM",
        "gguf_architecture": "lfm2",
        "terminal_ids": [7],
        "default_targets": "hybrid-ffn",
        "lora_alpha": 16,
        "official_gguf": {
            "repository": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
            "revision": "8ed288026e23958ad9dfa92d53ed773a8eee7125",
            "filename": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            "purpose": "base_speed_screen_only",
        },
        "sources": [
            "https://docs.liquid.ai/lfm/fine-tuning/trl",
            "https://unsloth.ai/docs/models/tutorials/lfm2.5",
        ],
    },
    "gemma4-e2b": {
        "repository": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "architecture": "Gemma4ForConditionalGeneration",
        "gguf_architecture": "gemma4",
        "terminal_ids": [1, 106, 50],
        "default_targets": "attention-ffn",
        "lora_alpha": 16,
        "official_gguf": {
            "repository": "google/gemma-4-E2B-it-qat-q4_0-gguf",
            "revision": "675cff42a74c774d6cb76f76d8eacb49b48c9b93",
            "filename": "gemma-4-E2B_q4_0-it.gguf",
            "purpose": "qat_base_speed_screen_only_not_native_adapter_base",
        },
        "sources": [
            "https://huggingface.co/docs/transformers/en/model_doc/gemma4",
            "https://unsloth.ai/docs/models/gemma-4",
        ],
    },
}
ASSET_FILES = (
    "config.json", "generation_config.json", "chat_template.jinja",
    "tokenizer_config.json", "tokenizer.json",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def value_hash(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, values):
    raw = "".join(compact(row) + "\n" for row in values).encode()
    Path(path).write_bytes(gzip.compress(raw, mtime=0) if str(path).endswith(".gz") else raw)


def new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    result = {}
    for name in ("torch", "transformers", "peft", "tokenizers", "safetensors", "accelerate"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def acquire(model_name, output):
    profile = PROFILES[model_name]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior = read(output / "assets.json") if (output / "assets.json").exists() else None
    if prior:
        verify_assets(output, model_name)
        return prior
    for filename in ASSET_FILES:
        url = f"https://huggingface.co/{profile['repository']}/resolve/{profile['revision']}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(80_000_001)
        require(len(content) <= 80_000_000, "Tokenizer asset exceeds the 80MB limit.")
        # Always fetch from the pin before creating a receipt for pre-existing files.
        (output / filename).write_bytes(content)
    require(read(output / "config.json")["architectures"] == [profile["architecture"]],
            "Unexpected native model architecture.")
    receipt = {
        "version": VERSION, "model": model_name,
        "repository": profile["repository"], "revision": profile["revision"],
        "files_sha256": {name: sha(output / name) for name in ASSET_FILES},
        "weights_downloaded": False,
    }
    write(output / "assets.json", receipt)
    return receipt


def verify_assets(path, model_name):
    receipt, profile = read(Path(path) / "assets.json"), PROFILES[model_name]
    require(receipt["model"] == model_name and receipt["repository"] == profile["repository"]
            and receipt["revision"] == profile["revision"], "Tokenizer model pin differs.")
    require(set(receipt["files_sha256"]) == set(ASSET_FILES), "Incomplete tokenizer receipt.")
    for name, expected in receipt["files_sha256"].items():
        require(sha(Path(path) / name) == expected, f"Tokenizer asset changed: {name}")
    return receipt


def tokenizer_from(path, model_name):
    from transformers import AutoTokenizer

    verify_assets(path, model_name)
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def prompt_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def encode_completion(tokenizer, messages, text, terminal_ids):
    prefix = prompt_ids(tokenizer, messages)
    complete = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": text}],
        tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=False,
    )
    require(complete[:len(prefix)] == prefix,
            "Native full conversation does not preserve the generation prefix boundary.")
    suffix = complete[len(prefix):]
    terminal_index = next((i for i, token in enumerate(suffix) if token in terminal_ids), None)
    require(terminal_index is not None, "Native assistant template has no configured terminal.")
    completion = suffix[:terminal_index + 1]
    require(tokenizer.decode(completion[:-1], skip_special_tokens=False) == text,
            "Assistant template changed target text or injected a special terminal.")
    require(not tokenizer.decode(suffix[terminal_index + 1:], skip_special_tokens=False).strip(),
            "Unexpected content after assistant terminal.")
    ids = prefix + completion
    return {
        "input_ids": ids, "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix) + completion,
        "completion_start": len(prefix), "completion_text": text,
        "terminal_token_id": completion[-1],
    }


def validate_case(case):
    require(case["split"] == case["source"]["split"] == "train", "Only TRAIN cases may be prepared.")
    require(case["language"] in ("en", "ko"), "Unexpected case language.")
    require(case.get("judgment", {}).get("status", "edit_required") in ("edit_required", "preserve"),
            "Uncertain labels cannot become training abstentions.")
    require(case["messages"] == formats.messages(case["input"]), "Canonical saved messages changed.")
    references = [case["primary"], *case.get("acceptable", [])]
    if "accepted_predictions" in case:
        require(case["accepted_predictions"] == case.get("acceptable", []), "Accepted references differ.")
    for action in references:
        text = formats.serialize(case["input"], action, "json")
        require(formats.parse(text, case["input"], "json").model_dump() == action,
                "Seven-action semantic round trip failed.")


def validate_encoded(row, terminal_ids, context_tokens, max_completion):
    ids, start, labels = row["input_ids"], row["completion_start"], row["labels"]
    require(type(start) is int and 0 < start < len(ids), "Invalid assistant boundary.")
    require(all(type(value) is int and value >= 0 for value in ids), "Invalid token IDs.")
    require(labels == [-100] * start + ids[start:] and row["attention_mask"] == [1] * len(ids),
            "Completion-only mask changed.")
    require(ids[-1] in terminal_ids and not set(ids[start:-1]).intersection(terminal_ids),
            "Invalid completion terminal.")
    require(len(ids) <= context_tokens and len(ids) - start <= max_completion
            and start <= context_tokens - max_completion, "Case exceeds the declared prompt/output budget.")


def prepare(model_name, cases_path, assets, output, *, context_tokens=6144, epochs=3,
            accumulation=4, learning_rate=5e-5, seed=42, targets=None):
    require(context_tokens in (4096, 6144, 8192) and epochs > 0 and accumulation > 0,
            "Explicit positive training and supported context budgets are required.")
    require(0 < learning_rate <= 0.01, "Invalid learning rate.")
    output = new_directory(output)
    profile = PROFILES[model_name]
    tokenizer = tokenizer_from(assets, model_name)
    cases = rows(cases_path)
    require(cases and len({case["id"] for case in cases}) == len(cases), "Empty or duplicate TRAIN cases.")
    encoded, over_budget = [], []
    for case in cases:
        validate_case(case)
        text = formats.serialize(case["input"], case["primary"], "json")
        row = encode_completion(tokenizer, case["messages"], text, profile["terminal_ids"])
        row.update(case_id=case["id"], language=case["language"], action=case["primary"]["action"],
                   semantic_input_sha256=value_hash(formats.contract.wire_snapshot(case["input"])),
                   messages=case["messages"])
        try:
            validate_encoded(row, profile["terminal_ids"], context_tokens, 512)
        except ValueError as error:
            over_budget.append({"case_id": case["id"], "prompt_tokens": row["completion_start"],
                                "completion_tokens": len(row["input_ids"]) - row["completion_start"],
                                "error": str(error)})
        encoded.append(row)
    lengths = [row["completion_start"] for row in encoded]
    report = {
        "version": VERSION, "model": model_name, "requested_rows": len(cases),
        "retained_rows": len(encoded), "excluded_rows": 0, "truncation": False,
        "context_tokens": context_tokens, "max_prompt_tokens": max(lengths),
        "min_prompt_tokens": min(lengths), "over_budget": over_budget,
        "counts": dict(sorted(Counter(f"{r['language']}:{r['action']}" for r in encoded).items())),
    }
    write(output / "preparation.json", report)
    require(not over_budget, "Full cohort exceeds budget; inspect preparation.json and explicitly revise the budget. No rows were excluded.")
    order, randomizer = [], random.Random(seed)
    for _ in range(epochs):
        indices = list(range(len(encoded)))
        randomizer.shuffle(indices)
        order.extend(indices)
    write_rows(output / "train.jsonl.gz", encoded)
    config = {
        "version": VERSION, "model": model_name, "base": profile,
        "assets_path": str(Path(assets).resolve()), "assets": verify_assets(assets, model_name),
        "source_cases_sha256": sha(cases_path), "train_rows_sha256": sha(output / "train.jsonl.gz"),
        "contract_version": formats.contract.VERSION, "format": "json",
        "context_tokens": context_tokens, "max_completion_tokens": 512,
        "training": {"epochs": epochs, "seed": seed, "gradient_accumulation": accumulation,
                     "learning_rate": learning_rate, "rank": 16, "alpha": profile["lora_alpha"],
                     "dropout": 0.0, "weight_decay": 0.01, "max_grad_norm": 1.0,
                     "target_mode": targets or profile["default_targets"], "sample_order": order,
                     "micro_batch_size": 1, "packing": False, "truncation": False},
        "generation": {"max_new_tokens": 512, "do_sample": False, "num_beams": 1,
                       "use_cache": True, "eos_token_id": profile["terminal_ids"],
                       "pad_token_id": tokenizer.pad_token_id, "bos_token_id": tokenizer.bos_token_id},
        "preparation_environment": environment(), "final_test_bodies_read": False,
    }
    write(output / "configuration.json", config)
    return report


def prepared(path, assets_override=None):
    path = Path(path).resolve()
    config = read(path / "configuration.json")
    require(config["version"] == VERSION and config["base"] == PROFILES[config["model"]], "Model profile changed.")
    require(config["contract_version"] == formats.contract.VERSION, "Action contract changed.")
    require(sha(path / "train.jsonl.gz") == config["train_rows_sha256"], "Prepared token rows changed.")
    assets = Path(assets_override or config["assets_path"])
    require(verify_assets(assets, config["model"]) == config["assets"], "Tokenizer identity changed.")
    encoded = rows(path / "train.jsonl.gz")
    for row in encoded:
        validate_encoded(row, config["generation"]["eos_token_id"], config["context_tokens"], config["max_completion_tokens"])
    order = config["training"]["sample_order"]
    require(Counter(order) == Counter({i: config["training"]["epochs"] for i in range(len(encoded))}),
            "Training must expose every cohort row exactly once per declared epoch.")
    return config, encoded, assets


def select_targets(model, model_name, mode):
    if model_name == "lfm25-1.2b":
        require(mode in ("attention", "hybrid-ffn"), "Unsupported LFM target mode.")
        suffixes = {"q_proj", "k_proj", "v_proj", "out_proj"}
        if mode == "hybrid-ffn":
            suffixes |= {"in_proj", "w1", "w2", "w3"}
        prefix = "model.layers."
    else:
        require(mode in ("attention", "attention-ffn"), "Unsupported Gemma target mode.")
        suffixes = {"q_proj", "k_proj", "v_proj", "o_proj"}
        if mode == "attention-ffn":
            suffixes |= {"gate_proj", "up_proj", "down_proj"}
        prefix = "model.language_model.layers."
    selected = {}
    for name, module in model.named_modules():
        if name.startswith(prefix) and name.rsplit(".", 1)[-1] in suffixes:
            if model_name == "lfm25-1.2b" and mode == "attention" and ".self_attn." not in name:
                continue
            require(type(module).__name__ == "Linear", f"LoRA target is not linear: {name}")
            selected[name] = [module.out_features, module.in_features]
    require(selected and {name.rsplit(".", 1)[-1] for name in selected} == suffixes,
            "Actual LoRA target inventory does not cover the requested modules.")
    return dict(sorted(selected.items()))


def smoke_selection(cases_path, preparations, output):
    source_sha = sha(cases_path)
    lengths, cells = {}, {}
    for preparation in preparations:
        config, encoded, _ = prepared(preparation)
        require(config["source_cases_sha256"] == source_sha, "Smoke token inventories must bind the same TRAIN cohort.")
        for row in encoded:
            lengths.setdefault(row["case_id"], {})[config["model"]] = row["completion_start"]
            cells[row["case_id"]] = (row["language"], row["action"])
    require(all(len(values) == len(preparations) for values in lengths.values()), "Smoke inventories have different case IDs.")
    by_cell = {}
    for identity in sorted(lengths, key=lambda key: (max(lengths[key].values()), key)):
        by_cell.setdefault(cells[identity], identity)
    expected = {(language, action) for language in ("en", "ko") for action in formats.contract.ACTIONS}
    require(set(by_cell) == expected, "Smoke requires all fourteen action/language cells.")
    selected = list(by_cell.values())
    longest = [identity for identity in sorted(lengths, key=lambda key: (-max(lengths[key].values()), key))
               if identity not in selected][:2]
    selected += longest
    require(len(selected) == 16, "Smoke needs fourteen cells plus two distinct longest remaining prompts.")
    cases = {case["id"]: case for case in rows(cases_path) if case["id"] in selected}
    output = new_directory(output)
    write_rows(output / "cases.jsonl.gz", [cases[identity] for identity in selected])
    report = {"version": VERSION, "status": "engineering_train_only_selection",
              "source_cases_sha256": source_sha, "selected": len(selected),
              "rule": "Shortest maximum model-token length per action/language cell, then two longest unselected cases; lexical ID breaks ties.",
              "case_ids": selected, "longest_additions": longest,
              "prompt_tokens": {identity: lengths[identity] for identity in selected},
              "case_sha256": {identity: value_hash(cases[identity]) for identity in selected},
              "references_preserved": True, "final_test_bodies_read": False}
    write(output / "selection.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire_parser = sub.add_parser("acquire")
    acquire_parser.add_argument("--model", choices=PROFILES, required=True)
    acquire_parser.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--model", choices=PROFILES, required=True)
    for name in ("cases", "assets", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--context-tokens", type=int, default=6144)
    prep.add_argument("--epochs", type=int, default=3)
    prep.add_argument("--accumulation", type=int, default=4)
    prep.add_argument("--learning-rate", type=float, default=5e-5)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--targets", choices=("attention", "hybrid-ffn", "attention-ffn"))
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--cases", type=Path, required=True)
    smoke.add_argument("--preparations", type=Path, nargs="+", required=True)
    smoke.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        result = acquire(args.model, args.output)
    elif args.command == "smoke":
        result = smoke_selection(args.cases, args.preparations, args.output)
    else:
        result = prepare(args.model, args.cases, args.assets, args.output,
                         context_tokens=args.context_tokens, epochs=args.epochs,
                         accumulation=args.accumulation, learning_rate=args.learning_rate,
                         seed=args.seed, targets=args.targets)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
