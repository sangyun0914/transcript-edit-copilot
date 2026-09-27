"""Two-update, pinned passive SFT smoke test and separate fresh-process reload."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import time
from pathlib import Path


VERSION = "passive-sft-smoke-v1"
LIBRARIES = (
    "python",
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
    "safetensors",
)
MODULE_PATTERN = re.compile(
    r"model\.(?:language_model\.)?layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
)
SLOTS = [
    (action, language)
    for action in ("no_action", "correct_text", "split_paragraph", "merge_paragraph")
    for language in ("en", "ko")
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def loading_diagnostics(value):
    if isinstance(value, dict):
        return {key: loading_diagnostics(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(
            (loading_diagnostics(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, (list, tuple)):
        return [loading_diagnostics(item) for item in value]
    return value


def child(directory, name):
    path = Path(name)
    require(not path.is_absolute(), "Portable input paths must be relative.")
    resolved = (directory / path).resolve()
    require(resolved.is_relative_to(directory.resolve()), "Input path escapes its root.")
    return resolved


def check_file(path, expected):
    require(
        isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
        "An explicit SHA256 is required.",
    )
    require(path.is_file() and digest(path) == expected, f"Bound file changed: {path}")


def validate_configuration(config):
    require(config["version"] == VERSION, "Unsupported smoke configuration version.")
    require(config["candidate_id"] == "qwen35-9b", "This smoke selects qwen35-9b only.")
    require(set(config["libraries"]) == set(LIBRARIES), "Pin the complete runtime library set.")
    require(
        all(
            isinstance(v, str) and v and not any(c in v for c in "*<>=,")
            for v in config["libraries"].values()
        ),
        "Library versions must be exact, not ranges.",
    )
    absent = config.get("absent_libraries", [])
    require(
        isinstance(absent, list)
        and all(
            isinstance(name, str) and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
            for name in absent
        )
        and len(set(absent)) == len(absent)
        and not set(absent) & set(LIBRARIES),
        "Absent libraries must be unique package names outside the pinned library set.",
    )
    model, train, generation = (config[key] for key in ("model", "training", "generation"))
    require(
        model["class"] == "Qwen3_5ForConditionalGeneration"
        and model["dtype"] == "bfloat16"
        and model["quantization"] in ("nf4", "none")
        and type(model["bnb_4bit_use_double_quant"]) is bool
        and model["attn_implementation"] in ("eager", "sdpa")
        and model["gradient_checkpointing"] is True,
        "Unsupported model, precision, attention or checkpointing setting.",
    )
    targets = model["target_modules"]
    require(
        isinstance(targets, list)
        and targets
        and len(set(targets)) == len(targets)
        and all(isinstance(name, str) and MODULE_PATTERN.fullmatch(name) for name in targets),
        "Use unique exact text attention/MLP module names; vision/GDN are excluded.",
    )
    ignored = model["ignored_native_tensor_names"]
    require(
        isinstance(ignored, list)
        and len(set(ignored)) == len(ignored)
        and all(isinstance(name, str) and name.startswith("mtp.") for name in ignored),
        "Only explicit native MTP tensor exclusions are supported.",
    )
    fixed = {
        "micro_batch_size": 1,
        "gradient_accumulation": 4,
        "max_updates": 2,
        "rank": 16,
        "alpha": 16,
        "dropout": 0,
        "bias": "none",
        "optimizer": "adamw_torch",
        "scheduler": "constant",
        "max_sequence_tokens": 4096,
        "max_completion_tokens": 512,
        "loss_implementation": "completion_positions_cross_entropy",
        "normalization": "mean_completion_loss_per_row_then_mean_microbatches",
        "truncation": False,
        "packing": False,
    }
    require(all(train.get(k) == v for k, v in fixed.items()), "The bounded smoke recipe changed.")
    require(type(train["seed"]) is int and 0 <= train["seed"] < 2**32, "Invalid seed.")
    for key, lower, upper in (
        ("learning_rate", 0, 1e-4),
        ("weight_decay", -1e-12, 0.1),
        ("max_grad_norm", 0, 10),
    ):
        value = train[key]
        require(
            type(value) in (int, float) and math.isfinite(value) and lower < value <= upper,
            f"Invalid bounded {key}.",
        )
    require(
        generation["max_new_tokens"] == 512
        and generation["do_sample"] is False
        and generation["use_cache"] is True
        and generation["num_beams"] == 1
        and generation["eos_token_id"] == generation["pad_token_id"],
        "Use explicit greedy EOS/padding and the frozen 512-token reserve.",
    )


def environment(config):
    actual = {"python": platform.python_version()}
    absent = config.get("absent_libraries", [])
    for library in (*LIBRARIES[1:], *absent):
        try:
            actual[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            actual[library] = None
    differences = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in config["libraries"].items()
        if actual[key] != value
    }
    differences.update(
        {
            key: {"expected": None, "actual": actual[key]}
            for key in absent
            if actual[key] is not None
        }
    )
    return {"actual": actual, "differences": differences, "matched": not differences}


def validate_smoke(config, prepared, cases, messages, tokens, inputs):
    train = config["training"]
    order = prepared["proposed_smoke"]["order"]
    require(
        len(cases) == len(messages) == len(tokens) == len(inputs) == 8
        and len(set(order)) == 8
        and train["sample_order"] == order,
        "Smoke must visit the exact eight preregistered rows once.",
    )
    eos = prepared["tokenizer"]["eos_token_id"]
    vocab = prepared["tokenizer"]["vocab_size"]
    require(
        config["generation"]["eos_token_id"] == eos, "Generation EOS differs from training EOS."
    )
    rows = []
    for index, (case, message, token, model_input) in enumerate(
        zip(cases, messages, tokens, inputs, strict=True)
    ):
        identity, meta, values = case["id"], case["metadata"], token["primary"]
        require(
            identity == order[index] == message["id"] == token["id"],
            "Smoke order or identity changed.",
        )
        require(
            meta["split"] == token["split"] == "train"
            and meta["training_eligible"] is True
            and (case["expectedOutput"]["action"], meta["language"]) == SLOTS[index]
            and isinstance(meta["source_group_id"], str)
            and bool(meta["source_group_id"])
            and meta["sampling_cluster"].startswith("train:"),
            "Non-train source or incorrect action/language smoke cell.",
        )
        prefix, ids, labels, attention = (
            values[k] for k in ("completion_start", "input_ids", "labels", "attention_mask")
        )
        require(
            type(prefix) is int
            and 0 < prefix <= 3584
            and 0 < len(ids) - prefix <= train["max_completion_tokens"]
            and len(ids) <= train["max_sequence_tokens"]
            and all(type(i) is int and 0 <= i < vocab for i in ids)
            and labels == [-100] * prefix + ids[prefix:]
            and all(type(i) is int for i in labels)
            and attention == [1] * len(ids)
            and all(type(i) is int for i in attention)
            and ids[-1] == eos
            and model_input == {k: values[k] for k in ("input_ids", "attention_mask", "labels")},
            "Invalid completion mask, EOS, attention or unpadded smoke input.",
        )
        require(
            len(token["messages"]) == 2
            and [m["role"] for m in token["messages"]] == ["system", "user"]
            and json.loads(token["references"][0]) == case["expectedOutput"]
            and message["messages"]
            == [*token["messages"], {"role": "assistant", "content": token["references"][0]}]
            and token["input_sha256"] == meta["input_sha256"],
            "Smoke prompt or target binding changed.",
        )
        rows.append({"case_id": identity, **values})
    require(
        sum(len(r["input_ids"]) for r in rows) == train["max_total_sequence_tokens"]
        and sum(len(r["input_ids"]) - r["completion_start"] for r in rows)
        == train["max_total_completion_tokens"],
        "The preregistered total sequence/completion budget changed.",
    )
    return rows


def preflight(config_path):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, read(config_path)
    validate_configuration(config)
    check_file(Path(__file__).resolve(), config["trainer_sha256"])
    bound = {}
    for name in ("selection", "preparation"):
        binding = config["bindings"][name]
        path = child(root, binding["path"])
        check_file(path, binding["sha256"])
        bound[name] = (path, read(path))
    selection = bound["selection"][1]
    prep_path, receipt = bound["preparation"]
    require(
        selection["selected_t1_candidate_ids"] == [config["candidate_id"]]
        and selection["bindings"]["train_only_preparation"]["sha256"] == digest(prep_path)
        and receipt["candidate_id"] == config["candidate_id"]
        and receipt["release_manifest_sha256"] == selection["bindings"]["release_manifest_sha256"],
        "Selection, release and preparation do not bind the same candidate.",
    )
    for name, expected in receipt["files_sha256"].items():
        check_file(child(prep_path.parent, name), expected)
    prepared = read(prep_path.parent / "configuration.json")
    native, selected_native = config["base"], selection["selected_native_identity"]
    require(
        native["repository"]
        == selected_native["base_repository"]
        == prepared["native_base"]["repository"]
        and native["revision"]
        == selected_native["base_revision"]
        == prepared["native_base"]["revision"]
        and re.fullmatch(r"[0-9a-f]{40}", native["revision"]),
        "Native base repository/revision differs from the selection.",
    )
    native_path = child(root, native["path"])
    files = native["files_sha256"]
    require("config.json" in files and "LICENSE" in files, "Bind native configuration and license.")
    for name, expected in files.items():
        check_file(child(native_path, name), expected)
    absent = native["absent_files"]
    require(
        len(set(absent)) == len(absent) and not set(absent) & set(files),
        "Conflicting native absence bindings.",
    )
    for name in absent:
        require(not child(native_path, name).exists(), f"Pinned absent file appeared: {name}")
    require(
        "generation_config.json" in files or "generation_config.json" in absent,
        "Bind generation config presence or absence.",
    )
    if "model.safetensors.index.json" in files:
        index = read(native_path / "model.safetensors.index.json")
        shards = set(index["weight_map"].values())
        require(shards and shards <= files.keys(), "Unbound native weight shard.")
        require(
            shards == {name for name in files if name.endswith(".safetensors")},
            "Unexpected native weight file.",
        )
        require(
            set(config["model"]["ignored_native_tensor_names"])
            == {name for name in index["weight_map"] if name.startswith("mtp.")},
            "The preregistered unused MTP tensor census differs from the native index.",
        )
    else:
        require("model.safetensors" in files, "Bind native safetensors weights and index.")
    tokenizer = config["tokenizer"]
    require(
        tokenizer["files_sha256"]
        == prepared["tokenizer"]["files_sha256"]
        == selected_native["tokenizer_files_sha256"]
        and tokenizer["template_sha256"] == tokenizer["files_sha256"]["chat_template.jinja"]
        and files["config.json"] == tokenizer["files_sha256"]["config.json"],
        "Native model/tokenizer/template differs from the frozen input identity.",
    )
    for name, expected in tokenizer["files_sha256"].items():
        check_file(child(child(root, tokenizer["path"]), name), expected)
    bodies = [
        read_rows(prep_path.parent / name)
        for name in (
            "smoke-cases.jsonl",
            "smoke-messages.jsonl",
            "smoke-tokens.jsonl",
            "smoke-model-inputs.jsonl",
        )
    ]
    rows = validate_smoke(config, prepared, *bodies)
    native_config = read(native_path / "config.json")
    summary = {
        "version": VERSION,
        "status": "configuration_and_inputs_verified",
        "configuration_sha256": digest(config_path),
        "trainer_sha256": digest(Path(__file__)),
        "preparation_sha256": digest(prep_path),
        "native_weight_files": len(files),
        "rows": len(rows),
        "sequence_tokens": sum(len(r["input_ids"]) for r in rows),
        "completion_tokens": sum(len(r["input_ids"]) - r["completion_start"] for r in rows),
        "native_config_eos_token_id": native_config.get("text_config", native_config).get(
            "eos_token_id"
        ),
        "explicit_training_and_generation_eos_token_id": config["generation"]["eos_token_id"],
        "environment": environment(config),
        "training_updates": 0,
        "padding": "None: batch size one, exact complete frozen sequences.",
        "final_test_bodies_read": 0,
    }
    return config, root, rows, bodies[2], summary


def tensor_hash(tensor, torch):
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_records(state, torch):
    result = {}
    for name, tensor in state.items():
        require(torch.isfinite(tensor).all().item(), f"Non-finite adapter tensor: {name}")
        result[name] = {
            "sha256": tensor_hash(tensor, torch),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return result


def completion_loss(model, row, torch, device):
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    prefix = row["completion_start"]
    positions = torch.arange(prefix - 1, ids.shape[1] - 1, device=device)
    logits = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=False,
        logits_to_keep=positions,
        return_dict=True,
    ).logits
    targets = torch.tensor(row["labels"][prefix:], dtype=torch.long, device=device)
    require(
        logits.shape[:2] == (1, len(targets)),
        "Completion-position logits have unexpected dimensions.",
    )
    return torch.nn.functional.cross_entropy(logits[0].float(), targets, reduction="mean")


def train_updates(model, rows, optimizer, torch, training, device, on_update, on_row=None):
    accumulation = training["gradient_accumulation"]
    require(
        len(rows) == accumulation * training["max_updates"], "Incomplete smoke accumulation group."
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    updates, group_losses, consumed = [], [], 0
    for row in rows:
        loss = completion_loss(model, row, torch, device)
        require(torch.isfinite(loss).item(), f"Non-finite loss at {row['case_id']}.")
        group_losses.append(float(loss.detach()))
        (loss / accumulation).backward()
        del loss
        require(
            all(p.grad is None or torch.isfinite(p.grad).all().item() for p in trainable),
            f"Non-finite gradient at {row['case_id']}.",
        )
        consumed += 1
        if on_row:
            on_row(
                {
                    "microbatch": consumed,
                    "case_id": row["case_id"],
                    "sequence_tokens": len(row["input_ids"]),
                    "completion_tokens": len(row["input_ids"]) - row["completion_start"],
                    "loss": group_losses[-1],
                }
            )
        if consumed % accumulation:
            continue
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                trainable,
                training["max_grad_norm"],
                error_if_nonfinite=True,
            )
        )
        require(
            math.isfinite(gradient_norm) and gradient_norm > 0,
            "Missing or non-finite gradient norm.",
        )
        optimizer.step()
        require(
            all(torch.isfinite(p).all().item() for p in trainable),
            "Optimizer produced non-finite tensors.",
        )
        optimizer.zero_grad(set_to_none=True)
        event = {
            "update": len(updates) + 1,
            "consumed_rows": consumed,
            "case_ids": [r["case_id"] for r in rows[consumed - accumulation : consumed]],
            "row_losses": group_losses,
            "mean_row_loss": sum(group_losses) / accumulation,
            "gradient_norm_before_clip": gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        updates.append(event)
        on_update(event)
        group_losses = []
    return updates


def load_runtime(config, root, encoded):
    require(
        environment(config)["matched"], "Installed library versions differ from the configuration."
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        Qwen3_5ForConditionalGeneration,
        set_seed,
    )

    require(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "A CUDA BF16-capable GPU is required.",
    )
    set_seed(config["training"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        child(root, config["tokenizer"]["path"]), local_files_only=True, trust_remote_code=False
    )
    require(
        tokenizer.eos_token_id == config["generation"]["eos_token_id"],
        "Loaded tokenizer EOS differs.",
    )
    for record in encoded:
        values = record["primary"]
        prefix = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        completion = tokenizer.encode(record["references"][0], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        require(
            prefix == values["input_ids"][: values["completion_start"]]
            and completion == values["input_ids"][values["completion_start"] :],
            "Loaded tokenizer no longer reproduces frozen prompt/target IDs.",
        )
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "local_files_only": True,
        "trust_remote_code": False,
        "attn_implementation": config["model"]["attn_implementation"],
        "output_loading_info": True,
    }
    if config["model"]["quantization"] == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=config["model"]["bnb_4bit_use_double_quant"],
        )
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        child(root, config["base"]["path"]), **kwargs
    )
    loading = loading_diagnostics(loading)
    require(
        not any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs"))
        and set(loading.get("unexpected_keys", []))
        <= set(config["model"]["ignored_native_tensor_names"]),
        f"Native model loading mismatch: {loading}",
    )
    modules = dict(model.named_modules())
    available = {name for name in modules if MODULE_PATTERN.fullmatch(name)}
    require(
        available == set(config["model"]["target_modules"]),
        "Configured text target modules differ from the actual native model.",
    )
    require(
        all(isinstance(modules[name], torch.nn.Linear) for name in available),
        "A target is not a supported linear module.",
    )
    device = torch.device("cuda:0")
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "capability": list(torch.cuda.get_device_capability(0)),
        "cuda_version": torch.version.cuda,
        "targets": {
            name: {
                "class": type(modules[name]).__name__,
                "in_features": modules[name].in_features,
                "out_features": modules[name].out_features,
            }
            for name in sorted(available)
        },
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "loading_info": loading,
        "tokenizer_parity_rows": len(encoded),
        "ignored_native_tensor_names": config["model"]["ignored_native_tensor_names"],
    }
    return torch, model, tokenizer, device, info


def enable_training(model, config):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    checkpointing = {"use_reentrant": False}
    if config["model"]["quantization"] == "nf4":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=checkpointing
        )
    else:
        model.requires_grad_(False)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing)
        model.enable_input_require_grads()
    training = config["training"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=training["rank"],
            lora_alpha=training["alpha"],
            lora_dropout=training["dropout"],
            bias=training["bias"],
            target_modules=config["model"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.peft_config["default"].base_model_name_or_path = config["base"]["repository"]
    model.peft_config["default"].revision = config["base"]["revision"]
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    actual_targets = {
        name.split(".lora_")[0].removeprefix("base_model.model.") for name in trainable
    }
    require(
        trainable
        and all(".lora_A." in name or ".lora_B." in name for name in trainable)
        and actual_targets == set(config["model"]["target_modules"])
        and len(trainable) == 2 * len(actual_targets),
        "Unexpected trainable tensors or missing adapters.",
    )
    model.train()
    return model, trainable


def prepare_reload_base(model, config):
    if config["model"]["quantization"] == "nf4":
        from peft import prepare_model_for_kbit_training

        # Match the training preparation's FP32 non-quantized parameters as well as NF4 weights.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return model


def generate_smoke(model, tokenizer, rows, config, torch, device, on_result=None):
    from transformers import GenerationConfig

    model.eval()
    model.gradient_checkpointing_disable()
    generation = GenerationConfig(**config["generation"], bos_token_id=tokenizer.bos_token_id)
    results = []
    for row in rows:
        prefix = row["input_ids"][: row["completion_start"]]
        ids = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation
            )
        output_ids = generated[0, len(prefix) :].tolist()
        results.append(
            {
                "case_id": row["case_id"],
                "prompt_ids_sha256": hashlib.sha256(
                    json.dumps(prefix, separators=(",", ":")).encode()
                ).hexdigest(),
                "output_ids": output_ids,
                "text": tokenizer.decode(output_ids, skip_special_tokens=True),
                "finish_reason": "stop"
                if output_ids and output_ids[-1] == generation.eos_token_id
                else "length",
            }
        )
        if on_result:
            on_result(results[-1])
    return results


def save_receipt(output, info):
    write(output / "run.json", info)
    write(
        output / "manifest.json",
        {
            "version": VERSION,
            "status": info["status"],
            "files_sha256": {
                str(path.relative_to(output)): digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )


def run(config_path, output, previous=None):
    config, root, rows, encoded, verified = preflight(config_path)
    require(
        verified["environment"]["matched"],
        "Installed library versions differ from the pinned configuration.",
    )
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh attempt output directory.")
    require(
        not output.is_relative_to(root),
        "Run outputs must be outside the immutable input directory.",
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    info = {
        **verified,
        "status": "loading",
        "mode": "fresh_reload" if previous else "train",
        "started_unix": started,
        "pid": os.getpid(),
        "optimizer_state_resumed": False,
        "sample_order": config["training"]["sample_order"],
        "training_updates": 0,
        "scope": "Engineering smoke; no model selection, deployment or dev/test quality claim.",
    }
    write(output / "configuration.json", config)
    (output / "trainer.py").write_bytes(Path(__file__).read_bytes())
    write(output / "run.json", info)
    try:
        from peft import PeftModel, get_peft_model_state_dict

        torch, model, tokenizer, device, runtime = load_runtime(config, root, encoded)
        write(output / "runtime.json", runtime)

        def generation_result(result):
            with (output / "smoke-generation-events.jsonl").open("a") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "event": "smoke_generation_complete",
                        "case_id": result["case_id"],
                        "output_tokens": len(result["output_ids"]),
                        "finish_reason": result["finish_reason"],
                    }
                ),
                flush=True,
            )

        if previous:
            previous = Path(previous).resolve()
            manifest = read(previous / "manifest.json")
            require(
                manifest["status"] == "training_complete_fresh_reload_pending",
                "Only a completed training smoke can be reloaded.",
            )
            for name, expected in manifest["files_sha256"].items():
                check_file(child(previous, name), expected)
            saved = read(previous / "run.json")
            require(
                saved["configuration_sha256"] == verified["configuration_sha256"]
                and saved["pid"] != os.getpid(),
                "Fresh reload requires the same configuration and a separate process.",
            )
            model = prepare_reload_base(model, config)
            model = PeftModel.from_pretrained(
                model, previous / "adapter", is_trainable=False, local_files_only=True
            )
            loaded = tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            require(
                loaded == read(previous / "adapter-tensors.json"),
                "Freshly loaded adapter tensors differ from saved tensors.",
            )
            actual = generate_smoke(
                model, tokenizer, rows, config, torch, device, generation_result
            )
            expected = read(previous / "smoke-generation.json")
            write(output / "smoke-generation.json", actual)
            parity = {
                "adapter_tensors_identical": True,
                "cases": [
                    {"case_id": a["case_id"], "identical": a == b}
                    for a, b in zip(actual, expected, strict=True)
                ],
                "optimizer_state_resumed": False,
            }
            write(output / "reload-parity.json", parity)
            require(
                all(row["identical"] for row in parity["cases"]),
                "Greedy output changed after fresh reload.",
            )
            info.update(
                status="fresh_reload_passed_local_export_pending",
                previous_manifest_sha256=digest(previous / "manifest.json"),
                reloaded_training_updates=saved["training_updates"],
            )
        else:
            model, trainable = enable_training(model, config)
            initial = tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            write(
                output / "initialization.json",
                {
                    "trainable_tensors": trainable,
                    "trainable_parameters": sum(trainable.values()),
                    "initial_adapter_tensors": initial,
                },
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )

            def record(event):
                info["training_updates"] = event["update"]
                with (output / "training-events.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                write(output / "run.json", info)
                print(json.dumps({"event": "optimizer_update", **event}), flush=True)

            def record_row(event):
                with (output / "training-rows.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                print(json.dumps({"event": "microbatch_complete", **event}), flush=True)

            info["status"] = "training"
            write(output / "run.json", info)
            updates = train_updates(
                model, rows, optimizer, torch, config["training"], device, record, record_row
            )
            final = tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            require(set(initial) == set(final), "Adapter tensor census changed.")
            changed = [name for name in final if final[name] != initial[name]]
            require(changed, "No adapter tensors changed after the bounded updates.")
            model.save_pretrained(
                output / "adapter", safe_serialization=True, save_embedding_layers=False
            )
            tokenizer.save_pretrained(output / "tokenizer")
            adapter_config = read(output / "adapter/adapter_config.json")
            require(
                adapter_config["base_model_name_or_path"] == config["base"]["repository"]
                and adapter_config["revision"] == config["base"]["revision"],
                "Saved adapter lost native base provenance.",
            )
            write(output / "adapter-tensors.json", final)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                    "python_rng_state": random.getstate(),
                    "completed_updates": len(updates),
                    "next_row_index": len(rows),
                    "configuration_sha256": verified["configuration_sha256"],
                    "resume_supported_by_this_smoke": False,
                },
                output / "training-state.pt",
            )
            write(
                output / "smoke-generation.json",
                generate_smoke(model, tokenizer, rows, config, torch, device, generation_result),
            )
            info.update(
                status="training_complete_fresh_reload_pending",
                training_updates=len(updates),
                changed_adapter_tensors=changed,
                consumed_rows=len(rows),
                unique_cases=len({r["case_id"] for r in rows}),
                resume_supported=False,
            )
        info.update(
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
            seconds=time.time() - started,
        )
        save_receipt(output, info)
        return info
    except BaseException as error:
        info.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            seconds=time.time() - started,
        )
        if "torch" in locals() and torch.cuda.is_available():
            info.update(
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
            )
        write(output / "run.json", info)
        save_receipt(output, info)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "reload"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        if command != "preflight":
            sub.add_argument("--output", type=Path, required=True)
        if command == "reload":
            sub.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.config)[-1]
    else:
        result = run(args.config, args.output, getattr(args, "run", None))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
