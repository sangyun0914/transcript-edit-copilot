"""Pinned Gemma BF16/FP32-LoRA smoke, curriculum, and separate fresh reload."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time

import prepare_passive_gemma as preparation
import prepare_passive_curriculum as curriculum_preparation
import prepare_passive_sft as smoke_preparation
import train_passive_curriculum as curriculum
import train_passive_sft as smoke


VERSION = "passive-gemma-sft-v1"
REPOSITORY = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
MODULE_PATTERN = re.compile(
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
)
SMOKE_BINDINGS = (
    "smoke_configuration",
    "smoke_training_manifest",
    "smoke_run",
    "smoke_reload_manifest",
    "smoke_reload_run",
    "smoke_reload_parity",
)
SOURCE_MODULES = (
    ("trainer.py", "trainer_sha256", __file__),
    ("train_passive_sft.py", "smoke_trainer_sha256", smoke.__file__),
    ("train_passive_curriculum.py", "curriculum_trainer_sha256", curriculum.__file__),
    ("prepare_passive_gemma.py", "preparer_sha256", preparation.__file__),
    (
        "prepare_passive_curriculum.py",
        "curriculum_preparer_sha256",
        curriculum_preparation.__file__,
    ),
    ("prepare_passive_sft.py", "smoke_preparer_sha256", smoke_preparation.__file__),
)
require = smoke.require
bound = curriculum.bound


def target_names():
    return sorted(
        f"model.language_model.layers.{layer}.{component}"
        for layer in range(30)
        for component in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            *(("self_attn.v_proj",) if layer % 6 != 5 else ()),
        )
    )


def validate_configuration(config):
    require(
        config["version"] == VERSION
        and config["candidate_id"] == "gemma26"
        and config["phase"] in ("smoke", "substantive"),
        "Select a registered Gemma engineering or substantive phase.",
    )
    require(
        set(config["libraries"]) == set(smoke.LIBRARIES)
        and all(
            isinstance(v, str) and v and not any(c in v for c in "*<>=,")
            for v in config["libraries"].values()
        )
        and config.get("absent_libraries") == ["torchao"],
        "Pin every runtime version and the incompatible unused torchao absence.",
    )
    fixed_model = {
        "class": "Gemma4ForConditionalGeneration",
        "dtype": "bfloat16",
        "adapter_dtype": "float32",
        "quantization": "none",
        "attn_implementation": "sdpa",
        "experts_implementation": "eager",
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
        "ignored_native_tensor_names": [],
        "expected_trainable_parameters": 18585600,
    }
    require(
        all(config["model"].get(k) == v for k, v in fixed_model.items())
        and config["model"]["target_modules"] == target_names()
        and isinstance(config["model"]["gpu_name_contains"], str)
        and bool(config["model"]["gpu_name_contains"].strip())
        and type(config["model"]["minimum_gpu_bytes"]) is int
        and config["model"]["minimum_gpu_bytes"] >= 75000000000,
        "Gemma requires its exact 205 text linear targets and BF16-base/FP32-adapter regime.",
    )
    require(
        config["base"]["repository"] == REPOSITORY
        and config["base"]["revision"] == REVISION,
        "Use the original pinned native Gemma weights.",
    )
    phase = config["phase"]
    fixed = {
        "seed": 3407,
        "sampling_seed": "3407",
        "micro_batch_size": 1,
        "gradient_accumulation": 4,
        "max_updates": 2 if phase == "smoke" else 203,
        "draw_count": 8 if phase == "smoke" else 809,
        "rank": 16,
        "alpha": 16,
        "dropout": 0,
        "bias": "none",
        "learning_rate": 1e-4,
        "optimizer": "adamw_torch",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "scheduler": "constant",
        "wall_limit_seconds": 21600,
        "max_sequence_tokens": 4096,
        "max_completion_tokens": 512,
        "loss_implementation": "completion_positions_cross_entropy",
        "normalization": "mean_completion_loss_per_row_then_mean_microbatches",
        "final_partial_accumulation": "actual_microbatch_count",
        "fresh_native_initialization": True,
        "truncation": False,
        "packing": False,
        "recovery_updates": [] if phase == "smoke" else [50, 100, 150],
    }
    train = config["training"]
    require(
        all(train.get(k) == v for k, v in fixed.items())
        and type(train["seed"]) is int
        and type(train["sampling_seed"]) is str
        and len(train["sample_order"]) == train["draw_count"],
        "The preregistered Gemma recipe or phase budget changed.",
    )
    require(
        config["generation"]
        == {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
            "num_beams": 1,
            "eos_token_id": [1, 106, 50],
            "pad_token_id": 0,
            "bos_token_id": 2,
        },
        "Pin greedy generation with all native terminal IDs; training label EOS remains 1.",
    )


def validate_protocol(protocol, config, release_sha256):
    expected = {
        "view": "json-curriculum",
        "draw_count": 809,
        "sampling_seed": "3407",
        "model_seed": 3407,
        "effective_batch": 4,
        "updates": 203,
        "final_accumulation": 1,
        "development_during_training": False,
        "checkpoint_selection": "final_update_203_only",
        "recovery_updates": [50, 100, 150],
        "max_wall_seconds": 21600,
    }
    require(
        protocol["version"] == "passive-gemma-evaluation-v1"
        and protocol["candidate_id"] == config["candidate_id"]
        and protocol["numerical_regime"]
        == {
            "base_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "quantization": "none",
        }
        and all(
            protocol["engineering_smoke"].get(k) == v
            for k, v in {
                "draw_count": 8,
                "updates": 2,
                "fresh_reload_required": True,
            }.items()
        )
        and protocol["engineering_smoke"]["native_gpu"]
        == config["model"]["gpu_name_contains"]
        and protocol["engineering_smoke"]["minimum_gpu_bytes"]
        == config["model"]["minimum_gpu_bytes"]
        and protocol["engineering_smoke"]["actual_peak_memory_must_be_measured"] is True
        and all(protocol["training_plan"].get(k) == v for k, v in expected.items())
        and protocol["b2"]["release_manifest"]["sha256"] == release_sha256
        and protocol["final_test_bodies_read"] is False
        and protocol["learning_signal_policy"]["no_auto_extra_training"] is True
        and protocol["learning_signal_policy"]["deployment_promotion"] is False,
        "Bind the fixed Gemma protocol before training; no development-driven continuation.",
    )
    require(
        all(
            protocol["generation_policy"].get(k) == v
            for k, v in config["generation"].items()
            if k != "use_cache"
        )
        and protocol["generation_policy"]["label_eos_token_id"] == 1
        and protocol["generation_policy"]["thinking"] is False
        and protocol["generation_policy"]["raw_tokens_and_terminal_id_retained"]
        is True,
        "Generation protocol differs from the registered native terminal-token policy.",
    )


def validate_assets(config, root, pin):
    require(
        pin["version"] == "passive-native-pin-v1"
        and pin["candidate_id"] == "gemma26"
        and pin["repository"] == REPOSITORY
        and pin["revision"] == REVISION
        and {k: v["sha256"] for k, v in pin["files"].items()}
        == config["base"]["files_sha256"]
        and pin["license"]["identifier"] == "apache-2.0"
        and "README.md" in config["base"]["files_sha256"],
        "Native API pin, license evidence or asset inventory differs.",
    )
    for section in ("base", "tokenizer"):
        directory = smoke.child(root, config[section]["path"])
        for name, expected in config[section]["files_sha256"].items():
            smoke.check_file(smoke.child(directory, name), expected)
    native = smoke.child(root, config["base"]["path"])
    require(
        config["tokenizer"]["template_sha256"]
        == config["tokenizer"]["files_sha256"]["chat_template.jinja"]
        and all(
            config["base"]["files_sha256"][k] == v
            for k, v in config["tokenizer"]["files_sha256"].items()
        ),
        "The frozen tokenizer and native revision differ.",
    )
    absent = config["base"]["absent_files"]
    require(
        len(absent) == len(set(absent)) and not set(absent) & set(pin["files"]),
        "Conflicting native absence pins.",
    )
    for name in absent:
        require(
            not smoke.child(native, name).exists(),
            f"Pinned absent file appeared: {name}",
        )
    index = smoke.read(native / "model.safetensors.index.json")["weight_map"]
    require(
        set(index.values())
        == {name for name in pin["files"] if name.endswith(".safetensors")}
        and {
            name.removesuffix(".weight")
            for name in index
            if name.endswith(".weight")
            and MODULE_PATTERN.fullmatch(name.removesuffix(".weight"))
        }
        == set(config["model"]["target_modules"]),
        "Native shard or text target census differs.",
    )
    native_config = smoke.read(native / "config.json")
    require(
        native_config["architectures"] == ["Gemma4ForConditionalGeneration"]
        and native_config["text_config"]["vocab_size"] == 262144
        and native_config["text_config"]["eos_token_id"] == 1
        and smoke.read(native / "generation_config.json")["eos_token_id"]
        == [1, 106, 50],
        "Native architecture or termination identity changed.",
    )


def validate_smoke_gate(config, root, release_sha256):
    evidence = {
        name: bound(config["bindings"][name], root)[1] for name in SMOKE_BINDINGS
    }
    hashes = {name: config["bindings"][name]["sha256"] for name in SMOKE_BINDINGS}
    original = evidence["smoke_configuration"]
    validate_configuration(original)
    require(original["phase"] == "smoke", "The prerequisite must be a Gemma smoke.")
    for key in ("candidate_id", "model", "libraries", "absent_libraries", "generation"):
        require(
            original[key] == config[key],
            f"Completed Gemma smoke setting differs: {key}",
        )
    for section, keys in (
        ("base", ("repository", "revision", "files_sha256", "absent_files")),
        ("tokenizer", ("files_sha256", "template_sha256")),
    ):
        require(
            all(original[section][key] == config[section][key] for key in keys),
            f"Completed smoke input differs: {section}",
        )
    for _, key, _ in SOURCE_MODULES:
        require(
            original[key] == config[key],
            "Completed smoke source implementation changed.",
        )
    for name in ("preparation", "native_pin", "evaluation_protocol"):
        require(
            original["bindings"][name]["sha256"] == config["bindings"][name]["sha256"],
            f"Completed smoke binding differs: {name}",
        )
    train, run = evidence["smoke_training_manifest"], evidence["smoke_run"]
    reload, reload_run = evidence["smoke_reload_manifest"], evidence["smoke_reload_run"]
    parity = evidence["smoke_reload_parity"]
    require(
        train["version"]
        == run["version"]
        == reload["version"]
        == reload_run["version"]
        == VERSION
        and train["status"] == run["status"] == "training_complete_fresh_reload_pending"
        and run["phase"] == "smoke"
        and run["mode"] == "train"
        and run["training_updates"] == 2
        and run["optimizer_committed_rows"] == 8
        and run["configuration_sha256"] == hashes["smoke_configuration"]
        and run["release_manifest_sha256"] == release_sha256
        and run["final_test_bodies_read"] == 0
        and run["optimizer_state_resumed"] is False
        and train["files_sha256"]["run.json"] == hashes["smoke_run"]
        and train["files_sha256"]["configuration.json"] == hashes["smoke_configuration"]
        and reload["status"]
        == reload_run["status"]
        == "fresh_reload_passed_local_export_pending"
        and reload["files_sha256"]["run.json"] == hashes["smoke_reload_run"]
        and reload["files_sha256"]["reload-parity.json"]
        == hashes["smoke_reload_parity"]
        and reload_run["previous_manifest_sha256"] == hashes["smoke_training_manifest"]
        and reload_run["configuration_sha256"] == hashes["smoke_configuration"]
        and reload_run["mode"] == "fresh_reload"
        and reload_run["phase"] == "smoke"
        and reload_run["reloaded_training_updates"] == 2
        and reload_run["pid"] != run["pid"]
        and parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and [r["case_id"] for r in parity["cases"]]
        == original["training"]["sample_order"]
        and all(r["identical"] is True for r in parity["cases"]),
        "Substantive Gemma requires its actual completed two-update smoke and exact fresh reload.",
    )


def validate_prepared(config, bundle):
    prepared = bundle["configuration"]
    require(
        prepared["candidate_id"] == "gemma26"
        and prepared["training_view"] == "json-curriculum"
        and prepared["native_base"]["repository"] == REPOSITORY
        and prepared["native_base"]["revision"] == REVISION
        and prepared["tokenizer"]["files_sha256"] == config["tokenizer"]["files_sha256"]
        and prepared["tokenizer"]["eos_token_id"] == 1
        and prepared["tokenizer"]["vocab_size"] == 262144
        and prepared["sampling"]
        == {
            "algorithm": "weighted_indices_v2",
            "seed": "3407",
            "seed_type": "str",
            "replacement": True,
            "draw_count": 809,
            "weights_applied_to_loss": False,
        },
        "Prepared native tokenizer or fixed curriculum sampler differs.",
    )
    planned = bundle["smoke_configuration"]["proposed_smoke"]
    validation = {
        "generation": {"eos_token_id": 1},
        "training": {
            **config["training"],
            "sample_order": planned["order"],
            **{
                key: planned[key]
                for key in ("max_total_sequence_tokens", "max_total_completion_tokens")
            },
        },
    }
    diagnostics = smoke.validate_smoke(
        validation,
        bundle["smoke_configuration"],
        bundle["smoke_cases"],
        bundle["smoke_messages"],
        bundle["smoke_tokens"],
        bundle["smoke_inputs"],
    )
    if config["phase"] == "smoke":
        rows = curriculum.selected_rows(
            {
                "cases": bundle["smoke_cases"],
                "tokens": bundle["smoke_tokens"],
                "draw_plan": [
                    {"draw_index": i, "case_id": row["case_id"], "train_row_index": i}
                    for i, row in enumerate(diagnostics)
                ],
            }
        )
    else:
        rows = curriculum.selected_rows(bundle)
    training = config["training"]
    require(
        len(rows) == training["draw_count"]
        and [row["case_id"] for row in rows] == training["sample_order"]
        and sum(len(row["input_ids"]) for row in rows)
        == training["max_total_sequence_tokens"]
        and sum(len(row["input_ids"]) - row["completion_start"] for row in rows)
        == training["max_total_completion_tokens"],
        "Selected phase differs from its exact prepared exposure budget.",
    )
    return rows, diagnostics


def preflight(config_path):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, smoke.read(config_path)
    validate_configuration(config)
    for _, key, module in SOURCE_MODULES:
        smoke.check_file(Path(module), config[key])
    manifest_path, manifest = bound(config["bindings"]["preparation"], root)
    bundle = preparation.verify_bundle(manifest_path.parent)
    require(bundle["manifest"] == manifest, "Preparation changed during verification.")
    _, selection = bound(config["bindings"]["selection"], root)
    require(
        selection["selected_t1_candidate_ids"] == ["gemma26"]
        and selection["bindings"]["train_only_preparation"]["sha256"]
        == smoke.digest(manifest_path)
        and selection["bindings"]["release_manifest_sha256"]
        == manifest["release_manifest_sha256"]
        and selection["selected_native_identity"]["base_repository"] == REPOSITORY
        and selection["selected_native_identity"]["base_revision"] == REVISION,
        "Selection is not bound to this Gemma preparation/release/native revision.",
    )
    _, protocol = bound(config["bindings"]["evaluation_protocol"], root)
    validate_protocol(protocol, config, manifest["release_manifest_sha256"])
    _, pin = bound(config["bindings"]["native_pin"], root)
    validate_assets(config, root, pin)
    rows, diagnostics = validate_prepared(config, bundle)
    if config["phase"] == "substantive":
        validate_smoke_gate(config, root, manifest["release_manifest_sha256"])
    summary = {
        "version": VERSION,
        "phase": config["phase"],
        "candidate_id": "gemma26",
        "status": "configuration_and_inputs_verified",
        "configuration_sha256": smoke.digest(config_path),
        **{key: config[key] for _, key, _ in SOURCE_MODULES},
        "preparation_sha256": smoke.digest(manifest_path),
        "release_manifest_sha256": manifest["release_manifest_sha256"],
        "evaluation_protocol_sha256": config["bindings"]["evaluation_protocol"][
            "sha256"
        ],
        "native_pin_sha256": config["bindings"]["native_pin"]["sha256"],
        "environment": smoke.environment(config),
        "training_updates": 0,
        "draw_count": len(rows),
        "unique_planned_cases": len({r["case_id"] for r in rows}),
        "sequence_tokens": config["training"]["max_total_sequence_tokens"],
        "completion_tokens": config["training"]["max_total_completion_tokens"],
        "diagnostic_rows": len(diagnostics),
        "final_test_bodies_read": 0,
        "padding": "None: batch size one, exact frozen complete sequences.",
    }
    return config, root, bundle, rows, diagnostics, summary


def parameter_census(model):
    dtypes, devices = Counter(), Counter()
    for parameter in model.parameters():
        dtypes[str(parameter.dtype)] += parameter.numel()
        devices[str(parameter.device)] += parameter.numel()
    return {
        "parameter_elements_by_dtype": dict(sorted(dtypes.items())),
        "parameter_elements_by_device": dict(sorted(devices.items())),
    }


def validate_tokenizer(tokenizer, encoded):
    require(
        tokenizer.eos_token_id == 1
        and tokenizer.pad_token_id == 0
        and tokenizer.bos_token_id == 2,
        "Loaded Gemma tokenizer special tokens differ.",
    )
    for record in encoded:
        value = record["primary"]
        prefix = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        completion = tokenizer.encode(
            record["references"][0], add_special_tokens=False
        ) + [1]
        require(
            prefix == value["input_ids"][: value["completion_start"]]
            and completion == value["input_ids"][value["completion_start"] :],
            "Actual Gemma tokenizer differs from frozen prompt/target IDs.",
        )


def load_runtime(config, root, encoded):
    require(smoke.environment(config)["matched"], "Runtime library versions changed.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, set_seed

    require(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "A CUDA BF16-capable GPU is required.",
    )
    gpu = torch.cuda.get_device_properties(0)
    require(
        config["model"]["gpu_name_contains"] == gpu.name
        and gpu.total_memory >= config["model"]["minimum_gpu_bytes"],
        "The allocated GPU differs from the preregistered exact name or minimum visible memory.",
    )
    set_seed(config["training"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        smoke.child(root, config["tokenizer"]["path"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    validate_tokenizer(tokenizer, encoded)
    model, loading = Gemma4ForConditionalGeneration.from_pretrained(
        smoke.child(root, config["base"]["path"]),
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation=config["model"]["attn_implementation"],
        experts_implementation=config["model"]["experts_implementation"],
        output_loading_info=True,
    )
    loading = smoke.loading_diagnostics(loading)
    require(
        not any(
            loading.get(k)
            for k in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ),
        f"Native Gemma loading mismatch: {loading}",
    )
    require(
        model.config.text_config._attn_implementation
        == config["model"]["attn_implementation"]
        and model.config.text_config._experts_implementation
        == config["model"]["experts_implementation"],
        "Loaded text attention or expert execution backend differs from its pin.",
    )
    modules = dict(model.named_modules())
    available = {name for name in modules if MODULE_PATTERN.fullmatch(name)}
    require(
        available == set(config["model"]["target_modules"])
        and all(type(modules[name]) is torch.nn.Linear for name in available),
        "Actual Gemma linear module census differs from the pinned 205 targets.",
    )
    require(
        all(
            p.device.type == "cuda"
            and p.device.index == 0
            and p.dtype == torch.bfloat16
            for p in model.parameters()
        ),
        "Every native parameter must stay BF16 on the one GPU; offloading is not registered.",
    )
    expert_parameters = {
        name: list(p.shape)
        for name, p in model.named_parameters()
        if ".experts." in name
    }
    require(
        len(expert_parameters) == 60
        and all(len(shape) == 3 for shape in expert_parameters.values()),
        "Gemma fused expert census changed.",
    )
    runtime = {
        "gpu": gpu.name,
        "gpu_total_bytes": gpu.total_memory,
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
        "base_parameter_census": parameter_census(model),
        "expert_parameters": expert_parameters,
        "loading_info": loading,
        "tokenizer_parity_rows": len(encoded),
        "ignored_native_tensor_names": [],
        "numerical_regime": "native_bfloat16_lora_float32",
        "text_attn_implementation": model.config.text_config._attn_implementation,
        "text_experts_implementation": model.config.text_config._experts_implementation,
        "native_generation_config": smoke.read(
            smoke.child(root, config["base"]["path"]) / "generation_config.json"
        ),
        "effective_generation": config["generation"],
        "completion_label_eos_token_id": 1,
        "allocated_after_load_bytes": torch.cuda.memory_allocated(),
        "reserved_after_load_bytes": torch.cuda.memory_reserved(),
    }
    return torch, model, tokenizer, torch.device("cuda:0"), runtime


def enable_training(model, config, torch):
    from peft import LoraConfig, get_peft_model

    model.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    train = config["training"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=train["rank"],
            lora_alpha=train["alpha"],
            lora_dropout=train["dropout"],
            bias=train["bias"],
            target_modules=config["model"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
        autocast_adapter_dtype=True,
    )
    model.peft_config["default"].base_model_name_or_path = config["base"]["repository"]
    model.peft_config["default"].revision = config["base"]["revision"]
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    trainable = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    actual = {
        name.split(".lora_")[0].removeprefix("base_model.model.") for name in trainable
    }
    require(
        actual == set(config["model"]["target_modules"])
        and len(trainable) == 2 * len(actual)
        and sum(trainable.values()) == config["model"]["expected_trainable_parameters"]
        and all((".lora_A." in n or ".lora_B." in n) for n in trainable),
        "Unexpected Gemma trainable adapter census.",
    )
    require(
        all(
            p.dtype == (torch.float32 if p.requires_grad else torch.bfloat16)
            for p in model.parameters()
        ),
        "Native BF16/expert tensors were upcast or adapter tensors were not FP32.",
    )
    model.train()
    return model, trainable


def generate_smoke(model, tokenizer, rows, config, torch, device, on_result=None):
    from transformers import GenerationConfig

    model.eval()
    model.gradient_checkpointing_disable()
    generation = GenerationConfig(**config["generation"])
    results = []
    for row in rows:
        prefix = row["input_ids"][: row["completion_start"]]
        ids = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                generation_config=generation,
            )
        output_ids = generated[0, len(prefix) :].tolist()
        terminal = output_ids[-1] if output_ids else None
        result = {
            "case_id": row["case_id"],
            "prompt_ids_sha256": hashlib.sha256(
                json.dumps(prefix, separators=(",", ":")).encode()
            ).hexdigest(),
            "output_ids": output_ids,
            "terminal_token_id": terminal,
            "text": tokenizer.decode(output_ids, skip_special_tokens=True),
            "text_raw": tokenizer.decode(output_ids, skip_special_tokens=False),
            "finish_reason": "stop"
            if terminal in generation.eos_token_id
            else "length",
        }
        results.append(result)
        if on_result:
            on_result(result)
    return results


def manifest(path, status):
    smoke.write(
        path / "manifest.json",
        {
            "version": VERSION,
            "status": status,
            "files_sha256": {
                str(p.relative_to(path)): smoke.digest(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and p != path / "manifest.json"
            },
        },
    )


def save_checkpoint(
    output, name, model, optimizer, torch, config_sha, exposure, reason
):
    require(
        not exposure.optimizer_commit_uncertain,
        "An interrupted optimizer step cannot be marked as a safe checkpoint.",
    )
    path = output / name
    path.mkdir(parents=True, exist_ok=False)
    curriculum.save_training_state(path, model, optimizer, torch, config_sha, exposure)
    smoke.write(
        path / "checkpoint.json",
        {
            "status": "recovery_checkpoint_complete",
            "reason": reason,
            "configuration_sha256": config_sha,
            "exposure": exposure.summary(),
            "resume_supported": False,
            "optimizer_state_resumed": False,
            "partial_gradients_saved": False,
        },
    )
    manifest(path, "recovery_checkpoint_complete")
    return str(path.relative_to(output))


def verify_previous(previous, config, config_sha):
    receipt = smoke.read(previous / "manifest.json")
    required = {
        "run.json",
        "configuration.json",
        "runtime.json",
        "exposure.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter-tensors.json",
        "smoke-generation.json",
        "training-state.pt",
        *(name for name, _, _ in SOURCE_MODULES),
    }
    require(
        receipt["version"] == VERSION
        and receipt["status"] == "training_complete_fresh_reload_pending"
        and required <= set(receipt["files_sha256"]),
        "Reload requires a completed Gemma run with all consumed files hash-bound.",
    )
    for name, expected in receipt["files_sha256"].items():
        smoke.check_file(smoke.child(previous, name), expected)
    saved, exposure = (
        smoke.read(previous / "run.json"),
        smoke.read(previous / "exposure.json"),
    )
    require(
        saved["configuration_sha256"] == config_sha
        and receipt["files_sha256"]["configuration.json"] == config_sha
        and saved["status"] == receipt["status"]
        and saved["mode"] == "train"
        and saved["phase"] == config["phase"]
        and saved["pid"] != os.getpid()
        and saved["training_updates"]
        == exposure["completed_updates"]
        == config["training"]["max_updates"]
        and saved["optimizer_committed_rows"]
        == exposure["phases"]["optimizer_committed"]["rows"]
        == config["training"]["draw_count"]
        and exposure["optimizer_commit_uncertain"] is False
        and exposure["pending_backward_rows_not_committed"] == 0,
        "Fresh reload requires this exact fully committed phase and a separate process.",
    )
    return receipt


def snapshot_configuration(source, destination, expected_sha256):
    contents = Path(source).read_bytes()
    require(
        hashlib.sha256(contents).hexdigest() == expected_sha256,
        "Configuration changed after preflight.",
    )
    Path(destination).write_bytes(contents)


def run(config_path, output, previous=None):
    config, root, bundle, rows, diagnostics, verified = preflight(config_path)
    require(
        verified["environment"]["matched"],
        "Installed environment differs from the pinned runtime.",
    )
    output = Path(output).resolve()
    previous = Path(previous).resolve() if previous else None
    require(
        not output.exists()
        and not output.is_relative_to(root)
        and (previous is None or not output.is_relative_to(previous)),
        "Use a fresh output outside immutable input and prior run directories.",
    )
    if previous:
        verify_previous(previous, config, verified["configuration_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    exposure = curriculum.Exposure()
    info = {
        **verified,
        "status": "loading",
        "mode": "fresh_reload" if previous else "train",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "optimizer_state_resumed": False,
        "initialization": "original_native_base_with_saved_lora"
        if previous
        else "original_native_base_with_fresh_lora",
        "resume_supported": False,
        "wall_limit_seconds": config["training"]["wall_limit_seconds"],
        "wall_limit_origin": "Immediately before runtime/model loading; excludes input hash preflight.",
        "wall_limit_enforcement": "Between complete operations and before optimizer steps; in-flight operations/checkpoints may finish after the deadline.",
        "recovery_checkpoints": [],
        "scope": "Train-only fixed Gemma experiment; no held-out selection or promotion.",
    }
    snapshot_configuration(
        config_path, output / "configuration.json", verified["configuration_sha256"]
    )
    for name, _, module in SOURCE_MODULES:
        (output / name).write_bytes(Path(module).read_bytes())
    smoke.write(output / "run.json", info)
    model = optimizer = torch = None

    def deadline():
        curriculum.check_deadline(started, config["training"]["wall_limit_seconds"])

    try:
        import torch
        from peft import PeftModel, get_peft_model_state_dict

        _, model, tokenizer, device, runtime = load_runtime(
            config, root, bundle["tokens"]
        )
        smoke.write(output / "runtime.json", runtime)
        deadline()

        def generated(result):
            curriculum.append(output / "smoke-generation-events.jsonl", result)
            print(
                json.dumps(
                    {
                        "event": "smoke_generation_complete",
                        "case_id": result["case_id"],
                        "output_tokens": len(result["output_ids"]),
                    }
                ),
                flush=True,
            )
            deadline()

        if previous:
            model.requires_grad_(False)
            model = PeftModel.from_pretrained(
                model,
                previous / "adapter",
                is_trainable=False,
                local_files_only=True,
                autocast_adapter_dtype=True,
            )
            require(
                all(
                    p.dtype == (torch.float32 if ".lora_" in n else torch.bfloat16)
                    for n, p in model.named_parameters()
                ),
                "Reload precision regime changed.",
            )
            actual = smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            require(
                actual == smoke.read(previous / "adapter-tensors.json"),
                "Reloaded adapter tensors differ.",
            )
            results = generate_smoke(
                model, tokenizer, diagnostics, config, torch, device, generated
            )
            smoke.write(output / "smoke-generation.json", results)
            expected = smoke.read(previous / "smoke-generation.json")
            parity = {
                "adapter_tensors_identical": True,
                "optimizer_state_resumed": False,
                "cases": [
                    {"case_id": a["case_id"], "identical": a == b}
                    for a, b in zip(results, expected, strict=True)
                ],
            }
            smoke.write(output / "reload-parity.json", parity)
            require(
                all(r["identical"] for r in parity["cases"]),
                "Greedy output changed after fresh reload.",
            )
            info.update(
                status="fresh_reload_passed_local_export_pending",
                previous_manifest_sha256=smoke.digest(previous / "manifest.json"),
                reloaded_training_updates=config["training"]["max_updates"],
            )
        else:
            model, trainable = enable_training(model, config, torch)
            initial = smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            runtime["after_adapter_parameter_census"] = parameter_census(model)
            smoke.write(output / "runtime.json", runtime)
            smoke.write(
                output / "initialization.json",
                {
                    "initialization": "original_native_base_with_fresh_lora",
                    "smoke_adapter_loaded": False,
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

            def record_row(event):
                event["elapsed_seconds"] = time.monotonic() - started
                curriculum.append(output / "training-rows.jsonl", event)
                print(json.dumps({"event": "microbatch", **event}), flush=True)

            def record_update(event):
                event["elapsed_seconds"] = time.monotonic() - started
                info.update(
                    training_updates=exposure.updates, seconds=event["elapsed_seconds"]
                )
                curriculum.append(output / "training-events.jsonl", event)
                smoke.write(output / "exposure.json", exposure.summary())
                smoke.write(output / "run.json", info)
                if exposure.updates in config["training"]["recovery_updates"]:
                    path = save_checkpoint(
                        output,
                        f"checkpoint-{exposure.updates:06d}",
                        model,
                        optimizer,
                        torch,
                        verified["configuration_sha256"],
                        exposure,
                        "preregistered_fixed_interval",
                    )
                    info["recovery_checkpoints"].append(path)
                    smoke.write(output / "run.json", info)
                    print(
                        json.dumps(
                            {"event": "recovery_checkpoint_complete", "path": path}
                        ),
                        flush=True,
                    )
                print(json.dumps({"event": "optimizer_update", **event}), flush=True)

            info["status"] = "training"
            smoke.write(output / "run.json", info)
            curriculum.train_draws(
                model,
                rows,
                optimizer,
                torch,
                config["training"],
                device,
                exposure,
                record_row,
                record_update,
                deadline,
            )
            deadline()
            final = curriculum.save_training_state(
                output,
                model,
                optimizer,
                torch,
                verified["configuration_sha256"],
                exposure,
            )
            require(set(initial) == set(final), "Adapter tensor census changed.")
            changed = [name for name in final if initial[name] != final[name]]
            require(changed, "No Gemma adapter tensors changed.")
            saved_config = smoke.read(output / "adapter/adapter_config.json")
            require(
                saved_config["base_model_name_or_path"] == REPOSITORY
                and saved_config["revision"] == REVISION,
                "Saved adapter lost native provenance.",
            )
            tokenizer.save_pretrained(output / "tokenizer")
            deadline()
            results = generate_smoke(
                model, tokenizer, diagnostics, config, torch, device, generated
            )
            smoke.write(output / "smoke-generation.json", results)
            smoke.write(output / "exposure.json", exposure.summary())
            info.update(
                status="training_complete_fresh_reload_pending",
                training_updates=exposure.updates,
                consumed_rows=exposure.phases["backward"]["rows"],
                optimizer_committed_rows=exposure.phases["optimizer_committed"]["rows"],
                changed_adapter_tensors=changed,
                unique_cases=len(
                    exposure.phases["optimizer_committed"]["counts"]["case_id"]
                ),
            )
        info.update(
            seconds=time.monotonic() - started,
            finished_unix=time.time(),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        )
        smoke.write(output / "run.json", info)
        manifest(output, info["status"])
        return info
    except BaseException as error:
        info.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            seconds=time.monotonic() - started,
            finished_unix=time.time(),
            training_updates=exposure.updates,
        )
        smoke.write(output / "exposure.json", exposure.summary())
        if (
            not previous
            and model is not None
            and optimizer is not None
            and torch is not None
        ):
            try:
                optimizer.zero_grad(set_to_none=True)
                path = save_checkpoint(
                    output,
                    f"recovery-after-{exposure.updates:06d}",
                    model,
                    optimizer,
                    torch,
                    verified["configuration_sha256"],
                    exposure,
                    "failure_or_wall_limit_partial_gradients_discarded",
                )
                info["recovery_checkpoints"].append(path)
            except BaseException as checkpoint_error:
                info["recovery_checkpoint_error"] = (
                    f"{type(checkpoint_error).__name__}: {checkpoint_error}"
                )
        if torch is not None and torch.cuda.is_available():
            info.update(
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
            )
        smoke.write(output / "run.json", info)
        manifest(output, "failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "reload"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name != "preflight":
            command.add_argument("--output", required=True, type=Path)
        if name == "reload":
            command.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    result = (
        preflight(args.config)[-1]
        if args.command == "preflight"
        else run(args.config, args.output, getattr(args, "run", None))
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
