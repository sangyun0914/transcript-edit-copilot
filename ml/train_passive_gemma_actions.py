"""Train-only Gemma T1 weight continuation with a fresh optimizer and fresh reload."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import prepare_passive_actions_training as preparation
import train_passive_gemma as gemma


VERSION = "passive-gemma-actions-sft-v1"
CONTRACT_VERSION = "passive-nep-actions-v1"
REPOSITORY, REVISION = gemma.REPOSITORY, gemma.REVISION
smoke, curriculum = gemma.smoke, gemma.curriculum
require, bound = smoke.require, curriculum.bound
INITIALIZATION = "verified_t1_adapter_fresh_optimizer"
ANCHOR_PINS = {
    "anchor_configuration": "545f77eaee18b535cd4e991917bbc7100bddc0f11bccbf4d82660cb046ee97b9",
    "anchor_training_manifest": "a6a035984cd6443d0aa341921abb8f753a01f91ce121996ac0c7d5921c325243",
    "anchor_reload_manifest": "6204367caef727a9815b6b10c0c63806f315abac599aca2ad2f7fce4b507144c",
    "anchor_final_verification": "376c9e27211e0f2d2f795b18503bdb0e7b1bcd33c2a8de7fddb4f839d7c88230",
}
ANCHOR_BINDINGS = (
    *ANCHOR_PINS,
    "anchor_run",
    "anchor_runtime",
    "anchor_exposure",
    "anchor_adapter_tensors",
    "anchor_reload_run",
    "anchor_reload_parity",
)
SOURCE_MODULES = (
    ("trainer.py", "trainer_sha256", __file__),
    ("prepare_passive_actions_training.py", "preparer_sha256", preparation.__file__),
    ("train_passive_gemma.py", "gemma_trainer_sha256", gemma.__file__),
    ("train_passive_sft.py", "smoke_trainer_sha256", smoke.__file__),
    ("train_passive_curriculum.py", "curriculum_trainer_sha256", curriculum.__file__),
    ("prepare_passive_gemma.py", "gemma_preparer_sha256", gemma.preparation.__file__),
    (
        "prepare_passive_curriculum.py",
        "curriculum_preparer_sha256",
        gemma.curriculum_preparation.__file__,
    ),
    (
        "prepare_passive_sft.py",
        "smoke_preparer_sha256",
        gemma.smoke_preparation.__file__,
    ),
)
COMMON_TRAINING = {
    "seed": 3407,
    "sampling_seed": 3407,
    "micro_batch_size": 1,
    "gradient_accumulation": 4,
    "rank": 16,
    "alpha": 16,
    "dropout": 0,
    "bias": "none",
    "learning_rate": 1e-4,
    "optimizer": "adamw_torch",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "scheduler": "constant",
    "wall_limit_seconds": 3600,
    "max_sequence_tokens": 4096,
    "max_completion_tokens": 512,
    "loss_implementation": "completion_positions_cross_entropy",
    "normalization": "mean_completion_loss_per_row_then_mean_microbatches",
    "final_partial_accumulation": "actual_microbatch_count",
    "initialization": INITIALIZATION,
    "optimizer_state_resumed": False,
    "passes": 1,
    "replacement": False,
    "truncation": False,
    "packing": False,
    "recovery_updates": [],
}
GENERATION = {
    "max_new_tokens": 512,
    "do_sample": False,
    "use_cache": True,
    "num_beams": 1,
    "eos_token_id": [1, 106, 50],
    "pad_token_id": 0,
    "bos_token_id": 2,
}


def phase_budget(phase):
    return (
        {
            "draw_count": 8,
            "max_updates": 2,
            "max_total_sequence_tokens": 10993,
            "max_total_completion_tokens": 106,
        }
        if phase == "smoke"
        else {
            "draw_count": 42,
            "max_updates": 11,
            "max_total_sequence_tokens": 56276,
            "max_total_completion_tokens": 598,
        }
    )


def same_assets(left, right):
    for section, keys in (
        ("base", ("repository", "revision", "files_sha256", "absent_files")),
        ("tokenizer", ("files_sha256", "template_sha256")),
    ):
        require(
            all(left[section][key] == right[section][key] for key in keys),
            f"Native {section} identities differ.",
        )


def validate_configuration(config, anchor=None):
    require(
        config["version"] == VERSION
        and config["candidate_id"] == "gemma26"
        and config["contract_version"] == CONTRACT_VERSION
        and config["phase"] in ("smoke", "substantive"),
        "Unknown all-action Gemma stage.",
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
        "target_modules": gemma.target_names(),
    }
    require(
        all(config["model"].get(k) == v for k, v in fixed_model.items())
        and isinstance(config["model"]["gpu_name_contains"], str)
        and config["model"]["gpu_name_contains"].strip()
        and type(config["model"]["minimum_gpu_bytes"]) is int
        and config["model"]["minimum_gpu_bytes"] >= 75000000000
        and config["base"]["repository"] == REPOSITORY
        and config["base"]["revision"] == REVISION,
        "Require the original Gemma BF16 base and exact 205 FP32-LoRA targets.",
    )
    require(
        set(config["libraries"]) == set(smoke.LIBRARIES)
        and all(
            isinstance(v, str) and v and not any(c in v for c in "*<>=,")
            for v in config["libraries"].values()
        )
        and config["absent_libraries"] == ["torchao"],
        "Pin all runtime versions and incompatible torchao absence.",
    )
    fixed = COMMON_TRAINING | phase_budget(config["phase"])
    training = config["training"]
    require(
        all(training.get(k) == v for k, v in fixed.items())
        and type(training["seed"]) is int
        and type(training["sampling_seed"]) is int
        and len(training["sample_order"]) == training["draw_count"]
        and len(set(training["sample_order"])) == training["draw_count"]
        and config["generation"] == GENERATION,
        "Fixed continuation recipe, exact phase budget or native terminals changed.",
    )
    for _, key, _ in SOURCE_MODULES:
        require(
            isinstance(config[key], str)
            and len(config[key]) == 64
            and set(config[key]) <= set("0123456789abcdef"),
            "Pin every source file.",
        )
    require(
        set(config["anchor"]["files_sha256"])
        == {"adapter_config.json", "adapter_model.safetensors"},
        "Copy only the exact T1 adapter, not optimizer state or another adapter.",
    )
    if anchor is not None:
        gemma.validate_configuration(anchor)
        require(anchor["phase"] == "substantive", "Anchor must be genuine final T1.")
        same_assets(config, anchor)
        for key in ("libraries", "absent_libraries", "generation"):
            require(config[key] == anchor[key], f"T1 runtime recipe changed: {key}.")
        for key, original_key in (
            ("gemma_trainer_sha256", "trainer_sha256"),
            ("smoke_trainer_sha256", "smoke_trainer_sha256"),
            ("curriculum_trainer_sha256", "curriculum_trainer_sha256"),
            ("gemma_preparer_sha256", "preparer_sha256"),
            ("curriculum_preparer_sha256", "curriculum_preparer_sha256"),
            ("smoke_preparer_sha256", "smoke_preparer_sha256"),
        ):
            require(
                config[key] == anchor[original_key], "Frozen T1 helper source changed."
            )
        hardware = {"gpu_name_contains", "minimum_gpu_bytes"}
        require(
            {k: v for k, v in config["model"].items() if k not in hardware}
            == {k: v for k, v in anchor["model"].items() if k not in hardware},
            "T1 native/adapter numerical regime changed.",
        )


def validate_protocol(protocol, config, prepared):
    phases = {}
    for phase, key in (("smoke", "smoke"), ("substantive", "training")):
        value = prepared[key]
        phases[phase] = {
            "draw_count": value["draw_count"],
            "updates": value["optimizer_updates"],
            "final_accumulation": value["final_update_rows"],
            "sequence_tokens": value["max_total_sequence_tokens"],
            "completion_tokens": value["max_total_completion_tokens"],
            "sample_order": value["sample_order"],
        }
    require(
        protocol["version"] == "passive-actions-gemma-training-protocol-v1"
        and protocol["candidate_id"] == config["candidate_id"]
        and protocol["contract_version"] == CONTRACT_VERSION
        and protocol["preparation_sha256"]
        == config["bindings"]["preparation"]["sha256"]
        and protocol["anchor"]
        == {
            name.removeprefix("anchor_") + "_sha256": digest
            for name, digest in ANCHOR_PINS.items()
        }
        and protocol["phases"] == phases
        and protocol["training_recipe"] == COMMON_TRAINING
        and protocol["generation"] == GENERATION
        and protocol["label_eos_token_id"] == 1
        and protocol["development"]
        == {
            "during_training": False,
            "case_count": 14,
            "checkpoint_selection": "final_update_11_only",
            "no_auto_extra_training": True,
        }
        and protocol["final_test_bodies_read"] is False,
        "Preregister the exact train-only continuation and final-only diagnostic.",
    )


def validate_anchor(config, root):
    evidence = {
        name: bound(config["bindings"][name], root)[1] for name in ANCHOR_BINDINGS
    }
    hashes = {name: config["bindings"][name]["sha256"] for name in ANCHOR_BINDINGS}
    require(
        all(hashes[name] == sha for name, sha in ANCHOR_PINS.items()),
        "Only the independently verified original final T1 is an anchor.",
    )
    original = evidence["anchor_configuration"]
    validate_configuration(config, original)
    train, reload = (
        evidence["anchor_training_manifest"],
        evidence["anchor_reload_manifest"],
    )
    for name, file in (
        ("anchor_configuration", "configuration.json"),
        ("anchor_run", "run.json"),
        ("anchor_runtime", "runtime.json"),
        ("anchor_exposure", "exposure.json"),
        ("anchor_adapter_tensors", "adapter-tensors.json"),
    ):
        require(
            train["files_sha256"].get(file) == hashes[name],
            "Consumed anchor metadata is absent from its immutable manifest.",
        )
    require(
        reload["files_sha256"].get("run.json") == hashes["anchor_reload_run"]
        and reload["files_sha256"].get("reload-parity.json")
        == hashes["anchor_reload_parity"]
        and reload["files_sha256"].get("configuration.json")
        == hashes["anchor_configuration"],
        "Consumed anchor reload metadata is not hash-bound.",
    )
    run, fresh, parity = (
        evidence[name]
        for name in ("anchor_run", "anchor_reload_run", "anchor_reload_parity")
    )
    require(
        train["version"] == reload["version"] == gemma.VERSION
        and train["status"] == run["status"] == "training_complete_fresh_reload_pending"
        and run["mode"] == "train"
        and run["phase"] == "substantive"
        and run["configuration_sha256"] == hashes["anchor_configuration"]
        and run["training_updates"] == 203
        and run["optimizer_committed_rows"] == 809
        and reload["status"]
        == fresh["status"]
        == "fresh_reload_passed_local_export_pending"
        and fresh["mode"] == "fresh_reload"
        and fresh["previous_manifest_sha256"] == hashes["anchor_training_manifest"]
        and fresh["configuration_sha256"] == hashes["anchor_configuration"]
        and fresh["reloaded_training_updates"] == 203
        and fresh["pid"] != run["pid"]
        and fresh["started_unix"] > run["finished_unix"]
        and parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and len(parity["cases"]) == 8
        and all(c["identical"] is True for c in parity["cases"]),
        "The exact final T1 training and independent fresh reload must have passed.",
    )
    exposure = evidence["anchor_exposure"]
    require(
        exposure["completed_updates"] == 203
        and exposure["optimizer_commit_uncertain"] is False
        and exposure["pending_backward_rows_not_committed"] == 0,
        "Anchor optimizer exposure is incomplete.",
    )
    for phase in curriculum.PHASES:
        value = exposure["phases"][phase]
        require(
            value["rows"] == 809
            and value["sequence_tokens"] == 1851933
            and value["completion_tokens"] == 13395
            and value["counts"]["case_id"]
            == dict(Counter(original["training"]["sample_order"])),
            "Anchor train exposure differs from its registered draw stream.",
        )
    verification = evidence["anchor_final_verification"]
    require(
        verification["status"] == "passed"
        and verification["verification_kind"]
        == "verified_complete_substantive_training_and_fresh_reload"
        and all(v is True for v in verification["checks"].values()),
        "Require the genuine complete independent T1 verification.",
    )
    for key, name in (
        ("configuration", "anchor_configuration"),
        ("training_manifest", "anchor_training_manifest"),
        ("reload_manifest", "anchor_reload_manifest"),
    ):
        require(
            verification["bindings"][key]["sha256"] == hashes[name],
            "T1 independent verification binds a different run.",
        )
    runtime = evidence["anchor_runtime"]
    require(
        set(runtime["targets"]) == set(config["model"]["target_modules"])
        and runtime["numerical_regime"] == "native_bfloat16_lora_float32",
        "Anchor runtime target/precision census differs.",
    )
    adapter = smoke.child(root, config["anchor"]["path"])
    for file, expected in config["anchor"]["files_sha256"].items():
        require(
            train["files_sha256"].get("adapter/" + file) == expected,
            "Anchor adapter does not belong to final T1.",
        )
        smoke.check_file(smoke.child(adapter, file), expected)
    peft = smoke.read(adapter / "adapter_config.json")
    require(
        peft["base_model_name_or_path"] == REPOSITORY
        and peft["revision"] == REVISION
        and peft["r"] == peft["lora_alpha"] == 16
        and peft["lora_dropout"] == 0
        and peft["bias"] == "none"
        and peft["peft_type"] == "LORA"
        and set(peft["target_modules"]) == set(config["model"]["target_modules"]),
        "The anchor adapter architecture/scaling/provenance changed.",
    )
    tensors = evidence["anchor_adapter_tensors"]
    require(len(tensors) == 410, "Require all 410 anchor tensor records.")
    return adapter, tensors, hashes


def flatten_row(row, index):
    require(row["draw_index"] == index, "Prepared draw indexes must be contiguous.")
    return {**row, **row["provenance"]}


def validate_prepared(config, bundle):
    prepared = bundle["configuration"]
    require(
        prepared["version"] == "passive-actions-gemma-training-preparation-v1"
        and prepared["status"] == "frozen_train_only_actions_preparation"
        and prepared["candidate_id"] == "gemma26"
        and prepared["contract_version"] == CONTRACT_VERSION
        and prepared["native_base"] == {"repository": REPOSITORY, "revision": REVISION}
        and prepared["tokenizer"]["files_sha256"] == config["tokenizer"]["files_sha256"]
        and prepared["tokenizer"]["eos_token_id"] == 1
        and prepared["sampling"]
        == {
            "algorithm": "random.Random(3407).shuffle(sorted(train_case_ids))",
            "seed": 3407,
            "seed_type": "int",
            "replacement": False,
            "passes": 1,
            "draw_count": 42,
        }
        and prepared["initialization"]
        == "exact_verified_T1_adapter_on_original_native_base"
        and prepared["optimizer_initialization"] == "fresh_AdamW_not_T1_state_resume",
        "Only the new exact train-only all-action native preparation is admitted.",
    )
    full = [flatten_row(row, i) for i, row in enumerate(bundle["rows"])]
    diagnostics = [flatten_row(row, i) for i, row in enumerate(bundle["diagnostics"])]
    for phase, rows, key in (
        ("smoke", diagnostics, "smoke"),
        ("substantive", full, "training"),
    ):
        fixed, value = phase_budget(phase), prepared[key]
        require(
            len(rows) == fixed["draw_count"]
            and len({r["case_id"] for r in rows}) == len(rows)
            and [r["case_id"] for r in rows] == value["sample_order"]
            and value["optimizer_updates"] == fixed["max_updates"]
            and value["final_update_rows"] == (4 if phase == "smoke" else 2)
            and sum(len(r["input_ids"]) for r in rows)
            == fixed["max_total_sequence_tokens"]
            and sum(len(r["input_ids"]) - r["completion_start"] for r in rows)
            == fixed["max_total_completion_tokens"],
            "Prepared phase budget/order differs.",
        )
        for row in rows:
            ids, prefix = row["input_ids"], row["completion_start"]
            require(
                0 < prefix < len(ids) <= 4096
                and len(ids) - prefix <= 512
                and ids[-1] == 1
                and row["attention_mask"] == [1] * len(ids)
                and row["labels"] == [-100] * prefix + ids[prefix:]
                and all(type(t) is int and 0 <= t < 262144 for t in ids)
                and all(
                    isinstance(row[k], str) and row[k] for k in curriculum.CATEGORIES
                ),
                "Complete sequence masks, EOS or provenance differ.",
            )
    require(
        len(bundle["tokens"]) == len(bundle["cases"]) == 42
        and all(c["split"] == "train" for c in bundle["cases"])
        and Counter(r["action"] for r in full)
        == {
            "no_action": 6,
            "correct_text": 6,
            "delete_span": 6,
            "trim_silence": 6,
            "remove_silence": 6,
            "split_paragraph": 6,
            "merge_paragraph": 6,
        }
        and Counter(r["language"] for r in full) == {"en": 21, "ko": 21}
        and [r["case_id"] for r in diagnostics] == prepared["diagnostic_case_order"],
        "Require all 42 train cases and eight fixed all-action diagnostics.",
    )
    rows = diagnostics if config["phase"] == "smoke" else full
    require(
        config["training"]["sample_order"] == [r["case_id"] for r in rows],
        "Configuration changed the prepared phase order.",
    )
    return rows, diagnostics


def validate_smoke_gate(config, root):
    evidence = {
        name: bound(config["bindings"][name], root)[1] for name in gemma.SMOKE_BINDINGS
    }
    hashes = {name: config["bindings"][name]["sha256"] for name in gemma.SMOKE_BINDINGS}
    original = evidence["smoke_configuration"]
    validate_configuration(original)
    require(
        original["phase"] == "smoke", "Require this new-contract engineering smoke."
    )
    same_assets(config, original)
    for key in ("model", "libraries", "absent_libraries", "generation"):
        require(
            config[key] == original[key], "Substantive differs from qualified smoke."
        )
    require(
        config["anchor"]["files_sha256"] == original["anchor"]["files_sha256"],
        "Substantive differs from the qualified smoke anchor.",
    )
    for _, key, _ in SOURCE_MODULES:
        require(config[key] == original[key], "Smoke and substantive sources differ.")
    for name in (*ANCHOR_BINDINGS, "preparation", "native_pin", "evaluation_protocol"):
        require(
            config["bindings"][name]["sha256"] == original["bindings"][name]["sha256"],
            "Smoke and substantive input/protocol lineage differs.",
        )
    train, run, reload, fresh, parity = (
        evidence[name]
        for name in (
            "smoke_training_manifest",
            "smoke_run",
            "smoke_reload_manifest",
            "smoke_reload_run",
            "smoke_reload_parity",
        )
    )
    require(
        train["version"] == reload["version"] == VERSION
        and train["files_sha256"].get("configuration.json")
        == hashes["smoke_configuration"]
        and train["files_sha256"].get("run.json") == hashes["smoke_run"]
        and reload["files_sha256"].get("run.json") == hashes["smoke_reload_run"]
        and reload["files_sha256"].get("reload-parity.json")
        == hashes["smoke_reload_parity"]
        and train["status"] == run["status"] == "training_complete_fresh_reload_pending"
        and run["phase"] == "smoke"
        and run["mode"] == "train"
        and run["training_updates"] == 2
        and run["optimizer_committed_rows"] == 8
        and run["initialization"] == INITIALIZATION
        and run["optimizer_state_resumed"] is False
        and run["configuration_sha256"] == hashes["smoke_configuration"]
        and reload["status"]
        == fresh["status"]
        == "fresh_reload_passed_local_export_pending"
        and fresh["mode"] == "fresh_reload"
        and fresh["phase"] == "smoke"
        and fresh["configuration_sha256"] == hashes["smoke_configuration"]
        and fresh["previous_manifest_sha256"] == hashes["smoke_training_manifest"]
        and fresh["reloaded_training_updates"] == 2
        and fresh["pid"] != run["pid"]
        and fresh["started_unix"] > run["finished_unix"]
        and parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and [r["case_id"] for r in parity["cases"]]
        == original["training"]["sample_order"]
        and all(r["identical"] is True for r in parity["cases"]),
        "New substantive continuation requires the actual complete smoke and separate reload.",
    )


def preflight(config_path):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, smoke.read(config_path)
    validate_configuration(config)
    for _, key, module in SOURCE_MODULES:
        smoke.check_file(Path(module), config[key])
    prep_path, prep_manifest = bound(config["bindings"]["preparation"], root)
    bundle = preparation.verify_bundle(prep_path.parent)
    require(
        bundle["manifest"] == prep_manifest,
        "Prepared manifest changed during readback.",
    )
    rows, diagnostics = validate_prepared(config, bundle)
    _, protocol = bound(config["bindings"]["evaluation_protocol"], root)
    validate_protocol(protocol, config, bundle["configuration"])
    _, pin = bound(config["bindings"]["native_pin"], root)
    gemma.validate_assets(config, root, pin)
    anchor_dir, anchor_tensors, anchor_hashes = validate_anchor(config, root)
    if config["phase"] == "substantive":
        validate_smoke_gate(config, root)
    summary = {
        "version": VERSION,
        "candidate_id": "gemma26",
        "phase": config["phase"],
        "contract_version": CONTRACT_VERSION,
        "status": "configuration_and_inputs_verified",
        "configuration_sha256": smoke.digest(config_path),
        **{key: config[key] for _, key, _ in SOURCE_MODULES},
        "preparation_sha256": smoke.digest(prep_path),
        "native_pin_sha256": config["bindings"]["native_pin"]["sha256"],
        "evaluation_protocol_sha256": config["bindings"]["evaluation_protocol"][
            "sha256"
        ],
        "anchor_evidence_sha256": anchor_hashes,
        "anchor_adapter_sha256": config["anchor"]["files_sha256"][
            "adapter_model.safetensors"
        ],
        "environment": smoke.environment(config),
        "training_updates": 0,
        "draw_count": len(rows),
        "unique_planned_cases": len(rows),
        "diagnostic_rows": 8,
        "sequence_tokens": config["training"]["max_total_sequence_tokens"],
        "completion_tokens": config["training"]["max_total_completion_tokens"],
        "development_bodies_read": 0,
        "final_test_bodies_read": 0,
        "padding": "None: batch one, full frozen sequences.",
    }
    return config, root, bundle, rows, diagnostics, anchor_dir, anchor_tensors, summary


def load_adapter(model, adapter_dir, expected, config, torch, training):
    from peft import PeftModel, get_peft_model_state_dict

    require(
        not isinstance(model, PeftModel) and not hasattr(model, "peft_config"),
        "Load exactly one adapter on the original unmerged native base.",
    )
    model.requires_grad_(False)
    model = PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=training,
        local_files_only=True,
        autocast_adapter_dtype=True,
    )
    require(
        set(model.peft_config) == {"default"},
        "No stacked or alternate adapter is admitted.",
    )
    actual = smoke.tensor_records(
        get_peft_model_state_dict(model, save_embedding_layers=False), torch
    )
    require(
        actual == expected,
        "Loaded adapter differs from the exact bound tensor records.",
    )
    require(
        all(
            p.dtype == (torch.float32 if ".lora_" in name else torch.bfloat16)
            for name, p in model.named_parameters()
        ),
        "Native BF16/adapter FP32 changed.",
    )
    adapter = model.peft_config["default"]
    require(
        adapter.r == config["training"]["rank"]
        and adapter.lora_alpha == config["training"]["alpha"]
        and adapter.lora_dropout == 0
        and adapter.bias == "none"
        and set(adapter.target_modules) == set(config["model"]["target_modules"]),
        "Loaded adapter configuration differs from the fixed training recipe.",
    )
    trainable = {
        name: p.numel() for name, p in model.named_parameters() if p.requires_grad
    }
    if training:
        targets = {
            name.split(".lora_")[0].removeprefix("base_model.model.")
            for name in trainable
        }
        require(
            targets == set(config["model"]["target_modules"])
            and len(trainable) == 2 * len(targets)
            and sum(trainable.values())
            == config["model"]["expected_trainable_parameters"]
            and all(".lora_A." in n or ".lora_B." in n for n in trainable),
            "Only the exact saved adapter tensors may receive gradients.",
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        model.train()
    else:
        require(not trainable, "Fresh reload must be inference-only.")
        model.eval()
    adapter.base_model_name_or_path, adapter.revision = REPOSITORY, REVISION
    return model, trainable, actual


def manifest(directory, status):
    smoke.write(
        directory / "manifest.json",
        {
            "version": VERSION,
            "status": status,
            "files_sha256": {
                str(p.relative_to(directory)): smoke.digest(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file() and p != directory / "manifest.json"
            },
        },
    )


def verify_previous(previous, config, config_sha, rows):
    receipt = smoke.read(previous / "manifest.json")
    required = {
        "run.json",
        "configuration.json",
        "runtime.json",
        "initialization.json",
        "exposure.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "adapter-tensors.json",
        "smoke-generation.json",
        "training-state.pt",
        "training-rows.jsonl",
        "training-events.jsonl",
        *(n for n, _, _ in SOURCE_MODULES),
    }
    require(
        receipt["version"] == VERSION
        and receipt["status"] == "training_complete_fresh_reload_pending"
        and required <= set(receipt["files_sha256"]),
        "Reload requires a complete all-action run with every consumed file bound.",
    )
    actual_files = {
        str(p.relative_to(previous)) for p in previous.rglob("*") if p.is_file()
    }
    require(
        actual_files == set(receipt["files_sha256"]) | {"manifest.json"},
        "Previous run has unbound or missing files.",
    )
    for name, expected in receipt["files_sha256"].items():
        smoke.check_file(smoke.child(previous, name), expected)
    saved = smoke.read(previous / "run.json")
    expected_exposure = curriculum.Exposure()
    for row in rows:
        for phase in curriculum.PHASES:
            expected_exposure.record(phase, row)
    expected_exposure.updates = config["training"]["max_updates"]
    require(
        smoke.read(previous / "exposure.json") == expected_exposure.summary()
        and saved["status"] == receipt["status"]
        and saved["version"] == VERSION
        and saved["mode"] == "train"
        and saved["phase"] == config["phase"]
        and saved["initialization"] == INITIALIZATION
        and saved["optimizer_state_resumed"] is False
        and saved["configuration_sha256"]
        == receipt["files_sha256"]["configuration.json"]
        == config_sha
        and saved["pid"] != os.getpid()
        and saved["finished_unix"] < time.time()
        and saved["training_updates"] == config["training"]["max_updates"]
        and saved["optimizer_committed_rows"] == config["training"]["draw_count"],
        "Reload requires the exact complete phase and a subsequent independent process.",
    )
    for name, key, _ in SOURCE_MODULES:
        require(
            receipt["files_sha256"][name] == config[key],
            "Saved runtime source differs.",
        )
    return receipt


def save_recovery(output, model, optimizer, torch, config_sha, exposure):
    require(
        not exposure.optimizer_commit_uncertain,
        "Interrupted optimizer step cannot be recorded as a safe checkpoint.",
    )
    path = output / f"recovery-after-{exposure.updates:06d}"
    path.mkdir(exist_ok=False)
    curriculum.save_training_state(path, model, optimizer, torch, config_sha, exposure)
    smoke.write(
        path / "checkpoint.json",
        {
            "status": "recovery_checkpoint_complete",
            "configuration_sha256": config_sha,
            "reason": "failure_or_wall_limit_partial_gradients_discarded",
            "exposure": exposure.summary(),
            "optimizer_state_resumed": False,
            "resume_supported": False,
            "partial_gradients_saved": False,
        },
    )
    manifest(path, "recovery_checkpoint_complete")
    return path.name


def run(config_path, output, previous=None):
    config, root, bundle, rows, diagnostics, anchor_dir, anchor_tensors, verified = (
        preflight(config_path)
    )
    require(
        verified["environment"]["matched"], "Runtime differs from pinned libraries."
    )
    output = Path(output).resolve()
    previous = Path(previous).resolve() if previous else None
    require(
        not output.exists()
        and not output.is_relative_to(root)
        and (previous is None or not output.is_relative_to(previous)),
        "Use a fresh output outside immutable inputs and the prior run.",
    )
    if previous:
        verify_previous(previous, config, verified["configuration_sha256"], rows)
    output.mkdir(parents=True, exist_ok=False)
    started, exposure = time.monotonic(), curriculum.Exposure()
    info = {
        **verified,
        "status": "loading",
        "mode": "fresh_reload" if previous else "train",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "optimizer_state_resumed": False,
        "initialization": "native_base_with_saved_actions_adapter"
        if previous
        else INITIALIZATION,
        "historical_anchor_updates": 203,
        "historical_anchor_draws": 809,
        "resume_supported": False,
        "recovery_checkpoints": [],
        "wall_limit_seconds": config["training"]["wall_limit_seconds"],
        "wall_limit_origin": "Before model loading; excludes hash preflight.",
        "wall_limit_enforcement": "Between operations and before optimizer steps; in-flight operations may finish after the limit.",
        "scope": "Train-only fixed all-action continuation; no dev selection or acoustic quality claim.",
    }
    gemma.snapshot_configuration(
        config_path, output / "configuration.json", verified["configuration_sha256"]
    )
    for name, key, module in SOURCE_MODULES:
        smoke.check_file(Path(module), config[key])
        (output / name).write_bytes(Path(module).read_bytes())
    smoke.write(output / "run.json", info)
    model = optimizer = torch = None

    def deadline():
        curriculum.check_deadline(started, config["training"]["wall_limit_seconds"])

    try:
        torch, model, tokenizer, device, runtime = gemma.load_runtime(
            config, root, bundle["tokens"]
        )
        deadline()
        adapter_dir = previous / "adapter" if previous else anchor_dir
        expected = (
            smoke.read(previous / "adapter-tensors.json")
            if previous
            else anchor_tensors
        )
        model, trainable, initial = load_adapter(
            model, adapter_dir, expected, config, torch, not previous
        )
        runtime.update(
            after_adapter_parameter_census=gemma.parameter_census(model),
            initial_adapter_tensors=initial,
            anchor_adapter_sha256=verified["anchor_adapter_sha256"],
        )
        smoke.write(output / "runtime.json", runtime)

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

        if not previous:
            smoke.write(
                output / "initialization.json",
                {
                    "initialization": INITIALIZATION,
                    "anchor_adapter_loaded": True,
                    "smoke_adapter_loaded": False,
                    "optimizer_state_resumed": False,
                    "anchor_adapter_sha256": verified["anchor_adapter_sha256"],
                    "anchor_training_manifest_sha256": ANCHOR_PINS[
                        "anchor_training_manifest"
                    ],
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
            require(
                not optimizer.state, "Optimizer must start with no historical moments."
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
            require(set(initial) == set(final), "Adapter tensor inventory changed.")
            changed = [name for name in final if final[name] != initial[name]]
            require(changed, "No loaded T1 adapter tensor changed.")
            peft = smoke.read(output / "adapter/adapter_config.json")
            require(
                peft["base_model_name_or_path"] == REPOSITORY
                and peft["revision"] == REVISION,
                "Saved adapter lost original native provenance.",
            )
            tokenizer.save_pretrained(output / "tokenizer")
            results = gemma.generate_smoke(
                model, tokenizer, diagnostics, config, torch, device, generated
            )
            smoke.write(output / "smoke-generation.json", results)
            smoke.write(output / "exposure.json", exposure.summary())
            info.update(
                status="training_complete_fresh_reload_pending",
                training_updates=exposure.updates,
                consumed_rows=exposure.phases["backward"]["rows"],
                optimizer_committed_rows=exposure.phases["optimizer_committed"]["rows"],
                unique_cases=len(
                    exposure.phases["optimizer_committed"]["counts"]["case_id"]
                ),
                changed_adapter_tensors=changed,
            )
        else:
            results = gemma.generate_smoke(
                model, tokenizer, diagnostics, config, torch, device, generated
            )
            smoke.write(output / "smoke-generation.json", results)
            expected_results = smoke.read(previous / "smoke-generation.json")
            parity = {
                "adapter_tensors_identical": True,
                "optimizer_state_resumed": False,
                "cases": [
                    {"case_id": a["case_id"], "identical": a == b}
                    for a, b in zip(results, expected_results, strict=True)
                ],
            }
            smoke.write(output / "reload-parity.json", parity)
            require(
                len(parity["cases"]) == 8
                and all(r["identical"] for r in parity["cases"]),
                "Greedy output changed after independent fresh reload.",
            )
            info.update(
                status="fresh_reload_passed_local_export_pending",
                previous_manifest_sha256=smoke.digest(previous / "manifest.json"),
                reloaded_training_updates=config["training"]["max_updates"],
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
                info["recovery_checkpoints"].append(
                    save_recovery(
                        output,
                        model,
                        optimizer,
                        torch,
                        verified["configuration_sha256"],
                        exposure,
                    )
                )
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
