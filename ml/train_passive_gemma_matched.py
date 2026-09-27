"""Fresh-native Gemma matched JSON/Rewrite smoke, SFT, and separate reload."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import prepare_passive_gemma_matched as preparation
import prepare_passive_matched as matched_preparation
import train_passive_gemma as gemma


VERSION = "passive-gemma-matched-sft-v1"
FORMATS = ("json", "rewrite")
REPOSITORY, REVISION = gemma.REPOSITORY, gemma.REVISION
smoke, curriculum = gemma.smoke, gemma.curriculum
require, bound = smoke.require, curriculum.bound
load_runtime = gemma.load_runtime
enable_training = gemma.enable_training
parameter_census = gemma.parameter_census
generate_smoke = gemma.generate_smoke
snapshot_configuration = gemma.snapshot_configuration
SMOKE_BINDINGS = gemma.SMOKE_BINDINGS
T1_BINDINGS = (
    "t1_configuration",
    "t1_training_manifest",
    "t1_run",
    "t1_reload_manifest",
    "t1_reload_run",
    "t1_reload_parity",
    "t1_export_receipt",
    "t1_evaluation_execution",
    "t1_evaluation_summary",
    "t1_evaluation_verification",
)
SOURCE_MODULES = (
    ("trainer.py", "trainer_sha256", __file__),
    ("train_passive_gemma.py", "gemma_trainer_sha256", gemma.__file__),
    ("train_passive_sft.py", "smoke_trainer_sha256", smoke.__file__),
    ("train_passive_curriculum.py", "curriculum_trainer_sha256", curriculum.__file__),
    ("prepare_passive_gemma_matched.py", "preparer_sha256", preparation.__file__),
    (
        "prepare_passive_matched.py",
        "matched_preparer_sha256",
        matched_preparation.__file__,
    ),
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


def same_assets(left, right):
    for section, keys in (
        ("base", ("repository", "revision", "files_sha256", "absent_files")),
        ("tokenizer", ("files_sha256", "template_sha256")),
    ):
        require(
            all(left[section][key] == right[section][key] for key in keys),
            f"Native {section} binding changed.",
        )


def validate_configuration(config, original=None):
    require(
        config["version"] == VERSION
        and config["candidate_id"] == "gemma26"
        and config["format"] in FORMATS
        and config["phase"] in ("smoke", "substantive"),
        "Unknown matched Gemma phase/format.",
    )
    require(
        set(config["libraries"]) == set(smoke.LIBRARIES)
        and all(
            isinstance(v, str) and v and not any(c in v for c in "*<>=,")
            for v in config["libraries"].values()
        )
        and config["absent_libraries"] == ["torchao"],
        "Pin every runtime library and incompatible torchao absence.",
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
        and config["base"]["repository"] == REPOSITORY
        and config["base"]["revision"] == REVISION,
        "Gemma BF16/FP32 205-target model recipe changed.",
    )
    train, is_smoke = config["training"], config["phase"] == "smoke"
    fixed_training = {
        "seed": 3407,
        "sampling_seed": "3407",
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
        "wall_limit_seconds": 21600,
        "max_sequence_tokens": 4096,
        "max_completion_tokens": 512,
        "loss_implementation": "completion_positions_cross_entropy",
        "normalization": "mean_completion_loss_per_row_then_mean_microbatches",
        "final_partial_accumulation": "actual_microbatch_count",
        "fresh_native_initialization": True,
        "truncation": False,
        "packing": False,
    }
    require(
        all(train.get(k) == v for k, v in fixed_training.items()),
        "The fixed optimizer, initialization, or completion-only loss recipe changed.",
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
        "Retain all native generation terminals separately from label EOS 1.",
    )
    hardware = {"gpu_name_contains", "minimum_gpu_bytes"}
    require(
        isinstance(config["model"]["gpu_name_contains"], str)
        and config["model"]["gpu_name_contains"].strip()
        and type(config["model"]["minimum_gpu_bytes"]) is int
        and config["model"]["minimum_gpu_bytes"] >= 75000000000,
        "Preregister a BF16-capable GPU with sufficient memory.",
    )
    require(
        train["max_updates"] == (2 if is_smoke else 201)
        and train["draw_count"] == (8 if is_smoke else 802)
        and len(train["sample_order"]) == train["draw_count"]
        and train["recovery_updates"] == ([] if is_smoke else [50, 100, 150])
        and type(train["seed"]) is int
        and type(train["sampling_seed"]) is str,
        "Matched phase draw/update/recovery budget changed.",
    )
    for key in ("max_total_sequence_tokens", "max_total_completion_tokens"):
        require(
            type(train[key]) is int and train[key] > 0,
            "Bind exact positive token budgets.",
        )
    for _, key, _ in SOURCE_MODULES:
        require(
            isinstance(config[key], str)
            and len(config[key]) == 64
            and all(c in "0123456789abcdef" for c in config[key]),
            "Every runtime source requires a SHA256 pin.",
        )
    if original is None:
        return
    gemma.validate_configuration(original)
    require(
        original["phase"] == "substantive",
        "Bind the completed substantive Gemma T1 configuration.",
    )
    for key in ("libraries", "absent_libraries", "generation"):
        require(
            config[key] == original[key],
            f"Verified Gemma numerical/runtime recipe changed: {key}",
        )
    same_assets(config, original)
    require(
        {k: v for k, v in config["model"].items() if k not in hardware}
        == {k: v for k, v in original["model"].items() if k not in hardware},
        "Gemma BF16/FP32 205-target model recipe changed.",
    )
    variable = {
        "max_updates",
        "draw_count",
        "sample_order",
        "max_total_sequence_tokens",
        "max_total_completion_tokens",
        "recovery_updates",
    }
    require(
        {k: v for k, v in config["training"].items() if k not in variable}
        == {k: v for k, v in original["training"].items() if k not in variable},
        "The fixed T1 optimizer/loss recipe changed.",
    )
    for current, prior in (
        ("gemma_trainer_sha256", "trainer_sha256"),
        ("gemma_preparer_sha256", "preparer_sha256"),
        ("smoke_trainer_sha256", "smoke_trainer_sha256"),
        ("curriculum_trainer_sha256", "curriculum_trainer_sha256"),
        ("curriculum_preparer_sha256", "curriculum_preparer_sha256"),
        ("smoke_preparer_sha256", "smoke_preparer_sha256"),
    ):
        require(config[current] == original[prior], "Frozen T1 helper source changed.")


def validate_protocol(protocol, config, release_sha256):
    expected = {
        "candidate": "gemma26",
        "draw_count": 802,
        "sampling_seed": "3407",
        "model_seed": 3407,
        "effective_batch": 4,
        "updates": 201,
        "final_accumulation": 2,
        "independent_fresh_native_runs": True,
        "development_during_training": False,
        "checkpoint_selection": "final_update_201_only",
        "recovery_updates": [50, 100, 150],
        "max_wall_seconds": 21600,
    }
    require(
        protocol["version"] == "passive-gemma-matched-evaluation-v1"
        and protocol["candidate_id"] == "gemma26"
        and protocol["formats"] == list(FORMATS)
        and all(protocol["training_plan"].get(k) == v for k, v in expected.items()),
        "Bind the fixed independent 802-draw/201-update matched experiment.",
    )
    require(
        protocol["numerical_regime"]
        == {
            "base_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "quantization": "none",
        },
        "Matched numerical regime changed.",
    )
    engineering = protocol["engineering_smoke"]
    require(
        all(
            engineering.get(k) == v
            for k, v in {
                "draw_count": 8,
                "updates": 2,
                "fresh_reload_required": True,
                "one_smoke_per_format": True,
                "actual_peak_memory_must_be_measured": True,
            }.items()
        )
        and engineering["native_gpu"] == config["model"]["gpu_name_contains"]
        and engineering["minimum_gpu_bytes"] == config["model"]["minimum_gpu_bytes"],
        "Both formats require their registered engineering smoke/reload.",
    )
    require(
        protocol["b2"]["release_manifest"]["sha256"] == release_sha256
        and protocol["training_rows"] == 802
        and protocol["registered_requests_per_condition"] == 286
        and protocol["same_case_sampling_order_required"] is True
        and protocol["conditions"]
        == [
            "matched-json-unconstrained",
            "matched-rewrite-unconstrained",
            "matched-json-constrained",
        ]
        and protocol["primary_format_comparison"]["left"]
        == "matched-json-unconstrained"
        and protocol["primary_format_comparison"]["right"]
        == "matched-rewrite-unconstrained"
        and protocol["final_test_bodies_read"] is False
        and protocol["learning_signal_policy"]["no_auto_extra_training"] is True
        and protocol["learning_signal_policy"]["deployment_promotion"] is False,
        "Matched comparison, release, or held-out scope changed.",
    )
    generation = protocol["generation_policy"]
    require(
        all(
            generation.get(k) == v
            for k, v in config["generation"].items()
            if k != "use_cache"
        )
        and generation["label_eos_token_id"] == 1
        and generation["thinking"] is False
        and generation["raw_tokens_and_terminal_id_retained"] is True,
        "Native completion EOS/termination policy changed.",
    )
    require(
        set(protocol["t1_evidence"]) == set(T1_BINDINGS)
        and all(
            protocol["t1_evidence"][name]["sha256"]
            == config["bindings"][name]["sha256"]
            for name in T1_BINDINGS
        ),
        "T1 evidence differs from its preregistered metadata pins.",
    )


def validate_t1_evidence(config, root, release_sha256):
    evidence = {name: bound(config["bindings"][name], root)[1] for name in T1_BINDINGS}
    hashes = {name: config["bindings"][name]["sha256"] for name in T1_BINDINGS}
    _, protocol = bound(config["bindings"]["evaluation_protocol"], root)
    original = evidence["t1_configuration"]
    validate_configuration(config, original)
    manifest, run = evidence["t1_training_manifest"], evidence["t1_run"]
    re_manifest, re_run = evidence["t1_reload_manifest"], evidence["t1_reload_run"]
    parity, export = evidence["t1_reload_parity"], evidence["t1_export_receipt"]
    require(
        manifest["version"] == run["version"] == gemma.VERSION
        and manifest["status"]
        == run["status"]
        == "training_complete_fresh_reload_pending"
        and manifest["files_sha256"]["run.json"] == hashes["t1_run"]
        and manifest["files_sha256"]["configuration.json"]
        == run["configuration_sha256"]
        == hashes["t1_configuration"]
        and run["phase"] == "substantive"
        and run["mode"] == "train"
        and run["training_updates"] == 203
        and run["optimizer_committed_rows"] == 809
        and run["release_manifest_sha256"] == release_sha256
        and run["native_pin_sha256"]
        == original["bindings"]["native_pin"]["sha256"]
        == config["bindings"]["native_pin"]["sha256"]
        and run["preparation_sha256"] == original["bindings"]["preparation"]["sha256"]
        and run["evaluation_protocol_sha256"] == protocol["t1_protocol"]["sha256"]
        and run["trainer_sha256"] == config["gemma_trainer_sha256"]
        and run["initialization"] == "original_native_base_with_fresh_lora"
        and run["optimizer_state_resumed"] is False
        and run["final_test_bodies_read"] == 0,
        "T2 requires the exact completed final Gemma T1 run.",
    )
    require(
        re_manifest["version"] == re_run["version"] == gemma.VERSION
        and re_manifest["status"]
        == re_run["status"]
        == "fresh_reload_passed_local_export_pending"
        and re_manifest["files_sha256"]["run.json"] == hashes["t1_reload_run"]
        and re_manifest["files_sha256"]["configuration.json"]
        == hashes["t1_configuration"]
        and re_manifest["files_sha256"]["reload-parity.json"]
        == hashes["t1_reload_parity"]
        and re_run["configuration_sha256"] == hashes["t1_configuration"]
        and re_run["previous_manifest_sha256"] == hashes["t1_training_manifest"]
        and re_run["evaluation_protocol_sha256"] == run["evaluation_protocol_sha256"]
        and re_run["phase"] == "substantive"
        and re_run["mode"] == "fresh_reload"
        and re_run["reloaded_training_updates"] == 203
        and re_run["pid"] != run["pid"]
        and re_run["started_unix"] > run["finished_unix"]
        and re_run["optimizer_state_resumed"] is False
        and parity["optimizer_state_resumed"] is False
        and parity["adapter_tensors_identical"] is True
        and len(parity["cases"]) == len({r["case_id"] for r in parity["cases"]}) == 8
        and all(r["identical"] is True for r in parity["cases"]),
        "T1 requires an independent later exact tensor/output reload.",
    )
    require(
        export["version"] == "passive-gemma-sft-export-v1"
        and export["status"]
        == "tensor_verified_paired_export_runtime_preflight_pending"
        and export["training_record"]["sha256"]
        in (hashes["t1_run"], hashes["t1_training_manifest"])
        and export["base_repository"] == REPOSITORY
        and export["base_revision"] == REVISION
        and export["native_generation_stop_ids"] == [1, 106, 50]
        and set(export["artifacts"])
        == {"base", "adapter", "merged", "base_quantized", "merged_quantized"},
        "T1 paired export does not belong to the final native-trained adapter.",
    )
    execution, summary = (
        evidence["t1_evaluation_execution"],
        evidence["t1_evaluation_summary"],
    )
    require(
        execution["version"] == "passive-gemma-execution-v1"
        and execution["protocol_sha256"] == run["evaluation_protocol_sha256"]
        and execution["export_receipt"]["sha256"] == hashes["t1_export_receipt"]
        and execution["training_configuration"]["sha256"] == hashes["t1_configuration"]
        and execution["fresh_reload"]["sha256"] == hashes["t1_reload_manifest"]
        and summary["version"] == "passive-gemma-evaluation-v1"
        and summary["protocol_sha256"] == run["evaluation_protocol_sha256"]
        and summary["execution_sha256"] == hashes["t1_evaluation_execution"]
        and summary["registered_new_requests"] == 778
        and summary["registered_conditions"]
        == ["native-base-full", "native-base-icl", "sft-full", "sft-icl"]
        and summary["paired_export"]["artifacts"]
        == {
            "native_base": export["artifacts"]["base_quantized"],
            "sft": export["artifacts"]["merged_quantized"],
        },
        "T1 completed evaluation lost its native pair or final checkpoint lineage.",
    )
    counts = {
        "native-base-full": 289,
        "native-base-icl": 100,
        "sft-full": 289,
        "sft-icl": 100,
    }
    require(
        set(summary["runs"]) == set(summary["provenance"]) == set(counts),
        "All four registered T1 conditions are required.",
    )
    for name, count in counts.items():
        row = summary["runs"][name]
        require(
            row["run_status"] == summary["provenance"][name]["status"] == "completed"
            and row["expected_fresh_requests"] == row["observed_fresh_results"] == count
            and row["missing_requests"] == row["request_errors"] == 0,
            "T1 engineering evaluation is incomplete; quality thresholds are not a launch gate.",
        )
    verified = evidence["t1_evaluation_verification"]
    checks = verified["condition_checks"]
    require(
        verified["version"] == "passive-gemma-final-evaluation-verification-v1"
        and verified["status"] == "passed"
        and verified["registered_requests"]
        == verified["completed_verified_requests"]
        == 778
        and verified["root_exec_exit_code"] == 0
        and verified["final_report"]["sha256"] == hashes["t1_evaluation_summary"]
        and verified["final_test_bodies_read"] == 0
        and verified["deployment_promotion"] is False
        and len(checks) == len(counts)
        and {r["condition"]: r["requests"] for r in checks} == counts
        and all(
            r["receipt"]["sha256"]
            == summary["provenance"][r["condition"]]["receipt_sha256"]
            for r in checks
        ),
        "Bind the independent metadata-only verification of all 778 completed T1 requests.",
    )


def validate_prepared(config, bundle):
    prepared = bundle["configuration"]
    require(
        prepared["candidate_id"] == "gemma26"
        and prepared["frozen_train_rows"] == 802
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
            "draw_count": 802,
            "weights_applied_to_loss": False,
        },
        "Matched native/tokenizer/sampler binding changed.",
    )
    require(
        prepared["generation_metadata"]
        == {
            "completion_label_eos_token_id": 1,
            "native_generation_stop_ids": [1, 106, 50],
            "bos_token_id": 2,
            "pad_token_id": 0,
            "actual_generation_policy_verified_by_preparation": False,
        },
        "Completion EOS and native generation stops must remain distinct.",
    )
    common = prepared["training"]
    require(
        common["micro_batch_size"] == 1
        and common["gradient_accumulation"] == 4
        and common["optimizer_updates"] == 201
        and common["final_update_rows"] == 2
        and common["final_update_normalization"] == "mean_over_actual_microbatch_count"
        and common["initialization"]
        == "fresh_adapter_on_exact_native_base_not_smoke_or_T1_adapter",
        "Matched training budget/initialization changed.",
    )
    fmt, training = config["format"], config["training"]
    view, payload = prepared["views"][fmt], bundle["views"][fmt]
    require(
        view["training_view"] == fmt
        and len(bundle["cases"])
        == len(payload["tokens"])
        == len(bundle["draw_plan"])
        == 802,
        "Choose the exact admitted802 format view.",
    )
    full_rows = curriculum.selected_rows({**bundle, "tokens": payload["tokens"]})
    require(
        [r["case_id"] for r in full_rows] == common["sample_order"]
        and sum(len(r["input_ids"]) for r in full_rows)
        == view["max_total_sequence_tokens"]
        and sum(len(r["input_ids"]) - r["completion_start"] for r in full_rows)
        == view["max_total_completion_tokens"],
        "Prepared fixed draw order or format budget changed.",
    )
    diagnostics = payload["diagnostics"]["rows"]
    order = view["diagnostic_case_order"]
    require(
        len(diagnostics) == len(set(order)) == 8
        and [r["case_id"] for r in diagnostics] == order
        and all(prepared["views"][f]["diagnostic_case_order"] == order for f in FORMATS)
        and sum(len(r["input_ids"]) for r in diagnostics)
        == view["diagnostic_sequence_tokens"]
        and sum(len(r["input_ids"]) - r["completion_start"] for r in diagnostics)
        == view["diagnostic_completion_tokens"],
        "Both formats require the same eight matched diagnostics.",
    )
    if config["phase"] == "smoke":
        diagnostic = payload["diagnostics"]
        rows = curriculum.selected_rows(
            {
                "cases": diagnostic["cases"],
                "tokens": diagnostic["tokens"],
                "draw_plan": [
                    {"draw_index": i, "case_id": case_id, "train_row_index": i}
                    for i, case_id in enumerate(order)
                ],
            }
        )
    else:
        rows = full_rows
    require(
        [r["case_id"] for r in rows] == training["sample_order"]
        and len(rows) == training["draw_count"]
        and sum(len(r["input_ids"]) for r in rows)
        == training["max_total_sequence_tokens"]
        and sum(len(r["input_ids"]) - r["completion_start"] for r in rows)
        == training["max_total_completion_tokens"],
        "Configured phase must consume exactly its frozen rows and tokens.",
    )
    for row in full_rows + diagnostics:
        ids, start = row["input_ids"], row["completion_start"]
        require(
            type(start) is int
            and 0 < start <= 3584
            and 0 < len(ids) - start <= 512
            and len(ids) <= 4096
            and all(type(v) is int and 0 <= v < 262144 for v in ids)
            and ids[-1] == 1
            and row["labels"] == [-100] * start + ids[start:]
            and row["attention_mask"] == [1] * len(ids),
            "Exact completion-only mask, token bounds or label EOS changed.",
        )
    return rows, diagnostics


def validate_smoke_gate(config, root, release_sha256, diagnostic_order=None):
    if diagnostic_order is None:
        path, manifest = bound(config["bindings"]["preparation"], root)
        prepared_path = path.parent / "configuration.json"
        smoke.check_file(prepared_path, manifest["files_sha256"]["configuration.json"])
        diagnostic_order = smoke.read(prepared_path)["views"][config["format"]][
            "diagnostic_case_order"
        ]
    evidence = {
        name: bound(config["bindings"][name], root)[1] for name in SMOKE_BINDINGS
    }
    hashes = {name: config["bindings"][name]["sha256"] for name in SMOKE_BINDINGS}
    original = evidence["smoke_configuration"]
    _, t1 = bound(config["bindings"]["t1_configuration"], root)
    validate_configuration(original, t1)
    require(
        original["phase"] == "smoke"
        and original["format"] == config["format"]
        and original["training"]["sample_order"] == diagnostic_order,
        "Each substantive arm requires its own format's eight-case smoke.",
    )
    same_assets(config, original)
    for key in ("candidate_id", "model", "libraries", "absent_libraries", "generation"):
        require(original[key] == config[key], "Smoke runtime/model recipe differs.")
    for _, key, _ in SOURCE_MODULES:
        require(
            original[key] == config[key], "Smoke and substantive source pins differ."
        )
    for name in ("preparation", "native_pin", "evaluation_protocol", *T1_BINDINGS):
        require(
            original["bindings"][name]["sha256"] == config["bindings"][name]["sha256"],
            "Smoke evidence/preparation/protocol changed.",
        )
    train, run = evidence["smoke_training_manifest"], evidence["smoke_run"]
    reload, reload_run, parity = (
        evidence["smoke_reload_manifest"],
        evidence["smoke_reload_run"],
        evidence["smoke_reload_parity"],
    )
    require(
        train["version"]
        == run["version"]
        == reload["version"]
        == reload_run["version"]
        == VERSION
        and train["status"] == run["status"] == "training_complete_fresh_reload_pending"
        and run["mode"] == "train"
        and run["phase"] == "smoke"
        and run["format"] == config["format"]
        and run["training_updates"] == 2
        and run["optimizer_committed_rows"] == 8
        and run["consumed_rows"] == 8
        and run["peak_allocated_gpu_bytes"] > 0
        and run["peak_reserved_gpu_bytes"] >= run["peak_allocated_gpu_bytes"]
        and bool(run["changed_adapter_tensors"])
        and run["configuration_sha256"]
        == train["files_sha256"]["configuration.json"]
        == hashes["smoke_configuration"]
        and train["files_sha256"]["run.json"] == hashes["smoke_run"]
        and run["release_manifest_sha256"] == release_sha256
        and run["preparation_sha256"] == config["bindings"]["preparation"]["sha256"]
        and run["native_pin_sha256"] == config["bindings"]["native_pin"]["sha256"]
        and run["evaluation_protocol_sha256"]
        == config["bindings"]["evaluation_protocol"]["sha256"]
        and run["optimizer_state_resumed"] is False
        and run["final_test_bodies_read"] == 0,
        "Substantive SFT requires its genuine completed two-update smoke.",
    )
    require(
        reload["status"]
        == reload_run["status"]
        == "fresh_reload_passed_local_export_pending"
        and reload["files_sha256"]["run.json"] == hashes["smoke_reload_run"]
        and reload["files_sha256"]["configuration.json"]
        == hashes["smoke_configuration"]
        and reload["files_sha256"]["reload-parity.json"]
        == hashes["smoke_reload_parity"]
        and reload_run["previous_manifest_sha256"] == hashes["smoke_training_manifest"]
        and reload_run["configuration_sha256"] == hashes["smoke_configuration"]
        and reload_run["mode"] == "fresh_reload"
        and reload_run["phase"] == "smoke"
        and reload_run["format"] == config["format"]
        and reload_run["reloaded_training_updates"] == 2
        and reload_run["pid"] != run["pid"]
        and reload_run["started_unix"] > run["finished_unix"]
        and reload_run["optimizer_state_resumed"] is False
        and parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and [r["case_id"] for r in parity["cases"]] == diagnostic_order
        and all(r["identical"] is True for r in parity["cases"]),
        "The same-format smoke requires a subsequent independent exact reload.",
    )


def preflight(config_path):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, smoke.read(config_path)
    for _, key, module in SOURCE_MODULES:
        smoke.check_file(Path(module), config[key])
    evidence = {name: bound(config["bindings"][name], root)[1] for name in T1_BINDINGS}
    hashes = {name: config["bindings"][name]["sha256"] for name in T1_BINDINGS}
    validate_configuration(config, evidence["t1_configuration"])
    manifest_path, prepared_manifest = bound(config["bindings"]["preparation"], root)
    require(
        prepared_manifest["version"] == "passive-sft-gemma-matched-preparation-v1"
        and prepared_manifest["status"] == "frozen_matched_train_only_preparation",
        "Use a frozen matched Gemma train-only bundle.",
    )
    bundle = preparation.verify_bundle(manifest_path.parent)
    require(
        bundle["manifest"] == prepared_manifest,
        "Preparation changed during verification.",
    )
    release_sha = prepared_manifest["release_manifest_sha256"]
    _, protocol = bound(config["bindings"]["evaluation_protocol"], root)
    validate_protocol(protocol, config, release_sha)
    validate_t1_evidence(config, root, release_sha)
    _, pin = bound(config["bindings"]["native_pin"], root)
    gemma.validate_assets(config, root, pin)
    rows, diagnostics = validate_prepared(config, bundle)
    if config["phase"] == "substantive":
        validate_smoke_gate(
            config, root, release_sha, [r["case_id"] for r in diagnostics]
        )
    summary = {
        "version": VERSION,
        "phase": config["phase"],
        "format": config["format"],
        "stage": "matched_format_sft",
        "training_view": config["format"],
        "candidate_id": "gemma26",
        "status": "configuration_and_inputs_verified",
        "configuration_sha256": smoke.digest(config_path),
        **{key: config[key] for _, key, _ in SOURCE_MODULES},
        "preparation_sha256": smoke.digest(manifest_path),
        "release_manifest_sha256": release_sha,
        "evaluation_protocol_sha256": config["bindings"]["evaluation_protocol"][
            "sha256"
        ],
        "native_pin_sha256": config["bindings"]["native_pin"]["sha256"],
        "t1_evidence_sha256": hashes,
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


def verify_previous(previous, config, config_sha, rows=None, diagnostics=None):
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
        "initialization.json",
        "training-rows.jsonl",
        "training-events.jsonl",
        *(name for name, _, _ in SOURCE_MODULES),
    }
    require(
        receipt["version"] == VERSION
        and receipt["status"] == "training_complete_fresh_reload_pending"
        and required <= set(receipt["files_sha256"]),
        "Reload requires a completed Gemma run with all consumed files hash-bound.",
    )
    require(
        set(receipt["files_sha256"])
        == {
            str(p.relative_to(previous))
            for p in previous.rglob("*")
            if p.is_file() and p != previous / "manifest.json"
        },
        "Reload run inventory contains missing or unbound files.",
    )
    for name, expected in receipt["files_sha256"].items():
        smoke.check_file(smoke.child(previous, name), expected)
    for name, key, _ in SOURCE_MODULES:
        require(
            receipt["files_sha256"][name] == config[key],
            "Reload runtime source changed.",
        )
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
        and saved["format"] == config["format"]
        and saved["stage"] == "matched_format_sft"
        and saved["pid"] != os.getpid()
        and saved["finished_unix"] < time.time()
        and saved["optimizer_state_resumed"] is False
        and saved["initialization"] == "original_native_base_with_fresh_lora"
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
    training = config["training"]
    for phase in curriculum.PHASES:
        actual = exposure["phases"][phase]
        require(
            actual["rows"] == training["draw_count"]
            and actual["counts"]["case_id"] == dict(Counter(training["sample_order"]))
            and actual["sequence_tokens"] == training["max_total_sequence_tokens"]
            and actual["completion_tokens"] == training["max_total_completion_tokens"],
            "Reload exposure differs from the complete registered format draw stream.",
        )
    if rows is not None:
        expected = curriculum.Exposure()
        for row in rows:
            for phase in curriculum.PHASES:
                expected.record(phase, row)
        expected.updates = training["max_updates"]
        require(
            exposure == expected.summary(),
            "Source/episode/token exposure differs from frozen rows.",
        )
    if diagnostics is not None:
        generations = smoke.read(previous / "smoke-generation.json")
        require(
            [r["case_id"] for r in generations] == [r["case_id"] for r in diagnostics],
            "Reload must use the exact eight same-format diagnostics.",
        )
    return receipt


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
        verify_previous(
            previous, config, verified["configuration_sha256"], rows, diagnostics
        )
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
        "scope": "Train-only independent fresh-native matched-format experiment; no held-out selection or promotion.",
    }
    snapshot_configuration(
        config_path, output / "configuration.json", verified["configuration_sha256"]
    )
    for name, key, module in SOURCE_MODULES:
        snapshot_configuration(module, output / name, config[key])
    smoke.write(output / "run.json", info)
    model = optimizer = torch = None

    def deadline():
        curriculum.check_deadline(started, config["training"]["wall_limit_seconds"])

    try:
        import torch
        from peft import PeftModel, get_peft_model_state_dict

        _, model, tokenizer, device, runtime = load_runtime(
            config, root, bundle["views"][config["format"]]["tokens"]
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
                    "t1_adapter_loaded": False,
                    "other_format_adapter_loaded": False,
                    "format": config["format"],
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
