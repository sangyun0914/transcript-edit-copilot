"""Independent matched JSON/Rewrite SFT with one bound shared draw stream."""

import argparse
import json
import os
from pathlib import Path
import time

import prepare_passive_matched as preparation
import prepare_passive_sft as smoke_preparation
import train_passive_curriculum as curriculum
import train_passive_sft as smoke


VERSION = "passive-sft-matched-v1"
FORMATS = ("json", "rewrite")
T1_BINDINGS = (
    "t1_training_manifest",
    "t1_run",
    "t1_reload_manifest",
    "t1_reload_run",
    "t1_reload_parity",
    "t1_export_receipt",
    "t1_local_preflight",
)
Exposure = curriculum.Exposure
check_deadline = curriculum.check_deadline
train_draws = curriculum.train_draws
save_training_state = curriculum.save_training_state
append = curriculum.append
bound = curriculum.bound


def validate_configuration(config, original):
    smoke.require(config["version"] == VERSION, "Unknown matched configuration.")
    smoke.require(config["format"] in FORMATS, "Choose one registered matched format.")
    smoke.validate_configuration(original)
    for name in ("candidate_id", "model", "libraries", "generation"):
        smoke.require(
            config[name] == original[name], f"Verified native identity/settings changed: {name}"
        )
    smoke.require(
        config.get("absent_libraries", []) == original.get("absent_libraries", []),
        "Runtime absence pins changed.",
    )
    for section, keys in (
        ("base", ("repository", "revision", "files_sha256", "absent_files")),
        ("tokenizer", ("files_sha256", "template_sha256")),
    ):
        smoke.require(
            all(config[section][key] == original[section][key] for key in keys),
            f"Native/tokenizer binding changed: {section}",
        )
    training = config["training"]
    fixed = {
        "seed": 3407,
        "sampling_seed": "3407",
        "micro_batch_size": 1,
        "gradient_accumulation": 4,
        "max_updates": 201,
        "draw_count": 802,
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
        "recovery_updates": [50, 100, 150],
    }
    smoke.require(
        all(training.get(key) == value for key, value in fixed.items()),
        "The preregistered matched recipe changed.",
    )
    smoke.require(
        type(training["seed"]) is int and type(training["sampling_seed"]) is str,
        "Keep model seed integer and sampler seed string.",
    )
    smoke.require(len(training["sample_order"]) == 802, "Exactly 802 shared draws are required.")


def validate_protocol(protocol, config, release_sha256):
    expected_plan = {
        "candidate": config["candidate_id"],
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
    expected_gate = {
        "t1_final_updates": 203,
        "t1_committed_draws": 809,
        "fresh_reload_adapter_and_outputs": "exact",
        "paired_export": "tensor_verified_paired_export_runtime_preflight_pending",
        "local_preflight_status": "template_preflight_passed_generation_unrun",
        "full_t1_development_evaluation_may_run_in_parallel": True,
    }
    smoke.require(
        protocol["version"] == "passive-matched-evaluation-v1"
        and protocol["formats"] == list(FORMATS)
        and all(protocol["training_plan"].get(key) == value for key, value in expected_plan.items())
        and all(protocol["launch_gate"].get(key) == value for key, value in expected_gate.items())
        and protocol["b2"]["release_manifest"]["sha256"] == release_sha256
        and protocol["conditions"]
        == [
            "matched-json-unconstrained",
            "matched-rewrite-unconstrained",
            "matched-json-constrained",
        ]
        and protocol["primary_format_comparison"]["left"] == "matched-json-unconstrained"
        and protocol["primary_format_comparison"]["right"] == "matched-rewrite-unconstrained"
        and protocol["registered_requests_per_condition"] == 286
        and protocol["training_rows"] == 802
        and protocol["same_case_sampling_order_required"] is True
        and protocol["final_test_bodies_read"] is False
        and protocol["learning_signal_policy"]["no_auto_extra_training"] is True
        and protocol["learning_signal_policy"]["deployment_promotion"] is False,
        "Evaluation protocol differs from the fixed matched-format experiment.",
    )


def validate_t1_evidence(config, protocol, evidence, hashes, release_sha256):
    manifest, run = evidence["t1_training_manifest"], evidence["t1_run"]
    reload_manifest, reload_run = evidence["t1_reload_manifest"], evidence["t1_reload_run"]
    parity = evidence["t1_reload_parity"]
    export, local = evidence["t1_export_receipt"], evidence["t1_local_preflight"]
    smoke.require(
        manifest["version"] == run["version"] == curriculum.VERSION
        and manifest["status"] == run["status"] == "training_complete_fresh_reload_pending"
        and manifest["files_sha256"]["run.json"] == hashes["t1_run"]
        and manifest["files_sha256"]["configuration.json"] == run["configuration_sha256"]
        and run["mode"] == "train"
        and run["training_updates"] == 203
        and run["optimizer_committed_rows"] == 809
        and run["trainer_sha256"] == config["curriculum_trainer_sha256"]
        and run["smoke_trainer_sha256"] == config["smoke_trainer_sha256"]
        and run["release_manifest_sha256"] == release_sha256
        and run["evaluation_protocol_sha256"] == protocol["t1_protocol"]["sha256"]
        and run["final_test_bodies_read"] == 0,
        "T2 requires this release's completed final 203-update T1 run.",
    )
    smoke.require(
        reload_manifest["version"] == reload_run["version"] == curriculum.VERSION
        and reload_manifest["status"]
        == reload_run["status"]
        == "fresh_reload_passed_local_export_pending"
        and reload_manifest["files_sha256"]["run.json"] == hashes["t1_reload_run"]
        and reload_manifest["files_sha256"]["reload-parity.json"] == hashes["t1_reload_parity"]
        and reload_run["previous_manifest_sha256"] == hashes["t1_training_manifest"]
        and reload_run["configuration_sha256"] == run["configuration_sha256"]
        and reload_run["evaluation_protocol_sha256"] == run["evaluation_protocol_sha256"]
        and reload_run["mode"] == "fresh_reload"
        and reload_run["reloaded_training_updates"] == 203
        and reload_run["pid"] != run["pid"]
        and reload_run["optimizer_state_resumed"] is False
        and parity["optimizer_state_resumed"] is False
        and parity["adapter_tensors_identical"] is True
        and len(parity["cases"]) == len({row["case_id"] for row in parity["cases"]}) == 8
        and all(row["identical"] is True for row in parity["cases"]),
        "T1 fresh reload must bind the same completed run with exact tensor/output parity.",
    )
    smoke.require(
        export["version"] == "passive-sft-export-v1"
        and export["status"] == protocol["launch_gate"]["paired_export"]
        and export["training_record"]["sha256"]
        in (hashes["t1_run"], hashes["t1_training_manifest"])
        and export["base_repository"] == config["base"]["repository"]
        and export["base_revision"] == config["base"]["revision"]
        and set(export["artifacts"])
        == {"base", "adapter", "merged", "base_quantized", "merged_quantized"},
        "T1 paired tensor-verified export does not bind the completed native training run.",
    )
    smoke.require(
        local["version"] == "passive-curriculum-evaluation-v1"
        and local["condition"] == "sft-full"
        and local["model_role"] == "sft"
        and local["status"] == protocol["launch_gate"]["local_preflight_status"]
        and local["error"] is None
        and local["export_receipt_sha256"] == hashes["t1_export_receipt"]
        and local["training_configuration_sha256"] == run["configuration_sha256"]
        and local["fresh_reload_manifest_sha256"] == hashes["t1_reload_manifest"]
        and local["protocol_sha256"] == run["evaluation_protocol_sha256"]
        and local["registered_requests"] == local["requested_cases"] == 289
        and "manifest.json" in local["files_sha256"],
        "T1 local template preflight must bind the same trained export; generation remains unrun.",
    )


def validate_prepared(config, bundle):
    prepared = bundle["configuration"]
    smoke.require(
        prepared["candidate_id"] == config["candidate_id"]
        and prepared["frozen_train_rows"] == 802
        and prepared["sampling"]
        == {
            "algorithm": "weighted_indices_v2",
            "seed": "3407",
            "seed_type": "str",
            "replacement": True,
            "draw_count": 802,
            "weights_applied_to_loss": False,
        },
        "Use the preregistered shared 802-draw matched sampler.",
    )
    training, view = config["training"], prepared["views"][config["format"]]
    smoke.require(
        prepared["training"]["micro_batch_size"] == 1
        and prepared["training"]["gradient_accumulation"] == 4
        and prepared["training"]["optimizer_updates"] == training["max_updates"] == 201
        and prepared["training"]["final_update_rows"] == 2
        and prepared["training"]["final_update_normalization"]
        == "mean_over_actual_microbatch_count"
        and prepared["training"]["sample_order"] == training["sample_order"]
        and prepared["training"]["initialization"]
        == "fresh_adapter_on_exact_native_base_not_smoke_or_T1_adapter"
        and view["training_view"] == config["format"]
        and all(
            training[key] == view[key]
            for key in ("max_total_sequence_tokens", "max_total_completion_tokens")
        ),
        "Matched-format exposure, fresh initialization or final normalization changed.",
    )
    smoke.require(
        all(
            prepared["native_base"][key] == config["base"][key]
            for key in ("repository", "revision")
        )
        and prepared["tokenizer"]["files_sha256"] == config["tokenizer"]["files_sha256"]
        and prepared["tokenizer"]["eos_token_id"] == config["generation"]["eos_token_id"],
        "Matched native/tokenizer identity changed.",
    )
    encoded = bundle["views"][config["format"]]["tokens"]
    rows = curriculum.selected_rows({**bundle, "tokens": encoded})
    smoke.require(
        len(encoded) == len(rows) == 802
        and [row["case_id"] for row in rows] == training["sample_order"]
        and sum(len(row["input_ids"]) for row in rows) == training["max_total_sequence_tokens"]
        and sum(len(row["input_ids"]) - row["completion_start"] for row in rows)
        == training["max_total_completion_tokens"],
        "Actual matched draw stream differs from the exact format-specific token budget.",
    )
    diagnostics = bundle["views"][config["format"]]["diagnostics"]["rows"]
    order = view["diagnostic_case_order"]
    smoke.require(
        len(diagnostics) == len(set(order)) == 8
        and [row["case_id"] for row in diagnostics] == order
        and all(prepared["views"][fmt]["diagnostic_case_order"] == order for fmt in FORMATS)
        and sum(len(row["input_ids"]) for row in diagnostics) == view["diagnostic_sequence_tokens"]
        and sum(len(row["input_ids"]) - row["completion_start"] for row in diagnostics)
        == view["diagnostic_completion_tokens"],
        "Both formats require the same eight registered matched training diagnostics.",
    )
    return rows, diagnostics


def preflight(config_path):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, smoke.read(config_path)
    modules = (
        (__file__, "trainer_sha256"),
        (smoke.__file__, "smoke_trainer_sha256"),
        (curriculum.__file__, "curriculum_trainer_sha256"),
        (preparation.__file__, "preparer_sha256"),
        (curriculum.preparation.__file__, "curriculum_preparer_sha256"),
        (smoke_preparation.__file__, "smoke_preparer_sha256"),
    )
    for path, key in modules:
        smoke.check_file(Path(path), config[key])
    _, original = bound(config["bindings"]["smoke_configuration"], root)
    smoke.require(
        original["trainer_sha256"] == config["smoke_trainer_sha256"],
        "Original smoke used a different helper implementation.",
    )
    validate_configuration(config, original)
    manifest_path, manifest = bound(config["bindings"]["preparation"], root)
    bundle = preparation.verify_bundle(manifest_path.parent)
    smoke.require(bundle["manifest"] == manifest, "Preparation changed during verification.")
    smoke.require(
        config["bindings"]["smoke_configuration"]["sha256"]
        == manifest["bindings"]["smoke_configuration"]["sha256"],
        "Original smoke configuration differs from the completed preparation evidence.",
    )
    protocol_path, protocol = bound(config["bindings"]["evaluation_protocol"], root)
    release_sha = manifest["release_manifest_sha256"]
    validate_protocol(protocol, config, release_sha)
    evidence, hashes = {}, {}
    for name in T1_BINDINGS:
        _, evidence[name] = bound(config["bindings"][name], root)
        hashes[name] = config["bindings"][name]["sha256"]
    validate_t1_evidence(config, protocol, evidence, hashes, release_sha)
    curriculum.validate_assets(config, root)
    rows, diagnostics = validate_prepared(config, bundle)
    training = config["training"]
    summary = {
        "version": VERSION,
        "format": config["format"],
        "status": "configuration_and_inputs_verified",
        "configuration_sha256": smoke.digest(config_path),
        "trainer_sha256": config["trainer_sha256"],
        "smoke_trainer_sha256": config["smoke_trainer_sha256"],
        "curriculum_trainer_sha256": config["curriculum_trainer_sha256"],
        "preparation_sha256": smoke.digest(manifest_path),
        "release_manifest_sha256": release_sha,
        "evaluation_protocol_sha256": smoke.digest(protocol_path),
        "t1_launch_evidence_sha256": hashes,
        "t1_local_gate_status": evidence["t1_local_preflight"]["status"],
        "t1_full_development_evaluation_required_for_launch": False,
        "environment": smoke.environment(config),
        "training_updates": 0,
        "draw_count": len(rows),
        "unique_planned_cases": len({row["case_id"] for row in rows}),
        "sequence_tokens": training["max_total_sequence_tokens"],
        "completion_tokens": training["max_total_completion_tokens"],
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
                str(file.relative_to(path)): smoke.digest(file)
                for file in sorted(path.rglob("*"))
                if file.is_file() and file != path / "manifest.json"
            },
        },
    )


def save_checkpoint(output, name, model, optimizer, torch, config_sha, exposure, reason):
    smoke.require(
        not exposure.optimizer_commit_uncertain,
        "An interrupted optimizer step cannot be marked as a safe recovery checkpoint.",
    )
    path = output / name
    path.mkdir(parents=True, exist_ok=False)
    save_training_state(path, model, optimizer, torch, config_sha, exposure)
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
            "scope": "Durable recovery evidence; no development evaluation or checkpoint selection.",
        },
    )
    manifest(path, "recovery_checkpoint_complete")
    return str(path.relative_to(output))


def run(config_path, output, previous=None):
    config, root, bundle, rows, diagnostics, verified = preflight(config_path)
    smoke.require(
        verified["environment"]["matched"], "Installed environment differs from the pinned runtime."
    )
    output = Path(output).resolve()
    smoke.require(
        not output.exists() and not output.is_relative_to(root),
        "Use a fresh output outside immutable input directories.",
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    exposure = Exposure()
    info = {
        **verified,
        "status": "loading",
        "mode": "fresh_reload" if previous else "train",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "optimizer_state_resumed": False,
        "initialization": "original_native_base_with_fresh_lora",
        "resume_supported": False,
        "wall_limit_seconds": config["training"]["wall_limit_seconds"],
        "wall_limit_origin": "Immediately before runtime/model loading; excludes input hash preflight.",
        "wall_limit_enforcement": "Checked between complete operations and before optimizer steps; an in-flight operation/checkpoint may finish after the deadline.",
        "recovery_checkpoints": [],
        "scope": "Independent matched-format train-only SFT; no dev-driven selection, deployment promotion or final-test evaluation.",
    }
    (output / "configuration.json").write_bytes(Path(config_path).read_bytes())
    for name, module in (
        ("trainer.py", __file__),
        ("train_passive_sft.py", smoke.__file__),
        ("prepare_passive_matched.py", preparation.__file__),
        ("train_passive_curriculum.py", curriculum.__file__),
        ("prepare_passive_curriculum.py", curriculum.preparation.__file__),
        ("prepare_passive_sft.py", smoke_preparation.__file__),
    ):
        (output / name).write_bytes(Path(module).read_bytes())
    smoke.write(output / "run.json", info)
    model = optimizer = torch = None

    def deadline():
        check_deadline(started, config["training"]["wall_limit_seconds"])

    try:
        from peft import PeftModel, get_peft_model_state_dict

        torch, model, tokenizer, device, runtime = smoke.load_runtime(
            config, root, bundle["views"][config["format"]]["tokens"]
        )
        smoke.write(output / "runtime.json", runtime)
        deadline()

        def generated(result):
            append(output / "smoke-generation-events.jsonl", result)
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
            previous = Path(previous).resolve()
            record = smoke.read(previous / "manifest.json")
            smoke.require(
                record["version"] == VERSION
                and record["status"] == "training_complete_fresh_reload_pending",
                "Reload only a completed matched run.",
            )
            for name, expected in record["files_sha256"].items():
                smoke.check_file(smoke.child(previous, name), expected)
            saved = smoke.read(previous / "run.json")
            smoke.require(
                saved["configuration_sha256"] == verified["configuration_sha256"]
                and saved["pid"] != os.getpid()
                and saved["training_updates"] == 201,
                "Reload requires the same completed configuration and a separate process.",
            )
            model = smoke.prepare_reload_base(model, config)
            model = PeftModel.from_pretrained(
                model, previous / "adapter", is_trainable=False, local_files_only=True
            )
            actual = smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            smoke.require(
                actual == smoke.read(previous / "adapter-tensors.json"),
                "Reloaded adapter tensors differ.",
            )
            deadline()
            results = smoke.generate_smoke(
                model, tokenizer, diagnostics, config, torch, device, generated
            )
            smoke.write(output / "smoke-generation.json", results)
            expected = smoke.read(previous / "smoke-generation.json")
            parity = {
                "adapter_tensors_identical": True,
                "cases": [
                    {"case_id": a["case_id"], "identical": a == b}
                    for a, b in zip(results, expected, strict=True)
                ],
                "optimizer_state_resumed": False,
            }
            smoke.write(output / "reload-parity.json", parity)
            smoke.require(
                all(value["identical"] for value in parity["cases"]),
                "Greedy output changed after fresh reload.",
            )
            info.update(
                status="fresh_reload_passed_local_export_pending",
                previous_manifest_sha256=smoke.digest(previous / "manifest.json"),
                reloaded_training_updates=201,
            )
        else:
            model, trainable = smoke.enable_training(model, config)
            initial = smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            smoke.write(
                output / "initialization.json",
                {
                    "initialization": "original_native_base_with_fresh_lora",
                    "trainable_tensors": trainable,
                    "trainable_parameters": sum(trainable.values()),
                    "initial_adapter_tensors": initial,
                },
            )
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )

            def record_row(event):
                append(output / "training-rows.jsonl", event)
                print(json.dumps({"event": "microbatch", **event}), flush=True)

            def record_update(event):
                info["training_updates"] = exposure.updates
                append(output / "training-events.jsonl", event)
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
                        json.dumps({"event": "recovery_checkpoint_complete", "path": path}),
                        flush=True,
                    )
                print(json.dumps({"event": "optimizer_update", **event}), flush=True)

            info["status"] = "training"
            smoke.write(output / "run.json", info)
            train_draws(
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
            final = save_training_state(
                output, model, optimizer, torch, verified["configuration_sha256"], exposure
            )
            smoke.require(set(initial) == set(final), "Adapter tensor census changed.")
            changed = [name for name in final if initial[name] != final[name]]
            smoke.require(changed, "No adapter tensors changed.")
            adapter_config = smoke.read(output / "adapter/adapter_config.json")
            smoke.require(
                adapter_config["base_model_name_or_path"] == config["base"]["repository"]
                and adapter_config["revision"] == config["base"]["revision"],
                "Saved adapter lost native provenance.",
            )
            tokenizer.save_pretrained(output / "tokenizer")
            deadline()
            results = smoke.generate_smoke(
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
                unique_cases=len(exposure.phases["optimizer_committed"]["counts"]["case_id"]),
            )
        info.update(
            seconds=time.monotonic() - started,
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
            training_updates=exposure.updates,
        )
        smoke.write(output / "exposure.json", exposure.summary())
        if not previous and model is not None and optimizer is not None and torch is not None:
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
                info["recovery_policy"] = (
                    "Use only an earlier complete recovery checkpoint, or initialize a new attempt "
                    "from the original native base; this trainer does not automatically resume."
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
