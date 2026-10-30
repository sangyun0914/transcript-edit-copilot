"""Fresh native Gemma seven-action LoRA, variable budgets, and independent reload."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import prepare_gemma_actions_v3 as preparation
import train_passive_gemma as gemma
import gemma_sft_checkpoint as recovery
from train_passive_gemma_actions import load_adapter

VERSION = "gemma-actions-v3-sft"
TRAINED = "training_complete_fresh_reload_pending"
RELOADED = "fresh_reload_passed_local_export_pending"
smoke, curriculum = gemma.smoke, gemma.curriculum
require, digest, read, write = smoke.require, smoke.digest, smoke.read, smoke.write


def validate_configuration(config):
    require(
        config["version"] == VERSION
        and config["candidate_id"] == "gemma26"
        and config["format"] in preparation.formats.FORMATS
        and config["format_version"] == preparation.formats.VERSION
        and config["contract_version"] == preparation.formats.contract.VERSION,
        "Unknown seven-action Gemma configuration.",
    )
    require(
        config["base"]["repository"] == gemma.REPOSITORY
        and config["base"]["revision"] == gemma.REVISION,
        "Use the pinned native Gemma revision.",
    )
    require(
        set(config["libraries"]) == set(smoke.LIBRARIES)
        and config["absent_libraries"] == ["torchao"]
        and all(
            isinstance(value, str) and value and not any(c in value for c in "*<>=,")
            for value in config["libraries"].values()
        ),
        "Pin every runtime library exactly and exclude incompatible torchao.",
    )
    expected_model = {
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
        all(config["model"].get(key) == value for key, value in expected_model.items())
        and config["model"]["minimum_gpu_bytes"] >= 75000000000
        and bool(config["model"]["gpu_name_contains"]),
        "Require exact Gemma text LoRA targets, native BF16 and FP32 adapter regime.",
    )
    training = config["training"]
    fixed = {
        "micro_batch_size": 1,
        "rank": 16,
        "alpha": 16,
        "dropout": 0,
        "bias": "none",
        "optimizer": "adamw_torch",
        "scheduler": "constant",
        "max_sequence_tokens": 4096,
        "max_completion_tokens": 512,
        "fresh_native_initialization": True,
        "truncation": False,
        "packing": False,
    }
    require(
        all(training.get(key) == value for key, value in fixed.items()),
        "Unsupported training regime.",
    )
    require(
        type(training["seed"]) is int
        and type(training["gradient_accumulation"]) is int
        and training["gradient_accumulation"] > 0
        and training["draw_count"] > 0
        and len(training["sample_order"]) == training["draw_count"]
        and training["max_updates"]
        == math.ceil(training["draw_count"] / training["gradient_accumulation"])
        and all(
            math.isfinite(training[key]) and training[key] > 0
            for key in ("learning_rate", "max_grad_norm", "wall_limit_seconds")
        ),
        "Invalid explicit draw/update/optimizer budget.",
    )
    require(
        config["generation"] == preparation.GENERATION,
        "Native generation terminal policy changed.",
    )
    require(
        config["phase"] in ("engineering", "substantive")
        and (config["phase"] == "engineering" or config["protocol_sha256"]),
        "Substantive training requires a bound protocol.",
    )


def preflight(config_path, *, assets=True):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, read(config_path)
    validate_configuration(config)
    bundle = preparation.verify_bundle(root)
    require(
        config == bundle["configs"][config["format"]],
        "Use the immutable prepared configuration.",
    )
    for name, expected in config["sources_sha256"].items():
        smoke.check_file(Path(__file__).with_name(name), expected)
    smoke.check_file(root / "native-pin.json", config["native_pin_sha256"])
    if config["protocol_sha256"]:
        smoke.check_file(root / "protocol.json", config["protocol_sha256"])
        protocol = read(root / "protocol.json")
        require(
            protocol.get("final_test_bodies_read") is False,
            "Training protocol must forbid final-test access.",
        )
        if config["phase"] == "substantive":
            binding = read(root / "source-binding.json")
            require(
                protocol["dataset"]["train_cases_sha256"]
                == binding["train_cases_sha256"]
                and all(
                    config["training"][key] == value
                    for key, value in protocol["training_plan"].items()
                )
                and protocol.get("eligibility_format", "common")
                == binding.get("eligibility_format", "common"),
                "Substantive training differs from the registered dataset or plan.",
            )
    if assets:
        gemma.validate_assets(config, root, read(root / "native-pin.json"))
    records = bundle["tokens"][config["format"]]
    indexed = {
        case["id"]: (case, record)
        for case, record in zip(bundle["cases"], records, strict=True)
    }
    rows = [
        preparation.flatten(*indexed[identity], index)
        for index, identity in enumerate(config["training"]["sample_order"])
    ]
    diagnostics = [
        preparation.flatten(*indexed[identity], index)
        for index, identity in enumerate(config["diagnostic_order"])
    ]
    require(
        diagnostics
        and len({row["case_id"] for row in diagnostics}) == len(diagnostics),
        "Unique train diagnostics required.",
    )
    return (
        config,
        root,
        bundle,
        rows,
        diagnostics,
        {
            "version": VERSION,
            "configuration_sha256": digest(config_path),
            "preparation_sha256": digest(root / "manifest.json"),
            "environment": smoke.environment(config),
            "status": "configuration_and_inputs_verified",
            "native_assets_verified": assets,
            "train_rows": len(bundle["cases"]),
            "draws": len(rows),
            "final_test_bodies_read": False,
        },
    )


def manifest(directory, status, version=VERSION):
    write(
        directory / "manifest.json",
        {
            "version": version,
            "status": status,
            "files_sha256": {
                str(path.relative_to(directory)): digest(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file()
                and path.name != "manifest.json"
                and "__pycache__" not in path.parts
            },
        },
    )


def verified_run(directory, status=TRAINED, *, version=VERSION):
    directory = Path(directory).resolve()
    receipt = read(directory / "manifest.json")
    require(
        receipt["version"] == version and receipt["status"] == status,
        "Run has not reached the required state.",
    )
    for name, expected in receipt["files_sha256"].items():
        smoke.check_file(smoke.child(directory, name), expected)
    info = read(directory / "run.json")
    require(
        info["status"] == status
        and info["configuration_sha256"]
        == receipt["files_sha256"]["configuration.json"],
        "Saved run/configuration binding differs.",
    )
    return receipt, info


def verify_adapter_serialization(directory, expected, torch):
    from safetensors.torch import load_file

    actual = smoke.tensor_records(
        load_file(str(Path(directory) / "adapter_model.safetensors"), device="cpu"),
        torch,
    )
    require(
        actual == expected,
        "Saved adapter bytes differ from the in-memory tensor records.",
    )
    return {
        "status": "passed",
        "tensor_count": len(actual),
        "adapter_sha256": digest(Path(directory) / "adapter_model.safetensors"),
    }


def save_adapter_checkpoint(output, model, torch, config_sha, exposure, reason):
    from peft import get_peft_model_state_dict

    require(
        not exposure.optimizer_commit_uncertain,
        "Cannot checkpoint an uncertain optimizer commit.",
    )
    directory = output / f"checkpoint-{exposure.updates:06d}"
    require(not directory.exists(), "Checkpoint output already exists.")
    directory.mkdir()
    state = smoke.tensor_records(
        get_peft_model_state_dict(model, save_embedding_layers=False), torch
    )
    model.save_pretrained(
        directory / "adapter", safe_serialization=True, save_embedding_layers=False
    )
    write(directory / "adapter-tensors.json", state)
    write(
        directory / "serialization.json",
        verify_adapter_serialization(directory / "adapter", state, torch),
    )
    write(
        directory / "checkpoint.json",
        {
            "status": "adapter_checkpoint_saved",
            "reason": reason,
            "configuration_sha256": config_sha,
            "exposure": exposure.summary(),
            "optimizer_state_saved": False,
            "resume_supported": False,
            "quality_evaluated": False,
        },
    )
    manifest(directory, "adapter_checkpoint_saved")
    return str(directory.relative_to(output))


def verify_reload(previous, reloaded, *, version=VERSION):
    receipt, info = verified_run(previous, version=version)
    _, reload_info = verified_run(reloaded, RELOADED, version=version)
    parity = read(Path(reloaded) / "reload-parity.json")
    require(
        reload_info["previous_manifest_sha256"]
        == digest(Path(previous) / "manifest.json")
        and reload_info["configuration_sha256"] == info["configuration_sha256"]
        and reload_info["pid"] != info["pid"]
        and parity["adapter_tensors_identical"] is True
        and parity["cases"]
        and all(row["identical"] for row in parity["cases"]),
        "Require an independent process with identical adapter and generation results.",
    )
    return receipt, info


def new_output(output, root, previous=None):
    output = Path(output).resolve()
    require(
        not output.exists()
        and not output.is_relative_to(root)
        and (previous is None or not output.is_relative_to(Path(previous).resolve())),
        "Use a new output directory outside immutable inputs and previous run.",
    )
    output.mkdir(parents=True)
    return output


def run(config_path, output, previous=None, *, resume_checkpoint=None, checkpoint_dir=None):
    config, root, bundle, rows, diagnostics, verified = preflight(config_path)
    require(verified["environment"]["matched"], "Pinned GPU training runtime differs.")
    require(not previous or (resume_checkpoint is None and checkpoint_dir is None),
            "Independent reload cannot resume training or write optimizer checkpoints.")
    previous = Path(previous).resolve() if previous else None
    if previous:
        receipt, prior = verified_run(previous)
        require(
            prior["configuration_sha256"] == verified["configuration_sha256"]
            and prior["pid"] != os.getpid(),
            "Reload must run in a different process against the exact same configuration.",
        )
        exposure = read(previous / "exposure.json")
        require(
            prior["training_updates"] == config["training"]["max_updates"]
            and exposure["phases"]["optimizer_committed"]["rows"] == len(rows),
            "Previous run exposure is incomplete.",
        )
    output = new_output(output, root, previous)
    started, exposure = time.monotonic(), curriculum.Exposure()
    prior_elapsed = 0
    info = {
        **verified,
        "mode": "fresh_reload" if previous else "train",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "status": "loading",
        "training_updates": 0,
        "optimizer_state_resumed": False,
        "initialization": "saved_adapter_on_native_base"
        if previous
        else "fresh_native_lora",
        "resume_supported": not bool(previous),
        "resume_history": [],
        "resumed_elapsed_seconds": 0,
        "recovery_checkpoint_format": recovery.VERSION,
        "wall_limit_seconds": config["training"]["wall_limit_seconds"],
        "wall_limit_origin": "Cumulative retained checkpoint time plus this attempt from before model loading; lost uncheckpointed work is not known.",
        "wall_limit_enforcement": "Before forwards and optimizer steps; completed-checkpoint serialization and diagnostics do not add updates.",
    }
    gemma.snapshot_configuration(
        config_path, output / "configuration.json", verified["configuration_sha256"]
    )
    write(output / "run.json", info)
    torch = None

    def elapsed():
        return prior_elapsed + time.monotonic() - started

    def deadline():
        curriculum.check_deadline(started - prior_elapsed, config["training"]["wall_limit_seconds"])

    try:
        if not previous:
            checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir else output.with_name(output.name + "-checkpoints")
            require(not checkpoint_dir.exists() and checkpoint_dir != output
                    and not checkpoint_dir.is_relative_to(output) and not checkpoint_dir.is_relative_to(root),
                    "Use a fresh checkpoint directory outside the run and immutable preparation.")
            require(type(config["training"]["checkpoint_every_updates"]) is int
                    and config["training"]["checkpoint_every_updates"] > 0, "Require a positive frozen checkpoint interval.")
            info["checkpoint_directory"] = str(checkpoint_dir)
        torch, model, tokenizer, device, runtime = gemma.load_runtime(
            config, root, bundle["tokens"][config["format"]]
        )
        from peft import get_peft_model_state_dict

        deadline()
        if previous:
            expected = read(previous / "adapter-tensors.json")
            serialization = verify_adapter_serialization(
                previous / "adapter", expected, torch
            )
            model, trainable, initial = load_adapter(
                model, previous / "adapter", expected, config, torch, False
            )
        else:
            model, trainable = gemma.enable_training(model, config, torch)
            initial = smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            serialization = None
        runtime.update(
            after_adapter_parameter_census=gemma.parameter_census(model),
            initial_adapter_tensors=initial,
        )
        write(output / "runtime.json", runtime)
        if not previous:
            initialization = {
                "initialization": "fresh_native_lora",
                "trainable_tensors": trainable,
                "trainable_parameters": sum(trainable.values()),
                "initial_adapter_tensors": initial,
            }
            write(output / "initialization.json", initialization)
            parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
            optimizer = torch.optim.AdamW(
                list(parameters.values()),
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )
            require(
                not optimizer.state, "Fresh optimizer must have no resumed moments."
            )
            history = {"rows": [], "updates": []}
            start_offset = 0
            if resume_checkpoint is not None:
                prior_checkpoint, exposure, history = recovery.restore(
                    resume_checkpoint, config, rows, info, runtime, parameters, initialization, optimizer, torch)
                prior_elapsed = prior_checkpoint["elapsed_seconds"]
                start_offset = prior_checkpoint["draws_completed"]
                info.update(optimizer_state_resumed=True, initialization="checkpoint_adapter_optimizer_rng_on_native_base",
                            training_updates=exposure.updates, resumed_elapsed_seconds=prior_elapsed,
                            resume_history=[*prior_checkpoint["resume_history"], {
                                "checkpoint": str(Path(resume_checkpoint).resolve()),
                                "checkpoint_sha256": digest(Path(resume_checkpoint) / "checkpoint.json"),
                                "source_run": prior_checkpoint["source_run"], "source_pid": prior_checkpoint["source_pid"],
                                "updates_completed": exposure.updates, "draws_completed": start_offset,
                                "elapsed_seconds": prior_elapsed}])
                for name, events in (("training-rows.jsonl", history["rows"]), ("training-events.jsonl", history["updates"])):
                    for event in events:
                        curriculum.append(output / name, event)

            def checkpoint():
                info["latest_checkpoint"] = recovery.save(
                    checkpoint_dir, output, config, rows, info, runtime, parameters,
                    initialization, optimizer, exposure, history, elapsed(), torch)
                write(output / "run.json", info)
                print(json.dumps({"checkpoint_saved": info["latest_checkpoint"]}), flush=True)

            write(output / "exposure.json", exposure.summary())
            checkpoint()

            def record_row(event):
                event["elapsed_seconds"] = elapsed()
                curriculum.append(output / "training-rows.jsonl", event)
                history["rows"].append(event)

            def record_update(event):
                event["elapsed_seconds"] = elapsed()
                require(
                    event["gradient_norm_before_clip"] > 0,
                    "Training update has zero gradient.",
                )
                curriculum.append(output / "training-events.jsonl", event)
                history["updates"].append(event)
                write(output / "exposure.json", exposure.summary())
                info.update(status="training", training_updates=exposure.updates)
                write(output / "run.json", info)
                if (
                    exposure.updates % config["training"]["checkpoint_every_updates"]
                    == 0
                    or exposure.phases["optimizer_committed"]["rows"] == len(rows)
                ):
                    checkpoint()
                print(json.dumps({"event": "optimizer_update", **event}), flush=True)

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
                start_offset=start_offset,
            )
            recovery.validate_history(config, rows, exposure.updates, len(rows), history, elapsed(), exposure.summary())
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
            require(changed, "No adapter tensor changed.")
            serialization = verify_adapter_serialization(
                output / "adapter", final, torch
            )
            tokenizer.save_pretrained(output / "tokenizer")
            info.update(
                status=TRAINED,
                training_updates=exposure.updates,
                changed_adapter_tensors=changed,
                optimizer_committed_rows=exposure.phases["optimizer_committed"]["rows"],
            )
        results = gemma.generate_smoke(
            model,
            tokenizer,
            diagnostics,
            config,
            torch,
            device,
            lambda event: curriculum.append(output / "generation-events.jsonl", event),
        )
        write(output / "smoke-generation.json", results)
        write(output / "adapter-serialization.json", serialization)
        if previous:
            expected_results = read(previous / "smoke-generation.json")
            parity = {
                "adapter_tensors_identical": True,
                "optimizer_state_resumed": False,
                "cases": [
                    {"case_id": a["case_id"], "identical": a == b}
                    for a, b in zip(results, expected_results, strict=True)
                ],
            }
            write(output / "reload-parity.json", parity)
            require(
                all(row["identical"] for row in parity["cases"]),
                "Greedy outputs changed after independent reload.",
            )
            info.update(
                status=RELOADED,
                previous_manifest_sha256=digest(previous / "manifest.json"),
                reloaded_training_updates=config["training"]["max_updates"],
            )
        write(output / "exposure.json", exposure.summary())
        info.update(
            seconds=elapsed(),
            execution_seconds_this_attempt=time.monotonic() - started,
            finished_unix=time.time(),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        )
        write(output / "run.json", info)
        manifest(output, info["status"])
        return info
    except BaseException as error:
        info.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            seconds=elapsed(),
            execution_seconds_this_attempt=time.monotonic() - started,
            training_updates=exposure.updates,
        )
        if torch is not None:
            info.update(
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
            )
        write(output / "exposure.json", exposure.summary())
        write(output / "run.json", info)
        manifest(output, "failed")
        raise


def export_configuration(training, reloaded, template_path, output):
    """Bind a new adapter to the unchanged native Gemma exporter and converter pin."""
    import export_passive_gemma as exporter

    training, reloaded = Path(training).resolve(), Path(reloaded).resolve()
    version = read(training / "manifest.json")["version"]
    require(
        version in (VERSION, "gemma-actions-v3-dpo"),
        "Only verified v3 SFT/DPO runs can be exported.",
    )
    receipt, _ = verify_reload(training, reloaded, version=version)
    previous = exporter.configuration(template_path)
    config = read(training / "configuration.json")
    for section in ("base",):
        require(
            all(
                previous[section][key] == config[section][key]
                for key in ("repository", "revision", "files_sha256")
            ),
            "Export template native base differs from training.",
        )
    result = copy.deepcopy(previous)
    result["adapter"] = {
        "path": str(training / "adapter"),
        "base_model_name_or_path": gemma.REPOSITORY,
        "files_sha256": {
            name: receipt["files_sha256"]["adapter/" + name]
            for name in ("adapter_config.json", "adapter_model.safetensors")
        },
    }
    result["training_record"] = {
        "path": str(training / "manifest.json"),
        "sha256": digest(training / "manifest.json"),
    }
    preparation.write(Path(output), result)
    return {
        "status": "export_configuration_written_not_exported",
        "path": str(Path(output).resolve()),
        "configuration_sha256": digest(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "reload"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "preflight":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--metadata-only", action="store_true")
        if name == "reload":
            command.add_argument("--run", type=Path, required=True)
        if name == "run":
            command.add_argument("--resume-checkpoint", type=Path)
            command.add_argument("--checkpoint-dir", type=Path)
    export = commands.add_parser("export-config")
    for name in ("training", "reloaded", "template", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.config, assets=not args.metadata_only)[-1]
    elif args.command == "export-config":
        result = export_configuration(
            args.training, args.reloaded, args.template, args.output
        )
    else:
        result = run(args.config, args.output, getattr(args, "run", None),
                     resume_checkpoint=getattr(args, "resume_checkpoint", None),
                     checkpoint_dir=getattr(args, "checkpoint_dir", None))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
