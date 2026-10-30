"""One preregistered 809-draw passive JSON SFT run and separate fresh reload."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time

import prepare_passive_curriculum as preparation
import prepare_passive_sft as smoke_preparation
import train_passive_sft as smoke


VERSION = "passive-sft-curriculum-v1"
PHASES = ("forward", "backward", "optimizer_committed")
CATEGORIES = ("case_id", "source_id", "source_group_id", "episode_id", "language", "action")


class WallLimitExceeded(RuntimeError):
    pass


def check_deadline(started, limit, clock=time.monotonic):
    if clock() - started >= limit:
        raise WallLimitExceeded(
            "The preregistered wall limit elapsed; no further update is allowed."
        )


def bound(binding, root):
    path = smoke.child(root, binding["path"])
    smoke.check_file(path, binding["sha256"])
    return path, smoke.read(path)


def validate_configuration(config, original):
    smoke.require(config["version"] == VERSION, "Unknown curriculum configuration.")
    smoke.validate_configuration(original)
    for name in ("candidate_id", "model", "libraries", "generation"):
        smoke.require(
            config[name] == original[name], f"Verified smoke identity/settings changed: {name}"
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
        "max_updates": 203,
        "draw_count": 809,
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
        "The preregistered curriculum recipe changed.",
    )
    smoke.require(
        type(training["seed"]) is int and type(training["sampling_seed"]) is str,
        "Keep model seed integer and sampler seed string.",
    )
    smoke.require(
        len(training["sample_order"]) == 809, "Exactly 809 fixed replacement draws are required."
    )


def validate_assets(config, root):
    for section in ("base", "tokenizer"):
        directory = smoke.child(root, config[section]["path"])
        for name, expected in config[section]["files_sha256"].items():
            smoke.check_file(smoke.child(directory, name), expected)
    native = smoke.child(root, config["base"]["path"])
    for name in config["base"]["absent_files"]:
        smoke.require(
            not smoke.child(native, name).exists(), f"Pinned absent native file appeared: {name}"
        )
    index = smoke.read(native / "model.safetensors.index.json")["weight_map"]
    smoke.require(
        set(index.values())
        == {name for name in config["base"]["files_sha256"] if name.endswith(".safetensors")}
        and set(config["model"]["ignored_native_tensor_names"])
        == {name for name in index if name.startswith("mtp.")},
        "Native shards or preregistered MTP exclusions changed.",
    )


def validate_protocol(protocol, config, release_sha256):
    plan = protocol["training_plan"]
    expected = {
        "candidate": config["candidate_id"],
        "view": "json-curriculum",
        "draw_count": 809,
        "sampling_seed": "3407",
        "model_seed": 3407,
        "effective_batch": 4,
        "updates": 203,
        "final_accumulation": 1,
        "development_during_training": False,
    }
    smoke.require(
        protocol["version"] == "passive-curriculum-evaluation-v1"
        and all(plan.get(key) == value for key, value in expected.items())
        and protocol["conditions"] == ["native-base-full", "native-base-icl", "sft-full", "sft-icl"]
        and protocol["b2"]["release_manifest"]["sha256"] == release_sha256
        and protocol["final_test_bodies_read"] is False
        and protocol["learning_signal_policy"]["no_auto_extra_training"] is True
        and protocol["learning_signal_policy"]["deployment_promotion"] is False,
        "Evaluation protocol differs from the preregistered fixed train-only experiment.",
    )


def selected_rows(bundle):
    rows = []
    for position, draw in enumerate(bundle["draw_plan"]):
        index = draw["train_row_index"]
        case, encoded = bundle["cases"][index], bundle["tokens"][index]
        smoke.require(
            draw["draw_index"] == position and case["id"] == encoded["id"] == draw["case_id"],
            "Draw row binding changed.",
        )
        metadata = case["metadata"]
        rows.append(
            {
                "case_id": case["id"],
                "draw_index": position,
                **{
                    key: metadata[key]
                    for key in ("source_id", "source_group_id", "episode_id", "language")
                },
                "action": case["expectedOutput"]["action"],
                **encoded["primary"],
            }
        )
    return rows


def preflight(config_path):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, smoke.read(config_path)
    for path, expected in (
        (Path(__file__), config["trainer_sha256"]),
        (Path(smoke.__file__), config["smoke_trainer_sha256"]),
        (Path(preparation.__file__), config["preparer_sha256"]),
        (Path(smoke_preparation.__file__), config["smoke_preparer_sha256"]),
    ):
        smoke.check_file(path, expected)
    _, original = bound(config["bindings"]["smoke_configuration"], root)
    smoke.require(
        original["trainer_sha256"] == config["smoke_trainer_sha256"],
        "Original smoke used a different helper implementation.",
    )
    validate_configuration(config, original)
    manifest_path, manifest = bound(config["bindings"]["preparation"], root)
    bundle = preparation.verify_bundle(manifest_path.parent)
    smoke.require(
        bundle["manifest"] == manifest, "Preparation manifest changed during verification."
    )
    smoke.require(
        config["bindings"]["smoke_configuration"]["sha256"]
        == manifest["bindings"]["smoke_configuration"]["sha256"],
        "Original smoke configuration differs from the completed preparation evidence.",
    )
    protocol_path, protocol = bound(config["bindings"]["evaluation_protocol"], root)
    validate_protocol(protocol, config, manifest["release_manifest_sha256"])
    prepared = bundle["configuration"]
    smoke.require(
        prepared["candidate_id"] == config["candidate_id"]
        and prepared["training_view"] == "json-curriculum"
        and prepared["sampling"]
        == {
            "algorithm": "weighted_indices_v2",
            "seed": "3407",
            "seed_type": "str",
            "replacement": True,
            "draw_count": 809,
            "weights_applied_to_loss": False,
        },
        "Use the preregistered train-only curriculum sampler.",
    )
    for key, prepared_key in (
        ("draw_count", "draw_count"),
        ("max_updates", "optimizer_updates"),
        ("max_total_sequence_tokens", "max_total_sequence_tokens"),
        ("max_total_completion_tokens", "max_total_completion_tokens"),
        ("sample_order", "sample_order"),
    ):
        value = (
            prepared["sampling"][prepared_key]
            if key == "draw_count"
            else prepared["training"][prepared_key]
        )
        smoke.require(config["training"][key] == value, f"Prepared exposure budget differs: {key}")
    smoke.require(
        prepared["training"]["final_update_rows"] == 1, "Final accumulation must contain one row."
    )
    for key in ("repository", "revision"):
        smoke.require(
            prepared["native_base"][key] == config["base"][key], "Prepared native identity changed."
        )
    smoke.require(
        prepared["tokenizer"]["files_sha256"] == config["tokenizer"]["files_sha256"],
        "Prepared tokenizer binding changed.",
    )
    validate_assets(config, root)
    rows = selected_rows(bundle)
    training = config["training"]
    smoke.require(
        len(rows) == 809
        and [row["case_id"] for row in rows] == training["sample_order"]
        and sum(len(row["input_ids"]) for row in rows) == training["max_total_sequence_tokens"]
        and sum(len(row["input_ids"]) - row["completion_start"] for row in rows)
        == training["max_total_completion_tokens"],
        "Actual fixed draw stream exceeds or differs from its exact budget.",
    )
    diagnostic_config = {
        "generation": config["generation"],
        "training": {
            **original["training"],
            "sample_order": bundle["smoke_configuration"]["proposed_smoke"]["order"],
        },
    }
    diagnostics = smoke.validate_smoke(
        diagnostic_config,
        bundle["smoke_configuration"],
        bundle["smoke_cases"],
        bundle["smoke_messages"],
        bundle["smoke_tokens"],
        bundle["smoke_inputs"],
    )
    summary = {
        "version": VERSION,
        "status": "configuration_and_inputs_verified",
        "configuration_sha256": smoke.digest(config_path),
        "trainer_sha256": config["trainer_sha256"],
        "smoke_trainer_sha256": config["smoke_trainer_sha256"],
        "preparation_sha256": smoke.digest(manifest_path),
        "release_manifest_sha256": manifest["release_manifest_sha256"],
        "evaluation_protocol_sha256": smoke.digest(protocol_path),
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


class Exposure:
    def __init__(self):
        self.updates = 0
        self.optimizer_commit_uncertain = False
        self.phases = {
            phase: {
                "rows": 0,
                "sequence_tokens": 0,
                "completion_tokens": 0,
                "counts": {key: Counter() for key in CATEGORIES},
            }
            for phase in PHASES
        }

    def record(self, phase, row):
        values = self.phases[phase]
        values["rows"] += 1
        values["sequence_tokens"] += len(row["input_ids"])
        values["completion_tokens"] += len(row["input_ids"]) - row["completion_start"]
        for key in CATEGORIES:
            values["counts"][key][row[key]] += 1

    def summary(self):
        return {
            "completed_updates": self.updates,
            "optimizer_commit_uncertain": self.optimizer_commit_uncertain,
            "pending_backward_rows_not_committed": self.phases["backward"]["rows"]
            - self.phases["optimizer_committed"]["rows"],
            "phases": {
                phase: {
                    **{
                        key: values[key] for key in ("rows", "sequence_tokens", "completion_tokens")
                    },
                    "unique_cases": len(values["counts"]["case_id"]),
                    "counts": {
                        key: dict(sorted(counts.items()))
                        for key, counts in values["counts"].items()
                    },
                }
                for phase, values in self.phases.items()
            },
        }


def train_draws(
    model,
    rows,
    optimizer,
    torch,
    training,
    device,
    exposure,
    on_row,
    on_update,
    deadline=lambda: None,
    *,
    start_offset=0,
):
    accumulation = training["gradient_accumulation"]
    smoke.require(
        len(rows) == training["draw_count"]
        and math.ceil(len(rows) / accumulation) == training["max_updates"],
        "Draw/update budget mismatch.",
    )
    smoke.require(
        type(start_offset) is int
        and 0 <= start_offset <= len(rows)
        and (start_offset % accumulation == 0 or start_offset == len(rows))
        and exposure.updates == math.ceil(start_offset / accumulation)
        and not exposure.optimizer_commit_uncertain
        and all(exposure.phases[phase]["rows"] == start_offset for phase in PHASES),
        "Resume requires an exact completed optimizer boundary and matching exposure.",
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    for offset in range(start_offset, len(rows), accumulation):
        group = rows[offset : offset + accumulation]
        losses = []
        for row in group:
            deadline()
            event = {key: row[key] for key in ("draw_index", *CATEGORIES)}
            event.update(
                status="failed",
                forward_complete=False,
                backward_complete=False,
                sequence_tokens=len(row["input_ids"]),
                completion_tokens=len(row["input_ids"]) - row["completion_start"],
                accumulation_divisor=len(group),
            )
            try:
                loss = smoke.completion_loss(model, row, torch, device)
                exposure.record("forward", row)
                event["forward_complete"] = True
                smoke.require(
                    torch.isfinite(loss).item(), f"Non-finite loss at draw {row['draw_index']}."
                )
                event["loss"] = float(loss.detach())
                (loss / len(group)).backward()
                del loss
                exposure.record("backward", row)
                event["backward_complete"] = True
                smoke.require(
                    all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters),
                    "Non-finite gradients; current accumulation is not committed.",
                )
                losses.append(event["loss"])
                event["status"] = "forward_backward_complete"
            except BaseException as error:
                event.update(error_type=type(error).__name__, error=str(error))
                raise
            finally:
                on_row(event)
            deadline()
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, training["max_grad_norm"], error_if_nonfinite=True
            )
        )
        smoke.require(math.isfinite(norm) and norm >= 0, "Non-finite gradient norm.")
        deadline()
        exposure.optimizer_commit_uncertain = True
        optimizer.step()
        smoke.require(
            all(torch.isfinite(p).all().item() for p in parameters),
            "Optimizer produced non-finite adapter tensors.",
        )
        optimizer.zero_grad(set_to_none=True)
        for row in group:
            exposure.record("optimizer_committed", row)
        exposure.updates += 1
        exposure.optimizer_commit_uncertain = False
        on_update(
            {
                "update": exposure.updates,
                "committed_rows": offset + len(group),
                "case_ids": [row["case_id"] for row in group],
                "microbatches": len(group),
                "row_losses": losses,
                "mean_row_loss": sum(losses) / len(group),
                "gradient_norm_before_clip": norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )


def optimizer_is_finite(optimizer, torch):
    return all(
        not torch.is_tensor(value) or torch.isfinite(value).all().item()
        for state in optimizer.state.values()
        for value in state.values()
    )


def save_training_state(path, model, optimizer, torch, config_sha, exposure):
    from peft import get_peft_model_state_dict

    records = smoke.tensor_records(
        get_peft_model_state_dict(model, save_embedding_layers=False), torch
    )
    smoke.require(
        optimizer_is_finite(optimizer, torch), "Recovery optimizer contains non-finite tensors."
    )
    model.save_pretrained(path / "adapter", safe_serialization=True, save_embedding_layers=False)
    smoke.write(path / "adapter-tensors.json", records)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng_state": random.getstate(),
            "configuration_sha256": config_sha,
            "exposure": exposure.summary(),
            "next_committed_draw_index": exposure.phases["optimizer_committed"]["rows"],
            "resume_supported": False,
            "partial_gradients_saved": False,
        },
        path / "training-state.pt",
    )
    return records


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


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


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
        "scope": "Fixed train-only SFT run; no dev-driven selection, deployment promotion or final-test evaluation.",
    }
    smoke.write(output / "configuration.json", config)
    for name, module in (
        ("trainer.py", __file__),
        ("train_passive_sft.py", smoke.__file__),
        ("prepare_passive_curriculum.py", preparation.__file__),
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
            config, root, bundle["tokens"]
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
                "Reload only a completed curriculum run.",
            )
            for name, expected in record["files_sha256"].items():
                smoke.check_file(smoke.child(previous, name), expected)
            saved = smoke.read(previous / "run.json")
            smoke.require(
                saved["configuration_sha256"] == verified["configuration_sha256"]
                and saved["pid"] != os.getpid()
                and saved["training_updates"] == 203,
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
                reloaded_training_updates=203,
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
