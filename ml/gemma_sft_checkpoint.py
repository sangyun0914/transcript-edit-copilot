"""Atomic optimizer-boundary recovery for the frozen Gemma draw stream."""

import math
import os
from pathlib import Path
import shutil
import tempfile

import train_passive_curriculum as curriculum
import train_small_actions_final as primitives

smoke = curriculum.smoke
require, read, write, sha = smoke.require, smoke.read, smoke.write, smoke.digest
VERSION = "gemma-actions-v3-sft-checkpoint-v1"
FILES = {"configuration.json", "runtime.json", "history.json", "state.pt"}


def runtime_identity(runtime):
    return {key: value for key, value in runtime.items()
            if key not in {"allocated_after_load_bytes", "reserved_after_load_bytes"}}


def validate_history(config, rows, updates, draws, history, elapsed, exposure):
    settings = config["training"]
    accumulation = settings["gradient_accumulation"]
    require(type(updates) is int and type(draws) is int
            and 0 <= updates <= settings["max_updates"]
            and draws == min(updates * accumulation, len(rows))
            and len(rows) == settings["draw_count"]
            and settings["max_updates"] == math.ceil(len(rows) / accumulation)
            and [row["case_id"] for row in rows] == settings["sample_order"]
            and [row["draw_index"] for row in rows] == list(range(len(rows))),
            "Checkpoint cursor differs from the frozen optimizer-step boundary.")
    require(set(history) == {"rows", "updates"} and len(history["rows"]) == draws
            and len(history["updates"]) == updates and math.isfinite(elapsed) and elapsed >= 0,
            "Checkpoint history is incomplete or elapsed time is invalid.")
    rebuilt = curriculum.Exposure()
    previous_row_time = previous_update_time = 0
    for index, (row, event) in enumerate(zip(rows[:draws], history["rows"], strict=True)):
        divisor = min(accumulation, len(rows) - index // accumulation * accumulation)
        require(all(event[key] == row[key] for key in ("draw_index", *curriculum.CATEGORIES))
                and event["status"] == "forward_backward_complete"
                and event["forward_complete"] is True and event["backward_complete"] is True
                and event["sequence_tokens"] == len(row["input_ids"])
                and event["completion_tokens"] == len(row["input_ids"]) - row["completion_start"]
                and event["accumulation_divisor"] == divisor
                and math.isfinite(event["loss"])
                and previous_row_time <= event["elapsed_seconds"] <= elapsed,
                "Checkpoint row history differs from the committed draw prefix.")
        previous_row_time = event["elapsed_seconds"]
        for phase in curriculum.PHASES:
            rebuilt.record(phase, row)
    for update, event in enumerate(history["updates"], 1):
        start, end = (update - 1) * accumulation, min(update * accumulation, len(rows))
        group = rows[start:end]
        losses = [row["loss"] for row in history["rows"][start:end]]
        require(event["update"] == update and event["committed_rows"] == end
                and event["case_ids"] == [row["case_id"] for row in group]
                and event["microbatches"] == len(group) and event["row_losses"] == losses
                and event["mean_row_loss"] == sum(losses) / len(losses)
                and math.isfinite(event["gradient_norm_before_clip"]) and event["gradient_norm_before_clip"] > 0
                and event["learning_rate"] == settings["learning_rate"]
                and previous_update_time <= event["elapsed_seconds"] <= elapsed
                and history["rows"][end - 1]["elapsed_seconds"] <= event["elapsed_seconds"],
                "Checkpoint update history differs from the frozen sample order or optimizer.")
        previous_update_time = event["elapsed_seconds"]
    rebuilt.updates = updates
    require(rebuilt.summary() == exposure, "Checkpoint exposure differs from its complete committed prefix.")
    return rebuilt


def identity(config, info, runtime):
    return {"configuration_sha256": info["configuration_sha256"],
            "preparation_sha256": info["preparation_sha256"],
            "sources_sha256": config["sources_sha256"],
            "native_pin_sha256": config["native_pin_sha256"],
            "runtime": runtime_identity(runtime)}


def save(parent, output, config, rows, info, runtime, trainable, initialization,
         optimizer, exposure, history, elapsed, torch):
    draws = exposure.phases["optimizer_committed"]["rows"]
    validate_history(config, rows, exposure.updates, draws, history, elapsed, exposure.summary())
    require(trainable and all(p.dtype == torch.float32 and p.grad is None for p in trainable.values()),
            "Checkpoint requires FP32 trainable tensors with no pending gradients.")
    require(initialization["initialization"] == "fresh_native_lora"
            and initialization["initial_adapter_tensors"] == runtime["initial_adapter_tensors"],
            "Checkpoint must preserve the original fresh adapter evidence.")
    parent, output = Path(parent), Path(output)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"checkpoint-{exposure.updates:06d}"
    require(not destination.exists(), "Checkpoint already exists; use a fresh checkpoint directory.")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
    try:
        for name in ("configuration.json", "runtime.json"):
            shutil.copyfile(output / name, temporary / name)
        write(temporary / "history.json", history)
        state = primitives.cpu_state({"trainable": trainable, "optimizer": optimizer.state_dict(),
                                      "rng": primitives.rng_state(torch)}, torch)
        primitives.validate_tensors(state["trainable"], trainable, torch, "trainable")
        primitives.validate_optimizer(state["optimizer"], optimizer, trainable, exposure.updates, torch)
        primitives.validate_rng_state(state["rng"], torch)
        torch.save(state, temporary / "state.pt")
        manifest = {"version": VERSION, "status": "optimizer_step_complete", **identity(config, info, runtime),
                    "source_run": str(output.resolve()), "source_pid": os.getpid(),
                    "draws": len(rows), "updates_completed": exposure.updates, "draws_completed": draws,
                    "elapsed_seconds": elapsed, "exposure": exposure.summary(),
                    "trainable_names": list(trainable), "initialization": initialization,
                    "resume_history": info["resume_history"], "final_test_bodies_read": False,
                    "optimizer_state_saved": True, "resume_supported": True,
                    "files_sha256": {name: sha(temporary / name) for name in sorted(FILES)}}
        require(manifest["files_sha256"]["configuration.json"] == info["configuration_sha256"],
                "Run configuration changed before checkpoint publication.")
        write(temporary / "checkpoint.json", manifest)
        for path in temporary.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        primitives.sync_directory(temporary)
        temporary.rename(destination)
        primitives.sync_directory(parent)
        primitives.sync_directory(parent.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"path": str(destination.resolve()), "sha256": sha(destination / "checkpoint.json"),
            "updates_completed": exposure.updates, "draws_completed": draws}


def restore(path, config, rows, info, runtime, trainable, initialization, optimizer, torch):
    path = Path(path).resolve()
    require(path.is_dir() and {p.name for p in path.iterdir()} == FILES | {"checkpoint.json"}
            and all(p.is_file() and not p.is_symlink() for p in path.iterdir()),
            "Incomplete checkpoint file inventory.")
    manifest = read(path / "checkpoint.json")
    require(manifest["version"] == VERSION and manifest["status"] == "optimizer_step_complete"
            and path.name == f"checkpoint-{manifest['updates_completed']:06d}"
            and set(manifest["files_sha256"]) == FILES and manifest["resume_supported"] is True
            and manifest["optimizer_state_saved"] is True and manifest["final_test_bodies_read"] is False,
            "Incomplete, unpublished or legacy adapter-only checkpoint.")
    for name, expected in manifest["files_sha256"].items():
        require(sha(path / name) == expected, f"Checkpoint file SHA256 differs: {name}")
    require(all(manifest[key] == value for key, value in identity(config, info, runtime).items()),
            "Checkpoint configuration, preparation, source or runtime identity differs.")
    require(read(path / "configuration.json") == config
            and manifest["files_sha256"]["configuration.json"] == info["configuration_sha256"]
            and runtime_identity(read(path / "runtime.json")) == runtime_identity(runtime),
            "Checkpoint configuration/runtime file binding differs.")
    require(manifest["draws"] == len(rows) and manifest["trainable_names"] == list(trainable)
            and manifest["initialization"] == initialization,
            "Checkpoint trainable inventory or original fresh initialization differs.")
    history = read(path / "history.json")
    exposure = validate_history(config, rows, manifest["updates_completed"], manifest["draws_completed"],
                                history, manifest["elapsed_seconds"], manifest["exposure"])
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    require(set(state) == {"trainable", "optimizer", "rng"}, "Incomplete checkpoint tensor/optimizer/RNG state.")
    primitives.validate_tensors(state["trainable"], trainable, torch, "trainable")
    primitives.validate_optimizer(state["optimizer"], optimizer, trainable, exposure.updates, torch)
    primitives.validate_rng_state(state["rng"], torch)
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state["trainable"][name])
    optimizer.load_state_dict(state["optimizer"])
    primitives.restore_rng_state(state["rng"], torch)
    return manifest, exposure, history
