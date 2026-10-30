"""Train, independently reload, and evaluate pinned small seven-action adapters."""

import argparse
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import small_actions_final as data

CHECKPOINT_VERSION = "small-actions-final-sft-checkpoint-v1"
CHECKPOINT_FILES = {"configuration.json", "runtime.json", "updates.jsonl", "state.pt"}


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(data.compact(value) + "\n")


def snapshot_implementation(output):
    root = Path(__file__).resolve().parent
    files = {Path(__file__).resolve(), Path(data.__file__).resolve(),
             Path(data.formats.__file__).resolve(), Path(data.formats.contract.__file__).resolve(),
             data.formats.contract.PROMPT_PATH.resolve(),
             *(root / name for name in ("caption_facts.py", "replay_session.py", "session_cases.py", "nep.py",
                                       "nep_v2.py", "nep_v3.py", "nep_v4.py", "model_input.py",
                                       "context_policy.py", "format_comparison.py"))}
    destination = Path(output) / "implementation"
    destination.mkdir()
    for source in sorted(files):
        shutil.copyfile(source, destination / source.name)
    return {path.name: data.sha(path) for path in sorted(destination.iterdir())}


def load_base(config, assets, *, device="cuda", cache_dir=None):
    import torch
    import transformers

    data.require(device in ("cuda", "cpu"), "Use explicit CUDA training or CPU export.")
    if device == "cuda":
        data.require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                     "This native BF16 path requires a BF16-capable CUDA GPU; no dtype fallback.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    transformers.set_seed(config["training"]["seed"])
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    tokenizer = data.tokenizer_from(assets, config["model"])
    cls = getattr(transformers, config["base"]["architecture"])
    model, diagnostics = cls.from_pretrained(
        config["base"]["repository"], revision=config["base"]["revision"],
        cache_dir=cache_dir, trust_remote_code=False, dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="sdpa", output_loading_info=True,
    )
    data.require(not any(diagnostics.get(key) for key in
                         ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
                 f"Native checkpoint loading mismatch: {diagnostics}")
    targets = data.select_targets(model, config["model"], config["training"]["target_mode"])
    runtime = {
        "environment": data.environment(), "base": config["base"], "dtype": "bfloat16",
        "attention": "sdpa", "targets": targets,
        "loading": {key: sorted(value) if isinstance(value, set) else value
                    for key, value in diagnostics.items()},
        "parameters": sum(p.numel() for p in model.parameters()), "device": device,
    }
    if device == "cuda":
        runtime.update(gpu=torch.cuda.get_device_name(),
                       gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                       cuda_version=torch.version.cuda)
    return torch, model, tokenizer, runtime


def attach_lora(model, config, targets):
    from peft import LoraConfig, get_peft_model

    model.requires_grad_(False)
    train = config["training"]
    model = get_peft_model(model, LoraConfig(
        r=train["rank"], lora_alpha=train["alpha"], lora_dropout=train["dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=list(targets),
    ), autocast_adapter_dtype=True)
    model.peft_config["default"].base_model_name_or_path = config["base"]["repository"]
    model.peft_config["default"].revision = config["base"]["revision"]
    return model


def adapter_state(model, torch):
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model, save_embedding_layers=False)
    data.require(state and all(".lora_A." in name or ".lora_B." in name for name in state),
                 "Only ordinary LoRA tensors may be saved.")
    return {name: value.detach().float().cpu().contiguous() for name, value in state.items()}


def verify_adapter_file(model, adapter, torch):
    from safetensors.torch import load_file

    expected = adapter_state(model, torch)
    actual = load_file(str(Path(adapter) / "adapter_model.safetensors"), device="cpu")
    data.require(set(actual) == set(expected), "Saved adapter tensor inventory differs.")
    data.require(all(actual[key].dtype == torch.float32 and torch.equal(actual[key], expected[key])
                     and torch.isfinite(actual[key]).all().item() for key in actual),
                 "Saved adapter is not an exact finite FP32 copy.")
    return {"tensor_count": len(actual), "parameters": sum(t.numel() for t in actual.values()),
            "sha256": data.sha(Path(adapter) / "adapter_model.safetensors")}


def resolved_adapter_targets(model, adapter_config):
    from peft import LoraConfig
    from peft.tuners.tuners_utils import check_target_module_exists

    # PEFT can replace full target names with equivalent suffixes during injection.
    saved = LoraConfig(**adapter_config)
    return {name for name, _ in model.named_modules() if check_target_module_exists(saved, name)}


def load_adapter(model, training, config, torch, *, trainable=False):
    from peft import PeftModel

    training = Path(training)
    info = data.read(training / "run.json")
    data.require(info["status"] == "training_complete_reload_pending", "Incomplete training run.")
    data.require(data.sha(training / "configuration.json") == info["configuration_sha256"],
                 "Training configuration changed.")
    data.require(data.read(training / "configuration.json") == config, "Adapter model/data configuration differs.")
    data.require(data.sha(training / "adapter/adapter_model.safetensors") == info["adapter"]["sha256"],
                 "Adapter bytes changed.")
    if info.get("kind") == "dpo":
        from train_small_actions_final_dpo import verify_training

        verify_training(training)
    adapter_config = data.read(training / "adapter/adapter_config.json")
    data.require(adapter_config["base_model_name_or_path"] == config["base"]["repository"]
                 and adapter_config["revision"] == config["base"]["revision"], "LoRA base pin differs.")
    settings = config["training"]
    data.require(adapter_config["r"] == settings["rank"] and adapter_config["lora_alpha"] == settings["alpha"]
                 and adapter_config["lora_dropout"] == settings["dropout"] and adapter_config["bias"] == "none"
                 and not any(adapter_config.get(key) for key in ("use_rslora", "use_dora", "rank_pattern", "alpha_pattern", "modules_to_save", "lora_bias"))
                 and resolved_adapter_targets(model, adapter_config)
                 == set(data.select_targets(model, config["model"], settings["target_mode"])),
                 "Saved LoRA execution settings differ from the frozen SFT configuration.")
    model = PeftModel.from_pretrained(model, training / "adapter", is_trainable=trainable,
                                     autocast_adapter_dtype=True)
    verify_adapter_file(model, training / "adapter", torch)
    return model


def completion_logps(model, row, torch, device="cuda"):
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    start = row["completion_start"]
    positions = torch.arange(start - 1, ids.shape[1] - 1, device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=positions, return_dict=True).logits
    targets = torch.tensor(row["labels"][start:], dtype=torch.long, device=device)
    data.require(logits.shape[:2] == (1, len(targets)), "Completion-position logits have wrong shape.")
    return -torch.nn.functional.cross_entropy(logits[0].float(), targets, reduction="none")


def generate_one(model, tokenizer, prefix, config, torch, *, device="cuda"):
    from transformers import GenerationConfig

    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                generation_config=GenerationConfig(**config["generation"]))
    if device == "cuda":
        torch.cuda.synchronize()
    tokens = output[0, len(prefix):].tolist()
    return {
        "output_ids": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=True),
        "text_raw": tokenizer.decode(tokens, skip_special_tokens=False),
        "finish_reason": "stop" if tokens and tokens[-1] in config["generation"]["eos_token_id"] else "length",
        "generation_ms": (time.perf_counter() - started) * 1000,
    }


def diagnostic_indices(rows):
    chosen = {}
    for index, row in enumerate(rows):
        chosen.setdefault((row["language"], row["action"] == "no_action"), index)
    return list(chosen.values())


def diagnostics(model, tokenizer, rows, config, torch):
    model.eval()
    model.gradient_checkpointing_disable()
    result = []
    for index in diagnostic_indices(rows):
        row = rows[index]
        with torch.inference_mode():
            loss = -completion_logps(model, row, torch).mean().item()
        generated = generate_one(model, tokenizer, row["input_ids"][:row["completion_start"]], config, torch)
        result.append({"case_id": row["case_id"], "loss": loss, **generated})
    return result


def verify_diagnostics(expected, actual):
    data.require(len(expected) == len(actual) > 0, "Reload diagnostics are incomplete.")
    for before, after in zip(expected, actual, strict=True):
        data.require(before["case_id"] == after["case_id"] and before["output_ids"] == after["output_ids"],
                     "Independent reload changed deterministic diagnostic generation.")
        data.require(math.isfinite(after["loss"]) and math.isclose(before["loss"], after["loss"], rel_tol=1e-3, abs_tol=1e-4),
                     "Independent reload changed diagnostic loss.")


def checkpoint_runtime(runtime):
    return {key: runtime[key] for key in
            ("environment", "base", "dtype", "attention", "targets", "parameters", "device", "cuda_version")
            if key in runtime}


def validate_progress(config, rows, updates, draws, history, elapsed):
    order, accumulation = config["training"]["sample_order"], config["training"]["gradient_accumulation"]
    data.require(type(accumulation) is int and accumulation > 0 and order
                 and all(type(index) is int and 0 <= index < len(rows) for index in order),
                 "Invalid frozen training order or accumulation.")
    data.require(type(updates) is int and type(draws) is int
                 and 0 <= updates <= math.ceil(len(order) / accumulation)
                 and draws == min(updates * accumulation, len(order)),
                 "Checkpoint cursor is not a completed optimizer-step boundary.")
    data.require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0
                 and len(history) == updates, "Checkpoint update history or elapsed time is incomplete.")
    previous = 0
    for update, event in enumerate(history, 1):
        group = order[(update - 1) * accumulation:update * accumulation]
        data.require(event["update"] == update and event["draws_completed"] == min(update * accumulation, len(order))
                     and event["case_ids"] == [rows[index]["case_id"] for index in group],
                     "Checkpoint update history differs from the frozen sample order.")
        data.require(all(math.isfinite(event[key]) for key in ("loss", "gradient_norm", "elapsed_seconds"))
                     and event["gradient_norm"] >= 0 and previous <= event["elapsed_seconds"] <= elapsed,
                     "Checkpoint update history contains invalid metrics or elapsed time.")
        previous = event["elapsed_seconds"]


def cpu_state(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(item, torch) for item in value)
    return value


def rng_state(torch):
    import numpy as np

    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"generator": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3], "gauss": numpy_state[4]},
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def numpy_rng_state(value):
    import numpy as np

    data.require(set(value) == {"generator", "keys", "position", "has_gauss", "gauss"}
                 and value["generator"] == "MT19937" and len(value["keys"]) == 624
                 and all(type(key) is int and 0 <= key < 2 ** 32 for key in value["keys"])
                 and type(value["position"]) is int and 0 <= value["position"] <= 624
                 and value["has_gauss"] in (0, 1) and math.isfinite(value["gauss"]),
                 "Invalid checkpoint NumPy RNG state.")
    return (value["generator"], np.asarray(value["keys"], dtype=np.uint32),
            value["position"], value["has_gauss"], value["gauss"])


def validate_rng_state(state, torch):
    import numpy as np

    data.require(set(state) == {"python", "numpy", "cpu", "cuda"}, "Incomplete checkpoint RNG state.")
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(numpy_rng_state(state["numpy"]))
    current = rng_state(torch)
    data.require(len(state["cuda"]) == len(current["cuda"]), "Checkpoint CUDA RNG device inventory differs.")
    for name, saved, expected in [("cpu", state["cpu"], current["cpu"]),
                                  *[(f"cuda:{i}", saved, current["cuda"][i]) for i, saved in enumerate(state["cuda"])]]:
        data.require(isinstance(saved, torch.Tensor) and saved.dtype == torch.uint8
                     and saved.shape == expected.shape, "Invalid checkpoint torch RNG state.")
        torch.Generator(device=name).set_state(saved)


def restore_rng_state(state, torch):
    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(numpy_rng_state(state["numpy"]))
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_tensors(actual, expected, torch, label):
    data.require(set(actual) == set(expected), f"Checkpoint {label} tensor inventory differs.")
    data.require(all(isinstance(actual[name], torch.Tensor) and actual[name].shape == tensor.shape
                     and actual[name].dtype == tensor.dtype and torch.isfinite(actual[name]).all().item()
                     for name, tensor in expected.items()), f"Checkpoint {label} tensors are invalid.")


def validate_optimizer(state, optimizer, trainable, updates, torch):
    expected = optimizer.state_dict()
    data.require(set(state) == {"state", "param_groups"} and state["param_groups"] == expected["param_groups"],
                 "Checkpoint optimizer settings or parameter order differ.")
    identities = [identity for group in expected["param_groups"] for identity in group["params"]]
    data.require(set(state["state"]) == (set(identities) if updates else set()),
                 "Checkpoint optimizer state is incomplete.")
    for identity, parameter in zip(identities, trainable.values(), strict=True):
        if not updates:
            continue
        saved = state["state"][identity]
        data.require(set(saved) == {"step", "exp_avg", "exp_avg_sq"}
                     and isinstance(saved["step"], torch.Tensor) and saved["step"].ndim == 0
                     and saved["step"].item() == updates, "Checkpoint optimizer step count differs.")
        validate_tensors({key: saved[key] for key in ("exp_avg", "exp_avg_sq")},
                         {key: parameter for key in ("exp_avg", "exp_avg_sq")}, torch, "optimizer")
        data.require((saved["exp_avg_sq"] >= 0).all().item(), "Invalid checkpoint optimizer second moment.")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(output, config, rows, info, runtime, trainable, initial, optimizer, history, elapsed, torch,
                    *, checkpoint_dir=None):
    validate_progress(config, rows, info["updates_completed"], info["draws_completed"], history, elapsed)
    parent = Path(checkpoint_dir) if checkpoint_dir else Path(output).with_name(Path(output).name + "-checkpoints")
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"checkpoint-{info['updates_completed']:06d}"
    data.require(not destination.exists(), "Checkpoint already exists; use a fresh output directory.")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
    try:
        for name in ("configuration.json", "runtime.json"):
            shutil.copyfile(Path(output) / name, temporary / name)
        data.write_rows(temporary / "updates.jsonl", history)
        state = cpu_state({"trainable": trainable, "initial_adapter": initial,
                           "optimizer": optimizer.state_dict(), "rng": rng_state(torch)}, torch)
        validate_tensors(state["trainable"], trainable, torch, "trainable")
        validate_optimizer(state["optimizer"], optimizer, trainable, info["updates_completed"], torch)
        torch.save(state, temporary / "state.pt")
        manifest = {"version": CHECKPOINT_VERSION, "status": "optimizer_step_complete",
                    "configuration_sha256": info["configuration_sha256"],
                    "implementation_sha256": info["implementation_sha256"], "runtime": checkpoint_runtime(runtime),
                    "source_run": str(Path(output).resolve()), "source_pid": os.getpid(),
                    "rows": len(rows), "draws": len(config["training"]["sample_order"]),
                    "updates_completed": info["updates_completed"], "draws_completed": info["draws_completed"],
                    "elapsed_seconds": elapsed, "trainable_names": list(trainable),
                    "resume_history": info["resume_history"], "final_test_bodies_read": False,
                    "files_sha256": {name: data.sha(temporary / name) for name in sorted(CHECKPOINT_FILES)}}
        data.write(temporary / "checkpoint.json", manifest)
        for path in temporary.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        sync_directory(temporary)
        temporary.rename(destination)
        sync_directory(parent)
        sync_directory(parent.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"path": str(destination), "sha256": data.sha(destination / "checkpoint.json"),
            "updates_completed": info["updates_completed"], "draws_completed": info["draws_completed"]}


def restore_checkpoint(path, config, rows, info, runtime, trainable, initial, optimizer, torch):
    path = Path(path).resolve()
    data.require(path.is_dir() and {item.name for item in path.iterdir()} == CHECKPOINT_FILES | {"checkpoint.json"}
                 and all(item.is_file() and not item.is_symlink() for item in path.iterdir()),
                 "Incomplete checkpoint file inventory.")
    manifest = data.read(path / "checkpoint.json")
    data.require(manifest["version"] == CHECKPOINT_VERSION and manifest["status"] == "optimizer_step_complete"
                 and path.name == f"checkpoint-{manifest['updates_completed']:06d}"
                 and set(manifest["files_sha256"]) == CHECKPOINT_FILES, "Incomplete or unpublished checkpoint.")
    for name, expected in manifest["files_sha256"].items():
        data.require(data.sha(path / name) == expected, f"Checkpoint file SHA256 differs: {name}")
    data.require(manifest["configuration_sha256"] == info["configuration_sha256"]
                 == manifest["files_sha256"]["configuration.json"] and data.read(path / "configuration.json") == config,
                 "Checkpoint frozen configuration differs.")
    data.require(manifest["implementation_sha256"] == info["implementation_sha256"],
                 "Checkpoint trainer implementation differs.")
    data.require(manifest["runtime"] == checkpoint_runtime(runtime)
                 == checkpoint_runtime(data.read(path / "runtime.json")), "Checkpoint runtime settings differ.")
    data.require(manifest["rows"] == len(rows) and manifest["draws"] == len(config["training"]["sample_order"])
                 and manifest["trainable_names"] == list(trainable)
                 and manifest["final_test_bodies_read"] is False, "Checkpoint training inventory differs.")
    history = data.rows(path / "updates.jsonl")
    validate_progress(config, rows, manifest["updates_completed"], manifest["draws_completed"], history, manifest["elapsed_seconds"])
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    data.require(set(state) == {"trainable", "initial_adapter", "optimizer", "rng"}, "Incomplete checkpoint state.")
    validate_tensors(state["trainable"], trainable, torch, "trainable")
    validate_tensors(state["initial_adapter"], initial, torch, "initial adapter")
    validate_optimizer(state["optimizer"], optimizer, trainable, manifest["updates_completed"], torch)
    validate_rng_state(state["rng"], torch)
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state["trainable"][name])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"], torch)
    return manifest, state["initial_adapter"], history


def train(preparation, output, *, assets=None, cache_dir=None, checkpoint_every=0, resume_checkpoint=None,
          checkpoint_dir=None):
    data.require(type(checkpoint_every) is int and checkpoint_every >= 0, "--checkpoint-every must be nonnegative.")
    data.require(checkpoint_dir is None or checkpoint_every > 0, "--checkpoint-dir requires checkpoint saving.")
    config, rows, assets = data.prepared(preparation, assets)
    output = data.new_directory(output)
    data.write(output / "configuration.json", config)
    info = {"version": data.VERSION, "status": "loading", "pid": os.getpid(),
            "configuration_sha256": data.sha(output / "configuration.json"),
            "implementation_sha256": snapshot_implementation(output),
            "rows": len(rows), "draws": len(config["training"]["sample_order"]),
            "updates_completed": 0, "draws_completed": 0, "final_test_bodies_read": False,
            "initialization": "fresh_seeded_adapter_fresh_optimizer", "resume_history": [],
            "checkpoint_every_updates": checkpoint_every, "resumed_elapsed_seconds": 0}
    data.write(output / "run.json", info)
    started = time.monotonic()
    prior_elapsed, history = 0, []

    def elapsed():
        return prior_elapsed + time.monotonic() - started

    def checkpoint():
        info["latest_checkpoint"] = save_checkpoint(
            output, config, rows, info, runtime, trainable, initial, optimizer, history, elapsed(), torch,
            checkpoint_dir=checkpoint_dir)
        data.write(output / "run.json", info)
        print(data.compact({"checkpoint_saved": info["latest_checkpoint"]}), flush=True)

    try:
        if checkpoint_every:
            checkpoint_dir = (Path(checkpoint_dir).resolve() if checkpoint_dir
                              else output.with_name(output.name + "-checkpoints"))
            data.require(checkpoint_dir != output and output not in checkpoint_dir.parents,
                         "Checkpoints must be outside the final training output directory.")
            data.require(not checkpoint_dir.exists(), "Use a fresh checkpoint directory.")
            info["checkpoint_directory"] = str(checkpoint_dir)
        torch, model, tokenizer, runtime = load_base(config, assets, cache_dir=cache_dir)
        data.write(output / "runtime.json", runtime)
        for row in rows:
            data.require(data.prompt_ids(tokenizer, row["messages"]) == row["input_ids"][:row["completion_start"]],
                         "Loaded tokenizer changed prepared prompt tokens.")
        model = attach_lora(model, config, runtime["targets"])
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        data.require(trainable and len(trainable) == 2 * len(runtime["targets"])
                     and all(".lora_" in name and p.dtype == torch.float32 for name, p in trainable.items()),
                     "Unexpected trainable parameters or adapter dtype.")
        initial = cpu_state(adapter_state(model, torch), torch)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        settings = config["training"]
        optimizer = torch.optim.AdamW(trainable.values(), lr=settings["learning_rate"],
                                      weight_decay=settings["weight_decay"])
        order, accumulation = settings["sample_order"], settings["gradient_accumulation"]
        if resume_checkpoint is not None:
            prior, initial, history = restore_checkpoint(
                resume_checkpoint, config, rows, info, runtime, trainable, initial, optimizer, torch)
            prior_elapsed = prior["elapsed_seconds"]
            info.update(updates_completed=prior["updates_completed"], draws_completed=prior["draws_completed"],
                        initialization="checkpoint_adapter_optimizer_rng", resumed_elapsed_seconds=prior_elapsed,
                        resume_history=[*prior["resume_history"], {
                            "checkpoint": str(Path(resume_checkpoint).resolve()),
                            "checkpoint_sha256": data.sha(Path(resume_checkpoint) / "checkpoint.json"),
                            "source_run": prior["source_run"], "source_pid": prior["source_pid"],
                            "updates_completed": prior["updates_completed"], "draws_completed": prior["draws_completed"],
                            "elapsed_seconds": prior_elapsed}])
            data.write_rows(output / "updates.jsonl", history)
        validate_progress(config, rows, info["updates_completed"], info["draws_completed"], history, elapsed())
        info["status"] = "training"
        data.write(output / "run.json", info)
        if checkpoint_every:
            checkpoint()
        for offset in range(info["draws_completed"], len(order), accumulation):
            group = order[offset:offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in group:
                row = rows[index]
                loss = -completion_logps(model, row, torch).mean()
                data.require(torch.isfinite(loss).item(), "Non-finite training loss.")
                (loss / len(group)).backward()
                losses.append(loss.detach().item())
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), settings["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            info["updates_completed"] += 1
            info["draws_completed"] += len(group)
            event = {"update": info["updates_completed"], "draws_completed": info["draws_completed"],
                     "case_ids": [rows[i]["case_id"] for i in group], "loss": sum(losses) / len(losses),
                     "gradient_norm": float(norm), "elapsed_seconds": elapsed()}
            append(output / "updates.jsonl", event)
            history.append(event)
            if info["updates_completed"] == 1 or info["updates_completed"] % 10 == 0:
                print(data.compact(event), flush=True)
            if checkpoint_every and (info["updates_completed"] % checkpoint_every == 0
                                     or info["draws_completed"] == len(order)):
                checkpoint()
        validate_progress(config, rows, info["updates_completed"], info["draws_completed"], history, elapsed())
        data.require(info["draws_completed"] == len(order), "Training did not finish the frozen optimizer budget.")
        final = adapter_state(model, torch)
        data.require(any(not torch.equal(initial[name], value) for name, value in final.items()), "Training did not change adapter weights.")
        model.save_pretrained(output / "adapter", safe_serialization=True, save_embedding_layers=False)
        info["adapter"] = verify_adapter_file(model, output / "adapter", torch)
        data.write(output / "diagnostics.json", diagnostics(model, tokenizer, rows, config, torch))
        info.update(status="training_complete_reload_pending", seconds=elapsed(),
                    execution_seconds_this_attempt=time.monotonic() - started,
                    peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated())
        data.write(output / "run.json", info)
        return info
    except BaseException as error:
        info.update(status="failed_incomplete", error_type=type(error).__name__, error=str(error),
                    seconds=elapsed(), execution_seconds_this_attempt=time.monotonic() - started)
        data.write(output / "run.json", info)
        raise


def reload(preparation, training, output, *, assets=None, cache_dir=None):
    config, rows, assets = data.prepared(preparation, assets)
    training, output = Path(training).resolve(), data.new_directory(output)
    prior = data.read(training / "run.json")
    data.require(prior.get("kind", "sft") == "sft", "Use train_small_actions_final_dpo.py reload for DPO pair diagnostics.")
    data.require(prior["pid"] != os.getpid(), "Reload must run in a separate fresh process.")
    torch, model, tokenizer, runtime = load_base(config, assets, cache_dir=cache_dir)
    data.require(runtime["environment"] == data.read(training / "runtime.json")["environment"],
                 "Independent reload uses a different library environment.")
    model = load_adapter(model, training, config, torch)
    actual = diagnostics(model, tokenizer, rows, config, torch)
    data.write(output / "diagnostics.json", actual)
    verify_diagnostics(data.read(training / "diagnostics.json"), actual)
    result = {"version": data.VERSION, "status": "independent_reload_passed", "pid": os.getpid(),
              "training_configuration_sha256": data.sha(training / "configuration.json"),
              "adapter_sha256": prior["adapter"]["sha256"], "diagnostics": len(actual),
              "runtime": runtime, "final_test_bodies_read": False}
    data.write(output / "run.json", result)
    return result


def verified_reload(training, reloaded):
    training, reloaded = Path(training), Path(reloaded)
    report = data.read(reloaded / "run.json")
    data.require(report["status"] == "independent_reload_passed"
                 and report["training_configuration_sha256"] == data.sha(training / "configuration.json")
                 and report["adapter_sha256"] == data.sha(training / "adapter/adapter_model.safetensors"),
                 "Export/inference requires the same independently reloaded adapter.")
    if (training / "dpo_configuration.json").exists():
        from train_small_actions_final_dpo import verify_training

        info, dpo = verify_training(training)
        data.require(report.get("kind") == "dpo"
                     and report.get("dpo_configuration_sha256") == info["dpo_configuration_sha256"]
                     and report.get("parent_adapter_sha256") == dpo["parent"]["adapter_sha256"]
                     and report.get("pair_diagnostics") == info["rows"],
                     "DPO requires its independently reloaded pair diagnostics and exact SFT parent.")
    return report


def model_identity(training, reloaded):
    verified_reload(training, reloaded)
    training, reloaded = Path(training), Path(reloaded)
    info = data.read(training / "run.json")
    return {"kind": info.get("kind", "sft"),
            "configuration_sha256": data.sha(training / "configuration.json"),
            "adapter_sha256": data.sha(training / "adapter/adapter_model.safetensors"),
            "reload_run_sha256": data.sha(reloaded / "run.json")}


def validate_requests(requests, prepared_rows):
    data.require(requests and len({r.get("case_id", r.get("id")) for r in requests}) == len(requests), "Duplicate or empty requests.")
    train_rows = {row["case_id"]: row for row in prepared_rows}
    for request in requests:
        data.require(request["split"] in ("train", "dev")
                     and not {"primary", "acceptable", "accepted_predictions", "judgment", "rationale"}.intersection(request),
                     "Inference accepts label-free TRAIN or exposed DEV requests only.")
        data.require(request["messages"] == data.formats.messages(request["input"]), "Saved inference prompt differs.")
        if request["split"] == "train":
            trained = train_rows.get(request.get("case_id", request.get("id")))
            data.require(trained is not None and trained["messages"] == request["messages"]
                         and trained["semantic_input_sha256"] == data.value_hash(
                             data.formats.contract.wire_snapshot(request["input"])),
                         "TRAIN inference must match the prepared training cohort exactly.")


def infer(preparation, requests, output, *, training=None, reloaded=None, assets=None, cache_dir=None):
    config, prepared_rows, assets = data.prepared(preparation, assets)
    data.require(bool(training) == bool(reloaded), "Supply both training and independent reload for SFT inference.")
    if training:
        identity = model_identity(training, reloaded)
    else:
        identity = None
    request_path = Path(requests)
    requests = data.rows(request_path)
    validate_requests(requests, prepared_rows)
    output = data.new_directory(output)
    torch, model, tokenizer, runtime = load_base(config, assets, cache_dir=cache_dir)
    if training:
        model = load_adapter(model, training, config, torch)
    model.eval()
    model.gradient_checkpointing_disable()
    data.write(output / "runtime.json", runtime)
    failures = 0
    for request in requests:
        prefix = data.prompt_ids(tokenizer, request["messages"])
        data.require(len(prefix) <= config["context_tokens"] - config["max_completion_tokens"],
                     "Case exceeds declared context; no truncation or row exclusion is allowed.")
        row = {"case_id": request.get("case_id", request.get("id")), "input_sha256": data.value_hash(data.formats.contract.wire_snapshot(request["input"])),
               "messages_sha256": data.value_hash(request["messages"]), "format": "json", "prefix_tokens": len(prefix),
               "prefix_ids_sha256": data.value_hash(prefix)}
        try:
            row.update(status="completed", **generate_one(model, tokenizer, prefix, config, torch))
        except Exception as error:
            row.update(status="request_failure", error_type=type(error).__name__, error=str(error))
            failures += 1
        append(output / "observations.jsonl", row)
        print(data.compact({"case_id": row["case_id"], "status": row["status"]}), flush=True)
    result = {"version": data.VERSION, "status": "inference_complete", "requested": len(requests),
              "failures": failures, "kind": identity["kind"] if identity else "base", "schema_constrained": False,
              "model_identity": identity, "observations_sha256": data.sha(output / "observations.jsonl"),
              "configuration_sha256": data.sha(Path(preparation) / "configuration.json"),
              "request_sha256": data.sha(request_path), "splits": sorted({row["split"] for row in requests}),
              "final_test_bodies_read": False}
    data.write(output / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "reload", "infer"))
    for name in ("preparation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("training", "reloaded", "requests", "assets", "cache-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save an initial checkpoint, every N completed updates, and the final update; 0 disables saving.")
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Resume a complete checkpoint into a fresh --output directory with the same frozen preparation.")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Fresh checkpoint directory outside --output; defaults to the sibling <output>-checkpoints.")
    args = parser.parse_args()
    shared = {"assets": args.assets, "cache_dir": args.cache_dir}
    data.require(args.command == "train" or (args.checkpoint_every == 0 and args.resume_checkpoint is None
                                            and args.checkpoint_dir is None),
                 "Checkpoint options apply only to train.")
    if args.command == "train":
        result = train(args.preparation, args.output, checkpoint_every=args.checkpoint_every,
                       resume_checkpoint=args.resume_checkpoint, checkpoint_dir=args.checkpoint_dir, **shared)
    elif args.command == "reload":
        data.require(args.training, "--training is required for reload.")
        result = reload(args.preparation, args.training, args.output, **shared)
    else:
        data.require(args.requests, "--requests is required for inference.")
        result = infer(args.preparation, args.requests, args.output, training=args.training, reloaded=args.reloaded, **shared)
    print(data.compact(result), flush=True)


if __name__ == "__main__":
    main()
