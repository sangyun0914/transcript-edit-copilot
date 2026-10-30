"""Cache exact SFT reference scores, train standard DPO, and independently reload.

configuration.json remains the native model/data identity for existing serving.
Only dpo_configuration.json describes the DPO optimizer budget and parent.
"""

import argparse
import math
import os
from pathlib import Path
import shutil
import time

import prepare_small_actions_final_dpo as prep
import small_actions_final as data
import train_small_actions_final as sft


def verify_training(training):
    training = Path(training)
    info, config = data.read(training / "run.json"), data.read(training / "dpo_configuration.json")
    data.require(info["kind"] == "dpo" and info["status"] == "training_complete_reload_pending"
                 and info["dpo_configuration_sha256"] == data.sha(training / "dpo_configuration.json")
                 and info["configuration_sha256"] == config["parent"]["configuration_sha256"]
                 == data.sha(training / "configuration.json")
                 and info["parent_adapter_sha256"] == config["parent"]["adapter_sha256"]
                 and data.sha(training / "plan.md") == config["plan_sha256"]
                 and info["adapter"]["sha256"] == data.sha(training / "adapter/adapter_model.safetensors")
                 and info["rows"] == config["pairs"]
                 and info["draws_completed"] == len(config["training"]["sample_order"])
                 and info["updates_completed"] == config["training"]["updates"],
                 "DPO parent, actual optimizer budget or saved adapter changed/incomplete.")
    return info, config


def snapshot(preparation, output):
    for name in ("configuration.json", "dpo_configuration.json", "plan.md"):
        shutil.copyfile(Path(preparation) / name, output / name)
    hashes = sft.snapshot_implementation(output)
    for source in (Path(__file__), Path(prep.__file__)):
        destination = output / "implementation" / source.name
        shutil.copyfile(source, destination)
        hashes[source.name] = data.sha(destination)
    return hashes


def load_parent(config, dpo, assets, training, reloaded, *, cache_dir=None, trainable=False):
    data.require(sft.model_identity(training, reloaded) == dpo["parent"], "Reference must be the exact selected frozen SFT parent.")
    torch, model, tokenizer, runtime = sft.load_base(config, assets, cache_dir=cache_dir)
    data.require(runtime["environment"] == data.read(Path(training) / "runtime.json")["environment"],
                 "DPO must use the independently verified SFT library environment.")
    model = sft.load_adapter(model, training, config, torch, trainable=trainable)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            data.require(module.p == 0, "This exact-parent DPO path requires zero dropout, as registered for SFT.")
    model.eval()
    model.gradient_checkpointing_disable()
    return torch, model, tokenizer, runtime


def summed_logprob(model, row, torch):
    value = sft.completion_logps(model, row, torch).sum()
    data.require(torch.isfinite(value).item(), "Non-finite summed completion log probability.")
    return value


def dpo_loss(chosen, rejected, reference, beta, torch):
    margin = beta * ((chosen - rejected) - (reference["chosen_logprob"] - reference["rejected_logprob"]))
    return torch.nn.functional.softplus(-margin), margin


def pair_scores(model, pair, torch):
    with torch.no_grad():
        return {side + "_logprob": float(summed_logprob(model, pair[side], torch)) for side in prep.SIDES}


def close_scores(expected, actual):
    for key in ("chosen_logprob", "rejected_logprob"):
        data.require(math.isfinite(actual[key]) and math.isclose(expected[key], actual[key], rel_tol=1e-5, abs_tol=1e-4),
                     "Exact frozen parent/reference or independent reload score parity failed.")


def reference(preparation, training, reloaded, output, *, assets=None, cache_dir=None):
    config, dpo, pairs, assets = prep.prepared(preparation, assets)
    output = data.new_directory(output)
    info = {"version": prep.VERSION, "kind": "dpo_reference", "status": "loading", "pid": os.getpid(),
            "parent": dpo["parent"], "dpo_configuration_sha256": data.sha(Path(preparation) / "dpo_configuration.json"),
            "pairs_sha256": dpo["pairs_sha256"], "scored": 0, "final_test_bodies_read": False}
    data.write(output / "run.json", info)
    try:
        torch, model, tokenizer, runtime = load_parent(config, dpo, assets, training, reloaded, cache_dir=cache_dir)
        data.require(not any(parameter.requires_grad for parameter in model.parameters()), "Reference policy is not frozen.")
        data.write(output / "runtime.json", runtime)
        for pair in pairs:
            data.require(data.prompt_ids(tokenizer, pair["messages"]) == pair["chosen"]["input_ids"][:pair["chosen"]["completion_start"]],
                         "Native tokenizer changed the frozen pair prefix.")
            record = {"pair_id": pair["pair_id"], "pair_sha256": data.value_hash(pair), **pair_scores(model, pair, torch)}
            sft.append(output / "reference.jsonl", record)
            info["scored"] += 1
            if info["scored"] % 25 == 0:
                print(data.compact({"reference_pairs_scored": info["scored"]}), flush=True)
        info.update(status="frozen_reference_complete", reference_sha256=data.sha(output / "reference.jsonl"))
        data.write(output / "run.json", info)
        return info
    except BaseException as error:
        info.update(status="failed_incomplete", error_type=type(error).__name__, error=str(error))
        data.write(output / "run.json", info)
        raise


def reference_cache(directory, preparation, dpo, pairs):
    directory = Path(directory)
    info = data.read(directory / "run.json")
    data.require(info["status"] == "frozen_reference_complete" and info["parent"] == dpo["parent"]
                 and info["dpo_configuration_sha256"] == data.sha(Path(preparation) / "dpo_configuration.json")
                 and info["pairs_sha256"] == dpo["pairs_sha256"] and info["scored"] == len(pairs)
                 and info["reference_sha256"] == data.sha(directory / "reference.jsonl"),
                 "Reference cache is incomplete or belongs to a different parent/pair configuration.")
    records = data.rows(directory / "reference.jsonl")
    data.require(len(records) == len(pairs), "Incomplete reference scores.")
    for pair, record in zip(pairs, records, strict=True):
        data.require(record["pair_id"] == pair["pair_id"] and record["pair_sha256"] == data.value_hash(pair)
                     and all(math.isfinite(record[side + "_logprob"]) and record[side + "_logprob"] <= 0 for side in prep.SIDES),
                     "Reference score binding changed or is non-finite.")
    return records, info


def diagnostic_indices(pairs):
    chosen = {}
    for index, pair in enumerate(pairs):
        chosen.setdefault((pair["language"], pair["chosen_action"]), index)
    longest = max(range(len(pairs)), key=lambda i: max(len(pairs[i][side]["input_ids"]) for side in prep.SIDES))
    return sorted({*chosen.values(), longest})


def diagnostics(model, tokenizer, pairs, references, config, dpo, torch):
    model.eval()
    model.gradient_checkpointing_disable()
    generated_indices = set(diagnostic_indices(pairs))
    result = []
    for index, (pair, reference) in enumerate(zip(pairs, references, strict=True)):
        values = pair_scores(model, pair, torch)
        gap = values["chosen_logprob"] - values["rejected_logprob"]
        ref_gap = reference["chosen_logprob"] - reference["rejected_logprob"]
        margin = dpo["training"]["beta"] * (gap - ref_gap)
        row = {"pair_id": pair["pair_id"], "case_id": pair["case_id"], "pair_sha256": data.value_hash(pair),
               **values, "policy_logprob_gap": gap, "reference_logprob_gap": ref_gap,
               "preference_margin": margin, "dpo_loss": max(0, -margin) + math.log1p(math.exp(-abs(margin))),
               "reference_chosen_logprob": reference["chosen_logprob"],
               "reference_rejected_logprob": reference["rejected_logprob"]}
        for side in prep.SIDES:
            count = len(pair[side]["input_ids"]) - pair[side]["completion_start"]
            row[side + "_completion_tokens"] = count
            row[side + "_nll_mean"] = -values[side + "_logprob"] / count
            row["reference_" + side + "_nll_mean"] = -reference[side + "_logprob"] / count
        if index in generated_indices:
            completion = pair["chosen"]
            row["generation"] = sft.generate_one(model, tokenizer, completion["input_ids"][:completion["completion_start"]], config, torch)
            row["parent_output_ids"] = pair["parent_output_ids"]
            row["parent_text"] = pair["parent_text"]
            row["output_changed"] = row["generation"]["output_ids"] != pair["parent_output_ids"]
        result.append(row)
    return result


def summarize_diagnostics(rows):
    generated = [row for row in rows if "generation" in row]
    return {"scored_pairs": len(rows), "output_change_sample": len(generated),
            "output_changed": sum(row["output_changed"] for row in generated),
            "output_sample_rule": "First pair per chosen action/language, plus the longest native pair.",
            "mean_preference_margin": math.fsum(row["preference_margin"] for row in rows) / len(rows),
            "mean_dpo_loss": math.fsum(row["dpo_loss"] for row in rows) / len(rows),
            "mean_chosen_nll_change": math.fsum(row["chosen_nll_mean"] - row["reference_chosen_nll_mean"] for row in rows) / len(rows),
            "mean_rejected_nll_change": math.fsum(row["rejected_nll_mean"] - row["reference_rejected_nll_mean"] for row in rows) / len(rows),
            "scope": "TRAIN fitting diagnostic only; adoption requires the separately frozen full DEV rules."}


def verify_diagnostics(expected, actual):
    data.require(len(expected) == len(actual) > 0, "Incomplete DPO reload diagnostics.")
    for before, after in zip(expected, actual, strict=True):
        data.require(before["pair_id"] == after["pair_id"] and before["pair_sha256"] == after["pair_sha256"], "Reload pairs changed.")
        close_scores(before, after)
        data.require(("generation" in before) == ("generation" in after), "Reload output sample changed.")
        if "generation" in before:
            data.require(before["generation"]["output_ids"] == after["generation"]["output_ids"],
                         "Independent reload changed deterministic DPO generation.")


def backward_pair(model, pair, reference, beta, divisor, torch):
    # Recompute chosen after rejected backward to keep only one activation graph.
    with torch.no_grad():
        scored = summed_logprob(model, pair["chosen"], torch)
    rejected = summed_logprob(model, pair["rejected"], torch)
    loss, margin = dpo_loss(scored, rejected, reference, beta, torch)
    coefficient = (-beta * torch.sigmoid(-margin.detach())).detach()
    event = {"pair_id": pair["pair_id"], "loss": float(loss.detach()), "margin": float(margin.detach()),
             "chosen_logprob": float(scored), "rejected_logprob": float(rejected.detach())}
    (loss / divisor).backward()
    del loss, margin, rejected
    chosen = summed_logprob(model, pair["chosen"], torch)
    data.require(torch.isclose(chosen.detach(), scored, rtol=1e-5, atol=1e-4).item(),
                 "Sequential DPO recomputation changed chosen scores.")
    (coefficient * chosen / divisor).backward()
    return event


def train(preparation, training, reloaded, reference, output, *, assets=None, cache_dir=None):
    config, dpo, pairs, assets = prep.prepared(preparation, assets)
    references, ref_info = reference_cache(reference, preparation, dpo, pairs)
    output = data.new_directory(output)
    implementation = snapshot(preparation, output)
    info = {"version": prep.VERSION, "kind": "dpo", "status": "loading", "pid": os.getpid(),
            "configuration_sha256": data.sha(output / "configuration.json"),
            "dpo_configuration_sha256": data.sha(output / "dpo_configuration.json"),
            "reference_run_sha256": data.sha(Path(reference) / "run.json"),
            "parent_adapter_sha256": dpo["parent"]["adapter_sha256"], "implementation_sha256": implementation,
            "rows": len(pairs), "draws": len(dpo["training"]["sample_order"]),
            "draws_completed": 0, "updates_completed": 0, "initialization": "exact_sft_adapter_fresh_optimizer",
            "final_test_bodies_read": False}
    data.write(output / "run.json", info)
    started = time.monotonic()
    try:
        data.require(ref_info["pid"] != os.getpid(), "Cache reference scores in a separate process before training.")
        torch, model, tokenizer, runtime = load_parent(config, dpo, assets, training, reloaded, cache_dir=cache_dir, trainable=True)
        data.require(runtime["environment"] == data.read(Path(reference) / "runtime.json")["environment"], "Reference environment changed.")
        data.write(output / "runtime.json", runtime)
        for pair, ref in zip(pairs, references, strict=True):
            actual = pair_scores(model, pair, torch)
            close_scores(ref, actual)
            sft.append(output / "initial-scores.jsonl", {"pair_id": pair["pair_id"], **actual,
                        "preference_margin": dpo["training"]["beta"] * (
                            actual["chosen_logprob"] - actual["rejected_logprob"] - ref["chosen_logprob"] + ref["rejected_logprob"])})
        info["initial_reference_parity_pairs"] = len(pairs)
        initial = sft.adapter_state(model, torch)
        trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
        data.require(trainable and all(".lora_" in name and value.dtype == torch.float32 for name, value in trainable.items()),
                     "DPO may update only the existing FP32 SFT LoRA tensors.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        settings = dpo["training"]
        torch.manual_seed(settings["seed"])
        optimizer = torch.optim.AdamW(trainable.values(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        order, accumulation = settings["sample_order"], settings["gradient_accumulation"]
        for offset in range(0, len(order), accumulation):
            group = order[offset:offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            events = [backward_pair(model, pairs[index], references[index], settings["beta"], len(group), torch) for index in group]
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), settings["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            info["updates_completed"] += 1
            info["draws_completed"] += len(group)
            event = {"update": info["updates_completed"], "draws_completed": info["draws_completed"],
                     "gradient_norm": float(norm), "pairs": events, "elapsed_seconds": time.monotonic() - started}
            sft.append(output / "updates.jsonl", event)
            if info["updates_completed"] == 1 or info["updates_completed"] % 10 == 0:
                print(data.compact({"update": info["updates_completed"], "draws_completed": info["draws_completed"]}), flush=True)
        final = sft.adapter_state(model, torch)
        data.require(any(not torch.equal(initial[name], value) for name, value in final.items()), "DPO did not change any adapter weights.")
        model.save_pretrained(output / "adapter", safe_serialization=True, save_embedding_layers=False)
        info["adapter"] = sft.verify_adapter_file(model, output / "adapter", torch)
        actual = diagnostics(model, tokenizer, pairs, references, config, dpo, torch)
        data.write(output / "diagnostics.json", actual)
        data.write(output / "diagnostic-summary.json", summarize_diagnostics(actual))
        data.require(sft.model_identity(training, reloaded) == dpo["parent"], "Frozen SFT parent changed during DPO.")
        info.update(status="training_complete_reload_pending", seconds=time.monotonic() - started,
                    peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated())
        data.write(output / "run.json", info)
        verify_training(output)
        return info
    except BaseException as error:
        info.update(status="failed_incomplete", error_type=type(error).__name__, error=str(error), seconds=time.monotonic() - started)
        data.write(output / "run.json", info)
        raise


def reload(preparation, training, reference, output, *, assets=None, cache_dir=None):
    config, dpo, pairs, assets = prep.prepared(preparation, assets)
    prior, saved = verify_training(training)
    references, _ = reference_cache(reference, preparation, dpo, pairs)
    data.require(saved == dpo and prior["pid"] != os.getpid()
                 and prior["reference_run_sha256"] == data.sha(Path(reference) / "run.json"),
                 "DPO reload needs a fresh process and the same frozen preparation/reference cache.")
    output = data.new_directory(output)
    torch, model, tokenizer, runtime = sft.load_base(config, assets, cache_dir=cache_dir)
    data.require(runtime["environment"] == data.read(Path(training) / "runtime.json")["environment"], "Independent DPO reload environment differs.")
    model = sft.load_adapter(model, training, config, torch)
    actual = diagnostics(model, tokenizer, pairs, references, config, dpo, torch)
    data.write(output / "diagnostics.json", actual)
    verify_diagnostics(data.read(Path(training) / "diagnostics.json"), actual)
    result = {"version": prep.VERSION, "kind": "dpo", "status": "independent_reload_passed", "pid": os.getpid(),
              "training_configuration_sha256": prior["configuration_sha256"],
              "dpo_configuration_sha256": prior["dpo_configuration_sha256"],
              "adapter_sha256": prior["adapter"]["sha256"], "parent_adapter_sha256": dpo["parent"]["adapter_sha256"],
              "pair_diagnostics": len(actual), "runtime": runtime, "final_test_bodies_read": False}
    data.write(output / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reference", "train", "reload"))
    for name in ("preparation", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("reloaded", "reference", "assets", "cache-dir"):
        parser.add_argument("--" + name, type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "reference":
        data.require(args["reloaded"] is not None, "--reloaded must identify the SFT parent reload.")
        args.pop("reference")
        result = reference(**args)
    elif command == "train":
        data.require(args["reloaded"] is not None and args["reference"] is not None, "--reloaded and --reference are required.")
        result = train(**args)
    else:
        data.require(args["reference"] is not None, "--reference is required.")
        args.pop("reloaded")
        result = reload(**args)
    print(data.compact(result), flush=True)


if __name__ == "__main__":
    main()
