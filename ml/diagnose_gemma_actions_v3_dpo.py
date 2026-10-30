"""Score every frozen TRAIN preference before and after DPO; not a quality eval."""

import argparse
import gc
import json
import math
from pathlib import Path
import time

import train_gemma_actions_v3_dpo as dpo

VERSION = "gemma-actions-v3-dpo-training-diagnostic-1"
REFERENCE_ATOL = 1e-4
INITIAL_LOSS_ATOL = 1e-5
require, read, write, digest = dpo.require, dpo.read, dpo.write, dpo.digest


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path)}


def validate_completed_run(config, pairs, training, reloaded, config_sha, reference_sha):
    require(
        all(info["configuration_sha256"] == config_sha
            and info["reference_manifest_sha256"] == reference_sha
            for info in (training, reloaded)),
        "Training/reload belongs to another DPO configuration or reference cache.",
    )
    require(
        training["mode"] == "train" and reloaded["mode"] == "fresh_reload"
        and training["training_updates"] == config["training"]["max_updates"]
        and reloaded["reloaded_training_updates"] == config["training"]["max_updates"]
        and training["optimizer_committed_rows"] == len(pairs)
        and training["initialization"] == reloaded["initialization"]
        == "frozen_sft_adapter_fresh_optimizer",
        "Require the complete registered DPO pass and its independent reload.",
    )


def score_pairs(model, pairs, references, config, torch, device, stage, emit):
    require(stage in {"initial", "final"}, "Unknown diagnostic stage.")
    beta = config["training"]["beta"]
    records = []
    with torch.no_grad():
        for pair in pairs:
            reference = references[pair["pair_id"]]
            direct = {
                side: dpo.objective.summed_logprob(model, pair[side], torch, device)
                for side in dpo.SIDES
            }
            loss, margin = dpo.objective.dpo_loss(
                direct["chosen"], direct["rejected"], reference, beta, torch
            )
            values = {side: float(direct[side]) for side in dpo.SIDES}
            loss, margin = float(loss), float(margin)
            require(
                all(math.isfinite(value) and value <= 0 for value in values.values())
                and math.isfinite(loss) and loss >= 0 and math.isfinite(margin),
                "Non-finite or invalid policy likelihood/loss.",
            )
            errors = {
                side: abs(values[side] - reference[side + "_logprob"])
                for side in dpo.SIDES
            }
            row = {
                "stage": stage, "pair_id": pair["pair_id"], "case_id": pair["case_id"],
                "pair_binding": dpo.pair_binding(pair), "input_sha256": pair["input_sha256"],
                "policy_chosen_logprob": values["chosen"],
                "policy_rejected_logprob": values["rejected"],
                "reference_chosen_logprob": reference["chosen_logprob"],
                "reference_rejected_logprob": reference["rejected_logprob"],
                "policy_logprob_gap": values["chosen"] - values["rejected"],
                "chosen_reward": beta * (values["chosen"] - reference["chosen_logprob"]),
                "rejected_reward": beta * (values["rejected"] - reference["rejected_logprob"]),
                "preference_margin": margin, "dpo_loss": loss,
                "completion_tokens": {
                    side: len(pair[side]["input_ids"]) - pair[side]["completion_start"]
                    for side in dpo.SIDES
                },
            }
            if stage == "initial":
                row.update(
                    reference_absolute_errors=errors,
                    reference_parity_passed=all(error <= REFERENCE_ATOL for error in errors.values())
                    and abs(loss - math.log(2)) < INITIAL_LOSS_ATOL,
                )
            emit(row)
            records.append(row)
            if stage == "initial":
                require(row["reference_parity_passed"],
                        "Initial anchor differs from its reference cache; stop before final scoring.")
    return records


def verify_census(pairs, records, stage):
    require(
        pairs and len(records) == len(pairs)
        and [row["pair_id"] for row in records] == [pair["pair_id"] for pair in pairs]
        and len({row["pair_id"] for row in records}) == len(pairs)
        and all(row["stage"] == stage and row["case_id"] == pair["case_id"]
                and row["pair_binding"] == dpo.pair_binding(pair)
                and row["input_sha256"] == pair["input_sha256"]
                for pair, row in zip(pairs, records, strict=True)),
        "Every frozen pair must be scored exactly once in registered order.",
    )


def summarize(records, beta):
    count = len(records)
    require(count > 0, "An empty preference set has no diagnostic accuracy.")
    positive = sum(row["preference_margin"] > 0 for row in records)
    near_zero = 2 * beta * REFERENCE_ATOL
    return {
        "pairs": count,
        "mean_dpo_loss": math.fsum(row["dpo_loss"] for row in records) / count,
        "mean_preference_margin": math.fsum(row["preference_margin"] for row in records) / count,
        "preference_accuracy": {"numerator": positive, "denominator": count, "value": positive / count},
        "zero_margin_pairs": sum(row["preference_margin"] == 0 for row in records),
        "near_zero_margin_pairs": sum(abs(row["preference_margin"]) <= near_zero for row in records),
        "near_zero_absolute_margin_threshold": near_zero,
        "policy_raw_preference_accuracy": {
            "numerator": sum(row["policy_logprob_gap"] > 0 for row in records),
            "denominator": count,
            "value": sum(row["policy_logprob_gap"] > 0 for row in records) / count,
        },
    }


def compare_scores(pairs, initial, final, beta):
    verify_census(pairs, initial, "initial")
    verify_census(pairs, final, "final")
    require(all(row.get("reference_parity_passed") is True for row in initial),
            "Initial direct reference parity must pass for every pair.")
    before, after = summarize(initial, beta), summarize(final, beta)
    return {
        "initial": before, "final": after,
        "mean_loss_change_final_minus_initial": after["mean_dpo_loss"] - before["mean_dpo_loss"],
        "mean_margin_change_final_minus_initial": after["mean_preference_margin"] - before["mean_preference_margin"],
        "pairs_with_lower_final_loss": sum(b["dpo_loss"] < a["dpo_loss"] for a, b in zip(initial, final, strict=True)),
        "pairs_with_higher_final_margin": sum(b["preference_margin"] > a["preference_margin"] for a, b in zip(initial, final, strict=True)),
        "initial_reference_max_absolute_error": max(
            error for row in initial for error in row["reference_absolute_errors"].values()
        ),
    }


def score_stage(stage, config, root, sft_config, sft_root, pairs, references, training, output, info):
    torch = model = tokenizer = None
    try:
        torch, model, tokenizer, device, runtime, _, _ = dpo.load(
            config, root, sft_config, sft_root, pairs, False,
            previous=training if stage == "final" else None,
        )
        require(not any(parameter.requires_grad for parameter in model.parameters()),
                "Diagnostic policies must be frozen.")
        write(output / f"{stage}-runtime.json", runtime)

        def emit(row):
            dpo.sft.curriculum.append(output / f"{stage}-scores.jsonl", row)
            info[stage + "_scored"] += 1

        records = score_pairs(model, pairs, references, config, torch, device, stage, emit)
        verify_census(pairs, records, stage)
        return records
    finally:
        del model, tokenizer
        gc.collect()
        if torch is not None:
            torch.cuda.empty_cache()


def run(config_path, reference, training, reloaded, output):
    config_path, reference, training, reloaded, output = (
        Path(path).resolve() for path in (config_path, reference, training, reloaded, output)
    )
    protected = (config_path.parent, reference, training, reloaded)
    require(not output.exists() and not any(output.is_relative_to(path) for path in protected),
            "Use a fresh diagnostic directory outside all frozen inputs and model runs.")
    config, root, sft_config, sft_root, pairs, _, verified = dpo.preflight(config_path)
    references = dpo.verify_cache(reference, config, pairs)
    _, trained = dpo.sft.verify_reload(training, reloaded, version=dpo.VERSION)
    fresh = read(reloaded / "run.json")
    config_sha, reference_sha = digest(config_path), digest(reference / "manifest.json")
    validate_completed_run(config, pairs, trained, fresh, config_sha, reference_sha)
    output.mkdir(parents=True)
    info = {
        "version": VERSION, "status": "registered", "scope": "train_optimization_only",
        "train_only": True, "final_test_bodies_read": False,
        "generalization_evaluated": False, "model_selection_changed": False,
        "requested_pairs_per_stage": len(pairs), "initial_scored": 0, "final_scored": 0,
        "unique_train_cases": len({pair["case_id"] for pair in pairs}),
        "configuration_sha256": config_sha, "anchor": config["anchor"], "beta": config["training"]["beta"],
        "bindings": {name: binding(path) for name, path in (
            ("configuration", config_path), ("preparation", root / "manifest.json"),
            ("reference", reference / "manifest.json"), ("training", training / "manifest.json"),
            ("reload", reloaded / "manifest.json"),
        )},
        "implementation_sha256": {Path(module.__file__).name: digest(module.__file__)
                                   for module in (dpo, dpo.sft, dpo.objective)},
        "diagnostic_implementation_sha256": digest(__file__),
        "verified_preparation": verified,
        "initial_reference_absolute_tolerance": REFERENCE_ATOL,
        "initial_log2_loss_absolute_tolerance": INITIAL_LOSS_ATOL,
        "definitions": {
            "logprob": "Summed completion log probability, including EOS, excluding prompt; not length normalized.",
            "preference_margin": "beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)).",
            "preference_accuracy": "Fraction of all frozen pairs with strictly positive reference-relative margin. Exact ties count as zero; near-zero counts are reported separately.",
            "policy_raw_preference_accuracy": "Fraction with policy_chosen > policy_rejected, without reference subtraction; different completion lengths can affect this value.",
            "initial": "Directly rescore the original frozen SFT anchor on every pair and require reference-cache parity.",
            "final": "Rescore the verified final DPO training adapter, bound to an independent successful reload, on the identical pairs.",
        },
        "limitations": [
            "These exact pairs trained the adapter. Loss/margin improvement measures fitting of the DPO objective, not held-out editing quality or generalization.",
            "No generation, semantic review, DEV evaluation, local serving or app validation is performed.",
        ],
    }
    write(output / "run-registration.json", info)
    started = time.monotonic()
    try:
        initial = score_stage("initial", config, root, sft_config, sft_root, pairs, references, training, output, info)
        final = score_stage("final", config, root, sft_config, sft_root, pairs, references, training, output, info)
        report = {"version": VERSION, "scope": info["scope"], "generalization_evaluated": False,
                  "interpretation": info["limitations"][0], "definitions": info["definitions"],
                  **compare_scores(pairs, initial, final, info["beta"])}
        write(output / "report.json", report)
        info.update(status="diagnostic_complete", seconds=time.monotonic() - started)
        write(output / "run.json", info)
        dpo.sft.manifest(output, info["status"], VERSION)
        return {"status": info["status"], "output": str(output), **report}
    except BaseException as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error),
                    seconds=time.monotonic() - started)
        write(output / "run.json", info)
        dpo.sft.manifest(output, "failed", VERSION)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "training", "reloaded", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    args["config_path"] = args.pop("config")
    print(json.dumps(run(**args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
