"""Bound train-only DPO from the fixed T1 adapter with one activation graph."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

import prepare_passive_preferences as preparation
import train_passive_curriculum as curriculum
import train_passive_sft as smoke

VERSION = "passive-dpo-v1"
SIDES = ("chosen", "rejected")
CATEGORIES = (
    "pair_id",
    "case_id",
    "source_id",
    "source_group_id",
    "episode_id",
    "language",
    "chosen_action",
)
PHASES = (
    "chosen_scored",
    "rejected_forward",
    "rejected_backward",
    "chosen_forward",
    "chosen_backward",
    "pair_backward_complete",
    "optimizer_committed",
)


def profile_pairs(pairs):
    smoke.require(len(pairs) >= 8, "Eight qualified pairs are needed for the engineering profile.")
    return sorted(
        pairs,
        key=lambda pair: (
            -max(len(pair[side]["input_ids"]) for side in SIDES),
            -max(len(pair[side]["input_ids"]) - pair[side]["completion_start"] for side in SIDES),
            pair["pair_id"],
        ),
    )[:8]


def summed_logprob(model, encoded, torch, device):
    ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    start = encoded["completion_start"]
    logits = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=False,
        logits_to_keep=torch.arange(start - 1, ids.shape[1] - 1, device=device),
        return_dict=True,
    ).logits
    targets = torch.tensor(encoded["labels"][start:], dtype=torch.long, device=device)
    smoke.require(logits.shape[:2] == (1, len(targets)), "Unexpected completion logit dimensions.")
    value = -torch.nn.functional.cross_entropy(logits[0].float(), targets, reduction="sum")
    smoke.require(torch.isfinite(value).item(), "Non-finite summed completion log probability.")
    return value


def dpo_loss(chosen, rejected, reference, beta, torch):
    ref = torch.as_tensor(
        [reference["chosen_logprob"], reference["rejected_logprob"]],
        dtype=chosen.dtype,
        device=chosen.device,
    )
    margin = beta * ((chosen - rejected) - (ref[0] - ref[1]))
    return torch.nn.functional.softplus(-margin), margin


def sequential_backward(
    model, pair, reference, torch, device, settings, divisor, event, record, attempt=lambda _: None
):
    attempt("chosen_scored")
    with torch.no_grad():
        scored = summed_logprob(model, pair["chosen"], torch, device)
    record("chosen_scored", pair, "chosen")
    attempt("rejected_forward")
    rejected = summed_logprob(model, pair["rejected"], torch, device)
    record("rejected_forward", pair, "rejected")
    loss, margin = dpo_loss(scored, rejected, reference, settings["beta"], torch)
    smoke.require(torch.isfinite(loss).item(), "Non-finite DPO loss.")
    coefficient = (-settings["beta"] * torch.sigmoid(-margin.detach())).detach()
    event.update(
        loss=float(loss.detach()),
        preference_margin=float(margin.detach()),
        chosen_logprob=float(scored),
        rejected_logprob=float(rejected.detach()),
        reference_chosen_logprob=reference["chosen_logprob"],
        reference_rejected_logprob=reference["rejected_logprob"],
        chosen_gradient_coefficient=float(coefficient),
    )
    attempt("rejected_backward")
    (loss / divisor).backward()
    record("rejected_backward", pair, "rejected")
    del loss, margin, rejected
    attempt("chosen_forward")
    chosen = summed_logprob(model, pair["chosen"], torch, device)
    record("chosen_forward", pair, "chosen")
    error = float((chosen.detach() - scored).abs())
    event["chosen_recompute_absolute_error"] = error
    smoke.require(
        torch.isclose(
            chosen.detach(),
            scored,
            rtol=settings["chosen_recompute_rtol"],
            atol=settings["chosen_recompute_atol"],
        ).item(),
        "Chosen scoring/recomputation differs; sequential DPO gradient is unsafe.",
    )
    attempt("chosen_backward")
    (coefficient * chosen / divisor).backward()
    record("chosen_backward", pair, "chosen")
    del chosen, scored, coefficient


class Exposure:
    def __init__(self):
        self.updates = 0
        self.optimizer_commit_uncertain = False
        self.attempted_compute_calls = Counter()
        self.phases = {
            phase: {
                "rows": 0,
                "sequence_tokens": 0,
                "completion_tokens": 0,
                "counts": {key: Counter() for key in CATEGORIES},
            }
            for phase in PHASES
        }

    def record(self, phase, pair, side=None):
        values = self.phases[phase]
        values["rows"] += 1
        for name in (side,) if side else SIDES:
            row = pair[name]
            values["sequence_tokens"] += len(row["input_ids"])
            values["completion_tokens"] += len(row["input_ids"]) - row["completion_start"]
        metadata = {**pair["metadata"], "pair_id": pair["pair_id"], "case_id": pair["case_id"]}
        for key in CATEGORIES:
            values["counts"][key][metadata[key]] += 1

    def summary(self):
        return {
            "completed_updates": self.updates,
            "optimizer_commit_uncertain": self.optimizer_commit_uncertain,
            "attempted_compute_calls": dict(sorted(self.attempted_compute_calls.items())),
            "completed_compute_calls": {phase: self.phases[phase]["rows"] for phase in PHASES[:5]},
            "physical_compute_accounting": "Token totals count completed finite forward/backward stages; interrupted operations may perform additional unmeasured partial work.",
            "pending_complete_backward_pairs_not_committed": self.phases["pair_backward_complete"][
                "rows"
            ]
            - self.phases["optimizer_committed"]["rows"],
            "partial_backward_pairs_not_completed": self.phases["rejected_backward"]["rows"]
            - self.phases["pair_backward_complete"]["rows"],
            "actual_policy_forward_sequence_tokens": sum(
                self.phases[name]["sequence_tokens"]
                for name in ("chosen_scored", "rejected_forward", "chosen_forward")
            ),
            "actual_policy_forward_completion_tokens": sum(
                self.phases[name]["completion_tokens"]
                for name in ("chosen_scored", "rejected_forward", "chosen_forward")
            ),
            "actual_policy_backward_sequence_tokens": sum(
                self.phases[name]["sequence_tokens"]
                for name in ("rejected_backward", "chosen_backward")
            ),
            "actual_policy_backward_completion_tokens": sum(
                self.phases[name]["completion_tokens"]
                for name in ("rejected_backward", "chosen_backward")
            ),
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


def train_pairs(
    model,
    pairs,
    references,
    optimizer,
    torch,
    settings,
    device,
    exposure,
    on_pair,
    on_update,
    deadline=lambda: None,
):
    accumulation = settings["gradient_accumulation"]
    smoke.require(
        len(pairs) == settings["pair_count"]
        and math.ceil(len(pairs) / accumulation) == settings["max_updates"]
        and len({pair["pair_id"] for pair in pairs}) == len(pairs)
        and set(references) >= {pair["pair_id"] for pair in pairs},
        "DPO pair order/count/reference budget mismatch.",
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    for offset in range(0, len(pairs), accumulation):
        group = pairs[offset : offset + accumulation]
        losses = []
        for index, pair in enumerate(group, offset):
            deadline()
            event = {
                "pair_index": index,
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "metadata": pair["metadata"],
                "accumulation_divisor": len(group),
                "status": "failed",
                "attempted_phases": [],
                "completed_phases": [],
            }

            def record(phase, current, side):
                exposure.record(phase, current, side)
                event["completed_phases"].append(phase)
                deadline()

            def attempt(phase):
                exposure.attempted_compute_calls[phase] += 1
                event["attempted_phases"].append(phase)

            try:
                sequential_backward(
                    model,
                    pair,
                    references[pair["pair_id"]],
                    torch,
                    device,
                    settings,
                    len(group),
                    event,
                    record,
                    attempt,
                )
                smoke.require(
                    all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                        for parameter in parameters
                    ),
                    "Non-finite DPO gradients; the accumulation is not committed.",
                )
                exposure.record("pair_backward_complete", pair)
                event["status"] = "pair_backward_complete"
                losses.append(event["loss"])
            except BaseException as error:
                event.update(error_type=type(error).__name__, error=str(error))
                raise
            finally:
                on_pair(event)
            deadline()
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                parameters, settings["max_grad_norm"], error_if_nonfinite=True
            )
        )
        smoke.require(math.isfinite(norm) and norm >= 0, "Non-finite DPO gradient norm.")
        deadline()
        exposure.optimizer_commit_uncertain = True
        optimizer.step()
        smoke.require(
            all(torch.isfinite(parameter).all().item() for parameter in parameters),
            "Optimizer produced non-finite DPO adapter tensors.",
        )
        optimizer.zero_grad(set_to_none=True)
        for pair in group:
            exposure.record("optimizer_committed", pair)
        exposure.updates += 1
        exposure.optimizer_commit_uncertain = False
        on_update(
            {
                "update": exposure.updates,
                "committed_pairs": offset + len(group),
                "pair_ids": [pair["pair_id"] for pair in group],
                "microbatches": len(group),
                "mean_pair_loss": sum(losses) / len(losses),
                "pair_losses": losses,
                "gradient_norm_before_clip": norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )


def record_from_manifest(directory, manifest, name):
    path = smoke.child(directory, name)
    smoke.check_file(path, manifest["files_sha256"][name])
    return smoke.read(path)


def manifest_directory(binding, root):
    path, receipt = curriculum.bound(binding, root)
    return path.parent, receipt


def validate_anchor(config, anchor_config, anchor_summary, diagnostics, root):
    train_dir, train_manifest = manifest_directory(
        config["bindings"]["anchor_training_manifest"], root
    )
    reload_dir, reload_manifest = manifest_directory(
        config["bindings"]["anchor_reload_manifest"], root
    )
    smoke.require(
        train_manifest["version"] == reload_manifest["version"] == curriculum.VERSION
        and train_manifest["status"] == "training_complete_fresh_reload_pending"
        and reload_manifest["status"] == "fresh_reload_passed_local_export_pending",
        "DPO anchors only the completed T1 curriculum and its exact fresh reload.",
    )
    training = record_from_manifest(train_dir, train_manifest, "run.json")
    reloaded = record_from_manifest(reload_dir, reload_manifest, "run.json")
    for directory, manifest in ((train_dir, train_manifest), (reload_dir, reload_manifest)):
        smoke.require(
            record_from_manifest(directory, manifest, "configuration.json") == anchor_config,
            "Saved anchor configuration changed.",
        )
    smoke.require(
        training["mode"] == "train"
        and training["training_updates"] == 203
        and training["optimizer_committed_rows"] == 809
        and training["configuration_sha256"] == anchor_summary["configuration_sha256"]
        and training["trainer_sha256"] == config["curriculum_trainer_sha256"]
        and training["preparation_sha256"] == anchor_summary["preparation_sha256"]
        and training["final_test_bodies_read"] == 0
        and reloaded["mode"] == "fresh_reload"
        and reloaded["reloaded_training_updates"] == 203
        and reloaded["configuration_sha256"] == training["configuration_sha256"]
        and reloaded["previous_manifest_sha256"]
        == config["bindings"]["anchor_training_manifest"]["sha256"]
        and reloaded["pid"] != training["pid"]
        and reloaded["optimizer_state_resumed"] is False,
        "DPO must start from the same final T1 adapter, with a separate exact reload.",
    )
    exposure = record_from_manifest(train_dir, train_manifest, "exposure.json")
    committed = exposure["phases"]["optimizer_committed"]
    smoke.require(
        exposure["completed_updates"] == 203
        and exposure["optimizer_commit_uncertain"] is False
        and exposure["pending_backward_rows_not_committed"] == 0
        and committed["rows"] == 809
        and committed["counts"]["case_id"]
        == dict(Counter(anchor_config["training"]["sample_order"]))
        and committed["sequence_tokens"] == anchor_config["training"]["max_total_sequence_tokens"]
        and committed["completion_tokens"]
        == anchor_config["training"]["max_total_completion_tokens"],
        "Anchor committed exposure is not the fixed final T1 stream.",
    )
    parity = record_from_manifest(reload_dir, reload_manifest, "reload-parity.json")
    before = record_from_manifest(train_dir, train_manifest, "smoke-generation.json")
    after = record_from_manifest(reload_dir, reload_manifest, "smoke-generation.json")
    order = [row["case_id"] for row in diagnostics]
    smoke.require(
        parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and [row["case_id"] for row in parity["cases"]] == order
        and all(row["identical"] is True for row in parity["cases"])
        and before == after
        and [row["case_id"] for row in before] == order,
        "Anchor reload lacks exact adapter and fixed eight-output parity.",
    )
    for name in (
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        "adapter-tensors.json",
    ):
        smoke.check_file(smoke.child(train_dir, name), train_manifest["files_sha256"][name])
    adapter = record_from_manifest(train_dir, train_manifest, "adapter/adapter_config.json")
    smoke.require(
        adapter["base_model_name_or_path"] == config["base"]["repository"]
        and adapter["revision"] == config["base"]["revision"]
        and adapter["r"] == adapter["lora_alpha"] == 16
        and adapter["lora_dropout"] == 0,
        "Anchor adapter identity or rank changed.",
    )
    return train_dir, train_manifest


def validate_configuration(config, anchor, prepared, pairs):
    smoke.require(config["version"] == VERSION, "Unknown DPO configuration.")
    for name in ("candidate_id", "base", "tokenizer", "model", "libraries", "generation"):
        smoke.require(config[name] == anchor[name], f"DPO native/runtime identity differs: {name}")
    smoke.require(
        config.get("absent_libraries", []) == anchor.get("absent_libraries", []),
        "DPO runtime absence pins changed.",
    )
    settings = config["training"]
    fixed = {
        "seed": 3407,
        "beta": 0.1,
        "micro_batch_size": 1,
        "gradient_accumulation": 4,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "optimizer": "adamw_torch",
        "scheduler": "constant",
        "rank": 16,
        "alpha": 16,
        "dropout": 0,
        "bias": "none",
        "wall_limit_seconds": 21600,
        "max_sequence_tokens": 4096,
        "max_completion_tokens": 512,
        "loss_implementation": "three_forward_two_backward_dpo",
        "log_probability": "summed_completion_including_eos",
        "normalization": "mean_pair_loss_per_actual_microbatch_group",
        "chosen_recompute_atol": 1e-4,
        "chosen_recompute_rtol": 1e-6,
        "sampling_replacement": False,
        "weights_applied_to_loss": False,
        "training_passes": 1,
        "truncation": False,
        "packing": False,
        "initialization": "exact_final_T1_adapter",
        "reference": "exact_final_T1_adapter",
        "recovery_updates": [25, 50],
    }
    smoke.require(
        all(settings.get(key) == value for key, value in fixed.items()),
        "The preregistered DPO recipe changed.",
    )
    smoke.require(
        type(settings["seed"]) is int
        and prepared["status"] == "qualified_ready"
        and prepared["readiness"]["passed"] is True
        and 32 <= len(pairs) <= 256
        and settings["pair_count"] == prepared["pair_count"] == len(pairs)
        and settings["pair_order"] == prepared["pair_order"] == [pair["pair_id"] for pair in pairs]
        and settings["max_updates"] == prepared["max_updates"] == math.ceil(len(pairs) / 4)
        and settings["final_accumulation"]
        == prepared["final_accumulation"]
        == (len(pairs) % 4 or 4)
        and settings["max_total_sequence_tokens"] == prepared["sequence_tokens"]
        and settings["max_total_completion_tokens"] == prepared["completion_tokens"]
        and prepared["eos_token_id"] == config["generation"]["eos_token_id"],
        "Qualified pair readiness, exact exposure or EOS differs.",
    )
    selected = profile_pairs(pairs)
    smoke.require(
        config["profile"]
        == {
            "pair_count": 8,
            "max_updates": 2,
            "pair_order": [pair["pair_id"] for pair in selected],
            "selection": "largest_sequence_then_completion_then_pair_id",
            "discard_adapter_before_substantive": True,
        },
        "Use the fixed largest eight-pair engineering profile and restart T1 afterward.",
    )


def preflight(config_path):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, smoke.read(config_path)
    for filename, key in (
        (__file__, "trainer_sha256"),
        (preparation.__file__, "preparer_sha256"),
        (curriculum.__file__, "curriculum_trainer_sha256"),
        (smoke.__file__, "smoke_trainer_sha256"),
        (curriculum.preparation.__file__, "curriculum_preparer_sha256"),
        (curriculum.smoke_preparation.__file__, "smoke_preparer_sha256"),
    ):
        smoke.check_file(Path(filename), config[key])
    anchor_path, anchor = curriculum.bound(config["bindings"]["anchor_configuration"], root)
    smoke.require(
        anchor_path.parent == root, "Keep the unchanged anchor config beside the DPO config."
    )
    _, _, _, _, diagnostics, anchor_summary = curriculum.preflight(anchor_path)
    anchor_prep, _ = curriculum.bound(config["bindings"]["anchor_preparation"], root)
    candidate_manifest, _ = curriculum.bound(config["bindings"]["candidate_generation"], root)
    prep_path, prep_manifest = curriculum.bound(config["bindings"]["preparation"], root)
    smoke.require(
        config["bindings"]["anchor_preparation"]["sha256"] == anchor_summary["preparation_sha256"],
        "Preference source preparation differs from the T1 anchor.",
    )
    bundle = preparation.verify_bundle(
        prep_path.parent, anchor_prep.parent, candidate_manifest.parent
    )
    prepared, pairs = bundle["configuration"], bundle["pairs"]
    smoke.require(
        bundle["manifest"] == prep_manifest, "Preference manifest changed during verification."
    )
    validate_configuration(config, anchor, prepared, pairs)
    for name, field in (
        ("anchor_configuration", "anchor_configuration_sha256"),
        ("anchor_training_manifest", "anchor_training_manifest_sha256"),
        ("anchor_reload_manifest", "anchor_reload_manifest_sha256"),
    ):
        smoke.require(
            config["bindings"][name]["sha256"] == prepared[field],
            "Preferences are tied to a different SFT anchor.",
        )
    smoke.require(
        prep_manifest["bindings"]["release_manifest_sha256"]
        == anchor_summary["release_manifest_sha256"]
        and prep_manifest["bindings"]["candidate_manifest_sha256"]
        == config["bindings"]["candidate_generation"]["sha256"],
        "Preference candidates/release changed.",
    )
    protocol_path, protocol = curriculum.bound(config["bindings"]["evaluation_protocol"], root)
    validate_protocol(protocol, config, anchor_summary)
    anchor_dir, anchor_manifest = validate_anchor(config, anchor, anchor_summary, diagnostics, root)
    summary = {
        "version": VERSION,
        "status": "configuration_and_inputs_verified",
        "configuration_sha256": smoke.digest(config_path),
        "trainer_sha256": config["trainer_sha256"],
        "preparation_sha256": smoke.digest(prep_path),
        "evaluation_protocol_sha256": smoke.digest(protocol_path),
        "anchor_configuration_sha256": anchor_summary["configuration_sha256"],
        "anchor_training_manifest_sha256": config["bindings"]["anchor_training_manifest"]["sha256"],
        "anchor_reload_manifest_sha256": config["bindings"]["anchor_reload_manifest"]["sha256"],
        "anchor_adapter_sha256": anchor_manifest["files_sha256"][
            "adapter/adapter_model.safetensors"
        ],
        "candidate_manifest_sha256": config["bindings"]["candidate_generation"]["sha256"],
        "release_manifest_sha256": anchor_summary["release_manifest_sha256"],
        "environment": smoke.environment(config),
        "pair_count": len(pairs),
        "training_updates": 0,
        "sequence_tokens": prepared["sequence_tokens"],
        "completion_tokens": prepared["completion_tokens"],
        "diagnostic_rows": len(diagnostics),
        "final_test_bodies_read": 0,
        "preference_origin": "AI",
        "actual_tokenizer_parity": False,
        "tokenizer_parity_scope": "Static bindings and masks only; actual all-candidate decoding and selected-pair tokenization run before model loading.",
    }
    return config, root, pairs, diagnostics, anchor_dir, summary


def cache_rows(directory, config, verified, pairs):
    directory = Path(directory).resolve()
    receipt = smoke.read(directory / "manifest.json")
    smoke.require(
        receipt["version"] == VERSION and receipt["status"] == "reference_cache_complete",
        "Use a completed exact-T1 reference cache.",
    )
    smoke.require(
        {"run.json", "reference-logps.jsonl"} <= set(receipt["files_sha256"]),
        "Every consumed reference cache payload must be hash-bound.",
    )
    for name, expected in receipt["files_sha256"].items():
        smoke.check_file(smoke.child(directory, name), expected)
    run = smoke.read(directory / "run.json")
    keys = (
        "configuration_sha256",
        "preparation_sha256",
        "anchor_training_manifest_sha256",
        "anchor_adapter_sha256",
        "anchor_configuration_sha256",
    )
    smoke.require(
        all(run[key] == verified[key] for key in keys)
        and run["mode"] == "reference_precompute"
        and run["training_updates"] == 0
        and run["reference_updates"] == 0
        and run["reference_adapter_unchanged"] is True
        and run["actual_tokenizer_parity"] is True
        and run["completed_pairs"] == len(pairs)
        and run["reference_forward_attempts"]
        == run["reference_forward_completions"]
        == 2 * len(pairs)
        and run["reference_forward_sequence_tokens"] == verified["sequence_tokens"]
        and run["reference_forward_completion_tokens"] == verified["completion_tokens"]
        and run["reference_source"] == "exact_final_T1_adapter"
        and run["log_probability"] == "summed_completion_including_eos",
        "The reference cache belongs to another model, policy, data or reduction.",
    )
    rows = [
        json.loads(line) for line in (directory / "reference-logps.jsonl").read_text().splitlines()
    ]
    smoke.require(
        [row["pair_id"] for row in rows] == config["training"]["pair_order"],
        "Reference cache pair order changed.",
    )
    for row, pair in zip(rows, pairs, strict=True):
        smoke.require(row["case_id"] == pair["case_id"], "Reference cache state changed.")
        for side in SIDES:
            value = row[side + "_logprob"]
            smoke.require(
                type(value) in (int, float) and math.isfinite(value) and value <= 0,
                "A cached summed log probability is invalid.",
            )
            smoke.require(
                row[side + "_completion_tokens"]
                == len(pair[side]["input_ids"]) - pair[side]["completion_start"],
                "Reference cache omitted or added completion/EOS tokens.",
            )
    return {row["pair_id"]: row for row in rows}, smoke.digest(directory / "manifest.json"), run


def encoded_sides(pairs):
    return [
        {
            "id": pair["case_id"],
            "messages": pair["messages"],
            "references": [pair[side + "_text"]],
            "primary": pair[side],
        }
        for pair in pairs
        for side in SIDES
    ]


def load_adapter(config, root, pairs, adapter_dir, expected, training):
    smoke.require(
        verify_actual_tokens(config, root) == pairs,
        "Verified preference pairs changed before loading.",
    )
    from peft import PeftModel, get_peft_model_state_dict

    torch, model, tokenizer, device, runtime = smoke.load_runtime(
        config, root, encoded_sides(pairs)
    )
    model = smoke.prepare_reload_base(model, config)
    model = PeftModel.from_pretrained(
        model, adapter_dir, is_trainable=training, local_files_only=True
    )
    actual = smoke.tensor_records(
        get_peft_model_state_dict(model, save_embedding_layers=False), torch
    )
    smoke.require(
        actual == expected, "The loaded adapter differs from the exact bound tensor census."
    )
    smoke.require(
        all(not isinstance(module, torch.nn.Dropout) or module.p == 0 for module in model.modules())
        and getattr(model.config.text_config, "attention_dropout", 0) == 0,
        "Sequential DPO requires zero dropout for chosen recomputation parity.",
    )
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        trainable = {
            name: parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        targets = {name.split(".lora_")[0].removeprefix("base_model.model.") for name in trainable}
        smoke.require(
            len(trainable) == 2 * len(config["model"]["target_modules"])
            and targets == set(config["model"]["target_modules"])
            and all(".lora_A." in name or ".lora_B." in name for name in trainable),
            "Unexpected DPO trainable parameters; reference/base weights must stay frozen.",
        )
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
        trainable = {}
    runtime["actual_rejected_tokenization_verified"] = True
    runtime["initial_adapter_tensors"] = actual
    runtime["trainable_tensors"] = trainable
    return torch, model, tokenizer, device, runtime


def validate_protocol(protocol, config, anchor):
    plan = protocol["training_plan"]
    fixed = {
        "candidate": config["candidate_id"],
        "method": "DPO",
        "preference_origin": "AI",
        "beta": 0.1,
        "learning_rate": 1e-5,
        "optimizer": "adamw_torch",
        "weight_decay": 0.01,
        "scheduler": "constant",
        "seed": 3407,
        "micro_batch_pairs": 1,
        "gradient_accumulation": 4,
        "max_grad_norm": 1.0,
        "epochs": 1,
        "replacement": False,
        "weights_applied_to_loss": False,
        "maximum_pairs": 256,
        "maximum_updates": 64,
        "final_partial_accumulation": "actual_pair_count",
        "max_wall_seconds": 21600,
        "development_during_training": False,
        "checkpoint_selection": "final_registered_update_only",
        "recovery_updates": [25, 50],
    }
    smoke.require(
        protocol["version"] == "passive-dpo-evaluation-v1"
        and all(plan.get(key) == value for key, value in fixed.items())
        and protocol["anchor"]["candidate"] == config["candidate_id"]
        and protocol["anchor"]["sft_configuration_sha256"] == anchor["configuration_sha256"]
        and protocol["anchor"]["sft_protocol_sha256"] == anchor["evaluation_protocol_sha256"]
        and protocol["anchor"]["sft_updates"] == 203
        and protocol["anchor"]["sft_draws"] == 809
        and protocol["reference"]["precompute_before_updates"] is True
        and protocol["reference"]["completion_logprob"] == "sum_including_eos"
        and protocol["reference"]["prompt_loss"] is False
        and protocol["reference"]["adapter_disabled"] is False
        and protocol["reference"]["reference_updates"] == 0
        and protocol["engineering_profile"]["pairs"] == 8
        and protocol["engineering_profile"]["updates"] == 2
        and protocol["engineering_profile"]["fresh_reload_required"] is True
        and protocol["engineering_profile"]["profile_weights_used_for_substantive_training"]
        is False
        and protocol["memory_policy"]["algorithm"]
        == "chosen_no_grad_then_rejected_backward_then_chosen_recomputed_backward"
        and protocol["memory_policy"]["forward_passes_per_pair"] == 3
        and protocol["memory_policy"]["backward_passes_per_pair"] == 2
        and protocol["memory_policy"]["simultaneous_activation_graphs"] == 1
        and protocol["memory_policy"]["dropout"] == 0
        and protocol["memory_policy"]["chosen_recompute_rtol"]
        == config["training"]["chosen_recompute_rtol"]
        and protocol["memory_policy"]["chosen_recompute_atol"]
        == config["training"]["chosen_recompute_atol"]
        and protocol["memory_policy"]["abort_on_recompute_mismatch"] is True
        and protocol["launch_gate"]
        == {
            "minimum_pairs": 32,
            "minimum_source_ids": 8,
            "required_languages": ["en", "ko"],
            "chosen_edit_and_no_action_required": True,
            "actual_native_tokenizer_parity": True,
            "reference_cache_verified": True,
            "engineering_profile_finite_and_reload_passed": True,
        }
        and protocol["evaluation"]["conditions"] == ["sft-full", "dpo-full"]
        and protocol["evaluation"]["requests_per_condition"] == 289
        and protocol["evaluation"]["json_schema_constrained"] is True
        and protocol["learning_signal_policy"]["no_auto_extra_training"] is True
        and protocol["learning_signal_policy"]["deployment_promotion"] is False
        and protocol["final_test_bodies_read"] is False,
        "The fixed-anchor DPO protocol changed.",
    )


def verify_actual_tokens(config, root):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        smoke.child(root, config["tokenizer"]["path"]),
        local_files_only=True,
        trust_remote_code=False,
    )
    paths = {
        name: smoke.child(root, config["bindings"][name]["path"]).parent
        for name in ("preparation", "anchor_preparation", "candidate_generation")
    }
    checked = preparation.verify_bundle(
        paths["preparation"],
        paths["anchor_preparation"],
        paths["candidate_generation"],
        tokenizer=tokenizer,
    )
    smoke.require(
        checked["rejected_tokenization_verified"] is True,
        "Actual candidate decoding and rejected tokenization must be verified.",
    )
    return checked["pairs"]


def manifest(output, status):
    smoke.write(
        output / "manifest.json",
        {
            "version": VERSION,
            "status": status,
            "files_sha256": {
                str(path.relative_to(output)): smoke.digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path != output / "manifest.json"
            },
        },
    )


def snapshot(output, config_path, config, info):
    (output / "configuration.json").write_bytes(Path(config_path).read_bytes())
    for name, filename in (
        ("trainer.py", __file__),
        ("prepare_passive_preferences.py", preparation.__file__),
        ("train_passive_curriculum.py", curriculum.__file__),
        ("train_passive_sft.py", smoke.__file__),
        ("prepare_passive_curriculum.py", curriculum.preparation.__file__),
        ("prepare_passive_sft.py", curriculum.smoke_preparation.__file__),
    ):
        (output / name).write_bytes(Path(filename).read_bytes())
    smoke.write(output / "run.json", info)


def new_output(output, root, *inputs):
    output = Path(output).resolve()
    smoke.require(
        not output.exists()
        and not any(output.is_relative_to(Path(path).resolve()) for path in (root, *inputs)),
        "Use a fresh output outside all immutable input directories.",
    )
    output.mkdir(parents=True)
    return output


def finish(output, info, started, torch):
    info["seconds"] = time.monotonic() - started
    if torch is not None and torch.cuda.is_available():
        info.update(
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        )
    smoke.write(output / "run.json", info)
    manifest(output, info["status"])


def reference(config_path, output):
    config, root, pairs, _, anchor_dir, verified = preflight(config_path)
    smoke.require(
        verified["environment"]["matched"], "Use the pinned runtime before precomputation."
    )
    output = new_output(output, root)
    started, torch = time.monotonic(), None
    info = {
        **verified,
        "mode": "reference_precompute",
        "status": "loading",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "completed_pairs": 0,
        "reference_source": "exact_final_T1_adapter",
        "reference_updates": 0,
        "log_probability": "summed_completion_including_eos",
        "reference_adapter_unchanged": False,
        "actual_tokenizer_parity": False,
        "reference_forward_attempts": 0,
        "reference_forward_completions": 0,
        "reference_forward_sequence_tokens": 0,
        "reference_forward_completion_tokens": 0,
        "wall_limit_seconds": config["training"]["wall_limit_seconds"],
        "scope": "Fixed T1 reference scores before DPO; no policy or reference update.",
    }
    snapshot(output, config_path, config, info)
    try:
        from peft import get_peft_model_state_dict

        expected = smoke.read(anchor_dir / "adapter-tensors.json")
        torch, model, _, device, runtime = load_adapter(
            config, root, pairs, anchor_dir / "adapter", expected, False
        )
        smoke.write(output / "runtime.json", runtime)
        info["actual_tokenizer_parity"] = True
        info["runtime_gpu"] = runtime["gpu"]
        for pair in pairs:
            curriculum.check_deadline(started, config["training"]["wall_limit_seconds"])
            row = {"pair_id": pair["pair_id"], "case_id": pair["case_id"]}
            for side in SIDES:
                info["reference_forward_attempts"] += 1
                event = {
                    "pair_id": pair["pair_id"],
                    "case_id": pair["case_id"],
                    "side": side,
                    "status": "failed",
                    "sequence_tokens": len(pair[side]["input_ids"]),
                    "completion_tokens": len(pair[side]["input_ids"])
                    - pair[side]["completion_start"],
                }
                try:
                    with torch.inference_mode():
                        row[side + "_logprob"] = float(
                            summed_logprob(model, pair[side], torch, device)
                        )
                    row[side + "_completion_tokens"] = event["completion_tokens"]
                    info["reference_forward_completions"] += 1
                    info["reference_forward_sequence_tokens"] += event["sequence_tokens"]
                    info["reference_forward_completion_tokens"] += event["completion_tokens"]
                    event["status"] = "reference_forward_complete"
                except BaseException as error:
                    event.update(error_type=type(error).__name__, error=str(error))
                    raise
                finally:
                    curriculum.append(output / "reference-events.jsonl", event)
                curriculum.check_deadline(started, config["training"]["wall_limit_seconds"])
            curriculum.append(output / "reference-logps.jsonl", row)
            info["completed_pairs"] += 1
            info["status"] = "precomputing_reference"
            smoke.write(output / "run.json", info)
            print(
                json.dumps(
                    {
                        "event": "reference_pair",
                        "pair_id": pair["pair_id"],
                        "completed_pairs": info["completed_pairs"],
                    }
                ),
                flush=True,
            )
        actual = smoke.tensor_records(
            get_peft_model_state_dict(model, save_embedding_layers=False), torch
        )
        smoke.require(actual == expected, "Reference precomputation changed the fixed T1 adapter.")
        info.update(status="reference_cache_complete", reference_adapter_unchanged=True)
    except BaseException as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        finish(output, info, started, torch)
    return info


def verified_run(directory, status):
    directory = Path(directory).resolve()
    receipt = smoke.read(directory / "manifest.json")
    smoke.require(
        receipt["version"] == VERSION and receipt["status"] == status,
        "DPO evidence is incomplete or from another trainer.",
    )
    required = {"run.json", "smoke-generation.json"}
    if status == "training_complete_fresh_reload_pending":
        required |= {
            "exposure.json",
            "adapter-tensors.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
        }
    elif status == "fresh_reload_passed_local_export_pending":
        required.add("reload-parity.json")
    smoke.require(
        required <= set(receipt["files_sha256"]),
        "Every consumed training/reload payload must be hash-bound.",
    )
    for name, expected in receipt["files_sha256"].items():
        smoke.check_file(smoke.child(directory, name), expected)
    return directory, receipt, smoke.read(directory / "run.json")


def validate_profile(profile, profile_reload, config, verified, reference_sha):
    directory, _, run = verified_run(profile, "training_complete_fresh_reload_pending")
    reload_dir, _, reloaded = verified_run(
        profile_reload, "fresh_reload_passed_local_export_pending"
    )
    smoke.require(
        run["mode"] == "train"
        and run["experiment_phase"] == "engineering_profile"
        and run["configuration_sha256"] == verified["configuration_sha256"]
        and run["anchor_adapter_sha256"] == verified["anchor_adapter_sha256"]
        and run["reference_cache_sha256"] == reference_sha
        and run["training_updates"] == 2
        and run["optimizer_committed_pairs"] == 8
        and run["pair_order"] == config["profile"]["pair_order"]
        and run["actual_tokenizer_parity"] is True
        and run["changed_adapter_tensors"]
        and run["pid"] != os.getpid()
        and reloaded["mode"] == "fresh_reload"
        and reloaded["experiment_phase"] == "engineering_profile"
        and reloaded["configuration_sha256"] == verified["configuration_sha256"]
        and reloaded["previous_manifest_sha256"] == smoke.digest(directory / "manifest.json")
        and reloaded["reloaded_training_updates"] == 2
        and reloaded["pid"] != run["pid"],
        "Substantive DPO requires this exact two-update profile and an independent reload.",
    )
    exposure = smoke.read(directory / "exposure.json")
    smoke.require(
        exposure["completed_updates"] == 2
        and exposure["optimizer_commit_uncertain"] is False
        and exposure["pending_complete_backward_pairs_not_committed"] == 0
        and exposure["partial_backward_pairs_not_completed"] == 0
        and exposure["phases"]["optimizer_committed"]["counts"]["pair_id"]
        == dict(Counter(config["profile"]["pair_order"])),
        "Engineering profile did not commit the exact eight pairs.",
    )
    parity = smoke.read(reload_dir / "reload-parity.json")
    before, after = (
        smoke.read(directory / "smoke-generation.json"),
        smoke.read(reload_dir / "smoke-generation.json"),
    )
    smoke.require(
        parity["adapter_tensors_identical"] is True
        and parity["optimizer_state_resumed"] is False
        and len(parity["cases"]) == len({row["case_id"] for row in parity["cases"]}) == 8
        and all(row["identical"] is True for row in parity["cases"])
        and before == after
        and len(before) == 8
        and [row["case_id"] for row in before] == [row["case_id"] for row in parity["cases"]],
        "Engineering profile reload lost exact tensor/output parity.",
    )
    return {
        "profile_manifest_sha256": smoke.digest(directory / "manifest.json"),
        "profile_reload_manifest_sha256": smoke.digest(reload_dir / "manifest.json"),
    }


def save_checkpoint(output, name, model, optimizer, torch, config_sha, exposure, reason):
    smoke.require(
        not exposure.optimizer_commit_uncertain,
        "An interrupted optimizer step cannot be marked as a safe recovery checkpoint.",
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


def run(config_path, reference_path, output, phase, profile=None, profile_reload=None):
    config, root, all_pairs, diagnostics, anchor_dir, verified = preflight(config_path)
    smoke.require(verified["environment"]["matched"], "Use the pinned DPO runtime.")
    references, reference_sha, reference_run = cache_rows(
        reference_path, config, verified, all_pairs
    )
    smoke.require(phase in ("engineering_profile", "substantive"), "Unknown DPO training phase.")
    gates = {}
    if phase == "substantive":
        smoke.require(
            profile is not None and profile_reload is not None,
            "Substantive DPO requires a completed independent engineering profile/reload.",
        )
        gates = validate_profile(profile, profile_reload, config, verified, reference_sha)
    pairs = profile_pairs(all_pairs) if phase == "engineering_profile" else all_pairs
    settings = {
        **config["training"],
        "pair_count": len(pairs),
        "max_updates": math.ceil(len(pairs) / 4),
    }
    output = new_output(
        output, root, reference_path, *([profile, profile_reload] if profile else [])
    )
    started, torch, model, optimizer = time.monotonic(), None, None, None
    exposure = Exposure()
    info = {
        **verified,
        **gates,
        "mode": "train",
        "experiment_phase": phase,
        "status": "loading",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "initialization": "exact_final_T1_adapter",
        "reference_source": "exact_final_T1_adapter",
        "reference_cache_sha256": reference_sha,
        "pair_count": len(pairs),
        "pair_order": [pair["pair_id"] for pair in pairs],
        "optimizer_state_resumed": False,
        "resume_supported": False,
        "actual_tokenizer_parity": False,
        "sequence_tokens": sum(len(pair[side]["input_ids"]) for pair in pairs for side in SIDES),
        "completion_tokens": sum(
            len(pair[side]["input_ids"]) - pair[side]["completion_start"]
            for pair in pairs
            for side in SIDES
        ),
        "wall_limit_seconds": settings["wall_limit_seconds"],
        "wall_limit_origin": "Immediately before runtime/model loading; excludes input hash preflight.",
        "wall_limit_enforcement": "Checked between complete operations and before optimizer steps; in-flight operations/checkpoints may finish after the deadline.",
        "recovery_checkpoints": [],
        "scope": "Fixed same-state AI preference experiment; engineering weights never initialize substantive DPO. No development selection or final-test evaluation.",
    }
    snapshot(output, config_path, config, info)

    def deadline():
        curriculum.check_deadline(started, settings["wall_limit_seconds"])

    try:
        from peft import get_peft_model_state_dict

        initial = smoke.read(anchor_dir / "adapter-tensors.json")
        torch, model, tokenizer, device, runtime = load_adapter(
            config, root, all_pairs, anchor_dir / "adapter", initial, True
        )
        smoke.require(
            runtime["gpu"] == reference_run["runtime_gpu"],
            "Use the same GPU architecture as the frozen NF4 reference cache.",
        )
        smoke.write(output / "runtime.json", runtime)
        smoke.write(
            output / "initialization.json",
            {
                "initialization": "exact_final_T1_adapter",
                "anchor_adapter_sha256": verified["anchor_adapter_sha256"],
                "initial_adapter_tensors": initial,
                "trainable_tensors": runtime["trainable_tensors"],
                "profile_adapter_loaded": False,
            },
        )
        info["actual_tokenizer_parity"] = True
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )
        deadline()

        def record_pair(event):
            curriculum.append(output / "training-pairs.jsonl", event)
            print(
                json.dumps(
                    {
                        "event": "pair",
                        "pair_id": event["pair_id"],
                        "pair_index": event["pair_index"],
                        "status": event["status"],
                        "loss": event.get("loss"),
                        "chosen_recompute_absolute_error": event.get(
                            "chosen_recompute_absolute_error"
                        ),
                    }
                ),
                flush=True,
            )

        def record_update(event):
            info["training_updates"] = exposure.updates
            curriculum.append(output / "training-events.jsonl", event)
            smoke.write(output / "exposure.json", exposure.summary())
            smoke.write(output / "run.json", info)
            if phase == "substantive" and exposure.updates in settings["recovery_updates"]:
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
                    json.dumps({"event": "recovery_checkpoint_complete", "path": path}), flush=True
                )
            print(json.dumps({"event": "optimizer_update", **event}), flush=True)

        info["status"] = "training"
        smoke.write(output / "run.json", info)
        train_pairs(
            model,
            pairs,
            references,
            optimizer,
            torch,
            settings,
            device,
            exposure,
            record_pair,
            record_update,
            deadline,
        )
        deadline()
        final = curriculum.save_training_state(
            output, model, optimizer, torch, verified["configuration_sha256"], exposure
        )
        smoke.require(set(initial) == set(final), "DPO changed the adapter tensor census.")
        changed = [name for name in final if final[name] != initial[name]]
        smoke.require(changed, "DPO did not change any adapter tensor.")
        adapter_config = smoke.read(output / "adapter/adapter_config.json")
        smoke.require(
            adapter_config["base_model_name_or_path"] == config["base"]["repository"]
            and adapter_config["revision"] == config["base"]["revision"],
            "DPO lost native adapter provenance.",
        )
        tokenizer.save_pretrained(output / "tokenizer")

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

        deadline()
        smoke.write(
            output / "smoke-generation.json",
            smoke.generate_smoke(model, tokenizer, diagnostics, config, torch, device, generated),
        )
        smoke.write(output / "exposure.json", exposure.summary())
        info.update(
            status="training_complete_fresh_reload_pending",
            training_updates=exposure.updates,
            optimizer_committed_pairs=exposure.phases["optimizer_committed"]["rows"],
            changed_adapter_tensors=changed,
            unique_cases=len(exposure.phases["optimizer_committed"]["counts"]["case_id"]),
        )
        smoke.require(
            smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            == final,
            "Diagnostic generation changed the DPO adapter.",
        )
    except BaseException as error:
        info.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            training_updates=exposure.updates,
        )
        smoke.write(output / "exposure.json", exposure.summary())
        if model is not None and optimizer is not None and torch is not None:
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
                    "Use only earlier complete recovery evidence; no automatic optimizer resume."
                )
        raise
    finally:
        finish(output, info, started, torch)
    return info


def reload_run(config_path, previous, output):
    config, root, pairs, diagnostics, _, verified = preflight(config_path)
    smoke.require(verified["environment"]["matched"], "Use the pinned fresh-reload runtime.")
    previous, _, saved = verified_run(previous, "training_complete_fresh_reload_pending")
    phase = saved["experiment_phase"]
    selected = profile_pairs(pairs) if phase == "engineering_profile" else pairs
    smoke.require(
        phase in ("engineering_profile", "substantive")
        and saved["mode"] == "train"
        and saved["configuration_sha256"] == verified["configuration_sha256"]
        and saved["pid"] != os.getpid()
        and saved["training_updates"] == math.ceil(len(selected) / 4)
        and saved["optimizer_committed_pairs"] == len(selected)
        and saved["pair_order"] == [pair["pair_id"] for pair in selected],
        "Reload requires this phase's final completed DPO checkpoint in a fresh process.",
    )
    output = new_output(output, root, previous)
    started, torch = time.monotonic(), None
    info = {
        **verified,
        "mode": "fresh_reload",
        "experiment_phase": phase,
        "status": "loading",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "optimizer_state_resumed": False,
        "actual_tokenizer_parity": False,
        "reference_cache_sha256": saved["reference_cache_sha256"],
        "previous_manifest_sha256": smoke.digest(previous / "manifest.json"),
        "scope": "Fresh inference reload; no optimizer state or policy updates.",
    }
    snapshot(output, config_path, config, info)

    def deadline():
        curriculum.check_deadline(started, config["training"]["wall_limit_seconds"])

    try:
        expected = smoke.read(previous / "adapter-tensors.json")
        torch, model, tokenizer, device, runtime = load_adapter(
            config, root, pairs, previous / "adapter", expected, False
        )
        smoke.write(output / "runtime.json", runtime)
        info["actual_tokenizer_parity"] = True
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

        observed = smoke.generate_smoke(
            model, tokenizer, diagnostics, config, torch, device, generated
        )
        smoke.write(output / "smoke-generation.json", observed)
        original = smoke.read(previous / "smoke-generation.json")
        parity = {
            "adapter_tensors_identical": True,
            "optimizer_state_resumed": False,
            "cases": [
                {"case_id": a["case_id"], "identical": a == b}
                for a, b in zip(observed, original, strict=True)
            ],
        }
        smoke.write(output / "reload-parity.json", parity)
        smoke.require(
            len(parity["cases"]) == 8 and all(row["identical"] for row in parity["cases"]),
            "DPO greedy output changed after fresh reload.",
        )
        info.update(
            status="fresh_reload_passed_local_export_pending",
            reloaded_training_updates=saved["training_updates"],
        )
    except BaseException as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        finish(output, info, started, torch)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "reference", "profile", "run", "reload"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name != "preflight":
            command.add_argument("--output", required=True, type=Path)
        if name in ("profile", "run"):
            command.add_argument("--reference", required=True, type=Path)
        if name == "run":
            command.add_argument("--profile", required=True, type=Path)
            command.add_argument("--profile-reload", required=True, type=Path)
        if name == "reload":
            command.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.config)[-1]
    elif args.command == "reference":
        result = reference(args.config, args.output)
    elif args.command == "reload":
        result = reload_run(args.config, args.run, args.output)
    else:
        result = run(
            args.config,
            args.reference,
            args.output,
            "engineering_profile" if args.command == "profile" else "substantive",
            getattr(args, "profile", None),
            getattr(args, "profile_reload", None),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
