"""Gemma seven-action candidates, bound reference cache, DPO and fresh reload."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

import prepare_gemma_actions_v3 as prep
import prepare_passive_actions_v3 as dataset
import train_gemma_actions_v3 as sft
import train_passive_dpo as objective
from grading import transcript_media

VERSION = "gemma-actions-v3-dpo"
SIDES = ("chosen", "rejected")
require, digest, read, write = sft.require, sft.digest, sft.read, sft.write


def candidate_identity(config, training, reloaded):
    receipt, info = sft.verify_reload(training, reloaded)
    require(
        info["configuration_sha256"] == digest(config),
        "Candidate SFT configuration differs.",
    )
    return {
        "anchor_training_manifest_sha256": digest(Path(training) / "manifest.json"),
        "anchor_reload_manifest_sha256": digest(Path(reloaded) / "manifest.json"),
        "anchor_configuration_sha256": digest(config),
        "anchor_adapter_sha256": receipt["files_sha256"][
            "adapter/adapter_model.safetensors"
        ],
    }


def candidates(config_path, training, reloaded, output):
    config, root, bundle, _, _, verified = sft.preflight(config_path)
    identity = candidate_identity(config_path, training, reloaded)
    output = sft.new_output(output, root, training)
    started = time.monotonic()
    info = {
        "version": VERSION,
        "mode": "candidate_generation",
        "status": "loading",
        "pid": os.getpid(),
        "configuration_sha256": digest(config_path),
        **identity,
        "train_only": True,
        "final_test_bodies_read": False,
    }
    sft.gemma.snapshot_configuration(
        config_path, output / "configuration.json", info["configuration_sha256"]
    )
    write(output / "run.json", info)
    try:
        torch, model, tokenizer, device, runtime = sft.gemma.load_runtime(
            config, root, bundle["tokens"][config["format"]]
        )
        model, _, _ = sft.load_adapter(
            model,
            Path(training) / "adapter",
            read(Path(training) / "adapter-tensors.json"),
            config,
            torch,
            False,
        )
        write(output / "runtime.json", runtime)
        by_id = {case["id"]: case for case in bundle["cases"]}
        records = bundle["tokens"][config["format"]]
        unique_rows = [
            prep.flatten(by_id[record["id"]], record, index)
            for index, record in enumerate(records)
        ]
        emitted = []

        def emit(value):
            case = by_id[value["case_id"]]
            result = {
                **value,
                "candidate_id": f"{value['case_id']}:greedy:0",
                "split": "train",
                "format": config["format"],
                "input_sha256": prep.value_hash(
                    prep.formats.contract.wire_snapshot(case["input"])
                ),
                **identity,
                "parsed_action": None,
                "validation_error": None,
            }
            try:
                result["parsed_action"] = prep.formats.parse(
                    value["text"],
                    case["input"],
                    config["format"],
                    finish_reason=value["finish_reason"],
                ).model_dump()
            except ValueError as error:
                result["validation_error"] = str(error)
            sft.curriculum.append(output / "candidates.jsonl", result)
            emitted.append(result)
            print(
                json.dumps(
                    {
                        "event": "candidate",
                        "case_id": value["case_id"],
                        "valid": result["parsed_action"] is not None,
                    }
                ),
                flush=True,
            )

        sft.gemma.generate_smoke(
            model, tokenizer, unique_rows, config, torch, device, emit
        )
        info.update(
            status="candidates_complete",
            candidates=len(emitted),
            seconds=time.monotonic() - started,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        )
        write(output / "run.json", info)
        sft.manifest(output, info["status"], VERSION)
        return info
    except BaseException as error:
        info.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            seconds=time.monotonic() - started,
        )
        write(output / "run.json", info)
        sft.manifest(output, "failed", VERSION)
        raise


def encoded_completion(tokenizer, case, value, format_name):
    text = (
        value
        if isinstance(value, str)
        else prep.formats.serialize(case["input"], value, format_name)
    )
    prefix = prep.formats.prefix_ids(tokenizer, case["input"], format_name)
    completion = tokenizer.encode(text, add_special_tokens=False)
    require(
        text.strip() and not set(completion) & {1, 106, 50},
        "DPO text is empty or contains native terminal tokens.",
    )
    completion += [1]
    require(
        0 < len(prefix) <= prep.formats.INPUT_LIMIT
        and len(completion) <= prep.formats.OUTPUT_LIMIT,
        "DPO completion exceeds the fixed budget; never truncate.",
    )
    return {
        "input_ids": prefix + completion,
        "attention_mask": [1] * (len(prefix) + len(completion)),
        "labels": [-100] * len(prefix) + completion,
        "completion_start": len(prefix),
        "completion_text": text,
        "completion_tokens": len(completion),
        "eos_token_id": 1,
    }


def prepare_pair(pair, case, candidate, tokenizer, config, candidate_sha, *, saved_projection=False):
    require(
        pair["case_id"] == case["id"] == candidate["case_id"]
        and candidate["split"] == case["split"] == "train",
        "DPO requires the same train state on both sides.",
    )
    input_hash = prep.value_hash(prep.formats.contract.wire_snapshot(case["input"]))
    require(
        pair["input_sha256"] == candidate["input_sha256"] == input_hash
        and pair["candidate_id"] == candidate["candidate_id"]
        and pair["candidate_run_sha256"] == candidate_sha,
        "Preference candidate or visible input binding differs.",
    )
    judgment = pair["judgment"]
    require(
        judgment["status"] == "preferred"
        and judgment["actor"] == "ai_generated"
        and judgment["rationale"].strip(),
        "Only explicitly justified AI preferences enter this release; ties and uncertainty are excluded.",
    )
    values = {
        side: pair[side] if side in pair else pair[side + "_text"] for side in SIDES
    }
    candidate_present = any(
        value == candidate["parsed_action"]
        if isinstance(value, dict)
        else value == candidate["text"]
        for value in values.values()
    )
    require(candidate_present, "Pair must include its actual frozen SFT candidate.")
    require(
        isinstance(values["chosen"], dict),
        "The preferred side must be an executable typed action.",
    )
    prep.formats.serialize(case["input"], values["chosen"], config["format"])
    if isinstance(values["rejected"], dict):
        prep.formats.serialize(case["input"], values["rejected"], config["format"])
    if saved_projection:
        from prepare_final_unified_dataset import bound_evaluation_bundle
        bundle = bound_evaluation_bundle(case)
    else:
        bundle = dataset.runtime_bundle(
            case, prep.formats.contract.PassiveContextBuilder(tokenizer)
        )
    require(bundle["input"] == case["input"], "Preference runtime input changed.")
    chosen_after = prep.formats.contract.apply_passive_edit(
        case["document"], bundle, values["chosen"]
    )
    try:
        rejected_action = (
            values["rejected"]
            if isinstance(values["rejected"], dict)
            else prep.formats.parse(
                values["rejected"], case["input"], config["format"]
            ).model_dump()
        )
        rejected_after = prep.formats.contract.apply_passive_edit(
            case["document"], bundle, rejected_action
        )
    except (ValueError, KeyError):
        rejected_after = None
    require(
        rejected_after is None
        or transcript_media(chosen_after) != transcript_media(rejected_after),
        "Equivalent executed outcomes are ties, not preferences.",
    )
    encoded = {
        side: encoded_completion(tokenizer, case, value, config["format"])
        for side, value in values.items()
    }
    require(
        encoded["chosen"]["input_ids"] != encoded["rejected"]["input_ids"],
        "Identical outputs are ties, not preferences.",
    )
    require(
        encoded["chosen"]["input_ids"][: encoded["chosen"]["completion_start"]]
        == encoded["rejected"]["input_ids"][: encoded["rejected"]["completion_start"]],
        "DPO prompt prefixes differ.",
    )
    return {
        "pair_id": pair["pair_id"],
        "case_id": case["id"],
        "input_sha256": input_hash,
        "candidate_id": candidate["candidate_id"],
        "candidate_run_sha256": candidate_sha,
        "judgment": judgment,
        "messages": prep.formats.messages(case["input"], config["format"]),
        "metadata": {
            "source_id": case["source"]["id"],
            "source_group_id": case["source"]["source_group"],
            "episode_id": case.get("episode_id", case["id"]),
            "language": case["language"],
            "chosen_action": values["chosen"]["action"],
        },
        **encoded,
    }


ANCHOR_FILES = {
    "train-manifest.json": "manifest.json",
    "train-run.json": "run.json",
    "configuration.json": "configuration.json",
    "runtime.json": "runtime.json",
    "exposure.json": "exposure.json",
    "adapter-tensors.json": "adapter-tensors.json",
    "adapter/adapter_config.json": "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors": "adapter/adapter_model.safetensors",
}


def prepare_pairs(
    config_path,
    training,
    reloaded,
    candidate_path,
    pair_path,
    output,
    *,
    phase="engineering",
    max_pairs=None,
    beta=0.1,
    learning_rate=5e-6,
    accumulation=4,
    protocol=None,
    saved_projection=False,
):
    from transformers import AutoTokenizer

    config, root, bundle, _, _, _ = sft.preflight(config_path, assets=False)
    identity = candidate_identity(config_path, training, reloaded)
    _, candidate_info = sft.verified_run(
        candidate_path, "candidates_complete", version=VERSION
    )
    require(
        all(candidate_info[key] == value for key, value in identity.items()),
        "Candidate anchor differs.",
    )
    candidate_sha = digest(Path(candidate_path) / "manifest.json")
    raw_pairs = prep.rows(pair_path)
    require(
        raw_pairs and len({pair["pair_id"] for pair in raw_pairs}) == len(raw_pairs),
        "Nonempty unique pair IDs required.",
    )
    candidates_by_id = {
        value["candidate_id"]: value
        for value in prep.rows(Path(candidate_path) / "candidates.jsonl")
    }
    cases = {case["id"]: case for case in bundle["cases"]}
    tokenizer = AutoTokenizer.from_pretrained(
        root / config["tokenizer"]["path"],
        local_files_only=True,
        trust_remote_code=False,
    )
    values = [
        prepare_pair(
            pair,
            cases[pair["case_id"]],
            candidates_by_id[pair["candidate_id"]],
            tokenizer,
            config,
            candidate_sha,
            saved_projection=saved_projection,
        )
        for pair in raw_pairs
    ]
    require(
        len(
            {
                (
                    pair["case_id"],
                    prep.value_hash(pair["chosen"]["input_ids"]),
                    prep.value_hash(pair["rejected"]["input_ids"]),
                )
                for pair in values
            }
        )
        == len(values),
        "Duplicate same-state preference pairs inflate exposure.",
    )
    if max_pairs:
        require(
            phase == "engineering" and max_pairs > 0,
            "Pair slicing is engineering-only.",
        )
        values = sorted(
            values,
            key=lambda pair: (
                -max(len(pair[side]["input_ids"]) for side in SIDES),
                pair["pair_id"],
            ),
        )[:max_pairs]
    require(
        phase in ("engineering", "substantive")
        and (phase == "engineering" or protocol),
        "Substantive DPO requires protocol binding.",
    )
    require(
        math.isfinite(beta)
        and beta > 0
        and math.isfinite(learning_rate)
        and learning_rate > 0
        and accumulation > 0,
        "Invalid DPO hyperparameters.",
    )
    output = sft.new_output(output, root, training)
    shutil.copytree(
        root,
        output / "sft",
        ignore=shutil.ignore_patterns("*.safetensors", "__pycache__"),
    )
    anchor = output / "anchor"
    anchor.mkdir()
    for destination, source in ANCHOR_FILES.items():
        target = anchor / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(training) / source, target)
    for source in ("manifest.json", "run.json", "reload-parity.json"):
        shutil.copyfile(Path(reloaded) / source, anchor / ("reload-" + source))
    if protocol:
        shutil.copyfile(protocol, output / "protocol.json")
    preparation_sources = prep.source_closure([Path(__file__).name])
    (output / "preparation-source").mkdir()
    for name in preparation_sources:
        shutil.copyfile(
            Path(__file__).with_name(name), output / "preparation-source" / name
        )
    for path in Path(__file__).parent.glob("prompt*.txt"):
        shutil.copyfile(path, output / "preparation-source" / path.name)
    prep.write_rows(output / "pairs.jsonl.gz", values)
    prep.write(
        output / "preference-input-binding.json",
        {
            "judgments_sha256": digest(pair_path),
            "candidate_manifest_sha256": candidate_sha,
            "original_pairs": len(raw_pairs),
            "prepared_pairs": len(values),
        },
    )
    settings = {
        **config["training"],
        "learning_rate": learning_rate,
        "beta": beta,
        "gradient_accumulation": accumulation,
        "pair_count": len(values),
        "max_updates": math.ceil(len(values) / accumulation),
        "pair_order": [pair["pair_id"] for pair in values],
        "chosen_recompute_rtol": 1e-6,
        "chosen_recompute_atol": 1e-4,
        "fresh_native_initialization": False,
        "initialization": "frozen_sft_adapter_fresh_optimizer",
    }
    for key in (
        "sample_order",
        "draw_count",
        "max_total_sequence_tokens",
        "max_total_completion_tokens",
        "action_exposures",
        "language_exposures",
        "unique_cases",
        "passes",
        "replacement",
        "final_update_rows",
    ):
        settings.pop(key, None)
    result = {
        **copy.deepcopy(config),
        "version": VERSION,
        "phase": phase,
        "training": settings,
        "sft_config_path": f"sft/config-{config['format']}.json",
        "sft_preparation_sha256": digest(root / "manifest.json"),
        "pairs_sha256": digest(output / "pairs.jsonl.gz"),
        "protocol_sha256": digest(output / "protocol.json") if protocol else None,
        "preparation_sources_sha256": {
            name: digest(output / "preparation-source" / name)
            for name in preparation_sources
        },
        "anchor": identity,
        "anchor_files_sha256": {
            str(path.relative_to(anchor)): digest(path)
            for path in sorted(anchor.rglob("*"))
            if path.is_file()
        },
    }
    if saved_projection:
        result["saved_projection"] = True
    prep.write(output / "configuration.json", result)
    sft.manifest(output, "dpo_prepared_train_only", VERSION)
    preflight(output / "configuration.json", assets=False)
    return {
        "status": "dpo_prepared_train_only",
        "pairs": len(values),
        "configuration_sha256": digest(output / "configuration.json"),
    }


def validate_pair_tokens(pair):
    prefixes = []
    for side in SIDES:
        row = pair[side]
        ids, start = row["input_ids"], row["completion_start"]
        require(
            type(start) is int
            and 0 < start <= prep.formats.INPUT_LIMIT
            and start < len(ids) <= prep.formats.CONTEXT_LIMIT
            and len(ids) - start <= prep.formats.OUTPUT_LIMIT
            and all(type(token) is int and 0 <= token < 262144 for token in ids)
            and row["attention_mask"] == [1] * len(ids)
            and row["labels"] == [-100] * start + ids[start:]
            and ids[-1] == row["eos_token_id"] == 1
            and not set(ids[start:-1]) & {1, 106, 50}
            and row["completion_tokens"] == len(ids) - start,
            "DPO token/masking/EOS binding differs.",
        )
        prefixes.append(ids[:start])
    require(
        prefixes[0] == prefixes[1]
        and pair["chosen"]["input_ids"] != pair["rejected"]["input_ids"],
        "DPO same-state/tie constraint failed.",
    )


def preflight(config_path, *, assets=True):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, read(config_path)
    receipt = read(root / "manifest.json")
    require(
        config["version"] == receipt["version"] == VERSION
        and receipt["status"] == "dpo_prepared_train_only",
        "Unknown DPO preparation.",
    )
    for name, expected in receipt["files_sha256"].items():
        sft.smoke.check_file(sft.smoke.child(root, name), expected)
    sft_config, sft_root, bundle, _, diagnostics, verified = sft.preflight(
        sft.smoke.child(root, config["sft_config_path"]), assets=assets
    )
    require(
        digest(sft_root / "manifest.json") == config["sft_preparation_sha256"]
        and digest(root / "pairs.jsonl.gz") == config["pairs_sha256"]
        and config["anchor"]["anchor_configuration_sha256"]
        == verified["configuration_sha256"],
        "DPO SFT input binding differs.",
    )
    for key in (
        "base",
        "tokenizer",
        "model",
        "generation",
        "libraries",
        "absent_libraries",
        "format",
        "contract_version",
        "format_version",
    ):
        require(
            config[key] == sft_config[key],
            f"DPO native/format/runtime identity differs: {key}",
        )
    anchor = root / "anchor"
    for name, expected in config["anchor_files_sha256"].items():
        sft.smoke.check_file(sft.smoke.child(anchor, name), expected)
    original = read(anchor / "train-manifest.json")
    require(
        digest(anchor / "train-manifest.json")
        == config["anchor"]["anchor_training_manifest_sha256"]
        and digest(anchor / "reload-manifest.json")
        == config["anchor"]["anchor_reload_manifest_sha256"]
        and original["status"] == sft.TRAINED
        and original["version"] == sft.VERSION,
        "Frozen reference is not a verified v3 SFT anchor.",
    )
    for destination, source in ANCHOR_FILES.items():
        if source != "manifest.json":
            require(
                digest(anchor / destination) == original["files_sha256"][source],
                "Anchor evidence/adapter drift.",
            )
    reload_manifest, reload_info = (
        read(anchor / "reload-manifest.json"),
        read(anchor / "reload-run.json"),
    )
    for name in ("run.json", "reload-parity.json"):
        require(
            digest(anchor / ("reload-" + name))
            == reload_manifest["files_sha256"][name],
            "Anchor reload evidence drift.",
        )
    require(
        reload_info["status"] == sft.RELOADED
        and reload_info["previous_manifest_sha256"]
        == digest(anchor / "train-manifest.json")
        and reload_info["pid"] != read(anchor / "train-run.json")["pid"]
        and all(
            value["identical"]
            for value in read(anchor / "reload-reload-parity.json")["cases"]
        ),
        "SFT fresh reload was not verified.",
    )
    require(
        digest(anchor / "adapter/adapter_model.safetensors")
        == config["anchor"]["anchor_adapter_sha256"],
        "Reference adapter changed.",
    )
    if config["protocol_sha256"]:
        sft.smoke.check_file(root / "protocol.json", config["protocol_sha256"])
        require(
            read(root / "protocol.json").get("final_test_bodies_read") is False,
            "DPO must exclude final-test access.",
        )
    require(
        config["phase"] in ("engineering", "substantive")
        and (config["phase"] == "engineering" or config["protocol_sha256"]),
        "Missing DPO protocol.",
    )
    pairs = prep.rows(root / "pairs.jsonl.gz")
    settings = config["training"]
    require(
        pairs
        and len({pair["pair_id"] for pair in pairs}) == len(pairs)
        and [pair["pair_id"] for pair in pairs] == settings["pair_order"]
        and len(pairs) == settings["pair_count"]
        and settings["gradient_accumulation"] > 0
        and math.ceil(len(pairs) / settings["gradient_accumulation"])
        == settings["max_updates"]
        and settings["dropout"] == 0
        and settings["beta"] > 0
        and math.isfinite(settings["beta"]),
        "DPO pair/update settings differ.",
    )
    cases = {case["id"]: case for case in bundle["cases"]}
    for pair in pairs:
        validate_pair_tokens(pair)
        case = cases[pair["case_id"]]
        if config.get("saved_projection") is True:
            from prepare_final_unified_dataset import bound_evaluation_bundle
            bound_evaluation_bundle(case)
        require(
            pair["messages"] == prep.formats.messages(case["input"], config["format"])
            and pair["input_sha256"]
            == prep.value_hash(prep.formats.contract.wire_snapshot(case["input"])),
            "DPO visible input changed.",
        )
    return (
        config,
        root,
        sft_config,
        sft_root,
        pairs,
        diagnostics,
        {
            **{
                key: value
                for key, value in verified.items()
                if key not in ("train_rows", "draws")
            },
            "version": VERSION,
            "configuration_sha256": digest(config_path),
            "anchor_sft_train_rows": verified["train_rows"],
            "anchor_sft_draws": verified["draws"],
            "planned_preference_pairs": len(pairs),
        },
    )


def encoded_sides(pairs):
    return [
        {
            "messages": pair["messages"],
            "references": [pair[side]["completion_text"]],
            "primary": pair[side],
        }
        for pair in pairs
        for side in SIDES
    ]


def pair_binding(pair):
    return prep.value_hash(
        {
            key: pair[key]
            for key in ("pair_id", "case_id", "input_sha256", "messages", *SIDES)
        }
    )


def cache_binding(config, pairs):
    return {
        "configuration_sha256": prep.value_hash(config),
        "anchor": config["anchor"],
        "pairs_sha256": config["pairs_sha256"],
        "format_version": config["format_version"],
        "pair_bindings": {pair["pair_id"]: pair_binding(pair) for pair in pairs},
    }


def verify_cache(directory, config, pairs):
    _, info = sft.verified_run(directory, "reference_cache_complete", version=VERSION)
    require(
        info["binding"] == cache_binding(config, pairs),
        "Reference cache belongs to different inputs, outputs or SFT anchor.",
    )
    records = prep.rows(Path(directory) / "reference.jsonl")
    require(
        len(records) == len(pairs)
        and len({row["pair_id"] for row in records}) == len(pairs),
        "Reference row count or IDs differ.",
    )
    by_id = {row["pair_id"]: row for row in records}
    for pair in pairs:
        row = by_id[pair["pair_id"]]
        require(
            row["pair_binding"] == pair_binding(pair)
            and all(
                type(row[side + "_logprob"]) in (int, float)
                and math.isfinite(row[side + "_logprob"])
                and row[side + "_logprob"] <= 0
                for side in SIDES
            ),
            "Invalid or stale completion reference probability.",
        )
    return by_id


def load(config, root, sft_config, sft_root, pairs, training, previous=None):
    torch, model, tokenizer, device, runtime = sft.gemma.load_runtime(
        sft_config, sft_root, encoded_sides(pairs)
    )
    adapter_root = Path(previous) if previous else root / "anchor"
    expected = read(adapter_root / "adapter-tensors.json")
    sft.verify_adapter_serialization(adapter_root / "adapter", expected, torch)
    model, trainable, initial = sft.load_adapter(
        model, adapter_root / "adapter", expected, config, torch, training
    )
    runtime.update(
        after_adapter_parameter_census=sft.gemma.parameter_census(model),
        initial_adapter_tensors=initial,
    )
    return torch, model, tokenizer, device, runtime, trainable, initial


def reference(config_path, output):
    config, root, sft_config, sft_root, pairs, _, verified = preflight(config_path)
    output = sft.new_output(output, root)
    started = time.monotonic()
    info = {
        **verified,
        "mode": "reference",
        "status": "loading",
        "pid": os.getpid(),
        "binding": cache_binding(config, pairs),
    }
    sft.gemma.snapshot_configuration(
        config_path, output / "configuration.json", verified["configuration_sha256"]
    )
    write(output / "run.json", info)
    try:
        torch, model, _, device, runtime, _, _ = load(
            config, root, sft_config, sft_root, pairs, False
        )
        require(
            not any(parameter.requires_grad for parameter in model.parameters()),
            "Reference model must be frozen.",
        )
        write(output / "runtime.json", runtime)
        records = []
        with torch.no_grad():
            for pair in pairs:
                record = {
                    "pair_id": pair["pair_id"],
                    "pair_binding": pair_binding(pair),
                }
                for side in SIDES:
                    record[side + "_logprob"] = float(
                        objective.summed_logprob(model, pair[side], torch, device)
                    )
                sft.curriculum.append(output / "reference.jsonl", record)
                records.append(record)
        info.update(
            status="reference_cache_complete",
            pairs=len(records),
            seconds=time.monotonic() - started,
            scored_sequence_tokens=sum(
                len(pair[side]["input_ids"]) for pair in pairs for side in SIDES
            ),
            scored_completion_tokens=sum(
                len(pair[side]["input_ids"]) - pair[side]["completion_start"]
                for pair in pairs
                for side in SIDES
            ),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        )
        write(output / "run.json", info)
        sft.manifest(output, info["status"], VERSION)
        verify_cache(output, config, pairs)
        return info
    except BaseException as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error))
        write(output / "run.json", info)
        sft.manifest(output, "failed", VERSION)
        raise


def check_reference_direct(model, pairs, references, torch, device, beta=0.1):
    results = []
    with torch.no_grad():
        for pair in sorted(pairs, key=lambda value: value["pair_id"])[
            : min(2, len(pairs))
        ]:
            direct = {
                side: objective.summed_logprob(model, pair[side], torch, device)
                for side in SIDES
            }
            reference = references[pair["pair_id"]]
            errors = {
                side: abs(float(direct[side]) - reference[side + "_logprob"])
                for side in SIDES
            }
            require(
                all(error <= 1e-4 for error in errors.values()),
                "Loaded anchor's direct probabilities differ from reference cache.",
            )
            loss, margin = objective.dpo_loss(
                direct["chosen"], direct["rejected"], reference, beta, torch
            )
            require(
                abs(float(loss) - math.log(2)) < 1e-5,
                "Initial policy/reference DPO loss differs from log(2).",
            )
            results.append(
                {
                    "pair_id": pair["pair_id"],
                    "absolute_errors": errors,
                    "loss": float(loss),
                    "margin": float(margin),
                }
            )
    return results


def run(config_path, reference_path, output, previous=None):
    config, root, sft_config, sft_root, pairs, diagnostics, verified = preflight(
        config_path
    )
    references = verify_cache(reference_path, config, pairs)
    previous = Path(previous).resolve() if previous else None
    if previous:
        _, previous_info = sft.verified_run(previous, version=VERSION)
        require(
            previous_info["configuration_sha256"] == verified["configuration_sha256"]
            and previous_info["pid"] != os.getpid()
            and previous_info["reference_manifest_sha256"]
            == digest(Path(reference_path) / "manifest.json")
            and previous_info["training_updates"] == config["training"]["max_updates"],
            "DPO reload process/input binding differs.",
        )
    output = sft.new_output(output, root, previous)
    started, exposure = time.monotonic(), objective.Exposure()
    info = {
        **verified,
        "mode": "fresh_reload" if previous else "train",
        "status": "loading",
        "pid": os.getpid(),
        "training_updates": 0,
        "reference_manifest_sha256": digest(Path(reference_path) / "manifest.json"),
        "optimizer_state_resumed": False,
        "initialization": "frozen_sft_adapter_fresh_optimizer",
    }
    sft.gemma.snapshot_configuration(
        config_path, output / "configuration.json", verified["configuration_sha256"]
    )
    write(output / "run.json", info)

    def deadline():
        sft.curriculum.check_deadline(started, config["training"]["wall_limit_seconds"])

    try:
        torch, model, tokenizer, device, runtime, _, initial = load(
            config, root, sft_config, sft_root, pairs, not previous, previous
        )
        write(output / "runtime.json", runtime)
        deadline()
        if not previous:
            write(
                output / "reference-direct-parity.json",
                check_reference_direct(
                    model, pairs, references, torch, device, config["training"]["beta"]
                ),
            )
            optimizer = torch.optim.AdamW(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                lr=config["training"]["learning_rate"],
                weight_decay=config["training"]["weight_decay"],
            )
            require(not optimizer.state, "DPO starts with fresh optimizer moments.")

            def pair_event(event):
                sft.curriculum.append(
                    output / "pair-events.jsonl",
                    {**event, "elapsed_seconds": time.monotonic() - started},
                )

            def update_event(event):
                require(
                    event["gradient_norm_before_clip"] > 0,
                    "DPO has zero policy gradient.",
                )
                sft.curriculum.append(output / "training-events.jsonl", event)
                write(output / "exposure.json", exposure.summary())
                info.update(status="training", training_updates=exposure.updates)
                write(output / "run.json", info)
                if (
                    exposure.updates % config["training"]["checkpoint_every_updates"]
                    == 0
                ):
                    sft.save_adapter_checkpoint(
                        output,
                        model,
                        torch,
                        verified["configuration_sha256"],
                        exposure,
                        "registered_periodic_dpo_checkpoint",
                    )
                print(json.dumps({"event": "dpo_update", **event}), flush=True)

            objective.train_pairs(
                model,
                pairs,
                references,
                optimizer,
                torch,
                config["training"],
                device,
                exposure,
                pair_event,
                update_event,
                deadline,
            )
            final = sft.curriculum.save_training_state(
                output,
                model,
                optimizer,
                torch,
                verified["configuration_sha256"],
                exposure,
            )
            changed = [name for name in final if final[name] != initial[name]]
            require(
                set(initial) == set(final) and changed,
                "DPO did not change the exact selected SFT adapter tensors.",
            )
            write(
                output / "adapter-serialization.json",
                sft.verify_adapter_serialization(output / "adapter", final, torch),
            )
            tokenizer.save_pretrained(output / "tokenizer")
            info.update(
                status=sft.TRAINED,
                training_updates=exposure.updates,
                changed_adapter_tensors=changed,
                optimizer_committed_rows=exposure.phases["optimizer_committed"]["rows"],
            )
        results = sft.gemma.generate_smoke(
            model,
            tokenizer,
            diagnostics,
            config,
            torch,
            device,
            lambda event: sft.curriculum.append(
                output / "generation-events.jsonl", event
            ),
        )
        write(output / "smoke-generation.json", results)
        if previous:
            expected = read(previous / "smoke-generation.json")
            parity = {
                "adapter_tensors_identical": True,
                "optimizer_state_resumed": False,
                "cases": [
                    {"case_id": actual["case_id"], "identical": actual == old}
                    for actual, old in zip(results, expected, strict=True)
                ],
            }
            write(output / "reload-parity.json", parity)
            require(
                all(row["identical"] for row in parity["cases"]),
                "DPO independent reload generation differs.",
            )
            info.update(
                status=sft.RELOADED,
                previous_manifest_sha256=digest(previous / "manifest.json"),
                reloaded_training_updates=config["training"]["max_updates"],
            )
        write(output / "exposure.json", exposure.summary())
        info.update(
            seconds=time.monotonic() - started,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        )
        write(output / "run.json", info)
        sft.manifest(output, info["status"], VERSION)
        return info
    except BaseException as error:
        info.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            seconds=time.monotonic() - started,
            training_updates=exposure.updates,
        )
        write(output / "exposure.json", exposure.summary())
        write(output / "run.json", info)
        sft.manifest(output, "failed", VERSION)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("candidates")
    for name in ("config", "training", "reloaded", "output"):
        c.add_argument("--" + name, type=Path, required=True)
    p = commands.add_parser("prepare-pairs")
    for name in ("config", "training", "reloaded", "candidates", "pairs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--phase", choices=("engineering", "substantive"), default="engineering"
    )
    p.add_argument("--max-pairs", type=int)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--protocol", type=Path)
    for name in ("preflight", "reference", "run", "reload"):
        p = commands.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name == "preflight":
            p.add_argument("--metadata-only", action="store_true")
        else:
            p.add_argument("--output", type=Path, required=True)
        if name in ("run", "reload"):
            p.add_argument("--reference", type=Path, required=True)
        if name == "reload":
            p.add_argument("--run", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["config_path"] = args.pop("config")
    if command == "preflight":
        result = preflight(args["config_path"], assets=not args["metadata_only"])[-1]
    elif command == "prepare-pairs":
        args["candidate_path"], args["pair_path"] = (
            args.pop("candidates"),
            args.pop("pairs"),
        )
        result = prepare_pairs(**args)
    elif command in ("run", "reload"):
        args["reference_path"] = args.pop("reference")
        if "run" in args:
            args["previous"] = args.pop("run")
        result = run(**args)
    else:
        result = globals()[command](**args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
