"""Bound B002 DPO over its original candidates and the frozen 22-state supplement.

The original SFT preparation, source closure and DPO module remain immutable.
Reference/train/reload orchestration is adapted from train_gemma_actions_v3_dpo;
the objective, strict pair preparation, cache binding and adapter loader are reused.
This entry point has a separate version and revalidates both input bundles.
"""

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

import generate_gemma_actions_v3_supplement as supplement
import train_gemma_actions_v3_dpo as dpo

VERSION = "gemma-actions-v3-dpo-supplemented"
PROTOCOL_VERSION = VERSION + "-protocol-1"
PREPARED = "dpo_prepared_train_only"
ORIGINAL_CANDIDATE_SHA256 = "4844e5b753c6ca1a59fdabbabe57545a4e7e77c776ab562ee8fc9e2a620a805d"
ORIGINAL_PREFERENCES_SHA256 = "5a660d4a3d32fcec67462963154584b217f753feb054693a5b682d59f336db17"
FIXED_TRAINING = {
    "beta": 0.1,
    "learning_rate": 5e-6,
    "passes": 1,
    "gradient_accumulation": 4,
}
sft, prep, objective, SIDES = dpo.sft, dpo.prep, dpo.objective, dpo.SIDES
require, digest, read, write = dpo.require, dpo.digest, dpo.read, dpo.write


def load_tokenizer(config, root):
    from transformers import AutoTokenizer

    directory = sft.smoke.child(root, config["tokenizer"]["path"])
    for name, expected in config["tokenizer"]["files_sha256"].items():
        sft.smoke.check_file(sft.smoke.child(directory, name), expected)
    return AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )


def verify_original_candidates(path, config, bundle, identity, tokenizer):
    path = Path(path).resolve()
    require(
        digest(path / "manifest.json") == ORIGINAL_CANDIDATE_SHA256,
        "The original 848-candidate run must remain unchanged.",
    )
    receipt, info = sft.verified_run(
        path, "candidates_complete", version=dpo.VERSION
    )
    require(
        info["mode"] == "candidate_generation"
        and info["train_only"] is True
        and info["final_test_bodies_read"] is False
        and all(info[key] == value for key, value in identity.items())
        and info["configuration_sha256"] == identity["anchor_configuration_sha256"],
        "Original candidate anchor or scope differs.",
    )
    candidates = prep.rows(path / "candidates.jsonl")
    require(
        len(candidates) == len(bundle["cases"]) == info["candidates"] == 848
        and len({row["candidate_id"] for row in candidates}) == len(candidates)
        and [row["case_id"] for row in candidates]
        == [case["id"] for case in bundle["cases"]],
        "Original candidate IDs, count or order differ.",
    )
    for case, record, candidate in zip(
        bundle["cases"], bundle["tokens"]["json"], candidates, strict=True
    ):
        primary = record["primary"]
        prefix = primary["input_ids"][: primary["completion_start"]]
        output_ids = candidate["output_ids"]
        terminal = output_ids[-1] if output_ids else None
        require(
            candidate["candidate_id"] == f"{case['id']}:greedy:0"
            and candidate["split"] == case["split"] == "train"
            and candidate["format"] == "json"
            and all(candidate[key] == value for key, value in identity.items())
            and candidate["input_sha256"]
            == prep.value_hash(prep.formats.contract.wire_snapshot(case["input"]))
            and prefix == prep.formats.prefix_ids(tokenizer, case["input"], "json")
            and candidate["prompt_ids_sha256"] == prep.value_hash(prefix)
            and 0 < len(output_ids) <= config["generation"]["max_new_tokens"]
            and all(type(token) is int and 0 <= token < 262144 for token in output_ids)
            and not set(output_ids[:-1]) & set(config["generation"]["eos_token_id"])
            and candidate["terminal_token_id"] == terminal
            and candidate["finish_reason"]
            == ("stop" if terminal in config["generation"]["eos_token_id"] else "length")
            and candidate["text"] == tokenizer.decode(output_ids, skip_special_tokens=True)
            and candidate["text_raw"] == tokenizer.decode(output_ids, skip_special_tokens=False),
            "Original candidate prompt, raw output or input binding differs.",
        )
        parsed, error = None, None
        try:
            parsed = prep.formats.parse(
                candidate["text"], case["input"], "json",
                finish_reason=candidate["finish_reason"],
            ).model_dump()
        except ValueError as caught:
            error = str(caught)
        require(
            candidate["parsed_action"] == parsed
            and bool(candidate["validation_error"]) == bool(error),
            "Original candidate validation differs.",
        )
    return {"root": path, "manifest": receipt, "info": info, "candidates": candidates,
            "identity": identity, "manifest_sha256": ORIGINAL_CANDIDATE_SHA256}


def index_run(run, cases, input_binding):
    candidates = run["candidates"]
    require(
        len({case["id"] for case in cases}) == len(cases)
        and len({row["candidate_id"] for row in candidates}) == len(candidates)
        and [row["case_id"] for row in candidates] == [case["id"] for case in cases],
        "Duplicate candidate IDs or candidate/input roster mismatch.",
    )
    return {
        **run,
        "input_bundle": input_binding,
        "cases_by_id": {case["id"]: case for case in cases},
        "candidates_by_id": {row["candidate_id"]: row for row in candidates},
    }


def candidate_registry(config, root, bundle, identity, candidate_path,
                       supplement_candidate_path, input_path, tokenizer):
    inputs = supplement.verify_supplement(input_path, identity)
    original = verify_original_candidates(candidate_path, config, bundle, identity, tokenizer)
    extra = supplement.verify_candidate_run(
        supplement_candidate_path, identity, input_path=inputs["root"], tokenizer=tokenizer
    )
    require(
        extra["manifest_sha256"] != original["manifest_sha256"]
        and extra["supplement"]["bindings"] == inputs["bindings"]
        and not {case["id"] for case in bundle["cases"]}
        & {case["id"] for case in inputs["cases"]}
        and not {
            prep.value_hash(prep.formats.contract.wire_snapshot(case["input"]))
            for case in bundle["cases"]
        } & {
            prep.value_hash(prep.formats.contract.wire_snapshot(case["input"]))
            for case in inputs["cases"]
        },
        "Supplement and original train states must remain separate and nonoverlapping.",
    )
    original_binding = {
        "kind": "original_sft",
        "manifest_sha256": digest(root / "manifest.json"),
        "configuration_sha256": identity["anchor_configuration_sha256"],
        "cases_sha256": bundle["receipt"]["files_sha256"]["train-cases.jsonl"],
    }
    runs = [
        index_run(original, bundle["cases"], original_binding),
        index_run(extra, inputs["cases"], {"kind": "supplement", **inputs["bindings"]}),
    ]
    return {run["manifest_sha256"]: run for run in runs}, inputs


def read_preferences(paths, registry):
    require(len(paths) == 2, "Supply original and supplement preference files in that order.")
    run_order = list(registry)
    require(
        run_order[0] == ORIGINAL_CANDIDATE_SHA256
        and digest(paths[0]) == ORIGINAL_PREFERENCES_SHA256,
        "Preserve the original 51 preference judgments and their byte order.",
    )
    inputs, raw = [], []
    for index, (path, candidate_sha) in enumerate(zip(paths, run_order, strict=True)):
        values = prep.rows(path)
        require(
            (index != 0 or len(values) == 51)
            and all(pair["candidate_run_sha256"] == candidate_sha for pair in values),
            "A judgment file is bound to the wrong candidate run.",
        )
        entry = {
            "path": f"preference-inputs/{index:03d}.jsonl" + (".gz" if Path(path).suffix == ".gz" else ""),
            "sha256": digest(path), "pairs": len(values),
            "candidate_manifest_sha256": candidate_sha,
        }
        inputs.append(entry)
        raw.extend((pair, entry["sha256"]) for pair in values)
    require(
        raw and len({pair["pair_id"] for pair, _ in raw}) == len(raw),
        "Nonempty globally unique pair IDs required.",
    )
    return inputs, raw


def prepare_preferences(raw, registry, tokenizer, config):
    values = []
    for pair, judgment_sha in raw:
        candidate_sha = pair["candidate_run_sha256"]
        require(candidate_sha in registry, "Preference refers to an unregistered candidate run.")
        run = registry[candidate_sha]
        require(
            pair["case_id"] in run["cases_by_id"]
            and pair["candidate_id"] in run["candidates_by_id"],
            "Preference state/candidate is not in its bound run's input roster.",
        )
        case, candidate = run["cases_by_id"][pair["case_id"]], run["candidates_by_id"][pair["candidate_id"]]
        value = dpo.prepare_pair(pair, case, candidate, tokenizer, config, candidate_sha)
        rejected = pair.get("rejected")
        if not isinstance(rejected, dict):
            try:
                rejected = prep.formats.parse(
                    rejected if rejected is not None else pair["rejected_text"], case["input"], "json"
                ).model_dump()
            except ValueError:
                rejected = None
        value.update(input_bundle=run["input_bundle"], judgments_sha256=judgment_sha)
        chosen_preserve = value["metadata"]["chosen_action"] == "no_action"
        rejected_preserve = rejected and rejected["action"] == "no_action"
        direction = (
            "preserve_over_edit" if chosen_preserve and rejected is not None
            else "edit_over_preserve" if rejected_preserve
            else "edit_over_edit" if rejected is not None
            else "preserve_over_invalid" if chosen_preserve
            else "edit_over_invalid"
        )
        value["metadata"].update(
            input_bundle_kind=run["input_bundle"]["kind"],
            candidate_run_sha256=candidate_sha,
            semantic_label=case.get("judgment", {}).get("status", "edit_required"),
            primary_action=case["primary"]["action"],
            rejected_action=rejected["action"] if rejected else "invalid",
            preference_direction=direction,
            stratum=case.get("lineage", {}).get("stratum", "unrecorded"),
            parent_stratum=case.get("lineage", {}).get("parent_stratum", "not_applicable"),
        )
        values.append(value)
    reject_duplicate_pairs(values)
    return values


def reject_duplicate_pairs(pairs):
    require(
        len({pair["pair_id"] for pair in pairs}) == len(pairs)
        and len({
            (pair["input_sha256"], prep.value_hash(pair["chosen"]["input_ids"]),
             prep.value_hash(pair["rejected"]["input_ids"]))
            for pair in pairs
        }) == len(pairs),
        "Duplicate same-state preference pairs inflate exposure across candidate runs.",
    )


def verify_prepared_pairs(pairs, raw, registry, tokenizer, config):
    rebuilt = prepare_preferences(raw, registry, tokenizer, config)
    require(pairs == rebuilt, "Prepared pairs differ from their strict original judgments and candidate runs.")
    for pair in pairs:
        dpo.validate_pair_tokens(pair)


def pair_census(pairs):
    columns = (
        "source_id", "source_group_id", "language", "chosen_action", "rejected_action",
        "semantic_label", "preference_direction", "input_bundle_kind", "candidate_run_sha256",
        "stratum", "parent_stratum",
    )
    cases = Counter(pair["case_id"] for pair in pairs)
    return {
        "pairs": len(pairs), "unique_states": len(cases),
        "state_exposures": dict(sorted(cases.items())),
        "repeated_states": {key: value for key, value in sorted(cases.items()) if value > 1},
        "by": {key: dict(sorted(Counter(pair["metadata"][key] for pair in pairs).items()))
               for key in columns},
        "paired_sequence_tokens": sum(len(pair[side]["input_ids"]) for pair in pairs for side in SIDES),
        "paired_completion_tokens": sum(pair[side]["completion_tokens"] for pair in pairs for side in SIDES),
        "planned_policy_forward_sequence_tokens": sum(
            2 * len(pair["chosen"]["input_ids"]) + len(pair["rejected"]["input_ids"]) for pair in pairs
        ),
        "human_preferences": False, "train_only": True, "final_test_bodies_read": False,
    }


def training_settings(config, pairs):
    settings = {
        **config["training"], **FIXED_TRAINING,
        "pair_count": len(pairs), "max_updates": math.ceil(len(pairs) / 4),
        "pair_order": [pair["pair_id"] for pair in pairs],
        "chosen_recompute_rtol": 1e-6, "chosen_recompute_atol": 1e-4,
        "fresh_native_initialization": False,
        "initialization": "frozen_sft_adapter_fresh_optimizer",
    }
    for key in (
        "sample_order", "draw_count", "max_total_sequence_tokens", "max_total_completion_tokens",
        "action_exposures", "language_exposures", "unique_cases", "replacement", "final_update_rows",
        "draw_order_sha256", "draw_order_mode",
    ):
        settings.pop(key, None)
    return settings


def validate_protocol(protocol, identity, registry, inputs, preferences, pairs):
    expected = {
        "version": PROTOCOL_VERSION,
        "final_test_bodies_read": False,
        "anchor": identity,
        "candidate_manifest_sha256": list(registry),
        "preference_sha256": [entry["sha256"] for entry in preferences],
        "supplement_manifest_sha256": inputs["bindings"]["manifest_sha256"],
        "training_plan": {
            **FIXED_TRAINING, "pair_count": len(pairs),
            "max_updates": math.ceil(len(pairs) / 4),
        },
    }
    require(
        all(protocol.get(key) == value for key, value in expected.items()),
        "Protocol must bind both candidate runs, final judgments/count and the bounded DPO plan.",
    )


def source_names():
    return sorted(set(prep.source_closure([Path(__file__).name])) | {
        path.name for path in Path(__file__).parent.glob("prompt*.txt")
    })


def copy_receipt_tree(source, target, receipt):
    target.mkdir(parents=True, exist_ok=False)
    for name in ["manifest.json", *receipt["files_sha256"]]:
        original = sft.smoke.child(source, name)
        destination = sft.smoke.child(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)


def registry_binding(registry):
    return [
        {"manifest_sha256": sha, "path": f"candidate-runs/{sha}",
         "input_bundle": run["input_bundle"], "candidates": len(run["candidates"])}
        for sha, run in registry.items()
    ]


def prepare_pairs(config_path, training, reloaded, candidate_path,
                  supplement_candidate_path, input_path, pair_paths, protocol, output):
    config, root, bundle, _, _, _ = sft.preflight(config_path, assets=False)
    require(config["format"] == "json", "The selected B002 DPO anchor is JSON only.")
    identity = dpo.candidate_identity(config_path, training, reloaded)
    tokenizer = load_tokenizer(config, root)
    registry, inputs = candidate_registry(
        config, root, bundle, identity, candidate_path, supplement_candidate_path, input_path, tokenizer
    )
    preferences, raw = read_preferences(pair_paths, registry)
    pairs = prepare_preferences(raw, registry, tokenizer, config)
    validate_protocol(read(protocol), identity, registry, inputs, preferences, pairs)
    output = Path(output).resolve()
    for immutable in [root, Path(training), Path(reloaded), inputs["root"],
                      *(run["root"] for run in registry.values())]:
        require(not output.is_relative_to(immutable.resolve()), "Output must be outside immutable inputs.")
    output = sft.new_output(output, root, training)
    shutil.copytree(root, output / "sft", ignore=shutil.ignore_patterns("*.safetensors", "__pycache__"))
    copy_receipt_tree(inputs["root"], output / "supplement", inputs["manifest"])
    for sha, run in registry.items():
        target = output / "candidate-runs" / sha
        copy_receipt_tree(run["root"], target, run["manifest"])
    anchor = output / "anchor"
    anchor.mkdir()
    for destination, source in dpo.ANCHOR_FILES.items():
        target = anchor / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(training) / source, target)
    for name in ("manifest.json", "run.json", "reload-parity.json"):
        shutil.copyfile(Path(reloaded) / name, anchor / ("reload-" + name))
    for entry, path in zip(preferences, pair_paths, strict=True):
        target = output / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(protocol, output / "protocol.json")
    (output / "preparation-source").mkdir()
    sources = source_names()
    for name in sources:
        shutil.copyfile(Path(__file__).with_name(name), output / "preparation-source" / name)
    prep.write_rows(output / "pairs.jsonl.gz", pairs)
    prep.write(output / "pair-census.json", pair_census(pairs))
    binding = {
        "candidate_runs": registry_binding(registry), "preferences": preferences,
        "supplement": inputs["bindings"], "original_pairs": 51,
        "prepared_pairs": len(pairs), "final_test_bodies_read": False,
    }
    prep.write(output / "preference-input-binding.json", binding)
    result = {
        **copy.deepcopy(config), "version": VERSION, "phase": "substantive",
        "training": training_settings(config, pairs),
        "sft_config_path": "sft/config-json.json",
        "sft_preparation_sha256": digest(root / "manifest.json"),
        "pairs_sha256": digest(output / "pairs.jsonl.gz"),
        "protocol_sha256": digest(output / "protocol.json"),
        "census_sha256": digest(output / "pair-census.json"),
        "preference_input_binding_sha256": digest(output / "preference-input-binding.json"),
        "candidate_runs": binding["candidate_runs"], "preference_inputs": preferences,
        "supplement": {"path": "supplement", **inputs["bindings"]},
        "preparation_sources_sha256": {
            name: digest(output / "preparation-source" / name) for name in sources
        },
        "runtime_implementation": {
            "entrypoint": Path(__file__).name,
            "orchestration_adapted_from": Path(dpo.__file__).name,
            "frozen_dpo_sha256": config["sources_sha256"][Path(dpo.__file__).name],
            "objective_and_adapter_implementation_reused": True,
        },
        "anchor": identity,
        "anchor_files_sha256": {
            str(path.relative_to(anchor)): digest(path)
            for path in sorted(anchor.rglob("*")) if path.is_file()
        },
    }
    prep.write(output / "configuration.json", result)
    sft.manifest(output, PREPARED, VERSION)
    try:
        preflight(output / "configuration.json", assets=False)
    except BaseException as error:
        write(output / "preparation-error.json", {"error_type": type(error).__name__, "error": str(error)})
        sft.manifest(output, "failed", VERSION)
        raise
    return {"status": PREPARED, "pairs": len(pairs), "census": pair_census(pairs),
            "configuration_sha256": digest(output / "configuration.json")}


def verify_anchor(root, config, sft_config):
    anchor = root / "anchor"
    for name, expected in config["anchor_files_sha256"].items():
        sft.smoke.check_file(sft.smoke.child(anchor, name), expected)
    original = read(anchor / "train-manifest.json")
    require(
        digest(anchor / "train-manifest.json") == config["anchor"]["anchor_training_manifest_sha256"]
        and digest(anchor / "reload-manifest.json") == config["anchor"]["anchor_reload_manifest_sha256"]
        and digest(anchor / "configuration.json") == config["anchor"]["anchor_configuration_sha256"]
        and original["status"] == sft.TRAINED and original["version"] == sft.VERSION
        and read(anchor / "configuration.json") == sft_config,
        "Frozen reference is not the unchanged verified B002 SFT anchor.",
    )
    for destination, source in dpo.ANCHOR_FILES.items():
        if source != "manifest.json":
            require(digest(anchor / destination) == original["files_sha256"][source],
                    "Anchor evidence/adapter drift.")
    receipt, info = read(anchor / "reload-manifest.json"), read(anchor / "reload-run.json")
    for name in ("run.json", "reload-parity.json"):
        require(digest(anchor / ("reload-" + name)) == receipt["files_sha256"][name],
                "Anchor reload evidence drift.")
    parity = read(anchor / "reload-reload-parity.json")
    train_info = read(anchor / "train-run.json")
    require(
        receipt["version"] == sft.VERSION and receipt["status"] == info["status"] == sft.RELOADED
        and info["previous_manifest_sha256"] == digest(anchor / "train-manifest.json")
        and info["configuration_sha256"] == train_info["configuration_sha256"]
        == config["anchor"]["anchor_configuration_sha256"]
        and info["pid"] != train_info["pid"]
        and parity["adapter_tensors_identical"] is True
        and parity["cases"] and all(row["identical"] for row in parity["cases"])
        and digest(anchor / "adapter/adapter_model.safetensors") == config["anchor"]["anchor_adapter_sha256"],
        "SFT fresh reload or adapter identity was not verified.",
    )


def preflight(config_path, *, assets=True):
    config_path = Path(config_path).resolve()
    root, config = config_path.parent, read(config_path)
    receipt = read(root / "manifest.json")
    require(
        config["version"] == receipt["version"] == VERSION and receipt["status"] == PREPARED,
        "Unknown supplemented DPO preparation.",
    )
    for name, expected in receipt["files_sha256"].items():
        sft.smoke.check_file(sft.smoke.child(root, name), expected)
    require(
        config["preparation_sources_sha256"].keys() == set(source_names()),
        "Supplemented preparation source closure differs.",
    )
    for name, expected in config["preparation_sources_sha256"].items():
        sft.smoke.check_file(root / "preparation-source" / name, expected)
        sft.smoke.check_file(Path(__file__).with_name(name), expected)
    sft_config, sft_root, bundle, _, diagnostics, verified = sft.preflight(
        sft.smoke.child(root, config["sft_config_path"]), assets=assets
    )
    require(
        digest(sft_root / "manifest.json") == config["sft_preparation_sha256"]
        and config["anchor"]["anchor_configuration_sha256"] == verified["configuration_sha256"],
        "Original SFT preparation/configuration binding differs.",
    )
    for key, value in sft_config.items():
        if key not in {"version", "phase", "training", "protocol_sha256"}:
            require(config[key] == value, f"Frozen SFT configuration changed: {key}")
    require(
        config["runtime_implementation"] == {
            "entrypoint": Path(__file__).name,
            "orchestration_adapted_from": Path(dpo.__file__).name,
            "frozen_dpo_sha256": sft_config["sources_sha256"][Path(dpo.__file__).name],
            "objective_and_adapter_implementation_reused": True,
        }, "DPO implementation provenance differs.",
    )
    verify_anchor(root, config, sft_config)
    for name, key in (
        ("pairs.jsonl.gz", "pairs_sha256"), ("protocol.json", "protocol_sha256"),
        ("pair-census.json", "census_sha256"),
        ("preference-input-binding.json", "preference_input_binding_sha256"),
    ):
        sft.smoke.check_file(root / name, config[key])
    registered = config["candidate_runs"]
    require(
        len(registered) == 2 and len({entry["manifest_sha256"] for entry in registered}) == 2,
        "Exactly two separately bound candidate runs are required.",
    )
    for entry in registered:
        sft.smoke.check_file(sft.smoke.child(root, entry["path"]) / "manifest.json", entry["manifest_sha256"])
    tokenizer = load_tokenizer(sft_config, sft_root)
    registry, inputs = candidate_registry(
        sft_config, sft_root, bundle, config["anchor"],
        *(sft.smoke.child(root, entry["path"]) for entry in registered),
        sft.smoke.child(root, config["supplement"]["path"]), tokenizer,
    )
    require(
        registered == registry_binding(registry)
        and config["supplement"] == {"path": "supplement", **inputs["bindings"]},
        "Prepared candidate/input bundle provenance differs.",
    )
    preference_paths = [sft.smoke.child(root, entry["path"]) for entry in config["preference_inputs"]]
    preferences, raw = read_preferences(preference_paths, registry)
    require(preferences == config["preference_inputs"], "Frozen preference sources differ.")
    pairs = prep.rows(root / "pairs.jsonl.gz")
    verify_prepared_pairs(pairs, raw, registry, tokenizer, sft_config)
    expected_binding = {
        "candidate_runs": registry_binding(registry), "preferences": preferences,
        "supplement": inputs["bindings"], "original_pairs": 51,
        "prepared_pairs": len(pairs), "final_test_bodies_read": False,
    }
    require(
        read(root / "preference-input-binding.json") == expected_binding
        and read(root / "pair-census.json") == pair_census(pairs)
        and config["phase"] == "substantive"
        and config["training"] == training_settings(sft_config, pairs),
        "DPO census, final pair order or bounded training settings differ.",
    )
    validate_protocol(read(root / "protocol.json"), config["anchor"], registry, inputs, preferences, pairs)
    return config, root, sft_config, sft_root, pairs, diagnostics, {
        **{key: value for key, value in verified.items() if key not in ("train_rows", "draws")},
        "version": VERSION, "configuration_sha256": digest(config_path),
        "anchor_sft_train_rows": verified["train_rows"], "anchor_sft_draws": verified["draws"],
        "supplement_train_rows": len(inputs["cases"]), "planned_preference_pairs": len(pairs),
        "candidate_manifest_sha256": list(registry),
        "runtime_implementation": config["runtime_implementation"],
    }


def verify_cache(directory, config, pairs):
    _, info = sft.verified_run(directory, "reference_cache_complete", version=VERSION)
    require(info["binding"] == dpo.cache_binding(config, pairs),
            "Reference cache belongs to different inputs, outputs or SFT anchor.")
    records = prep.rows(Path(directory) / "reference.jsonl")
    require(len(records) == len(pairs) and len({row["pair_id"] for row in records}) == len(pairs),
            "Reference row count or IDs differ.")
    by_id = {row["pair_id"]: row for row in records}
    require(set(by_id) == {pair["pair_id"] for pair in pairs}, "Reference pair roster differs.")
    for pair in pairs:
        row = by_id[pair["pair_id"]]
        require(
            row["pair_binding"] == dpo.pair_binding(pair)
            and all(type(row[side + "_logprob"]) in (int, float)
                    and math.isfinite(row[side + "_logprob"]) and row[side + "_logprob"] <= 0
                    for side in SIDES),
            "Invalid or stale completion reference probability.",
        )
    return by_id


def check_reference_direct(model, pairs, references, torch, device, beta=0.1):
    results = []
    for candidate_sha in dict.fromkeys(pair["candidate_run_sha256"] for pair in pairs):
        selected = [pair for pair in pairs if pair["candidate_run_sha256"] == candidate_sha]
        results.extend(
            {**row, "candidate_run_sha256": candidate_sha}
            for row in dpo.check_reference_direct(model, selected, references, torch, device, beta)
        )
    return results


def export_configuration(training, reloaded, template_path, output):
    import export_passive_gemma as exporter

    training, reloaded = Path(training).resolve(), Path(reloaded).resolve()
    receipt, _ = sft.verify_reload(training, reloaded, version=VERSION)
    previous = exporter.configuration(template_path)
    config = read(training / "configuration.json")
    require(config["version"] == VERSION, "Export requires the supplemented DPO configuration.")
    require(
        all(previous["base"][key] == config["base"][key]
            for key in ("repository", "revision", "files_sha256")),
        "Export template native base differs from training.",
    )
    result = copy.deepcopy(previous)
    result["adapter"] = {
        "path": str(training / "adapter"), "base_model_name_or_path": sft.gemma.REPOSITORY,
        "files_sha256": {name: receipt["files_sha256"]["adapter/" + name]
                         for name in ("adapter_config.json", "adapter_model.safetensors")},
    }
    result["training_record"] = {
        "path": str(training / "manifest.json"), "sha256": digest(training / "manifest.json")
    }
    prep.write(Path(output), result)
    return {"status": "export_configuration_written_not_exported", "path": str(Path(output).resolve()),
            "configuration_sha256": digest(output)}


def export_adapter(config_path, reloaded, output):
    import export_gemma_actions_v3 as lora
    import export_passive_gemma as exporter

    config = exporter.configuration(config_path)
    train_root = Path(config["training_record"]["path"]).parent
    require(read(train_root / "manifest.json")["version"] == VERSION,
            "Require a completed supplemented DPO run.")
    sft.verify_reload(train_root, reloaded, version=VERSION)
    train_config = read(train_root / "configuration.json")
    require(train_config["version"] == VERSION, "Export training configuration version differs.")
    for module in (lora, exporter):
        sft.smoke.check_file(Path(module.__file__), train_config["sources_sha256"][Path(module.__file__).name])
    output = Path(output).resolve()
    require(
        not output.is_relative_to(train_root.resolve())
        and not output.is_relative_to(Path(reloaded).resolve()),
        "Export output must be outside immutable training and reload evidence.",
    )
    started = time.monotonic()
    report = exporter.preflight(config_path, output)
    try:
        bound_config, _ = exporter.bound_run(config_path, output)
        name, command = next(
            (name, command) for name, command in exporter.conversion_commands(bound_config, output)
            if name == "convert-adapter"
        )
        exporter.run_command(output, name, command)
        result = lora.verify_adapter_tensors(
            Path(config["adapter"]["path"]), exporter.paths(output)["adapter"],
            report["pairs"], report["alpha"], exporter.gguf_module(config),
        )
        receipt = {
            "version": VERSION + "-lora-export",
            "status": "adapter_conversion_verified_serving_pending", **result,
            "configuration_sha256": exporter.sha256(output / "configuration.json"),
            "native_pin_sha256": report["native_pin_sha256"],
            "training_manifest_sha256": exporter.sha256(train_root / "manifest.json"),
            "reload_manifest_sha256": exporter.sha256(Path(reloaded) / "manifest.json"),
            "base_repository": exporter.BASE_REPOSITORY, "base_revision": exporter.BASE_REVISION,
            "runtime_revision": exporter.RUNTIME_REVISION,
            "contract_version": train_config["contract_version"], "format_version": train_config["format_version"],
            "output_format": train_config["format"], "training_phase": train_config["phase"],
            "implementation_sha256": exporter.sha256(Path(__file__)),
            "tensor_verifier_source_sha256": exporter.sha256(Path(lora.__file__)),
            "orchestration_adapted_from": Path(lora.__file__).name,
            "seconds": time.monotonic() - started, "full_base_or_merged_copy_created": False,
            "serving_requirement": "Use the verified native-derived base GGUF with this adapter at scale 1.0; validate base+LoRA inference separately.",
            "scope": "Exact conversion of every FP32 LoRA tensor. This does not certify merged quantization, base+LoRA predictions, quality or activation.",
        }
        exporter.write_new(output / "adapter-receipt.json", receipt)
        return receipt
    except BaseException as error:
        exporter.write_new(output / "adapter-export-failure.json", {
            "version": VERSION + "-lora-export", "status": "failed",
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def reference(config_path, output):
    config, root, sft_config, sft_root, pairs, _, verified = preflight(config_path)
    output = sft.new_output(output, root)
    started = time.monotonic()
    info = {
        **verified,
        "mode": "reference",
        "status": "loading",
        "pid": os.getpid(),
        "binding": dpo.cache_binding(config, pairs),
    }
    sft.gemma.snapshot_configuration(
        config_path, output / "configuration.json", verified["configuration_sha256"]
    )
    write(output / "run.json", info)
    try:
        torch, model, _, device, runtime, _, _ = dpo.load(
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
                    "pair_binding": dpo.pair_binding(pair),
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


def run(config_path, reference_path, output, previous=None):
    config, root, sft_config, sft_root, pairs, diagnostics, verified = preflight(
        config_path
    )
    references = verify_cache(reference_path, config, pairs)
    require(
        read(Path(reference_path) / "run.json")["pid"] != os.getpid(),
        "Compute the reference cache in an independent process before policy training or reload.",
    )
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
        torch, model, tokenizer, device, runtime, _, initial = dpo.load(
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
            require(
                exposure.updates == config["training"]["max_updates"]
                and exposure.phases["optimizer_committed"]["rows"] == len(pairs),
                "The bounded DPO run must commit exactly one pass over its frozen pairs.",
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
    prepare = commands.add_parser("prepare-pairs")
    for name in ("config", "training", "reloaded", "candidates", "supplement-candidates",
                 "input-plan", "protocol", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--pairs", type=Path, action="append", required=True)
    for name in ("preflight", "reference", "run", "reload"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "preflight":
            command.add_argument("--metadata-only", action="store_true")
        else:
            command.add_argument("--output", type=Path, required=True)
        if name in ("run", "reload"):
            command.add_argument("--reference", type=Path, required=True)
        if name == "reload":
            command.add_argument("--run", type=Path, required=True)
    export = commands.add_parser("export-config")
    for name in ("training", "reloaded", "template", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    adapter = commands.add_parser("export-adapter")
    for name in ("config", "reloaded", "output"):
        adapter.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "export-config":
        args["template_path"] = args.pop("template")
        result = export_configuration(**args)
    else:
        args["config_path"] = args.pop("config")
        if command == "export-adapter":
            result = export_adapter(**args)
        elif command == "preflight":
            result = preflight(args["config_path"], assets=not args["metadata_only"])[-1]
        elif command == "prepare-pairs":
            args["candidate_path"] = args.pop("candidates")
            args["supplement_candidate_path"] = args.pop("supplement_candidates")
            args["input_path"] = args.pop("input_plan")
            args["pair_paths"] = args.pop("pairs")
            result = prepare_pairs(**args)
        elif command in ("run", "reload"):
            args["reference_path"] = args.pop("reference")
            if "run" in args:
                args["previous"] = args.pop("run")
            result = run(**args)
        else:
            result = reference(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


