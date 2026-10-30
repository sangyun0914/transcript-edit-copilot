"""Generate the frozen 22-case supplement with the unchanged B002 SFT anchor."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import prepare_dpo_supplement_actions_v3 as frozen
import prepare_gemma_actions_v3 as prep
import train_gemma_actions_v3 as sft
import train_gemma_actions_v3_dpo as dpo

VERSION = dpo.VERSION
SUPPLEMENT_SCHEMA = "actions-v3-dpo-supplement-candidates-1"
SUPPLEMENT_MANIFEST_SHA256 = "67a11cd68e6b76c0de0a01437ecd6c111b54ec9e5e90a8a3747c5ab569950fbb"
ANCHOR_KEYS = (
    "anchor_training_manifest_sha256",
    "anchor_reload_manifest_sha256",
    "anchor_configuration_sha256",
    "anchor_adapter_sha256",
)
RAW_KEYS = (
    "case_id", "prompt_ids_sha256", "output_ids", "terminal_token_id",
    "text", "text_raw", "finish_reason",
)
PLANNED_GENERATION = {
    "format": "json",
    "candidates_per_case": 1,
    "sampling": "deterministic_greedy",
    "maximum_new_tokens": 512,
    "repeat_draws": False,
    "retain_all_raw_outputs_failures_and_ties": True,
}
require, digest, read, write = sft.require, sft.digest, sft.read, sft.write


def exact_files(root, receipt):
    paths = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "Bundle symlinks are forbidden.")
    actual = {
        str(path.relative_to(root))
        for path in paths
        if path.is_file() and path != root / "manifest.json"
    }
    require(actual == set(receipt["files_sha256"]), "Bundle file inventory differs.")
    for name, expected in receipt["files_sha256"].items():
        sft.smoke.check_file(sft.smoke.child(root, name), expected)


def verify_supplement(input_path, expected_identity=None):
    """Read the pinned whole input bundle; never accept an arbitrary cases file."""
    path = Path(input_path)
    require(not path.is_symlink(), "Supplement input cannot be a symlink.")
    if path.is_file():
        require(path.name == "candidate-input-plan.json", "Use the frozen input plan or its directory.")
        path = path.parent
    root = path.resolve()
    sft.smoke.check_file(root / "manifest.json", SUPPLEMENT_MANIFEST_SHA256)
    manifest = read(root / "manifest.json")
    require(
        manifest["version"] == frozen.VERSION
        and manifest["status"] == "inputs_frozen_candidates_not_generated"
        and manifest["cases"] == 22
        and manifest["train_only"] is True
        and manifest["final_test_bodies_read"] is False,
        "Unknown supplement input manifest.",
    )
    exact_files(root, manifest)
    plan = read(root / "candidate-input-plan.json")
    require(
        plan["version"] == manifest["version"]
        and plan["status"] == manifest["status"]
        and plan["cases"] == 22
        and plan["planned_generation"] == PLANNED_GENERATION
        and plan["generated_candidates"] == plan["generated_preference_pairs"] == 0
        and plan["dev_bodies_read"] is False
        and plan["final_test_bodies_read"] is False
        and plan["model_predictions_consulted"] is False,
        "Supplement plan or generation policy differs.",
    )
    identity = plan["anchor"]
    require(
        set(identity) == set(ANCHOR_KEYS)
        and identity == manifest["anchor"]
        and all(identity[key] == plan["original_anchor_identity"][key.removeprefix("anchor_")] for key in ANCHOR_KEYS),
        "Supplement anchor identity differs.",
    )
    if expected_identity is not None:
        require(identity == expected_identity, "Supplement differs from the verified SFT anchor.")
    for key, name in (
        ("anchor_configuration", "anchor/configuration.json"),
        ("input_cases", "cases.jsonl"),
        ("case_roster", "case-roster.jsonl"),
    ):
        require(
            plan[key] == {"path": name, "sha256": manifest["files_sha256"][name]},
            "Supplement plan payload binding differs: " + key,
        )
    require(set(plan["evidence"]) == set(frozen.FILES), "Supplement evidence inventory differs.")
    evidence = {}
    for key, original_path in frozen.FILES.items():
        snapshot = (
            "cases.jsonl" if key == "cases"
            else "anchor/configuration.json" if key == "configuration"
            else f"evidence/{key}.json"
        )
        entry = plan["evidence"][key]
        require(
            entry == {
                "path": original_path,
                "sha256": manifest["files_sha256"][snapshot],
                "bytes": (root / snapshot).stat().st_size,
                "snapshot": snapshot,
            },
            "Supplement evidence binding differs: " + key,
        )
        if key in frozen.TRUSTED_SHA256:
            require(entry["sha256"] == frozen.TRUSTED_SHA256[key], "Pinned supplement evidence differs: " + key)
        if key != "cases":
            evidence[key] = read(root / snapshot)
    require(
        evidence["selection"]["identity"] == plan["original_anchor_identity"]
        and identity["anchor_configuration_sha256"] == plan["anchor_configuration"]["sha256"]
        and identity["anchor_training_manifest_sha256"] == plan["evidence"]["training"]["sha256"]
        and identity["anchor_reload_manifest_sha256"] == plan["evidence"]["reload"]["sha256"]
        and identity["anchor_adapter_sha256"] == evidence["training"]["files_sha256"]["adapter/adapter_model.safetensors"],
        "Supplement anchor evidence differs.",
    )
    roster, counts = frozen.read_roster(root / "cases.jsonl", evidence)
    require(
        roster == prep.rows(root / "case-roster.jsonl")
        and [row["case_id"] for row in roster] == plan["case_order"]
        and counts == plan["counts"] == manifest["counts"],
        "Supplement roster, order, exposure or census differs.",
    )
    cases = prep.rows(root / "cases.jsonl")
    require(
        [case["id"] for case in cases] == plan["case_order"]
        and len(cases) == len(set(plan["case_order"])) == 22,
        "Require all 22 supplement cases in frozen order.",
    )
    return {
        "root": root,
        "manifest": manifest,
        "plan": plan,
        "cases": cases,
        "roster": roster,
        "identity": identity,
        "bindings": {
            "manifest_sha256": SUPPLEMENT_MANIFEST_SHA256,
            "plan_sha256": manifest["files_sha256"]["candidate-input-plan.json"],
            "cases_sha256": manifest["files_sha256"]["cases.jsonl"],
            "roster_sha256": manifest["files_sha256"]["case-roster.jsonl"],
            "configuration_sha256": identity["anchor_configuration_sha256"],
        },
    }


def prompt_record(case, record):
    prefix = record["primary"]["input_ids"][: record["primary"]["completion_start"]]
    messages = prep.formats.messages(case["input"], "json")
    return {
        "case_id": case["id"],
        "input_sha256": prep.value_hash(prep.formats.contract.wire_snapshot(case["input"])),
        "messages": messages,
        "prompt_sha256": prep.value_hash(messages),
        "input_ids": prefix,
        "completion_start": len(prefix),
        "prompt_ids_sha256": prep.value_hash(prefix),
    }


def tokenize_supplement(supplement, tokenizer):
    cases = supplement["cases"]
    admitted, encoded, exclusions = prep.tokenize_cases(cases, tokenizer, selected_formats=("json",))
    records = encoded["json"]
    require(
        not exclusions
        and [case["id"] for case in admitted] == supplement["plan"]["case_order"]
        and [record["id"] for record in records] == supplement["plan"]["case_order"]
        and len(records) == 22,
        "All 22 supplement cases must tokenize in order; dropping or truncation is forbidden.",
    )
    builder = prep.formats.contract.PassiveContextBuilder(tokenizer)
    prompts = []
    for case, record in zip(cases, records, strict=True):
        prep.validate_record(case, record, "json")
        require(dpo.dataset.runtime_bundle(case, builder)["input"] == case["input"], "Supplement runtime input changed.")
        prompts.append(prompt_record(case, record))
    sft.gemma.validate_tokenizer(tokenizer, records)
    return records, prompts


def generation_rows(prompts):
    # Strip labels and reference completions before handing rows to the frozen generator.
    return [
        {"case_id": row["case_id"], "input_ids": list(row["input_ids"]), "completion_start": len(row["input_ids"])}
        for row in prompts
    ]


def parsed_output(text, case, finish_reason):
    try:
        return prep.formats.parse(text, case["input"], "json", finish_reason=finish_reason).model_dump(), None
    except ValueError as error:
        return None, str(error)


def candidate_record(value, case, prompt, supplement):
    require(
        value["case_id"] == case["id"]
        and value["prompt_ids_sha256"] == prompt["prompt_ids_sha256"],
        "Generated case or actual prompt hash differs.",
    )
    parsed, error = parsed_output(value["text"], case, value["finish_reason"])
    return {
        **value,
        "candidate_id": f"{case['id']}:greedy:0",
        "split": "train",
        "format": "json",
        "input_sha256": prompt["input_sha256"],
        "prompt_sha256": prompt["prompt_sha256"],
        "output_ids_sha256": prep.value_hash(value["output_ids"]),
        "raw_output_sha256": frozen.bytes_hash(value["text_raw"].encode()),
        **supplement["identity"],
        "supplement_input": supplement["bindings"],
        "parsed_action": parsed,
        "validation_error": error,
        "generation_error": None,
    }


def validate_candidate(value, case, prompt, supplement, tokenizer=None):
    require(
        value["case_id"] == case["id"]
        and value["candidate_id"] == f"{case['id']}:greedy:0"
        and value["split"] == case["split"] == "train"
        and value["format"] == "json"
        and value["supplement_input"] == supplement["bindings"]
        and all(value[key] == supplement["identity"][key] for key in ANCHOR_KEYS)
        and value["generation_error"] is None,
        "Candidate identity, supplement binding or generation status differs.",
    )
    require(
        all(value[key] == prompt[key] for key in ("input_sha256", "prompt_sha256", "prompt_ids_sha256")),
        "Candidate input or prompt binding differs.",
    )
    ids = value["output_ids"]
    require(
        isinstance(ids, list)
        and 0 < len(ids) <= prep.GENERATION["max_new_tokens"]
        and all(type(token) is int and 0 <= token < 262144 for token in ids)
        and not set(ids[:-1]) & set(prep.GENERATION["eos_token_id"])
        and value["output_ids_sha256"] == prep.value_hash(ids)
        and isinstance(value["text"], str)
        and isinstance(value["text_raw"], str)
        and value["raw_output_sha256"] == frozen.bytes_hash(value["text_raw"].encode()),
        "Candidate raw output or tokens differ.",
    )
    terminal = ids[-1] if ids else None
    require(
        value["terminal_token_id"] == terminal
        and value["finish_reason"] == ("stop" if terminal in prep.GENERATION["eos_token_id"] else "length")
        and (terminal in prep.GENERATION["eos_token_id"] or len(ids) == prep.GENERATION["max_new_tokens"]),
        "Candidate terminal or finish reason differs.",
    )
    if tokenizer is not None:
        require(
            value["text"] == tokenizer.decode(ids, skip_special_tokens=True)
            and value["text_raw"] == tokenizer.decode(ids, skip_special_tokens=False),
            "Candidate text differs from the raw generated tokens.",
        )
    parsed, error = parsed_output(value["text"], case, value["finish_reason"])
    require(value["parsed_action"] == parsed and value["validation_error"] == error, "Candidate parse receipt differs.")


def source_snapshot(output):
    names = prep.source_closure([Path(__file__).name])
    (output / "sources").mkdir()
    result = {}
    for name in names:
        source = Path(__file__).with_name(name)
        shutil.copyfile(source, output / "sources" / name)
        result[name] = digest(source)
    return result


def run_manifest(output, status):
    write(output / "manifest.json", {
        "version": VERSION,
        "status": status,
        "files_sha256": {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path != output / "manifest.json"
        },
    })


def validate_candidate_rows(supplement, records, prompts, values, tokenizer=None):
    order = supplement["plan"]["case_order"]
    require(
        [row["id"] for row in records] == order
        and [row["case_id"] for row in prompts] == order
        and [row["case_id"] for row in values] == order
        and len({row["candidate_id"] for row in values}) == len(values) == 22,
        "Complete candidates require the exact 22-case ordered roster.",
    )
    builder = prep.formats.contract.PassiveContextBuilder(tokenizer) if tokenizer is not None else None
    for case, record, prompt, value in zip(supplement["cases"], records, prompts, values, strict=True):
        prep.validate_case(case)
        prep.validate_record(case, record, "json")
        require(prompt == prompt_record(case, record), "Saved prompt contains changed input, tokens or reference leakage.")
        if tokenizer is not None:
            require(dpo.dataset.runtime_bundle(case, builder)["input"] == case["input"], "Supplement runtime input changed on readback.")
        validate_candidate(value, case, prompt, supplement, tokenizer)
    if tokenizer is not None:
        sft.gemma.validate_tokenizer(tokenizer, records)


def verify_candidate_run(path, expected_identity=None, *, input_path=None, tokenizer=None):
    """Verify complete candidate provenance; pass the pinned tokenizer for token readback."""
    root = Path(path).resolve()
    manifest, info = sft.verified_run(root, "candidates_complete", version=VERSION)
    exact_files(root, manifest)
    supplement = verify_supplement(root / "supplement", expected_identity)
    if input_path is not None:
        external = verify_supplement(input_path, supplement["identity"])
        require(external["bindings"] == supplement["bindings"], "Candidate supplemental input differs.")
    identity = supplement["identity"]
    require(
        info["version"] == VERSION
        and info["mode"] == "supplemental_candidate_generation"
        and info["supplement_schema"] == SUPPLEMENT_SCHEMA
        and info["supplement"] == supplement["bindings"]
        and info["case_order"] == supplement["plan"]["case_order"]
        and info["candidates"] == 22
        and info["generation"] == prep.GENERATION
        and info["train_only"] is True
        and info["final_test_bodies_read"] is False
        and info["configuration_sha256"] == identity["anchor_configuration_sha256"]
        and all(info[key] == identity[key] for key in ANCHOR_KEYS),
        "Supplement candidate run metadata differs.",
    )
    require(
        digest(root / "anchor-training-manifest.json") == identity["anchor_training_manifest_sha256"]
        and digest(root / "anchor-reload-manifest.json") == identity["anchor_reload_manifest_sha256"]
        and read(root / "anchor-training-manifest.json")["files_sha256"]["adapter/adapter_model.safetensors"] == identity["anchor_adapter_sha256"],
        "Candidate anchor snapshots differ.",
    )
    preflight = read(root / "original-sft-preflight.json")
    require(
        preflight["status"] == "configuration_and_inputs_verified"
        and preflight["native_assets_verified"] is True
        and preflight["train_rows"] == 848
        and preflight["configuration_sha256"] == identity["anchor_configuration_sha256"]
        and preflight["preparation_sha256"] == supplement["plan"]["evidence"]["preparation"]["sha256"],
        "Original 848-case SFT preflight evidence differs.",
    )
    names = prep.source_closure([Path(__file__).name])
    require(set(info["sources_sha256"]) == set(names), "Candidate source closure differs.")
    for name in names:
        expected = info["sources_sha256"][name]
        sft.smoke.check_file(root / "sources" / name, expected)
        sft.smoke.check_file(Path(__file__).with_name(name), expected)
    require(info["execution_source_sha256"] == info["sources_sha256"][Path(__file__).name], "Candidate entry point binding differs.")
    require("runtime.json" in manifest["files_sha256"], "Candidate runtime evidence is missing.")
    records = prep.rows(root / "tokens-json.jsonl.gz")
    prompts = prep.rows(root / "prompts.jsonl")
    values = prep.rows(root / "candidates.jsonl")
    require(
        prep.rows(root / "raw-generations.jsonl") == [{key: value[key] for key in RAW_KEYS} for value in values],
        "Raw generation returns differ from the saved candidates.",
    )
    validate_candidate_rows(supplement, records, prompts, values, tokenizer)
    return {
        "root": root,
        "manifest": manifest,
        "info": info,
        "candidates": values,
        "supplement": supplement,
        "identity": identity,
        "manifest_sha256": digest(root / "manifest.json"),
        "tokenizer_reverified": tokenizer is not None,
    }


def candidates(config_path, training, reloaded, input_plan, output):
    config, sft_root, sft_bundle, _, _, verified = sft.preflight(config_path)
    identity = dpo.candidate_identity(config_path, training, reloaded)
    supplement = verify_supplement(input_plan, identity)
    require(
        config["format"] == "json"
        and config["generation"] == prep.GENERATION
        and len(sft_bundle["cases"]) == 848
        and {case["id"] for case in sft_bundle["cases"]}.isdisjoint(supplement["plan"]["case_order"]),
        "Keep the original 848-case SFT anchor separate from all 22 supplement inputs.",
    )
    destination = Path(output).resolve()
    require(
        not Path(output).is_symlink()
        and not destination.is_relative_to(supplement["root"])
        and not destination.is_relative_to(Path(reloaded).resolve()),
        "Output must be outside immutable supplemental and reload inputs.",
    )
    output = sft.new_output(output, sft_root, training)
    started = time.monotonic()
    emitted, active_case, active_prompt = [], None, None
    info = {
        "version": VERSION,
        "mode": "supplemental_candidate_generation",
        "supplement_schema": SUPPLEMENT_SCHEMA,
        "status": "loading",
        "pid": os.getpid(),
        "configuration_sha256": digest(config_path),
        **identity,
        "supplement": supplement["bindings"],
        "case_order": supplement["plan"]["case_order"],
        "generation": config["generation"],
        "train_only": True,
        "final_test_bodies_read": False,
    }
    write(output / "run.json", info)
    try:
        sft.gemma.snapshot_configuration(config_path, output / "configuration.json", info["configuration_sha256"])
        shutil.copytree(supplement["root"], output / "supplement")
        verify_supplement(output / "supplement", identity)
        shutil.copyfile(Path(training) / "manifest.json", output / "anchor-training-manifest.json")
        shutil.copyfile(Path(reloaded) / "manifest.json", output / "anchor-reload-manifest.json")
        write(output / "original-sft-preflight.json", verified)
        info["sources_sha256"] = source_snapshot(output)
        info["execution_source_sha256"] = info["sources_sha256"][Path(__file__).name]
        write(output / "run.json", info)
        torch, model, tokenizer, device, runtime = sft.gemma.load_runtime(config, sft_root, sft_bundle["tokens"]["json"])
        model, _, _ = sft.load_adapter(model, Path(training) / "adapter", read(Path(training) / "adapter-tensors.json"), config, torch, False)
        write(output / "runtime.json", runtime)
        records, prompts = tokenize_supplement(supplement, tokenizer)
        prep.write_rows(output / "tokens-json.jsonl.gz", records)
        prep.write_rows(output / "prompts.jsonl", prompts)
        info["status"] = "generating"
        write(output / "run.json", info)
        rows = zip(supplement["cases"], prompts, generation_rows(prompts), strict=True)
        for index, (case, prompt, row) in enumerate(rows):
            active_case = case["id"]
            active_prompt = prompt

            def emit(value):
                sft.curriculum.append(output / "raw-generations.jsonl", value)
                result = candidate_record(value, case, prompt, supplement)
                sft.curriculum.append(output / "candidates.jsonl", result)
                emitted.append(result)
                print(json.dumps({"event": "supplement_candidate", "case_id": case["id"], "valid": result["parsed_action"] is not None}), flush=True)

            sft.gemma.generate_smoke(model, tokenizer, [row], config, torch, device, emit)
            require(len(emitted) == index + 1, "Generator did not emit exactly one candidate per case.")
        require([row["case_id"] for row in emitted] == info["case_order"], "Generator did not complete the exact 22-case roster.")
        info.update(
            status="candidates_complete",
            candidates=len(emitted),
            seconds=time.monotonic() - started,
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        )
        write(output / "run.json", info)
        run_manifest(output, info["status"])
        verify_candidate_run(output, identity, tokenizer=tokenizer)
        return info
    except BaseException as error:
        failed_stage = info["status"]
        info.update(
            status="failed",
            candidates=len(emitted),
            failed_case_id=active_case,
            incomplete_case_ids=[case_id for case_id in info["case_order"] if case_id not in {row["case_id"] for row in emitted}],
            error_type=type(error).__name__,
            error=str(error),
            seconds=time.monotonic() - started,
        )
        write(output / "failure.json", {
            "stage": failed_stage,
            "case_id": active_case,
            "input_sha256": active_prompt["input_sha256"] if active_prompt else None,
            "prompt_ids_sha256": active_prompt["prompt_ids_sha256"] if active_prompt else None,
            "supplement_input": supplement["bindings"],
            **identity,
            "error": {"type": type(error).__name__, "message": str(error)},
            "completed_case_ids": [row["case_id"] for row in emitted],
            "incomplete_case_ids": info["incomplete_case_ids"],
            "raw_outputs": "Completed model returns are retained in raw-generations.jsonl; an interrupted call has no invented output.",
        })
        write(output / "run.json", info)
        run_manifest(output, "failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("candidates")
    verify = sub.add_parser("verify")
    for command in (generate, verify):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--training", type=Path, required=True)
        command.add_argument("--reloaded", type=Path, required=True)
    generate.add_argument("--input-plan", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    verify.add_argument("--candidates", type=Path, required=True)
    verify.add_argument("--input-plan", type=Path)
    inputs = sub.add_parser("verify-input")
    inputs.add_argument("--input-plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "candidates":
        result = candidates(args.config, args.training, args.reloaded, args.input_plan, args.output)
    elif args.command == "verify-input":
        bundle = verify_supplement(args.input_plan)
        result = {"status": "supplement_inputs_verified", "cases": len(bundle["cases"]), "bindings": bundle["bindings"]}
    else:
        from transformers import AutoTokenizer

        config, root, original, _, _, _ = sft.preflight(args.config, assets=False)
        identity = dpo.candidate_identity(args.config, args.training, args.reloaded)
        tokenizer = AutoTokenizer.from_pretrained(root / config["tokenizer"]["path"], local_files_only=True, trust_remote_code=False)
        sft.gemma.validate_tokenizer(tokenizer, original["tokens"]["json"])
        checked = verify_candidate_run(args.candidates, identity, input_path=args.input_plan, tokenizer=tokenizer)
        result = {"status": "supplement_candidates_verified", "candidates": len(checked["candidates"]), "manifest_sha256": checked["manifest_sha256"], "tokenizer_reverified": checked["tokenizer_reverified"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
