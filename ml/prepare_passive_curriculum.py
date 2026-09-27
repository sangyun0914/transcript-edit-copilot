"""Freeze a train-only weighted curriculum and verify its portable payloads."""

import argparse
import bisect
import json
import math
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import prepare_passive_sft as smoke

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-sft-curriculum-preparation-v1"
CANDIDATE = "qwen35-9b"
TRAIN_ROWS = 809
PAYLOADS = {
    "train-cases.jsonl": "train-cases.jsonl",
    "train-messages.jsonl": "train-messages.jsonl",
    "train-tokens.jsonl.gz": "tokenized/qwen35-9b/json-curriculum-train.jsonl.gz",
    "source-groups.json": "source-groups.json",
    "prompt-json.txt": "prompt-json.txt",
}
EVIDENCE = {
    "release": "evidence/release-manifest.json",
    "selection": "evidence/selection.json",
    "smoke_completion": "evidence/smoke-completion.json",
    "smoke_configuration": "evidence/smoke-configuration.json",
}
SMOKE_PAYLOADS = {
    "smoke-cases.jsonl",
    "smoke-messages.jsonl",
    "smoke-tokens.jsonl",
    "smoke-model-inputs.jsonl",
    "configuration.json",
    "exposure-census.json",
    "source/prepare_passive_sft.py",
    "source/test_prepare_passive_sft.py",
    "native-metadata/config.json",
    "native-metadata/tokenizer_config.json",
    "native-metadata/chat_template.jinja",
}
SOURCE_FILES = (
    "prepare_passive_curriculum.py",
    "test_prepare_passive_curriculum.py",
    "prepare_passive_sft.py",
    "passive_bundle_v2.py",
)
SMOKE_GATES = {
    "frozen_data_independent_readback",
    "B2_reconstruction_and_selection",
    "native_weight_hashes_and_actual_modules",
    "two_full_length_optimizer_updates",
    "fresh_process_exact_adapter_and_output_reload",
    "paired_native_derived_F16_Q4_tensor_export",
    "local_serving_prefix_eos_and_adapter_validity",
}
require = smoke.require
sha256 = smoke.sha256
json_bytes = smoke.json_bytes


def child(directory, name):
    value = Path(name)
    require(not value.is_absolute(), "Bundle paths must be relative.")
    path = (directory / value).resolve()
    require(path.is_relative_to(directory.resolve()), "Bundle path escapes its root.")
    return path


def read_bound(path, expected):
    require(
        isinstance(expected, str)
        and len(expected) == 64
        and all(value in "0123456789abcdef" for value in expected),
        "An explicit SHA256 is required.",
    )
    content = Path(path).read_bytes()
    require(sha256(content) == expected, f"Bound file changed: {path}")
    return content


def weighted_draw_indices(cases, count, seed):
    require(
        type(seed) is str and bool(seed),
        "The sampling seed must be an explicit string.",
    )
    require(type(count) is int and count > 0 and cases, "Invalid weighted draw count.")
    cumulative, total = [], 0.0
    for case in cases:
        meta = case["metadata"]
        weight = meta["sampling_weight"]
        require(
            meta["split"] == "train"
            and meta.get("training_eligible") is True
            and type(weight) in (int, float)
            and math.isfinite(weight)
            and weight > 0,
            "Only eligible train rows with positive finite weights may be sampled.",
        )
        total += weight
        cumulative.append(total)
    require(math.isfinite(total), "The total sampling weight is not finite.")
    rng = random.Random(seed)
    return [bisect.bisect_right(cumulative, rng.random() * total) for _ in range(count)]


def exposure(cases, tokens, indices):
    frequencies = Counter(indices)
    lengths = [len(row["primary"]["input_ids"]) for row in tokens]
    prompts = [row["primary"]["completion_start"] for row in tokens]
    names = {
        "language": lambda case: case["metadata"]["language"],
        "action": lambda case: case["expectedOutput"]["action"],
        "action_language": lambda case: (
            case["expectedOutput"]["action"] + "/" + case["metadata"]["language"]
        ),
        "source_group": lambda case: case["metadata"]["source_group_id"],
        "source": lambda case: case["metadata"]["source_id"],
        "episode": lambda case: case["metadata"]["episode_id"],
        "sampling_cluster": lambda case: case["metadata"]["sampling_cluster"],
        "release_stratum": lambda case: case["metadata"].get(
            "release_stratum", "unspecified"
        ),
        "kind": lambda case: case["metadata"].get("kind", "unspecified"),
    }
    strata = {}
    for name, key in names.items():
        groups = defaultdict(list)
        for index in indices:
            groups[key(cases[index])].append(index)
        strata["by_" + name] = {
            value: {
                "draws": len(members),
                "unique_cases": len(set(members)),
                "sequence_tokens": sum(lengths[i] for i in members),
                "prompt_tokens": sum(prompts[i] for i in members),
                "completion_tokens": sum(lengths[i] - prompts[i] for i in members),
            }
            for value, members in sorted(groups.items())
        }
    return {
        "status": "fixed_planned_exposure_not_observed_training",
        "available_unique_cases": len(cases),
        "draws": len(indices),
        "unique_cases": len(frequencies),
        "undrawn_cases": len(cases) - len(frequencies),
        "repeat_draws": len(indices) - len(frequencies),
        "repeated_case_ids": sum(count > 1 for count in frequencies.values()),
        "maximum_case_multiplicity": max(frequencies.values()),
        "case_draw_counts": {
            case["id"]: frequencies[index] for index, case in enumerate(cases)
        },
        "case_multiplicity_histogram": {
            str(count): frequency
            for count, frequency in sorted(
                Counter(frequencies[index] for index in range(len(cases))).items()
            )
        },
        "sequence_tokens": sum(lengths[i] for i in indices),
        "prompt_tokens": sum(prompts[i] for i in indices),
        "completion_tokens": sum(lengths[i] - prompts[i] for i in indices),
        "sampled_case_order_sha256": sha256(
            smoke.jsonl_bytes(cases[i]["id"] for i in indices)
        ),
        **strata,
    }


def configuration(prepared, cases, tokens, indices, seed):
    realized = exposure(cases, tokens, indices)
    return {
        "version": VERSION,
        "status": "frozen_train_only_preparation",
        "candidate_id": CANDIDATE,
        "native_base": prepared["native_base"],
        "tokenizer": prepared["tokenizer"],
        "training_view": "json-curriculum",
        "frozen_train_rows": len(cases),
        "loss_mask": "completion_only_including_eos",
        "input_limit": 3584,
        "output_limit": 512,
        "context_limit": 4096,
        "truncation": False,
        "packing": False,
        "sampling": {
            "algorithm": "weighted_indices_v2",
            "seed": seed,
            "seed_type": "str",
            "replacement": True,
            "draw_count": len(indices),
            "weights_applied_to_loss": False,
        },
        "training": {
            "micro_batch_size": 1,
            "gradient_accumulation": 4,
            "optimizer_updates": math.ceil(len(indices) / 4),
            "final_update_rows": len(indices) % 4 or 4,
            "final_update_normalization": "mean_over_actual_microbatch_count",
            "max_total_sequence_tokens": realized["sequence_tokens"],
            "max_total_completion_tokens": realized["completion_tokens"],
            "sample_order": [cases[index]["id"] for index in indices],
            "initialization": "fresh_adapter_on_exact_native_base_not_smoke_adapter",
        },
        "diagnostics": {
            "preparation": "smoke/manifest.json",
            "case_order": prepared["proposed_smoke"]["order"],
            "scope": "The unchanged eight train states are fixed engineering diagnostics, not held-out quality evidence.",
        },
        "training_run": False,
        "dev_test_record_bodies_copied": 0,
        "scope": "One fixed weighted primary-target stream. Draws are not unique-state coverage or independent evidence; substantive outcome remains unmeasured.",
    }


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_bytes())
    require(
        manifest["version"] == VERSION
        and manifest["status"] == "frozen_train_only_preparation"
        and manifest["candidate_id"] == CANDIDATE,
        "Unexpected curriculum preparation identity.",
    )
    expected = manifest["files_sha256"]
    allowed = (
        set(PAYLOADS)
        | set(EVIDENCE.values())
        | {
            "configuration.json",
            "draw-plan.json",
            "exposure-census.json",
            "smoke/manifest.json",
        }
        | {"source/" + name for name in SOURCE_FILES}
        | {"smoke/" + name for name in SMOKE_PAYLOADS}
    )
    require(
        set(expected) == allowed,
        "Only the registered train-only payload inventory is allowed.",
    )
    observed = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }
    require(
        observed == set(expected) | {"manifest.json"},
        "Bundle inventory differs from its manifest.",
    )
    contents = {
        name: read_bound(child(directory, name), value)
        for name, value in expected.items()
    }
    require(
        sha256(Path(__file__).read_bytes())
        == expected["source/prepare_passive_curriculum.py"]
        and sha256(Path(smoke.__file__).read_bytes())
        == expected["source/prepare_passive_sft.py"],
        "Loaded portable verifier code differs from its pinned snapshot.",
    )
    evidence = {}
    for name, filename in EVIDENCE.items():
        binding = manifest["bindings"][name]
        require(
            binding["path"] == filename and expected[filename] == binding["sha256"],
            "Evidence binding differs.",
        )
        evidence[name] = json.loads(contents[filename])
    release, selection, completion, completed_config = (
        evidence[key] for key in EVIDENCE
    )
    require(
        release["version"] == "passive-dataset-completion-v2"
        and release["status"] == "frozen_supported_scope"
        and manifest["release_manifest_sha256"] == expected[EVIDENCE["release"]]
        and selection["bindings"]["release_manifest_sha256"]
        == manifest["release_manifest_sha256"]
        and selection["selected_t1_candidate_ids"] == [CANDIDATE],
        "Selection and frozen release disagree.",
    )
    for name, original in PAYLOADS.items():
        require(
            expected[name] == release["files_sha256"][original],
            "A training payload differs from the frozen release.",
        )
    require(
        expected["source/passive_bundle_v2.py"]
        == release["dependencies_sha256"].get("ml/passive_bundle_v2.py"),
        "The weighted sampler source differs from the frozen release.",
    )
    smoke_binding = manifest["smoke_preparation"]
    require(
        smoke_binding["path"] == "smoke/manifest.json"
        and smoke_binding["sha256"]
        == expected[smoke_binding["path"]]
        == selection["bindings"]["train_only_preparation"]["sha256"],
        "Fixed smoke preparation differs from the selection.",
    )
    smoke_manifest = json.loads(contents[smoke_binding["path"]])
    require(
        set(smoke_manifest["files_sha256"]) == SMOKE_PAYLOADS,
        "Unexpected original smoke payload inventory.",
    )
    for name, value in smoke_manifest["files_sha256"].items():
        require(expected["smoke/" + name] == value, "A fixed smoke payload changed.")
    require(
        smoke_manifest["release_manifest_sha256"] == manifest["release_manifest_sha256"]
        and smoke_manifest["candidate_id"] == CANDIDATE,
        "The smoke diagnostics belong to another release or candidate.",
    )
    completed_binding = manifest["bindings"]["smoke_configuration"]
    require(
        completion["version"] == "passive-t1-feasibility-completion-v1"
        and completion["status"]
        == "passed_engineering_roundtrip_substantive_sft_preregistration_pending"
        and set(completion["gates"]) == SMOKE_GATES
        and all(value is True for value in completion["gates"].values())
        and completion["training"]["training_updates"] == 2
        and completion["training"]["unique_cases"] == 8
        and completion["files_sha256"][completed_binding["source_path"]]
        == completed_binding["sha256"]
        and completed_config["version"] == "passive-sft-smoke-v1"
        and completed_config["bindings"]["preparation"]["sha256"]
        == smoke_binding["sha256"],
        "The successful smoke round trip is not bound to these diagnostics.",
    )
    config = json.loads(contents["configuration.json"])
    prepared = json.loads(contents["smoke/configuration.json"])
    native = selection["selected_native_identity"]
    require(
        config["candidate_id"] == CANDIDATE
        and config["native_base"]["repository"]
        == native["base_repository"]
        == completion["candidate"]
        == completed_config["base"]["repository"]
        and config["native_base"]["revision"]
        == native["base_revision"]
        == completion["revision"]
        == completed_config["base"]["revision"]
        == manifest["native_revision"]
        and config["tokenizer"]["files_sha256"] == native["tokenizer_files_sha256"],
        "Native candidate/tokenizer identity differs.",
    )
    for name, original in PAYLOADS.items():
        relative = next(
            (
                path
                for path in smoke_manifest["inputs_sha256"]
                if path.endswith("/" + original)
            ),
            None,
        )
        require(
            relative and smoke_manifest["inputs_sha256"][relative] == expected[name],
            "Smoke and curriculum frozen input bindings differ.",
        )
    lines = {
        "cases": smoke.jsonl_lines(contents["train-cases.jsonl"]),
        "messages": smoke.jsonl_lines(contents["train-messages.jsonl"]),
        "tokens": smoke.jsonl_lines(contents["train-tokens.jsonl.gz"], compressed=True),
    }
    records = {
        name: [json.loads(line) for line in values] for name, values in lines.items()
    }
    cases, messages, tokens = (
        records[name] for name in ("cases", "messages", "tokens")
    )
    require(len(cases) == TRAIN_ROWS, "Use all 809 frozen training rows.")
    groups = json.loads(contents["source-groups.json"])
    smoke.validate_train(
        cases,
        messages,
        tokens,
        groups,
        contents["prompt-json.txt"].decode().removesuffix("\n"),
        prepared["tokenizer"]["eos_token_id"],
        prepared["tokenizer"]["vocab_size"],
    )
    for name in ("cases", "messages", "tokens", "inputs"):
        filename = (
            "smoke-model-inputs.jsonl" if name == "inputs" else f"smoke-{name}.jsonl"
        )
        records["smoke_" + name] = [
            json.loads(line)
            for line in smoke.jsonl_lines(contents["smoke/" + filename])
        ]
    selected = smoke.select_smoke(cases, tokens)
    require(
        len(selected) == 8
        and prepared["proposed_smoke"]["order"]
        == [cases[index]["id"] for index in selected]
        == completed_config["training"]["sample_order"],
        "Fixed train diagnostic selection changed.",
    )
    for name in ("cases", "messages", "tokens"):
        require(
            contents[f"smoke/smoke-{name}.jsonl"]
            == b"".join(lines[name][i] for i in selected),
            "A fixed diagnostic row is not byte-identical to frozen training.",
        )
    require(
        records["smoke_inputs"]
        == [
            {
                key: tokens[i]["primary"][key]
                for key in ("input_ids", "attention_mask", "labels")
            }
            for i in selected
        ],
        "Fixed smoke model inputs changed.",
    )
    sampling = config["sampling"]
    require(
        sampling["draw_count"] == TRAIN_ROWS and sampling["seed_type"] == "str",
        "Use exactly 809 draws and a string sampling seed.",
    )
    indices = weighted_draw_indices(cases, TRAIN_ROWS, sampling["seed"])
    plan = [
        {
            "draw_index": position,
            "case_id": cases[index]["id"],
            "train_row_index": index,
        }
        for position, index in enumerate(indices)
    ]
    require(
        json.loads(contents["draw-plan.json"]) == plan,
        "Frozen weighted draw order changed.",
    )
    require(
        config == configuration(prepared, cases, tokens, indices, sampling["seed"]),
        "Curriculum configuration or token/update budget changed.",
    )
    realized = exposure(cases, tokens, indices)
    require(
        json.loads(contents["exposure-census.json"])
        == {"available_train": smoke.census(cases, tokens), "fixed_stream": realized},
        "Curriculum exposure census changed.",
    )
    inventory = [
        {"path": name, "bytes": len(content), "sha256": expected[name]}
        for name, content in sorted(contents.items())
    ]
    require(
        manifest["inventory"] == inventory
        and manifest["payload_bytes"] == sum(row["bytes"] for row in inventory),
        "Portable byte inventory changed.",
    )
    return {
        "manifest": manifest,
        "configuration": config,
        **records,
        "draw_plan": plan,
        "smoke_configuration": prepared,
        "verification": {
            "status": "passed_train_only_curriculum_readback",
            "train_rows": len(cases),
            "draws": len(plan),
            "unique_drawn_cases": realized["unique_cases"],
            "repeat_draws": realized["repeat_draws"],
            "sequence_tokens": realized["sequence_tokens"],
            "completion_tokens": realized["completion_tokens"],
            "optimizer_updates": 203,
            "final_update_rows": 1,
            "payload_files": len(expected),
            "payload_bytes": manifest["payload_bytes"],
            "dev_test_record_bodies_read": 0,
            "training_updates_executed": 0,
        },
    }


def prepare(
    release,
    selection,
    smoke_completion,
    smoke_preparation,
    output,
    *,
    release_sha256,
    selection_sha256,
    smoke_completion_sha256,
    seed,
    project_root=ROOT,
):
    root = Path(project_root).resolve()
    release, selection, smoke_completion, smoke_preparation = (
        Path(p).resolve()
        for p in (release, selection, smoke_completion, smoke_preparation)
    )
    output = Path(output).absolute()
    require(
        not output.exists() and not output.is_symlink(), "Use a fresh output directory."
    )
    output = output.resolve()
    require(
        not any(p.name == "frozen" for p in (output, *output.parents)),
        "Never publish inside a frozen release.",
    )
    require(
        not output.is_relative_to(release)
        and not output.is_relative_to(smoke_preparation),
        "Never publish inside a frozen input or original smoke preparation.",
    )
    require(
        all(
            p.is_relative_to(root)
            for p in (release, selection, smoke_completion, smoke_preparation)
        ),
        "Inputs must stay within the project root.",
    )
    require(type(seed) is str and seed, "Sampling seed must be an explicit string.")
    inputs = {}

    def bound(path, expected):
        path = Path(path).resolve()
        require(path.is_relative_to(root), "Input escapes the project root.")
        content = read_bound(path, expected)
        inputs[str(path.relative_to(root))] = expected
        return content

    files = {
        EVIDENCE["release"]: bound(release / "manifest.json", release_sha256),
        EVIDENCE["selection"]: bound(selection, selection_sha256),
        EVIDENCE["smoke_completion"]: bound(smoke_completion, smoke_completion_sha256),
    }
    released, selected, completion = (
        json.loads(files[EVIDENCE[key]])
        for key in ("release", "selection", "smoke_completion")
    )
    prep_hash = selected["bindings"]["train_only_preparation"]["sha256"]
    files["smoke/manifest.json"] = bound(smoke_preparation / "manifest.json", prep_hash)
    prep_manifest = json.loads(files["smoke/manifest.json"])
    require(
        set(prep_manifest["files_sha256"]) == SMOKE_PAYLOADS,
        "Unexpected original smoke payload inventory.",
    )
    for name, expected in prep_manifest["files_sha256"].items():
        files["smoke/" + name] = bound(child(smoke_preparation, name), expected)
    config_names = [
        name
        for name in completion["files_sha256"]
        if name.startswith("input/configuration") and name.endswith(".json")
    ]
    require(
        len(config_names) == 1,
        "Successful smoke must bind one exact run configuration.",
    )
    completed_name = config_names[0]
    files[EVIDENCE["smoke_configuration"]] = bound(
        child(smoke_completion.parent, completed_name),
        completion["files_sha256"][completed_name],
    )
    for name, original in PAYLOADS.items():
        files[name] = bound(release / original, released["files_sha256"][original])
    cases = [json.loads(line) for line in smoke.jsonl_lines(files["train-cases.jsonl"])]
    tokens = [
        json.loads(line)
        for line in smoke.jsonl_lines(files["train-tokens.jsonl.gz"], compressed=True)
    ]
    require(len(cases) == TRAIN_ROWS, "Use exactly 809 frozen train rows.")
    indices = weighted_draw_indices(cases, TRAIN_ROWS, seed)
    from passive_bundle_v2 import weighted_indices

    require(
        indices == weighted_indices(cases, TRAIN_ROWS, seed),
        "Portable sampler differs from the frozen weighted_indices semantics.",
    )
    prepared = json.loads(files["smoke/configuration.json"])
    files["configuration.json"] = json_bytes(
        configuration(prepared, cases, tokens, indices, seed)
    )
    files["draw-plan.json"] = json_bytes(
        [
            {
                "draw_index": position,
                "case_id": cases[index]["id"],
                "train_row_index": index,
            }
            for position, index in enumerate(indices)
        ]
    )
    files["exposure-census.json"] = json_bytes(
        {
            "available_train": smoke.census(cases, tokens),
            "fixed_stream": exposure(cases, tokens, indices),
        }
    )
    for name in SOURCE_FILES:
        path = Path(__file__).with_name(name)
        content = path.read_bytes()
        files["source/" + name] = content
        inputs[str(path.resolve().relative_to(root))] = sha256(content)
    require(
        files["source/prepare_passive_sft.py"]
        == files["smoke/source/prepare_passive_sft.py"],
        "Original preparation validator has changed.",
    )
    bindings = {
        name: {"path": filename, "sha256": sha256(files[filename])}
        for name, filename in EVIDENCE.items()
    }
    bindings["smoke_configuration"]["source_path"] = completed_name
    manifest = {
        "version": VERSION,
        "status": "frozen_train_only_preparation",
        "candidate_id": CANDIDATE,
        "native_revision": selected["selected_native_identity"]["base_revision"],
        "release_manifest_sha256": release_sha256,
        "bindings": bindings,
        "smoke_preparation": {"path": "smoke/manifest.json", "sha256": prep_hash},
        "inputs_sha256": inputs,
        "files_sha256": {
            name: sha256(content) for name, content in sorted(files.items())
        },
        "inventory": [
            {"path": name, "bytes": len(content), "sha256": sha256(content)}
            for name, content in sorted(files.items())
        ],
        "payload_bytes": sum(map(len, files.values())),
        "portable_dependencies": [
            "Python standard library",
            "source/prepare_passive_sft.py",
        ],
        "sampler_provenance": "Locally compared with ml/passive_bundle_v2.py weighted_indices; portable replay uses identical cumulative summation, bisect_right and random.Random(str seed).",
        "training_run": False,
        "dev_test_record_bodies_copied": 0,
        "verification_scope": "Exact frozen train bytes, source split membership, masks/references/messages, fixed diagnostics, draw replay and token/update/exposure accounting. No retokenization, new semantic audit or model execution.",
        "preparation_runtime": {"python": sys.version},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent)
    )
    try:
        for name, content in files.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
        verified = verify_bundle(staging)["verification"]
        for name, expected in inputs.items():
            read_bound(root / name, expected)
        require(
            not output.exists(), "Publication destination appeared during validation."
        )
        staging.rename(output)
        return {
            **verified,
            "directory": str(output),
            "manifest_sha256": sha256((output / "manifest.json").read_bytes()),
        }
    except BaseException as error:
        if staging.exists():
            with (staging / "failure.json").open("xb") as stream:
                stream.write(
                    json_bytes(
                        {
                            "status": "failed_before_publication",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    for name in (
        "release",
        "selection",
        "smoke-completion",
        "smoke-preparation",
        "output",
    ):
        build.add_argument("--" + name, type=Path, required=True)
    for name in (
        "release-sha256",
        "selection-sha256",
        "smoke-completion-sha256",
        "seed",
    ):
        build.add_argument("--" + name, required=True)
    check = sub.add_parser("verify")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_bundle(args.directory)["verification"]
    else:
        result = prepare(
            args.release,
            args.selection,
            args.smoke_completion,
            args.smoke_preparation,
            args.output,
            release_sha256=args.release_sha256,
            selection_sha256=args.selection_sha256,
            smoke_completion_sha256=args.smoke_completion_sha256,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
