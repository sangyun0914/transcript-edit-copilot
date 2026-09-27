"""Freeze matched JSON/Rewrite training views with one shared weighted stream."""

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import prepare_passive_curriculum as curriculum
import prepare_passive_sft as smoke

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-sft-matched-preparation-v1"
FORMATS = ("json", "rewrite")
TRAIN_ROWS = 802
PAYLOADS = {
    "train-cases.jsonl": "matched/train-cases.jsonl",
    "json-train-tokens.jsonl.gz": "tokenized/qwen35-9b/json-train.jsonl.gz",
    "rewrite-train-tokens.jsonl.gz": "tokenized/qwen35-9b/rewrite-train.jsonl.gz",
    "source-groups.json": "source-groups.json",
    "prompt-json.txt": "prompt-json.txt",
    "prompt-rewrite.txt": "prompt-rewrite.txt",
    "matched-eligibility.json": "matched/eligibility.json",
}
SOURCES = (
    "prepare_passive_matched.py",
    "test_prepare_passive_matched.py",
    "prepare_passive_curriculum.py",
    "prepare_passive_sft.py",
    "passive_bundle_v2.py",
    "passive_formats_v2.py",
)
require = smoke.require
sha256 = smoke.sha256
json_bytes = smoke.json_bytes
child = curriculum.child
read_bound = curriculum.read_bound


def exported_messages(tokens):
    return [
        {
            "id": row["id"],
            "sampling_weight": row["sampling_weight"],
            "messages": [
                *row["messages"],
                {"role": "assistant", "content": row["references"][0]},
            ],
        }
        for row in tokens
    ]


def selected_diagnostics(cases, views):
    choices = defaultdict(list)
    for index, case in enumerate(cases):
        cell = (case["expectedOutput"]["action"], case["metadata"]["language"])
        length = max(len(views[fmt][index]["primary"]["input_ids"]) for fmt in FORMATS)
        choices[cell].append((-length, case["id"], index))
    require(
        set(choices) == set(smoke.SLOTS),
        "Missing matched action/language diagnostic cell.",
    )
    return [min(choices[cell])[2] for cell in smoke.SLOTS]


def validate_views(cases, views, groups, prompts, tokenizer):
    require(
        len(cases) == TRAIN_ROWS
        and all(len(views[fmt]) == TRAIN_ROWS for fmt in FORMATS),
        "Both matched views must contain the same 802 training rows.",
    )
    smoke.validate_train(
        cases,
        exported_messages(views["json"]),
        views["json"],
        groups,
        prompts["json"],
        tokenizer["eos_token_id"],
        tokenizer["vocab_size"],
    )
    eos, vocab = tokenizer["eos_token_id"], tokenizer["vocab_size"]
    for case, direct, rewrite in zip(
        cases, views["json"], views["rewrite"], strict=True
    ):
        require(
            set(rewrite) == set(direct)
            and all(
                rewrite[key] == direct[key]
                for key in ("id", "split", "sampling_weight", "input_sha256")
            )
            and rewrite["messages"]
            == [
                {"role": "system", "content": prompts["rewrite"]},
                direct["messages"][1],
            ]
            and len(rewrite["references"]) == len(direct["references"]),
            "Matched identity, evidence, weight or reference census differs.",
        )
        for text in rewrite["references"]:
            reference = json.loads(text)
            require(
                isinstance(reference, dict)
                and set(reference) == {"format_version", "scope", "after"}
                and type(reference["format_version"]) is int
                and reference["format_version"] == 1
                and isinstance(reference["scope"], list)
                and isinstance(reference["after"], list),
                "Unexpected frozen scoped Rewrite envelope.",
            )
        values = rewrite["primary"]
        require(
            set(values)
            == {"input_ids", "attention_mask", "labels", "completion_start"},
            "Unexpected Rewrite training fields.",
        )
        ids, prefix = values["input_ids"], values["completion_start"]
        require(
            type(prefix) is int
            and 0 < prefix <= 3584
            and 0 < len(ids) - prefix <= 512
            and len(ids) <= 4096
            and all(type(value) is int and 0 <= value < vocab for value in ids)
            and ids[-1] == eos
            and values["labels"] == [-100] * prefix + ids[prefix:]
            and all(type(value) is int for value in values["labels"])
            and values["attention_mask"] == [1] * len(ids)
            and all(type(value) is int for value in values["attention_mask"]),
            "Invalid Rewrite completion mask, attention, token bounds or EOS.",
        )
        require(
            len(rewrite["alternative_completion_ids"])
            == len(case["acceptableOutputs"]),
            "Rewrite alternative-token census differs.",
        )
        for completion in rewrite["alternative_completion_ids"]:
            require(
                0 < len(completion) <= 512
                and prefix + len(completion) <= 4096
                and completion[-1] == eos
                and all(
                    type(value) is int and 0 <= value < vocab for value in completion
                ),
                "Invalid Rewrite alternative completion.",
            )


def semantic_readback(cases, views):
    from passive_formats_v2 import messages, parse, serialize

    references = 0
    for case in cases:
        references += 1 + len(case["acceptableOutputs"])
    for fmt in FORMATS:
        for case, row in zip(cases, views[fmt], strict=True):
            actions = [case["expectedOutput"], *case["acceptableOutputs"]]
            require(
                row["messages"] == messages(case["input"], fmt),
                "Frozen format input no longer reproduces canonical evidence.",
            )
            require(
                row["references"]
                == [serialize(case["input"], action, fmt) for action in actions],
                "Frozen format references differ from the admitted actions.",
            )
            require(
                [
                    parse(text, case["input"], fmt).model_dump()
                    for text in row["references"]
                ]
                == actions,
                "JSON/Rewrite semantic roundtrip failed.",
            )
    return {
        "status": "passed_local_paired_reference_readback",
        "train_cases": len(cases),
        "references_per_format": references,
        "format_reference_checks": 2 * references,
        "canonical_message_checks": 2 * len(cases),
        "training_primary_targets_only": True,
        "retokenized": False,
        "dev_test_bodies_read": 0,
        "scope": "Locally reconstructed canonical messages, exact serialized alternatives and parsed admitted actions. Portable readback binds these checks and exact frozen bytes; the trainer separately verifies actual tokenizer IDs.",
    }


def configuration(original, cases, views, indices, selected, seed):
    result = {
        "version": VERSION,
        "status": "frozen_matched_train_only_preparation",
        "candidate_id": curriculum.CANDIDATE,
        "native_base": original["native_base"],
        "tokenizer": original["tokenizer"],
        "frozen_train_rows": len(cases),
        "formats": list(FORMATS),
        "loss_mask": "completion_only_including_eos",
        "input_limit": 3584,
        "output_limit_each_format": 512,
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
            "sample_order": [cases[i]["id"] for i in indices],
            "initialization": "fresh_adapter_on_exact_native_base_not_smoke_or_T1_adapter",
        },
        "views": {},
        "diagnostic_selection": "One per action/language cell; longest maximum primary sequence across JSON and Rewrite, then lexicographic case ID. These are shared train-only engineering diagnostics.",
        "original_smoke": {
            "path": "original-smoke/manifest.json",
            "used_for_matched_diagnostics": False,
            "reason": "Two original smoke states are outside the frozen common matched set; the original preparation remains unchanged as provenance.",
        },
        "training_run": False,
        "dev_test_record_bodies_copied": 0,
        "comparison_scope": "Both formats use the same source evidence, admitted actions and weighted case order. Sequence and completion-token exposure differs by representation and is reported separately. T1 full curriculum is not the matched JSON control.",
    }
    for fmt in FORMATS:
        planned = curriculum.exposure(cases, views[fmt], indices)
        diagnostics = curriculum.exposure(cases, views[fmt], selected)
        result["views"][fmt] = {
            "training_view": fmt,
            "max_total_sequence_tokens": planned["sequence_tokens"],
            "max_total_completion_tokens": planned["completion_tokens"],
            "diagnostic_case_order": [cases[i]["id"] for i in selected],
            "diagnostic_sequence_tokens": diagnostics["sequence_tokens"],
            "diagnostic_completion_tokens": diagnostics["completion_tokens"],
        }
    return result


def expected_names():
    return (
        set(PAYLOADS)
        | set(curriculum.EVIDENCE.values())
        | {
            "configuration.json",
            "draw-plan.json",
            "exposure-census.json",
            "semantic-readback.json",
            "original-smoke/manifest.json",
            "diagnostics/cases.jsonl",
        }
        | {"source/" + name for name in SOURCES}
        | {"original-smoke/" + name for name in curriculum.SMOKE_PAYLOADS}
        | {
            f"diagnostics/{fmt}-{kind}.jsonl"
            for fmt in FORMATS
            for kind in ("messages", "tokens", "inputs")
        }
    )


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_bytes())
    require(
        manifest["version"] == VERSION
        and manifest["status"] == "frozen_matched_train_only_preparation"
        and manifest["candidate_id"] == curriculum.CANDIDATE,
        "Unexpected matched preparation identity.",
    )
    expected = manifest["files_sha256"]
    require(
        set(expected) == expected_names(),
        "Only the registered matched train-only inventory is allowed.",
    )
    require(
        {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file()
        }
        == set(expected) | {"manifest.json"},
        "Matched bundle inventory changed.",
    )
    files = {
        name: read_bound(child(directory, name), value)
        for name, value in expected.items()
    }
    for module, name in (
        (sys.modules[__name__], "prepare_passive_matched.py"),
        (curriculum, "prepare_passive_curriculum.py"),
        (smoke, "prepare_passive_sft.py"),
    ):
        require(
            sha256(Path(module.__file__).read_bytes()) == expected["source/" + name],
            "Loaded matched verifier code differs from its pinned source.",
        )
    evidence = {}
    for name, filename in curriculum.EVIDENCE.items():
        binding = manifest["bindings"][name]
        require(
            binding["path"] == filename and binding["sha256"] == expected[filename],
            "Evidence binding changed.",
        )
        evidence[name] = json.loads(files[filename])
    release, selection, completion, completed_config = (
        evidence[name] for name in curriculum.EVIDENCE
    )
    require(
        release["version"] == "passive-dataset-completion-v2"
        and release["status"] == "frozen_supported_scope"
        and manifest["release_manifest_sha256"]
        == expected[curriculum.EVIDENCE["release"]]
        == selection["bindings"]["release_manifest_sha256"]
        and selection["selected_t1_candidate_ids"] == [curriculum.CANDIDATE],
        "Frozen release and selected native identity disagree.",
    )
    for name, original in PAYLOADS.items():
        require(
            expected[name] == release["files_sha256"][original],
            "Matched data differs from its exact frozen payload.",
        )
    for name in ("passive_bundle_v2.py", "passive_formats_v2.py"):
        require(
            expected["source/" + name] == release["dependencies_sha256"]["ml/" + name],
            "Frozen sampler or format implementation changed.",
        )
    original_binding = manifest["original_smoke_preparation"]
    require(
        original_binding["path"] == "original-smoke/manifest.json"
        and original_binding["sha256"]
        == expected[original_binding["path"]]
        == selection["bindings"]["train_only_preparation"]["sha256"],
        "Original smoke preparation changed.",
    )
    original_manifest = json.loads(files[original_binding["path"]])
    require(
        set(original_manifest["files_sha256"]) == curriculum.SMOKE_PAYLOADS
        and original_manifest["candidate_id"] == curriculum.CANDIDATE
        and original_manifest["release_manifest_sha256"]
        == manifest["release_manifest_sha256"],
        "Unexpected original smoke identity.",
    )
    for name, value in original_manifest["files_sha256"].items():
        require(
            expected["original-smoke/" + name] == value,
            "Original smoke payload changed.",
        )
    require(
        files["source/prepare_passive_sft.py"]
        == files["original-smoke/source/prepare_passive_sft.py"],
        "Original preparation validator changed.",
    )
    original = json.loads(files["original-smoke/configuration.json"])
    native = selection["selected_native_identity"]
    completed_binding = manifest["bindings"]["smoke_configuration"]
    require(
        completion["version"] == "passive-t1-feasibility-completion-v1"
        and completion["status"]
        == "passed_engineering_roundtrip_substantive_sft_preregistration_pending"
        and set(completion["gates"]) == curriculum.SMOKE_GATES
        and all(value is True for value in completion["gates"].values())
        and completion["training"]["training_updates"] == 2
        and completion["training"]["unique_cases"] == 8
        and completion["files_sha256"][completed_binding["source_path"]]
        == completed_binding["sha256"]
        and completed_config["version"] == "passive-sft-smoke-v1"
        and completed_config["bindings"]["preparation"]["sha256"]
        == original_binding["sha256"],
        "The successful engineering smoke is not bound to these inputs.",
    )
    require(
        original["native_base"]["repository"]
        == native["base_repository"]
        == completion["candidate"]
        == completed_config["base"]["repository"]
        and original["native_base"]["revision"]
        == native["base_revision"]
        == completion["revision"]
        == completed_config["base"]["revision"]
        == manifest["native_revision"]
        and original["tokenizer"]["files_sha256"] == native["tokenizer_files_sha256"],
        "Matched native/tokenizer identity changed.",
    )
    case_lines = smoke.jsonl_lines(files["train-cases.jsonl"])
    cases = [json.loads(line) for line in case_lines]
    token_lines = {
        fmt: smoke.jsonl_lines(files[f"{fmt}-train-tokens.jsonl.gz"], compressed=True)
        for fmt in FORMATS
    }
    tokens = {fmt: [json.loads(line) for line in token_lines[fmt]] for fmt in FORMATS}
    prompts = {
        fmt: files[f"prompt-{fmt}.txt"].decode().removesuffix("\n") for fmt in FORMATS
    }
    validate_views(
        cases,
        tokens,
        json.loads(files["source-groups.json"]),
        prompts,
        original["tokenizer"],
    )
    require(
        json.loads(files["matched-eligibility.json"])["qualified"]["train"]
        == TRAIN_ROWS,
        "Matched qualification census changed.",
    )
    config = json.loads(files["configuration.json"])
    seed = config["sampling"]["seed"]
    indices = curriculum.weighted_draw_indices(cases, TRAIN_ROWS, seed)
    selected = selected_diagnostics(cases, tokens)
    plan = [
        {
            "draw_index": position,
            "case_id": cases[index]["id"],
            "train_row_index": index,
        }
        for position, index in enumerate(indices)
    ]
    require(
        json.loads(files["draw-plan.json"]) == plan,
        "The shared weighted case order changed.",
    )
    require(
        config == configuration(original, cases, tokens, indices, selected, seed),
        "Matched budgets or diagnostic configuration changed.",
    )
    exposure = {
        fmt: {
            "available_train": smoke.census(cases, tokens[fmt]),
            "fixed_stream": curriculum.exposure(cases, tokens[fmt], indices),
        }
        for fmt in FORMATS
    }
    require(
        json.loads(files["exposure-census.json"]) == exposure,
        "Matched exposure census changed.",
    )
    semantic = json.loads(files["semantic-readback.json"])
    references = sum(1 + len(case["acceptableOutputs"]) for case in cases)
    require(
        semantic["status"] == "passed_local_paired_reference_readback"
        and semantic["train_cases"] == TRAIN_ROWS
        and semantic["references_per_format"] == references
        and semantic["format_reference_checks"] == 2 * references
        and semantic["canonical_message_checks"] == 2 * TRAIN_ROWS
        and semantic["dev_test_bodies_read"] == 0,
        "Paired reference readback receipt changed.",
    )
    require(
        files["diagnostics/cases.jsonl"]
        == b"".join(case_lines[index] for index in selected),
        "Matched diagnostic case bytes changed.",
    )
    views = {}
    for fmt in FORMATS:
        messages = exported_messages(tokens[fmt])
        diagnostic_tokens = [tokens[fmt][index] for index in selected]
        diagnostic_messages = [messages[index] for index in selected]
        diagnostic_inputs = [
            {
                key: row["primary"][key]
                for key in ("input_ids", "attention_mask", "labels")
            }
            for row in diagnostic_tokens
        ]
        require(
            files[f"diagnostics/{fmt}-tokens.jsonl"]
            == b"".join(token_lines[fmt][index] for index in selected),
            "Matched diagnostic token bytes changed.",
        )
        require(
            files[f"diagnostics/{fmt}-messages.jsonl"]
            == smoke.jsonl_bytes(diagnostic_messages)
            and files[f"diagnostics/{fmt}-inputs.jsonl"]
            == smoke.jsonl_bytes(diagnostic_inputs),
            "Matched diagnostic messages or masks changed.",
        )
        views[fmt] = {
            "messages": messages,
            "tokens": tokens[fmt],
            "diagnostics": {
                "cases": [cases[index] for index in selected],
                "messages": diagnostic_messages,
                "tokens": diagnostic_tokens,
                "inputs": diagnostic_inputs,
                "rows": [
                    {"case_id": row["id"], **row["primary"]}
                    for row in diagnostic_tokens
                ],
            },
        }
    inventory = [
        {"path": name, "bytes": len(content), "sha256": expected[name]}
        for name, content in sorted(files.items())
    ]
    require(
        manifest["inventory"] == inventory
        and manifest["payload_bytes"] == sum(row["bytes"] for row in inventory),
        "Matched byte inventory changed.",
    )
    return {
        "manifest": manifest,
        "configuration": config,
        "cases": cases,
        "draw_plan": plan,
        "original_smoke_configuration": original,
        "views": views,
        "verification": {
            "status": "passed_matched_train_only_readback",
            "train_rows": TRAIN_ROWS,
            "shared_draws_per_format": len(plan),
            "unique_drawn_cases": len(set(indices)),
            "repeat_draws": len(indices) - len(set(indices)),
            "optimizer_updates_per_format": config["training"]["optimizer_updates"],
            "final_update_rows": config["training"]["final_update_rows"],
            "views": {fmt: config["views"][fmt] for fmt in FORMATS},
            "payload_files": len(files),
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
        Path(value).resolve()
        for value in (release, selection, smoke_completion, smoke_preparation)
    )
    output = Path(output).absolute()
    require(
        not output.exists() and not output.is_symlink(),
        "Use a fresh matched output directory.",
    )
    output = output.resolve()
    require(
        not output.is_relative_to(release)
        and not output.is_relative_to(smoke_preparation)
        and not any(path.name == "frozen" for path in (output, *output.parents)),
        "Never publish inside frozen or original smoke inputs.",
    )
    require(type(seed) is str and seed, "Use an explicit string sampling seed.")
    inputs = {}

    def bound(path, expected):
        path = Path(path).resolve()
        require(path.is_relative_to(root), "Input escapes the project root.")
        content = read_bound(path, expected)
        inputs[str(path.relative_to(root))] = expected
        return content

    files = {
        curriculum.EVIDENCE["release"]: bound(
            release / "manifest.json", release_sha256
        ),
        curriculum.EVIDENCE["selection"]: bound(selection, selection_sha256),
        curriculum.EVIDENCE["smoke_completion"]: bound(
            smoke_completion, smoke_completion_sha256
        ),
    }
    released, selected, completion = (
        json.loads(files[curriculum.EVIDENCE[key]])
        for key in ("release", "selection", "smoke_completion")
    )
    prep_sha = selected["bindings"]["train_only_preparation"]["sha256"]
    files["original-smoke/manifest.json"] = bound(
        smoke_preparation / "manifest.json", prep_sha
    )
    original_manifest = json.loads(files["original-smoke/manifest.json"])
    require(
        set(original_manifest["files_sha256"]) == curriculum.SMOKE_PAYLOADS,
        "Unexpected original smoke inventory.",
    )
    for name, value in original_manifest["files_sha256"].items():
        files["original-smoke/" + name] = bound(child(smoke_preparation, name), value)
    names = [
        name
        for name in completion["files_sha256"]
        if name.startswith("input/configuration") and name.endswith(".json")
    ]
    require(len(names) == 1, "Successful smoke must bind one run configuration.")
    completed_name = names[0]
    files[curriculum.EVIDENCE["smoke_configuration"]] = bound(
        child(smoke_completion.parent, completed_name),
        completion["files_sha256"][completed_name],
    )
    for name, original in PAYLOADS.items():
        files[name] = bound(release / original, released["files_sha256"][original])
    cases_lines = smoke.jsonl_lines(files["train-cases.jsonl"])
    cases = [json.loads(line) for line in cases_lines]
    token_lines = {
        fmt: smoke.jsonl_lines(files[f"{fmt}-train-tokens.jsonl.gz"], compressed=True)
        for fmt in FORMATS
    }
    views = {fmt: [json.loads(line) for line in token_lines[fmt]] for fmt in FORMATS}
    original = json.loads(files["original-smoke/configuration.json"])
    validate_views(
        cases,
        views,
        json.loads(files["source-groups.json"]),
        {
            fmt: files[f"prompt-{fmt}.txt"].decode().removesuffix("\n")
            for fmt in FORMATS
        },
        original["tokenizer"],
    )
    indices = curriculum.weighted_draw_indices(cases, TRAIN_ROWS, seed)
    from passive_bundle_v2 import weighted_indices

    require(
        indices == weighted_indices(cases, TRAIN_ROWS, seed),
        "Shared sampler differs from the frozen implementation.",
    )
    files["semantic-readback.json"] = json_bytes(semantic_readback(cases, views))
    selected = selected_diagnostics(cases, views)
    files["configuration.json"] = json_bytes(
        configuration(original, cases, views, indices, selected, seed)
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
            fmt: {
                "available_train": smoke.census(cases, views[fmt]),
                "fixed_stream": curriculum.exposure(cases, views[fmt], indices),
            }
            for fmt in FORMATS
        }
    )
    files["diagnostics/cases.jsonl"] = b"".join(
        cases_lines[index] for index in selected
    )
    for fmt in FORMATS:
        diagnostic_tokens = [views[fmt][index] for index in selected]
        files[f"diagnostics/{fmt}-tokens.jsonl"] = b"".join(
            token_lines[fmt][index] for index in selected
        )
        files[f"diagnostics/{fmt}-messages.jsonl"] = smoke.jsonl_bytes(
            exported_messages(diagnostic_tokens)
        )
        files[f"diagnostics/{fmt}-inputs.jsonl"] = smoke.jsonl_bytes(
            {
                key: row["primary"][key]
                for key in ("input_ids", "attention_mask", "labels")
            }
            for row in diagnostic_tokens
        )
    for name in SOURCES:
        path = Path(__file__).with_name(name)
        files["source/" + name] = path.read_bytes()
        inputs[str(path.resolve().relative_to(root))] = sha256(files["source/" + name])
    bindings = {
        name: {"path": filename, "sha256": sha256(files[filename])}
        for name, filename in curriculum.EVIDENCE.items()
    }
    bindings["smoke_configuration"]["source_path"] = completed_name
    manifest = {
        "version": VERSION,
        "status": "frozen_matched_train_only_preparation",
        "candidate_id": curriculum.CANDIDATE,
        "native_revision": original["native_base"]["revision"],
        "release_manifest_sha256": release_sha256,
        "bindings": bindings,
        "original_smoke_preparation": {
            "path": "original-smoke/manifest.json",
            "sha256": prep_sha,
        },
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
            "source/prepare_passive_curriculum.py",
            "source/prepare_passive_sft.py",
        ],
        "training_run": False,
        "dev_test_record_bodies_copied": 0,
        "preparation_runtime": {"python": sys.version},
        "scope": "Matched 802-state primary-target format comparison preparation. Eventual T2 launch must separately bind completed substantive T1 and its fixed experiment protocol. Neither format may inherit the learned T1/smoke adapter.",
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
        result = verify_bundle(staging)["verification"]
        for name, value in inputs.items():
            read_bound(root / name, value)
        require(
            not output.exists(), "Matched publication path appeared during validation."
        )
        staging.rename(output)
        return {
            **result,
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
    result = (
        verify_bundle(args.directory)["verification"]
        if args.command == "verify"
        else prepare(
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
