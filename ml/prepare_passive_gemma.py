"""Freeze Gemma's train-only smoke and fixed 809-draw curriculum data."""

import argparse
import json
import tempfile
from pathlib import Path

import prepare_passive_curriculum as curriculum
import prepare_passive_sft as smoke

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-sft-gemma-preparation-v1"
STATUS = "frozen_train_only_preparation"
CANDIDATE = "gemma26"
REPOSITORY = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
RELEASE_SHA256 = "15d4d27d4ce1fde74a4cd993f17362c7061aa115ec7547d3bf60b0a99c22c222"
QWEN_MANIFEST_SHA256 = (
    "6810caf2a2ac377bdc999e0fa16485d28a6399b0f9511e2326d7c79607828a5e"
)
FROZEN_HELPERS = {
    "prepare_passive_curriculum.py": "433e7835f5ab2c417217ebc9a11ae46ef0a2b4ab67756e01570c27898f5a541c",
    "prepare_passive_sft.py": "73431b13981cf1a063f1e8f9c8481b248e017728db67869babe35cc86a26f5f1",
    "passive_bundle_v2.py": "2a0a4b94f5480df6149f32e9dd79f36a2e330026e94562c61082c932c99eb0f5",
}
SOURCE_FILES = (
    "prepare_passive_gemma.py",
    "test_prepare_passive_gemma.py",
    *FROZEN_HELPERS,
)
PAYLOADS = {
    "train-cases.jsonl": "train-cases.jsonl",
    "train-messages.jsonl": "train-messages.jsonl",
    "train-tokens.jsonl.gz": "tokenized/gemma26/json-curriculum-train.jsonl.gz",
    "source-groups.json": "source-groups.json",
    "prompt-json.txt": "prompt-json.txt",
}
EVIDENCE = {
    "release": "evidence/release-manifest.json",
    "model_registry": "evidence/model-registry.json",
    "tokenizer_registry": "evidence/tokenizer-registry.json",
    "qwen_preparation": "evidence/qwen-preparation-manifest.json",
    "qwen_draw_plan": "evidence/qwen-draw-plan.json",
}
TOKENIZER_FILES = {
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
}
METADATA_FILES = TOKENIZER_FILES - {"tokenizer.json"}
ALLOWED = (
    set(PAYLOADS)
    | set(EVIDENCE.values())
    | {"source/" + name for name in SOURCE_FILES}
    | {"native-metadata/" + name for name in METADATA_FILES}
    | {
        "configuration.json",
        "smoke-configuration.json",
        "draw-plan.json",
        "exposure-census.json",
        "smoke-cases.jsonl",
        "smoke-messages.jsonl",
        "smoke-tokens.jsonl",
        "smoke-model-inputs.jsonl",
    }
)
require = smoke.require
sha256 = smoke.sha256
read_bound = curriculum.read_bound
child = curriculum.child


def native_identity(candidate):
    return {
        "repository": candidate["base_repository"],
        "revision": candidate["base_revision"],
        "weights_verified_by_this_preparation": False,
    }


def tokenizer_identity(candidate):
    return {
        "id": CANDIDATE,
        "files_sha256": candidate["tokenizer_files_sha256"],
        "eos_token_id": 1,
        "vocab_size": 262144,
    }


def smoke_configuration(candidate, cases, tokens, selected):
    realized = curriculum.exposure(cases, tokens, selected)
    return {
        "version": VERSION,
        "status": "planned_smoke_not_executed",
        "candidate_id": CANDIDATE,
        "native_base": native_identity(candidate),
        "tokenizer": tokenizer_identity(candidate),
        "proposed_smoke": {
            "micro_batch_size": 1,
            "gradient_accumulation": 4,
            "optimizer_updates": 2,
            "final_update_rows": 4,
            "rows": 8,
            "order": [cases[index]["id"] for index in selected],
            "max_total_sequence_tokens": realized["sequence_tokens"],
            "max_total_completion_tokens": realized["completion_tokens"],
            "sampling": "Each selected row once; no resampling or loss weighting.",
        },
        "selection_rule": "One primary-action/language cell; longest complete sequence, then lexicographic case ID.",
        "selection": [
            {
                "case_id": cases[index]["id"],
                "train_row_index": index,
                "action": cases[index]["expectedOutput"]["action"],
                "language": cases[index]["metadata"]["language"],
                "sequence_tokens": len(tokens[index]["primary"]["input_ids"]),
                "prompt_tokens": tokens[index]["primary"]["completion_start"],
                "completion_tokens": len(tokens[index]["primary"]["input_ids"])
                - tokens[index]["primary"]["completion_start"],
            }
            for index in selected
        ],
        "scope": "Train-only engineering diagnostics; no held-out quality or completed training claim.",
    }


def configuration(candidate, cases, tokens, indices, selected):
    prepared = smoke_configuration(candidate, cases, tokens, selected)
    value = curriculum.configuration(prepared, cases, tokens, indices, "3407")
    value.update(version=VERSION, candidate_id=CANDIDATE)
    value["diagnostics"]["preparation"] = "smoke-configuration.json"
    value["diagnostics"]["scope"] = prepared["scope"]
    value["launch_gates"] = {
        "native_weights_verified": False,
        "actual_tokenizer_parity_verified_by_preparation": False,
        "smoke_training_and_fresh_reload_completed": False,
        "substantive_requires_successful_gemma_smoke_and_reload": True,
    }
    value["comparison"] = {
        "draw_order": "Exact same 809 case draws as the frozen Qwen9B curriculum.",
        "qwen_preparation_manifest_sha256": QWEN_MANIFEST_SHA256,
        "scope": "Common training states and draw order; model-specific token counts differ.",
        "control": "Use a paired export of this native Gemma base for SFT attribution; prior public QAT GGUF is an external reference.",
    }
    value["public_quantized_artifact"] = {
        "model_sha256": candidate["model_sha256"],
        "repository": candidate["quant_repository"],
        "revision": candidate["quant_revision"],
        "is_native_training_base": False,
        "verified_or_loaded_by_this_preparation": False,
    }
    value["generation_metadata"] = {
        "completion_label_eos_token_id": 1,
        "native_config_eos_token_id": [1, 106],
        "native_config_pad_token_id": 0,
        "generation_config_policy": "Actual generation stop/pad policy belongs to the separately pinned run configuration.",
    }
    return value


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_bytes())
    require(
        manifest.get("version") == VERSION
        and manifest.get("status") == STATUS
        and manifest.get("candidate_id") == CANDIDATE
        and manifest.get("native_revision") == REVISION,
        "Unexpected Gemma preparation identity.",
    )
    expected = manifest["files_sha256"]
    require(
        set(expected) == ALLOWED, "Only the registered train-only inventory is allowed."
    )
    paths = list(directory.rglob("*"))
    require(
        not any(path.is_symlink() for path in paths), "Bundle symlinks are not allowed."
    )
    require(
        {str(path.relative_to(directory)) for path in paths if path.is_file()}
        == ALLOWED | {"manifest.json"},
        "Physical bundle inventory differs.",
    )
    content = {
        name: read_bound(child(directory, name), pin) for name, pin in expected.items()
    }
    for name, pin in FROZEN_HELPERS.items():
        require(expected["source/" + name] == pin, "A frozen helper source changed.")
    for name, source in (
        ("prepare_passive_gemma.py", __file__),
        ("prepare_passive_curriculum.py", curriculum.__file__),
        ("prepare_passive_sft.py", smoke.__file__),
    ):
        require(
            sha256(Path(source).read_bytes()) == expected["source/" + name],
            "Loaded verifier source changed.",
        )
    evidence = {}
    for key, filename in EVIDENCE.items():
        require(
            manifest["bindings"][key]
            == {"path": filename, "sha256": expected[filename]},
            "Evidence binding changed.",
        )
        evidence[key] = json.loads(content[filename])
    release, registry, tokenizer_registry = (
        evidence[name] for name in ("release", "model_registry", "tokenizer_registry")
    )
    require(
        expected[EVIDENCE["release"]]
        == manifest["release_manifest_sha256"]
        == RELEASE_SHA256
        and release["version"] == "passive-dataset-completion-v2"
        and release["status"] == "frozen_supported_scope"
        and registry["version"] == "passive-models-v2",
        "Use the exact immutable completion-v2 release and model registry.",
    )
    matches = [row for row in registry["candidates"] if row["id"] == CANDIDATE]
    require(len(matches) == 1, "One Gemma model registration is required.")
    candidate = matches[0]
    tokenizers = [
        row for row in tokenizer_registry["candidates"] if row["id"] == CANDIDATE
    ]
    require(len(tokenizers) == 1, "One Gemma tokenizer registration is required.")
    tokenizer = tokenizers[0]
    tokenizer_pin = registry["tokenizers"]
    require(
        candidate["base_repository"] == tokenizer["repository"] == REPOSITORY
        and candidate["base_revision"] == tokenizer["revision"] == REVISION
        and candidate["tokenizer_id"] == CANDIDATE
        and not candidate.get("adapter_path")
        and candidate["tokenizer_path"] == tokenizer["path"]
        and candidate["tokenizer_files_sha256"] == tokenizer["files_sha256"]
        and set(tokenizer["files_sha256"]) == TOKENIZER_FILES
        and tokenizer_pin["registry_sha256"]
        == expected[EVIDENCE["tokenizer_registry"]]
        == release["dependencies_sha256"][tokenizer_pin["registry_path"]]
        and candidate["template_sha256"]
        == tokenizer["files_sha256"]["chat_template.jinja"],
        "Native Gemma tokenizer, revision or template identity changed.",
    )
    for name, pin in tokenizer["files_sha256"].items():
        require(
            release["dependencies_sha256"][str(Path(tokenizer["path"]) / name)] == pin,
            "Tokenizer is not frozen by the release.",
        )
        if name in METADATA_FILES:
            require(
                expected["native-metadata/" + name] == pin, "Native metadata changed."
            )
    native = json.loads(content["native-metadata/config.json"])
    token_config = json.loads(content["native-metadata/tokenizer_config.json"])
    require(
        native["model_type"] == "gemma4"
        and native["architectures"] == ["Gemma4ForConditionalGeneration"]
        and native["text_config"]["vocab_size"] == 262144
        and native["text_config"]["eos_token_id"] == 1
        and native["text_config"]["pad_token_id"] == 0
        and native["eos_token_id"] == [1, 106]
        and token_config["eos_token"] == "<eos>",
        "Gemma native vocabulary, EOS or architecture changed.",
    )
    for name, original in PAYLOADS.items():
        require(
            expected[name] == release["files_sha256"][original],
            "Frozen train payload changed.",
        )
    require(
        expected["source/passive_bundle_v2.py"]
        == release["dependencies_sha256"]["ml/passive_bundle_v2.py"],
        "Frozen weighted sampler changed.",
    )
    qwen = evidence["qwen_preparation"]
    require(
        expected[EVIDENCE["qwen_preparation"]] == QWEN_MANIFEST_SHA256
        and qwen["candidate_id"] == "qwen35-9b"
        and qwen["release_manifest_sha256"] == RELEASE_SHA256
        and qwen["files_sha256"]["draw-plan.json"]
        == expected[EVIDENCE["qwen_draw_plan"]]
        and all(
            qwen["files_sha256"][name] == expected[name]
            for name in ("train-cases.jsonl", "train-messages.jsonl")
        ),
        "Common-state Qwen comparison binding changed.",
    )
    lines = {
        name: smoke.jsonl_lines(content[filename], compressed=name == "tokens")
        for name, filename in (
            ("cases", "train-cases.jsonl"),
            ("messages", "train-messages.jsonl"),
            ("tokens", "train-tokens.jsonl.gz"),
        )
    }
    rows = {
        name: [json.loads(line) for line in values] for name, values in lines.items()
    }
    cases, messages, tokens = (rows[name] for name in ("cases", "messages", "tokens"))
    require(len(cases) == 809, "Use all 809 frozen training states.")
    smoke.validate_train(
        cases,
        messages,
        tokens,
        json.loads(content["source-groups.json"]),
        content["prompt-json.txt"].decode().removesuffix("\n"),
        1,
        262144,
    )
    indices = curriculum.weighted_draw_indices(cases, 809, "3407")
    plan = [
        {"draw_index": draw, "case_id": cases[index]["id"], "train_row_index": index}
        for draw, index in enumerate(indices)
    ]
    require(
        json.loads(content["draw-plan.json"]) == evidence["qwen_draw_plan"] == plan,
        "Use the exact shared string-seed draw order.",
    )
    selected = smoke.select_smoke(cases, tokens)
    for name in ("cases", "messages", "tokens"):
        value = content[f"smoke-{name}.jsonl"]
        require(
            value == b"".join(lines[name][i] for i in selected),
            "Smoke rows must preserve frozen train bytes and longest-slot selection.",
        )
        rows["smoke_" + name] = [json.loads(line) for line in smoke.jsonl_lines(value)]
    rows["smoke_inputs"] = [
        json.loads(line)
        for line in smoke.jsonl_lines(content["smoke-model-inputs.jsonl"])
    ]
    require(
        rows["smoke_inputs"]
        == [
            {
                key: tokens[i]["primary"][key]
                for key in ("input_ids", "attention_mask", "labels")
            }
            for i in selected
        ],
        "Smoke model inputs changed.",
    )
    config = configuration(candidate, cases, tokens, indices, selected)
    smoke_config = smoke_configuration(candidate, cases, tokens, selected)
    require(
        json.loads(content["configuration.json"]) == config
        and json.loads(content["smoke-configuration.json"]) == smoke_config,
        "Preparation configuration or phase budgets changed.",
    )
    census = {
        "available_train": smoke.census(cases, tokens),
        "fixed_stream": curriculum.exposure(cases, tokens, indices),
        "smoke": curriculum.exposure(cases, tokens, selected),
    }
    require(
        json.loads(content["exposure-census.json"]) == census,
        "Exposure census changed.",
    )
    inventory = [
        {"path": name, "bytes": len(data), "sha256": expected[name]}
        for name, data in sorted(content.items())
    ]
    require(
        manifest["inventory"] == inventory
        and manifest["payload_bytes"] == sum(row["bytes"] for row in inventory),
        "Portable byte inventory changed.",
    )
    return {
        "manifest": manifest,
        "configuration": config,
        "smoke_configuration": smoke_config,
        **rows,
        "draw_plan": plan,
        "verification": {
            "status": "passed_train_only_gemma_readback",
            "train_rows": 809,
            "draws": 809,
            "unique_drawn_cases": census["fixed_stream"]["unique_cases"],
            "repeat_draws": census["fixed_stream"]["repeat_draws"],
            "sequence_tokens": census["fixed_stream"]["sequence_tokens"],
            "completion_tokens": census["fixed_stream"]["completion_tokens"],
            "optimizer_updates": 203,
            "final_update_rows": 1,
            "smoke_rows": 8,
            "smoke_sequence_tokens": census["smoke"]["sequence_tokens"],
            "smoke_completion_tokens": census["smoke"]["completion_tokens"],
            "payload_files": len(expected),
            "payload_bytes": manifest["payload_bytes"],
            "dev_test_record_bodies_read": 0,
            "training_updates_executed": 0,
            "actual_tokenizer_parity": False,
        },
    }


def prepare(
    release, registry, qwen_preparation, output, *, registry_sha256, project_root=ROOT
):
    root = Path(project_root).resolve()
    release, registry, qwen_preparation = (
        Path(p).resolve() for p in (release, registry, qwen_preparation)
    )
    output = Path(output).absolute()
    require(
        not output.exists() and not output.is_symlink(), "Use a fresh output directory."
    )
    output = output.resolve()
    require(
        not any(p.name == "frozen" for p in (output, *output.parents))
        and not any(output.is_relative_to(p) for p in (release, qwen_preparation)),
        "Output must be outside immutable inputs.",
    )
    inputs = {}

    def bound(path, pin):
        path = Path(path).resolve()
        require(path.is_relative_to(root), "Input escapes the project root.")
        data = read_bound(path, pin)
        inputs[str(path.relative_to(root))] = pin
        return data

    files = {
        EVIDENCE["release"]: bound(release / "manifest.json", RELEASE_SHA256),
        EVIDENCE["model_registry"]: bound(registry, registry_sha256),
        EVIDENCE["qwen_preparation"]: bound(
            qwen_preparation / "manifest.json", QWEN_MANIFEST_SHA256
        ),
    }
    released, registered, qwen = (
        json.loads(files[EVIDENCE[key]])
        for key in ("release", "model_registry", "qwen_preparation")
    )
    tokenizer_binding = registered["tokenizers"]
    files[EVIDENCE["tokenizer_registry"]] = bound(
        root / tokenizer_binding["registry_path"], tokenizer_binding["registry_sha256"]
    )
    candidate = next(row for row in registered["candidates"] if row["id"] == CANDIDATE)
    files[EVIDENCE["qwen_draw_plan"]] = bound(
        qwen_preparation / "draw-plan.json", qwen["files_sha256"]["draw-plan.json"]
    )
    for name, original in PAYLOADS.items():
        files[name] = bound(release / original, released["files_sha256"][original])
    for name, pin in candidate["tokenizer_files_sha256"].items():
        data = bound(root / candidate["tokenizer_path"] / name, pin)
        if name in METADATA_FILES:
            files["native-metadata/" + name] = data
        if name == "tokenizer.json":
            require(
                [
                    token["id"]
                    for token in json.loads(data)["added_tokens"]
                    if token["content"] == "<eos>"
                ]
                == [1],
                "Actual native EOS lookup differs.",
            )
    lines = {
        name: smoke.jsonl_lines(files[filename], compressed=name == "tokens")
        for name, filename in (
            ("cases", "train-cases.jsonl"),
            ("messages", "train-messages.jsonl"),
            ("tokens", "train-tokens.jsonl.gz"),
        )
    }
    cases, messages, tokens = (
        [json.loads(line) for line in lines[name]]
        for name in ("cases", "messages", "tokens")
    )
    indices = curriculum.weighted_draw_indices(cases, 809, "3407")
    selected = smoke.select_smoke(cases, tokens)
    for name in ("cases", "messages", "tokens"):
        files[f"smoke-{name}.jsonl"] = b"".join(lines[name][i] for i in selected)
    files["smoke-model-inputs.jsonl"] = smoke.jsonl_bytes(
        {
            key: tokens[i]["primary"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        for i in selected
    )
    files["smoke-configuration.json"] = smoke.json_bytes(
        smoke_configuration(candidate, cases, tokens, selected)
    )
    files["configuration.json"] = smoke.json_bytes(
        configuration(candidate, cases, tokens, indices, selected)
    )
    files["draw-plan.json"] = smoke.json_bytes(
        [
            {
                "draw_index": draw,
                "case_id": cases[index]["id"],
                "train_row_index": index,
            }
            for draw, index in enumerate(indices)
        ]
    )
    files["exposure-census.json"] = smoke.json_bytes(
        {
            "available_train": smoke.census(cases, tokens),
            "fixed_stream": curriculum.exposure(cases, tokens, indices),
            "smoke": curriculum.exposure(cases, tokens, selected),
        }
    )
    for name in SOURCE_FILES:
        files["source/" + name] = Path(__file__).with_name(name).read_bytes()
    inventory = [
        {"path": name, "bytes": len(data), "sha256": sha256(data)}
        for name, data in sorted(files.items())
    ]
    manifest = {
        "version": VERSION,
        "status": STATUS,
        "candidate_id": CANDIDATE,
        "native_revision": REVISION,
        "release_manifest_sha256": RELEASE_SHA256,
        "bindings": {
            key: {"path": name, "sha256": sha256(files[name])}
            for key, name in EVIDENCE.items()
        },
        "inputs_sha256": inputs,
        "files_sha256": {row["path"]: row["sha256"] for row in inventory},
        "inventory": inventory,
        "payload_bytes": sum(row["bytes"] for row in inventory),
        "scope": "Train-only data preparation. Native weights, smoke training/reload and full-training completion are separate gates.",
        "portable_verification": "Standard library; run source/prepare_passive_gemma.py verify BUNDLE.",
    }

    def publish(directory):
        for name, data in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        with (directory / "manifest.json").open("xb") as stream:
            stream.write(smoke.json_bytes(manifest))

    with tempfile.TemporaryDirectory(
        prefix="gemma-preparation-validation-"
    ) as temporary:
        publish(Path(temporary))
        verify_bundle(temporary)
    output.mkdir(parents=True, exist_ok=False)
    publish(output)
    return verify_bundle(output)["verification"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--release", type=Path, required=True)
    create.add_argument("--registry", type=Path, required=True)
    create.add_argument("--registry-sha256", required=True)
    create.add_argument("--qwen-preparation", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_bundle(args.directory)["verification"]
    else:
        result = prepare(
            args.release,
            args.registry,
            args.qwen_preparation,
            args.output,
            registry_sha256=args.registry_sha256,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
