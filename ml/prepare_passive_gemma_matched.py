"""Freeze the existing Gemma matched train views and their shared 802-draw plan."""

import argparse
import json
from pathlib import Path
import tempfile

import prepare_passive_curriculum as curriculum
import prepare_passive_gemma as gemma
import prepare_passive_matched as matched
import prepare_passive_sft as smoke

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-sft-gemma-matched-preparation-v1"
STATUS = "frozen_matched_train_only_preparation"
CANDIDATE = "gemma26"
REPOSITORY = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
RELEASE_SHA256 = "15d4d27d4ce1fde74a4cd993f17362c7061aa115ec7547d3bf60b0a99c22c222"
REFERENCE_SHA256 = "9f93f1f9281e59b092502ee89782599d4fbf5769186cdbe5436e9e40f90d8bfc"
TRAIN_ROWS = 802
UNIQUE_DRAWS = 444
FORMATS = ("json", "rewrite")
FROZEN_HELPERS = {
    "prepare_passive_matched.py": "0e0e3be76fcbcd3182a41afa78c22b8fd98c4ee052132b4919ae1dfeee16ed7a",
    "prepare_passive_gemma.py": "fd4fc1a1d787c88d674a2096608c44f96873f8db4cc1dfa86d945ea7945c083b",
    **gemma.FROZEN_HELPERS,
    "passive_formats_v2.py": "dc890899e89dd937879c919e768b9d9492a7e080caae0e8e43936e84ee27e2df",
}
SOURCE_FILES = (
    "prepare_passive_gemma_matched.py",
    "test_prepare_passive_gemma_matched.py",
    *FROZEN_HELPERS,
)
PAYLOADS = {
    "train-cases.jsonl": "matched/train-cases.jsonl",
    "json-train-tokens.jsonl.gz": "tokenized/gemma26/json-train.jsonl.gz",
    "rewrite-train-tokens.jsonl.gz": "tokenized/gemma26/rewrite-train.jsonl.gz",
    "source-groups.json": "source-groups.json",
    "prompt-json.txt": "prompt-json.txt",
    "prompt-rewrite.txt": "prompt-rewrite.txt",
    "matched-eligibility.json": "matched/eligibility.json",
}
EVIDENCE = {
    "release": "evidence/release-manifest.json",
    "model_registry": "evidence/model-registry.json",
    "tokenizer_registry": "evidence/tokenizer-registry.json",
    "matched_reference": "evidence/qwen-matched-manifest.json",
    "matched_draw_plan": "evidence/qwen-matched-draw-plan.json",
}
ALLOWED = (
    set(PAYLOADS)
    | set(EVIDENCE.values())
    | {"source/" + name for name in SOURCE_FILES}
    | {"native-metadata/" + name for name in gemma.METADATA_FILES}
    | {
        "configuration.json",
        "draw-plan.json",
        "exposure-census.json",
        "semantic-readback.json",
        "diagnostics/cases.jsonl",
    }
    | {
        f"diagnostics/{fmt}-{kind}.jsonl"
        for fmt in FORMATS
        for kind in ("messages", "tokens", "inputs")
    }
)
require = smoke.require
sha256 = smoke.sha256
child = curriculum.child
read_bound = curriculum.read_bound
semantic_readback = matched.semantic_readback


def configuration(candidate, cases, tokens, indices, selected):
    views = {}
    for fmt in FORMATS:
        planned = curriculum.exposure(cases, tokens[fmt], indices)
        diagnostics = curriculum.exposure(cases, tokens[fmt], selected)
        views[fmt] = {
            "training_view": fmt,
            "max_total_sequence_tokens": planned["sequence_tokens"],
            "max_total_completion_tokens": planned["completion_tokens"],
            "diagnostic_case_order": [cases[index]["id"] for index in selected],
            "diagnostic_sequence_tokens": diagnostics["sequence_tokens"],
            "diagnostic_completion_tokens": diagnostics["completion_tokens"],
        }
    return {
        "version": VERSION,
        "status": STATUS,
        "candidate_id": CANDIDATE,
        "native_base": gemma.native_identity(candidate),
        "tokenizer": gemma.tokenizer_identity(candidate),
        "frozen_train_rows": TRAIN_ROWS,
        "formats": list(FORMATS),
        "loss_mask": "completion_only_including_eos",
        "input_limit": 3584,
        "output_limit_each_format": 512,
        "context_limit": 4096,
        "truncation": False,
        "packing": False,
        "sampling": {
            "algorithm": "weighted_indices_v2",
            "seed": "3407",
            "seed_type": "str",
            "replacement": True,
            "draw_count": TRAIN_ROWS,
            "weights_applied_to_loss": False,
        },
        "training": {
            "micro_batch_size": 1,
            "gradient_accumulation": 4,
            "optimizer_updates": 201,
            "final_update_rows": 2,
            "final_update_normalization": "mean_over_actual_microbatch_count",
            "sample_order": [cases[index]["id"] for index in indices],
            "initialization": "fresh_adapter_on_exact_native_base_not_smoke_or_T1_adapter",
        },
        "views": views,
        "diagnostic_selection": "One per action/language cell; longest maximum Gemma primary sequence across JSON and Rewrite, then lexicographic case ID. Shared train-only diagnostics, each used once for an eight-row/two-update engineering smoke.",
        "generation_metadata": {
            "completion_label_eos_token_id": 1,
            "native_generation_stop_ids": [1, 106, 50],
            "bos_token_id": 2,
            "pad_token_id": 0,
            "actual_generation_policy_verified_by_preparation": False,
        },
        "matched_reference_manifest_sha256": REFERENCE_SHA256,
        "training_run": False,
        "dev_test_record_bodies_copied": 0,
        "comparison_scope": "Exact common matched cases, weights and 802-case draw order. Gemma-specific token exposure differs by format. The T1 full-curriculum adapter is not the matched JSON control.",
    }


def validate_identity(content, expected, manifest):
    evidence = {}
    for key, filename in EVIDENCE.items():
        require(
            manifest["bindings"][key]
            == {"path": filename, "sha256": expected[filename]},
            "Evidence binding changed.",
        )
        evidence[key] = json.loads(content[filename])
    release, registry, token_registry = (
        evidence[key] for key in ("release", "model_registry", "tokenizer_registry")
    )
    require(
        expected[EVIDENCE["release"]]
        == manifest["release_manifest_sha256"]
        == RELEASE_SHA256
        and release["version"] == "passive-dataset-completion-v2"
        and release["status"] == "frozen_supported_scope"
        and registry["version"] == "passive-models-v2",
        "Use the exact frozen release and registered Gemma identity.",
    )
    candidates = [row for row in registry["candidates"] if row["id"] == CANDIDATE]
    tokenizers = [row for row in token_registry["candidates"] if row["id"] == CANDIDATE]
    require(
        len(candidates) == len(tokenizers) == 1, "One Gemma registration is required."
    )
    candidate, tokenizer = candidates[0], tokenizers[0]
    token_pin = registry["tokenizers"]
    require(
        candidate["base_repository"] == tokenizer["repository"] == REPOSITORY
        and candidate["base_revision"] == tokenizer["revision"] == REVISION
        and candidate["tokenizer_id"] == CANDIDATE
        and not candidate.get("adapter_path")
        and candidate["tokenizer_path"] == tokenizer["path"]
        and candidate["tokenizer_files_sha256"] == tokenizer["files_sha256"]
        and set(tokenizer["files_sha256"]) == gemma.TOKENIZER_FILES
        and token_pin["registry_sha256"]
        == expected[EVIDENCE["tokenizer_registry"]]
        == release["dependencies_sha256"][token_pin["registry_path"]]
        and candidate["template_sha256"]
        == tokenizer["files_sha256"]["chat_template.jinja"],
        "Gemma native revision, tokenizer or template differs from the frozen release.",
    )
    for name, pin in tokenizer["files_sha256"].items():
        require(
            release["dependencies_sha256"][str(Path(tokenizer["path"]) / name)] == pin,
            "Tokenizer is outside the frozen release.",
        )
        if name in gemma.METADATA_FILES:
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
        "Gemma architecture, vocabulary or label EOS changed.",
    )
    for name, original in PAYLOADS.items():
        require(
            expected[name] == release["files_sha256"][original],
            "Frozen train payload changed.",
        )
    for name in ("passive_bundle_v2.py", "passive_formats_v2.py"):
        require(
            expected["source/" + name] == release["dependencies_sha256"]["ml/" + name],
            "Frozen sampler or format source changed.",
        )
    reference = evidence["matched_reference"]
    require(
        expected[EVIDENCE["matched_reference"]] == REFERENCE_SHA256
        and reference["version"] == "passive-sft-matched-preparation-v1"
        and reference["status"] == STATUS
        and reference["candidate_id"] == "qwen35-9b"
        and reference["release_manifest_sha256"] == RELEASE_SHA256
        and reference["files_sha256"]["draw-plan.json"]
        == expected[EVIDENCE["matched_draw_plan"]]
        and all(
            reference["files_sha256"][name] == expected[name]
            for name in (
                "train-cases.jsonl",
                "source-groups.json",
                "matched-eligibility.json",
            )
        ),
        "Common matched case, weight or draw-plan provenance changed.",
    )
    require(
        json.loads(content["matched-eligibility.json"])["qualified"]
        == {"train": TRAIN_ROWS, "dev": 286},
        "Keep the frozen common 802-train/286-development roster.",
    )
    return candidate, evidence


def read_views(content):
    case_lines = smoke.jsonl_lines(content["train-cases.jsonl"])
    token_lines = {
        fmt: smoke.jsonl_lines(content[f"{fmt}-train-tokens.jsonl.gz"], compressed=True)
        for fmt in FORMATS
    }
    cases = [json.loads(line) for line in case_lines]
    tokens = {fmt: [json.loads(line) for line in token_lines[fmt]] for fmt in FORMATS}
    return cases, tokens, case_lines, token_lines


def planned_views(content, candidate):
    cases, tokens, case_lines, token_lines = read_views(content)
    matched.validate_views(
        cases,
        tokens,
        json.loads(content["source-groups.json"]),
        {
            fmt: content[f"prompt-{fmt}.txt"].decode().removesuffix("\n")
            for fmt in FORMATS
        },
        gemma.tokenizer_identity(candidate),
    )
    indices = curriculum.weighted_draw_indices(cases, TRAIN_ROWS, "3407")
    require(
        len(set(indices)) == UNIQUE_DRAWS,
        "The fixed stream must retain 444 unique states.",
    )
    selected = matched.selected_diagnostics(cases, tokens)
    plan = [
        {"draw_index": pos, "case_id": cases[index]["id"], "train_row_index": index}
        for pos, index in enumerate(indices)
    ]
    require(
        plan == json.loads(content[EVIDENCE["matched_draw_plan"]]),
        "Gemma must retain the exact common matched string-seed draw order.",
    )
    views, diagnostics = {}, {}
    diagnostics["diagnostics/cases.jsonl"] = b"".join(
        case_lines[index] for index in selected
    )
    for fmt in FORMATS:
        messages = matched.exported_messages(tokens[fmt])
        selected_tokens = [tokens[fmt][index] for index in selected]
        selected_messages = [messages[index] for index in selected]
        inputs = [
            {
                key: row["primary"][key]
                for key in ("input_ids", "attention_mask", "labels")
            }
            for row in selected_tokens
        ]
        diagnostics[f"diagnostics/{fmt}-tokens.jsonl"] = b"".join(
            token_lines[fmt][index] for index in selected
        )
        diagnostics[f"diagnostics/{fmt}-messages.jsonl"] = smoke.jsonl_bytes(
            selected_messages
        )
        diagnostics[f"diagnostics/{fmt}-inputs.jsonl"] = smoke.jsonl_bytes(inputs)
        views[fmt] = {
            "messages": messages,
            "tokens": tokens[fmt],
            "diagnostics": {
                "cases": [cases[index] for index in selected],
                "messages": selected_messages,
                "tokens": selected_tokens,
                "inputs": inputs,
                "rows": [
                    {"case_id": row["id"], **row["primary"]} for row in selected_tokens
                ],
            },
        }
    config = configuration(candidate, cases, tokens, indices, selected)
    exposure = {
        fmt: {
            "available_train": smoke.census(cases, tokens[fmt]),
            "fixed_stream": curriculum.exposure(cases, tokens[fmt], indices),
            "smoke": curriculum.exposure(cases, tokens[fmt], selected),
        }
        for fmt in FORMATS
    }
    return cases, tokens, plan, config, views, diagnostics, exposure


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    require(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        "The bundle manifest must be an ordinary file.",
    )
    manifest = json.loads(manifest_path.read_bytes())
    require(
        manifest["version"] == VERSION
        and manifest["status"] == STATUS
        and manifest["candidate_id"] == CANDIDATE
        and manifest["native_revision"] == REVISION
        and manifest["dev_test_record_bodies_copied"] == 0,
        "Unexpected Gemma matched train-only preparation identity.",
    )
    expected = manifest["files_sha256"]
    require(
        set(expected) == ALLOWED,
        "Only the registered matched train-only inventory is allowed.",
    )
    paths = list(directory.rglob("*"))
    require(
        not any(path.is_symlink() for path in paths)
        and {str(path.relative_to(directory)) for path in paths if path.is_file()}
        == ALLOWED | {"manifest.json"},
        "Physical bundle inventory changed or contains a symlink.",
    )
    content = {
        name: read_bound(child(directory, name), pin) for name, pin in expected.items()
    }
    for name, pin in FROZEN_HELPERS.items():
        require(
            expected["source/" + name] == pin, "A frozen preparation helper changed."
        )
    for name, source in (
        ("prepare_passive_gemma_matched.py", __file__),
        ("prepare_passive_matched.py", matched.__file__),
        ("prepare_passive_gemma.py", gemma.__file__),
        ("prepare_passive_curriculum.py", curriculum.__file__),
        ("prepare_passive_sft.py", smoke.__file__),
    ):
        require(
            sha256(Path(source).read_bytes()) == expected["source/" + name],
            "Loaded verifier source differs from its bound snapshot.",
        )
    candidate, _ = validate_identity(content, expected, manifest)
    cases, tokens, plan, config, views, diagnostics, exposure = planned_views(
        content, candidate
    )
    require(
        json.loads(content["draw-plan.json"]) == plan,
        "Shared draw-plan replay differs.",
    )
    require(
        json.loads(content["configuration.json"]) == config,
        "Gemma format or phase budgets changed.",
    )
    require(
        json.loads(content["exposure-census.json"]) == exposure,
        "Gemma exposure census differs.",
    )
    require(
        all(content[name] == value for name, value in diagnostics.items()),
        "Shared diagnostics changed.",
    )
    semantic = json.loads(content["semantic-readback.json"])
    references = sum(1 + len(case["acceptableOutputs"]) for case in cases)
    require(
        semantic["status"] == "passed_local_paired_reference_readback"
        and semantic["train_cases"] == TRAIN_ROWS
        and semantic["references_per_format"] == references
        and semantic["format_reference_checks"] == 2 * references
        and semantic["canonical_message_checks"] == 2 * TRAIN_ROWS
        and semantic["training_primary_targets_only"] is True
        and semantic["retokenized"] is False
        and semantic["dev_test_bodies_read"] == 0,
        "Paired semantic readback evidence changed.",
    )
    inventory = [
        {"path": name, "bytes": len(data), "sha256": expected[name]}
        for name, data in sorted(content.items())
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
        "views": views,
        "verification": {
            "status": "passed_matched_train_only_gemma_readback",
            "train_rows": TRAIN_ROWS,
            "shared_draws_per_format": len(plan),
            "unique_drawn_cases": UNIQUE_DRAWS,
            "repeat_draws": len(plan) - UNIQUE_DRAWS,
            "optimizer_updates_per_format": 201,
            "final_update_rows": 2,
            "views": config["views"],
            "payload_files": len(content),
            "payload_bytes": manifest["payload_bytes"],
            "actual_tokenizer_parity_verified": False,
            "dev_test_record_bodies_read": 0,
            "training_updates_executed": 0,
        },
    }


def prepare(
    release,
    registry,
    reference_preparation,
    output,
    *,
    registry_sha256,
    project_root=ROOT,
):
    root = Path(project_root).resolve()
    release, registry, reference_preparation = (
        Path(path).resolve() for path in (release, registry, reference_preparation)
    )
    output = Path(output).absolute()
    require(
        not output.exists() and not output.is_symlink(), "Use a fresh output directory."
    )
    output = output.resolve()
    require(
        not any(part.name == "frozen" for part in (output, *output.parents))
        and not any(
            output.is_relative_to(path) for path in (release, reference_preparation)
        ),
        "Publish outside frozen inputs and existing reference preparation.",
    )
    inputs = {}

    def bound(path, expected):
        path = Path(path).resolve()
        require(path.is_relative_to(root), "Input escapes the project root.")
        value = read_bound(path, expected)
        inputs[str(path.relative_to(root))] = expected
        return value

    files = {
        EVIDENCE["release"]: bound(release / "manifest.json", RELEASE_SHA256),
        EVIDENCE["model_registry"]: bound(registry, registry_sha256),
        EVIDENCE["matched_reference"]: bound(
            reference_preparation / "manifest.json", REFERENCE_SHA256
        ),
    }
    released, registered, reference = (
        json.loads(files[EVIDENCE[key]])
        for key in ("release", "model_registry", "matched_reference")
    )
    tokenizer_binding = registered["tokenizers"]
    files[EVIDENCE["tokenizer_registry"]] = bound(
        root / tokenizer_binding["registry_path"], tokenizer_binding["registry_sha256"]
    )
    candidates = [row for row in registered["candidates"] if row["id"] == CANDIDATE]
    require(len(candidates) == 1, "One Gemma registration is required.")
    candidate = candidates[0]
    files[EVIDENCE["matched_draw_plan"]] = bound(
        reference_preparation / "draw-plan.json",
        reference["files_sha256"]["draw-plan.json"],
    )
    for name, original in PAYLOADS.items():
        files[name] = bound(release / original, released["files_sha256"][original])
    for name, pin in candidate["tokenizer_files_sha256"].items():
        value = bound(root / candidate["tokenizer_path"] / name, pin)
        if name in gemma.METADATA_FILES:
            files["native-metadata/" + name] = value
        if name == "tokenizer.json":
            require(
                [
                    row["id"]
                    for row in json.loads(value)["added_tokens"]
                    if row["content"] == "<eos>"
                ]
                == [1],
                "Actual native label EOS lookup differs.",
            )
    for name in SOURCE_FILES:
        source = Path(__file__).with_name(name)
        files["source/" + name] = source.read_bytes()
        if name in FROZEN_HELPERS:
            require(
                sha256(files["source/" + name]) == FROZEN_HELPERS[name],
                "Frozen helper source changed.",
            )
    for name, pin in released["dependencies_sha256"].items():
        if name.startswith("ml/") and Path(name).suffix in {
            ".py",
            ".cjs",
            ".json",
            ".txt",
        }:
            bound(root / name, pin)
    cases, tokens, plan, config, _, diagnostics, exposure = planned_views(
        files, candidate
    )
    files.update(diagnostics)
    for name, value in (
        ("configuration.json", config),
        ("draw-plan.json", plan),
        ("exposure-census.json", exposure),
        ("semantic-readback.json", semantic_readback(cases, tokens)),
    ):
        files[name] = smoke.json_bytes(value)
    inventory = [
        {"path": name, "bytes": len(value), "sha256": sha256(value)}
        for name, value in sorted(files.items())
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
        "dev_test_record_bodies_copied": 0,
        "scope": "Existing train-only Gemma matched bytes and common draw order; metadata describes the 286-dev cohort without reading its bodies. Native weights, actual tokenizer parity, training, reload and serving remain separate gates.",
        "portable_verification": "Standard library; run source/prepare_passive_gemma_matched.py verify BUNDLE.",
    }

    def publish(directory):
        for name, value in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(value)
        with (directory / "manifest.json").open("xb") as stream:
            stream.write(smoke.json_bytes(manifest))

    with tempfile.TemporaryDirectory(prefix="gemma-matched-readback-") as temporary:
        publish(Path(temporary))
        verify_bundle(temporary)
    output.mkdir(parents=True, exist_ok=False)
    publish(output)
    return verify_bundle(output)["verification"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    for name in ("release", "registry", "reference-preparation", "output"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--registry-sha256", required=True)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_bundle(args.directory)["verification"]
    else:
        result = prepare(
            args.release,
            args.registry,
            args.reference_preparation,
            args.output,
            registry_sha256=args.registry_sha256,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
