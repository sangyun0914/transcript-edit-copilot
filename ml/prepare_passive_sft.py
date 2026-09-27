"""Prepare a pinned train-only SFT smoke bundle; no trainer or model is loaded."""

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("no_action", "correct_text", "split_paragraph", "merge_paragraph")
LANGUAGES = ("en", "ko")
SLOTS = tuple((action, language) for action in ACTIONS for language in LANGUAGES)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()


def jsonl_bytes(values):
    return "".join(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for value in values
    ).encode()


def jsonl_lines(content, compressed=False):
    if compressed:
        content = gzip.decompress(content)
    lines = [line for line in content.splitlines(keepends=True) if line.strip()]
    require(
        all(line.endswith(b"\n") for line in lines),
        "Frozen JSONL rows need terminating newlines.",
    )
    return lines


def validate_train(cases, messages, tokens, groups, prompt, eos_id, vocab_size):
    require(
        len(cases) == len(messages) == len(tokens) > 0, "Train payload lengths differ."
    )
    identities = [case["id"] for case in cases]
    require(len(set(identities)) == len(identities), "Duplicate train case ID.")
    for case, exported, encoded in zip(cases, messages, tokens, strict=True):
        identity, meta = case["id"], case["metadata"]
        require(
            exported["id"] == encoded["id"] == identity,
            "Train payload identity/order differs.",
        )
        require(
            meta["split"] == encoded["split"] == "train"
            and meta.get("training_eligible") is True
            and groups.get(meta["source_group_id"]) == "train",
            "Non-train or ineligible source entered smoke preparation.",
        )
        action = case["expectedOutput"]["action"]
        require(
            action in ACTIONS
            and meta["language"] in LANGUAGES
            and case["input"]["language"]
            in {"en": ("en", "eng"), "ko": ("ko", "kor")}[meta["language"]]
            and action in case["input"]["allowed_actions"]
            and set(case["input"]["allowed_actions"]) <= set(ACTIONS),
            "Unsupported train action/language or gated action.",
        )
        weight = meta["sampling_weight"]
        require(
            type(weight) in (int, float)
            and math.isfinite(weight)
            and weight > 0
            and weight == encoded["sampling_weight"] == exported["sampling_weight"],
            "Train sampling weight differs or is not positive/finite.",
        )
        require(
            all(
                isinstance(meta.get(key), str) and meta[key]
                for key in (
                    "source_id",
                    "source_group_id",
                    "episode_id",
                    "sampling_cluster",
                )
            )
            and meta["sampling_cluster"].startswith("train:")
            and re.fullmatch(r"[0-9a-f]{64}", meta["input_sha256"]) is not None
            and encoded["input_sha256"] == meta["input_sha256"],
            "Train provenance/input binding is incomplete.",
        )
        references = [case["expectedOutput"], *case["acceptableOutputs"]]
        require(
            [json.loads(text) for text in encoded["references"]] == references
            and all(reference["action"] in ACTIONS for reference in references),
            "Train primary/alternative references differ.",
        )
        require(
            [message["role"] for message in encoded["messages"]] == ["system", "user"]
            and encoded["messages"][0]["content"] == prompt
            and exported["messages"]
            == [
                *encoded["messages"],
                {"role": "assistant", "content": encoded["references"][0]},
            ],
            "Frozen train messages/assistant target differ.",
        )
        values = encoded["primary"]
        require(
            set(values)
            == {"input_ids", "attention_mask", "labels", "completion_start"},
            "Unexpected primary training fields.",
        )
        inputs, labels, prefix = (
            values["input_ids"],
            values["labels"],
            values["completion_start"],
        )
        require(
            isinstance(inputs, list)
            and isinstance(labels, list)
            and type(prefix) is int
            and 0 < prefix <= 3584
            and 0 < len(inputs) - prefix <= 512
            and len(inputs) <= 4096
            and len(labels) == len(inputs)
            and all(type(token) is int and 0 <= token < vocab_size for token in inputs)
            and all(type(label) is int for label in labels)
            and labels[:prefix] == [-100] * prefix
            and labels[prefix:] == inputs[prefix:]
            and values["attention_mask"] == [1] * len(inputs)
            and all(type(value) is int for value in values["attention_mask"])
            and inputs[-1] == eos_id,
            "Invalid completion-only mask, token bounds, attention mask or EOS.",
        )
        alternatives = encoded["alternative_completion_ids"]
        require(
            len(alternatives) == len(references) - 1, "Alternative token count differs."
        )
        for completion in alternatives:
            require(
                isinstance(completion, list)
                and 0 < len(completion) <= 512
                and prefix + len(completion) <= 4096
                and completion[-1] == eos_id
                and all(
                    type(token) is int and 0 <= token < vocab_size
                    for token in completion
                ),
                "Invalid alternative completion tokens.",
            )


def select_smoke(cases, tokens):
    choices = defaultdict(list)
    for index, (case, encoded) in enumerate(zip(cases, tokens, strict=True)):
        slot = (case["expectedOutput"]["action"], case["metadata"]["language"])
        choices[slot].append((-len(encoded["primary"]["input_ids"]), case["id"], index))
    require(
        set(choices) == set(SLOTS),
        "Missing or unsupported action-language smoke slots.",
    )
    return [min(choices[slot])[2] for slot in SLOTS]


def census(cases, tokens):
    weights = [case["metadata"]["sampling_weight"] for case in cases]
    lengths = [len(row["primary"]["input_ids"]) for row in tokens]
    prefixes = [row["primary"]["completion_start"] for row in tokens]
    completions = [
        length - prefix for length, prefix in zip(lengths, prefixes, strict=True)
    ]
    mass = math.fsum(weights)
    groups = defaultdict(lambda: {"rows": 0, "sampling_mass": 0.0})
    for case, weight in zip(cases, weights, strict=True):
        group = groups[case["metadata"]["source_group_id"]]
        group["rows"] += 1
        group["sampling_mass"] += weight / mass
    return {
        "unique_cases": len(cases),
        "source_groups": dict(sorted(groups.items())),
        "unique_sources": len({case["metadata"]["source_id"] for case in cases}),
        "unique_episodes": len({case["metadata"]["episode_id"] for case in cases}),
        "sampling_clusters": dict(
            sorted(
                Counter(case["metadata"]["sampling_cluster"] for case in cases).items()
            )
        ),
        "action_language_rows": {
            f"{action}/{language}": sum(
                case["expectedOutput"]["action"] == action
                and case["metadata"]["language"] == language
                for case in cases
            )
            for action, language in SLOTS
        },
        "sampling_weight_sum": mass,
        "primary_tokens": {
            name: {
                "minimum": min(values),
                "maximum": max(values),
                "unweighted_sum": sum(values),
                "expected_per_weighted_draw": math.fsum(
                    weight * value
                    for weight, value in zip(weights, values, strict=True)
                )
                / mass,
            }
            for name, values in (
                ("sequence", lengths),
                ("masked_prompt", prefixes),
                ("completion", completions),
            )
        },
    }


def prepare(
    release,
    registry,
    candidate_id,
    native_revision,
    output,
    *,
    release_sha256,
    registry_sha256,
    project_root=ROOT,
):
    root, release, registry = map(
        lambda path: Path(path).resolve(), (project_root, release, registry)
    )
    output = Path(output).absolute()
    require(
        not output.exists() and not output.is_symlink(),
        "Output must be a fresh directory.",
    )
    output = output.resolve()
    require(
        not output.is_relative_to(release)
        and not any(parent.name == "frozen" for parent in (output, *output.parents)),
        "Output cannot be inside a frozen release.",
    )
    bindings = {}

    def bound(path, expected):
        path = Path(path).resolve()
        require(path.is_relative_to(root), "Input path escapes the project root.")
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "An explicit SHA256 binding is required.",
        )
        content = path.read_bytes()
        require(sha256(content) == expected, f"Bound file hash drift: {path}")
        bindings[str(path.relative_to(root))] = expected
        return content

    manifest = json.loads(bound(release / "manifest.json", release_sha256))
    model_registry = json.loads(bound(registry, registry_sha256))
    require(
        manifest.get("version") == "passive-dataset-completion-v2"
        and manifest.get("status") == "frozen_supported_scope"
        and model_registry.get("version") == "passive-models-v2",
        "Use a frozen completion-v2 release and pinned passive-models-v2 registry.",
    )
    candidates = [
        row for row in model_registry["candidates"] if row["id"] == candidate_id
    ]
    require(len(candidates) == 1, "Choose one explicit registered candidate ID.")
    candidate = candidates[0]
    require(
        not candidate.get("adapter_path"),
        "This preparer requires a native base candidate.",
    )
    tokenizer_binding = model_registry["tokenizers"]
    tokenizer_name = tokenizer_binding["registry_path"]
    require(
        manifest["dependencies_sha256"].get(tokenizer_name)
        == tokenizer_binding["registry_sha256"],
        "Release and model registry bind different tokenizer registries.",
    )
    tokenizer_registry = json.loads(
        bound(root / tokenizer_name, tokenizer_binding["registry_sha256"])
    )
    tokenizer_candidates = [
        row
        for row in tokenizer_registry["candidates"]
        if row["id"] == candidate["tokenizer_id"]
    ]
    require(
        len(tokenizer_candidates) == 1,
        "Missing or duplicate pinned tokenizer candidate.",
    )
    tokenizer = tokenizer_candidates[0]
    require(
        re.fullmatch(r"[0-9a-f]{40}", native_revision) is not None
        and native_revision == candidate["base_revision"] == tokenizer["revision"]
        and candidate["base_repository"] == tokenizer["repository"]
        and candidate["tokenizer_path"] == tokenizer["path"]
        and candidate["tokenizer_files_sha256"] == tokenizer["files_sha256"],
        "Native revision/repository/tokenizer identity differs from the frozen pin.",
    )
    tokenizer_files = {}
    require(
        set(tokenizer["files_sha256"])
        == {
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.jinja",
        },
        "Unexpected pinned tokenizer file set.",
    )
    for name, expected in tokenizer["files_sha256"].items():
        relative = str(Path(tokenizer["path"]) / name)
        require(
            manifest["dependencies_sha256"].get(relative) == expected,
            "Tokenizer file is not bound by the release.",
        )
        tokenizer_files[name] = bound(root / relative, expected)
    require(
        candidate["template_sha256"]
        == tokenizer["files_sha256"]["chat_template.jinja"],
        "Pinned template binding differs.",
    )
    native_config = json.loads(tokenizer_files["config.json"])
    vocab_size = native_config.get("text_config", native_config)["vocab_size"]
    require(
        type(vocab_size) is int and vocab_size > 0, "Native vocabulary size is missing."
    )
    token_config = json.loads(tokenizer_files["tokenizer_config.json"])
    eos = token_config["eos_token"]
    eos = eos["content"] if isinstance(eos, dict) else eos
    eos_ids = [
        token["id"]
        for token in json.loads(tokenizer_files["tokenizer.json"])["added_tokens"]
        if token["content"] == eos
    ]
    require(len(eos_ids) == 1, "Pinned tokenizer must resolve one EOS ID.")

    def payload(name):
        return bound(release / name, manifest["files_sha256"].get(name))

    case_lines = jsonl_lines(payload("train-cases.jsonl"))
    message_lines = jsonl_lines(payload("train-messages.jsonl"))
    token_name = f"tokenized/{candidate['tokenizer_id']}/json-curriculum-train.jsonl.gz"
    token_lines = jsonl_lines(payload(token_name), compressed=True)
    cases = [json.loads(line) for line in case_lines]
    messages = [json.loads(line) for line in message_lines]
    tokens = [json.loads(line) for line in token_lines]
    groups = json.loads(payload("source-groups.json"))
    prompt = payload("prompt-json.txt").decode().removesuffix("\n")
    validate_train(cases, messages, tokens, groups, prompt, eos_ids[0], vocab_size)
    selected = select_smoke(cases, tokens)
    full_census = census(cases, tokens)
    smoke_census = census(
        [cases[index] for index in selected], [tokens[index] for index in selected]
    )
    selection = [
        {
            "case_id": cases[index]["id"],
            "train_row_index": index,
            "action": action,
            "language": language,
            "sequence_tokens": len(tokens[index]["primary"]["input_ids"]),
            "prompt_tokens": tokens[index]["primary"]["completion_start"],
            "completion_tokens": len(tokens[index]["primary"]["input_ids"])
            - tokens[index]["primary"]["completion_start"],
        }
        for (action, language), index in zip(SLOTS, selected, strict=True)
    ]
    config = {
        "version": "passive-sft-preparation-v1",
        "status": "data_ready_trainer_not_implemented",
        "training_run": False,
        "candidate_selection_status": "not_selected_for_t1",
        "scope": "Preparation-only candidate example; does not bind a finalist or authorize training.",
        "candidate_id": candidate_id,
        "native_base": {
            "repository": candidate["base_repository"],
            "revision": native_revision,
            "weights_verified_by_this_preparation": False,
        },
        "tokenizer": {
            "id": tokenizer["id"],
            "files_sha256": tokenizer["files_sha256"],
            "eos_token_id": eos_ids[0],
            "vocab_size": vocab_size,
        },
        "public_quantized_artifact": {
            "model_sha256": candidate.get("model_sha256"),
            "provenance": candidate.get("quantization_provenance"),
            "verified_or_loaded_by_this_preparation": False,
            "is_native_training_base": False,
        },
        "training_view": "json-curriculum",
        "loss_mask": "completion_only_including_eos",
        "input_limit": 3584,
        "output_limit": 512,
        "context_limit": 4096,
        "truncation": False,
        "packing": False,
        "proposed_smoke": {
            "micro_batch_size": 1,
            "gradient_accumulation": 4,
            "optimizer_updates": 2,
            "lora_rank": 16,
            "rows": 8,
            "order": [row["case_id"] for row in selection],
            "sampling": "Each selected row once in the declared action-language slot order.",
            "sampling_weight_use": "Retained for provenance; no resampling or loss weighting in smoke.",
        },
        "unresolved_prerequisites": {
            "trainer_implementation": None,
            "native_weight_files_and_sha256": None,
            "profiled_device_and_memory": None,
            "adapter_target_modules": None,
            "precision_and_base_quantization": None,
            "optimizer": None,
            "learning_rate_and_scheduler": None,
            "lora_alpha_dropout_and_bias": None,
            "training_seed": None,
            "training_library_versions": None,
            "save_fresh_reload_parity": None,
            "merge_quantize_serve_contract": None,
        },
        "selection_rule": "One primary-action × language cell; longest full primary sequence, then lexicographic case ID.",
        "selection": selection,
    }
    budgets = []
    for exposures in (1, 2):
        draws = len(cases) * exposures
        budgets.append(
            {
                "nominal_exposures": exposures,
                "proposed_sampled_rows": draws,
                "proposed_optimizer_updates": math.ceil(draws / 4),
                "final_update_microbatches": draws % 4 or 4,
                "expected_completion_tokens": draws
                * full_census["primary_tokens"]["completion"][
                    "expected_per_weighted_draw"
                ],
                "expected_sequence_tokens": draws
                * full_census["primary_tokens"]["sequence"][
                    "expected_per_weighted_draw"
                ],
            }
        )
    exposure = {
        "status": "proposal_only_no_sampling_or_training_performed",
        "full_train": full_census,
        "selected_smoke": smoke_census,
        "full_training_budget_proposals": budgets,
        "policy": (
            "Full-training proposals assume continuous weighted replacement draws, batch 1 and "
            "accumulation 4, with one final partial update at the end of the entire draw stream. "
            "Normalize that final update by its actual microbatch count. Weights apply through "
            "sampling only, never again through loss multiplication. Draws are not guaranteed "
            "unique-row visits; report realized unique cases/clusters and tokens after training. "
            "Expected token counts are not observed training work or independent evidence."
        ),
    }
    files = {
        "smoke-cases.jsonl": b"".join(case_lines[index] for index in selected),
        "smoke-messages.jsonl": b"".join(message_lines[index] for index in selected),
        "smoke-tokens.jsonl": b"".join(token_lines[index] for index in selected),
        "smoke-model-inputs.jsonl": jsonl_bytes(
            {
                key: tokens[index]["primary"][key]
                for key in ("input_ids", "attention_mask", "labels")
            }
            for index in selected
        ),
        "configuration.json": json_bytes(config),
        "exposure-census.json": json_bytes(exposure),
        "source/prepare_passive_sft.py": Path(__file__).read_bytes(),
        "source/test_prepare_passive_sft.py": Path(__file__)
        .with_name("test_prepare_passive_sft.py")
        .read_bytes(),
    }
    for name in ("config.json", "tokenizer_config.json", "chat_template.jinja"):
        files[f"native-metadata/{name}"] = tokenizer_files[name]
    receipt = {
        "version": "passive-sft-preparation-receipt-v1",
        "status": config["status"],
        "candidate_selection_status": config["candidate_selection_status"],
        "candidate_id": candidate_id,
        "native_revision": native_revision,
        "release_id": manifest["release_id"],
        "release_manifest_sha256": release_sha256,
        "model_registry_sha256": registry_sha256,
        "inputs_sha256": bindings,
        "files_sha256": {name: sha256(content) for name, content in files.items()},
        "source_snapshot": "source/prepare_passive_sft.py",
        "bundled_test_argv": [
            "python3",
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            "source",
            "-p",
            "test_prepare_passive_sft.py",
            "-v",
        ],
        "validation": {
            "train_rows": len(cases),
            "selected_smoke_rows": len(selected),
            "all_train_order_masks_messages_references_weights_and_eos_checked": True,
            "only_train_record_bodies_read": True,
            "selected_frozen_jsonl_row_bytes_preserved": True,
            "dev_test_record_bodies_copied": 0,
            "tokenizer_or_model_loaded": False,
            "inference_requests": 0,
            "training_updates": 0,
            "retokenized": False,
            "token_provenance": "Exact frozen payload hashes; structural checks do not re-run frozen semantic qualification or tokenization.",
        },
        "preparation_runtime": {
            "python": sys.version,
            "dependencies": "Python standard library only",
        },
        "reproduce_argv_from_project_root": [
            "python3",
            "ml/prepare_passive_sft.py",
            "--release",
            str(release.relative_to(root)),
            "--release-sha256",
            release_sha256,
            "--registry",
            str(registry.relative_to(root)),
            "--registry-sha256",
            registry_sha256,
            "--candidate",
            candidate_id,
            "--native-revision",
            native_revision,
            "--output",
            "<fresh-output-directory>",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(json_bytes(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--native-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        args.release,
        args.registry,
        args.candidate,
        args.native_revision,
        args.output,
        release_sha256=args.release_sha256,
        registry_sha256=args.registry_sha256,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_selection_status": result["candidate_selection_status"],
                "candidate_id": args.candidate,
                "output": str(args.output.resolve()),
                "manifest_sha256": sha256((args.output / "manifest.json").read_bytes()),
            }
        )
    )


if __name__ == "__main__":
    main()
