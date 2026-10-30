"""Prepare a portable train-only seven-action Gemma JSON/Rewrite comparison."""

import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import passive_formats_actions_v3 as formats
import train_passive_gemma as gemma

VERSION = "gemma-actions-v3-preparation"
STATUS = "prepared_train_only_not_trained"
DRAW_ORDER_VERSION = "gemma-actions-v3-explicit-draw-order-1"
require, digest, read, child = (
    gemma.smoke.require,
    gemma.smoke.digest,
    gemma.smoke.read,
    gemma.smoke.child,
)
TOKENIZER_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
)
GENERATION = {
    "max_new_tokens": 512,
    "do_sample": False,
    "use_cache": True,
    "num_beams": 1,
    "eos_token_id": [1, 106, 50],
    "pad_token_id": 0,
    "bos_token_id": 2,
}


def value_hash(value):
    return hashlib.sha256(formats.compact(value).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rows(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, values):
    raw = ("\n".join(formats.compact(value) for value in values) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def references(case):
    values = [case["primary"], *case.get("acceptable", [])]
    return list({formats.compact(value): value for value in values}.values())


def validate_case(case):
    require(
        case["split"] == case["source"]["split"] == "train",
        "Only train case bodies are admitted.",
    )
    require(
        case.get("judgment", {}).get("status", "edit_required")
        in ("edit_required", "preserve"),
        "Unknown/inapplicable labels cannot become training abstentions.",
    )
    require(
        case["messages"] == formats.messages(case["input"]),
        "Case messages differ from visible input.",
    )
    require(case["language"] in ("en", "ko"), "Unexpected training language.")
    for action in references(case):
        for name in formats.FORMATS:
            text = formats.serialize(case["input"], action, name)
            require(
                formats.parse(text, case["input"], name).model_dump() == action,
                "Reference format round trip changed semantics.",
            )


def tokenize_cases(cases, tokenizer, selected_formats=formats.FORMATS):
    require(
        tuple(selected_formats) in (tuple(formats.FORMATS), ("json",)),
        "Use the matched formats or the full JSON cohort.",
    )
    require(
        len({case["id"] for case in cases}) == len(cases), "Duplicate training IDs."
    )
    admitted, encoded, exclusions = [], {name: [] for name in selected_formats}, []
    for case in cases:
        validate_case(case)
        counts = {
            name: len(formats.prefix_ids(tokenizer, case["input"], name))
            for name in formats.FORMATS
        }
        if any(counts[name] > formats.INPUT_LIMIT for name in selected_formats):
            exclusions.append(
                {
                    "case_id": case["id"],
                    "reason": "common_input_budget"
                    if len(selected_formats) == 2
                    else "json_input_budget",
                    "input_tokens": counts,
                }
            )
            continue
        values = {}
        try:
            for name in selected_formats:
                targets = [
                    formats.encode(tokenizer, case["input"], action, name)
                    for action in references(case)
                ]
                values[name] = {
                    "id": case["id"],
                    "split": "train",
                    "format": name,
                    "messages": formats.messages(case["input"], name),
                    "semantic_input_sha256": value_hash(
                        formats.contract.wire_snapshot(case["input"])
                    ),
                    "references": [target["completion_text"] for target in targets],
                    "primary": targets[0],
                }
        except ValueError as error:
            exclusions.append(
                {
                    "case_id": case["id"],
                    "reason": "common_completion_budget_or_terminal"
                    if len(selected_formats) == 2
                    else "json_completion_budget_or_terminal",
                    "error": str(error),
                }
            )
            continue
        admitted.append(case)
        for name in selected_formats:
            encoded[name].append(values[name])
    require(admitted, "No jointly eligible train cases.")
    return admitted, encoded, exclusions


def validate_record(case, record, format_name):
    require(
        record["id"] == case["id"]
        and record["split"] == "train"
        and record["format"] == format_name,
        "Token row identity differs.",
    )
    require(
        record["messages"] == formats.messages(case["input"], format_name)
        and record["semantic_input_sha256"]
        == value_hash(formats.contract.wire_snapshot(case["input"]))
        and record["references"]
        == [
            formats.serialize(case["input"], action, format_name)
            for action in references(case)
        ],
        "Token row visible input or reference identity differs.",
    )
    value = record["primary"]
    ids, start = value["input_ids"], value["completion_start"]
    require(
        type(start) is int
        and 0 < start <= formats.INPUT_LIMIT
        and start < len(ids) <= formats.CONTEXT_LIMIT
        and len(ids) - start <= formats.OUTPUT_LIMIT
        and all(type(token) is int and 0 <= token < 262144 for token in ids),
        "Invalid native token boundaries.",
    )
    require(
        value["labels"] == [-100] * start + ids[start:]
        and value["attention_mask"] == [1] * len(ids)
        and ids[-1] == value["eos_token_id"] == 1
        and not set(ids[start:-1]) & {1, 106, 50}
        and value["completion_tokens"] == len(ids) - start
        and value["completion_text"] == record["references"][0],
        "Completion masking/EOS/reference mismatch.",
    )


def flatten(case, record, index):
    return {
        **record["primary"],
        "case_id": case["id"],
        "draw_index": index,
        "source_id": case["source"]["id"],
        "source_group_id": case["source"]["source_group"],
        "episode_id": case.get("episode_id", case["id"]),
        "language": case["language"],
        "action": case["primary"]["action"],
    }


def budget(cases, tokens, order, accumulation):
    by_id = {
        case["id"]: (case, token) for case, token in zip(cases, tokens, strict=True)
    }
    selected = [
        flatten(*by_id[identity], index) for index, identity in enumerate(order)
    ]
    return {
        "draw_count": len(order),
        "max_updates": math.ceil(len(order) / accumulation),
        "final_update_rows": (len(order) - 1) % accumulation + 1,
        "max_total_sequence_tokens": sum(len(row["input_ids"]) for row in selected),
        "max_total_completion_tokens": sum(
            len(row["input_ids"]) - row["completion_start"] for row in selected
        ),
        "sample_order": order,
        "unique_cases": len(set(order)),
        "action_exposures": dict(
            sorted(Counter(row["action"] for row in selected).items())
        ),
        "language_exposures": dict(
            sorted(Counter(row["language"] for row in selected).items())
        ),
    }


def explicit_draw_order(path, cases, training_plan):
    require(Path(path).is_file(), "Missing registered explicit draw-order file.")
    value = read(path)
    require(
        isinstance(value, dict)
        and set(value) == {"version", "sample_order"}
        and value["version"] == DRAW_ORDER_VERSION
        and isinstance(value["sample_order"], list)
        and value["sample_order"]
        and all(isinstance(identity, str) for identity in value["sample_order"]),
        "Invalid explicit draw-order document.",
    )
    order = value["sample_order"]
    require(
        set(order) == {case["id"] for case in cases},
        "Explicit draw order must cover every eligible case and no foreign/skipped case.",
    )
    settings = {
        "draw_order_mode": "explicit",
        "draw_order_sha256": digest(path),
        "passes": 1,
        "replacement": len(order) != len(set(order)),
    }
    require(
        training_plan.get("sample_order") == order
        and all(
            training_plan.get(key) == expected for key, expected in settings.items()
        ),
        "Explicit draw-order hash/order/settings differ from the registered plan.",
    )
    return order, settings


def source_closure(entries):
    directory = Path(__file__).parent
    pending, found = list(entries), set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = directory / name
        require(path.is_file(), f"Missing portable source: {name}")
        found.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for module in names:
                local = str(module).split(".")[0] + ".py"
                if (directory / local).is_file():
                    pending.append(local)
    return sorted(found)


def manifest(directory, status=STATUS, selected_formats=formats.FORMATS):
    write(
        directory / "manifest.json",
        {
            "version": VERSION,
            "status": status,
            "train_only": True,
            "formats": list(selected_formats),
            "final_test_bodies_read": False,
            "files_sha256": {
                str(path.relative_to(directory)): digest(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file()
                and path.name != "manifest.json"
                and "__pycache__" not in path.parts
            },
        },
    )


def verify_bundle(directory):
    directory = Path(directory).resolve()
    receipt = read(directory / "manifest.json")
    require(
        receipt["version"] == VERSION
        and receipt["status"] == STATUS
        and receipt["train_only"] is True
        and receipt["final_test_bodies_read"] is False,
        "Unknown training preparation receipt.",
    )
    for name, expected in receipt["files_sha256"].items():
        gemma.smoke.check_file(child(directory, name), expected)
    cases = rows(directory / "train-cases.jsonl")
    require(
        cases and len({case["id"] for case in cases}) == len(cases),
        "Nonempty unique train cases required.",
    )
    selected_formats = receipt.get("formats", list(formats.FORMATS))
    require(
        tuple(selected_formats) in (tuple(formats.FORMATS), ("json",)),
        "Unregistered training formats.",
    )
    tokens = {
        name: rows(directory / f"tokens-{name}.jsonl.gz") for name in selected_formats
    }
    for case in cases:
        validate_case(case)
    for name, records in tokens.items():
        require(len(records) == len(cases), "Training row count differs.")
        for case, record in zip(cases, records, strict=True):
            validate_record(case, record, name)
    configs = {
        name: read(directory / f"config-{name}.json") for name in selected_formats
    }
    protocol_path = directory / "protocol.json"
    training_plan = (
        read(protocol_path).get("training_plan", {}) if protocol_path.is_file() else {}
    )
    draw_order_path = directory / "draw-order.json"
    if (
        draw_order_path.exists()
        or "draw_order_sha256" in training_plan
        or "draw_order_mode" in training_plan
        or any(
            "draw_order_sha256" in config["training"]
            or "draw_order_mode" in config["training"]
            for config in configs.values()
        )
    ):
        order, settings = explicit_draw_order(draw_order_path, cases, training_plan)
        require(
            all(
                config.get("phase") == "substantive"
                and config.get("protocol_sha256") == digest(protocol_path)
                and config["training"]["sample_order"] == order
                and all(
                    config["training"].get(key) == value
                    for key, value in settings.items()
                )
                for config in configs.values()
            ),
            "Prepared explicit draw order differs from its registered file.",
        )
    require(
        all(
            config["training"]["sample_order"]
            == configs["json"]["training"]["sample_order"]
            for config in configs.values()
        ),
        "Matched format draw orders differ.",
    )
    for name, config in configs.items():
        if config.get("phase") == "substantive":
            require(
                config["training"]["learning_rate"]
                == training_plan.get("learning_rate", 1e-4),
                "Prepared learning rate differs from its registered protocol.",
            )
        expected = budget(
            cases,
            tokens[name],
            config["training"]["sample_order"],
            config["training"]["gradient_accumulation"],
        )
        require(
            all(config["training"][key] == value for key, value in expected.items()),
            "Actual prepared token/draw budget differs.",
        )
    return {"receipt": receipt, "cases": cases, "tokens": tokens, "configs": configs}


def prepare(
    cases_path,
    native_pin,
    native,
    runtime_template,
    output,
    *,
    phase="engineering",
    max_cases=None,
    passes=1,
    seed=3407,
    accumulation=4,
    gpu_name=None,
    protocol=None,
    eligibility_format="common",
    draw_order=None,
    learning_rate=1e-4,
):
    output = Path(output).resolve()
    require(not output.exists(), "Use a new immutable preparation directory.")
    require(
        phase in ("engineering", "substantive") and passes > 0 and accumulation > 0,
        "Invalid training plan.",
    )
    require(
        type(learning_rate) in (int, float)
        and math.isfinite(learning_rate)
        and learning_rate > 0,
        "Learning rate must be finite and positive.",
    )
    require(
        phase == "engineering" or protocol is not None,
        "Substantive training requires a frozen protocol binding.",
    )
    require(eligibility_format in ("common", "json"), "Unknown training cohort.")
    selected_formats = formats.FORMATS if eligibility_format == "common" else ("json",)
    registration = read(protocol) if protocol else {}
    explicit_registered = any(
        key in registration.get("training_plan", {})
        for key in ("draw_order_sha256", "draw_order_mode")
    )
    require(
        (draw_order is not None) == explicit_registered,
        "Explicit draw-order input and registered plan must be supplied together.",
    )
    if draw_order is not None:
        require(
            phase == "substantive" and passes == 1 and max_cases is None,
            "Explicit draws require a substantive protocol, passes=1 and no case slicing.",
        )
    if phase == "substantive":
        require(
            registration["final_test_bodies_read"] is False
            and registration["dataset"]["train_cases_sha256"] == digest(cases_path)
            and registration["training_plan"]["seed"] == seed
            and registration["training_plan"]["passes"] == passes
            and registration["training_plan"]["gradient_accumulation"] == accumulation,
            "Substantive input or draw plan differs from its registered protocol.",
        )
        require(
            registration.get("eligibility_format", "common") == eligibility_format,
            "Training format cohort differs from the registered protocol.",
        )
        require(
            registration["training_plan"].get("learning_rate", 1e-4) == learning_rate,
            "Learning rate differs from the registered protocol.",
        )
    template, pin = read(runtime_template), read(native_pin)
    require(
        pin["repository"] == gemma.REPOSITORY and pin["revision"] == gemma.REVISION,
        "Native Gemma identity changed.",
    )
    for name in TOKENIZER_FILES:
        gemma.smoke.check_file(Path(native) / name, pin["files"][name]["sha256"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        native, local_files_only=True, trust_remote_code=False
    )
    all_cases = rows(cases_path)
    cases, tokens, excluded = tokenize_cases(all_cases, tokenizer, selected_formats)
    order = sorted(case["id"] for case in cases)
    random.Random(seed).shuffle(order)
    if max_cases is not None:
        require(
            phase == "engineering" and max_cases > 0,
            "Training case slicing is engineering-only.",
        )
        # A train-only engineering slice first covers observed action/language cells.
        by_id = {case["id"]: case for case in cases}
        cells, first, rest = set(), [], []
        for identity in order:
            case = by_id[identity]
            cell = (case["primary"]["action"], case["language"])
            (rest if cell in cells else first).append(identity)
            cells.add(cell)
        selected = set((first + rest)[:max_cases])
        order = [identity for identity in order if identity in selected]
        indices = [index for index, case in enumerate(cases) if case["id"] in selected]
        cases = [cases[index] for index in indices]
        tokens = {
            name: [records[index] for index in indices]
            for name, records in tokens.items()
        }
    draw_settings = {}
    if draw_order is None:
        order *= passes
    else:
        order, draw_settings = explicit_draw_order(
            draw_order, cases, registration["training_plan"]
        )
    output.mkdir(parents=True)
    if draw_order is not None:
        shutil.copyfile(draw_order, output / "draw-order.json")
        gemma.smoke.check_file(
            output / "draw-order.json", draw_settings["draw_order_sha256"]
        )
    write_rows(output / "train-cases.jsonl", cases)
    write(output / "exclusions.json", excluded)
    write(
        output / "source-binding.json",
        {
            "train_cases_sha256": digest(cases_path),
            "original_train_rows": len(all_cases),
            "admitted_rows": len(cases),
            "common_budget_exclusions": len(excluded),
            "eligibility_format": eligibility_format,
            "selected_formats": list(selected_formats),
            "budget_exclusions": len(excluded),
            "max_cases": max_cases,
        },
    )
    shutil.copyfile(native_pin, output / "native-pin.json")
    if protocol:
        shutil.copyfile(protocol, output / "protocol.json")
    (output / "native").mkdir()
    for name in pin["files"]:
        if not name.endswith(".safetensors") and (Path(native) / name).is_file():
            gemma.smoke.check_file(Path(native) / name, pin["files"][name]["sha256"])
            shutil.copyfile(Path(native) / name, output / "native" / name)
    source_files = source_closure(
        [
            "prepare_gemma_actions_v3.py",
            "train_gemma_actions_v3.py",
            "train_gemma_actions_v3_dpo.py",
            "acquire_passive_gemma.py",
            "export_gemma_actions_v3.py",
            "infer_gemma_actions_v3.py",
            "prepare_actions_v3_preferences.py",
        ]
    )
    (output / "source").mkdir()
    for name in source_files:
        shutil.copyfile(Path(__file__).with_name(name), output / "source" / name)
    for path in Path(__file__).parent.glob("prompt*.txt"):
        shutil.copyfile(path, output / "source" / path.name)
    for name in selected_formats:
        write_rows(output / f"tokens-{name}.jsonl.gz", tokens[name])
        config = {
            "version": "gemma-actions-v3-sft",
            "candidate_id": "gemma26",
            "phase": phase,
            "format": name,
            "contract_version": formats.contract.VERSION,
            "format_version": formats.VERSION,
            "libraries": template["libraries"],
            "absent_libraries": template["absent_libraries"],
            "model": copy.deepcopy(template["model"]),
            "generation": GENERATION,
            "base": {**template["base"], "path": "native"},
            "tokenizer": {**template["tokenizer"], "path": "native"},
            "native_pin_sha256": digest(output / "native-pin.json"),
            "protocol_sha256": digest(output / "protocol.json") if protocol else None,
            "sources_sha256": {
                path.name: digest(path)
                for path in sorted((output / "source").iterdir())
            },
            "training": {
                "seed": seed,
                "sampling_seed": seed,
                "micro_batch_size": 1,
                "gradient_accumulation": accumulation,
                "rank": 16,
                "alpha": 16,
                "dropout": 0,
                "bias": "none",
                "learning_rate": learning_rate,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "optimizer": "adamw_torch",
                "scheduler": "constant",
                "wall_limit_seconds": 21600,
                "checkpoint_every_updates": 25,
                "max_sequence_tokens": 4096,
                "max_completion_tokens": 512,
                "fresh_native_initialization": True,
                "truncation": False,
                "packing": False,
                "passes": passes,
                "replacement": False,
                **draw_settings,
                **budget(cases, tokens[name], order, accumulation),
            },
            "diagnostic_order": list(dict.fromkeys(order))[: min(7, len(cases))],
        }
        if gpu_name:
            config["model"]["gpu_name_contains"] = gpu_name
        write(output / f"config-{name}.json", config)
    manifest(output, selected_formats=selected_formats)
    verify_bundle(output)
    return {
        "status": STATUS,
        "cases": len(cases),
        "draws": len(order),
        "formats": list(selected_formats),
        "manifest_sha256": digest(output / "manifest.json"),
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for key in ("cases", "native-pin", "native", "runtime-template", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--phase", choices=("engineering", "substantive"), default="engineering"
    )
    p.add_argument("--max-cases", type=int)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gpu-name")
    p.add_argument("--protocol", type=Path)
    p.add_argument("--eligibility-format", choices=("common", "json"), default="common")
    p.add_argument("--draw-order", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("directory", type=Path)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["cases_path"] = args.pop("cases")
        print(json.dumps(prepare(**args)))
    else:
        bundle = verify_bundle(args["directory"])
        print(json.dumps({"status": "verified", "cases": len(bundle["cases"])}))


if __name__ == "__main__":
    main()
