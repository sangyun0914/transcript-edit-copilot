"""Extract and verify the reviewed 42-row Gemma all-action training view.

The portable verifier uses only the standard library and exact reviewed train
bytes. It never imports the 56-scene authoring module or reads development bodies.
Native tokenizer verification and external anchor inspection occur locally during
preparation; runtime training must repeat native parity and anchor checks.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import struct

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-actions-gemma-training-preparation-v1"
STATUS = "frozen_train_only_actions_preparation"
CONTRACT = "passive-nep-actions-v1"
PILOT = "artifacts/datasets/nep-passive-actions-v1/pilot-draft-attempt-02"
PILOT_SHA = "37c15f589c7e163d1d7c642c9f632ab699424a488e10520a067883971a449742"
TRAIN_SHA = "eac97fca3e41f84dcd20fa79151efcf44b1578b35ae9f4e37797c800fde9c0b5"
TOKENS_SHA = "117f43904ffb723a7339936041870f5a9943a01250ebac229116df922d4611a4"
STAGE = "artifacts/benchmarks/passive-actions-v1/training-handoff/proposed-stage-attempt-01.json"
STAGE_SHA = "e66bf370a8f2699db13afac01e208abc022e9ff47be0f73f4b8cc66c9f244570"
REVIEW = "artifacts/benchmarks/passive-actions-v1/pilot-independent-review-attempt-01/review.json"
REVIEW_SHA = "2a8a933d549db43a87c3be7e30d23d5e4cb9c70099ff23896d7d347a9ac64ec3"
PROMPT_SHA = "34d3851779be239cd468d99d8c2718d4b591ed454e26a95480acd59bc5a3a98b"
TRAIN = "artifacts/training/passive-t1-gemma26-sft-v1/sft-attempt-01/train"
RELOAD = "artifacts/training/passive-t1-gemma26-sft-v1/sft-attempt-01/reload"
TRAIN_MANIFEST_SHA = "a6a035984cd6443d0aa341921abb8f753a01f91ce121996ac0c7d5921c325243"
RELOAD_MANIFEST_SHA = "6204367caef727a9815b6b10c0c63806f315abac599aca2ad2f7fce4b507144c"
FINAL = (
    "artifacts/training/passive-t1-gemma26-sft-v1/final-verification-attempt-01.json"
)
FINAL_SHA = "376c9e27211e0f2d2f795b18503bdb0e7b1bcd33c2a8de7fddb4f839d7c88230"
NATIVE_PIN = (
    "artifacts/training/passive-t1-gemma26-smoke-v1/native-acquisition/native-pin.json"
)
NATIVE_PIN_SHA = "3741c6d8e6f41a55cd47a90ca17c38221cd98df49be308e7d8a4235dece06fa5"
REPOSITORY = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
ADAPTER_SHA = "04c04afdb9b59d7c5b55a6137a4e886a241046239c7102f97a1257e59b1e79e9"
ADAPTER_BYTES = 74403016
ACTIONS = (
    "no_action",
    "correct_text",
    "delete_span",
    "trim_silence",
    "remove_silence",
    "split_paragraph",
    "merge_paragraph",
)
TRAIN_METADATA = {
    "configuration.json": "anchor/configuration.json",
    "run.json": "anchor/train-run.json",
    "runtime.json": "anchor/runtime.json",
    "exposure.json": "anchor/exposure.json",
    "adapter/adapter_config.json": "anchor/adapter-config.json",
    "adapter-tensors.json": "anchor/adapter-tensors.json",
}
RELOAD_METADATA = {
    "run.json": "anchor/reload-run.json",
    "reload-parity.json": "anchor/reload-parity.json",
}
FIXED_EVIDENCE = {
    "prompt.txt": ("ml/prompt-passive-actions-v1.txt", PROMPT_SHA),
    "evidence/proposed-stage.json": (STAGE, STAGE_SHA),
    "anchor/train-manifest.json": (TRAIN + "/manifest.json", TRAIN_MANIFEST_SHA),
    "anchor/reload-manifest.json": (RELOAD + "/manifest.json", RELOAD_MANIFEST_SHA),
    "anchor/final-verification.json": (FINAL, FINAL_SHA),
    "anchor/native-pin.json": (NATIVE_PIN, NATIVE_PIN_SHA),
}
ALLOWED = {
    *FIXED_EVIDENCE,
    *TRAIN_METADATA.values(),
    *RELOAD_METADATA.values(),
    "train-cases.jsonl",
    "train-tokens.jsonl.gz",
    "training-rows.jsonl.gz",
    "configuration.json",
    "draw-plan.json",
    "anchor.json",
    "native-tokenizer-verification.json",
    "evidence/pilot-review-reference.json",
    "evidence/pilot-manifest-reference.json",
    "source/prepare_passive_actions_training.py",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, values):
    raw = "".join(
        json.dumps(v, ensure_ascii=False, separators=(",", ":")) + "\n" for v in values
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0))


def copy_bound(source, target, expected):
    raw = source.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == expected,
        f"Bound input differs: {source.name}",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(raw)


def pilot_reference():
    return {
        "path": PILOT + "/manifest.json",
        "sha256": PILOT_SHA,
        "train_files_sha256": {
            "cases/train.jsonl": TRAIN_SHA,
            "tokenized/gemma26/json-curriculum-train.jsonl.gz": TOKENS_SHA,
        },
        "generator_source_sha256": "371cdfbf9a68d26dd8d15baa8a5fc08a604b64d02ab283b5d3bcb26484b71f3b",
        "contract_source_sha256": "375408411ae960037f932ad6c9162efcb7a982bdc1822d8fc6ca84910fac8a23",
        "original_manifest_copied": False,
    }


def target_names():
    return sorted(
        f"model.language_model.layers.{layer}.{component}"
        for layer in range(30)
        for component in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            *(("self_attn.v_proj",) if layer % 6 != 5 else ()),
        )
    )


def validate_cases(cases, tokens, prompt_sha256):
    require(len(cases) == len(tokens) == 42, "Exactly 42 training rows are required.")
    require(len({c["id"] for c in cases}) == 42, "Training case IDs must be unique.")
    cells, groups = {}, set()
    result = []
    for case, record in zip(cases, tokens, strict=True):
        require(
            case["split"] == record["split"] == case["source"]["split"] == "train"
            and case["id"] == record["id"],
            "Train-only row identity differs.",
        )
        action, language = case["primary"]["action"], case["language"]
        require(
            action in ACTIONS and language in ("en", "ko"),
            "Unexpected action/language.",
        )
        require(
            case["source"]["natural_audio"] is False
            and case["source"]["acoustically_verified"] is False,
            "Authored symbolic provenance cannot become an acoustic claim.",
        )
        cells[(action, language)] = cells.get((action, language), 0) + 1
        groups.add(case["source"]["source_group"])
        messages = record["messages"]
        require(
            messages == case["messages"]
            and len(messages) == 2
            and [m["role"] for m in messages] == ["system", "user"]
            and hashlib.sha256(messages[0]["content"].encode()).hexdigest()
            == prompt_sha256,
            "Canonical fixed-policy messages differ.",
        )
        value = json.loads(messages[1]["content"])
        require(
            value["contract_version"] == CONTRACT
            and value["allowed_actions"] == list(ACTIONS)
            and "task_brief" not in value
            and "reference" not in value,
            "Observable input contract differs.",
        )
        primary = record["primary"]
        ids, start = primary["input_ids"], primary["completion_start"]
        require(
            type(start) is int
            and 0 < start <= 3584
            and start < len(ids) <= 4096
            and len(ids) - start <= 512
            and all(type(n) is int and 0 <= n < 262144 for n in ids),
            "Token sequence or completion boundary is invalid.",
        )
        require(
            primary["eos_token_id"] == ids[-1] == 1
            and 1 not in ids[start:-1]
            and primary["completion_tokens"] == len(ids) - start
            and json.loads(primary["completion_text"]) == case["primary"]
            and record["prefix_ids_sha256"] == json_digest(ids[:start]),
            "Native completion, EOS, reference or prefix binding differs.",
        )
        result.append(
            {
                "case_id": case["id"],
                "input_ids": list(ids),
                "attention_mask": [1] * len(ids),
                "labels": [-100] * start + ids[start:],
                "completion_start": start,
                "provenance": {
                    "source_id": case["source"]["id"],
                    "source_group_id": case["source"]["source_group"],
                    "episode_id": case["id"],
                    "language": language,
                    "action": action,
                },
            }
        )
    require(
        len(cells) == 14 and set(cells.values()) == {3} and len(groups) == 6,
        "Balanced 42-row, six-group training census differs.",
    )
    return result


def planned(cases, tokens, stage):
    order = sorted(case["id"] for case in cases)
    random.Random(3407).shuffle(order)
    substantive, smoke = stage["proposed_substantive"], stage["proposed_smoke"]
    require(
        stage["proposed_recipe"]["sampling_seed_type"] == "int"
        and stage["proposed_recipe"]["one_pass_order_algorithm"]
        == "random.Random(3407).shuffle(sorted(train_case_ids))"
        and substantive["sample_order"] == order,
        "Fixed one-pass integer-seed order differs.",
    )
    by_id = {c["id"]: (c, t) for c, t in zip(cases, tokens, strict=True)}
    diagnostics = []
    for cell in smoke["cell_order"]:
        choices = [
            (len(t["primary"]["input_ids"]), c["id"])
            for c, t in by_id.values()
            if c["language"] == cell["language"]
            and c["primary"]["action"] == cell["action"]
        ]
        diagnostics.append(sorted(choices, key=lambda x: (-x[0], x[1]))[0][1])
    require(
        len(diagnostics) == len(set(diagnostics)) == 8
        and diagnostics
        == smoke["sample_order"]
        == smoke["diagnostics_order"]
        == substantive["diagnostics_order"],
        "Train-only diagnostics differ.",
    )
    phases = {}
    for name, chosen, expected, counts in (
        ("training", order, substantive, (42, 11, 2, 56276, 598)),
        ("smoke", diagnostics, smoke, (8, 2, 4, 10993, 106)),
    ):
        seq = sum(len(by_id[case_id][1]["primary"]["input_ids"]) for case_id in chosen)
        comp = sum(
            by_id[case_id][1]["primary"]["completion_tokens"] for case_id in chosen
        )
        actual = (
            len(chosen),
            math.ceil(len(chosen) / 4),
            (len(chosen) - 1) % 4 + 1,
            seq,
            comp,
        )
        require(
            actual == counts
            and actual
            == tuple(
                expected[k]
                for k in (
                    "draw_count",
                    "updates",
                    "final_accumulation",
                    "sequence_tokens",
                    "completion_tokens",
                )
            ),
            "Registered exact sequence/completion/update budgets differ.",
        )
        phases[name] = dict(
            zip(
                (
                    "draw_count",
                    "optimizer_updates",
                    "final_update_rows",
                    "max_total_sequence_tokens",
                    "max_total_completion_tokens",
                ),
                actual,
                strict=True,
            )
        )
        phases[name]["sample_order"] = chosen
    return phases


def validate_anchor_metadata(directory):
    train, reload = (
        read(directory / "anchor/train-manifest.json"),
        read(directory / "anchor/reload-manifest.json"),
    )
    for manifest, files in ((train, TRAIN_METADATA), (reload, RELOAD_METADATA)):
        for original, target in files.items():
            require(
                digest(directory / target) == manifest["files_sha256"][original],
                "Historical anchor payload does not match its original manifest.",
            )
    config, run, reloaded, parity, runtime, tensor_records, adapter, final = (
        read(directory / ("anchor/" + name))
        for name in (
            "configuration.json",
            "train-run.json",
            "reload-run.json",
            "reload-parity.json",
            "runtime.json",
            "adapter-tensors.json",
            "adapter-config.json",
            "final-verification.json",
        )
    )
    require(
        config["version"] == "passive-gemma-sft-v1"
        and config["phase"] == "substantive"
        and config["base"]["repository"] == REPOSITORY
        and config["base"]["revision"] == REVISION
        and config["training"]["max_updates"] == 203
        and config["training"]["draw_count"] == 809,
        "Anchor is not the genuine final T1 Gemma adapter.",
    )
    require(
        run["configuration_sha256"]
        == digest(directory / "anchor/configuration.json")
        == reloaded["configuration_sha256"]
        and run["training_updates"] == 203
        and run["consumed_rows"] == run["optimizer_committed_rows"] == 809
        and reloaded["status"] == "fresh_reload_passed_local_export_pending"
        and reloaded["previous_manifest_sha256"] == TRAIN_MANIFEST_SHA
        and reloaded["reloaded_training_updates"] == 203
        and run["pid"] != reloaded["pid"]
        and run["finished_unix"] <= reloaded["started_unix"]
        and parity["adapter_tensors_identical"] is True
        and len(parity["cases"]) == 8
        and all(c["identical"] for c in parity["cases"]),
        "Anchor final training/separate reload evidence differs.",
    )
    expected_names = {
        "base_model.model." + name + f".lora_{side}.weight"
        for name in target_names()
        for side in ("A", "B")
    }
    require(
        set(tensor_records) == expected_names
        and len(tensor_records) == 410
        and sorted(adapter["target_modules"]) == target_names()
        and adapter["r"] == adapter["lora_alpha"] == 16
        and adapter["revision"] == REVISION
        and adapter["base_model_name_or_path"] == REPOSITORY
        and all(r["dtype"] == "torch.float32" for r in tensor_records.values())
        and sum(math.prod(r["shape"]) for r in tensor_records.values()) == 18585600,
        "Anchor adapter scope, count, shapes or dtype differs.",
    )
    require(
        final["status"] == "passed"
        and final["training_updates"] == 203
        and final["committed_draws"] == 809
        and final["finite_adapter_tensors"] == 410
        and runtime["completion_label_eos_token_id"] == 1
        and config["generation"]["eos_token_id"] == [1, 106, 50]
        and train["files_sha256"]["adapter/adapter_model.safetensors"] == ADAPTER_SHA,
        "Independent historical final proof or native terminal bindings differ.",
    )
    return config, tensor_records


def inspect_anchor(path, records):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "Anchor changed during readback.",
    )
    require(
        len(raw) == ADAPTER_BYTES and hashlib.sha256(raw).hexdigest() == ADAPTER_SHA,
        "Actual external T1 adapter bytes differ.",
    )
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_size])
    header.pop("__metadata__", None)
    require(set(header) == set(records), "Saved adapter tensor names differ.")
    body = memoryview(raw)[8 + header_size :]
    spans = []
    for name, record in records.items():
        info = header[name]
        start, end = info["data_offsets"]
        require(
            info["dtype"] == "F32"
            and info["shape"] == record["shape"]
            and 0 <= start < end <= len(body)
            and end - start == math.prod(info["shape"]) * 4,
            "Saved adapter tensor shape/dtype/span differs.",
        )
        data = body[start:end]
        require(
            tensor_sha256(record["dtype"], record["shape"], data) == record["sha256"]
            and all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", data)),
            "Actual anchor tensor digest differs or contains a nonfinite value.",
        )
        spans.append((start, end))
    spans.sort()
    require(
        spans[0][0] == 0
        and spans[-1][1] == len(body)
        and all(a[1] == b[0] for a, b in zip(spans, spans[1:])),
        "Tensor payload spans have gaps or overlaps.",
    )
    return {
        "status": "actual_external_adapter_verified",
        "path": str(path.relative_to(ROOT)),
        "sha256": ADAPTER_SHA,
        "bytes": len(raw),
        "finite_fp32_tensors": len(records),
        "parameters": 18585600,
        "targets": 205,
        "copied_into_preparation": False,
        "stat_identity_stable_during_read": True,
    }


def tensor_sha256(dtype, shape, raw):
    header = json.dumps({"dtype": dtype, "shape": shape}, sort_keys=True).encode()
    value = hashlib.sha256(header)
    value.update(raw)
    return value.hexdigest()


def verify_native_tokenizer(tokenizer, tokens):
    require(
        (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id)
        == (1, 0, 2),
        "Gemma native special tokens differ.",
    )
    for record in tokens:
        primary = record["primary"]
        prefix = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        completion = tokenizer.encode(
            primary["completion_text"], add_special_tokens=False
        ) + [1]
        rendered = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        require(
            prefix == primary["input_ids"][: primary["completion_start"]]
            and completion == primary["input_ids"][primary["completion_start"] :]
            and rendered == record["rendered_prompt"],
            "Actual native tokenizer parity failed.",
        )
    return {
        "status": "all_actual_native_train_rows_verified",
        "rows": len(tokens),
        "prefix_and_completion_and_rendered_text_identical": True,
        "label_eos_token_id": 1,
        "local_files_only": True,
        "model_weights_loaded": False,
    }


def configuration(phases, anchor_config):
    return {
        "version": VERSION,
        "status": STATUS,
        "candidate_id": "gemma26",
        "contract_version": CONTRACT,
        "training_view": "json-curriculum",
        "native_base": {"repository": REPOSITORY, "revision": REVISION},
        "tokenizer": {
            "files_sha256": anchor_config["tokenizer"]["files_sha256"],
            "eos_token_id": 1,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "vocab_size": 262144,
        },
        "sampling": {
            "algorithm": "random.Random(3407).shuffle(sorted(train_case_ids))",
            "seed": 3407,
            "seed_type": "int",
            "replacement": False,
            "passes": 1,
            "draw_count": 42,
        },
        **phases,
        "diagnostic_case_order": phases["smoke"]["sample_order"],
        "initialization": "exact_verified_T1_adapter_on_original_native_base",
        "optimizer_initialization": "fresh_AdamW_not_T1_state_resume",
        "loss_mask": "completion_only_including_eos",
        "truncation": False,
        "packing": False,
        "dev_test_record_bodies_copied": 0,
    }


def derive(cases, tokens, stage, prompt_sha256):
    all_rows = validate_cases(cases, tokens, prompt_sha256)
    phases = planned(cases, tokens, stage)
    lookup = {row["case_id"]: row for row in all_rows}
    original_index = {case["id"]: i for i, case in enumerate(cases)}
    ordered = [
        {**lookup[case_id], "draw_index": i}
        for i, case_id in enumerate(phases["training"]["sample_order"])
    ]
    diagnostics = [
        {**lookup[case_id], "draw_index": i}
        for i, case_id in enumerate(phases["smoke"]["sample_order"])
    ]
    plan = [
        {
            "draw_index": i,
            "case_id": row["case_id"],
            "train_row_index": original_index[row["case_id"]],
        }
        for i, row in enumerate(ordered)
    ]
    return phases, ordered, diagnostics, plan


def verify_bundle(directory):
    directory = Path(directory).resolve()
    require(
        not any(p.is_symlink() for p in directory.rglob("*")),
        "Preparation cannot contain symlinks.",
    )
    inventory = {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    }
    require(
        inventory == ALLOWED | {"manifest.json"},
        "Preparation inventory is not the exact train-only allowlist.",
    )
    manifest = read(directory / "manifest.json")
    require(
        manifest["version"] == VERSION
        and manifest["status"] == STATUS
        and set(manifest["files_sha256"]) == ALLOWED,
        "Preparation manifest schema/inventory differs.",
    )
    for relative, expected in manifest["files_sha256"].items():
        require(
            digest(directory / relative) == expected,
            f"Preparation payload changed: {relative}",
        )
    require(
        digest(Path(__file__))
        == manifest["preparer_sha256"]
        == digest(directory / "source/prepare_passive_actions_training.py"),
        "Portable verifier source differs from its frozen snapshot.",
    )
    for relative, (_, expected) in FIXED_EVIDENCE.items():
        require(
            digest(directory / relative) == expected,
            "Fixed historical evidence differs.",
        )
    pilot = read(directory / "evidence/pilot-manifest-reference.json")
    require(
        pilot == pilot_reference(), "Reviewed train-only provenance projection differs."
    )
    require(
        read(directory / "evidence/pilot-review-reference.json")
        == {
            "path": REVIEW,
            "sha256": REVIEW_SHA,
            "status_verified_locally": "passed",
            "manifest_sha256": PILOT_SHA,
            "review_body_copied": False,
        },
        "Reviewed pilot evidence reference is missing.",
    )
    require(
        digest(directory / "train-cases.jsonl")
        == TRAIN_SHA
        == pilot["train_files_sha256"]["cases/train.jsonl"]
        and digest(directory / "train-tokens.jsonl.gz")
        == TOKENS_SHA
        == pilot["train_files_sha256"][
            "tokenized/gemma26/json-curriculum-train.jsonl.gz"
        ],
        "Exact reviewed training bytes differ.",
    )
    config, tensor_records = validate_anchor_metadata(directory)
    anchor = read(directory / "anchor.json")
    require(
        anchor["status"] == "actual_external_adapter_verified"
        and anchor["sha256"] == ADAPTER_SHA
        and anchor["bytes"] == ADAPTER_BYTES
        and anchor["finite_fp32_tensors"] == 410
        and anchor["copied_into_preparation"] is False,
        "Actual historical anchor readback is absent.",
    )
    cases, tokens = (
        rows(directory / "train-cases.jsonl"),
        rows(directory / "train-tokens.jsonl.gz"),
    )
    phases, ordered, diagnostics, plan = derive(
        cases,
        tokens,
        read(directory / "evidence/proposed-stage.json"),
        hashlib.sha256(
            (directory / "prompt.txt").read_text().strip().encode()
        ).hexdigest(),
    )
    prepared = configuration(phases, config)
    require(
        read(directory / "configuration.json") == prepared
        and read(directory / "draw-plan.json") == plan
        and rows(directory / "training-rows.jsonl.gz") == ordered,
        "Derived masks, order, provenance or exact budget differs.",
    )
    native = read(directory / "native-tokenizer-verification.json")
    require(
        native["status"] == "all_actual_native_train_rows_verified"
        and native["rows"] == 42
        and native["prefix_and_completion_and_rendered_text_identical"] is True,
        "Local native-tokenizer verification is missing.",
    )
    return {
        "manifest": manifest,
        "configuration": prepared,
        "cases": cases,
        "tokens": [
            {**record, "references": [record["primary"]["completion_text"]]}
            for record in tokens
        ],
        "rows": ordered,
        "draw_plan": plan,
        "diagnostics": diagnostics,
        "anchor": anchor,
        "anchor_tensor_records": tensor_records,
        "verification": {
            "status": "passed_train_only_actions_preparation",
            "train_rows": 42,
            "smoke_rows": 8,
            "optimizer_updates": 11,
            "final_update_rows": 2,
            "dev_test_record_bodies_read": 0,
            "model_weights_loaded": False,
            "external_anchor_rehashed_by_portable_verifier": False,
        },
    }


def prepare(output):
    from transformers import AutoTokenizer

    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh preparation directory.")
    initial_source = digest(Path(__file__))
    output.mkdir(parents=True)
    for relative, (source, expected) in FIXED_EVIDENCE.items():
        copy_bound(ROOT / source, output / relative, expected)
    require(
        digest(ROOT / PILOT / "manifest.json") == PILOT_SHA,
        "Original reviewed pilot manifest differs.",
    )
    pilot = read(ROOT / PILOT / "manifest.json")
    reference = pilot_reference()
    require(
        all(
            pilot["files_sha256"][name] == expected
            for name, expected in reference["train_files_sha256"].items()
        )
        and pilot["sources_sha256"]["ml/prepare_passive_actions.py"]
        == reference["generator_source_sha256"]
        and pilot["sources_sha256"]["ml/passive_actions.py"]
        == reference["contract_source_sha256"],
        "Original pilot does not bind the exact train-only extraction/source.",
    )
    write(output / "evidence/pilot-manifest-reference.json", reference)
    require(
        digest(ROOT / REVIEW) == REVIEW_SHA, "Original pilot review metadata differs."
    )
    review = read(ROOT / REVIEW)
    require(
        review["status"] == "passed"
        and review["manifest_sha256"] == PILOT_SHA
        and review["semantic_review"][
            "all_current_reference_previews_defensible_under_fixed_prompt"
        ]
        is True,
        "Original pilot review did not pass.",
    )
    write(
        output / "evidence/pilot-review-reference.json",
        {
            "path": REVIEW,
            "sha256": REVIEW_SHA,
            "status_verified_locally": "passed",
            "manifest_sha256": PILOT_SHA,
            "review_body_copied": False,
        },
    )
    train, reload = (
        read(output / "anchor/train-manifest.json"),
        read(output / "anchor/reload-manifest.json"),
    )
    for root, manifest, files in (
        (TRAIN, train, TRAIN_METADATA),
        (RELOAD, reload, RELOAD_METADATA),
    ):
        for original, target in files.items():
            copy_bound(
                ROOT / root / original,
                output / target,
                manifest["files_sha256"][original],
            )
    for original, target, expected in (
        ("cases/train.jsonl", "train-cases.jsonl", TRAIN_SHA),
        (
            "tokenized/gemma26/json-curriculum-train.jsonl.gz",
            "train-tokens.jsonl.gz",
            TOKENS_SHA,
        ),
    ):
        copy_bound(ROOT / PILOT / original, output / target, expected)
    config, records = validate_anchor_metadata(output)
    anchor = inspect_anchor(ROOT / TRAIN / "adapter/adapter_model.safetensors", records)
    write(output / "anchor.json", anchor)
    tokenizer_path = (
        ROOT / "artifacts/datasets/nep-release-v1/comparison-tokenizers-v2/gemma26"
    )
    for filename, expected in config["tokenizer"]["files_sha256"].items():
        require(
            digest(tokenizer_path / filename) == expected,
            "Actual native tokenizer files differ.",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    cases, tokens = (
        rows(output / "train-cases.jsonl"),
        rows(output / "train-tokens.jsonl.gz"),
    )
    write(
        output / "native-tokenizer-verification.json",
        verify_native_tokenizer(tokenizer, tokens),
    )
    phases, ordered, _, plan = derive(
        cases,
        tokens,
        read(output / "evidence/proposed-stage.json"),
        hashlib.sha256(
            (output / "prompt.txt").read_text().strip().encode()
        ).hexdigest(),
    )
    write(output / "configuration.json", configuration(phases, config))
    write(output / "draw-plan.json", plan)
    write_rows(output / "training-rows.jsonl.gz", ordered)
    copy_bound(
        Path(__file__),
        output / "source/prepare_passive_actions_training.py",
        initial_source,
    )
    manifest = {
        "version": VERSION,
        "status": STATUS,
        "candidate_id": "gemma26",
        "contract_version": CONTRACT,
        "pilot_manifest_sha256": PILOT_SHA,
        "preparer_sha256": initial_source,
        "files_sha256": {p: digest(output / p) for p in sorted(ALLOWED)},
        "train_rows": 42,
        "smoke_rows": 8,
        "dev_test_record_bodies_copied": 0,
        "pilot_generator_source_copied": False,
        "native_weights_copied": False,
        "anchor_adapter_copied": False,
        "training_updates_executed": 0,
        "anchor_manifest_sha256": TRAIN_MANIFEST_SHA,
        "initialization_scope": "Each phase warm-loads the genuine final T1 adapter; fresh optimizer; no smoke weight carryover.",
        "limitations": "Authored symbolic source-timed proposal pilot; reference choices are not exhaustive and no acoustic-safety claim is made.",
    }
    write(output / "manifest.json", manifest)
    return verify_bundle(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    bundle = verify_bundle(args.output) if args.verify_only else prepare(args.output)
    print(
        json.dumps(
            {
                "manifest_sha256": digest(args.output / "manifest.json"),
                **bundle["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
