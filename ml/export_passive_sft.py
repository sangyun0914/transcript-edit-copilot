"""Export one pinned Qwen3.5-9B LoRA as a paired native-derived GGUF experiment.

Run preflight, convert, verify, quantize and receipt in separate processes with
the same --config and --output. Inputs are local, hash-bound and never modified.
No acquisition, training, inference, dataset access or model promotion occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

import export_qwen06
from export_qwen06 import check_hash, sha256, write_new

VERSION = "passive-sft-export-v1"
BASE_REPOSITORY = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
RUNTIME_REVISION = "427291b5b34cd914a31b3fd3b61a68f6184f4b9f"
PROJECTIONS = {
    ("self_attn", "q_proj"): "attn_q",
    ("self_attn", "k_proj"): "attn_k",
    ("self_attn", "v_proj"): "attn_v",
    ("self_attn", "o_proj"): "attn_output",
    ("mlp", "gate_proj"): "ffn_gate",
    ("mlp", "up_proj"): "ffn_up",
    ("mlp", "down_proj"): "ffn_down",
}
GGUF_FILES = {
    "base": "base-F16.gguf",
    "adapter": "adapter-F32.gguf",
    "merged": "merged-F16.gguf",
    "base_quantized": "base-Q4_K_M.gguf",
    "merged_quantized": "merged-Q4_K_M.gguf",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(path, parent):
    path = Path(path)
    return (path if path.is_absolute() else parent / path).resolve()


def relative_file(directory, name):
    path = (directory / name).resolve()
    require(
        not Path(name).is_absolute() and path.is_relative_to(directory.resolve()),
        f"Bound file escapes its directory: {name}",
    )
    return path


def check_files(directory, files):
    require(
        isinstance(files, dict) and files, "A nonempty file/hash binding is required."
    )
    for name, expected in files.items():
        require(re.fullmatch(r"[0-9a-f]{64}", expected), f"Invalid SHA256: {name}")
        check_hash(relative_file(directory, name), expected)


def tensor_header(path):
    """Read shapes without materializing native multi-gigabyte tensors."""
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, f"Missing safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        require(0 < length <= 64 * 1024 * 1024, "Invalid safetensors header length.")
        header = json.loads(stream.read(length))
    data_size = path.stat().st_size - 8 - length
    result = {}
    sizes = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    ranges = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = value["shape"], value["data_offsets"]
        require(
            value["dtype"] in sizes
            and isinstance(shape, list)
            and all(type(n) is int and n > 0 for n in shape)
            and len(offsets) == 2
            and all(type(n) is int for n in offsets)
            and 0 <= offsets[0] <= offsets[1] <= data_size
            and offsets[1] - offsets[0] == math.prod(shape) * sizes[value["dtype"]],
            f"Invalid safetensors tensor metadata: {name}",
        )
        ranges.append(tuple(offsets))
        result[name] = {"shape": shape, "dtype": value["dtype"]}
    ranges.sort()
    require(
        bool(ranges)
        and ranges[0][0] == 0
        and ranges[-1][1] == data_size
        and all(left[1] == right[0] for left, right in zip(ranges, ranges[1:])),
        "Safetensors data has a hole, overlap or trailing bytes.",
    )
    return result


def native_inventory(directory, files):
    index_name = "model.safetensors.index.json"
    require(index_name in files, "Bind the native shard index.")
    weight_map = json.loads((directory / index_name).read_text())["weight_map"]
    shards = set(weight_map.values())
    require(shards and shards <= set(files), "Bind every native weight shard.")
    require(
        shards == {name for name in files if name.endswith(".safetensors")},
        "Native binding contains weights outside the declared shard index.",
    )
    inventory = {}
    for shard in sorted(shards):
        for name, metadata in tensor_header(relative_file(directory, shard)).items():
            require(name not in inventory, f"Duplicate native tensor: {name}")
            require(
                weight_map.get(name) == shard, f"Native index/header mismatch: {name}"
            )
            inventory[name] = {**metadata, "shard": shard}
    require(
        set(inventory) == set(weight_map), "Native index references missing tensors."
    )
    return inventory


def mapped_native_name(name, layers):
    match = re.fullmatch(
        r"model\.language_model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)\.weight",
        name,
    )
    require(
        match is not None,
        f"Only ordinary text attention/MLP adapters are supported: {name}",
    )
    layer, family, projection = match.groups()
    require(
        0 <= int(layer) < layers and (family, projection) in PROJECTIONS,
        f"Unsupported adapter target: {name}",
    )
    return f"blk.{layer}.{PROJECTIONS[(family, projection)]}.weight"


def validate_adapter(
    config,
    inventory,
    adapter,
    layers,
    expected_base,
    *,
    trained_modules=None,
    ignored_native_names=(),
):
    require(
        config.get("base_model_name_or_path") == expected_base,
        "Adapter base identity differs.",
    )
    require(
        config.get("revision") in (None, BASE_REVISION),
        "Adapter native revision differs.",
    )
    require(
        config.get("peft_type") == "LORA" and config.get("bias") == "none",
        "Use an ordinary bias-free LoRA.",
    )
    for key in (
        "lora_bias",
        "fan_in_fan_out",
        "use_dora",
        "use_rslora",
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "layer_replication",
        "target_parameters",
        "alora_invocation_tokens",
        "trainable_token_indices",
        "use_qalora",
        "use_bdlora",
        "velora_config",
        "layers_to_transform",
        "layers_pattern",
        "exclude_modules",
        "megatron_config",
        "loftq_config",
        "eva_config",
        "corda_config",
    ):
        require(not config.get(key), f"Unsupported adapter configuration: {key}")
    require(
        config.get("init_lora_weights", True) in (True, False, "gaussian"),
        "Initialization that can replace native base weights is unsupported.",
    )
    rank, alpha = config.get("r"), config.get("lora_alpha")
    require(type(rank) is int and rank > 0, "LoRA rank must be a positive integer.")
    require(
        type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0,
        "LoRA alpha must be positive and finite.",
    )
    targets = config.get("target_modules")
    require(
        isinstance(targets, str)
        and bool(targets)
        or isinstance(targets, list)
        and bool(targets)
        and all(isinstance(target, str) and target for target in targets),
        "Bind target_modules as an explicit list or regex.",
    )

    def selected(module):
        if isinstance(targets, str):
            return re.fullmatch(targets, module) is not None
        return any(
            module == target or module.endswith("." + target) for target in targets
        )

    ignored = set(ignored_native_names)
    require(
        len(ignored) == len(ignored_native_names)
        and ignored == {name for name in inventory if name.startswith("mtp.")},
        "Native MTP exclusions differ from the preregistered training census.",
    )
    # PEFT may compact qualified module names to suffixes when saving. Native MTP
    # weights were never loaded into the trained model and are not adapter targets.
    intended = {
        name
        for name in inventory
        if name not in ignored and name.endswith(".weight") and selected(name[:-7])
    }
    require(intended, "Adapter target configuration matches no native weights.")
    for name in intended:
        mapped_native_name(name, layers)
    if trained_modules is not None:
        require(
            len(set(trained_modules)) == len(trained_modules)
            and intended == {name + ".weight" for name in trained_modules},
            "Saved adapter targets differ from the actual trained text module census.",
        )
    pairs = {}
    for name, meta in adapter.items():
        match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", name)
        require(match is not None, f"Unsupported adapter tensor: {name}")
        module, side = match.groups()
        native = module + ".weight"
        mapped = mapped_native_name(native, layers)
        require(native in intended, f"Undeclared or absent native target: {native}")
        shape = inventory[native]["shape"]
        require(
            len(shape) == 2 and inventory[native]["dtype"] in {"F16", "BF16", "F32"},
            f"Expected a floating native matrix: {native}",
        )
        expected = [rank, shape[1]] if side == "A" else [shape[0], rank]
        require(
            meta["shape"] == expected and meta["dtype"] in {"F32", "F16", "BF16"},
            f"Adapter/native shape or dtype mismatch: {name}",
        )
        pair = pairs.setdefault(
            native,
            {
                "gguf_name": mapped,
                "shape": shape,
                "shard": inventory[native]["shard"],
                "tensors": {},
            },
        )
        require(side not in pair["tensors"], f"Duplicate adapter side: {name}")
        pair["tensors"][side] = name
    require(
        set(pairs) == intended
        and all(set(p["tensors"]) == {"A", "B"} for p in pairs.values()),
        "Adapter tensors do not cover exactly the declared native targets.",
    )
    return pairs


def training_census(config, inventory):
    record_path = Path(config["training_record"]["path"])
    check_hash(record_path, config["training_record"]["sha256"])
    record = json.loads(record_path.read_text())
    manifest_path = (
        record_path
        if "files_sha256" in record
        else record_path.parent / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files_sha256"]
    directory = manifest_path.parent
    bound = {str(manifest_path): sha256(manifest_path)}
    values = {}
    for name in ("run.json", "configuration.json", "runtime.json"):
        path = directory / name
        check_hash(path, files[name])
        bound[str(path)] = files[name]
        values[name] = json.loads(path.read_text())
    run, training, runtime = (
        values[name] for name in ("run.json", "configuration.json", "runtime.json")
    )
    require(
        record_path == manifest_path or record_path == directory / "run.json",
        "Bind a training run record or its manifest.",
    )
    require(
        run["mode"] == "train"
        and run["training_updates"] > 0
        and run["status"] == "training_complete_fresh_reload_pending"
        and run["configuration_sha256"] == files["configuration.json"],
        "The module census must belong to a completed, bound training run.",
    )
    require(
        training["base"]["repository"] == config["base"]["repository"]
        and training["base"]["revision"] == config["base"]["revision"],
        "Training census belongs to a different native model.",
    )
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        require(
            files["adapter/" + name] == config["adapter"]["files_sha256"][name],
            "The module census belongs to a different saved adapter.",
        )
    declared = training["model"]["target_modules"]
    targets = runtime["targets"]
    require(
        declared
        and len(set(declared)) == len(declared)
        and set(targets) == set(declared),
        "Runtime text module census differs from the training configuration.",
    )
    ignored = training["model"]["ignored_native_tensor_names"]
    require(
        len(set(ignored)) == len(ignored)
        and set(ignored)
        == set(runtime["ignored_native_tensor_names"])
        == {name for name in inventory if name.startswith("mtp.")},
        "Native, runtime and preregistered MTP exclusions differ.",
    )
    for name, value in targets.items():
        require(
            name + ".weight" in inventory
            and value["class"] in {"Linear", "Linear4bit"}
            and inventory[name + ".weight"]["shape"]
            == [value["out_features"], value["in_features"]],
            f"Native tensor shape differs from the trained runtime module: {name}",
        )
    return {
        "files_sha256": bound,
        "targets": targets,
        "ignored_native_tensor_names": ignored,
        "rank": training["training"]["rank"],
        "alpha": training["training"]["alpha"],
    }


def configuration(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    require(config.get("version") == VERSION, "Unknown export configuration.")
    require(
        config["base"]["repository"] == BASE_REPOSITORY
        and config["base"]["revision"] == BASE_REVISION,
        "This bounded export supports only the selected native Qwen3.5-9B revision.",
    )
    require(
        config["runtime"]["revision"] == RUNTIME_REVISION,
        "Use the verified b10816 conversion/runtime revision.",
    )
    for key in ("base", "adapter"):
        config[key]["path"] = str(resolve(config[key]["path"], path.parent))
    config["training_record"]["path"] = str(
        resolve(config["training_record"]["path"], path.parent)
    )
    runtime = config["runtime"]
    runtime["source_path"] = str(resolve(runtime["source_path"], path.parent))
    # Python discovers its virtual environment through the invoked executable path.
    # Resolving .venv/bin/python's symlink discards that environment.
    runtime["python"] = os.path.abspath(path.parent / runtime["python"])
    for tool in ("llama-export-lora", "llama-quantize"):
        runtime["tools"][tool]["path"] = str(
            resolve(runtime["tools"][tool]["path"], path.parent)
        )
    runtime["support_files_sha256"] = {
        str(resolve(name, path.parent)): expected
        for name, expected in runtime.get("support_files_sha256", {}).items()
    }
    return config


def validate_inputs(config):
    for key in ("base", "adapter"):
        check_files(Path(config[key]["path"]), config[key]["files_sha256"])
    required = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
    }
    require(
        required <= set(config["base"]["files_sha256"]),
        "Bind native config, index, tokenizer, template and license.",
    )
    require(
        {"adapter_config.json", "adapter_model.safetensors"}
        <= set(config["adapter"]["files_sha256"]),
        "Bind adapter configuration and weights.",
    )
    record = config["training_record"]
    check_hash(Path(record["path"]), record["sha256"])
    runtime = config["runtime"]
    source = Path(runtime["source_path"])
    source_files = runtime["source_files_sha256"]
    needed = {"convert_hf_to_gguf.py", "convert_lora_to_gguf.py"} | {
        str(path.relative_to(source))
        for directory in ("conversion", "gguf-py")
        for path in (source / directory).rglob("*.py")
    }
    require(needed <= set(source_files), "Bind all converter and GGUF Python sources.")
    check_files(source, source_files)
    require(Path(runtime["python"]).is_file(), "Converter Python is unavailable.")
    for tool in runtime["tools"].values():
        check_hash(Path(tool["path"]), tool["sha256"])
    for name, expected in runtime["support_files_sha256"].items():
        check_hash(Path(name), expected)


def preflight(config_path, output):
    config = configuration(config_path)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "configuration.json", config)
    try:
        validate_inputs(config)
        base = Path(config["base"]["path"])
        native_config = json.loads((base / "config.json").read_text())
        require(
            native_config.get("model_type") == "qwen3_5"
            and not native_config.get("quantization_config"),
            "Expected original, unquantized Qwen3.5 native weights.",
        )
        layers = native_config["text_config"]["num_hidden_layers"]
        require(
            type(layers) is int and layers > 0, "Native text layer count is invalid."
        )
        inventory = native_inventory(base, config["base"]["files_sha256"])
        census = training_census(config, inventory)
        adapter = Path(config["adapter"]["path"])
        adapter_config = json.loads((adapter / "adapter_config.json").read_text())
        require(
            adapter_config["r"] == census["rank"]
            and adapter_config["lora_alpha"] == census["alpha"],
            "Saved adapter rank/scaling differs from the training configuration.",
        )
        pairs = validate_adapter(
            adapter_config,
            inventory,
            tensor_header(adapter / "adapter_model.safetensors"),
            layers,
            config["adapter"].get("base_model_name_or_path", BASE_REPOSITORY),
            trained_modules=list(census["targets"]),
            ignored_native_names=census["ignored_native_tensor_names"],
        )
        report = {
            "status": "passed",
            "version": VERSION,
            "config_sha256": sha256(output / "configuration.json"),
            "implementation_sha256": sha256(Path(__file__)),
            "shared_helpers_sha256": sha256(Path(export_qwen06.__file__)),
            "native_tensor_count": len(inventory),
            "text_layers": layers,
            "adapter_tensor_count": 2 * len(pairs),
            "pairs": pairs,
            "training_census": census,
            "template_sha256": sha256(base / "chat_template.jinja"),
            "rank": adapter_config["r"],
            "alpha": adapter_config["lora_alpha"],
            "scope": "Hashes, complete native shard index and actual adapter/native tensor shapes; no tensor arithmetic or inference.",
        }
        write_new(output / "preflight.json", report)
        return report
    except Exception as error:
        write_new(
            output / "preflight-error.json", {"status": "failed", "error": str(error)}
        )
        raise


def bound_run(config_path, output):
    config = configuration(config_path)
    require(
        config == json.loads((output / "configuration.json").read_text()),
        "Export configuration changed after preflight.",
    )
    report = json.loads((output / "preflight.json").read_text())
    require(report["status"] == "passed", "Preflight must pass.")
    check_hash(output / "configuration.json", report["config_sha256"])
    check_hash(Path(__file__), report["implementation_sha256"])
    check_hash(Path(export_qwen06.__file__), report["shared_helpers_sha256"])
    for name, expected in report["training_census"]["files_sha256"].items():
        check_hash(Path(name), expected)
    validate_inputs(config)
    return config, report


def run_command(output, name, argv):
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    stem = logs / name
    env = os.environ.copy()
    overrides = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
    }
    env.update(overrides)
    command = [str(value) for value in argv]
    write_new(
        stem.with_suffix(".command.json"),
        {
            "argv": command,
            "environment_overrides": overrides,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    with stem.with_suffix(".log").open("x") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=stream)
    write_new(
        stem.with_suffix(".result.json"),
        {
            "returncode": result.returncode,
            "log_sha256": sha256(stem.with_suffix(".log")),
        },
    )
    require(
        result.returncode == 0,
        f"{name} failed; preserve this attempt and use a new output directory.",
    )


def paths(output):
    return {key: output / name for key, name in GGUF_FILES.items()}


def refuse_existing(*values):
    for path in values:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite export artifact: {path}")


def artifact(path):
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def conversion_commands(config, output):
    runtime, target = config["runtime"], paths(output)
    source = Path(runtime["source_path"])
    return [
        (
            "convert-base",
            [
                runtime["python"],
                source / "convert_hf_to_gguf.py",
                config["base"]["path"],
                "--outfile",
                target["base"],
                "--outtype",
                "f16",
            ],
        ),
        (
            "convert-adapter",
            [
                runtime["python"],
                source / "convert_lora_to_gguf.py",
                config["adapter"]["path"],
                "--base",
                config["base"]["path"],
                "--outfile",
                target["adapter"],
                "--outtype",
                "f32",
            ],
        ),
        (
            "merge",
            [
                runtime["tools"]["llama-export-lora"]["path"],
                "-m",
                target["base"],
                "--lora",
                target["adapter"],
                "-o",
                target["merged"],
                "-t",
                "4",
            ],
        ),
    ]


def convert(config_path, output):
    config, _ = bound_run(config_path, output)
    target = paths(output)
    refuse_existing(
        output / "conversion.json",
        *(target[key] for key in ("base", "adapter", "merged")),
    )
    for name, command in conversion_commands(config, output):
        run_command(output, name, command)
    report = {
        "status": "converted_tensor_verification_pending",
        "artifacts": {
            key: artifact(target[key]) for key in ("base", "adapter", "merged")
        },
    }
    write_new(output / "conversion.json", report)
    return report


def gguf_module(config):
    source = Path(config["runtime"]["source_path"]) / "gguf-py"
    sys.path.insert(0, str(source))
    import gguf

    require(
        Path(gguf.__file__).resolve().is_relative_to(source),
        "A different GGUF package is already loaded.",
    )
    return gguf


def matching_metadata(left, right):
    def fields(reader):
        return {
            name: field.contents()
            for name, field in reader.fields.items()
            if not name.startswith("GGUF.")
        }

    require(fields(left) == fields(right), "Base and adapted GGUF metadata differ.")


def verify(config_path, output):
    config, preflight_report = bound_run(config_path, output)
    refuse_existing(output / "tensor-verification.json")
    import numpy as np
    import peft
    import torch
    from peft.tuners.lora.layer import Linear
    from safetensors import safe_open
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    gguf = gguf_module(config)
    converted = json.loads((output / "conversion.json").read_text())
    for value in converted["artifacts"].values():
        check_hash(Path(value["path"]), value["sha256"])
    target = paths(output)
    readers = {
        key: gguf.GGUFReader(target[key]) for key in ("base", "adapter", "merged")
    }
    tensors = {
        key: {tensor.name: tensor for tensor in reader.tensors}
        for key, reader in readers.items()
    }
    for reader in readers.values():
        require(
            reader.get_field("general.architecture").contents() == "qwen35",
            "GGUF architecture changed.",
        )
    matching_metadata(readers["base"], readers["merged"])
    require(
        set(tensors["base"]) == set(tensors["merged"]),
        "Base/merged tensor inventories differ.",
    )
    require(
        readers["adapter"].get_field("adapter.lora.alpha").contents()
        == preflight_report["alpha"],
        "GGUF LoRA scaling changed.",
    )
    template = readers["base"].get_field("tokenizer.chat_template").contents()
    require(
        template.encode()
        == (Path(config["base"]["path"]) / "chat_template.jinja").read_bytes(),
        "Native/GGUF chat templates differ.",
    )
    adapter_dir = Path(config["adapter"]["path"])
    original = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")
    pairs = preflight_report["pairs"]
    require(
        len(original) == len(tensors["adapter"]) == 2 * len(pairs),
        "Adapter tensor count changed.",
    )
    expected_adapter = {
        f"{pair['gguf_name']}.lora_{side.lower()}"
        for pair in pairs.values()
        for side in ("A", "B")
    }
    require(
        set(tensors["adapter"]) == expected_adapter,
        "Converted adapter target set differs.",
    )
    lora_config = peft.LoraConfig.from_pretrained(adapter_dir)
    reports = []
    with torch.no_grad():
        for native_name, pair in sorted(pairs.items()):
            name = pair["gguf_name"]
            base_tensor, merged_tensor = tensors["base"][name], tensors["merged"][name]
            require(
                base_tensor.tensor_type
                == merged_tensor.tensor_type
                == gguf.GGMLQuantizationType.F16,
                f"Expected F16 adapted base matrix: {name}",
            )
            base, merged = base_tensor.data, merged_tensor.data
            with safe_open(
                Path(config["base"]["path"]) / pair["shard"],
                framework="pt",
                device="cpu",
            ) as native:
                require(
                    np.array_equal(native.get_tensor(native_name).half().numpy(), base),
                    f"Native/F16 target mismatch: {name}",
                )
            require(
                list(base.shape) == list(merged.shape) == pair["shape"],
                f"Converted target shape differs: {name}",
            )
            ab = {}
            for side, tensor_name in pair["tensors"].items():
                value = original[tensor_name].float()
                converted_tensor = tensors["adapter"][f"{name}.lora_{side.lower()}"]
                require(
                    converted_tensor.tensor_type == gguf.GGMLQuantizationType.F32
                    and np.array_equal(value.numpy(), converted_tensor.data)
                    and bool(torch.isfinite(value).all()),
                    f"Converted adapter value mismatch/nonfinite: {tensor_name}",
                )
                ab[side] = value
            linear = torch.nn.Linear(
                base.shape[1], base.shape[0], bias=False, device="meta"
            )
            linear.weight = torch.nn.Parameter(
                torch.from_numpy(base.astype(np.float32))
            )
            layer = Linear(
                linear,
                adapter_name="default",
                config=lora_config,
                r=preflight_report["rank"],
                lora_alpha=preflight_report["alpha"],
            )
            layer.lora_A["default"].weight.copy_(ab["A"])
            layer.lora_B["default"].weight.copy_(ab["B"])
            layer.merge(safe_merge=True)
            expected = layer.get_base_layer().weight.half().numpy()
            error = np.abs(merged.astype(np.float32) - expected.astype(np.float32))
            spacing = np.spacing(np.abs(expected)).astype(np.float32)
            require(
                np.isfinite(expected).all()
                and np.isfinite(merged).all()
                and np.all(error <= spacing),
                f"Merge differs from PEFT by more than one F16 ULP: {name}",
            )
            reports.append(
                {
                    "name": name,
                    "native_name": native_name,
                    "shape": pair["shape"],
                    "changed_elements": int(np.count_nonzero(merged != base)),
                    "max_abs_peft_f16_error": float(error.max()),
                }
            )
            del layer, linear, expected, error, spacing
    adapted = {pair["gguf_name"] for pair in pairs.values()}
    unchanged = []
    for name, base in tensors["base"].items():
        other = tensors["merged"][name]
        require(
            base.tensor_type == other.tensor_type
            and np.array_equal(base.shape, other.shape),
            f"Base/merged tensor type/shape differs: {name}",
        )
        if name not in adapted:
            require(
                np.array_equal(base.data, other.data) and np.isfinite(base.data).all(),
                f"Untouched tensor changed or nonfinite: {name}",
            )
            unchanged.append(name)
    report = {
        "status": "passed",
        "adapter_tensors_exact": len(original),
        "native_targets_exact": len(pairs),
        "merged_targets": reports,
        "unadapted_tensors_exact": unchanged,
        "oracle": f"PEFT {peft.__version__} Linear.merge, F32 accumulation of native-F16 base and adapter, at most one F16 ULP",
        "torch_version": torch.__version__,
        "template_sha256": preflight_report["template_sha256"],
        "eos_token_id": readers["base"]
        .get_field("tokenizer.ggml.eos_token_id")
        .contents(),
        "artifacts": converted["artifacts"],
        "scope": "Fresh-process tensor verification only; NF4 training/reload and native-F16 deployment are different numerical conditions.",
    }
    write_new(output / "tensor-verification.json", report)
    return report


def quantize(config_path, output):
    config, _ = bound_run(config_path, output)
    report = json.loads((output / "tensor-verification.json").read_text())
    require(
        report["status"] == "passed",
        "Fresh tensor verification must pass before quantization.",
    )
    target = paths(output)
    refuse_existing(
        output / "quantization.json",
        target["base_quantized"],
        target["merged_quantized"],
    )
    for key in ("base", "merged"):
        check_hash(target[key], report["artifacts"][key]["sha256"])
        run_command(
            output,
            f"quantize-{key}",
            [
                config["runtime"]["tools"]["llama-quantize"]["path"],
                target[key],
                target[f"{key}_quantized"],
                "Q4_K_M",
                "4",
            ],
        )
    result = {
        "status": "quantized_pair_verification_pending",
        "quantization": "Q4_K_M",
        "importance_matrix": None,
        "threads": 4,
        "artifacts": {
            key: artifact(target[key]) for key in ("base_quantized", "merged_quantized")
        },
    }
    write_new(output / "quantization.json", result)
    return result


def quantized_changes(base, merged, adapted):
    import numpy as np

    require(set(base) == set(merged), "Quantized tensor inventories differ.")
    changed = []
    for name, tensor in base.items():
        other = merged[name]
        require(
            tensor.tensor_type == other.tensor_type
            and np.array_equal(tensor.shape, other.shape),
            f"Quantization recipe differs: {name}",
        )
        if not np.array_equal(tensor.data, other.data):
            require(name in adapted, f"Untouched quantized tensor changed: {name}")
            changed.append(name)
    return sorted(changed)


def receipt(config_path, output):
    config, preflight_report = bound_run(config_path, output)
    refuse_existing(output / "receipt.json")
    verification = json.loads((output / "tensor-verification.json").read_text())
    quantization = json.loads((output / "quantization.json").read_text())
    require(verification["status"] == "passed", "Tensor verification must pass.")
    artifacts = {**verification["artifacts"], **quantization["artifacts"]}
    for value in artifacts.values():
        check_hash(Path(value["path"]), value["sha256"])
    gguf = gguf_module(config)
    readers = {
        key: gguf.GGUFReader(artifacts[key]["path"])
        for key in ("base_quantized", "merged_quantized")
    }
    matching_metadata(readers["base_quantized"], readers["merged_quantized"])
    for reader in readers.values():
        require(
            reader.get_field("general.architecture").contents() == "qwen35",
            "Quantized architecture changed.",
        )
        require(
            reader.get_field("tokenizer.chat_template").contents().encode()
            == (Path(config["base"]["path"]) / "chat_template.jinja").read_bytes(),
            "Quantization changed the template.",
        )
        require(
            reader.get_field("tokenizer.ggml.eos_token_id").contents()
            == verification["eos_token_id"],
            "Quantization changed EOS.",
        )
    tensors = {
        key: {tensor.name: tensor for tensor in reader.tensors}
        for key, reader in readers.items()
    }
    adapted = {pair["gguf_name"] for pair in preflight_report["pairs"].values()}
    changed = quantized_changes(
        tensors["base_quantized"], tensors["merged_quantized"], adapted
    )
    report = {
        "version": VERSION,
        "status": "tensor_verified_paired_export_runtime_preflight_pending",
        "configuration_sha256": preflight_report["config_sha256"],
        "tensor_verification_sha256": sha256(output / "tensor-verification.json"),
        "training_record": config["training_record"],
        "base_repository": BASE_REPOSITORY,
        "base_revision": BASE_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "artifacts": artifacts,
        "quantized_changed_targets": changed,
        "quantized_unchanged_targets": sorted(adapted - set(changed)),
        "quantized_unadapted_tensors_bit_identical": len(tensors["base_quantized"])
        - len(adapted),
        "limitations": [
            "Text-only export; vision/GDN/embedding adapters are rejected.",
            "The paired baseline is derived from the exact same native weights/conversion as the merged model; the public B2 GGUF is a separate comparator.",
            "Original native weights are rounded to F16 before adding F32 LoRA deltas. NF4/BF16 training or reload is a separate numerical condition; tensor verification does not prove generation parity.",
            "Small trained deltas can disappear under F16 or Q4_K_M rounding; unchanged adapted targets are reported, not relabeled as model improvement.",
            "Frozen input-prefix parity, native/F16/quantized generation, EOS/schema checks, local serving and quality evaluation remain unexecuted.",
        ],
    }
    write_new(output / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "convert", "verify", "quantize", "receipt")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    function = {
        "preflight": preflight,
        "convert": convert,
        "verify": verify,
        "quantize": quantize,
        "receipt": receipt,
    }[args.command]
    report = function(args.config.resolve(), args.output.resolve())
    print(
        json.dumps({"status": report["status"], "output": str(args.output.resolve())})
    )


if __name__ == "__main__":
    main()
