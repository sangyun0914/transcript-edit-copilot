"""Export the fixed Gemma4 26B text LoRA into a verified native-derived GGUF pair.

Architecture-neutral file/converter helpers come from the frozen Qwen exporter.
Gemma target selection, native pin and tensor checks live in this sibling only.
No acquisition, training, inference, dataset access or model promotion occurs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re

import export_passive_sft as shared
import export_qwen06
from export_qwen06 import check_hash, sha256, write_new
from export_passive_sft import (
    RUNTIME_REVISION,
    require,
    resolve,
    check_files,
    tensor_header,
    native_inventory,
    mapped_native_name,
    training_census,
    paths,
    refuse_existing,
    run_command,
    artifact,
    conversion_commands,
    gguf_module,
    matching_metadata,
    quantized_changes,
)

VERSION = "passive-gemma-sft-export-v1"
BASE_REPOSITORY = "google/gemma-4-26B-A4B-it"
BASE_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
TEXT_TARGET = re.compile(
    r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight$"
)
GENERATION_STOP_IDS = [1, 106, 50]


def target_names(native_config):
    text = native_config["text_config"]
    require(
        native_config.get("model_type") == "gemma4"
        and native_config.get("architectures") == ["Gemma4ForConditionalGeneration"]
        and not native_config.get("quantization_config")
        and text["num_hidden_layers"] == 30
        and text["attention_k_eq_v"] is True,
        "Use the pinned unquantized Gemma4 26B architecture.",
    )
    result = set()
    for index, kind in enumerate(text["layer_types"]):
        projections = ["q_proj", "k_proj", "o_proj"]
        if kind == "sliding_attention":
            projections.append("v_proj")
        else:
            require(kind == "full_attention", "Unexpected Gemma attention layer type.")
        prefix = f"model.language_model.layers.{index}"
        result.update(f"{prefix}.self_attn.{name}.weight" for name in projections)
        result.update(
            f"{prefix}.mlp.{name}_proj.weight" for name in ("gate", "up", "down")
        )
    require(len(result) == 205, "Gemma requires exactly 205 ordinary text targets.")
    return result


def validate_native_pin(config):
    binding = config["native_pin"]
    path = Path(binding["path"])
    check_hash(path, binding["sha256"])
    pin = json.loads(path.read_text())
    require(
        pin["version"] == "passive-native-pin-v1"
        and pin["candidate_id"] == "gemma26"
        and pin["repository"] == BASE_REPOSITORY
        and pin["revision"] == BASE_REVISION
        and {name: value["sha256"] for name, value in pin["files"].items()}
        == config["base"]["files_sha256"],
        "Export native files differ from the exact Gemma acquisition pin.",
    )
    check_hash(path.parent / "native-api.json", pin["api_sha256"])
    generation = json.loads(
        (Path(config["base"]["path"]) / "generation_config.json").read_text()
    )
    require(
        generation["eos_token_id"] == GENERATION_STOP_IDS
        and generation["bos_token_id"] == 2
        and generation["pad_token_id"] == 0,
        "Keep native EOS, turn and tool-handoff terminal IDs.",
    )
    return pin


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
        config.get("revision") == BASE_REVISION,
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

    require(
        not ignored_native_names,
        "Gemma has no native MTP exclusions.",
    )
    # PEFT compacts saved targets to suffixes that also occur in the vision tower.
    # Only the independently recorded text module census may select adapter weights.
    intended = {
        name
        for name in inventory
        if TEXT_TARGET.fullmatch(name) and selected(name[:-7])
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


def configuration(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    require(config.get("version") == VERSION, "Unknown export configuration.")
    require(
        config["base"]["repository"] == BASE_REPOSITORY
        and config["base"]["revision"] == BASE_REVISION,
        "This bounded export supports only the selected native Gemma4-26B-A4B revision.",
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
    config["native_pin"]["path"] = str(
        resolve(config["native_pin"]["path"], path.parent)
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
        "README.md",
        "generation_config.json",
    }
    require(
        required <= set(config["base"]["files_sha256"]),
        "Bind native config, index, tokenizer, template, generation defaults and model card.",
    )
    validate_native_pin(config)
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
            native_config.get("model_type") == "gemma4"
            and not native_config.get("quantization_config"),
            "Expected original, unquantized Gemma4 native weights.",
        )
        layers = native_config["text_config"]["num_hidden_layers"]
        require(
            type(layers) is int and layers > 0, "Native text layer count is invalid."
        )
        inventory = native_inventory(base, config["base"]["files_sha256"])
        expected_targets = target_names(native_config)
        require(
            {name for name in inventory if TEXT_TARGET.fullmatch(name)}
            == expected_targets,
            "Native tensor headers must contain the exact 205 text targets and five shared K/V layers.",
        )
        census = training_census(config, inventory)
        require(
            {name + ".weight" for name in census["targets"]} == expected_targets,
            "Training census must adapt exactly the registered 205 ordinary Gemma text modules.",
        )
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
            "architecture_neutral_helpers_sha256": sha256(Path(shared.__file__)),
            "native_tensor_count": len(inventory),
            "text_layers": layers,
            "adapter_tensor_count": 2 * len(pairs),
            "pairs": pairs,
            "training_census": census,
            "template_sha256": sha256(base / "chat_template.jinja"),
            "rank": adapter_config["r"],
            "alpha": adapter_config["lora_alpha"],
            "native_generation_stop_ids": GENERATION_STOP_IDS,
            "native_pin_sha256": config["native_pin"]["sha256"],
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
    check_hash(Path(shared.__file__), report["architecture_neutral_helpers_sha256"])
    for name, expected in report["training_census"]["files_sha256"].items():
        check_hash(Path(name), expected)
    validate_inputs(config)
    return config, report


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
            reader.get_field("general.architecture").contents() == "gemma4",
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
        "native_generation_stop_ids": GENERATION_STOP_IDS,
        "scope": "Fresh-process tensor verification only; BF16 training/reload and native-F16 deployment are different numerical conditions.",
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
        "commands": {
            key: {
                suffix: artifact(output / "logs" / f"quantize-{key}.{suffix}")
                for suffix in ("command.json", "result.json", "log")
            }
            for key in ("base", "merged")
        },
        "quantizer_fallbacks": {
            key: [
                line
                for line in (output / "logs" / f"quantize-{key}.log")
                .read_text()
                .splitlines()
                if "not divisible by" in line or "fallback" in line.lower()
            ]
            for key in ("base", "merged")
        },
        "artifacts": {
            key: artifact(target[key]) for key in ("base_quantized", "merged_quantized")
        },
    }
    write_new(output / "quantization.json", result)
    return result


def receipt(config_path, output):
    config, preflight_report = bound_run(config_path, output)
    refuse_existing(output / "receipt.json")
    verification = json.loads((output / "tensor-verification.json").read_text())
    quantization = json.loads((output / "quantization.json").read_text())
    require(verification["status"] == "passed", "Tensor verification must pass.")
    require(
        quantization["quantization"] == "Q4_K_M"
        and quantization["importance_matrix"] is None
        and quantization["threads"] == 4,
        "Keep the same registered quantization recipe for the pair.",
    )
    for files in quantization["commands"].values():
        for value in files.values():
            check_hash(Path(value["path"]), value["sha256"])
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
            reader.get_field("general.architecture").contents() == "gemma4",
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
        "quantization_sha256": sha256(output / "quantization.json"),
        "training_record": config["training_record"],
        "native_pin": config["native_pin"],
        "base_repository": BASE_REPOSITORY,
        "base_revision": BASE_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "artifacts": artifacts,
        "quantized_changed_targets": changed,
        "quantized_unchanged_targets": sorted(adapted - set(changed)),
        "quantized_unadapted_tensors_bit_identical": len(tensors["base_quantized"])
        - len(adapted),
        "native_generation_stop_ids": GENERATION_STOP_IDS,
        "quantizer_fallbacks": quantization["quantizer_fallbacks"],
        "quantized_tensor_type_census": {
            key: dict(Counter(tensor.tensor_type.name for tensor in reader.tensors))
            for key, reader in readers.items()
        },
        "limitations": [
            "Text-only export; vision/router/fused-expert/embedding adapters are rejected.",
            "The paired baseline is derived from the exact same native weights/conversion as the merged model; the public B2 GGUF is a separate comparator.",
            "Original native weights are rounded to F16 before adding F32 LoRA deltas. BF16 training or reload is a separate numerical condition; tensor verification does not prove generation parity.",
            "Q4_K_M is a mixed recipe. Gemma expert/shared-MLP widths can invoke the pinned quantizer's Q5_0/Q8_0 shape fallbacks; actual type census and file bytes are retained for both models. Historical vendor QAT Q4_0 is a separate comparator.",
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
