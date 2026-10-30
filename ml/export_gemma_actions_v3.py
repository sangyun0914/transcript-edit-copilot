"""Export and exactly verify only the Gemma LoRA GGUF, without full model copies."""

import argparse
from pathlib import Path
import time

import export_passive_gemma as exporter
import train_gemma_actions_v3 as training

VERSION = "gemma-actions-v3-lora-export"


def verify_adapter_tensors(adapter_path, tensor_path, pairs, alpha, gguf):
    import numpy as np
    from safetensors.torch import load_file

    reader = gguf.GGUFReader(str(tensor_path))
    require = exporter.require
    require(
        reader.get_field("general.architecture").contents() == "gemma4",
        "Converted adapter architecture differs.",
    )
    require(
        reader.get_field("adapter.lora.alpha").contents() == alpha,
        "Converted adapter scaling differs.",
    )
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected = {
        f"{pair['gguf_name']}.lora_{side.lower()}"
        for pair in pairs.values()
        for side in ("A", "B")
    }
    original = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    require(
        set(tensors) == expected and len(original) == len(tensors) == 2 * len(pairs),
        "Converted adapter target inventory differs.",
    )
    verified = []
    for pair in pairs.values():
        for side, tensor_name in pair["tensors"].items():
            converted = tensors[f"{pair['gguf_name']}.lora_{side.lower()}"]
            value = original[tensor_name].float().numpy()
            require(
                converted.tensor_type == gguf.GGMLQuantizationType.F32
                and np.isfinite(value).all()
                and np.array_equal(value, converted.data),
                f"Converted FP32 adapter differs: {tensor_name}",
            )
            verified.append(tensor_name)
    return {
        "adapter_tensor_count": len(verified),
        "native_text_targets": len(pairs),
        "all_tensors_exact": True,
        "adapter_sha256": exporter.sha256(adapter_path / "adapter_model.safetensors"),
        "gguf_sha256": exporter.sha256(tensor_path),
        "gguf_bytes": tensor_path.stat().st_size,
    }


def run(config_path, reloaded, output):
    config = exporter.configuration(config_path)
    train_root = Path(config["training_record"]["path"]).parent
    version = training.read(train_root / "manifest.json")["version"]
    exporter.require(
        version in (training.VERSION, "gemma-actions-v3-dpo"),
        "Require a new v3 training run.",
    )
    training.verify_reload(train_root, reloaded, version=version)
    train_config = training.read(train_root / "configuration.json")
    output = Path(output).resolve()
    started = time.monotonic()
    preflight = exporter.preflight(config_path, output)
    try:
        bound_config, _ = exporter.bound_run(config_path, output)
        name, command = next(
            (name, command)
            for name, command in exporter.conversion_commands(bound_config, output)
            if name == "convert-adapter"
        )
        exporter.run_command(output, name, command)
        result = verify_adapter_tensors(
            Path(config["adapter"]["path"]),
            exporter.paths(output)["adapter"],
            preflight["pairs"],
            preflight["alpha"],
            exporter.gguf_module(config),
        )
        receipt = {
            "version": VERSION,
            "status": "adapter_conversion_verified_serving_pending",
            **result,
            "configuration_sha256": exporter.sha256(output / "configuration.json"),
            "native_pin_sha256": preflight["native_pin_sha256"],
            "training_manifest_sha256": exporter.sha256(train_root / "manifest.json"),
            "reload_manifest_sha256": exporter.sha256(Path(reloaded) / "manifest.json"),
            "base_repository": exporter.BASE_REPOSITORY,
            "base_revision": exporter.BASE_REVISION,
            "runtime_revision": exporter.RUNTIME_REVISION,
            "contract_version": train_config["contract_version"],
            "format_version": train_config["format_version"],
            "output_format": train_config["format"],
            "training_phase": train_config["phase"],
            "implementation_sha256": exporter.sha256(Path(__file__)),
            "seconds": time.monotonic() - started,
            "full_base_or_merged_copy_created": False,
            "serving_requirement": "Use the verified native-derived base GGUF with this adapter at scale 1.0; validate base+LoRA inference separately.",
            "scope": "Exact conversion of every FP32 LoRA tensor. This does not certify merged quantization, base+LoRA predictions, quality or activation.",
        }
        exporter.write_new(output / "adapter-receipt.json", receipt)
        return receipt
    except BaseException as error:
        exporter.write_new(
            output / "adapter-export-failure.json",
            {
                "version": VERSION,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reloaded", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    import json

    print(
        json.dumps(run(args.config, args.reloaded, args.output), indent=2), flush=True
    )


if __name__ == "__main__":
    main()
