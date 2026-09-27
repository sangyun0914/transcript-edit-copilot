"""Fresh-process tensor audit of the Qwen06 GGUF export, without model inference."""

from __future__ import annotations

import hashlib
import json
import re
import sys

from export_qwen06 import (
    ADAPTER,
    BASE,
    BASE_WEIGHT,
    BASE_WEIGHT_SHA256,
    ROOT,
    SOURCE,
    check_hash,
    sha256,
    validate_adapter_config,
    verify_archive,
    write_new,
)


def mapped_adapter_name(name: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"base_model\.model\.model\.language_model\.layers\.(\d+)\."
        r"(mlp|self_attn)\.([a-z_]+)\.lora_([AB])\.weight",
        name,
    )
    if not match:
        raise ValueError(f"Unexpected adapter tensor: {name}")
    layer, family, projection, side = match.groups()
    names = {
        ("mlp", "gate_proj"): "ffn_gate",
        ("mlp", "up_proj"): "ffn_up",
        ("mlp", "down_proj"): "ffn_down",
        ("self_attn", "q_proj"): "attn_q",
        ("self_attn", "k_proj"): "attn_k",
        ("self_attn", "v_proj"): "attn_v",
        ("self_attn", "o_proj"): "attn_output",
    }
    key = (family, projection)
    if key not in names or not 0 <= int(layer) < 24:
        raise ValueError(f"Unsupported adapter tensor: {name}")
    base = f"blk.{layer}.{names[key]}.weight"
    return base, f"{base}.lora_{side.lower()}"


def verify() -> dict:
    import numpy as np
    import torch
    import peft
    from peft.tuners.lora.layer import Linear
    from safetensors import safe_open
    from safetensors.torch import load_file

    sys.path.insert(0, str(SOURCE / "gguf-py"))
    import gguf

    torch.set_num_threads(4)
    archive = verify_archive()
    config = json.loads((ADAPTER / "adapter_config.json").read_text())
    validate_adapter_config(config)
    peft_config = peft.LoraConfig.from_pretrained(ADAPTER)
    check_hash(BASE / BASE_WEIGHT, BASE_WEIGHT_SHA256)
    original = load_file(ADAPTER / "adapter_model.safetensors", device="cpu")
    if len(original) != 192:
        raise ValueError("Expected all 192 archived LoRA tensors")
    paths = {
        "base": ROOT / "gguf/Qwen3.5-2B-original-15852e8-F16.gguf",
        "adapter": ROOT / "gguf/qwen06-adapter-F32.gguf",
        "merged": ROOT / "gguf/qwen06-merged-F16.gguf",
    }
    readers = {name: gguf.GGUFReader(path) for name, path in paths.items()}
    tensors = {
        name: {t.name: t for t in reader.tensors} for name, reader in readers.items()
    }
    for reader in readers.values():
        if reader.get_field("general.architecture").contents() != "qwen35":
            raise ValueError("Expected QWEN35 architecture")
    if readers["adapter"].get_field("adapter.lora.alpha").contents() != 16:
        raise ValueError("Adapter scaling changed")
    if len(tensors["adapter"]) != 192:
        raise ValueError("Converted adapter tensor count changed")
    if set(tensors["base"]) != set(tensors["merged"]):
        raise ValueError("Merged GGUF does not have the exact base tensor set")
    metadata_equal = all(
        readers["merged"].get_field(name) is not None
        and field.contents() == readers["merged"].get_field(name).contents()
        for name, field in readers["base"].fields.items()
        if not name.startswith("GGUF.")
    )
    if not metadata_equal:
        raise ValueError("Base and merged model metadata differ")
    template = readers["base"].get_field("tokenizer.chat_template").contents()
    if template.encode() != (BASE / "chat_template.jinja").read_bytes():
        raise ValueError("GGUF template does not match original native template")
    pairs = {}
    for name, tensor in original.items():
        base_name, converted_name = mapped_adapter_name(name)
        converted = tensors["adapter"][converted_name]
        if converted.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f"Adapter precision changed: {name}")
        if not np.array_equal(tensor.numpy(), converted.data):
            raise ValueError(f"Adapter values changed: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite adapter tensor: {name}")
        pairs.setdefault(base_name, {})[name.rsplit(".lora_", 1)[1][0]] = name
    if len(pairs) != 96 or any(set(p) != {"A", "B"} for p in pairs.values()):
        raise ValueError("Incomplete LoRA pair coverage")
    reports = []
    with (
        torch.no_grad(),
        safe_open(BASE / BASE_WEIGHT, framework="pt", device="cpu") as native,
    ):
        for base_name, names in sorted(pairs.items()):
            native_name = (
                names["A"]
                .removeprefix("base_model.model.")
                .replace(".lora_A.weight", ".weight")
            )
            base = tensors["base"][base_name].data
            merged = tensors["merged"][base_name].data
            if not np.array_equal(native.get_tensor(native_name).half().numpy(), base):
                raise ValueError(
                    f"Base target tensor does not match original native weights: {base_name}"
                )
            a, b = original[names["A"]], original[names["B"]]
            if a.shape[0] != 16 or b.shape[1] != 16:
                raise ValueError(f"Rank mismatch: {base_name}")
            linear = torch.nn.Linear(
                base.shape[1], base.shape[0], bias=False, device="meta"
            )
            linear.weight = torch.nn.Parameter(
                torch.from_numpy(base.astype(np.float32))
            )
            layer = Linear(
                linear, adapter_name="default", config=peft_config, r=16, lora_alpha=16
            )
            layer.lora_A["default"].weight.copy_(a)
            layer.lora_B["default"].weight.copy_(b)
            layer.merge(safe_merge=True)
            expected = layer.get_base_layer().weight.half().numpy()
            absolute_error = np.abs(
                merged.astype(np.float32) - expected.astype(np.float32)
            )
            spacing = np.spacing(np.abs(expected)).astype(np.float32)
            if not np.isfinite(merged).all() or not np.all(absolute_error <= spacing):
                raise ValueError(
                    f"Merged tensor differs from PEFT by more than one F16 ULP: {base_name}"
                )
            changed = int(np.count_nonzero(merged != base))
            if changed == 0:
                raise ValueError(f"Adapter had no effect: {base_name}")
            reports.append(
                {
                    "name": base_name,
                    "native_name": native_name,
                    "shape": list(base.shape),
                    "elements": int(base.size),
                    "changed_elements": changed,
                    "exact_peft_elements": int(np.count_nonzero(merged == expected)),
                    "max_abs_peft_f16_error": float(absolute_error.max()),
                    "max_peft_f16_ulps": float((absolute_error / spacing).max()),
                }
            )
            del linear, layer, expected, absolute_error, spacing
    unchanged = []
    for name, tensor in tensors["base"].items():
        if name in pairs:
            continue
        if not np.array_equal(tensor.data, tensors["merged"][name].data):
            raise ValueError(f"Unadapted tensor changed: {name}")
        if not np.isfinite(tensor.data).all():
            raise ValueError(f"Nonfinite base tensor: {name}")
        unchanged.append(name)
    return {
        "status": "passed",
        "scope": "Fresh process, all tensor values; no predictor or dev/test requests",
        "oracle": f"PEFT {peft.__version__} Linear.merge(safe_merge=True), float32 accumulation then F16",
        "tolerance": "At most one F16 ULP per element for CPU matrix accumulation order; all F32 adapter values exact",
        "torch_version": torch.__version__,
        "archive_verification": archive,
        "native_base_sha256": BASE_WEIGHT_SHA256,
        "adapter_tensors_exact": len(original),
        "base_target_tensors_exact": len(pairs),
        "merged_tensor_count": len(tensors["merged"]),
        "merged_targets": reports,
        "unadapted_tensors_exact": unchanged,
        "metadata_equal": metadata_equal,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "eos_token_id": readers["base"]
        .get_field("tokenizer.ggml.eos_token_id")
        .contents(),
        "artifacts": {
            path.name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths.values()
        },
        "limitations": [
            "Text-only GGUF: native vision tensors are excluded by the pinned converter.",
            "This verifies weight export, not Transformers/llama.cpp generation parity or quality.",
            "Archived continue_adapter.py loaded the original base in BF16 (load_in_4bit=False, load_in_16bit=True); export applies the adapter to original native weights converted to F16 before Q4_K_M.",
        ],
    }


if __name__ == "__main__":
    report = verify()
    write_new(ROOT / "tensor-verification.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "adapter_tensors_exact",
                    "base_target_tensors_exact",
                    "merged_tensor_count",
                    "metadata_equal",
                    "template_sha256",
                    "eos_token_id",
                )
            },
            indent=2,
        )
    )
