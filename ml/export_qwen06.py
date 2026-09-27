"""Prepare the archived Qwen06 adapter with its original base; never run inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "artifacts/model-preparation/qwen06-v1"
BASE_REPO = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
LLAMA_REVISION = "427291b5b34cd914a31b3fd3b61a68f6184f4b9f"
SOURCE = ROOT / f"llama.cpp-{LLAMA_REVISION}"
BASE = ROOT / "base-native"
ADAPTER = REPO / "artifacts/iterations/colab-l4-run-06/adapter"
RUNTIME = REPO / "artifacts/nep-probe/llama-b10816"
PYTHON = ROOT / ".venv/bin/python"
BASE_WEIGHT = "model.safetensors-00001-of-00001.safetensors"
BASE_WEIGHT_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
ADAPTER_SHA256 = "c0c170e76d5115f5bb91fd337ba51cd1343e4cacbf1e566f42015c362142bfb7"
PROTOCOL_SHA256 = "992794a48c41c883634bfa09b74f6890a74e667bc36f1c93a9a73a77bcedf79c"
SOURCE_SHA256 = "056ae92c363e5c761e6c03102f4aa6c1551e134f9e322eea1a4cf3f6cf87f585"
TOOLS_SHA256 = {
    "llama-export-lora": "325a1a576e7014c3cbe539df48334eb74913e6801c0a28204e5c971f21681372",
    "llama-quantize": "0ece83905109d9a3345c3e5b81f32082d30c59bb802a0762ce9439767b0fc1d7",
    "libllama-quantize-impl.dylib": "8005805741e80e78f801d48ee5c22a0bbbf0bda0536b3f3927e66e7749ee7cda",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: expected {expected}, got {actual}")
    return actual


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def run(name: str, argv: list[str | Path], *, offline: bool = False) -> None:
    name = name.replace(".", "-")
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    index = 1
    while (logs / f"{name}-{index:03}.command.json").exists():
        index += 1
    stem = logs / f"{name}-{index:03}"
    command = [str(item) for item in argv]
    env = os.environ.copy()
    additions = {
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if offline:
        additions.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    env.update(additions)
    receipt = {
        "argv": command,
        "cwd": str(REPO),
        "environment_overrides": additions,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    implementations = [Path(__file__).resolve()] + [
        Path(x) for x in command if x.endswith(".py")
    ]
    snapshots = stem.with_suffix(".implementation")
    snapshots.mkdir()
    receipt["implementation_sha256"] = {}
    for implementation in set(implementations):
        if implementation.is_file():
            shutil.copyfile(implementation, snapshots / implementation.name)
            receipt["implementation_sha256"][str(implementation)] = sha256(
                implementation
            )
    write_new(stem.with_suffix(".command.json"), receipt)
    print(f"Running {name}; log {stem.with_suffix('.log')}", flush=True)
    with stem.with_suffix(".log").open("x") as stream:
        completed = subprocess.run(
            command, cwd=REPO, env=env, stdout=stream, stderr=stream
        )
    write_new(
        stem.with_suffix(".result.json"),
        {
            "returncode": completed.returncode,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "log_sha256": sha256(stem.with_suffix(".log")),
        },
    )
    if completed.returncode:
        raise RuntimeError(f"{name} failed; retained {stem.with_suffix('.log')}")


def verify_archive() -> dict:
    registry = json.loads((REPO / "ml/passive-models-v1.json").read_text())
    candidate = next(x for x in registry["candidates"] if x["id"] == "qwen06")
    if (
        candidate["base_repository"] != BASE_REPO
        or candidate["base_revision"] != BASE_REVISION
    ):
        raise ValueError("Registry does not match the original Qwen06 base")
    protocol = ADAPTER.parent / "protocol.json"
    check_hash(protocol, PROTOCOL_SHA256)
    saved = json.loads(protocol.read_text())
    if saved["model"] != BASE_REPO or saved["revision"] != BASE_REVISION:
        raise ValueError("Original protocol does not match the base revision")
    verified = {}
    for name, expected in candidate["adapter_files_sha256"].items():
        verified[name] = check_hash(REPO / name, expected)
    check_hash(ADAPTER / "adapter_model.safetensors", ADAPTER_SHA256)
    return {"protocol_sha256": PROTOCOL_SHA256, "adapter_files_sha256": verified}


def validate_adapter_config(config: dict) -> None:
    if config.get("base_model_name_or_path") != BASE_REPO:
        raise ValueError("Unexpected adapter base")
    required = {
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 16,
        "bias": "none",
        "lora_bias": False,
        "fan_in_fan_out": False,
        "use_dora": False,
        "use_rslora": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Unsupported adapter configuration: {key}")
    for key in (
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
    ):
        if config.get(key):
            raise ValueError(f"Unsupported adapter configuration: {key}")


def verify_converter_source() -> dict:
    archive = ROOT / f"acquisition/llama-{LLAMA_REVISION}.tar.gz"
    check_hash(archive, SOURCE_SHA256)
    checked = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            relative = Path(*Path(member.name).parts[1:])
            if not member.isfile() or relative.suffix != ".py":
                continue
            if relative.parts[0] not in (
                "conversion",
                "gguf-py",
            ) and relative.name not in (
                "convert_hf_to_gguf.py",
                "convert_lora_to_gguf.py",
            ):
                continue
            stream = source.extractfile(member)
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
            checked[str(relative)] = check_hash(SOURCE / relative, expected)
    registry = json.loads((REPO / "ml/passive-models-v1.json").read_text())
    for path, expected in registry["runtime"]["files_sha256"].items():
        check_hash(REPO / path, expected)
    for name, expected in TOOLS_SHA256.items():
        check_hash(RUNTIME / name, expected)
    return checked


def acquire() -> None:
    verification = verify_archive()
    validate_adapter_config(json.loads((ADAPTER / "adapter_config.json").read_text()))
    api_path = ROOT / "acquisition/base-api.json"
    api = json.loads(api_path.read_text())
    if api["sha"] != BASE_REVISION or api["id"] != BASE_REPO:
        raise ValueError("Base API response is not for the exact original revision")
    BASE.mkdir(parents=True, exist_ok=True)
    files = {}
    for item in sorted(api["siblings"], key=lambda x: (x["size"], x["rfilename"])):
        name = item["rfilename"]
        if Path(name).name != name:
            raise ValueError(f"Unexpected non-flat base path: {name}")
        target = BASE / name
        if not target.exists():
            partial = target.with_name(target.name + ".partial")
            run(
                "download-" + name.replace(".", "-"),
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--silent",
                    "--show-error",
                    "--continue-at",
                    "-",
                    "--write-out",
                    "%{http_code} %{size_download} %{time_total}\n",
                    f"https://huggingface.co/{BASE_REPO}/resolve/{BASE_REVISION}/{name}",
                    "--output",
                    partial,
                ],
            )
            verify_download(partial, item)
            partial.rename(target)
        verify_download(target, item)
        files[name] = {
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
            "git_blob_id": item["blobId"],
            "lfs": item.get("lfs"),
        }
        print(f"Verified {name}: {files[name]['sha256']}", flush=True)
    check_hash(BASE / BASE_WEIGHT, BASE_WEIGHT_SHA256)
    write_new(
        ROOT / "acquisition/verified-base.json",
        {
            "repository": BASE_REPO,
            "revision": BASE_REVISION,
            "files": files,
            "api_sha256": sha256(api_path),
            "archive_verification": verification,
        },
    )


def verify_download(path: Path, item: dict) -> None:
    size = path.stat().st_size
    if size != item["size"]:
        raise ValueError(f"Size mismatch for {path}")
    if item.get("lfs"):
        check_hash(path, item["lfs"]["sha256"])
    else:
        digest = hashlib.sha1(f"blob {size}\0".encode() + path.read_bytes()).hexdigest()
        if digest != item["blobId"]:
            raise ValueError(f"Git blob mismatch for {path}")


def convert() -> None:
    verify_converter_source()
    verify_archive()
    validate_adapter_config(json.loads((ADAPTER / "adapter_config.json").read_text()))
    check_hash(BASE / BASE_WEIGHT, BASE_WEIGHT_SHA256)
    output = ROOT / "gguf"
    output.mkdir(exist_ok=True)
    base_output = output / "Qwen3.5-2B-original-15852e8-F16.gguf"
    adapter_output = output / "qwen06-adapter-F32.gguf"
    merged_output = output / "qwen06-merged-F16.gguf"
    for path in (base_output, adapter_output, merged_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite conversion artifact: {path}")
    run(
        "convert-base",
        [
            PYTHON,
            SOURCE / "convert_hf_to_gguf.py",
            BASE,
            "--outfile",
            base_output,
            "--outtype",
            "f16",
        ],
        offline=True,
    )
    run(
        "convert-adapter",
        [
            PYTHON,
            SOURCE / "convert_lora_to_gguf.py",
            ADAPTER,
            "--base",
            BASE,
            "--outfile",
            adapter_output,
            "--outtype",
            "f32",
        ],
        offline=True,
    )
    run(
        "merge-gguf",
        [
            RUNTIME / "llama-export-lora",
            "-m",
            base_output,
            "--lora",
            adapter_output,
            "-o",
            merged_output,
            "-t",
            "4",
        ],
        offline=True,
    )


def quantize() -> None:
    verify_converter_source()
    verification = json.loads((ROOT / "tensor-verification.json").read_text())
    if verification["status"] != "passed":
        raise ValueError("Tensor verification must pass before quantization")
    for name in ("Qwen3.5-2B-original-15852e8", "qwen06-merged"):
        original = ROOT / "gguf" / f"{name}-F16.gguf"
        check_hash(original, verification["artifacts"][original.name]["sha256"])
        output = ROOT / "gguf" / f"{name}-Q4_K_M.gguf"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite quantized artifact: {output}")
        run(
            f"quantize-{name}",
            [RUNTIME / "llama-quantize", original, output, "Q4_K_M", "4"],
            offline=True,
        )


def receipt() -> None:
    import sys

    import numpy as np

    sys.path.insert(0, str(SOURCE / "gguf-py"))
    import gguf

    verify_archive()
    verify_converter_source()
    tensor_report = json.loads((ROOT / "tensor-verification.json").read_text())
    if tensor_report["status"] != "passed":
        raise ValueError("Tensor verification did not pass")
    models = {
        "qwen06": ROOT / "gguf/qwen06-merged-Q4_K_M.gguf",
        "qwen06-base-control": ROOT / "gguf/Qwen3.5-2B-original-15852e8-Q4_K_M.gguf",
    }
    readers = {name: gguf.GGUFReader(path) for name, path in models.items()}
    by_name = {
        name: {t.name: t for t in reader.tensors} for name, reader in readers.items()
    }
    base = by_name["qwen06-base-control"]
    merged = by_name["qwen06"]
    if set(base) != set(merged) or len(base) != 335:
        raise ValueError("Quantized tensor sets differ")
    changed = []
    for name, original in base.items():
        target = merged[name]
        if original.tensor_type != target.tensor_type or not np.array_equal(
            original.shape, target.shape
        ):
            raise ValueError(f"Quantization recipe mismatch: {name}")
        if not np.array_equal(original.data, target.data):
            changed.append(name)
    expected = {item["name"] for item in tensor_report["merged_targets"]}
    if set(changed) != expected:
        raise ValueError("Quantized differences do not match all 96 LoRA targets")
    for name, reader in readers.items():
        if reader.get_field("general.architecture").contents() != "qwen35":
            raise ValueError(f"Unexpected quantized architecture: {name}")
        template = reader.get_field("tokenizer.chat_template").contents().encode()
        if hashlib.sha256(template).hexdigest() != tensor_report["template_sha256"]:
            raise ValueError(f"Quantization changed the template: {name}")
        if (
            reader.get_field("tokenizer.ggml.eos_token_id").contents()
            != tensor_report["eos_token_id"]
        ):
            raise ValueError(f"Quantization changed EOS: {name}")
    artifacts = {}
    for path in sorted((ROOT / "gguf").glob("*.gguf")):
        artifacts[path.name] = {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        if path.name in tensor_report["artifacts"]:
            if (
                artifacts[path.name]["sha256"]
                != tensor_report["artifacts"][path.name]["sha256"]
            ):
                raise ValueError(f"Verified F16/adapter artifact changed: {path}")
    candidates = []
    for name, path in models.items():
        item = artifacts[path.name]
        candidates.append(
            {
                "id": name,
                "tokenizer_id": "qwen06",
                "base_repository": BASE_REPO,
                "base_revision": BASE_REVISION,
                "model_path": item["path"],
                "model_sha256": item["sha256"],
                "model_bytes": item["bytes"],
                "status": "ready_local_weights",
                "quantization": "Q4_K_M",
                "quant_repository": None,
                "quant_revision": None,
                "base_weights_status": "verified_original_native_weights",
                "quantization_provenance": (
                    "Local pinned b10816 conversion; original native base to F16, original F32 LoRA merged before Q4_K_M; no importance matrix. See preparation_receipt."
                    if name == "qwen06"
                    else "Local pinned b10816 conversion; original native base to F16 then Q4_K_M; no adapter and no importance matrix. See preparation_receipt."
                ),
                "preparation_receipt": str((ROOT / "receipt.json").relative_to(REPO)),
                "gguf_template_sha256": tensor_report["template_sha256"],
                "gguf_template_matches_pinned_native": True,
                "runtime_status": "passive_template_tokenizer_and_generation_smoke_not_run",
                "paired_base_required": name == "qwen06",
                "paired_base_id": "qwen06-base-control" if name == "qwen06" else None,
                "license": {
                    "identifier": "apache-2.0",
                    "source": f"https://huggingface.co/{BASE_REPO}/blob/{BASE_REVISION}/LICENSE",
                    "local_license_file": str((BASE / "LICENSE").relative_to(REPO)),
                    "sha256": sha256(BASE / "LICENSE"),
                    "status": "pinned_local_snapshot",
                },
            }
        )
    write_new(
        ROOT / "receipt.json",
        {
            "version": "qwen06-local-export-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "tensor_verified_artifacts_ready_runtime_preflight_pending",
            "base": json.loads((ROOT / "acquisition/verified-base.json").read_text()),
            "adapter": {
                "path": str(ADAPTER.relative_to(REPO)),
                "sha256": ADAPTER_SHA256,
                "protocol_sha256": PROTOCOL_SHA256,
                "historical_status": "failed diagnostic; provisional control, not promoted",
            },
            "converter": {
                "repository": "ggml-org/llama.cpp",
                "revision": LLAMA_REVISION,
                "tag": "b10816",
                "source_archive_sha256": SOURCE_SHA256,
                "python_source_receipt": "converter-source-verification.json",
                "tools_sha256": TOOLS_SHA256,
                "dependencies_lock": "requirements.lock",
                "dependencies_lock_sha256": sha256(ROOT / "requirements.lock"),
            },
            "artifacts": artifacts,
            "registry_candidates": candidates,
            "verification": {
                "tensor_receipt": "tensor-verification.json",
                "sha256": sha256(ROOT / "tensor-verification.json"),
                "quantized_tensor_count": 335,
                "quantized_changed_tensors": changed,
                "quantized_unadapted_tensors_bit_identical": 239,
                "scope": "Fresh tensor reload; no inference, no exposed dev or final-test inputs",
            },
            "commands": sorted(
                str(p.relative_to(ROOT)) for p in (ROOT / "logs").glob("*.command.json")
            ),
            "limitations": tensor_report["limitations"]
            + [
                "Complete frozen token-ID parity, stopping/schema checks and B1 screen remain unrun for both artifacts.",
                "Q4_K_M quantization uses no calibration/importance matrix; historical training used unquantized BF16, so quantized runtime tokens are not expected to be byte-identical to historical CUDA reload tokens.",
            ],
        },
    )
    print(
        json.dumps(
            {"status": "ready_for_runtime_preflight", "candidates": candidates},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["acquire", "convert", "quantize", "receipt"]
    )
    args = parser.parse_args()
    if args.command == "acquire":
        acquire()
    elif args.command == "convert":
        convert()
    elif args.command == "quantize":
        quantize()
    else:
        receipt()


if __name__ == "__main__":
    main()
