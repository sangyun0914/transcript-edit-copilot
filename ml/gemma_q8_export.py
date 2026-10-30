"""One opt-in, lazy direct conversion of the pinned native Gemma base to Q8_0."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/final-gemma26-q8-product-comparison-2026-10-25.md"
VERSION = "gemma26-final-direct-q8-export-1"
REGIME = "native_derived_q8_0_base_fp32_lora"
CONFIG = ROOT / "artifacts/training/passive-t1-gemma26-sft-v1/export-attempt-01/configuration.json"
CONFIG_SHA = "07ca5586a1cb354099362b8fc4568d4a1b0451a527b04b624df9c3dc5511ed3d"
REVISION = "427291b5b34cd914a31b3fd3b61a68f6184f4b9f"
REPOSITORY = "google/gemma-4-26B-A4B-it"
NATIVE_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
RECIPE = {"outtype": "q8_0", "lazy": True, "use_temp_file": False, "merge_adapter": False,
          "max_attempts": 1, "timeout_seconds": 3600}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def bound(reference):
    path = Path(reference["path"])
    require(path.is_file() and sha(path) == reference["sha256"], "Bound Q8 input changed.")
    return path


def child(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root), "Q8 inventory path escapes its root.")
    return path


def verify_tree(root, files):
    require(isinstance(files, dict) and files, "Empty pinned Q8 input inventory.")
    for name, expected in files.items():
        path = child(root, name)
        require(path.is_file() and sha(path) == expected, "Pinned Q8 input changed: " + name)


def gguf_inventory(path, converter):
    # GGUFReader maps the file; only headers are accessed here, never tensor values.
    package = (Path(converter) / "gguf-py").resolve()
    sys.path.insert(0, str(package))
    import gguf
    require(Path(gguf.__file__).resolve().is_relative_to(package), "Wrong GGUF metadata reader was imported.")
    reader = gguf.GGUFReader(str(path))
    metadata = {name: field.contents() for name, field in reader.fields.items()
                if name.startswith(("gemma4.", "tokenizer.")) or name == "general.architecture"}
    rows = [{"name": t.name, "shape": [int(n) for n in t.shape], "type": t.tensor_type.name,
             "bytes": int(t.n_bytes)} for t in reader.tensors]
    require(len(rows) == len({r["name"] for r in rows}), "Duplicate GGUF tensor names.")
    return {"architecture": reader.get_field("general.architecture").contents(),
            "file_type": reader.get_field("general.file_type").contents(),
            "semantic_metadata_sha256": digest(metadata), "metadata_keys": sorted(metadata),
            "tensor_count": len(rows), "tensor_types": dict(sorted(Counter(r["type"] for r in rows).items())),
            "tensors": rows}


def validate_inventory(actual, expected):
    require(actual["architecture"] == expected["architecture"] == "gemma4"
            and actual["file_type"] == 7 and actual["tensor_count"] == expected["tensor_count"] == 658,
            "Direct Q8 must be the complete Gemma4 text base with MOSTLY_Q8_0 metadata.")
    require(actual["semantic_metadata_sha256"] == expected["semantic_metadata_sha256"]
            and actual["metadata_keys"] == expected["metadata_keys"], "Q8 architecture/tokenizer/template/terminal metadata changed.")
    names = {r["name"]: r["shape"] for r in actual["tensors"]}
    require(len(names) == len(actual["tensors"]) == 658
            and names == {r["name"]: r["shape"] for r in expected["tensors"]},
            "Q8 tensor names or shapes differ from the pinned native-derived inventory.")
    census = dict(sorted(Counter(r["type"] for r in actual["tensors"]).items()))
    require(census == actual["tensor_types"] and set(census) <= {"Q8_0", "F32"}
            and census.get("Q8_0", 0) > 0 and census.get("F32", 0) > 0
            and all(type(r["bytes"]) is int and r["bytes"] > 0 for r in actual["tensors"]),
            "Q8 export contains a different quantization, fallback tensor type or invalid byte census.")


def register_base(output, config=CONFIG):
    config, output = Path(config).resolve(), Path(output).resolve()
    require(not output.exists() and sha(config) == CONFIG_SHA, "Use a fresh registration and the original pinned converter configuration.")
    value = read(config)
    require(value["base"]["repository"] == REPOSITORY and value["base"]["revision"] == NATIVE_REVISION
            and value["runtime"]["revision"] == REVISION, "Native/converter revision changed.")
    verify_tree(value["runtime"]["source_path"], value["runtime"]["source_files_sha256"])
    old_receipt = config.parent / "receipt.json"
    old = read(old_receipt)
    base = old["artifacts"]["base_quantized"]
    require(sha(base["path"]) == base["sha256"] and Path(base["path"]).stat().st_size == base["bytes"],
            "Pinned native-derived GGUF inventory reference changed.")
    expected = gguf_inventory(base["path"], value["runtime"]["source_path"])
    result = {"version": VERSION, "status": "registered_before_conversion", "created_at": utc(),
        "plan": binding(PLAN), "implementation": binding(__file__), "configuration": binding(config),
        "native_pin": value["native_pin"], "native": value["base"],
        "converter": {k: value["runtime"][k] for k in ("revision", "source_path", "source_files_sha256")},
        "inventory_reference": {"receipt": binding(old_receipt), "gguf": base, "inventory": expected},
        "recipe": RECIPE, "adapter_independent_base_only": True, "model_execution": False,
        "final_test_bodies_read": False, "body_access_authorized": False}
    bound(result["native_pin"])
    write(output, result)
    return result


def validate_registration(value):
    require(value["version"] == VERSION and value["status"] == "registered_before_conversion"
            and value["configuration"]["sha256"] == CONFIG_SHA
            and value["plan"]["sha256"] == sha(PLAN) and value["implementation"]["sha256"] == sha(__file__)
            and value["native"]["repository"] == REPOSITORY and value["native"]["revision"] == NATIVE_REVISION
            and value["converter"]["revision"] == REVISION and value["recipe"] == RECIPE
            and value["adapter_independent_base_only"] is True
            and all(value[k] is False for k in ("model_execution", "final_test_bodies_read", "body_access_authorized")),
            "Q8 conversion registration, scope, recipe or implementation changed.")


def utc():
    return datetime.now(timezone.utc).isoformat()


def convert(registration, registration_sha256, output, *, native_root=None, converter_root=None):
    registration, output = Path(registration).resolve(), Path(output).resolve()
    require(sha(registration) == registration_sha256 and not output.exists(), "Use the exact registration and one fresh conversion directory.")
    value = read(registration)
    validate_registration(value)
    native = Path(native_root or value["native"]["path"]).resolve()
    converter = Path(converter_root or value["converter"]["source_path"]).resolve()
    require(not output.is_relative_to(native) and not output.is_relative_to(converter), "Output overlaps pinned inputs.")
    verify_tree(converter, value["converter"]["source_files_sha256"])
    verify_tree(native, value["native"]["files_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(output.parent).free >= 32_000_000_000, "At least32GB free is required for the single Q8 output.")
    output.mkdir()
    write(registration.with_name(registration.stem + ".attempt.json"),
          {"registration_sha256": registration_sha256, "output": str(output), "created_at": utc(),
           "max_attempts": 1})
    shutil.copyfile(registration, output / "registration.json")
    partial = output / "base-Q8_0.partial.gguf"
    command = [sys.executable, str(converter / "convert_hf_to_gguf.py"), str(native),
               "--outfile", str(partial), "--outtype", "q8_0"]
    info = {"version": VERSION, "status": "starting", "started_at": utc(), "argv": command,
        "host": platform.platform(), "python": sys.version, "native_path": str(native),
        "converter_path": str(converter), "registration_sha256": registration_sha256,
        "environment": {name: importlib.metadata.version(name) for name in
                        ("torch", "numpy", "transformers", "safetensors", "sentencepiece")},
        "disk_free_before": shutil.disk_usage(output).free, "pid": None, "exit_code": None,
        "recipe": RECIPE, "model_execution": False}
    started, process = time.monotonic(), None
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("NO_LOCAL_GGUF", None)
    try:
        with (output / "conversion.log").open("x") as log:
            process = subprocess.Popen(command, cwd=converter, stdout=log, stderr=subprocess.STDOUT, env=env)
            info["pid"] = process.pid
            write(output / "launch.json", info)
            info["exit_code"] = process.wait(timeout=RECIPE["timeout_seconds"])
        require(info["exit_code"] == 0 and partial.is_file(), "Pinned direct Q8 converter did not complete successfully.")
        inventory = gguf_inventory(partial, converter)
        validate_inventory(inventory, value["inventory_reference"]["inventory"])
        write(output / "gguf-inventory.json", inventory)
        final = output / "base-Q8_0.gguf"
        partial.rename(final)
        info.update(status="completed", ended_at=utc(), seconds=time.monotonic() - started,
                    disk_free_after=shutil.disk_usage(output).free)
        write(output / "conversion.json", info)
        receipt = {"version": VERSION, "status": "conversion_verified_serving_pending", "numerical_regime": REGIME,
            "base_repository": REPOSITORY, "base_revision": NATIVE_REVISION, "runtime_revision": REVISION,
            "native_pin_sha256": value["native_pin"]["sha256"], "registration_sha256": registration_sha256,
            "artifacts": {"base_quantized": {"path": final.name, "bytes": final.stat().st_size, "sha256": sha(final)}},
            "files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "tensor_count": inventory["tensor_count"], "tensor_types": inventory["tensor_types"],
            "recipe": RECIPE, "model_execution": False, "quality_claimed": False, "final_test_bodies_read": False}
        write(output / "receipt.json", receipt)
        return receipt
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        info.update(status="failed", ended_at=utc(), seconds=time.monotonic() - started,
                    exit_code=None if process is None else process.returncode,
                    error_type=type(error).__name__, error=str(error))
        write(output / "failure.json", info)
        raise


def verify_export(receipt_path):
    receipt_path = Path(receipt_path).resolve()
    root, receipt = receipt_path.parent, read(receipt_path)
    require(receipt["version"] == VERSION and receipt["status"] == "conversion_verified_serving_pending"
            and receipt["numerical_regime"] == REGIME and receipt["recipe"] == RECIPE
            and receipt["base_repository"] == REPOSITORY and receipt["base_revision"] == NATIVE_REVISION
            and receipt["runtime_revision"] == REVISION and receipt["tensor_count"] == 658
            and all(receipt[k] is False for k in ("model_execution", "quality_claimed", "final_test_bodies_read")),
            "Q8 receipt is incomplete, mislabeled or uses another recipe/base.")
    expected = {p.name for p in root.iterdir() if p.is_file() and p != receipt_path}
    require(set(receipt["files_sha256"]) == expected and "failure.json" not in expected,
            "Q8 export has failed, missing or unexpected files.")
    verify_tree(root, receipt["files_sha256"])
    value = read(root / "registration.json")
    validate_registration(value)
    require(sha(root / "registration.json") == receipt["registration_sha256"]
            and receipt["native_pin_sha256"] == value["native_pin"]["sha256"], "Q8 export lost its prospective native registration.")
    inventory, execution = read(root / "gguf-inventory.json"), read(root / "conversion.json")
    validate_inventory(inventory, value["inventory_reference"]["inventory"])
    require(execution["status"] == "completed" and execution["exit_code"] == 0
            and execution["registration_sha256"] == receipt["registration_sha256"]
            and receipt["tensor_types"] == inventory["tensor_types"], "Q8 execution or census changed.")
    artifact = receipt["artifacts"]["base_quantized"]
    path = child(root, artifact["path"])
    require(path.name == "base-Q8_0.gguf" and path.stat().st_size == artifact["bytes"]
            and receipt["files_sha256"].get(path.name) == artifact["sha256"], "Q8 final artifact differs from its receipt.")
    return receipt, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register-base")
    reg.add_argument("--output", type=Path, required=True)
    reg.add_argument("--config", type=Path, default=CONFIG)
    run = sub.add_parser("convert")
    for name in ("registration", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--registration-sha256", required=True)
    for name in ("native-root", "converter-root"):
        run.add_argument("--" + name, type=Path)
    check = sub.add_parser("verify-export")
    check.add_argument("--receipt-path", type=Path, required=True)
    args = vars(parser.parse_args())
    name = args.pop("command").replace("-", "_")
    result = globals()[name](**args)
    print(json.dumps(result[0] if name == "verify_export" else result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
