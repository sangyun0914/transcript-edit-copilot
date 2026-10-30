"""Assemble a final-only runner beside each model's immutable training sources."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
VERSION = "task-test-actions-v3-runtime-1"
MANIFEST = "runtime-manifest.json"
ENTRY = "ml/infer_task_test_actions_v3.py"
SHARED = ("passive_actions.py", "passive_formats_actions_v3.py", "prepare_actions_v3_conditions.py",
          "prompt-passive-actions-v1.txt")


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def child(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Runtime source path escaped its package.")
    return path


def source_plan(config_path):
    config_path = Path(config_path).resolve()
    config, preparation = read(config_path), read(config_path.parent / "manifest.json")
    require(preparation["train_only"] is True and preparation["final_test_bodies_read"] is False,
            "Use an immutable train-only model preparation.")
    require(preparation["files_sha256"].get(config_path.name) == file_sha(config_path),
            "Configuration is not bound by the preparation manifest.")
    sources = {}
    for name, expected in config["sources_sha256"].items():
        require(Path(name).name == name and name not in {".", ".."}, "Invalid frozen source filename.")
        path = config_path.parent / "source" / name
        require(file_sha(path) == expected and preparation["files_sha256"].get("source/" + name) == expected,
                "Frozen preparation source changed.")
        sources["ml/" + name] = {"path": path, "sha256": expected, "origin": "frozen_preparation"}
    require(all("ml/" + name in sources for name in SHARED), "Missing shared predictor contract source.")
    for name in SHARED:
        require(sources["ml/" + name]["sha256"] == file_sha(ROOT / "ml" / name),
                "Frozen model and final-test controller disagree on the predictor contract.")
    pending = [ENTRY, "ml/assemble_task_test_actions_v3.py", *sources]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if name not in sources:
            path = child(ROOT, name)
            require(path.is_file(), "Missing final-only runtime dependency.")
            sources[name] = {"path": path, "sha256": file_sha(path), "origin": "final_only_overlay"}
        path = sources[name]["path"]
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                for directory in ("ml", "server"):
                    target = str(Path(directory) / Path(*module.split(".")).with_suffix(".py"))
                    if target in sources or (ROOT / target).is_file():
                        pending.append(target)
    return config, dict(sorted(sources.items()))


def assemble(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config, plan = source_plan(config_path)
    require(not output.exists() and not output.is_relative_to(config_path.parent),
            "Use a new runtime directory outside the frozen model preparation.")
    output.mkdir(parents=True)
    for name, value in plan.items():
        destination = child(output, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(value["path"], destination)
    manifest = {
        "version": VERSION, "status": "assembled_model_specific_runtime", "scope": "final_test",
        "created_at": datetime.now(timezone.utc).isoformat(), "entrypoint": ENTRY,
        "configuration_sha256": file_sha(config_path),
        "preparation_manifest_sha256": file_sha(config_path.parent / "manifest.json"),
        "configuration_sources_sha256": config["sources_sha256"],
        "source_sha256": {name: value["sha256"] for name, value in plan.items()},
        "source_origin": {name: value["origin"] for name, value in plan.items()},
        "final_test_bodies_read": False, "model_weights_read": False,
    }
    write(output / MANIFEST, manifest)
    return load_runtime(output)


def load_runtime(directory, expected_sha256=None):
    root = Path(directory).resolve()
    manifest_path = root / MANIFEST
    if expected_sha256 is not None:
        require(file_sha(manifest_path) == expected_sha256, "Frozen execution runtime manifest changed.")
    manifest = read(manifest_path)
    require(manifest["version"] == VERSION and manifest["scope"] == "final_test"
            and manifest["status"] == "assembled_model_specific_runtime" and manifest["entrypoint"] == ENTRY
            and manifest["final_test_bodies_read"] is False and manifest["model_weights_read"] is False,
            "Invalid model-specific runtime manifest.")
    require(set(manifest["source_origin"]) == set(manifest["source_sha256"])
            and set(manifest["source_origin"].values()) <= {"frozen_preparation", "final_only_overlay"},
            "Runtime source origins changed.")
    if "runtime_family" in manifest:
        import small_final_test_runtime as small
        import gemma_final_test_runtime as gemma
        trained_roles = {"sft"} if manifest.get("dpo_omission_sha256") is not None else {"sft", "sft_dpo"}
        require(manifest["runtime_family"] in {small.FAMILY, gemma.FAMILY}
                and set(manifest["training_implementation_sha256"]) == trained_roles
                and all(manifest["training_implementation_sha256"].values()),
                "Final runtime must preserve every trained role's original source evidence.")
        if manifest["runtime_family"] == small.FAMILY:
            require(manifest["source_policy"] == "explicit_frozen_inference_runtime_with_separate_training_source_provenance"
                    and not manifest["configuration_sources_sha256"]
                    and set(manifest["source_origin"].values()) == {"final_only_overlay"},
                    "Small inference runtime must preserve separate original training-source evidence.")
        else:
            require(manifest["source_policy"] == "frozen_training_sources_with_final_inference_overlay"
                    and manifest["configuration_sources_sha256"], "Gemma runtime must retain its frozen training sources.")
    for name, expected in manifest["source_sha256"].items():
        require(file_sha(child(root, name)) == expected, "Execution runtime source changed.")
    for name, expected in manifest["configuration_sources_sha256"].items():
        require(manifest["source_sha256"].get("ml/" + name) == expected
                and manifest["source_origin"].get("ml/" + name) == "frozen_preparation",
                "A frozen training source was overridden by the final-only overlay.")
    require(all(str(path.relative_to(root)) in manifest["source_sha256"]
                for directory in (root / "ml", root / "server") for path in directory.rglob("*.py")),
            "Unregistered Python source can shadow the runtime imports.")
    result = {"runtime_manifest_sha256": file_sha(manifest_path),
            "configuration_sha256": manifest["configuration_sha256"],
            "preparation_manifest_sha256": manifest["preparation_manifest_sha256"],
            "source_sha256": manifest["source_sha256"]}
    if "runtime_family" in manifest:
        result["runtime_family"] = manifest["runtime_family"]
        result["inference_policy_sha256"] = manifest["inference_policy_sha256"]
    return result


def metadata_preflight(runtime, config_path, output):
    identity = load_runtime(runtime)
    config_path = Path(config_path).resolve()
    require(file_sha(config_path) == identity["configuration_sha256"]
            and file_sha(config_path.parent / "manifest.json") == identity["preparation_manifest_sha256"],
            "Runtime is bound to another model preparation.")
    import train_gemma_actions_v3 as sft
    require(Path(sft.__file__).resolve() == Path(runtime).resolve() / "ml/train_gemma_actions_v3.py",
            "Run preflight through the assembled runtime's script, using normal sibling imports.")
    _, _, _, _, _, verified = sft.preflight(config_path, assets=False)
    value = {"status": "metadata_preflight_passed", **identity, "verified": verified,
             "actual_trainer_source_sha256": file_sha(sft.__file__),
             "actual_preparer_source_sha256": file_sha(sft.preparation.__file__),
             "native_model_loaded": False, "gpu_used": False, "final_test_bodies_read": False}
    write(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assembly = commands.add_parser("assemble")
    assembly.add_argument("--config", required=True)
    assembly.add_argument("--output", required=True)
    check = commands.add_parser("metadata-preflight")
    check.add_argument("--runtime", required=True)
    check.add_argument("--config", required=True)
    check.add_argument("--output", required=True)
    args = parser.parse_args()
    result = (assemble(args.config, args.output) if args.command == "assemble"
              else metadata_preflight(args.runtime, args.config, args.output))
    print(json.dumps({key: value for key, value in result.items() if key != "source_sha256"}))


if __name__ == "__main__":
    main()
