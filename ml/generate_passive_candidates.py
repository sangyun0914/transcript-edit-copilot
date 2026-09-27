"""Generate one fixed train-only candidate per state from the final T1 adapter."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import train_passive_curriculum as curriculum
import train_passive_sft as smoke
import prepare_passive_preferences as preferences

VERSION = "passive-preference-candidates-v1"


def verified_manifest(directory, status):
    directory = Path(directory).resolve()
    manifest = smoke.read(directory / "manifest.json")
    smoke.require(
        manifest.get("version") == curriculum.VERSION
        and manifest.get("status") == status,
        "Preference generation requires completed T1 training and fresh reload.",
    )
    required = {"run.json", "smoke-generation.json"}
    required |= (
        {
            "adapter-tensors.json",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
        }
        if status == "training_complete_fresh_reload_pending"
        else {"reload-parity.json"}
    )
    smoke.require(
        required <= set(manifest["files_sha256"]), "Unbound required anchor evidence."
    )
    for name, expected in manifest["files_sha256"].items():
        smoke.check_file(smoke.child(directory, name), expected)
    return manifest


def validate_anchor(training, reloaded, parity, training_sha, config_sha):
    smoke.require(
        training.get("mode") == "train"
        and training.get("status") == "training_complete_fresh_reload_pending"
        and training.get("training_updates") == 203
        and training.get("optimizer_committed_rows") == 809
        and training.get("configuration_sha256") == config_sha,
        "Use the fixed final T1 checkpoint, never smoke or recovery checkpoints.",
    )
    smoke.require(
        reloaded.get("mode") == "fresh_reload"
        and reloaded.get("status") == "fresh_reload_passed_local_export_pending"
        and reloaded.get("reloaded_training_updates") == 203
        and reloaded.get("previous_manifest_sha256") == training_sha
        and reloaded.get("configuration_sha256") == config_sha
        and reloaded.get("pid") != training.get("pid")
        and parity.get("adapter_tensors_identical") is True
        and len(parity.get("cases", [])) == 8
        and len({row["case_id"] for row in parity["cases"]}) == 8
        and all(row.get("identical") is True for row in parity["cases"]),
        "The candidate source must have passed an independent exact reload.",
    )


def unique_rows(bundle):
    cases, encoded = bundle["cases"], bundle["tokens"]
    smoke.require(
        len(cases) == len(encoded) == 809
        and len({case["id"] for case in cases}) == 809,
        "Generate once per unique frozen training state, not per weighted draw.",
    )
    result = []
    for case, tokens in zip(cases, encoded, strict=True):
        smoke.require(
            case["metadata"]["split"] == "train"
            and case["metadata"]["training_eligible"] is True
            and case["id"] == tokens["id"]
            and tokens["split"] == "train",
            "Candidate generation cannot consume development or test states.",
        )
        result.append({"case_id": case["id"], **tokens["primary"]})
    return result


def run(config_path, training_directory, reload_directory, protocol_path, output):
    config_path, protocol_path = (
        Path(config_path).resolve(),
        Path(protocol_path).resolve(),
    )
    training_directory, reload_directory = (
        Path(value).resolve() for value in (training_directory, reload_directory)
    )
    config, root, bundle, _, _, verified = curriculum.preflight(config_path)
    protocol = smoke.read(protocol_path)
    config_sha = verified["configuration_sha256"]
    policy_binding = protocol["qualification_policy"]
    policy_path = smoke.child(protocol_path.parent, policy_binding["path"])
    smoke.check_file(policy_path, policy_binding["sha256"])
    policy = smoke.read(policy_path)
    preferences.policy_check(policy)
    smoke.require(
        protocol.get("version") == VERSION
        and protocol.get("generator_sha256") == smoke.digest(Path(__file__))
        and protocol.get("configuration_sha256") == config_sha
        and protocol.get("preparation_sha256") == verified["preparation_sha256"]
        and protocol.get("generation") == config["generation"]
        and protocol.get("candidate_count") == 809
        and protocol.get("candidates_per_state") == 1
        and protocol.get("state_order") == "frozen_training_order"
        and protocol.get("candidate_source") == "final_T1_NF4_adapter"
        and protocol.get("development_or_test_candidates") is False
        and protocol.get("qualification_policy_sha256") == policy_binding["sha256"]
        and protocol.get("qualifier_sha256") == smoke.digest(Path(preferences.__file__))
        and policy.get("qualifier_sha256") == protocol["qualifier_sha256"]
        and protocol.get("wall_limit_seconds") == 21600,
        "Use the fixed train-only candidate generation protocol.",
    )
    smoke.require(
        verified["environment"]["matched"], "Use the pinned Colab environment."
    )
    training_manifest = verified_manifest(
        training_directory, "training_complete_fresh_reload_pending"
    )
    verified_manifest(reload_directory, "fresh_reload_passed_local_export_pending")
    training = smoke.read(training_directory / "run.json")
    reloaded = smoke.read(reload_directory / "run.json")
    parity = smoke.read(reload_directory / "reload-parity.json")
    validate_anchor(
        training,
        reloaded,
        parity,
        smoke.digest(training_directory / "manifest.json"),
        config_sha,
    )
    smoke.require(
        smoke.read(training_directory / "smoke-generation.json")
        == smoke.read(reload_directory / "smoke-generation.json"),
        "Fresh reload evidence changed.",
    )
    rows = unique_rows(bundle)
    output = Path(output).resolve()
    smoke.require(
        not output.exists()
        and not output.is_relative_to(root)
        and not output.is_relative_to(training_directory)
        and not output.is_relative_to(reload_directory),
        "Use a fresh candidate output directory outside all immutable inputs.",
    )
    output.mkdir(parents=True)
    started = time.monotonic()
    info = {
        "version": VERSION,
        "status": "loading",
        "mode": "candidate_generation_only",
        "pid": os.getpid(),
        "protocol_sha256": smoke.digest(protocol_path),
        "configuration_sha256": config_sha,
        "preparation_sha256": verified["preparation_sha256"],
        "training_manifest_sha256": smoke.digest(training_directory / "manifest.json"),
        "reload_manifest_sha256": smoke.digest(reload_directory / "manifest.json"),
        "generator_sha256": smoke.digest(Path(__file__)),
        "curriculum_trainer_sha256": smoke.digest(Path(curriculum.__file__)),
        "smoke_trainer_sha256": smoke.digest(Path(smoke.__file__)),
        "candidate_source": "final_T1_NF4_adapter",
        "requested_candidates": 809,
        "completed_candidates": 0,
        "training_updates": 0,
        "development_or_test_record_bodies_read": 0,
        "environment": verified["environment"],
        "scope": "Train-only model outputs for later AI preference qualification; candidates are not automatically negatives or preference pairs.",
    }
    smoke.write(output / "run.json", info)
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    (output / "generator.py").write_bytes(Path(__file__).read_bytes())
    torch = None
    try:
        from peft import PeftModel, get_peft_model_state_dict

        torch, model, tokenizer, device, runtime = smoke.load_runtime(
            config, root, bundle["tokens"]
        )
        model = smoke.prepare_reload_base(model, config)
        model = PeftModel.from_pretrained(
            model,
            training_directory / "adapter",
            is_trainable=False,
            local_files_only=True,
        )
        model.requires_grad_(False)
        expected = smoke.read(training_directory / "adapter-tensors.json")
        smoke.require(
            smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            == expected,
            "The preference generator loaded a different adapter.",
        )
        smoke.write(output / "runtime.json", runtime)

        def emitted(row):
            expected_row = rows[info["completed_candidates"]]
            smoke.require(
                row["case_id"] == expected_row["case_id"], "Candidate order changed."
            )
            prefix = expected_row["input_ids"][: expected_row["completion_start"]]
            smoke.require(
                row["prompt_ids_sha256"]
                == hashlib.sha256(
                    json.dumps(prefix, separators=(",", ":")).encode()
                ).hexdigest(),
                "Candidate prompt differs from its frozen training state.",
            )
            curriculum.append(output / "candidates.jsonl", row)
            info["completed_candidates"] += 1
            info["status"] = "generating"
            smoke.write(output / "run.json", info)
            print(
                json.dumps(
                    {
                        "completed": info["completed_candidates"],
                        "case_id": row["case_id"],
                        "tokens": len(row["output_ids"]),
                        "finish_reason": row["finish_reason"],
                    }
                ),
                flush=True,
            )
            curriculum.check_deadline(started, protocol["wall_limit_seconds"])

        curriculum.check_deadline(started, protocol["wall_limit_seconds"])
        smoke.generate_smoke(model, tokenizer, rows, config, torch, device, emitted)
        smoke.require(
            info["completed_candidates"] == 809, "Incomplete candidate generation."
        )
        smoke.require(
            smoke.tensor_records(
                get_peft_model_state_dict(model, save_embedding_layers=False), torch
            )
            == expected,
            "Inference changed the fixed SFT adapter.",
        )
        smoke.require(
            smoke.read(training_directory / "manifest.json") == training_manifest,
            "Training evidence changed while generating candidates.",
        )
        info["status"] = "completed_train_candidates_unqualified"
    except BaseException as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        info["seconds"] = time.monotonic() - started
        if torch is not None:
            info.update(
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
            )
        smoke.write(output / "run.json", info)
        smoke.write(
            output / "manifest.json",
            {
                "version": VERSION,
                "status": info["status"],
                "files_sha256": {
                    file.name: smoke.digest(file)
                    for file in sorted(output.iterdir())
                    if file.is_file() and file.name != "manifest.json"
                },
            },
        )
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "training", "reload", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, args.training, args.reload, args.protocol, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
