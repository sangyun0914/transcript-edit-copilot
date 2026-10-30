"""Prepare the conditional unified Gemma26 fallback without training or TEST access."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import platform
import shutil
import time

import prepare_gemma_actions_v3 as prep
import train_gemma_actions_v3 as trainer

VERSION = "gemma26-final-conditional-fallback-2"
RUNTIME_VERSION = "gemma26-final-owned-runtime-inspection-1"
MODELS = ("lfm25-1.2b", "gemma4-e2b")
LR_RECOVERY_PLAN = Path(__file__).resolve().parents[1] / "docs/plans/final-small-lr-recovery-2026-10-25.md"
LR_RECOVERY_PLAN_SHA = "de8ec9f80935568058f4421035532561f7d014bededaa1bb383c5957bfc0d2ae"
EXECUTION_PLAN = Path(__file__).resolve().parents[1] / "docs/plans/final-gemma26-execution-2026-10-25.md"
ROWS, EPOCHS, SEED, ACCUMULATION, LEARNING_RATE = 1261, 3, 42, 4, 5e-5
require, read, write, sha = prep.require, prep.read, prep.write, prep.digest


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def bound(value):
    path = Path(value["path"])
    require(path.is_file() and sha(path) == value["sha256"], "Bound file changed.")
    return path


def epoch_order(case_ids, configurations, expected_rows=ROWS):
    require(
        len(case_ids) == len(set(case_ids)) == expected_rows,
        "Require every unique final TRAIN row.",
    )
    orders = []
    for config in configurations:
        training = config["training"]
        require(
            all(
                training[key] == value
                for key, value in {
                    "epochs": EPOCHS,
                    "seed": SEED,
                    "gradient_accumulation": ACCUMULATION,
                    "learning_rate": LEARNING_RATE,
                    "micro_batch_size": 1,
                    "packing": False,
                    "truncation": False,
                }.items()
            ),
            "Small configuration differs from the registered three-epoch recipe.",
        )
        order = training["sample_order"]
        require(
            len(order) == EPOCHS * expected_rows
            and all(type(index) is int for index in order)
            and all(
                sorted(order[start : start + expected_rows])
                == list(range(expected_rows))
                for start in range(0, len(order), expected_rows)
            ),
            "Each epoch must visit every final TRAIN row exactly once.",
        )
        orders.append(order)
    require(len(orders) == 2 and orders[0] == orders[1], "Small epoch orders differ.")
    return [case_ids[index] for index in orders[0]]


def validate_lr_recovery(original, recovery):
    require(original["model"] == recovery["model"] == "lfm25-1.2b", "The additional candidate must be LFM.")
    expected = copy.deepcopy(original)
    expected["training"]["learning_rate"] = 2e-4
    require(recovery == expected, "The registered LFM recovery changes only learning rate to 2e-4.")


def draft(release, small_configurations, native_pin, native, historical_config, output, lr_recovery_configuration):
    from transformers import AutoTokenizer

    output, release = Path(output).resolve(), Path(release).resolve()
    require(not output.exists(), "Use a new conditional readiness directory.")
    metadata = read(release)
    require(
        metadata["counts"]["train"] == ROWS and metadata["test_bodies_read"] is False,
        "Use the final unified TRAIN release without TEST access.",
    )
    cases_path = release.parent / "cases/train.jsonl"
    require(
        sha(cases_path) == metadata["files_sha256"]["cases/train.jsonl"],
        "Final TRAIN release bytes changed.",
    )
    cases = prep.rows(cases_path)
    configurations = [read(path) for path in small_configurations]
    require(
        [config["model"] for config in configurations] == list(MODELS),
        "Supply LFM and E2B configurations in their registered order.",
    )
    validate_lr_recovery(configurations[0], read(lr_recovery_configuration))
    require(sha(LR_RECOVERY_PLAN) == LR_RECOVERY_PLAN_SHA, "The prospective LFM recovery plan changed.")
    case_ids = [case["id"] for case in cases]
    for path, config in zip(small_configurations, configurations, strict=True):
        train_rows = Path(path).parent / "train.jsonl.gz"
        require(
            config["source_cases_sha256"] == sha(cases_path)
            and config["train_rows_sha256"] == sha(train_rows)
            and config["final_test_bodies_read"] is False,
            "Small configuration belongs to a different TRAIN release.",
        )
        saved = prep.rows(train_rows)
        require(
            [row["case_id"] for row in saved] == case_ids
            and all(
                row["messages"] == case["messages"]
                for row, case in zip(saved, cases, strict=True)
            ),
            "Small sample indices do not map to the same saved case messages.",
        )
    order = epoch_order(case_ids, configurations)
    pin, historical = read(native_pin), read(historical_config)
    require(
        pin["repository"] == prep.gemma.REPOSITORY
        and pin["revision"] == prep.gemma.REVISION,
        "The historical Gemma26 native identity changed.",
    )
    trainer.validate_configuration(historical)
    for name in prep.TOKENIZER_FILES:
        require(
            sha(Path(native) / name) == pin["files"][name]["sha256"],
            "Tokenizer changed.",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        native, local_files_only=True, trust_remote_code=False
    )
    admitted, tokens, excluded = prep.tokenize_cases(cases, tokenizer, ("json",))
    require(
        admitted == cases and len(admitted) == ROWS and not excluded,
        "The full fallback cohort must fit without exclusions or truncation.",
    )
    records = tokens["json"]
    for case, record in zip(cases, records, strict=True):
        prep.validate_record(case, record, "json")
    budget = prep.budget(cases, records, order, ACCUMULATION)
    require(
        budget["draw_count"] == 3783
        and budget["max_updates"] == 946
        and budget["final_update_rows"] == 3,
        "Exact final fallback optimizer budget changed.",
    )
    output.mkdir(parents=True)
    write(
        output / "draw-order.json",
        {"version": prep.DRAW_ORDER_VERSION, "sample_order": order},
    )
    validation = {
        "original_train_rows": ROWS,
        "admitted_rows": ROWS,
        "excluded_rows": 0,
        "history_lengths": dict(
            sorted(
                Counter(
                    str(case.get("history_length", {}).get("length", 3))
                    for case in cases
                ).items()
            )
        ),
        "saved_inputs_and_messages_preserved": True,
        "all_accepted_references_validated": sum(
            len(prep.references(case)) for case in cases
        ),
        "min_prompt_tokens": min(row["primary"]["completion_start"] for row in records),
        "max_prompt_tokens": max(row["primary"]["completion_start"] for row in records),
        "max_primary_completion_tokens": max(
            len(row["primary"]["input_ids"]) - row["primary"]["completion_start"]
            for row in records
        ),
        "max_primary_sequence_tokens": max(
            len(row["primary"]["input_ids"]) for row in records
        ),
        "training_completion_eos_token_id": 1,
        "generation_terminal_ids": list(prep.GENERATION["eos_token_id"]),
        "weights_loaded": False,
        "training_started": False,
    }
    write(output / "token-validation.json", validation)
    protocol = {
        "version": VERSION,
        "status": "conditional_runtime_unbound_not_training_authorized",
        "final_test_bodies_read": False,
        "eligibility_format": "json",
        "dataset": {"release": binding(release), "train_cases_sha256": sha(cases_path)},
        "training_plan": {
            "seed": SEED,
            "sampling_seed": SEED,
            "passes": 1,
            "gradient_accumulation": ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "draw_order_mode": "explicit",
            "draw_order_sha256": sha(output / "draw-order.json"),
            "replacement": True,
            **budget,
        },
        "epoch_semantics": {
            "epochs": EPOCHS,
            "unique_rows_per_epoch": ROWS,
            "each_case_exposures": EPOCHS,
            "order_policy": "Exact row indices from both frozen small configurations, mapped to saved final TRAIN IDs.",
            "replacement_field": "Legacy explicit-stream flag for repeated IDs across epochs; no within-epoch replacement.",
        },
        "terminal_policy": {
            "training": "JSON completion plus historical Gemma26 EOS1, completion-only loss.",
            "generation": prep.GENERATION,
            "comparison_note": "Preserves the validated Gemma26 recipe; it is not the small-native chat-template completion terminal policy.",
        },
        "fallback_condition": "Original LFM, E2B and the registered fresh LFM LR2e-4 candidate must each complete all 471 native DEV requests and fail the unchanged numeric gate; root owns activation and model selection.",
        "lr_recovery_plan": binding(LR_RECOVERY_PLAN),
        "execution_plan": binding(EXECUTION_PLAN),
        "runtime_inspection": None,
        "historical_runtime_is_not_current_evidence": True,
    }
    write(output / "protocol-draft.json", protocol)
    receipt = {
        "version": VERSION,
        "status": protocol["status"],
        "final_test_bodies_read": False,
        "training_started": False,
        "runtime_environment": None,
        "gpu_name": None,
        "selected_model": None,
        "inputs": {
            "release": binding(release),
            "train_cases": binding(cases_path),
            "native_pin": binding(native_pin),
            "historical_config": binding(historical_config),
            "small_configurations": [binding(path) for path in small_configurations],
            "lr_recovery_configuration": binding(lr_recovery_configuration),
            "lr_recovery_plan": binding(LR_RECOVERY_PLAN),
            "execution_plan": binding(EXECUTION_PLAN),
        },
        "native_metadata_path": str(Path(native).resolve()),
        "source_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "prepare_gemma_actions_v3.py",
                "train_gemma_actions_v3.py",
                "train_passive_gemma_actions.py",
                "train_passive_gemma.py",
                "passive_formats_actions_v3.py",
            )
        },
        "files_sha256": {
            path.name: sha(path) for path in output.iterdir() if path.is_file()
        },
    }
    write(output / "manifest.json", receipt)
    return {
        "status": receipt["status"],
        "draws": 3783,
        "updates": 946,
        "output": str(output),
        **validation,
    }


def inspect_runtime(output, owned_runtime_id):
    import torch

    require(
        bool(owned_runtime_id.strip()), "An actual owned G4 runtime ID is required."
    )
    require(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "Inspect an actual BF16-capable CUDA runtime.",
    )
    gpu = torch.cuda.get_device_properties(0)
    actual = trainer.smoke.environment(
        {"libraries": {}, "absent_libraries": ["torchao"]}
    )["actual"]
    require(
        gpu.total_memory >= 75000000000 and actual["torchao"] is None,
        "Runtime memory or torchao requirement failed.",
    )
    value = {
        "version": RUNTIME_VERSION,
        "status": "owned_runtime_observed_without_model_load",
        "owned_runtime_id": owned_runtime_id,
        "observed_unix": time.time(),
        "hostname": platform.node(),
        "libraries": {key: actual[key] for key in trainer.smoke.LIBRARIES},
        "absent_libraries": ["torchao"],
        "gpu": {
            "name": gpu.name,
            "total_memory": gpu.total_memory,
            "bf16_supported": True,
        },
        "weights_loaded": False,
        "training_started": False,
        "final_test_bodies_read": False,
    }
    write(output, value)
    return value


def runtime_template(historical, inspection):
    require(
        inspection["version"] == RUNTIME_VERSION
        and inspection["status"] == "owned_runtime_observed_without_model_load"
        and bool(inspection["owned_runtime_id"])
        and inspection["final_test_bodies_read"] is False
        and inspection["weights_loaded"] is False
        and inspection["training_started"] is False,
        "Require a future actual owned runtime inspection.",
    )
    libraries, gpu = inspection["libraries"], inspection["gpu"]
    require(
        set(libraries) == set(trainer.smoke.LIBRARIES)
        and all(isinstance(value, str) and value for value in libraries.values())
        and all(
            libraries[key] == historical["libraries"][key]
            for key in libraries
            if key != "python"
        )
        and libraries["python"].split(".")[:2]
        == historical["libraries"]["python"].split(".")[:2]
        and inspection["absent_libraries"] == ["torchao"]
        and isinstance(gpu["name"], str)
        and bool(gpu["name"])
        and gpu["total_memory"] >= 75000000000
        and gpu["bf16_supported"] is True,
        "Actual runtime differs from the supported numerical recipe; inspect before changing it.",
    )
    result = {
        key: copy.deepcopy(historical[key])
        for key in ("libraries", "absent_libraries", "model", "base", "tokenizer")
    }
    result["libraries"] = libraries
    result["model"]["gpu_name_contains"] = gpu["name"]
    return result


def failed_development(summary_path, configuration_binding, output):
    from evaluate_small_actions_final import summarize

    summary = read(summary_path)
    report_path, mapping_path = (
        bound(summary["report"]),
        bound(summary["cohort_mapping"]),
    )
    report = read(report_path)
    require(
        summary["model_identity"]["kind"] == "sft"
        and summary["model_identity"]["configuration_sha256"]
        == configuration_binding["sha256"]
        and report["model_identity"] == summary["model_identity"]
        and report["numerical_regime"] == "native_bf16_fp32_lora"
        and report["schema_constrained"] is False
        and report["final_test"] is False
        and report["excluded_before_inference"] == 0
        and summary["overall"]["requests"] == 471
        and {
            name: value["metrics"]["requests"]
            for name, value in summary["cohorts"].items()
        }
        == {"canonical_dev322": 322, "supplemental_dev149": 149},
        "Require the exact candidate's complete native unified DEV result.",
    )
    run = read(bound(report["inference_run"]))
    require(
        run["status"] == "inference_complete", "Candidate inference did not complete."
    )
    checked = summarize(report_path, mapping_path, output)
    require(
        checked == summary
        and checked["minimum_numeric_quality_gate"]["passed"] is False,
        "Every prerequisite candidate must have actually failed the registered DEV gate.",
    )
    return binding(summary_path)


def prepare(draft_manifest, runtime_inspection, small_summaries, output, lr_recovery_summary):
    output = Path(output).resolve()
    require(not output.exists(), "Use a new immutable activated preparation directory.")
    receipt = read(draft_manifest)
    root = Path(draft_manifest).resolve().parent
    require(
        receipt["version"] == VERSION
        and receipt["status"] == "conditional_runtime_unbound_not_training_authorized",
        "Unknown conditional draft.",
    )
    for name, expected in receipt["files_sha256"].items():
        require(sha(root / name) == expected, "Conditional readiness file changed.")
    for name, expected in receipt["source_sha256"].items():
        require(
            sha(Path(__file__).with_name(name)) == expected,
            "Fallback source changed; make a new draft.",
        )
    inputs = receipt["inputs"]
    for key in ("release", "train_cases", "native_pin", "historical_config"):
        bound(inputs[key])
    for value in inputs["small_configurations"]:
        bound(value)
    recovery_config = bound(inputs["lr_recovery_configuration"])
    recovery_plan = bound(inputs["lr_recovery_plan"])
    bound(inputs["execution_plan"])
    require(sha(recovery_plan) == LR_RECOVERY_PLAN_SHA, "The registered recovery decision changed.")
    validate_lr_recovery(read(bound(inputs["small_configurations"][0])), read(recovery_config))
    template = runtime_template(
        read(bound(inputs["historical_config"])), read(runtime_inspection)
    )
    require(
        len(small_summaries) == 2, "Require both completed small native DEV summaries."
    )
    output.mkdir(parents=True)
    failures = [
        failed_development(summary, config, output / f"verified-{model}-summary.json")
        for model, summary, config in zip(
            MODELS, small_summaries, inputs["small_configurations"], strict=True
        )
    ]
    recovery_failure = failed_development(
        lr_recovery_summary, inputs["lr_recovery_configuration"],
        output / "verified-lfm-lr2e4-summary.json",
    )
    protocol = read(root / "protocol-draft.json")
    protocol.update(
        status="conditional_prerequisites_bound_not_trained",
        runtime_inspection=binding(runtime_inspection),
        completed_small_failure_summaries=failures,
        completed_lr_recovery_failure_summary=recovery_failure,
        readiness_manifest=binding(draft_manifest),
    )
    write(output / "protocol.json", protocol)
    write(output / "runtime-template.json", template)
    shutil.copyfile(runtime_inspection, output / "runtime-inspection.json")
    result = prep.prepare(
        bound(inputs["train_cases"]),
        bound(inputs["native_pin"]),
        receipt["native_metadata_path"],
        output / "runtime-template.json",
        output / "preparation",
        phase="substantive",
        passes=1,
        seed=SEED,
        accumulation=ACCUMULATION,
        learning_rate=LEARNING_RATE,
        protocol=output / "protocol.json",
        eligibility_format="json",
        draw_order=root / "draw-order.json",
    )
    source = read(output / "preparation/source-binding.json")
    require(
        source["original_train_rows"] == source["admitted_rows"] == ROWS
        and source["budget_exclusions"] == 0,
        "No final TRAIN exclusion is allowed.",
    )
    config, _, _, _, _, verified = trainer.preflight(
        output / "preparation/config-json.json", assets=False
    )
    require(
        config["training"] == {**config["training"], **protocol["training_plan"]},
        "Frozen fallback plan changed.",
    )
    write(
        output / "activation.json",
        {
            "version": VERSION,
            "status": "prepared_not_trained",
            "preparation": result,
            "metadata_preflight": verified,
            "training_started": False,
            "final_test_bodies_read": False,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("draft")
    for key in ("release", "native-pin", "native", "historical-config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--small-configurations", nargs=2, type=Path, required=True)
    p.add_argument("--lr-recovery-configuration", type=Path, required=True)
    p = sub.add_parser("inspect-runtime")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--owned-runtime-id", required=True)
    p = sub.add_parser("prepare")
    for key in ("draft-manifest", "runtime-inspection", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--small-summaries", nargs=2, type=Path, required=True)
    p.add_argument("--lr-recovery-summary", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command").replace("-", "_")
    print(json.dumps(globals()[command](**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
