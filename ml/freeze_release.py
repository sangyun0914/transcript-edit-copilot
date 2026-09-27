"""Freeze only a fully verified, independently admitted train/dev draft."""

import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from uuid import uuid4

from dataset_release import verify
from dataset_release_inventory import ROOT, digest, read, rows, write
from iteration import jsonl
from nep_v2 import ACTIONS
from replay_session import session_hash
from release_audit import audit_release
from reviewed_audio import CUTS
from source_qualification import TOKENIZER


def validate_admission(directory, admission):
    cases = rows(directory / "train-cases.jsonl") + rows(directory / "dev-cases.jsonl")
    if admission["draft_manifest_sha256"] != digest(directory / "manifest.json"):
        raise ValueError("Independent review belongs to a different draft.")
    if admission.get("status") != "passed_independent_census":
        raise ValueError("A passing independent census is required.")
    artifacts = {}
    for name in ("registration", "judgments"):
        path = (ROOT / admission[f"review_{name}_path"]).resolve()
        if (
            not path.is_relative_to(ROOT)
            or digest(path) != admission[f"review_{name}_sha256"]
        ):
            raise ValueError("Independent review evidence is missing or changed.")
        artifacts[name] = path
    registration = read(artifacts["registration"])
    case_ids = {c["id"] for c in cases}
    if (
        registration["case_count"] != len(cases)
        or len(registration["ordered_case_ids"]) != len(cases)
        or set(registration["ordered_case_ids"]) != case_ids
        or len(registration["case_map"]) != len(cases)
        or set(registration["case_map"].values()) != case_ids
    ):
        raise ValueError("Independent census registration differs from the population.")
    for split in ("train", "dev"):
        path = directory / f"{split}-cases.jsonl"
        if registration["population_files"].get(str(path.relative_to(ROOT))) != digest(
            path
        ):
            raise ValueError("Independent review population hash changed.")
    judgments = rows(artifacts["judgments"])
    keys = [r["audit_key"] for r in judgments]
    if len(set(keys)) != len(keys) or set(keys) != set(registration["case_map"]):
        raise ValueError("Input-first judgments do not cover the complete census.")
    by_id = {c["id"]: c for c in cases}
    for row in judgments:
        case = by_id[registration["case_map"][row["audit_key"]]]
        if row["input_sha256"] != session_hash(case["input"]) or not row.get(
            "input_visible_evidence"
        ):
            raise ValueError(
                "Input-first judgment is unbound or lacks visible evidence."
            )
    reviewed = {r["case_id"]: r for r in admission["cases"]}
    if len(reviewed) != len(admission["cases"]) or set(reviewed) != {
        c["id"] for c in cases
    }:
        raise ValueError(
            "Independent review must cover every admitted case exactly once."
        )
    if admission.get("unresolved"):
        raise ValueError("Unresolved reference review blocks freeze.")
    for case in cases:
        review = reviewed[case["id"]]
        if (
            review["verdict"] != "accept"
            or review["input_sha256"] != session_hash(case["input"])
            or review["reference_sha256"]
            != session_hash([case["expectedOutput"], *case["acceptableOutputs"]])
            or not review.get("reason")
        ):
            raise ValueError(f"Unaccepted or stale reference review: {case['id']}")
    if {c["expectedOutput"]["action"] for c in cases} != set(ACTIONS):
        raise ValueError(
            "A required action has no positive reference, even in supported scope."
        )
    if {c["metadata"]["language"] for c in cases} != {"en", "ko"}:
        raise ValueError("Both project languages must be represented.")
    return cases


def freeze(directory, output, admission_path, evidence_paths, tokenizer):
    if output.exists():
        raise ValueError("Frozen releases are immutable; choose a new version.")
    verification = verify(directory, tokenizer)
    admission = read(admission_path)
    cases = validate_admission(directory, admission)
    protocol = read(directory / "test-protocol.json")
    split_manifest = read(directory / "split-manifest.json")
    if (
        protocol.get("generated_cases") != 0
        or protocol.get("reserved_test_bodies_opened_by_this_auditor") is not False
        or read(directory / "audit.json").get("test_bodies_opened") is not False
        or {r["source_group_id"] for r in protocol["source_allocations"]}
        != set(split_manifest["reserved_test_groups"])
        or protocol.get("artifact_kind")
        != "protocol_only_no_test_bodies_no_test_labels"
    ):
        raise ValueError("Final-test design or sealed-source allocation is incomplete.")
    evidence = [read(p) for p in evidence_paths]
    for report in evidence:
        if (
            report.get("manifest_sha256") != digest(directory / "manifest.json")
            or report.get("status") != "passed"
        ):
            raise ValueError("Execution/audio audit is stale or did not pass.")
    required_audits = {"episode_causality", "semantic_timing", "full_caption_export"}
    if {r.get("audit_kind") for r in evidence} != required_audits:
        raise ValueError(
            "Freeze requires episode, media/semantic and full-caption-export audits."
        )
    old_manifest = read(directory / "manifest.json")
    recording_cuts = [
        c
        for c in cases
        if c["expectedOutput"]["action"] in CUTS
        and c["metadata"]["media_kind"] == "recording"
    ]
    counts = {
        "cases": len(cases),
        "splits": dict(Counter(c["metadata"]["split"] for c in cases)),
        "actions": dict(Counter(c["expectedOutput"]["action"] for c in cases)),
        "languages": dict(Counter(c["metadata"]["language"] for c in cases)),
        "natural_audio_cuts": len(recording_cuts),
        "symbolic_cuts": sum(
            c["expectedOutput"]["action"] in CUTS
            and c["metadata"]["media_kind"] == "authored_symbolic_timeline"
            for c in cases
        ),
        "unique_reviewed_operations": len(
            {
                q
                for c in recording_cuts
                for q in c["metadata"]["audio_qualification_ids"]
            }
        ),
    }
    for field, value in counts.items():
        if old_manifest.get(field) != value:
            raise ValueError(f"Manifest count differs from admitted rows: {field}")
    if not recording_cuts or any(
        not c["metadata"]["audio_qualification_ids"] for c in recording_cuts
    ):
        raise ValueError("Natural audio cut coverage has no qualified reference.")

    destination = output
    output = destination.with_name(f".{destination.name}.staging-{uuid4().hex}")
    shutil.copytree(directory, output)
    provenance = output / "freeze-provenance"
    provenance.mkdir()
    shutil.copyfile(directory / "manifest.json", provenance / "draft-manifest.json")
    shutil.copyfile(admission_path, provenance / "independent-admission.json")
    for name in ("registration", "judgments"):
        path = ROOT / admission[f"review_{name}_path"]
        shutil.copyfile(path, provenance / path.name)
    for path in evidence_paths:
        shutil.copyfile(path, provenance / path.name)
    release_id = f"nep-release-v1-d4-schema4-2026-09-26-{digest(directory / 'manifest.json')[:12]}"
    for case in cases:
        case["metadata"].update(
            training_eligible=case["metadata"]["split"] == "train",
            review_status="agent_validated_observable_rule",
            release_status="frozen_supported_scope",
            release_id=release_id,
            independent_review="freeze-provenance/independent-admission.json",
        )
    for split in ("train", "dev"):
        jsonl(
            output / f"{split}-cases.jsonl",
            [c for c in cases if c["metadata"]["split"] == split],
        )
    diversity = audit_release(
        cases,
        rows(output / "contexts/qwen3.5-2b.jsonl"),
        rows(output / "executions.jsonl"),
    )
    (output / "diversity-audit.json").write_text(
        json.dumps(diversity, ensure_ascii=False, indent=2) + "\n"
    )

    coverage = read(output / "scenario-coverage.json")
    exclusions = rows(output / "exclusions.jsonl")
    for decision in coverage["decisions"]:
        admitted = decision["draft_case_ids"]
        decision.update(
            admitted_case_ids=admitted,
            status="admitted_supported_scope" if admitted else "documented_gap",
        )
        decision["excluded_candidates"] = [
            r["id"] for r in exclusions if decision["id"] in r["scenario_ids"]
        ]
        if decision["id"] == "A01":
            decision.update(
                status="eligibility_only_natural_hard_negative_missing",
                evidence_scope="The admitted clean-cut case carries invalid endpoint and unqualified-boundary exclusion probes. It does not establish perceptual detection of hazardous consonant/ending cuts in natural audio.",
            )
        if (
            decision["remaining"]
            == "Input-only semantic audit and broader independent coverage remain before freeze."
        ):
            decision["remaining"] = (
                "The listed references passed the independent census. Additional source/condition coverage is not implied by admission."
            )
    coverage.update(
        coverage_complete=False,
        D4_complete=True,
        specifications_with_admitted_evidence=sum(
            bool(r["admitted_case_ids"]) for r in coverage["decisions"]
        ),
        interpretation="Admission validates listed decisions only. A symbolic positive is not natural-audio qualification, and populated content IDs do not imply every planned acoustic/language contrast is complete.",
    )
    (output / "scenario-coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n"
    )
    feedback = {
        f"H{number:02d}": [
            c["id"] for c in cases if f"H{number:02d}" in c["metadata"]["feedback_ids"]
        ]
        for number in range(1, 11)
    }
    write(output / "feedback-coverage.json", feedback)
    recipe = read(output / "recipe.json")
    write(
        output / "support-limits.json",
        {
            "content_limits": recipe["gaps"],
            "episode_limits": {
                e["id"]: e.get("remaining", [])
                for e in recipe["episodes"]
                if e.get("remaining")
            },
            "natural_audio": f"{counts['unique_reviewed_operations']} reviewed operations on original recordings; natural backchannel removal, abandoned-start removal and complete silence removal remain unqualified. Symbolic references do not fill these acoustic gaps.",
            "reference_scope": "AI-authored editing goals and interaction outcomes; no natural user preference/prevalence claim.",
            "runtime_scope": "Offline/PredictionSession serialization parity and pinned HF tokenizer views are verified. Actual GGUF/server chat-template parity remains a runtime preflight before B1/L0 predictions.",
            "test_scope": "The protocol is frozen; final-test bodies and answers remain unopened and ungenerated until E1.",
        },
    )
    by_action = {}
    for action in ACTIONS:
        selected = [c for c in cases if c["expectedOutput"]["action"] == action]
        by_action[action] = {
            "case_ids": [c["id"] for c in selected],
            "language_split_counts": dict(
                Counter(
                    f"{c['metadata']['language']}:{c['metadata']['split']}"
                    for c in selected
                )
            ),
            "natural_recording_cases": [
                c["id"] for c in selected if c["metadata"]["media_kind"] == "recording"
            ],
            "symbolic_cases": [
                c["id"]
                for c in selected
                if c["metadata"]["media_kind"] == "authored_symbolic_timeline"
            ],
        }
    write(output / "action-coverage.json", by_action)
    inventory = read(output / "source-inventory.json")
    for source in [
        *inventory["current_sources"],
        *inventory.get("authored_sources", []),
    ]:
        group = source.get("source_group_id", source.get("id"))
        admitted = [c["id"] for c in cases if c["metadata"]["source_group_id"] == group]
        source.update(
            release_admission_status="frozen_supported_scope",
            reference_status="listed_references_independently_validated_and_frozen",
            source_group_id=group,
            draft_case_ids=admitted,
            admitted_case_ids=admitted,
        )
    recipe_path = old_manifest["code_and_recipe_sha256"]
    for fixture in inventory["fixtures"]:
        fixture["release_status"] = "role_reviewed_at_D4"
        if fixture["path"] in recipe_path and fixture["path"].endswith(
            "dataset-release-v3.json"
        ):
            fixture["admitted_case_ids"] = [c["id"] for c in cases]
            fixture["role"] = (
                "Current v4 release recipe with reviewed recordings and explicitly symbolic scripts."
            )
        elif fixture["path"].endswith("source-qualification-v1.json"):
            fixture["admitted_case_ids"] = [
                c["id"] for c in cases if c["metadata"]["retained_from"]
            ]
    (output / "source-inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n"
    )
    protocol.update(
        status="D4_design_frozen_E1_binding_pending", bound_release_id=release_id
    )
    (output / "test-protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    audit = read(output / "audit.json")
    audit.update(
        semantic_audit="passed_independent_full_census",
        training_eligible=True,
        independent_audit_sha256=digest(admission_path),
        D4_complete=True,
        supported_scope_only=True,
        final_test_cases=0,
        deterministic_draft_verification=verification,
    )
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    subprocess.run(
        ["node", "ml/verify_source_caption_export.cjs", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    export_report = read(output / "caption-export-check.json")
    staging_prefix = str(output.relative_to(ROOT)) + "/"
    final_prefix = str(destination.relative_to(ROOT)) + "/"
    export_report["inputs_sha256"] = {
        final_prefix + name[len(staging_prefix) :]
        if name.startswith(staging_prefix)
        else name: value
        for name, value in export_report["inputs_sha256"].items()
    }
    (output / "caption-export-check.json").write_text(
        json.dumps(export_report, ensure_ascii=False, indent=2) + "\n"
    )
    manifest = {
        **old_manifest,
        **counts,
        "release_id": release_id,
        "status": "frozen_supported_scope",
        "remaining_gates": [],
        "D4_complete": True,
        "coverage_complete": False,
        "draft_path": str(directory.relative_to(ROOT)),
        "draft_manifest_sha256": digest(directory / "manifest.json"),
        "freeze_tool_sha256": digest(Path(__file__)),
        "independent_review_sha256": digest(admission_path),
        "final_test_generated_cases": 0,
        "planned_final_test_request_slots": protocol["allocation_interpretation"][
            "planned_request_slots"
        ],
        "training_status": "not_trained_on_this_release",
        "sampling_policy": read(output / "diversity-audit.json")["weight_policy"],
    }
    manifest["code_and_recipe_sha256"] = {
        **manifest["code_and_recipe_sha256"],
        str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__)),
    }
    manifest["files_sha256"] = {
        str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != output / "manifest.json"
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    frozen_verification = verify(output, tokenizer)
    if destination.exists():
        raise ValueError(
            "A frozen release appeared during verification; it will not be overwritten."
        )
    output.rename(destination)
    return {
        "release_id": release_id,
        "splits": manifest["splits"],
        "D4_complete": True,
        "coverage_complete": False,
        "manifest_sha256": digest(destination / "manifest.json"),
        "verification": frozen_verification,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    print(
        json.dumps(
            freeze(
                args.directory.resolve(),
                args.output.resolve(),
                args.admission.resolve(),
                [p.resolve() for p in args.evidence],
                tokenizer,
            ),
            indent=2,
        )
    )
