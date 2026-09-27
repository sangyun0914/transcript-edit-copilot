"""Audit archived train/dev diversity without changing cases or opening final tests."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEST_FAMILIES = {
    "name_consistency": {
        "history_supported",
        "history_absent",
        "history_conflicting",
        "clean",
        "protected",
        "dismissed_same",
        "dismissed_other",
        "changed_evidence",
        "expired",
        "invalidated",
        "accepted_then_undone",
        "modified_acceptance",
        "future_rejection",
        "two_opportunities",
    },
    "obvious_text_error": {"clear_error"},
    "silence_shortening": {"trim_history", "trim_dismissed", "trim_modified"},
    "silence_removal": {"remove_pause"},
    "caption_split": {"split"},
    "caption_merge": {"merge"},
    "immediate_word_repetition": {"delete_repeat"},
}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def value_hash(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_sha256"]].append(row["case_id"])
    return {
        "rows": len(rows),
        "unique_input_target_pairs": len(pairs),
        "unique_serialized_message_target_pairs": len(
            {r["messages_sha256"] for r in rows}
        ),
        "source_groups": sorted({r["group"] for r in rows}),
        "scenario_variants": dict(sorted(Counter(r["variant"] for r in rows).items())),
        "primary_actions": dict(sorted(Counter(r["action"] for r in rows).items())),
        "duplicate_clusters": [ids for ids in pairs.values() if len(ids) > 1],
    }


def verify_sources(root):
    folder = root / "data/derived/nep/source-windows/iteration-03"
    manifest = json.loads((folder / "manifest.json").read_text())
    windows = read_rows(folder / "windows.jsonl")
    checked = {}

    def verify(path, expected):
        relative = str(path.relative_to(root))
        if relative not in checked:
            checked[relative] = file_hash(path)
        if checked[relative] != expected:
            raise ValueError(f"Source hash mismatch: {relative}")

    verify(folder / "windows.jsonl", manifest["windows_sha256"])
    for window in windows:
        if window["reserved_split"] not in {"train", "dev"}:
            raise ValueError("Audit must not read reserved final-test material.")
        verify(folder / window["audio_preview"], window["audio_sha256"])
        verify(root / window["source_video"], window["video_sha256"])
    acquisition = json.loads((root / "data/raw/ocw/acquisition.json").read_text())
    ocw = []
    for source in acquisition["sources"]:
        if source["split"] not in {"train", "dev"}:
            raise ValueError("Unexpected reserved source in acquisition manifest.")
        for name in ("video.mp4", "captions.vtt"):
            verify(
                root / "data/raw/ocw" / source["id"] / name,
                source["files"][name]["sha256"],
            )
        ocw.append(
            {
                "id": source["id"],
                "split": source["split"],
                "duration_seconds": float(source["media"]["format"]["duration"]),
                "page_url": source["page_url"],
            }
        )
    return {
        "windows": len(windows),
        "windows_per_group": dict(Counter(w["source_group_id"] for w in windows)),
        "references": sum(w["expectedOutput"] is not None for w in windows),
        "training_eligible": sum(w["training_eligible"] for w in windows),
        "ocw_acquired": ocw,
        "verified_files": checked,
        "scope": "File integrity and admission metadata; no listening or acoustic qualification.",
    }


def audit(root):
    accumulated = {"train": [], "dev": []}
    runs, inputs, latest_cases = [], {}, {}
    for number in range(1, 6):
        run_id = f"colab-l4-run-{number:02}"
        bundle = root / "artifacts/iterations" / run_id / "input-bundle"
        prompt_path = bundle / "prompt.txt"
        prompt = prompt_path.read_text()
        inputs[str(prompt_path.relative_to(root))] = file_hash(prompt_path)
        run = {"run": run_id}
        for split in accumulated:
            path = bundle / f"{split}-cases.jsonl"
            cases = read_rows(path)
            inputs[str(path.relative_to(root))] = file_hash(path)
            rows = []
            for case in cases:
                metadata = case["metadata"]
                if metadata["split"] != split:
                    raise ValueError(f"Unexpected split: {case['id']}")
                messages = [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            case["input"], ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            case["expectedOutput"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ]
                rows.append(
                    {
                        "run": run_id,
                        "case_id": case["id"],
                        "group": metadata["source_group_id"],
                        "variant": metadata["scenario_variant"],
                        "action": case["expectedOutput"]["action"],
                        "input_sha256": value_hash(case["input"]),
                        "target_sha256": value_hash(case["expectedOutput"]),
                        "pair_sha256": value_hash(
                            [case["input"], case["expectedOutput"]]
                        ),
                        "messages_sha256": value_hash(messages),
                        "prompt_input_sha256": value_hash(messages[:2]),
                        "acceptable_outputs_sha256": value_hash(
                            case.get("acceptableOutputs", [])
                        ),
                    }
                )
            previous = {r["pair_sha256"] for r in accumulated[split]}
            run[split] = summarize(rows)
            run[split]["new_input_target_pairs"] = len(
                {r["pair_sha256"] for r in rows} - previous
            )
            accumulated[split].extend(rows)
            if number == 5:
                latest_cases[split] = rows
        runs.append(run)

    latest_families = {}
    seen_variants = set()
    for family, variants in LATEST_FAMILIES.items():
        subset = [r for r in latest_cases["train"] if r["variant"] in variants]
        latest_families[family] = summarize(subset)
        seen_variants |= {r["variant"] for r in subset}
    if seen_variants != {r["variant"] for r in latest_cases["train"]}:
        raise ValueError("Update the explicit content-family audit mapping.")

    kept, excluded, seen = [], [], {}
    for row in latest_cases["train"]:
        signature = row["messages_sha256"]
        if signature in seen:
            if (
                row["acceptable_outputs_sha256"]
                != seen[signature]["acceptable_outputs_sha256"]
            ):
                raise ValueError(
                    "Duplicate SFT pair has a different grading-reference set."
                )
            excluded.append(
                {
                    "case_id": row["case_id"],
                    "same_training_messages_as": seen[signature]["case_id"],
                }
            )
        else:
            seen[signature] = row
            kept.append(row["case_id"])

    conflicts = []
    for split, rows in accumulated.items():
        by_input = defaultdict(list)
        for row in rows:
            by_input[row["prompt_input_sha256"]].append(row)
        for same in by_input.values():
            if len({r["target_sha256"] for r in same}) > 1:
                conflicts.append(
                    {"split": split, "case_ids": [r["case_id"] for r in same]}
                )
    cross_inputs = {r["input_sha256"] for r in accumulated["train"]} & {
        r["input_sha256"] for r in accumulated["dev"]
    }
    cross_groups = {r["group"] for r in accumulated["train"]} & {
        r["group"] for r in accumulated["dev"]
    }
    return {
        "audit_version": 1,
        "scope": "Saved Qwen runs 01–05 train/dev only; regression and reserved final test excluded.",
        "pair_definition": "Canonical JSON input + primary expectedOutput, excluding metadata/IDs. Also check exact serialized system/user/assistant messages before tokenization.",
        "limitations": [
            "Different pairs or declared groups are not proof of independent editing use cases.",
            "Content-family mapping is an explicit editorial taxonomy, not an automatic semantic equivalence metric.",
            "No perceptual audio, translation, or pretraining-contamination audit is claimed.",
            "Existing predictions and row-weighted experiment scores remain unchanged.",
        ],
        "inputs_sha256": inputs,
        "runs": runs,
        "cumulative": {split: summarize(rows) for split, rows in accumulated.items()},
        "identical_train_dev_inputs": len(cross_inputs),
        "shared_train_dev_source_groups": sorted(cross_groups),
        "conflicting_primary_labels_for_identical_messages": conflicts,
        "latest_content_families": latest_families,
        "latest_train_unique_selection": {
            "status": "Audit-only selection, not a new dataset or training run; temporal regressions remain in archives.",
            "keep_case_ids": kept,
            "redundant_for_sft": excluded,
        },
        "source_material": verify_sources(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/audits/2026-09-25-diversity/report.json",
    )
    args = parser.parse_args()
    report = audit(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "train_unique": report["cumulative"]["train"][
                    "unique_input_target_pairs"
                ],
                "dev_unique": report["cumulative"]["dev"]["unique_input_target_pairs"],
                "latest_train_unique": report["runs"][-1]["train"][
                    "unique_input_target_pairs"
                ],
                "verified_source_files": len(
                    report["source_material"]["verified_files"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
