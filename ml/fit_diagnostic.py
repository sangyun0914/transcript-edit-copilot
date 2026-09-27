"""Prepare a correction-fit subset without relabeling development or source groups."""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate import read_rows
from iteration import HERE, jsonl, messages, sha


def verified_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f"Source manifest mismatch: {relative}")
    return manifest


def build(recipe_path, source, preferences, output):
    recipe = json.loads(recipe_path.read_text())
    if sha(source / "manifest.json") != recipe["source_manifest_sha256"]:
        raise ValueError("The frozen training/development source changed.")
    source_manifest = verified_manifest(source)
    preference_manifest = verified_manifest(preferences)
    if preference_manifest["recipe_sha256"] != recipe["preference_recipe_sha256"]:
        raise ValueError("The preference diagnostic recipe changed.")
    all_train = read_rows(source / "train-cases.jsonl")
    requested = {
        (g, s) for g in recipe["train_groups"] for s in recipe["train_scenarios"]
    }
    train = [
        c
        for c in all_train
        if (c["metadata"]["source_group_id"], c["metadata"]["scenario_variant"])
        in requested
    ]
    actual = [
        (c["metadata"]["source_group_id"], c["metadata"]["scenario_variant"])
        for c in train
    ]
    if set(actual) != requested or len(actual) != len(requested):
        raise ValueError(
            "Every requested training scenario must exist exactly once in training."
        )
    original_dev = read_rows(source / "dev-cases.jsonl")
    preference_cases = read_rows(preferences / "dev-cases.jsonl")
    dev = original_dev + preference_cases
    groups = {}
    for split, cases in (("train", train), ("dev", dev)):
        for case in cases:
            metadata = case["metadata"]
            group = metadata["source_group_id"]
            if (
                metadata["split"] != split
                or metadata["training_eligible"] is not (split == "train")
                or group in source_manifest["reserved_test_groups"]
                or groups.get(group, split) != split
            ):
                raise ValueError(
                    "Diagnostic selection would cross a split or reserved group."
                )
            groups[group] = split
    if set(preference_manifest["groups"]) & set(source_manifest["groups"]):
        raise ValueError("Preference scripts must not reuse earlier source groups.")
    if len({c["id"] for c in train + dev}) != len(train + dev):
        raise ValueError("Diagnostic case IDs must be unique.")
    prompt = (source / "prompt.txt").read_text()
    if (
        prompt != (preferences / "prompt.txt").read_text()
        or prompt != (HERE / "prompt-v2.txt").read_text()
    ):
        raise ValueError("Keep the same canonical prompt for this fit comparison.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "sessions").mkdir()
    (output / "recipe.json").write_bytes(recipe_path.read_bytes())
    (output / "prompt.txt").write_text(prompt)
    (output / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    executions = []
    for root, cases in (
        (source, train + original_dev),
        (preferences, preference_cases),
    ):
        ids = {c["id"] for c in cases}
        source_executions = [
            r for r in read_rows(root / "executions.jsonl") if r["id"] in ids
        ]
        if (
            len(source_executions) != len(ids)
            or {r["id"] for r in source_executions} != ids
        ):
            raise ValueError("Every diagnostic case needs one execution fixture.")
        executions.extend(source_executions)
        for case in cases:
            metadata = case["metadata"]
            paths = [
                metadata["session"],
                metadata.get("history_session"),
                *metadata.get("history_sessions", []),
            ]
            for relative in filter(None, paths):
                original, target = (
                    (root / relative).resolve(),
                    (output / relative).resolve(),
                )
                if not original.is_relative_to(
                    (root / "sessions").resolve()
                ) or not target.is_relative_to((output / "sessions").resolve()):
                    raise ValueError("Journal path escapes the session directory.")
                if target.exists() and target.read_bytes() != original.read_bytes():
                    raise ValueError("Journal path collision.")
                target.write_bytes(original.read_bytes())
    jsonl(output / "executions.jsonl", executions)
    jsonl(output / "preference-cases.jsonl", preference_cases)
    for split, cases in (("train", train), ("dev", dev)):
        jsonl(output / f"{split}-cases.jsonl", cases)
        jsonl(output / f"{split}.jsonl", [messages(c, prompt) for c in cases])
    manifest = {
        "generator": "correction-fit-v1",
        "generator_sha256": sha(Path(__file__)),
        "recipe_sha256": sha(recipe_path),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "preference_manifest_sha256": sha(preferences / "manifest.json"),
        "status": "prepared_not_trained",
        "question": recipe["question"],
        "scope": recipe["scope"],
        "groups": groups,
        "reserved_test_groups": source_manifest["reserved_test_groups"],
        "counts": {"train": len(train), "dev": len(dev)},
        "development_cohorts": {
            "retained_iteration03": len(original_dev),
            "pause_preferences": len(preference_cases),
        },
        "actions_by_split": {
            split: dict(Counter(c["expectedOutput"]["action"] for c in cases))
            for split, cases in (("train", train), ("dev", dev))
        },
        "training_exposure": {
            "steps": recipe["steps"],
            "batch_size": 1,
            "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
            "nominal_example_presentations": recipe["steps"]
            * recipe["gradient_accumulation_steps"],
            "nominal_epochs": recipe["steps"]
            * recipe["gradient_accumulation_steps"]
            / len(train),
        },
        "files": {
            p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preferences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.recipe, args.source, args.preferences, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
