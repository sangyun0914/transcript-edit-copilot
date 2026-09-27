"""Rebuild passive candidates and audit replay, serving parity and actual caption exports."""

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from passive_contract import (
    PassiveContextBuilder,
    PassivePredictionSession,
    apply_passive_edit,
    canonical_messages,
    committed_transition,
    parse_snapshot,
    wire_snapshot,
)
from passive_dataset import (
    ROOT,
    RESERVED,
    apply_corruptions,
    digest,
    episode_steps,
    load_sources,
    read,
    write,
)
from replay_session import replay, session_hash
from verify_release_export import Exporter, verify_export_delta, verify_operation


def rows(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def tokenizers(dependencies=None):
    from transformers import AutoTokenizer

    registry_path = (
        ROOT
        / "artifacts/datasets/nep-release-v1/comparison-tokenizers-v2/registry.json"
    )
    registry = read(registry_path)
    if dependencies is not None:
        dependencies[str(registry_path.relative_to(ROOT))] = digest(registry_path)
    for candidate in registry["candidates"]:
        for name, expected in candidate["files_sha256"].items():
            path = ROOT / candidate["path"] / name
            if digest(path) != expected:
                raise ValueError("Pinned tokenizer file changed.")
            if dependencies is not None:
                dependencies[str(path.relative_to(ROOT))] = expected
    return {
        c["id"]: AutoTokenizer.from_pretrained(ROOT / c["path"], local_files_only=True)
        for c in registry["candidates"]
    }


def validate_inventory(directory, registries):
    dependencies = {
        str((directory / name).resolve().relative_to(ROOT)): digest(directory / name)
        for name in (
            "train-candidates.jsonl",
            "dev-candidates.jsonl",
            "recipe.json",
            "exclusions.jsonl",
        )
    }
    sources = load_sources(registries)
    candidates = []
    for split in ("train", "dev"):
        batch = rows(directory / f"{split}-candidates.jsonl")
        if any(row["metadata"]["split"] != split for row in batch):
            raise ValueError("Candidate split differs from its file placement.")
        candidates.extend(batch)
    specs = read(directory / "recipe.json")["episodes"]
    recipes = {s["id"]: s for s in specs}
    if len(recipes) != len(specs):
        raise ValueError("Duplicate episode recipe identity.")
    exclusions = rows(directory / "exclusions.jsonl")
    excluded = {row["episode_id"] for row in exclusions}
    if len(excluded) != len(exclusions) or not excluded <= recipes.keys():
        raise ValueError("Unknown or duplicate episode exclusions.")
    by_episode = defaultdict(list)
    groups, seen_ids, input_splits = {}, set(), {}
    for case in candidates:
        meta = case["metadata"]
        if case["id"] in seen_ids:
            raise ValueError("Duplicate case identity.")
        seen_ids.add(case["id"])
        group, split = meta["source_group_id"], meta["split"]
        if group in RESERVED or split not in {"train", "dev"}:
            raise ValueError("Reserved material cannot enter passive verification.")
        source = sources.get(meta["source_id"])
        if source is None or (group, split, meta["language"]) != (
            source["source_group_id"],
            source["split"],
            source["language"],
        ):
            raise ValueError(
                "Candidate source identity differs from its registered parent."
            )
        if group in groups and groups[group] != split:
            raise ValueError("Source group crosses splits.")
        groups[group] = split
        key = session_hash(canonical_messages(case["input"]))
        if key != meta["input_sha256"]:
            raise ValueError("Cached predictor-input hash is forged or stale.")
        if key in input_splits and input_splits[key] != split:
            raise ValueError("Identical predictor input crosses splits.")
        input_splits[key] = split
        if parse_snapshot(wire_snapshot(case["input"])).model_dump() != case["input"]:
            raise ValueError("Compact wire is not a lossless representation.")
        by_episode[meta["episode_id"]].append(case)
    if set(by_episode) != recipes.keys() - excluded:
        raise ValueError("Missing, unregistered or excluded episode candidate rows.")
    saved_episodes = {}
    for registry in registries:
        path = Path(registry).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Source registry is outside this project.")
        dependencies[str(path.relative_to(ROOT))] = digest(path)
    for episode_id, cases in by_episode.items():
        relative = f"episodes/{episode_id}.json"
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or any(
            c["reproduction"]["episode"] != relative for c in cases
        ):
            raise ValueError("Episode reproduction path changed or escapes its draft.")
        saved = read(path)
        if saved["spec"] != recipes[episode_id]:
            raise ValueError("Saved episode differs from the declared recipe.")
        steps = episode_steps(saved["spec"])
        if sorted(c["metadata"]["episode_step"] for c in cases) != list(
            range(len(steps))
        ):
            raise ValueError("Missing or duplicated recipe step, including no_action.")
        if any(
            c["id"] != episode_id + f"-s{c['metadata']['episode_step']:02d}"
            or c["reproduction"]["step"] != c["metadata"]["episode_step"]
            for c in cases
        ):
            raise ValueError("Candidate identity does not match its episode step.")
        source = sources[saved["spec"]["source_id"]]
        if any(c["metadata"]["source_id"] != source["id"] for c in cases):
            raise ValueError("Case and episode disagree on source identity.")
        source_path = (ROOT / source["document_path"]).resolve()
        if (
            not source_path.is_relative_to(ROOT)
            or saved["source_document"] != source["document_path"]
        ):
            raise ValueError("Reproduction changed its registered source document.")
        source_hash = digest(source_path)
        if source_hash != saved["source_document_sha256"]:
            raise ValueError("Reproduction source document changed.")
        dependencies[str(path.relative_to(ROOT))] = digest(path)
        dependencies[str(source_path.relative_to(ROOT))] = source_hash
        saved_episodes[episode_id] = saved
    return candidates, by_episode, saved_episodes, dependencies


def verify(directory, output, registries):
    candidates, by_episode, saved_episodes, dependencies = validate_inventory(
        directory, registries
    )
    old_manifest_path = ROOT / "artifacts/datasets/nep-release-v1/frozen/manifest.json"
    old_manifest = read(old_manifest_path)
    dependencies[str(old_manifest_path.relative_to(ROOT))] = digest(old_manifest_path)
    for name, expected in old_manifest["code_and_recipe_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Pinned executor/export dependency changed: {name}")
        dependencies[name] = expected
    loaded = tokenizers(dependencies)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(ROOT):
            path = Path(filename).resolve()
            dependencies[str(path.relative_to(ROOT))] = digest(path)
    for name in (
        "ml/passive_contract.py",
        "ml/prompt-passive-v1.txt",
        "ml/passive_dataset.py",
        "ml/verify_passive_dataset.py",
        "ml/passive_training.py",
    ):
        dependencies[name] = digest(ROOT / name)
    builder = PassiveContextBuilder(loaded["qwen06"], budget_tokenizers=loaded)
    exporter = Exporter()
    checks, errors, parity = [], [], 0
    try:
        for episode_id, cases in by_episode.items():
            saved = saved_episodes[episode_id]
            source = ROOT / saved["source_document"]
            if digest(source) != saved["source_document_sha256"]:
                raise ValueError("Reconstruction source changed.")
            document = apply_corruptions(
                read(source), saved["spec"].get("corruptions", [])
            )
            initial = copy.deepcopy(document)
            steps = episode_steps(saved["spec"])
            session = PassivePredictionSession(episode_id, builder)
            history = []
            for case in sorted(cases, key=lambda c: c["metadata"]["episode_step"]):
                step = case["metadata"]["episode_step"]
                try:
                    bundle = session.prepare(
                        document,
                        steps[step]["focus_uuid"],
                        as_of=step,
                        history=history,
                        caption_profile=saved["spec"].get("caption_profile"),
                    )
                    if canonical_messages(bundle["input"]) != canonical_messages(
                        case["input"]
                    ):
                        raise ValueError(
                            "Serving preparation differs from saved training context."
                        )
                    if (
                        session_hash(document)
                        != case["metadata"]["document_before_hash"]
                    ):
                        raise ValueError("Before-state hash changed.")
                    parity += 1
                    line_length = bundle["input"]["caption_profile"][
                        "export_line_length"
                    ]
                    before_export = exporter.export(document, line_length)
                    for index, action in enumerate(
                        [case["expectedOutput"], *case["acceptableOutputs"]]
                    ):
                        after = apply_passive_edit(document, bundle, action)
                        operation = verify_operation(
                            document, after, bundle["mapping"], action
                        )
                        export_audit = verify_export_delta(
                            before_export,
                            exporter.export(after, line_length),
                            operation,
                            action["action"],
                        )
                        checks.append(
                            {
                                "case_id": case["id"],
                                "reference_index": index,
                                "status": "passed",
                                **export_audit,
                            }
                        )
                    after = apply_passive_edit(document, bundle, case["expectedOutput"])
                    if session_hash(after) != case["metadata"]["document_after_hash"]:
                        raise ValueError("After-state hash changed.")
                    if document != after:
                        history.append(
                            committed_transition(
                                document, after, case["expectedOutput"], step + 1
                            )
                        )
                    document = after
                except (ValueError, AssertionError) as error:
                    errors.append({"case_id": case["id"], "error": str(error)})
                    break
            journal = copy.deepcopy(saved["journal"])
            journal[0]["document"] = initial
            replayed, summary = replay(journal)
            if replayed != document or not summary["complete"]:
                errors.append(
                    {
                        "episode_id": episode_id,
                        "error": "Rehydrated whole episode does not reproduce all primary outcomes.",
                    }
                )
    finally:
        exporter.close()
    for name, expected in dependencies.items():
        if digest(ROOT / name) != expected:
            raise ValueError(
                f"A verification dependency changed during the audit: {name}"
            )
    result = {
        "status": "passed" if not errors else "failed",
        "directory": str(directory),
        "candidate_hashes": {
            name: digest(directory / name)
            for name in (
                "train-candidates.jsonl",
                "dev-candidates.jsonl",
                "recipe.json",
            )
        },
        "code_hashes": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                ROOT / "ml/passive_contract.py",
                ROOT / "ml/prompt-passive-v1.txt",
                ROOT / "ml/passive_dataset.py",
                ROOT / "ml/verify_passive_dataset.py",
                ROOT / "ml/passive_training.py",
            ]
        },
        "dependencies_sha256": dependencies,
        "candidates": len(candidates),
        "actual_serving_message_parity_checks": parity,
        "replayed_episodes": len(by_episode),
        "accepted_outcomes_exported": len(checks),
        "primary_actions": dict(
            Counter(c["expectedOutput"]["action"] for c in candidates)
        ),
        "errors": errors,
        "outcomes": checks,
        "semantic_admission": "separate_independent_review_required",
        "audio_listening": False,
    }
    write(output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "outcomes",
                    "code_hashes",
                    "candidate_hashes",
                    "dependencies_sha256",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    verify(args.directory, args.output, args.sources)
