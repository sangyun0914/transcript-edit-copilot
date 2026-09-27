"""Recheck a frozen source collection and its actual serialized training inputs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_dataset_diversity import value_hash, verify_sources
from context_policy import ContextBuilder
from grading import grade
from model_input import messages, token_ids
from replay_session import replay, session_hash
from session_cases import apply_edit
from source_qualification import (
    ROOT,
    TOKENIZER,
    digest,
    focus_at,
    media_intervals,
    write,
)
from train_adapter import training_row


def verify(directory, tokenizer):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"Frozen file mismatch: {name}")
    for name, expected in manifest["inputs_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Generation input changed: {name}")
    export = json.loads((directory / "caption-export-check.json").read_text())
    if export["status"] != "passed":
        raise ValueError("Actual caption export check is required.")
    for name, expected in export["inputs_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Caption export check is stale.")

    def rows(name):
        return [json.loads(s) for s in (directory / name).read_text().splitlines()]

    cases = rows("train-cases.jsonl") + rows("dev-cases.jsonl")
    executions = {r["id"]: r for r in rows("executions.jsonl")}
    contexts = {r["case_id"]: r for r in rows("contexts.jsonl")}
    recipes = {
        r["id"]: r for r in json.loads((directory / "recipe.json").read_text())["cases"]
    }
    prompt = (directory / "prompt.txt").read_text()
    builder = ContextBuilder(tokenizer, prompt)
    pairs, groups, negative_checks, semantic_checks = set(), {}, [], []
    for case in cases:
        meta = case["metadata"]
        group, split = meta["source_group_id"], meta["split"]
        if groups.setdefault(group, split) != split or split not in {"train", "dev"}:
            raise ValueError("Source group crosses splits.")
        if meta["training_eligible"] != (split == "train"):
            raise ValueError("Incorrect training eligibility.")
        state = executions[case["id"]]
        document = state["before"]
        timed = [i for i in document["content"] if "sourceStart" in i]
        focus = focus_at(timed, recipes[case["id"]]["focus_seconds"])
        bundle = builder.build(
            document, focus["uuid"], case["input"]["task_brief"], as_of=0
        )
        if bundle["input"] != case["input"] or bundle["mapping"] != state["mapping"]:
            raise ValueError(
                "Runtime input cannot be reproduced from the full document."
            )
        expected = case["expectedOutput"]
        for edit in [expected, *case["acceptableOutputs"]]:
            after = apply_edit(document, bundle, edit)
            if media_intervals(document) != media_intervals(after):
                raise ValueError("Source intervals changed.")
            if not grade(case, edit, state)["outcome_match"]:
                raise ValueError("An accepted reference fails the actual grader.")
        events = rows(meta["session"])
        after, _ = replay(events)
        if session_hash(after) != state["after_hash"]:
            raise ValueError("Replay hash mismatch.")
        context = contexts[case["id"]]
        prefix = token_ids(tokenizer, case["input"], prompt)
        if context["input_ids"] != prefix or context["messages"] != messages(
            case["input"], prompt
        ):
            raise ValueError("Stored context differs from model input.")
        if context["token_count"] != len(prefix) or len(prefix) + 256 > 4096:
            raise ValueError("Context budget mismatch.")
        if split == "train" and context["training_sequence"] != training_row(
            tokenizer, case, prompt
        ):
            raise ValueError("Training sequence differs from training code.")
        key = session_hash({"input": case["input"], "target": expected})
        if key in pairs:
            raise ValueError("Duplicate input/reference.")
        pairs.add(key)
        if case["id"] == "source-ko-hesitation-text-only":
            for bad in [
                {"action": "delete_span", "target": expected["target"]},
                {"action": "split_paragraph", "target": expected["target"]},
            ]:
                try:
                    apply_edit(document, bundle, bad)
                except ValueError as error:
                    negative_checks.append(
                        {
                            "case_id": case["id"],
                            "attempt": bad["action"],
                            "rejected_because": str(error),
                            "reference": "source-only capability probe, not an extra no_action training row",
                        }
                    )
                else:
                    raise ValueError(
                        "Unverified audio or incomplete-paragraph operation was accepted."
                    )
        bad = None
        if case["id"] == "source-ko-negative-weight-display":
            bad = {**expected, "text": "W 1.5는"}
        elif case["id"] == "source-ko-agreement-turn":
            item = next(
                i for i in case["input"]["items"] if i["text"] == "맞아요. 맞아요."
            )
            bad = {"action": "correct_text", "target": [item["id"]], "text": "맞아요."}
        if bad:
            apply_edit(document, bundle, bad)
            if grade(case, bad, state)["outcome_match"]:
                raise ValueError(
                    "A meaning/purpose-changing edit passed the reference grader."
                )
            semantic_checks.append(
                {"case_id": case["id"], "valid_but_unacceptable": bad}
            )
        if (
            case["id"] == "source-en-caption-continuation"
            and case["input"]["cursor_before"] != expected["target"][0]
        ):
            raise ValueError("A cue-boundary cursor selected the preceding cue.")
    historical = {"train": [], "dev": []}
    archive_hashes = {}
    for number in range(1, 6):
        for split in historical:
            path = (
                ROOT
                / f"artifacts/iterations/colab-l4-run-{number:02d}/input-bundle/{split}-cases.jsonl"
            )
            archive_hashes[str(path.relative_to(ROOT))] = digest(path)
            historical[split].extend(
                json.loads(s) for s in path.read_text().splitlines()
            )
    counts = {}
    for split, old in historical.items():
        current = [c for c in cases if c["metadata"]["split"] == split]
        other = historical["dev" if split == "train" else "train"]
        if {c["metadata"]["source_group_id"] for c in current} & {
            c["metadata"]["source_group_id"] for c in other
        }:
            raise ValueError(
                "New source group appears in the opposite historical split."
            )
        if {value_hash(c["input"]) for c in current} & {
            value_hash(c["input"]) for c in other
        }:
            raise ValueError("New input appears in the opposite historical split.")
        old_pairs = {value_hash([c["input"], c["expectedOutput"]]) for c in old}
        new_pairs = {value_hash([c["input"], c["expectedOutput"]]) for c in current}
        counts[split] = {
            "historical_unique": len(old_pairs),
            "new_unique": len(new_pairs - old_pairs),
            "prepared_union_unique": len(old_pairs | new_pairs),
        }
    source_hash_check = verify_sources(ROOT)
    report = {
        "status": "passed",
        "cases": len(cases),
        "unique_input_target_pairs": len(pairs),
        "splits": dict(Counter(c["metadata"]["split"] for c in cases)),
        "source_groups": groups,
        "caption_export": "passed_actual_app_exporter",
        "negative_capability_checks": negative_checks,
        "semantic_controls": semantic_checks,
        "historical_overlap_check": counts,
        "historical_input_sha256": archive_hashes,
        "original_source_files_reverified": len(source_hash_check["verified_files"]),
        "checks": [
            "frozen archive hashes",
            "full-document cursor reconstruction",
            "training/inference exact serialization",
            "reference and alternative grading",
            "source media interval preservation",
            "apply/undo/redo replay",
            "cross-split source exclusion",
            "actual caption export",
        ],
        "model_evaluation_run": False,
        "test_contents_accessed": False,
    }
    write(directory / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=ROOT / "artifacts/iterations/source-qualification-v1",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    print(
        json.dumps(
            verify(args.directory.resolve(), tokenizer), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
