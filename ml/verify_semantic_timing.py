"""Check executable semantic distractors and render cuts from saved editor states."""

import argparse
from collections import defaultdict
from pathlib import Path

from audio_review import RATE, load_pcm
from dataset_release_inventory import ROOT, digest, rows, write
from grading import grade
from replay_session import session_hash
from reviewed_audio import CUTS, verify_media_edit
from session_cases import apply_edit
from source_qualification import media_intervals


def render_intervals(samples, origin, source_hash, intervals):
    import numpy as np

    parts = []
    for source, start, length in intervals:
        if source != source_hash:
            raise ValueError("Unexpected source in the reviewed audio state.")
        left = max(0, round((start - origin) * RATE))
        right = min(len(samples), round((start + length - origin) * RATE))
        if left < right:
            parts.append(samples[left:right])
    return np.concatenate(parts) if parts else np.empty(0, dtype=samples.dtype)


def verify(directory):
    import numpy as np

    cases = rows(directory / "train-cases.jsonl") + rows(directory / "dev-cases.jsonl")
    executions = {e["id"]: e for e in rows(directory / "executions.jsonl")}
    rendered, distractors, contrasts = [], [], defaultdict(list)
    for case in cases:
        meta = case["metadata"]
        execution = executions[case["id"]]
        before = execution["before"]
        bundle = {
            "input": case["input"],
            "mapping": execution["mapping"],
            "document_hash": session_hash(before),
            **(
                {"caption_authority": execution["caption_authority"]}
                if "caption_authority" in execution
                else {}
            ),
        }
        for action in [case["expectedOutput"], *case["acceptableOutputs"]]:
            if (
                action["action"] not in CUTS
                or meta.get("media_kind") == "authored_symbolic_timeline"
            ):
                continue
            after = apply_edit(before, bundle, action)
            proof = verify_media_edit(
                before,
                after,
                bundle["mapping"],
                action,
                execution["audio_qualifications"],
            )
            folder = ROOT / proof["render_directory"]
            original = load_pcm(folder / "before.wav")
            approved = load_pcm(folder / "after.wav")
            original_state = render_intervals(
                original,
                proof["preview_origin_seconds"],
                proof["source_sha256"],
                media_intervals(before),
            )
            executed_state = render_intervals(
                original,
                proof["preview_origin_seconds"],
                proof["source_sha256"],
                media_intervals(after),
            )
            if not np.array_equal(original_state, original) or not np.array_equal(
                executed_state, approved
            ):
                raise ValueError(
                    "Executed editor source intervals do not reproduce the listened render."
                )
            rendered.append(
                {
                    "id": case["id"],
                    "qualification_id": proof["id"],
                    "render_sha256": digest(folder / "after.wav"),
                    "sample_exact_editor_render": True,
                    "removed_samples": len(original) - len(approved),
                }
            )
        for row in meta.get("counterfactual_audit", []):
            action = row["prediction"]
            apply_edit(before, bundle, action)
            if grade(case, action, execution)["outcome_match"]:
                raise ValueError("Counterfactual is not a negative under this brief.")
            distractors.append(
                {
                    "id": case["id"],
                    "action": action["action"],
                    "eligible": True,
                    "audio_qualification_id": row["audio_qualification_id"],
                    "scope": "Semantic distinction is author-audited; this check proves eligibility, not meaning.",
                }
            )
        if meta.get("semantic_contrast_id"):
            contrasts[meta["semantic_contrast_id"]].append(case)
    paired = []
    for identity, pair in contrasts.items():
        if len(pair) != 2 or {c["expectedOutput"]["action"] for c in pair} != {
            "no_action",
            "trim_silence",
        }:
            raise ValueError(
                "Pacing contrast must contain one keep and one shorten judgment."
            )
        if (
            len(
                {
                    (
                        c["metadata"]["source_group_id"],
                        c["metadata"]["split"],
                        c["metadata"]["focus_uuid"],
                    )
                    for c in pair
                }
            )
            != 1
        ):
            raise ValueError("Matched contrast differs in source, split or cursor.")
        cut = next(c for c in pair if c["expectedOutput"]["action"] == "trim_silence")
        keep = next(c for c in pair if c["expectedOutput"]["action"] == "no_action")
        if cut["expectedOutput"] not in [
            c["prediction"] for c in keep["metadata"]["counterfactual_audit"]
        ]:
            raise ValueError(
                "Preservation case does not challenge the same eligible cut."
            )
        left, right = [{i["id"]: i for i in c["input"]["items"]} for c in pair]
        common = left.keys() & right.keys()
        if any(left[i] != right[i] for i in common):
            raise ValueError("Contrast leaks a label through changed item evidence.")
        paired.append(
            {
                "id": identity,
                "cases": [c["id"] for c in pair],
                "same_items": left == right,
                "shared_item_count": len(common),
                "item_counts": [len(left), len(right)],
                "same_source_cursor_and_eligible_cut": True,
                "interpretation": "Correlated editorial-purpose contrast, not two independent sources.",
            }
        )
    semantic = [c for c in cases if c["id"].startswith("semantic-")]
    baselines = {}
    for name in ["always_abstain", "trim_verified_pauses_over_1200_to_900"]:
        predictions = []
        for case in semantic:
            prediction = {"action": "no_action"}
            if name != "always_abstain":
                gaps = [
                    i
                    for i in case["input"]["items"]
                    if i["type"] == "non_text"
                    and i["boundary_status"] == "verified"
                    and i["audio_class"] == "verified_silence"
                    and 1200 < i["duration_ms"] <= 5000
                ]
                if gaps:
                    gap = min(
                        gaps,
                        key=lambda i: abs(i["id"] - case["input"]["cursor_before"]),
                    )
                    prediction = {
                        "action": "trim_silence",
                        "target": [gap["id"]],
                        "keep_ms": 900,
                    }
            predictions.append(
                {
                    "id": case["id"],
                    "prediction": prediction,
                    "match": grade(case, prediction, executions[case["id"]])[
                        "outcome_match"
                    ],
                }
            )
        baselines[name] = {
            "matches": sum(r["match"] for r in predictions),
            "cases": len(predictions),
            "predictions": predictions,
        }
    return {
        "audit_kind": "semantic_timing",
        "status": "passed",
        "manifest_sha256": digest(directory / "manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "rendered_references": rendered,
        "eligible_distractors": distractors,
        "paired_contrasts": paired,
        "input_only_rule_baselines": baselines,
        "human_listening_scope": "Four exact original operations; synthetic briefs, labels and interaction outcomes are not human preference data.",
        "training_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.directory.resolve())
    write(args.output.resolve(), result)
    print(
        {
            "status": result["status"],
            "rendered_references": len(result["rendered_references"]),
            "eligible_distractors": len(result["eligible_distractors"]),
            "paired_contrasts": result["paired_contrasts"],
        }
    )
