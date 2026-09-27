"""Generate and score matched, replayable pause-preference development cases."""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate import evaluate, read_rows
from grading import load_executions
from iteration import HERE, jsonl, sha
from replay_session import session_hash
from session_cases import apply_edit, replayable_session, snapshot_for
from synthetic_pilot import document_for, execute_history, symbolic_evidence


def scenarios(group):
    english = group["language"] == "en"
    brief = (
        "Review pauses in this passage. Only shorten verified pauses lasting at least 3 seconds; "
        "leave shorter pauses and all words unchanged. Use the retained duration established by "
        "consistent recent pause edits. If that duration is missing or conflicting, leave the "
        "passage unchanged. Explicit instructions for this passage override previous edits."
        if english
        else "이 문맥의 쉼을 검토하세요. 3초 이상인 검증된 쉼만 줄이고 더 짧은 쉼과 모든 단어는 "
        "그대로 두세요. 일관된 최근 쉼 편집에서 정한 남길 길이를 사용하세요. 남길 길이의 근거가 "
        "없거나 서로 충돌하면 그대로 두세요. 현재 문맥에 대한 명시적 지시를 이전 편집보다 우선하세요."
    )

    def history(keep):
        return [
            {
                "action": "trim_silence",
                "before_ms": before,
                "after_ms": keep,
                "origin": "synthetic",
            }
            for before in (4000, 3000)
        ]

    def make(family, edits, keep=None, duration=3500, suffix="", **extra):
        a, b = group["sentences"]
        return {
            "id": f"{group['id']}-{family}",
            "family": family,
            "language": group["language"],
            "brief": brief + suffix,
            "history": edits,
            "paragraphs": [
                [
                    {"id": "left", "text": a, "ms": 2000},
                    {"id": "pause", "text": "", "ms": duration},
                    {"id": "right", "text": b, "ms": 2000},
                ]
            ],
            "expected": {"action": "no_action"}
            if keep is None
            else {
                "action": "trim_silence",
                "target": [1],
                "keep_ms": keep,
            },
            **extra,
        }

    for keep in (500, 900, 1500):
        yield make(f"history-{keep}", history(keep), keep, matched=True)
    yield make("single-history-1500", history(1500)[-1:], 1500)
    yield make("absent-history", [])
    yield make(
        "unrelated-history",
        [
            {
                "action": "correct_text",
                "origin": "synthetic",
                "before": "recieve" if english else "확읹",
                "after": "receive" if english else "확인",
            }
        ],
    )
    yield make("conflicting-history", [history(500)[0], history(1500)[1]])
    for duration in (2999, 3000, 3001):
        yield make(
            f"cutoff-{duration}",
            history(1500),
            1500 if duration >= 3000 else None,
            duration,
        )
    yield make("protected-pause", history(1500), protected=True)
    yield make("unverified-pause", history(1500), unverified=True)
    yield make("over-contract-limit", history(1500), duration=5001)
    explicit = (
        " For this passage, retain 1.5 seconds in each eligible pause."
        if english
        else " 현재 문맥에서 조건에 맞는 쉼은 1.5초를 남기세요."
    )
    yield make("explicit-duration", [], 1500, suffix=explicit)
    yield make("explicit-overrides-history", history(500), 1500, suffix=explicit)
    preserve = (
        " Preserve this passage's pauses exactly as recorded."
        if english
        else " 현재 문맥의 쉼은 녹음된 길이 그대로 보존하세요."
    )
    yield make("preserve-overrides-history", history(500), suffix=preserve)


def build(recipe_path, output):
    recipe = json.loads(recipe_path.read_text())
    groups = recipe["groups"]
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("Preference source groups must be unique.")
    if any(g["language"] not in ("en", "ko") for g in groups):
        raise ValueError("Preference diagnostics require English or Korean.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "sessions").mkdir()
    (output / "recipe.json").write_bytes(recipe_path.read_bytes())
    (output / "prompt.txt").write_bytes((HERE / "prompt-v2.txt").read_bytes())
    lineage = {
        "generator": "pause-preferences-v1",
        "recipe_sha256": sha(recipe_path),
        "generator_sha256": sha(Path(__file__)),
        "contract_sha256": sha(HERE / "nep_v2.py"),
        "executor_sha256": sha(HERE / "session_cases.py"),
        "history_builder_sha256": sha(HERE / "synthetic_pilot.py"),
    }
    cases, executions = [], []
    for group in groups:
        for spec in scenarios(group):
            before, paths = execute_history(document_for(spec), spec, output)
            evidence = {} if spec.get("unverified") else symbolic_evidence(before)
            bundle = snapshot_for(
                before, "left", spec["brief"], spec["history"], evidence, window_size=3
            )
            if spec.get("protected"):
                bundle["input"]["protected_ids"] = [1]
            after = apply_edit(before, bundle, spec["expected"])
            session = f"sessions/{spec['id']}.jsonl"
            jsonl(
                output / session,
                replayable_session(before, after, spec["expected"]["action"], lineage),
            )
            cases.append(
                {
                    "id": spec["id"],
                    "input": bundle["input"],
                    "expectedOutput": spec["expected"],
                    "metadata": {
                        **lineage,
                        "split": "dev",
                        "source_group_id": group["id"],
                        "scenario_variant": spec["family"],
                        "language": group["language"],
                        "comparison_group": group["id"]
                        if spec.get("matched")
                        else None,
                        "content_origin": "independent_ai_authored_script",
                        "target_origin": "ai_generated_recipe",
                        "history_origin": "executed_synthetic",
                        "actor": "synthetic",
                        "training_eligible": False,
                        "requires_regeneration": False,
                        "review_status": "agent_validated_observable_rule",
                        "timing_origin": "symbolic_fixture_not_acoustic_evidence",
                        "audio_quality": "not_established",
                        "threshold_origin": "explicit_brief",
                        "session": session,
                        "history_sessions": paths,
                        "before_hash": session_hash(before),
                    },
                }
            )
            executions.append(
                {
                    "id": spec["id"],
                    "before": before,
                    "after": after,
                    "mapping": bundle["mapping"],
                }
            )
    jsonl(output / "dev-cases.jsonl", cases)
    jsonl(output / "executions.jsonl", executions)
    manifest = {
        **lineage,
        "question": recipe["question"],
        "scope": recipe["scope"],
        "groups": {g["id"]: "dev" for g in groups},
        "cases": len(cases),
        "actions": dict(Counter(c["expectedOutput"]["action"] for c in cases)),
        "files": {
            p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def score(cases, results, executions):
    ids = [r["case_id"] for r in results]
    if len(set(ids)) != len(ids):
        raise ValueError("Use one prediction per case for paired preference scoring.")
    if set(ids) != {c["id"] for c in cases}:
        raise ValueError("Preference results must cover exactly the supplied cases.")
    report = evaluate(cases, results, executions)
    rows = {r["case_id"]: r for r in report["rows"]}
    paired = {}
    for case in cases:
        group = case["metadata"].get("comparison_group")
        if group:
            paired.setdefault(group, []).append(case)
    groups = []
    for group, members in paired.items():
        by_duration = {c["expectedOutput"]["keep_ms"]: c for c in members}
        if set(by_duration) != {500, 900, 1500} or len(members) != 3:
            raise ValueError(
                "Each history comparison needs 500, 900 and 1500 ms cases."
            )
        inputs = [
            {k: v for k, v in c["input"].items() if k != "recent_edits"}
            for c in members
        ]
        if any(value != inputs[0] for value in inputs):
            raise ValueError("Matched preference inputs differ beyond recent edits.")
        groups.append(
            {
                "source_group_id": group,
                "both_500_and_1500_correct": all(
                    rows[by_duration[k]["id"]]["outcome_match"] is True
                    for k in (500, 1500)
                ),
                "all_three_durations_correct": all(
                    rows[c["id"]]["outcome_match"] is True for c in members
                ),
                "predictions": {
                    str(k): rows[c["id"]].get("prediction")
                    for k, c in by_duration.items()
                },
            }
        )
    report["matched_history"] = {
        "groups": len(groups),
        "both_500_and_1500_correct": sum(
            g["both_500_and_1500_correct"] for g in groups
        ),
        "all_three_durations_correct": sum(
            g["all_three_durations_correct"] for g in groups
        ),
        "rows": groups,
    }
    report["always_abstain_matches"] = sum(
        c["expectedOutput"]["action"] == "no_action" for c in cases
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("build")
    generate.add_argument("--recipe", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    grading = commands.add_parser("score")
    grading.add_argument("--cases", type=Path, required=True)
    grading.add_argument("--results", type=Path, required=True)
    grading.add_argument("--executions", type=Path, required=True)
    grading.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        report = build(args.recipe, args.output)
    else:
        report = score(
            read_rows(args.cases),
            read_rows(args.results),
            load_executions(args.executions),
        )
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("files", "rows")},
            ensure_ascii=False,
            indent=2,
        )
    )
