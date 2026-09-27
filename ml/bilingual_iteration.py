"""Build unselected bilingual history diagnostics with matched observable contexts."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from evaluate import read_rows
from iteration import HERE, jsonl, messages, sha
from replay_session import session_hash
from session_cases import apply_edit, replayable_session, snapshot_for
from synthetic_pilot import document_for, execute_history, symbolic_evidence


def scenarios(group):
    english = group["language"] == "en"
    brief = (
        "Review this passage for the next useful edit. Preserve meaning, names, emphasis and language. "
        "Use consistent spelling when prior corrections establish it. If evidence is insufficient or conflicting, leave the passage unchanged."
        if english
        else "이 문맥에서 다음으로 필요한 편집 하나를 검토하세요. 의미와 이름, 강조와 언어를 보존하세요. "
        "이전 교정에서 확인된 표기는 일관되게 사용하되 근거가 부족하거나 서로 충돌하면 그대로 두세요."
    )
    relevant = {
        "action": "correct_text",
        "before": group["name"],
        "after": group["canonical"],
        "origin": "synthetic",
    }
    unrelated = {
        "action": "correct_text",
        "before": group["typo"],
        "after": group["correct"],
        "origin": "synthetic",
    }
    conditions = {
        "relevant": [relevant],
        "absent": [],
        "unrelated": [unrelated],
        "conflicting": [relevant, {**relevant, "after": group["alternate"]}],
    }

    def make(family, paragraphs, *, history=(), instruction=brief, **extra):
        return {
            "id": f"{group['id']}-{family}",
            "family": family,
            "language": group["language"],
            "brief": instruction,
            "paragraphs": [
                [
                    {"id": f"w{p}-{i}", "text": word, "ms": 1200 if not word else 400}
                    for i, word in enumerate(words)
                ]
                for p, words in enumerate(paragraphs)
            ],
            "focus": "w0-0",
            "history": list(history),
            "expected": {"action": "no_action"},
            **extra,
        }

    def passage(name):
        return [
            word.replace("{name}", name).replace("{foil}", group["foil"])
            for word in group["passage"]
        ]

    target_index = next(
        i for i, word in enumerate(group["passage"]) if "{name}" in word
    )
    for context in ("short", "long"):
        paragraphs = [passage(group["name"])]
        p = 0
        if context == "long":
            paragraphs = [
                group["background"][:2],
                paragraphs[0],
                group["background"][2:],
            ]
            p = 1
        for condition, history in conditions.items():
            target = f"w{p}-{target_index}"
            yield make(
                f"history-{condition}-{context}",
                paragraphs,
                history=history,
                expected={
                    "action": "correct_text",
                    "target": [target],
                    "text": group["canonical"],
                }
                if condition == "relevant"
                else {"action": "no_action"},
                history_condition=condition,
                context_size=context,
                comparison_group=f"{group['id']}-{context}",
            )
    target = f"w0-{target_index}"
    yield make("already-correct", [passage(group["canonical"])], history=[relevant])
    yield make(
        "protected-name",
        [passage(group["name"])],
        history=[relevant],
        protected=[target],
    )
    yield make(
        "verbatim-name",
        [passage(group["name"])],
        history=[relevant],
        instruction=brief
        + (
            " This passage is a verbatim quotation; preserve its spelling."
            if english
            else " 현재 문맥은 원문 인용이므로 표기를 그대로 보존하세요."
        ),
    )
    yield make(
        "missing-name",
        [passage("[unclear name]" if english else "[이름 불명]")],
        history=[relevant],
    )
    typo_index = next(
        i for i, word in enumerate(group["typo_passage"]) if "{typo}" in word
    )
    words = [word.replace("{typo}", group["typo"]) for word in group["typo_passage"]]
    yield make(
        "clear-text-error",
        [words],
        expected={
            "action": "correct_text",
            "target": [f"w0-{typo_index}"],
            "text": group["typo_passage"][typo_index].replace(
                "{typo}", group["correct"]
            ),
        },
    )
    a, b = group["sentences"]
    yield make(
        "remove-hesitation",
        [[a, group["filler"], b]],
        qualified_fixture=True,
        instruction=brief
        + (
            " Remove the hesitation between the completed sentences; keep their wording."
            if english
            else " 완결된 두 문장 사이의 머뭇거림만 제거하고 문장 내용은 유지하세요."
        ),
        expected={"action": "delete_span", "target": ["w0-1"]},
    )
    pause_history = [
        {
            "action": "trim_silence",
            "before_ms": 1200,
            "after_ms": 400,
            "origin": "synthetic",
        }
    ]
    yield make(
        "trim-pause",
        [[a, "", b]],
        qualified_fixture=True,
        history=pause_history,
        instruction=brief
        + (
            " Follow the pause duration established by the recent edit."
            if english
            else " 최근 편집에서 정한 쉼 길이를 동일하게 적용하세요."
        ),
        expected={"action": "trim_silence", "target": ["w0-1"], "keep_ms": 400},
    )
    yield make(
        "remove-interruption",
        [[a, "", b]],
        qualified_fixture=True,
        instruction=brief
        + (
            " The pause is a recording interruption. Remove it completely."
            if english
            else " 두 문장 사이의 쉼은 녹음 중단으로 생긴 구간입니다. 이 구간을 완전히 제거하세요."
        ),
        expected={"action": "remove_silence", "target": ["w0-1"]},
    )
    yield make(
        "split-caption",
        [[a, b]],
        instruction=brief
        + (
            " Keep each complete sentence in a separate caption."
            if english
            else " 완결된 문장마다 자막을 나누세요."
        ),
        expected={"action": "split_paragraph", "target": ["w0-1"]},
    )
    yield make(
        "merge-caption",
        [[group["fragment"][0]], [group["fragment"][1]]],
        instruction=brief
        + (
            " Repair the caption break in the middle of this single sentence."
            if english
            else " 하나의 문장 중간에 잘못 나뉜 자막 경계를 합치세요."
        ),
        expected={"action": "merge_paragraph", "target": ["w1-0"]},
    )
    yield make(
        "keep-emphasis",
        [["very" if english else "정말", "very" if english else "정말", b]],
        qualified_fixture=True,
        instruction=brief
        + (
            " The repetition is deliberate emphasis."
            if english
            else " 반복된 표현은 의도적으로 강조한 말입니다."
        ),
    )
    yield make("unknown-gap", [[a, "", b]], history=pause_history)


def build(recipe_path, retained, output):
    recipe = json.loads(recipe_path.read_text())
    manifest = json.loads((retained / "manifest.json").read_text())
    if sha(retained / "manifest.json") != recipe["retained_manifest_sha256"]:
        raise ValueError("Retained development manifest changed.")
    for name, expected in manifest["files"].items():
        path = (retained / name).resolve()
        if not path.is_relative_to(retained.resolve()) or sha(path) != expected:
            raise ValueError("Retained development file differs from its manifest.")
    retained_cases = read_rows(retained / "dev-cases.jsonl")
    groups = {c["metadata"]["source_group_id"]: "dev" for c in retained_cases}
    reserved = set(recipe["reserved_test_groups"])
    for group in recipe["groups"]:
        if (
            group["id"] in groups
            or group["id"] in reserved
            or group["split"] not in ("train", "dev")
        ):
            raise ValueError(
                "Source/script groups must be unique and cannot reuse a reserved test group."
            )
        if group["language"] not in ("en", "ko"):
            raise ValueError("This bilingual recipe requires English or Korean.")
        if len({group["name"], group["canonical"], group["alternate"]}) != 3:
            raise ValueError("History alternatives must be distinct.")
        groups[group["id"]] = group["split"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "sessions").mkdir()
    (output / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    (output / "recipe.json").write_bytes(recipe_path.read_bytes())
    lineage = {
        "generator": "bilingual-history-v1",
        "recipe_sha256": sha(recipe_path),
        "generator_sha256": sha(Path(__file__)),
        "contract_sha256": sha(HERE / "nep_v2.py"),
        "executor_sha256": sha(HERE / "session_cases.py"),
        "history_builder_sha256": sha(HERE / "synthetic_pilot.py"),
    }
    cases, executions = [], []
    for group in recipe["groups"]:
        for spec in scenarios(group):
            document = document_for(spec)
            visible_count = sum(
                i["type"] not in ("paragraph_start", "paragraph_end")
                for i in document["content"]
            )
            document, history_paths = execute_history(document, spec, output)
            proof = symbolic_evidence(document) if spec.get("qualified_fixture") else {}
            # Prior edits execute outside this fixed window, so their resulting text cannot reveal the answer.
            bundle = snapshot_for(
                document,
                spec["focus"],
                spec["brief"],
                spec["history"],
                proof,
                window_size=visible_count,
            )
            aliases = {v["uuid"]: int(k) for k, v in bundle["mapping"].items()}
            bundle["input"]["protected_ids"] = [
                aliases[t] for t in spec.get("protected", [])
            ]
            expected = copy.deepcopy(spec["expected"])
            if "target" in expected:
                expected["target"] = [aliases[t] for t in expected["target"]]
            after = apply_edit(document, bundle, expected)
            session = f"sessions/{spec['id']}.jsonl"
            jsonl(
                output / session,
                replayable_session(document, after, expected["action"], lineage),
            )
            cases.append(
                {
                    "id": spec["id"],
                    "input": bundle["input"],
                    "expectedOutput": expected,
                    "metadata": {
                        **lineage,
                        "split": group["split"],
                        "source_group_id": group["id"],
                        "language": group["language"],
                        "domain": group["domain"],
                        "scenario_variant": spec["family"],
                        "history_condition": spec.get("history_condition"),
                        "context_size": spec.get("context_size"),
                        "comparison_group": spec.get("comparison_group"),
                        "content_origin": "independent_ai_authored_script",
                        "target_origin": "ai_generated_recipe",
                        "history_origin": "executed_synthetic",
                        "history_sessions": history_paths,
                        "actor": "synthetic",
                        "training_eligible": group["split"] == "train",
                        "requires_regeneration": False,
                        "review_status": "agent_validated_observable_rule",
                        "timing_origin": "symbolic_fixture_not_acoustic_evidence",
                        "audio_quality": "not_established",
                        "action_scope": "unselected_current_passage",
                        "session": session,
                        "before_hash": session_hash(document),
                    },
                }
            )
            executions.append(
                {
                    "id": spec["id"],
                    "before": document,
                    "mapping": bundle["mapping"],
                    "after": after,
                }
            )
    new_cases = list(cases)
    for case in retained_cases:
        paths = [
            case["metadata"]["session"],
            case["metadata"].get("history_session"),
            *case["metadata"].get("history_sessions", []),
        ]
        for relative in filter(None, paths):
            source = (retained / relative).resolve()
            destination = (output / relative).resolve()
            if not source.is_relative_to(
                retained.resolve()
            ) or not destination.is_relative_to((output / "sessions").resolve()):
                raise ValueError("Journal path escapes its session directory.")
            destination.write_bytes(source.read_bytes())
    cases.extend(retained_cases)
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Case IDs must be unique.")
    retained_ids = {c["id"] for c in retained_cases}
    executions.extend(
        r for r in read_rows(retained / "executions.jsonl") if r["id"] in retained_ids
    )
    prompt = (HERE / "prompt-v2.txt").read_text()
    (output / "prompt.txt").write_text(prompt)
    jsonl(output / "executions.jsonl", executions)
    jsonl(
        output / "new-dev-cases.jsonl",
        [c for c in new_cases if c["metadata"]["split"] == "dev"],
    )
    for split in ("train", "dev"):
        selected = [c for c in cases if c["metadata"]["split"] == split]
        jsonl(output / f"{split}-cases.jsonl", selected)
        jsonl(output / f"{split}.jsonl", [messages(c, prompt) for c in selected])
    result = {
        **lineage,
        "iteration": 3,
        "status": "data_ready_training_pending",
        "question": recipe["question"],
        "groups": groups,
        "reserved_test_groups": recipe["reserved_test_groups"],
        "counts": dict(Counter(c["metadata"]["split"] for c in cases)),
        "new_counts": dict(Counter(c["metadata"]["split"] for c in new_cases)),
        "retained_development": len(retained_cases),
        "scope": recipe["scope"],
        "actions_by_split": {
            split: dict(
                Counter(
                    c["expectedOutput"]["action"]
                    for c in new_cases
                    if c["metadata"]["split"] == split
                )
            )
            for split in ("train", "dev")
        },
        "languages_by_split": {
            split: dict(
                Counter(
                    c["input"]["language"]
                    for c in new_cases
                    if c["metadata"]["split"] == split
                )
            )
            for split in ("train", "dev")
        },
        "files": {
            p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob("*")
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.recipe, args.retained, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "files"},
            ensure_ascii=False,
            indent=2,
        )
    )
