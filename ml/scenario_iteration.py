"""Generate a small mixed-action iteration with explicit evidence and abstention pairs."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from iteration import HERE, jsonl, messages, sha
from replay_session import session_hash
from session_cases import apply_edit, replayable_session, snapshot_for
from synthetic_pilot import document_for, execute_history, symbolic_evidence


def scenarios(group):
    term, corrupt = group["term"], group["corrupt"]
    history = [
        {
            "action": "correct_text",
            "before": corrupt,
            "after": term,
            "origin": "synthetic",
        }
    ]

    def case(name, paragraphs, brief, expected=None, **extra):
        return {
            "id": group["id"] + "-" + name,
            "family": name,
            "paragraphs": [
                [
                    {"id": f"w{p}-{i}", "text": text, "ms": 1200 if not text else 400}
                    for i, text in enumerate(words)
                ]
                for p, words in enumerate(paragraphs)
            ],
            "focus": "w0-0",
            "brief": brief,
            "expected": expected or {"action": "no_action"},
            **extra,
        }

    correction = {"action": "correct_text", "target": ["w0-0"], "text": term}
    common = "선택한 표현만 검토하세요. 의미와 말투를 보존하고 필요한 수정만 하세요."
    yield case(
        "history-correction",
        [[corrupt, group["tail"]]],
        common,
        correction,
        history=history,
        reason="The same misspelling was corrected in an executed earlier edit.",
    )
    yield case(
        "truncated-reference",
        [group["truncated"]],
        "선택한 자막의 누락된 글자와 문장부호를 다음 확인된 대본에 맞추세요: "
        + group["reference"],
        {
            "action": "correct_text",
            "target": [f"w0-{i}" for i in range(len(group["truncated"]))],
            "text": group["reference"],
        },
        selection=[f"w0-{i}" for i in range(len(group["truncated"]))],
        reason="The observable brief supplies the exact AI-authored reference; repair display text only.",
    )
    yield case(
        "fragmented-word",
        [group["fragmented"]],
        "선택한 단어의 잘못된 분절을 교정하세요. 이 문맥의 용어는 ‘"
        + group["joined"]
        + "’입니다.",
        {"action": "correct_text", "target": ["w0-0", "w0-1"], "text": group["joined"]},
        selection=["w0-0", "w0-1"],
        reason="Two erroneous source items form the explicitly established term; preserve both audio intervals.",
    )
    yield case(
        "delete-requested-filler",
        [[group["filler"], group["start"]]],
        "첫머리의 ‘"
        + group["filler"]
        + "’는 불필요한 머뭇거림입니다. 해당 소리를 잘라내고 ‘"
        + group["start"]
        + "’부터 시작하세요.",
        {"action": "delete_span", "target": ["w0-0"]},
        qualified_fixture=True,
        reason="Explicit synthetic intent and symbolic verified boundaries permit this execution example, not a real-audio quality claim.",
    )
    yield case(
        "shorten-pause",
        [[group["start"], "", group["end"]]],
        f"현재 두 항목 사이의 무음은 {group['keep_ms']}밀리초만 남겨 템포를 맞추세요.",
        {"action": "trim_silence", "target": ["w0-1"], "keep_ms": group["keep_ms"]},
        focus="w0-1",
        qualified_fixture=True,
        reason="Requested positive retained duration is shorter than the symbolic 1200 ms silence.",
    )
    yield case(
        "remove-pause",
        [[group["start"], "", group["end"]]],
        "두 항목은 한 흐름으로 연결합니다. 사이의 무음을 남김없이 잘라내세요.",
        {"action": "remove_silence", "target": ["w0-1"]},
        focus="w0-1",
        qualified_fixture=True,
        reason="The observable brief requests full removal, distinguishing it from shortening.",
    )
    yield case(
        "split-sentences",
        [group["sentences"]],
        "완결된 문장마다 자막을 하나씩 만드세요. 두 문장이 한 자막에 붙었습니다.",
        {"action": "split_paragraph", "target": ["w0-1"]},
        focus="w0-1",
        reason="Two complete short sentences and an explicit per-sentence brief support this boundary.",
    )
    yield case(
        "merge-fragments",
        [[group["merge"][0]], [group["merge"][1]]],
        "현재 짧은 한 문장이 두 자막으로 갈라졌습니다. 한 문장짜리 자막으로 합치세요.",
        {"action": "merge_paragraph", "target": ["w1-0"]},
        focus="w1-0",
        reason="Complete adjacent fragments share speaker/language and form the requested short sentence.",
    )
    yield case(
        "already-correct",
        [[term, group["tail"]]],
        common,
        history=history,
        reason="The earlier desired spelling is already present; do not repeat it.",
    )
    yield case(
        "protected-corruption",
        [[corrupt, group["tail"]]],
        common,
        history=history,
        protected=["w0-0"],
        reason="Protection overrides a supported correction.",
    )
    yield case(
        "verbatim-corruption",
        [[corrupt, group["tail"]]],
        common
        + " 현재 선택은 원문을 그대로 인용한 부분입니다. 틀려 보여도 이 표기는 보존하세요.",
        history=history,
        reason="Explicit verbatim intent overrides the earlier spelling preference.",
    )
    yield case(
        "missing-name",
        [[group["unknown_name"], "담당자가 발표합니다."]],
        "인식되지 않은 이름 일부가 있습니다. 명단도 원음도 제공되지 않았습니다. 추측으로 채우지 마세요.",
        reason="The intended name is unavailable. Abstention is a policy target, not a claim about the true name.",
    )
    yield case(
        "missing-content",
        [[group["missing"]]],
        "문장 중 일부가 누락됐지만 대본과 원음이 없습니다. 사실이나 단어를 추측해 복원하지 마세요.",
        reason="Several completions fit; preserve the incomplete transcript rather than invent content.",
    )
    yield case(
        "unfinished-sentence",
        [[group["unfinished"]]],
        "화자가 아직 말하는 중이며 뒤 문맥이 없습니다. 발화가 끝나기 전에는 문장이나 자막 경계를 확정하지 마세요.",
        reason="The observed unfinished utterance is not evidence for completion or punctuation.",
    )
    yield case(
        "unverified-gap",
        [[group["start"], "", group["end"]]],
        "말소리가 없는 것으로 확인된 무음만 줄이세요. 이 빈 구간의 원음은 아직 확인하지 못했습니다.",
        focus="w0-1",
        reason="Missing transcript text does not establish silence or qualified cut boundaries.",
    )
    yield case(
        "meaningful-filler",
        [[group["filler"], group["start"]]],
        "첫 ‘"
        + group["filler"]
        + "’는 고민하는 감정을 전달하는 의도된 표현입니다. 말투와 쉼을 보존하세요.",
        qualified_fixture=True,
        reason="The same filler with explicit expressive intent must remain, even with symbolic boundary eligibility.",
    )


def build(recipe_path, retained, output):
    recipe = json.loads(recipe_path.read_text())
    if sha(retained / "manifest.json") != recipe["retained_manifest_sha256"]:
        raise ValueError("Retained development data changed.")
    retained_cases = [
        json.loads(line)
        for line in (retained / "dev-cases.jsonl").read_text().splitlines()
    ]
    groups = {c["metadata"]["source_group_id"]: "dev" for c in retained_cases}
    for group in recipe["groups"]:
        if group["id"] in groups or group["split"] not in ("train", "dev"):
            raise ValueError(
                "Source/scenario groups must be assigned uniquely before generation."
            )
        groups[group["id"]] = group["split"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "sessions").mkdir()
    lineage = {
        "generator": "scenario-iteration-v1",
        "recipe_sha256": sha(recipe_path),
        "generator_sha256": sha(Path(__file__)),
        "contract_sha256": sha(HERE / "nep_v2.py"),
        "executor_sha256": sha(HERE / "session_cases.py"),
    }
    cases, executions = [], []
    for group in recipe["groups"]:
        for spec in scenarios(group):
            document, history_paths = execute_history(document_for(spec), spec, output)
            proof = symbolic_evidence(document) if spec.get("qualified_fixture") else {}
            bundle = snapshot_for(
                document, spec["focus"], spec["brief"], spec.get("history", []), proof
            )
            aliases = {v["uuid"]: int(k) for k, v in bundle["mapping"].items()}
            bundle["input"]["selection"] = [
                aliases[t] for t in spec.get("selection", [spec["focus"]])
            ]
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
                        "scenario_variant": spec["family"],
                        "content_origin": "ai_authored_scenario",
                        "target_origin": "ai_generated_recipe",
                        "history_origin": "executed_synthetic",
                        "history_sessions": history_paths,
                        "actor": "synthetic",
                        "training_eligible": group["split"] == "train",
                        "requires_regeneration": False,
                        "review_status": "agent_validated_observable_rule",
                        "timing_origin": "symbolic_fixture_not_acoustic_evidence",
                        "audio_quality": "not_established",
                        "rationale": spec["reason"],
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
    retained_ids = {c["id"] for c in retained_cases}
    for case in retained_cases:
        for session in [
            case["metadata"]["session"],
            case["metadata"].get("history_session"),
        ]:
            if session:
                (output / session).write_bytes((retained / session).read_bytes())
    cases.extend(retained_cases)
    executions.extend(
        row
        for row in map(
            json.loads, (retained / "executions.jsonl").read_text().splitlines()
        )
        if row["id"] in retained_ids
    )
    prompt = (HERE / "prompt-v2.txt").read_text()
    (output / "prompt.txt").write_text(prompt)
    (output / "recipe.json").write_bytes(recipe_path.read_bytes())
    (output / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    jsonl(output / "executions.jsonl", executions)
    for split in ("train", "dev"):
        selected = [c for c in cases if c["metadata"]["split"] == split]
        jsonl(output / f"{split}-cases.jsonl", selected)
        jsonl(output / f"{split}.jsonl", [messages(c, prompt) for c in selected])
    manifest = {
        **lineage,
        "iteration": 2,
        "question": recipe["question"],
        "status": "data_ready_training_pending",
        "groups": groups,
        "counts": dict(Counter(c["metadata"]["split"] for c in cases)),
        "actions_by_split": {
            split: dict(
                Counter(
                    c["expectedOutput"]["action"]
                    for c in cases
                    if c["metadata"]["split"] == split
                )
            )
            for split in ("train", "dev")
        },
        "retained_manifest_sha256": recipe["retained_manifest_sha256"],
        "scope": "Constructed STT-like corruptions and symbolic actions; no measured real STT quality, audio qualification or sealed test",
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--retained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.recipe, args.retained, args.output), ensure_ascii=False, indent=2
        )
    )
