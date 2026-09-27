"""Build one small, replay-validated training/development iteration from AI recipes."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from replay_session import session_hash
from session_cases import apply_edit, load_project, replayable_session, snapshot_for
from synthetic_pilot import baseline_report, document_for

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def source_excerpt(document, canonical):
    matches = [i["uuid"] for i in document["content"] if i.get("text") == canonical][:2]
    if len(matches) != 2:
        raise ValueError(f"Need two original occurrences of {canonical}.")
    content, paragraph = [], []
    for item in document["content"]:
        paragraph.append(item)
        if item["type"] == "paragraph_end":
            if any(i["uuid"] in matches for i in paragraph):
                content.extend(paragraph)
            paragraph = []
    return {
        "content": copy.deepcopy(content),
        "metadata": copy.deepcopy(document["metadata"]),
    }, matches


def emit_case(
    document, ids, term, variant, case_id, group, split, origin, output, lineage
):
    document = copy.deepcopy(document)
    by_id = {i["uuid"]: i for i in document["content"]}
    prior, target = ids
    if variant != "already_correct":
        by_id[target]["text"] = term["variant"]
    brief = "선택한 표현만 검토하세요. 의미와 말투를 유지하고, 최근에 확립된 표기가 있으면 일관되게 적용하세요. 필요한 수정이 없으면 유지하세요."
    history = []
    if variant == "brief_supported":
        brief += f" 이번 작업의 표준 표기는 ‘{term['canonical']}’입니다."
    else:
        by_id[prior]["text"] = term["variant"]
        previous = snapshot_for(document, prior, brief, window_size=24)
        alias = next(
            int(k) for k, v in previous["mapping"].items() if v["uuid"] == prior
        )
        action = {
            "action": "correct_text",
            "target": [alias],
            "text": term["canonical"],
        }
        after = apply_edit(document, previous, action)
        jsonl(
            output / "sessions" / f"{case_id}.history.jsonl",
            replayable_session(document, after, "correct_text", lineage),
        )
        history = [
            {
                "action": "correct_text",
                "before": term["variant"],
                "after": term["canonical"],
                "origin": "synthetic",
            }
        ]
        document = after
    if variant == "verbatim":
        brief += " 단, 현재 선택은 인용된 원문 표기이므로 이전 교정과 달라도 그대로 보존하세요."
    bundle = snapshot_for(document, target, brief, history, window_size=24)
    alias = next(int(k) for k, v in bundle["mapping"].items() if v["uuid"] == target)
    bundle["input"]["selection"] = [alias]
    bundle["input"]["protected_ids"] = [
        i["id"]
        for i in bundle["input"]["items"]
        if i["id"] != alias or variant == "protected"
    ]
    expected = (
        {"action": "correct_text", "target": [alias], "text": term["canonical"]}
        if variant in ("history_supported", "brief_supported")
        else {"action": "no_action"}
    )
    after = apply_edit(document, bundle, expected)
    session_path = f"sessions/{case_id}.jsonl"
    jsonl(
        output / session_path,
        replayable_session(document, after, expected["action"], lineage),
    )
    metadata = {
        "split": split,
        "source_group_id": group,
        "scenario_variant": variant,
        "content_origin": origin,
        "context_origin": "constructed_variant",
        "history_origin": "executed_synthetic",
        "target_origin": "ai_generated_recipe",
        "actor": "synthetic",
        "training_eligible": split == "train",
        "review_status": "agent_validated_observable_rule",
        "requires_regeneration": False,
        "timing_origin": "source_timings_unverified"
        if origin == "permitted_real_media"
        else "symbolic_fixture",
        "audio_quality": "not_established",
        "action_scope": "selected_text_only",
        "session": session_path,
        "before_hash": session_hash(document),
        "history_session": f"sessions/{case_id}.history.jsonl" if history else None,
        **lineage,
    }
    return {
        "id": case_id,
        "input": bundle["input"],
        "expectedOutput": expected,
        "metadata": metadata,
    }, {"id": case_id, "before": document, "mapping": bundle["mapping"], "after": after}


def messages(case, prompt):
    metadata = case["metadata"]
    if (
        metadata.get("requires_regeneration")
        or metadata["target_origin"] != "ai_generated_recipe"
        or case["expectedOutput"] is None
    ):
        raise ValueError("Only validated, regenerated AI targets can be exported.")
    return {
        "messages": [
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
                    case["expectedOutput"], ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]
    }


def build(recipe_path, project, output):
    recipe = json.loads(recipe_path.read_text())
    if recipe["source_group"]["split"] != "train":
        raise ValueError("This iteration uses its real source group for training only.")
    variants = recipe["variants"]
    allowed = {
        "history_supported",
        "brief_supported",
        "already_correct",
        "protected",
        "verbatim",
    }
    if (
        not variants
        or len(set(variants)) != len(variants)
        or not set(variants) <= allowed
    ):
        raise ValueError("Use unique, supported scenario variants.")
    if sha(project) != recipe["source_group"]["project_sha256"]:
        raise ValueError("Source project differs from the assigned source group.")
    groups = {recipe["source_group"]["id"]: "train"}
    for scenario in recipe["development_scenarios"]:
        if scenario["group"] in groups:
            raise ValueError(
                "Source/scenario groups must be assigned uniquely before generation."
            )
        groups[scenario["group"]] = "dev"
    output.mkdir(parents=True, exist_ok=False)
    (output / "sessions").mkdir()
    lineage = {
        "generator": "iteration-v1",
        "recipe_sha256": sha(recipe_path),
        "generator_sha256": sha(Path(__file__)),
        "contract_sha256": sha(HERE / "nep_v2.py"),
        "executor_sha256": sha(HERE / "session_cases.py"),
    }
    (output / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    (output / "recipe.json").write_bytes(recipe_path.read_bytes())
    document = load_project(project)
    cases, executions = [], []
    sources = []
    for index, term in enumerate(recipe["training_terms"]):
        excerpt, ids = source_excerpt(document, term["canonical"])
        sources.append(
            (
                f"source-{index}",
                excerpt,
                ids,
                term,
                recipe["source_group"]["id"],
                "train",
                "permitted_real_media",
                {**lineage, "project_sha256": sha(project)},
            )
        )
    for index, term in enumerate(recipe["development_scenarios"]):
        words = [
            {"id": "prior", "text": term["canonical"]},
            {"id": "intro", "text": term["context"][0]},
            {"id": "target", "text": term["canonical"]},
            {"id": "tail", "text": " ".join(term["context"][1:])},
        ]
        synthetic = document_for({"paragraphs": [words[:1], words[1:]]})
        sources.append(
            (
                f"dev-{index}",
                synthetic,
                ["prior", "target"],
                term,
                term["group"],
                "dev",
                "ai_authored_scenario",
                lineage,
            )
        )
    for prefix, before, ids, term, group, split, origin, provenance in sources:
        for variant in recipe["variants"]:
            case, execution = emit_case(
                before,
                ids,
                term,
                variant,
                prefix + "-" + variant,
                group,
                split,
                origin,
                output,
                provenance,
            )
            cases.append(case)
            executions.append(execution)
    prompt = (HERE / "prompt-v2.txt").read_text()
    (output / "prompt.txt").write_text(prompt)
    for split in ("train", "dev"):
        selected = [c for c in cases if c["metadata"]["split"] == split]
        jsonl(output / f"{split}-cases.jsonl", selected)
        jsonl(output / f"{split}.jsonl", [messages(c, prompt) for c in selected])
    jsonl(output / "executions.jsonl", executions)
    manifest = {
        "iteration": 1,
        "question": recipe["question"],
        "status": "data_ready_training_pending",
        "groups": groups,
        "counts": dict(Counter(c["metadata"]["split"] for c in cases)),
        "actions": dict(Counter(c["expectedOutput"]["action"] for c in cases)),
        "scope": "Small selected-text learning experiment; constructed corruptions, no acoustic positives or held-out test claim",
        "next": "Run a short adapter experiment, compare base/adapter on dev plus existing seven-action regressions, then revise generation from failures.",
        "baselines": {
            split: baseline_report(
                [c for c in cases if c["metadata"]["split"] == split]
            )
            for split in ("train", "dev")
        },
        "files": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        **lineage,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", type=Path, default=HERE / "fixtures/iteration-01.json"
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.recipe, args.project, args.output)
    print(
        json.dumps(
            {k: manifest[k] for k in ("status", "counts", "actions", "scope")}, indent=2
        )
    )
