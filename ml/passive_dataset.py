"""Build and audit observable-input editing episodes without changing the D4 seed."""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from replay_session import replay, session_hash
from session_cases import document_records, replayable_session, transaction
from verify_release_export import verify_operation

ROOT = Path(__file__).resolve().parents[1]
RESERVED = {"mit-801sc-f2016", "future-en-aviation", "future-ko-climate"}

# These are display corruptions, not claims about mistakes made by an ASR service.
SPELLING = {
    "different": "diffrent",
    "because": "becuase",
    "probably": "probaly",
    "actually": "actualy",
    "something": "somthing",
    "important": "importnat",
    "example": "exmaple",
    "question": "quesiton",
    "people": "peopel",
    "little": "littel",
    "between": "betwen",
    "design": "desgin",
    "remote": "remtoe",
    "button": "buton",
    "buttons": "butons",
    "product": "prodcut",
    "function": "funciton",
    "information": "informaiton",
    "screen": "screeen",
    "control": "contorl",
    "basically": "basicly",
    "perhaps": "perhpas",
    "possible": "possibel",
    "maybe": "mayeb",
    "suppose": "supopse",
    "think": "thnik",
    "remember": "remeber",
    "understand": "understnad",
    "number": "nubmer",
    "separate": "seperate",
    "difference": "diffrence",
    "computer": "comptuer",
    "another": "antoher",
    "together": "togehter",
    "anything": "anythnig",
    "everything": "everythnig",
    "really": "realy",
    "already": "alraedy",
    "should": "shoudl",
    "would": "woudl",
    "could": "coudl",
    "might": "mihgt",
    "however": "howver",
    "whether": "whehter",
    "through": "throuhg",
    "obviously": "obviuosly",
    "specific": "specfic",
    "similar": "simliar",
}
KO_SPELLING = {
    "생각": "생갹",
    "그러니까": "그러니깐까",
    "그런데": "그런댸",
    "이렇게": "이럭게",
    "이거는": "이거느는",
    "예를": "예르를",
    "정확히": "정학히",
    "중요한": "중요하한",
    "가능한": "가능하한",
    "계산": "계샨",
    "확인": "확닌",
    "결과": "결곽",
    "사람": "사럄",
    "이야기": "이야긔",
    "그래서": "그래셔",
    "말씀": "말씁",
    "부분": "부뷴",
    "방법": "방뻡",
    "개발": "개뱔",
    "경제": "경졔",
    "정부": "정뷰",
    "오늘": "오늟",
    "지난": "지난난",
    "말했다": "말햇다",
    "발표": "발푠",
    "교육": "교윢",
    "환경": "환굥",
    "문화": "문홰",
    "기술": "기슐",
    "기업": "기엽업",
    "지역": "지역역",
    "이후": "이훟",
    "당시": "당싀",
    "문제": "문졔",
    "설명": "설먕",
    "시간": "시갼",
    "대통령": "대통렁",
    "대학": "대햑",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path, values):
    Path(path).write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values)
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mutations(item, language):
    text = item.get("text", "")
    if item["type"] != "text" or "\n" in text:
        return []
    values = []
    if language.startswith("en"):
        for match in re.finditer(r"\b[A-Za-z]+\b", text):
            original = match[0]
            bad = SPELLING.get(original.lower())
            if bad:
                if original[0].isupper():
                    bad = bad.capitalize()
                values.append(
                    {
                        "uuid": item["uuid"],
                        "original": text,
                        "replacement": text[: match.start()]
                        + bad
                        + text[match.end() :],
                        "mechanism": "synthetic_display_spelling",
                        "lexeme": original.lower(),
                    }
                )
    else:
        for original, bad in KO_SPELLING.items():
            if original in text:
                values.append(
                    {
                        "uuid": item["uuid"],
                        "original": text,
                        "replacement": text.replace(original, bad, 1),
                        "mechanism": "synthetic_display_spelling",
                        "lexeme": original,
                    }
                )
    return values


def mine_restorations(source, max_episodes=24):
    """At most one episode per disjoint 96-item region, with actual edit history."""
    if max_episodes < 1:
        raise ValueError("The episode cap must be positive.")
    items = [i for i in source["document"]["content"] if "length" in i]
    candidates, seen_words = [], Counter()
    for low in range(0, len(items), 96):
        local = items[low : low + 64]
        possibilities = [
            (n, m)
            for n, item in enumerate(local)
            for m in mutations(item, source["language"])
        ]
        if not possibilities:
            continue
        possibilities.sort(
            key=lambda pair: (seen_words[pair[1]["lexeme"]], abs(pair[0] - 24), pair[0])
        )
        first_position, first = possibilities[0]
        second = next(
            (
                m
                for n, m in possibilities
                if m["uuid"] != first["uuid"]
                and m["lexeme"] != first["lexeme"]
                and abs(n - first_position) <= 20
            ),
            None,
        )
        chosen = [first] + ([second] if second else [])
        for m in chosen:
            seen_words[m["lexeme"]] += 1
        candidates.append(
            {
                "id": f"{source['id']}-restore-{low // 96:03d}",
                "source_id": source["id"],
                "kind": "source_derived_restoration",
                "corruptions": chosen,
                "source_region": [
                    local[0]["sourceStart"],
                    local[-1]["sourceStart"] + local[-1]["length"],
                ],
                "rationale": "Repair only the injected unambiguous display spelling; retain surrounding speech, uncertainty, numbers, repetitions and all source intervals.",
            }
        )
    if len(candidates) > max_episodes:
        if max_episodes == 1:
            return [candidates[len(candidates) // 2]]
        indices = [
            round(i * (len(candidates) - 1) / (max_episodes - 1))
            for i in range(max_episodes)
        ]
        candidates = [candidates[n] for n in indices]
    return candidates


def load_sources(registries):
    sources, groups = {}, {}
    for registry in registries:
        for row in read(registry)["sources"]:
            group, split = row["source_group_id"], row["split"]
            if group in RESERVED or split not in {"train", "dev"}:
                raise ValueError("Reserved or unknown split cannot be loaded.")
            if group in groups and groups[group] != split:
                raise ValueError("A source family crosses splits.")
            groups[group] = split
            path = ROOT / row["document_path"]
            if row.get("document_sha256") and digest(path) != row["document_sha256"]:
                raise ValueError("Source document hash changed.")
            source = {**row, "document": read(path)}
            if row["id"] in sources:
                raise ValueError("Duplicate source identity.")
            document_records(source["document"])
            sources[row["id"]] = source
    return sources


def apply_corruptions(document, changes):
    result = copy.deepcopy(document)
    indexed = {i["uuid"]: i for i in result["content"]}
    for change in changes:
        item = indexed[change["uuid"]]
        if item.get("text") != change["original"]:
            raise ValueError("Synthetic before-image differs from source.")
        item["text"] = change["replacement"]
    return result


def alias(bundle, identity):
    return next(
        int(key) for key, item in bundle["mapping"].items() if item["uuid"] == identity
    )


def compact_transaction(before, after, seq, revision, action, **extra):
    event = transaction(before, after, seq, revision, action, **extra)
    left, right = before["content"], after["content"]
    start = 0
    while start < min(len(left), len(right)) and left[start] == right[start]:
        start += 1
    stop = 0
    while (
        stop < min(len(left), len(right)) - start
        and left[-stop - 1] == right[-stop - 1]
    ):
        stop += 1
    event["patch"].update(
        index=start,
        remove=left[start : len(left) - stop if stop else None],
        insert=right[start : len(right) - stop if stop else None],
    )
    return event


def episode_steps(spec):
    if spec.get("steps"):
        return copy.deepcopy(spec["steps"])
    steps = [
        {
            "focus_uuid": m["uuid"],
            "action": "correct_text",
            "target_uuids": [m["uuid"]],
            "text": m["original"],
        }
        for m in spec["corruptions"]
    ]
    if spec["kind"] == "source_derived_restoration":
        steps.append(
            {"focus_uuid": spec["corruptions"][-1]["uuid"], "action": "no_action"}
        )
    return steps


def build_episode(spec, source, builder, output):
    from passive_contract import (
        apply_passive_edit,
        canonical_messages,
        committed_transition,
    )

    document = apply_corruptions(source["document"], spec.get("corruptions", []))
    initial = copy.deepcopy(document)
    history, cases, journal = [], [], []
    steps = episode_steps(spec)
    journal.append(
        {
            "kind": "start",
            "seq": 0,
            "schema_version": 2,
            "hash_format": "typed-json-f64-v1",
            "actor": "synthetic",
            "collection_mode": "synthetic",
            "generator": "passive-dataset-v1",
            "document": initial,
            "document_hash": session_hash(initial),
            "sources": [source["source_group_id"]],
        }
    )
    for n, step in enumerate(steps):
        bundle = builder.build(
            document,
            step["focus_uuid"],
            as_of=n,
            history=history,
            caption_profile=spec.get("caption_profile"),
        )
        prediction = {
            k: v for k, v in step.items() if k in {"action", "text", "keep_ms"}
        }
        if step.get("target_uuids"):
            prediction["target"] = [alias(bundle, u) for u in step["target_uuids"]]
        after = apply_passive_edit(document, bundle, prediction)
        verify_operation(document, after, bundle["mapping"], prediction)
        alternatives = []
        indexed = {i["uuid"]: i for i in document["content"]}
        visible = {i["uuid"] for i in bundle["mapping"].values()}
        for change in spec.get("corruptions", []):
            if (
                change["uuid"] not in visible
                or indexed[change["uuid"]]["text"] != change["replacement"]
            ):
                continue
            alternative = {
                "action": "correct_text",
                "target": [alias(bundle, change["uuid"])],
                "text": change["original"],
            }
            if alternative != prediction:
                alternate_after = apply_passive_edit(document, bundle, alternative)
                verify_operation(
                    document, alternate_after, bundle["mapping"], alternative
                )
                alternatives.append(alternative)
        # Undo/redo runs through the journal parser, independently of the executor.
        undo_log = replayable_session(
            document, after, prediction["action"], [source["source_group_id"]]
        )
        if replay(undo_log)[0] != after:
            raise ValueError("Undo/redo replay failed.")
        identifier = spec["id"] + f"-s{n:02d}"
        case = {
            "id": identifier,
            "input": bundle["input"],
            "expectedOutput": prediction,
            "acceptableOutputs": alternatives,
            "metadata": {
                "split": source["split"],
                "source_group_id": source["source_group_id"],
                "source_id": source["id"],
                "language": source["language"],
                "episode_id": spec["id"],
                "episode_step": n,
                "actor": "synthetic",
                "history_origin": "executed_reconstruction",
                "kind": spec["kind"],
                "media_kind": source.get("media_kind", "recording"),
                "source_region": spec.get("source_region"),
                "decision_rationale": spec["rationale"],
                "context": bundle["context"],
                "input_sha256": session_hash(canonical_messages(bundle["input"])),
                "training_eligible": False,
                "review_status": "pending_independent_admission",
                "document_before_hash": session_hash(document),
                "document_after_hash": session_hash(after),
            },
            "reproduction": {"episode": f"episodes/{spec['id']}.json", "step": n},
        }
        cases.append(case)
        if document != after:
            history.append(committed_transition(document, after, prediction, n + 1))
            journal.append(
                compact_transaction(
                    document, after, len(journal), len(history), prediction["action"]
                )
            )
        document = after
    journal.append(
        {
            "kind": "end",
            "seq": len(journal),
            "revision": len(history),
            "document_hash": session_hash(document),
            "transactions": len(history),
        }
    )
    if replay(journal)[0] != document:
        raise ValueError("Whole episode replay failed.")
    # The source document is stored once. Rehydration restores the exact start event.
    journal[0].pop("document")
    write(
        output / "episodes" / f"{spec['id']}.json",
        {
            "spec": spec,
            "source_document": source["document_path"],
            "source_document_sha256": digest(ROOT / source["document_path"]),
            "journal": journal,
        },
    )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--max-episodes", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory for each candidate generation.")
    sources = load_sources(args.sources)
    args.output.mkdir(parents=True)
    if args.recipe:
        specs = read(args.recipe)["episodes"]
    else:
        specs = [
            e
            for source in sources.values()
            for e in mine_restorations(source, args.max_episodes)
        ]
    write(
        args.output / "recipe.json",
        {"version": "passive-dataset-v1", "episodes": specs},
    )
    from transformers import AutoTokenizer
    from passive_contract import PassiveContextBuilder

    registry = read(
        ROOT
        / "artifacts/datasets/nep-release-v1/comparison-tokenizers-v2/registry.json"
    )
    tokenizers = {
        c["id"]: AutoTokenizer.from_pretrained(ROOT / c["path"], local_files_only=True)
        for c in registry["candidates"]
    }
    builder = PassiveContextBuilder(tokenizers["qwen06"], budget_tokenizers=tokenizers)
    cases, excluded = [], []
    for spec in specs:
        try:
            cases.extend(
                build_episode(spec, sources[spec["source_id"]], builder, args.output)
            )
        except (ValueError, StopIteration) as error:
            excluded.append(
                {
                    "episode_id": spec["id"],
                    "reason": str(error),
                    "status": "not_admitted",
                }
            )
    for split in ("train", "dev"):
        jsonl(
            args.output / f"{split}-candidates.jsonl",
            [c for c in cases if c["metadata"]["split"] == split],
        )
    jsonl(args.output / "exclusions.jsonl", excluded)
    write(
        args.output / "census.json",
        {
            "status": "candidates_only_not_a_release",
            "rows": len(cases),
            "splits": dict(Counter(c["metadata"]["split"] for c in cases)),
            "languages": dict(Counter(c["metadata"]["language"] for c in cases)),
            "actions": dict(Counter(c["expectedOutput"]["action"] for c in cases)),
            "source_groups": dict(
                Counter(c["metadata"]["source_group_id"] for c in cases)
            ),
            "episodes": len({c["metadata"]["episode_id"] for c in cases}),
            "excluded_episodes": len(excluded),
            "training_eligible": 0,
            "test_rows": 0,
        },
    )
    print(json.dumps(read(args.output / "census.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
