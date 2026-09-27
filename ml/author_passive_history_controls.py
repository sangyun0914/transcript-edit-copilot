"""Bind reviewed source opportunities to small, independently reviewed history controls."""

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from passive_contract import canonical_messages, parse_snapshot, validate_prediction
from passive_dataset import ROOT, digest, episode_steps, load_sources, read, write
from replay_session import session_hash
from session_cases import document_records
from verify_passive_dataset import rows


def bound_reviews(batches, dependencies):
    reviewed = {}
    for batch in batches:
        directory = ROOT / batch["path"]
        files = [directory / f"{split}-candidates.jsonl" for split in ("train", "dev")]
        candidates = {case["id"]: case for path in files for case in rows(path)}
        recipe_path = directory / "recipe.json"
        dependencies[str(recipe_path.relative_to(ROOT))] = digest(recipe_path)
        specs = {spec["id"]: spec for spec in read(recipe_path)["episodes"]}
        for name in batch["reviews"]:
            review_path = ROOT / name
            dependencies[name] = digest(review_path)
            review = read(review_path)
            hashes = review["candidate_files_sha256"]
            if {ROOT / p for p in hashes} != set(files):
                raise ValueError("A review must bind both exact candidate files.")
            for path, expected in hashes.items():
                if digest(ROOT / path) != expected:
                    raise ValueError("Reviewed candidate bytes changed.")
                dependencies[path] = expected
            for decision in review["cases"]:
                case = candidates[decision["case_id"]]
                if decision["input_sha256"] != session_hash(
                    canonical_messages(case["input"])
                ):
                    raise ValueError("Review is bound to another predictor input.")
                if case["id"] in reviewed:
                    raise ValueError("Duplicate reviewed case identity.")
                if decision["decision"] not in {"admit", "exclude"}:
                    raise ValueError("Only resolved prior decisions can seed controls.")
                reviewed[case["id"]] = {
                    "case": case,
                    "decision": decision,
                    "spec": specs[case["metadata"]["episode_id"]],
                    "draft": batch["path"],
                    "review": name,
                    "review_sha256": dependencies[name],
                }
    return reviewed


def action_key(case, action):
    return case["metadata"]["source_id"], session_hash(action)


def evidence(item):
    return {
        "draft": item["draft"],
        "review": item["review"],
        "review_sha256": item["review_sha256"],
        "case_id": item["case"]["id"],
        "input_sha256": item["decision"]["input_sha256"],
    }


def author(config):
    dependencies = {name: digest(ROOT / name) for name in config["sources"]}
    sources = load_sources([ROOT / name for name in config["sources"]])
    prior = bound_reviews(config["restoration_drafts"], dependencies)
    native = bound_reviews(config["native_drafts"], dependencies)
    native_actions, rejected_actions = {}, set()
    for item in native.values():
        case, spec = item["case"], item["spec"]
        action = case["expectedOutput"]
        if (
            spec["kind"] != "native_source_repair"
            or spec.get("corruptions")
            or len(episode_steps(spec)) != 1
            or case["input"]["recent_edits"]
            or action["action"]
            not in {"correct_text", "split_paragraph", "merge_paragraph"}
        ):
            raise ValueError("Continuation evidence must be a fresh native positive.")
        key = action_key(case, action)
        if item["decision"]["decision"] == "admit":
            native_actions[key] = item
        else:
            rejected_actions.add(key)
    for case_id in config.get("required_excluded_native_case_ids", []):
        if native[case_id]["decision"]["decision"] != "exclude":
            raise ValueError("A required semantic rejection was not preserved.")

    empty, continuation = defaultdict(list), defaultdict(list)
    deferred = []
    for item in prior.values():
        case, decision, spec = item["case"], item["decision"], item["spec"]
        if case["expectedOutput"]["action"] != "no_action":
            continue
        steps = episode_steps(spec)
        if (
            spec["kind"] != "source_derived_restoration"
            or steps[-1]["action"] != "no_action"
        ):
            raise ValueError("Controls require a complete original restoration recipe.")
        if case["metadata"]["episode_step"] != len(steps) - 1:
            raise ValueError("The original reviewed stop is not the terminal state.")
        source = sources[spec["source_id"]]
        if any(
            case["metadata"][key] != source[key]
            for key in ("source_group_id", "split", "language")
        ):
            raise ValueError("Original case changed its registered source role.")
        if decision.get("duplicate_of_case_id"):
            continue
        group = source["source_group_id"]
        if decision["decision"] == "admit":
            empty[group].append(item)
            continue
        suggestions = list(decision.get("suggested_native_fixes", []))
        if decision.get("suggested_native_fix"):
            suggestions.append(decision["suggested_native_fix"])
        for action in suggestions:
            key = action_key(case, action)
            if key in rejected_actions:
                deferred.append(
                    {
                        "case_id": case["id"],
                        "action": action,
                        "reason": "native_secondary_semantic_rejection",
                    }
                )
                continue
            if key not in native_actions:
                deferred.append(
                    {
                        "case_id": case["id"],
                        "action": action,
                        "reason": "no_bound_native_secondary_admission",
                    }
                )
                continue
            prefix_ids = [spec["id"] + f"-s{n:02d}" for n in range(len(steps) - 1)]
            if not all(prior[p]["decision"]["decision"] == "admit" for p in prefix_ids):
                deferred.append(
                    {
                        "case_id": case["id"],
                        "action": action,
                        "reason": "prefix_not_already_admitted",
                    }
                )
                continue
            validate_prediction(
                json.dumps(action, ensure_ascii=False),
                parse_snapshot(case["input"]),
                None,
                None,
                "stop",
            )
            continuation[group].append((item, action, native_actions[key]))

    specs = []
    for group in sorted(set(empty) | set(continuation)):
        if empty[group]:
            item = min(empty[group], key=lambda x: x["case"]["id"])
            case, original = item["case"], item["spec"]
            fingerprint = session_hash([case["id"], "empty_history"])[:12]
            specs.append(
                {
                    "id": f"{original['source_id']}-history-empty-{fingerprint}",
                    "source_id": original["source_id"],
                    "kind": "source_history_empty_control",
                    "source_region": copy.deepcopy(original["source_region"]),
                    "corruptions": [],
                    "steps": [copy.deepcopy(episode_steps(original)[-1])],
                    "rationale": "Revisit the previously admitted stopping opportunity on the original source with no edit history. The complete fresh context requires independent review.",
                    "history_control": {
                        "variant": "empty_history_stop",
                        "original": evidence(item),
                        "row_roles": [
                            {
                                "episode_step": 0,
                                "role": "novel_control",
                                "original_case_id": case["id"],
                                "required_admission": "fresh_independent_review",
                            }
                        ],
                    },
                }
            )
        if continuation[group]:
            item, action, native_item = min(
                continuation[group],
                key=lambda x: (
                    x[1]["action"] != "correct_text",
                    -x[0]["case"]["metadata"]["episode_step"],
                    x[0]["case"]["id"],
                    session_hash(x[1]),
                ),
            )
            case, original = item["case"], item["spec"]
            source = sources[original["source_id"]]
            records, mapping = document_records(source["document"])
            indexed = {record["id"]: record for record in records}
            visible = {record["id"]: record for record in case["input"]["items"]}
            if any(indexed[t]["text"] != visible[t]["text"] for t in action["target"]):
                raise ValueError(
                    "Terminal target is not restored original-source text."
                )
            steps = episode_steps(original)[:-1]
            final_step = {
                "focus_uuid": episode_steps(original)[-1]["focus_uuid"],
                "action": action["action"],
                "target_uuids": [mapping[str(t)]["uuid"] for t in action["target"]],
            }
            if "text" in action:
                final_step["text"] = action["text"]
            steps.append(final_step)
            fingerprint = session_hash([case["id"], action, "continue_after_repairs"])[
                :12
            ]
            specs.append(
                {
                    "id": f"{original['source_id']}-history-continue-{fingerprint}",
                    "source_id": original["source_id"],
                    "kind": "source_history_action_continuation",
                    "source_region": copy.deepcopy(original["source_region"]),
                    "corruptions": copy.deepcopy(original["corruptions"]),
                    "steps": steps,
                    "rationale": "Replay the original injected spelling repairs, then perform the independently supported native edit still visible at the original stop focus. Prefix candidates are reconstruction evidence only and must be excluded; no stop is appended.",
                    "history_control": {
                        "variant": "action_after_restoration",
                        "original": evidence(item),
                        "native_secondary": evidence(native_item),
                        "row_roles": [
                            {
                                "episode_step": n,
                                "role": "represented_prefix"
                                if n < len(steps) - 1
                                else "novel_control",
                                "original_case_id": original["id"] + f"-s{n:02d}",
                                "required_admission": "exclude"
                                if n < len(steps) - 1
                                else "fresh_independent_review",
                            }
                            for n in range(len(steps))
                        ],
                    },
                }
            )
    if len(specs) > 40:
        raise ValueError("The authorized bound is at most 40 novel states.")
    counts = Counter(
        (sources[s["source_id"]]["source_group_id"], s["history_control"]["variant"])
        for s in specs
    )
    if any(value > 1 for value in counts.values()):
        raise ValueError("More than one control of a kind in one source group.")
    for name, expected in dependencies.items():
        if digest(ROOT / name) != expected:
            raise ValueError("An authoring dependency changed during selection.")
    return {
        "version": "passive-history-controls-v1",
        "scope": "Candidate controls only; natural source opportunities retain their original group, split and region. No fabricated history or appended stopping state.",
        "episodes": specs,
        "novel_state_count": len(specs),
        "novel_state_scope": "Prospective additional admissions, not a count of unseen inputs: continuation terminals re-adjudicate the original excluded stopping input; empty-history controls require a fresh canonical comparison and semantic review.",
        "represented_prefix_count": sum(len(s["steps"]) - 1 for s in specs),
        "dependencies_sha256": dependencies,
        "deferred_findings": deferred,
        "unrepresented_groups": sorted(
            {s["source_group_id"] for s in sources.values()} - {g for g, _ in counts}
        ),
    }


def annotate(directory):
    directory = (ROOT / directory).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("A control draft must be inside this project.")
    specs = {s["id"]: s for s in read(directory / "recipe.json")["episodes"]}
    prior = {}
    for spec in specs.values():
        original = spec["history_control"]["original"]
        if digest(ROOT / original["review"]) != original["review_sha256"]:
            raise ValueError("The original review changed before annotation.")
        if original["draft"] not in prior:
            base = ROOT / original["draft"]
            prior[original["draft"]] = {
                c["id"]: c
                for split in ("train", "dev")
                for c in rows(base / f"{split}-candidates.jsonl")
            }
    roles = []
    for split in ("train", "dev"):
        path = directory / f"{split}-candidates.jsonl"
        cases = rows(path)
        for case in cases:
            spec = specs[case["metadata"]["episode_id"]]
            control = spec["history_control"]
            role = control["row_roles"][case["metadata"]["episode_step"]]
            old = prior[control["original"]["draft"]][role["original_case_id"]]
            same_input = canonical_messages(case["input"]) == canonical_messages(
                old["input"]
            )
            if role["role"] == "represented_prefix" and (
                not same_input or case["expectedOutput"] != old["expectedOutput"]
            ):
                raise ValueError("Represented prefix differs from its original case.")
            if (
                control["variant"] == "action_after_restoration"
                and role["role"] == "novel_control"
                and not same_input
            ):
                raise ValueError("Continuation changed the original stopping input.")
            if (
                control["variant"] == "empty_history_stop"
                and case["input"]["recent_edits"]
            ):
                raise ValueError("An empty-history control contains history.")
            case["metadata"]["history_control"] = {
                **role,
                "variant": control["variant"],
                "original_draft": control["original"]["draft"],
                "same_canonical_input_as_original": same_input,
            }
            roles.append(
                {
                    "case_id": case["id"],
                    "input_sha256": case["metadata"]["input_sha256"],
                    **case["metadata"]["history_control"],
                }
            )
        path.write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
        )
    write(
        directory / "row-roles.json",
        {
            "candidate_files_sha256": {
                str(
                    (directory / f"{split}-candidates.jsonl").relative_to(ROOT)
                ): digest(directory / f"{split}-candidates.jsonl")
                for split in ("train", "dev")
            },
            "counts": dict(Counter(r["role"] for r in roles)),
            "rows": roles,
        },
    )
    return Counter(r["role"] for r in roles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--annotate-draft", type=Path)
    args = parser.parse_args()
    if args.annotate_draft:
        print(json.dumps(dict(annotate(args.annotate_draft))))
    else:
        if not args.config or not args.output or args.output.exists():
            raise ValueError("Provide a config and new output recipe path.")
        result = author(read(args.config))
        result["config_sha256"] = digest(args.config)
        write(args.output, result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "novel_state_count",
                        "represented_prefix_count",
                        "unrepresented_groups",
                    )
                }
            )
        )
