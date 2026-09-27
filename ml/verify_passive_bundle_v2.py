"""Independent readback of frozen passive payloads, admission, tokens and sampling.

This verifier never freezes, edits candidates, trains, or inspects held-out bodies.
Its optional output is a new audit file outside the frozen directory.
"""

import argparse
import copy
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from passive_contract import (
    ALLOWED_ACTIONS,
    PROMPT,
    PassiveContextBuilder,
    PassiveFeedbackLedger,
    canonical_messages,
)
from passive_dataset import RESERVED, ROOT, digest
from passive_formats_v2 import (
    CONTEXT_LIMIT,
    INPUT_LIMIT,
    OUTPUT_LIMIT,
    REWRITE_PROMPT,
    messages,
    parse,
    serialize,
)
from passive_training_v2 import training_epoch
from replay_session import replay, session_hash
from verify_passive_dataset import tokenizers

VERSION = "passive-completion-independent-readback-v2"
PARENT = ROOT / "artifacts/datasets/nep-passive-v1/frozen"
WEIGHT_FIELDS = {"sampling_weight", "sampling_cluster"}
ADMISSION_FIELDS = WEIGHT_FIELDS | {
    "training_eligible",
    "review_status",
    "review_file",
    "review_reason",
    "release_stratum",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def project_path(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), f"Evidence escapes the project: {value}")
    return path


def version(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class PhysicalHashes:
    def __init__(self):
        self.checked = {}

    def check(self, path, expected):
        path = Path(path).resolve()
        current = version(path)
        if path not in self.checked:
            actual = digest(path)
            require(version(path) == current, f"File changed while hashing: {path}")
            self.checked[path] = actual, current
        actual, original = self.checked[path]
        require(current == original, f"File changed during readback: {path}")
        require(actual == expected, f"Physical SHA256 mismatch: {path}")

    def stable(self):
        for path, (_, original) in self.checked.items():
            require(version(path) == original, f"File changed during readback: {path}")


def verify_inventory(directory, manifest, hashes):
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    require(
        files == set(manifest["files_sha256"]) | {"manifest.json"},
        "Frozen physical inventory differs from the manifest, including unlisted files.",
    )
    for name, expected in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        require(
            path.is_relative_to(directory), "Frozen payload path escapes its release."
        )
        hashes.check(path, expected)


def independent_weights(cases):
    """Calculate component membership and masses independently of the packager."""
    by_split = defaultdict(list)
    for case in cases:
        by_split[case["metadata"]["split"]].append(case)
    expected, masses = {}, {}
    for split, split_cases in by_split.items():
        by_source = defaultdict(list)
        for case in split_cases:
            meta = case["metadata"]
            by_source[
                (meta["language"], meta["source_group_id"], meta["source_id"])
            ].append(case)
        opportunities = defaultdict(lambda: defaultdict(list))
        for (language, group, _), members in by_source.items():
            regions = [c["metadata"]["source_region"] for c in members]
            if None in regions:
                require(
                    all(
                        c["metadata"]["media_kind"] == "authored_symbolic_timeline"
                        for c in members
                    ),
                    "A recording has no bounded source opportunity.",
                )
                opportunities[language][group].append(members)
                continue
            pending = sorted(members, key=lambda c: c["metadata"]["source_region"][0])
            connected, right_edge = [], None
            for case in pending:
                start, stop = case["metadata"]["source_region"]
                require(
                    all(
                        type(n) in (int, float) and math.isfinite(n)
                        for n in (start, stop)
                    )
                    and start >= 0
                    and stop > start,
                    "Invalid sampling interval.",
                )
                if right_edge is None or start >= right_edge:
                    if connected:
                        opportunities[language][group].append(connected)
                    connected, right_edge = [case], stop
                else:
                    connected.append(case)
                    right_edge = max(right_edge, stop)
            if connected:
                opportunities[language][group].append(connected)
        label_members = defaultdict(set)
        for language, groups in opportunities.items():
            for group, components in groups.items():
                for component in components:
                    weight = (
                        1
                        / len(opportunities)
                        / len(groups)
                        / len(components)
                        / len(component)
                    )
                    labels = {c["metadata"]["sampling_cluster"] for c in component}
                    require(
                        len(labels) == 1,
                        "Dependent rows have different sampling clusters.",
                    )
                    label = next(iter(labels))
                    members = {c["id"] for c in component}
                    require(
                        not label_members[label],
                        "Independent opportunities share a cluster ID.",
                    )
                    label_members[label] = members
                    for case in component:
                        expected[case["id"]] = weight
        language_mass = defaultdict(float)
        for case in split_cases:
            meta = case["metadata"]
            actual = meta["sampling_weight"]
            require(
                type(actual) in (int, float) and math.isfinite(actual) and actual > 0,
                "Nonpositive or nonfinite sampling weight.",
            )
            require(
                math.isclose(
                    actual, expected[case["id"]], rel_tol=1e-12, abs_tol=1e-15
                ),
                f"Incorrect independent opportunity weight: {case['id']}",
            )
            language_mass[meta["language"]] += actual
        require(
            math.isclose(sum(language_mass.values()), 1, rel_tol=1e-12),
            "Split mass is not one.",
        )
        masses[split] = dict(language_mass)
    return expected, masses


def independent_draws(cases, count, seed):
    rng = random.Random(seed)
    weights = [c["metadata"]["sampling_weight"] for c in cases]
    total = sum(weights)
    selected = []
    for _ in range(count):
        draw, accumulated = rng.random() * total, 0.0
        for case, weight in zip(cases, weights, strict=True):
            accumulated += weight
            if draw < accumulated:
                selected.append(case["id"])
                break
        else:
            raise ValueError("Sampler rounding produced no selected case.")
    return selected


def semantic_fields_equal(left, right):
    return all(
        left[k] == right[k]
        for k in ("id", "input", "expectedOutput", "acceptableOutputs")
    )


def verify_admission(manifest, cases, excluded, hashes):
    dependencies = manifest["dependencies_sha256"]
    configurations = []
    for name in dependencies:
        if name.endswith(".json") and "freeze" in Path(name).name:
            value = read(project_path(name))
            if "parent_release" in value and "supplements" in value:
                configurations.append((name, value))
    require(
        len(configurations) == 1, "Expected one bound completion freeze configuration."
    )
    config_name, config = configurations[0]
    require(
        project_path(config["parent_release"]) == PARENT.resolve(),
        "Unexpected parent release.",
    )
    frozen = {c["id"]: c for c in cases}
    parent_rows = [
        c for split in ("train", "dev") for c in rows(PARENT / f"{split}-cases.jsonl")
    ]
    expected_ids = set()
    for original in parent_rows:
        require(original["id"] in frozen, "A parent case disappeared.")
        actual = frozen[original["id"]]
        require(
            semantic_fields_equal(original, actual),
            f"Parent predictor/reference value changed: {original['id']}",
        )
        require(
            actual["metadata"]["release_stratum"] == "curriculum_v1",
            "Parent stratum changed.",
        )
        for key, value in original["metadata"].items():
            if key not in WEIGHT_FIELDS:
                require(
                    actual["metadata"].get(key) == value,
                    f"Parent metadata changed: {original['id']}:{key}",
                )
        require(
            actual["reproduction"] == original["reproduction"],
            "Parent reproduction changed.",
        )
        expected_ids.add(original["id"])
    expected_exclusions, supplement_versions, session_episodes = {}, Counter(), {}
    checkpoint_observations = 0
    for spec in config["supplements"]:
        directory = project_path(spec["directory"])
        verification_path, review_path = (
            project_path(spec["verification"]),
            project_path(spec["review"]),
        )
        for path in (verification_path, review_path):
            name = str(path.relative_to(ROOT))
            require(
                name in dependencies,
                "An admission proof is outside the frozen dependencies.",
            )
            hashes.check(path, dependencies[name])
        verification, review = read(verification_path), read(review_path)
        candidate_hashes = {
            str((directory / f"{split}-candidates.jsonl").relative_to(ROOT)): digest(
                directory / f"{split}-candidates.jsonl"
            )
            for split in ("train", "dev")
        }
        require(
            review["candidate_files_sha256"] == candidate_hashes,
            "Semantic review binds different candidate files.",
        )
        for name, expected_hash in review.get("review_evidence_sha256", {}).items():
            require(
                dependencies.get(name) == expected_hash,
                "Combined semantic review loses its component evidence binding.",
            )
            hashes.check(project_path(name), expected_hash)
        candidate_rows = [
            c
            for split in ("train", "dev")
            for c in rows(directory / f"{split}-candidates.jsonl")
        ]
        decisions = {d["case_id"]: d for d in review["cases"]}
        require(
            len(decisions) == len(review["cases"]) == len(candidate_rows),
            "Review census/identity duplication.",
        )
        require(
            set(decisions) == {c["id"] for c in candidate_rows},
            "Review misses or invents a candidate.",
        )
        expected_outcomes = {
            (c["id"], index)
            for c in candidate_rows
            for index in range(1 + len(c["acceptableOutputs"]))
        }
        outcomes = verification["outcomes"]
        require(
            verification["status"] == "passed" and verification["errors"] == [],
            "Supplement proof failed.",
        )
        require(
            len(outcomes) == len(expected_outcomes)
            and {(o["case_id"], o["reference_index"]) for o in outcomes}
            == expected_outcomes
            and all(o["status"] == "passed" for o in outcomes),
            "Supplement proof omits a primary or alternative.",
        )
        parity_count = len(candidate_rows)
        if verification["version"] == "passive-source-checkpoints-v2":
            exclusion_path = directory / "exclusions.jsonl"
            relative = str(exclusion_path.relative_to(ROOT))
            expected_hash = verification["dependencies_sha256"].get(relative)
            require(
                expected_hash is not None
                and verification["candidate_hashes"].get("exclusions.jsonl")
                == expected_hash
                and dependencies.get(relative) == expected_hash,
                "Unresolved checkpoint dispositions lost their physical binding.",
            )
            hashes.check(exclusion_path, expected_hash)
            unresolved = rows(exclusion_path)
            unresolved_ids = {e["case_id"] for e in unresolved}
            candidate_ids = {c["id"] for c in candidate_rows}
            observed_ids = verification["observed_case_ids"]
            recipe_ids = {
                e["id"] + "-s00" for e in read(directory / "recipe.json")["episodes"]
            }
            require(
                len(unresolved_ids) == len(unresolved)
                and not unresolved_ids & candidate_ids
                and len(set(observed_ids)) == len(observed_ids)
                and set(observed_ids) == candidate_ids | unresolved_ids == recipe_ids,
                "Checkpoint observed/candidate/unresolved partition changed.",
            )
            require(
                verification["scheduled_checkpoints"]
                == verification["observed_contexts"]
                == len(observed_ids)
                and verification["unresolved_or_unsupported"] == len(unresolved)
                and set(verification["excluded_case_ids"]) == unresolved_ids,
                "Checkpoint observation/exclusion census changed.",
            )
            for row in unresolved:
                require(
                    row["case_id"] not in frozen
                    and row["status"] in {"unresolved", "unsupported"}
                    and bool(row["reason"].strip())
                    and bool(row["input_sha256"]),
                    "An unresolved observation was relabeled or silently admitted.",
                )
                expected_exclusions[row["case_id"]] = {
                    **row,
                    "decision": "exclude",
                    "exclusion_stage": "checkpoint_qualification",
                    "release_stratum": "source_first_checkpoints",
                    "directory": str(directory.relative_to(ROOT)),
                }
            parity_count = len(observed_ids)
            checkpoint_observations += parity_count
        require(
            verification["candidates"] == len(candidate_rows)
            and verification["actual_serving_message_parity_checks"] == parity_count
            and verification["replayed_episodes"]
            == len({c["metadata"]["episode_id"] for c in candidate_rows})
            and verification["accepted_outcomes_exported"] == len(expected_outcomes),
            "Supplement parity/replay/export census differs.",
        )
        require(
            len(verification["case_ids"]) == len(candidate_rows)
            and set(verification["case_ids"]) == {c["id"] for c in candidate_rows},
            "Mechanical proof has a different candidate identity census.",
        )
        if verification["version"] == "passive-sessions-v2":
            require(
                verification["full_ledger_as_of_parity_checks"] == len(candidate_rows),
                "Session proof omits as_of checks.",
            )
        for name, expected_hash in verification["dependencies_sha256"].items():
            require(
                dependencies.get(name) == expected_hash,
                "Frozen bundle loses a mechanical evidence binding.",
            )
        for original in candidate_rows:
            decision = decisions[original["id"]]
            require(
                decision["input_sha256"] == original["metadata"]["input_sha256"],
                "Semantic decision input changed.",
            )
            require(
                decision["decision"] in {"admit", "exclude"},
                "Unresolved decision entered freeze.",
            )
            if decision["decision"] == "exclude":
                require(
                    original["id"] not in frozen,
                    "Excluded supplement row was admitted.",
                )
                expected_exclusions[original["id"]] = {
                    **decision,
                    "directory": spec["directory"],
                    "exclusion_stage": "independent_candidate_review",
                    "release_stratum": spec["stratum"],
                }
                continue
            require(
                original["id"] in frozen and original["id"] not in expected_ids,
                "Missing or duplicate admitted identity.",
            )
            actual = frozen[original["id"]]
            require(
                semantic_fields_equal(original, actual),
                "Supplement input/reference changed after review.",
            )
            for key, value in original["metadata"].items():
                if key not in ADMISSION_FIELDS:
                    require(
                        actual["metadata"].get(key) == value,
                        f"Supplement provenance changed: {key}",
                    )
            require(
                actual["metadata"]["review_file"] == spec["review"]
                and actual["metadata"]["review_reason"] == decision["reason"],
                "Frozen review attribution changed.",
            )
            require(
                actual["reproduction"]
                == {
                    **original["reproduction"],
                    "candidate_directory": spec["directory"],
                },
                "Frozen reconstruction pointer changed.",
            )
            expected_ids.add(original["id"])
            supplement_versions[verification["version"]] += 1
            if verification["version"] == "passive-sessions-v2":
                path = directory / original["reproduction"]["episode"]
                session_episodes.setdefault(path, []).append(actual)
    require(
        set(frozen) == expected_ids,
        "Release contains an unreviewed or unregistered extra row.",
    )
    require(
        len(excluded) == len(expected_exclusions)
        and {e["case_id"]: e for e in excluded} == expected_exclusions,
        "Frozen supplement exclusions differ from independent decisions.",
    )
    return (
        config_name,
        config,
        len(parent_rows),
        dict(supplement_versions),
        session_episodes,
        checkpoint_observations,
    )


def check_encoded(values, prefix, completion):
    require(
        values
        == {
            "input_ids": prefix + completion,
            "attention_mask": [1] * (len(prefix) + len(completion)),
            "labels": [-100] * len(prefix) + completion,
            "completion_start": len(prefix),
        },
        "Token IDs, completion boundary, attention mask or loss labels differ from exact retokenization.",
    )
    require(
        0 < len(prefix) <= INPUT_LIMIT
        and 0 < len(completion) <= OUTPUT_LIMIT
        and len(prefix) + len(completion) <= CONTEXT_LIMIT,
        "Common input/output reserve exceeded.",
    )


def check_mass(actual, contributions, count):
    total = sum(contributions["language"].values())
    require(actual["unique_rows"] == count, "Token-mass row census differs.")
    require(
        math.isclose(
            actual["mean_completion_tokens_under_sampler"], total, rel_tol=1e-12
        ),
        "Expected completion length differs.",
    )
    observed = actual["expected_primary_loss_token_fractions"]
    require(set(observed) == set(contributions), "Missing token-mass category.")
    for category, values in contributions.items():
        require(set(observed[category]) == set(values), "Missing token-mass group.")
        for key, value in values.items():
            require(
                math.isclose(
                    observed[category][key], value / total, rel_tol=1e-12, abs_tol=1e-15
                ),
                "Reported token fractions differ from actual completions and row weights.",
            )


def verify_tokenized(directory, cases, matched, loaded):
    counts, maxima, reference_encodings, loader_checks = {}, defaultdict(Counter), 0, []
    mass_artifact = read(directory / "training-token-mass.json")["views"]
    expected_mass_views = set()
    for name, tokenizer in loaded.items():
        prefix_cache = {}
        for case in cases:
            counts.setdefault(case["id"], {})[name] = {}
            for fmt in ("json", "rewrite"):
                prepared = messages(case["input"], fmt)
                prefix = tokenizer.apply_chat_template(
                    prepared,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prefix_cache[(case["id"], fmt)] = prefix
                counts[case["id"]][name][fmt] = len(prefix)
        for variant, fmt in (
            ("json-curriculum", "json"),
            ("json", "json"),
            ("rewrite", "rewrite"),
        ):
            view = cases if variant == "json-curriculum" else matched
            train_cases, train_primary = [], {}
            for split in ("train", "dev"):
                selected = [c for c in view if c["metadata"]["split"] == split]
                encoded = rows(
                    directory / "tokenized" / name / f"{variant}-{split}.jsonl.gz"
                )
                require(
                    len(encoded) == len(selected), "Tokenized view omits or adds cases."
                )
                contributions = defaultdict(Counter)
                for record, case in zip(encoded, selected, strict=True):
                    require(
                        set(record)
                        == {
                            "id",
                            "split",
                            "sampling_weight",
                            "input_sha256",
                            "messages",
                            "references",
                            "primary",
                            "alternative_completion_ids",
                        },
                        "Unexpected tokenized payload schema.",
                    )
                    require(
                        record["id"] == case["id"] and record["split"] == split,
                        "Tokenized identity/order/split changed.",
                    )
                    require(
                        record["input_sha256"] == case["metadata"]["input_sha256"]
                        and record["sampling_weight"]
                        == case["metadata"]["sampling_weight"],
                        "Tokenized input hash/weight changed.",
                    )
                    require(
                        record["messages"] == messages(case["input"], fmt),
                        "Tokenized messages differ from exact canonical evidence.",
                    )
                    actions = [case["expectedOutput"], *case["acceptableOutputs"]]
                    references = [
                        serialize(case["input"], action, fmt) for action in actions
                    ]
                    require(
                        record["references"] == references,
                        "Serialized primary/alternatives changed.",
                    )
                    prefix = prefix_cache[(case["id"], fmt)]
                    completions = [
                        tokenizer.encode(text, add_special_tokens=False)
                        + [tokenizer.eos_token_id]
                        for text in references
                    ]
                    require(
                        tokenizer.eos_token_id is not None,
                        "Pinned tokenizer lacks EOS.",
                    )
                    for completion in completions:
                        require(
                            0 < len(completion) <= OUTPUT_LIMIT
                            and len(prefix) + len(completion) <= CONTEXT_LIMIT,
                            "A reference does not fit the common reserve.",
                        )
                    check_encoded(record["primary"], prefix, completions[0])
                    require(
                        record["alternative_completion_ids"] == completions[1:],
                        "Alternative completion tokens changed.",
                    )
                    reference_encodings += len(actions)
                    maxima[name][f"{fmt}_input"] = max(
                        maxima[name][f"{fmt}_input"], len(prefix)
                    )
                    maxima[name][f"{fmt}_output"] = max(
                        maxima[name][f"{fmt}_output"], max(map(len, completions))
                    )
                    if split == "train":
                        train_cases.append(case)
                        train_primary[case["id"]] = record["primary"]
                        contribution = case["metadata"]["sampling_weight"] * len(
                            completions[0]
                        )
                        for category in (
                            "language",
                            "source_group_id",
                            "episode_id",
                            "kind",
                            "release_stratum",
                        ):
                            contributions[category][case["metadata"][category]] += (
                                contribution
                            )
                if split == "train":
                    key = f"{name}:{variant}"
                    expected_mass_views.add(key)
                    check_mass(mass_artifact[key], contributions, len(selected))
            seed = "independent-readback-weighted-loader-v2"
            expected_ids = independent_draws(train_cases, len(train_cases), seed)
            sampled = training_epoch(
                directory, name, variant, seed=seed, samples=len(train_cases)
            )
            require(
                [row["case_id"] for row in sampled] == expected_ids,
                "Actual loader did not consume frozen weighted probabilities.",
            )
            for row in sampled:
                require(
                    row == {"case_id": row["case_id"], **train_primary[row["case_id"]]},
                    "Actual loader changed encoded training content.",
                )
            loader_checks.append(
                {
                    "candidate": name,
                    "view": variant,
                    "rows": len(sampled),
                    "selected_case_ids_sha256": session_hash(expected_ids),
                }
            )
        print(
            f"Verified {name}: exact token payloads, masks, token mass and all three weighted training views.",
            flush=True,
        )
    require(
        set(mass_artifact) == expected_mass_views,
        "Token-mass report contains a missing or unknown view.",
    )
    return counts, dict(maxima), reference_encodings, loader_checks


def verify(directory, output=None):
    directory = Path(directory).resolve()
    require(
        directory.is_relative_to(ROOT), "Frozen bundle must stay inside this project."
    )
    if output is not None:
        output = Path(output).resolve()
        require(
            not output.is_relative_to(directory) and not output.exists(),
            "Use a new audit file outside the immutable release.",
        )
    manifest_path = directory / "manifest.json"
    manifest, hashes = read(manifest_path), PhysicalHashes()
    require(
        manifest["version"] == "passive-dataset-completion-v2"
        and manifest["status"] == "frozen_supported_scope",
        "Unexpected frozen bundle schema/status.",
    )
    hashes.check(manifest_path, digest(manifest_path))
    verify_inventory(directory, manifest, hashes)
    for name, expected_hash in manifest["dependencies_sha256"].items():
        hashes.check(project_path(name), expected_hash)
    parent_manifest = read(PARENT / "manifest.json")
    hashes.check(PARENT / "manifest.json", manifest["parent_manifest_sha256"])
    verify_inventory(PARENT, parent_manifest, hashes)
    for name, expected_hash in parent_manifest["dependencies_sha256"].items():
        hashes.check(project_path(name), expected_hash)
    for field in ("test_protocol", "coverage_audit"):
        path = project_path(manifest[field])
        require(
            str(path.relative_to(ROOT)) in manifest["dependencies_sha256"],
            f"Unbound {field}.",
        )
    protocol = read(project_path(manifest["test_protocol"]))
    require(
        protocol["generated_cases"] == 0
        and protocol["reserved_test_bodies_opened_by_this_auditor"] is False,
        "A reserved test was generated or exposed before its future binding.",
    )
    print(
        f"Rehashed {len(hashes.checked)} unique physical files, including the parent release.",
        flush=True,
    )
    by_split = {
        split: rows(directory / f"{split}-cases.jsonl") for split in ("train", "dev")
    }
    cases = by_split["train"] + by_split["dev"]
    excluded = rows(directory / "exclusions.jsonl")
    ids, signatures, groups, semantic_references = set(), set(), {}, 0
    for split, values in by_split.items():
        for case in values:
            meta = case["metadata"]
            require(case["id"] not in ids, "Duplicate frozen case identity.")
            ids.add(case["id"])
            require(
                meta["split"] == split
                and meta["training_eligible"] is (split == "train"),
                "Training/development eligibility changed.",
            )
            require(
                meta["source_group_id"] not in RESERVED,
                "Reserved source leaked into the release.",
            )
            require(
                groups.setdefault(meta["source_group_id"], split) == split,
                "Source family crosses train/development.",
            )
            if meta["source_group_id"] == "shuka-local-recording":
                require(
                    split == "dev", "Shuka source left its assigned development split."
                )
            require(
                meta["review_status"] == "independent_observable_input_review",
                "Unadmitted row in release.",
            )
            signature = session_hash(canonical_messages(case["input"]))
            require(
                signature == meta["input_sha256"] and signature not in signatures,
                "Stale or duplicate canonical input.",
            )
            signatures.add(signature)
            require(
                messages(case["input"], "json")[1]
                == messages(case["input"], "rewrite")[1],
                "Output format changes predictor evidence.",
            )
            for action in [case["expectedOutput"], *case["acceptableOutputs"]]:
                require(
                    action["action"] in ALLOWED_ACTIONS,
                    "Gated audio action entered passive data.",
                )
                for fmt in ("json", "rewrite"):
                    require(
                        parse(
                            serialize(case["input"], action, fmt), case["input"], fmt
                        ).model_dump()
                        == action,
                        "Reference JSON/Rewrite semantic roundtrip failed.",
                    )
                semantic_references += 1
    require(
        groups == read(directory / "source-groups.json"),
        "Frozen source-group map differs from rows.",
    )
    (
        config_name,
        config,
        parent_count,
        strata,
        session_episodes,
        checkpoint_observations,
    ) = verify_admission(manifest, cases, excluded, hashes)
    require(
        manifest["release_id"] == config["release_id"],
        "Release identity differs from its bound configuration.",
    )
    _, masses = independent_weights(cases)
    for split, selected in by_split.items():
        exported = rows(directory / f"{split}-messages.jsonl")
        require(len(exported) == len(selected), "Message export row count differs.")
        for row, case in zip(exported, selected, strict=True):
            expected = canonical_messages(case["input"])
            if split == "train":
                expected += [
                    {
                        "role": "assistant",
                        "content": serialize(
                            case["input"], case["expectedOutput"], "json"
                        ),
                    }
                ]
            require(
                row
                == {
                    "id": case["id"],
                    "messages": expected,
                    "sampling_weight": case["metadata"]["sampling_weight"],
                },
                "Frozen message bytes or weights differ from semantic cases.",
            )
    require(
        (directory / "prompt-json.txt").read_text() == PROMPT + "\n"
        and (directory / "prompt-rewrite.txt").read_text() == REWRITE_PROMPT + "\n",
        "Prompt bytes changed.",
    )
    matched_by_split = {
        s: rows(directory / "matched" / f"{s}-cases.jsonl") for s in ("train", "dev")
    }
    matched = matched_by_split["train"] + matched_by_split["dev"]
    by_id = {c["id"]: c for c in cases}
    require(len({c["id"] for c in matched}) == len(matched), "Duplicate matched input.")
    for split, values in matched_by_split.items():
        for case in values:
            require(
                case["id"] in by_id and case["metadata"]["split"] == split,
                "Matched view invents or moves a case.",
            )
            original = copy.deepcopy(by_id[case["id"]])
            for field in WEIGHT_FIELDS:
                original["metadata"][field] = case["metadata"][field]
            require(
                case == original,
                "Matched view changed evidence, references or provenance.",
            )
    _, matched_masses = independent_weights(matched)
    loaded_dependencies = {}
    loaded = tokenizers(loaded_dependencies)
    require(len(loaded) == 5, "Expected all five pinned tokenizer candidates.")
    for name, expected_hash in loaded_dependencies.items():
        require(
            manifest["dependencies_sha256"].get(name) == expected_hash,
            "Tokenizer dependency is outside the frozen manifest.",
        )
    counts, maxima, reference_encodings, loaders = verify_tokenized(
        directory, cases, matched, loaded
    )
    eligible = {
        c["id"]
        for c in cases
        if all(
            n <= INPUT_LIMIT
            for formats in counts[c["id"]].values()
            for n in formats.values()
        )
    }
    require(
        {c["id"] for c in matched} == eligible,
        "Matched subset was not selected solely by the common input budget.",
    )
    eligibility = read(directory / "matched/eligibility.json")
    require(
        eligibility["qualified"]
        == dict(Counter(c["metadata"]["split"] for c in matched))
        and eligibility["measured_maxima"] == maxima,
        "Matched budget census/maxima differs from actual encodings.",
    )
    overflow = {e["case_id"]: e for e in eligibility["excluded"]}
    require(
        len(overflow) == len(eligibility["excluded"])
        and set(overflow) == ids - eligible,
        "Matched overflow list differs from actual input budgets.",
    )
    for identity, evidence in overflow.items():
        require(
            evidence["input_token_counts"] == counts[identity]
            and evidence["reason"] == "input_budget_overflow_no_reference_used"
            and evidence["retained_in_json_curriculum"] is True,
            "Input-only overflow evidence is incorrect.",
        )
    builder = PassiveContextBuilder(loaded["qwen06"], budget_tokenizers=loaded)
    session_checks = 0
    for path, batch in session_episodes.items():
        saved = read(path)
        actual, summary = replay(saved["journal"])
        require(
            summary["complete"] and actual == saved["final_document"],
            "Session whole journal no longer replays.",
        )
        ledger = PassiveFeedbackLedger.restore(saved["feedback_ledger"])
        checkpoints = {c["case_id"]: c for c in saved["checkpoints"]}
        for case in batch:
            checkpoint = checkpoints[case["id"]]
            bundle = builder.build(
                checkpoint["before"],
                checkpoint["focus_uuid"],
                as_of=checkpoint["as_of"],
                history=saved["committed_transitions"],
                feedback=ledger,
                caption_profile=case["input"]["caption_profile"],
            )
            require(
                canonical_messages(bundle["input"])
                == canonical_messages(case["input"]),
                "Admitted session input lost as_of/review dynamics.",
            )
            require(
                session_hash(checkpoint["before"])
                == case["metadata"]["document_before_hash"],
                "Session checkpoint document changed.",
            )
            session_checks += 1
    sample = read(directory / "training-sampling-example.json")
    require(
        sample["case_ids"]
        == independent_draws(by_split["train"], len(by_split["train"]), sample["seed"]),
        "Saved sampling example differs from actual frozen weights.",
    )
    screen = read(directory / "screening-protocol.json")
    screen_ids = screen["selected_case_ids"]
    dev_ids = {c["id"] for c in by_split["dev"]}
    require(
        len(set(screen_ids)) == len(screen_ids)
        and set(screen_ids) <= dev_ids
        and len(screen_ids) + screen["unfilled_slots"] == screen["request_budget"] == 36
        and screen["model_predictions_consulted"] is False,
        "Screening slots are duplicated, leaked or miscounted.",
    )
    source_first = {
        c["id"]
        for c in by_split["dev"]
        if c["metadata"]["release_stratum"] == "source_first_checkpoints"
    }
    require(
        source_first <= set(screen_ids),
        "Screening omitted a required source-first DEV checkpoint.",
    )
    require(
        {by_id[i]["metadata"]["source_group_id"] for i in screen_ids}
        == {c["metadata"]["source_group_id"] for c in by_split["dev"]},
        "Screening source-family coverage differs.",
    )
    available_cells = {
        (c["metadata"]["language"], c["expectedOutput"]["action"])
        for c in by_split["dev"]
    }
    require(
        {
            (by_id[i]["metadata"]["language"], by_id[i]["expectedOutput"]["action"])
            for i in screen_ids
        }
        == available_cells,
        "Screening omits an available language/action cell.",
    )
    census = read(directory / "census.json")
    derived = {
        "rows": len(cases),
        "splits": {s: len(v) for s, v in by_split.items()},
        "stratum_splits": dict(
            Counter(
                f"{c['metadata']['release_stratum']}:{c['metadata']['split']}"
                for c in cases
            )
        ),
        "action_splits": dict(
            Counter(
                f"{c['expectedOutput']['action']}:{c['metadata']['split']}"
                for c in cases
            )
        ),
        "language_splits": dict(
            Counter(
                f"{c['metadata']['language']}:{c['metadata']['split']}" for c in cases
            )
        ),
        "source_group_splits": dict(Counter(groups.values())),
        "registered_sources": len({c["metadata"]["source_id"] for c in cases}),
        "episodes": len({c["metadata"]["episode_id"] for c in cases}),
        "known_outcomes": semantic_references,
        "feedback_rows": sum(bool(c["input"]["recent_feedback"]) for c in cases),
        "selection_rows": sum(bool(c["input"]["selection"]) for c in cases),
        "multi_target_corrections": sum(
            c["expectedOutput"]["action"] == "correct_text"
            and len(c["expectedOutput"]["target"]) > 1
            for c in cases
        ),
        "matched_format_splits": dict(Counter(c["metadata"]["split"] for c in matched)),
        "format_input_exclusions": len(overflow),
        "supplement_review_exclusions": sum(
            e["exclusion_stage"] == "independent_candidate_review" for e in excluded
        ),
        "checkpoint_registered_observations": checkpoint_observations,
        "checkpoint_unresolved_exclusions": sum(
            e["exclusion_stage"] == "checkpoint_qualification" for e in excluded
        ),
        "audio_cut_rows": 0,
        "test_rows": 0,
        "training_run": False,
        "full_seven_action_coverage_complete": False,
    }
    require(
        all(census[k] == value for k, value in derived.items()),
        "Frozen census differs from admitted payloads.",
    )
    require(
        census["release_id"] == manifest["release_id"],
        "Census release identity differs.",
    )
    hashes.stable()
    result = {
        "version": VERSION,
        "status": "passed_independent_bundle_readback",
        "directory": str(directory),
        "manifest_sha256": digest(manifest_path),
        "verifier_sha256": digest(Path(__file__)),
        "freeze_configuration": config_name,
        "release_files_rehashed": len(manifest["files_sha256"]),
        "dependencies_physically_rehashed": len(manifest["dependencies_sha256"]),
        "unique_physical_files_rehashed": len(hashes.checked),
        "parent_inputs_references_and_provenance_unchanged": parent_count,
        "independently_bound_supplement_rows": strata,
        "census": derived,
        "semantic_references_roundtripped_in_both_formats": semantic_references,
        "reference_token_encodings_checked": reference_encodings,
        "exact_message_checks": len(cases),
        "completion_masks": "Every primary prompt token masked; every completion token including EOS unmasked. Exact retokenization of all primary/alternative references.",
        "sampling_weight_sums": masses,
        "matched_sampling_weight_sums": matched_masses,
        "actual_weighted_training_loader_checks": loaders,
        "session_full_ledger_as_of_rebuilds": session_checks,
        "token_mass_views_verified": len(loaders),
        "screening_dev_slots_checked": len(screen_ids),
        "no_semantic_relabeling": True,
        "training_run": False,
        "test_bodies_opened": False,
        "audio_listening": False,
        "limitations": config["limitations"],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.output), ensure_ascii=False, indent=2))
