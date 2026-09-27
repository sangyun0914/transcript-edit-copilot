"""Qualify fixed, same-state TRAIN candidates as explicitly AI-labeled preferences."""

import argparse
from collections import Counter, defaultdict, deque
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random

import prepare_passive_curriculum as curriculum
import prepare_passive_sft as data
import train_passive_sft as smoke

VERSION = "passive-preference-preparation-v1"
POLICY_VERSION = "passive-preference-qualification-v1"
ROOT = Path(__file__).resolve().parents[1]
REASONS = {
    "format_invalid",
    "semantic_invalid",
    "repeated_dismissed_edit",
    "missed_required_edit",
    "unnecessary_edit_on_preserve_state",
}
PAYLOADS = {
    "pairs.jsonl",
    "qualification.jsonl",
    "selection-exclusions.jsonl",
    "configuration.json",
    "policy.json",
    "source/prepare_passive_preferences.py",
}
CANDIDATE_PAYLOADS = {
    "run.json",
    "protocol.json",
    "generator.py",
    "runtime.json",
    "candidates.jsonl",
}
GRADING_SOURCES = (
    "passive_benchmark.py",
    "passive_contract.py",
    "passive_dataset.py",
    "context_feedback_report.py",
    "session_cases.py",
    "replay_session.py",
    "grading.py",
)


def classify(case, candidate):
    from passive_benchmark import evaluate_result

    smoke.require(
        case["metadata"]["split"] == "train",
        "Never qualify a development/test preference.",
    )
    references = [case["expectedOutput"], *case.get("acceptableOutputs", [])]
    chosen = evaluate_result(case, json.dumps(references[0], ensure_ascii=False))
    smoke.require(
        chosen["raw_valid"]
        and chosen["raw_grade"]["outcome_match"]
        and (references[0]["action"] == "no_action" or chosen["displayed"]),
        "The primary training target is not an eligible accepted outcome.",
    )
    if candidate.get("finish_reason") != "stop":
        return {"status": "excluded", "reason": "unfinished_or_failed_generation"}
    rejected = evaluate_result(case, candidate["text"], candidate["finish_reason"])
    if not rejected["raw_valid"]:
        reason = rejected["error_type"]
        smoke.require(
            reason in {"format_invalid", "semantic_invalid"}, "Unknown invalidity."
        )
    elif rejected["raw_grade"]["outcome_match"]:
        if rejected.get("suppression_reason"):
            return {
                "status": "excluded",
                "reason": "accepted_reference_conflicts_with_repeat_gate",
            }
        return {"status": "tie", "reason": "both_are_acceptable_execution_outcomes"}
    elif rejected.get("suppression_reason"):
        reason = "repeated_dismissed_edit"
    elif rejected["prediction"]["action"] == "no_action" and all(
        reference["action"] != "no_action" for reference in references
    ):
        reason = "missed_required_edit"
    elif all(reference["action"] == "no_action" for reference in references):
        reason = "unnecessary_edit_on_preserve_state"
    else:
        return {
            "status": "excluded",
            "reason": "valid_unmatched_edit_requires_semantic_review",
        }
    return {
        "status": "qualified",
        "reason": reason,
        "rejected_grade": rejected,
        "chosen_grade": chosen,
    }


def pair_identity(messages, chosen, rejected):
    return hashlib.sha256(data.jsonl_bytes([messages, chosen, rejected])).hexdigest()


def select_pairs(pairs, maximum=256, seed="3407-dpo-v1"):
    smoke.require(
        type(maximum) is int and maximum > 0 and type(seed) is str, "Invalid selection."
    )
    unique, excluded = {}, []
    for pair in pairs:
        identity = pair_identity(
            pair["messages"], pair["chosen_text"], pair["rejected_text"]
        )
        if identity in unique:
            excluded.append(
                {
                    "case_id": pair["case_id"],
                    "reason": "exact_same_state_pair_duplicate",
                    "duplicate_of": unique[identity]["case_id"],
                }
            )
        else:
            smoke.require(pair["pair_id"] == identity, "Pair identity changed.")
            unique[identity] = pair
    groups = defaultdict(list)
    for pair in unique.values():
        meta = pair["metadata"]
        groups[(meta["chosen_action"], meta["language"])].append(pair)
    queues = {
        key: deque(
            sorted(
                values,
                key=lambda pair: hashlib.sha256(
                    (seed + ":selection:" + pair["pair_id"]).encode()
                ).hexdigest(),
            )
        )
        for key, values in sorted(groups.items())
    }
    selected = []
    while len(selected) < maximum and any(queues.values()):
        for values in queues.values():
            if values and len(selected) < maximum:
                selected.append(values.popleft())
    for values in queues.values():
        excluded.extend(
            {"case_id": pair["case_id"], "reason": "fixed_pair_budget"}
            for pair in values
        )
    random.Random(seed + ":order").shuffle(selected)
    return selected, excluded


def readiness(pairs):
    metadata = [pair["metadata"] for pair in pairs]
    checks = {
        "at_least_32_unique_pairs": len(pairs) >= 32,
        "at_least_8_sources": len({row["source_id"] for row in metadata}) >= 8,
        "both_languages": {row["language"] for row in metadata} == {"en", "ko"},
        "chosen_abstention_present": any(
            row["chosen_action"] == "no_action" for row in metadata
        ),
        "chosen_edit_present": any(
            row["chosen_action"] != "no_action" for row in metadata
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def encoded_rejected(tokenizer, record, text):
    prefix = record["primary"]["input_ids"][: record["primary"]["completion_start"]]
    completion = tokenizer.encode(text, add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    if not completion or len(completion) > 512 or len(prefix) > 3584:
        return None
    return {
        "input_ids": prefix + completion,
        "attention_mask": [1] * (len(prefix) + len(completion)),
        "labels": [-100] * len(prefix) + completion,
        "completion_start": len(prefix),
    }


def validate_pairs(pairs, eos_id, vocab_size):
    identities, states = set(), set()
    for pair in pairs:
        identity, state = pair["pair_id"], pair["case_id"]
        smoke.require(
            identity not in identities and state not in states, "Repeated pair/state."
        )
        identities.add(identity)
        states.add(state)
        smoke.require(
            pair["preference_origin"] == "AI"
            and pair["reason"] in REASONS
            and pair["metadata"]["split"] == "train"
            and pair["metadata"]["language"] in {"en", "ko"}
            and all(
                pair["metadata"].get(key)
                for key in (
                    "source_id",
                    "source_group_id",
                    "episode_id",
                    "chosen_action",
                )
            )
            and pair["chosen_text"] != pair["rejected_text"]
            and identity
            == pair_identity(
                pair["messages"], pair["chosen_text"], pair["rejected_text"]
            ),
            "Preference identity, provenance or qualification changed.",
        )
        prefixes = []
        for name in ("chosen", "rejected"):
            encoded = pair[name]
            ids, start = encoded["input_ids"], encoded["completion_start"]
            smoke.require(
                type(start) is int
                and 0 < start <= 3584
                and 0 < len(ids) - start <= 512
                and len(ids) <= 4096
                and all(type(token) is int and 0 <= token < vocab_size for token in ids)
                and encoded["attention_mask"] == [1] * len(ids)
                and encoded["labels"] == [-100] * start + ids[start:]
                and ids[-1] == eos_id,
                "Preference masks, EOS or length budget changed; never truncate.",
            )
            prefixes.append(ids[:start])
        smoke.require(
            prefixes[0] == prefixes[1]
            and hashlib.sha256(
                json.dumps(prefixes[0], separators=(",", ":")).encode()
            ).hexdigest()
            == pair["prompt_ids_sha256"],
            "Chosen and rejected must condition on exactly the same frozen prefix.",
        )


def policy_check(policy):
    smoke.require(
        policy.get("version") == POLICY_VERSION
        and policy.get("maximum_pairs") == 256
        and policy.get("selection_seed") == "3407-dpo-v1"
        and policy.get("minimum_pairs") == 32
        and policy.get("minimum_sources") == 8
        and policy.get("required_languages") == ["en", "ko"]
        and policy.get("both_chosen_edit_and_abstention") is True
        and policy.get("uncertain_valid_edits") == "exclude"
        and policy.get("ties") == "exclude"
        and policy.get("unfinished_generations") == "exclude"
        and policy.get("synthetic_negative_fallback") is False
        and policy.get("preference_origin") == "AI"
        and policy.get("training_epochs") == 1
        and policy.get("sampling_replacement") is False,
        "Preference qualification must be registered before candidate generation.",
    )
    smoke.require(
        set(policy.get("grading_sources_sha256", {})) == set(GRADING_SOURCES)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef")
            for value in [
                policy.get("qualifier_sha256"),
                *policy["grading_sources_sha256"].values(),
            ]
        ),
        "Bind the actual qualifier and grading sources before candidate generation.",
    )


def load_candidates(
    directory, bundle, config_sha, policy_sha, preparation_sha, tokenizer=None
):
    directory = Path(directory).resolve()
    manifest = smoke.read(directory / "manifest.json")
    smoke.require(
        manifest.get("version") == "passive-preference-candidates-v1"
        and manifest.get("status") == "completed_train_candidates_unqualified"
        and set(manifest["files_sha256"]) == CANDIDATE_PAYLOADS,
        "Use only the exact completed train candidate inventory.",
    )
    smoke.require(
        {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file()
        }
        == CANDIDATE_PAYLOADS | {"manifest.json"},
        "Unexpected candidate bundle file.",
    )
    for name, expected in manifest["files_sha256"].items():
        smoke.check_file(smoke.child(directory, name), expected)
    run = smoke.read(directory / "run.json")
    protocol = smoke.read(directory / "protocol.json")
    smoke.require(
        run["configuration_sha256"] == config_sha
        and run["preparation_sha256"] == preparation_sha
        and run["completed_candidates"] == run["requested_candidates"] == 809
        and run["training_updates"] == 0
        and run["candidate_source"] == "final_T1_NF4_adapter"
        and run["protocol_sha256"] == smoke.digest(directory / "protocol.json")
        and protocol["qualification_policy_sha256"] == policy_sha
        and protocol["generator_sha256"]
        == manifest["files_sha256"]["generator.py"]
        == run["generator_sha256"],
        "Candidate generation is not bound to this anchor and preregistered policy.",
    )
    rows = [
        json.loads(line)
        for line in (directory / "candidates.jsonl").read_text().splitlines()
    ]
    smoke.require(
        [row["case_id"] for row in rows] == [case["id"] for case in bundle["cases"]],
        "Require exactly one candidate per frozen train state, in order.",
    )
    eos_id, vocab_size = (
        bundle["configuration"]["tokenizer"][key]
        for key in ("eos_token_id", "vocab_size")
    )
    for encoded, candidate in zip(bundle["tokens"], rows, strict=True):
        prefix = encoded["primary"]["input_ids"][
            : encoded["primary"]["completion_start"]
        ]
        output_ids = candidate["output_ids"]
        smoke.require(
            isinstance(candidate["text"], str)
            and isinstance(output_ids, list)
            and len(output_ids) <= 512
            and all(
                type(token) is int and 0 <= token < vocab_size for token in output_ids
            )
            and candidate["prompt_ids_sha256"]
            == hashlib.sha256(
                json.dumps(prefix, separators=(",", ":")).encode()
            ).hexdigest()
            and candidate["finish_reason"]
            == ("stop" if output_ids and output_ids[-1] == eos_id else "length"),
            "Candidate token IDs, prefix or termination differs from its frozen train state.",
        )
        if tokenizer is not None:
            smoke.require(
                tokenizer.decode(output_ids, skip_special_tokens=True)
                == candidate["text"],
                "Candidate text does not decode from its actual generated IDs.",
            )
    return run, rows


def metadata(case):
    meta = case["metadata"]
    return {
        **{
            key: meta[key]
            for key in (
                "source_id",
                "source_group_id",
                "episode_id",
                "language",
                "split",
            )
        },
        "chosen_action": case["expectedOutput"]["action"],
        "original_sampling_weight": meta["sampling_weight"],
    }


def verify_bundle(directory, anchor_directory, candidate_directory, tokenizer=None):
    directory = Path(directory).resolve()
    manifest = smoke.read(directory / "manifest.json")
    smoke.require(
        manifest.get("version") == VERSION
        and set(manifest["files_sha256"]) == PAYLOADS,
        "Unknown preference inventory.",
    )
    for name, expected in manifest["files_sha256"].items():
        smoke.check_file(smoke.child(directory, name), expected)
    smoke.require(
        {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file()
        }
        == set(manifest["files_sha256"]) | {"manifest.json"}
        and smoke.digest(Path(__file__))
        == manifest["files_sha256"]["source/prepare_passive_preferences.py"],
        "Preference payload inventory or verifier implementation changed.",
    )
    pairs = [
        json.loads(line)
        for line in (directory / "pairs.jsonl").read_text().splitlines()
    ]
    config = smoke.read(directory / "configuration.json")
    policy = smoke.read(directory / "policy.json")
    policy_check(policy)
    anchor_directory, candidate_directory = (
        Path(anchor_directory).resolve(),
        Path(candidate_directory).resolve(),
    )
    smoke.check_file(
        anchor_directory / "manifest.json", manifest["bindings"]["preparation_sha256"]
    )
    smoke.check_file(
        candidate_directory / "manifest.json",
        manifest["bindings"]["candidate_manifest_sha256"],
    )
    bundle = curriculum.verify_bundle(anchor_directory)
    generated_run, candidates = load_candidates(
        candidate_directory,
        bundle,
        manifest["bindings"]["configuration_sha256"],
        smoke.digest(directory / "policy.json"),
        manifest["bindings"]["preparation_sha256"],
        tokenizer,
    )
    smoke.require(
        manifest["bindings"]["release_manifest_sha256"]
        == bundle["manifest"]["release_manifest_sha256"]
        and manifest["bindings"]["policy_sha256"]
        == smoke.digest(directory / "policy.json")
        and policy["qualifier_sha256"]
        == manifest["files_sha256"]["source/prepare_passive_preferences.py"]
        and config["anchor_training_manifest_sha256"]
        == generated_run["training_manifest_sha256"]
        and config["anchor_reload_manifest_sha256"]
        == generated_run["reload_manifest_sha256"]
        and config["eos_token_id"]
        == bundle["configuration"]["tokenizer"]["eos_token_id"]
        and config["vocab_size"] == bundle["configuration"]["tokenizer"]["vocab_size"],
        "Preference provenance or tokenizer differs from its actual anchor and candidates.",
    )
    decisions = [
        json.loads(line)
        for line in (directory / "qualification.jsonl").read_text().splitlines()
    ]
    smoke.require(
        [row["case_id"] for row in decisions]
        == [case["id"] for case in bundle["cases"]],
        "Qualification must account for every frozen training candidate.",
    )
    qualified, lookup = [], {}
    for case, encoded, candidate, decision in zip(
        bundle["cases"], bundle["tokens"], candidates, decisions, strict=True
    ):
        lookup[case["id"]] = (case, encoded, candidate, decision)
        if decision["status"] == "qualified":
            smoke.require(
                decision["reason"] in REASONS and candidate["finish_reason"] == "stop",
                "An unqualified candidate entered the selection pool.",
            )
            qualified.append(
                {
                    "pair_id": pair_identity(
                        encoded["messages"], encoded["references"][0], candidate["text"]
                    ),
                    "case_id": case["id"],
                    "messages": encoded["messages"],
                    "chosen_text": encoded["references"][0],
                    "rejected_text": candidate["text"],
                    "metadata": metadata(case),
                }
            )
    selected, exclusions = select_pairs(qualified)
    smoke.require(
        [pair["pair_id"] for pair in pairs] == [pair["pair_id"] for pair in selected]
        and exclusions
        == [
            json.loads(line)
            for line in (directory / "selection-exclusions.jsonl")
            .read_text()
            .splitlines()
        ],
        "Fixed pair selection, duplicate removal or order changed.",
    )
    for pair in pairs:
        case, encoded, candidate, decision = lookup[pair["case_id"]]
        smoke.require(
            pair["messages"] == encoded["messages"]
            and pair["chosen_text"] == encoded["references"][0]
            and pair["chosen"] == encoded["primary"]
            and pair["metadata"] == metadata(case)
            and pair["rejected_text"] == candidate["text"]
            and pair["reason"] == decision["reason"]
            and pair["prompt_ids_sha256"] == candidate["prompt_ids_sha256"]
            and pair["rejected_tokenization"]
            == {
                "method": "retokenize_decoded_text_then_append_eos",
                "generated_output_ids_sha256": hashlib.sha256(
                    data.jsonl_bytes(candidate["output_ids"])
                ).hexdigest(),
                "generated_tokens": len(candidate["output_ids"]),
            },
            "A pair differs from its actual train anchor or generated candidate.",
        )
        if tokenizer is not None:
            smoke.require(
                pair["rejected"]
                == encoded_rejected(tokenizer, encoded, candidate["text"]),
                "Rejected completion encoding differs from the actual tokenizer.",
            )
    validate_pairs(pairs, config["eos_token_id"], config["vocab_size"])
    smoke.require(
        len(pairs) <= 256
        and manifest["status"]
        == config["status"]
        == (
            "qualified_ready"
            if readiness(pairs)["passed"]
            else "insufficient_qualified_data"
        )
        and config["anchor_configuration_sha256"]
        == manifest["bindings"]["configuration_sha256"]
        and config["pair_order"] == [pair["pair_id"] for pair in pairs]
        and config["pair_count"] == len(pairs)
        and config["max_updates"] == math.ceil(len(pairs) / 4)
        and config["final_accumulation"] == (len(pairs) % 4 or 4)
        and config["readiness"] == readiness(pairs)
        and config["sequence_tokens"]
        == sum(
            len(pair[side]["input_ids"])
            for pair in pairs
            for side in ("chosen", "rejected")
        )
        and config["completion_tokens"]
        == sum(
            len(pair[side]["input_ids"]) - pair[side]["completion_start"]
            for pair in pairs
            for side in ("chosen", "rejected")
        ),
        "Pair exposure budgets or readiness changed.",
    )
    smoke.require(
        config["selected_pair_census"]
        == {
            key: dict(Counter(pair["metadata"][key] for pair in pairs))
            for key in (
                "source_group_id",
                "source_id",
                "episode_id",
                "chosen_action",
                "language",
            )
        }
        and config["selected_pair_reasons"]
        == dict(Counter(pair["reason"] for pair in pairs))
        and config["qualification_counts"]
        == dict(Counter(row["reason"] for row in decisions)),
        "Preference qualification or selected-pair census changed.",
    )
    return {
        "manifest": manifest,
        "configuration": config,
        "pairs": pairs,
        "policy": policy,
        "rejected_tokenization_verified": tokenizer is not None,
    }


def prepare(release, config_path, candidates, policy_path, output):
    from passive_benchmark import _execution
    from transformers import AutoTokenizer

    release, config_path, candidates, policy_path, output = (
        Path(value).resolve()
        for value in (release, config_path, candidates, policy_path, output)
    )
    smoke.require(
        all(
            path.is_relative_to(ROOT)
            for path in (release, config_path, candidates, policy_path, output)
        ),
        "Keep preference inputs and outputs within this project.",
    )
    smoke.require(
        not output.exists() and not output.is_relative_to(release),
        "Use a fresh output directory.",
    )
    policy = smoke.read(policy_path)
    policy_check(policy)
    config = smoke.read(config_path)
    config_sha = smoke.digest(config_path)
    prep_binding = config["bindings"]["preparation"]
    prep_manifest_path = smoke.child(config_path.parent, prep_binding["path"])
    smoke.check_file(prep_manifest_path, prep_binding["sha256"])
    bundle = curriculum.verify_bundle(prep_manifest_path.parent)
    manifest = smoke.read(release / "manifest.json")
    smoke.require(
        smoke.digest(release / "manifest.json")
        == bundle["manifest"]["release_manifest_sha256"],
        "Preference source differs from T1's frozen release.",
    )
    frozen_grading_sources = {
        name: expected
        for name, expected in manifest["dependencies_sha256"].items()
        if name.startswith("ml/") and Path(name).suffix in {".py", ".txt"}
    }
    smoke.require(frozen_grading_sources, "Frozen grading dependency pins are missing.")
    for name, expected in frozen_grading_sources.items():
        smoke.check_file(ROOT / name, expected)
    smoke.require(
        policy["qualifier_sha256"] == smoke.digest(Path(__file__)),
        "Qualifier source changed.",
    )
    for name, expected in policy["grading_sources_sha256"].items():
        smoke.check_file(Path(__file__).with_name(name), expected)
    native = smoke.child(config_path.parent, config["tokenizer"]["path"])
    for name, expected in config["tokenizer"]["files_sha256"].items():
        smoke.check_file(smoke.child(native, name), expected)
    tokenizer = AutoTokenizer.from_pretrained(
        native, local_files_only=True, trust_remote_code=False
    )
    smoke.require(
        tokenizer.eos_token_id == config["generation"]["eos_token_id"], "EOS changed."
    )
    generated_run, generated = load_candidates(
        candidates,
        bundle,
        config_sha,
        smoke.digest(policy_path),
        prep_binding["sha256"],
        tokenizer,
    )
    dependencies = {}

    def read_bound(path):
        path = Path(path).resolve()
        smoke.require(
            path.is_relative_to(ROOT), "Execution evidence escapes the project."
        )
        name = str(path.relative_to(ROOT))
        expected = manifest["dependencies_sha256"][name]
        smoke.check_file(path, expected)
        dependencies[name] = expected
        return smoke.read(path)

    qualified, decisions = [], []
    for case, encoded, candidate in zip(
        bundle["cases"], bundle["tokens"], generated, strict=True
    ):
        current = copy.deepcopy(case)
        current["_execution"] = _execution(current, read_bound)
        decision = {"case_id": case["id"], **classify(current, candidate)}
        if decision["status"] == "qualified":
            prefix = encoded["primary"]["input_ids"][
                : encoded["primary"]["completion_start"]
            ]
            prefix_sha = hashlib.sha256(
                json.dumps(prefix, separators=(",", ":")).encode()
            ).hexdigest()
            smoke.require(
                candidate["prompt_ids_sha256"] == prefix_sha, "Candidate state changed."
            )
            actual_prefix = tokenizer.apply_chat_template(
                encoded["messages"],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            smoke.require(
                actual_prefix == prefix,
                "Actual tokenizer no longer reproduces the same state.",
            )
            rejected = encoded_rejected(tokenizer, encoded, candidate["text"])
            if rejected is None:
                decision.update(status="excluded", reason="rejected_completion_budget")
            else:
                text = encoded["references"][0]
                chosen_completion = tokenizer.encode(text, add_special_tokens=False) + [
                    tokenizer.eos_token_id
                ]
                smoke.require(
                    chosen_completion == encoded["primary"]["input_ids"][len(prefix) :],
                    "Chosen completion differs from its frozen encoding.",
                )
                qualified.append(
                    {
                        "pair_id": pair_identity(
                            encoded["messages"], text, candidate["text"]
                        ),
                        "case_id": case["id"],
                        "metadata": metadata(case),
                        "messages": encoded["messages"],
                        "chosen_text": text,
                        "rejected_text": candidate["text"],
                        "chosen": encoded["primary"],
                        "rejected": rejected,
                        "prompt_ids_sha256": prefix_sha,
                        "preference_origin": "AI",
                        "reason": decision["reason"],
                        "rejected_tokenization": {
                            "method": "retokenize_decoded_text_then_append_eos",
                            "generated_output_ids_sha256": hashlib.sha256(
                                data.jsonl_bytes(candidate["output_ids"])
                            ).hexdigest(),
                            "generated_tokens": len(candidate["output_ids"]),
                        },
                    }
                )
        decisions.append(decision)
    pairs, selection_exclusions = select_pairs(qualified)
    vocab_size = bundle["configuration"]["tokenizer"]["vocab_size"]
    validate_pairs(pairs, tokenizer.eos_token_id, vocab_size)
    ready = readiness(pairs)
    prepared = {
        "version": VERSION,
        "status": "qualified_ready"
        if ready["passed"]
        else "insufficient_qualified_data",
        "anchor_configuration_sha256": config_sha,
        "anchor_training_manifest_sha256": generated_run["training_manifest_sha256"],
        "anchor_reload_manifest_sha256": generated_run["reload_manifest_sha256"],
        "pair_count": len(pairs),
        "pair_order": [pair["pair_id"] for pair in pairs],
        "max_updates": math.ceil(len(pairs) / 4),
        "final_accumulation": len(pairs) % 4 or 4,
        "sequence_tokens": sum(
            len(pair[side]["input_ids"])
            for pair in pairs
            for side in ("chosen", "rejected")
        ),
        "completion_tokens": sum(
            len(pair[side]["input_ids"]) - pair[side]["completion_start"]
            for pair in pairs
            for side in ("chosen", "rejected")
        ),
        "eos_token_id": tokenizer.eos_token_id,
        "vocab_size": vocab_size,
        "readiness": ready,
        "preference_origin": "AI",
        "one_pair_per_state": True,
        "sampling_replacement": False,
        "weights_applied_to_loss": False,
        "training_updates_executed": 0,
        "development_or_test_record_bodies_read": 0,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "pydantic")
        },
        "selection_counts": dict(
            Counter(pair["metadata"]["chosen_action"] for pair in pairs)
        ),
        "selected_pair_census": {
            key: dict(Counter(pair["metadata"][key] for pair in pairs))
            for key in (
                "source_group_id",
                "source_id",
                "episode_id",
                "chosen_action",
                "language",
            )
        },
        "selected_pair_reasons": dict(Counter(pair["reason"] for pair in pairs)),
        "qualification_counts": dict(Counter(row["reason"] for row in decisions)),
    }
    payloads = {
        "pairs.jsonl": data.jsonl_bytes(pairs),
        "qualification.jsonl": data.jsonl_bytes(decisions),
        "selection-exclusions.jsonl": data.jsonl_bytes(selection_exclusions),
        "configuration.json": data.json_bytes(prepared),
        "policy.json": policy_path.read_bytes(),
        "source/prepare_passive_preferences.py": Path(__file__).read_bytes(),
    }
    result_manifest = {
        "version": VERSION,
        "status": prepared["status"],
        "bindings": {
            "release_manifest_sha256": smoke.digest(release / "manifest.json"),
            "preparation_sha256": prep_binding["sha256"],
            "configuration_sha256": config_sha,
            "candidate_manifest_sha256": smoke.digest(candidates / "manifest.json"),
            "policy_sha256": smoke.digest(policy_path),
        },
        "execution_dependencies_sha256": dependencies,
        "frozen_grading_sources_sha256": frozen_grading_sources,
        "files_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in payloads.items()
        },
        "scope": "AI-generated same-state preference diagnostic. Synthetic labels are not human preferences; unknown alternatives and censored generations are excluded.",
    }
    output.mkdir(parents=True)
    for name, content in payloads.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
    smoke.write(output / "manifest.json", result_manifest)
    verify_bundle(output, prep_manifest_path.parent, candidates, tokenizer)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    for name in ("release", "config", "candidates", "policy", "output"):
        build.add_argument("--" + name, type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("directory", type=Path)
    check.add_argument("--anchor", type=Path, required=True)
    check.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    result = (
        verify_bundle(args.directory, args.anchor, args.candidates)["configuration"]
        if args.command == "verify"
        else prepare(
            args.release, args.config, args.candidates, args.policy, args.output
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
