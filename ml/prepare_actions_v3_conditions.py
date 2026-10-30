"""Freeze train-only demonstrations and separately qualified development conditions.

Only model-visible history changes. Source mappings and the runtime feedback gate
remain intact. Canonical coverage never shrinks to the ICL or ablation intersection.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from context_feedback_report import history_rule
from grading import transcript_media
import passive_actions as contract
import passive_formats_actions_v3 as formats
import prepare_passive_actions_v3 as dataset
from replay_session import session_hash
from verify_release_export import verify_operation

VERSION = "passive-actions-v3-conditions-1"
COHORT_POLICY = "matched_common_json_rewrite"
CONDITIONS = {
    "canonical_full_context": (),
    "train_icl": (),
    "context_history": ("recent_feedback",),
    "context_only": ("recent_edits", "recent_feedback"),
}
FEEDBACK_GATE = "unchanged_runtime_journal"
require, digest, sha256 = dataset.require, dataset.digest, dataset.sha256


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def input_hash(snapshot):
    return digest(contract.wire_snapshot(snapshot))


def project_input(snapshot, condition_id):
    require(condition_id in CONDITIONS, "Unregistered context condition.")
    result = contract.parse_snapshot(snapshot).model_dump()
    for field in CONDITIONS[condition_id]:
        result[field] = []
    return contract.parse_snapshot(result).model_dump()


def native_prefix(tokenizer, messages):
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    require(
        isinstance(value, list)
        and value
        and all(type(token) is int and 0 <= token < 262144 for token in value),
        "Native tokenizer returned invalid prefix IDs.",
    )
    return value


def demonstration(case, tokenizer, release_binding):
    require(case["split"] == case["source"]["split"] == "train", "ICL is train-only.")
    outputs = {
        name: formats.serialize(case["input"], case["primary"], name)
        for name in formats.FORMATS
    }
    prefixes = {
        name: native_prefix(tokenizer, formats.messages(case["input"], name))
        for name in formats.FORMATS
    }
    sizes = {
        name: len(prefixes[name])
        + len(tokenizer.encode(outputs[name], add_special_tokens=False))
        for name in formats.FORMATS
    }
    return {
        "case_id": case["id"],
        "split": "train",
        "language": case["language"],
        "source_group": case["source"]["source_group"],
        "source_sha256": digest(case["source"]),
        "case_sha256": digest(case),
        "input": copy.deepcopy(case["input"]),
        "input_sha256": input_hash(case["input"]),
        "action": copy.deepcopy(case["primary"]),
        "outputs_sha256": {name: digest(value) for name, value in outputs.items()},
        "native_prefix_ids_sha256": {
            name: digest(value) for name, value in prefixes.items()
        },
        "selection_tokens": sizes,
        "accepted_predictions_sha256": digest(case["accepted_predictions"]),
        "release_binding": copy.deepcopy(release_binding),
    }


def select_demonstrations(train_cases, tokenizer, release_binding):
    candidates = [
        demonstration(case, tokenizer, release_binding) for case in train_cases
    ]
    choices = []
    for edit_language in ("en", "ko"):
        pair = []
        for language, editing in (
            (edit_language, True),
            ("ko" if edit_language == "en" else "en", False),
        ):
            pool = [
                item
                for item in candidates
                if item["language"] == language
                and (item["action"]["action"] != "no_action") == editing
            ]
            if not pool:
                break
            pair.append(
                min(
                    pool,
                    key=lambda item: (
                        max(item["selection_tokens"].values()),
                        item["case_id"],
                    ),
                )
            )
        if len(pair) == 2:
            choices.append(
                sorted(pair, key=lambda item: (item["language"], item["case_id"]))
            )
    require(choices, "ICL needs a bilingual train edit/abstention pair.")
    return min(
        choices,
        key=lambda pair: (
            sum(max(item["selection_tokens"].values()) for item in pair),
            [item["case_id"] for item in pair],
        ),
    )


def condition_specs(demonstrations):
    return {
        name: {
            "version": VERSION,
            "id": name,
            "removed_fields": list(removed),
            "feedback_gate": FEEDBACK_GATE,
            "demonstrations": copy.deepcopy(demonstrations)
            if name == "train_icl"
            else [],
            "selection": "train_only_shortest_bilingual_edit_and_abstention_v1"
            if name == "train_icl"
            else None,
        }
        for name, removed in CONDITIONS.items()
    }


def expected_messages(snapshot, format_name, condition):
    require(
        set(condition)
        == {
            "version",
            "id",
            "removed_fields",
            "feedback_gate",
            "demonstrations",
            "selection",
        }
        and condition["version"] == VERSION
        and condition["id"] in CONDITIONS
        and condition["removed_fields"] == list(CONDITIONS[condition["id"]])
        and condition["feedback_gate"] == FEEDBACK_GATE,
        "Condition differs from the registered model-visible transformation.",
    )
    require(
        project_input(snapshot, condition["id"]) == snapshot,
        "Ablated fields are still present.",
    )
    canonical = formats.messages(snapshot, format_name)
    demos = condition["demonstrations"]
    if condition["id"] != "train_icl":
        require(
            not demos and condition["selection"] is None, "Unexpected demonstrations."
        )
        return canonical
    require(
        len(demos) == 2
        and len({item["case_id"] for item in demos}) == 2
        and {item["language"] for item in demos} == {"en", "ko"}
        and {item["action"]["action"] == "no_action" for item in demos} == {False, True}
        and condition["selection"]
        == "train_only_shortest_bilingual_edit_and_abstention_v1",
        "ICL requires the fixed bilingual train edit/abstention pair.",
    )
    messages = [canonical[0]]
    for demo in demos:
        require(
            demo["split"] == "train"
            and demo["source_group"] not in dataset.RESERVED
            and demo["input_sha256"] == input_hash(demo["input"])
            and demo["input"]["language"] == demo["language"],
            "Demonstration split or input binding changed.",
        )
        text = formats.serialize(demo["input"], demo["action"], format_name)
        require(
            digest(text) == demo["outputs_sha256"][format_name], "ICL answer changed."
        )
        messages.extend(
            (
                formats.messages(demo["input"], format_name)[1],
                {"role": "assistant", "content": text},
            )
        )
    return [*messages, canonical[1]]


def validate_request(row):
    require(
        not {
            "primary",
            "acceptable",
            "accepted_predictions",
            "judgment",
            "rationale",
        }.intersection(row)
        and row["split"] == "dev",
        "Requests contain current-case labels or a non-development input.",
    )
    require(
        row["condition_sha256"] == digest(row["condition"])
        and row["input_sha256"] == input_hash(row["input"])
        and row["messages"]
        == expected_messages(row["input"], row["format"], row["condition"])
        and row["messages_sha256"] == digest(row["messages"])
        and row["prefix_ids_sha256"] == digest(row["prefix_ids"]),
        "Registered condition, prompt or token binding changed.",
    )
    require(
        0 < len(row["prefix_ids"]) <= formats.INPUT_LIMIT
        and all(
            type(token) is int and 0 <= token < 262144 for token in row["prefix_ids"]
        ),
        "Condition exceeds the complete native input budget.",
    )


def baseline_prediction(snapshot, name):
    snapshot = contract.parse_snapshot(snapshot).model_dump()
    if name == "always_no_action":
        return {"action": "no_action"}
    require(name == "consistent_history_rule", "Unknown deterministic baseline.")
    return history_rule(snapshot)


def qualify(case, condition, projected, decision, release_sha256):
    changed = [field for field in condition["removed_fields"] if case["input"][field]]
    if not changed:
        require(decision is None, "Unchanged input must retain its reviewed labels.")
        return {
            "status": "qualified",
            "basis": "unchanged_reviewed_model_input",
            "judgment": case["judgment"],
            "accepted_predictions": case["accepted_predictions"],
            "rationale": case["rationale"],
        }
    if decision is None:
        return {
            "status": "needs_review",
            "reason": "removed_context_may_change_necessity",
            "changed_fields": changed,
        }
    require(
        decision["case_id"] == case["id"]
        and decision["condition_id"] == condition["id"]
        and decision["condition_sha256"] == digest(condition)
        and decision["original_input_sha256"] == input_hash(case["input"])
        and decision["input_sha256"] == input_hash(projected)
        and decision["release_manifest_sha256"] == release_sha256
        and decision["model_predictions_consulted"] is False
        and decision["actor"] in ("ai_generated", "human")
        and decision["reviewer_id"].strip()
        and decision["rationale"].strip()
        and decision["status"] in ("qualified", "uncertain", "inapplicable"),
        "Condition qualification is not a bound input-only judgment.",
    )
    if decision["status"] != "qualified":
        require(
            not decision.get("accepted_predictions"),
            "Unknown is not an accepted no_action.",
        )
        return {
            "status": decision["status"],
            "reason": decision["rationale"],
            "qualification": decision,
        }
    refs = decision["accepted_predictions"]
    judgment = decision["judgment"]
    require(
        judgment["status"] in ("edit_required", "preserve")
        and refs
        and len({digest(value) for value in refs}) == len(refs)
        and all(
            (value["action"] == "no_action") == (judgment["status"] == "preserve")
            for value in refs
        ),
        "Qualified references disagree about whether the projected input needs an edit.",
    )
    return {
        "status": "qualified",
        "basis": "independent_projected_input_review",
        "judgment": judgment,
        "accepted_predictions": refs,
        "rationale": decision["rationale"],
        "qualification": decision,
    }


def conditioned_case(case, bundle, condition, qualification):
    projected = project_input(case["input"], condition["id"])
    bundle = copy.deepcopy(bundle)
    bundle["input"] = projected
    bundle["context"]["prompt_hash"] = session_hash(
        contract.canonical_messages(projected)
    )
    refs = qualification["accepted_predictions"]
    ledger = case["runtime"].get("feedback_journal")
    ledger = contract.PassiveFeedbackLedger.restore(ledger) if ledger else None
    results = []
    for reference in refs:
        require(
            not ledger
            or not ledger.suppressed(
                case["document"], bundle, reference, case["runtime"]["sequence"]
            ),
            "A qualified edit is suppressed by the unchanged runtime feedback gate.",
        )
        after = contract.apply_passive_edit(case["document"], bundle, reference)
        verify_operation(case["document"], after, bundle["mapping"], reference)
        results.append(digest(transcript_media(after)))
    return {
        **case,
        "input": projected,
        "original_input": copy.deepcopy(case["input"]),
        "original_case_sha256": digest(case),
        "original_input_sha256": input_hash(case["input"]),
        "messages": formats.messages(projected),
        "bundle": bundle,
        "condition": condition,
        "condition_sha256": digest(condition),
        "primary": refs[0],
        "acceptable": refs,
        "accepted_predictions": refs,
        "accepted_result_sha256": results,
        "judgment": qualification["judgment"],
        "rationale": qualification["rationale"],
        "condition_qualification": qualification,
    }


def case_stratum(case):
    lineage, source = case.get("lineage", {}), case["source"]
    return (
        lineage.get("stratum")
        or lineage.get("scenario_family")
        or source.get("kind")
        or source.get("provenance")
        or "unspecified"
    )


def coverage(cases):
    dimensions = {
        "language": lambda c: c["language"],
        "reference_action": lambda c: c["primary"]["action"],
        "source_group": lambda c: c["source"]["source_group"],
        "stratum": case_stratum,
    }
    return {
        key: dict(sorted(Counter(value(case) for case in cases).items()))
        for key, value in dimensions.items()
    }


def verify_reference_proofs(release, manifest, cases, builder):
    """Reuse hash-bound replay evidence; absent evidence takes the full validation path."""
    release = Path(release).resolve()
    files = manifest["files_sha256"]
    for split in {case["split"] for case in cases}:
        require(split in ("train", "dev"), "Reference proofs exclude final-test cases.")
        name = f"cases/{split}.jsonl"
        require(sha256(release / name) == files[name], "Reference case file changed.")
    sources = {
        str((dataset.ROOT / name).resolve()): expected
        for name, expected in manifest.get("source_sha256", {}).items()
    }
    anchors = [Path(contract.__file__).resolve(), contract.PROMPT_PATH.resolve()]
    checked_sources = 0
    for name, expected in sources.items():
        path = Path(name)
        if path.suffix == ".py" or path in anchors:
            require(
                path.is_relative_to(dataset.ROOT), "Proof source escaped the checkout."
            )
            require(
                sha256(path) == expected,
                "Reference proof contract or prompt changed.",
            )
            checked_sources += 1
    proof_name = "reference-executions.jsonl"
    reusable = proof_name in files and all(str(path) in sources for path in anchors)
    proofs = {}
    if reusable:
        require(
            manifest.get("contract_version") == contract.VERSION
            and set(manifest.get("counts", {})) <= {"train", "dev"},
            "Reference proof scope or contract is not exposed train/development.",
        )
        path = (release / proof_name).resolve()
        require(
            path.is_relative_to(release) and sha256(path) == files[proof_name],
            "Reference proof file changed.",
        )
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                proof = json.loads(line)
                require(proof["id"] not in proofs, "Duplicate reference proof IDs.")
                proofs[proof["id"]] = {
                    key: value for key, value in proof.items() if key != "journal"
                }
                proofs[proof["id"]].update(
                    journal=bool(proof["journal"]),
                    bound_sha256=digest(
                        {key: value for key, value in proof.items() if key != "id"}
                    ),
                )
    cached, replayed = 0, 0
    for case in cases:
        snapshot = contract.parse_snapshot(case["input"]).model_dump()
        refs = case["accepted_predictions"]
        status = case["judgment"]["status"]
        require(
            case["split"] == case["source"]["split"]
            and case["source"]["source_group"] not in dataset.RESERVED
            and contract.canonical_messages(snapshot) == case["messages"]
            and status in ("edit_required", "preserve")
            and refs
            and refs == case["acceptable"]
            and refs[0] == case["primary"]
            and all(
                (value["action"] == "no_action") == (status == "preserve")
                for value in refs
            ),
            "Reference split, visible prompt or judgment linkage changed.",
        )
        if not reusable or "execution_sha256" not in case:
            dataset.validate_case(case, builder)
            replayed += 1
            continue
        require(case["id"] in proofs, "A bound case is missing its reference proof.")
        proof = proofs[case["id"]]
        require(
            proof["bound_sha256"] == case["execution_sha256"]
            and proof["prediction"] == case["primary"]
            and proof["before_sha256"] == session_hash(case["document"])
            and proof["changed"] == (case["primary"]["action"] != "no_action")
            and proof["source_integrity"]
            == "verified_exact_operation_and_outside_intervals",
            "Reference execution proof belongs to another case or operation.",
        )
        if proof["changed"]:
            require(
                proof["journal"]
                and proof["hydrate_start_document_from"] == "case.document"
                and proof["replay"]["complete"] is True
                and proof["replay"]["transactions"] == 3
                and proof["replay"]["final_hash"] == proof["after_sha256"],
                "Stored apply/undo/redo proof is incomplete.",
            )
        else:
            require(
                not proof["journal"]
                and proof["replay"] is None
                and proof["after_sha256"] == proof["before_sha256"],
                "Stored abstention changed the document.",
            )
        cached += 1
    return {
        "cached_cases": cached,
        "replayed_cases": replayed,
        "proof_file_sha256": files.get(proof_name) if reusable else None,
        "current_contract_sha256": sha256(anchors[0]),
        "current_prompt_sha256": sha256(anchors[1]),
        "verified_code_sources": checked_sources,
        "alternative_execution": "Selected ICL cases and admitted development references execute separately.",
    }


def prepare(release, review, output, qualifications=None, *, tokenizer=None):
    release, output = Path(release).resolve(), Path(output).resolve()
    require(not output.exists(), "Use a fresh condition preparation directory.")
    manifest = json.loads((release / "manifest.json").read_text())
    release_binding = {
        "manifest": binding(release / "manifest.json"),
        "semantic_review": binding(review),
    }
    evidence = json.loads(Path(review).read_text())
    require(
        evidence["status"] == "passed"
        and evidence["manifest_sha256"] == release_binding["manifest"]["sha256"]
        and evidence["model_predictions_consulted"] is False,
        "Require frozen input-only review before development outputs.",
    )
    split_cases = {}
    for split in ("train", "dev"):
        name = f"cases/{split}.jsonl"
        require(
            sha256(release / name) == manifest["files_sha256"][name],
            "Train/development cases changed.",
        )
        split_cases[split] = dataset.rows(release / name)
        require(
            split_cases[split]
            and len({case["id"] for case in split_cases[split]})
            == len(split_cases[split]),
            "Empty or duplicate split roster.",
        )
        require(
            all(
                case["split"] == case["source"]["split"] == split
                and case["source"]["source_group"] not in dataset.RESERVED
                for case in split_cases[split]
            ),
            "Split or reserved source contamination.",
        )
        release_binding[f"{split}_cases_sha256"] = manifest["files_sha256"][name]
    train, development = split_cases["train"], split_cases["dev"]
    require(
        not {case["source"]["source_group"] for case in train}.intersection(
            case["source"]["source_group"] for case in development
        ),
        "Train and development source groups overlap.",
    )
    require(
        not {case["id"] for case in train}.intersection(
            case["id"] for case in development
        ),
        "Train and development IDs overlap.",
    )
    tokenizer, tokenizer_identity = tokenizer or dataset.legacy.tokenizer()
    builder = contract.PassiveContextBuilder(tokenizer)
    proof_verification = verify_reference_proofs(
        release, manifest, [*train, *development], builder
    )
    demos = select_demonstrations(train, tokenizer, release_binding)
    selected = {item["case_id"] for item in demos}
    for case in train:
        if case["id"] in selected:
            dataset.validate_case(case, builder)
    require(
        not {item["input_sha256"] for item in demos}.intersection(
            input_hash(case["input"]) for case in development
        ),
        "A demonstration duplicates a development input.",
    )
    conditions = condition_specs(demos)
    decisions = dataset.rows(qualifications) if qualifications else []
    indexed = {(row["case_id"], row["condition_id"]): row for row in decisions}
    require(
        len(indexed) == len(decisions)
        and all(
            identity in {case["id"] for case in development} and name in CONDITIONS
            for identity, name in indexed
        ),
        "Duplicate or foreign condition qualification.",
    )
    prepared = {name: [] for name in CONDITIONS}
    requests = {
        name: {format_name: [] for format_name in formats.FORMATS}
        for name in CONDITIONS
    }
    eligibility, queue = [], []
    for case in development:
        bundle = dataset.runtime_bundle(case, builder)
        require(bundle["input"] == case["input"], "Development runtime window changed.")
        for name, condition in conditions.items():
            projected = project_input(case["input"], name)
            qualification = qualify(
                case,
                condition,
                projected,
                indexed.get((case["id"], name)),
                release_binding["manifest"]["sha256"],
            )
            event = {
                "case_id": case["id"],
                "condition_id": name,
                "language": case["language"],
                "source_group": case["source"]["source_group"],
                "stratum": case_stratum(case),
                "original_input_sha256": input_hash(case["input"]),
                "input_sha256": input_hash(projected),
                "condition_sha256": digest(condition),
                "status": qualification["status"],
            }
            if qualification["status"] != "qualified":
                event["reason"] = qualification["reason"]
                eligibility.append(event)
                if qualification["status"] == "needs_review":
                    queue.append(
                        {
                            **event,
                            "release_manifest_sha256": release_binding["manifest"][
                                "sha256"
                            ],
                            "removed_fields": condition["removed_fields"],
                            "original_input": case["input"],
                            "projected_input": projected,
                            "original_judgment": case["judgment"],
                            "original_references": case["accepted_predictions"],
                            "original_rationale": case["rationale"],
                            "runtime_feedback_gate": FEEDBACK_GATE,
                        }
                    )
                continue
            messages = {
                fmt: expected_messages(projected, fmt, condition)
                for fmt in formats.FORMATS
            }
            prefixes = {
                fmt: native_prefix(tokenizer, value) for fmt, value in messages.items()
            }
            event["input_tokens"] = {fmt: len(value) for fmt, value in prefixes.items()}
            if any(len(value) > formats.INPUT_LIMIT for value in prefixes.values()):
                eligibility.append(
                    {
                        **event,
                        "status": "excluded",
                        "reason": "common_format_input_budget",
                    }
                )
                continue
            value = conditioned_case(case, bundle, condition, qualification)
            value["bundle"]["context"]["input_tokens"] = event["input_tokens"]["json"]
            value["bundle"]["context"]["condition_token_counts"] = event["input_tokens"]
            prepared[name].append(value)
            eligibility.append(
                {**event, "status": "admitted", "basis": qualification["basis"]}
            )
            for fmt in formats.FORMATS:
                row = {
                    "case_id": case["id"],
                    "split": "dev",
                    "format": fmt,
                    "input": projected,
                    "input_sha256": input_hash(projected),
                    "original_input_sha256": input_hash(case["input"]),
                    "messages": messages[fmt],
                    "messages_sha256": digest(messages[fmt]),
                    "prefix_ids": prefixes[fmt],
                    "prefix_ids_sha256": digest(prefixes[fmt]),
                    "condition": condition,
                    "condition_sha256": digest(condition),
                }
                validate_request(row)
                requests[name][fmt].append(row)
    require(
        prepared["canonical_full_context"],
        "No canonical common-format development cases.",
    )
    comparisons = {}
    for comparison, names in {
        "icl_matched": ("canonical_full_context", "train_icl"),
        "feedback_visibility_matched": ("canonical_full_context", "context_history"),
        "history_visibility_matched": ("context_history", "context_only"),
        "all_context_matched": (
            "canonical_full_context",
            "context_history",
            "context_only",
        ),
    }.items():
        shared = set.intersection(
            *({case["id"] for case in prepared[name]} for name in names)
        )
        shared.intersection_update(
            case["id"] for case in prepared["canonical_full_context"]
        )
        kept = [
            case for case in prepared["canonical_full_context"] if case["id"] in shared
        ]
        comparisons[comparison] = {
            "conditions": list(names),
            "case_ids": [case["id"] for case in kept],
            "requests_per_condition": len(kept),
            "canonical_label_coverage": coverage(kept),
            "coverage_by_condition": {
                name: coverage(
                    [case for case in prepared[name] if case["id"] in shared]
                )
                for name in names
            },
            "changed_input_case_ids": {
                name: [
                    case["id"]
                    for case in prepared[name]
                    if case["id"] in shared
                    and case["original_input_sha256"] != input_hash(case["input"])
                ]
                for name in names
            },
            "excluded_case_ids": [
                case["id"] for case in development if case["id"] not in shared
            ],
        }
    sources = {
        str((dataset.ROOT / path).resolve()): value
        for path, value in dataset.legacy.source_closure(
            [__file__, contract.PROMPT_PATH]
        ).items()
    }
    output.mkdir(parents=True)
    dataset.write(
        output / "registry.json",
        {
            "version": VERSION,
            "cohort_policy": COHORT_POLICY,
            "status": "registered_before_inference",
            "conditions": conditions,
            "demonstrated_actions": sorted(
                {item["action"]["action"] for item in demos}
            ),
            "demonstrations_cover_all_seven_actions": False,
            "selection_used_development_outputs": False,
            "model_matrix": {
                "always_no_action": "canonical_full_context",
                "consistent_history_rule": "canonical_full_context",
                "native_base": "canonical_full_context",
                "native_base_icl": "train_icl",
                "native_sft": "canonical_full_context",
                "native_dpo_from_same_sft": "canonical_full_context",
            },
            "rules_supported_edit_actions": ["correct_text", "trim_silence"],
            "comparisons": comparisons,
            "tokenizer": tokenizer_identity,
            "release_binding": release_binding,
            "reference_proof_verification": proof_verification,
        },
    )
    for name, cases in prepared.items():
        directory = output / name
        dataset.write_rows(directory / "cases.jsonl", cases)
        for fmt, values in requests[name].items():
            dataset.write_rows(directory / f"requests-{fmt}.jsonl", values)
        dataset.write_rows(
            directory / "exclusions.jsonl",
            [
                event
                for event in eligibility
                if event["condition_id"] == name and event["status"] != "admitted"
            ],
        )
        dataset.write(
            directory / "manifest.json",
            {
                "version": VERSION,
                "status": "prepared_not_inferred",
                "cohort_policy": COHORT_POLICY,
                "condition": conditions[name],
                "condition_sha256": digest(conditions[name]),
                "registry": binding(output / "registry.json"),
                "dataset_manifest": release_binding["manifest"],
                "semantic_review": release_binding["semantic_review"],
                "requests": len(cases),
                "excluded": len(development) - len(cases),
                "case_ids": [case["id"] for case in cases],
                "coverage": coverage(cases),
                "tokenizer": tokenizer_identity,
                "final_test_bodies_read": False,
                "sources_sha256": sources,
                "files_sha256": {
                    path.name: sha256(path)
                    for path in directory.iterdir()
                    if path.is_file()
                },
            },
        )
    dataset.write_rows(output / "eligibility.jsonl", eligibility)
    dataset.write_rows(output / "review-queue.jsonl", queue)
    dataset.write(
        output / "manifest.json",
        {
            "version": VERSION,
            "status": "prepared_with_explicit_condition_eligibility",
            "cohort_policy": COHORT_POLICY,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_binding": release_binding,
            "reference_proof_verification": proof_verification,
            "qualifications": binding(qualifications) if qualifications else None,
            "canonical_requests": len(prepared["canonical_full_context"]),
            "condition_requests": {
                name: len(cases) for name, cases in prepared.items()
            },
            "review_queue": len(queue),
            "canonical_not_restricted_by_icl_or_ablation": True,
            "native_budget": formats.INPUT_LIMIT,
            "final_test_bodies_read": False,
            "sources_sha256": sources,
            "files_sha256": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return {
        "output": str(output),
        "condition_requests": {name: len(cases) for name, cases in prepared.items()},
        "review_queue": len(queue),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--qualifications", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.release, args.review, args.output, args.qualifications),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
