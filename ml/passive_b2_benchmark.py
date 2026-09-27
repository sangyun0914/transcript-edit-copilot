"""B2 development views and reporting without changing the frozen release.

Only condition_messages() belongs in a predictor request. Execution evidence,
references and comparability decisions are evaluator-only. This module never
loads a test body, starts inference, downloads a tokenizer or changes a label.
"""

import argparse
import copy
import gzip
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import passive_benchmark as b1
from passive_contract import canonical_messages
from passive_dataset import RESERVED
from passive_formats_v2 import serialize
from prepare_passive_b2_prefixes import (
    common_input_eligibility as common_input_eligibility,
)
from replay_session import session_hash

ROOT = b1.ROOT
DEFAULT_RELEASE = ROOT / "artifacts/datasets/nep-passive-completion-v2/frozen"
CONDITIONS = ("full", "context_only", "history_only", "icl")
DEV_REQUESTS = 289


def _manifest(release):
    manifest = json.loads((release / "manifest.json").read_text())
    if (
        manifest.get("version") != "passive-dataset-completion-v2"
        or manifest.get("status") != "frozen_supported_scope"
    ):
        raise ValueError("Use the frozen completion-v2 supported-scope release.")
    return manifest


def _load_split(release, split):
    if split not in {"train", "dev"}:
        raise ValueError("B2 only reads the exposed train/dev payloads.")
    release = Path(release).resolve()
    manifest = _manifest(release)
    names = (f"{split}-cases.jsonl", f"{split}-messages.jsonl", "source-groups.json")
    payloads = {
        name: b1._checked_bytes(release / name, manifest["files_sha256"].get(name))
        for name in names
    }
    cases = [
        json.loads(line) for line in payloads[names[0]].splitlines() if line.strip()
    ]
    messages = [
        json.loads(line) for line in payloads[names[1]].splitlines() if line.strip()
    ]
    encoded = {row["id"]: row for row in messages}
    groups = json.loads(payloads["source-groups.json"])
    if (
        len({case["id"] for case in cases}) != len(cases)
        or len(encoded) != len(messages)
        or set(encoded) != {case["id"] for case in cases}
    ):
        raise ValueError("Duplicate or missing exposed cases/messages.")
    for case in cases:
        meta = case["metadata"]
        if (
            meta["split"] != split
            or groups.get(meta["source_group_id"]) != split
            or meta["source_group_id"] in RESERVED
        ):
            raise ValueError("Case source is outside its registered exposed split.")
        messages = canonical_messages(case["input"])
        encoded_messages = messages
        if split == "train":
            encoded_messages = messages + [
                {
                    "role": "assistant",
                    "content": serialize(case["input"], case["expectedOutput"], "json"),
                }
            ]
        if (
            encoded[case["id"]]["messages"] != encoded_messages
            or session_hash(messages) != meta["input_sha256"]
        ):
            raise ValueError("Frozen messages differ from exact passive serving input.")
    return cases, manifest


def _bind_execution(case, manifest):
    def read_bound(path):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Execution evidence escapes the project.")
        return json.loads(
            b1._checked_bytes(
                path, manifest["dependencies_sha256"].get(str(path.relative_to(ROOT)))
            )
        )

    case = copy.deepcopy(case)
    case["_execution"] = b1._execution(case, read_bound)
    return case


def load_dev(release=DEFAULT_RELEASE, *, expected_count=DEV_REQUESTS):
    """Load every dev case in frozen file order, with checked execution evidence."""
    cases, manifest = _load_split(release, "dev")
    if len(cases) != expected_count:
        raise ValueError(f"Expected the full {expected_count}-case development roster.")
    return [_bind_execution(case, manifest) for case in cases]


def select_demonstrations(release=DEFAULT_RELEASE):
    """Two fixed train-only examples/language; no dev input or answer retrieval.

    Choose one no_action and one correct_text with empty history/feedback by the
    smallest worst-tokenizer frozen input length, then lexical case ID. Selection
    examines only train rows and the split registry; it is not query-dependent.
    """
    cases, manifest = _load_split(release, "train")
    result = {}
    for language in ("en", "ko"):
        result[language] = []
        for action in ("no_action", "correct_text"):
            pool = [
                case
                for case in cases
                if case["metadata"]["language"] == language
                and case["expectedOutput"]["action"] == action
                and not case["input"]["recent_edits"]
                and not case["input"]["recent_feedback"]
            ]
            if not pool:
                raise ValueError(
                    f"No registered train demonstration: {language}/{action}"
                )
            selected = min(
                pool,
                key=lambda case: (
                    max(
                        case["metadata"]["context"]["comparison_token_counts"].values()
                    ),
                    case["id"],
                ),
            )
            result[language].append(_bind_execution(selected, manifest))
    return result


def feedback_condition(case):
    feedback = case["input"]["recent_feedback"]
    return (
        "+".join(
            sorted(
                {
                    f"{item['outcome']}:{'same' if item['same_context'] else 'changed'}"
                    for item in feedback
                }
            )
        )
        if feedback
        else "none"
    )


def condition_case(case, condition):
    """Remove only predictor history fields, keeping the real application/gate state.

    A reference is unscored whenever nonempty evidence was removed. This is a
    conservative comparability mask, not a new no_action label or an adjudication
    that the original edit becomes wrong. Unchanged inputs keep their references.
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown B2 condition.")
    if "_b2" in case:
        if case["_b2"]["condition"] != condition:
            raise ValueError("Build each condition from the original frozen case.")
        return copy.deepcopy(case)
    value = copy.deepcopy(case)
    value["input"] = copy.deepcopy(case["input"])
    fields = {
        "full": (),
        "icl": (),
        "context_only": ("recent_edits", "recent_feedback"),
        "history_only": ("recent_feedback",),
    }[condition]
    removed = [field for field in fields if value["input"][field]]
    for field in fields:
        value["input"][field] = []
    value["_b2"] = {
        "condition": condition,
        "original_input_sha256": session_hash(canonical_messages(case["input"])),
        "input_sha256": session_hash(canonical_messages(value["input"])),
        "reference_comparable": not removed,
        "reference_status": "removed_evidence_unadjudicated"
        if removed
        else "original_evidence",
        "removed_evidence_fields": removed,
        "original_primary_action": (case.get("expectedOutput") or {}).get(
            "action", "unknown"
        ),
        "original_feedback_condition": feedback_condition(case),
    }
    if removed:
        value["expectedOutput"] = None
        value["acceptableOutputs"] = []
    return value


def condition_messages(case, condition="full", demonstrations=()):
    """Return request-only messages, optionally with two fixed train demonstrations."""
    view = condition_case(case, condition)
    messages = canonical_messages(view["input"])
    if condition != "icl":
        if demonstrations:
            raise ValueError("Demonstrations belong only to the ICL condition.")
        return messages
    if len(demonstrations) != 2:
        raise ValueError("ICL requires the preregistered two demonstrations.")
    output = [messages[0]]
    for demonstration, action in zip(
        demonstrations, ("no_action", "correct_text"), strict=True
    ):
        meta, snapshot = demonstration["metadata"], demonstration["input"]
        if (
            meta["split"] != "train"
            or meta["source_group_id"] in RESERVED
            or meta["source_group_id"] == case["metadata"]["source_group_id"]
            or meta["language"] != case["metadata"]["language"]
            or snapshot["recent_edits"]
            or snapshot["recent_feedback"]
            or demonstration["expectedOutput"]["action"] != action
        ):
            raise ValueError(
                "Demonstrations must use the registered train-only language/action slots."
            )
        output.append(canonical_messages(snapshot)[1])
        output.append(
            {
                "role": "assistant",
                "content": serialize(snapshot, demonstration["expectedOutput"], "json"),
            }
        )
    return output + [messages[1]]


def _ordered(cases, results):
    registered = {case["id"] for case in cases}
    received = {row["case_id"]: row for row in results}
    if (
        len(registered) != len(cases)
        or len(received) != len(results)
        or set(received) - registered
    ):
        raise ValueError(
            "Require unique registered cases and at most one result per case."
        )
    for identity, row in received.items():
        if row.get("reused_from"):
            if row.get("error_type") == "missing_result" or row.get(
                "not_requested_reason"
            ):
                raise ValueError(
                    "Reuse requires an observed prior result, not a missing placeholder."
                )
            row = dict(row)
            row.setdefault(
                "inherited_complete_json_ms",
                row.get("complete_json_ms", row.get("latency_ms")),
            )
            row["complete_json_ms"], row["latency_ms"] = None, None
            received[identity] = row
    ordered = []
    for case in cases:
        if case.get("_b2", {}).get("input_budget_eligible") is False:
            if (
                case["id"] in received
                and received[case["id"]].get("not_requested_reason")
                != "input_budget_excluded"
            ):
                raise ValueError(
                    "An input-excluded case must not contain a model result."
                )
            ordered.append(
                {"case_id": case["id"], "not_requested_reason": "input_budget_excluded"}
            )
        else:
            ordered.append(
                received.get(
                    case["id"],
                    {
                        "case_id": case["id"],
                        "error_type": "missing_result",
                        "error": "Registered request has no result.",
                    },
                )
            )
    return ordered


def input_qualified_cases(cases, eligibility):
    """Mask preregistered input exclusions in copies for symmetric quality comparison."""
    if eligibility is None:
        return cases
    ids = {case["id"] for case in cases}
    eligible = set(eligibility["eligible_case_ids"])
    excluded = {row["case_id"] for row in eligibility["excluded"]}
    if (
        eligibility["registered_requests"] != len(cases)
        or eligible & excluded
        or eligible | excluded != ids
    ):
        raise ValueError("Input eligibility must partition the full registered roster.")
    views = []
    for case in cases:
        view = copy.deepcopy(case)
        view.setdefault("_b2", {})["input_budget_eligible"] = case["id"] in eligible
        if case["id"] in excluded:
            view["_b2"].setdefault(
                "original_primary_action",
                (case.get("expectedOutput") or {}).get("action", "unknown"),
            )
            view["_b2"]["reference_status"] = "input_budget_excluded"
            view["expectedOutput"], view["acceptableOutputs"] = None, []
        views.append(view)
    return views


def _metrics(cases, rows):
    metrics = b1._metrics(cases, rows)
    known_shown = metrics["displayed"] - metrics["displayed_unknown"]
    excluded = sum(
        row.get("not_requested_reason") == "input_budget_excluded" for row in rows
    )
    reused = sum(bool(row.get("reused_from")) for row in rows)
    metrics["observed_results"] -= excluded
    metrics["invalid_results"] -= excluded
    metrics.update(
        registered_requests=len(cases),
        input_budget_excluded=excluded,
        input_qualified_cases=len(cases) - excluded,
        reused_quality_rows=reused,
        expected_fresh_requests=len(cases) - excluded - reused,
        observed_fresh_results=metrics["observed_results"] - reused,
        quality_scored_cases=metrics["known_references"],
        quality_unscored_cases=metrics["unknown_references"],
        displayed_reference_known=known_shown,
        displayed_useful_precision=metrics["displayed_useful_edits"] / known_shown
        if known_shown
        else None,
        displayed_coverage=metrics["displayed"] / len(cases) if cases else None,
        raw_outcome_agreement=metrics["raw_outcome_matches"]
        / metrics["known_references"]
        if metrics["known_references"]
        else None,
    )
    return metrics


def summarize_b2(cases, results, *, eligibility=None):
    """All-request validity/latency and explicitly reference-qualified quality."""
    cases = input_qualified_cases(cases, eligibility)
    rows = _ordered(cases, results)
    report = {
        "scope": "Exposed synthetic development; correlated source/episode states; audio actions gated.",
        "precision_scope": "Reference-known displayed edits only; unknown displays reported separately.",
        "action_stratum": "Frozen primary reference; acceptable alternatives can use another action.",
        **_metrics(cases, rows),
        "missing_case_ids": [
            row["case_id"] for row in rows if row.get("error_type") == "missing_result"
        ],
        "quality_unscored_case_ids": [
            case["id"] for case in cases if not b1._references(case)
        ],
        "input_budget_excluded_case_ids": [
            case["id"]
            for case in cases
            if case.get("_b2", {}).get("input_budget_eligible") is False
        ],
        "input_budget_policy": "Input exclusions are eligibility decisions, not model failures; fixed registration retained.",
        "latency_reuse_policy": "Reused rows inherit quality only; complete_json_ms is unmeasured and inherited latency is not pooled.",
    }

    def action(case):
        return case.get("_b2", {}).get(
            "original_primary_action",
            (case.get("expectedOutput") or {}).get("action", "unknown"),
        )

    def input_length(case):
        counts = case["metadata"].get("context", {}).get("comparison_token_counts", {})
        return (
            ("over_2048" if max(counts.values()) > 2048 else "at_most_2048")
            if counts
            else "unmeasured"
        )

    grouping = {
        "action": action,
        "language": lambda case: case["metadata"]["language"],
        "action_language": lambda case: (
            f"{action(case)}|{case['metadata']['language']}"
        ),
        "feedback_condition": lambda case: case.get("_b2", {}).get(
            "original_feedback_condition", feedback_condition(case)
        ),
        "reference_status": lambda case: case.get("_b2", {}).get(
            "reference_status", "original_evidence"
        ),
        "source": lambda case: case["metadata"]["source_group_id"],
        "episode": lambda case: case["metadata"]["episode_id"],
        "stratum": lambda case: case["metadata"].get("release_stratum", "unspecified"),
        "kind": lambda case: case["metadata"].get("kind", "unspecified"),
        "input_length": input_length,
    }
    for name, key in grouping.items():
        groups = defaultdict(list)
        for case, row in zip(cases, rows, strict=True):
            groups[key(case)].append((case, row))
        report[f"by_{name}"] = {
            label: _metrics([case for case, _ in group], [row for _, row in group])
            for label, group in sorted(groups.items())
        }
    report["independent_source_groups"] = len(report["by_source"])
    report["sequential_replay"] = sequence_report(cases, rows)
    return report


def sequence_report(cases, results):
    """Fixed teacher-forced replay checkpoints, never an on-policy user simulation."""
    rows = {row["case_id"]: row for row in _ordered(cases, results)}
    groups = defaultdict(list)
    for case in cases:
        if case["metadata"].get("release_stratum") == "feedback_sessions":
            groups[case["metadata"]["episode_id"]].append(case)
    episodes = {}
    for name, states in sorted(groups.items()):
        states.sort(key=lambda case: case["metadata"]["episode_step"])
        checkpoints = [case["metadata"]["episode_step"] for case in states]
        if len(set(checkpoints)) != len(checkpoints):
            raise ValueError("Sequence checkpoints must have unique episode steps.")
        batch = [rows[case["id"]] for case in states]
        metrics = _metrics(states, batch)
        episodes[name] = {
            "case_ids": [case["id"] for case in states],
            "feedback_conditions": sorted(
                {
                    case.get("_b2", {}).get(
                        "original_feedback_condition", feedback_condition(case)
                    )
                    for case in states
                }
            ),
            **metrics,
            "all_checkpoint_outcomes_match": metrics["raw_outcome_matches"]
            == len(states)
            if metrics["known_references"] == len(states)
            else None,
        }
    return {
        "scope": "Existing synthetic journals replayed at fixed states; predictions do not choose future states.",
        "request_reuse": "Same quality observations as full dev, not additional independent samples.",
        "unmeasured": [
            "on_policy_success",
            "repair_actions",
            "human_time_savings",
            "natural_preference_accuracy",
        ],
        "episodes": episodes,
    }


def compare_runs(
    cases, results_by_name, *, cases_by_name=None, eligibility_by_name=None
):
    """Paired run differences on common references; descriptive source uncertainty."""
    ids = [case["id"] for case in cases]
    views, ordered = {}, {}
    for name, results in results_by_name.items():
        view = (cases_by_name or {}).get(name, cases)
        if [case["id"] for case in view] != ids:
            raise ValueError(
                "All paired conditions must preserve the same registered case order."
            )
        view = input_qualified_cases(view, (eligibility_by_name or {}).get(name))
        views[name], ordered[name] = view, _ordered(view, results)
    reports = {
        name: summarize_b2(views[name], results)
        for name, results in results_by_name.items()
    }
    pairs = {}
    for left, right in itertools.combinations(sorted(results_by_name), 2):
        groups = defaultdict(list)
        for index, case in enumerate(cases):
            groups[case["metadata"]["source_group_id"]].append(index)
        source_results = {}
        for source, indices in sorted(groups.items()):
            common = [
                i
                for i in indices
                if b1._references(views[left][i]) and b1._references(views[right][i])
            ]
            counts = dict(
                both_match=0, left_only_match=0, right_only_match=0, neither_match=0
            )
            for i in common:
                matches = [
                    ordered[name][i].get("raw_valid") is True
                    and (ordered[name][i].get("raw_grade") or {}).get("outcome_match")
                    is True
                    for name in (left, right)
                ]
                key = (
                    "both_match"
                    if all(matches)
                    else "left_only_match"
                    if matches[0]
                    else "right_only_match"
                    if matches[1]
                    else "neither_match"
                )
                counts[key] += 1
            source_results[source] = {
                "registered_requests": len(indices),
                "common_reference_cases": len(common),
                "unscored_cases": len(indices) - len(common),
                **counts,
                "right_minus_left_agreement": (
                    counts["right_only_match"] - counts["left_only_match"]
                )
                / len(common)
                if common
                else None,
                "left": _metrics(
                    [views[left][i] for i in common], [ordered[left][i] for i in common]
                ),
                "right": _metrics(
                    [views[right][i] for i in common],
                    [ordered[right][i] for i in common],
                ),
            }
        deltas = [
            row["right_minus_left_agreement"]
            for row in source_results.values()
            if row["right_minus_left_agreement"] is not None
        ]
        leave_one_out = (
            [(sum(deltas) - delta) / (len(deltas) - 1) for delta in deltas]
            if len(deltas) > 1
            else []
        )
        common_count = sum(
            row["common_reference_cases"] for row in source_results.values()
        )
        pairs[f"{left}__vs__{right}"] = {
            "left": left,
            "right": right,
            "registered_requests": len(cases),
            "common_reference_cases": common_count,
            "by_source": source_results,
            "source_macro_agreement_delta": sum(deltas) / len(deltas)
            if deltas
            else None,
            "leave_one_source_out_macro_delta_range": [
                min(leave_one_out),
                max(leave_one_out),
            ]
            if leave_one_out
            else None,
            "uncertainty": "Descriptive cluster variation and sensitivity only; not a confidence interval or significance test.",
        }
    return {"version": "passive-b2-comparison-v1", "runs": reports, "pairs": pairs}


def prepare_protocol(release, output):
    """Write a new preregistration; never overwrite previous evidence."""
    release, output = Path(release).resolve(), Path(output).resolve()
    if output.is_relative_to(release) or any(
        parent.name == "frozen" and parent.is_relative_to(ROOT / "artifacts/datasets")
        for parent in (output, *output.parents)
    ):
        raise ValueError("B2 preparation belongs outside every frozen release.")
    cases, _ = _load_split(release, "dev")
    if len(cases) != DEV_REQUESTS:
        raise ValueError(
            "Protocol preparation requires the complete 289-case dev roster."
        )
    demos = select_demonstrations(release)
    registry, requests = [], []
    for case in cases:
        views = {}
        for condition in CONDITIONS:
            value = condition_case(case, condition)
            messages = condition_messages(
                value,
                condition,
                demos[case["metadata"]["language"]] if condition == "icl" else (),
            )
            requests.append(
                {
                    "case_id": case["id"],
                    "condition": condition,
                    "messages": messages,
                    "messages_sha256": session_hash(messages),
                }
            )
            views[condition] = {
                **value["_b2"],
                "messages_sha256": session_hash(messages),
                "identical_to_full_request": messages
                == canonical_messages(case["input"]),
            }
        registry.append(
            {
                "case_id": case["id"],
                "source_group_id": case["metadata"]["source_group_id"],
                "episode_id": case["metadata"]["episode_id"],
                "language": case["metadata"]["language"],
                "conditions": views,
            }
        )
    manifest = _manifest(release)
    tokenizer_registry = (
        "artifacts/datasets/nep-release-v1/comparison-tokenizers-v2/registry.json"
    )
    b1._checked_bytes(
        ROOT / tokenizer_registry,
        manifest["dependencies_sha256"].get(tokenizer_registry),
    )
    protocol = {
        "version": "passive-b2-preregistration-v1",
        "status": "design_registered_candidate_and_runtime_binding_pending",
        "release_manifest_sha256": hashlib.sha256(
            (release / "manifest.json").read_bytes()
        ).hexdigest(),
        "preparation_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "tokenizer_registry": {
            "path": tokenizer_registry,
            "sha256": manifest["dependencies_sha256"][tokenizer_registry],
        },
        "b2_predictions_consulted": False,
        "preparation_execution_scope": "All dev input/split/message hashes checked; selected train demonstrations replayed. Full dev execution reconstruction is performed by load_dev before benchmark inference.",
        "final_test_bodies_read": False,
        "scope": "Full exposed dev; passive text/caption; synthetic sessions and preferences; audio actions gated.",
        "registered_requests": len(cases),
        "case_order": "Exact frozen dev-cases.jsonl file order",
        "case_ids": [case["id"] for case in cases],
        "independent_source_groups": len(
            {case["metadata"]["source_group_id"] for case in cases}
        ),
        "candidate_binding": {
            "candidate_count": [2, 3],
            "selected_candidates": [],
            "icl_anchor_candidate": None,
            "required_before_any_b2_prediction": True,
            "selection": "Complete B1 plus bounded native-sampling diagnostics first. Record runtime/template/language/format viability and every exclusion. Do not infer a final ranking from B1. Bind 2-3 viable candidates and one ICL/history-only anchor before B2 results.",
        },
        "conditions": {
            "full": "All 289 requests for each bound candidate; recent history and causal feedback unchanged.",
            "context_only": "All 289 registered states for each candidate; remove recent_edits and recent_feedback only. Reuse explicitly byte-identical full requests; generate the changed requests once.",
            "history_only": "One bound anchor; remove recent_feedback only. Reuse explicitly byte-identical full requests; generate only changed requests.",
            "icl": "One bound anchor; full evidence plus fixed train-only two-shot examples. Input-only common-budget qualification precedes prediction; compare full and ICL on exactly the same qualified subset.",
        },
        "baseline_candidates": ["always_no_action", "consistent_history_rule"],
        "baseline_scope": "Both deterministic baselines use the original full 289 states and actual repeat gate.",
        "ablations": {
            "rule": "References become unscored when any nonempty evidence is removed. They do not become no_action. No postprediction adjudication or relabeling in this version.",
            "comparison": "Report fixed289 execution/failure/display denominators and the unchanged-evidence quality subset separately; this conservative subset does not estimate causal history benefit.",
            "full_application_state_retained": True,
            "counts": {
                condition: {
                    "reference_comparable": sum(
                        row["conditions"][condition]["reference_comparable"]
                        for row in registry
                    ),
                    "identical_request_reuse": sum(
                        row["conditions"][condition]["identical_to_full_request"]
                        for row in registry
                    ),
                }
                for condition in CONDITIONS
            },
        },
        "icl": {
            "shots_per_language": 2,
            "selection": "Train rows only: empty recent edits/feedback, slots no_action then correct_text; smallest worst-tokenizer frozen input count, lexical ID tie-break. No dev query, reference, lexical similarity, embedding or answer retrieval.",
            "demo_case_ids": {
                language: [case["id"] for case in values]
                for language, values in demos.items()
            },
            "caption_demonstrations": "No split/merge demonstrations in this bounded two-shot condition; report that limitation.",
            "budget": "Freeze complete full/ICL prefix counts and common_input_eligibility before inference, using the bound comparison tokenizer(s). No recropping; excluded input slots remain visible against 289. Runtime failures do not alter the eligible subset.",
            "input_budget_status": "pending_offline_or_runtime_tokenization_no_generation",
        },
        "runtime": {
            "context_tokens": 4096,
            "input_tokens": 3584,
            "output_tokens": 512,
            "temperature": 0,
            "seed": 42,
            "reasoning": "off",
            "schema_constrained": True,
            "cache_requested": False,
            "request_timeout_seconds": 120,
            "automatic_retries": 0,
            "policy": "One predictor at a time. Pin weights/runtime/tokenizers/templates/sampling and verify prompt token parity. Preserve every failure/timeout/cancellation; no rerun replaces a failed primary result. Stochastic/native runs are separate diagnostics.",
        },
        "request_budget": {
            "model_full": "289 * candidates",
            "model_context_additional": sum(
                not row["conditions"]["context_only"]["identical_to_full_request"]
                for row in registry
            ),
            "model_history_only_additional_anchor": sum(
                not row["conditions"]["history_only"]["identical_to_full_request"]
                for row in registry
            ),
            "model_icl_additional_anchor_max": len(cases),
            "max_for_two_candidates": 1101,
            "max_for_three_candidates": 1501,
            "deterministic_baseline_requests": 2 * len(cases),
            "sequential_additional_requests": 0,
        },
        "sequential_replay": {
            "episodes": {
                name: {key: episode[key] for key in ("case_ids", "feedback_conditions")}
                for name, episode in sequence_report(cases, [])["episodes"].items()
            },
            "scope": "Six fixed synthetic dev sessions, fifteen teacher-forced checkpoints replayed with original committed transitions and past-only ledger. Reuse main predictions; do not count them twice. The model does not drive subsequent states.",
            "coverage_limit": "Expiry, invalidation, dismissal, undo and committed corrections/boundaries are present; this is not full accept/dismiss/modify/undo on-policy coverage. Repair actions, human savings and natural preference accuracy remain unmeasured.",
        },
        "reporting": {
            "quality": "Raw format/semantic validity and outcomes, actual shown precision/coverage, unnecessary displays, missed useful edits, correct abstention, gate suppressions, all failures with fixed denominators.",
            "strata": [
                "action*language",
                "feedback_condition",
                "source",
                "episode",
                "release_stratum",
                "kind",
                "input_length",
                "reference_status",
            ],
            "uncertainty": "Per-source paired counts, source macro differences and leave-one-source-out sensitivity; eight groups, no row-independent significance claims.",
            "latency": "All measured requests including failures; missing latencies remain unmeasured. Reused rows are inherited observations, not extra latency samples. Cold load/RSS/swap/cache and editor state recorded. Uncontrolled resource timing is diagnostic, not official trigger-to-preview latency.",
            "selection": "At most two T1 candidates: execution integrity and local feasibility first, then unnecessary/meaning-changing edits, useful recall and controlled latency/memory Pareto tradeoff. No invented weighted score; no-action-only or gated invalids do not win by abstention.",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "case-registry.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in registry)
    )
    protocol["case_registry_sha256"] = hashlib.sha256(
        (output / "case-registry.jsonl").read_bytes()
    ).hexdigest()
    (output / "requests.jsonl.gz").write_bytes(
        gzip.compress(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in requests
            ).encode(),
            mtime=0,
        )
    )
    protocol["requests_sha256"] = hashlib.sha256(
        (output / "requests.jsonl.gz").read_bytes()
    ).hexdigest()
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = prepare_protocol(args.release, args.output)
    print(
        json.dumps(
            {
                key: protocol[key]
                for key in (
                    "version",
                    "status",
                    "registered_requests",
                    "request_budget",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
