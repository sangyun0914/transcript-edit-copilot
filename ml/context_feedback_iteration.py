"""Generate replayed bilingual cursor/feedback states through the shared context builder."""

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from context_policy import (
    ContextBuilder,
    ContextPolicy,
    FeedbackLedger,
    encode,
    messages,
)
from iteration import HERE, jsonl, sha
from nep_v2 import Envelope
from nep_v3 import Snapshot, validate_prediction
from replay_session import replay, session_hash
from session_cases import apply_edit, replayable_session
from synthetic_pilot import document_for, execute_history, symbolic_evidence

FAMILIES = (
    "history_supported",
    "history_absent",
    "history_conflicting",
    "clean",
    "clear_error",
    "protected",
    "dismissed_same",
    "dismissed_other",
    "changed_evidence",
    "expired",
    "invalidated",
    "accepted_then_undone",
    "modified_acceptance",
    "future_rejection",
    "two_opportunities",
    "trim_history",
    "trim_dismissed",
    "trim_modified",
    "split",
    "merge",
    "delete_repeat",
    "remove_pause",
)


def case_state(group, family, builder, output):
    case_id = f"{group['id']}-{family}"
    english = group["language"] == "en"
    brief = (
        "Suggest one useful transcript edit near the cursor. Preserve meaning and deliberate emphasis. "
        "Use established spelling, respect recent decisions, and leave uncertain names unchanged."
        if english
        else "커서 주변에서 필요한 편집 하나를 제안하세요. 의미와 의도적인 강조를 보존하세요. "
        "확인된 표기를 일관되게 사용하고 최근 결정을 존중하되 불확실한 이름은 그대로 두세요."
    )
    clean = family in {
        "clean",
        "clear_error",
        "trim_history",
        "trim_dismissed",
        "trim_modified",
        "split",
        "merge",
        "delete_repeat",
        "remove_pause",
    }
    paragraphs = []
    for p, text in enumerate(group["paragraphs"]):
        text = text.replace(
            "{name}", group["canonical"] if clean else group["name"]
        ).replace("{term}", group["term"])
        paragraphs.append(
            [
                {"id": f"w{p}-{n}", "text": word, "ms": 250 + (n % 5) * 75}
                for n, word in enumerate(text.split())
            ]
        )
    central = paragraphs[2]
    name_word = next(
        w for w in central if w["text"] in {group["name"], group["canonical"]}
    )
    term_word = next(w for w in central if w["text"] == group["term"])
    sentence_boundary = next(
        n + 1 for n, w in enumerate(central[:-1]) if w["text"].endswith(".")
    )
    focus_slot = [2, sentence_boundary + 2, max(1, sentence_boundary - 3)][
        int(session_hash(group["id"])[:2], 16) % 3
    ]
    focus_uuid = central[focus_slot]["id"]
    target_uuid, replacement = name_word["id"], group["canonical"]
    protected = []
    if family in {"clear_error", "two_opportunities"}:
        term_word["text"] = group["typo"]
        if family == "clear_error":
            target_uuid, replacement = term_word["id"], group["term"]
    if family.startswith("trim_") or family == "remove_pause":
        target_uuid = "current-pause"
        central.insert(sentence_boundary, {"id": target_uuid, "text": "", "ms": 3500})
        if family == "remove_pause":
            brief += (
                " For this passage, remove verified pauses completely."
                if english
                else " 이 구간에서는 확인된 무음을 완전히 제거하세요."
            )
        else:
            brief += (
                " Shorten verified pauses of at least 3000ms to the retained duration consistently chosen recently; without that evidence, keep them."
                if english
                else " 확인된 3000ms 이상의 무음은 최근 일관되게 선택한 길이로 줄이되 그 근거가 없으면 보존하세요."
            )
    if family == "split":
        target_uuid = central[sentence_boundary]["id"]
        brief += (
            " Keep each complete sentence in its own caption."
            if english
            else " 각 완결된 문장을 별도의 자막 문단으로 나누세요."
        )
    if family == "merge":
        break_at = 4
        target_uuid = central[break_at]["id"]
        paragraphs[2:3] = [central[:break_at], central[break_at:]]
        brief += (
            " Join adjacent captions that split a sentence."
            if english
            else " 한 문장을 중간에서 나눈 인접 자막 문단은 합치세요."
        )
    if family == "delete_repeat":
        position = next(
            (n for n, w in enumerate(central) if w["text"] in ("the", "동안", "함께")),
            1,
        )
        duplicate = {**central[position], "id": "accidental-repeat"}
        central.insert(position, duplicate)
        target_uuid = duplicate["id"]
        brief += (
            " Remove accidental immediate duplicate words, preserving deliberate emphasis."
            if english
            else " 의도적인 강조는 보존하고 실수로 바로 반복된 단어는 제거하세요."
        )
    history = [
        {
            "action": "correct_text",
            "origin": "synthetic",
            "before": group["name"],
            "after": group["canonical"],
        }
    ]
    if clean or family == "history_absent":
        history = []
    if family == "history_conflicting":
        history.append({**history[0], "after": group["alternate"]})
    if family.startswith("trim_"):
        history = [
            {
                "action": "trim_silence",
                "origin": "synthetic",
                "before_ms": 3000,
                "after_ms": 1500,
            }
        ]
    document = document_for({"language": group["language"], "paragraphs": paragraphs})
    document, history_paths = execute_history(
        document,
        {
            "id": case_id,
            "brief": brief,
            "history": history,
            "language": group["language"],
        },
        output,
    )
    history_events = [
        {"sequence": n + 1, "edit": edit} for n, edit in enumerate(history)
    ]
    ledger = FeedbackLedger(case_id)
    if family == "protected":
        protected = [target_uuid]

    def snapshot(sequence=30):
        return builder.build(
            document,
            focus_uuid,
            brief,
            as_of=sequence,
            history=history_events,
            feedback=ledger,
            protected=protected,
            evidence=symbolic_evidence(document),
        )

    def alias(bundle, uuid):
        return next(int(k) for k, v in bundle["mapping"].items() if v["uuid"] == uuid)

    bundle = snapshot()
    target = alias(bundle, target_uuid)
    candidate = {"action": "correct_text", "target": [target], "text": replacement}
    if family.startswith("trim_"):
        candidate = {"action": "trim_silence", "target": [target], "keep_ms": 1500}
    elif family in {"split", "merge", "delete_repeat", "remove_pause"}:
        candidate = {
            "action": {
                "split": "split_paragraph",
                "merge": "merge_paragraph",
                "delete_repeat": "delete_span",
                "remove_pause": "remove_silence",
            }[family],
            "target": [target],
        }
    feedback_families = {
        "dismissed_same",
        "dismissed_other",
        "changed_evidence",
        "expired",
        "invalidated",
        "accepted_then_undone",
        "modified_acceptance",
        "future_rejection",
        "trim_dismissed",
        "trim_modified",
    }
    if family in feedback_families:
        proposal = copy.deepcopy(candidate)
        proposal_bundle = bundle
        if family == "dismissed_other":
            other = next(
                i
                for i in bundle["input"]["items"]
                if i["text"] == group["foil"] or i["text"].startswith(group["foil"])
            )
            proposal = {
                "action": "correct_text",
                "target": [other["id"]],
                "text": "Other spelling" if english else "다른 표기",
            }
        if family == "trim_modified":
            proposal_bundle = builder.build(
                document, "h0", brief, as_of=20, evidence=symbolic_evidence(document)
            )
            proposal = {
                "action": "trim_silence",
                "target": [alias(proposal_bundle, "h0")],
                "keep_ms": 500,
            }
            # The executed history already retained 1500ms. Replay the earlier proposal on its actual 3000ms pre-state.
            events = [
                json.loads(s)
                for s in (output / history_paths[0]).read_text().splitlines()
            ]
            earlier = events[0]["document"]
            proposal_bundle = builder.build(
                earlier, "h0", brief, as_of=20, evidence=symbolic_evidence(earlier)
            )
            proposal["target"] = [alias(proposal_bundle, "h0")]
            ledger.propose("proposal", 20, earlier, proposal_bundle, proposal)
        else:
            ledger.propose("proposal", 20, document, proposal_bundle, proposal)
        ledger.record("proposal", "shown", 21)
        outcome = {
            "expired": "expired",
            "invalidated": "invalidated",
            "accepted_then_undone": "accepted",
            "modified_acceptance": "modified",
            "trim_modified": "modified",
        }.get(family, "dismissed")
        decision_sequence = 40 if family == "future_rejection" else 22
        if outcome in {"accepted", "modified"}:
            applied = copy.deepcopy(proposal)
            if family == "modified_acceptance":
                applied["text"] = group["alternate"]
            if family == "trim_modified":
                applied["keep_ms"] = 1500
            ledger.record(
                "proposal",
                outcome,
                decision_sequence,
                applied=applied,
                transaction_id="transaction",
                reason="parameter" if family == "trim_modified" else "unspecified",
            )
            if family != "trim_modified":
                before = document
                after = apply_edit(document, bundle, applied)
                events = replayable_session(before, after, applied["action"], {})
                if family == "accepted_then_undone":
                    events = events[:3] + [
                        {
                            "kind": "end",
                            "seq": 3,
                            "revision": 2,
                            "document_hash": session_hash(before),
                            "transactions": 2,
                        }
                    ]
                    ledger.record(
                        "proposal", "undone", 23, transaction_id="transaction"
                    )
                document, _ = replay(events)
                path = f"sessions/{case_id}.acceptance.jsonl"
                jsonl(output / path, events)
                history_paths.append(path)
        else:
            ledger.record("proposal", outcome, decision_sequence)
        if family == "changed_evidence":
            brief += (
                f" A newly checked name register confirms {group['canonical']} for {group['name']}."
                if english
                else f" 새로 확인한 이름 명부에서 {group['name']}의 표기가 {group['canonical']}임을 확인했습니다."
            )
        bundle = snapshot()
        candidate["target"] = [alias(bundle, target_uuid)]
    abstain = family in {
        "history_absent",
        "history_conflicting",
        "clean",
        "protected",
        "dismissed_same",
        "accepted_then_undone",
        "modified_acceptance",
        "trim_dismissed",
    }
    expected = {"action": "no_action"} if abstain else candidate
    alternatives = []
    if family == "two_opportunities":
        alternatives.append(
            {
                "action": "correct_text",
                "target": [alias(bundle, term_word["id"])],
                "text": group["term"],
            }
        )
    return {
        "id": case_id,
        "group": group,
        "family": family,
        "document": document,
        "bundle": bundle,
        "history": history_events,
        "ledger": ledger,
        "focus": focus_uuid,
        "brief": brief,
        "protected": protected,
        "history_paths": history_paths,
        "expected": expected,
        "alternatives": alternatives,
    }


def build(recipe_path, tokenizer, output):
    recipe = json.loads(recipe_path.read_text())
    groups = {g["id"]: g["split"] for g in recipe["groups"]}
    if len(groups) != len(recipe["groups"]) or set(groups) & set(
        recipe["reserved_test_groups"]
    ):
        raise ValueError("Group IDs must be unique and exclude reserved tests.")
    if any(split not in {"train", "dev"} for split in groups.values()):
        raise ValueError("This generator is for exposed train/dev only.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "sessions").mkdir()
    (output / "feedback").mkdir()
    prompt = (HERE / "prompt-v3.txt").read_text()
    builder = ContextBuilder(tokenizer, prompt)
    cases, executions, contexts, sweeps = [], [], [], []
    for group in recipe["groups"]:
        for family in FAMILIES:
            state = case_state(group, family, builder, output)
            bundle, expected = state["bundle"], state["expected"]
            request = Envelope(
                document_id=state["id"],
                request_id="reference",
                revision=0,
                interaction_epoch=0,
            )
            for edit in [expected, *state["alternatives"]]:
                validate_prediction(
                    json.dumps(edit),
                    Snapshot.model_validate(bundle["input"]),
                    request,
                    request,
                    "stop",
                )
            after = apply_edit(state["document"], bundle, expected)
            events = replayable_session(
                state["document"], after, expected["action"], {}
            )
            session = f"sessions/{state['id']}.jsonl"
            jsonl(output / session, events)
            feedback = f"feedback/{state['id']}.json"
            (output / feedback).write_text(
                json.dumps(state["ledger"].dump(), ensure_ascii=False, indent=2) + "\n"
            )
            case = {
                "id": state["id"],
                "input": bundle["input"],
                "expectedOutput": expected,
                "acceptableOutputs": state["alternatives"],
                "metadata": {
                    "split": group["split"],
                    "source_group_id": group["id"],
                    "scenario_variant": family,
                    "actor": "synthetic",
                    "target_origin": "ai_generated_recipe",
                    "content_origin": "independent_ai_authored_script",
                    "context_origin": "shared_cursor_policy_v1",
                    "history_origin": "executed_synthetic",
                    "training_eligible": group["split"] == "train",
                    "requires_regeneration": False,
                    "review_status": "agent_validated_observable_rule",
                    "timing_origin": "synthetic_word_items_not_measured_alignment",
                    "audio_quality": "not_established",
                    "session": session,
                    "history_sessions": state["history_paths"],
                    "feedback_journal": feedback,
                    "context": bundle["context"],
                    "focus_origin": "source_group_selects_early_middle_or_late_cursor_before_reference_selection",
                },
            }
            cases.append(case)
            executions.append(
                {
                    "id": state["id"],
                    "before": state["document"],
                    "mapping": bundle["mapping"],
                }
            )
            ids = encode(tokenizer, case["input"], prompt)
            contexts.append(
                {
                    "case_id": case["id"],
                    "messages": messages(case["input"], prompt),
                    "input_ids": ids,
                    "token_count": len(ids),
                }
            )
            for tokens in (2048, 3840):
                for limit in (0, 3, 8):
                    policy = ContextPolicy(
                        input_tokens=tokens, history_limit=limit, feedback_limit=limit
                    )
                    sample = ContextBuilder(tokenizer, prompt, policy).build(
                        state["document"],
                        state["focus"],
                        state["brief"],
                        as_of=30,
                        history=state["history"],
                        feedback=state["ledger"],
                        protected=state["protected"],
                        evidence=symbolic_evidence(state["document"]),
                    )
                    visible = {i["id"] for i in sample["input"]["items"]}
                    sweeps.append(
                        {
                            "case_id": case["id"],
                            "input_budget": tokens,
                            "history_limit": limit,
                            **sample["context"],
                            "reference_targets_visible": set(expected.get("target", []))
                            <= visible,
                        }
                    )
    for split in ("train", "dev"):
        selected = [c for c in cases if c["metadata"]["split"] == split]
        jsonl(output / f"{split}-cases.jsonl", selected)
    jsonl(output / "executions.jsonl", executions)
    jsonl(output / "contexts.jsonl", contexts)
    jsonl(output / "context-sweep.jsonl", sweeps)
    (output / "prompt.txt").write_text(prompt)
    (output / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    (output / "recipe.json").write_bytes(recipe_path.read_bytes())
    by_split = {
        split: [c for c in cases if c["metadata"]["split"] == split]
        for split in ("train", "dev")
    }
    signatures = {
        s: {json.dumps(c["input"], sort_keys=True, ensure_ascii=False) for c in cs}
        for s, cs in by_split.items()
    }
    texts = {
        s: {
            re.sub(
                r"\s+", " ", " ".join(i["text"] for i in c["input"]["items"])
            ).casefold()
            for c in cs
        }
        for s, cs in by_split.items()
    }
    if signatures["train"] & signatures["dev"] or texts["train"] & texts["dev"]:
        raise ValueError("Exact input or full transcript crosses splits.")
    report = {
        "version": 1,
        "status": "generated_validated_not_trained",
        "question": recipe["question"],
        "scope": recipe["scope"],
        "groups": groups,
        "reserved_test_groups": recipe["reserved_test_groups"],
        "counts": {s: len(cs) for s, cs in by_split.items()},
        "actions": {
            s: dict(Counter(c["expectedOutput"]["action"] for c in cs))
            for s, cs in by_split.items()
        },
        "languages": {
            s: dict(Counter(c["input"]["language"] for c in cs))
            for s, cs in by_split.items()
        },
        "policy": bundle["context"]["policy"],
        "test_cases": 0,
        "code_sha256": {
            name: sha(HERE / name)
            for name in (
                "context_feedback_iteration.py",
                "context_policy.py",
                "model_input.py",
                "nep_v3.py",
                "session_cases.py",
                "prompt-v3.txt",
            )
        },
        "files": {
            str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", type=Path, default=HERE / "fixtures/context-feedback-v1.json"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    result = build(
        args.recipe,
        AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True),
        args.output,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("counts", "actions", "languages", "status")},
            ensure_ascii=False,
        )
    )
