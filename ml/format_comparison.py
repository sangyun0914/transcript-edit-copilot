"""Versioned JSON/action and source-preserving scoped-after-state comparison views."""

import json
from typing import Annotated, Literal

from pydantic import Field

from model_input import messages, parse_snapshot, token_ids, validate_prediction
from nep import StrictModel
from nep_v2 import Envelope
from replay_session import session_hash

FORMAT_VERSION = "nep-format-comparison-v1"
BUDGETS = {
    "context_tokens": 4096,
    "common_input_tokens": 3584,
    "output_tokens": {"json": 256, "rewrite": 512},
    "completion_encoding": "plain UTF-8 JSON tokenization plus tokenizer EOS; generation prefix includes the pinned chat template",
}


class FullAudio(StrictModel):
    mode: Literal["full"]


class EdgeAudio(StrictModel):
    mode: Literal["edges"]
    keep_ms: int = Field(gt=0)


class AfterItem(StrictModel):
    ids: list[int] = Field(min_length=1, max_length=256)
    text: str | None = Field(max_length=2000)
    audio: Annotated[FullAudio | EdgeAudio, Field(discriminator="mode")]
    boundary_before: bool


class Rewrite(StrictModel):
    format_version: Literal[1]
    scope: list[int] = Field(max_length=256)
    after: list[AfterItem] = Field(max_length=256)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def request_for_format():
    return Envelope(
        document_id="format", request_id="format", revision=0, interaction_epoch=0
    )


def source_item(snapshot, item):
    fact = next(f for f in snapshot.caption_facts if f.paragraph == item.paragraph)
    return {
        "ids": [item.id],
        "text": item.text if item.type == "text" else None,
        "audio": {"mode": "full"},
        "boundary_before": item.id == fact.first_id,
    }


def serialize_rewrite(snapshot, action, policy=None):
    snapshot = parse_snapshot(snapshot) if isinstance(snapshot, dict) else snapshot
    if snapshot.schema_version != 4:
        raise ValueError("Scoped Rewrite v1 requires the v4 source/boundary snapshot.")
    request = request_for_format()
    edit = validate_prediction(
        compact(action), snapshot, request, request, "stop", policy
    )
    value = {"format_version": 1, "scope": [], "after": []}
    if edit.action == "no_action":
        return value
    positions = {i.id: n for n, i in enumerate(snapshot.items)}
    scope = edit.target
    if edit.action in ("split_paragraph", "merge_paragraph"):
        scope = [snapshot.items[positions[scope[0]] - 1].id, scope[0]]
    selected = [snapshot.items[positions[i]] for i in scope]
    after = [source_item(snapshot, i) for i in selected]
    if edit.action in ("delete_span", "remove_silence"):
        after = []
    elif edit.action == "correct_text":
        after = [{**after[0], "ids": list(scope), "text": edit.text}]
    elif edit.action == "trim_silence":
        after[0]["audio"] = {"mode": "edges", "keep_ms": edit.keep_ms}
    else:
        after[-1]["boundary_before"] = edit.action == "split_paragraph"
    return Rewrite.model_validate(
        {**value, "scope": scope, "after": after}
    ).model_dump()


def parse_rewrite(text, snapshot, request, current, finish_reason, policy=None):
    snapshot = parse_snapshot(snapshot) if isinstance(snapshot, dict) else snapshot
    if snapshot.schema_version != 4:
        raise ValueError("Scoped Rewrite v1 requires v4.")
    if request != current or finish_reason != "stop":
        raise ValueError("Stale or incomplete Rewrite response.")
    rewrite = Rewrite.model_validate_json(text, strict=True).model_dump()
    scope, after = rewrite["scope"], rewrite["after"]
    by_id = {i.id: i for i in snapshot.items}
    positions = {i.id: n for n, i in enumerate(snapshot.items)}
    if not scope:
        if after:
            raise ValueError("An empty scope cannot create source content.")
        action = {"action": "no_action"}
    else:
        if any(i not in by_id for i in scope):
            raise ValueError("Rewrite references an unknown source alias.")
        indices = [positions[i] for i in scope]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("Rewrite scope must be contiguous, ordered and unique.")
        before = [source_item(snapshot, by_id[i]) for i in scope]
        if not after:
            action = {
                "action": "remove_silence"
                if len(scope) == 1 and by_id[scope[0]].type == "non_text"
                else "delete_span",
                "target": scope,
            }
        elif len(after) == 1 and after[0]["ids"] == scope:
            state = after[0]
            if state["boundary_before"] != before[0]["boundary_before"]:
                raise ValueError(
                    "Rewrite combines a caption change with another operation."
                )
            if state["audio"]["mode"] == "edges":
                if len(scope) != 1 or state["text"] != before[0]["text"]:
                    raise ValueError(
                        "A trim cannot change text or merge source aliases."
                    )
                action = {
                    "action": "trim_silence",
                    "target": scope,
                    "keep_ms": state["audio"]["keep_ms"],
                }
            else:
                action = {
                    "action": "correct_text",
                    "target": scope,
                    "text": state["text"],
                }
        elif len(scope) == len(after) == 2:
            expected = [dict(i) for i in before]
            expected[1]["boundary_before"] = not expected[1]["boundary_before"]
            if after != expected:
                raise ValueError(
                    "Rewrite changes immutable source state or performs multiple operations."
                )
            action = {
                "action": "split_paragraph"
                if after[1]["boundary_before"]
                else "merge_paragraph",
                "target": [scope[1]],
            }
        else:
            raise ValueError("Rewrite must express exactly one supported transaction.")
    edit = validate_prediction(
        compact(action), snapshot, request, current, finish_reason, policy
    )
    if serialize_rewrite(snapshot, edit.model_dump(), policy) != rewrite:
        raise ValueError("Rewrite contains redundant scope or extra source changes.")
    return edit


def build_format_comparison(cases, tokenizers, json_prompt, rewrite_prompt):
    """Reject overflow; never use references to choose or shrink a context window."""
    if not tokenizers:
        raise ValueError("Pin comparison tokenizers before measuring format budgets.")
    contexts = {
        name: {identity: [] for identity in tokenizers} for name in ("json", "rewrite")
    }
    maxima = {
        name: {identity: {"input": 0, "output": 0} for identity in tokenizers}
        for name in contexts
    }
    targets, request = [], request_for_format()
    for case in cases:
        snapshot = parse_snapshot(case["input"])
        actions = [case["expectedOutput"], *case.get("acceptableOutputs", [])]
        if any(a is None for a in actions):
            raise ValueError("Unresolved references are not format-comparison cases.")
        outputs = {
            "json": [compact(a) for a in actions],
            "rewrite": [compact(serialize_rewrite(snapshot, a)) for a in actions],
        }
        for raw, rewritten in zip(actions, outputs["rewrite"], strict=True):
            if (
                parse_rewrite(
                    rewritten, snapshot, request, request, "stop"
                ).model_dump()
                != raw
            ):
                raise ValueError(
                    "Rewrite and JSON no longer represent the same transaction."
                )
        targets.append(
            {
                "case_id": case["id"],
                "semantic_input_sha256": session_hash(case["input"]),
                "raw_targets": actions,
                "outputs": outputs,
            }
        )
        for name, prompt in (("json", json_prompt), ("rewrite", rewrite_prompt)):
            for identity, tokenizer in tokenizers.items():
                prefix = token_ids(tokenizer, case["input"], prompt)
                if tokenizer.eos_token_id is None:
                    raise ValueError("A pinned comparison tokenizer has no EOS token.")
                completions = [
                    tokenizer.encode(s, add_special_tokens=False)
                    + [tokenizer.eos_token_id]
                    for s in outputs[name]
                ]
                largest = max(map(len, completions))
                if (
                    len(prefix) > BUDGETS["common_input_tokens"]
                    or largest > BUDGETS["output_tokens"][name]
                ):
                    raise ValueError(
                        f"{case['id']} {identity} {name}: input {len(prefix)} or output {largest} exceeds the frozen format reserve."
                    )
                contexts[name][identity].append(
                    {
                        "case_id": case["id"],
                        "messages": messages(case["input"], prompt),
                        "input_ids": prefix,
                        "output_ids_including_eos": completions,
                        "completion_labels": [
                            [-100] * len(prefix) + completion
                            for completion in completions
                        ],
                        "semantic_input_sha256": session_hash(case["input"]),
                        "raw_target_sha256": session_hash(actions),
                    }
                )
                maxima[name][identity]["input"] = max(
                    maxima[name][identity]["input"], len(prefix)
                )
                maxima[name][identity]["output"] = max(
                    maxima[name][identity]["output"], largest
                )
    return {
        "spec": {
            "version": FORMAT_VERSION,
            "budgets": BUDGETS,
            "measured_maxima": maxima,
            "cases": len(cases),
            "tokenizer_ids": list(tokenizers),
            "roundtrip": "passed_all_references",
            "training_status": "not_run",
            "allocation_basis": "Preliminary serialization of 52 restored train/dev states across the five pinned tokenizers required at most 44 JSON / 86 Rewrite completion tokens including EOS. A 3072-token shared input cropped three other existing references, so the common input was fixed at 3584 and Rewrite reserve at 512 before final case auditing; no model outcomes were consulted.",
            "parity_scope": "Canonical v4 evidence losslessly serialized through model_input.messages for offline data and PredictionSession; all raw targets roundtrip through both output formats. Hugging Face token IDs use pinned files/templates. Actual llama-server/GGUF template and tokenizer parity is not certified by this audit and requires L0/B1 runtime preflight.",
        },
        "targets": targets,
        "contexts": contexts,
    }
