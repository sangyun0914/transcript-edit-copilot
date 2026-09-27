"""Passive inputs derived from editor state, shared by offline preparation and serving.

Callers supply authoritative editor documents/settings and recorded state transitions.
This module validates their consistency, not the identity of the human or simulated
actor. No final/reference document or authored task brief enters predictor messages.
The runtime adapter below is Python preparation/validation, not live editor wiring or
proof of a particular inference server's chat-template implementation.
"""

import copy
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

import model_input
import nep_v4
from caption_facts import (
    CaptionProfile,
    ParagraphFacts,
    caption_facts,
    verify_caption_facts,
)
from context_policy import ContextBudgetError, FeedbackLedger
from nep import StrictModel
from nep_v2 import (
    ACTIONS as ACTIONS,
    EDIT as EDIT,
    Action,
    Envelope as Envelope,
    RecentEdit,
)
from nep_v3 import Feedback
from replay_session import session_hash
from session_cases import apply_edit, document_records

VERSION = "passive-nep-v1"
PROMPT_PATH = Path(__file__).with_name("prompt-passive-v1.txt")
PROMPT = PROMPT_PATH.read_text().strip()
ALLOWED_ACTIONS = ("no_action", "correct_text", "split_paragraph", "merge_paragraph")
GATED_ACTIONS = ("delete_span", "trim_silence", "remove_silence")
OUTPUT_SCHEMA = EDIT.json_schema()
_INTERNAL_BRIEF = "Preserve meaning, language, uncertainty and all source audio."
ITEM_FIELDS = model_input.ITEM_FIELDS[:-2]
_WIRE_FIELDS = {
    "item_fields",
    "caption_fact_fields",
    "caption_metric_fields",
    "complete_caption_fields",
    "complete_caption_facts",
}


@dataclass(frozen=True)
class PassivePolicy:
    version: str = VERSION
    max_items: int = 64
    input_tokens: int = 3584
    context_tokens: int = 4096
    output_tokens: int = 512
    history_limit: int = 3
    feedback_limit: int = 3

    def __post_init__(self):
        if self.version != VERSION:
            raise ValueError("Unknown passive policy version.")
        if any(
            type(value) is not int
            for key, value in asdict(self).items()
            if key != "version"
        ):
            raise ValueError("Passive context limits must be integers.")
        if not 1 <= self.max_items <= 256:
            raise ValueError("Item budget must be 1–256.")
        if not 0 <= self.history_limit <= 8 or not 0 <= self.feedback_limit <= 8:
            raise ValueError("History and feedback limits must be 0–8.")
        if min(self.input_tokens, self.output_tokens) < 1:
            raise ValueError("Token budgets must be positive.")
        if self.input_tokens + self.output_tokens > self.context_tokens:
            raise ValueError("Input and output reserves exceed the context budget.")


class PassiveItem(StrictModel):
    id: int = Field(ge=0)
    type: Literal["text", "non_text"]
    text: str
    duration_ms: int = Field(gt=0)
    paragraph: int = Field(ge=0)
    speaker: str | None
    language: str
    source_backed: bool
    source_run: int = Field(ge=0)


def _v4_data(value):
    result = copy.deepcopy(value)
    result.pop("contract_version")
    result.update(schema_version=4, task_brief=_INTERNAL_BRIEF)
    result["items"] = [
        {
            **item,
            "boundary_status": "unverified",
            "audio_class": "speech" if item["type"] == "text" else "unknown",
        }
        for item in result["items"]
    ]
    return result


class PassiveSnapshot(StrictModel):
    contract_version: Literal["passive-nep-v1"]
    task: Literal["predict_next_transcript_edit"]
    language: str
    mode: Literal["transcript_cleanup"]
    cursor_before: int
    selection: list[int]
    recent_edits: list[RecentEdit] = Field(max_length=8)
    recent_feedback: list[Feedback] = Field(max_length=8)
    items: list[PassiveItem] = Field(min_length=1, max_length=256)
    allowed_actions: list[Action]
    protected_ids: list[int] = Field(max_length=0)
    complete_paragraphs: list[int]
    caption_profile: CaptionProfile
    caption_facts: list[ParagraphFacts] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def references_and_eligibility(self):
        if tuple(self.allowed_actions) != ALLOWED_ACTIONS:
            raise ValueError(
                "Passive v1 audio cuts are gated without a runtime verifier."
            )
        if len(self.selection) != len(set(self.selection)):
            raise ValueError("Selection must not contain duplicate aliases.")
        if any(f.protected for f in self.caption_facts):
            raise ValueError("Passive v1 has no caller-authored protection overrides.")
        nep_v4.Snapshot.model_validate(_v4_data(self.model_dump()))
        return self


def parse_snapshot(value):
    """Parse either the strict readable contract or its lossless compact wire form."""
    if isinstance(value, PassiveSnapshot):
        value = value.model_dump()
    value = copy.deepcopy(value)
    if not isinstance(value, dict):
        raise ValueError("Passive input must be an object.")
    if "item_fields" not in value:
        return PassiveSnapshot.model_validate(value)
    if set(value) - (PassiveSnapshot.model_fields.keys() | _WIRE_FIELDS):
        raise ValueError("Unknown or privileged passive wire input keys.")
    if value.get("contract_version") != VERSION or value.pop("item_fields") != list(
        ITEM_FIELDS
    ):
        raise ValueError("Unknown passive wire version or item layout.")
    items = [dict(zip(ITEM_FIELDS, row, strict=True)) for row in value["items"]]
    for item in items:
        PassiveItem.model_validate(item)
    value["items"] = items
    legacy = _v4_data(value)
    legacy["item_fields"] = list(model_input.ITEM_FIELDS)
    legacy["items"] = [
        [item[key] for key in model_input.ITEM_FIELDS] for item in legacy["items"]
    ]
    parsed = model_input.parse_snapshot(legacy).model_dump()
    result = {
        key: parsed[key]
        for key in PassiveSnapshot.model_fields
        if key != "contract_version"
    }
    result["contract_version"] = VERSION
    result["items"] = [
        {key: item[key] for key in ITEM_FIELDS} for item in parsed["items"]
    ]
    return PassiveSnapshot.model_validate(result)


def wire_snapshot(value):
    snapshot = parse_snapshot(value).model_dump()
    compact = model_input.wire_snapshot(_v4_data(snapshot))
    result = {
        key: compact[key]
        for key in (*PassiveSnapshot.model_fields, *_WIRE_FIELDS)
        if key != "contract_version"
    }
    result["contract_version"] = VERSION
    result["item_fields"] = list(ITEM_FIELDS)
    result["items"] = [[item[key] for key in ITEM_FIELDS] for item in snapshot["items"]]
    # Explicit field order keeps canonical bytes independent of set/hash iteration.
    return {key: result[key] for key in sorted(result)}


def canonical_messages(value):
    return [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                wire_snapshot(value), ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]


def token_ids(tokenizer, value):
    return tokenizer.apply_chat_template(
        canonical_messages(value),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def validate_prediction(text, snapshot, request, current, finish_reason):
    return nep_v4.validate_prediction(
        text,
        nep_v4.Snapshot.model_validate(_v4_data(parse_snapshot(snapshot).model_dump())),
        request,
        current,
        finish_reason,
    )


def _records(document):
    if not isinstance(document, dict) or set(document) != {"content", "metadata"}:
        raise ValueError("Use an editor state with content and metadata only.")
    if not isinstance(document["metadata"], dict) or not isinstance(
        document["content"], list
    ):
        raise ValueError("Malformed editor state.")
    inside = False
    for item in document["content"]:
        if not isinstance(item, dict) or not isinstance(item.get("uuid"), str):
            raise ValueError("Document items need their actual identities.")
        if item["type"] == "paragraph_start":
            if inside:
                raise ValueError("Nested paragraph start.")
            inside = True
        elif item["type"] == "paragraph_end":
            if not inside:
                raise ValueError("Unmatched paragraph end.")
            inside = False
        elif item["type"] in {"text", "non_text", "artificial_silence"}:
            if not inside or not math.isfinite(item["length"]) or item["length"] <= 0:
                raise ValueError(
                    "Timed items need a positive finite duration inside a paragraph."
                )
            if item["type"] == "text" and not isinstance(item.get("text"), str):
                raise ValueError("Display text must be a string.")
            if "source" in item and (
                not math.isfinite(item["sourceStart"]) or item["sourceStart"] < 0
            ):
                raise ValueError("Invalid original source interval.")
        else:
            raise ValueError("Unknown editor item type.")
    if inside:
        raise ValueError("Unclosed paragraph.")
    records, mapping = document_records(document)
    if not records:
        raise ValueError("A passive window requires a current timed item.")
    return records, mapping


def _legacy_bundle(document, bundle):
    snapshot = parse_snapshot(bundle["input"]).model_dump()
    if bundle["document_hash"] != session_hash(document):
        raise ValueError("Stale source document.")
    records, mapping = _records(document)
    expected = []
    for item in snapshot["items"]:
        alias = item["id"]
        if (
            alias >= len(records)
            or bundle["mapping"].get(str(alias)) != mapping[str(alias)]
        ):
            raise ValueError("Alias mapping differs from the current document.")
        expected.append({key: records[alias][key] for key in ITEM_FIELDS})
    if snapshot["items"] != expected or set(bundle["mapping"]) != {
        str(i["id"]) for i in snapshot["items"]
    }:
        raise ValueError("Visible input differs from the actual current document.")
    if bundle.get("caption_authority") != {
        "version": 1,
        "protected_uuids": [],
        "profile": snapshot["caption_profile"],
    }:
        raise ValueError("Caption settings differ from their runtime authority.")
    result = {**bundle, "input": _v4_data(snapshot)}
    verify_caption_facts(document, result)
    return result


def apply_passive_edit(document, bundle, prediction):
    """Execute only after rebinding every visible fact and caption summary to this state."""
    return apply_edit(document, _legacy_bundle(document, bundle), prediction)


class CommittedTransition(StrictModel):
    sequence: int = Field(gt=0)
    origin: Literal["human", "ai_computer_use", "synthetic"]
    transaction_id: str = Field(min_length=1)
    before: dict
    after: dict
    prediction: dict
    undo_of: str | None = None

    @model_validator(mode="after")
    def actual_state_delta(self):
        from verify_release_export import verify_operation

        before, after = (
            (self.after, self.before) if self.undo_of else (self.before, self.after)
        )
        records, mapping = _records(before)
        _records(after)
        edit = EDIT.validate_python(self.prediction, strict=True)
        if edit.action == "no_action":
            raise ValueError("Abstention is not a committed editing transaction.")
        aliases = edit.target
        if aliases != sorted(set(aliases)) or any(
            str(a) not in mapping for a in aliases
        ):
            raise ValueError("History targets are not actual ordered unique aliases.")
        indices = [mapping[str(a)]["index"] for a in aliases]
        if edit.action not in {
            "split_paragraph",
            "merge_paragraph",
        } and indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("History span crosses hidden items or paragraph markers.")
        if (
            edit.action == "trim_silence"
            and edit.keep_ms >= records[aliases[0]]["duration_ms"]
        ):
            raise ValueError("Committed trim must shorten its actual interval.")
        verify_operation(before, after, mapping, edit.model_dump())
        return self


def committed_transition(
    before,
    after,
    prediction,
    sequence,
    origin="synthetic",
    transaction_id=None,
    *,
    undo_of=None,
):
    value = {
        "sequence": sequence,
        "origin": origin,
        "before": before,
        "after": after,
        "prediction": prediction,
        "undo_of": undo_of,
    }
    value["transaction_id"] = transaction_id or session_hash(value)
    return CommittedTransition.model_validate(value).model_dump()


def _history(document, history, as_of):
    if type(as_of) is not int or as_of < 0:
        raise ValueError("Prediction sequence must be a nonnegative integer.")
    transitions = [CommittedTransition.model_validate(value) for value in history]
    sequences = [t.sequence for t in transitions]
    if sequences != sorted(set(sequences)) or len(
        {t.transaction_id for t in transitions}
    ) != len(transitions):
        raise ValueError("Committed history must have unique chronological identities.")
    active, by_id, past = [], {}, []
    for transition in transitions:
        if transition.sequence > as_of:
            continue
        if past and session_hash(past[-1].after) != session_hash(transition.before):
            raise ValueError(
                "Committed history is not an actual consecutive state chain."
            )
        if transition.undo_of:
            original = by_id.get(transition.undo_of)
            if original is None or not active or active[-1] != original:
                raise ValueError(
                    "Undo must reverse the latest committed active transaction."
                )
            if (
                transition.before != original.after
                or transition.after != original.before
                or transition.prediction != original.prediction
            ):
                raise ValueError("Undo does not restore its actual prior state.")
            active.pop()
        else:
            active.append(transition)
        past.append(transition)
        by_id[transition.transaction_id] = transition
    if past and session_hash(past[-1].after) != session_hash(document):
        raise ValueError(
            "The final committed history state is not the current document."
        )
    summaries = []
    for transition in active:
        records, _ = _records(transition.before)
        edit = transition.prediction
        selected = [records[a] for a in edit["target"]]
        before_text = " ".join(i["text"] for i in selected if i["type"] == "text")
        fields = {
            "action": edit["action"],
            "origin": transition.origin,
            "before": before_text,
            "after": edit.get("text"),
        }
        if edit["action"] in GATED_ACTIONS:
            fields.update(
                before_ms=sum(i["duration_ms"] for i in selected),
                after_ms=edit.get("keep_ms", 0),
            )
        summaries.append(RecentEdit.model_validate(fields).model_dump())
    return past, summaries


class PassiveFeedbackLedger:
    """Actual proposal/outcome records; acceptance and undo require committed deltas."""

    def __init__(self, document_id):
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Feedback needs the actual session document identity.")
        self.document_id = document_id
        self._ledger = FeedbackLedger(document_id)
        self._commits, self._undos, self._records = {}, {}, []

    def propose(self, proposal_id, sequence, document, bundle, prediction):
        if type(sequence) is not int or sequence < bundle["context"]["as_of"]:
            raise ValueError("Proposal sequence predates its context.")
        legacy = _legacy_bundle(document, bundle)
        self._ledger.propose(proposal_id, sequence, document, legacy, prediction)
        self._records.append(
            copy.deepcopy(
                {
                    "kind": "propose",
                    "proposal_id": proposal_id,
                    "sequence": sequence,
                    "document": document,
                    "bundle": bundle,
                    "prediction": prediction,
                }
            )
        )

    def record(
        self,
        proposal_id,
        outcome,
        sequence,
        *,
        origin="synthetic",
        reason="unspecified",
        transition=None,
        applied=None,
        transaction_id=None,
    ):
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Feedback sequence must be a nonnegative integer.")
        event = {
            "proposal_id": proposal_id,
            "outcome": outcome,
            "sequence": sequence,
            "origin": origin,
            "reason": reason,
        }
        committed = None
        if outcome in {"accepted", "modified", "undone"}:
            if transition is None:
                raise ValueError(
                    "Accepted or undone feedback requires an actual committed transition."
                )
            committed = CommittedTransition.model_validate(transition)
            if committed.sequence != sequence or committed.origin != origin:
                raise ValueError(
                    "Feedback contradicts the committed sequence or actor."
                )
            if outcome == "undone":
                original = self._commits.get(proposal_id)
                if original is None or (
                    committed.undo_of != original["transaction_id"]
                    or committed.before != original["after"]
                    or committed.after != original["before"]
                    or committed.prediction != original["prediction"]
                ):
                    raise ValueError(
                        "Undo feedback does not reverse its accepted transaction."
                    )
                expected_id = original["transaction_id"]
                if applied is not None:
                    raise ValueError(
                        "Undo carries its state transition, not a new applied edit."
                    )
            else:
                if committed.undo_of is not None:
                    raise ValueError("Acceptance cannot describe an undo.")
                proposal = self._ledger.proposals[proposal_id]
                old_bundle = {
                    "input": proposal["snapshot"],
                    "mapping": proposal["mapping"],
                    "document_hash": proposal["document_hash"],
                    "caption_authority": proposal["caption_authority"],
                }
                if (
                    apply_edit(committed.before, old_bundle, committed.prediction)
                    != committed.after
                ):
                    raise ValueError(
                        "Accepted feedback differs from the executed proposal state."
                    )
                if applied is not None and applied != committed.prediction:
                    raise ValueError(
                        "Applied feedback differs from the committed edit."
                    )
                applied = committed.prediction
                expected_id = committed.transaction_id
            if transaction_id is not None and transaction_id != expected_id:
                raise ValueError(
                    "Feedback transaction identity contradicts its state transition."
                )
            transaction_id = expected_id
        elif (
            transition is not None or applied is not None or transaction_id is not None
        ):
            raise ValueError("Only an acceptance or undo has a committed transaction.")
        self._ledger.record(**event, applied=applied, transaction_id=transaction_id)
        if committed:
            (self._undos if outcome == "undone" else self._commits)[proposal_id] = (
                committed.model_dump()
            )
        self._records.append(
            copy.deepcopy(
                {
                    "kind": "record",
                    **event,
                    "transition": committed.model_dump() if committed else None,
                    "applied": applied,
                    "transaction_id": transaction_id,
                }
            )
        )

    def _verify_history(self, past, as_of):
        actual = {t.transaction_id: t.model_dump() for t in past}
        for event in self._ledger.events:
            if event["sequence"] > as_of:
                continue
            if event["outcome"] in {"accepted", "modified", "undone"}:
                records = self._undos if event["outcome"] == "undone" else self._commits
                transition = records[event["proposal_id"]]
                if actual.get(transition["transaction_id"]) != transition:
                    raise ValueError(
                        "Feedback acceptance/undo is absent from actual committed history."
                    )

    def context(self, document, mapping, as_of, limit):
        return self._ledger.context(document, _INTERNAL_BRIEF, mapping, as_of, limit)

    def suppressed(self, document, bundle, prediction, as_of):
        return self._ledger.suppressed(
            document, _legacy_bundle(document, bundle), prediction, as_of
        )

    def dump(self):
        return copy.deepcopy(
            {
                "contract_version": VERSION,
                "document_id": self.document_id,
                "records": self._records,
            }
        )

    @classmethod
    def restore(cls, value):
        if (
            set(value) != {"contract_version", "document_id", "records"}
            or value["contract_version"] != VERSION
        ):
            raise ValueError("Unknown passive feedback journal format.")
        ledger = cls(value["document_id"])
        for value_ in value["records"]:
            event = copy.deepcopy(value_)
            kind = event.pop("kind")
            if kind == "propose":
                ledger.propose(**event)
            elif kind == "record":
                ledger.record(**event)
            else:
                raise ValueError("Unknown passive feedback journal event.")
        return ledger


class PassiveContextBuilder:
    def __init__(self, tokenizer, budget_tokenizers=None, policy=None):
        self.tokenizer = tokenizer
        self.budget_tokenizers = dict(budget_tokenizers or {})
        self.policy = policy or PassivePolicy()
        if not isinstance(self.policy, PassivePolicy):
            raise ValueError("Use the separately versioned PassivePolicy.")

    def build(
        self,
        document,
        focus_uuid,
        *,
        as_of,
        selection=(),
        history=(),
        feedback=None,
        caption_profile=None,
    ):
        records, mapping = _records(document)
        past, summaries = _history(document, history, as_of)
        if feedback is not None:
            if not isinstance(feedback, PassiveFeedbackLedger):
                raise ValueError("Use actual passive proposal/outcome records.")
            feedback._verify_history(past, as_of)
        summaries = (
            summaries[-self.policy.history_limit :] if self.policy.history_limit else []
        )
        by_uuid = {v["uuid"]: int(k) for k, v in mapping.items()}
        if (
            focus_uuid not in by_uuid
            or not set(selection) <= by_uuid.keys()
            or len(selection) != len(set(selection))
        ):
            raise ValueError(
                "Cursor and selection must reference current unique timed items."
            )
        focus, selected = by_uuid[focus_uuid], [by_uuid[u] for u in selection]
        low, high = min([focus, *selected]), max([focus, *selected])
        if high - low + 1 > self.policy.max_items:
            raise ContextBudgetError("Selection and cursor exceed the item budget.")
        start = max(
            0,
            min(
                focus - self.policy.max_items // 2, len(records) - self.policy.max_items
            ),
        )
        start = max(min(start, low), high - self.policy.max_items + 1)
        end = min(len(records), start + self.policy.max_items)
        count = end - start
        totals = Counter(i["paragraph"] for i in records)
        profile = CaptionProfile.model_validate(caption_profile or {}).model_dump()
        while True:
            items = records[start:end]
            local_map = {str(i["id"]): mapping[str(i["id"])] for i in items}
            visible = Counter(i["paragraph"] for i in items)
            value = PassiveSnapshot.model_validate(
                {
                    "contract_version": VERSION,
                    "task": "predict_next_transcript_edit",
                    "language": records[focus]["language"],
                    "mode": "transcript_cleanup",
                    "cursor_before": focus,
                    "selection": selected,
                    "recent_edits": summaries,
                    "recent_feedback": feedback.context(
                        document, local_map, as_of, self.policy.feedback_limit
                    )
                    if feedback
                    else [],
                    "items": [{key: i[key] for key in ITEM_FIELDS} for i in items],
                    "allowed_actions": list(ALLOWED_ACTIONS),
                    "protected_ids": [],
                    "complete_paragraphs": [
                        p for p, n in visible.items() if totals[p] == n
                    ],
                    "caption_profile": profile,
                    "caption_facts": caption_facts(
                        document, records, items, [], profile
                    ),
                }
            ).model_dump()
            tokens = token_ids(self.tokenizer, value)
            comparison = {
                name: len(token_ids(t, value))
                for name, t in self.budget_tokenizers.items()
            }
            if max([len(tokens), *comparison.values()]) <= self.policy.input_tokens:
                break
            left, right = start < low, end - 1 > high
            if not left and not right:
                raise ContextBudgetError(
                    "Mandatory state and history exceed the complete passive prompt budget."
                )
            if left and (not right or focus - start > end - 1 - focus):
                start += 1
            else:
                end -= 1
        return {
            "input": value,
            "mapping": local_map,
            "document_hash": session_hash(document),
            "caption_authority": {
                "version": 1,
                "protected_uuids": [],
                "profile": profile,
            },
            "context": {
                "policy": asdict(self.policy),
                "as_of": as_of,
                "total_items": len(records),
                "candidate_items": count,
                "visible_items": len(items),
                "start_index": start,
                "end_index_exclusive": end,
                "visible_paragraphs": len(visible),
                "input_tokens": len(tokens),
                "comparison_token_counts": comparison,
                "prompt_hash": session_hash(canonical_messages(value)),
                "budget_reduced": count != len(items),
            },
        }


class PassivePredictionSession:
    """Use identical preparation at inference; cache issued bundles against later tampering."""

    def __init__(self, document_id, builder):
        self.builder = builder
        self.feedback = PassiveFeedbackLedger(document_id)
        self._prepared = set()

    def prepare(self, document, focus_uuid, **kwargs):
        bundle = self.builder.build(
            document, focus_uuid, feedback=self.feedback, **kwargs
        )
        self._prepared.add(session_hash(bundle))
        return bundle

    def validate(self, text, document, bundle, request, current, finish_reason, as_of):
        if (
            request.document_id != self.feedback.document_id
            or session_hash(bundle) not in self._prepared
        ):
            raise ValueError("Different session or modified/unissued runtime context.")
        if as_of != bundle["context"]["as_of"]:
            raise ValueError("Prediction sequence differs from its prepared context.")
        _legacy_bundle(document, bundle)
        edit = validate_prediction(
            text, bundle["input"], request, current, finish_reason
        )
        suppressed = self.feedback.suppressed(
            document, bundle, edit.model_dump(), as_of
        )
        return {
            "prediction": edit.model_dump(),
            "show": edit.action != "no_action" and not suppressed,
            "suppressed": suppressed,
        }
