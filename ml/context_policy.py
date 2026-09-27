"""Shared offline/runtime context construction and session-scoped proposal feedback."""

import copy
import json
from collections import Counter
from dataclasses import asdict, dataclass

from model_input import (
    messages,
    parse_snapshot,
    token_ids as encode,
    validate_prediction,
)

from nep_v2 import ACTIONS, EDIT, Envelope, RecentEdit
from nep_v3 import Feedback
from replay_session import session_hash
from session_cases import document_records


@dataclass(frozen=True)
class ContextPolicy:
    version: str = "cursor-feedback-v1"
    max_items: int = 64
    input_tokens: int = 3840
    context_tokens: int = 4096
    output_tokens: int = 256
    history_limit: int = 3
    feedback_limit: int = 3
    schema_version: int = 3

    def __post_init__(self):
        if self.schema_version not in (3, 4):
            raise ValueError("Context builder supports schema version 3 or 4.")
        if not 1 <= self.max_items <= 256:
            raise ValueError("Item limit must be 1–256.")
        if not 0 <= self.history_limit <= 8 or not 0 <= self.feedback_limit <= 8:
            raise ValueError("History and feedback limits must be 0–8.")
        if self.input_tokens < 1 or self.output_tokens < 1:
            raise ValueError("Token budgets must be positive.")
        if self.input_tokens + self.output_tokens > self.context_tokens:
            raise ValueError("Reserve output tokens inside the runtime context limit.")


class ContextBudgetError(ValueError):
    pass


def evidence_fingerprint(
    document, target_uuids, brief, version="paragraph-v1", action=None
):
    paragraphs, current = [], None
    for item in document["content"]:
        if item["type"] == "paragraph_start":
            current = []
            paragraphs.append(current)
        if current is not None:
            current.append(item)
    indices = [
        n for n, p in enumerate(paragraphs) if any(i["uuid"] in target_uuids for i in p)
    ]
    if len(
        {i["uuid"] for n in indices for i in paragraphs[n]} & set(target_uuids)
    ) != len(target_uuids):
        return None
    if version == "target-neighborhood-v2" and action not in {
        "split_paragraph",
        "merge_paragraph",
    }:
        neighborhoods = []
        for n in indices:
            paragraph = paragraphs[n]
            timed = [
                i
                for i in paragraph
                if i["type"] in {"text", "non_text", "artificial_silence"}
            ]
            targets = [n for n, i in enumerate(timed) if i["uuid"] in target_uuids]
            neighborhoods.append(
                {
                    "paragraph": paragraph[0],
                    "items": timed[max(0, min(targets) - 2) : max(targets) + 3],
                }
            )
        return session_hash({"neighborhoods": neighborhoods, "brief": brief})
    if version not in {"paragraph-v1", "target-neighborhood-v2"}:
        raise ValueError("Unknown feedback evidence policy.")
    return session_hash(
        {"paragraphs": [paragraphs[n] for n in indices], "brief": brief}
    )


def correction_effect(document, bundle, prediction):
    from grading import transcript_media
    from session_cases import apply_edit

    target = bundle["mapping"][str(prediction["target"][0])]["index"]
    paragraph_id = next(
        i["uuid"]
        for i in reversed(document["content"][:target])
        if i["type"] == "paragraph_start"
    )
    after = apply_edit(document, bundle, prediction)
    content, inside = [], False
    for item in after["content"]:
        if item["type"] == "paragraph_start":
            inside = item["uuid"] == paragraph_id
        if inside:
            content.append(item)
    return session_hash(
        {
            "paragraph_id": paragraph_id,
            "effect": transcript_media({"content": content, "metadata": {}}),
        }
    )


class FeedbackLedger:
    def __init__(self, document_id):
        self.document_id = document_id
        self.proposals = {}
        self.events = []

    def propose(self, proposal_id, sequence, document, bundle, prediction):
        if proposal_id in self.proposals or sequence < 0:
            raise ValueError("Proposal identity must be new and sequence nonnegative.")
        from caption_facts import verify_caption_facts

        verify_caption_facts(document, bundle)
        request = Envelope(
            document_id=self.document_id,
            request_id=proposal_id,
            revision=0,
            interaction_epoch=0,
        )
        edit = validate_prediction(
            json.dumps(prediction),
            parse_snapshot(bundle["input"]),
            request,
            request,
            "stop",
        )
        if edit.action == "no_action":
            raise ValueError("Abstention is not a displayed proposal.")
        if bundle["document_hash"] != session_hash(document):
            raise ValueError("Cannot record a stale proposal.")
        targets = [bundle["mapping"][str(t)]["uuid"] for t in edit.target]
        evidence_targets = list(targets)
        evidence_version = (
            "target-neighborhood-v2"
            if bundle["input"]["schema_version"] == 4
            else "paragraph-v1"
        )
        if edit.action == "merge_paragraph":
            position = next(
                n
                for n, i in enumerate(bundle["input"]["items"])
                if i["id"] == edit.target[0]
            )
            previous = bundle["input"]["items"][position - 1]["id"]
            evidence_targets.append(bundle["mapping"][str(previous)]["uuid"])
        self.proposals[proposal_id] = copy.deepcopy(
            {
                "sequence": sequence,
                "prediction": edit.model_dump(),
                "targets": targets,
                "evidence_targets": evidence_targets,
                "evidence_hash": evidence_fingerprint(
                    document,
                    evidence_targets,
                    bundle["input"]["task_brief"],
                    evidence_version,
                    edit.action,
                ),
                **(
                    {"evidence_version": evidence_version}
                    if evidence_version != "paragraph-v1"
                    else {}
                ),
                "input_hash": session_hash(bundle["input"]),
                "document_hash": bundle["document_hash"],
                "before": " ".join(
                    i["text"]
                    for i in bundle["input"]["items"]
                    if i["id"] in edit.target
                ),
                "snapshot": bundle["input"],
                "mapping": bundle["mapping"],
                **(
                    {"caption_authority": bundle["caption_authority"]}
                    if "caption_authority" in bundle
                    else {}
                ),
                "correction_effect": correction_effect(document, bundle, prediction)
                if edit.action == "correct_text"
                else None,
            }
        )

    def record(
        self,
        proposal_id,
        outcome,
        sequence,
        *,
        origin="synthetic",
        reason="unspecified",
        applied=None,
        transaction_id=None,
    ):
        if outcome not in {
            "shown",
            "accepted",
            "dismissed",
            "modified",
            "undone",
            "expired",
            "invalidated",
        }:
            raise ValueError("Unknown proposal outcome.")
        if origin not in {"human", "ai_computer_use", "synthetic"}:
            raise ValueError("Unknown actor.")
        if reason not in {"unspecified", "timing", "target", "parameter", "preference"}:
            raise ValueError("Unknown feedback reason.")
        proposal = self.proposals[proposal_id]
        prior = [e for e in self.events if e["proposal_id"] == proposal_id]
        last = prior[-1]["outcome"] if prior else None
        if sequence <= proposal["sequence"] or (
            self.events and sequence <= self.events[-1]["sequence"]
        ):
            raise ValueError("Feedback events must be strictly chronological.")
        if outcome == "shown":
            if prior:
                raise ValueError("A proposal may be exposed only once.")
        elif outcome == "invalidated" and last is None:
            pass
        elif last is None:
            raise ValueError("An unshown proposal cannot receive user feedback.")
        elif outcome == "undone":
            if last not in {"accepted", "modified"} or not transaction_id:
                raise ValueError("Undo must reference an accepted transaction.")
            if transaction_id != prior[-1]["transaction_id"]:
                raise ValueError(
                    "Undo transaction does not match the accepted proposal."
                )
        elif last != "shown":
            raise ValueError("This proposal already has a terminal decision.")
        if outcome in {"accepted", "modified"}:
            if not transaction_id or applied is None:
                raise ValueError(
                    "Accepted feedback requires the applied edit and transaction."
                )
            request = Envelope(
                document_id=self.document_id,
                request_id=proposal_id,
                revision=0,
                interaction_epoch=0,
            )
            validate_prediction(
                json.dumps(applied),
                parse_snapshot(proposal["snapshot"]),
                request,
                request,
                "stop",
            )
            if outcome == "accepted" and applied != proposal["prediction"]:
                raise ValueError(
                    "A changed edit must be recorded as modified acceptance."
                )
            if applied["action"] == "no_action":
                raise ValueError("Acceptance must apply an edit.")
        elif applied is not None:
            raise ValueError("Only acceptance carries an applied edit.")
        if outcome == "undone":
            applied = prior[-1]["applied"]
        self.events.append(
            copy.deepcopy(
                {
                    "proposal_id": proposal_id,
                    "outcome": outcome,
                    "sequence": sequence,
                    "origin": origin,
                    "reason": reason,
                    "applied": applied,
                    "transaction_id": transaction_id,
                }
            )
        )

    def outcomes(self, as_of):
        latest = {}
        for event in self.events:
            if event["sequence"] <= as_of:
                latest[event["proposal_id"]] = event
        return sorted(
            (e for e in latest.values() if e["outcome"] != "shown"),
            key=lambda e: e["sequence"],
        )

    def context(self, document, brief, mapping, as_of, limit):
        by_uuid = {v["uuid"]: int(k) for k, v in mapping.items()}
        result = []
        for event in self.outcomes(as_of)[-limit:] if limit else []:
            proposal = self.proposals[event["proposal_id"]]
            if not any(
                e["proposal_id"] == event["proposal_id"]
                and e["outcome"] == "shown"
                and e["sequence"] <= as_of
                for e in self.events
            ):
                continue
            edit, applied = proposal["prediction"], event["applied"] or {}
            same = (
                evidence_fingerprint(
                    document,
                    proposal["evidence_targets"],
                    brief,
                    proposal.get("evidence_version", "paragraph-v1"),
                    proposal["prediction"]["action"],
                )
                == proposal["evidence_hash"]
            )
            result.append(
                Feedback(
                    action=edit["action"],
                    outcome=event["outcome"],
                    origin=event["origin"],
                    before=proposal["before"],
                    proposed_text=edit.get("text"),
                    proposed_keep_ms=edit.get("keep_ms"),
                    applied_text=applied.get("text"),
                    applied_keep_ms=applied.get("keep_ms"),
                    visible_targets=[
                        by_uuid[u] for u in proposal["targets"] if u in by_uuid
                    ],
                    same_context=same,
                    reason=event["reason"],
                ).model_dump()
            )
        return result

    def suppressed(self, document, bundle, prediction, as_of):
        edit = EDIT.validate_python(prediction)
        if edit.action == "no_action":
            return False
        targets = [bundle["mapping"][str(t)]["uuid"] for t in edit.target]
        current_aliases = {v["uuid"]: int(k) for k, v in bundle["mapping"].items()}
        parameters = edit.model_dump(exclude={"target"})
        for event in self.outcomes(as_of):
            if event["outcome"] not in {"dismissed", "undone"}:
                continue
            proposal = self.proposals[event["proposal_id"]]
            refused = (
                event["applied"]
                if event["outcome"] == "undone"
                else proposal["prediction"]
            )
            refused_targets = [
                proposal["mapping"][str(t)]["uuid"] for t in refused["target"]
            ]
            rejected = {k: v for k, v in refused.items() if k != "target"}
            equivalent = targets == refused_targets and parameters == rejected
            if (
                event["outcome"] == "dismissed"
                and refused["action"] == edit.action == "correct_text"
            ):
                equivalent = (
                    equivalent
                    or correction_effect(document, bundle, prediction)
                    == proposal["correction_effect"]
                )
            if (
                event["outcome"] == "undone"
                and refused["action"] == edit.action == "correct_text"
                and all(u in current_aliases for u in refused_targets)
            ):
                restored = {
                    **refused,
                    "target": [current_aliases[u] for u in refused_targets],
                }
                try:
                    equivalent = equivalent or correction_effect(
                        document, bundle, prediction
                    ) == correction_effect(document, bundle, restored)
                except ValueError:
                    pass
            if not equivalent:
                continue
            if (
                evidence_fingerprint(
                    document,
                    proposal["evidence_targets"],
                    bundle["input"]["task_brief"],
                    proposal.get("evidence_version", "paragraph-v1"),
                    proposal["prediction"]["action"],
                )
                == proposal["evidence_hash"]
            ):
                return True
        return False

    def dump(self):
        return copy.deepcopy(
            {
                "version": 1,
                "document_id": self.document_id,
                "proposals": self.proposals,
                "events": self.events,
            }
        )

    @classmethod
    def restore(cls, value):
        if value.get("version") != 1:
            raise ValueError("Unknown feedback journal version.")
        ledger = cls(value["document_id"])
        for key, proposal in value["proposals"].items():
            if session_hash(proposal["snapshot"]) != proposal["input_hash"]:
                raise ValueError("Proposal snapshot hash mismatch.")
            request = Envelope(
                document_id=ledger.document_id,
                request_id=key,
                revision=0,
                interaction_epoch=0,
            )
            validate_prediction(
                json.dumps(proposal["prediction"]),
                parse_snapshot(proposal["snapshot"]),
                request,
                request,
                "stop",
            )
            targets = [
                proposal["mapping"][str(t)]["uuid"]
                for t in proposal["prediction"]["target"]
            ]
            if targets != proposal["targets"]:
                raise ValueError("Proposal target mapping mismatch.")
            ledger.proposals[key] = copy.deepcopy(proposal)
        for event in value["events"]:
            ledger.record(
                event["proposal_id"],
                event["outcome"],
                event["sequence"],
                origin=event["origin"],
                reason=event["reason"],
                applied=event["applied"]
                if event["outcome"] in {"accepted", "modified"}
                else None,
                transaction_id=event["transaction_id"],
            )
            if ledger.events[-1] != event:
                raise ValueError("Feedback replay differs from recorded outcome.")
        return ledger


class ContextBuilder:
    def __init__(
        self,
        tokenizer,
        prompt,
        policy=None,
        budget_tokenizers=None,
        budget_prompts=None,
    ):
        self.tokenizer, self.prompt = tokenizer, prompt
        self.policy = policy or ContextPolicy()
        self.budget_tokenizers = budget_tokenizers or {}
        self.budget_prompts = budget_prompts or {}

    def build(
        self,
        document,
        focus_uuid,
        brief,
        *,
        as_of,
        history=(),
        feedback=None,
        selection=(),
        protected=(),
        evidence=None,
        allowed_actions=None,
        caption_profile=None,
    ):
        if as_of < 0:
            raise ValueError("Prediction sequence must be nonnegative.")
        records, mapping = document_records(document, evidence)
        indices = {v["uuid"]: int(k) for k, v in mapping.items()}
        if (
            focus_uuid not in indices
            or not set(selection) <= indices.keys()
            or not set(protected) <= indices.keys()
        ):
            raise ValueError(
                "Focus, selection and protections must reference current timed items."
            )
        focus = indices[focus_uuid]
        selected = [indices[u] for u in selection]
        required = [focus, *selected]
        low, high = min(required), max(required)
        if high - low + 1 > self.policy.max_items:
            raise ContextBudgetError("Selection and cursor exceed the item budget.")
        sequences = [e["sequence"] for e in history]
        if sequences != sorted(set(sequences)) or any(n < 0 for n in sequences):
            raise ValueError("Committed history must be strictly chronological.")
        past = [
            RecentEdit.model_validate(e["edit"]).model_dump()
            for e in history
            if e["sequence"] <= as_of
        ]
        past = past[-self.policy.history_limit :] if self.policy.history_limit else []
        start = max(
            0,
            min(
                focus - self.policy.max_items // 2, len(records) - self.policy.max_items
            ),
        )
        start = min(start, low)
        start = max(start, high - self.policy.max_items + 1)
        end = min(len(records), start + self.policy.max_items)
        counts = Counter(i["paragraph"] for i in records)
        authority = None
        if self.policy.schema_version == 4:
            from caption_facts import CaptionProfile, caption_facts

            profile = CaptionProfile.model_validate(caption_profile or {}).model_dump()
            authority = {
                "version": 1,
                "protected_uuids": list(protected),
                "profile": profile,
            }
        elif caption_profile is not None:
            raise ValueError("Caption profiles require schema version 4.")
        candidate_count = end - start
        while True:
            items = records[start:end]
            local_map = {str(i["id"]): mapping[str(i["id"])] for i in items}
            visible = Counter(i["paragraph"] for i in items)
            value = parse_snapshot(
                {
                    "schema_version": self.policy.schema_version,
                    "task": "predict_next_transcript_edit",
                    "language": records[focus]["language"],
                    "mode": "transcript_cleanup",
                    "task_brief": brief,
                    "cursor_before": focus,
                    "selection": selected,
                    "recent_edits": past,
                    "recent_feedback": feedback.context(
                        document, brief, local_map, as_of, self.policy.feedback_limit
                    )
                    if feedback
                    else [],
                    "items": items,
                    "allowed_actions": ACTIONS
                    if allowed_actions is None
                    else allowed_actions,
                    "protected_ids": [
                        indices[u] for u in protected if start <= indices[u] < end
                    ],
                    "complete_paragraphs": [
                        p for p, n in visible.items() if counts[p] == n
                    ],
                    **(
                        {
                            "caption_profile": profile,
                            "caption_facts": caption_facts(
                                document,
                                records,
                                items,
                                [indices[u] for u in protected],
                                profile,
                            ),
                        }
                        if authority
                        else {}
                    ),
                }
            ).model_dump()
            tokens = encode(self.tokenizer, value, self.prompt)
            comparison_counts = {
                name: len(encode(tokenizer, value, self.prompt))
                for name, tokenizer in self.budget_tokenizers.items()
            }
            format_counts = {
                format_name: {
                    name: len(encode(tokenizer, value, prompt))
                    for name, tokenizer in {
                        "primary": self.tokenizer,
                        **self.budget_tokenizers,
                    }.items()
                }
                for format_name, prompt in self.budget_prompts.items()
            }
            if (
                max(
                    [
                        len(tokens),
                        *comparison_counts.values(),
                        *[
                            n
                            for counts_ in format_counts.values()
                            for n in counts_.values()
                        ],
                    ]
                )
                <= self.policy.input_tokens
            ):
                break
            can_left, can_right = start < low, end - 1 > high
            if not can_left and not can_right:
                raise ContextBudgetError(
                    "Mandatory items and history exceed the full-prompt token budget."
                )
            if can_left and (not can_right or focus - start > end - 1 - focus):
                start += 1
            else:
                end -= 1
        return {
            "input": value,
            "document_hash": session_hash(document),
            "mapping": local_map,
            **({"caption_authority": authority} if authority else {}),
            "context": {
                "policy": {
                    k: v
                    for k, v in asdict(self.policy).items()
                    if k != "schema_version" or v != 3
                },
                "as_of": as_of,
                "total_items": len(records),
                "candidate_items": candidate_count,
                "visible_items": len(items),
                "start_index": start,
                "end_index_exclusive": end,
                "visible_paragraphs": len(visible),
                "input_tokens": len(tokens),
                "prompt_hash": session_hash(messages(value, self.prompt)),
                "budget_reduced": candidate_count != len(items),
                **(
                    {"comparison_token_counts": comparison_counts}
                    if self.budget_tokenizers
                    else {}
                ),
                **(
                    {"format_token_counts": format_counts}
                    if self.budget_prompts
                    else {}
                ),
            },
        }


class PredictionSession:
    """Runtime adapter uses the same builder as dataset generation; it never applies edits."""

    def __init__(self, document_id, builder):
        self.builder = builder
        self.feedback = FeedbackLedger(document_id)

    def prepare(self, document, focus_uuid, brief, **kwargs):
        return self.builder.build(
            document, focus_uuid, brief, feedback=self.feedback, **kwargs
        )

    def validate(
        self,
        text,
        document,
        bundle,
        request,
        current,
        finish_reason,
        as_of,
        output_format="json",
    ):
        if request.document_id != self.feedback.document_id or bundle[
            "document_hash"
        ] != session_hash(document):
            raise ValueError("Stale or different runtime document.")
        from caption_facts import verify_caption_facts

        verify_caption_facts(document, bundle)
        validator = validate_prediction
        if output_format == "rewrite":
            from format_comparison import parse_rewrite

            validator = parse_rewrite
        elif output_format != "json":
            raise ValueError("Unknown prediction output format.")
        edit = validator(
            text,
            parse_snapshot(bundle["input"]),
            request,
            current,
            finish_reason,
        )
        suppressed = self.feedback.suppressed(
            document, bundle, edit.model_dump(), as_of
        )
        return {
            "prediction": edit.model_dump(),
            "show": edit.action != "no_action" and not suppressed,
            "suppressed": suppressed,
        }
