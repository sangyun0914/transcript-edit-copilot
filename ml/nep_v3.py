"""Versioned past-feedback input; edit semantics remain the validated v2 contract."""

from typing import Literal

from pydantic import Field, model_validator

import nep_v2
from nep import StrictModel
from nep_v2 import ACTIONS, EDIT, Envelope, Policy  # noqa: F401

VERSION = 3


class Feedback(StrictModel):
    action: nep_v2.Action
    outcome: Literal[
        "accepted", "dismissed", "modified", "undone", "expired", "invalidated"
    ]
    origin: Literal["human", "ai_computer_use", "synthetic"]
    before: str
    proposed_text: str | None = None
    proposed_keep_ms: int | None = Field(default=None, gt=0)
    applied_text: str | None = None
    applied_keep_ms: int | None = Field(default=None, gt=0)
    visible_targets: list[int]
    same_context: bool
    reason: Literal["unspecified", "timing", "target", "parameter", "preference"] = (
        "unspecified"
    )


class Snapshot(nep_v2.Snapshot):
    schema_version: Literal[3]
    recent_feedback: list[Feedback] = Field(max_length=8)

    @model_validator(mode="after")
    def feedback_references(self):
        ids = {i.id for i in self.items}
        for feedback in self.recent_feedback:
            if (
                len(feedback.visible_targets) != len(set(feedback.visible_targets))
                or not set(feedback.visible_targets) <= ids
            ):
                raise ValueError(
                    "Feedback references unknown or duplicate visible targets."
                )
        return self


def legacy_snapshot(value):
    parsed = Snapshot.model_validate(value).model_dump()
    parsed.pop("recent_feedback")
    parsed["schema_version"] = 2
    return nep_v2.Snapshot.model_validate(parsed)


def validate_prediction(text, snapshot, request, current, finish_reason, policy=None):
    return nep_v2.validate_prediction(
        text,
        legacy_snapshot(snapshot.model_dump()),
        request,
        current,
        finish_reason,
        policy,
    )
