"""The offline NEP contract. Predictions are validated, never applied to media."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class NoAction(StrictModel):
    action: Literal["no_action"]


class DeleteSpan(StrictModel):
    action: Literal["delete_span"]
    target: list[int] = Field(min_length=1, max_length=8)


class CorrectText(StrictModel):
    action: Literal["correct_text"]
    target: list[int] = Field(min_length=1, max_length=1)
    text: str = Field(min_length=1, max_length=120)


class TrimSilence(StrictModel):
    action: Literal["trim_silence"]
    target: list[int] = Field(min_length=1, max_length=1)
    keep_ms: Literal[200, 300, 500]


Edit = Annotated[
    NoAction | DeleteSpan | CorrectText | TrimSilence, Field(discriminator="action")
]
EDIT = TypeAdapter(Edit)
Action = Literal["no_action", "delete_span", "correct_text", "trim_silence"]


class Item(StrictModel):
    id: int = Field(ge=0)
    type: Literal["text", "non_text"]
    text: str = ""
    duration_ms: int = Field(gt=0)
    paragraph: int = Field(ge=0)
    speaker: str | None
    language: str
    source_backed: bool = True
    source_run: int = Field(ge=0)
    boundary_status: Literal["verified", "unverified", "needs_alignment", "overlap"]
    audio_class: Literal["speech", "verified_silence", "unknown", "audio_event"]


class Snapshot(StrictModel):
    task: Literal["predict_next_transcript_edit"]
    language: str
    mode: Literal["transcript_cleanup"]
    cursor_before: int | None
    selection: list[int]
    recent_edits: list[dict] = Field(max_length=8)
    items: list[Item] = Field(min_length=1, max_length=64)
    allowed_actions: list[Action]
    protected_ids: list[int]

    @model_validator(mode="after")
    def aliases(self):
        ids = {item.id for item in self.items}
        if len(ids) != len(self.items):
            raise ValueError("Item aliases must be unique.")
        referenced = set(self.selection + self.protected_ids)
        if self.cursor_before is not None:
            referenced.add(self.cursor_before)
        if not referenced <= ids:
            raise ValueError("Snapshot references an unknown item.")
        if "no_action" not in self.allowed_actions:
            raise ValueError("Abstention must remain available.")
        return self


class Envelope(StrictModel):
    version: Literal[1] = 1
    document_id: str
    request_id: str
    revision: int = Field(ge=0)
    interaction_epoch: int = Field(ge=0)


def validate_prediction(
    text: str,
    snapshot: Snapshot,
    request: Envelope,
    current: Envelope,
    finish_reason: str,
) -> Edit:
    if request != current:
        raise ValueError("Stale document or interaction context.")
    if finish_reason != "stop":
        raise ValueError("Incomplete model response.")
    edit = EDIT.validate_json(text, strict=True)
    if edit.action not in snapshot.allowed_actions:
        raise ValueError("Action is not allowed.")
    if isinstance(edit, NoAction):
        return edit
    by_id = {item.id: item for item in snapshot.items}
    if any(target not in by_id for target in edit.target):
        raise ValueError("Unknown target.")
    if set(edit.target) & set(snapshot.protected_ids):
        raise ValueError("Protected target.")
    selected = [by_id[target] for target in edit.target]
    if isinstance(edit, CorrectText):
        if (
            selected[0].type != "text"
            or not edit.text.strip()
            or edit.text == selected[0].text
        ):
            raise ValueError("Text correction requires a changed, nonempty text item.")
        return edit
    if any(
        not item.source_backed or item.boundary_status != "verified"
        for item in selected
    ):
        raise ValueError("Audio cuts require verified source boundaries.")
    if isinstance(edit, TrimSilence):
        item = selected[0]
        if item.type != "non_text" or item.audio_class != "verified_silence":
            raise ValueError("Unknown non-text and audio events are not silence.")
        if not edit.keep_ms < item.duration_ms <= 5000:
            raise ValueError(
                "Pause trim must shorten an interval of at most five seconds."
            )
    if isinstance(edit, DeleteSpan):
        positions = [
            next(i for i, item in enumerate(snapshot.items) if item.id == target)
            for target in edit.target
        ]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError("Deletion targets must be contiguous, ordered and unique.")
        if any(
            item.type != "text" or item.audio_class != "speech" for item in selected
        ):
            raise ValueError("Deletion accepts speech items only.")
        if (
            len(
                {
                    (item.paragraph, item.speaker, item.language, item.source_run)
                    for item in selected
                }
            )
            != 1
        ):
            raise ValueError(
                "Deletion cannot cross paragraph, speaker, language or source-run boundaries."
            )
        if any(item.speaker is None for item in selected):
            raise ValueError(
                "Uncertain speaker boundaries are not eligible for audio cuts."
            )
        if sum(item.duration_ms for item in selected) > 2000:
            raise ValueError("Deletion exceeds the initial two-second limit.")
    return edit
