"""Version 2 offline contract; eligibility is separate from semantic usefulness."""

import unicodedata
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from nep import Item, NoAction, StrictModel

VERSION = 2
Action = Literal[
    "no_action",
    "correct_text",
    "delete_span",
    "trim_silence",
    "remove_silence",
    "split_paragraph",
    "merge_paragraph",
]
ACTIONS = list(Action.__args__)


class DeleteSpan(StrictModel):
    action: Literal["delete_span"]
    target: list[int] = Field(min_length=1, max_length=256)


class CorrectText(StrictModel):
    action: Literal["correct_text"]
    target: list[int] = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=2000)


class TrimSilence(StrictModel):
    action: Literal["trim_silence"]
    target: list[int] = Field(min_length=1, max_length=1)
    keep_ms: int = Field(gt=0)


class RemoveSilence(StrictModel):
    action: Literal["remove_silence"]
    target: list[int] = Field(min_length=1, max_length=1)


class SplitParagraph(StrictModel):
    action: Literal["split_paragraph"]
    target: list[int] = Field(min_length=1, max_length=1)


class MergeParagraph(StrictModel):
    action: Literal["merge_paragraph"]
    target: list[int] = Field(min_length=1, max_length=1)


Edit = Annotated[
    NoAction
    | CorrectText
    | DeleteSpan
    | TrimSilence
    | RemoveSilence
    | SplitParagraph
    | MergeParagraph,
    Field(discriminator="action"),
]
EDIT = TypeAdapter(Edit)


class RecentEdit(StrictModel):
    action: Action
    origin: Literal["human", "ai_computer_use", "synthetic"]
    before: str | None = None
    after: str | None = None
    before_ms: int | None = Field(default=None, gt=0)
    after_ms: int | None = Field(default=None, ge=0)


class Snapshot(StrictModel):
    schema_version: Literal[2]
    task: Literal["predict_next_transcript_edit"]
    language: str
    mode: Literal["transcript_cleanup"]
    task_brief: str = Field(max_length=2000)
    cursor_before: int | None
    selection: list[int]
    recent_edits: list[RecentEdit] = Field(max_length=8)
    items: list[Item] = Field(min_length=1, max_length=256)
    allowed_actions: list[Action]
    protected_ids: list[int]
    complete_paragraphs: list[int]

    @model_validator(mode="after")
    def references(self):
        ids = {item.id for item in self.items}
        referenced = set(self.selection + self.protected_ids)
        if self.cursor_before is not None:
            referenced.add(self.cursor_before)
        if len(ids) != len(self.items) or not referenced <= ids:
            raise ValueError("Duplicate aliases or unknown snapshot reference.")
        if "no_action" not in self.allowed_actions:
            raise ValueError("Abstention must remain available.")
        paragraphs = [item.paragraph for item in self.items]
        runs = [p for i, p in enumerate(paragraphs) if i == 0 or p != paragraphs[i - 1]]
        if len(runs) != len(set(runs)):
            raise ValueError("Paragraphs must be contiguous.")
        if not set(self.complete_paragraphs) <= set(paragraphs):
            raise ValueError("Unknown complete paragraph.")
        return self


class Envelope(StrictModel):
    version: Literal[2] = 2
    document_id: str
    request_id: str
    revision: int = Field(ge=0)
    interaction_epoch: int = Field(ge=0)


class Policy(StrictModel):
    max_delete_ms: int = Field(default=2000, gt=0)
    max_pause_ms: int = Field(default=5000, gt=0)


def validate_prediction(
    text: str,
    snapshot: Snapshot,
    request: Envelope,
    current: Envelope,
    finish_reason: str,
    policy: Policy | None = None,
) -> Edit:
    policy = policy or Policy()
    if request != current:
        raise ValueError("Stale document or interaction context.")
    if finish_reason != "stop":
        raise ValueError("Incomplete model response.")
    edit = EDIT.validate_json(text, strict=True)
    if edit.action not in snapshot.allowed_actions:
        raise ValueError("Action is not allowed.")
    if isinstance(edit, NoAction):
        return edit
    positions = {item.id: i for i, item in enumerate(snapshot.items)}
    if any(t not in positions for t in edit.target):
        raise ValueError("Unknown target.")
    indices = [positions[t] for t in edit.target]
    first, last = indices[0], indices[-1]
    if indices != list(range(first, first + len(indices))):
        raise ValueError("Targets must be contiguous, ordered and unique.")
    if set(edit.target) & set(snapshot.protected_ids):
        raise ValueError("Protected target.")
    selected = snapshot.items[first : last + 1]
    if isinstance(edit, (SplitParagraph, MergeParagraph)):
        if first == 0:
            raise ValueError("Caption edits need the previous item in context.")
        previous, current_item = snapshot.items[first - 1], selected[0]
        same = previous.paragraph == current_item.paragraph
        if isinstance(edit, SplitParagraph) != same:
            raise ValueError("Caption boundary already exists or is absent.")
        affected_paragraphs = {previous.paragraph, current_item.paragraph}
        if not affected_paragraphs <= set(snapshot.complete_paragraphs):
            raise ValueError("Caption edits require complete affected paragraphs.")
        affected = [i for i in snapshot.items if i.paragraph in affected_paragraphs]
        if any(i.id in snapshot.protected_ids for i in affected):
            raise ValueError("Caption edit affects protected content.")
        if (
            len({(i.speaker, i.language) for i in affected}) != 1
            or not previous.speaker
            or not previous.language
            or previous.language == "und"
        ):
            raise ValueError(
                "Caption edit cannot cross unknown/different speakers or languages."
            )
        if isinstance(edit, SplitParagraph):
            if not all(
                any(i.type == "text" for i in part)
                for part in (
                    [
                        i
                        for i in snapshot.items[:first]
                        if i.paragraph == previous.paragraph
                    ],
                    [
                        i
                        for i in snapshot.items[first:]
                        if i.paragraph == previous.paragraph
                    ],
                )
            ):
                raise ValueError("A split needs text on both sides.")
        return edit
    if len({(i.paragraph, i.speaker, i.language, i.source_run) for i in selected}) != 1:
        raise ValueError(
            "Edit crosses paragraph, speaker, language or source-run boundaries."
        )
    if not all(i.source_backed for i in selected):
        raise ValueError("Edit requires source-backed targets.")
    if isinstance(edit, CorrectText):
        original = " ".join(i.text for i in selected if i.type == "text")
        if (
            not original
            or not edit.text.strip()
            or unicodedata.normalize("NFC", original)
            == unicodedata.normalize("NFC", edit.text)
        ):
            raise ValueError("Correction requires changed, nonempty display text.")
        return edit
    if any(i.boundary_status != "verified" for i in selected):
        raise ValueError("Audio cuts require verified source boundaries.")
    if isinstance(edit, (TrimSilence, RemoveSilence)):
        item = selected[0]
        if item.type != "non_text" or item.audio_class != "verified_silence":
            raise ValueError("Unknown gaps and audio events are not verified silence.")
        if item.duration_ms > policy.max_pause_ms:
            raise ValueError("Pause exceeds the configured duration policy.")
        if isinstance(edit, TrimSilence) and edit.keep_ms >= item.duration_ms:
            raise ValueError("A trim must shorten the interval.")
    else:
        if not selected[0].speaker:
            raise ValueError("Audio cuts require a known speaker.")
        if not any(i.type == "text" for i in selected) or any(
            (i.type == "text" and i.audio_class != "speech")
            or (i.type == "non_text" and i.audio_class != "verified_silence")
            for i in selected
        ):
            raise ValueError(
                "Deletion accepts speech and verified intervening silence only."
            )
        if sum(i.duration_ms for i in selected) > policy.max_delete_ms:
            raise ValueError("Deletion exceeds the configured duration policy.")
    return edit
