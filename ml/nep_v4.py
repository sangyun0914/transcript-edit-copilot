"""V4 adds trusted full-paragraph caption facts without claiming cropped text is visible."""

from typing import Literal

from pydantic import Field, model_validator

import nep_v2
import nep_v3
from caption_facts import CaptionProfile, ParagraphFacts, export_metrics
from nep import NoAction
from nep_v2 import ACTIONS, EDIT, Envelope, Policy  # noqa: F401

VERSION = 4


class Snapshot(nep_v3.Snapshot):
    schema_version: Literal[4]
    caption_profile: CaptionProfile
    caption_facts: list[ParagraphFacts] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def caption_references(self):
        if [f.paragraph for f in self.caption_facts] != list(
            dict.fromkeys(i.paragraph for i in self.items)
        ):
            raise ValueError(
                "Each visible paragraph needs exactly one trusted summary."
            )
        for fact in self.caption_facts:
            local = [i for i in self.items if i.paragraph == fact.paragraph]
            ids = [i.id for i in local]
            if (
                ids != list(range(ids[0], ids[-1] + 1))
                or not fact.first_id <= ids[0] <= ids[-1] <= fact.last_id
            ):
                raise ValueError(
                    "Visible items must be a contiguous part of the full paragraph."
                )
            complete = ids[0] == fact.first_id and ids[-1] == fact.last_id
            if any(
                (i.speaker, i.language) != (fact.speaker, fact.language) for i in local
            ):
                raise ValueError(
                    "Visible speaker/language contradict full-paragraph facts."
                )
            if complete != (fact.paragraph in self.complete_paragraphs):
                raise ValueError(
                    "Complete-paragraph flag contradicts the full-document range."
                )
            if len(fact.visible_start_ms) != len(local):
                raise ValueError(
                    "Caption offsets must cover exactly the visible items."
                )
            visible_text = sum(i.type == "text" for i in local)
            if (
                fact.prefix_text_items + visible_text + fact.suffix_text_items
                != fact.text_items
            ):
                raise ValueError(
                    "Caption text counts do not match visible and omitted content."
                )
            if complete and (fact.prefix_text_items or fact.suffix_text_items):
                raise ValueError("Complete paragraphs cannot have omitted text.")
            targets = [s.target for s in fact.split_exports]
            expected = (
                [i for i in ids if i != fact.first_id]
                if self.caption_profile.export_line_length is not None
                else []
            )
            if targets != expected:
                raise ValueError(
                    "Export metrics must cover the visible split boundaries."
                )
        return self


def legacy_snapshot(value):
    parsed = Snapshot.model_validate(value).model_dump()
    for key in ("caption_profile", "caption_facts", "recent_feedback"):
        parsed.pop(key)
    parsed["schema_version"] = 2
    return nep_v2.Snapshot.model_validate(parsed)


def validate_prediction(text, snapshot, request, current, finish_reason, policy=None):
    if request != current:
        raise ValueError("Stale document or interaction context.")
    if finish_reason != "stop":
        raise ValueError("Incomplete model response.")
    edit = EDIT.validate_json(text, strict=True)
    if edit.action not in snapshot.allowed_actions:
        raise ValueError("Action is not allowed.")
    if isinstance(edit, NoAction) or edit.action not in (
        "split_paragraph",
        "merge_paragraph",
    ):
        return nep_v2.validate_prediction(
            text,
            legacy_snapshot(snapshot.model_dump()),
            request,
            current,
            finish_reason,
            policy,
        )
    positions = {i.id: n for n, i in enumerate(snapshot.items)}
    target = edit.target[0]
    if target not in positions or positions[target] == 0:
        raise ValueError("Caption edits need their target and previous item visible.")
    position = positions[target]
    previous, current_item = snapshot.items[position - 1 : position + 1]
    splitting = edit.action == "split_paragraph"
    if splitting != (previous.paragraph == current_item.paragraph):
        raise ValueError("Caption boundary already exists or is absent.")
    affected = {previous.paragraph, current_item.paragraph}
    facts = [f for f in snapshot.caption_facts if f.paragraph in affected]
    if any(f.protected for f in facts):
        raise ValueError(
            "Caption edit affects protected content, including offscreen content."
        )
    if (
        (previous.speaker, previous.language)
        != (current_item.speaker, current_item.language)
        or not previous.speaker
        or not previous.language
        or previous.language == "und"
    ):
        raise ValueError(
            "Caption edit cannot cross unknown/different speakers or languages."
        )
    profile = snapshot.caption_profile
    if splitting:
        # Adjacent text supplies the local grouping evidence; summaries never supply hidden wording.
        local_before = snapshot.items[max(0, position - 3) : position]
        if current_item.type != "text" or not any(
            i.type == "text" and i.paragraph == current_item.paragraph
            for i in local_before
        ):
            raise ValueError("A caption split needs nearby visible text on both sides.")
        fact = facts[0]
        local = [i for i in snapshot.items if i.paragraph == fact.paragraph]
        split_position = next(n for n, i in enumerate(local) if i.id == target)
        left_text = fact.prefix_text_items + sum(
            i.type == "text" for i in local[:split_position]
        )
        if not 0 < left_text < fact.text_items:
            raise ValueError("A split needs text on both sides in the full paragraph.")
        if profile.export_line_length is None:
            left_ms = fact.visible_start_ms[split_position]
            minimum = min(left_ms, fact.duration_ms - left_ms)
        else:
            minimum = next(
                s.after.min_ms for s in fact.split_exports if s.target == target
            )
        if minimum < profile.min_cue_ms:
            raise ValueError(
                "Caption split creates an exported cue below the profile minimum."
            )
    else:
        if not affected <= set(snapshot.complete_paragraphs):
            raise ValueError(
                "Insufficient visible context for merging complete paragraph meanings."
            )
        merged, elapsed = [], 0
        for fact in facts:
            local = [i for i in snapshot.items if i.paragraph == fact.paragraph]
            ends = [*fact.visible_start_ms[1:], fact.duration_ms]
            for item, start, end in zip(
                local, fact.visible_start_ms, ends, strict=True
            ):
                merged.append(
                    {
                        "type": item.type,
                        "text": item.text,
                        "length": item.duration_ms / 1000,
                        "_caption_start_ms": elapsed + start,
                        "_caption_end_ms": elapsed + end,
                    }
                )
            elapsed += fact.duration_ms
        result = export_metrics([merged], profile.export_line_length)
        if (
            result["min_ms"] < profile.min_cue_ms
            or result["max_ms"] > profile.max_merge_cue_ms
            or result["max_text_utf16"] > profile.max_merge_text_utf16
        ):
            raise ValueError("Merged exported cues exceed the caption profile.")
    return edit
