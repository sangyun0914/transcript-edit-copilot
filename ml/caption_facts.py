"""Document-derived caption structure; metrics follow the editor's VTT splitter."""

from typing import Literal
from math import floor

from pydantic import Field, model_validator

from nep import StrictModel


class CaptionProfile(StrictModel):
    version: Literal[1] = 1
    export_line_length: int | None = Field(default=None, gt=0)
    min_cue_ms: int = Field(default=1000, gt=0)
    max_merge_cue_ms: int = Field(default=7000, gt=0)
    max_merge_text_utf16: int = Field(default=84, gt=0)


class CueMetrics(StrictModel):
    count: int = Field(ge=0)
    min_ms: int = Field(ge=0)
    max_ms: int = Field(ge=0)
    max_text_utf16: int = Field(ge=0)


class SplitMetrics(StrictModel):
    target: int = Field(ge=0)
    after: CueMetrics


class ParagraphFacts(StrictModel):
    paragraph: int = Field(ge=0)
    first_id: int = Field(ge=0)
    last_id: int = Field(ge=0)
    speaker: str | None
    language: str
    text_items: int = Field(ge=0)
    text_utf16: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    protected: bool
    prefix_text_items: int = Field(ge=0)
    prefix_text_utf16: int = Field(ge=0)
    suffix_text_items: int = Field(ge=0)
    suffix_text_utf16: int = Field(ge=0)
    visible_start_ms: list[int]
    exported: CueMetrics
    split_exports: list[SplitMetrics]

    @model_validator(mode="after")
    def consistent_ranges(self):
        if self.last_id < self.first_id:
            raise ValueError("Paragraph alias range is reversed.")
        if self.visible_start_ms != sorted(self.visible_start_ms) or any(
            n < 0 or n > self.duration_ms for n in self.visible_start_ms
        ):
            raise ValueError("Caption offsets must be ordered inside the paragraph.")
        return self


def utf16_units(text):
    return len(text.encode("utf-16-le")) // 2


def text_metrics(items):
    words = [i.get("text", "") for i in items if i["type"] == "text"]
    return len(words), utf16_units(" ".join(words))


def vtt_ms(seconds):
    # The actual writer floors the fractional part, rather than rounding seconds * 1000.
    return floor(seconds) * 1000 + floor((seconds % 1) * 1000)


def export_metrics(paragraphs, line_length=None):
    """Use real item lengths, not invented word timing; match contentToVtt's split rule."""
    cues, elapsed = [], 0.0
    for paragraph in paragraphs:
        current, characters = [], 0
        for item in paragraph:
            start_ms = vtt_ms(elapsed)
            elapsed += item["length"]
            item = {
                **item,
                "_caption_start_ms": item.get("_caption_start_ms", start_ms),
                "_caption_end_ms": item.get("_caption_end_ms", vtt_ms(elapsed)),
            }
            if item["type"] == "text":
                count = utf16_units(item["text"])
                if line_length is not None and characters + count > line_length:
                    if current:
                        cues.append(current)
                    current, characters = [], 0
                characters += count
            current.append(item)
        if current:
            cues.append(current)
    durations = [
        cue[-1]["_caption_end_ms"] - cue[0]["_caption_start_ms"] for cue in cues
    ]
    return CueMetrics(
        count=len(cues),
        min_ms=min(durations, default=0),
        max_ms=max(durations, default=0),
        max_text_utf16=max((text_metrics(c)[1] for c in cues), default=0),
    ).model_dump()


def caption_facts(document, records, items, protected_ids, profile):
    profile = CaptionProfile.model_validate(profile)
    timed, elapsed = [], 0.0
    for item in document["content"]:
        if item["type"] in ("text", "non_text", "artificial_silence"):
            start_ms = vtt_ms(elapsed)
            elapsed += item["length"]
            timed.append(
                {
                    **item,
                    "_caption_start_ms": start_ms,
                    "_caption_end_ms": vtt_ms(elapsed),
                }
            )
    if len(timed) != len(records):
        raise ValueError("Caption records differ from the timed document.")
    visible = {i["id"] for i in items}
    result = []
    for paragraph in dict.fromkeys(i["paragraph"] for i in items):
        members = [r for r in records if r["paragraph"] == paragraph]
        local = [r for r in members if r["id"] in visible]
        first, last = members[0]["id"], members[-1]["id"]
        full = timed[first : last + 1]
        start, end = local[0]["id"] - first, local[-1]["id"] - first + 1
        prefix, suffix = text_metrics(full[:start]), text_metrics(full[end:])
        total_text = text_metrics(full)
        paragraph_start = full[0]["_caption_start_ms"]
        offsets = [i["_caption_start_ms"] - paragraph_start for i in full[start:end]]
        splits = []
        if profile.export_line_length is not None:
            for record in local:
                position = record["id"] - first
                if position:
                    splits.append(
                        {
                            "target": record["id"],
                            "after": export_metrics(
                                [full[:position], full[position:]],
                                profile.export_line_length,
                            ),
                        }
                    )
        result.append(
            ParagraphFacts(
                paragraph=paragraph,
                first_id=first,
                last_id=last,
                speaker=members[0]["speaker"],
                language=members[0]["language"],
                text_items=total_text[0],
                text_utf16=total_text[1],
                duration_ms=full[-1]["_caption_end_ms"] - paragraph_start,
                protected=bool(set(protected_ids) & {r["id"] for r in members}),
                prefix_text_items=prefix[0],
                prefix_text_utf16=prefix[1],
                suffix_text_items=suffix[0],
                suffix_text_utf16=suffix[1],
                visible_start_ms=offsets,
                exported=export_metrics([full], profile.export_line_length),
                split_exports=splits,
            ).model_dump()
        )
    return result


def verify_caption_facts(document, bundle):
    if bundle["input"].get("schema_version") != 4:
        return
    from session_cases import document_records

    authority = bundle.get("caption_authority")
    if not authority or authority.get("version") != 1:
        raise ValueError("V4 execution requires document-derived caption authority.")
    records, mapping = document_records(document)
    by_uuid = {v["uuid"]: int(k) for k, v in mapping.items()}
    if not set(authority["protected_uuids"]) <= by_uuid.keys():
        raise ValueError("Caption protection authority references a missing item.")
    protected = [by_uuid[u] for u in authority["protected_uuids"]]
    value = bundle["input"]
    if value["caption_profile"] != authority["profile"]:
        raise ValueError("Caption profile differs from runtime authority.")
    actual = caption_facts(
        document, records, value["items"], protected, authority["profile"]
    )
    if actual != value["caption_facts"]:
        raise ValueError("Caption facts differ from the full current document.")
    visible = {i["id"] for i in value["items"]}
    if set(value["protected_ids"]) != set(protected) & visible:
        raise ValueError("Visible protections differ from caption authority.")
