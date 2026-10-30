"""Experimental, preview-only source-timed proposals for all seven passive actions.

This separately versioned contract validates editor mapping and reversible proposals;
it does not certify acoustic boundaries or classify an unknown gap as silence.
Existing passive v1 sources, prompts and scientific evidence remain unchanged.
"""

import copy
import hashlib
import json
import marshal
import math
import unicodedata
from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, PrivateAttr, model_validator

import model_input
import nep_v4
from caption_facts import (
    CaptionProfile,
    ParagraphFacts,
    caption_facts,
    text_metrics,
    verify_caption_facts,
)
from context_policy import (
    ContextBudgetError,
    FeedbackLedger,
    evidence_fingerprint,
)
from nep import StrictModel
from nep_v2 import (
    ACTIONS as ACTIONS,
    EDIT as EDIT,
    Action,
    Envelope as Envelope,
    RecentEdit,
)
from nep_v3 import Feedback
from replay_session import session_hash as _session_hash
from session_cases import document_records

VERSION = "passive-nep-actions-v1"
PROMPT_PATH = Path(__file__).with_name("prompt-passive-actions-v1.txt")
PROMPT = PROMPT_PATH.read_text().strip()
ALLOWED_ACTIONS = tuple(ACTIONS)
GATED_ACTIONS = ()
CUT_ACTIONS = ("delete_span", "trim_silence", "remove_silence")
OUTPUT_SCHEMA = EDIT.json_schema()
_INTERNAL_BRIEF = "Fixed passive-nep-actions-v1 source-timed preview policy."
ITEM_FIELDS = (
    *model_input.ITEM_FIELDS,
    "source_id",
    "source_start_ms",
    "source_end_ms",
)
_WIRE_FIELDS = {
    "item_fields",
    "caption_fact_fields",
    "caption_metric_fields",
    "complete_caption_fields",
    "complete_caption_facts",
}

_validation_context = ContextVar("passive_validation_context", default=None)
_CACHE_LIMITS = {
    "records": 8,
    "source": 2,
    "hash": 32,
    "bundle": 4,
    "transition": 128,
    "summary": 128,
    "messages": 128,
}


def _cache_key(value):
    try:
        # This private, in-memory key preserves bool/int and list/tuple types.
        # Version 2 ignores shared-object identities; it is never read or persisted
        # and does not replace the public typed-json document/prompt hashes.
        return hashlib.sha256(marshal.dumps(value, 2)).digest()
    except (ValueError, TypeError):
        return None


class _ValidationCache:
    def __init__(self):
        self.values = {name: OrderedDict() for name in _CACHE_LIMITS}
        self.lock = RLock()

    def get(self, kind, key):
        with self.lock:
            values = self.values[kind]
            if key is not None and key in values:
                values.move_to_end(key)
                return True, values[key]
        return False, None

    def put(self, kind, key, value):
        if key is None:
            return
        with self.lock:
            values = self.values[kind]
            values[key] = value
            values.move_to_end(key)
            while len(values) > _CACHE_LIMITS[kind]:
                values.popitem(last=False)


def _checkpoint():
    context = _validation_context.get()
    if context is not None and context[1] is not None:
        context[1]()


def _cached(kind, *, copy_on_hit=False):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            context = _validation_context.get()
            if context is None:
                return function(*args, **kwargs)
            _checkpoint()
            key = _cache_key([list(args), kwargs])
            found, value = context[0].get(kind, key)
            if found:
                return copy.deepcopy(value) if copy_on_hit else value
            value = function(*args, **kwargs)
            _checkpoint()
            context[0].put(kind, key, copy.deepcopy(value))
            return value

        return wrapped

    return decorate


@_cached("hash")
def session_hash(value):
    return _session_hash(value)


def _session_cached(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with self.validation_scope():
            return function(self, *args, **kwargs)

    return wrapped


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
    boundary_status: Literal["verified", "unverified", "needs_alignment", "overlap"]
    audio_class: Literal["speech", "verified_silence", "unknown", "audio_event"]
    source_id: str | None
    source_start_ms: float | None = Field(ge=0, allow_inf_nan=False)
    source_end_ms: float | None = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def actual_interval(self):
        if self.source_backed:
            if (
                not self.source_id
                or self.source_start_ms is None
                or self.source_end_ms is None
                or self.source_end_ms <= self.source_start_ms
            ):
                raise ValueError(
                    "Source-backed items need their positive original interval."
                )
            if abs((self.source_end_ms - self.source_start_ms) - self.duration_ms) > 1:
                raise ValueError(
                    "Display duration contradicts the original source interval."
                )
        elif any(
            v is not None
            for v in (self.source_id, self.source_start_ms, self.source_end_ms)
        ):
            raise ValueError(
                "An unbacked item cannot claim an original source interval."
            )
        return self


def _v4_data(value):
    result = copy.deepcopy(value)
    result.pop("contract_version")
    result.update(schema_version=4, task_brief=_INTERNAL_BRIEF)
    result["items"] = [
        {key: item[key] for key in model_input.ITEM_FIELDS} for item in result["items"]
    ]
    return result


class PassiveSnapshot(StrictModel):
    contract_version: Literal["passive-nep-actions-v1"]
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
                "The all-action proposal policy has one fixed set of seven actions."
            )
        if len(self.selection) != len(set(self.selection)):
            raise ValueError("Selection must not contain duplicate aliases.")
        if any(f.protected for f in self.caption_facts):
            raise ValueError(
                "This passive policy has no caller-authored protection overrides."
            )
        aliases = [item.id for item in self.items]
        if aliases != list(range(aliases[0], aliases[0] + len(aliases))):
            raise ValueError(
                "Visible aliases must be a contiguous ordered document window."
            )
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
    result["items"] = items
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


@_cached("messages", copy_on_hit=True)
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


def token_count(tokenizer, value):
    count = getattr(tokenizer, "count_chat_template_tokens", None)
    if count is None:
        return len(token_ids(tokenizer, value))
    result = count(
        canonical_messages(value),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if type(result) is not int or result < 0:
        raise ValueError("Native tokenizer returned an invalid token count.")
    return result


def validate_prediction(text, snapshot, request, current, finish_reason):
    snapshot = parse_snapshot(snapshot)
    if request != current:
        raise ValueError("Stale document or interaction context.")
    if finish_reason != "stop":
        raise ValueError("Incomplete model response.")
    edit = EDIT.validate_json(text, strict=True)
    if edit.action == "no_action":
        return edit
    if edit.action in {"split_paragraph", "merge_paragraph"}:
        return nep_v4.validate_prediction(
            text,
            nep_v4.Snapshot.model_validate(_v4_data(snapshot.model_dump())),
            request,
            current,
            finish_reason,
        )
    positions = {item.id: n for n, item in enumerate(snapshot.items)}
    if any(alias not in positions for alias in edit.target):
        raise ValueError("Unknown target.")
    indices = [positions[alias] for alias in edit.target]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Targets must be contiguous, ordered and unique.")
    selected = [snapshot.items[n] for n in indices]
    if (
        len(
            {
                (i.paragraph, i.speaker, i.language, i.source_run, i.source_id)
                for i in selected
            }
        )
        != 1
    ):
        raise ValueError(
            "Edit crosses paragraph, speaker, language or source-run boundaries."
        )
    if not all(i.source_backed for i in selected):
        raise ValueError("Edit requires source-backed targets.")
    if any(
        abs(a.source_end_ms - b.source_start_ms) > 0.001
        for a, b in zip(selected, selected[1:])
    ):
        raise ValueError("Targets cross a source discontinuity.")
    if any(i.audio_class == "audio_event" for i in selected):
        raise ValueError("A proposal cannot discard or relabel a known audio event.")
    if edit.action == "correct_text":
        original = " ".join(i.text for i in selected if i.type == "text")
        if (
            not original
            or not edit.text.strip()
            or unicodedata.normalize("NFC", original)
            == unicodedata.normalize("NFC", edit.text)
        ):
            raise ValueError("Correction requires changed, nonempty display text.")
        return edit
    if any(i.boundary_status in {"needs_alignment", "overlap"} for i in selected):
        raise ValueError(
            "Known conflicting source timing requires alignment before a cut proposal."
        )
    if not selected[0].speaker or selected[0].language in {"", "und"}:
        raise ValueError("A cut proposal needs a known speaker and language.")
    if edit.action in {"trim_silence", "remove_silence"}:
        item = selected[0]
        if item.type != "non_text" or item.audio_class not in {
            "unknown",
            "verified_silence",
        }:
            raise ValueError(
                "Pause operations require a gap, not known speech or an audio event."
            )
        if edit.action == "trim_silence" and edit.keep_ms >= (
            item.source_end_ms - item.source_start_ms
        ):
            raise ValueError(
                "A trim must retain a positive, strictly shorter interval."
            )
    elif not any(i.type == "text" for i in selected) or any(
        i.type == "non_text" and i.audio_class == "speech" for i in selected
    ):
        raise ValueError(
            "Word deletion needs visible text and cannot hide known gap speech."
        )
    return edit


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _overlaps(start, end, region_start, region_end):
    return (
        start <= region_start <= end
        if region_start == region_end
        else start < region_end and region_start < end
    )


def _source_metadata(document):
    metadata = document["metadata"]
    provenance = metadata.get("transcriptions", [])
    if not isinstance(provenance, list):
        raise ValueError("Malformed transcription provenance.")
    records = list(provenance)
    legacy = metadata.get("transcription")
    if legacy is not None:
        if not isinstance(legacy, dict) or not isinstance(
            legacy.get("items", {}), dict
        ):
            raise ValueError("Malformed legacy transcription provenance.")
        if legacy.get("items") and (
            not isinstance(legacy.get("source"), str)
            or not legacy["source"]
            or any(not isinstance(value, dict) for value in legacy["items"].values())
        ):
            raise ValueError(
                "Legacy temporal evidence needs its original source identity."
            )
        records.append(
            {
                **legacy,
                "regions": [
                    {
                        **legacy["items"][item["uuid"]],
                        "sourceStart": item["sourceStart"],
                        "length": item["length"],
                    }
                    for item in document["content"]
                    if item.get("source") == legacy.get("source")
                    and item["uuid"] in legacy.get("items", {})
                ],
            }
        )
    regions, annotations = [], []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Malformed transcription provenance.")
        for field in ("regions", "annotations"):
            if not isinstance(record.get(field, []), list):
                raise ValueError("Malformed transcription regions or annotations.")
        if not record.get("regions") and not record.get("annotations"):
            continue
        if not isinstance(record.get("source"), str) or not record["source"]:
            raise ValueError("Temporal evidence needs its original source identity.")
        for region in record.get("regions", []):
            if (
                not isinstance(region, dict)
                or not _number(region.get("sourceStart"))
                or not _number(region.get("length"))
                or region["sourceStart"] < 0
                or region["length"] <= 0
                or region.get("boundary_status")
                not in {"unverified", "needs_alignment", "overlap"}
            ):
                raise ValueError("Malformed source boundary evidence.")
            regions.append({**region, "source": record["source"]})
        for note in record.get("annotations", []):
            if (
                not isinstance(note, dict)
                or note.get("type") not in {"audio_event", "unaligned_word"}
                or not _number(note.get("start"))
                or not _number(note.get("end"))
                or note["start"] < 0
                or note["end"] < note["start"]
            ):
                raise ValueError("Malformed source audio annotation.")
            annotations.append({**note, "source": record["source"]})
    return regions, annotations


class _IntervalIndex:
    def __init__(self, intervals):
        self.points = []
        self.starts = []
        self.max_ends = []
        for start, end in sorted(intervals):
            if start == end:
                self.points.append(start)
            else:
                self.starts.append(start)
                self.max_ends.append(
                    max(end, self.max_ends[-1] if self.max_ends else end)
                )

    def overlaps(self, start, end):
        position = bisect_left(self.starts, end)
        # A preceding long region can overlap even when later, shorter ones do not.
        if position and self.max_ends[position - 1] > start:
            return True
        position = bisect_left(self.points, start)
        return position < len(self.points) and self.points[position] <= end


def _source_metadata_index(document):
    legacy = document["metadata"].get("transcription")
    if isinstance(legacy, dict) and legacy.get("items"):
        return _build_metadata_index(document)
    return _metadata_index(document["metadata"])


@_cached("source")
def _metadata_index(metadata):
    return _build_metadata_index({"metadata": metadata, "content": []})


def _build_metadata_index(document):
    regions, annotations = _source_metadata(document)
    intervals = defaultdict(lambda: defaultdict(list))
    for region in regions:
        intervals[region["source"]][region["boundary_status"]].append(
            (region["sourceStart"], region["sourceStart"] + region["length"])
        )
    for note in annotations:
        intervals[note["source"]][note["type"]].append((note["start"], note["end"]))
    return {
        source: {kind: _IntervalIndex(values) for kind, values in kinds.items()}
        for source, kinds in intervals.items()
    }


@_cached("records")
def _validated_records(document):
    if not isinstance(document, dict) or set(document) != {"content", "metadata"}:
        raise ValueError("Use an editor state with content and metadata only.")
    if not isinstance(document["metadata"], dict) or not isinstance(
        document["content"], list
    ):
        raise ValueError("Malformed editor state.")
    inside = False
    for position, item in enumerate(document["content"]):
        if position % 256 == 0:
            _checkpoint()
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("uuid"), str)
            or not item["uuid"]
        ):
            raise ValueError("Document items need their actual identities.")
        kind = item.get("type")
        if kind == "paragraph_start":
            if inside:
                raise ValueError("Nested paragraph start.")
            inside = True
        elif kind == "paragraph_end":
            if not inside:
                raise ValueError("Unmatched paragraph end.")
            inside = False
        elif kind in {"text", "non_text", "artificial_silence"}:
            if (
                not inside
                or not _number(item.get("length"))
                or item["length"] <= 0
                or not math.isfinite(item["length"] * 1000)
            ):
                raise ValueError(
                    "Timed items need a positive finite duration inside a paragraph."
                )
            if kind == "text" and not isinstance(item.get("text"), str):
                raise ValueError("Display text must be a string.")
            if "source" in item and (
                not isinstance(item["source"], str)
                or not item["source"]
                or not _number(item.get("sourceStart"))
                or item["sourceStart"] < 0
                or not math.isfinite((item["sourceStart"] + item["length"]) * 1000)
            ):
                raise ValueError("Invalid original source interval.")
        else:
            raise ValueError("Unknown editor item type.")
    if inside:
        raise ValueError("Unclosed paragraph.")
    records, mapping = document_records(document)
    source_metadata = _source_metadata_index(document)
    for record in records:
        if record["id"] % 256 == 0:
            _checkpoint()
        item = document["content"][mapping[str(record["id"])]["index"]]
        backed = record["source_backed"]
        record.update(
            source_id=item.get("source"),
            source_start_ms=item["sourceStart"] * 1000 if backed else None,
            source_end_ms=(item["sourceStart"] + item["length"]) * 1000
            if backed
            else None,
        )
        asr = item.get("asr", {})
        if not isinstance(asr, dict):
            raise ValueError("Malformed item ASR metadata.")
        statuses = [
            value["boundary_status"]
            for value in (item, asr)
            if "boundary_status" in value
        ] or ["unverified"]
        classes = [
            value["audio_class"] for value in (item, asr) if "audio_class" in value
        ] or [record["audio_class"]]
        if any(
            value not in {"unverified", "needs_alignment", "overlap"}
            for value in statuses
        ) or any(
            value not in {"speech", "unknown", "audio_event"} for value in classes
        ):
            raise ValueError(
                "This proposal contract has no authority to certify acoustic evidence."
            )
        audio = (
            "audio_event"
            if "audio_event" in classes
            else "speech"
            if "speech" in classes
            else "unknown"
        )
        if backed:
            start, end = item["sourceStart"], item["sourceStart"] + item["length"]
            evidence = {
                kind
                for kind, intervals in source_metadata.get(item["source"], {}).items()
                if intervals.overlaps(start, end)
            }
            statuses.extend(evidence & {"unverified", "needs_alignment", "overlap"})
            if "audio_event" in evidence:
                audio = "audio_event"
            if "unaligned_word" in evidence:
                statuses.append("needs_alignment")
        record["boundary_status"] = (
            "needs_alignment"
            if "needs_alignment" in statuses
            else "overlap"
            if "overlap" in statuses
            else "unverified"
        )
        record["audio_class"] = audio
        PassiveItem.model_validate(record)
    return records, mapping


def _records(document, *, allow_empty=False):
    records, mapping = _validated_records(document)
    if not records and not allow_empty:
        raise ValueError("A passive window requires a current timed item.")
    return records, mapping


@_cached("bundle")
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
    """Build a deterministic preview document; no media or editor state is mutated."""
    _legacy_bundle(document, bundle)
    request = Envelope(
        document_id=bundle["document_hash"],
        request_id="preview",
        revision=0,
        interaction_epoch=0,
    )
    edit = validate_prediction(
        json.dumps(prediction), bundle["input"], request, request, "stop"
    )
    result = copy.deepcopy(document)
    if edit.action == "no_action":
        return result
    content = result["content"]
    indices = [bundle["mapping"][str(t)]["index"] for t in edit.target]
    first, last = indices[0], indices[-1]
    selected = content[first : last + 1]
    if edit.action not in ("split_paragraph", "merge_paragraph"):
        if indices != list(range(first, last + 1)):
            raise ValueError("Targets cross paragraph markers or hidden items.")
        if any(i.get("source") != selected[0].get("source") for i in selected):
            raise ValueError("Targets cross original sources.")
        if any(
            abs(a["sourceStart"] + a["length"] - b["sourceStart"]) > 1e-6
            for a, b in zip(selected, selected[1:])
        ):
            raise ValueError("Targets cross a source discontinuity.")
    key = session_hash({"before": bundle["document_hash"], "edit": prediction})

    def new_id(suffix):
        return str(uuid5(NAMESPACE_URL, key + suffix))

    if edit.action in ("delete_span", "remove_silence"):
        del content[first : last + 1]
    elif edit.action == "correct_text":
        content[first : last + 1] = [
            {
                **selected[0],
                "type": "text",
                "text": edit.text,
                "conf": 1,
                "length": sum(i["length"] for i in selected),
            }
        ]
    elif edit.action == "trim_silence":
        gap = selected[0]
        half = edit.keep_ms / 2000
        if 2 * half >= gap["length"]:
            raise ValueError(
                "Rounded model duration cannot lengthen a source interval."
            )
        content[first : first + 1] = [
            {**gap, "length": half},
            {
                **gap,
                "uuid": new_id("right"),
                "sourceStart": gap["sourceStart"] + gap["length"] - half,
                "length": half,
            },
        ]
    elif edit.action == "split_paragraph":
        paragraph = next(
            i for i in reversed(content[:first]) if i["type"] == "paragraph_start"
        )
        content[first:first] = [
            {"type": "paragraph_end", "uuid": new_id("end")},
            {**paragraph, "uuid": new_id("start")},
        ]
    else:
        if [i["type"] for i in content[first - 2 : first]] != [
            "paragraph_end",
            "paragraph_start",
        ]:
            raise ValueError("Merge target must be the first item after a boundary.")
        del content[first - 2 : first]
    return result


def _correction_effect(document, bundle, prediction):
    from grading import transcript_media

    target = bundle["mapping"][str(prediction["target"][0])]["index"]
    paragraph_id = next(
        item["uuid"]
        for item in reversed(document["content"][:target])
        if item["type"] == "paragraph_start"
    )
    after = apply_passive_edit(document, bundle, prediction)
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


class CommittedTransition(StrictModel):
    sequence: int = Field(gt=0)
    origin: Literal["human", "ai_computer_use", "synthetic"]
    transaction_id: str = Field(min_length=1)
    before: dict
    after: dict
    prediction: dict
    undo_of: str | None = None
    _state_hashes: tuple[str, str] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def actual_state_delta(self):
        from verify_release_export import verify_operation

        context = _validation_context.get()
        key = (
            _cache_key(
                {field: getattr(self, field) for field in type(self).model_fields}
            )
            if context is not None
            else None
        )
        if context is not None:
            found, hashes = context[0].get("transition", key)
            if found:
                _checkpoint()
                self._state_hashes = hashes
                return self
        before, after = (
            (self.after, self.before) if self.undo_of else (self.before, self.after)
        )
        _, mapping = _records(before)
        _records(after, allow_empty=True)
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
            and edit.keep_ms / 1000
            >= before["content"][mapping[str(aliases[0])]["index"]]["length"]
        ):
            raise ValueError("Committed trim must shorten its actual interval.")
        verify_operation(before, after, mapping, edit.model_dump())
        _checkpoint()
        if context is not None:
            self._state_hashes = (session_hash(self.before), session_hash(self.after))
            context[0].put("transition", key, self._state_hashes)
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


@_cached("summary")
def _history_summary(before, edit, origin):
    records, _ = _records(before)
    selected = [records[a] for a in edit["target"]]
    fields = {
        "action": edit["action"],
        "origin": origin,
        "before": " ".join(i["text"] for i in selected if i["type"] == "text"),
        "after": edit.get("text"),
    }
    if edit["action"] in CUT_ACTIONS:
        fields.update(
            before_ms=sum(i["duration_ms"] for i in selected),
            after_ms=edit.get("keep_ms", 0),
        )
    return RecentEdit.model_validate(fields).model_dump()


def _history(document, history, as_of):
    if type(as_of) is not int or as_of < 0:
        raise ValueError("Prediction sequence must be a nonnegative integer.")
    if isinstance(history, _SessionHistory):
        return history.context(document, as_of)
    transitions = [CommittedTransition.model_validate(value) for value in history]
    sequences = [t.sequence for t in transitions]
    if sequences != sorted(set(sequences)) or len(
        {t.transaction_id for t in transitions}
    ) != len(transitions):
        raise ValueError("Committed history must have unique chronological identities.")
    active, by_id, past = [], {}, []
    for transition in transitions:
        _checkpoint()
        if transition.sequence > as_of:
            continue
        if past and (
            past[-1]._state_hashes[1]
            if past[-1]._state_hashes is not None
            else session_hash(past[-1].after)
        ) != (
            transition._state_hashes[0]
            if transition._state_hashes is not None
            else session_hash(transition.before)
        ):
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
    if past and (
        past[-1]._state_hashes[1]
        if past[-1]._state_hashes is not None
        else session_hash(past[-1].after)
    ) != session_hash(document):
        raise ValueError(
            "The final committed history state is not the current document."
        )
    summaries = [
        _history_summary(transition.before, transition.prediction, transition.origin)
        for transition in active
    ]
    return past, summaries


class HistoryResyncRequired(ValueError):
    pass


class HistoryLimitError(ValueError):
    pass


def _history_key(value):
    normalized = {"undo_of": None, **value}
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    if json.loads(encoded) != normalized:
        raise ValueError(
            "Incremental history requires JSON objects, arrays and scalar values."
        )
    return hashlib.sha256(encoded.encode()).digest()


def _own_transition(transition, previous):
    before = transition.before
    if previous is not None and _cache_key(before) == _cache_key(previous.after):
        before = previous.after
    else:
        before = copy.deepcopy(before)
    old_items = {item["uuid"]: (item, _cache_key(item)) for item in before["content"]}
    content = []
    for position, item in enumerate(transition.after["content"]):
        if position % 256 == 0:
            _checkpoint()
        old = old_items.get(item["uuid"])
        content.append(
            old[0] if old and old[1] == _cache_key(item) else copy.deepcopy(item)
        )
    metadata = transition.after["metadata"]
    metadata = (
        before["metadata"]
        if _cache_key(metadata) == _cache_key(before["metadata"])
        else copy.deepcopy(metadata)
    )
    return transition.model_copy(
        update={
            "before": before,
            "after": {"content": content, "metadata": metadata},
            "prediction": copy.deepcopy(transition.prediction),
        }
    )


@dataclass(frozen=True)
class _HistoryEntry:
    key: bytes
    transition: "CommittedTransition | _CommittedProof"
    summary: dict | None


class _CommittedProof(StrictModel):
    sequence: int = Field(gt=0)
    origin: Literal["human", "ai_computer_use", "synthetic"]
    transaction_id: str = Field(min_length=1)
    before_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    after_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    prediction: dict
    undo_of: str | None = None

    @property
    def _state_hashes(self):
        return self.before_hash, self.after_hash


def _transition_proof(transition):
    if isinstance(transition, _CommittedProof):
        return transition
    before, after = transition._state_hashes or (
        session_hash(transition.before),
        session_hash(transition.after),
    )
    return _CommittedProof(
        sequence=transition.sequence,
        origin=transition.origin,
        transaction_id=transition.transaction_id,
        before_hash=before,
        after_hash=after,
        prediction=copy.deepcopy(transition.prediction),
        undo_of=transition.undo_of,
    )


class _SessionHistory:
    limit = 512

    def __init__(self, compact=False):
        self.entries, self.active = [], []
        self.by_id, self.known_keys = {}, {}
        self.parent = None
        self.compact = compact
        self._prefix = 0

    @property
    def tip(self):
        last = self.entries[-1].transition if self.entries else None
        return {
            "transaction_id": last.transaction_id if last else None,
            "sequence": last.sequence if last else 0,
            "count": len(self.entries),
        }

    def _append(self, entry):
        transition = entry.transition
        if transition.transaction_id in self.by_id:
            raise ValueError(
                "Committed history must have unique chronological identities."
            )
        if self.entries:
            previous = self.entries[-1].transition
            if transition.sequence <= previous.sequence:
                raise ValueError(
                    "Committed history must have unique chronological identities."
                )
            if previous._state_hashes[1] != transition._state_hashes[0]:
                raise ValueError(
                    "Committed history is not an actual consecutive state chain."
                )
        if transition.undo_of:
            original = self.by_id.get(transition.undo_of)
            if original is None or not self.active or self.active[-1] is not original:
                raise ValueError(
                    "Undo must reverse the latest committed active transaction."
                )
            original_transition = original.transition
            if (
                transition._state_hashes != original_transition._state_hashes[::-1]
                or transition.prediction != original_transition.prediction
            ):
                raise ValueError("Undo does not restore its actual prior state.")
            self.active.pop()
        else:
            self.active.append(entry)
        self.entries.append(entry)
        self.by_id[transition.transaction_id] = entry
        self.known_keys[transition.transaction_id] = entry.key

    def stage(self, values, as_of, base):
        if type(as_of) is not int or as_of < 0:
            raise ValueError("Prediction sequence must be a nonnegative integer.")
        earlier_base = None
        if base is not None and base != self.tip["transaction_id"] and self.compact:
            earlier_base = next(
                (
                    index + 1
                    for index, entry in enumerate(self.entries)
                    if entry.transition.transaction_id == base
                ),
                None,
            )
        if base is not None and base != self.tip["transaction_id"] and earlier_base is None:
            raise HistoryResyncRequired(
                "History base is not the last acknowledged transaction."
            )
        candidate = _SessionHistory(self.compact)
        candidate.parent = self
        candidate.known_keys = dict(self.known_keys)
        keyed = [(value, _history_key(value)) for value in values]
        for value, key in keyed:
            known = self.known_keys.get(value.get("transaction_id"))
            if known is not None and known != key:
                raise ValueError(
                    "An acknowledged transaction was changed under the same identity."
                )
        if earlier_base is not None:
            overlap = self.entries[earlier_base:]
            if len(keyed) < len(overlap) or any(
                key != entry.key for (_, key), entry in zip(keyed, overlap)
            ):
                raise HistoryResyncRequired(
                    "History retry does not include the previously committed suffix."
                )
            keyed = keyed[len(overlap) :]
        prefix = len(self.entries) if base is not None else 0
        if base is None:
            while prefix < min(len(keyed), len(self.entries)):
                if keyed[prefix][1] != self.entries[prefix].key:
                    break
                prefix += 1
            keyed = keyed[prefix:]
        candidate._prefix = prefix
        if prefix == len(self.entries):
            candidate.entries = list(self.entries)
            candidate.active = list(self.active)
            candidate.by_id = dict(self.by_id)
        else:
            for entry in self.entries[:prefix]:
                candidate._append(entry)
        for value, key in keyed:
            _checkpoint()
            identity = value.get("transaction_id")
            if (
                not self.compact
                and identity not in candidate.known_keys
                and len(candidate.known_keys) >= self.limit
            ):
                raise HistoryLimitError(
                    "Committed history is full; start a fresh copilot session."
                )
            entry = self.by_id.get(identity)
            if entry is None:
                transition = CommittedTransition.model_validate(value)
                if transition._state_hashes is None:
                    transition._state_hashes = (
                        session_hash(transition.before),
                        session_hash(transition.after),
                    )
                summary = (
                    None
                    if transition.undo_of
                    else _history_summary(
                        transition.before, transition.prediction, transition.origin
                    )
                )
                previous = (
                    candidate.entries[-1].transition if candidate.entries else None
                )
                entry = _HistoryEntry(
                    key,
                    _transition_proof(transition)
                    if self.compact
                    else _own_transition(transition, previous),
                    copy.deepcopy(summary),
                )
            candidate._append(entry)
        if candidate.entries and candidate.entries[-1].transition.sequence > as_of:
            raise ValueError(
                "Incremental history cannot acknowledge a future transaction."
            )
        return candidate

    def compact_copy(self):
        candidate = _SessionHistory(compact=True)
        candidate.known_keys = dict(self.known_keys)
        for entry in self.entries:
            candidate._append(
                _HistoryEntry(
                    entry.key, _transition_proof(entry.transition), entry.summary
                )
            )
        return candidate

    def receipt(self, full=False):
        if not self.compact:
            raise ValueError("Only validated compact history has a runtime receipt.")
        prefix = 0 if full else self._prefix
        return {
            "format": "validated-history-v1",
            "prefix": prefix,
            "entries": [
                {
                    "key": entry.key.hex(),
                    "transition": entry.transition.model_dump(),
                    "summary": entry.summary,
                }
                for entry in self.entries[prefix:]
            ],
        }

    def restore_receipt(self, value):
        if (
            set(value) != {"format", "prefix", "entries"}
            or value["format"] != "validated-history-v1"
            or type(value["prefix"]) is not int
            or not 0 <= value["prefix"] <= len(self.entries)
        ):
            raise ValueError("Unknown validated history receipt.")
        candidate = _SessionHistory(compact=True)
        candidate.known_keys = dict(self.known_keys)
        for entry in self.entries[: value["prefix"]]:
            candidate._append(entry)
        for entry in value["entries"]:
            transition = _CommittedProof.model_validate(entry["transition"])
            key = bytes.fromhex(entry["key"])
            known = candidate.known_keys.get(transition.transaction_id)
            if len(key) != 32 or (known is not None and known != key):
                raise ValueError("A validated history identity changed.")
            summary = entry["summary"]
            if transition.undo_of is None:
                RecentEdit.model_validate(summary)
            elif summary is not None:
                raise ValueError("Undo history cannot introduce an edit summary.")
            candidate._append(_HistoryEntry(key, transition, copy.deepcopy(summary)))
        return candidate

    def context(self, document, as_of):
        if self.entries and (
            self.entries[-1].transition.sequence > as_of
            or self.entries[-1].transition._state_hashes[1] != session_hash(document)
        ):
            raise ValueError(
                "The final committed history state is not the current document."
            )
        return (
            [entry.transition for entry in self.entries],
            [entry.summary for entry in self.active],
        )


class _ProposalLedger(FeedbackLedger):
    def propose(self, proposal_id, sequence, document, bundle, prediction):
        if proposal_id in self.proposals or sequence < 0:
            raise ValueError("Proposal identity must be new and sequence nonnegative.")
        _legacy_bundle(document, bundle)
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
        evidence_version = "target-neighborhood-v2"
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
                    _INTERNAL_BRIEF,
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
                "correction_effect": _correction_effect(document, bundle, prediction)
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


class PassiveFeedbackLedger:
    """Actual proposal/outcome records; acceptance and undo require committed deltas."""

    def __init__(self, document_id):
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Feedback needs the actual session document identity.")
        self.document_id = document_id
        self._ledger = _ProposalLedger(document_id)
        self._commits, self._undos, self._records = {}, {}, []

    def _save_record(self, value):
        self._records.append(copy.deepcopy(value))

    def _commit_value(self, transition):
        return transition.model_dump()

    def _reverses(self, transition, original):
        return (
            transition.before == original["after"]
            and transition.after == original["before"]
        )

    def propose(self, proposal_id, sequence, document, bundle, prediction):
        if type(sequence) is not int or sequence < bundle["context"]["as_of"]:
            raise ValueError("Proposal sequence predates its context.")
        self._ledger.propose(proposal_id, sequence, document, bundle, prediction)
        self._save_record(
            {
                "kind": "propose",
                "proposal_id": proposal_id,
                "sequence": sequence,
                "document": document,
                "bundle": bundle,
                "prediction": prediction,
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
                    or not self._reverses(committed, original)
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
                    apply_passive_edit(
                        committed.before, old_bundle, committed.prediction
                    )
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
                self._commit_value(committed)
            )
        self._save_record(
            {
                "kind": "record",
                **event,
                "transition": self._commit_value(committed) if committed else None,
                "applied": applied,
                "transaction_id": transaction_id,
            }
        )

    def _verify_history(self, past, as_of):
        required = []
        for event in self._ledger.events:
            if event["sequence"] > as_of:
                continue
            if event["outcome"] in {"accepted", "modified", "undone"}:
                records = self._undos if event["outcome"] == "undone" else self._commits
                required.append(records[event["proposal_id"]])
        identities = {transition["transaction_id"] for transition in required}
        actual = {
            t.transaction_id: self._commit_value(t)
            for t in past
            if t.transaction_id in identities
        }
        for transition in required:
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


class CompactFeedbackLedger(PassiveFeedbackLedger):
    """Product receipts retain verified window/proof data, never source documents."""

    storage_format = "passive-feedback-validated-v1"

    @classmethod
    def from_ledger(cls, ledger):
        result = cls(ledger.document_id)
        result._ledger = copy.copy(ledger._ledger)
        result._ledger.proposals = dict(ledger._ledger.proposals)
        result._ledger.events = list(ledger._ledger.events)
        for name in ("_commits", "_undos"):
            setattr(
                result,
                name,
                {
                    key: _transition_proof(
                        CommittedTransition.model_construct(**value)
                    ).model_dump()
                    for key, value in getattr(ledger, name).items()
                },
            )
        return result

    def fork(self):
        result = copy.copy(self)
        result._ledger = copy.copy(self._ledger)
        # Existing proposals, events and proofs are append-only owned values.
        result._ledger.proposals = dict(self._ledger.proposals)
        result._ledger.events = list(self._ledger.events)
        result._commits, result._undos = dict(self._commits), dict(self._undos)
        result._records = []
        return result

    def _commit_value(self, transition):
        return _transition_proof(transition).model_dump()

    def _reverses(self, transition, original):
        return _transition_proof(transition)._state_hashes == (
            original["after_hash"],
            original["before_hash"],
        )

    def suppressed(self, document, bundle, prediction, as_of):
        _legacy_bundle(document, bundle)
        edit = EDIT.validate_python(prediction)
        if edit.action == "no_action":
            return False
        targets = [bundle["mapping"][str(target)]["uuid"] for target in edit.target]
        aliases = {value["uuid"]: int(key) for key, value in bundle["mapping"].items()}
        parameters = edit.model_dump(exclude={"target"})
        effects, fingerprints = {}, {}

        def effect(value):
            key = _cache_key(value)
            if key not in effects:
                effects[key] = _correction_effect(document, bundle, value)
            return effects[key]

        for event in self._ledger.outcomes(as_of):
            if event["outcome"] not in {"dismissed", "undone"}:
                continue
            proposal = self._ledger.proposals[event["proposal_id"]]
            evidence_key = (
                tuple(proposal["evidence_targets"]),
                proposal.get("evidence_version", "paragraph-v1"),
                proposal["prediction"]["action"],
            )
            if evidence_key not in fingerprints:
                fingerprints[evidence_key] = evidence_fingerprint(
                    document,
                    proposal["evidence_targets"],
                    _INTERNAL_BRIEF,
                    evidence_key[1],
                    evidence_key[2],
                )
            if fingerprints[evidence_key] != proposal["evidence_hash"]:
                continue
            refused = (
                event["applied"]
                if event["outcome"] == "undone"
                else proposal["prediction"]
            )
            refused_targets = [
                proposal["mapping"][str(target)]["uuid"] for target in refused["target"]
            ]
            if targets == refused_targets and parameters == {
                key: value for key, value in refused.items() if key != "target"
            }:
                return True
            if refused["action"] == edit.action == "correct_text":
                if (
                    event["outcome"] == "dismissed"
                    and effect(prediction) == proposal["correction_effect"]
                ):
                    return True
                if event["outcome"] == "undone" and all(
                    target in aliases for target in refused_targets
                ):
                    restored = {
                        **refused,
                        "target": [aliases[target] for target in refused_targets],
                    }
                    try:
                        if effect(prediction) == effect(restored):
                            return True
                    except ValueError:
                        pass
        return False

    def _save_record(self, value):
        if value["kind"] == "propose":
            value = {
                "kind": "propose",
                "proposal_id": value["proposal_id"],
                "proposal": self._ledger.proposals[value["proposal_id"]],
            }
        self._records = [
            copy.deepcopy({"storage_format": self.storage_format, **value})
        ]

    def restore_record(self, value):
        # Only the verified durable product journal supplies these receipts. A client
        # still submits full transitions through record(), where execution is checked.
        if value.get("storage_format") != self.storage_format:
            event = copy.deepcopy(value)
            kind = event.pop("kind")
            if kind == "propose":
                self.propose(**event)
            elif kind == "record":
                self.record(**event)
            else:
                raise ValueError("Unknown passive feedback event.")
            return
        event = copy.deepcopy(value)
        event.pop("storage_format")
        kind = event.pop("kind")
        if kind == "propose":
            identity, proposal = event["proposal_id"], event["proposal"]
            if identity in self._ledger.proposals:
                raise ValueError("A validated proposal identity was reused.")
            parse_snapshot(proposal["snapshot"])
            EDIT.validate_python(proposal["prediction"])
            self._ledger.proposals[identity] = proposal
        elif kind == "record":
            transition = event.pop("transition")
            proof = _CommittedProof.model_validate(transition) if transition else None
            if event["outcome"] in {"accepted", "modified", "undone"}:
                if proof is None or (
                    proof.sequence != event["sequence"]
                    or proof.origin != event["origin"]
                ):
                    raise ValueError("Validated feedback contradicts its transition.")
                if event["outcome"] == "undone":
                    original = self._commits[event["proposal_id"]]
                    if (
                        proof.undo_of != original["transaction_id"]
                        or not self._reverses(proof, original)
                        or proof.prediction != original["prediction"]
                    ):
                        raise ValueError(
                            "Validated undo does not reverse its acceptance."
                        )
                elif (
                    proof.undo_of
                    or proof.prediction != event["applied"]
                    or proof.transaction_id != event["transaction_id"]
                ):
                    raise ValueError("Validated acceptance differs from its edit.")
            elif proof is not None:
                raise ValueError("Only acceptance or undo has a validated transition.")
            self._ledger.record(**event)
            if proof is not None:
                target = self._undos if event["outcome"] == "undone" else self._commits
                target[event["proposal_id"]] = proof.model_dump()
        else:
            raise ValueError("Unknown validated feedback event.")
        self._records = []

    def dump(self):
        return copy.deepcopy(
            {
                "storage_format": self.storage_format,
                "contract_version": VERSION,
                "document_id": self.document_id,
                "proposals": self._ledger.proposals,
                "events": self._ledger.events,
                "commits": self._commits,
                "undos": self._undos,
            }
        )


class _CaptionWindow:
    def __init__(self, document, records, items, profile):
        self.facts = caption_facts(document, records, items, [], profile)
        self.items = defaultdict(list)
        for item in items:
            self.items[item["paragraph"]].append(item)

    def select(self, items):
        visible = defaultdict(list)
        for item in items:
            visible[item["paragraph"]].append(item)
        result = []
        for facts in self.facts:
            paragraph = facts["paragraph"]
            if paragraph not in visible:
                continue
            original, local = self.items[paragraph], visible[paragraph]
            start = local[0]["id"] - original[0]["id"]
            end = local[-1]["id"] - original[0]["id"] + 1
            value = {
                **facts,
                "visible_start_ms": facts["visible_start_ms"][start:end],
                "split_exports": [
                    split
                    for split in facts["split_exports"]
                    if local[0]["id"] <= split["target"] <= local[-1]["id"]
                ],
            }
            for side, removed in (
                ("prefix", original[:start]),
                ("suffix", original[end:]),
            ):
                count, units = text_metrics(removed)
                old_count = facts[f"{side}_text_items"]
                value[f"{side}_text_items"] = old_count + count
                value[f"{side}_text_utf16"] = (
                    facts[f"{side}_text_utf16"] + units + int(bool(old_count and count))
                )
            result.append(value)
        return result


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
        captions = _CaptionWindow(document, records, records[start:end], profile)
        while True:
            _checkpoint()
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
                    "caption_facts": captions.select(items),
                }
            ).model_dump()
            count_tokens = token_count(self.tokenizer, value)
            comparison = {
                name: token_count(t, value)
                for name, t in self.budget_tokenizers.items()
            }
            if max([count_tokens, *comparison.values()]) <= self.policy.input_tokens:
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
            "mapping": copy.deepcopy(local_map),
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
                "input_tokens": count_tokens,
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
        self._validation_cache = _ValidationCache()
        self._history_state = _SessionHistory()
        self._history_persisted = False

    def enable_compact_runtime(self):
        if not isinstance(self.feedback, CompactFeedbackLedger):
            self.feedback = CompactFeedbackLedger.from_ledger(self.feedback)
            self._history_state = self._history_state.compact_copy()

    @contextmanager
    def validation_scope(self, check_cancelled=None):
        current = _validation_context.get()
        if current is not None and current[0] is self._validation_cache:
            yield
            return
        token = _validation_context.set((self._validation_cache, check_cancelled))
        try:
            _checkpoint()
            yield
            _checkpoint()
        finally:
            _validation_context.reset(token)

    @_session_cached
    def prepare(self, document, focus_uuid, **kwargs):
        bundle = self.builder.build(
            document, focus_uuid, feedback=self.feedback, **kwargs
        )
        self._prepared.add(session_hash(bundle))
        if type(self.builder) is PassiveContextBuilder:
            key = _cache_key([[document, bundle], {}])
            self._validation_cache.put(
                "bundle",
                key,
                copy.deepcopy({**bundle, "input": _v4_data(bundle["input"])}),
            )
        return bundle

    @_session_cached
    def prepare_incremental(
        self, document, focus_uuid, *, history=(), history_base=None, **kwargs
    ):
        candidate = self._history_state.stage(history, kwargs["as_of"], history_base)
        bundle = self.prepare(document, focus_uuid, history=candidate, **kwargs)
        return bundle, candidate

    def commit_history(self, candidate):
        if candidate.parent is not self._history_state:
            raise HistoryResyncRequired(
                "History changed before this prediction completed."
            )
        candidate.parent = None
        self._history_state = candidate

    @_session_cached
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
