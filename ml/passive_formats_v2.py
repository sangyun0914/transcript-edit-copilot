"""Matched passive JSON/Rewrite views with one shared, unchanged evidence window."""

import json

from format_comparison import parse_rewrite as parse_v4_rewrite
from format_comparison import serialize_rewrite as serialize_v4_rewrite
from nep_v4 import Snapshot
from passive_contract import (
    PROMPT,
    Envelope,
    _v4_data,
    canonical_messages,
    parse_snapshot,
    validate_prediction,
)

VERSION = "passive-format-comparison-v2"
INPUT_LIMIT = 3584
OUTPUT_LIMIT = 512
CONTEXT_LIMIT = 4096
REWRITE_INSTRUCTIONS = """Return one scoped after-state JSON, no prose: {\"format_version\":1,\"scope\":[aliases],\"after\":[{\"ids\":[aliases],\"text\":\"display text\",\"audio\":{\"mode\":\"full\"},\"boundary_before\":false}]}. For no_action both arrays are empty. scope contains ordered contiguous visible aliases. Preserve existing IDs, audio and everything outside scope. A correction combines scope into one after item with changed text and the original first boundary. A split/merge scopes exactly previous and target items, retains their text/audio, and toggles only the second boundary. Original boundaries come from caption_facts.first_id; non_text text is null. No other changes or extra keys."""


def rewrite_prompt():
    policy, output_section = PROMPT.split("The seven output forms are:\n", 1)
    _, safeguards = output_section.split("\nOnly allowed_actions", 1)
    safeguards = ("Only allowed_actions" + safeguards).rsplit(
        "\nReturn exactly one JSON object", 1
    )[0]
    return policy + safeguards + "\n\n" + REWRITE_INSTRUCTIONS


REWRITE_PROMPT = rewrite_prompt()


def messages(snapshot, format_name="json"):
    result = canonical_messages(snapshot)
    if format_name == "rewrite":
        result[0] = {"role": "system", "content": REWRITE_PROMPT}
    elif format_name != "json":
        raise ValueError("Unknown passive output format.")
    return result


def request():
    return Envelope(
        document_id="format", request_id="format", revision=0, interaction_epoch=0
    )


def serialize(snapshot, action, format_name="json"):
    envelope = request()
    checked = validate_prediction(
        json.dumps(action, ensure_ascii=False), snapshot, envelope, envelope, "stop"
    )
    value = checked.model_dump()
    if format_name == "rewrite":
        legacy = Snapshot.model_validate(_v4_data(parse_snapshot(snapshot).model_dump()))
        value = serialize_v4_rewrite(legacy, value)
    elif format_name != "json":
        raise ValueError("Unknown passive output format.")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse(text, snapshot, format_name="json", *, envelope=None, current=None,
          finish_reason="stop"):
    envelope = envelope or request()
    current = current or envelope
    if format_name == "json":
        return validate_prediction(text, snapshot, envelope, current, finish_reason)
    if format_name != "rewrite":
        raise ValueError("Unknown passive output format.")
    legacy = Snapshot.model_validate(_v4_data(parse_snapshot(snapshot).model_dump()))
    return parse_v4_rewrite(text, legacy, envelope, current, finish_reason)


def prefix_ids(tokenizer, snapshot, format_name="json"):
    return tokenizer.apply_chat_template(
        messages(snapshot, format_name), tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def eligibility(snapshot, tokenizers):
    """Selection uses inputs alone; targets cannot move or crop a comparison window."""
    counts = {
        name: {fmt: len(prefix_ids(tok, snapshot, fmt)) for fmt in ("json", "rewrite")}
        for name, tok in tokenizers.items()
    }
    return all(n <= INPUT_LIMIT for row in counts.values() for n in row.values()), counts


def encode(tokenizer, snapshot, action, format_name="json"):
    prefix = prefix_ids(tokenizer, snapshot, format_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("A training tokenizer must declare EOS.")
    completion = tokenizer.encode(
        serialize(snapshot, action, format_name), add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    if len(prefix) > INPUT_LIMIT or len(completion) > OUTPUT_LIMIT:
        raise ValueError("Input/output exceeds the common format budget; never truncate.")
    return {
        "input_ids": prefix + completion,
        "attention_mask": [1] * (len(prefix) + len(completion)),
        "labels": [-100] * len(prefix) + completion,
        "completion_start": len(prefix),
    }
