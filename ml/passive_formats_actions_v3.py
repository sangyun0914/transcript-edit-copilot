"""Seven-action JSON and scoped Rewrite over exactly the same visible input."""

import json

import passive_actions as contract
from format_comparison import Rewrite, source_item

VERSION = "passive-formats-actions-v3"
INPUT_LIMIT, OUTPUT_LIMIT, CONTEXT_LIMIT = 3584, 512, 4096
FORMATS = ("json", "rewrite")
REWRITE_INSTRUCTIONS = """Return one scoped after-state JSON, no prose: {"format_version":1,"scope":[aliases],"after":[{"ids":[aliases],"text":"display text","audio":{"mode":"full"},"boundary_before":false}]}. For no_action both arrays are empty. scope contains ordered contiguous visible aliases. Preserve existing IDs and everything outside scope. A correction combines scope into one after item with changed text, full audio and the original first boundary. A deletion or complete pause removal has an empty after array. A pause trim keeps the same item with audio {"mode":"edges","keep_ms":positive_integer}. A split/merge scopes exactly previous and target items, retains their text/audio, and toggles only the second boundary. Original boundaries come from caption_facts.first_id; non_text text is null. Exactly one supported action is allowed; no extra keys or source changes."""


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def rewrite_prompt():
    policy, output = contract.PROMPT.split("The seven output forms are:\n", 1)
    _, safeguards = output.split("\n\n", 1)
    safeguards = safeguards.rsplit("\nReturn exactly one JSON object", 1)[0]
    return policy + safeguards + "\n\n" + REWRITE_INSTRUCTIONS


REWRITE_PROMPT = rewrite_prompt()


def messages(snapshot, format_name="json"):
    result = contract.canonical_messages(snapshot)
    if format_name == "rewrite":
        result[0] = {"role": "system", "content": REWRITE_PROMPT}
    elif format_name != "json":
        raise ValueError("Unknown seven-action format.")
    return result


def request():
    return contract.Envelope(
        document_id="format", request_id="format", revision=0, interaction_epoch=0
    )


def rewrite_value(snapshot, edit):
    snapshot = contract.parse_snapshot(snapshot)
    if edit["action"] == "no_action":
        return {"format_version": 1, "scope": [], "after": []}
    by_id = {item.id: item for item in snapshot.items}
    positions = {item.id: index for index, item in enumerate(snapshot.items)}
    scope = list(edit["target"])
    if edit["action"] in ("split_paragraph", "merge_paragraph"):
        scope.insert(0, snapshot.items[positions[scope[0]] - 1].id)
    after = [source_item(snapshot, by_id[alias]) for alias in scope]
    if edit["action"] in ("delete_span", "remove_silence"):
        after = []
    elif edit["action"] == "correct_text":
        after = [{**after[0], "ids": scope, "text": edit["text"]}]
    elif edit["action"] == "trim_silence":
        after[0]["audio"] = {"mode": "edges", "keep_ms": edit["keep_ms"]}
    else:
        after[-1]["boundary_before"] = edit["action"] == "split_paragraph"
    return Rewrite.model_validate(
        {"format_version": 1, "scope": scope, "after": after}
    ).model_dump()


def serialize(snapshot, action, format_name="json"):
    envelope = request()
    checked = contract.validate_prediction(
        compact(action), snapshot, envelope, envelope, "stop"
    )
    if format_name == "json":
        return compact(checked.model_dump())
    if format_name == "rewrite":
        return compact(rewrite_value(snapshot, checked.model_dump()))
    raise ValueError("Unknown seven-action format.")


def parse(
    text,
    snapshot,
    format_name="json",
    *,
    envelope=None,
    current=None,
    finish_reason="stop",
):
    envelope = envelope or request()
    current = current or envelope
    if format_name == "json":
        return contract.validate_prediction(
            text, snapshot, envelope, current, finish_reason
        )
    if format_name != "rewrite":
        raise ValueError("Unknown seven-action format.")
    if envelope != current or finish_reason != "stop":
        raise ValueError("Stale or incomplete Rewrite response.")
    snapshot = contract.parse_snapshot(snapshot)
    value = Rewrite.model_validate_json(text, strict=True).model_dump()
    scope, after = value["scope"], value["after"]
    by_id = {item.id: item for item in snapshot.items}
    positions = {item.id: index for index, item in enumerate(snapshot.items)}
    if not scope:
        action = {"action": "no_action"}
    else:
        if any(alias not in by_id for alias in scope):
            raise ValueError("Unknown Rewrite source alias.")
        indices = [positions[alias] for alias in scope]
        if indices != list(range(indices[0], indices[0] + len(scope))):
            raise ValueError("Rewrite scope must be contiguous, ordered and unique.")
        if not after:
            action = {
                "action": "remove_silence"
                if len(scope) == 1 and by_id[scope[0]].type == "non_text"
                else "delete_span",
                "target": scope,
            }
        elif len(after) == 1 and after[0]["ids"] == scope:
            state = after[0]
            if state["audio"]["mode"] == "edges":
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
            action = {
                "action": "split_paragraph"
                if after[1]["boundary_before"]
                else "merge_paragraph",
                "target": [scope[1]],
            }
        else:
            raise ValueError("Rewrite must express exactly one transaction.")
    checked = contract.validate_prediction(
        compact(action), snapshot, envelope, current, finish_reason
    )
    if rewrite_value(snapshot, checked.model_dump()) != value:
        raise ValueError(
            "Rewrite changed immutable state, added scope or combined operations."
        )
    return checked


def prefix_ids(tokenizer, snapshot, format_name="json"):
    return tokenizer.apply_chat_template(
        messages(snapshot, format_name),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode(tokenizer, snapshot, action, format_name="json"):
    if (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id) != (
        1,
        0,
        2,
    ):
        raise ValueError("Require the pinned Gemma native special tokens.")
    prefix = prefix_ids(tokenizer, snapshot, format_name)
    text = serialize(snapshot, action, format_name)
    completion = tokenizer.encode(text, add_special_tokens=False)
    if any(token in (1, 106, 50) for token in completion):
        raise ValueError("Reference contains a native terminal token.")
    completion += [1]
    if not 0 < len(prefix) <= INPUT_LIMIT or len(completion) > OUTPUT_LIMIT:
        raise ValueError(
            "Common format budget exceeded; inputs and completions are never truncated."
        )
    return {
        "input_ids": prefix + completion,
        "attention_mask": [1] * (len(prefix) + len(completion)),
        "labels": [-100] * len(prefix) + completion,
        "completion_start": len(prefix),
        "completion_text": text,
        "completion_tokens": len(completion),
        "eos_token_id": 1,
    }
