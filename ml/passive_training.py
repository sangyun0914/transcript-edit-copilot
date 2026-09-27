"""Training examples use exactly the passive serving messages, with completion-only loss."""

import json

from passive_contract import (
    Envelope,
    canonical_messages,
    token_ids,
    validate_prediction,
)


def training_messages(case):
    if case["metadata"].get("training_eligible") is not True:
        raise ValueError("Only independently admitted training rows may be exported.")
    if case["metadata"]["split"] != "train":
        raise ValueError("Development and reserved cases are not training examples.")
    request = Envelope(
        document_id=case["id"], request_id="training", revision=0, interaction_epoch=0
    )
    validate_prediction(
        json.dumps(case["expectedOutput"], ensure_ascii=False),
        case["input"],
        request,
        request,
        "stop",
    )
    return canonical_messages(case["input"]) + [
        {
            "role": "assistant",
            "content": json.dumps(
                case["expectedOutput"], ensure_ascii=False, separators=(",", ":")
            ),
        }
    ]


def training_row(tokenizer, case, max_length=4096, max_output_tokens=512):
    messages = training_messages(case)
    prefix = token_ids(tokenizer, case["input"])
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must declare an EOS token.")
    completion = tokenizer.encode(messages[-1]["content"], add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    if (
        len(prefix) > max_length - max_output_tokens
        or len(completion) > max_output_tokens
        or len(prefix) + len(completion) > max_length
    ):
        raise ValueError("Never truncate an editing target or its inference context.")
    return {
        "input_ids": prefix + completion,
        "attention_mask": [1] * (len(prefix) + len(completion)),
        "labels": [-100] * len(prefix) + completion,
    }
