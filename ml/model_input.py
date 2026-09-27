"""One message and token serialization path for generation, training and inference."""

import json
import importlib


ITEM_FIELDS = (
    "id",
    "type",
    "text",
    "duration_ms",
    "paragraph",
    "speaker",
    "language",
    "source_backed",
    "source_run",
    "boundary_status",
    "audio_class",
)
CAPTION_FIELDS = (
    "paragraph",
    "first_id",
    "last_id",
    "duration_ms",
    "protected",
    "prefix_text_items",
    "prefix_text_utf16",
    "suffix_text_items",
    "suffix_text_utf16",
    "visible_start_ms",
    "exported",
    "split_exports",
)
METRIC_FIELDS = ("count", "min_ms", "max_ms", "max_text_utf16")
COMPLETE_FIELDS = ("paragraph", "duration_ms", "visible_start_ms")


def wire_snapshot(snapshot):
    """Losslessly omit repeated item keys in v4; canonical stored snapshots stay readable."""
    if snapshot.get("schema_version") != 4:
        return snapshot
    result = dict(snapshot)
    result["item_fields"] = list(ITEM_FIELDS)
    result["items"] = [[item[key] for key in ITEM_FIELDS] for item in snapshot["items"]]
    result["caption_fact_fields"] = list(CAPTION_FIELDS)
    result["caption_metric_fields"] = list(METRIC_FIELDS)
    result["complete_caption_fields"] = list(COMPLETE_FIELDS)
    result["complete_caption_facts"] = []
    result["caption_facts"] = []
    for fact in snapshot["caption_facts"]:
        if (
            fact["paragraph"] in snapshot["complete_paragraphs"]
            and snapshot["caption_profile"]["export_line_length"] is None
        ):
            result["complete_caption_facts"].append(
                [fact[key] for key in COMPLETE_FIELDS]
            )
            continue
        row = dict(fact)
        row["exported"] = [fact["exported"][key] for key in METRIC_FIELDS]
        row["split_exports"] = [
            [s["target"], *[s["after"][key] for key in METRIC_FIELDS]]
            for s in fact["split_exports"]
        ]
        result["caption_facts"].append([row[key] for key in CAPTION_FIELDS])
    return result


def parse_snapshot(value):
    value = dict(value)
    if value.get("schema_version") == 4 and "item_fields" in value:
        if value.pop("item_fields") != list(ITEM_FIELDS):
            raise ValueError("Unknown v4 item wire layout.")
        value["items"] = [
            dict(zip(ITEM_FIELDS, row, strict=True)) for row in value["items"]
        ]
        if value.pop("caption_fact_fields") != list(CAPTION_FIELDS) or value.pop(
            "caption_metric_fields"
        ) != list(METRIC_FIELDS):
            raise ValueError("Unknown v4 caption wire layout.")
        if value.pop("complete_caption_fields") != list(COMPLETE_FIELDS):
            raise ValueError("Unknown complete caption wire layout.")
        facts = [
            dict(zip(CAPTION_FIELDS, row, strict=True))
            for row in value["caption_facts"]
        ]
        for row in value.pop("complete_caption_facts"):
            fact = dict(zip(COMPLETE_FIELDS, row, strict=True))
            local = [i for i in value["items"] if i["paragraph"] == fact["paragraph"]]
            if (
                fact["paragraph"] not in value["complete_paragraphs"]
                or value["caption_profile"]["export_line_length"] is not None
            ):
                raise ValueError(
                    "Compact complete facts require complete text without automatic line splitting."
                )
            fact.update(
                first_id=local[0]["id"],
                last_id=local[-1]["id"],
                protected=any(i["id"] in value["protected_ids"] for i in local),
                prefix_text_items=0,
                prefix_text_utf16=0,
                suffix_text_items=0,
                suffix_text_utf16=0,
                exported=None,
                split_exports=[],
            )
            facts.append(fact)
        for fact in facts:
            local = [i for i in value["items"] if i["paragraph"] == fact["paragraph"]]
            text = [i["text"] for i in local if i["type"] == "text"]
            groups = sum(
                (
                    fact["prefix_text_items"] > 0,
                    bool(text),
                    fact["suffix_text_items"] > 0,
                )
            )
            fact.update(
                speaker=local[0]["speaker"],
                language=local[0]["language"],
                text_items=fact["prefix_text_items"]
                + len(text)
                + fact["suffix_text_items"],
                text_utf16=fact["prefix_text_utf16"]
                + len(" ".join(text).encode("utf-16-le")) // 2
                + fact["suffix_text_utf16"]
                + max(0, groups - 1),
            )
            fact["exported"] = (
                dict(zip(METRIC_FIELDS, fact["exported"], strict=True))
                if fact["exported"] is not None
                else {
                    "count": 1,
                    "min_ms": fact["duration_ms"],
                    "max_ms": fact["duration_ms"],
                    "max_text_utf16": fact["text_utf16"],
                }
            )
            fact["split_exports"] = [
                {"target": s[0], "after": dict(zip(METRIC_FIELDS, s[1:], strict=True))}
                for s in fact["split_exports"]
            ]
        paragraph_order = list(dict.fromkeys(i["paragraph"] for i in value["items"]))
        value["caption_facts"] = sorted(
            facts, key=lambda f: paragraph_order.index(f["paragraph"])
        )
    return importlib.import_module(
        f"nep_v{value['schema_version']}"
    ).Snapshot.model_validate(value)


def validate_prediction(text, snapshot, *args, **kwargs):
    return importlib.import_module(
        f"nep_v{snapshot.schema_version}"
    ).validate_prediction(text, snapshot, *args, **kwargs)


def messages(snapshot, prompt):
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                wire_snapshot(snapshot), ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]


def token_ids(tokenizer, snapshot, prompt):
    return tokenizer.apply_chat_template(
        messages(snapshot, prompt),
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
