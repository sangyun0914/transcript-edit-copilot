"""Trusted V3 snapshot serialization and an offline, undoable reference executor."""

import copy
import json
import zipfile
from collections import Counter
from uuid import NAMESPACE_URL, uuid5

from nep_v2 import ACTIONS, Envelope, Snapshot
from replay_session import replay, session_hash


def load_project(path):
    with zipfile.ZipFile(path) as archive:
        document = json.loads(archive.read("document.json"))
    if document.get("version") != 3:
        raise ValueError("Use a V3 project.")
    return {
        "content": document["content"],
        "metadata": {
            k: v
            for k, v in document["metadata"].items()
            if k in ("transcription", "transcriptions")
        },
    }


def document_records(document, evidence=None):
    evidence = evidence or {}
    records, mapping = [], {}
    paragraph, speaker, language, run, previous = -1, None, "und", 0, None
    seen = set()
    for index, item in enumerate(document["content"]):
        if item["uuid"] in seen:
            raise ValueError("Document UUIDs must be unique.")
        seen.add(item["uuid"])
        if item["type"] == "paragraph_start":
            paragraph += 1
            speaker, language = item.get("speaker"), item.get("language") or "und"
        elif item["type"] in ("text", "non_text", "artificial_silence"):
            if paragraph < 0 or item["length"] <= 0:
                raise ValueError("Invalid timed document item.")
            backed = "source" in item
            if previous is not None and (
                not backed
                or "source" not in previous
                or item["source"] != previous["source"]
                or abs(
                    item["sourceStart"] - previous["sourceStart"] - previous["length"]
                )
                > 1e-6
            ):
                run += 1
            proof = evidence.get(item["uuid"])
            if proof and proof["item_hash"] != session_hash(item):
                raise ValueError("Boundary evidence does not match the source item.")
            alias = len(records)
            records.append(
                {
                    "id": alias,
                    "type": "text" if item["type"] == "text" else "non_text",
                    "text": item.get("text", ""),
                    "duration_ms": max(1, round(item["length"] * 1000)),
                    "paragraph": paragraph,
                    "speaker": speaker,
                    "language": language,
                    "source_backed": backed,
                    "source_run": run,
                    "boundary_status": proof.get("boundary_status", "verified")
                    if proof
                    else "unverified",
                    "audio_class": proof["audio_class"]
                    if proof
                    else ("speech" if item["type"] == "text" else "unknown"),
                }
            )
            mapping[str(alias)] = {
                "index": index,
                "uuid": item["uuid"],
                "item_hash": session_hash(item),
            }
            previous = item
    return records, mapping


def snapshot_for(
    document, focus_uuid, brief, history=(), evidence=None, window_size=64
):
    if not 1 <= window_size <= 256:
        raise ValueError("Use a window of 1–256 items.")
    records, mapping = document_records(document, evidence)
    focus = next((int(k) for k, v in mapping.items() if v["uuid"] == focus_uuid), None)
    if focus is None:
        raise ValueError("Focus must reference a timed document item.")
    start = max(0, min(focus - window_size // 2, len(records) - window_size))
    items = records[start : start + window_size]
    counts = Counter(i["paragraph"] for i in records)
    visible = Counter(i["paragraph"] for i in items)
    snapshot = Snapshot.model_validate(
        {
            "schema_version": 2,
            "task": "predict_next_transcript_edit",
            "language": records[focus]["language"],
            "mode": "transcript_cleanup",
            "task_brief": brief,
            "cursor_before": focus,
            "selection": [],
            "recent_edits": list(history)[-8:],
            "items": items,
            "allowed_actions": ACTIONS,
            "protected_ids": [],
            "complete_paragraphs": [p for p, n in visible.items() if counts[p] == n],
        }
    )
    return {
        "input": snapshot.model_dump(),
        "document_hash": session_hash(document),
        "mapping": {str(i["id"]): mapping[str(i["id"])] for i in items},
    }


def apply_edit(document, bundle, prediction, policy=None):
    if session_hash(document) != bundle["document_hash"]:
        raise ValueError("Stale source document.")
    from caption_facts import verify_caption_facts
    from model_input import parse_snapshot, validate_prediction as validate_versioned

    snapshot = parse_snapshot(bundle["input"])
    verify_caption_facts(document, bundle)
    for item in snapshot.items:
        mapped = bundle["mapping"][str(item.id)]
        current = document["content"][mapped["index"]]
        if (
            current["uuid"] != mapped["uuid"]
            or session_hash(current) != mapped["item_hash"]
        ):
            raise ValueError("Alias mapping does not match the source document.")
    request = Envelope(
        document_id=bundle["document_hash"],
        request_id="offline",
        revision=0,
        interaction_epoch=0,
    )
    edit = validate_versioned(
        json.dumps(prediction), snapshot, request, request, "stop", policy
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


def transaction(before, after, seq, revision, action, origin="synthetic", **extra):
    return {
        "kind": "transaction",
        "seq": seq,
        "revision": revision,
        "interaction_epoch": revision,
        "action_type": f"synthetic/{action}",
        "operation": action,
        "origin": origin,
        "actor": "synthetic",
        "before_hash": session_hash(before),
        "after_hash": session_hash(after),
        "patch": {
            "index": 0,
            "remove": before["content"],
            "insert": after["content"],
            "previous_metadata": before["metadata"],
            "metadata": after["metadata"],
        },
        **extra,
    }


def replayable_session(before, after, action, source_info):
    events = [
        {
            "kind": "start",
            "seq": 0,
            "schema_version": 2,
            "hash_format": "typed-json-f64-v1",
            "actor": "synthetic",
            "collection_mode": "synthetic",
            "generator": "synthetic-pilot-v1",
            "document": before,
            "document_hash": session_hash(before),
            "sources": source_info,
        }
    ]
    if before != after:
        events.extend(
            [
                transaction(before, after, 1, 1, action),
                transaction(after, before, 2, 2, action, origin="undo"),
                transaction(before, after, 3, 3, action, origin="redo"),
            ]
        )
    events.append(
        {
            "kind": "end",
            "seq": len(events),
            "revision": len(events) - 1,
            "document_hash": session_hash(after),
            "transactions": len(events) - 1,
        }
    )
    actual, summary = replay(events)
    if actual != after or not summary["complete"]:
        raise ValueError("Synthetic session failed replay.")
    return events
