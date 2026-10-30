"""Verify and replay a local editor journal without loading its media."""

import argparse
import copy
import hashlib
import json
import math
import struct
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1024)
def _key_order(key):
    return key.encode("utf-16-be")


def canonical(value):
    if isinstance(value, str):
        return json.encoder.encode_basestring(value)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("Session numbers must be finite.")
        return "#" + struct.pack(">d", 0 if value == 0 else value).hex()
    if isinstance(value, list):
        return "[" + ",".join(canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value, key=_key_order)
        return (
            "{"
            + ",".join(canonical(key) + ":" + canonical(value[key]) for key in keys)
            + "}"
        )
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def session_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def replay(events):
    if (
        not events
        or events[0].get("kind") != "start"
        or events[0].get("schema_version") not in (1, 2)
        or events[0].get("hash_format") != "typed-json-f64-v1"
    ):
        raise ValueError("Unsupported editing session.")
    actor = events[0].get("actor")
    if events[0]["schema_version"] == 2 and actor not in (
        "human",
        "ai_computer_use",
        "synthetic",
    ):
        raise ValueError("Missing or unknown session actor.")
    document = copy.deepcopy(events[0]["document"])
    if session_hash(document) != events[0]["document_hash"]:
        raise ValueError("Initial snapshot hash mismatch.")
    revision, ended = 0, False
    for seq, event in enumerate(events):
        if ended or event["seq"] != seq:
            raise ValueError("Missing, reordered or post-end session event.")
        if event["kind"] not in (
            "start",
            "context",
            "opportunity",
            "source",
            "transaction",
            "end",
        ):
            raise ValueError("Unknown session event.")
        if event["kind"] == "start" and seq != 0:
            raise ValueError("Repeated session start.")
        if event["kind"] == "transaction":
            if events[0]["schema_version"] == 2 and event.get("actor") != actor:
                raise ValueError("Transaction actor differs from the session actor.")
            revision += 1
            if (
                event["revision"] != revision
                or session_hash(document) != event["before_hash"]
            ):
                raise ValueError("Transaction history mismatch.")
            patch = event["patch"]
            index, removed = patch["index"], patch["remove"]
            if type(index) is not int or not 0 <= index <= len(document["content"]):
                raise ValueError("Invalid patch index.")
            if document["content"][index : index + len(removed)] != removed:
                raise ValueError("Patch does not match original items.")
            if (
                "previous_metadata" in patch
                and document["metadata"] != patch["previous_metadata"]
            ):
                raise ValueError("Metadata before-image mismatch.")
            document["content"][index : index + len(removed)] = copy.deepcopy(
                patch["insert"]
            )
            if "metadata" in patch:
                document["metadata"] = copy.deepcopy(patch["metadata"])
            if session_hash(document) != event["after_hash"]:
                raise ValueError("Replayed snapshot hash mismatch.")
        if event["kind"] in ("opportunity", "end"):
            if (
                event["revision"] != revision
                or session_hash(document) != event["document_hash"]
            ):
                raise ValueError("Checkpoint hash mismatch.")
        ended = event["kind"] == "end"
    return document, {
        "events": len(events),
        "transactions": revision,
        "complete": ended,
        "final_hash": session_hash(document),
        "collection_mode": events[0].get("collection_mode"),
        "actor": actor if events[0]["schema_version"] == 2 else "legacy_unknown",
    }


def read_events(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    _, summary = replay(read_events(args.session))
    print(json.dumps(summary, indent=2))
