"""Load registered authored timelines without representing them as real audio."""

from uuid import NAMESPACE_URL, uuid5

from dataset_release_inventory import ROOT, digest, read
from replay_session import session_hash


def scripted_sources(manifest_path):
    registration = read(ROOT / manifest_path)
    registered_ids = [g["source_id"] for g in registration["groups"]]
    group_ids = [g["id"] for g in registration["groups"]]
    if len(set(registered_ids)) != len(registered_ids) or len(set(group_ids)) != len(
        group_ids
    ):
        raise ValueError("Duplicate authored source or group registration.")
    path = ROOT / registration["sources_file"]
    scripts = read(path)
    registered = {g["source_id"]: g for g in registration["groups"]}
    source_ids = [s["id"] for s in scripts["sources"]]
    if len(set(source_ids)) != len(source_ids) or set(source_ids) != set(registered):
        raise ValueError("Authored sources differ from their preregistered groups.")
    sources, windows = {}, []
    for script in scripts["sources"]:
        group = registered[script["id"]]
        content, evidence, markers, position = [], {}, {}, 0
        source_hash = session_hash(script)

        def uid(name):
            return str(uuid5(NAMESPACE_URL, f"{script['id']}/{name}"))

        for p, paragraph in enumerate(script["paragraphs"]):
            content.append(
                {
                    "type": "paragraph_start",
                    "uuid": uid(f"p{p}"),
                    "speaker": paragraph["speaker"],
                    "language": group["language"],
                }
            )
            for n, item in enumerate(paragraph["items"]):
                if item["ms"] <= 0:
                    raise ValueError("Symbolic duration must be positive.")
                text = item.get("text")
                value = {
                    "type": "text" if text else "non_text",
                    "uuid": uid(f"p{p}/i{n}"),
                    "source": f"symbolic:{source_hash}",
                    "sourceStart": position / 1000,
                    "length": item["ms"] / 1000,
                }
                if text:
                    value.update(text=text, conf=1)
                if item.get("marker"):
                    if item["marker"] in markers:
                        raise ValueError("Duplicate script marker.")
                    markers[item["marker"]] = position / 1000
                content.append(value)
                evidence[value["uuid"]] = {
                    "item_hash": session_hash(value),
                    "boundary_status": "verified",
                    "audio_class": "speech" if text else "verified_silence",
                    "origin": "authored_symbolic_interval_not_acoustic_evidence",
                }
                position += item["ms"]
            content.append({"type": "paragraph_end", "uuid": uid(f"e{p}")})
        document = {
            "content": content,
            "metadata": {
                "provenance": "authored_symbolic_timeline",
                "script_sha256": source_hash,
            },
        }
        sources[script["id"]] = {
            "id": script["id"],
            "source_group_id": group["id"],
            "reserved_split": group["split"],
            "language": group["language"],
            "document": document,
            "boundary_evidence": evidence,
            "markers": markers,
            "media_kind": "authored_symbolic_timeline",
            "source_path": path,
            "video": None,
            "video_sha256": None,
            "source_identity_sha256": source_hash,
            "transcript_sha256": digest(path),
            "transcript_origin": "independent_ai_authored_script",
            "itemization": "authored_symbolic_timestamps_not_natural_word_boundaries",
            "license": "Project-authored synthetic text; no source recording.",
        }
        windows.append(
            {
                "id": f"{script['id']}-01",
                "source_id": script["id"],
                "start_seconds": 0,
                "end_seconds": position / 1000,
                "source_group_id": group["id"],
                "reserved_split": group["split"],
            }
        )
    return sources, windows
