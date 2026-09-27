"""Bind listening observations to exact renders, then qualify regenerated edits."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from dataset_release_inventory import ROOT, digest, read, resolve_local, write
from replay_session import session_hash
from source_qualification import media_intervals

CUTS = {"delete_span", "trim_silence", "remove_silence"}


def bind_observation(observation, evidence, evidence_sha256):
    if (
        observation["id"] != evidence["id"]
        or observation["source_sha256"] != evidence["source_sha256"]
        or observation["evidence_sha256"] != evidence_sha256
        or observation.get("human_listened") is not True
        or observation.get("decision") != "accept"
        or evidence["role"] != "natural_candidate"
    ):
        raise ValueError(
            "Listening observation does not approve this exact source/render."
        )


def regenerate(config_path, output):
    from verify_audio_review import verify

    config = read(config_path)
    observations_path = resolve_local(ROOT, config["observations"])
    if digest(observations_path) != config["observations_sha256"]:
        raise ValueError("Listening observations changed.")
    observations = read(observations_path)["cases"]
    by_id = {o["id"]: o for o in observations}
    if len(by_id) != len(observations):
        raise ValueError("Duplicate listening observation.")
    verifications = [
        verify(resolve_local(ROOT, p)) for p in config["review_directories"]
    ]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(observations_path, output / "observations.json")
    qualifications, dispositions = [], []
    for spec in config["qualifications"]:
        path = resolve_local(ROOT, spec["evidence"])
        evidence = read(path)
        observation = by_id[spec["id"]]
        bind_observation(observation, evidence, digest(path))
        if spec.get("withheld_reason"):
            dispositions.append(
                {
                    "id": spec["id"],
                    "status": "withheld",
                    "reason": spec["withheld_reason"],
                    "review_observation": observation,
                    "evidence_sha256": digest(path),
                }
            )
            continue
        # Case-level acceptance does not select a preferred alternate duration.
        render = next(r for r in evidence["renders"] if r["id"] == "proposed")
        conflicts = render["annotation_conflicts"]
        if evidence["action"] == "delete_span":
            conflicts = [
                a
                for a in conflicts
                if not (
                    a["tag"] == "w"
                    and a["start"] < a["end"]
                    and any(
                        i.get("text") == a["text"]
                        and abs(i["sourceStart"] - a["start"]) < 1e-7
                        and abs(i["sourceStart"] + i["length"] - a["end"]) < 1e-7
                        for i in evidence["selected_items"]
                    )
                )
            ]
        if conflicts:
            raise ValueError("Accepted cut still intersects a native event.")
        qualifications.append(
            {
                "id": spec["id"],
                "source_id": evidence["source_id"],
                "source_group_id": evidence["source_group_id"],
                "split": evidence["split"],
                "source_sha256": evidence["source_sha256"],
                "items": evidence["selected_items"],
                "action": evidence["action"],
                "keep_ms": render["keep_ms"],
                "removed_source_interval": render["removed_source_interval"],
                "render_directory": str(
                    path.parent.parent.relative_to(ROOT) / render["folder"]
                ),
                "preview_origin_seconds": evidence["preview_origin_seconds"],
                "evidence": spec["evidence"],
                "evidence_sha256": digest(path),
                "review_observation": observation,
                "scope": spec["scope"],
                "label_origin": "regenerated_ai_recipe_not_copied_human_label",
                "agent_listened": False,
                "human_listened": True,
                "training_eligible": False,
            }
        )
    if {q["id"] for q in qualifications + dispositions} != set(by_id):
        raise ValueError("Every observation needs an explicit disposition.")
    result = {
        "version": "reviewed-audio-v1",
        "qualifications": qualifications,
        "withheld": dispositions,
        "prior_render_verification": verifications,
        "scope": "Only listed operations and parameters; no blanket acoustic certificate.",
        "training_eligible": False,
        "config": str(config_path.relative_to(ROOT)),
        "input_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [config_path, observations_path, Path(__file__)]
        },
    }
    write(output / "qualifications.json", result)
    return {"qualified_operations": len(qualifications), "training_eligible": False}


def load_qualifications(path):
    data = read(path)
    for name, expected in data["input_sha256"].items():
        if digest(resolve_local(ROOT, name)) != expected:
            raise ValueError(f"Qualification input changed: {name}")
    for verification in data["prior_render_verification"]:
        folder = resolve_local(ROOT, verification["directory"])
        if digest(folder / "manifest.json") != verification["manifest_sha256"]:
            raise ValueError("Parent review manifest changed.")
        from audio_review import check_files

        check_files(folder, read(folder / "manifest.json"))
    for proof in data["qualifications"]:
        evidence = resolve_local(ROOT, proof["evidence"])
        original = read(evidence)
        bind_observation(proof["review_observation"], original, digest(evidence))
        render = next(r for r in original["renders"] if r["id"] == "proposed")
        expected = {
            "id": original["id"],
            "source_id": original["source_id"],
            "source_group_id": original["source_group_id"],
            "split": original["split"],
            "source_sha256": original["source_sha256"],
            "items": original["selected_items"],
            "action": original["action"],
            "keep_ms": render["keep_ms"],
            "removed_source_interval": render["removed_source_interval"],
            "preview_origin_seconds": original["preview_origin_seconds"],
            "render_directory": str(
                evidence.parent.parent.relative_to(ROOT) / render["folder"]
            ),
            "evidence_sha256": digest(evidence),
        }
        if any(proof.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Qualification broadens or changes the reviewed operation."
            )
    config = read(resolve_local(ROOT, data["config"]))
    expected_ids = {
        s["id"] for s in config["qualifications"] if not s.get("withheld_reason")
    }
    if (
        len(data["qualifications"]) != len(expected_ids)
        or {p["id"] for p in data["qualifications"]} != expected_ids
    ):
        raise ValueError(
            "Qualification includes a withheld, missing or duplicate candidate."
        )
    return data["qualifications"]


def attach_qualifications(sources, qualifications):
    for proof in qualifications:
        source = sources[proof["source_id"]]
        if (
            source["video_sha256"] != proof["source_sha256"]
            or source["source_group_id"] != proof["source_group_id"]
            or source["reserved_split"] != proof["split"]
        ):
            raise ValueError(
                "Reviewed source or split differs from the release source."
            )
        source.setdefault("audio_qualifications", []).append(proof)
        by_id = {i["uuid"]: i for i in source["document"]["content"]}
        for item in proof["items"]:
            if by_id.get(item["uuid"]) != item:
                raise ValueError("Reviewed native item changed.")
            source.setdefault("boundary_evidence", {})[item["uuid"]] = {
                "item_hash": session_hash(item),
                "boundary_status": "verified",
                "audio_class": "speech"
                if item["type"] == "text"
                else "verified_silence",
                "qualification_id": proof["id"],
                "scope": "Release references restricted to the exact reviewed operation.",
            }


def removed_interval(document, mapping, action):
    selected = [document["content"][mapping[str(t)]["index"]] for t in action["target"]]
    start = selected[0]["sourceStart"]
    end = selected[-1]["sourceStart"] + selected[-1]["length"]
    if action["action"] == "trim_silence":
        half = action["keep_ms"] / 2000
        start, end = start + half, end - half
    return selected, [start, end]


def expected_media_after_cut(document, source_id, interval):
    start, end = interval
    result = []
    for source, left, length in media_intervals(document):
        right = left + length
        if source != source_id or right <= start or left >= end:
            result.append([source, left, length])
            continue
        if left < start:
            result.append([source, left, start - left])
        if right > end:
            result.append([source, end, right - end])
    return [[s, round(t, 6), round(d, 6)] for s, t, d in result]


def verify_media_edit(before, after, mapping, action, qualifications):
    if action["action"] not in CUTS:
        if media_intervals(before) != media_intervals(after):
            raise ValueError("Non-cut action changed original media intervals.")
        return None
    selected, interval = removed_interval(before, mapping, action)
    proof = next(
        (
            p
            for p in qualifications
            if p["action"] == action["action"]
            and p["keep_ms"] == action.get("keep_ms")
            and p["items"] == selected
            and all(
                abs(a - b) < 1e-7
                for a, b in zip(interval, p["removed_source_interval"])
            )
        ),
        None,
    )
    if proof is None:
        raise ValueError(
            "Cut lacks an exact reviewed operation and source-item binding."
        )
    actual = media_intervals(after)
    expected = expected_media_after_cut(before, selected[0]["source"], interval)
    # Aggregation is rounded to microseconds; two independently summed runs can differ by one.
    if len(actual) != len(expected) or any(
        a[0] != e[0] or any(abs(x - y) > 1.1e-6 for x, y in zip(a[1:], e[1:]))
        for a, e in zip(actual, expected)
    ):
        raise ValueError("Cut changed media outside its reviewed removal interval.")
    return copy.deepcopy(proof)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(regenerate(args.config.resolve(), args.output.resolve()), indent=2)
    )
