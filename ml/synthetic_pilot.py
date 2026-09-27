"""Build AI-authored contract fixtures with executable synthetic sessions, not human labels."""

import argparse
import copy
import hashlib
import json
import wave
import zipfile
from collections import Counter
from pathlib import Path

from nep_v2 import Envelope, Snapshot, validate_prediction
from replay_session import session_hash
from session_cases import apply_edit, replayable_session, snapshot_for

HERE = Path(__file__).resolve().parent


def document_for(spec):
    content, cursor = [], 0
    for p, words in enumerate(spec["paragraphs"]):
        content.append(
            {
                "type": "paragraph_start",
                "uuid": f"p{p}",
                "speaker": "speaker",
                "language": spec.get("language", "ko"),
            }
        )
        for word in words:
            text, duration = word["text"], word.get("ms", 400)
            item = {
                "type": "text" if text else "non_text",
                "uuid": word["id"],
                "source": "fixture",
                "sourceStart": cursor / 1000,
                "length": duration / 1000,
            }
            if text:
                item.update(text=text, conf=None)
            content.append(item)
            cursor += duration
        content.append({"type": "paragraph_end", "uuid": f"e{p}"})
    return {"content": content, "metadata": {}}


def symbolic_evidence(document):
    # For state-transition tests only; silent placeholders cannot qualify real cuts.
    return {
        i["uuid"]: {
            "item_hash": session_hash(i),
            "audio_class": "speech" if i["type"] == "text" else "verified_silence",
        }
        for i in document["content"]
        if "source" in i
    }


def execute_history(document, spec, output):
    """Execute earlier edits elsewhere in this document, keeping each replay chain."""
    history = spec.get("history", [])
    if not history:
        return document, []
    language = spec.get("language", "ko")
    english = language in ("en", "eng")
    paragraphs = []
    for index, edit in enumerate(history):
        if edit["action"] == "correct_text":
            words = [{"id": f"h{index}", "text": edit["before"]}]
        elif edit["action"] == "trim_silence":
            words = [
                {
                    "id": f"h{index}-left",
                    "text": "Previous item." if english else "이전 항목입니다.",
                },
                {"id": f"h{index}", "text": "", "ms": edit["before_ms"]},
                {
                    "id": f"h{index}-right",
                    "text": "Next item." if english else "다음 항목입니다.",
                },
            ]
        else:
            raise ValueError("Add an executable history recipe for this action.")
        paragraphs.append(words)
    extra = document_for({"paragraphs": paragraphs, "language": language})
    offset = sum(i.get("length", 0) for i in document["content"])
    for item in extra["content"]:
        if "sourceStart" in item:
            item["sourceStart"] += offset
        else:
            item["uuid"] = "history-" + item["uuid"]
    document["content"].extend(extra["content"])
    paths = []
    for index, edit in enumerate(history):
        bundle = snapshot_for(
            document,
            f"h{index}",
            spec["brief"],
            history[:index],
            symbolic_evidence(document),
        )
        target = next(
            int(k) for k, v in bundle["mapping"].items() if v["uuid"] == f"h{index}"
        )
        prediction = {"action": edit["action"], "target": [target]}
        if edit["action"] == "correct_text":
            prediction["text"] = edit["after"]
        else:
            prediction["keep_ms"] = edit["after_ms"]
        after = apply_edit(document, bundle, prediction)
        events = replayable_session(document, after, edit["action"], {})
        relative = f"sessions/{spec['id']}.history-{index + 1}.jsonl"
        (output / relative).write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
        )
        paths.append(relative)
        document = after
    return document, paths


def build(spec_path, output):
    specs = json.loads(spec_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "sessions").mkdir()
    (output / "projects").mkdir()
    cases, sessions, review_cases, media, dispositions = [], [], [], {}, []
    for spec in specs:
        document, history_paths = execute_history(document_for(spec), spec, output)
        proof = (
            symbolic_evidence(document) if spec.get("qualified_fixture", False) else {}
        )
        bundle = snapshot_for(
            document, spec["focus"], spec["brief"], spec.get("history", []), proof
        )
        aliases = {v["uuid"]: int(k) for k, v in bundle["mapping"].items()}
        expected = copy.deepcopy(spec.get("expected"))
        proposal = copy.deepcopy(spec.get("proposal", expected))
        for edit in (expected, proposal):
            if edit and "target" in edit:
                edit["target"] = [aliases[t] for t in edit["target"]]
        if spec.get("protected"):
            bundle["input"]["protected_ids"] = [aliases[t] for t in spec["protected"]]
        row = {
            "id": spec["id"],
            "input": bundle["input"],
            "expectedOutput": expected,
            "metadata": {
                "split": "calibration",
                "source_group_id": "ai-authored-contract-v2",
                "scenario_family": spec["family"],
                "content_origin": "ai_authored_symbolic_fixture",
                "history_origin": "executed_synthetic",
                "history_sessions": history_paths,
                "before_hash": session_hash(document),
                "target_origin": "ai_authored",
                "actor": "synthetic",
                "generator": "synthetic-pilot-v1",
                "generator_spec_sha256": hashlib.sha256(
                    spec_path.read_bytes()
                ).hexdigest(),
                "timing_origin": "symbolic_fixture_not_acoustic_evidence",
                "audio_quality": "not_established",
                "training_eligible": False,
                "position_s": next(
                    i["sourceStart"]
                    for i in document["content"]
                    if i["uuid"] == spec["focus"]
                ),
                "review_status": "needs_review"
                if expected is None
                else "synthetic_reference",
                "rationale": spec["reason"],
                "source_items": bundle["mapping"],
                "context_origin": "constructed_corruption"
                if spec.get("corrupted")
                else "ai_authored",
            },
        }
        if expected is not None:
            after = apply_edit(document, bundle, expected)
            events = replayable_session(document, after, expected["action"], {})
            session_path = output / "sessions" / f"{spec['id']}.jsonl"
            session_path.write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
            )
            sessions.append(
                {
                    "id": spec["id"],
                    "before": document,
                    "after": after,
                    "events": events,
                    "prediction": expected,
                    "mapping": bundle["mapping"],
                }
            )
        else:
            after = None
        cases.append(row)
        dispositions.append(
            {
                "case_id": spec["id"],
                "action": proposal["action"],
                "disposition": "needs_review"
                if expected is None
                else (
                    "abstain"
                    if expected["action"] == "no_action"
                    else "fixture_supported"
                ),
                "reason": spec["reason"],
            }
        )
        if expected is None:
            # A text-only uncertain card uses a valid silent WAV solely to exercise the workbench.
            audio_path = output / f"{spec['id']}.wav"
            duration = max(
                i["sourceStart"] + i["length"]
                for i in document["content"]
                if "source" in i
            )
            with wave.open(str(audio_path), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * round(duration * 16000))
            media[spec["id"]] = {
                "path": audio_path.name,
                "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            }
            source_items = {
                k: {
                    "uuid": v["uuid"],
                    "source": "fixture",
                    "start_s": document["content"][v["index"]]["sourceStart"],
                    "end_s": document["content"][v["index"]]["sourceStart"]
                    + document["content"][v["index"]]["length"],
                }
                for k, v in bundle["mapping"].items()
            }
            review_cases.append(
                {
                    **row,
                    "proposal": proposal,
                    "metadata": {
                        **row["metadata"],
                        "source_items": source_items,
                        "proposal_note": spec["reason"]
                        + " Symbolic fixture: no speech audio is available.",
                        "audio": {
                            "media_id": spec["id"],
                            "source": "fixture",
                            "start_s": 0,
                            "end_s": duration,
                        },
                    },
                }
            )
        if spec.get("export_project"):
            media_path = output / "projects" / f"{spec['id']}.wav"
            duration = max(
                i["sourceStart"] + i["length"]
                for i in document["content"]
                if "source" in i
            )
            with wave.open(str(media_path), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * round(duration * 16000))
            with zipfile.ZipFile(
                output / "projects" / f"{spec['id']}.audapolis", "w"
            ) as archive:
                archive.writestr(
                    "document.json",
                    json.dumps(
                        {
                            "version": 3,
                            **document,
                            "metadata": {
                                "display_video": False,
                                "display_speaker_names": True,
                            },
                        }
                    ),
                )
                archive.write(media_path, "sources/fixture")
    (output / "pilot-cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    )
    (output / "replay-fixtures.json").write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in review_cases)
    )
    manifest = {
        "schema_version": 1,
        "title": "Synthetic v2 ambiguity review",
        "note": "AI-authored symbolic fixtures; silent audio placeholders, not an acoustic benchmark. "
        "Review feedback must improve generation, followed by regeneration; do not train on exported labels.",
        "media": media,
        "counts": dict(Counter(c["proposal"]["action"] for c in review_cases)),
        "label_policy": "feedback_then_regenerate",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    report = {
        "cases": len(cases),
        "replayed_sessions": len(sessions),
        "review_queue": len(review_cases),
        "action_counts": dict(
            Counter(c["expectedOutput"]["action"] for c in cases if c["expectedOutput"])
        ),
        "scope": "unsealed symbolic calibration; no acoustic or natural-user quality claims",
        "dispositions": dispositions,
        "baselines": baseline_report(cases),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def repeated_correction(snapshot):
    for recent in reversed(snapshot["recent_edits"]):
        if (
            recent["action"] == "correct_text"
            and recent.get("before")
            and recent.get("after")
        ):
            for item in snapshot["items"]:
                if (
                    item["text"] == recent["before"]
                    and item["id"] not in snapshot["protected_ids"]
                ):
                    return {
                        "action": "correct_text",
                        "target": [item["id"]],
                        "text": recent["after"],
                    }
    return {"action": "no_action"}


def baseline_report(cases):
    results = {}
    for name, predict in (
        ("always_no_action", lambda _: {"action": "no_action"}),
        ("repeated_correction", repeated_correction),
    ):
        rows = []
        for case in cases:
            prediction = predict(case["input"])
            request = Envelope(
                document_id=case["id"],
                request_id="rule",
                revision=0,
                interaction_epoch=0,
            )
            try:
                validate_prediction(
                    json.dumps(prediction),
                    Snapshot.model_validate(case["input"]),
                    request,
                    request,
                    "stop",
                )
                valid = True
            except ValueError:
                valid = False
            rows.append(
                {
                    "id": case["id"],
                    "prediction": prediction,
                    "valid": valid,
                    "labeled": case["expectedOutput"] is not None,
                    "match": prediction == case["expectedOutput"],
                }
            )
        labeled = [r for r in rows if r["labeled"]]
        shown = [
            r for r in rows if r["valid"] and r["prediction"]["action"] != "no_action"
        ]
        results[name] = {
            "opportunities": len(rows),
            "labeled": len(labeled),
            "draft_matches": sum(r["match"] for r in labeled),
            "shown": len(shown),
            "shown_matches": sum(r["match"] for r in shown),
            "shown_unknown": sum(not r["labeled"] for r in shown),
            "rows": rows,
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=HERE / "fixtures/synthetic-v2.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.spec, args.output)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("baselines", "dispositions")},
            indent=2,
        )
    )
