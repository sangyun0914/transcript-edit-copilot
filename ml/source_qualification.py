"""Qualify purpose-selected real-source references without inventing word timings."""

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import wave
from array import array
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from context_policy import ContextBuilder
from grading import transcript_media
from iteration import jsonl
from model_input import messages, token_ids
from replay_session import replay, session_hash
from session_cases import apply_edit, document_records, load_project, replayable_session
from source_windows import caption_cues
from train_adapter import training_row

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "ml/fixtures/source-qualification-v1.json"
TOKENIZER = ROOT / "artifacts/iterations/colab-l4-run-04/tokenizer"
EVIDENCE = ROOT / "artifacts/source-qualification-v1/evidence"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def cue_document(source, path):
    cues = caption_cues(path.read_text())
    content, speaker = [], None

    def uid(value):
        return str(uuid5(NAMESPACE_URL, f"{source['id']}/{value}"))

    for index, cue in enumerate(cues):
        if cue["end"] <= cue["start"]:
            raise ValueError("Nonpositive publisher cue.")
        if index and cues[index - 1]["end"] > cue["start"] + 1e-6:
            raise ValueError("Overlapping cues need a different import representation.")
        label = re.match(r"^([A-Z][A-Z ]+):", cue["text"])
        if label:
            speaker = label[1]
        content.append(
            {
                "type": "paragraph_start",
                "uuid": uid(f"p{index}"),
                "speaker": speaker,
                "language": source["language"],
            }
        )
        content.append(
            {
                "type": "text",
                "uuid": uid(f"c{index}"),
                "text": cue["text"],
                "source": source["video_sha256"],
                "sourceStart": cue["start"],
                "length": cue["end"] - cue["start"],
                "conf": 1,
            }
        )
        if index + 1 < len(cues) and cues[index + 1]["start"] - cue["end"] > 1e-6:
            content.append(
                {
                    "type": "non_text",
                    "uuid": uid(f"g{index}"),
                    "source": source["video_sha256"],
                    "sourceStart": cue["end"],
                    "length": cues[index + 1]["start"] - cue["end"],
                }
            )
        content.append({"type": "paragraph_end", "uuid": uid(f"e{index}")})
    return {"content": content, "metadata": {}}


def sources_for(recipe):
    windows_path = ROOT / recipe["windows"]
    windows = [json.loads(line) for line in windows_path.read_text().splitlines()]
    local = {
        row["id"]: row
        for row in json.loads(
            (ROOT / "data/manifests/local-iteration-03-sources.json").read_text()
        )["sources"]
    }
    sources = {}
    for window in windows:
        key = window["source_id"]
        if window["source_group_id"] in recipe["reserved_test_groups"]:
            raise ValueError("Reserved test material cannot enter qualification.")
        if window["reserved_split"] not in {"train", "dev"}:
            raise ValueError("Qualification is restricted to exposed train/dev.")
        if key in sources:
            if sources[key]["reserved_split"] != window["reserved_split"]:
                raise ValueError("Source crosses splits.")
            continue
        video = Path(window["source_video"])
        if digest(video) != window["video_sha256"]:
            raise ValueError("Source media differs from its acquisition record.")
        source = {**window, "id": key, "video": video}
        if key in local:
            path = ROOT / local[key]["project"]
            document = load_project(path)
            document["metadata"] = {}
            itemization = "original_scribe_items_and_paragraphs_no_retiming"
            speaker_origin = "original_scribe_diarization_unverified"
        else:
            path = ROOT / "data/raw/ocw" / key / "captions.vtt"
            document = cue_document(source, path)
            itemization = "one_publisher_cue_per_text_item_plus_actual_intercue_gaps"
            speaker_origin = (
                "explicit_publisher_speaker_labels_carried_until_next_label"
            )
        if digest(path) != window["transcript_sha256"]:
            raise ValueError("Source transcript differs from its acquisition record.")
        records, mapping = document_records(document)
        sources[key] = {
            **source,
            "document": document,
            "records": records,
            "mapping": mapping,
            "transcript_path": path,
            "itemization": itemization,
            "speaker_origin": speaker_origin,
        }
    return sources, windows


def energy_summary(path):
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
            1,
            2,
            16000,
        ):
            raise ValueError("Expected prepared 16 kHz mono signed-16-bit PCM.")
        samples = array("h", audio.readframes(audio.getnframes()))
    longest, current, low = 0, 0, 0
    for start in range(0, len(samples), 320):
        block = samples[start : start + 320]
        rms = math.sqrt(sum(x * x for x in block) / len(block)) / 32768
        if rms < 10 ** (-45 / 20):
            current += len(block)
            low += len(block)
            longest = max(longest, current)
        else:
            current = 0
    return {
        "threshold_dbfs": -45,
        "block_ms": 20,
        "low_energy_percent": round(low / len(samples) * 100, 2),
        "longest_low_energy_seconds": round(longest / 16000, 3),
        "qualification": "candidate_signal_only_not_silence_or_safe_cut_evidence",
    }


def triage(recipe, sources, windows):
    rows = []
    for window in windows:
        index = int(window["id"].rsplit("-", 1)[-1]) - 1
        choice = recipe["window_decisions"][window["source_id"]][index]
        source = sources[window["source_id"]]
        items = [i for i in source["document"]["content"] if "sourceStart" in i]
        start, end = window["start_seconds"], window["end_seconds"]
        visible = [
            i
            for i in items
            if i["sourceStart"] < end and i["sourceStart"] + i["length"] > start
        ]
        audio = (ROOT / recipe["windows"]).parent / window["audio_preview"]
        if digest(audio) != window["audio_sha256"]:
            raise ValueError("Prepared audio preview changed.")
        gaps = [i["length"] for i in visible if i["type"] == "non_text"]
        rows.append(
            {
                "id": window["id"],
                "source_id": source["id"],
                "source_group_id": source["source_group_id"],
                "split": source["reserved_split"],
                "language": source["language"],
                "start_seconds": start,
                "end_seconds": end,
                **choice,
                "timed_items": len(visible),
                "text_items": sum(i["type"] == "text" for i in visible),
                "whitespace_units": sum(
                    len(i.get("text", "").split()) for i in visible
                ),
                "edge_crossing_items": sum(
                    i["sourceStart"] < start or i["sourceStart"] + i["length"] > end
                    for i in visible
                ),
                "gaps_ge_3_seconds": sum(x >= 3 for x in gaps),
                "largest_gap_seconds": round(max(gaps, default=0), 3),
                "audio_signal": energy_summary(audio),
                "selected_cases": [
                    c["id"]
                    for c in recipe["cases"]
                    if c["source_id"] == source["id"] and c["window"] == index + 1
                ],
                "itemization": source["itemization"],
                "audio_preview": relative(audio),
                "audio_sha256": window["audio_sha256"],
                "cut_boundary_status": "unverified",
                "project_use_status": "user_authorized",
            }
        )
    return rows


def media_intervals(document):
    result = []
    for item in document["content"]:
        if "sourceStart" not in item:
            continue
        segment = [item["source"], item["sourceStart"], item["length"]]
        if (
            result
            and result[-1][0] == segment[0]
            and abs(result[-1][1] + result[-1][2] - segment[1]) < 1e-7
        ):
            result[-1][2] += segment[2]
        else:
            result.append(segment)
    return [
        [source, round(start, 6), round(length, 6)] for source, start, length in result
    ]


def caption_summary(document):
    return [
        {
            "text": p["text"],
            "speaker": p["speaker"],
            "language": p["language"],
            "duration_seconds": round(sum(s[2] for s in p["media"]), 3),
        }
        for p in transcript_media(document)["paragraphs"]
    ]


def focus_at(timed, seconds):
    starts = [i for i in timed if abs(i["sourceStart"] - seconds) < 1e-6]
    if starts:
        return starts[0]
    inside = [
        i for i in timed if i["sourceStart"] <= seconds < i["sourceStart"] + i["length"]
    ]
    return (
        inside[0]
        if inside
        else min(timed, key=lambda i: abs(i["sourceStart"] - seconds))
    )


def build(recipe_path, tokenizer, output):
    recipe = json.loads(recipe_path.read_text())
    if (output / "manifest.json").exists():
        raise ValueError("Frozen qualification output exists; choose a new version.")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("sessions", "audio", "evidence"):
        (output / name).mkdir(exist_ok=True)
    sources, windows = sources_for(recipe)
    triaged = triage(recipe, sources, windows)
    prompt = (ROOT / "ml/prompt-v3.txt").read_text()
    builder = ContextBuilder(tokenizer, prompt)
    cases, contexts, executions = [], [], []
    keys, ids = set(), set()
    for spec in recipe["cases"]:
        source = sources[spec["source_id"]]
        document = source["document"]
        timed = [i for i in document["content"] if "sourceStart" in i]
        window = next(
            w
            for w in windows
            if w["source_id"] == source["id"]
            and w["id"].endswith(f"-{spec['window']:02d}")
        )
        if not window["start_seconds"] <= spec["focus_seconds"] < window["end_seconds"]:
            raise ValueError(
                f"{spec['id']}: cursor must be inside nominated source window."
            )
        focus = focus_at(timed, spec["focus_seconds"])
        bundle = builder.build(document, focus["uuid"], spec["brief"], as_of=0)
        selected = []
        expected = {"action": spec["action"]}
        if expected["action"] != "no_action":
            selected = [
                i for i in timed if abs(i["sourceStart"] - spec["target_start"]) < 1e-6
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"{spec['id']}: target start must match an original item."
                )
            if "target_end" in spec:
                selected = [
                    i
                    for i in timed
                    if i["sourceStart"] >= spec["target_start"] - 1e-6
                    and i["sourceStart"] < spec["target_end"] - 1e-6
                ]
                if (
                    abs(
                        selected[-1]["sourceStart"]
                        + selected[-1]["length"]
                        - spec["target_end"]
                    )
                    > 1e-6
                ):
                    raise ValueError(
                        "Target end must match an original item; no fractional-word cuts."
                    )
            aliases = {v["uuid"]: int(k) for k, v in bundle["mapping"].items()}
            expected["target"] = [aliases[i["uuid"]] for i in selected]
            if "text" in spec:
                expected["text"] = spec["text"]
        alternatives = []
        for alternative in spec.get("alternative_targets", []):
            target = next(
                i
                for i in timed
                if abs(i["sourceStart"] - alternative["start_seconds"]) < 1e-6
            )
            alias = next(
                int(k)
                for k, v in bundle["mapping"].items()
                if v["uuid"] == target["uuid"]
            )
            edit = {
                "action": "correct_text",
                "target": [alias],
                "text": alternative["text"],
            }
            if media_intervals(document) != media_intervals(
                apply_edit(document, bundle, edit)
            ):
                raise ValueError("Alternative changed media.")
            alternatives.append(edit)
        after = apply_edit(document, bundle, expected)
        if media_intervals(document) != media_intervals(after):
            raise ValueError(
                "This collection permits only media-preserving references."
            )
        if expected["action"] in {"split_paragraph", "merge_paragraph"}:
            before_words = [i for i in document["content"] if "sourceStart" in i]
            after_words = [i for i in after["content"] if "sourceStart" in i]
            if before_words != after_words:
                raise ValueError("Caption operation changed timed content.")
        evidence = []
        for item in spec.get("evidence", []):
            path = EVIDENCE / item["file"]
            shutil.copyfile(path, output / "evidence" / path.name)
            evidence.append(
                {
                    **item,
                    "file": f"evidence/{path.name}",
                    "sha256": digest(path),
                    "origin": "agent_inspected_source_frame",
                    "predictor_visibility": "observation_in_task_brief_not_image_input",
                }
            )
        preview_start = max(
            0,
            min(
                spec["focus_seconds"] - 6,
                spec.get("target_start", spec["focus_seconds"]) - 2,
            ),
        )
        preview_end = max(
            spec["focus_seconds"] + 8, spec.get("target_end", spec["focus_seconds"]) + 2
        )
        preview = output / "audio" / f"{spec['id']}.wav"
        if not preview.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-ss",
                    str(preview_start),
                    "-i",
                    str(source["video"]),
                    "-t",
                    str(preview_end - preview_start),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(preview),
                ],
                check=True,
            )
        visible = [document["content"][v["index"]] for v in bundle["mapping"].values()]
        context = {
            **bundle["context"],
            "text_items": sum(i["type"] == "text" for i in visible),
            "whitespace_units": sum(len(i.get("text", "").split()) for i in visible),
            "complete_paragraphs": len(bundle["input"]["complete_paragraphs"]),
            "source_start_seconds": min(i["sourceStart"] for i in visible),
            "source_end_seconds": max(i["sourceStart"] + i["length"] for i in visible),
            "itemization": source["itemization"],
        }
        case = {
            "id": spec["id"],
            "input": bundle["input"],
            "expectedOutput": expected,
            "acceptableOutputs": alternatives,
            "metadata": {
                "split": source["reserved_split"],
                "language": source["language"],
                "source_group_id": source["source_group_id"],
                "parent_source_id": source["id"],
                "scenario_family": spec["scenario_family"],
                "scenario_variant": spec["id"].removeprefix("source-"),
                "purpose": spec["purpose"],
                "decision_factor": spec["decision_factor"],
                "adds_coverage": spec["adds_coverage"],
                "template_id": spec["id"],
                "actor": "synthetic",
                "target_origin": "ai_generated_recipe",
                "content_origin": source["transcript_origin"],
                "timing_origin": source["itemization"],
                "history_origin": "none_no_fabricated_user_history",
                "context_origin": "shared_cursor_policy_v1",
                "interactive_state_origin": "purpose_and_cursor_authored_by_agent_on_unchanged_source_document",
                "training_eligible": source["reserved_split"] == "train",
                "requires_regeneration": False,
                "review_status": "agent_validated_observable_rule",
                "reference_scope": "AI-authored purpose-conditioned reference; no human judgment or acoustic-cut qualification claim.",
                "boundary_status": "unverified",
                "source_window": window["id"],
                "source_context": context,
                "source_video": relative(source["video"]),
                "video_sha256": source["video_sha256"],
                "transcript_sha256": source["transcript_sha256"],
                "source_page": source.get("source_page"),
                "creator": source.get("creator"),
                "license": source["license"],
                "license_url": source.get("license_url"),
                "project_use_authorization": recipe["authorization"],
                "speaker_origin": source["speaker_origin"],
                "evidence": evidence,
                "adaptations": "Authored task/cursor/reference; source text and time intervals unchanged before reference application. Local ASR metadata omitted from editor state; MIT cue segmentation imported explicitly.",
                "preview": {
                    "file": f"audio/{preview.name}",
                    "sha256": digest(preview),
                    "source_start_seconds": preview_start,
                    "source_end_seconds": preview_end,
                    "scope": "Local listening excerpt, not the full model context and not independently listened acoustic validation.",
                },
                "session": f"sessions/{spec['id']}.jsonl",
            },
        }
        key = session_hash(
            {"messages": messages(bundle["input"], prompt), "target": expected}
        )
        if key in keys or spec["id"] in ids:
            raise ValueError("Duplicate complete messages/reference or case ID.")
        keys.add(key)
        ids.add(spec["id"])
        events = replayable_session(
            document,
            after,
            expected["action"],
            {source["id"]: {"video_sha256": source["video_sha256"]}},
        )
        restored, _ = replay(events)
        if restored != after:
            raise ValueError("Apply/undo/redo replay mismatch.")
        jsonl(output / case["metadata"]["session"], events)
        sequence = training_row(tokenizer, case, prompt)
        prefix = token_ids(tokenizer, bundle["input"], prompt)
        if sequence["input_ids"][: len(prefix)] != prefix or len(prefix) + 256 > 4096:
            raise ValueError("Training/inference prefix or context budget mismatch.")
        msg = messages(bundle["input"], prompt)
        contexts.append(
            {
                "case_id": spec["id"],
                "split": source["reserved_split"],
                "messages": msg,
                "messages_sha256": hashlib.sha256(
                    json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode()
                ).hexdigest(),
                "input_ids": prefix,
                "token_count": len(prefix),
                "rendered_prompt": tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                **(
                    {"training_sequence": sequence}
                    if source["reserved_split"] == "train"
                    else {}
                ),
            }
        )
        executions.append(
            {
                "id": spec["id"],
                "before": document,
                "mapping": bundle["mapping"],
                "after_hash": session_hash(after),
                "caption_changes": {
                    "before": caption_summary(document),
                    "after": caption_summary(after),
                }
                if expected["action"] in {"split_paragraph", "merge_paragraph"}
                else None,
            }
        )
        cases.append(case)
        print(
            json.dumps(
                {
                    "id": spec["id"],
                    "action": expected["action"],
                    "items": context["visible_items"],
                    "text_items": context["text_items"],
                    "whitespace_units": context["whitespace_units"],
                    "tokens": len(prefix),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for split in ("train", "dev"):
        jsonl(
            output / f"{split}-cases.jsonl",
            [c for c in cases if c["metadata"]["split"] == split],
        )
    jsonl(output / "contexts.jsonl", contexts)
    jsonl(output / "executions.jsonl", executions)
    jsonl(output / "triage.jsonl", triaged)
    (output / "prompt.txt").write_text(prompt)
    shutil.copyfile(recipe_path, output / "recipe.json")
    shutil.copyfile(ROOT / recipe["authorization"], output / "source-use.json")
    report = {
        "version": recipe["version"],
        "status": "reference_collection_prepared_not_trained",
        "windows_triaged": len(triaged),
        "dispositions": dict(Counter(r["decision"] for r in triaged)),
        "references": len(cases),
        "unique_message_target_pairs": len(keys),
        "splits": dict(Counter(c["metadata"]["split"] for c in cases)),
        "actions": dict(Counter(c["expectedOutput"]["action"] for c in cases)),
        "families": dict(Counter(c["metadata"]["scenario_family"] for c in cases)),
        "family_requirements": recipe["families"],
        "sampling": recipe["sampling"],
        "qualified_audio_cut_references": 0,
        "test_contents_accessed": False,
        "checks": [
            "31 original media/caption/window files remain hashed by the prior audit; source and selected preview hashes rechecked",
            "source groups retained in train/dev",
            "shared production context builder and saved tokenizer",
            "exact training/inference prefix",
            "deduplicated full messages plus targets",
            "reference contract validation",
            "apply/undo/redo replay",
            "all reference media intervals preserved",
            "caption references preserve every timed item",
        ],
        "sources": [
            {
                "id": s["id"],
                "source_group_id": s["source_group_id"],
                "split": s["reserved_split"],
                "items": len(s["records"]),
                "paragraphs": len({i["paragraph"] for i in s["records"]}),
                "largest_paragraph_items": max(
                    Counter(i["paragraph"] for i in s["records"]).values()
                ),
                "itemization": s["itemization"],
                "video_sha256": s["video_sha256"],
                "transcript_sha256": s["transcript_sha256"],
            }
            for s in sources.values()
        ],
        "context_ranges": {
            key: [
                min(c["metadata"]["source_context"][key] for c in cases),
                max(c["metadata"]["source_context"][key] for c in cases),
            ]
            for key in (
                "visible_items",
                "text_items",
                "whitespace_units",
                "input_tokens",
                "complete_paragraphs",
            )
        },
        "inputs_sha256": {
            relative(p): digest(p)
            for p in [
                recipe_path,
                ROOT / recipe["authorization"],
                ROOT / recipe["windows"],
                ROOT / "ml/prompt-v3.txt",
                ROOT / "ml/source_qualification.py",
                *TOKENIZER.iterdir(),
            ]
            if p.is_file()
        },
    }
    write(output / "qualification-report.json", report)
    write(
        output / "manifest.json",
        {
            **report,
            "files_sha256": {
                str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/iterations/source-qualification-v1",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    result = build(args.recipe.resolve(), tokenizer, args.output.resolve())
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in {
                    "references",
                    "splits",
                    "actions",
                    "context_ranges",
                    "windows_triaged",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
