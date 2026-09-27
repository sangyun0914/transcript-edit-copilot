"""Independently audit every accepted release reference through actual Audapolis export."""

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from dataset_release_inventory import ROOT, digest, read, rows, write
from replay_session import replay, session_hash
from session_cases import apply_edit

TIMED = {"text", "non_text", "artificial_silence"}
CUTS = {"delete_span", "trim_silence", "remove_silence"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-8)


def timed(document):
    return [item for item in document["content"] if item["type"] in TIMED]


def duration(document):
    return math.fsum(item["length"] for item in timed(document))


def verify_operation(before, after, mapping, action):
    """Check the intended mutation and untouched prefix/suffix without using executor output as truth."""
    name = action["action"]
    require(
        before["metadata"] == after["metadata"], "An edit changed document metadata."
    )
    if name == "no_action":
        require(before == after, "Abstention changed the document.")
        return {
            "removed_seconds": 0.0,
            "affected_paragraphs": [],
            "target_end_seconds": 0.0,
        }
    content, result = before["content"], after["content"]
    indices = [mapping[str(alias)]["index"] for alias in action["target"]]
    first, last = indices[0], indices[-1]
    selected = content[first : last + 1]
    affected, current, target_end = [], None, 0.0
    for index, item in enumerate(content):
        if item["type"] == "paragraph_start":
            current = item["uuid"]
        if first <= index <= last and current not in affected:
            affected.append(current)
        if item["type"] in TIMED and index <= last:
            target_end += item["length"]
    removed = 0.0
    if name in {"delete_span", "remove_silence"}:
        require(
            result == content[:first] + content[last + 1 :],
            "Deletion changed content outside its exact span.",
        )
        removed = math.fsum(item["length"] for item in selected)
    elif name == "trim_silence":
        require(first == last, "Trim spans multiple items.")
        require(
            result[:first] == content[:first]
            and result[first + 2 :] == content[first + 1 :],
            "Trim changed surrounding items.",
        )
        left, right = result[first : first + 2]
        original = selected[0]
        retained = action["keep_ms"] / 1000
        for piece in (left, right):
            require(
                piece["type"] == original["type"]
                and piece.get("source") == original.get("source"),
                "Trim changed the original source or item type.",
            )
            require(
                close(piece["length"], retained / 2),
                "Trim did not keep both outside edges equally.",
            )
            require(
                {
                    k: v
                    for k, v in piece.items()
                    if k not in {"uuid", "length", "sourceStart"}
                }
                == {
                    k: v
                    for k, v in original.items()
                    if k not in {"uuid", "length", "sourceStart"}
                },
                "Trim changed non-timing item fields.",
            )
        require(
            left["uuid"] == original["uuid"] and right["uuid"] != original["uuid"],
            "Trim lost or duplicated item identity.",
        )
        require(
            left["sourceStart"] == original["sourceStart"]
            and close(
                right["sourceStart"] + right["length"],
                original["sourceStart"] + original["length"],
            ),
            "Trim moved an outer source boundary.",
        )
        removed = original["length"] - retained
    elif name == "correct_text":
        require(
            result[:first] == content[:first]
            and result[first + 1 :] == content[last + 1 :],
            "Correction changed content outside its target.",
        )
        replacement = result[first]
        require(
            replacement["type"] == "text" and replacement["text"] == action["text"],
            "Correction display text differs from its reference.",
        )
        require(
            replacement["uuid"] == selected[0]["uuid"]
            and replacement["source"] == selected[0]["source"]
            and replacement["sourceStart"] == selected[0]["sourceStart"],
            "Correction lost its original source identity.",
        )
        require(
            close(
                replacement["length"], math.fsum(item["length"] for item in selected)
            ),
            "Text-only correction changed audio duration.",
        )
        require(
            all(item["type"] in TIMED for item in selected),
            "Correction implicitly deleted a paragraph marker.",
        )
        for left, right in zip(selected, selected[1:]):
            require(
                left["source"] == right["source"]
                and close(left["sourceStart"] + left["length"], right["sourceStart"]),
                "Correction crossed an immutable source discontinuity.",
            )
    elif name == "split_paragraph":
        require(
            result[:first] == content[:first]
            and result[first + 2 :] == content[first:],
            "Split changed text, audio, or another boundary.",
        )
        marker = next(
            item
            for item in reversed(content[:first])
            if item["type"] == "paragraph_start"
        )
        require(
            result[first]["type"] == "paragraph_end"
            and result[first + 1]["type"] == "paragraph_start",
            "Split did not create a balanced boundary.",
        )
        require(
            {k: v for k, v in result[first + 1].items() if k != "uuid"}
            == {k: v for k, v in marker.items() if k != "uuid"},
            "Split changed paragraph speaker/language.",
        )
    elif name == "merge_paragraph":
        require(
            [item["type"] for item in content[first - 2 : first]]
            == ["paragraph_end", "paragraph_start"],
            "Merge target is not at a paragraph boundary.",
        )
        require(
            result == content[: first - 2] + content[first:],
            "Merge changed items outside its exact boundary.",
        )
        previous = next(
            item["uuid"]
            for item in reversed(content[: first - 2])
            if item["type"] == "paragraph_start"
        )
        affected.append(previous)
    else:
        raise ValueError("Unknown reference action.")
    require(
        close(duration(before) - duration(after), removed),
        "Executed timeline change differs from the exact source interval removal.",
    )
    if name in {"split_paragraph", "merge_paragraph"}:
        require(
            timed(before) == timed(after), "Caption grouping modified a timed item."
        )
    return {
        "removed_seconds": removed,
        "affected_paragraphs": affected,
        "target_end_seconds": target_end,
    }


class Exporter:
    def __init__(self):
        self.process = subprocess.Popen(
            ["node", str(Path(__file__).with_name("export_release_states.cjs"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.cache = {}

    def export(self, document, line_length):
        key = (session_hash(document), line_length)
        if key not in self.cache:
            self.process.stdin.write(
                json.dumps(
                    {"document": document, "line_length": line_length},
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.process.stdin.flush()
            value = self.process.stdout.readline()
            if not value:
                raise RuntimeError("Actual caption exporter stopped unexpectedly.")
            result = json.loads(value)
            require(
                result["status"] == "passed",
                result.get("error", "Caption export failed."),
            )
            self.cache[key] = result
        return self.cache[key]

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()
        error = self.process.stderr.read()
        self.process.stderr.close()
        require(self.process.returncode == 0, f"Caption helper failed: {error}")


def verify_export_delta(before, after, operation, action):
    if action == "no_action":
        require(
            before["export_sha256"] == after["export_sha256"],
            "Abstention changed exported VTT bytes.",
        )
    previous = {(cue["paragraph_id"], cue["segment"]): cue for cue in before["cues"]}
    new_empty, new_zero = [], []
    unchanged = 0
    max_quantization_drift = 0.0
    for cue in after["cues"]:
        key = (cue["paragraph_id"], cue["segment"])
        original = previous.get(key)
        if cue["empty"] and (original is None or not original["empty"]):
            new_empty.append(key)
        if cue["zero_duration"] and (original is None or not original["zero_duration"]):
            new_zero.append(key)
        if original is None or cue["paragraph_id"] in operation["affected_paragraphs"]:
            continue
        require(
            cue["timed_ids"] == original["timed_ids"]
            and cue["payload_sha256"] == original["payload_sha256"]
            and cue["text_sha256"] == original["text_sha256"],
            "Export changed an utterance outside the intended operation.",
        )
        shift = (
            operation["removed_seconds"]
            if original["start_seconds"] >= operation["target_end_seconds"] - 1e-8
            else 0.0
        )
        for edge in ("start", "end"):
            require(
                close(cue[f"{edge}_seconds"], original[f"{edge}_seconds"] - shift),
                "Unexpected timeline movement outside the target.",
            )
            drift = abs(cue[f"{edge}_ms"] - original[f"{edge}_ms"] + shift * 1000)
            require(drift <= 1.00001, "Export rounding drift exceeded one millisecond.")
            max_quantization_drift = max(max_quantization_drift, drift)
        unchanged += 1
    require(not new_empty, f"Operation introduced empty exported cues: {new_empty}")
    require(
        not new_zero, f"Operation introduced zero-duration exported cues: {new_zero}"
    )
    require(
        close(
            before["duration_seconds"] - after["duration_seconds"],
            operation["removed_seconds"],
        ),
        "Actual export clock does not reflect source interval removal.",
    )
    return {
        "before_cues": len(before["cues"]),
        "after_cues": len(after["cues"]),
        "baseline_empty_cues": sum(c["empty"] for c in before["cues"]),
        "retained_empty_cues": sum(c["empty"] for c in after["cues"]),
        "baseline_zero_duration_cues": sum(c["zero_duration"] for c in before["cues"]),
        "introduced_empty_cues": 0,
        "introduced_zero_duration_cues": 0,
        "untouched_cues_checked": unchanged,
        "max_outside_target_timestamp_quantization_drift_ms": round(
            max_quantization_drift, 9
        ),
        "removed_source_seconds": operation["removed_seconds"],
        "before_export_end_ms": before["exported_end_ms"],
        "after_export_end_ms": after["exported_end_ms"],
        "before_export_sha256": before["export_sha256"],
        "after_export_sha256": after["export_sha256"],
    }


def verify(directory):
    manifest_hash = digest(directory / "manifest.json")
    manifest = read(directory / "manifest.json")
    filenames = {"train-cases.jsonl", "dev-cases.jsonl", "executions.jsonl"}
    cases = rows(directory / "train-cases.jsonl") + rows(directory / "dev-cases.jsonl")
    executions = {row["id"]: row for row in rows(directory / "executions.jsonl")}
    filenames.update(case["metadata"]["session"] for case in cases)
    for filename in filenames:
        require(
            digest(directory / filename) == manifest["files_sha256"][filename],
            f"Release input hash mismatch: {filename}",
        )
    exporter = Exporter()
    results, errors = [], []
    try:
        for case in cases:
            execution = executions[case["id"]]
            before = execution["before"]
            bundle = {
                "input": case["input"],
                "mapping": execution["mapping"],
                "document_hash": session_hash(before),
                "caption_authority": execution["caption_authority"],
            }
            line_length = case["input"]["caption_profile"]["export_line_length"]
            for index, action in enumerate(
                [case["expectedOutput"], *case["acceptableOutputs"]]
            ):
                try:
                    after = apply_edit(before, bundle, action)
                    if index == 0:
                        journal = rows(directory / case["metadata"]["session"])
                        require(
                            journal[0]["document"] == before,
                            "Primary journal uses a different before-state.",
                        )
                        replayed, summary = replay(journal)
                        require(
                            summary["complete"]
                            and replayed == after
                            and session_hash(after) == execution["after_hash"],
                            "Primary journal/after hash differs from executed reference.",
                        )
                    operation = verify_operation(
                        before, after, execution["mapping"], action
                    )
                    left, right = (
                        exporter.export(before, line_length),
                        exporter.export(after, line_length),
                    )
                    result = verify_export_delta(
                        left, right, operation, action["action"]
                    )
                    results.append(
                        {
                            "case_id": case["id"],
                            "reference_index": index,
                            "reference_kind": "primary"
                            if index == 0
                            else "acceptable_alternative",
                            "action": action,
                            "status": "passed",
                            **result,
                        }
                    )
                except (ValueError, AssertionError) as error:
                    errors.append(
                        {
                            "case_id": case["id"],
                            "reference_index": index,
                            "error": str(error),
                        }
                    )
    finally:
        exporter.close()
    require(
        digest(directory / "manifest.json") == manifest_hash,
        "Release manifest changed during export audit.",
    )
    paths = [
        Path(__file__),
        Path(__file__).with_name("export_release_states.cjs"),
        ROOT / "app/src/core/webvtt.ts",
        ROOT / "app/src/state/editor/selectors.ts",
        ROOT / "app/node_modules/@audapolis/webvtt-writer/src/WebVtt.ts",
        ROOT / "app/node_modules/@audapolis/webvtt-writer/dist/WebVtt.js",
        ROOT / "app/node_modules/@audapolis/webvtt-writer/package.json",
        ROOT / "ml/session_cases.py",
        ROOT / "ml/replay_session.py",
        ROOT / "ml/model_input.py",
        ROOT / "ml/nep_v4.py",
        ROOT / "ml/caption_facts.py",
    ]
    return {
        "audit_kind": "full_caption_export",
        "status": "passed" if not errors else "failed",
        "manifest_sha256": manifest_hash,
        "release_directory": str(directory.relative_to(ROOT)),
        "primary_cases": len(cases),
        "acceptable_alternatives": sum(len(c["acceptableOutputs"]) for c in cases),
        "references_checked": len(results) + len(errors),
        "references_passed": len(results),
        "errors": errors,
        "action_counts": dict(Counter(row["action"]["action"] for row in results)),
        "unique_exported_states": len(exporter.cache),
        "case_reference_results": results,
        "export_settings": {
            "wordTimings": False,
            "includeSpeakerNames": False,
            "limitLineLength": sorted(
                {c["input"]["caption_profile"]["export_line_length"] for c in cases},
                key=str,
            ),
        },
        "verification": {
            "all_payloads": "Actual serialized VTT decoded payload equals every executed paragraph's text, including multi-speaker newlines.",
            "all_timestamps": "Exact sequential JavaScript document clock and actual writer flooring, cue ordering and complete timeline coverage.",
            "operation_isolation": "Independent exact prefix/suffix and source-edge checks for every raw reference; unaffected cue content stays byte-identical.",
            "baseline_empty_cues": "Existing non-text event cues are retained or intentionally shortened/removed; newly emptied utterance cues are rejected.",
            "floating_point_clock": "Regrouping a text span may shift a floored downstream timestamp by at most 1 ms despite identical audio intervals; such drift is measured per reference and never counted as a source edit.",
            "no_action": "Byte-identical exported VTT.",
            "perceptual_audio_video_quality": "Not inferred from export; operation-specific listening and source verification remain separate.",
        },
        "inputs_sha256": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "release_inputs_sha256": {
            name: manifest["files_sha256"][name] for name in sorted(filenames)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Keep previous audits; choose a new output path.")
    result = verify(args.directory.resolve())
    write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "audit_kind",
                    "primary_cases",
                    "acceptable_alternatives",
                    "references_checked",
                    "references_passed",
                    "errors",
                )
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)
