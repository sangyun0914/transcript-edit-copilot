"""Mine and inspect real audio proposals without certifying unheard cuts."""

import argparse
import copy
from dataclasses import asdict
import importlib.metadata
import json
import math
import re
import subprocess
import wave
from pathlib import Path

from dataset_release_inventory import ROOT, digest, read, write
from replay_session import session_hash

REGISTRATION = ROOT / "data/manifests/audio-review-v1.json"
RATE = 16000
FRAME = 512


def save_rows(path, values):
    with path.open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_pcm(path):
    import numpy as np

    with wave.open(str(path), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (
            RATE,
            1,
            2,
        ):
            raise ValueError("Expected 16 kHz mono signed-16-bit PCM.")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")


def save_pcm(path, values):
    if path.exists():
        raise FileExistsError(path)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(values.astype("<i2").tobytes())


def extract(path, start, duration, output):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-i",
            str(path),
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
        capture_output=True,
    )


def overlaps(left, right):
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def removal_interval(start, end, action, keep_ms=None):
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        raise ValueError("Invalid source interval.")
    if action == "trim_silence":
        if type(keep_ms) is not int or not 0 < keep_ms < round((end - start) * 1000):
            raise ValueError("A trim must retain a positive, shorter interval.")
        return [start + keep_ms / 2000, end - keep_ms / 2000]
    if action not in {"remove_silence", "delete_span"}:
        raise ValueError("Unsupported audio proposal.")
    return [start, end]


def cut_pcm(samples, removal, origin):
    """Use integer sample boundaries; preserve all samples outside the proposal."""
    import numpy as np

    start, end = [round((time - origin) * RATE) for time in removal]
    if not 0 < start < end < len(samples):
        raise ValueError("Cut lacks retained context on both sides.")
    after = np.concatenate((samples[:start], samples[end:]))
    return after, samples[start:end], [start, end]


def speech_probe(probabilities, start, end, origin):
    import numpy as np

    # Include every inference frame that intersects the requested source interval.
    first = max(0, math.floor((start - origin) * RATE / FRAME))
    last = min(len(probabilities), math.ceil((end - origin) * RATE / FRAME))
    if start < origin or first >= last:
        raise ValueError("Speech probe is outside its registered audio.")
    values = probabilities[first:last]
    return {
        "max_probability": round(float(np.max(values)), 5),
        "fraction_frames_ge_035": round(float(np.mean(values >= 0.35)), 5),
        "frames": len(values),
    }


def annotation_conflicts(annotations, interval):
    result = []
    for value in annotations:
        start, end = value["start"], value["end"]
        if (
            value["tag"] == "w"
            and start == end
            and not any(c.isalnum() for c in value["text"])
        ):
            continue
        bounds = [min(start, end), max(start, end)]
        if start == end:
            bounds = [start - 0.05, end + 0.05]
        if overlaps(bounds, interval):
            result.append(value)
    return result


def sources_and_windows():
    from ami_release import FOLDER, ami_sources
    from release_sources import calculus_sources
    from source_qualification import sources_for

    config = read(REGISTRATION)
    sources, windows = sources_for(read(ROOT / config["source_recipe"]))
    for window in windows:
        window["audio_path"] = str(
            ROOT
            / "data/derived/nep/source-windows/iteration-03"
            / window["audio_preview"]
        )
    extra, ami_windows = ami_sources()
    sources.update(extra)
    for index, window in enumerate(ami_windows, 1):
        window["audio_path"] = str(
            ROOT
            / f"data/derived/nep/source-windows/ami-pilot-v1/window-{index:02d}.wav"
        )
        window["headsets"] = [
            str(FOLDER / f"ES2002a.Headset-{i}.wav") for i in range(4)
        ]
    windows.extend(ami_windows)
    extra, calculus_windows = calculus_sources()
    sources.update(extra)
    for window in calculus_windows:
        window["audio_path"] = str(
            ROOT
            / "data/derived/local-tests/calculus-30min-v1"
            / window["source_id"]
            / "input.wav"
        )
    windows.extend(calculus_windows)
    if any(
        source["reserved_split"] not in {"train", "dev"} for source in sources.values()
    ):
        raise ValueError("Only exposed train/dev source groups are permitted.")
    return sources, windows


def scan(directory):
    import numpy as np
    from faster_whisper.vad import get_vad_model
    from faster_whisper.utils import get_assets_path

    directory.mkdir(parents=True, exist_ok=False)
    config = read(REGISTRATION)
    write(directory / "registration.json", config)
    sources, windows = sources_and_windows()
    (directory / "sources").mkdir()
    (directory / "vad").mkdir()
    (directory / "channels").mkdir()
    for identity, source in sources.items():
        write(
            directory / "sources" / f"{identity}.json",
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in source.items()
                if key not in {"records", "mapping"}
            },
        )
    model = get_vad_model()
    candidates = []
    for window in windows:
        path = Path(window["audio_path"])
        samples = load_pcm(path)
        duration = len(samples) / RATE
        start, end = window["start_seconds"], window["end_seconds"]
        if abs(duration - (end - start)) > 0.003:
            raise ValueError("Audio length differs from registered source window.")
        tracks = {"mix": path}
        for index, headset in enumerate(window.get("headsets", [])):
            output = directory / "channels" / f"{window['id']}-headset-{index}.wav"
            extract(headset, start, duration, output)
            tracks[f"headset-{index}"] = output
        window["tracks"] = {}
        probabilities = {}
        for name, track in tracks.items():
            pcm = load_pcm(track).astype(np.float32) / 32768
            padded = np.pad(pcm, (0, (-len(pcm)) % FRAME))
            probability = model(padded).reshape(-1)
            target = directory / "vad" / f"{window['id']}-{name}.npy"
            np.save(target, probability, allow_pickle=False)
            probabilities[name] = probability
            window["tracks"][name] = {
                "path": str(track),
                "sha256": digest(track),
                "probabilities": str(target.relative_to(directory)),
                "probabilities_sha256": digest(target),
            }
        source = sources[window["source_id"]]
        items = [
            item for item in source["document"]["content"] if "sourceStart" in item
        ]
        for position, item in enumerate(items):
            left, right = item["sourceStart"], item["sourceStart"] + item["length"]
            if left < start + 4 or right > end - 4:
                continue
            gap = (
                item["type"] == "non_text"
                and 900 < round((right - left) * 1000) <= 8000
            )
            filler = item["type"] == "text" and re.sub(
                r"[^\w]", "", item["text"]
            ).lower() in {"um", "uh", "erm", "er", "음", "어"}
            if not (gap or filler):
                continue
            action = "trim_silence" if gap else "delete_span"
            removed = removal_interval(left, right, action, 900 if gap else None)
            probes = {
                name: speech_probe(p, *removed, start)
                for name, p in probabilities.items()
            }
            conflicts = annotation_conflicts(
                source.get("native_annotations", []), removed
            )
            candidates.append(
                {
                    "id": f"{source['id']}-{item['uuid']}",
                    "window_id": window["id"],
                    "source_id": source["id"],
                    "source_group_id": source["source_group_id"],
                    "split": source["reserved_split"],
                    "language": source["language"],
                    "source_interval": [left, right],
                    "item_uuid": item["uuid"],
                    "item_hash": session_hash(item),
                    "item_type": item["type"],
                    "text": item.get("text", ""),
                    "action": action,
                    "keep_ms": 900 if gap else None,
                    "removed_interval": removed,
                    "vad_removed": probes,
                    "annotation_conflicts": conflicts,
                    "context_before": " ".join(
                        i.get("text", "")
                        for i in items[max(0, position - 14) : position]
                    ).strip(),
                    "context_after": " ".join(
                        i.get("text", "") for i in items[position + 1 : position + 15]
                    ).strip(),
                    "status": "candidate_only_unverified",
                    "training_eligible": False,
                }
            )
        print(
            json.dumps(
                {
                    "window": window["id"],
                    "tracks": len(tracks),
                    "candidates_so_far": len(candidates),
                }
            ),
            flush=True,
        )
    save_rows(directory / "candidates.jsonl", candidates)
    write(directory / "windows.json", windows)
    model_files = {str(p): digest(p) for p in Path(get_assets_path()).glob("*.onnx")}
    write(
        directory / "manifest.json",
        {
            "version": "audio-scan-v1",
            "registration_sha256": digest(REGISTRATION),
            "code_sha256": digest(Path(__file__)),
            "windows": len(windows),
            "candidate_count": len(candidates),
            "vad_models": model_files,
            "primary_audio_seconds": sum(
                w["end_seconds"] - w["start_seconds"] for w in windows
            ),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["faster-whisper", "onnxruntime", "numpy"]
            },
            "training_eligible": False,
            "files": {
                str(p.relative_to(directory)): digest(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            },
        },
    )


def check_files(directory, manifest):
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")


def normalized_text(value):
    return re.sub(r"[\W_]", "", value.casefold())


def signal_metrics(before, after, cuts):
    import numpy as np

    left, right = cuts
    jump = abs(float(after[left]) - float(after[left - 1])) / 32768
    edges = [before[left - 320 : left], before[right : right + 320]]
    return {
        "join_jump_dbfs": round(20 * math.log10(max(jump, 1 / 32768)), 2),
        "retained_edge_rms_dbfs": [
            round(
                20
                * math.log10(
                    max(
                        float(np.sqrt(np.mean((edge.astype(np.float64) / 32768) ** 2))),
                        1 / 32768,
                    )
                ),
                2,
            )
            for edge in edges
        ],
        "interpretation": "Signal diagnostics only; low jump and energy do not prove a natural join.",
    }


def source_selection(source, spec):
    items = [item for item in source["document"]["content"] if "sourceStart" in item]
    selected = [
        item
        for item in items
        if overlaps(
            [item["sourceStart"], item["sourceStart"] + item["length"]],
            [spec["target_start"], spec["target_end"]],
        )
        > 1e-6
    ]
    if (
        not selected
        or abs(selected[0]["sourceStart"] - spec["target_start"]) > 1e-6
        or abs(
            selected[-1]["sourceStart"] + selected[-1]["length"] - spec["target_end"]
        )
        > 1e-6
    ):
        raise ValueError("Review proposal must identify exact native item boundaries.")
    if any(
        abs(a["sourceStart"] + a["length"] - b["sourceStart"]) > 1e-6
        for a, b in zip(selected, selected[1:])
    ):
        raise ValueError("Review proposal crosses a source discontinuity.")
    return selected


def contract_probe(source, spec, selected):
    from session_cases import apply_edit, snapshot_for

    bundle = snapshot_for(
        source["document"],
        selected[0]["uuid"],
        spec["brief"],
        window_size=64,
        evidence=source.get("boundary_evidence", {}),
    )
    targets = {item["uuid"] for item in selected}
    action = {
        "action": spec["action"],
        "target": [
            int(key)
            for key, value in bundle["mapping"].items()
            if value["uuid"] in targets
        ],
    }
    if spec["action"] == "trim_silence":
        action["keep_ms"] = spec["keep_ms"]
    if len(action["target"]) != len(selected):
        raise ValueError("Review targets do not fit the diagnostic context.")
    try:
        apply_edit(source["document"], bundle, action)
    except ValueError as error:
        return {
            "allowed": False,
            "reason": str(error),
            "proposal": action,
            "input": bundle["input"],
            "document_hash": bundle["document_hash"],
        }
    raise ValueError("Unverified review candidate unexpectedly passed the executor.")


def render_check(folder, samples, removal, origin):
    import numpy as np

    after, removed, cuts = cut_pcm(samples, removal, origin)
    save_pcm(folder / "before.wav", samples)
    save_pcm(folder / "after.wav", after)
    save_pcm(folder / "removed.wav", removed)
    left, right = cuts
    save_pcm(
        folder / "join.wav",
        after[max(0, left - 2 * RATE) : min(len(after), left + 2 * RATE)],
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-i",
            str(folder / "before.wav"),
            "-filter_complex",
            f"[0:a]atrim=end_sample={left},asetpts=PTS-STARTPTS[a];"
            f"[0:a]atrim=start_sample={right},asetpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=0:a=1[out]",
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",
            str(folder / "ffmpeg-after.wav"),
        ],
        check=True,
        capture_output=True,
    )
    if not np.array_equal(load_pcm(folder / "ffmpeg-after.wav"), after):
        raise ValueError(
            "Independent FFmpeg render differs from the source-sample proposal."
        )
    restored = np.concatenate((after[:left], removed, after[left:]))
    if not np.array_equal(samples, restored):
        raise ValueError("Retained and removed PCM do not reconstruct the original.")
    return {
        "cut_samples": cuts,
        "sample_rate": RATE,
        "original_frames": len(samples),
        "edited_frames": len(after),
        "removed_frames": len(removed),
        "ffmpeg_sample_exact": True,
        "pcm_reconstruction_exact": True,
        "media_operation": "hard_concat_no_crossfade_matching_current_cut_semantics",
        "editor_transaction_executed": False,
        "signal": signal_metrics(samples, after, cuts),
    }


def asr_probe(model, path, language):
    segments, info = model.transcribe(
        str(path),
        language=language,
        task="transcribe",
        beam_size=5,
        temperature=0,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    segments = [asdict(segment) for segment in segments]
    return {
        "text": "".join(s["text"] for s in segments).strip(),
        "segments": segments,
        "language": info.language,
        "duration_seconds": info.duration,
        "prompt": None,
        "vad_filter": False,
        "beam_size": 5,
        "temperature": 0,
        "condition_on_previous_text": False,
        "audio_sha256": digest(path),
    }


def build(directory, recipe_path, model_directory):
    import numpy as np
    from faster_whisper import WhisperModel

    recipe = read(recipe_path)
    scan_directory = ROOT / recipe["scan_directory"]
    check_files(scan_directory, read(scan_directory / "manifest.json"))
    check_files(model_directory, read(model_directory / "manifest.json"))
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "recipe.json", recipe)
    windows = {window["id"]: window for window in read(scan_directory / "windows.json")}
    model = WhisperModel(
        str(model_directory / "whisper-small"),
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        local_files_only=True,
    )
    results, asr_cache = [], {}
    for spec in recipe["cases"]:
        source = read(scan_directory / "sources" / f"{spec['source_id']}.json")
        selected = source_selection(source, spec)
        window = windows[spec["window_id"]]
        if window["source_id"] != spec["source_id"]:
            raise ValueError("Candidate source and audio disagree.")
        track = window["tracks"]["mix"]
        if digest(Path(track["path"])) != track["sha256"]:
            raise ValueError("Source preview changed.")
        samples = load_pcm(Path(track["path"]))
        lower = max(
            0, round((spec["target_start"] - 4 - window["start_seconds"]) * RATE)
        )
        upper = min(
            len(samples),
            round((spec["target_end"] + 4 - window["start_seconds"]) * RATE),
        )
        origin = window["start_seconds"] + lower / RATE
        before = samples[lower:upper]
        result = {
            **spec,
            "source_group_id": source["source_group_id"],
            "split": source["reserved_split"],
            "language": source["language"],
            "source_sha256": source["video_sha256"],
            "selected_items": selected,
            "preview_origin_seconds": origin,
            "transcript_context": [
                {
                    "start": i["sourceStart"],
                    "end": i["sourceStart"] + i["length"],
                    "text": i.get("text", ""),
                    "type": i["type"],
                }
                for i in source["document"]["content"]
                if "sourceStart" in i
                and overlaps(
                    [i["sourceStart"], i["sourceStart"] + i["length"]],
                    [origin, origin + len(before) / RATE],
                )
            ],
            "contract": contract_probe(source, spec, selected),
            "training_eligible": False,
            "agent_listened": False,
            "human_listened": False,
            "renders": [],
        }
        for variant in [{"id": "proposed"}] + spec.get("variants", []):
            proposal = {**spec, **variant}
            folder = directory / spec["id"] / variant["id"]
            folder.mkdir(parents=True)
            removal = proposal.get("removal_override") or removal_interval(
                spec["target_start"],
                spec["target_end"],
                proposal["action"],
                proposal.get("keep_ms"),
            )
            rendered = {
                "id": variant["id"],
                "folder": str(folder.relative_to(directory)),
                "removed_source_interval": removal,
                "keep_ms": proposal.get("keep_ms"),
                "verification": render_check(folder, before, removal, origin),
                "vad_removed": {},
                "asr": {},
            }
            for name, value in window["tracks"].items():
                probability = np.load(
                    scan_directory / value["probabilities"], allow_pickle=False
                )
                rendered["vad_removed"][name] = speech_probe(
                    probability, *removal, window["start_seconds"]
                )
            rendered["annotation_conflicts"] = annotation_conflicts(
                source.get("native_annotations", []), removal
            )
            for kind in ["before", "after", "removed"]:
                path = folder / f"{kind}.wav"
                key = (digest(path), source["language"])
                if key not in asr_cache:
                    asr_cache[key] = asr_probe(model, path, source["language"])
                transcript = copy.deepcopy(asr_cache[key])
                write(folder / f"asr-{kind}.json", transcript)
                rendered["asr"][kind] = transcript["text"]
            rendered["asr_text_stable"] = normalized_text(
                rendered["asr"]["before"]
            ) == normalized_text(rendered["asr"]["after"])
            result["renders"].append(rendered)
        write(directory / spec["id"] / "evidence.json", result)
        results.append(result)
        print(
            json.dumps(
                {"case": spec["id"], "asr": result["renders"][0]["asr"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    save_rows(directory / "review-cases.jsonl", results)
    write(
        directory / "manifest.json",
        {
            "version": "audio-review-v1",
            "cases": len(results),
            "render_variants": sum(len(r["renders"]) for r in results),
            "unique_asr_probes": len(asr_cache),
            "remote_uploads": 0,
            "scribe_credits_used": 0,
            "scan_directory": str(scan_directory.relative_to(ROOT)),
            "scan_manifest_sha256": digest(scan_directory / "manifest.json"),
            "recipe_path": str(recipe_path.relative_to(ROOT)),
            "recipe_sha256": digest(recipe_path),
            "model_manifest_sha256": digest(model_directory / "manifest.json"),
            "code_sha256": digest(Path(__file__)),
            "training_eligible": False,
            "files": {
                str(p.relative_to(directory)): digest(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["scan", "build"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / "data/manifests/audio-review-cases-v1.json",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "artifacts/datasets/nep-release-v1/audio-models-v1",
    )
    args = parser.parse_args()
    if args.command == "scan":
        scan(args.directory.resolve())
    else:
        build(args.directory.resolve(), args.recipe.resolve(), args.models.resolve())


if __name__ == "__main__":
    main()
