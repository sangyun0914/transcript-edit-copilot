"""Transcribe preregistered training media, retaining raw results before normalization."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def acquire(registration, output):
    spec = json.loads(registration.read_text())
    if spec["split"] != "train" or not spec["selection_before_transcript_inspection"]:
        raise ValueError(
            "A training allocation and pre-inspection selection are required."
        )
    clips = spec["clips"]
    if (
        not clips
        or sum(c["duration_seconds"] for c in clips)
        > spec["maximum_transcribed_seconds"]
    ):
        raise ValueError("The registered transcription budget was exceeded.")
    if len({c["id"] for c in clips}) != len(clips):
        raise ValueError("Clip IDs must be unique.")
    catalog = json.loads((ROOT / spec["source_manifest"]).read_text())
    matches = [r for r in catalog["files"] if r["sha256"] == spec["source_sha256"]]
    if len(matches) != 1:
        raise ValueError("Expected one registered recording.")
    record = matches[0]
    media = ROOT / "data" / record["path"]
    if digest(media) != spec["source_sha256"]:
        raise ValueError("Recording differs from the registered SHA-256.")
    for clip in clips:
        if Path(clip["id"]).name != clip["id"] or clip["id"] in {".", ".."}:
            raise ValueError("Unsafe clip ID.")
        if (
            not 0
            <= clip["start_seconds"]
            < clip["start_seconds"] + clip["duration_seconds"]
            <= record["duration_seconds"]
        ):
            raise ValueError("Excerpt is outside the recording.")
    sys.path.insert(0, str(ROOT / "server"))
    from app.asr.scribe import (
        API_KEY_ENV,
        API_URL,
        MODEL_ID,
        is_configured,
        normalize_transcript,
    )

    import httpx

    if not is_configured() or MODEL_ID != spec["model"]:
        raise ValueError(
            "Registered Scribe model and configured credentials are required."
        )
    output.mkdir(parents=True, exist_ok=False)
    write(output / "registration.json", spec)
    records = []
    for clip in clips:
        folder = output / clip["id"]
        folder.mkdir()
        audio_path = folder / "input.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-ss",
                str(clip["start_seconds"]),
                "-i",
                str(media),
                "-t",
                str(clip["duration_seconds"]),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(audio_path),
            ],
            check=True,
        )
        request = {
            **clip,
            "source_id": spec["source_id"],
            "source_group_id": spec["source_group_id"],
            "split": spec["split"],
            "media_sha256": spec["source_sha256"],
            "audio_sha256": digest(audio_path),
            "registration_sha256": digest(registration),
            "provider": spec["provider"],
            "model": spec["model"],
            "source_time_offset_seconds": clip["start_seconds"],
            "request_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write(folder / "request.json", request)
        started = time.monotonic()
        try:
            options = {
                "model_id": MODEL_ID,
                "no_verbatim": "false",
                "timestamps_granularity": "word",
                "diarize": "true",
                "tag_audio_events": "true",
                "language_code": spec["language"],
            }
            with wave.open(str(audio_path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            with audio_path.open("rb") as stream:
                with httpx.Client(
                    timeout=httpx.Timeout(1800, connect=10, write=600, pool=10)
                ) as client:
                    response = client.post(
                        API_URL,
                        headers={"xi-api-key": os.environ[API_KEY_ENV]},
                        data=options,
                        files={"file": ("audio.wav", stream, "audio/wav")},
                    )
            response.raise_for_status()
            raw = response.json()
            write(folder / "raw-response.json", raw)
            content, annotations = normalize_transcript(raw, duration, clip["id"], True)
            metadata = {
                "version": 1,
                "provider": "elevenlabs",
                "model": MODEL_ID,
                "language_code": raw["language_code"],
                "duration": duration,
                "request": options,
                "raw_response": raw,
                "annotations": annotations,
            }
        except Exception as error:
            write(
                folder / "failed.json",
                {
                    "error_type": type(error).__name__,
                    "http_status": error.response.status_code
                    if isinstance(error, httpx.HTTPStatusError)
                    else None,
                    "provider_error": error.response.text[:2000]
                    if isinstance(error, httpx.HTTPStatusError)
                    else None,
                    "state": "request_failed_or_outcome_unknown_no_automatic_retry",
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            raise RuntimeError(
                "Acquisition failed; inspect the saved status/raw result before any paid retry."
            ) from None
        result_path = folder / "normalized-response.json"
        write(result_path, {"content": content, "transcription": metadata})
        result = {
            **request,
            "result": str(result_path.relative_to(output)),
            "result_sha256": digest(result_path),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "timed_items": sum(len(p["content"]) for p in content),
            "status": "source_only_not_cut_qualified",
        }
        write(folder / "acquisition.json", result)
        records.append(result)
        print(
            json.dumps(
                {
                    "clip": clip["id"],
                    "items": result["timed_items"],
                    "seconds": result["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    write(
        output / "acquisition.json",
        {
            "registration_sha256": digest(registration),
            "script_sha256": digest(Path(__file__)),
            "clips": records,
            "new_scribe_requests": len(records),
            "total_audio_seconds": sum(c["duration_seconds"] for c in clips),
            "admitted_cases": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.registration.resolve(), args.output.resolve())
