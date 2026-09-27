import math
import os
import wave
from typing import BinaryIO, Literal

import httpx
from pydantic import BaseModel, ConfigDict, FiniteFloat, ValidationError

API_KEY_ENV = "ELEVENLABS_PERSONAL_API_KEY"
MODEL_ID = "scribe_v2"
API_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class ScribeError(Exception):
    pass


class Word(BaseModel):
    model_config = ConfigDict(strict=True)

    text: str
    type: Literal["word", "spacing", "audio_event"]
    start: FiniteFloat | None = None
    end: FiniteFloat | None = None
    speaker_id: str | None = None


class Transcript(BaseModel):
    model_config = ConfigDict(strict=True)

    language_code: str
    text: str
    words: list[Word]


def is_configured() -> bool:
    return bool(os.environ.get(API_KEY_ENV, "").strip())


def normalize_transcript(
    raw: dict, duration: float, name: str, diarize: bool
) -> tuple[list[dict], list[dict]]:
    try:
        transcript = Transcript.model_validate(raw)
    except ValidationError:
        raise ScribeError("Scribe returned an unsupported transcript format.") from None
    if not math.isfinite(duration) or duration <= 0:
        raise ScribeError("The recording must have a positive duration.")

    timed = []
    for index, word in enumerate(transcript.words):
        if word.type == "spacing":
            continue
        if (
            word.start is None
            or word.end is None
            or not 0 <= word.start <= word.end <= duration
            or not word.text.strip()
        ):
            raise ScribeError(
                f"Scribe returned missing or invalid timing for word {index} "
                f"(start={word.start}, end={word.end}, duration={duration:.3f}); "
                "import was stopped."
            )
        timed.append((index, word))
    if not timed and transcript.text.strip():
        raise ScribeError("Scribe returned text without timed words; import was stopped.")
    speech = [(index, word) for index, word in timed if word.type == "word"]
    if speech and not any(word.start < word.end for _, word in speech):
        raise ScribeError("Scribe returned no words with usable duration; import was stopped.")

    speech.sort(key=lambda pair: pair[1].start)
    groups: list[list[tuple[int, Word]]] = []
    group_end = 0.0
    for pair in speech:
        word = pair[1]
        if word.start == word.end:
            continue
        if groups and word.start < group_end:
            groups[-1].append(pair)
        else:
            groups.append([pair])
        group_end = max(group_end, word.end)

    annotations = []
    for index, word in timed:
        if word.type == "word" and word.start < word.end:
            continue
        if word.type == "word":
            # Association never extends an audio interval or crosses a speaker boundary.
            group = next(
                (
                    group
                    for group in groups
                    if any(
                        anchor.start <= word.start <= anchor.end
                        and anchor.start < anchor.end
                        and anchor.speaker_id == word.speaker_id
                        for _, anchor in group
                    )
                ),
                None,
            )
            if group is not None:
                group.append((index, word))
                continue
        annotations.append(
            {
                "type": "audio_event" if word.type == "audio_event" else "unaligned_word",
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "word_indices": [index],
                "speaker_id": word.speaker_id,
            }
        )

    paragraphs: list[dict] = []

    def append(speaker: str, item: dict):
        if not paragraphs or paragraphs[-1]["speaker"] != speaker:
            paragraphs.append({"speaker": speaker, "content": []})
        paragraphs[-1]["content"].append(item)

    cursor = 0.0
    for group in groups:
        group.sort(key=lambda pair: (pair[1].start, pair[0]))
        start = min(word.start for _, word in group)
        end = max(word.end for _, word in group)
        needs_alignment = any(word.start == word.end for _, word in group)
        boundary_status = "unverified"
        if needs_alignment:
            boundary_status = "needs_alignment"
        elif len(group) > 1:
            boundary_status = "overlap"
        speakers = {word.speaker_id for _, word in group if word.speaker_id}
        speaker = name
        if diarize and speakers:
            if len(speakers) == 1:
                speaker = next(iter(speakers))
            else:
                speaker = ""
        if start > cursor:
            previous_speaker = paragraphs[-1]["speaker"] if paragraphs else speaker
            append(
                previous_speaker,
                {"type": "silence", "sourceStart": cursor, "length": start - cursor},
            )

        item = {
            "sourceStart": start,
            "length": end - start,
            "asr": {
                "word_indices": [index for index, _ in group],
                "boundary_status": boundary_status,
                "speaker_ids": sorted(speakers),
            },
        }
        words = [word.text.strip() for _, word in group if word.type == "word"]
        if words:
            item.update(type="word", word=" ".join(words), conf=None)
        else:
            item.update(type="silence")
        append(speaker, item)
        cursor = end

    if cursor < duration:
        speaker = paragraphs[-1]["speaker"] if paragraphs else name
        append(speaker, {"type": "silence", "sourceStart": cursor, "length": duration - cursor})
    return paragraphs, annotations


def transcribe(
    file: BinaryIO,
    name: str,
    language_code: str | None,
    diarize: bool,
    num_speakers: int | None,
) -> tuple[list[dict], dict]:
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise ScribeError(f"Set {API_KEY_ENV} in the backend environment to use Scribe.")
    try:
        file.seek(0)
        with wave.open(file, "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        raise ScribeError("Scribe import requires a valid PCM WAV file.") from None
    if duration < 0.1:
        raise ScribeError("Scribe requires at least 100 ms of audio.")
    file.seek(0)

    options = {
        "model_id": MODEL_ID,
        "no_verbatim": "false",
        "timestamps_granularity": "word",
        "diarize": str(diarize).lower(),
        "tag_audio_events": "true",
    }
    if language_code:
        options["language_code"] = language_code
    if diarize and num_speakers is not None:
        options["num_speakers"] = str(num_speakers)

    try:
        with httpx.Client(timeout=httpx.Timeout(600, connect=10, write=120, pool=10)) as client:
            response = client.post(
                API_URL,
                headers={"xi-api-key": api_key},
                data=options,
                files={"file": ("audio.wav", file, "audio/wav")},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ScribeError(
            f"ElevenLabs rejected the request (HTTP {exc.response.status_code}). "
            "Check the API key, account permissions, and available credits."
        ) from None
    except httpx.TimeoutException:
        raise ScribeError(
            "Scribe timed out. Check the account before retrying a billable upload."
        ) from None
    except httpx.RequestError:
        raise ScribeError(
            "Could not reach ElevenLabs. Check the backend's network connection."
        ) from None
    try:
        raw = response.json()
    except ValueError:
        raise ScribeError("Scribe returned an invalid JSON response.") from None

    content, annotations = normalize_transcript(raw, duration, name, diarize)
    return content, {
        "version": 1,
        "provider": "elevenlabs",
        "model": MODEL_ID,
        "language_code": raw["language_code"],
        "duration": duration,
        "request": options,
        "raw_response": raw,
        "annotations": annotations,
    }
