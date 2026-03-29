import array
import enum
import json
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import UploadFile
from pyannote.audio import Pipeline
from scipy.signal import resample_poly
from vosk import KaldiRecognizer, Model

from .models import models
from .tasks import Task, tasks

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Number of seconds that should be fed into vosk.
# Smaller = better progress estimates, but also slightly higher python overhead
VOSK_BLOCK_SIZE = 2


def _load_wav_mono_16k(file) -> tuple[np.ndarray, float]:
    """Load a WAV file and return (samples_int16, duration_seconds) at 16kHz mono."""
    data, sr = sf.read(file, dtype="float32", always_2d=True)
    # Mix to mono
    mono = data.mean(axis=1)
    # Resample to SAMPLE_RATE if needed
    if sr != SAMPLE_RATE:
        from math import gcd

        g = gcd(SAMPLE_RATE, sr)
        mono = resample_poly(mono, SAMPLE_RATE // g, sr // g).astype(np.float32)
    duration = float(len(mono) / SAMPLE_RATE)
    # Convert to int16 for Vosk
    samples_int16 = np.clip(mono * 32767, -32768, 32767).astype(np.int16)
    return samples_int16, duration


class _AudioSliceable:
    """Minimal wrapper providing pydub-like slicing for int16 sample arrays."""

    def __init__(self, samples: np.ndarray, sample_rate: int):
        self._samples = samples
        self._sample_rate = sample_rate

    @property
    def duration_seconds(self) -> float:
        return len(self._samples) / self._sample_rate

    def get_array_of_samples(self) -> array.array:
        return array.array("h", self._samples.tobytes())

    def slice(self, start_sec: float, end_sec: float) -> "_AudioSliceable":
        start_idx = int(start_sec * self._sample_rate)
        end_idx = int(end_sec * self._sample_rate)
        return _AudioSliceable(self._samples[start_idx:end_idx], self._sample_rate)


@dataclass
class DiarSegment:
    start: float
    length: float
    speaker_id: str


_diarization_pipeline: Optional[Pipeline] = None


def _get_diarization_pipeline() -> Pipeline:
    """Lazy-load and cache the pyannote speaker diarization pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required for speaker diarization. "
            "Create a free account at https://huggingface.co, accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-community-1, "
            "and set HF_TOKEN to your access token."
        )

    logger.info("Loading pyannote speaker diarization pipeline...")
    _diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    logger.info("Diarization pipeline loaded.")
    return _diarization_pipeline


class TranscriptionState(str, enum.Enum):
    QUEUED = "queued"
    LOADING_TRANSCRIPTION_MODEL = "loading transcription model"
    LOADING = "loading"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    DONE = "done"


@dataclass
class TranscriptionTask(Task):
    filename: str
    state: TranscriptionState
    total: float = 0
    processed: float = 0
    content: Optional[dict] = None
    progress: float = 0

    def set_transcription_progress(self, processed):
        self.processed += processed
        self.progress = self.processed / self.total


def transcribe_raw_data(model: Model, name, audio, offset, duration, process_callback):
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    finished = False
    processed = offset
    while not finished:
        block_start = processed
        block_end = processed + VOSK_BLOCK_SIZE
        if block_end > offset + duration:
            block_end = offset + duration
            finished = True
        block = audio.slice(block_start, block_end)
        rec.AcceptWaveform(block.get_array_of_samples().tobytes())
        processed = block_end
        process_callback(processed - block_start)

    vosk_result = json.loads(rec.FinalResult())
    return transform_vosk_result(name, vosk_result, duration, offset)


EPSILON = 0.00001


def process_audio(
    transcription_model: str,
    file: UploadFile,
    fileName: str,
    task_uuid: str,
    diarize: bool,
    diarize_max_speakers: Optional[int],
):
    task = tasks.get(task_uuid)

    content = transcribe(
        task,
        transcription_model,
        file,
        fileName,
        task_uuid,
        diarize,
        diarize_max_speakers,
    )

    task.content = content
    task.state = TranscriptionState.DONE


def transcribe(
    task: TranscriptionTask,
    transcription_model: str,
    file: UploadFile,
    fileName: str,
    task_uuid: str,
    diarize: bool,
    diarize_max_speakers: Optional[int],
):
    task.state = TranscriptionState.LOADING_TRANSCRIPTION_MODEL

    # TODO: Set error state if model does not exist
    model = models.get(transcription_model)

    samples, duration = _load_wav_mono_16k(file)
    audio = _AudioSliceable(samples, SAMPLE_RATE)

    # TODO: can we make this atomic?
    task.total = duration
    task.processed = 0

    if not diarize:
        task.state = TranscriptionState.TRANSCRIBING
        return [
            transcribe_raw_data(
                model,
                fileName,
                audio,
                0,
                duration,
                task.set_transcription_progress,
            )
        ]

    else:
        task.state = TranscriptionState.DIARIZING
        try:
            pipeline = _get_diarization_pipeline()
            waveform = torch.from_numpy(samples).float().unsqueeze(0) / 32768.0
            pipeline_input = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

            pipeline_params: dict = {}
            if diarize_max_speakers is not None:
                pipeline_params["max_speakers"] = diarize_max_speakers

            output = pipeline(pipeline_input, **pipeline_params)

            # Use exclusive diarization (no overlapping turns) for cleaner transcription
            annotation = output.exclusive_speaker_diarization

            segments = [
                DiarSegment(start=turn.start, length=turn.end - turn.start, speaker_id=speaker)
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]
        except Exception:
            traceback.print_exc()
            segments = []

        if not segments:
            segments = [DiarSegment(start=0, length=duration, speaker_id="SPEAKER_00")]
        else:
            # Extend the last segment to cover the full audio duration
            last = segments[-1]
            last.length = duration - last.start

        with ThreadPoolExecutor() as executor:
            task.state = TranscriptionState.TRANSCRIBING
            return list(
                executor.map(
                    lambda segment: transcribe_raw_data(
                        model,
                        f"{segment.speaker_id} ({fileName})",
                        audio,
                        segment.start,
                        segment.length,
                        task.set_transcription_progress,
                    ),
                    segments,
                )
            )


def transform_vosk_result(name: str, result: dict, length: float, offset: float = 0) -> dict:
    content = []
    current_time = 0

    for word in result.get("result", []):
        word_start = word["start"]

        if word["start"] > current_time:
            if (word["start"] - current_time) > 10 * EPSILON:
                content.append(
                    {
                        "sourceStart": current_time + offset,
                        "length": word["start"] - current_time,
                        "type": "silence",
                    }
                )
            else:
                word_start = current_time

        content.append(
            {
                "sourceStart": word_start + offset,
                "length": word["end"] - word["start"],
                "type": "word",
                "word": word["word"],
                "conf": word["conf"],
            }
        )
        current_time = word["end"]
    if current_time < length:
        if (length - current_time) < 10 * EPSILON and content:
            content[-1]["length"] += length - current_time
        else:
            content.append(
                {
                    "sourceStart": current_time + offset,
                    "length": length - current_time,
                    "type": "silence",
                }
            )

    return {"speaker": name, "content": content}
