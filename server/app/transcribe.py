import enum
import logging
from dataclasses import dataclass
from typing import BinaryIO

from .asr import scribe
from .tasks import Task

logger = logging.getLogger(__name__)


class TranscriptionState(str, enum.Enum):
    QUEUED = "queued"
    LOADING_TRANSCRIPTION_MODEL = "loading transcription model"
    LOADING = "loading"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TranscriptionTask(Task):
    filename: str
    state: TranscriptionState
    total: float = 0
    processed: float = 0
    content: list[dict] | None = None
    progress: float = 0
    transcription: dict | None = None
    error: str | None = None

    def set_transcription_progress(self, processed):
        self.processed += processed
        self.progress = self.processed / self.total if self.total else 0


def process_audio(
    task: TranscriptionTask,
    file: BinaryIO,
    file_name: str,
    provider: str,
    transcription_model: str | None,
    language_code: str | None,
    diarize: bool,
    diarize_max_speakers: int | None,
):
    try:
        with file:
            if provider == "scribe":
                task.state = TranscriptionState.TRANSCRIBING
                task.content, task.transcription = scribe.transcribe(
                    file, file_name, language_code, diarize, diarize_max_speakers
                )
            else:
                from .asr.vosk import transcribe

                task.content = transcribe(
                    task,
                    transcription_model,
                    file,
                    file_name,
                    task.uuid,
                    diarize,
                    diarize_max_speakers,
                )
        task.progress = 1
        task.state = TranscriptionState.DONE
    except scribe.ScribeError as exc:
        task.error = str(exc)
        task.state = TranscriptionState.FAILED
    except Exception as exc:
        logger.error("Transcription failed: %s", type(exc).__name__)
        task.error = "Transcription failed. Check the backend configuration and input recording."
        task.state = TranscriptionState.FAILED
