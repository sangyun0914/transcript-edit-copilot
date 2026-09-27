import copy
import io
import json
import os
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import httpx

from app.asr import scribe
from app.transcribe import TranscriptionState, TranscriptionTask, process_audio

FIXTURE = json.loads((Path(__file__).parent / "fixtures/scribe-ko.json").read_text())


def wav_file(duration=5):
    file = io.BytesIO()
    with wave.open(file, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * int(16000 * duration))
    file.seek(0)
    return file


def response(status=200, raw=FIXTURE):
    return httpx.Response(status, json=raw, request=httpx.Request("POST", scribe.API_URL))


class NormalizationTests(unittest.TestCase):
    def normalize(self, words, duration=30):
        raw = {"language_code": "kor", "text": " ".join(w["text"] for w in words), "words": words}
        before = copy.deepcopy(raw)
        paragraphs, annotations = scribe.normalize_transcript(raw, duration, "sample", True)
        self.assertEqual(raw, before)
        items = [item for paragraph in paragraphs for item in paragraph["content"]]
        cursor = 0
        for item in items:
            self.assertAlmostEqual(item["sourceStart"], cursor)
            self.assertGreater(item["length"], 0)
            cursor += item["length"]
        self.assertAlmostEqual(cursor, duration)
        references = [i for item in items for i in item.get("asr", {}).get("word_indices", [])]
        references += [i for note in annotations for i in note["word_indices"]]
        self.assertEqual(
            sorted(references), [i for i, w in enumerate(words) if w["type"] != "spacing"]
        )
        return paragraphs, items, annotations

    def word(self, text, start, end, speaker="speaker_0", kind="word"):
        return {
            "text": text,
            "start": float(start),
            "end": float(end),
            "speaker_id": speaker,
            "type": kind,
        }

    def test_preserves_fillers_speakers_and_events_without_combining_their_audio(self):
        paragraphs, items, notes = self.normalize(FIXTURE["words"], 5)
        words = [item for item in items if item["type"] == "word"]
        self.assertEqual(
            [w["word"] for w in words],
            ["어", "오늘은", "데이타를", "분석합니다.", "네.", "다음으로", "가죠."],
        )
        self.assertTrue(all(w["conf"] is None for w in words))
        self.assertEqual(notes[0]["word_indices"], [5])
        self.assertEqual(notes[0]["type"], "audio_event")
        self.assertAlmostEqual(words[4]["length"], 0.3)
        self.assertEqual(paragraphs[-1]["speaker"], "speaker_1")
        self.assertNotIn("asr", items[2])

    def test_long_audio_event_does_not_collapse_word_boundaries(self):
        _, items, notes = self.normalize(
            [
                self.word("music", 0, 30, kind="audio_event"),
                self.word("one", 1, 2),
                self.word("two", 4, 5),
                self.word("three", 8, 9),
            ]
        )
        self.assertEqual([w["length"] for w in items if w["type"] == "word"], [1, 1, 1])
        self.assertEqual(notes[0]["end"], 30)

    def test_points_only_associate_with_containing_same_speaker_intervals(self):
        _, items, notes = self.normalize(
            [
                self.word("leading", 0, 0),
                self.word("anchor", 1, 2),
                self.word("inside", 1.5, 1.5),
                self.word("touching", 2, 2),
                self.word("other speaker", 1.75, 1.75, "speaker_1"),
                self.word("distant", 12, 12),
            ]
        )
        words = [item for item in items if item["type"] == "word"]
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0]["word"], "anchor inside touching")
        self.assertEqual(words[0]["length"], 1)
        self.assertEqual(words[0]["asr"]["boundary_status"], "needs_alignment")
        self.assertEqual([note["text"] for note in notes], ["leading", "other speaker", "distant"])
        self.assertEqual(notes[1]["speaker_id"], "speaker_1")

    def test_overlapping_speech_preserves_speaker_set_without_inventing_an_identity(self):
        paragraphs, items, notes = self.normalize(
            [
                self.word("one", 1, 2),
                self.word("two", 1.98, 2.5, "speaker_1"),
                self.word("later", 3, 4, "speaker_1"),
            ]
        )
        words = [item for item in items if item["type"] == "word"]
        self.assertEqual(words[0]["asr"]["speaker_ids"], ["speaker_0", "speaker_1"])
        self.assertEqual(words[0]["asr"]["boundary_status"], "overlap")
        self.assertEqual(words[0]["length"], 1.5)
        self.assertEqual(paragraphs[0]["speaker"], "")
        self.assertEqual(paragraphs[-1]["speaker"], "speaker_1")
        self.assertEqual(notes, [])

    def test_event_only_audio_and_empty_transcript_preserve_duration(self):
        for words in (
            [],
            [self.word("laughter", 1, 2, kind="audio_event")],
            [self.word("clap", 0, 0, kind="audio_event")],
        ):
            with self.subTest(words=words):
                _, items, _ = self.normalize(words)
                self.assertTrue(all(item["type"] == "silence" for item in items))

    def test_rejects_invalid_timing_and_unexpected_formats(self):
        for change in (
            {"start": -1},
            {"end": 6},
            {"start": None},
            {"end": 0.2},
            {"end": float("nan")},
            {"type": "unexpected"},
            {"end": 5.0005},
        ):
            raw = copy.deepcopy(FIXTURE)
            raw["words"][0].update(change)
            with self.subTest(change=change), self.assertRaises(scribe.ScribeError):
                scribe.normalize_transcript(raw, 5, "sample", True)
        for raw in (
            {"text": "untimed", "language_code": "ko", "words": []},
            {"transcripts": []},
            [],
        ):
            with self.subTest(raw=raw), self.assertRaises(scribe.ScribeError):
                scribe.normalize_transcript(raw, 5, "sample", True)

    def test_only_point_words_fail_without_inventing_durations(self):
        with self.assertRaisesRegex(scribe.ScribeError, "no words with usable duration"):
            self.normalize([self.word("point", 1, 1)])


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {scribe.API_KEY_ENV: "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_verbatim_request_and_raw_provenance(self):
        with patch.object(scribe.httpx, "Client") as factory:
            client = factory.return_value.__enter__.return_value
            client.post.return_value = response()
            file = wav_file()
            content, metadata = scribe.transcribe(file, "sample", "ko", True, 4)
            kwargs = client.post.call_args.kwargs
            self.assertEqual(
                kwargs["data"],
                {
                    "model_id": "scribe_v2",
                    "no_verbatim": "false",
                    "timestamps_granularity": "word",
                    "diarize": "true",
                    "tag_audio_events": "true",
                    "language_code": "ko",
                    "num_speakers": "4",
                },
            )
            self.assertIs(kwargs["files"]["file"][1], file)
            self.assertEqual(metadata["raw_response"], FIXTURE)
            self.assertNotIn("test-key", json.dumps(metadata))
            self.assertTrue(content)
            client.post.assert_called_once()

    def test_missing_key_fails_before_network(self):
        with (
            patch.dict(os.environ, {scribe.API_KEY_ENV: ""}),
            patch.object(scribe.httpx, "Client") as factory,
        ):
            with self.assertRaises(scribe.ScribeError):
                scribe.transcribe(wav_file(), "sample", "ko", True, None)
            factory.assert_not_called()

    def test_http_and_timeout_failures_are_safe_and_not_retried(self):
        for failure in (
            response(401, {"detail": "private provider data"}),
            response(429),
            httpx.ReadTimeout("sensitive request details"),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.object(scribe.httpx, "Client") as factory,
            ):
                client = factory.return_value.__enter__.return_value
                if isinstance(failure, Exception):
                    client.post.side_effect = failure
                else:
                    client.post.return_value = failure
                with self.assertRaises(scribe.ScribeError) as caught:
                    scribe.transcribe(wav_file(), "sample", "ko", True, None)
                self.assertNotIn("private provider", str(caught.exception))
                self.assertNotIn("sensitive request", str(caught.exception))
                client.post.assert_called_once()

    def test_failed_and_successful_tasks_close_the_owned_file(self):
        for error in (
            None,
            scribe.ScribeError("Account unavailable"),
            RuntimeError("private details"),
        ):
            file = wav_file()
            task = TranscriptionTask("sample", TranscriptionState.QUEUED)
            with (
                self.subTest(error=type(error).__name__),
                patch.object(scribe, "transcribe", side_effect=error, return_value=([{}], {})),
            ):
                process_audio(task, file, "sample", "scribe", None, "ko", True, None)
                self.assertTrue(file.closed)
                self.assertEqual(
                    task.state, TranscriptionState.FAILED if error else TranscriptionState.DONE
                )
                if error:
                    self.assertNotIn("private details", task.error)
                else:
                    self.assertEqual(task.progress, 1)


if __name__ == "__main__":
    unittest.main()
