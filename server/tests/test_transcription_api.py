import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from test_scribe import wav_file

with (
    tempfile.TemporaryDirectory() as directory,
    patch.dict(
        os.environ,
        {
            "AUDAPOLIS_DATA_DIR": directory,
            "AUDAPOLIS_CACHE_DIR": directory,
        },
    ),
):
    from app.main import AUTH_TOKEN, app
    from app.tasks import tasks


class TranscriptionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        tasks.tasks.clear()
        self.headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}

    def post(self, **params):
        return self.client.post(
            "/tasks/start_transcription/",
            params=params,
            headers=self.headers,
            files={"file": ("sample.wav", wav_file().getvalue(), "audio/wav")},
            data={"fileName": "sample.wav"},
        )

    def test_capabilities_are_authenticated_and_never_return_the_key(self):
        self.assertEqual(self.client.get("/transcription/capabilities/").status_code, 401)
        with (
            patch.dict(os.environ, {"ELEVENLABS_PERSONAL_API_KEY": "test-key"}),
            patch("app.main.local_asr_available", return_value=False),
        ):
            response = self.client.get("/transcription/capabilities/", headers=self.headers)
        self.assertEqual(response.json(), {"scribe": True, "vosk": False})

    def test_missing_local_runtime_fails_before_creating_a_task(self):
        with (
            patch("app.main.local_asr_available", return_value=False),
            patch("app.main.tempfile.TemporaryFile") as owned,
        ):
            response = self.post(provider="vosk", transcription_model="downloaded-model")
        self.assertEqual(response.status_code, 503)
        self.assertIn("uv sync --extra local-asr", response.json()["detail"])
        owned.assert_not_called()
        self.assertEqual(len(tasks.tasks), 0)

    def test_scribe_requires_no_local_model_and_background_job_owns_upload(self):
        files_seen = []

        def transcribe(file, *_):
            self.assertFalse(file.closed)
            self.assertEqual(file.read(4), b"RIFF")
            files_seen.append(file)
            return [{"speaker": "s0", "content": []}], {"model": "scribe_v2"}

        with (
            patch.dict(os.environ, {"ELEVENLABS_PERSONAL_API_KEY": "test-key"}),
            patch("app.asr.scribe.transcribe", side_effect=transcribe),
        ):
            started = self.post(provider="scribe", language_code="ko", diarize=True)
        self.assertEqual(started.status_code, 200)
        task = self.client.get(f"/tasks/{started.json()['uuid']}/", headers=self.headers).json()
        self.assertEqual(task["state"], "done")
        self.assertEqual(task["transcription"]["model"], "scribe_v2")
        self.assertTrue(files_seen[0].closed)

    def test_bad_configuration_and_options_do_not_queue_jobs(self):
        with patch.dict(os.environ, {"ELEVENLABS_PERSONAL_API_KEY": ""}):
            self.assertEqual(self.post(provider="scribe").status_code, 503)
        for params in (
            {"provider": "unknown"},
            {"provider": "vosk"},
            {"provider": "scribe", "diarize_max_speakers": 0},
            {"provider": "scribe", "language_code": "../ko"},
        ):
            with self.subTest(params=params):
                self.assertEqual(self.post(**params).status_code, 422)
        self.assertEqual(len(tasks.tasks), 0)


if __name__ == "__main__":
    unittest.main()
