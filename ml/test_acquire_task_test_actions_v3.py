import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
from uuid import UUID
import wave

import httpx

import acquire_task_test_actions_v3 as acquisition


@contextmanager
def extracted_fixture(root, media, window):
    payload = media.read_bytes()

    class Source(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            start, end = 0, len(payload) - 1
            if requested:
                first, last = requested.removeprefix("bytes=").split("-")
                start = int(first)
                end = min(int(last), end) if last else end
            self.send_response(206 if requested else 200)
            self.send_header(
                "Content-Type", "video/mp4" if media.suffix == ".mp4" else "audio/wav"
            )
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            try:
                self.wfile.write(payload[start : end + 1])
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Source)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        gate = {
            "binding": {"path": "fixture-only", "sha256": "fixture-only"},
            "windows": [window],
        }
        metadata = root / "metadata.json"
        metadata.write_text('{"kind":"synthetic_fixture_not_final_test"}\n')
        source = root / "source.json"
        source.write_text(
            json.dumps(
                {
                    "window_id": window["window_id"],
                    "freeze_sha256": "fixture-only",
                    "source_group": window["source_group"],
                    "recording_id": window["publisher_recording_id"],
                    "selection_before_words": True,
                    "model_predictions_seen": False,
                    "resolution_evidence": [acquisition.binding(metadata)],
                    "media_url": f"http://127.0.0.1:{server.server_address[1]}/{media.name}",
                }
            )
        )
        with (
            mock.patch.object(acquisition.sealed, "load_gate", return_value=gate),
            mock.patch.object(
                acquisition, "ledger_path", return_value=root / "ledger.json"
            ),
            mock.patch.object(
                acquisition, "public_media_url", side_effect=lambda url: url
            ),
        ):
            yield acquisition.extract(
                "fixture-only",
                source,
                root / "extracted",
                ffmpeg=shutil.which("ffmpeg"),
                ffprobe=shutil.which("ffprobe"),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class AcquisitionTests(unittest.TestCase):
    def test_gate_rejects_before_body_network_or_credential_access(self):
        for command, arguments in (
            (acquisition.extract, ("closed.json", "unopened-media.json", "unused")),
            (acquisition.transcribe, ("closed.json", "unopened-audio.json", "unused")),
        ):
            with (
                self.subTest(command=command.__name__),
                mock.patch.object(
                    acquisition.sealed, "load_gate", side_effect=ValueError("sealed")
                ),
                mock.patch.object(acquisition, "read") as read,
                mock.patch.object(acquisition.httpx, "Client") as client,
                mock.patch.object(acquisition.os.environ, "get") as environment,
            ):
                with self.assertRaisesRegex(ValueError, "sealed"):
                    command(*arguments)
                read.assert_not_called()
                client.assert_not_called()
                environment.assert_not_called()

    def test_failed_or_unknown_billable_attempt_cannot_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            acquisition.reserve_asr(path, "window-a", "pcm-hash", "first")
            with self.assertRaisesRegex(ValueError, "automatic retry"):
                acquisition.reserve_asr(path, "window-a", "pcm-hash", "second")
            saved = json.loads(path.read_text())["asr_attempts"]["window-a"]
            self.assertEqual(saved["output"], "first")
            self.assertEqual(saved["status"], "reserved_before_network_outcome_unknown")
            acquisition.reserve_asr(path, "window-b", "other-hash", "third")

    def test_transcribe_rejects_reassigned_source_before_pcm_or_credentials(self):
        gate = {
            "binding": {"path": "frozen.json", "sha256": "frozen"},
            "windows": [
                {
                    "window_id": "registered-window",
                    "source_group": "registered-source",
                    "publisher_recording_id": "registered-recording",
                    "language": "en",
                }
            ],
        }
        for group, source_id, error in (
            (
                "replacement-source",
                acquisition.sealed.opaque_source_id("registered-window"),
                "source identity changed",
            ),
            ("registered-source", "sealed:registered-window", "opaque source identity"),
            (
                "registered-source",
                acquisition.sealed.opaque_source_id("different-window"),
                "opaque source identity",
            ),
        ):
            acquired = {
                "status": "pcm_window_complete",
                "freeze": gate["binding"],
                "window_id": "registered-window",
                "source_group": group,
                "recording_id": "registered-recording",
                "language": "en",
                "source_id": source_id,
            }
            with (
                self.subTest(source_id=source_id),
                mock.patch.object(acquisition.sealed, "load_gate", return_value=gate),
                mock.patch.object(acquisition, "read", return_value=acquired),
                mock.patch.object(acquisition.sealed, "bound") as bound,
                mock.patch.object(acquisition.os.environ, "get") as environment,
                mock.patch.object(acquisition.httpx, "Client") as client,
            ):
                with self.assertRaisesRegex(ValueError, error):
                    acquisition.transcribe("frozen.json", "acquired.json", "unused")
                bound.assert_not_called()
                environment.assert_not_called()
                client.assert_not_called()

    def test_rejected_over_budget_bytes_remain_in_accounting(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(acquisition, "ENCODED_LIMIT", 10),
        ):
            path = Path(directory) / "ledger.json"
            acquisition.charge_bytes(path, "encoded", 8)
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                acquisition.charge_bytes(path, "encoded", 4)
            self.assertEqual(json.loads(path.read_text())["encoded_bytes"], 12)
            with self.assertRaisesRegex(ValueError, "budget exceeded"):
                acquisition.charge_bytes(path, "encoded", 0)

    def test_single_range_preserved_without_content_compression(self):
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(
                206, headers={"content-range": "bytes 16-19/100"}, content=b"abcd"
            )

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with acquisition.remote_response(
                client, "https://publisher.example/media.mp4", "GET", "bytes=16-19"
            ) as response:
                self.assertEqual(response.read(), b"abcd")
        self.assertEqual(requests[0].headers["range"], "bytes=16-19")
        self.assertEqual(requests[0].headers["accept-encoding"], "identity")
        with self.assertRaises(ValueError):
            acquisition.remote_response(
                None, "https://publisher.example/a", "GET", "bytes=1-2,4-5"
            )

    @unittest.skipUnless(
        os.environ.get("NEP_MEDIA_INTEGRATION") == "1",
        "Explicit synthetic loopback/FFmpeg check.",
    )
    def test_real_range_proxy_ffmpeg_preserves_registered_excerpt(self):
        with tempfile.TemporaryDirectory(
            dir=acquisition.ROOT / "artifacts"
        ) as directory:
            root = Path(directory)
            audio = root / "synthetic-zero-pcm.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\0" * 180 * 16000 * 2)
            window = {
                "window_id": "fixture-only",
                "source_group": "synthetic-fixture",
                "publisher_recording_id": "zero-pcm",
                "language": "en",
                "start_seconds": 45,
            }
            with extracted_fixture(root, audio, window) as result:
                self.assertEqual(result["status"], "pcm_window_complete")
                self.assertEqual(result["source_start_seconds"], 45)
                with wave.open(str(root / "extracted/audio.wav"), "rb") as stream:
                    self.assertEqual(
                        stream.readframes(90 * 16000), b"\0" * 90 * 16000 * 2
                    )
                transfers = [
                    json.loads(line)
                    for line in (root / "extracted/range-transfers.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertTrue(any(row["range"] for row in transfers))
                self.assertEqual(
                    sum(row["bytes_received"] for row in transfers),
                    result["encoded_bytes_received"],
                )
                self.assertEqual(
                    json.loads((root / "ledger.json").read_text())["encoded_bytes"],
                    result["encoded_bytes_received"],
                )

    @unittest.skipUnless(
        os.environ.get("NEP_MEDIA_INTEGRATION") == "1",
        "Explicit synthetic loopback/FFmpeg check.",
    )
    def test_aac_mp4_priming_fractional_seek_preserves_exact_pcm_and_opaque_identity(
        self,
    ):
        with tempfile.TemporaryDirectory(
            dir=acquisition.ROOT / "artifacts"
        ) as directory:
            root = Path(directory)
            media = root / "synthetic-aac-priming.mp4"
            tone = r"0.2*sin(2*PI*(440+440*gte(t\,90))*t)"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:r=2:d=181.237",
                    "-f",
                    "lavfi",
                    "-i",
                    f"aevalsrc={tone}|{tone}:s=44100:d=181.237:c=stereo",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",
                    "-movflags",
                    "+faststart",
                    "-n",
                    str(media),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            probe = json.loads(
                subprocess.check_output(
                    [
                        shutil.which("ffprobe"),
                        "-v",
                        "error",
                        "-select_streams",
                        "a:0",
                        "-show_streams",
                        "-show_packets",
                        "-read_intervals",
                        "%+#1",
                        "-of",
                        "json",
                        str(media),
                    ],
                    timeout=30,
                )
            )
            self.assertEqual(probe["streams"][0]["codec_name"], "aac")
            self.assertEqual(
                (probe["streams"][0]["sample_rate"], probe["streams"][0]["channels"]),
                ("44100", 2),
            )
            self.assertTrue(
                any(
                    row.get("skip_samples", 0) > 0
                    for packet in probe["packets"]
                    for row in packet.get("side_data_list", [])
                )
            )
            window = {
                "window_id": "readable-fixture-course-window",
                "source_group": "synthetic-fixture",
                "publisher_recording_id": "aac-priming",
                "language": "en",
                "start_seconds": None,
            }
            with extracted_fixture(root, media, window) as result:
                duration = json.loads(
                    (root / "extracted/duration-resolution.json").read_text()
                )["duration_seconds"]
                self.assertEqual(result["source_start_seconds"], duration / 2 - 45)
                self.assertNotEqual(
                    result["source_start_seconds"], int(result["source_start_seconds"])
                )
                self.assertEqual(result["window_id"], window["window_id"])
                self.assertEqual(
                    result["source_id"],
                    acquisition.sealed.opaque_source_id(window["window_id"]),
                )
                self.assertEqual(UUID(result["source_id"]).version, 5)
                self.assertNotIn(window["window_id"], result["source_id"])
                with wave.open(str(root / "extracted/audio.wav"), "rb") as stream:
                    self.assertEqual(
                        (
                            stream.getframerate(),
                            stream.getnchannels(),
                            stream.getsampwidth(),
                            stream.getnframes(),
                        ),
                        (16000, 1, 2, 1440000),
                    )
                    for offset, frequency in ((1, 440), (88, 880)):
                        stream.setpos(offset * 16000)
                        samples = struct.unpack("<16000h", stream.readframes(16000))
                        crossings = sum(
                            a <= 0 < b for a, b in zip(samples, samples[1:])
                        )
                        self.assertAlmostEqual(crossings, frequency, delta=2)
                self.assertEqual(
                    json.loads((root / "ledger.json").read_text())["pcm_bytes"],
                    (root / "extracted/audio.wav").stat().st_size,
                )


if __name__ == "__main__":
    unittest.main()
