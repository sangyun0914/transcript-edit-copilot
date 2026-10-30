"""Gated fixed-window media acquisition and single-attempt raw Scribe capture."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlsplit
import wave

import httpx

import prepare_task_test_actions_v3 as sealed

ROOT = sealed.ROOT
VERSION = "task-test-actions-v3-acquisition-1"
ENCODED_LIMIT = 512 * 1024 * 1024
PCM_LIMIT = 64 * 1024 * 1024
require, read, write, binding = sealed.require, sealed.read_json, sealed.write, sealed.binding


def now():
    return datetime.now(timezone.utc).isoformat()


def fresh_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and not path.exists(), "Use a fresh workspace output.")
    path.mkdir(parents=True)
    return path


@contextmanager
def ledger(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read(path) if path.exists() else {"encoded_bytes": 0, "pcm_bytes": 0, "asr_attempts": {}}
        yield data
        temporary = path.with_suffix(".pending")
        with temporary.open("w") as target:
            json.dump(data, target, ensure_ascii=False, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)


def ledger_path(gate):
    return ROOT / "artifacts/evaluation/passive-actions-v3/acquisition-ledgers" / (gate["binding"]["sha256"] + ".json")


def charge_bytes(path, kind, count):
    require(type(count) is int and count >= 0, "Invalid acquisition byte count.")
    key, limit = ("encoded_bytes", ENCODED_LIMIT) if kind == "encoded" else ("pcm_bytes", PCM_LIMIT)
    with ledger(path) as data:
        data[key] += count
        total = data[key]
    require(total <= limit, f"Frozen {kind} acquisition budget exceeded; retain received bytes and stop.")


def reserve_asr(path, window_id, pcm_sha256, output):
    with ledger(path) as data:
        require(window_id not in data["asr_attempts"],
                "This window already has a potentially billable attempt; automatic retry is forbidden.")
        data["asr_attempts"][window_id] = {
            "reserved_at": now(), "pcm_sha256": pcm_sha256, "output": str(output),
            "status": "reserved_before_network_outcome_unknown",
        }


def public_media_url(url):
    parsed = urlsplit(url)
    require(parsed.scheme in {"https", "http"} and parsed.hostname and not parsed.username
            and not parsed.password and parsed.hostname not in {"localhost", "127.0.0.1", "::1"},
            "Use the registered public publisher media URL without credentials.")
    return url


def remote_response(client, url, method, range_value=None):
    headers = {"Accept-Encoding": "identity"}
    if range_value:
        require(range_value.startswith("bytes=") and "," not in range_value, "Only a single media byte range is supported.")
        headers["Range"] = range_value
    return client.stream(method, url, headers=headers, follow_redirects=True)


class MediaProxy:
    """Meter only one bound upstream object; FFmpeg can seek without a full download."""

    def __init__(self, url, budget_path, output):
        self.url, self.budget_path, self.output = public_media_url(url), budget_path, output
        charge_bytes(budget_path, "encoded", 0)
        self.errors, self.total_bytes = [], 0
        self.lock = threading.Lock()
        self.client = httpx.Client(timeout=httpx.Timeout(30, connect=10), trust_env=False)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.transfer("HEAD")

            def do_GET(self):
                self.transfer("GET")

            def transfer(self, method):
                if self.path != "/media":
                    self.send_error(404)
                    return
                record = {"started_at": now(), "method": method, "range": self.headers.get("Range"),
                          "bytes_forwarded": 0, "bytes_received": 0}
                digest = hashlib.sha256()
                headers_sent = False
                try:
                    with remote_response(proxy.client, proxy.url, method, record["range"]) as response:
                        public_media_url(str(response.url))
                        response.raise_for_status()
                        require(response.status_code in {200, 206}, "Unexpected media response.")
                        require(response.headers.get("content-encoding", "identity") == "identity",
                                "Compressed transfer is incompatible with byte-range provenance.")
                        record.update(status=response.status_code, final_url=str(response.url),
                                      content_range=response.headers.get("content-range"),
                                      content_length=response.headers.get("content-length"))
                        self.send_response(response.status_code)
                        for name in ("content-type", "content-length", "content-range", "accept-ranges"):
                            if name in response.headers:
                                self.send_header(name, response.headers[name])
                        self.send_header("Connection", "close")
                        self.end_headers()
                        headers_sent = True
                        if method == "GET":
                            for chunk in response.iter_raw(65536):
                                record["bytes_received"] += len(chunk)
                                digest.update(chunk)
                                charge_bytes(proxy.budget_path, "encoded", len(chunk))
                                self.wfile.write(chunk)
                                record["bytes_forwarded"] += len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    record["client_finished_range"] = True
                except Exception as error:
                    record.update(error_type=type(error).__name__, error=str(error))
                    with proxy.lock:
                        proxy.errors.append(record)
                    if not headers_sent:
                        self.send_error(502)
                finally:
                    record.update(finished_at=now(), received_sha256=digest.hexdigest())
                    with proxy.lock:
                        proxy.total_bytes += record["bytes_received"]
                        with (proxy.output / "range-transfers.jsonl").open("a") as target:
                            target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}/media"

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.client.close()


def verify_pcm(path):
    with wave.open(str(path), "rb") as audio:
        require((audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes())
                == (16000, 1, 2, 90 * 16000), "A complete 90-second 16-kHz mono PCM16 excerpt is required.")


def run_media_command(argv, output, name):
    try:
        result = subprocess.run(argv, capture_output=True, timeout=180, check=False)
    except subprocess.TimeoutExpired as error:
        (output / (name + ".stdout")).write_bytes(error.stdout or b"")
        (output / (name + ".stderr")).write_bytes(error.stderr or b"")
        write(output / (name + ".command.json"), {"argv": argv, "returncode": None, "timed_out": True})
        raise
    (output / (name + ".stdout")).write_bytes(result.stdout)
    (output / (name + ".stderr")).write_bytes(result.stderr)
    write(output / (name + ".command.json"), {"argv": argv, "returncode": result.returncode})
    require(result.returncode == 0, f"{name} failed; retain the acquisition attempt.")
    return result.stdout


def extract(freeze, media_binding, output, *, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    gate = sealed.load_gate(freeze)
    source = read(media_binding)
    window = sealed.verify_window_identity(gate, source)
    require(source["freeze_sha256"] == gate["binding"]["sha256"]
            and source["selection_before_words"] is True
            and source["model_predictions_seen"] is False, "Media resolution changed the frozen allocation.")
    for evidence in source["resolution_evidence"]:
        sealed.bound(evidence)
    require(bool(source["resolution_evidence"]), "Publisher media resolution needs bound evidence.")
    output = fresh_output(output)
    write(output / "source-binding.json", source)
    info = {"version": VERSION, "status": "extracting", "freeze": gate["binding"],
            "media_binding": binding(media_binding), "started_at": now(), "window_id": window["window_id"]}
    write(output / "run-registration.json", info)
    pcm_charged = False
    try:
        proxy = MediaProxy(source["media_url"], ledger_path(gate), output)
        with proxy as url:
            probe = json.loads(run_media_command(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", url], output, "ffprobe"))
            duration = float(probe["format"]["duration"])
            require(math.isfinite(duration) and duration >= 180, "Recording is too short or has unknown duration.")
            start = window["start_seconds"] if window["start_seconds"] is not None else duration / 2 - 45
            require(start >= 0 and start + 90 <= duration + 0.001, "Registered excerpt exceeds the recording.")
            resolution = {"recording_id": source["recording_id"], "duration_seconds": duration,
                          "before_words": True, "resolved_at": now(), "probe": binding(output / "ffprobe.stdout")}
            write(output / "duration-resolution.json", resolution)
            run_media_command([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-ss", str(start),
                               "-i", url, "-t", "90", "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                               "-c:a", "pcm_s16le", "-n", str(output / "audio.wav")], output, "ffmpeg")
        require(not proxy.errors, "Upstream transfer failed; do not admit a partial excerpt.")
        pcm_charged = True
        charge_bytes(ledger_path(gate), "pcm", (output / "audio.wav").stat().st_size)
        verify_pcm(output / "audio.wav")
        info.update(status="pcm_window_complete", source_group=source["source_group"], language=window["language"],
                    recording_id=source["recording_id"], source_id=sealed.opaque_source_id(window["window_id"]),
                    registered_source_group=window["registered_source_group"],
                    registered_recording_id=window["registered_publisher_recording_id"],
                    source_resolution=window["source_resolution"],
                    source_allocation_status=window["source_allocation_status"],
                    duration_seconds=90, source_start_seconds=start,
                    pcm_audio=binding(output / "audio.wav"), duration_resolution=binding(output / "duration-resolution.json"),
                    encoded_bytes_received=proxy.total_bytes)
    except Exception as error:
        info.update(status="acquisition_failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if not pcm_charged and (output / "audio.wav").exists():
            try:
                charge_bytes(ledger_path(gate), "pcm", (output / "audio.wav").stat().st_size)
            except ValueError as error:
                info["additional_budget_error"] = str(error)
        info["finished_at"] = now()
        write(output / "receipt.json", info)
    return info


def scribe_options(language):
    require(language in {"en", "ko"}, "Unregistered ASR language.")
    return {"model_id": "scribe_v2", "no_verbatim": "false", "timestamps_granularity": "word",
            "diarize": "true", "tag_audio_events": "true", "language_code": language}


def transcribe(freeze, acquisition, output):
    gate = sealed.load_gate(freeze)
    acquired = read(acquisition)
    require(acquired["status"] == "pcm_window_complete" and acquired["freeze"] == gate["binding"],
            "ASR requires a complete excerpt acquired under this exact freeze.")
    window = sealed.verify_window_identity(gate, acquired)
    require(acquired.get("source_id") == sealed.opaque_source_id(window["window_id"]),
            "ASR requires the opaque source identity for this registered window.")
    audio = sealed.bound(acquired["pcm_audio"])
    verify_pcm(audio)
    options = scribe_options(window["language"])
    api_key = os.environ.get("ELEVENLABS_PERSONAL_API_KEY", "").strip()
    require(bool(api_key), "The existing Scribe credential is unavailable.")
    output = fresh_output(output)
    info = {"version": VERSION, "status": "reserved", "freeze": gate["binding"],
            "acquisition": binding(acquisition), "window_id": window["window_id"],
            "request_options": options, "request_started_at": now(), "automatic_retry": False}
    write(output / "run-registration.json", info)
    reserve_asr(ledger_path(gate), window["window_id"], acquired["pcm_audio"]["sha256"], output)
    try:
        with audio.open("rb") as stream, httpx.Client(timeout=httpx.Timeout(600, connect=10, write=120, pool=10),
                                                      trust_env=False) as client:
            response = client.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key": api_key},
                                   data=options, files={"file": ("audio.wav", stream, "audio/wav")})
        (output / "raw-response.json").write_bytes(response.content)
        info.update(http_status=response.status_code, response_received_at=now(), raw_response=binding(output / "raw-response.json"))
        response.raise_for_status()
        raw = response.json()
        provenance = {key: acquired[key] for key in ("window_id", "source_group", "language", "recording_id", "source_id",
                                                     "duration_seconds", "source_start_seconds", "pcm_audio", "duration_resolution",
                                                     "registered_source_group", "registered_recording_id", "source_resolution",
                                                     "source_allocation_status")}
        provenance.update(split="final_test", freeze_sha256=gate["binding"]["sha256"],
                          request_options=options, request_started_at=info["request_started_at"],
                          raw_response_sha256=info["raw_response"]["sha256"], acquisition=binding(acquisition))
        write(output / "provenance.json", provenance)
        require(isinstance(raw, dict), "Scribe returned a non-object response; preserve it without repair.")
        info.update(status="raw_asr_captured_import_pending", provenance=binding(output / "provenance.json"))
    except Exception as error:
        info.update(status="asr_failed_no_automatic_retry", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        info["finished_at"] = now()
        write(output / "receipt.json", info)
        with ledger(ledger_path(gate)) as data:
            data["asr_attempts"][window["window_id"]].update(status=info["status"], receipt=binding(output / "receipt.json"))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "transcribe"):
        command = commands.add_parser(name)
        command.add_argument("--freeze", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "extract":
            command.add_argument("--media-binding", required=True, type=Path)
            command.add_argument("--ffmpeg", default="ffmpeg")
            command.add_argument("--ffprobe", default="ffprobe")
        else:
            command.add_argument("--acquisition", required=True, type=Path)
    values = vars(parser.parse_args())
    command = values.pop("command")
    result = globals()[command](**values)
    print(json.dumps({"status": result["status"], "window_id": result["window_id"]}, indent=2))


if __name__ == "__main__":
    main()
