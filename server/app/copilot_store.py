"""Local append-only evidence for copilot requests and observed outcomes."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import platformdirs

if os.name == "nt":
    import msvcrt
else:
    import fcntl

SCHEMA = "copilot-feedback-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def evidence_hash(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


class EvidenceError(Exception):
    pass


def lock(stream, exclusive):
    if os.name == "nt":
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if exclusive else msvcrt.LK_RLCK, 1)
    else:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


class CopilotEvidenceStore:
    def __init__(self, directory=None):
        self.directory = (
            Path(directory)
            if directory is not None
            else Path(os.environ.get("AUDAPOLIS_DATA_DIR", platformdirs.user_data_dir("audapolis")))
            / "copilot-feedback"
        )
        self._verified = {}

    def path(self, document_id):
        return self.directory / f"{hashlib.sha256(document_id.encode('utf-8')).hexdigest()}.jsonl"

    @staticmethod
    def _signature(stream):
        stat = os.fstat(stream.fileno())
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _remember(self, path, stream, count, head):
        self._verified[path] = (self._signature(stream), count, head)

    def _head(self, path, stream, document_id):
        cached = self._verified.get(path)
        if cached is not None and cached[0] == self._signature(stream):
            return cached[1:]
        stream.seek(0)
        count, head = 0, None
        for record in self._iter_records(stream, document_id):
            count, head = record["seq"] + 1, record["event_hash"]
        return count, head

    def confirm(self, document_id, expected_head):
        try:
            path = self.path(document_id)
            with path.open("r+b") as stream:
                lock(stream, True)
                count, head = self._head(path, stream, document_id)
                if head != expected_head:
                    raise EvidenceError(
                        "Another writer changed this copilot session; start a new session."
                    )
                os.fsync(stream.fileno())
                self._remember(path, stream, count, head)
        except OSError as error:
            raise EvidenceError("Cannot confirm durable local copilot evidence.") from error

    def _iter_records(self, stream, document_id=None):
        previous = None
        try:
            for seq, line in enumerate(stream):
                if not line.endswith(b"\n"):
                    raise EvidenceError(
                        "Copilot evidence has an incomplete last record; preserve the file."
                    )
                record = json.loads(line)
                digest = record.pop("event_hash")
                if document_id is None:
                    document_id = record["document_id"]
                if (
                    record["schema"] != SCHEMA
                    or record["seq"] != seq
                    or record["previous_event_hash"] != previous
                    or evidence_hash(record) != digest
                    or (document_id is not None and record["document_id"] != document_id)
                ):
                    raise EvidenceError(
                        "Copilot evidence hash or event order differs; preserve the file."
                    )
                previous = digest
                yield {**record, "event_hash": digest}
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise EvidenceError("Copilot evidence is unreadable; preserve the file.") from error

    def _read(self, stream, document_id=None):
        records = list(self._iter_records(stream, document_id))
        stream.seek(0)
        raw = stream.read()
        return records, raw

    def scan(self, document_id, consume):
        try:
            path = self.path(document_id)
            with path.open("rb") as stream:
                lock(stream, False)
                count, head = 0, None
                for record in self._iter_records(stream, document_id):
                    consume(record)
                    count, head = record["seq"] + 1, record["event_hash"]
                self._remember(path, stream, count, head)
        except FileNotFoundError:
            return
        except OSError as error:
            raise EvidenceError("Cannot read local copilot evidence.") from error

    def read(self, document_id):
        try:
            with self.path(document_id).open("rb") as stream:
                lock(stream, False)
                records, raw = self._read(stream, document_id)
                self._remember(
                    self.path(document_id),
                    stream,
                    len(records),
                    records[-1]["event_hash"] if records else None,
                )
                return records, raw
        except FileNotFoundError:
            return [], b""
        except OSError as error:
            raise EvidenceError("Cannot read local copilot evidence.") from error

    def append(self, document_id, fields, expected_head):
        try:
            new_directory = not self.directory.exists()
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = self.path(document_id)
            with os.fdopen(
                os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600), "a+b"
            ) as stream:
                lock(stream, True)
                count, previous = self._head(path, stream, document_id)
                if previous != expected_head:
                    raise EvidenceError(
                        "Another writer changed this copilot session; start a new session."
                    )
                record = {
                    **fields,
                    "schema": SCHEMA,
                    "document_id": document_id,
                    "seq": count,
                    "recorded_at": now(),
                    "previous_event_hash": previous,
                }
                record["event_hash"] = evidence_hash(record)
                stream.write((encoded(record) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
                # Persist the directory entry as well when this is the first record.
                if not count and os.name != "nt":
                    for path_ in [
                        self.directory,
                        *([self.directory.parent] if new_directory else []),
                    ]:
                        directory = os.open(path_, os.O_RDONLY)
                        try:
                            os.fsync(directory)
                        finally:
                            os.close(directory)
                self._remember(path, stream, count + 1, record["event_hash"])
                return record
        except OSError as error:
            raise EvidenceError("Cannot durably save local copilot evidence.") from error

    def sessions(self):
        result = []
        for path in sorted(self.directory.glob("*.jsonl")):
            try:
                with path.open("rb") as stream:
                    lock(stream, False)
                    records, _ = self._read(stream)
                if not records:
                    continue
                document_id = records[0]["document_id"]
                if self.path(document_id) != path:
                    raise EvidenceError("Copilot evidence filename differs from its session.")
                pending = set()
                proposals = {}
                failed = {
                    event["request"]["request_id"]: event["error"]["code"]
                    for event in records
                    if event["kind"] == "prediction_failed"
                }
                for event in records:
                    kind = event["kind"]
                    if kind == "prediction_requested":
                        pending.add(event["request"]["request_id"])
                    elif kind in {"prediction_completed", "prediction_failed"}:
                        pending.discard(event["request"]["request_id"])
                    if kind == "prediction_completed" and event["response"]["proposal_id"]:
                        proposals[event["response"]["proposal_id"]] = (
                            "request_failed"
                            if event["request"]["request_id"] in failed
                            else "unobserved"
                        )
                    elif kind == "feedback":
                        proposals[event["feedback"]["proposal_id"]] = event["feedback"]["outcome"]
                result.append(
                    {
                        "document_id": document_id,
                        "filename": path.name,
                        "status": "verified",
                        "events": len(records),
                        "head_hash": records[-1]["event_hash"],
                        "created_at": records[0]["recorded_at"],
                        "updated_at": records[-1]["recorded_at"],
                        "unfinished_requests": sorted(pending),
                        "failed_requests": failed,
                        "proposal_outcomes": proposals,
                    }
                )
            except (EvidenceError, OSError, KeyError, TypeError) as error:
                result.append({"filename": path.name, "status": "unreadable", "error": str(error)})
        return result
