"""Local review workbench. Draft edits never mutate an editor project."""

import copy
import hashlib
import io
import json
import os
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

ASSETS = Path(__file__).parent / "review_ui"
ACTIONS = (
    "no_action",
    "correct_text",
    "delete_span",
    "trim_silence",
    "remove_silence",
    "split_paragraph",
    "merge_paragraph",
)


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal[
        "no_action",
        "correct_text",
        "delete_span",
        "trim_silence",
        "remove_silence",
        "split_paragraph",
        "merge_paragraph",
    ]
    target: list[int] = Field(default_factory=list, max_length=256)
    text: str | None = Field(default=None, max_length=2000)
    keep_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def parameters(self):
        if self.action == "no_action":
            if self.target or self.text is not None or self.keep_ms is not None:
                raise ValueError("No action has no target or parameters.")
        elif not self.target:
            raise ValueError("Choose a target.")
        if self.action == "correct_text":
            if not self.text or not self.text.strip():
                raise ValueError("A text correction needs replacement text.")
        elif self.text is not None:
            raise ValueError("Only text correction has replacement text.")
        if self.action == "trim_silence":
            if self.keep_ms is None or self.keep_ms <= 0:
                raise ValueError("Choose a positive retained duration, or remove silence.")
        elif self.keep_ms is not None:
            raise ValueError("Only silence trimming has a retained duration.")
        return self


def edit_dict(edit):
    result = edit.model_dump(exclude_none=True)
    if not result["target"]:
        del result["target"]
    return result


def validate_edit(case, edit):
    items = case["input"]["items"]
    if edit.action == "no_action":
        return []
    positions = {item["id"]: i for i, item in enumerate(items)}
    if any(target not in positions for target in edit.target):
        raise ValueError("A target is not in this snapshot.")
    indices = [positions[target] for target in edit.target]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Targets must be unique, ordered and contiguous.")
    if set(edit.target) & set(case["input"].get("protected_ids", [])):
        raise ValueError("A protected item cannot be changed.")
    selected = [items[i] for i in indices]
    if edit.action in ("split_paragraph", "merge_paragraph"):
        if len(selected) != 1 or indices[0] == 0:
            raise ValueError("Choose one interior boundary before an item.")
        previous, current = items[indices[0] - 1], selected[0]
        same = previous["paragraph"] == current["paragraph"]
        if edit.action == "split_paragraph" and not same:
            raise ValueError("There is already a caption boundary here.")
        if edit.action == "merge_paragraph":
            if same:
                raise ValueError("There is no caption boundary to merge here.")
            if any(previous.get(k) != current.get(k) for k in ("speaker", "language")):
                raise ValueError("Do not merge captions across speakers or languages.")
        affected = []
        for item in items[indices[0] :]:
            if item["paragraph"] != current["paragraph"]:
                break
            affected.append(item["id"])
        if set(affected) & set(case["input"].get("protected_ids", [])):
            raise ValueError("This caption change affects a protected item.")
        return []
    if (
        len(
            {
                (i["paragraph"], i.get("speaker"), i.get("language"), i["source_run"])
                for i in selected
            }
        )
        != 1
    ):
        raise ValueError("This edit crosses a paragraph, speaker, language or source run.")
    if edit.action == "correct_text":
        if not any(i["type"] == "text" for i in selected):
            raise ValueError("Select text to correct.")
        original = " ".join(i.get("text", "") for i in selected if i["type"] == "text")
        if edit.text == original:
            raise ValueError("The correction does not change the text.")
        return []
    if edit.action in ("trim_silence", "remove_silence"):
        if len(selected) != 1 or selected[0]["type"] != "non_text":
            raise ValueError("Select one non-text interval for silence review.")
        if edit.keep_ms is not None and edit.keep_ms >= selected[0]["duration_ms"]:
            raise ValueError("The retained pause must be shorter than the original.")
    elif not any(i["type"] == "text" for i in selected):
        raise ValueError("Speech deletion must include text.")
    warnings = []
    if any(i["boundary_status"] != "verified" for i in selected):
        warnings.append(
            "ASR boundaries are unverified. Audition this cut; it is not execution-ready."
        )
    if edit.action in ("trim_silence", "remove_silence"):
        if selected[0]["audio_class"] != "verified_silence":
            warnings.append("This gap may contain speech or a breath. Silence is not established.")
    return warnings


def preview_items(case, edit):
    """Materialize a display-only result, leaving source data unchanged."""
    validate_edit(case, edit)
    items = copy.deepcopy(case["input"]["items"])
    if edit.action == "no_action":
        return items
    indices = [next(i for i, item in enumerate(items) if item["id"] == t) for t in edit.target]
    first, last = indices[0], indices[-1]
    if edit.action in ("delete_span", "remove_silence"):
        del items[first : last + 1]
    elif edit.action == "correct_text":
        items[first]["text"] = edit.text
        items[first]["type"] = "text"
        items[first]["duration_ms"] = sum(i["duration_ms"] for i in items[first : last + 1])
        del items[first + 1 : last + 1]
    elif edit.action == "trim_silence":
        items[first]["duration_ms"] = edit.keep_ms
    elif edit.action == "split_paragraph":
        original = items[first]["paragraph"]
        for item in items[first:]:
            if item["paragraph"] != original:
                break
            item["paragraph"] = f"split-{original}"
    elif edit.action == "merge_paragraph":
        original, previous = items[first]["paragraph"], items[first - 1]["paragraph"]
        for item in items[first:]:
            if item["paragraph"] != original:
                break
            item["paragraph"] = previous
    return items


class Reveal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["assisted", "blind"]
    initial_judgment: Literal["edit_needed", "no_action", "unsure"] | None = None
    initial_note: str = Field(default="", max_length=2000)


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_hash: str
    base_revision: int = Field(ge=0)
    exposure_id: str
    verdict: Literal["acceptable", "reject", "adjust", "no_action", "unsure", "clear"]
    chosen_edit: Edit | None = None
    meaning: Literal["pass", "fail", "unsure"] | None = None
    audio: Literal["pass", "fail", "not_checked", "not_applicable"] = "not_checked"
    show_now: Literal["yes", "no", "unsure"] = "unsure"
    notes: str = Field(default="", max_length=4000)
    listened_before: bool = False
    listened_after: bool = False


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported review batch.")
        media_files = list(self.manifest.get("media", {}).values())
        media_files += list(self.manifest.get("videos", {}).values())
        for media in media_files:
            path = (self.root / media["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Media must remain inside the review batch.")
            with path.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != media["sha256"]:
                    raise ValueError("Review media hash mismatch.")
        rows = [
            json.loads(line)
            for line in (self.root / "cases.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.cases = {case["id"]: case for case in rows}
        if not rows or len(rows) != len(self.cases):
            raise ValueError("The batch needs unique, nonempty cases.")
        for case in rows:
            ids = [item["id"] for item in case["input"]["items"]]
            if len(ids) != len(set(ids)):
                raise ValueError("Item aliases must be unique.")
            validate_edit(case, Edit.model_validate(case["proposal"]))
        self.events_path = self.root / "reviews.jsonl"
        self.lock = threading.Lock()
        self.events = []
        if self.events_path.exists():
            for line in self.events_path.read_text().splitlines():
                if line.strip():
                    self.events.append(json.loads(line))

    def case(self, case_id):
        if case_id not in self.cases:
            raise HTTPException(404, "Unknown review case.")
        return self.cases[case_id]

    def history(self, case_id, kind):
        case_hash = digest(self.case(case_id))
        return [
            e
            for e in self.events
            if e["case_id"] == case_id and e["kind"] == kind and e["case_hash"] == case_hash
        ]

    def latest(self, case_id):
        labels = self.history(case_id, "label")
        return labels[-1] if labels else None

    def position(self, case):
        metadata = case["metadata"]
        cursor = str(case["input"].get("cursor_before"))
        return metadata.get(
            "position_s", metadata.get("source_items", {}).get(cursor, {}).get("start_s")
        )

    def video(self, case_id):
        case = self.case(case_id)
        audio = case["metadata"].get("audio", {})
        media = self.manifest.get("videos", {}).get(audio.get("media_id"))
        if media is None:
            return None
        path = (self.root / media["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Media must remain inside the review batch.")
        mime = {".mp4": "video/mp4", ".webm": "video/webm"}.get(path.suffix.lower())
        if mime is None:
            raise ValueError("Review video must be MP4 or WebM.")
        return path, mime, audio["start_s"], audio["end_s"]

    def append(self, kind, case_id, payload):
        event = {
            "kind": kind,
            "id": str(uuid4()),
            "case_id": case_id,
            "case_hash": digest(self.case(case_id)),
            "at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        self.events.append(event)
        return event

    def reveal(self, case_id, request):
        with self.lock:
            if request.mode == "blind" and request.initial_judgment is None:
                raise HTTPException(
                    422, "Record an independent judgment before revealing the draft."
                )
            independent = request.mode == "blind" and not self.history(case_id, "exposure")
            event = self.append(
                "exposure",
                case_id,
                {
                    **request.model_dump(),
                    "independent": independent,
                },
            )
            case = self.case(case_id)
            return {
                "exposure": event,
                "proposal": case["proposal"],
                "rationale": case["metadata"].get("proposal_note", ""),
                "warnings": validate_edit(case, Edit.model_validate(case["proposal"])),
            }

    def save(self, case_id, label):
        with self.lock:
            case = self.case(case_id)
            if label.case_hash != digest(case):
                raise HTTPException(409, "This case changed. Reload before labeling it.")
            latest = self.latest(case_id)
            revision = latest["revision"] if latest else 0
            if label.base_revision != revision:
                raise HTTPException(
                    409, "Another tab saved a label. Reload to avoid overwriting it."
                )
            exposure = next(
                (e for e in self.history(case_id, "exposure") if e["id"] == label.exposure_id), None
            )
            if not exposure:
                raise HTTPException(422, "Open the draft before reviewing it.")
            chosen = None
            if label.verdict == "acceptable":
                chosen = Edit.model_validate(case["proposal"])
            elif label.verdict == "adjust":
                if label.chosen_edit is None:
                    raise HTTPException(422, "Provide the adjusted edit.")
                chosen = label.chosen_edit
            elif label.verdict == "no_action":
                chosen = Edit(action="no_action")
            if chosen:
                validate_edit(case, chosen)
                cut = chosen.action in ("delete_span", "trim_silence", "remove_silence")
                if chosen.action != "no_action" and label.meaning != "pass":
                    raise HTTPException(
                        422, "Confirm meaning is preserved, or leave this case unsure."
                    )
                if cut and (
                    label.audio != "pass" or not label.listened_before or not label.listened_after
                ):
                    raise HTTPException(
                        422, "Listen to both versions and judge the cut before approving it."
                    )
            return self.append(
                "label",
                case_id,
                {
                    **label.model_dump(exclude={"case_hash", "chosen_edit"}),
                    "chosen_edit": edit_dict(chosen) if chosen else None,
                    "revision": revision + 1,
                    "independent": exposure["independent"],
                    "mode": exposure["mode"],
                    "initial_judgment": exposure["initial_judgment"],
                },
            )

    def audio(self, case_id, edit=None):
        case = self.case(case_id)
        audio = case["metadata"].get("audio")
        if not audio:
            raise HTTPException(404, "This case has no source audio.")
        media = self.manifest["media"][audio["media_id"]]
        path = (self.root / media["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Media must remain inside the review batch.")
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            start, end = audio["start_s"], audio["end_s"]
            if not 0 <= start < end <= source.getnframes() / rate + 1e-6 or end - start > 60:
                raise ValueError("Invalid review audio window.")
            ranges = [(round(start * rate), min(source.getnframes(), round(end * rate)))]
            if edit is not None:
                validate_edit(case, edit)
                if edit.action in ("delete_span", "trim_silence", "remove_silence"):
                    spans = [case["metadata"]["source_items"][str(t)] for t in edit.target]
                    if any(s["source"] != audio["source"] for s in spans):
                        raise ValueError("The audio target references a different source.")
                    if any(abs(a["end_s"] - b["start_s"]) > 1e-6 for a, b in zip(spans, spans[1:])):
                        raise ValueError("The cut crosses a source discontinuity.")
                    left, right = (
                        round(spans[0]["start_s"] * rate),
                        round(spans[-1]["end_s"] * rate),
                    )
                    if not ranges[0][0] <= left < right <= ranges[0][1]:
                        raise ValueError("The cut lies outside the excerpt.")
                    keep = round((edit.keep_ms or 0) * rate / 1000)
                    if keep >= right - left:
                        raise ValueError("The retained audio must be shorter than the source span.")
                    ranges = [
                        (ranges[0][0], left + keep // 2),
                        (right - (keep - keep // 2), ranges[0][1]),
                    ]
            output = io.BytesIO()
            with wave.open(output, "wb") as target:
                target.setparams(source.getparams())
                for left, right in ranges:
                    source.setpos(left)
                    target.writeframes(source.readframes(right - left))
            return output.getvalue()


def create_review_app(root):
    store = ReviewStore(root)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.review = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def local_browser_only(request: Request, call_next):
        origin = request.headers.get("origin")
        expected = str(request.base_url).rstrip("/")
        if origin and origin != expected:
            return JSONResponse({"detail": "Only the local review page may access this API."}, 403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site requests are not allowed."}, 403)
        if request.method not in ("GET", "HEAD"):
            if request.headers.get("x-review-request") != "1":
                return JSONResponse({"detail": "Missing local review request header."}, 403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "media-src 'self' blob:; img-src 'self'; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
        return response

    @app.exception_handler(ValueError)
    async def invalid_review(request, exc):
        return JSONResponse({"detail": str(exc)}, 422)

    @app.get("/")
    def index():
        return FileResponse(ASSETS / "index.html")

    @app.get("/assets/{name}")
    def asset(name: Literal["review.js", "review.css"]):
        return FileResponse(ASSETS / name)

    @app.get("/api/batch")
    def batch():
        rows = []
        for case_id, case in store.cases.items():
            label = store.latest(case_id)
            rows.append(
                {
                    "id": case_id,
                    "position_s": store.position(case),
                    "verdict": label["verdict"] if label else None,
                }
            )
        return {"title": store.manifest["title"], "note": store.manifest["note"], "cases": rows}

    @app.get("/api/cases/{case_id}")
    def get_case(case_id):
        case = store.case(case_id)
        latest = store.latest(case_id)
        video = store.video(case_id)
        return {
            "id": case_id,
            "case_hash": digest(case),
            "input": case["input"],
            "position_s": store.position(case),
            "context_origin": case["metadata"]["context_origin"],
            "has_observed_action": case["metadata"].get("observed_next_action") is not None,
            "has_audio": bool(case["metadata"].get("audio")),
            "video": {"start_s": video[2], "end_s": video[3]} if video else None,
            "revision": latest["revision"] if latest else 0,
            "reviewed": bool(latest and latest["verdict"] != "clear"),
            "previously_exposed": bool(store.history(case_id, "exposure")),
        }

    @app.post("/api/cases/{case_id}/reveal")
    def reveal(case_id, request: Reveal):
        return store.reveal(case_id, request)

    @app.post("/api/cases/{case_id}/preview")
    def preview(case_id, edit: Edit):
        case = store.case(case_id)
        return {"items": preview_items(case, edit), "warnings": validate_edit(case, edit)}

    @app.get("/api/cases/{case_id}/audio")
    def before_audio(case_id):
        return Response(store.audio(case_id), media_type="audio/wav")

    @app.get("/api/cases/{case_id}/video")
    def source_video(case_id):
        video = store.video(case_id)
        if video is None:
            raise HTTPException(404, "This case has no source video.")
        return FileResponse(video[0], media_type=video[1])

    @app.post("/api/cases/{case_id}/audio")
    def after_audio(case_id, edit: Edit):
        return Response(store.audio(case_id, edit), media_type="audio/wav")

    @app.post("/api/cases/{case_id}/labels")
    def label_case(case_id, label: Label):
        return store.save(case_id, label)

    @app.get("/api/cases/{case_id}/labels")
    def label_history(case_id):
        return store.history(case_id, "label")

    @app.get("/api/export")
    def export():
        rows = []
        for case_id, case in store.cases.items():
            label = store.latest(case_id)
            if label is None or label["verdict"] == "clear":
                continue
            rows.append(
                {
                    "id": case_id,
                    "input": case["input"],
                    "expectedOutput": None,
                    "metadata": {
                        **case["metadata"],
                        "review": label,
                        "draft_proposal": case["proposal"],
                        "action_schema": "review-v1",
                        "case_hash": digest(case),
                        "target_origin": "human_review_feedback",
                        "requires_regeneration": True,
                        "training_eligible": False,
                    },
                }
            )
        return Response(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="review-feedback.jsonl"'},
        )

    return app
