import copy
import hashlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.review import Edit, Label, Reveal, ReviewStore, create_review_app, digest, preview_items


def fixture(root):
    media = root / "source.wav"
    with wave.open(str(media), "wb") as audio:
        audio.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * 12000)
    items = []
    for index, text in enumerate(("This", "", "is", "a", "test", "caption")):
        items.append(
            {
                "id": index,
                "type": "text" if text else "non_text",
                "text": text,
                "duration_ms": 2000,
                "paragraph": 0 if index < 4 else 1,
                "speaker": "speaker",
                "language": "en",
                "source_run": 0,
                "source_backed": True,
                "boundary_status": "unverified",
                "audio_class": "speech" if text else "unknown",
            }
        )
    case = {
        "id": "test-case",
        "input": {"items": items, "protected_ids": [], "recent_edits": []},
        "expectedOutput": None,
        "proposal": {"action": "delete_span", "target": [0]},
        "metadata": {
            "context_origin": "test_fixture",
            "audio": {"media_id": "source", "source": "s", "start_s": 0, "end_s": 12},
            "source_items": {
                str(i): {"source": "s", "start_s": 2 * i, "end_s": 2 * i + 2} for i in range(6)
            },
        },
    }
    manifest = {
        "schema_version": 1,
        "title": "Test batch",
        "note": "Generated test fixture, no human labels.",
        "media": {
            "source": {
                "path": "source.wav",
                "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "cases.jsonl").write_text(json.dumps(case) + "\n")
    return case


class ReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.case = fixture(self.root)
        self.store = ReviewStore(self.root)

    def label(self, **changes):
        exposure = self.store.reveal("test-case", Reveal(mode="assisted"))["exposure"]
        fields = {
            "case_hash": digest(self.case),
            "base_revision": 0,
            "exposure_id": exposure["id"],
            "verdict": "unsure",
        }
        return Label(**(fields | changes))

    def test_full_review_scope_and_non_mutating_preview(self):
        original = copy.deepcopy(self.case)
        edits = [
            {"action": "no_action"},
            {"action": "correct_text", "target": [0, 1, 2, 3], "text": "These are"},
            {"action": "delete_span", "target": [0, 1, 2, 3]},
            {"action": "trim_silence", "target": [1], "keep_ms": 725},
            {"action": "remove_silence", "target": [1]},
            {"action": "split_paragraph", "target": [3]},
            {"action": "merge_paragraph", "target": [4]},
        ]
        for edit in edits:
            with self.subTest(edit=edit):
                self.assertIsInstance(preview_items(self.case, Edit(**edit)), list)
        self.assertEqual(self.case, original)
        merged = preview_items(self.case, Edit(**edits[-1]))
        self.assertEqual({i["paragraph"] for i in merged}, {0})
        corrected = preview_items(self.case, Edit(**edits[1]))
        self.assertEqual((corrected[0]["text"], corrected[0]["duration_ms"]), ("These are", 8000))

    def test_invalid_targets_and_boundaries_are_rejected(self):
        for edit in (
            {"action": "delete_span", "target": [0, 2]},
            {"action": "delete_span", "target": [0, 0]},
            {"action": "delete_span", "target": [99]},
            {"action": "delete_span", "target": [3, 4]},
            {"action": "trim_silence", "target": [1], "keep_ms": 2000},
            {"action": "remove_silence", "target": [2]},
            {"action": "split_paragraph", "target": [4]},
            {"action": "merge_paragraph", "target": [2]},
            {"action": "correct_text", "target": [2], "text": "is"},
            {"action": "no_action", "target": [0]},
        ):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                preview_items(self.case, Edit(**edit))
        self.case["input"]["items"][4]["speaker"] = "another"
        with self.assertRaises(ValueError):
            preview_items(self.case, Edit(action="merge_paragraph", target=[4]))
        self.case["input"]["protected_ids"] = [0]
        with self.assertRaises(ValueError):
            preview_items(self.case, Edit(action="delete_span", target=[0]))
        self.case["input"]["protected_ids"] = [3]
        with self.assertRaises(ValueError):
            preview_items(self.case, Edit(action="split_paragraph", target=[2]))

    def test_audio_previews_preserve_source_and_exact_retained_duration(self):
        original = (self.root / "source.wav").read_bytes()
        before = self.store.audio("test-case")
        correction = self.store.audio(
            "test-case", Edit(action="correct_text", target=[0], text="That")
        )
        self.assertEqual(before, correction)
        for action, keep, expected in (
            ("trim_silence", 725, 10725),
            ("remove_silence", None, 10000),
        ):
            result = self.store.audio("test-case", Edit(action=action, target=[1], keep_ms=keep))
            with wave.open(io.BytesIO(result), "rb") as audio:
                self.assertEqual(audio.getnframes(), expected)
        self.assertEqual((self.root / "source.wav").read_bytes(), original)
        self.store.cases["test-case"]["metadata"]["source_items"]["1"]["start_s"] += 0.1
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            self.store.audio("test-case", Edit(action="delete_span", target=[0, 1]))

    def test_cut_approval_requires_meaning_and_audition_but_keeps_uncertainty(self):
        label = self.label(verdict="acceptable", meaning="pass", audio="pass")
        with self.assertRaises(HTTPException):
            self.store.save("test-case", label)
        label.listened_before = label.listened_after = True
        saved = self.store.save("test-case", label)
        self.assertEqual(saved["chosen_edit"], self.case["proposal"])
        self.assertEqual(
            self.store.case("test-case")["input"]["items"][0]["boundary_status"], "unverified"
        )

    def test_reviews_resume_and_revision_conflicts_do_not_overwrite(self):
        first = self.label(verdict="unsure")
        self.store.save("test-case", first)
        restarted = ReviewStore(self.root)
        self.assertIsNone(restarted.latest("test-case")["chosen_edit"])
        with self.assertRaises(HTTPException) as error:
            restarted.save("test-case", first)
        self.assertEqual(error.exception.status_code, 409)
        changed = first.model_copy(update={"base_revision": 1, "case_hash": "stale"})
        with self.assertRaises(HTTPException):
            restarted.save("test-case", changed)
        saved = restarted.save(
            "test-case", first.model_copy(update={"base_revision": 1, "verdict": "no_action"})
        )
        self.assertEqual(saved["chosen_edit"], {"action": "no_action"})
        self.assertEqual(len(restarted.history("test-case", "label")), 2)

    def test_blind_exposure_cannot_be_reset_by_reload_or_mode_switch(self):
        request = Reveal(mode="blind", initial_judgment="unsure")
        self.assertTrue(self.store.reveal("test-case", request)["exposure"]["independent"])
        restarted = ReviewStore(self.root)
        self.assertFalse(restarted.reveal("test-case", request)["exposure"]["independent"])
        with self.assertRaises(HTTPException):
            restarted.reveal("test-case", Reveal(mode="blind"))

    def test_api_blinding_export_clear_and_cross_site_protection(self):
        with TestClient(create_review_app(self.root), base_url="http://127.0.0.1") as client:
            case = client.get("/api/cases/test-case").json()
            self.assertNotIn("proposal", case)
            self.assertNotIn("rationale", case)
            self.assertEqual(
                client.post("/api/cases/test-case/reveal", json={"mode": "assisted"}).status_code,
                403,
            )
            self.assertEqual(
                client.get("/api/batch", headers={"Origin": "https://example.com"}).status_code, 403
            )
            self.assertEqual(
                client.get("/api/batch", headers={"Host": "example.com"}).status_code, 400
            )
            headers = {"X-Review-Request": "1"}
            exposure = client.post(
                "/api/cases/test-case/reveal", json={"mode": "assisted"}, headers=headers
            ).json()["exposure"]
            label = {
                "case_hash": case["case_hash"],
                "base_revision": 0,
                "exposure_id": exposure["id"],
                "verdict": "reject",
            }
            self.assertEqual(
                client.post("/api/cases/test-case/labels", json=label, headers=headers).status_code,
                200,
            )
            exported = json.loads(client.get("/api/export").text)
            self.assertIsNone(exported["expectedOutput"])
            self.assertEqual(exported["metadata"]["draft_proposal"], self.case["proposal"])
            label.update(base_revision=1, verdict="clear")
            client.post(
                "/api/cases/test-case/labels", json=label, headers=headers
            ).raise_for_status()
            self.assertEqual(client.get("/api/export").text, "")
            self.assertEqual(len(client.get("/api/cases/test-case/labels").json()), 2)

    def test_changed_media_and_malformed_logs_fail_visibly(self):
        (self.root / "reviews.jsonl").write_text('{"kind":')
        with self.assertRaises(json.JSONDecodeError):
            ReviewStore(self.root)
        (self.root / "source.wav").write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ReviewStore(self.root)

    def test_approved_human_feedback_is_never_exported_as_a_training_target(self):
        with TestClient(create_review_app(self.root), base_url="http://127.0.0.1") as client:
            headers = {"X-Review-Request": "1"}
            case = client.get("/api/cases/test-case").json()
            exposure = client.post(
                "/api/cases/test-case/reveal", json={"mode": "assisted"}, headers=headers
            ).json()["exposure"]
            client.post(
                "/api/cases/test-case/labels",
                headers=headers,
                json={
                    "case_hash": case["case_hash"],
                    "base_revision": 0,
                    "exposure_id": exposure["id"],
                    "verdict": "adjust",
                    "chosen_edit": {"action": "correct_text", "target": [0], "text": "That"},
                    "meaning": "pass",
                },
            ).raise_for_status()
            row = json.loads(client.get("/api/export").text)
            self.assertIsNone(row["expectedOutput"])
            self.assertTrue(row["metadata"]["requires_regeneration"])
            self.assertFalse(row["metadata"]["training_eligible"])
            self.assertEqual(row["metadata"]["review"]["chosen_edit"]["text"], "That")

    def test_optional_video_supports_seeking_without_invalidating_review(self):
        self.store.save("test-case", self.label())
        with TestClient(create_review_app(self.root), base_url="http://127.0.0.1") as client:
            self.assertIsNone(client.get("/api/cases/test-case").json()["video"])
            self.assertEqual(client.get("/api/cases/test-case/video").status_code, 404)
        media = self.root / "source.mp4"
        media.write_bytes(b"video-range-fixture")
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["videos"] = {
            "source": {"path": media.name, "sha256": hashlib.sha256(media.read_bytes()).hexdigest()}
        }
        manifest_path.write_text(json.dumps(manifest))
        with TestClient(create_review_app(self.root), base_url="http://127.0.0.1") as client:
            case = client.get("/api/cases/test-case").json()
            self.assertTrue(case["reviewed"])
            self.assertEqual(case["case_hash"], digest(self.case))
            self.assertEqual(case["video"], {"start_s": 0, "end_s": 12})
            response = client.get("/api/cases/test-case/video", headers={"Range": "bytes=0-4"})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b"video")
            self.assertEqual(response.headers["content-type"], "video/mp4")
        media.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ReviewStore(self.root)


if __name__ == "__main__":
    unittest.main()
