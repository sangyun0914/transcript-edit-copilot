"""Synthetic metadata faults only; no publisher body, media, label or prediction."""

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import prepare_task_test_actions_v3 as sealed


class SourceResolutionTests(unittest.TestCase):
    def setUp(self):
        (sealed.ROOT / "artifacts").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="source-resolution-engineering-", dir=sealed.ROOT / "artifacts"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        groups = [*sealed.KO_COHORTS, sealed.KO_FALLBACK]
        reservations = {
            "groups": [
                {
                    "source_group": group,
                    "language": "ko",
                    "role": "metadata_failure_fallback_only"
                    if group == sealed.KO_FALLBACK
                    else "primary",
                }
                for group in groups
            ],
            "exposed_source_group_splits": {"engineering-train": "train"},
        }
        windows = [
            {
                "window_id": f"engineering-{index}-{i}",
                "source_group": group,
                "language": "ko",
                "publisher_recording_id": f"engineering-publisher-{index}-{i}",
                "window_seconds": 90,
                "body_opened": False,
                "start_seconds": None,
                "end_seconds": None,
                "center_seconds": None,
            }
            for index, group in enumerate(sealed.KO_COHORTS)
            for i in range(4)
        ]
        fallback = {
            "source_group": sealed.KO_FALLBACK,
            "role": "metadata_failure_fallback_only",
            "first_eight_metadata_candidates": [
                {
                    "publisher_recording_id": f"engineering-fallback-{i}",
                    "body_sha256": None,
                    "media_and_caption_bodies_requested": False,
                }
                for i in range(4)
            ],
        }
        self.write("source-reservations-proposed.json", reservations)
        self.write("natural-window-proposal.json", windows)
        self.write("kocw-1532697-recording-order.json", fallback)
        roster = [
            {
                "slot_id": window["window_id"] + f"-c{checkpoint}",
                "source_group": window["source_group"],
                "window_id": window["window_id"],
                "language": "ko",
                "checkpoint": checkpoint,
                "stratum": "natural_source_first"
                if checkpoint == 0
                else "natural_source_simulated_session",
            }
            for window in windows
            for checkpoint in range(3)
        ]
        sealed.write_rows(self.root / "request-slots-proposed.jsonl", roster)
        self.metadata = self.root / "metadata.json"
        sealed.write(
            self.metadata,
            {
                "reserved_final_test_bodies_read": False,
                "generated_test_cases": 0,
                "files_sha256": {
                    path.name: sealed.sha256(path) for path in self.root.iterdir()
                },
            },
        )
        self.allocation = sealed.metadata_allocation(self.metadata)
        evidence = self.write(
            "engineering-header-receipt.json",
            {"scope": "synthetic_header_fixture", "body_read": False},
        )
        cohorts = []
        for group in groups:
            selected = (
                fallback["first_eight_metadata_candidates"]
                if group == sealed.KO_FALLBACK
                else self.allocation["cohorts"][group]
            )
            records = [
                {
                    "publisher_recording_id": row["publisher_recording_id"],
                    "status": "metadata_available",
                    "media_url": "https://publisher.invalid/"
                    + row["publisher_recording_id"]
                    + ".mp4",
                    "duration_seconds": None,
                    "reason_code": None,
                    "before_words": True,
                    "media_body_read": False,
                    "captions_or_transcript_read": False,
                    "asr_requested": False,
                    "model_predictions_seen": False,
                    "recorded_at": sealed.now(),
                    "evidence": [evidence],
                }
                for row in selected
            ]
            cohorts.append({"source_group": group, "recordings": records})
        self.assessments = {
            "version": "task-test-actions-v3-source-assessments-1",
            "metadata_manifest_sha256": sealed.sha256(self.metadata),
            "cohorts": cohorts,
        }

    def write(self, name, value):
        path = self.root / name
        sealed.write(path, value)
        return sealed.binding(path)

    def fail(self, index):
        record = self.assessments["cohorts"][index]["recordings"][0]
        record.update(
            status="technical_failure",
            reason_code="media_endpoint_unavailable",
            media_url=None,
        )

    def resolve(self):
        path = self.write("assessments.json", self.assessments)
        binding = sealed.resolve_sources(
            self.metadata, path["path"], self.root / "resolution.json"
        )
        freeze = {
            key: value
            for key, value in sealed.read_json(binding["path"]).items()
            if key
            in {"metadata_manifest", "source_reservations", "roster", "natural_windows"}
        }
        freeze["source_resolution"] = binding
        resolution = sealed.load_source_resolution(freeze, datetime.now(timezone.utc))
        gate = {"windows": self.allocation["windows"], "source_resolution": resolution}
        return binding, freeze, gate

    def payload(self, window):
        return {
            "window_id": window["window_id"],
            "source_group": window["source_group"],
            "recording_id": window["publisher_recording_id"],
            "language": "ko",
            "registered_source_group": window["registered_source_group"],
            "registered_recording_id": window["registered_publisher_recording_id"],
            "source_resolution": window["source_resolution"],
            "media_url": window["media_url"],
        }

    def test_first_failed_cohort_only_and_both_source_denominators_are_preserved(self):
        self.fail(0)
        self.fail(1)
        _, _, gate = self.resolve()
        resolved = [
            sealed.effective_window(gate, row["window_id"])
            for row in self.allocation["windows"]
        ]
        self.assertEqual(
            [row["source_allocation_status"] for row in resolved],
            ["fallback"] * 4 + ["source_unavailable"] * 4,
        )
        self.assertEqual(
            [row["publisher_recording_id"] for row in resolved[:4]],
            [f"engineering-fallback-{i}" for i in range(4)],
        )
        sealed.verify_window_identity(gate, self.payload(resolved[0]))
        gate.update(
            binding={"sha256": "engineering-freeze"},
            frozen_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        slot = self.allocation["roster"][0]
        candidate = {
            "id": slot["slot_id"],
            "split": "final_test",
            "language": "ko",
            "source": {
                "split": "final_test",
                "source_group": sealed.KO_FALLBACK,
                "recording_id": resolved[0]["publisher_recording_id"],
            },
            "lineage": {
                "stratum": slot["stratum"],
                "window_id": slot["window_id"],
                "registered_source_group": resolved[0]["registered_source_group"],
                "registered_recording_id": resolved[0][
                    "registered_publisher_recording_id"
                ],
                "source_resolution": resolved[0]["source_resolution"],
            },
            "provenance": {
                "selection_before_words": True,
                "model_predictions_seen": False,
                "created_at": sealed.now(),
                "freeze_sha256": "engineering-freeze",
            },
        }
        sealed.candidate_identity(candidate, slot, gate)
        candidate["source"]["source_group"] = sealed.KO_COHORTS[0]
        with self.assertRaisesRegex(ValueError, "source identity changed"):
            sealed.candidate_identity(candidate, slot, gate)
        with self.assertRaisesRegex(ValueError, "remains unavailable"):
            sealed.verify_window_identity(gate, self.payload(resolved[4]))
        census = [
            {
                "id": row["window_id"] + f"-c{i}",
                "source_group": row["source_group"],
                "registered_source_group": row["registered_source_group"],
                "stratum": "natural_source_first",
                "language": "ko",
                "window_id": row["window_id"],
                "status": "source_unavailable"
                if row["source_allocation_status"] == "source_unavailable"
                else "admitted",
            }
            for row in resolved
            for i in range(3)
        ]
        summary = sealed.census_summary(census)
        self.assertEqual(summary["source_group"][sealed.KO_FALLBACK]["registered"], 12)
        self.assertEqual(
            summary["registered_source_group"][sealed.KO_COHORTS[0]]["registered"], 12
        )
        self.assertEqual(
            summary["source_group"][sealed.KO_COHORTS[1]]["dispositions"],
            {"source_unavailable": 12},
        )
        self.assertEqual(
            sum(x["registered"] for x in summary["source_group"].values()), 24
        )

    def test_biology_only_failure_uses_same_fixed_fallback_and_keeps_unknown_durations(
        self,
    ):
        self.fail(1)
        _, _, gate = self.resolve()
        primary = sealed.effective_window(gate, "engineering-0-0")
        fallback = sealed.effective_window(gate, "engineering-1-0")
        self.assertEqual(primary["source_group"], sealed.KO_COHORTS[0])
        self.assertEqual(fallback["source_group"], sealed.KO_FALLBACK)
        self.assertIsNone(primary["start_seconds"])
        self.assertIsNone(fallback["start_seconds"])
        self.assertEqual(fallback["window_seconds"], 90)

    def test_identity_locator_receipt_and_postfreeze_timing_faults_are_rejected(self):
        self.fail(0)
        binding, freeze, gate = self.resolve()
        window = sealed.effective_window(gate, "engineering-0-0")
        for field, value in (
            ("source_group", sealed.KO_COHORTS[0]),
            ("recording_id", "wrong-publisher"),
            ("registered_recording_id", "other-logical-recording"),
            ("source_resolution", None),
            ("media_url", "https://publisher.invalid/another.mp4"),
        ):
            with self.subTest(field=field):
                changed = {**self.payload(window), field: value}
                with self.assertRaises(ValueError):
                    sealed.verify_window_identity(gate, changed)
        created = datetime.fromisoformat(
            sealed.read_json(binding["path"])["created_at"]
        )
        with self.assertRaisesRegex(ValueError, "after the body freeze"):
            sealed.load_source_resolution(freeze, created - timedelta(seconds=1))
        changed = sealed.read_json(binding["path"])
        changed["windows"][0]["start_seconds"] = 5
        freeze["source_resolution"] = self.write("changed-resolution.json", changed)
        with self.assertRaisesRegex(ValueError, "timing allocation"):
            sealed.load_source_resolution(freeze, datetime.now(timezone.utc))

    def test_missing_receipt_quality_substitution_body_access_and_missing_duration_fail_closed(
        self,
    ):
        _, freeze, _ = self.resolve()
        freeze.pop("source_resolution")
        with self.assertRaisesRegex(ValueError, "before freezing body access"):
            sealed.load_source_resolution(freeze, datetime.now(timezone.utc))
        for patch in (
            {"status": "technical_failure", "reason_code": "poor_asr_quality"},
            {
                "status": "technical_failure",
                "reason_code": "verified_recording_too_short",
                "duration_seconds": None,
            },
            {"media_body_read": True},
        ):
            changed = copy.deepcopy(self.assessments)
            changed["cohorts"][0]["recordings"][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                sealed.validate_source_assessments(changed, self.allocation)


if __name__ == "__main__":
    unittest.main()
