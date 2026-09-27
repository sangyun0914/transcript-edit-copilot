import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import freeze_release
import scripted_release
from dataset_release_inventory import digest, read, write
from iteration import jsonl
from nep_v2 import ACTIONS
from release_media import verify_media_edit
from replay_session import session_hash
from reviewed_audio import CUTS


class ReleaseGateTests(unittest.TestCase):
    def test_duplicate_script_registrations_fail_before_opening_bodies(self):
        group = {"id": "group", "source_id": "script", "split": "dev", "language": "en"}
        for second in (
            {**group, "id": "other", "split": "train"},
            {**group, "source_id": "other"},
        ):
            with (
                self.subTest(second=second),
                patch.object(
                    scripted_release, "read", return_value={"groups": [group, second]}
                ) as reader,
                self.assertRaisesRegex(ValueError, "Duplicate authored"),
            ):
                scripted_release.scripted_sources("unused.json")
            self.assertEqual(reader.call_count, 1)

        registration = {"groups": [group], "sources_file": "unused-scripts.json"}
        with (
            patch.object(
                scripted_release,
                "read",
                side_effect=[
                    registration,
                    {"sources": [{"id": "script"}, {"id": "script"}]},
                ],
            ),
            self.assertRaisesRegex(ValueError, "preregistered groups"),
        ):
            scripted_release.scripted_sources("unused.json")

    def test_symbolic_verification_cannot_replace_recording_certificates(self):
        before = {
            "content": [
                {
                    "type": "text",
                    "source": "recording",
                    "sourceStart": 0.0,
                    "length": 1.0,
                    "text": "A",
                },
                {
                    "type": "non_text",
                    "source": "recording",
                    "sourceStart": 1.0,
                    "length": 2.0,
                },
                {
                    "type": "text",
                    "source": "recording",
                    "sourceStart": 3.0,
                    "length": 1.0,
                    "text": "B",
                },
            ],
            "metadata": {},
        }
        after = copy.deepcopy(before)
        after["content"].pop(1)
        mapping, action = (
            {"1": {"index": 1}},
            {"action": "remove_silence", "target": [1]},
        )
        with self.assertRaisesRegex(ValueError, "exact reviewed operation"):
            verify_media_edit(before, after, mapping, action, [])
        before["metadata"] = {
            "provenance": "authored_symbolic_timeline",
            "script_sha256": "script",
        }
        with self.assertRaisesRegex(
            ValueError, "cannot authorize edits to a recording"
        ):
            verify_media_edit(
                before, after, mapping, action, [], "authored_symbolic_timeline"
            )
        for item in before["content"]:
            item["source"] = "symbolic:script"
        after = copy.deepcopy(before)
        after["content"].pop(1)
        with self.assertRaisesRegex(ValueError, "cannot carry natural-audio"):
            verify_media_edit(
                before, after, mapping, action, [{}], "authored_symbolic_timeline"
            )
        verify_media_edit(
            before, after, mapping, action, [], "authored_symbolic_timeline"
        )
        after["content"].pop()
        with self.assertRaisesRegex(ValueError, "outside its declared effect"):
            verify_media_edit(
                before, after, mapping, action, [], "authored_symbolic_timeline"
            )


class FreezePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.directory = self.root / "draft"
        self.directory.mkdir()
        self.destination = self.root / "frozen"
        self.cases = [
            {
                "id": f"case-{n}",
                "input": {"n": n},
                "expectedOutput": {"action": action},
                "acceptableOutputs": [],
                "metadata": {
                    "split": "train" if n < 4 else "dev",
                    "language": "en" if n % 2 else "ko",
                    "feedback_ids": [],
                    "media_kind": "recording",
                    "audio_qualification_ids": [f"review-{n}"]
                    if action in CUTS
                    else [],
                    "source_group_id": f"group-{n}",
                    "retained_from": None,
                    "training_eligible": False,
                },
            }
            for n, action in enumerate(ACTIONS)
        ]
        for split in ("train", "dev"):
            jsonl(
                self.directory / f"{split}-cases.jsonl",
                [c for c in self.cases if c["metadata"]["split"] == split],
            )
        write(
            self.directory / "manifest.json",
            {
                "status": "draft",
                "cases": 7,
                "actions": {action: 1 for action in ACTIONS},
                "languages": {"en": 3, "ko": 4},
                "natural_audio_cuts": 3,
                "symbolic_cuts": 0,
                "unique_reviewed_operations": 3,
                "code_and_recipe_sha256": {},
                "splits": {"train": 4, "dev": 3},
            },
        )
        write(
            self.directory / "test-protocol.json",
            {
                "generated_cases": 0,
                "reserved_test_bodies_opened_by_this_auditor": False,
                "source_allocations": [{"source_group_id": "isolated-reservation"}],
                "artifact_kind": "protocol_only_no_test_bodies_no_test_labels",
                "allocation_interpretation": {"planned_request_slots": 1},
            },
        )
        write(
            self.directory / "split-manifest.json",
            {"reserved_test_groups": ["isolated-reservation"]},
        )
        write(self.directory / "scenario-coverage.json", {"decisions": []})
        write(
            self.directory / "source-inventory.json",
            {"current_sources": [], "fixtures": []},
        )
        write(self.directory / "audit.json", {"test_bodies_opened": False})
        write(self.directory / "recipe.json", {"gaps": {}, "episodes": []})
        (self.directory / "contexts").mkdir()
        for name in (
            "exclusions.jsonl",
            "executions.jsonl",
            "contexts/qwen3.5-2b.jsonl",
        ):
            jsonl(self.directory / name, [])
        self.registration_path = self.root / "registration.json"
        self.judgments_path = self.root / "judgments.jsonl"
        write(
            self.registration_path,
            {
                "case_count": len(self.cases),
                "ordered_case_ids": [c["id"] for c in self.cases],
                "case_map": {str(n): c["id"] for n, c in enumerate(self.cases)},
                "population_files": {
                    f"draft/{split}-cases.jsonl": digest(
                        self.directory / f"{split}-cases.jsonl"
                    )
                    for split in ("train", "dev")
                },
            },
        )
        jsonl(
            self.judgments_path,
            [
                {
                    "audit_key": str(n),
                    "input_sha256": session_hash(c["input"]),
                    "input_visible_evidence": {"observed": n},
                }
                for n, c in enumerate(self.cases)
            ],
        )
        self.admission = {
            "status": "passed_independent_census",
            "draft_manifest_sha256": digest(self.directory / "manifest.json"),
            "review_registration_path": "registration.json",
            "review_registration_sha256": digest(self.registration_path),
            "review_judgments_path": "judgments.jsonl",
            "review_judgments_sha256": digest(self.judgments_path),
            "cases": [
                {
                    "case_id": c["id"],
                    "input_sha256": session_hash(c["input"]),
                    "reference_sha256": session_hash([c["expectedOutput"]]),
                    "verdict": "accept",
                    "reason": "Fixture decision.",
                }
                for c in self.cases
            ],
        }
        self.admission_path = self.root / "admission.json"
        write(self.admission_path, self.admission)
        self.evidence = []
        for kind in ("episode_causality", "semantic_timing", "full_caption_export"):
            path = self.root / f"{kind}.json"
            write(
                path,
                {
                    "audit_kind": kind,
                    "status": "passed",
                    "manifest_sha256": digest(self.directory / "manifest.json"),
                },
            )
            self.evidence.append(path)
        self.code_path = self.root / "freeze-code.py"
        self.code_path.write_text("fixture freeze implementation identity\n")
        self.original_files = {
            str(p.relative_to(self.directory)): digest(p)
            for p in self.directory.rglob("*")
            if p.is_file()
        }

    def test_admission_rejects_missing_or_incomplete_independent_evidence(self):
        with patch.object(freeze_release, "ROOT", self.root):
            self.assertEqual(
                len(freeze_release.validate_admission(self.directory, self.admission)),
                7,
            )
            missing = {k: v for k, v in self.admission.items() if k != "status"}
            with self.assertRaisesRegex(ValueError, "passing independent census"):
                freeze_release.validate_admission(self.directory, missing)
            judgments = [
                json.loads(line)
                for line in self.judgments_path.read_text().splitlines()
            ]
            jsonl(self.judgments_path, judgments[:-1])
            incomplete = {
                **self.admission,
                "review_judgments_sha256": digest(self.judgments_path),
            }
            with self.assertRaisesRegex(ValueError, "complete census"):
                freeze_release.validate_admission(self.directory, incomplete)

    def run_freeze(self, failure=None):
        def exporter(command, **_kwargs):
            if failure == "export":
                raise RuntimeError("injected export failure")
            stage = Path(command[-1])
            write(
                stage / "caption-export-check.json",
                {
                    "inputs_sha256": {
                        str(
                            (stage / "train-cases.jsonl").relative_to(self.root)
                        ): digest(stage / "train-cases.jsonl")
                    }
                },
            )

        checks = []

        def verification(path, _tokenizer):
            checks.append(path)
            if len(checks) == 2:
                self.assertFalse(self.destination.exists())
                self.assertNotEqual(path, self.destination)
                if failure == "verify":
                    raise RuntimeError("injected final verification failure")
            return {"status": "fixture verification"}

        with ExitStack() as stack:
            stack.enter_context(patch.object(freeze_release, "ROOT", self.root))
            stack.enter_context(
                patch.object(freeze_release, "__file__", str(self.code_path))
            )
            stack.enter_context(
                patch.object(freeze_release, "verify", side_effect=verification)
            )
            stack.enter_context(
                patch.object(
                    freeze_release,
                    "audit_release",
                    return_value={"weight_policy": "fixture"},
                )
            )
            stack.enter_context(
                patch.object(freeze_release.subprocess, "run", side_effect=exporter)
            )
            result = freeze_release.freeze(
                self.directory,
                self.destination,
                self.admission_path,
                self.evidence,
                None,
            )
        return result, checks

    def test_failed_export_does_not_publish_or_modify_draft(self):
        with self.assertRaisesRegex(RuntimeError, "injected export failure"):
            self.run_freeze("export")
        self.assertFalse(self.destination.exists())
        self.assertEqual(
            self.original_files,
            {
                str(p.relative_to(self.directory)): digest(p)
                for p in self.directory.rglob("*")
                if p.is_file()
            },
        )

    def test_disclosed_test_body_exposure_blocks_freeze(self):
        for name, flag in (
            ("test-protocol.json", "reserved_test_bodies_opened_by_this_auditor"),
            ("audit.json", "test_bodies_opened"),
        ):
            with self.subTest(name=name):
                path = self.directory / name
                document = read(path)
                document[flag] = True
                path.write_text(json.dumps(document) + "\n")
                with self.assertRaisesRegex(ValueError, "Final-test"):
                    self.run_freeze()
                self.assertFalse(self.destination.exists())
                document[flag] = False
                path.write_text(json.dumps(document) + "\n")

    def test_missing_full_caption_export_review_blocks_freeze(self):
        self.evidence = [
            path
            for path in self.evidence
            if read(path)["audit_kind"] != "full_caption_export"
        ]
        with self.assertRaisesRegex(ValueError, "full-caption-export"):
            self.run_freeze()
        self.assertFalse(self.destination.exists())

    def test_stale_coverage_counts_block_freeze_even_with_current_review_hashes(self):
        path = self.directory / "manifest.json"
        manifest = read(path)
        for name, incorrect in (
            ("cases", 8),
            ("splits", {"train": 999, "dev": 0}),
            ("actions", {**manifest["actions"], "no_action": 2}),
            ("languages", {"en": 4, "ko": 3}),
            ("natural_audio_cuts", 4),
            ("symbolic_cuts", 1),
            ("unique_reviewed_operations", 4),
        ):
            with self.subTest(name=name):
                path.write_text(json.dumps({**manifest, name: incorrect}) + "\n")
                manifest_hash = digest(path)
                self.admission_path.write_text(
                    json.dumps(
                        {**self.admission, "draft_manifest_sha256": manifest_hash}
                    )
                    + "\n"
                )
                for evidence_path in self.evidence:
                    report = read(evidence_path)
                    report["manifest_sha256"] = manifest_hash
                    evidence_path.write_text(json.dumps(report) + "\n")
                with self.assertRaisesRegex(ValueError, "[Cc]ount"):
                    self.run_freeze()
                self.assertFalse(self.destination.exists())

    def test_failed_final_verification_does_not_publish(self):
        with self.assertRaisesRegex(
            RuntimeError, "injected final verification failure"
        ):
            self.run_freeze("verify")
        self.assertFalse(self.destination.exists())

    def test_only_fully_verified_stage_is_published_and_cannot_be_overwritten(self):
        result, checks = self.run_freeze()
        self.assertEqual(len(checks), 2)
        self.assertTrue(self.destination.exists())
        self.assertEqual(
            result["manifest_sha256"], digest(self.destination / "manifest.json")
        )
        report = read(self.destination / "caption-export-check.json")
        self.assertEqual(list(report["inputs_sha256"]), ["frozen/train-cases.jsonl"])
        with patch.object(freeze_release, "verify") as verifier:
            with self.assertRaisesRegex(ValueError, "immutable"):
                freeze_release.freeze(
                    self.directory,
                    self.destination,
                    self.admission_path,
                    self.evidence,
                    None,
                )
            verifier.assert_not_called()


if __name__ == "__main__":
    unittest.main()
