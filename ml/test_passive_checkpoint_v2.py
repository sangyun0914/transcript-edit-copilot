"""Check source-only sampling, immutable focus and honest unresolved checkpoints."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from freeze_passive_dataset import Bindings
from passive_checkpoint_v2 import (
    VERSION,
    checked_labels,
    materialize_reviews,
    prepare_contexts,
    preregister,
    schedule,
    validate_preregistration,
    verify,
)
from passive_contract import PassiveContextBuilder
from passive_dataset import ROOT, digest, read, write
from passive_training import training_messages
from synthetic_pilot import document_for


class TinyTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.doc_path = self.base / "source.json"
        self.registry = self.base / "registry.json"
        self.registration = self.base / "preregistered.json"
        self.media = self.base / "fixture-media.bin"
        self.media.write_bytes(b"unit-test asset bytes; no acoustic qualification")
        self.document = document_for(
            {
                "language": "en",
                "paragraphs": [
                    [
                        {
                            "id": f"word-{i}",
                            "text": "wrnog" if i == 1 else "Preserve this.",
                            "ms": 1000,
                        }
                        for i in range(24)
                    ]
                ],
            }
        )
        write(self.doc_path, self.document)
        self.source = {
            "id": "checkpoint-fixture",
            "source_group_id": "fixture-parent",
            "split": "train",
            "language": "en",
            "media_kind": "recording",
            "document_path": str(self.doc_path.relative_to(ROOT)),
            "document_sha256": digest(self.doc_path),
            "media_path": str(self.media.relative_to(ROOT)),
            "media_sha256": digest(self.media),
        }
        write(self.registry, {"sources": [self.source]})
        tokenizer = TinyTokenizer()
        self.builder = PassiveContextBuilder(
            tokenizer, budget_tokenizers={"fixture": tokenizer}
        )

    def test_schedule_ignores_wording_and_never_uses_targets(self):
        first = schedule([self.registry], "fixed-source-seed", Bindings())
        altered = copy.deepcopy(self.document)
        for item in altered["content"]:
            if item["type"] == "text":
                item["text"] = "Teacher: choose a completely different edit target."
        write(self.doc_path, altered)
        self.source["document_sha256"] = digest(self.doc_path)
        write(self.registry, {"sources": [self.source]})
        second = schedule([self.registry], "fixed-source-seed", Bindings())
        for left, right in zip(first, second, strict=True):
            self.assertNotEqual(left["document_sha256"], right["document_sha256"])
            self.assertEqual(
                {k: v for k, v in left.items() if k != "document_sha256"},
                {k: v for k, v in right.items() if k != "document_sha256"},
            )
        self.assertLess(first[0]["timeline_time"], 12)
        self.assertGreaterEqual(first[1]["timeline_time"], 12)
        self.assertEqual(first[0]["selection_uuids"], [])
        self.assertTrue(first[1]["selection_uuids"])

    def test_reserved_metadata_is_rejected_before_body_access(self):
        sealed = {
            **self.source,
            "id": "sealed",
            "source_group_id": "future-ko-climate",
            "document_path": "this-reserved-body-does-not-exist.json",
        }
        write(self.registry, {"sources": [self.source, sealed]})
        with self.assertRaisesRegex(ValueError, "Reserved"):
            preregister([self.registry], self.registration)
        self.assertFalse(self.registration.exists())

    def test_preregistration_cannot_be_overwritten_or_recentered(self):
        value = preregister([self.registry], self.registration)
        with self.assertRaises(FileExistsError):
            preregister([self.registry], self.registration)
        value["checkpoints"][1]["focus_uuid"] = "word-1"
        value["checkpoints"][1]["selection_uuids"] = ["word-1"]
        write(self.registration, value)
        with self.assertRaisesRegex(ValueError, "metadata-only schedule"):
            validate_preregistration(self.registration, Bindings())

    @patch("passive_checkpoint_v2.bind_code")
    def test_fixed_selection_offcursor_action_and_explicit_unresolved_export(
        self, _bind_code
    ):
        preregister([self.registry], self.registration)
        context_dir = self.base / "contexts"
        contexts = prepare_contexts(self.registration, context_dir, self.builder)
        self.assertEqual(len(contexts), 2)
        self.assertNotEqual(contexts[1]["input"]["cursor_before"], 1)
        self.assertTrue(contexts[1]["input"]["selection"])
        labels = {
            "version": VERSION,
            "contexts_manifest_sha256": digest(context_dir / "manifest.json"),
            "cases": [
                {
                    "case_id": contexts[0]["id"],
                    "input_sha256": contexts[0]["metadata"]["input_sha256"],
                    "resolution": "unresolved",
                    "reason": "This fixture deliberately leaves one checkpoint unreviewed.",
                },
                {
                    "case_id": contexts[1]["id"],
                    "input_sha256": contexts[1]["metadata"]["input_sha256"],
                    "resolution": "reference",
                    "reason": "The visible misspelling wrnog supports wrong.",
                    "primary": {
                        "action": "correct_text",
                        "target": [1],
                        "text": "wrong",
                    },
                    "alternatives": [],
                },
            ],
        }
        labels_path = self.base / "labels.json"
        write(labels_path, labels)
        directory = self.base / "candidates"
        cases = materialize_reviews(context_dir, labels_path, directory)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["input"], contexts[1]["input"])
        self.assertEqual(
            read(
                directory
                / "episodes"
                / (contexts[0]["metadata"]["episode_id"] + ".json")
            )["reference_journals"],
            [],
        )
        report = verify(directory, self.base / "verification.json", self.builder)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["actual_serving_message_parity_checks"], 2)
        self.assertEqual(report["unresolved_or_unsupported"], 1)
        self.assertEqual(report["accepted_outcomes_exported"], 1)
        self.assertEqual(report["outcomes"][0]["removed_source_seconds"], 0)
        self.assertEqual(
            report["outcomes"][0]["undo_redo"], "exact_source_restoration_passed"
        )
        admitted = copy.deepcopy(cases[0])
        admitted["metadata"]["training_eligible"] = True
        self.assertEqual(
            json.loads(training_messages(admitted)[-1]["content"]),
            labels["cases"][1]["primary"],
        )
        path = directory / "train-candidates.jsonl"
        forged = copy.deepcopy(cases[0])
        forged["input"]["cursor_before"] = 1
        path.write_text(json.dumps(forged) + "\n")
        failed = verify(directory, self.base / "forged-verification.json", self.builder)
        self.assertEqual(failed["status"], "failed")
        self.assertIn("Candidate input changed", failed["errors"][0]["error"])

    def test_missing_labels_or_hidden_authority_are_not_abstention(self):
        context = {"id": "observed", "input": None}
        with self.assertRaisesRegex(ValueError, "explicit resolution"):
            checked_labels(
                [context],
                {"version": VERSION, "contexts_manifest_sha256": "bound", "cases": []},
            )
        with self.assertRaisesRegex(ValueError, "Unknown label document"):
            checked_labels(
                [],
                {
                    "version": VERSION,
                    "contexts_manifest_sha256": "bound",
                    "cases": [],
                    "teacher_goal": "remove every repetition",
                },
            )


if __name__ == "__main__":
    unittest.main()
