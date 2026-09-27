"""Admission binds complete reviews, physical source evidence and stable inputs."""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import freeze_passive_dataset as admission
from passive_contract import PassiveContextBuilder, canonical_messages
from passive_dataset import ROOT, build_episode, digest, jsonl, read, write
from replay_session import session_hash
from synthetic_pilot import document_for


class TinyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, *args, **kwargs):
        return [1]

    def encode(self, text, **kwargs):
        return list(text.encode())


class FreezePassiveDatasetTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.directory = self.base / "draft"
        self.directory.mkdir()
        self.output = self.base / "frozen"
        self.review = self.base / "review.json"
        self.verification = self.base / "verification.json"
        self.registry = self.base / "sources.json"
        self.config = self.base / "config.json"
        self.media = self.base / "recording.bin"
        self.media.write_bytes(b"controlled fixture media bytes")
        self.native = self.base / "native.json"
        write(self.native, {"origin": "controlled fixture"})
        self.registration = self.base / "preregistration.json"
        write(self.registration, {"source_groups": ["fixture-train", "fixture-dev"]})
        self.authoring = self.base / "authoring.py"
        self.authoring.write_text('VERSION = "fixture-authoring-v1"\n')
        self.cases, self.specs, self.sources = [], [], []
        builder = PassiveContextBuilder(TinyTokenizer())
        for split, word in (("train", "Perhaps"), ("dev", "Possibly")):
            doc = document_for(
                {
                    "language": "en",
                    "paragraphs": [
                        [
                            {"id": split + "-a", "text": word, "ms": 2000},
                            {
                                "id": split + "-b",
                                "text": "something changed.",
                                "ms": 2000,
                            },
                        ]
                    ],
                }
            )
            document_path = self.base / (split + "-source.json")
            write(document_path, doc)
            source = {
                "id": split,
                "source_group_id": "fixture-" + split,
                "split": split,
                "language": "en",
                "document_path": self.relative(document_path),
                "document_sha256": digest(document_path),
                "media_path": self.relative(self.media),
                "media_download": {"sha256": digest(self.media)},
                "file_sha256": {self.relative(self.native): digest(self.native)},
            }
            self.sources.append(source)
            spec = {
                "id": split + "-episode",
                "source_id": split,
                "kind": "source_derived_restoration",
                "rationale": "A controlled visible spelling defect.",
                "source_region": [0, 4],
                "corruptions": [
                    {
                        "uuid": split + "-b",
                        "original": "something changed.",
                        "replacement": "somthing changed.",
                    }
                ],
            }
            self.specs.append(spec)
            self.cases.extend(
                build_episode(
                    spec, {**source, "document": doc}, builder, self.directory
                )
            )
            if split == "train":
                baseline = {
                    **spec,
                    "id": "train-clean-baseline",
                    "kind": "clean_history_control",
                    "corruptions": [],
                    "steps": [{"action": "no_action", "focus_uuid": "train-b"}],
                }
                self.specs.append(baseline)
                self.cases.extend(
                    build_episode(
                        baseline, {**source, "document": doc}, builder, self.directory
                    )
                )
        write(
            self.registry,
            {
                "sources": self.sources,
                "preregistration_path": self.relative(self.registration),
                "preregistration_sha256": digest(self.registration),
            },
        )
        write(self.directory / "recipe.json", {"episodes": self.specs})
        jsonl(self.directory / "exclusions.jsonl", [])
        for split in ("train", "dev"):
            jsonl(
                self.directory / f"{split}-candidates.jsonl",
                [c for c in self.cases if c["metadata"]["split"] == split],
            )
        dependencies = {
            self.relative(p): digest(p)
            for p in [
                *self.directory.glob("*.json*"),
                *self.directory.glob("episodes/*.json"),
                self.registry,
            ]
        }
        write(
            self.verification,
            {
                "status": "passed",
                "errors": [],
                "directory": str(self.directory),
                "dependencies_sha256": dependencies,
                "candidate_hashes": {
                    name: digest(self.directory / name)
                    for name in (
                        "train-candidates.jsonl",
                        "dev-candidates.jsonl",
                        "recipe.json",
                    )
                },
                "candidates": len(self.cases),
                "actual_serving_message_parity_checks": len(self.cases),
                "replayed_episodes": 3,
                "accepted_outcomes_exported": len(self.cases),
                "outcomes": [
                    {"case_id": c["id"], "reference_index": 0, "status": "passed"}
                    for c in self.cases
                ],
            },
        )
        write(
            self.review,
            {
                "candidate_files_sha256": {
                    self.relative(self.directory / f"{split}-candidates.jsonl"): digest(
                        self.directory / f"{split}-candidates.jsonl"
                    )
                    for split in ("train", "dev")
                },
                "cases": [
                    {
                        "case_id": c["id"],
                        "decision": "admit",
                        "reason": "Controlled fixture decision for admission tests.",
                        "source_id": c["metadata"]["source_id"],
                        "input_sha256": session_hash(canonical_messages(c["input"])),
                    }
                    for c in self.cases
                ],
            },
        )
        write(
            self.config,
            {
                "source_registries": [self.relative(self.registry)],
                "provenance_files_sha256": {
                    self.relative(self.authoring): digest(self.authoring)
                },
                "drafts": [
                    {
                        "path": self.relative(self.directory),
                        "verification": self.relative(self.verification),
                        "reviews": [self.relative(self.review)],
                    }
                ],
            },
        )

    def relative(self, path):
        return str(path.relative_to(ROOT))

    def run_freeze(self):
        loaded = {str(i): TinyTokenizer() for i in range(5)}
        with patch.object(admission, "tokenizers", return_value=loaded):
            with contextlib.redirect_stdout(io.StringIO()):
                admission.freeze(self.config, self.output)

    def assert_rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.run_freeze()
        self.assertFalse(self.output.exists())

    def test_freeze_preserves_splits_history_controls_and_asset_closure(self):
        with patch.object(admission, "digest", wraps=digest) as hashed:
            self.run_freeze()
        self.assertEqual(
            sum(call.args[0] == self.media for call in hashed.call_args_list), 1
        )
        census = read(self.output / "census.json")
        self.assertEqual(census["splits"], {"train": 3, "dev": 2})
        self.assertEqual(census["sampling_weight_sums"], {"train": 1.0, "dev": 1.0})
        self.assertEqual(
            census["history_action_counts"]["train:no_action:empty_history"], 1
        )
        manifest = read(self.output / "manifest.json")
        for path in (
            self.media,
            self.native,
            self.registry,
            self.registration,
            self.authoring,
        ):
            self.assertEqual(
                manifest["dependencies_sha256"][self.relative(path)], digest(path)
            )
        for name, expected in manifest["files_sha256"].items():
            self.assertEqual(digest(self.output / name), expected)
        messages = admission.rows(self.output / "train-messages.jsonl")
        self.assertEqual(len(messages), 3)
        self.assertTrue(all(c["id"].startswith("train") for c in messages))
        self.assertTrue(all(c["sampling_weight"] == 1 / 3 for c in messages))

    def test_partial_or_stale_review_file_binding_is_rejected(self):
        original = read(self.review)
        name = self.relative(self.directory / "dev-candidates.jsonl")
        for mode in ("missing", "stale"):
            with self.subTest(mode=mode):
                review = copy.deepcopy(original)
                if mode == "missing":
                    review["candidate_files_sha256"].pop(name)
                else:
                    review["candidate_files_sha256"][name] = "0" * 64
                write(self.review, review)
                self.assert_rejected("both exact candidate files")

    def test_review_requires_exact_input_source_and_complete_decisions(self):
        original = read(self.review)
        for mode in ("input", "source", "candidate_file", "missing", "duplicate"):
            with self.subTest(mode=mode):
                review = copy.deepcopy(original)
                first = review["cases"][0]
                if mode == "input":
                    first["input_sha256"] = session_hash(self.cases[0]["input"])
                elif mode == "source":
                    first["source_id"] = "another-recording"
                elif mode == "candidate_file":
                    first["candidate_file"] = self.relative(
                        self.directory / "dev-candidates.jsonl"
                    )
                elif mode == "missing":
                    review["cases"].pop()
                else:
                    review["cases"].append(copy.deepcopy(first))
                write(self.review, review)
                self.assert_rejected(
                    "different predictor input|identity differs|exactly one|Duplicated"
                )

    def test_partial_verification_census_and_duplicate_outcomes_are_rejected(self):
        original = read(self.verification)
        for mode in ("candidates", "outcomes", "episode_dependency"):
            with self.subTest(mode=mode):
                verification = copy.deepcopy(original)
                if mode == "candidates":
                    verification["candidates"] -= 1
                elif mode == "outcomes":
                    verification["outcomes"][-1] = copy.deepcopy(
                        verification["outcomes"][0]
                    )
                else:
                    verification["dependencies_sha256"].pop(
                        self.relative(self.directory / "episodes/train-episode.json")
                    )
                write(self.verification, verification)
                self.assert_rejected("every candidate|reconstruction evidence")

    def test_missing_registry_and_changed_physical_assets_are_rejected(self):
        config = read(self.config)
        write(self.config, {**config, "source_registries": []})
        self.assert_rejected("explicit source_registries")
        write(self.config, config)
        for path in (self.media, self.native, self.registration, self.authoring):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                self.assert_rejected("Evidence hash differs")
                path.write_bytes(original)

    def test_changed_evidence_during_tokenization_never_publishes_a_release(self):
        original = admission.training_row
        changed = False

        def mutate(tokenizer, case):
            nonlocal changed
            result = original(tokenizer, case)
            if not changed:
                self.native.write_text("changed while the release was being prepared")
                changed = True
            return result

        with patch.object(admission, "training_row", side_effect=mutate):
            self.assert_rejected("Evidence changed during freeze")
        self.assertFalse(list(self.base.glob(".frozen-*")))

    def test_registry_cannot_reassign_a_verified_episode_to_another_document(self):
        registry = read(self.registry)
        other = self.base / "other-source.json"
        write(other, {"unrelated": "document"})
        registry["sources"][0]["document_path"] = self.relative(other)
        registry["sources"][0]["document_sha256"] = digest(other)
        write(self.registry, registry)
        verification = read(self.verification)
        verification["dependencies_sha256"][self.relative(self.registry)] = digest(
            self.registry
        )
        write(self.verification, verification)
        self.assert_rejected("Source provenance or sampling region")


if __name__ == "__main__":
    unittest.main()
