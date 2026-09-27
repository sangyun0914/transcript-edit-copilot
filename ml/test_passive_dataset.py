"""Guard source isolation, honest corruptions, alternatives and episode replay."""

import copy
import json
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

from passive_dataset import (
    ROOT,
    apply_corruptions,
    build_episode,
    compact_transaction,
    digest,
    episode_steps,
    jsonl,
    load_sources,
    mine_restorations,
    mutations,
    write,
)
from replay_session import replay, session_hash


class PassiveDatasetTests(unittest.TestCase):
    def test_native_findings_rebuild_original_text_without_synthetic_history(self):
        from mine_reviewed_passive_repairs import repair_recipes
        from passive_contract import PassiveContextBuilder, canonical_messages
        from synthetic_pilot import document_for

        class TinyTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1]

        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            directory = Path(tmp)
            document = document_for(
                {
                    "language": "en",
                    "paragraphs": [
                        [{"id": "a", "text": "It was diffrent.", "ms": 3000}]
                    ],
                }
            )
            source = directory / "document.json"
            write(source, document)
            registry = directory / "sources.json"
            write(
                registry,
                {
                    "sources": [
                        {
                            "id": "source",
                            "source_group_id": "parent",
                            "split": "train",
                            "language": "en",
                            "document_path": str(source.relative_to(ROOT)),
                            "document_sha256": digest(source),
                        }
                    ]
                },
            )
            bundle = PassiveContextBuilder(TinyTokenizer()).build(
                document, "a", as_of=0
            )
            candidate = {
                "id": "stop",
                "input": bundle["input"],
                "expectedOutput": {"action": "no_action"},
                "metadata": {
                    "source_id": "source",
                    "source_region": [0, 3],
                    "kind": "source_derived_restoration",
                },
            }
            jsonl(directory / "train-candidates.jsonl", [candidate])
            jsonl(directory / "dev-candidates.jsonl", [])
            review = directory / "review.json"
            decision = {
                "case_id": "stop",
                "input_sha256": session_hash(canonical_messages(candidate["input"])),
                "reason": "Locally visible misspelling.",
                "suggested_native_fix": {
                    "action": "correct_text",
                    "target": [0],
                    "text": "It was different.",
                },
            }

            def save_review(row):
                write(
                    review,
                    {
                        "candidate_files_sha256": {
                            str(
                                (directory / f"{s}-candidates.jsonl").relative_to(ROOT)
                            ): digest(directory / f"{s}-candidates.jsonl")
                            for s in ("train", "dev")
                        },
                        "cases": [row],
                    },
                )

            save_review(decision)
            config = {
                "sources": [str(registry.relative_to(ROOT))],
                "drafts": [
                    {
                        "path": str(directory.relative_to(ROOT)),
                        "reviews": [str(review.relative_to(ROOT))],
                    }
                ],
            }
            mined = repair_recipes(config)
            spec = mined["episodes"][0]
            self.assertEqual(spec["corruptions"], [])
            self.assertEqual(len(episode_steps(spec)), 1)
            self.assertEqual(spec["steps"][0]["target_uuids"], ["a"])
            self.assertTrue(spec["finding"]["fresh_input_review_required"])
            config["drafts"][0]["reviews"].append(str(review.relative_to(ROOT)))
            self.assertEqual(len(repair_recipes(config)["episodes"]), 1)
            wrong = copy.deepcopy(decision)
            wrong["suggested_native_fix"]["target"] = [999]
            save_review(wrong)
            with self.assertRaisesRegex(ValueError, "visible native source text"):
                repair_recipes(config)

    def test_explicit_episode_steps_are_not_mutated_or_extended(self):
        spec = {
            "kind": "source_derived_restoration",
            "steps": [{"action": "no_action", "focus_uuid": "a"}],
        }
        first = episode_steps(spec)
        first.append({"action": "no_action", "focus_uuid": "b"})
        self.assertEqual(len(episode_steps(spec)), 1)
        self.assertEqual(len(spec["steps"]), 1)

    def test_inventory_rejects_forged_split_hash_and_missing_stop(self):
        from passive_contract import PassiveContextBuilder, PassivePolicy
        from synthetic_pilot import document_for
        from verify_passive_dataset import validate_inventory

        class TinyTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1]

        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            directory = Path(tmp)
            doc = document_for(
                {
                    "language": "en",
                    "paragraphs": [
                        [
                            {"id": "a", "text": "Perhaps", "ms": 2000},
                            {"id": "b", "text": "something changed.", "ms": 2000},
                        ]
                    ],
                }
            )
            source_path = directory / "source.json"
            write(source_path, doc)
            source = {
                "id": "fixture",
                "source_group_id": "fixture-parent",
                "split": "train",
                "language": "en",
                "document_path": str(source_path.relative_to(ROOT)),
                "document_sha256": digest(source_path),
                "document": doc,
            }
            registry = directory / "registry.json"
            write(
                registry,
                {"sources": [{k: v for k, v in source.items() if k != "document"}]},
            )
            spec = {
                "id": "fixture-episode",
                "source_id": "fixture",
                "kind": "source_derived_restoration",
                "rationale": "Visible spelling only.",
                "corruptions": [
                    {
                        "uuid": "b",
                        "original": "something changed.",
                        "replacement": "somthing changed.",
                    }
                ],
            }
            cases = build_episode(
                spec,
                source,
                PassiveContextBuilder(TinyTokenizer(), policy=PassivePolicy()),
                directory,
            )
            write(directory / "recipe.json", {"episodes": [spec]})
            jsonl(directory / "exclusions.jsonl", [])
            jsonl(directory / "dev-candidates.jsonl", [])

            def check(values):
                jsonl(directory / "train-candidates.jsonl", values)
                return validate_inventory(directory, [registry])

            self.assertEqual(len(check(cases)[0]), 2)
            bad = copy.deepcopy(cases)
            bad[0]["metadata"]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "file placement"):
                check(bad)
            bad = copy.deepcopy(cases)
            bad[0]["metadata"]["input_sha256"] = "forged"
            with self.assertRaisesRegex(ValueError, "forged or stale"):
                check(bad)
            bad = copy.deepcopy(cases)
            bad[0]["metadata"]["source_group_id"] = "future-en-aviation"
            with self.assertRaisesRegex(ValueError, "Reserved"):
                check(bad)
            with self.assertRaisesRegex(ValueError, "recipe step"):
                check(cases[:-1])

    def test_reserved_registration_is_rejected_before_body_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.json"
            path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "id": "sealed",
                                "source_group_id": "future-ko-climate",
                                "split": "train",
                                "document_path": "/does/not/exist/do-not-open.json",
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Reserved"):
                load_sources([path])

    def test_corruption_preserves_every_field_but_explicit_display_text(self):
        document = {
            "content": [
                {
                    "type": "text",
                    "uuid": "a",
                    "source": "recording",
                    "sourceStart": 2.4,
                    "length": 1.2,
                    "text": "I might not remember.",
                    "conf": 0.7,
                }
            ],
            "metadata": {},
        }
        changes = mutations(document["content"][0], "en")
        change = next(c for c in changes if c["lexeme"] == "remember")
        result = apply_corruptions(document, [change])
        self.assertEqual(result["content"][0]["text"], "I might not remeber.")
        result["content"][0]["text"] = document["content"][0]["text"]
        self.assertEqual(result, document)
        with self.assertRaisesRegex(ValueError, "before-image"):
            apply_corruptions(document, [{**change, "original": "unknown"}])

    def test_no_cross_speaker_overlap_text_injections(self):
        self.assertEqual(
            mutations(
                {"type": "text", "uuid": "a", "text": "A: think\nB: maybe"}, "en"
            ),
            [],
        )

    def test_episode_regions_are_disjoint_and_one_parent_is_not_multiplied(self):
        items = [
            {
                "type": "text",
                "uuid": str(n),
                "sourceStart": n,
                "length": 1,
                "text": "something" if n % 2 else "important",
            }
            for n in range(400)
        ]
        source = {
            "id": "one-recording",
            "language": "en",
            "document": {"content": items},
        }
        specs = mine_restorations(source, 3)
        self.assertEqual(len(specs), 3)
        self.assertEqual({s["source_id"] for s in specs}, {"one-recording"})
        for left, right in pairwise(specs):
            self.assertLessEqual(left["source_region"][1], right["source_region"][0])
        for spec in specs:
            self.assertEqual(
                len({c["uuid"] for c in spec["corruptions"]}), len(spec["corruptions"])
            )

    def test_compact_patch_replays_and_tampering_is_detected(self):
        before = {
            "content": [{"uuid": str(n), "text": str(n)} for n in range(8)],
            "metadata": {},
        }
        after = copy.deepcopy(before)
        after["content"][3]["text"] = "changed"
        event = compact_transaction(before, after, 1, 1, "correct_text")
        self.assertEqual(event["patch"]["index"], 3)
        self.assertEqual(len(event["patch"]["remove"]), 1)
        events = [
            {
                "kind": "start",
                "seq": 0,
                "schema_version": 2,
                "hash_format": "typed-json-f64-v1",
                "actor": "synthetic",
                "document": before,
                "document_hash": session_hash(before),
            },
            event,
            {
                "kind": "end",
                "seq": 2,
                "revision": 1,
                "document_hash": session_hash(after),
            },
        ]
        self.assertEqual(replay(events)[0], after)
        events[1]["patch"]["insert"][0]["text"] = "tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            replay(events)


if __name__ == "__main__":
    unittest.main()
