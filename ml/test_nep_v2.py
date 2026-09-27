import copy
import json
import tempfile
import unittest
from pathlib import Path

from nep_v2 import EDIT, Envelope, Policy, Snapshot, validate_prediction
from probe import runtime_schema
from replay_session import replay
from session_cases import apply_edit, snapshot_for
from synthetic_pilot import build, document_for, execute_history

HERE = Path(__file__).parent
REQUEST = Envelope(
    document_id="fixture", request_id="r", revision=0, interaction_epoch=0
)


class FullContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name) / "pilot"
        cls.report = build(HERE / "fixtures/synthetic-v2.json", cls.root)
        cls.cases = {
            c["id"]: c
            for c in map(
                json.loads, (cls.root / "pilot-cases.jsonl").read_text().splitlines()
            )
        }
        cls.sessions = {
            c["id"]: c
            for c in json.loads((cls.root / "replay-fixtures.json").read_text())
        }

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def validate(self, case_id, prediction=None, **kwargs):
        case = self.cases[case_id]
        return validate_prediction(
            json.dumps(prediction or case["expectedOutput"]),
            Snapshot.model_validate(kwargs.pop("snapshot", case["input"])),
            REQUEST,
            kwargs.pop("current", REQUEST),
            kwargs.pop("finish_reason", "stop"),
            **kwargs,
        )

    def test_full_scope_and_clean_transcript_abstention(self):
        self.assertEqual(
            set(self.report["action_counts"]),
            {
                "no_action",
                "correct_text",
                "delete_span",
                "trim_silence",
                "remove_silence",
                "split_paragraph",
                "merge_paragraph",
            },
        )
        for case in self.cases.values():
            if case["expectedOutput"] is not None:
                self.validate(case["id"])
        self.assertEqual(
            self.cases["clean-transcript"]["expectedOutput"], {"action": "no_action"}
        )
        self.assertEqual(
            self.cases["repeat-terminology"]["metadata"]["context_origin"],
            "constructed_corruption",
        )
        self.assertIsNone(self.cases["uncertain-name"]["expectedOutput"])
        self.assertEqual(
            json.loads((HERE / "edit-v2.schema.json").read_text()), EDIT.json_schema()
        )
        for fixture in map(
            json.loads, (HERE / "fixtures/execution-v2.jsonl").read_text().splitlines()
        ):
            self.assertEqual(fixture, self.sessions[fixture["id"]])

    def test_full_span_and_parameter_support_preserves_source_and_exact_undo(self):
        for row in self.sessions.values():
            with self.subTest(case=row["id"]):
                result, summary = replay(row["events"])
                self.assertEqual(result, row["after"])
                self.assertEqual(summary["actor"], "synthetic")
                if row["before"] != row["after"]:
                    undone, _ = replay(row["events"][:3])
                    self.assertEqual(undone, row["before"])
        correction = self.sessions["multi-item-correction"]
        before_span = correction["before"]["content"][1:3]
        after_item = correction["after"]["content"][1]
        self.assertEqual(after_item["length"], sum(i["length"] for i in before_span))
        trim = self.sessions["trim-pause"]
        left, right = trim["after"]["content"][2:4]
        original = trim["before"]["content"][2]
        self.assertAlmostEqual(left["length"] + right["length"], 0.725)
        self.assertEqual(left["sourceStart"], original["sourceStart"])
        self.assertAlmostEqual(
            right["sourceStart"] + right["length"],
            original["sourceStart"] + original["length"],
        )

    def test_histories_are_executed_before_target_states(self):
        for case in self.cases.values():
            previous_hash = None
            for relative in case["metadata"]["history_sessions"]:
                events = list(
                    map(json.loads, (self.root / relative).read_text().splitlines())
                )
                if previous_hash is not None:
                    self.assertEqual(events[0]["document_hash"], previous_hash)
                replay(events)
                previous_hash = events[-1]["document_hash"]
            if previous_hash is not None:
                self.assertEqual(previous_hash, case["metadata"]["before_hash"])

    def test_decoding_workaround_does_not_relax_validation(self):
        schema = EDIT.json_schema()
        decoding = runtime_schema(schema)
        self.assertNotIn(
            "maxLength", decoding["$defs"]["CorrectText"]["properties"]["text"]
        )
        self.assertEqual(
            schema["$defs"]["CorrectText"]["properties"]["text"]["maxLength"], 2000
        )
        with self.assertRaises(ValueError):
            self.validate(
                "repeat-terminology",
                {"action": "correct_text", "target": [1], "text": "a" * 2001},
            )

    def test_protections_unknown_audio_invalid_targets_and_policy_are_rejected(self):
        for case, edit in (
            (
                "protected-term",
                {"action": "correct_text", "target": [0], "text": "데이터"},
            ),
            ("unknown-gap", {"action": "remove_silence", "target": [1]}),
            (
                "multi-item-correction",
                {"action": "correct_text", "target": [0, 2], "text": "test"},
            ),
            (
                "multi-item-correction",
                {"action": "correct_text", "target": [0, 0], "text": "test"},
            ),
            ("trim-pause", {"action": "trim_silence", "target": [1], "keep_ms": 0}),
            ("trim-pause", {"action": "trim_silence", "target": [1], "keep_ms": 1200}),
            (
                "clean-transcript",
                {"action": "correct_text", "target": [0], "text": "오늘은"},
            ),
        ):
            with self.subTest(case=case, edit=edit), self.assertRaises(ValueError):
                self.validate(case, edit)
        with self.assertRaisesRegex(ValueError, "duration policy"):
            self.validate("delete-with-gap", policy=Policy(max_delete_ms=1000))
        snapshot = copy.deepcopy(self.cases["delete-with-gap"]["input"])
        snapshot["items"][1]["source_run"] = 99
        with self.assertRaisesRegex(ValueError, "source-run"):
            self.validate("delete-with-gap", snapshot=snapshot)

    def test_caption_changes_cannot_hide_affected_or_protected_content(self):
        for mutation in (
            lambda s: s.update(complete_paragraphs=[]),
            lambda s: s.update(protected_ids=[0]),
            lambda s: s["items"][1].update(speaker="other"),
            lambda s: s["items"][1].update(language="en"),
        ):
            snapshot = copy.deepcopy(self.cases["merge-caption"]["input"])
            mutation(snapshot)
            with self.assertRaises(ValueError):
                self.validate("merge-caption", snapshot=snapshot)
        with self.assertRaisesRegex(ValueError, "boundary"):
            self.validate("merge-caption", {"action": "split_paragraph", "target": [1]})

    def test_english_and_korean_corrections_and_captions_preserve_language(self):
        for language in ("en", "eng", "ko", "kor"):
            with self.subTest(language=language):
                english = language in ("en", "eng")
                words = (
                    ["Neurel", "networks.", "Next:", "attention."]
                    if english
                    else ["신경먕을", "학습합니다.", "다음은", "어텐션입니다."]
                )
                spec = {
                    "id": f"language-{language}",
                    "language": language,
                    "brief": "Preserve the spoken language.",
                    "paragraphs": [
                        [{"id": f"w{i}", "text": word} for i, word in enumerate(words)]
                    ],
                    "history": [
                        {
                            "action": "trim_silence",
                            "before_ms": 1200,
                            "after_ms": 400,
                            "origin": "synthetic",
                        }
                    ],
                }
                document = document_for(spec)
                history_document, paths = execute_history(
                    copy.deepcopy(document), spec, self.root
                )
                self.assertEqual(
                    {
                        i["language"]
                        for i in history_document["content"]
                        if i["type"] == "paragraph_start"
                    },
                    {language},
                )
                for relative in paths:
                    replay(
                        list(
                            map(
                                json.loads,
                                (self.root / relative).read_text().splitlines(),
                            )
                        )
                    )
                bundle = snapshot_for(document, "w0", "Preserve the spoken language.")
                self.assertEqual(bundle["input"]["language"], language)
                self.assertEqual(
                    {i["language"] for i in bundle["input"]["items"]}, {language}
                )
                correction = "Neural" if english else "신경망을"
                corrected = apply_edit(
                    document,
                    bundle,
                    {"action": "correct_text", "target": [0], "text": correction},
                )
                expected = copy.deepcopy(document)
                expected["content"][1].update(text=correction, conf=1)
                self.assertEqual(corrected, expected)
                split = apply_edit(
                    document, bundle, {"action": "split_paragraph", "target": [2]}
                )
                self.assertEqual(
                    [
                        i["language"]
                        for i in split["content"]
                        if i["type"] == "paragraph_start"
                    ],
                    [language, language],
                )
                merged = apply_edit(
                    split,
                    snapshot_for(split, "w2", "Keep the original caption grouping."),
                    {"action": "merge_paragraph", "target": [2]},
                )
                self.assertEqual(merged, document)

    def test_unknown_language_is_not_assumed_korean_or_safe_for_caption_edits(self):
        for language in (None, "", "und"):
            with self.subTest(language=language):
                document = copy.deepcopy(self.sessions["merge-caption"]["before"])
                for item in document["content"]:
                    if item["type"] == "paragraph_start":
                        if language is None:
                            item.pop("language")
                        else:
                            item["language"] = language
                bundle = snapshot_for(document, "w1", "Preserve uncertain boundaries.")
                self.assertEqual(bundle["input"]["language"], "und")
                self.assertEqual(
                    {i["language"] for i in bundle["input"]["items"]}, {"und"}
                )
                with self.assertRaisesRegex(ValueError, "languages"):
                    apply_edit(
                        document, bundle, {"action": "merge_paragraph", "target": [1]}
                    )

    def test_stale_requests_source_mappings_and_actor_corruption_fail(self):
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.validate(
                "split-caption",
                current=REQUEST.model_copy(update={"interaction_epoch": 1}),
            )
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.validate("split-caption", finish_reason="length")
        row = self.sessions["multi-item-correction"]
        bundle = snapshot_for(row["before"], "w0", "Keep wording")
        changed = copy.deepcopy(row["before"])
        changed["content"][1]["text"] = "different"
        with self.assertRaisesRegex(ValueError, "Stale"):
            apply_edit(changed, bundle, row["prediction"])
        bundle["mapping"]["0"]["uuid"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Alias"):
            apply_edit(row["before"], bundle, row["prediction"])
        events = copy.deepcopy(row["events"])
        events[1]["actor"] = "human"
        with self.assertRaisesRegex(ValueError, "actor"):
            replay(events)

    def test_actual_serializer_breaks_source_runs_and_never_invents_alignment(self):
        document = copy.deepcopy(self.sessions["multi-item-correction"]["before"])
        next(
            i for i in reversed(document["content"]) if i["type"] == "paragraph_start"
        )["language"] = "en"
        document["content"][2]["sourceStart"] += 2
        bundle = snapshot_for(document, "w0", "Preserve wording")
        self.assertEqual(bundle["input"]["language"], "ko")
        self.assertEqual(
            snapshot_for(document, "h0", "Preserve wording")["input"]["language"], "en"
        )
        self.assertEqual(
            {i["language"] for i in bundle["input"]["items"]}, {"ko", "en"}
        )
        self.assertNotEqual(
            bundle["input"]["items"][0]["source_run"],
            bundle["input"]["items"][1]["source_run"],
        )
        self.assertTrue(
            all(i["boundary_status"] == "unverified" for i in bundle["input"]["items"])
        )
        with self.assertRaises(ValueError):
            apply_edit(
                document,
                bundle,
                {"action": "correct_text", "target": [0, 1], "text": "머신 러닝을"},
            )


if __name__ == "__main__":
    unittest.main()
