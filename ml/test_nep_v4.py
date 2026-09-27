import copy
import json
import subprocess
from pathlib import Path
import unittest

from caption_facts import export_metrics
from context_policy import (
    ContextBuilder,
    ContextPolicy,
    FeedbackLedger,
    PredictionSession,
)
from format_comparison import (
    BUDGETS,
    build_format_comparison,
    compact,
    parse_rewrite,
    serialize_rewrite,
)
from model_input import messages, parse_snapshot
from nep_v2 import Envelope
from replay_session import session_hash
from session_cases import apply_edit
from synthetic_pilot import document_for, symbolic_evidence
from test_context_policy import Tokenizer


class ComparisonTokenizer(Tokenizer):
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return list(range(len(text.encode()) // 3 + 1))


def builder(max_items=64, **kwargs):
    return ContextBuilder(
        ComparisonTokenizer(),
        "shared rules",
        ContextPolicy(
            schema_version=4,
            version="cursor-caption-v2",
            max_items=max_items,
            input_tokens=10000,
            context_tokens=11024,
            output_tokens=1024,
        ),
        **kwargs,
    )


def short_document():
    return document_for(
        {
            "language": "en",
            "paragraphs": [
                [
                    {"id": "a", "text": "alpha", "ms": 600},
                    {"id": "gap", "text": "", "ms": 1400},
                    {"id": "b", "text": "beta", "ms": 900},
                    {"id": "c", "text": "gamma", "ms": 700},
                ],
                [
                    {"id": "d", "text": "delta", "ms": 1000},
                    {"id": "e", "text": "epsilon", "ms": 1000},
                ],
            ],
        }
    )


REQUEST = Envelope(document_id="doc", request_id="r", revision=0, interaction_epoch=0)


class CaptionContractTests(unittest.TestCase):
    def long_document(self):
        return document_for(
            {
                "language": "ko",
                "paragraphs": [
                    [
                        {"id": f"w{n}", "text": f"문장{n}😀", "ms": 200}
                        for n in range(120)
                    ],
                    [{"id": "following", "text": "다음 문장.", "ms": 1500}],
                ],
            }
        )

    def test_cropped_split_uses_full_facts_and_runtime_wire_parity(self):
        document = self.long_document()
        shared = builder(max_items=16)
        offline = shared.build(document, "w60", "문장 경계를 나누세요.", as_of=0)
        runtime = PredictionSession("doc", shared).prepare(
            document, "w60", "문장 경계를 나누세요.", as_of=0
        )
        self.assertEqual(offline, runtime)
        self.assertEqual(
            messages(offline["input"], "rules"), messages(runtime["input"], "rules")
        )
        wire = json.loads(messages(offline["input"], "rules")[1]["content"])
        self.assertIsInstance(wire["items"][0], list)
        self.assertEqual(parse_snapshot(wire).model_dump(), offline["input"])
        self.assertEqual(offline["input"]["complete_paragraphs"], [])
        fact = offline["input"]["caption_facts"][0]
        self.assertEqual(
            (fact["first_id"], fact["last_id"], fact["duration_ms"]), (0, 119, 23999)
        )
        after = apply_edit(
            document, offline, {"action": "split_paragraph", "target": [60]}
        )
        self.assertEqual(
            sum(i["type"] == "paragraph_start" for i in after["content"]), 3
        )
        self.assertEqual(
            [i for i in after["content"] if "source" in i],
            [i for i in document["content"] if "source" in i],
        )
        self.assertEqual(
            fact["text_utf16"],
            len(" ".join(f"문장{n}😀" for n in range(120)).encode("utf-16-le")) // 2,
        )

    def test_hidden_protection_tampered_facts_false_completeness_and_crop_merge_fail(
        self,
    ):
        doc = self.long_document()
        base = builder(max_items=16)
        protected = base.build(doc, "w60", "group", as_of=0, protected=["w1"])
        self.assertEqual(protected["input"]["protected_ids"], [])
        with self.assertRaisesRegex(ValueError, "protected"):
            apply_edit(doc, protected, {"action": "split_paragraph", "target": [60]})
        clean = base.build(doc, "w60", "group", as_of=0)
        tampered = copy.deepcopy(clean)
        tampered["input"]["caption_facts"][0]["duration_ms"] += 2000
        with self.assertRaisesRegex(ValueError, "full current document"):
            apply_edit(doc, tampered, {"action": "split_paragraph", "target": [60]})
        tampered = copy.deepcopy(clean)
        tampered["input"]["complete_paragraphs"] = [0]
        with self.assertRaisesRegex(ValueError, "Complete-paragraph"):
            parse_snapshot(tampered["input"])
        boundary = base.build(doc, "following", "group", as_of=0)
        with self.assertRaisesRegex(ValueError, "Insufficient visible context"):
            apply_edit(doc, boundary, {"action": "merge_paragraph", "target": [120]})

    def test_profile_accounts_for_export_splitting_and_rejects_short_cues(self):
        doc = short_document()
        base = builder()
        plain = base.build(doc, "b", "group", as_of=0)
        with self.assertRaisesRegex(ValueError, "below the profile minimum"):
            apply_edit(doc, plain, {"action": "split_paragraph", "target": [3]})
        split = base.build(
            doc, "b", "group", as_of=0, caption_profile={"export_line_length": 6}
        )
        fact = split["input"]["caption_facts"][0]
        self.assertEqual(fact["exported"]["count"], 3)
        with self.assertRaisesRegex(ValueError, "below the profile minimum"):
            apply_edit(doc, split, {"action": "split_paragraph", "target": [2]})
        timed = [i for i in doc["content"] if "source" in i][:4]
        self.assertEqual(export_metrics([timed], 6), fact["exported"])

    def test_caption_metrics_match_the_actual_exporter_clock_and_line_splitting(self):
        doc = short_document()
        paragraphs = []
        for item in doc["content"]:
            if item["type"] == "paragraph_start":
                paragraphs.append([])
            elif "source" in item:
                paragraphs[-1].append(item)
        for limit in (None, 6):
            actual = json.loads(
                subprocess.check_output(
                    [
                        "node",
                        str(Path(__file__).with_name("verify_caption_metrics.cjs")),
                    ],
                    input=json.dumps({"document": doc, "line_length": limit}),
                    text=True,
                )
            )
            self.assertEqual(export_metrics(paragraphs, limit), actual)
        plain = builder().build(doc, "b", "group", as_of=0)
        self.assertEqual(plain["input"]["caption_facts"][0]["duration_ms"], 3599)

    def test_v4_feedback_survives_unrelated_edit_in_same_paragraph(self):
        doc = self.long_document()
        base, ledger = builder(), FeedbackLedger("doc")
        first = base.build(doc, "w60", "preserve words", as_of=0)
        edit = {"action": "correct_text", "target": [60], "text": "문장60"}
        ledger.propose("p", 1, doc, first, edit)
        ledger.record("p", "shown", 2)
        ledger.record("p", "dismissed", 3, reason="target")
        unrelated = base.build(doc, "w20", "preserve words", as_of=3)
        changed = apply_edit(
            doc,
            unrelated,
            {"action": "correct_text", "target": [20, 21], "text": "수정된 다른 문장"},
        )
        new = base.build(changed, "w60", "preserve words", as_of=4, feedback=ledger)
        alias = next(int(k) for k, v in new["mapping"].items() if v["uuid"] == "w60")
        self.assertTrue(new["input"]["recent_feedback"][0]["same_context"])
        self.assertTrue(ledger.suppressed(changed, new, {**edit, "target": [alias]}, 4))
        adjacent = next(i for i in changed["content"] if i["uuid"] == "w59")
        adjacent["text"] = "새로운 근거"
        updated = base.build(changed, "w60", "preserve words", as_of=5, feedback=ledger)
        self.assertFalse(updated["input"]["recent_feedback"][0]["same_context"])


class RewriteTests(unittest.TestCase):
    def setUp(self):
        self.document = short_document()
        self.builder = builder()
        self.bundle = self.builder.build(
            self.document,
            "b",
            "preserve intent",
            as_of=0,
            evidence=symbolic_evidence(self.document),
        )
        self.snapshot = parse_snapshot(self.bundle["input"])
        self.actions = [
            {"action": "no_action"},
            {"action": "correct_text", "target": [2, 3], "text": "corrected phrase"},
            {"action": "delete_span", "target": [0, 1]},
            {"action": "trim_silence", "target": [1], "keep_ms": 725},
            {"action": "remove_silence", "target": [1]},
            {"action": "split_paragraph", "target": [2]},
            {"action": "merge_paragraph", "target": [4]},
        ]

    def test_all_seven_formats_reconstruct_the_exact_executed_transaction(self):
        session = PredictionSession("doc", self.builder)
        wire = json.loads(messages(self.bundle["input"], "rules")[1]["content"])
        self.assertEqual(parse_snapshot(wire), self.snapshot)
        for raw in self.actions:
            with self.subTest(action=raw["action"]):
                rewritten = serialize_rewrite(self.snapshot, raw)
                parsed = parse_rewrite(
                    compact(rewritten), self.snapshot, REQUEST, REQUEST, "stop"
                ).model_dump()
                self.assertEqual(parsed, raw)
                self.assertEqual(
                    apply_edit(self.document, self.bundle, raw),
                    apply_edit(self.document, self.bundle, parsed),
                )
                runtime = session.validate(
                    compact(rewritten),
                    self.document,
                    self.bundle,
                    REQUEST,
                    REQUEST,
                    "stop",
                    0,
                    output_format="rewrite",
                )
                self.assertEqual(runtime["prediction"], raw)
        after = apply_edit(self.document, self.bundle, self.actions[3])
        pieces = [
            i
            for i in after["content"]
            if i.get("uuid") == "gap" or i.get("type") == "non_text"
        ]
        self.assertAlmostEqual(sum(i["length"] for i in pieces), 0.725)
        self.assertEqual(pieces[0]["sourceStart"], 0.6)
        self.assertAlmostEqual(pieces[-1]["sourceStart"] + pieces[-1]["length"], 2.0)

    def test_extra_changes_forged_sources_and_stale_or_truncated_outputs_fail(self):
        raw = self.actions[1]
        rewrite = serialize_rewrite(self.snapshot, raw)
        for change in (
            lambda value: value["after"][0]["ids"].reverse(),
            lambda value: value["after"][0].update(boundary_before=True),
            lambda value: value["after"][0].update(
                audio={"mode": "edges", "keep_ms": 200}
            ),
            lambda value: value["after"][0].update(source="invented"),
        ):
            bad = copy.deepcopy(rewrite)
            change(bad)
            with self.assertRaises(ValueError):
                parse_rewrite(compact(bad), self.snapshot, REQUEST, REQUEST, "stop")
        for current, finish in (
            (REQUEST.model_copy(update={"revision": 1}), "stop"),
            (REQUEST, "length"),
        ):
            with self.assertRaises(ValueError):
                parse_rewrite(compact(rewrite), self.snapshot, REQUEST, current, finish)

    def test_fixed_budget_measurement_keeps_same_input_and_reference(self):
        cases = [
            {
                "id": f"case-{n}",
                "input": self.bundle["input"],
                "expectedOutput": action,
                "acceptableOutputs": [],
            }
            for n, action in enumerate(self.actions)
        ]
        comparison = build_format_comparison(
            cases,
            {"pinned-a": ComparisonTokenizer(), "pinned-b": ComparisonTokenizer()},
            "json rules",
            "rewrite rules",
        )
        self.assertEqual(comparison["spec"]["budgets"], BUDGETS)
        for n, target in enumerate(comparison["targets"]):
            self.assertEqual(target["raw_targets"], [self.actions[n]])
            left = comparison["contexts"]["json"]["pinned-a"][n]
            right = comparison["contexts"]["rewrite"]["pinned-b"][n]
            self.assertEqual(left["messages"][1], right["messages"][1])
            self.assertEqual(
                left["semantic_input_sha256"], session_hash(self.bundle["input"])
            )
            self.assertEqual(left["raw_target_sha256"], right["raw_target_sha256"])
        overflow = copy.deepcopy(cases[1])
        overflow["expectedOutput"]["text"] = "x" * 1000
        with self.assertRaisesRegex(ValueError, "exceeds the frozen format reserve"):
            build_format_comparison(
                [overflow], {"pinned": ComparisonTokenizer()}, "rules", "rules"
            )

    def test_common_window_is_limited_by_both_format_prompts(self):
        document = document_for(
            {
                "paragraphs": [
                    [{"id": f"w{n}", "text": "word", "ms": 300} for n in range(64)]
                ]
            }
        )
        policy = ContextPolicy(
            schema_version=4, version="cursor-caption-v2", input_tokens=1800
        )
        left = ContextBuilder(
            ComparisonTokenizer(),
            "short",
            policy,
            budget_prompts={"rewrite": "long " * 200},
        ).build(document, "w32", "purpose", as_of=0)
        right = ContextBuilder(ComparisonTokenizer(), "long " * 200, policy).build(
            document, "w32", "purpose", as_of=0
        )
        self.assertEqual(left["input"], right["input"])
        self.assertEqual(left["mapping"], right["mapping"])
        self.assertLessEqual(
            left["context"]["format_token_counts"]["rewrite"]["primary"], 1800
        )


if __name__ == "__main__":
    unittest.main()
