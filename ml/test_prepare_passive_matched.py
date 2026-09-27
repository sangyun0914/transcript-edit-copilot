import copy
import gzip
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_passive_matched as matched
import prepare_passive_sft as smoke
import test_prepare_passive_curriculum as curriculum_fixture
from passive_bundle_v2 import weighted_indices


class MatchedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = curriculum_fixture.CurriculumPreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        fixture = self.fixture
        for name in matched.SOURCES:
            (fixture.source / name).write_bytes(
                Path(matched.__file__).with_name(name).read_bytes()
            )
        source_patch = patch.object(
            matched, "__file__", str(fixture.source / "prepare_passive_matched.py")
        )
        source_patch.start()
        self.addCleanup(source_patch.stop)
        fixture.output = fixture.root / "matched"
        self.cases = fixture.cases[:802]
        self.views = {
            "json": copy.deepcopy(fixture.tokens[:802]),
            "rewrite": copy.deepcopy(fixture.tokens[:802]),
        }
        for index, row in enumerate(self.views["rewrite"]):
            row["messages"][0]["content"] = "fixed rewrite policy"
            row["references"] = ['{"format_version":1,"scope":[],"after":[]}']
            primary = row["primary"]
            prefix = primary["completion_start"] + index % 13
            completion = [9] * (3 + index % 5) + [1]
            primary.update(
                input_ids=[5] * prefix + completion,
                attention_mask=[1] * (prefix + len(completion)),
                labels=[-100] * prefix + completion,
                completion_start=prefix,
            )
        release = json.loads((fixture.release / "manifest.json").read_bytes())
        payloads = {
            "matched/train-cases.jsonl": smoke.jsonl_bytes(self.cases),
            "prompt-rewrite.txt": b"fixed rewrite policy\n",
            "matched/eligibility.json": smoke.json_bytes(
                {"qualified": {"train": 802, "dev": 286}}
            ),
            **{
                f"tokenized/qwen35-9b/{fmt}-train.jsonl.gz": gzip.compress(
                    smoke.jsonl_bytes(self.views[fmt]), mtime=0
                )
                for fmt in matched.FORMATS
            },
        }
        for name, content in payloads.items():
            release["files_sha256"][name] = fixture.save(
                fixture.release / name, content
            )
        release["dependencies_sha256"]["ml/passive_formats_v2.py"] = smoke.sha256(
            (fixture.source / "passive_formats_v2.py").read_bytes()
        )
        fixture.release_sha = fixture.save(
            fixture.release / "manifest.json", smoke.json_bytes(release)
        )
        manifest_path = fixture.preparation / "manifest.json"
        original = json.loads(manifest_path.read_bytes())
        original["release_manifest_sha256"] = fixture.release_sha
        prep_sha = fixture.save(manifest_path, smoke.json_bytes(original))
        selection = json.loads(fixture.selection.read_bytes())
        selection["bindings"].update(
            release_manifest_sha256=fixture.release_sha,
            train_only_preparation={"sha256": prep_sha},
        )
        fixture.selection_sha = fixture.save(
            fixture.selection, smoke.json_bytes(selection)
        )
        run_path = fixture.completed / "input/configuration-attempt-03.json"
        run = json.loads(run_path.read_bytes())
        run["bindings"]["preparation"]["sha256"] = prep_sha
        run_sha = fixture.save(run_path, smoke.json_bytes(run))
        completion = json.loads(fixture.completion.read_bytes())
        completion["files_sha256"]["input/configuration-attempt-03.json"] = run_sha
        fixture.completion_sha = fixture.save(
            fixture.completion, smoke.json_bytes(completion)
        )
        self.original = json.loads(
            (fixture.preparation / "configuration.json").read_bytes()
        )

    def prepare(self):
        fixture = self.fixture
        receipt = {
            "status": "passed_local_paired_reference_readback",
            "train_cases": 802,
            "references_per_format": 802,
            "format_reference_checks": 1604,
            "canonical_message_checks": 1604,
            "dev_test_bodies_read": 0,
        }
        # Structural fixtures deliberately omit complete passive snapshots.
        with patch.object(matched, "semantic_readback", return_value=receipt):
            return matched.prepare(
                fixture.release,
                fixture.selection,
                fixture.completion,
                fixture.preparation,
                fixture.output,
                release_sha256=fixture.release_sha,
                selection_sha256=fixture.selection_sha,
                smoke_completion_sha256=fixture.completion_sha,
                seed="3407",
                project_root=fixture.root,
            )

    def test_shared_stream_frozen_bytes_and_format_specific_budgets(self):
        result = self.prepare()
        fixture = self.fixture
        bundle = matched.verify_bundle(fixture.output)
        indices = weighted_indices(self.cases, 802, "3407")
        self.assertNotEqual(indices, weighted_indices(self.cases, 802, 3407))
        self.assertEqual(
            [row["train_row_index"] for row in bundle["draw_plan"]], indices
        )
        self.assertEqual(result["optimizer_updates_per_format"], 201)
        self.assertEqual(result["final_update_rows"], 2)
        self.assertEqual(bundle["configuration"]["training"]["optimizer_updates"], 201)
        selected = []
        for action, language in smoke.SLOTS:
            eligible = [
                i
                for i, case in enumerate(self.cases)
                if case["expectedOutput"]["action"] == action
                and case["metadata"]["language"] == language
            ]
            selected.append(
                min(
                    eligible,
                    key=lambda i: (
                        -max(
                            len(self.views[fmt][i]["primary"]["input_ids"])
                            for fmt in matched.FORMATS
                        ),
                        self.cases[i]["id"],
                    ),
                )
            )
        for fmt in matched.FORMATS:
            view = bundle["configuration"]["views"][fmt]
            self.assertEqual(
                view["diagnostic_case_order"], [self.cases[i]["id"] for i in selected]
            )
            self.assertEqual(
                view["max_total_sequence_tokens"],
                sum(len(self.views[fmt][i]["primary"]["input_ids"]) for i in indices),
            )
            self.assertEqual(
                view["max_total_completion_tokens"],
                sum(
                    len(self.views[fmt][i]["primary"]["input_ids"])
                    - self.views[fmt][i]["primary"]["completion_start"]
                    for i in indices
                ),
            )
        self.assertNotEqual(
            result["views"]["json"]["max_total_completion_tokens"],
            result["views"]["rewrite"]["max_total_completion_tokens"],
        )
        for target, source in matched.PAYLOADS.items():
            self.assertEqual(
                (fixture.output / target).read_bytes(),
                (fixture.release / source).read_bytes(),
            )
        for path in fixture.preparation.rglob("*"):
            if path.is_file():
                self.assertEqual(
                    path.read_bytes(),
                    (
                        fixture.output
                        / "original-smoke"
                        / path.relative_to(fixture.preparation)
                    ).read_bytes(),
                )
        with self.assertRaisesRegex(ValueError, "fresh matched"):
            self.prepare()

    def test_mismatched_evidence_order_weights_and_masks_rejected(self):
        changes = [
            lambda rows: rows[0]["messages"][1].update(content="different evidence"),
            lambda rows: rows.reverse(),
            lambda rows: rows[0].update(sampling_weight=99),
            lambda rows: rows[0]["primary"]["labels"].__setitem__(0, 5),
            lambda rows: rows[0].update(split="dev"),
        ]
        for change in changes:
            views = copy.deepcopy(self.views)
            change(views["rewrite"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                matched.validate_views(
                    self.cases,
                    views,
                    {"en": "train", "ko": "train"},
                    {"json": "fixed policy", "rewrite": "fixed rewrite policy"},
                    self.original["tokenizer"],
                )

    def test_rebound_draw_order_seed_and_update_budget_rejected(self):
        self.prepare()
        fixture = self.fixture
        plan = json.loads((fixture.output / "draw-plan.json").read_bytes())
        changed = copy.deepcopy(plan)
        changed[0]["train_row_index"] = (changed[0]["train_row_index"] + 1) % 802
        fixture.rebind("draw-plan.json", changed)
        with self.assertRaisesRegex(ValueError, "weighted case order"):
            matched.verify_bundle(fixture.output)
        fixture.rebind("draw-plan.json", plan)
        config = json.loads((fixture.output / "configuration.json").read_bytes())
        changed = copy.deepcopy(config)
        changed["sampling"]["seed"] = 3407
        fixture.rebind("configuration.json", changed)
        with self.assertRaisesRegex(ValueError, "explicit string"):
            matched.verify_bundle(fixture.output)
        changed = copy.deepcopy(config)
        changed["training"]["optimizer_updates"] = 202
        fixture.rebind("configuration.json", changed)
        with self.assertRaisesRegex(ValueError, "Matched budgets"):
            matched.verify_bundle(fixture.output)

    def test_unregistered_test_body_rejected_without_opening(self):
        self.prepare()
        output = self.fixture.output
        forbidden = output / "test-cases.jsonl"
        forbidden.write_bytes(b"synthetic forbidden sentinel\n")
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files_sha256"][forbidden.name] = "0" * 64
        manifest_path.write_bytes(smoke.json_bytes(manifest))
        original_read = Path.read_bytes

        def guarded(path):
            if path == forbidden:
                self.fail("The verifier opened an unregistered test body.")
            return original_read(path)

        with patch.object(Path, "read_bytes", guarded):
            with self.assertRaisesRegex(ValueError, "train-only inventory"):
                matched.verify_bundle(output)

    def test_semantic_readback_rejects_changed_rewrite_target(self):
        from passive_formats_v2 import messages, serialize
        from test_passive_contract import CORRECTION, builder, document

        snapshot = builder().build(document(), "b", as_of=0)["input"]
        case = {
            "input": snapshot,
            "expectedOutput": CORRECTION,
            "acceptableOutputs": [{"action": "no_action"}],
        }
        views = {
            fmt: [
                {
                    "messages": messages(snapshot, fmt),
                    "references": [
                        serialize(snapshot, action, fmt)
                        for action in [CORRECTION, {"action": "no_action"}]
                    ],
                }
            ]
            for fmt in matched.FORMATS
        }
        result = matched.semantic_readback([case], views)
        self.assertEqual(result["format_reference_checks"], 4)
        views["rewrite"][0]["references"].reverse()
        with self.assertRaisesRegex(ValueError, "admitted actions"):
            matched.semantic_readback([case], views)


if __name__ == "__main__":
    unittest.main()
