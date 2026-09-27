import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from colab_bundle import build as bundle
from bilingual_iteration import build as build_bilingual
from evaluate import evaluate, read_rows
from grading import grade
from iteration import build, sha
from replay_session import read_events, replay, session_hash
from synthetic_pilot import build as pilot, document_for
from scenario_iteration import build as build_scenarios
from train_adapter import load_bundle, training_row
from preference_diagnostic import build as build_preferences, score as score_preferences
from fit_diagnostic import build as build_fit

HERE = Path(__file__).parent


class IterationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        document = document_for(
            {
                "paragraphs": [
                    [{"id": "a", "text": "컴퓨터"}],
                    [{"id": "b", "text": "컴퓨터"}],
                ]
            }
        )
        self.project = self.root / "source.audapolis"
        with zipfile.ZipFile(self.project, "w") as archive:
            archive.writestr("document.json", json.dumps({"version": 3, **document}))
        self.recipe = self.root / "recipe.json"
        recipe = json.loads((HERE / "fixtures/iteration-01.json").read_text())
        recipe["source_group"]["project_sha256"] = sha(self.project)
        recipe["training_terms"] = [{"canonical": "컴퓨터", "variant": "컴퓨타"}]
        recipe["development_scenarios"] = recipe["development_scenarios"][:1]
        self.recipe.write_text(json.dumps(recipe))
        self.iteration = self.root / "iteration"
        self.report = build(self.recipe, self.project, self.iteration)
        self.regression = self.root / "regression"
        pilot(HERE / "fixtures/synthetic-v2.json", self.regression)

    def pack(self):
        archive = self.root / "bundle.zip"
        bundle(self.iteration, self.regression, archive)
        root = self.root / "unpacked"
        with zipfile.ZipFile(archive) as file:
            file.extractall(root)
        return root

    def test_generated_splits_provenance_and_executed_history(self):
        self.assertEqual(self.report["counts"], {"train": 5, "dev": 5})
        train, dev, regression = load_bundle(self.pack())
        self.assertEqual(len(regression), 17)
        self.assertFalse(
            {c["metadata"]["source_group_id"] for c in train}
            & {c["metadata"]["source_group_id"] for c in dev}
        )
        for case in train + dev:
            events = read_events(self.iteration / case["metadata"]["session"])
            _, summary = replay(events)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["actor"], "synthetic")
            history = case["metadata"]["history_session"]
            if history:
                replay(read_events(self.iteration / history))
        self.assertEqual(
            [c["expectedOutput"]["action"] for c in train].count("correct_text"), 2
        )

    def test_bundle_rejects_tampering_and_resealed_human_feedback(self):
        root = self.pack()
        path = root / "train-cases.jsonl"
        rows = read_rows(path)
        rows[0]["metadata"]["requires_regeneration"] = True
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_bundle(root)
        manifest = json.loads((root / "bundle.json").read_text())
        manifest["files"][path.name] = sha(path)
        (root / "bundle.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "feedback-only"):
            load_bundle(root)

    def test_source_group_cannot_be_reassigned_to_development(self):
        recipe = json.loads(self.recipe.read_text())
        recipe["development_scenarios"][0]["group"] = recipe["source_group"]["id"]
        self.recipe.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "assigned uniquely"):
            build(self.recipe, self.project, self.root / "bad")

    def test_loss_masks_prompt_and_refuses_truncation(self):
        class Tokenizer:
            eos_token_id = 9

            def apply_chat_template(self, conversation, **kwargs):
                assert kwargs["return_dict"] is False
                self.conversation = conversation
                return [1, 2, 3]

            def encode(self, text, **kwargs):
                return [4, 5]

        tokenizer = Tokenizer()
        case = read_rows(self.iteration / "train-cases.jsonl")[0]
        row = training_row(tokenizer, case, "system")
        self.assertEqual(row["labels"], [-100, -100, -100, 4, 5, 9])
        self.assertEqual(len(tokenizer.conversation), 2)
        self.assertNotIn("metadata", tokenizer.conversation[1]["content"])
        tokenizer.encode = lambda *a, **k: [4] * 4096
        with self.assertRaisesRegex(ValueError, "silently truncate"):
            training_row(tokenizer, case, "system")

    def test_equivalent_minimal_correction_and_conservative_grading(self):
        cases = {c["id"]: c for c in read_rows(self.regression / "pilot-cases.jsonl")}
        executions = {
            c["id"]: c
            for c in json.loads((self.regression / "replay-fixtures.json").read_text())
        }
        case = cases["multi-item-correction"]
        expected = case["expectedOutput"]
        print_expected = [
            i["text"] for i in case["input"]["items"] if i["id"] in expected["target"]
        ]
        self.assertEqual(len(print_expected), 2)
        minimal = {
            "action": "correct_text",
            "target": [expected["target"][1]],
            "text": expected["text"].split(" ", 1)[1],
        }
        result = grade(case, minimal, executions[case["id"]])
        self.assertFalse(result["exact_match"])
        self.assertTrue(result["outcome_match"])
        wrong = {**minimal, "text": "wrong"}
        self.assertFalse(grade(case, wrong, executions[case["id"]])["outcome_match"])
        unknown = copy.deepcopy(case)
        unknown["expectedOutput"] = None
        self.assertIsNone(grade(unknown, minimal)["outcome_match"])
        report = evaluate(
            [case],
            [{"case_id": case["id"], "valid": True, "prediction": minimal}],
            executions,
        )
        self.assertEqual(report["outcome_matches"], 1)
        invalid = {**minimal, "target": [9999]}
        report = evaluate(
            [case],
            [{"case_id": case["id"], "valid": True, "prediction": invalid}],
            executions,
        )
        self.assertEqual(report["valid"], 0)

    def test_harder_scenarios_replay_admit_and_keep_unsupported_content(self):
        recipe = json.loads((HERE / "fixtures/iteration-02.json").read_text())
        recipe["retained_manifest_sha256"] = sha(self.iteration / "manifest.json")
        path = self.root / "mixed-recipe.json"
        path.write_text(json.dumps(recipe))
        mixed = self.root / "mixed"
        report = build_scenarios(path, self.iteration, mixed)
        self.assertEqual(report["counts"], {"train": 32, "dev": 21})
        self.iteration = mixed
        packed = self.pack()
        train, dev, _ = load_bundle(packed)
        self.assertEqual(len({c["expectedOutput"]["action"] for c in train}), 7)
        executions = {r["id"] for r in read_rows(packed / "eval-executions.jsonl")}
        self.assertTrue({c["id"] for c in train + dev} <= executions)
        abstain_report = evaluate(
            train,
            [
                {
                    "case_id": c["id"],
                    "valid": True,
                    "prediction": {"action": "no_action"},
                }
                for c in train
            ],
            {},
        )
        self.assertEqual(abstain_report["outcome_matches"], 16)
        self.assertEqual(abstain_report["by_action"]["correct_text"]["missed_edits"], 6)
        clean = next(
            c for c in train if c["metadata"]["scenario_variant"] == "already-correct"
        )
        false_edit_report = evaluate(
            [clean],
            [
                {
                    "case_id": clean["id"],
                    "valid": True,
                    "prediction": {
                        "action": "correct_text",
                        "target": clean["input"]["selection"],
                        "text": "불필요한 변경",
                    },
                }
            ],
            {},
        )
        self.assertEqual(
            false_edit_report["by_action"]["no_action"]["unnecessary_proposals"], 1
        )
        for case in train + dev:
            replay(read_events(mixed / case["metadata"]["session"]))
            for history in case["metadata"].get("history_sessions", []):
                replay(read_events(mixed / history))
            if case["metadata"]["scenario_variant"] in (
                "missing-name",
                "missing-content",
                "unfinished-sentence",
                "unverified-gap",
            ):
                self.assertEqual(case["expectedOutput"], {"action": "no_action"})
        recipe["groups"][0]["id"] = recipe["groups"][2]["id"]
        path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "assigned uniquely"):
            build_scenarios(path, self.root / "iteration", self.root / "bad-mixed")

    def test_bilingual_histories_change_evidence_without_leaking_answers_or_targets(
        self,
    ):
        recipe = json.loads((HERE / "fixtures/iteration-03.json").read_text())
        recipe["retained_manifest_sha256"] = sha(self.iteration / "manifest.json")
        recipe["groups"] = [recipe["groups"][0], recipe["groups"][-1]]
        path = self.root / "bilingual.json"
        path.write_text(json.dumps(recipe))
        generated = self.root / "bilingual"
        report = build_bilingual(path, self.iteration, generated)
        self.assertEqual(report["new_counts"], {"train": 20, "dev": 20})
        self.iteration = generated
        train, dev, _ = load_bundle(self.pack())
        cases = [
            c
            for c in train + dev
            if c["metadata"].get("generator") == "bilingual-history-v1"
        ]
        matched = {}
        for case in cases:
            self.assertEqual(case["input"]["selection"], [])
            self.assertEqual(
                {i["language"] for i in case["input"]["items"]},
                {case["input"]["language"]},
            )
            _, summary = replay(read_events(generated / case["metadata"]["session"]))
            self.assertTrue(summary["complete"])
            for relative in case["metadata"]["history_sessions"]:
                history_document, _ = replay(read_events(generated / relative))
            if case["metadata"]["history_sessions"]:
                self.assertEqual(
                    session_hash(history_document), case["metadata"]["before_hash"]
                )
            comparison = case["metadata"]["comparison_group"]
            if comparison:
                matched.setdefault(comparison, []).append(case)
        for pair in matched.values():
            self.assertEqual(len(pair), 4)
            inputs = [
                {k: v for k, v in c["input"].items() if k != "recent_edits"}
                for c in pair
            ]
            self.assertTrue(all(value == inputs[0] for value in inputs))
            relevant = next(
                c for c in pair if c["metadata"]["history_condition"] == "relevant"
            )
            self.assertNotIn(
                relevant["expectedOutput"]["text"],
                " ".join(i["text"] for i in relevant["input"]["items"]),
            )
            self.assertNotEqual(
                relevant["expectedOutput"]["target"],
                [relevant["input"]["cursor_before"]],
            )
            self.assertTrue(
                all(
                    c["expectedOutput"]["action"] == "no_action"
                    for c in pair
                    if c is not relevant
                )
            )
        report = evaluate(
            cases,
            [
                {
                    "case_id": c["id"],
                    "valid": True,
                    "prediction": {"action": "no_action"},
                }
                for c in cases
            ],
            {},
        )
        self.assertEqual(set(report["by_language"]), {"en", "ko"})
        self.assertEqual(report["by_history_condition"]["relevant"]["missed_edits"], 4)

    def test_fit_subset_preserves_development_and_cannot_select_its_groups(self):
        recipe = json.loads((HERE / "fixtures/iteration-03.json").read_text())
        recipe["retained_manifest_sha256"] = sha(self.iteration / "manifest.json")
        recipe["groups"] = [recipe["groups"][0], recipe["groups"][-1]]
        path = self.root / "bilingual.json"
        path.write_text(json.dumps(recipe))
        source = self.root / "bilingual"
        build_bilingual(path, self.iteration, source)
        preferences = self.root / "preferences"
        build_preferences(HERE / "fixtures/pause-preferences-v1.json", preferences)
        fit_recipe = json.loads((HERE / "fixtures/correction-fit-v1.json").read_text())
        fit_recipe["source_manifest_sha256"] = sha(source / "manifest.json")
        fit_recipe["train_groups"] = [recipe["groups"][0]["id"]]
        fit_recipe["train_scenarios"] = [
            "history-relevant-short",
            "history-conflicting-short",
        ]
        fit_path = self.root / "fit.json"
        fit_path.write_text(json.dumps(fit_recipe))
        self.iteration = self.root / "fit"
        build_fit(fit_path, source, preferences, self.iteration)
        train, dev, regression = load_bundle(self.pack())
        self.assertEqual(len(train), 2)
        self.assertEqual(len(regression), 17)
        original_train = {c["id"]: c for c in read_rows(source / "train-cases.jsonl")}
        self.assertEqual(train, [original_train[c["id"]] for c in train])
        self.assertEqual(
            dev,
            read_rows(source / "dev-cases.jsonl")
            + read_rows(preferences / "dev-cases.jsonl"),
        )
        fit_recipe["train_groups"] = [recipe["groups"][-1]["id"]]
        fit_path.write_text(json.dumps(fit_recipe))
        with self.assertRaisesRegex(ValueError, "exactly once in training"):
            build_fit(fit_path, source, preferences, self.root / "bad-fit")

    def test_bilingual_generation_rejects_reserved_groups_and_changed_retention_data(
        self,
    ):
        recipe = json.loads((HERE / "fixtures/iteration-03.json").read_text())
        recipe["retained_manifest_sha256"] = sha(self.iteration / "manifest.json")
        original_id = recipe["groups"][0]["id"]
        recipe["groups"][0]["id"] = recipe["reserved_test_groups"][0]
        path = self.root / "bilingual.json"
        path.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, "reserved test"):
            build_bilingual(path, self.iteration, self.root / "bad-reserved")
        recipe["groups"][0]["id"] = original_id
        path.write_text(json.dumps(recipe))
        with (self.iteration / "dev-cases.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "differs from its manifest"):
            build_bilingual(path, self.iteration, self.root / "bad-retention")


class PreferenceDiagnosticTests(unittest.TestCase):
    def test_matched_durations_cutoff_and_executed_histories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "preferences"
            manifest = build_preferences(
                HERE / "fixtures/pause-preferences-v1.json", root
            )
            self.assertEqual(manifest["actions"], {"trim_silence": 32, "no_action": 32})
            cases = read_rows(root / "dev-cases.jsonl")
            executions = {r["id"]: r for r in read_rows(root / "executions.jsonl")}
            for case in cases:
                self.assertFalse(case["metadata"]["training_eligible"])
                self.assertEqual(case["input"]["selection"], [])
                self.assertEqual(len(case["input"]["items"]), 3)
                for path in case["metadata"]["history_sessions"]:
                    history_document, _ = replay(read_events(root / path))
                if case["metadata"]["history_sessions"]:
                    self.assertEqual(
                        session_hash(history_document), case["metadata"]["before_hash"]
                    )
                final, summary = replay(read_events(root / case["metadata"]["session"]))
                self.assertTrue(summary["complete"])
                self.assertEqual(
                    session_hash(final), session_hash(executions[case["id"]]["after"])
                )
            results = [
                {"case_id": c["id"], "valid": True, "prediction": c["expectedOutput"]}
                for c in cases
            ]
            report = score_preferences(cases, results, executions)
            self.assertEqual(report["outcome_matches"], 64)
            self.assertEqual(
                report["matched_history"]["all_three_durations_correct"], 4
            )
            wrong = copy.deepcopy(results)
            for row in wrong:
                if row["prediction"]["action"] == "trim_silence":
                    row["prediction"]["keep_ms"] = 400
            report = score_preferences(cases, wrong, executions)
            self.assertEqual(report["outcome_matches"], 32)
            self.assertEqual(report["matched_history"]["both_500_and_1500_correct"], 0)
            for case in cases:
                family = case["metadata"]["scenario_variant"]
                if family == "cutoff-2999":
                    self.assertEqual(case["expectedOutput"], {"action": "no_action"})
                elif family in ("cutoff-3000", "cutoff-3001"):
                    self.assertEqual(case["expectedOutput"]["keep_ms"], 1500)
            with self.assertRaisesRegex(ValueError, "one prediction"):
                score_preferences(cases, results + results[:1], executions)
            with self.assertRaisesRegex(ValueError, "exactly"):
                score_preferences(cases, results[:-1], executions)
            changed = copy.deepcopy(cases)
            changed[0]["input"]["task_brief"] += " Different instruction."
            with self.assertRaisesRegex(ValueError, "differ beyond"):
                score_preferences(changed, results, executions)


if __name__ == "__main__":
    unittest.main()
