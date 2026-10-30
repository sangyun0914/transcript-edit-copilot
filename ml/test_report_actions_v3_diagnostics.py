import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_actions_v3_diagnostics as diagnostics


def case(identity="edit", refs=None, cursor=1, lineage=None):
    refs = refs or [{"action": "correct_text", "target": [1], "text": "fixed"}]
    return {
        "id": identity,
        "split": "dev",
        "language": "en",
        "source": {
            "source_group": "fixture",
            "provenance": "symbolic",
            "natural_audio": False,
        },
        "lineage": {"stratum": "authored_symbolic", **(lineage or {})},
        "judgment": {
            "status": "preserve"
            if refs[0]["action"] == "no_action"
            else "edit_required"
        },
        "primary": refs[0],
        "accepted_predictions": refs,
        "input": {
            "cursor_before": cursor,
            "recent_edits": [],
            "items": [],
            "protected_ids": [],
            "allowed_actions": sorted(diagnostics.EDIT_ACTIONS),
        },
    }


def grade(source, **changes):
    meta = diagnostics.case_metadata(source)
    result = {
        **{
            key: meta[key]
            for key in (
                "case_id",
                "judgment",
                "reference_action",
                "language",
                "source_group",
                "stratum",
            )
        },
        **{key: False for key in diagnostics.BOOLEAN_FIELDS},
        "status": "request_failure",
        "predicted_action": None,
        "source_integrity": None,
    }
    result.update(changes)
    return result


def observation(source, prediction, format_name="json", **changes):
    return {
        "case_id": source["id"],
        "format": format_name,
        "status": "completed",
        "finish_reason": "stop",
        "text": json.dumps(prediction),
        **changes,
    }


def successful_grade(source):
    return grade(
        source,
        valid=True,
        displayed=True,
        useful=True,
        outcome_match=True,
        status="completed",
        predicted_action=source["primary"]["action"],
        source_integrity="passed",
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, cases, grades, observations, format_name="json"):
    release, graded, observed = root / "release", root / "grade", root / "observed"
    write_rows(release / "cases/dev.jsonl", cases)
    write_json(
        release / "manifest.json",
        {
            "counts": {"dev": len(cases)},
            "files_sha256": {"cases/dev.jsonl": sha(release / "cases/dev.jsonl")},
        },
    )
    write_rows(graded / "graded.jsonl", grades)
    write_rows(observed / "observations.jsonl", observations)
    n = len(grades)
    needed = sum(row["judgment"] == "edit_required" for row in grades)
    shown, useful = (
        sum(row["displayed"] for row in grades),
        sum(row["useful"] for row in grades),
    )
    pairs = {
        "precision": (useful, shown),
        "recall": (useful, needed),
        "preservation": (sum(row["preserved"] for row in grades), n - needed),
        "model_preservation": (
            sum(row["preserved"] and row["model_abstained"] for row in grades),
            n - needed,
        ),
        "coverage": (shown, n),
        "validity": (sum(row["valid"] for row in grades), n),
        "outcome_accuracy": (sum(row["outcome_match"] for row in grades), n),
    }
    saved = {
        "status": "graded_development",
        "final_test": False,
        "condition": "canonical_full_context",
        "format": format_name,
        "graded": {"sha256": sha(graded / "graded.jsonl")},
        "observations": {"sha256": sha(observed / "observations.jsonl")},
        "overall": {
            "requests": n,
            "status_counts": {
                status: sum(row["status"] == status for row in grades)
                for status in {row["status"] for row in grades}
            },
            **{
                key: {"numerator": a, "denominator": b, "value": a / b if b else None}
                for key, (a, b) in pairs.items()
            },
        },
    }
    write_json(graded / "report.json", saved)
    return release, [("fixture", graded, observed)]


def fixture_report(release, runs, output):
    with patch.object(
        diagnostics, "ADMITTED_DEV_SHA256", sha(release / "cases/dev.jsonl")
    ):
        return diagnostics.report(release, runs, output)


class DiagnosticsTests(unittest.TestCase):
    def test_action_and_target_must_match_the_same_accepted_alternative(self):
        source = case(
            refs=[
                {"action": "correct_text", "target": [1], "text": "fixed"},
                {"action": "delete_span", "target": [2]},
            ]
        )
        prediction = {"action": "correct_text", "target": [2], "text": "fixed"}
        row = diagnostics.diagnose_case(
            diagnostics.case_metadata(source),
            grade(source),
            observation(source, prediction),
            "json",
        )
        self.assertEqual(
            row["exact_diagnostics"],
            {
                "action_type_match": True,
                "full_target_span_match": True,
                "same_alternative_joint_match": False,
                "full_prediction_match": False,
            },
        )
        for prediction in (
            {"action": "correct_text", "target": [1, 2], "text": "fixed"},
            {"action": "no_action", "target": [1]},
            {"action": "correct_text", "target": [True], "text": "fixed"},
        ):
            row = diagnostics.diagnose_case(
                diagnostics.case_metadata(source),
                grade(source),
                observation(source, prediction),
                "json",
            )
            self.assertFalse(row["exact_diagnostics"]["full_target_span_match"])
            self.assertFalse(row["exact_diagnostics"]["same_alternative_joint_match"])

    def test_invalid_and_missing_observations_remain_in_edit_denominators(self):
        sources = [case(identity) for identity in ("good", "invalid", "missing")]
        grades = [
            successful_grade(sources[0]),
            grade(sources[1], status="invalid_output"),
            grade(sources[2]),
        ]
        observed = [
            observation(sources[0], sources[0]["primary"]),
            observation(sources[1], None, text="{bad"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            release, runs = fixture(Path(directory), sources, grades, observed)
            result = fixture_report(release, runs, Path(directory) / "output")["runs"][
                "fixture"
            ]["overall"]
        self.assertEqual(result["counts"]["requests"], 3)
        for name in (
            "action_type_match",
            "full_target_span_match",
            "same_alternative_joint_match",
        ):
            self.assertEqual(
                result["edit_diagnostics"][name],
                {"numerator": 1, "denominator": 3, "value": 1 / 3},
            )
        self.assertEqual(
            result["output_diagnostics"]["missing_observation_case_ids"], ["missing"]
        )
        self.assertEqual(
            result["output_diagnostics"]["json_syntax_parseability"]["denominator"], 3
        )
        self.assertEqual(result["canonical_metrics"]["recall"]["denominator"], 3)

    def test_cursor_provenance_is_exact_nested_boolean_and_preserve_is_not_a_position(
        self,
    ):
        refs = [
            {"action": "correct_text", "target": [1, 2], "text": "fixed"},
            {"action": "delete_span", "target": [4]},
        ]
        missing = case(
            refs=refs, cursor=4, lineage={"selection_method": "source_first"}
        )
        missing["cursor_selection_before_judgment"] = True
        meta = diagnostics.case_metadata(missing)
        self.assertEqual(meta["cursor_selection_before_judgment"], "unknown")
        self.assertEqual(
            meta["cursor_at_first_accepted_target"], "at_first_accepted_target"
        )
        explicit = case(
            refs=refs, cursor=2, lineage={"cursor_selection_before_judgment": False}
        )
        self.assertEqual(
            diagnostics.case_metadata(explicit)["cursor_selection_before_judgment"],
            "false",
        )
        self.assertEqual(
            diagnostics.case_metadata(explicit)["cursor_at_first_accepted_target"],
            "elsewhere",
        )
        self.assertEqual(
            diagnostics.case_metadata(case(cursor=True))[
                "cursor_at_first_accepted_target"
            ],
            "unknown",
        )
        preserve = case(refs=[{"action": "no_action"}])
        row = diagnostics.diagnose_case(
            diagnostics.case_metadata(preserve),
            grade(preserve),
            observation(preserve, {"action": "no_action"}),
            "json",
        )
        self.assertEqual(
            row["cursor_at_first_accepted_target"], "not_applicable_preserve"
        )
        self.assertIsNone(row["exact_diagnostics"]["full_target_span_match"])

    def test_preservation_flags_are_not_redefined_as_display_avoidance(self):
        sources = [
            case(identity, [{"action": "no_action"}])
            for identity in (
                "no_action",
                "bad_no_action",
                "bad_merge",
                "suppressed",
                "shown",
                "failed",
                "execution",
            )
        ]
        completed = {"status": "completed", "valid": True, "source_integrity": "passed"}
        grades = [
            grade(
                sources[0],
                **completed,
                predicted_action="no_action",
                model_abstained=True,
                preserved=True,
                outcome_match=True,
            ),
            grade(sources[1], status="invalid_output"),
            grade(sources[2], status="invalid_output"),
            grade(
                sources[3],
                **completed,
                predicted_action="delete_span",
                suppressed=True,
                preserved=True,
                outcome_match=True,
            ),
            grade(
                sources[4], **completed, predicted_action="delete_span", displayed=True
            ),
            grade(sources[5]),
            grade(
                sources[6],
                status="execution_failure",
                valid=True,
                source_integrity="failed",
                predicted_action="delete_span",
            ),
        ]
        predictions = [
            {"action": "no_action"},
            {"action": "no_action", "target": [1]},
            {"action": "merge_paragraph", "target": [1]},
            *[{"action": "delete_span", "target": [1]} for _ in range(4)],
        ]
        observed = [
            observation(source, prediction)
            for source, prediction in zip(sources, predictions)
        ]
        observed[5].update(status="timeout", text=None)
        with tempfile.TemporaryDirectory() as directory:
            release, runs = fixture(Path(directory), sources, grades, observed)
            result = fixture_report(release, runs, Path(directory) / "output")["runs"][
                "fixture"
            ]
        metrics = result["overall"]["canonical_metrics"]
        self.assertEqual(metrics["preservation"]["numerator"], 2)
        self.assertEqual(metrics["model_preservation"]["numerator"], 1)
        self.assertEqual(metrics["validity"]["numerator"], 4)
        output = result["overall"]["output_diagnostics"]
        self.assertEqual(output["preserve_no_display"]["numerator"], 6)
        self.assertEqual(output["execution_completed"]["numerator"], 3)
        self.assertEqual(
            output["parseable_but_canonically_invalid_case_ids"],
            ["bad_merge", "bad_no_action"],
        )
        dispositions = result["preservation_dispositions"]
        self.assertEqual(dispositions["displayed_unnecessary"]["count"], 1)
        self.assertEqual(dispositions["feedback_suppressed_preserved"]["count"], 1)
        self.assertEqual(
            dispositions["invalid_output"]["raw_action_counts"],
            {"merge_paragraph": 1, "no_action": 1},
        )

    def test_noncanonical_format_keeps_grades_but_excludes_json_diagnostics(self):
        source = case()
        with tempfile.TemporaryDirectory() as directory:
            release, runs = fixture(
                Path(directory),
                [source],
                [successful_grade(source)],
                [observation(source, source["primary"], format_name="rewrite")],
                "rewrite",
            )
            result = fixture_report(release, runs, Path(directory) / "output")["runs"][
                "fixture"
            ]
        self.assertEqual(result["overall"]["counts"]["useful"], 1)
        self.assertEqual(
            result["overall"]["edit_diagnostics"]["excluded_noncanonical_format"], 1
        )
        self.assertEqual(
            result["overall"]["edit_diagnostics"]["action_type_match"]["denominator"], 0
        )
        self.assertEqual(
            result["cases"][0]["observation"]["exclusion"],
            "noncanonical_format_not_decoded",
        )

    def test_json_syntax_is_separate_from_an_unambiguous_action_object(self):
        source = case()
        for text in (
            "[1, 2]",
            '{"action":"delete_span","action":"correct_text","target":[1],"text":"fixed"}',
        ):
            row = diagnostics.diagnose_case(
                diagnostics.case_metadata(source),
                grade(source),
                observation(source, None, text=text),
                "json",
            )
            self.assertTrue(row["observation"]["syntactically_parseable_json"])
            self.assertFalse(row["exact_diagnostics"]["same_alternative_joint_match"])
        parsed, _ = diagnostics.parse_observation(
            {"text": '{"action":"correct_text","target":[NaN]}'}, "json"
        )
        self.assertFalse(parsed["syntactically_parseable_json"])

    def test_immutable_inputs_hash_binding_and_fresh_output(self):
        source = case()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, runs = fixture(
                root,
                [source],
                [successful_grade(source)],
                [observation(source, source["primary"])],
            )
            inputs = [path for path in root.rglob("*") if path.is_file()]
            before = {str(path): sha(path) for path in inputs}
            output = root / "output"
            result = fixture_report(release, runs, output)
            self.assertEqual(
                result["input_bindings"]["dev_cases"]["sha256"],
                before[str(release / "cases/dev.jsonl")],
            )
            self.assertEqual(before, {str(path): sha(path) for path in inputs})
            saved_output = (output / "diagnostics.json").read_bytes()
            self.assertTrue((output / "diagnostics.md").is_file())
            with self.assertRaisesRegex(ValueError, "fresh"):
                fixture_report(release, runs, output)
            self.assertEqual(saved_output, (output / "diagnostics.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "immutable"):
                fixture_report(release, runs, release / "new-output")
            observations_path = runs[0][2] / "observations.jsonl"
            observations_path.write_text(
                observations_path.read_text() + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash differs"):
                fixture_report(release, runs, root / "changed-input")
            self.assertFalse((root / "changed-input").exists())

    def test_missing_graded_and_foreign_or_duplicate_observations_fail_closed(self):
        source = case()
        for problem in ("missing_grade", "foreign", "duplicate"):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                observed = observation(source, source["primary"])
                release, runs = fixture(
                    root, [source], [successful_grade(source)], [observed]
                )
                if problem == "missing_grade":
                    saved = json.loads((root / "grade/report.json").read_text())
                    saved["overall"]["requests"] = 2
                    write_json(root / "grade/report.json", saved)
                else:
                    extra = copy.deepcopy(observed)
                    if problem == "foreign":
                        extra["case_id"] = "foreign"
                    write_rows(root / "observed/observations.jsonl", [observed, extra])
                with self.assertRaisesRegex(
                    ValueError, "Missing graded|Foreign/duplicate"
                ):
                    fixture_report(release, runs, root / "output")
                self.assertFalse((root / "output").exists())

    def test_same_ids_with_changed_cursor_or_reference_and_rehashed_manifest_are_rejected(
        self,
    ):
        for changed_field in ("cursor", "reference"):
            with (
                self.subTest(changed_field=changed_field),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                original = case()
                release, runs = fixture(
                    root,
                    [original],
                    [successful_grade(original)],
                    [observation(original, original["primary"])],
                )
                admitted_hash = sha(release / "cases/dev.jsonl")
                changed = copy.deepcopy(original)
                if changed_field == "cursor":
                    changed["input"]["cursor_before"] = 99
                else:
                    changed["accepted_predictions"][0]["target"] = [99]
                    changed["primary"] = changed["accepted_predictions"][0]
                write_rows(release / "cases/dev.jsonl", [changed])
                manifest = json.loads((release / "manifest.json").read_text())
                manifest["files_sha256"]["cases/dev.jsonl"] = sha(
                    release / "cases/dev.jsonl"
                )
                write_json(release / "manifest.json", manifest)
                with patch.object(diagnostics, "ADMITTED_DEV_SHA256", admitted_hash):
                    with self.assertRaisesRegex(ValueError, "pinned admitted"):
                        diagnostics.report(release, runs, root / "output")
                self.assertFalse((root / "output").exists())

    def test_history_rule_keeps_whole_item_equality_protection_and_first_match_order(
        self,
    ):
        snapshot = {
            "recent_edits": [
                {"action": "correct_text", "before": "bad", "after": "good"},
                {"action": "trim_silence", "after_ms": 1000},
            ],
            "protected_ids": [2],
            "allowed_actions": ["correct_text", "trim_silence"],
            "items": [
                {
                    "id": 1,
                    "text": "prefix bad",
                    "audio_class": "speech",
                    "boundary_status": "verified",
                    "duration_ms": 4000,
                },
                {
                    "id": 2,
                    "text": "bad",
                    "audio_class": "speech",
                    "boundary_status": "verified",
                    "duration_ms": 4000,
                },
                {
                    "id": 3,
                    "text": "",
                    "audio_class": "verified_silence",
                    "boundary_status": "verified",
                    "duration_ms": 4000,
                },
                {
                    "id": 4,
                    "text": "bad",
                    "audio_class": "speech",
                    "boundary_status": "verified",
                    "duration_ms": 4000,
                },
            ],
        }
        result = diagnostics.history_rule_predicates(snapshot)
        self.assertTrue(result["schema_compatible"])
        self.assertEqual(result["whole_item_text_matches"], 1)
        self.assertEqual(result["correction_candidate_items"], 1)
        self.assertEqual(result["trim_candidate_items"], 1)
        self.assertEqual(result["first_raw_rule_action"], "trim_silence")
        del snapshot["items"][0]["boundary_status"]
        self.assertFalse(
            diagnostics.history_rule_predicates(snapshot)["schema_compatible"]
        )


if __name__ == "__main__":
    unittest.main()
