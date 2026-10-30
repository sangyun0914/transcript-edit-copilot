import copy
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import recompute_actions_v3_results as evidence


class EvidenceTests(unittest.TestCase):
    def relocated_reads(self, root):
        stack = ExitStack()

        def confined(function):
            def check(path):
                resolved = Path(path).resolve()
                self.assertTrue(
                    resolved.is_relative_to(root)
                    or resolved == Path(evidence.__file__).resolve(),
                    f"Read outside relocated evidence: {path}",
                )
                return function(path)

            return check

        for name in ("read", "rows", "sha256"):
            stack.enter_context(
                patch.object(
                    evidence, name, side_effect=confined(getattr(evidence, name))
                )
            )
        return stack

    def development_fixture(self, root):
        def save(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) + "\n")
            return {"path": str(path), "sha256": evidence.sha256(path)}

        def manifest(directory, bindings):
            return save(
                directory + "/manifest.json",
                {
                    "status": "inference_complete",
                    "files_sha256": {
                        Path(value["path"]).name: value["sha256"] for value in bindings
                    },
                },
            )

        identity = {"kind": "synthetic-engineering-fixture"}
        request = {
            "case_id": "fixture",
            "input_sha256": "input",
            "messages_sha256": "messages",
            "prefix_ids_sha256": "prefix",
            "format": "json",
        }
        requests = save("prepared/requests-json.jsonl", request)
        prepared = manifest(
            "prepared", [save("prepared/cases.jsonl", {"id": "fixture"}), requests]
        )
        observations = save(
            "run/observations.jsonl", {**request, "model_identity": identity}
        )
        info = save(
            "run/run.json",
            {
                "status": "inference_complete",
                "requested": 1,
                "observed": 1,
                "model_identity": identity,
                "requests_sha256": requests["sha256"],
            },
        )
        inference = manifest("run", [info, observations])
        record = {
            "case_id": "fixture",
            "judgment": "preserve",
            "displayed": False,
            "useful": False,
            "preserved": True,
            "model_abstained": True,
            "valid": True,
            "outcome_match": True,
            "status": "no_action",
        }
        report = save(
            "grade/report.json",
            {
                "final_test": False,
                "preparation": prepared,
                "observations": observations,
                "inference_manifest": inference,
                "inference_run": info,
                "graded": save("grade/graded.jsonl", record),
                "overall": {"requests": 1, **evidence.metric_counts([record])},
                "model_identity": identity,
                "numerical_regime": "synthetic-fixture",
                "condition": "fixture",
                "format": "json",
                "excluded_before_inference": [],
            },
        )
        save(
            "registry.json",
            {
                "recorded_workspace": str(root),
                "entries": [
                    {
                        "id": "fixture",
                        "state": "recorded",
                        "kind": "development",
                        "evidence": report,
                    }
                ],
            },
        )

    def ledger(self):
        return {
            "optimizer_commit_uncertain": False,
            "completed_updates": 2,
            "phases": {
                "optimizer_committed": {
                    "rows": 3,
                    "unique_cases": 2,
                    "sequence_tokens": 30,
                    "completion_tokens": 3,
                    "counts": {
                        "case_id": {"a": 2, "b": 1},
                        "action": {"no_action": 3},
                        "language": {"en": 3},
                        "source_group_id": {"one": 3},
                    },
                }
            },
        }, [
            {"update": 1, "committed_rows": 2, "case_ids": ["a", "b"]},
            {"update": 2, "committed_rows": 3, "case_ids": ["a"]},
        ]

    def test_unique_states_are_not_repeated_draws(self):
        ledger, events = self.ledger()
        result = evidence.committed_counts(ledger, events, False)
        self.assertEqual(
            (result["committed_draws_or_pairs"], result["unique_states"]), (3, 2)
        )

    def test_uncertain_or_missing_commit_cannot_be_counted(self):
        ledger, events = self.ledger()
        changed = copy.deepcopy(ledger)
        changed["optimizer_commit_uncertain"] = True
        with self.assertRaises(ValueError):
            evidence.committed_counts(changed, events, False)
        with self.assertRaises(ValueError):
            evidence.committed_counts(ledger, events[:1], False)

    def test_no_proposals_do_not_become_perfect_precision(self):
        records = [
            {
                "judgment": "edit_required",
                "displayed": False,
                "useful": False,
                "preserved": False,
                "model_abstained": True,
                "valid": True,
                "outcome_match": False,
            }
        ]
        metrics = evidence.metric_counts(records)
        self.assertIsNone(metrics["precision"]["value"])
        self.assertEqual(metrics["recall"]["value"], 0)
        self.assertEqual(metrics["outcome_accuracy"]["denominator"], 1)

    def test_relocated_development_follows_nested_absolute_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            original, moved = base / "original", base / "moved"
            self.development_fixture(original)
            shutil.copytree(original, moved)
            with self.relocated_reads(moved):
                result = evidence.recompute(
                    moved / "registry.json", moved, moved / "summary.json"
                )
            self.assertEqual(result["verified_entries"], 1)
            for path in original.rglob("*"):
                if path.is_file():
                    self.assertEqual(
                        path.read_bytes(),
                        (moved / path.relative_to(original)).read_bytes(),
                    )

    def test_damaged_relocated_evidence_cannot_fall_back_to_valid_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            original = base / "original"
            self.development_fixture(original)
            for corruption in ("missing", "changed"):
                with self.subTest(corruption=corruption):
                    moved = base / corruption
                    shutil.copytree(original, moved)
                    target = moved / "run/observations.jsonl"
                    if corruption == "missing":
                        target.unlink()
                        failure = FileNotFoundError
                    else:
                        target.write_text("changed\n")
                        failure = ValueError
                    self.assertTrue((original / "run/observations.jsonl").is_file())
                    with self.relocated_reads(moved), self.assertRaises(failure):
                        evidence.recompute(
                            moved / "registry.json", moved, moved / "summary.json"
                        )
                    self.assertFalse((moved / "summary.json").exists())

    def test_external_traversal_and_symlink_evidence_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            original, moved = base / "original", base / "moved"
            original.mkdir()
            moved.mkdir()
            external = base / "external.json"
            external.write_text("{}")
            (moved / "link.json").symlink_to(external)
            for path in (
                str(external),
                "../external.json",
                str(original / "../external.json"),
                "link.json",
            ):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    evidence.bound(
                        moved,
                        {"path": path, "sha256": evidence.sha256(external)},
                        original,
                    )
            manifest = moved / "manifest.json"
            for name in (str(external), "../external.json", "link.json"):
                manifest.write_text(
                    json.dumps({"files_sha256": {name: evidence.sha256(external)}})
                )
                with self.subTest(manifest_name=name), self.assertRaises(ValueError):
                    evidence.verify_manifest(manifest)

    def test_missing_or_reordered_failed_observation_is_rejected(self):
        info = {"requested": 2, "observed": 2, "model_identity": {"kind": "fixture"}}
        requests = [
            {
                "case_id": identity,
                "input_sha256": identity,
                "messages_sha256": identity,
                "prefix_ids_sha256": identity,
                "format": "json",
            }
            for identity in ("a", "b")
        ]
        observations = [
            {
                **row,
                "model_identity": info["model_identity"],
                "status": "request_failure",
            }
            for row in requests
        ]
        evidence.verify_evaluation_rows(requests, observations, requests, info)
        for damaged in (observations[:1], observations[::-1]):
            with self.assertRaises(ValueError):
                evidence.verify_evaluation_rows(requests, damaged, requests, info)

    def test_final_census_keeps_unknowns_exclusions_and_correlated_windows_distinct(
        self,
    ):
        dispositions = [
            {
                "id": identity,
                "source_group": "resolved-source",
                "registered_source_group": "original-source",
                "stratum": "natural",
                "language": "ko",
                "window_id": "same-window",
                "status": status,
            }
            for identity, status in (
                ("a", "admitted"),
                ("b", "admitted"),
                ("c", "unknown"),
                ("d", "source_unavailable"),
            )
        ]
        census = evidence.final_census(dispositions, ["a"], ["b"])
        self.assertEqual(
            census["source_group"]["resolved-source"],
            {
                "registered": 4,
                "admitted": 2,
                "requested": 1,
                "natural_windows": 1,
                "dispositions": {"admitted": 2, "unknown": 1, "source_unavailable": 1},
            },
        )
        self.assertIn("original-source", census["registered_source_group"])
        for requested, excluded in (
            (["a", "c"], ["b"]),
            (["a"], []),
            (["a"], ["a", "b"]),
        ):
            with self.assertRaises(ValueError):
                evidence.final_census(dispositions, requested, excluded)

    def test_final_count_reader_accepts_engineering_grade_but_rejects_missing_failure(
        self,
    ):
        from test_prepare_task_test_actions_v3 import SealedTaskTests, sealed

        fixture = SealedTaskTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        prepared = fixture.seal(fixture.prepare())
        observed, output = fixture.root / "observed", fixture.root / "graded"

        def predictor(request):
            if request["case_id"] == "engineering-edit":
                raise RuntimeError("Registered engineering request failure")
            return {"text": '{"action":"no_action"}', "finish_reason": "stop"}

        sealed.record_run(
            fixture.freeze_path,
            prepared,
            "sft",
            observed,
            predictor,
            tokenizer=fixture.tokenizer,
            predictor_identity={"model": "sft", "engineering_fixture_only": True},
        )
        sealed.grade(fixture.freeze_path, prepared, observed, output)
        result = evidence.final_evaluation_summary(
            sealed.ROOT, output / "manifest.json"
        )
        self.assertTrue(result["engineering_fixture_only"])
        self.assertEqual(
            (
                result["registered_slots"],
                result["canonical_admitted_cases"],
                result["requests"],
            ),
            (4, 2, 2),
        )
        self.assertEqual(result["metrics"]["recall"]["value"], 0)
        self.assertEqual(result["metrics"]["preservation"]["value"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            moved = Path(temporary).resolve()
            relocated = moved / fixture.root.relative_to(sealed.ROOT)
            shutil.copytree(fixture.root, relocated)
            relocated_manifest = relocated / "graded/manifest.json"
            with self.relocated_reads(moved):
                recounted = evidence.final_evaluation_summary(
                    moved, relocated_manifest, recorded_workspace=sealed.ROOT
                )
            self.assertEqual(recounted, result)
            (relocated / "freeze.json").unlink()
            self.assertTrue(fixture.freeze_path.is_file())
            with self.relocated_reads(moved), self.assertRaises(FileNotFoundError):
                evidence.final_evaluation_summary(
                    moved, relocated_manifest, recorded_workspace=sealed.ROOT
                )
        graded = output / "graded.jsonl"
        graded.write_text(graded.read_text().splitlines()[1] + "\n")
        manifest = evidence.read(output / "manifest.json")
        manifest["files_sha256"]["graded.jsonl"] = evidence.sha256(graded)
        (output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "ordered observation and grade"):
            evidence.final_evaluation_summary(sealed.ROOT, output / "manifest.json")

    def test_native_startup_failures_stay_counted_but_resigned_execution_gaps_are_rejected(
        self,
    ):
        import infer_task_test_actions_v3 as native
        from test_prepare_task_test_actions_v3 import SealedTaskTests, sealed

        fixture = SealedTaskTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        release = fixture.seal(fixture.prepare())
        portable, remote = fixture.root / "portable", fixture.root / "remote"
        authorization = sealed.export_execution(
            fixture.freeze_path, release, "native_base", portable
        )
        native.run(
            portable, authorization["sha256"], fixture.inference_config["path"], remote
        )
        admitted, graded = fixture.root / "admitted", fixture.root / "graded"
        sealed.admit_observations(
            fixture.freeze_path,
            release,
            portable,
            remote,
            admitted,
            tokenizer=fixture.tokenizer,
        )
        sealed.grade(fixture.freeze_path, release, admitted, graded)
        summary = evidence.final_evaluation_summary(
            sealed.ROOT, graded / "manifest.json"
        )
        self.assertEqual(summary["requests"], 2)
        self.assertFalse(
            summary["execution_evidence"]["native_model_execution_verified"]
        )
        self.assertEqual(
            summary["execution_evidence"]["failure_stages"],
            {"preflight_or_model_loading": 2},
        )
        self.assertEqual(summary["metrics"]["outcome_accuracy"]["numerator"], 0)
        backups = {
            path: path.read_bytes()
            for directory in (remote, admitted, graded)
            for path in directory.iterdir()
            if path.is_file()
        }

        def save(path, value):
            path.write_text(json.dumps(value))

        def refresh(directory):
            path = directory / "manifest.json"
            value = evidence.read(path)
            value["files_sha256"] = {
                name: evidence.sha256(directory / name)
                for name in value["files_sha256"]
            }
            save(path, value)

        for fault in (
            "missing_kind",
            "missing_remote_binding",
            "changed_original_observation",
            "asserted_identity_without_runtime",
            "completed_before_verified_loading",
        ):
            with self.subTest(fault=fault):
                for path, value in backups.items():
                    path.write_bytes(value)
                execution = evidence.read(admitted / "manifest.json")
                if fault == "missing_kind":
                    execution.pop("execution_kind")
                elif fault == "missing_remote_binding":
                    execution.pop("remote_manifest")
                elif fault == "changed_original_observation":
                    observations = list(evidence.rows(remote / "observations.jsonl"))
                    observations[0]["error"] = "Different original native outcome"
                    (remote / "observations.jsonl").write_text(
                        "".join(sealed.packed(row) + "\n" for row in observations)
                    )
                elif fault == "asserted_identity_without_runtime":
                    info = evidence.read(remote / "run.json")
                    info.update(
                        model_identity_verified=True,
                        actual_model_identity=info["model_identity"],
                    )
                    save(remote / "run.json", info)
                    save(admitted / "remote-run.json", info)
                else:
                    observations = list(evidence.rows(remote / "observations.jsonl"))
                    observations[0].update(
                        status="completed",
                        text='{"action":"no_action"}',
                        text_raw='{"action":"no_action"}',
                        output_ids=[1],
                        terminal_token_id=1,
                        finish_reason="stop",
                        model_identity_verified=True,
                        generation_attempted=True,
                        generation_ms=1.0,
                        timing_source="cuda_synchronized_generate",
                    )
                    for directory in (remote, admitted):
                        (directory / "observations.jsonl").write_text(
                            "".join(sealed.packed(row) + "\n" for row in observations)
                        )
                refresh(remote)
                if "remote_manifest" in execution:
                    execution["remote_manifest"] = sealed.binding(
                        remote / "manifest.json"
                    )
                save(admitted / "manifest.json", execution)
                refresh(admitted)
                grade_manifest = evidence.read(graded / "manifest.json")
                grade_manifest["observation_manifest"] = sealed.binding(
                    admitted / "manifest.json"
                )
                save(graded / "manifest.json", grade_manifest)
                with self.assertRaises(ValueError):
                    sealed.grade(
                        fixture.freeze_path,
                        release,
                        admitted,
                        fixture.root / ("rejected-" + fault),
                    )
                with self.assertRaises(ValueError):
                    evidence.final_evaluation_summary(
                        sealed.ROOT, graded / "manifest.json"
                    )


if __name__ == "__main__":
    unittest.main()
