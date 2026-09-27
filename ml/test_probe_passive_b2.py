import copy
import fcntl
import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import passive_b2_benchmark as b2
import probe_passive_b2 as runner
from passive_benchmark import evaluate_result
from passive_contract import canonical_messages
from replay_session import session_hash
from test_passive_contract import builder, document


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


class B2RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.patch_root = patch.object(runner, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.release = self.root / "release"
        write(self.release / "manifest.json", {})
        write(self.release / "screening-protocol.json", {})
        source = document()
        bundle = builder().build(source, "b", as_of=0)
        self.cases = []
        for index in range(289):
            identity = f"case-{index:03}"
            case = {
                "id": identity, "input": copy.deepcopy(bundle["input"]),
                "expectedOutput": {"action": "no_action"}, "acceptableOutputs": [],
                "metadata": {
                    "split": "dev", "source_group_id": "dev-source", "language": "en",
                    "episode_id": identity, "episode_step": 0, "release_stratum": "curriculum_v1",
                    "context": bundle["context"],
                },
                "_execution": {"before": source, "bundle": bundle, "feedback_ledger": None, "as_of": 0},
            }
            if index == 1:
                case["input"]["recent_edits"] = [{
                    "action": "correct_text", "origin": "synthetic", "before": "requier", "after": "require",
                }]
            case["metadata"]["input_sha256"] = session_hash(canonical_messages(case["input"]))
            self.cases.append(case)
        self.ids = [case["id"] for case in self.cases]
        self.demos = {}
        for language in ("en", "ko"):
            self.demos[language] = []
            for index, action in enumerate(("no_action", "correct_text")):
                demo = copy.deepcopy(self.cases[0])
                demo["id"] = f"train-{language}-{index}"
                demo["metadata"].update(split="train", language=language, source_group_id="train-source")
                demo["expectedOutput"] = ({"action": action} if index == 0 else {
                    "action": action, "target": [1], "text": "require",
                })
                self.demos[language].append(demo)
        self.directory = self.root / "b2"
        self.protocol_path = self.directory / "protocol.json"
        self.prefix_path = self.directory / "prefixes/manifest.json"
        self.execution_path = self.directory / "execution.json"
        self.registry_path = self.root / "model-registry.json"
        self.registration, self.requests = [], []
        for case in self.cases:
            conditions = {}
            for condition in b2.CONDITIONS:
                view = b2.condition_case(case, condition)
                messages = b2.condition_messages(view, condition, self.demos["en"] if condition == "icl" else ())
                self.requests.append({"case_id": case["id"], "condition": condition,
                                      "messages": messages, "messages_sha256": session_hash(messages)})
                conditions[condition] = {
                    **view["_b2"], "messages_sha256": session_hash(messages),
                    "identical_to_full_request": messages == canonical_messages(case["input"]),
                }
            self.registration.append({"case_id": case["id"], "source_group_id": "dev-source",
                                      "episode_id": case["id"], "language": "en", "conditions": conditions})
        tokenizers, models = [], []
        for name in ("a", "b", "c", "d", "e"):
            path = self.root / "tokenizers" / name
            for filename, value in (("config.json", {"vocab_size": 100}),
                                    ("tokenizer_config.json", {}), ("tokenizer.json", {}),
                                    ("chat_template.jinja", "template")):
                write(path / filename, value)
            files = {file.name: runner.probe.digest(file) for file in path.iterdir()}
            tokenizer = {"id": name, "path": str(path.relative_to(self.root)),
                         "repository": f"native/{name}", "revision": f"revision-{name}", "files_sha256": files}
            tokenizers.append(tokenizer)
            models.append({"id": name, "tokenizer_id": name, "base_repository": tokenizer["repository"],
                           "base_revision": tokenizer["revision"], "tokenizer_path": tokenizer["path"],
                           "tokenizer_files_sha256": files,
                           "template_path": str(Path(tokenizer["path"]) / "chat_template.jinja"),
                           "template_sha256": files["chat_template.jinja"]})
        token_registry = self.root / "tokenizers/registry.json"
        write(token_registry, {"candidates": tokenizers})
        token_binding = {"path": str(token_registry.relative_to(self.root)), "sha256": runner.probe.digest(token_registry)}
        self.registry = {
            "candidates": models, "tokenizers": {"registry_path": token_binding["path"],
                                                "registry_sha256": token_binding["sha256"]},
            "runtime": {"context_tokens": 4096, "input_tokens": 3584,
                        "chat_template_kwargs": {"enable_thinking": False}, "parallel_slots": 1},
        }
        write(self.registry_path, self.registry)
        self.protocol = {
            "version": "passive-b2-preregistration-v1", "b2_predictions_consulted": False,
            "final_test_bodies_read": False, "release_manifest_sha256": runner.probe.digest(self.release / "manifest.json"),
            "registered_requests": 289, "case_ids": self.ids,
            "conditions": dict.fromkeys(b2.CONDITIONS, "registered"), "runtime": runner.POLICY,
            "tokenizer_registry": token_binding, "baseline_candidates": list(runner.probe.BASELINES),
            "icl": {"demo_case_ids": {lang: [case["id"] for case in values] for lang, values in self.demos.items()}},
        }
        self.prefixes = {
            "version": "passive-b2-prefixes-v1", "status": "completed", "case_ids": self.ids,
            "conditions": list(b2.CONDITIONS), "tokenizer_registry": token_binding,
            "candidates": tokenizers, "candidate_ids": [row["id"] for row in tokenizers],
            "files_sha256": {}, "prefix_counts": {}, "full_frozen_parity": {},
            "common_icl_eligibility_file": "common-icl-eligibility.json",
        }
        self.frozen = {identity: [1, index % 97 + 2] for index, identity in enumerate(self.ids)}
        for tokenizer in tokenizers:
            name = tokenizer["id"]
            self.prefixes["prefix_counts"][name] = {}
            self.prefixes["full_frozen_parity"][name] = {"passed": 289, "expected": 289}
            for condition in b2.CONDITIONS:
                rows = []
                for request in self.requests:
                    if request["condition"] != condition:
                        continue
                    identity = request["case_id"]
                    tokens = self.frozen[identity] if condition != "icl" else [1, 2, 3]
                    if name == "e" and condition == "icl" and identity == self.ids[-1]:
                        tokens = [1] * 3585
                    rows.append({"case_id": identity, "input_ids": tokens,
                                 "messages_sha256": request["messages_sha256"]})
                filename = f"{name}/{condition}.jsonl.gz"
                write_rows(self.prefix_path.parent / filename, rows)
                self.prefixes["files_sha256"][filename] = runner.probe.digest(self.prefix_path.parent / filename)
                self.prefixes["prefix_counts"][name][condition] = {row["case_id"]: len(row["input_ids"]) for row in rows}
        counts = {name: {condition: value for condition, value in conditions.items() if condition in ("full", "icl")}
                  for name, conditions in self.prefixes["prefix_counts"].items()}
        eligibility = b2.common_input_eligibility(self.ids, counts)
        eligibility.update(candidate_ids=[row["id"] for row in tokenizers],
                           candidate_basis="All five preregistered tokenizers, fixed before B2 candidate choice.")
        eligibility_path = self.prefix_path.parent / "common-icl-eligibility.json"
        write(eligibility_path, eligibility)
        self.prefixes.update(common_icl_eligibility_sha256=runner.probe.digest(eligibility_path),
                             common_icl_eligible_requests=288, common_icl_excluded_requests=1)
        self.prefixes["files_sha256"][eligibility_path.name] = runner.probe.digest(eligibility_path)
        self.rebind()

    def rebind(self):
        write_rows(self.directory / "case-registry.jsonl", self.registration)
        write_rows(self.directory / "requests.jsonl.gz", self.requests)
        self.protocol.update(case_registry_sha256=runner.probe.digest(self.directory / "case-registry.jsonl"),
                             requests_sha256=runner.probe.digest(self.directory / "requests.jsonl.gz"))
        write(self.protocol_path, self.protocol)
        self.prefixes.update({key: self.protocol[key] for key in (
            "release_manifest_sha256", "case_registry_sha256", "requests_sha256",
        )})
        self.prefixes["protocol_sha256"] = runner.probe.digest(self.protocol_path)
        write(self.prefix_path, self.prefixes)
        write(self.execution_path, {"protocol_sha256": runner.probe.digest(self.protocol_path),
                                    "prefix_manifest_sha256": runner.probe.digest(self.prefix_path),
                                    "registry_sha256": runner.probe.digest(self.registry_path),
                                    "selected_candidate_ids": ["a", "b"], "anchor_candidate_id": "a"})

    def args(self, condition="full", candidate="a", reuse=None):
        return SimpleNamespace(release=self.release, registry=self.registry_path, protocol=self.protocol_path,
                               prefix_manifest=self.prefix_path, execution=self.execution_path, condition=condition,
                               candidate=candidate, reuse_full_run=reuse, timeout=120, unconstrained=False,
                               sampling_config=None, output=self.root / "new-run", stage="screen")

    def prepare(self, args):
        with (patch.object(b2, "load_dev", return_value=self.cases),
              patch.object(b2, "select_demonstrations", return_value=self.demos),
              patch.object(runner.probe, "frozen_prefixes", return_value=self.frozen)):
            return runner.prepare(args)

    def full_run(self):
        prepared = self.prepare(self.args())
        source = self.root / "full-run"
        (source / "implementation").mkdir(parents=True)
        for path in prepared["source_files"]:
            shutil.copy2(path, source / "implementation" / path.name)
        shutil.copy2(self.registry_path, source / "implementation/model-registry.json")
        rows = []
        for index, case in enumerate(prepared["cases"]):
            text = '{"action":"no_action"}'
            row = evaluate_result(case, text)
            row.update(raw_text=text, finish_reason="stop", complete_json_ms=100 + index,
                       response={"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": len(self.frozen[case["id"]])}})
            if index == 2:
                row.update(request_failed=True, error_type="request_failure", error="HTTP503")
            rows.append(row)
        write_rows(source / "results.jsonl", rows)
        write(source / "summary.json", {"registered_requests": 289, "observed_results": 289,
                                         "missing_requests": 0, "input_budget_excluded": 0, "reused_quality_rows": 0})
        write(source / "manifest.json", {
            "version": "passive-b2-runtime-v1", "status": "completed", "stage": "screen",
            "candidate": "a", "condition": "full", "case_ids": self.ids, "completed_requests": 289,
            "reused_results": 0, "registry_sha256": runner.probe.digest(self.registry_path),
            "sampling": runner.probe.sampling_options(self.registry_path),
            "candidate_binding": self.registry["candidates"][0], "tokenizer_binding": self.prefixes["candidates"][0],
            "runtime_binding": self.registry["runtime"], "schema_constrained": True,
            "release_manifest_sha256": self.protocol["release_manifest_sha256"],
            "screening_protocol_sha256": runner.probe.digest(self.release / "screening-protocol.json"),
            "reasoning": "off; pinned native template enable_thinking=false", "context_tokens": 4096,
            "cache_requested": False, "request_timeout_seconds": 120,
            "template_checks": {"requests": 289, "passed": 289}, "benchmark": prepared["binding"],
        })
        return source

    def test_full_roster_and_all_five_tokenizer_icl_mask_are_retained(self):
        full = self.prepare(self.args())
        self.assertEqual([case["id"] for case in full["cases"]], self.ids)
        icl = self.prepare(self.args("icl"))
        self.assertEqual([case["id"] for case in icl["cases"]], self.ids[:-1])
        report = icl["summarize"](icl["cases"], [])
        self.assertEqual(report["registered_requests"], 289)
        self.assertEqual(report["input_budget_excluded_case_ids"], [self.ids[-1]])
        self.assertEqual(report["missing_requests"], 288)
        baseline = self.prepare(self.args(candidate="always_no_action"))
        self.assertEqual(len(baseline["cases"]), 289)
        self.assertEqual(baseline["prefixes"], {})

    def test_conditions_require_execution_anchor_and_explicit_full_reuse(self):
        for args in (self.args("icl", "b"), self.args("history_only", "b", self.root / "run"),
                     self.args("context_only"), self.args("context_only", "always_no_action"),
                     self.args(reuse=self.root / "run"), self.args(candidate="c")):
            with self.subTest(condition=args.condition, candidate=args.candidate), self.assertRaises(ValueError):
                self.prepare(args)
        execution = json.loads(self.execution_path.read_text())
        execution["registry_sha256"] = "changed"
        write(self.execution_path, execution)
        with self.assertRaisesRegex(ValueError, "does not bind registry"):
            self.prepare(self.args())

    def test_matching_hashes_cannot_hide_changed_messages_or_roster(self):
        self.requests[1]["messages"][1]["content"] += " UNREGISTERED PROMPT"
        self.requests[1]["messages_sha256"] = session_hash(self.requests[1]["messages"])
        self.registration[0]["conditions"]["context_only"]["messages_sha256"] = self.requests[1]["messages_sha256"]
        self.rebind()
        with self.assertRaisesRegex(ValueError, "messages or evidence masks"):
            self.prepare(self.args())
        self.protocol["case_ids"] = self.ids[:-1]
        self.prefixes["case_ids"] = self.ids[:-1]
        self.rebind()
        with self.assertRaisesRegex(ValueError, "full original 289"):
            self.prepare(self.args())

    def test_changed_full_token_ids_fail_even_with_rebound_file_hashes(self):
        filename = "a/full.jsonl.gz"
        path = self.prefix_path.parent / filename
        rows = runner.read_rows(path)
        rows[0]["input_ids"] = [1, 99]
        write_rows(path, rows)
        self.prefixes["files_sha256"][filename] = runner.probe.digest(path)
        self.rebind()
        with self.assertRaisesRegex(ValueError, "prefix tokens"):
            self.prepare(self.args())

    def test_identical_requests_reuse_failed_rows_regrade_quality_and_clear_latency(self):
        source = self.full_run()
        rows = runner.read_rows(source / "results.jsonl")
        rows[0]["raw_grade"]["outcome_match"] = False
        rows[0]["displayed"] = True
        write_rows(source / "results.jsonl", rows)
        prepared = self.prepare(self.args("context_only", reuse=source))
        reused = prepared["reused_results"]
        self.assertEqual(set(reused), set(self.ids) - {self.ids[1]})
        self.assertTrue(reused[self.ids[0]]["raw_grade"]["outcome_match"])
        self.assertFalse(reused[self.ids[0]]["displayed"])
        self.assertIsNone(reused[self.ids[0]]["complete_json_ms"])
        self.assertEqual(reused[self.ids[0]]["inherited_complete_json_ms"], 100)
        failed = reused[self.ids[2]]
        self.assertTrue(failed["request_failed"])
        self.assertEqual(failed["error_type"], "request_failure")
        self.assertFalse(failed["raw_valid"])
        self.assertIsNone(failed["raw_grade"])
        self.assertEqual(failed["reused_from"]["results_sha256"], runner.probe.digest(source / "results.jsonl"))
        report = prepared["summarize"](prepared["cases"], list(reused.values()))
        self.assertEqual(report["latency_samples"], 0)
        self.assertEqual(report["expected_fresh_requests"], 1)

    def test_reuse_rejects_changed_sampling_template_and_nonfull_rosters(self):
        source = self.full_run()
        path = source / "manifest.json"
        original = json.loads(path.read_text())
        for changes in ({"sampling": {"temperature": 1}}, {"tokenizer_binding": {}},
                        {"case_ids": self.ids[:36]}, {"status": "failed"}):
            write(path, {**original, **changes})
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "matching full289"):
                self.prepare(self.args("context_only", reuse=source))

    def test_cli_uses_the_same_predictor_lock_before_preparation_or_execution(self):
        lock_path = self.root / "artifacts/nep-passive-b1.lock"
        lock_path.parent.mkdir()
        with (lock_path.open("a") as lock, patch.object(runner, "prepare") as prepare,
              patch.object(runner.probe, "run") as run, patch("sys.stderr")):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(SystemExit):
                runner.main(["--execution", str(self.execution_path), "--candidate", "a",
                             "--condition", "full", "--output", str(self.root / "output")])
            prepare.assert_not_called()
            run.assert_not_called()

    def test_registry_drift_during_preparation_cannot_change_execution_binding(self):
        load_prefixes = runner.load_prefixes

        def change_registry(*args, **kwargs):
            result = load_prefixes(*args, **kwargs)
            write(self.registry_path, {**self.registry, "note": "Changed during preparation"})
            return result

        with patch.object(runner, "load_prefixes", side_effect=change_registry):
            with self.assertRaisesRegex(ValueError, "inputs changed during preparation"):
                self.prepare(self.args())


if __name__ == "__main__":
    unittest.main()
