"""Engineering fixtures only: no reserved source body, media, or label is opened."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import passive_actions as contract
import prepare_actions_v3_conditions as conditions
import prepare_gemma_actions_v3 as training_preparation
import prepare_task_test_actions_v3 as sealed
from test_passive_actions import document


class FixtureTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return [2, *bytes.fromhex(sealed.digest(messages))]

    def encode(self, value, **kwargs):
        return list(value.encode())


class OverflowICLTokenizer(FixtureTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) > 2:
            return [2] * 4000
        return super().apply_chat_template(messages, **kwargs)


class ForbiddenBodyPath:
    def __fspath__(self):
        raise AssertionError("Body path was consulted before the gate passed.")


class SealedTaskTests(unittest.TestCase):
    def setUp(self):
        (sealed.ROOT / "artifacts").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="sealed-v3-engineering-", dir=sealed.ROOT / "artifacts"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tokenizer = FixtureTokenizer()
        train_cases = []
        builder = contract.PassiveContextBuilder(self.tokenizer)
        for language, action in (
            ("en", {"action": "correct_text", "target": [1], "text": "we"}),
            ("ko", {"action": "no_action"}),
        ):
            doc = document(language)
            next(item for item in doc["content"] if item["uuid"] == "a")["text"] = (
                "Training example" if language == "en" else "학습 예시"
            )
            snapshot = builder.build(doc, "b", as_of=0)["input"]
            train_cases.append(
                {
                    "id": "engineering-demo-" + language,
                    "split": "train",
                    "language": language,
                    "source": {
                        "split": "train",
                        "source_group": "engineering-train-" + language,
                    },
                    "input": snapshot,
                    "primary": action,
                    "accepted_predictions": [action],
                }
            )
        training_rows = self.root / "training/cases/train.jsonl"
        training_rows.parent.mkdir(parents=True)
        sealed.write_rows(training_rows, train_cases)
        self.training_manifest = self.save(
            "training/manifest.json",
            {
                "counts": {"train": 2, "dev": 0},
                "files_sha256": {"cases/train.jsonl": sealed.sha256(training_rows)},
            },
        )
        self.evidence = self.save(
            "engineering-evidence.json",
            {"status": "verified", "engineering_fixture_only": True},
        )
        groups = [
            {
                "source_group": name,
                "language": "en" if "en" in name else "ko",
                "role": "primary",
            }
            for name in sorted(sealed.dataset.RESERVED)
        ]
        groups.extend(
            {
                "source_group": f"engineering-heldout-{language}",
                "language": language,
                "role": "primary",
            }
            for language in ("en", "ko")
        )
        self.reservations = self.save(
            "reservations.json",
            {
                "groups": groups,
                "exposed_source_group_splits": {
                    "engineering-train-en": "train",
                    "engineering-train-ko": "train",
                },
                "exposed_inventory_sha256": {},
            },
        )
        self.roster = [
            {
                "slot_id": identity,
                "source_group": f"engineering-heldout-{language}",
                "language": language,
                "stratum": "authored_challenge",
                "admission": "not_materialized",
                "scene_plan": "Engineering scene " + language,
            }
            for identity, language in (
                ("engineering-edit", "en"),
                ("engineering-preserve", "ko"),
                ("engineering-unknown", "en"),
                ("engineering-unavailable", "ko"),
            )
        ]
        roster_path = self.root / "roster.jsonl"
        sealed.write_rows(roster_path, self.roster)
        windows = self.save("windows.json", [])
        metadata = self.save(
            "metadata.json",
            {
                "reserved_final_test_bodies_read": False,
                "generated_test_cases": 0,
                "selected_dataset_manifest_sha256": self.training_manifest["sha256"],
                "files_sha256": {
                    p.name: sealed.sha256(p)
                    for p in (
                        Path(self.reservations["path"]),
                        roster_path,
                        Path(windows["path"]),
                    )
                },
            },
        )
        tokenizer_root = self.root / "tokenizer"
        tokenizer_root.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (tokenizer_root / name).write_text(
                "engineering tokenizer pin; no native accuracy claim"
            )
        runtime = {
            "prompt_sha256": sealed.sha256(contract.PROMPT_PATH),
            "contract_sha256": sealed.sha256(contract.__file__),
            "formats_sha256": sealed.sha256(sealed.formats.__file__),
            "tokenizer": {
                "path": str(tokenizer_root),
                "eos_token_id": 1,
                "files_sha256": {
                    p.name: sealed.sha256(p) for p in tokenizer_root.iterdir()
                },
            },
            "generation": {**training_preparation.GENERATION, "enable_thinking": False},
            "cache_policy": "engineering fixture",
            "display_policy": "existing proposal ledger",
            "environment_receipt": self.evidence,
        }
        asr = {
            "provider": "elevenlabs",
            "model_id": "scribe_v2",
            "no_verbatim": False,
            "timestamps_granularity": "word",
            "diarize": True,
            "num_speakers": None,
            "tag_audio_events": True,
            "language_codes": ["en", "ko"],
            "pcm": {"sample_rate": 16000, "channels": 1, "sample_width": 2},
            "retry_policy": "no_automatic_billable_retry",
            "hosted_alias_limitation": "Hosted alias cannot pin remote weights.",
        }
        importer = {
            "normalizer_sha256": sealed.sha256(
                sealed.ROOT / "server/app/asr/scribe.py"
            ),
            "document_importer_sha256": sealed.sha256(
                sealed.ROOT / "app/src/core/transcription.ts"
            ),
            "paragraph_policy": "existing_importTranscript_speaker_groups",
            "source_clock": "excerpt_relative_with_original_offset_provenance",
            "cursor_policy": "nearest_timed_item_to_window_midpoint_ties_earlier",
            "window_policy": "90 seconds chosen from metadata",
            "subtitle_policy": "judge_evidence_only",
            "text_or_timing_repair": False,
            "checkpoint_policy": "actual edit then shown/dismissed",
            "fallback_policy": "no body-based substitution",
        }
        metrics = {
            "evaluator_sha256": sealed.sha256(sealed.evaluation.__file__),
            "all_request_failures_in_denominator": True,
            "source_stratum_denominators": True,
            "unknown_is_no_action": False,
            "cluster_unit": "source_group",
            "decision_thresholds": "descriptive engineering fixture",
            "selection_after_test": False,
        }
        exposed = self.save(
            "exposed-text-hashes.json",
            {
                "sha256": [
                    sealed.digest(sealed.sealed_v1.corpus_text(case["input"]))
                    for case in train_cases
                ]
            },
        )
        admission = {
            key: True
            for key in (
                "input_first",
                "all_seven_actions_reviewed",
                "independent_review",
                "audio_evidence_separate",
                "source_integrity_required",
                "replay_undo_export_required",
            )
        }
        admission.update(
            unknown_is_no_action=False,
            duplicate_policy="no exposed wording or duplicate complete inputs",
            actors={
                "preparer": "fixture-preparer",
                "author": "fixture-author",
                "reviewer": "fixture-reviewer",
                "grader": "fixture-grader",
            },
            generator_recipe=self.evidence,
            label_rubric=self.evidence,
            exclusions=self.evidence,
            exposed_text_hashes=exposed,
        )
        policies = {
            name: self.save(
                f"{name}-policy.json",
                {"scope": sealed.SCOPE, "status": "frozen", **value},
            )
            for name, value in zip(
                sealed.POLICIES,
                (runtime, asr, importer, metrics, admission),
                strict=True,
            )
        }
        models = {}
        self.inference_config = self.save(
            "prepared/engineering-inference-config.json",
            {
                "format": "json",
                "generation": training_preparation.GENERATION,
                "libraries": {"engineering": "fixture"},
            },
        )
        for role in sorted(sealed.MODEL_ROLES):
            value = {
                "role": role,
                "status": "verified",
                "identity": {"model": role, "engineering_fixture_only": True},
                "artifact_manifest": self.evidence,
                "reload_receipt": self.evidence,
                "inference_configuration": self.inference_config,
                "preparation_manifest": self.evidence,
            }
            if role == "sft_dpo":
                value["parent_identity_sha256"] = sealed.digest(
                    {"model": "sft", "engineering_fixture_only": True}
                )
            models[role] = self.save(f"model-{role}.json", value)
        demos = []
        for case in train_cases:
            demos.append(
                conditions.demonstration(
                    case, self.tokenizer, {"manifest": self.training_manifest}
                )
            )
        specs = conditions.condition_specs(demos)
        runs = [
            {
                "id": role,
                "role": role,
                "format": "json",
                "condition": specs[
                    "train_icl" if role == "base_icl" else "canonical_full_context"
                ],
            }
            for role in sorted(sealed.RUN_ROLES)
        ]
        selection = self.save(
            "selection.json",
            {
                "status": "frozen",
                "test_bodies_read": False,
                "selection_scope": "development",
                "selected_model": "sft",
            },
        )
        self.freeze = {
            "version": sealed.FREEZE_VERSION,
            "scope": sealed.SCOPE,
            "status": "frozen",
            "engineering_fixture_only": True,
            "body_access_authorized": True,
            "reserved_bodies_read_before_freeze": False,
            "frozen_at": "2026-01-01T00:00:00+00:00",
            "source_sha256": sealed.code_sources(),
            "metadata_manifest": metadata,
            "source_reservations": self.reservations,
            "roster": sealed.binding(roster_path),
            "natural_windows": windows,
            "training_dataset_manifest": self.training_manifest,
            "policies": policies,
            "selection_receipt": selection,
            "selected_model": "sft",
            "models": models,
            "runs": runs,
        }
        self.freeze_path = self.root / "freeze.json"
        sealed.write(self.freeze_path, self.freeze)

    def save(self, name, value):
        path = self.root / name
        sealed.write(path, value)
        return sealed.binding(path)

    def replace_freeze(self):
        self.freeze_path.write_text(json.dumps(self.freeze))

    def candidates(self):
        result, structures = [], {}
        for slot in self.roster:
            if slot["slot_id"] == "engineering-unavailable":
                result.append(
                    {
                        "id": slot["slot_id"],
                        "status": "source_unavailable",
                        "reason": "Engineering technical failure fixture.",
                    }
                )
                continue
            doc = document(slot["language"])
            if slot["slot_id"] == "engineering-unknown":
                next(item for item in doc["content"] if item["uuid"] == "b")["text"] = (
                    "Unresolved fixture term"
                )
            timed = sealed.sealed_v1.timed(doc)
            midpoint = (
                min(item["sourceStart"] for item in timed)
                + max(item["sourceStart"] + item["length"] for item in timed)
            ) / 2
            focus = min(
                enumerate(timed),
                key=lambda pair: (
                    abs(pair[1]["sourceStart"] + pair[1]["length"] / 2 - midpoint),
                    pair[0],
                ),
            )[1]["uuid"]
            if slot["language"] not in structures:
                structures[slot["language"]] = self.save(
                    "structure-" + slot["language"] + ".json",
                    {
                        "freeze_sha256": sealed.sha256(self.freeze_path),
                        "source_group": slot["source_group"],
                        "scene_plan": slot["scene_plan"],
                        "created_at": sealed.now(),
                        "wording_assigned": False,
                        "timed_slots": [
                            {
                                key: item[key]
                                for key in ("uuid", "source", "sourceStart", "length")
                            }
                            for item in timed
                        ],
                        "anchor": {
                            "kind": "midpoint",
                            "source": "fixture",
                            "seconds": midpoint,
                        },
                    },
                )
            result.append(
                {
                    "id": slot["slot_id"],
                    "status": "candidate",
                    "split": sealed.SCOPE,
                    "language": slot["language"],
                    "source": {
                        "source_group": slot["source_group"],
                        "split": sealed.SCOPE,
                    },
                    "document": doc,
                    "runtime": {
                        "focus_uuid": focus,
                        "sequence": 0,
                        "selection": [],
                        "history": [],
                        "feedback_journal": None,
                        "caption_profile": contract.CaptionProfile().model_dump(),
                    },
                    "lineage": {
                        "stratum": "authored_challenge",
                        "scenario_family": "engineering_only",
                        "selection_method": "fixture",
                        "preword_structure": structures[slot["language"]],
                    },
                    "provenance": {
                        "selection_before_words": True,
                        "model_predictions_seen": False,
                        "created_at": sealed.now(),
                        "freeze_sha256": sealed.sha256(self.freeze_path),
                    },
                }
            )
        path = self.root / "engineering-candidates.jsonl"
        sealed.write_rows(path, result)
        return path

    def prepare(self, tokenizer=None):
        destination = self.root / "inputs"
        self.tokenizer = tokenizer or self.tokenizer
        sealed.freeze_inputs(
            self.freeze_path, self.candidates(), destination, tokenizer=self.tokenizer
        )
        return destination

    def prelabel_audit(self, case):
        destination = self.root / "prelabel" / case["id"]
        if not destination.exists():
            gate = sealed.load_gate(self.freeze_path)
            sealed.publish(
                destination,
                {
                    "input-record.json": case,
                    "requests.json": sealed.make_requests(case, gate, self.tokenizer),
                },
                {
                    "status": "prelabel_input_frozen",
                    "freeze": gate["binding"],
                    "predictions_seen": False,
                },
            )
        return {
            "prelabel_input_record": sealed.binding(destination / "manifest.json"),
            "author_decided_at": sealed.now(),
            "reviewed_at": sealed.now(),
        }

    def references(self, inputs):
        cases = {row["id"]: row for row in sealed.rows(inputs / "inputs.jsonl")}
        result = []
        for entry in sealed.rows(inputs / "census.jsonl"):
            identity = entry["id"]
            if identity not in cases:
                result.append({"id": identity, "status": entry["status"]})
                continue
            case = cases[identity]
            prelabel = self.prelabel_audit(case)
            status = {
                "engineering-edit": "edit_required",
                "engineering-preserve": "preserve",
                "engineering-unknown": "unknown",
            }[identity]
            action = (
                {"action": "correct_text", "target": [1], "text": "we"}
                if status == "edit_required"
                else {"action": "no_action"}
            )
            result.append(
                {
                    "id": identity,
                    "status": status,
                    "input_record_sha256": case["input_record_sha256"],
                    "primary": action if status in sealed.KNOWN else None,
                    "accepted_predictions": [action] if status in sealed.KNOWN else [],
                    "reason": "Engineering unknown is deliberately excluded.",
                    "acoustic_evidence": "none_engineering_fixture",
                    "audit": {
                        "author": "fixture-author",
                        "reviewer": "fixture-reviewer",
                        "input_first": True,
                        **prelabel,
                        "predictions_seen": False,
                        "reviewed_actions": list(contract.ACTIONS),
                        "rationale": "Technical contract fixture; this is not a final-test semantic label.",
                        "input_record_sha256": case["input_record_sha256"],
                    },
                }
            )
        return result

    def seal(self, inputs, references=None, tokenizer=None):
        path = self.root / "engineering-references.jsonl"
        sealed.write_rows(
            path, references if references is not None else self.references(inputs)
        )
        destination = self.root / "sealed"
        sealed.seal_references(
            self.freeze_path,
            inputs,
            path,
            destination,
            tokenizer=tokenizer or self.tokenizer,
        )
        return destination

    def test_missing_freeze_and_policy_fail_before_body_path_is_consulted(self):
        self.freeze["body_access_authorized"] = False
        self.replace_freeze()
        with self.assertRaisesRegex(ValueError, "body access"):
            sealed.freeze_inputs(
                self.freeze_path, ForbiddenBodyPath(), self.root / "never"
            )
        self.freeze["body_access_authorized"] = True
        self.freeze["policies"].pop("asr")
        self.replace_freeze()
        with self.assertRaises(KeyError):
            sealed.freeze_inputs(
                self.freeze_path, ForbiddenBodyPath(), self.root / "never"
            )

    def test_model_lineage_source_closure_and_roster_tamper_fail_closed(self):
        gate = sealed.load_gate(self.freeze_path)
        self.assertEqual(len(gate["roster"]), 4)
        self.freeze["source_sha256"]["ml/prepare_task_test_actions_v3.py"] = "0" * 64
        self.replace_freeze()
        with self.assertRaisesRegex(ValueError, "code changed"):
            sealed.load_gate(self.freeze_path)
        self.freeze["source_sha256"] = sealed.code_sources()
        self.replace_freeze()
        Path(self.freeze["roster"]["path"]).write_text("[]")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            sealed.load_gate(self.freeze_path)

    def test_unknown_and_mixed_necessity_never_become_no_action(self):
        inputs = self.prepare()
        refs = self.references(inputs)
        refs[2]["accepted_predictions"] = [{"action": "no_action"}]
        with self.assertRaisesRegex(ValueError, "Unknown cannot"):
            self.seal(inputs, refs)
        (self.root / "engineering-references.jsonl").unlink()
        refs = self.references(inputs)
        refs[0]["accepted_predictions"].append({"action": "no_action"})
        with self.assertRaisesRegex(ValueError, "mixed necessity"):
            self.seal(inputs, refs)

    def test_distinct_reviewer_and_exact_input_binding_required(self):
        inputs = self.prepare()
        refs = self.references(inputs)
        refs[0]["audit"]["reviewer"] = "fixture-author"
        with self.assertRaisesRegex(ValueError, "whole-window review"):
            self.seal(inputs, refs)
        (self.root / "engineering-references.jsonl").unlink()
        refs[0]["audit"]["reviewer"] = "fixture-reviewer"
        refs[0]["input_record_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "input binding"):
            self.seal(inputs, refs)

    def test_actual_replay_export_request_failures_and_source_denominators(self):
        inputs = self.prepare()
        release = self.seal(inputs)
        proofs = sealed.rows(release / "reference-executions.jsonl")
        self.assertEqual(proofs[0]["references"][0]["replay"]["transactions"], 3)
        self.assertIn("caption_export", proofs[0]["references"][0])
        called = []

        def fails(request):
            called.append(request)
            self.assertNotIn("primary", request)
            raise RuntimeError("Engineering model failure")

        observed = self.root / "observed"
        identity = {"model": "native_base", "engineering_fixture_only": True}
        with self.assertRaisesRegex(ValueError, "actual frozen identity"):
            sealed.record_run(
                self.freeze_path,
                release,
                "native_base",
                observed,
                fails,
                tokenizer=self.tokenizer,
            )
        sealed.record_run(
            self.freeze_path,
            release,
            "native_base",
            observed,
            fails,
            tokenizer=self.tokenizer,
            predictor_identity=identity,
        )
        report = self.root / "graded"
        sealed.grade(self.freeze_path, release, observed, report)
        value = sealed.read_json(report / "report.json")
        self.assertEqual(len(called), 2)
        self.assertEqual(value["summary"]["status_counts"], {"request_failure": 2})
        self.assertEqual(value["summary"]["recall"]["denominator"], 1)
        self.assertEqual(value["summary"]["preservation"]["denominator"], 1)
        self.assertEqual(value["summary"]["validity"]["denominator"], 2)
        self.assertEqual(value["registered_slots"], 4)
        self.assertEqual(
            value["disposition_counts"],
            {"admitted": 2, "unknown": 1, "source_unavailable": 1},
        )
        self.assertEqual(
            value["source_stratum_denominators"]["language"]["ko"]["registered"], 2
        )
        self.assertEqual(
            value["source_stratum_denominators"]["language"]["ko"]["requested"], 1
        )
        manifest = sealed.read_json(observed / "manifest.json")
        (observed / "observations.jsonl").write_text("")
        manifest["files_sha256"]["observations.jsonl"] = sealed.sha256(
            observed / "observations.jsonl"
        )
        (observed / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Every request"):
            sealed.grade(self.freeze_path, release, observed, self.root / "missing")

    def test_icl_budget_does_not_shrink_canonical_cohort(self):
        tokenizer = OverflowICLTokenizer()
        inputs = self.prepare(tokenizer)
        release = self.seal(inputs, tokenizer=tokenizer)
        manifest = sealed.read_json(release / "manifest.json")
        self.assertEqual(len(manifest["case_ids_by_run"]["native_base"]), 2)
        self.assertEqual(manifest["case_ids_by_run"]["base_icl"], [])
        self.assertEqual(len(sealed.rows(release / "run-exclusions.jsonl")), 2)

    def test_interrupted_process_keeps_completed_observations_but_is_not_gradeable(
        self,
    ):
        inputs = self.prepare()
        release = self.seal(inputs)
        identity = {"model": "native_base", "engineering_fixture_only": True}
        count = 0

        def interrupted(request):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt()
            return {"text": '{"action":"no_action"}', "finish_reason": "stop"}

        destination = self.root / "interrupted"
        with self.assertRaises(KeyboardInterrupt):
            sealed.record_run(
                self.freeze_path,
                release,
                "native_base",
                destination,
                interrupted,
                tokenizer=self.tokenizer,
                predictor_identity=identity,
            )
        self.assertEqual(len(sealed.rows(destination / "observations.jsonl")), 1)
        self.assertEqual(
            sealed.read_json(destination / "run-registration.json")["requested"], 2
        )
        self.assertFalse((destination / "manifest.json").exists())
        with self.assertRaises(FileNotFoundError):
            sealed.grade(
                self.freeze_path, release, destination, self.root / "ungradeable"
            )

    def test_no_development_guard_is_relabelled_or_bypassed(self):
        inputs = self.prepare()
        request = sealed.rows(inputs / "requests-native_base.jsonl")[0]
        self.assertEqual(request["split"], "final_test")
        with self.assertRaisesRegex(ValueError, "non-development"):
            conditions.validate_request(request)
        path = inputs / "inputs.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "payload changed"):
            self.seal(inputs)

    def test_scribe_adapter_preserves_raw_words_overlap_and_unaligned_annotations(self):
        raw = {
            "language_code": "ko",
            "text": "가 나 다",
            "words": [
                {
                    "type": "word",
                    "text": "가",
                    "start": 0.2,
                    "end": 0.8,
                    "speaker_id": "s0",
                },
                {
                    "type": "word",
                    "text": "나",
                    "start": 0.4,
                    "end": 1.0,
                    "speaker_id": "s1",
                },
                {
                    "type": "word",
                    "text": "다",
                    "start": 1.2,
                    "end": 1.2,
                    "speaker_id": "s0",
                },
                {
                    "type": "audio_event",
                    "text": "[sound]",
                    "start": 1.4,
                    "end": 1.5,
                    "speaker_id": None,
                },
            ],
        }
        provenance = {
            "source_id": sealed.opaque_source_id("engineering-window"),
            "window_id": "engineering-window",
            "duration_seconds": 2.0,
            "language": "ko",
        }
        doc = sealed.normalize_scribe_response(raw, provenance)
        self.assertEqual(doc, sealed.normalize_scribe_response(raw, provenance))
        text = [item for item in doc["content"] if item["type"] == "text"]
        self.assertEqual(
            [(i["text"], i["sourceStart"], i["length"]) for i in text],
            [("가 나", 0.2, 0.8)],
        )
        metadata = doc["metadata"]["transcriptions"][0]
        self.assertEqual(metadata["raw_response"], raw)
        self.assertEqual(metadata["regions"][0]["boundary_status"], "overlap")
        self.assertEqual(
            [a["type"] for a in metadata["annotations"]],
            ["unaligned_word", "audio_event"],
        )
        raw["words"][0]["end"] = 4
        with self.assertRaisesRegex(Exception, "invalid timing"):
            sealed.normalize_scribe_response(raw, provenance)

    def test_icl_demonstrations_match_actual_train_rows_and_exposed_index(self):
        original = copy.deepcopy(self.freeze)
        run = next(row for row in self.freeze["runs"] if row["role"] == "base_icl")
        demos = copy.deepcopy(run["condition"]["demonstrations"])
        demos[0]["action"] = {"action": "no_action"}
        run["condition"] = conditions.condition_specs(demos)["train_icl"]
        self.replace_freeze()
        with self.assertRaisesRegex(ValueError, "actual TRAIN case"):
            sealed.load_gate(self.freeze_path)
        self.freeze = original
        exposed = self.root / "exposed-text-hashes.json"
        exposed.write_text(json.dumps({"sha256": []}))
        policy = self.root / "admission-policy.json"
        value = sealed.read_json(policy)
        value["exposed_text_hashes"] = sealed.binding(exposed)
        policy.write_text(json.dumps(value))
        self.freeze["policies"]["admission"] = sealed.binding(policy)
        self.replace_freeze()
        with self.assertRaisesRegex(ValueError, "exposed-text index"):
            sealed.load_gate(self.freeze_path)

    def test_reviewer_cannot_also_be_grader(self):
        policy = self.root / "admission-policy.json"
        value = sealed.read_json(policy)
        value["actors"]["grader"] = value["actors"]["reviewer"]
        policy.write_text(json.dumps(value))
        self.freeze["policies"]["admission"] = sealed.binding(policy)
        self.replace_freeze()
        with self.assertRaisesRegex(ValueError, "actors are not separated"):
            sealed.freeze_inputs(
                self.freeze_path, ForbiddenBodyPath(), self.root / "never"
            )

    def test_authored_source_clock_and_synthetic_history_are_enforced(self):
        original = sealed.rows(self.candidates())
        for fault in (
            "focus",
            "future_history",
            "human_history",
            "human_feedback",
            "shared_structure",
        ):
            with self.subTest(fault=fault):
                values = copy.deepcopy(original)
                candidate = values[0]
                if fault == "focus":
                    candidate["runtime"]["focus_uuid"] = "b"
                elif fault in {"future_history", "human_history"}:
                    before = candidate["document"]
                    bundle = contract.PassiveContextBuilder(self.tokenizer).build(
                        before, "c", as_of=0
                    )
                    prediction = {"action": "correct_text", "target": [1], "text": "we"}
                    after = contract.apply_passive_edit(before, bundle, prediction)
                    candidate["document"] = after
                    candidate["runtime"].update(
                        sequence=1,
                        history=[
                            contract.committed_transition(
                                before,
                                after,
                                prediction,
                                2 if fault == "future_history" else 1,
                                origin="human"
                                if fault == "human_history"
                                else "synthetic",
                            )
                        ],
                    )
                elif fault == "human_feedback":
                    bundle = contract.PassiveContextBuilder(self.tokenizer).build(
                        candidate["document"], "c", as_of=0
                    )
                    ledger = contract.PassiveFeedbackLedger("fixture-authored")
                    ledger.propose(
                        "p",
                        0,
                        candidate["document"],
                        bundle,
                        {"action": "correct_text", "target": [1], "text": "we"},
                    )
                    ledger.record("p", "shown", 1, origin="human")
                    candidate["runtime"].update(
                        sequence=1, feedback_journal=ledger.dump()
                    )
                else:
                    original_structure = Path(
                        values[2]["lineage"]["preword_structure"]["path"]
                    )
                    alternate = self.root / "another-structure.json"
                    alternate.write_bytes(original_structure.read_bytes())
                    values[2]["lineage"]["preword_structure"] = sealed.binding(
                        alternate
                    )
                path = self.root / f"candidates-{fault}.jsonl"
                sealed.write_rows(path, values)
                with self.assertRaisesRegex(
                    ValueError, "cursor|human|Future|shared structure"
                ):
                    sealed.freeze_inputs(
                        self.freeze_path,
                        path,
                        self.root / f"rejected-{fault}",
                        tokenizer=self.tokenizer,
                    )

    def test_resigned_input_payload_cannot_reuse_an_old_input_record_digest(self):
        inputs = self.prepare()
        values = sealed.rows(inputs / "inputs.jsonl")
        values[0]["provenance"]["model_predictions_seen"] = True
        path = inputs / "inputs.jsonl"
        path.write_text("".join(sealed.packed(row) + "\n" for row in values))
        manifest = sealed.read_json(inputs / "manifest.json")
        manifest["files_sha256"][path.name] = sealed.sha256(path)
        (inputs / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "input binding changed"):
            self.seal(inputs)

    def test_resigned_prelabel_record_cannot_change_the_reviewed_input(self):
        inputs = self.prepare()
        refs = self.references(inputs)
        receipt_path = Path(refs[0]["audit"]["prelabel_input_record"]["path"])
        record_path = receipt_path.parent / "input-record.json"
        record = sealed.read_json(record_path)
        record["provenance"]["selection_before_words"] = False
        record["input_record_sha256"] = sealed.digest(
            {
                key: value
                for key, value in record.items()
                if key != "input_record_sha256"
            }
        )
        record_path.write_text(json.dumps(record))
        receipt = sealed.read_json(receipt_path)
        receipt["files_sha256"][record_path.name] = sealed.sha256(record_path)
        receipt_path.write_text(json.dumps(receipt))
        refs[0]["audit"]["prelabel_input_record"] = sealed.binding(receipt_path)
        with self.assertRaisesRegex(ValueError, "Prelabel input record"):
            self.seal(inputs, refs)


class NaturalCheckpointsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SealedTaskTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        fixture = self.fixture
        self.root, self.tokenizer = fixture.root, fixture.tokenizer
        window_id = "engineering-natural-window"
        fixture.roster = [
            {
                "slot_id": f"natural-c{n}",
                "source_group": "engineering-heldout-en",
                "language": "en",
                "stratum": "natural_source_first"
                if n == 0
                else "natural_source_simulated_session",
                "admission": "not_materialized",
                "window_id": window_id,
                "checkpoint": n,
            }
            for n in range(3)
        ]
        roster_path, window_path = (
            self.root / "roster.jsonl",
            self.root / "windows.json",
        )
        roster_path.write_text(
            "".join(sealed.packed(row) + "\n" for row in fixture.roster)
        )
        window_path.write_text(
            json.dumps(
                [
                    {
                        "window_id": window_id,
                        "source_group": "engineering-heldout-en",
                        "publisher_recording_id": "engineering-recording",
                        "language": "en",
                        "start_seconds": 0,
                        "duration_seconds": 90,
                    }
                ]
            )
        )
        metadata_path = self.root / "metadata.json"
        metadata = sealed.read_json(metadata_path)
        metadata["files_sha256"].update(
            {p.name: sealed.sha256(p) for p in (roster_path, window_path)}
        )
        metadata_path.write_text(json.dumps(metadata))
        fixture.freeze.update(
            metadata_manifest=sealed.binding(metadata_path),
            roster=sealed.binding(roster_path),
            natural_windows=sealed.binding(window_path),
        )
        fixture.replace_freeze()
        doc = document()
        source = sealed.opaque_source_id(window_id)
        for item in sealed.sealed_v1.timed(doc):
            item["source"] = source
            item["sourceStart"] += 42
        next(item for item in doc["content"] if item["uuid"] == "c")["text"] = (
            "can begni."
        )
        imported_source = {
            "window_id": window_id,
            "source_group": "engineering-heldout-en",
            "recording_id": "engineering-recording",
            "language": "en",
            "source_id": source,
            "duration_seconds": 90,
            "split": "final_test",
            "engineering_fixture_only": True,
        }
        imported = sealed.publish(
            self.root / "synthetic-import",
            {"document.json": doc, "provenance.json": imported_source},
            {
                "status": "imported_unlabeled",
                "freeze": sealed.binding(fixture.freeze_path),
                "initial_input_edited": False,
                "acoustic_certification": "none_engineering_fixture",
            },
        )
        builder = contract.PassiveContextBuilder(self.tokenizer)
        self.actions = [
            {"action": "correct_text", "target": [1], "text": "we"},
            {"action": "correct_text", "target": [2], "text": "can begin."},
            {"action": "no_action"},
        ]
        ledger = contract.PassiveFeedbackLedger("engineering-natural-session")
        values, history, self.reference_rows = [], [], []
        gate, exporter = sealed.load_gate(fixture.freeze_path), sealed.Exporter()
        try:
            for n, slot in enumerate(fixture.roster):
                candidate = {
                    "id": slot["slot_id"],
                    "status": "candidate",
                    "split": "final_test",
                    "language": "en",
                    "source": {
                        "source_group": slot["source_group"],
                        "recording_id": "engineering-recording",
                        "split": "final_test",
                    },
                    "document": copy.deepcopy(doc),
                    "runtime": {
                        "focus_uuid": "c",
                        "sequence": (0, 1, 3)[n],
                        "selection": [],
                        "history": copy.deepcopy(history),
                        "feedback_journal": ledger.dump() if n == 2 else None,
                        "caption_profile": contract.CaptionProfile().model_dump(),
                    },
                    "lineage": {
                        "stratum": slot["stratum"],
                        "window_id": window_id,
                        "scenario_family": "synthetic_natural_pipeline_only",
                        "selection_method": "fixture",
                    },
                    "provenance": {
                        "selection_before_words": True,
                        "model_predictions_seen": False,
                        "created_at": sealed.now(),
                        "freeze_sha256": sealed.sha256(fixture.freeze_path),
                        "initial_import_manifest": imported,
                    },
                }
                values.append(candidate)
                sealed.candidate_identity(candidate, slot, gate)
                sealed.natural_checkpoint(
                    candidate, slot, gate, values[n - 1] if n else None
                )
                case = sealed.input_record(
                    candidate, sealed.dataset.runtime_bundle(candidate, builder)
                )
                prelabel = fixture.prelabel_audit(case)
                action = self.actions[n]
                reference = {
                    "id": case["id"],
                    "status": "preserve" if n == 2 else "edit_required",
                    "input_record_sha256": case["input_record_sha256"],
                    "primary": action,
                    "accepted_predictions": [action],
                    "acoustic_evidence": "none_engineering_fixture",
                    "audit": {
                        "author": "fixture-author",
                        "reviewer": "fixture-reviewer",
                        "input_first": True,
                        **prelabel,
                        "predictions_seen": False,
                        "reviewed_actions": list(contract.ACTIONS),
                        "rationale": "Synthetic pipeline fixture, not a held-out semantic judgment.",
                        "input_record_sha256": case["input_record_sha256"],
                    },
                }
                self.reference_rows.append(reference)
                sealed.reference_audit(reference, case, gate, self.tokenizer)
                sealed.reference_proof(case, action, exporter)
                if n == 0:
                    after = contract.apply_passive_edit(doc, case["bundle"], action)
                    history = [contract.committed_transition(doc, after, action, 1)]
                    doc = after
                elif n == 1:
                    ledger.propose("next", 1, doc, case["bundle"], action)
                    ledger.record("next", "shown", 2, origin="synthetic")
                    ledger.record("next", "dismissed", 3, origin="synthetic")
        finally:
            exporter.close()
        self.candidates_path = self.root / "natural-candidates.jsonl"
        sealed.write_rows(self.candidates_path, values)

    def prepare(self, name="natural-inputs"):
        output = self.root / name
        sealed.freeze_inputs(
            self.fixture.freeze_path,
            self.candidates_path,
            output,
            tokenizer=self.tokenizer,
        )
        return output

    def references(self, inputs):
        self.assertEqual(
            [
                row["input_record_sha256"]
                for row in sealed.rows(inputs / "inputs.jsonl")
            ],
            [row["input_record_sha256"] for row in self.reference_rows],
        )
        return copy.deepcopy(self.reference_rows)

    def test_natural_c0_c1_c2_qualify_replay_seal_grade_and_reject_unqualified_predecessors(
        self,
    ):
        inputs = self.prepare()
        cases = sealed.rows(inputs / "inputs.jsonl")
        self.assertEqual([row["runtime"]["sequence"] for row in cases], [0, 1, 3])
        self.assertTrue(cases[2]["input"]["recent_feedback"])
        refs = self.references(inputs)
        for prior in (0, 1):
            altered = copy.deepcopy(refs)
            altered[prior]["primary"]["text"] = "different qualified fixture wording"
            reference_path = self.root / f"unqualified-{prior}.jsonl"
            sealed.write_rows(reference_path, altered)
            with self.assertRaisesRegex(ValueError, "unqualified edit"):
                sealed.seal_references(
                    self.fixture.freeze_path,
                    inputs,
                    reference_path,
                    self.root / f"never-{prior}",
                    tokenizer=self.tokenizer,
                )
        release = self.fixture.seal(inputs, refs)
        observations, graded = (
            self.root / "natural-observed",
            self.root / "natural-grade",
        )
        sealed.record_run(
            self.fixture.freeze_path,
            release,
            "always_no_action",
            observations,
            tokenizer=self.tokenizer,
        )
        sealed.grade(self.fixture.freeze_path, release, observations, graded)
        report = sealed.read_json(graded / "report.json")
        self.assertEqual(report["canonical_admitted_cases"], 3)
        self.assertEqual(report["summary"]["recall"]["denominator"], 2)
        self.assertEqual(report["summary"]["preservation"]["numerator"], 1)
        self.assertEqual(report["natural_windows"], 1)

    def test_context_budget_exclusions_cannot_seed_dependent_checkpoints(self):
        original = sealed.make_requests

        def oversized(*args):
            value = original(*args)
            value["native_base"]["prefix_ids"] = [2] * (sealed.formats.INPUT_LIMIT + 1)
            return value

        for kind in ("builder", "native_prefix"):
            patch = (
                mock.patch.object(
                    sealed.dataset,
                    "runtime_bundle",
                    side_effect=sealed.ContextBudgetError("fixture budget"),
                )
                if kind == "builder"
                else mock.patch.object(sealed, "make_requests", side_effect=oversized)
            )
            with self.subTest(kind=kind), patch:
                output = self.prepare(kind)
            census = sealed.rows(output / "census.jsonl")
            self.assertEqual(
                [row["status"] for row in census],
                [
                    "context_ineligible",
                    "structural_state_unavailable",
                    "structural_state_unavailable",
                ],
            )
            self.assertEqual(sealed.rows(output / "inputs.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
