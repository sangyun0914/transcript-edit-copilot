import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_actions_v3_conditions as conditions
import prepare_passive_actions_v3 as dataset


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) > 2 and len(messages[-1]["content"]) > 4000:
            return [8] * 3585
        return [8] * 100

    def encode(self, text, **kwargs):
        return [256 + ord(character) for character in text]


class ConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (dataset.FIXTURES / "revision-004").is_dir():
            raise unittest.SkipTest("Local authored research data is not distributed.")
        cls.tokenizer = Tokenizer()
        cls.builder = conditions.contract.PassiveContextBuilder(cls.tokenizer)
        with patch.object(dataset, "FIXTURES", dataset.FIXTURES / "revision-004"):
            cases, _ = dataset.build_authored_cases(cls.builder)
        cls.cases = {case["id"]: case for case in cases}
        cls.train = [
            cls.cases["actions-v3-archive-transposition"],
            cls.cases["actions-v3-ko-garden-numbers"],
        ]
        cls.development = [
            cls.cases["actions-v3-ferry-spelling"],
            cls.cases["actions-v3-ko-tool-policy-trim_silence"],
            cls.cases["actions-v3-ferry-long-window"],
        ]

    def decision(self, case, condition, release_sha="fixture-release"):
        projected = conditions.project_input(case["input"], condition["id"])
        return {
            "case_id": case["id"],
            "condition_id": condition["id"],
            "condition_sha256": conditions.digest(condition),
            "original_input_sha256": conditions.input_hash(case["input"]),
            "input_sha256": conditions.input_hash(projected),
            "release_manifest_sha256": release_sha,
            "status": "qualified",
            "actor": "ai_generated",
            "reviewer_id": "unit-fixture-independent-review",
            "model_predictions_consulted": False,
            "judgment": {"status": "preserve"},
            "accepted_predictions": [{"action": "no_action"}],
            "rationale": "Fixture qualification: missing pacing evidence does not select a retained duration.",
        }

    def test_projection_preserves_exact_window_mapping_and_runtime_gate(self):
        case = self.development[1]
        before = copy.deepcopy(case)
        spec = conditions.condition_specs([])["context_only"]
        projected = conditions.project_input(case["input"], "context_only")
        for key in case["input"]:
            expected = [] if key in spec["removed_fields"] else case["input"][key]
            self.assertEqual(projected[key], expected)
        decision = self.decision(case, spec)
        qualified = conditions.qualify(
            case, spec, projected, decision, "fixture-release"
        )
        bundle = dataset.runtime_bundle(case, self.builder)
        prepared = conditions.conditioned_case(case, bundle, spec, qualified)
        self.assertEqual(case, before)
        self.assertEqual(prepared["bundle"]["mapping"], bundle["mapping"])
        self.assertEqual(prepared["document"], case["document"])
        self.assertEqual(prepared["runtime"], case["runtime"])
        self.assertEqual(prepared["input"]["items"], case["input"]["items"])
        self.assertEqual(prepared["primary"], {"action": "no_action"})
        self.assertNotEqual(prepared["primary"], case["primary"])

    def test_removed_evidence_needs_bound_review_and_unknown_is_not_abstention(self):
        case = self.development[1]
        spec = conditions.condition_specs([])["context_only"]
        projected = conditions.project_input(case["input"], "context_only")
        self.assertEqual(
            conditions.qualify(case, spec, projected, None, "fixture-release")[
                "status"
            ],
            "needs_review",
        )
        decision = self.decision(case, spec)
        for key in (
            "input_sha256",
            "original_input_sha256",
            "condition_sha256",
            "release_manifest_sha256",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                conditions.qualify(
                    case, spec, projected, {**decision, key: "wrong"}, "fixture-release"
                )
        uncertain = {**decision, "status": "uncertain", "accepted_predictions": []}
        self.assertEqual(
            conditions.qualify(case, spec, projected, uncertain, "fixture-release")[
                "status"
            ],
            "uncertain",
        )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            conditions.qualify(
                case,
                spec,
                projected,
                {**uncertain, "accepted_predictions": [{"action": "no_action"}]},
                "fixture-release",
            )

    def test_hidden_feedback_does_not_disable_actual_suppression(self):
        case = self.cases["actions-v3-river-dismissed-correction"]
        spec = conditions.condition_specs([])["context_history"]
        projection = conditions.project_input(case["input"], spec["id"])
        decision = self.decision(case, spec)
        decision["judgment"] = {"status": "edit_required"}
        decision["accepted_predictions"] = [
            {
                "action": "correct_text",
                "target": [0],
                "text": case["input"]["recent_feedback"][0]["proposed_text"],
            }
        ]
        qualified = conditions.qualify(
            case, spec, projection, decision, "fixture-release"
        )
        with self.assertRaisesRegex(ValueError, "unchanged runtime feedback gate"):
            conditions.conditioned_case(
                case, dataset.runtime_bundle(case, self.builder), spec, qualified
            )

    def test_icl_uses_fixed_train_examples_and_never_current_references(self):
        selected = conditions.select_demonstrations(
            self.train, self.tokenizer, {"manifest": "fixture"}
        )
        self.assertEqual(
            selected,
            conditions.select_demonstrations(
                list(reversed(self.train)), self.tokenizer, {"manifest": "fixture"}
            ),
        )
        self.assertEqual({row["language"] for row in selected}, {"en", "ko"})
        self.assertEqual(
            {row["action"]["action"] == "no_action" for row in selected}, {False, True}
        )
        with self.assertRaisesRegex(ValueError, "train-only"):
            conditions.select_demonstrations(
                [self.development[0], *self.train], self.tokenizer, {}
            )
        spec = conditions.condition_specs(selected)["train_icl"]
        case = copy.deepcopy(self.development[0])
        case["primary"]["text"] = "HIDDEN_DEVELOPMENT_ANSWER"
        case["rationale"] = "HIDDEN_RATIONALE"
        for format_name in conditions.formats.FORMATS:
            messages = conditions.expected_messages(case["input"], format_name, spec)
            self.assertEqual(
                [row["role"] for row in messages],
                ["system", "user", "assistant", "user", "assistant", "user"],
            )
            self.assertEqual(
                messages[-1], conditions.formats.messages(case["input"], format_name)[1]
            )
            self.assertNotIn("HIDDEN", json.dumps(messages))
            for index, demo in enumerate(selected):
                parsed = conditions.formats.parse(
                    messages[2 + 2 * index]["content"], demo["input"], format_name
                )
                self.assertEqual(parsed.model_dump(), demo["action"])

    def prepare_fixture(self, root, qualifications=None):
        release = root / "release"
        for split, cases in (("train", self.train), ("dev", self.development)):
            dataset.write_rows(release / "cases" / f"{split}.jsonl", cases)
        dataset.write(
            release / "manifest.json",
            {
                "files_sha256": {
                    f"cases/{split}.jsonl": conditions.sha256(
                        release / "cases" / f"{split}.jsonl"
                    )
                    for split in ("train", "dev")
                }
            },
        )
        review = root / "review.json"
        dataset.write(
            review,
            {
                "status": "passed",
                "manifest_sha256": conditions.sha256(release / "manifest.json"),
                "model_predictions_consulted": False,
            },
        )
        result = conditions.prepare(
            release,
            review,
            root / "prepared",
            qualifications,
            tokenizer=(self.tokenizer, {"kind": "test_double_not_native"}),
        )
        return result, root / "prepared", release, review

    def test_icl_overflow_does_not_shrink_canonical_and_exclusions_are_registered(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, output, _, _ = self.prepare_fixture(Path(temporary))
            self.assertEqual(result["condition_requests"]["canonical_full_context"], 3)
            self.assertEqual(result["condition_requests"]["train_icl"], 2)
            self.assertEqual(result["review_queue"], 1)
            self.assertEqual(
                result["comparisons"]["icl_matched"]["requests_per_condition"], 2
            )
            self.assertEqual(
                result["comparisons"]["all_context_matched"]["requests_per_condition"],
                2,
            )
            queue = dataset.rows(output / "review-queue.jsonl")
            self.assertEqual(queue[0]["case_id"], self.development[1]["id"])
            self.assertTrue(queue[0]["original_input"]["recent_edits"])
            self.assertEqual(queue[0]["projected_input"]["recent_edits"], [])
            exclusions = dataset.rows(output / "train_icl" / "exclusions.jsonl")
            self.assertEqual(exclusions[0]["case_id"], self.development[2]["id"])
            self.assertEqual(exclusions[0]["reason"], "common_format_input_budget")
            self.assertEqual(exclusions[0]["stratum"], "long_context_remote_spelling")
            request = dataset.rows(output / "train_icl" / "requests-json.jsonl")[0]
            conditions.validate_request(request)
            request["condition_sha256"] = "other"
            with self.assertRaises(ValueError):
                conditions.validate_request(request)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["canonical_not_restricted_by_icl_or_ablation"])
            self.assertEqual(manifest["cohort_policy"], "matched_common_json_rewrite")
            for name, expected in manifest["files_sha256"].items():
                self.assertEqual(conditions.sha256(output / name), expected)

    def test_qualified_changed_labels_are_saved_without_changing_source_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, first, release, review = self.prepare_fixture(Path(temporary))
            source_hash = conditions.sha256(release / "cases/dev.jsonl")
            row = dataset.rows(first / "review-queue.jsonl")[0]
            spec = conditions.condition_specs([])[row["condition_id"]]
            qualification = self.decision(
                self.development[1], spec, conditions.sha256(release / "manifest.json")
            )
            qualifications = Path(temporary) / "qualifications.jsonl"
            dataset.write_rows(qualifications, [qualification])
            second = Path(temporary) / "qualified"
            result = conditions.prepare(
                release,
                review,
                second,
                qualifications,
                tokenizer=(self.tokenizer, {"kind": "test_double_not_native"}),
            )
            self.assertEqual(result["review_queue"], 0)
            self.assertEqual(result["condition_requests"]["context_only"], 3)
            changed = dataset.rows(second / "context_only/cases.jsonl")[1]
            self.assertEqual(changed["primary"], {"action": "no_action"})
            self.assertEqual(
                changed["condition_qualification"]["basis"],
                "independent_projected_input_review",
            )
            self.assertEqual(
                conditions.sha256(release / "cases/dev.jsonl"), source_hash
            )

    def test_bound_proof_reuse_avoids_replay_but_rejects_changed_evidence(self):
        case = copy.deepcopy(self.train[0])
        proof = dataset.validate_case(case, self.builder)
        case["execution_sha256"] = conditions.digest(proof)
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary)
            dataset.write_rows(release / "cases/train.jsonl", [case])
            dataset.write_rows(
                release / "reference-executions.jsonl", [{"id": case["id"], **proof}]
            )
            manifest = {
                "contract_version": conditions.contract.VERSION,
                "counts": {"train": 1},
                "files_sha256": {
                    name: conditions.sha256(release / name)
                    for name in ("cases/train.jsonl", "reference-executions.jsonl")
                },
                "source_sha256": {
                    str(path): conditions.sha256(path)
                    for path in (
                        Path(conditions.contract.__file__).resolve(),
                        conditions.contract.PROMPT_PATH.resolve(),
                    )
                },
            }
            with patch.object(
                dataset,
                "validate_case",
                side_effect=AssertionError("Unexpected replay"),
            ):
                result = conditions.verify_reference_proofs(
                    release, manifest, [case], self.builder
                )
                self.assertEqual(result["cached_cases"], 1)
                self.assertEqual(result["replayed_cases"], 0)
                changed = copy.deepcopy(case)
                changed["document"]["metadata"]["tampered"] = True
                with self.assertRaisesRegex(ValueError, "another case"):
                    conditions.verify_reference_proofs(
                        release, manifest, [changed], self.builder
                    )
                changed_manifest = copy.deepcopy(manifest)
                changed_manifest["source_sha256"][
                    str(Path(conditions.contract.__file__).resolve())
                ] = "0" * 64
                with self.assertRaisesRegex(ValueError, "contract or prompt changed"):
                    conditions.verify_reference_proofs(
                        release, changed_manifest, [case], self.builder
                    )
            fallback = {**manifest, "source_sha256": {}}
            with patch.object(
                dataset, "validate_case", wraps=dataset.validate_case
            ) as validate:
                result = conditions.verify_reference_proofs(
                    release, fallback, [case], self.builder
                )
                self.assertEqual(result["replayed_cases"], 1)
                validate.assert_called_once()
            with (release / "reference-executions.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "proof file changed"):
                conditions.verify_reference_proofs(
                    release, manifest, [case], self.builder
                )


if __name__ == "__main__":
    unittest.main()
