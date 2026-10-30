import copy
import json
import unittest

import compare_actions_v3_results as comparison
import prepare_history_length_actions_v3 as controls
import test_prepare_history_length_actions_v3 as history_test_fixtures


class ComparisonTests(unittest.TestCase):
    def arms(self):
        records, inputs = {}, {}
        for i in range(4):
            identity = str(i)
            records[identity] = {
                "source_group": str(i // 2),
                "language": "en",
                "stratum": "symbolic",
                "judgment": "preserve" if i % 2 else "edit_required",
                "reference_action": "no_action" if i % 2 else "correct_text",
                "displayed": not i % 2,
                "useful": not i % 2,
                "preserved": bool(i % 2),
                "model_abstained": bool(i % 2),
                "outcome_match": True,
                "valid": True,
            }
            inputs[identity] = {
                "input": {
                    "text": identity,
                    "recent_edits": ["earlier correction"],
                    "recent_feedback": [],
                },
                "accepted_result_sha256": ["result-" + identity],
            }
        return records, inputs

    def test_model_comparison_rejects_changed_input_or_accepted_alternative(self):
        records, inputs = self.arms()
        for field, value in (
            ("input", {"text": "different"}),
            ("accepted_result_sha256", ["new alternate"]),
        ):
            changed = copy.deepcopy(inputs)
            changed["0"][field] = value
            with self.assertRaises(ValueError):
                comparison.matched(records, records, inputs, changed, "models")

    def test_context_removal_counts_changed_reference_and_rejects_replacement(self):
        records, inputs = self.arms()
        changed = copy.deepcopy(inputs)
        changed["0"]["input"]["recent_edits"] = []
        changed["0"]["accepted_result_sha256"] = ["unchanged document"]
        projected = copy.deepcopy(records)
        projected["0"].update(judgment="preserve", reference_action="no_action")
        identities, changed_ids, relabeled = comparison.matched(
            records, projected, inputs, changed, "context"
        )
        self.assertEqual((len(identities), changed_ids, relabeled), (4, ["0"], ["0"]))
        changed["0"]["input"]["recent_edits"] = ["replacement history"]
        with self.assertRaises(ValueError):
            comparison.matched(records, projected, inputs, changed, "context")

    def test_paired_source_resampling_retains_correlation_and_zero_differences(self):
        records, _ = self.arms()
        result = comparison.source_resampling(
            records, records, list(records), resamples=50
        )
        self.assertEqual(result["source_groups"], 2)
        for interval in result["percentile95_descriptive"].values():
            self.assertEqual((interval["low"], interval["high"]), (0, 0))
            self.assertEqual(interval["defined_resamples"], 50)

    def test_abstention_has_undefined_precision_and_one_group_has_no_interval(self):
        records, _ = self.arms()
        for row in records.values():
            row.update(displayed=False, useful=False)
        result = comparison.source_resampling(
            records, records, list(records), resamples=50
        )
        interval = result["percentile95_descriptive"]["precision"]
        self.assertIsNone(interval["low"])
        self.assertEqual(interval["undefined_resamples"], 50)
        result = comparison.source_resampling(
            records, records, ["0", "1"], resamples=50
        )
        for interval in result["percentile95_descriptive"].values():
            self.assertIsNone(interval["low"])
            self.assertEqual(interval["defined_resamples"], 0)

    def test_invalid_hidden_outputs_keep_precision_but_reduce_validity(self):
        records, _ = self.arms()
        degraded = copy.deepcopy(records)
        for identity in ("1", "3"):
            degraded[identity].update(
                valid=False, preserved=False, model_abstained=False, outcome_match=False
            )
        before = comparison.values(comparison.tallies(list(records.values())))
        after = comparison.values(comparison.tallies(list(degraded.values())))
        self.assertEqual((before["precision"], after["precision"]), (1, 1))
        self.assertEqual((before["validity"], after["validity"]), (1, 0.5))
        result = comparison.source_resampling(
            records, degraded, list(records), resamples=50
        )
        interval = result["percentile95_descriptive"]["validity"]
        self.assertEqual((interval["low"], interval["high"]), (-0.5, -0.5))


class HistoryLengthComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not controls.FIXTURE.is_file():
            raise unittest.SkipTest(
                "History diagnostics require local experiment fixtures."
            )
        fixture = json.loads(controls.FIXTURE.read_text())
        cls.records = {length: {} for length in controls.LENGTHS}
        cls.inputs = {length: {} for length in controls.LENGTHS}
        for episode in fixture["episodes"]:
            cases, requests, _ = controls.build_episode(
                fixture,
                episode,
                history_test_fixtures.Tokenizer(),
                controls.digest(fixture),
            )
            for length, case, request in zip(
                controls.LENGTHS, cases, requests, strict=True
            ):
                cls.inputs[length][case["id"]] = {
                    **request,
                    "accepted_result_sha256": case["accepted_result_sha256"],
                }
                cls.records[length][case["id"]] = {
                    "source_group": case["source"]["source_group"],
                    "language": case["language"],
                    "stratum": case["lineage"]["stratum"],
                    "scenario_family": episode["mechanism"],
                    "judgment": case["judgment"]["status"],
                    "reference_action": case["primary"]["action"],
                }

    def test_registered_suffixes_count_conditional_reference_changes(self):
        for left, right, changes in ((0, 3, 4), (0, 8, 6), (3, 8, 2)):
            identities, changed, relabeled = comparison.matched(
                self.records[left],
                self.records[right],
                self.inputs[left],
                self.inputs[right],
                "history-length",
            )
            self.assertEqual(
                (len(identities), len(changed), len(relabeled)), (8, 8, changes)
            )

    def test_rebound_arbitrary_history_subset_is_still_rejected(self):
        left, right = copy.deepcopy(self.inputs[3]), copy.deepcopy(self.inputs[8])
        identity = next(iter(left))
        changed = left[identity]
        changed["input"]["recent_edits"] = copy.deepcopy(
            right[identity]["input"]["recent_edits"][:3]
        )
        changed["input_sha256"] = controls.conditions.input_hash(changed["input"])
        changed["messages"] = controls.conditions.expected_messages(
            changed["input"], "json", changed["condition"]
        )
        changed["messages_sha256"] = controls.digest(changed["messages"])
        for request in (changed, right[identity]):
            request["history_length"]["projection_input_sha256"]["3"] = changed[
                "input_sha256"
            ]
        with self.assertRaisesRegex(ValueError, "exact chronological suffix"):
            comparison.matched(
                self.records[3], self.records[8], left, right, "history-length"
            )

    def test_history_mode_rejects_incomplete_or_mixed_arms(self):
        left = copy.deepcopy(self.inputs[3])
        identity = next(iter(left))
        left.pop(identity)
        with self.assertRaisesRegex(ValueError, "complete eight-episode"):
            comparison.matched(
                self.records[3], self.records[8], left, self.inputs[8], "history-length"
            )
        left = copy.deepcopy(self.inputs[3])
        left[identity] = copy.deepcopy(self.inputs[0][identity])
        with self.assertRaisesRegex(ValueError, "complete eight-episode"):
            comparison.matched(
                self.records[3], self.records[8], left, self.inputs[8], "history-length"
            )

    def test_history_mode_keeps_model_and_numerical_conditions_fixed(self):
        report = {
            "model_identity": {"kind": "native", "adapter": "selected"},
            "numerical_regime": "bf16",
            "format": "json",
            "schema_constrained": False,
        }
        comparison.validate_history_regime(report, report)
        for field, value in (
            ("model_identity", {"kind": "different"}),
            ("numerical_regime", "quantized"),
            ("format", "rewrite"),
            ("schema_constrained", True),
        ):
            with self.assertRaisesRegex(ValueError, "keep model"):
                comparison.validate_history_regime(report, {**report, field: value})


if __name__ == "__main__":
    unittest.main()
