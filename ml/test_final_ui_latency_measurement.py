"""Synthetic renderer clocks only; no actual UI/model or reserved dataset."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import final_ui_latency_measurement as measurement


def event(name, elapsed, outcome=None):
    value = {
        "event": name,
        "at_ms": 10000 + elapsed,
        "since_trigger_ms": elapsed,
        "trace_id": "synthetic-trace",
        "request_id": "synthetic-request",
        "trigger": "automatic",
    }
    if outcome:
        value["outcome"] = outcome
    return value


def fixtures(statuses):
    attempts, outcomes = [], []
    for index, (condition, status, latency) in enumerate(statuses):
        identity = f"synthetic-{index}"
        attempts.append(
            {
                "case_id": identity,
                "condition": condition,
                "language": "en",
                "seed": "en-edit",
                "operation": "focus",
                "intent": "suggestion",
            }
        )
        events = [event("trigger", 0), event("request", 150)]
        if status == "preview":
            events += [
                event("response", latency - 10, "proposal"),
                event("visible_complete_preview", latency),
            ]
        elif status == "no_action":
            events += [event("response", latency, "no_action")]
        elif status == "invalid":
            events += [event("error", latency)]
        else:
            events = [event("trigger", 0), event("debounce_cancelled", 100)]
        outcomes.append(
            {
                "case_id": identity,
                "language": "en",
                "status": status,
                "detail": None,
                "automatic_trigger": True,
                "trace_id": "synthetic-trace",
                "measurements": events,
                "interaction_window": {"start_ms": 9990, "end_ms": 20000},
                "input_events": [
                    {
                        "type": "mousedown",
                        "trusted": True,
                        "at_ms": 10000,
                        "target": "item-synthetic-focus",
                    }
                ],
                "preview_useful": True if status == "preview" else None,
                "usefulness_reason": "Synthetic fixture"
                if status == "preview"
                else None,
                "evidence": [{"path": "synthetic-only", "sha256": "fixture"}],
            }
        )
    ids = [row["case_id"] for row in attempts]
    return {
        "attempts": attempts,
        "planned_case_ids": ids,
        "selected_cases": {"en-edit": {"focus_uuid": "synthetic-focus"}},
    }, {
        "status": "observed",
        "launch_method": "npm start",
        "automatic_only": True,
        "runtime_condition": "cold_and_warm_separate",
        "primary_latency_metric": measurement.PRIMARY_METRIC,
        "latency_origin_amendment": measurement.binding(measurement.ORIGIN_PLAN),
        "planned_case_ids": ids,
        "outcomes": outcomes,
    }


class FinalUILatencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        for name in ("PLAN", "ORIGIN_PLAN"):
            path = Path(temporary.name) / (name + ".md")
            path.write_text("Synthetic plan for renderer clock tests only.\n")
            patch = mock.patch.object(measurement, name, path)
            patch.start()
            self.addCleanup(patch.stop)

    def test_nonpreviews_and_cold_stay_in_denominators_without_fake_preview_latency(
        self,
    ):
        roster, observed = fixtures(
            [
                ("process_cold", "preview", 4000),
                ("warm", "preview", 400),
                ("warm", "no_action", 200),
                ("warm", "invalid", 900),
                ("warm", "not_issued", None),
            ]
        )
        result = measurement.summarize(roster, observed)
        self.assertEqual(result["attempts"], 5)
        self.assertEqual(result["warm_attempts"], 4)
        self.assertEqual(result["warm_preview_latency"]["n"], 1)
        self.assertEqual(result["warm_preview_latency"]["p95_ms"], 400)
        self.assertEqual(result["warm_by_language"]["en"]["preview_latency"]["n"], 1)
        self.assertEqual(result["useful_previews"]["count"], 1)
        self.assertEqual(result["useful_previews"]["eligible_attempts"], 4)
        self.assertEqual(
            result["preview_under_1000ms_per_warm_attempt"],
            {"count": 1, "denominator": 4},
        )
        self.assertEqual(result["cold_startup"][0]["preview_ms"], 4000)
        roster, observed = fixtures([("warm", "no_action", 200)])
        result = measurement.summarize(roster, observed)
        self.assertIsNone(result["warm_preview_latency"]["p95_ms"])
        self.assertIsNone(result["latency_goal_on_observed_previews"])

    def test_nearest_rank_and_strict_thresholds_are_fixed(self):
        result = measurement.distribution([100, 500, 1000, 2000])
        self.assertEqual((result["p50_ms"], result["p95_ms"]), (750, 2000))
        self.assertEqual(result["under_500ms"]["share"], 0.25)
        self.assertEqual(result["under_1000ms"]["share"], 0.5)

    def test_committing_enter_includes_work_before_focus_unblocks_trace(self):
        roster, observed = fixtures([("warm", "preview", 400)])
        roster["attempts"][0]["operation"] = "append_period"
        observed["outcomes"][0]["interaction_window"]["start_ms"] = 9000
        observed["outcomes"][0]["input_events"] = [
            {
                "type": "keydown",
                "key": "Enter",
                "trusted": True,
                "editable": True,
                "at_ms": 9200,
            }
        ]
        result = measurement.summarize(roster, observed)
        self.assertEqual(result["warm_preview_latency"]["p95_ms"], 1200)
        self.assertEqual(result["warm_trace_component_latency"]["p95_ms"], 400)
        self.assertEqual(
            result["attempt_results"][0]["trace_origin_after_dom_input_ms"], 800
        )
        self.assertFalse(result["latency_goal_on_observed_previews"])
        self.assertEqual(
            result["latency_origin_amendment"], observed["latency_origin_amendment"]
        )

    def test_missing_or_ambiguous_origin_never_falls_back_to_trace_latency(self):
        roster, observed = fixtures([("warm", "preview", 400)] * 3)
        observed["outcomes"][0].pop("input_events")
        observed["outcomes"][1]["input_events"] *= 2
        result = measurement.summarize(roster, observed)
        self.assertEqual(result["warm_preview_latency"]["n"], 1)
        self.assertEqual(result["warm_trace_component_latency"]["n"], 3)
        self.assertEqual(result["primary_timing_coverage"]["available"], 1)
        self.assertEqual(result["primary_timing_coverage"]["missing"], 2)
        self.assertEqual(
            result["primary_timing_coverage"]["missing_case_ids"],
            ["synthetic-0", "synthetic-1"],
        )
        self.assertEqual(result["useful_previews"]["count"], 3)
        self.assertEqual(result["useful_previews"]["painted"], 3)
        self.assertIsNone(result["latency_goal_on_observed_previews"])
        self.assertIsNone(result["stretch_goal_on_observed_previews"])
        observed["outcomes"][0]["full_latency_ms"] = 400
        with self.assertRaisesRegex(ValueError, "Recorded primary timing"):
            measurement.summarize(roster, observed)

    def test_missing_outcomes_fake_paint_and_changed_clock_are_rejected(self):
        roster, original = fixtures(
            [("warm", "preview", 600), ("warm", "no_action", 200)]
        )
        changes = []
        value = copy.deepcopy(original)
        value["outcomes"].pop()
        changes.append(value)
        value = copy.deepcopy(original)
        value["outcomes"][0]["measurements"].pop()
        changes.append(value)
        value = copy.deepcopy(original)
        value["outcomes"][0]["measurements"][-1]["since_trigger_ms"] = 1
        changes.append(value)
        value = copy.deepcopy(original)
        value["outcomes"][0]["status"] = "no_action"
        changes.append(value)
        for observed in changes:
            with self.assertRaises(ValueError):
                measurement.summarize(roster, observed)


if __name__ == "__main__":
    unittest.main()
