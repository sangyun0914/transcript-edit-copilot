"""Essential source, evidence, metric, and pre-prediction gates; no model calls."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

import passive_actions as actions
import prepare_passive_actions_final as prep
import probe_passive_actions_final as probe
from test_prepare_passive_actions_final import Tokens, document


def fixture(prediction=None, feedback=None):
    before = prep.project_document(document(), prep.source_map([document()]))
    sequence = 2 if feedback else 0
    bundle = actions.PassiveContextBuilder(Tokens()).build(before, "a", as_of=sequence, feedback=feedback)
    prediction = prediction or {"action": "no_action"}
    after = actions.apply_passive_edit(before, bundle, prediction)
    runtime = {"document": before, "sequence": sequence, "feedback_journal": feedback.dump() if feedback else None}
    case = {"id": "case", "language": "en", "primary": prediction, "input": bundle["input"], "bundle": bundle,
            "reference_outcomes": [{"available": True, "outcome_sha256": prep.outcome_digest(after)}]}
    shown = prediction["action"] != "no_action"
    gateway = {"status": "proposal" if shown else "no_action", "prediction": prediction,
               "request": {"document_id": feedback.document_id if feedback else "case", "request_id": "r", "revision": 0, "interaction_epoch": 0},
               "context": bundle["context"], "target_uuids": [bundle["mapping"][str(i)]["uuid"] for i in prediction.get("target", [])],
               "result_document": after if shown else None, "document_hash": prep.session_hash(before),
               "result_hash": prep.session_hash(after) if shown else None, "model": {"sha256": "a" * 64}}
    return case, runtime, gateway


def configuration():
    return probe.pilot.LocalConfig(contract_version=actions.VERSION, endpoint="http://127.0.0.1:18087",
                                   model_id="test", model_path="/unused", model_sha256="a" * 64,
                                   chat_template_path="/unused", chat_template_sha256="b" * 64, runtime_build="test")


class MetricTests(unittest.TestCase):
    def test_all_denominators_scope_ambiguity_and_failures_remain_explicit(self):
        actions_by_id = {"edit": "correct_text", "failed": "split_caption", "keep": "no_action", "ambiguous": "no_action", "suppressed": "no_action"}
        cases = [{"id": key, "language": "en", "primary": {"action": action},
                  "applicability": {"full_action_preservation_admitted": key != "ambiguous"}}
                 for key, action in actions_by_id.items()]
        outcomes = [
            {"case_id": "edit", "status": "completed", "gateway": {"status": "proposal", "prediction": {"action": "correct_text"}}, "outcome_match": True},
            {"case_id": "failed", "status": "failed", "outcome_match": False, "evidence": {"response": {"usage": None}}},
            {"case_id": "keep", "status": "completed", "gateway": {"status": "no_action", "prediction": {"action": "no_action"}}, "outcome_match": True},
            {"case_id": "ambiguous", "status": "completed", "gateway": {"status": "proposal", "prediction": {"action": "remove_silence"}}, "outcome_match": False},
            {"case_id": "suppressed", "status": "completed", "gateway": {"status": "suppressed", "prediction": {"action": "correct_text"}}, "outcome_match": True},
        ]
        result = probe.summarize(cases, outcomes)
        self.assertEqual((result["useful"], result["displayed"], result["reference_edit_needed"], result["invalid"]), (1, 2, 2, 1))
        self.assertEqual((result["known_reference_precision"], result["known_reference_recall"], result["known_reference_f1"]), (.5, .5, .5))
        self.assertEqual(result["legacy_preservation"], {"numerator": 1, "denominator": 3})
        self.assertEqual(result["readmitted_full_action_preservation"], {"numerator": 1, "denominator": 2})
        self.assertEqual(result["display_coverage"], {"numerator": 2, "denominator": 5})
        self.assertEqual((result["cut_on_legacy_preserve"], result["cut_on_readmitted_preserve"]), (1, 0))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            probe.summarize(cases, outcomes[:-1])

    def test_raw_http_median_and_nearest_rank_p95_preserve_all_measured_rows(self):
        self.assertEqual(probe.quantiles([20, 1, 10, 2]), {"count": 4, "p50": 6., "p95": 20})
        with self.assertRaises(ValueError):
            probe.quantiles([float("nan")])


class SourceTests(unittest.TestCase):
    def test_source_result_reexecutes_and_rejects_target_or_media_corruption(self):
        case, runtime, gateway = fixture({"action": "correct_text", "target": [0], "text": "Check the light."})
        self.assertTrue(probe.source_result(case, runtime, gateway)["outcome_match"])
        wrong = copy.deepcopy(gateway)
        wrong["target_uuids"] = ["b"]
        with self.assertRaisesRegex(ValueError, "target mapping"):
            probe.source_result(case, runtime, wrong)
        wrong = copy.deepcopy(gateway)
        wrong["result_document"]["content"][1]["sourceStart"] += .1
        with self.assertRaisesRegex(ValueError, "source execution"):
            probe.source_result(case, runtime, wrong)

    def test_registration_requires_complete_pre_prediction_review_before_export_access(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "manifest.json").write_text("{}")
            review = {"status": "passed", "manifest_sha256": prep.sha(root / "manifest.json"), "cases_reviewed": 288, "model_predictions_consulted": False}
            with patch.object(probe, "BENCHMARK", root), patch.object(probe, "verify_preparation", return_value=({}, [], [], [])), patch.object(probe, "read", return_value=review):
                with self.assertRaisesRegex(ValueError, "pre-prediction all289"):
                    probe.register(root, {}, root / "new")
                review.update(cases_reviewed=289, model_predictions_consulted=True)
                with self.assertRaisesRegex(ValueError, "pre-prediction all289"):
                    probe.register(root, {}, root / "new")

    def test_rendered_prompt_has_no_fuzzy_normalization(self):
        adapter = probe.CapturingAdapter(configuration(), {"case": {"messages": [], "rendered_prompt": "exact"}})
        adapter.case_id = "case"
        with patch.object(probe.pilot.CapturingAdapter, "_json", return_value={"prompt": "exact"}):
            adapter._json(None, "POST", "/apply-template", json={"messages": []})
        with patch.object(probe.pilot.CapturingAdapter, "_json", return_value={"prompt": "exact\n"}):
            with self.assertRaisesRegex(ValueError, "rendered prompt"):
                adapter._json(None, "POST", "/apply-template", json={"messages": []})


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_feedback_document_identity_replays_through_actual_gateway(self):
        case, runtime, _ = fixture()
        doc, bundle = runtime["document"], case["bundle"]
        ledger = actions.PassiveFeedbackLedger("original-episode")
        ledger.propose("p", 0, doc, bundle, {"action": "correct_text", "target": [0], "text": "Check the light."})
        ledger.record("p", "shown", 1, origin="synthetic")
        ledger.record("p", "dismissed", 2, origin="synthetic")
        case, runtime, _ = fixture(feedback=ledger)

        class Adapter(Tokens):
            config = configuration()
            contract_version = actions.VERSION
            model = {"sha256": "a" * 64}

            def health(self):
                return {"status": "ready"}

            async def generate(self, messages, output_schema, timeout):
                return probe.pilot.Completion('{"action":"no_action"}', "stop")

        service = probe.pilot.CopilotService(Adapter())
        service._session(ledger.document_id).feedback = actions.PassiveFeedbackLedger.restore(ledger.dump())
        request = probe.pilot.PredictRequest(request={"document_id": ledger.document_id, "request_id": "r", "revision": 0, "interaction_epoch": 0},
                                            document=doc, focus_uuid="a", sequence=2, history=[], selection=[], trigger="explicit", deadline_ms=120000)
        gateway = await service.predict(request)
        self.assertTrue(probe.source_result(case, runtime, gateway)["outcome_match"])
        bad = copy.deepcopy(gateway)
        bad["request"]["document_id"] = case["id"]
        with self.assertRaisesRegex(ValueError, "feedback document identity"):
            probe.source_result(case, runtime, bad)

    async def test_successful_raw_api_evidence_is_strict_but_malformed_failures_stay_uncredited(self):
        case, runtime, gateway = fixture()
        config = configuration()
        token = {"messages": actions.canonical_messages(case["input"]), "primary": {"input_ids": [2, 3], "completion_start": 2}, "rendered_prompt": "exact"}
        adapter = probe.CapturingAdapter(config, {"case": token})
        adapter.case_id, adapter.last_rendered = "case", "exact"
        adapter.apply_chat_template = lambda *args, **kwargs: [2, 3]
        content = '{"action":"no_action"}'
        response = {"model": "test", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                    "__verbose": {"content": content, "tokens": [200, 106], "stop_type": "eos"}}
        client = httpx.AsyncClient(base_url=config.endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)))
        with patch.object(probe.pilot.httpx, "AsyncClient", return_value=client):
            await adapter.generate(token["messages"], actions.OUTPUT_SCHEMA, 1)
        row = {"case_id": "case", "status": "completed", "gateway": gateway, "evidence": adapter.observation, **probe.source_result(case, runtime, gateway)}
        probe.validate_saved_observation(case, runtime, token, row, content, config)
        wrong = copy.deepcopy(row)
        wrong["evidence"]["request"]["cache_prompt"] = True
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            probe.validate_saved_observation(case, runtime, token, wrong, content, config)
        failed = {"status": "failed", "outcome_match": False, "evidence": {"response": {"choices": [], "usage": None}}}
        original = json.dumps(failed, sort_keys=True)
        probe.validate_saved_observation(case, runtime, token, failed, None, config)
        self.assertEqual(json.dumps(failed, sort_keys=True), original)
        failed["outcome_match"] = True
        with self.assertRaisesRegex(ValueError, "received credit"):
            probe.validate_saved_observation(case, runtime, token, failed, None, config)


if __name__ == "__main__":
    unittest.main()
