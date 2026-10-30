import asyncio
import copy
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.copilot import (
    Completion,
    CopilotService,
    DisabledAdapter,
    FeedbackRequest,
    GatewayError,
    LocalConfig,
    LocalLlamaAdapter,
    PredictRequest,
    RequestIdentity,
    contract,
    create_copilot_router,
    inference_timings,
)
from app.copilot_store import EvidenceError, evidence_hash


def document(language="en"):
    words = (
        ("We", "requier", "another review.")
        if language == "en"
        else ("다시", "검토가", "필요합니다.")
    )
    return {
        "content": [
            {"type": "paragraph_start", "uuid": "p", "speaker": "s", "language": language},
            *[
                {
                    "type": "text",
                    "uuid": f"w{i}",
                    "text": word,
                    "length": 1.0,
                    "source": "media",
                    "sourceStart": float(i),
                    "conf": 0.9,
                }
                for i, word in enumerate(words)
            ],
            {"type": "paragraph_end", "uuid": "e"},
        ],
        "metadata": {"transcription": {"provider": "synthetic"}},
    }


CORRECTION = {"action": "correct_text", "target": [1], "text": "require"}


def prediction_request(**changes):
    body = {
        "request": {
            "document_id": "doc",
            "request_id": "r1",
            "revision": 0,
            "interaction_epoch": 0,
        },
        "document": document(),
        "focus_uuid": "w1",
        "selection": [],
        "sequence": 0,
        "history": [],
        "trigger": "explicit",
        "deadline_ms": 30000,
    }
    body.update(changes)
    return PredictRequest.model_validate(body)


class FakeAdapter:
    model = {"id": "test-only", "sha256": "a" * 64}

    def __init__(self):
        self.text = json.dumps(CORRECTION)
        self.finish_reason = "stop"
        self.calls = []
        self.block = False
        self.started = asyncio.Event()
        self.finished = False

    def health(self):
        return {"status": "ready", "model": self.model}

    def apply_chat_template(self, messages, **kwargs):
        return list(range(100))

    async def generate(self, messages, output_schema, timeout):
        self.calls.append(copy.deepcopy(messages))
        self.started.set()
        try:
            if self.block:
                await asyncio.Event().wait()
            return Completion(self.text, self.finish_reason)
        finally:
            self.finished = True


class IsolatedEvidence:
    def make_service(self, adapter):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return CopilotService(adapter, feedback_directory=directory.name)


class CopilotApiTests(IsolatedEvidence, unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.service = self.make_service(self.adapter)
        app = FastAPI()
        app.state.copilot = self.service

        def auth(request: Request):
            if request.headers.get("Authorization") != "Bearer test-only":
                raise HTTPException(401)

        app.include_router(create_copilot_router(auth))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer test-only"}

    def predict(self, body=None):
        return self.client.post(
            "/copilot/predict",
            headers=self.headers,
            json=(body or prediction_request()).model_dump(),
        )

    def feedback(self, proposal, outcome, sequence, **extra):
        return self.client.post(
            "/copilot/feedback",
            headers=self.headers,
            json={
                "document_id": "doc",
                "proposal_id": proposal,
                "outcome": outcome,
                "sequence": sequence,
                "origin": "human",
                **extra,
            },
        )

    def restart(self):
        self.service = CopilotService(self.adapter, self.service.evidence.directory)
        self.client.app.state.copilot = self.service

    def exported(self):
        response = self.client.get(
            "/copilot/feedback/export", params={"document_id": "doc"}, headers=self.headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        return [json.loads(line) for line in response.content.splitlines()]

    def test_all_routes_require_existing_authentication(self):
        self.assertEqual(self.client.get("/copilot/health").status_code, 401)
        for endpoint in ("sessions", "export"):
            self.assertEqual(self.client.get(f"/copilot/feedback/{endpoint}").status_code, 401)
        for endpoint in ("predict", "feedback", "cancel"):
            self.assertEqual(self.client.post(f"/copilot/{endpoint}", json={}).status_code, 401)

    def test_disabled_and_unqualified_models_do_not_predict(self):
        self.service.adapter = DisabledAdapter()
        health = self.client.get("/copilot/health", headers=self.headers).json()
        self.assertEqual(health["status"], "disabled")
        self.assertEqual(health["contract_version"], "passive-nep-v1")
        response = self.predict()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "unconfigured")
        self.assertEqual(self.adapter.calls, [])

    def test_source_bound_result_uses_shared_context_and_executor(self):
        original = document()
        response = self.predict()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "proposal")
        self.assertEqual(body["target_uuids"], ["w1"])
        self.assertEqual(body["document_hash"], contract().session_hash(original))
        self.assertEqual(body["result_hash"], contract().session_hash(body["result_document"]))
        self.assertEqual(body["result_document"]["content"][2]["text"], "require")
        for before, after in zip(original["content"], body["result_document"]["content"]):
            for key in ("source", "sourceStart", "length"):
                self.assertEqual(before.get(key), after.get(key))
        self.assertEqual(original, document())
        messages = self.adapter.calls[0]
        snapshot = contract().parse_snapshot(json.loads(messages[1]["content"]))
        self.assertEqual(
            snapshot.allowed_actions,
            ["no_action", "correct_text", "split_paragraph", "merge_paragraph"],
        )
        self.assertEqual(messages, contract().canonical_messages(snapshot))
        self.assertNotIn("task_brief", snapshot.model_dump())
        self.assertEqual(body["context"]["policy"]["input_tokens"], 3584)
        events = self.exported()
        self.assertEqual(
            [event["kind"] for event in events],
            [
                "prediction_requested",
                "prediction_prepared",
                "prediction_generated",
                "prediction_completed",
            ],
        )
        self.assertEqual(events[0]["input_request"], prediction_request().model_dump())
        self.assertEqual(events[0]["actor"], "unclassified")
        self.assertEqual(events[1]["messages"], messages)
        self.assertEqual(events[1]["messages_hash"], evidence_hash(messages))
        self.assertEqual(events[1]["bundle"]["document_hash"], body["document_hash"])
        self.assertEqual(events[1]["model"], self.adapter.model)
        self.assertEqual(events[2]["completion"]["text"], self.adapter.text)
        self.assertEqual(events[-1]["event_hash"], body["evidence"]["event_hash"])
        self.assertEqual(events[-1]["ledger_record"]["prediction"], CORRECTION)
        self.assertNotIn("Bearer", json.dumps(events))

    def test_duplicate_requests_and_regressing_identity_fail(self):
        self.assertEqual(self.predict().status_code, 200)
        self.assertEqual(self.predict().status_code, 409)
        request = prediction_request()
        request.request.request_id = "r2"
        request.request.revision = 4
        self.assertEqual(self.predict(request).status_code, 200)
        request.request.request_id = "r3"
        request.request.revision = 3
        self.assertEqual(self.predict(request).json()["detail"]["code"], "stale")

    def test_invalid_truncated_or_audio_changing_predictions_are_errors(self):
        for index, (text, finish) in enumerate(
            (
                ("not json", "stop"),
                (json.dumps(CORRECTION), "length"),
                ('{"action":"delete_span","target":[1]}', "stop"),
                ('{"action":"correct_text","target":[1],"text":"requier"}', "stop"),
                ('{"action":"correct_text","target":[99],"text":"word"}', "stop"),
            )
        ):
            self.adapter.text, self.adapter.finish_reason = text, finish
            body = prediction_request()
            body.request.request_id = f"invalid-{index}"
            with self.subTest(index=index):
                response = self.predict(body)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["detail"]["code"], "invalid")
        self.assertEqual(self.service.sessions["doc"].feedback.dump()["records"], [])

    def test_no_action_is_success_without_proposal_or_mutation(self):
        self.adapter.text = '{"action":"no_action"}'
        result = self.predict().json()
        self.assertEqual(result["status"], "no_action")
        self.assertIsNone(result["proposal_id"])
        self.assertIsNone(result["result_document"])
        self.assertEqual(self.service.sessions["doc"].feedback.dump()["records"], [])

    def test_diagnostic_invalid_completion_keeps_raw_evidence_without_credit(self):
        raw = {"request": {"messages": []}, "response": {"choices": ["invalid source target"]}}

        async def generate(*args):
            return Completion(
                '{"action":"correct_text","target":[999],"text":"word"}',
                "stop",
                {"runtime_cached_tokens": 11, "runtime_prompt_ms": 4.0},
                raw,
            )

        self.adapter.generate = generate
        response = self.predict()
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "invalid")
        self.assertEqual(detail["runtime_evidence"], raw)
        self.assertEqual(detail["timings"]["runtime_cached_tokens"], 11)
        self.assertGreaterEqual(detail["timings"]["total_ms"], 0)
        self.assertNotIn("prediction", detail)
        self.assertEqual(self.service.sessions["doc"].feedback.dump()["records"], [])

    def test_dismissal_suppresses_equivalent_proposal_and_requires_shown(self):
        proposal = self.predict().json()["proposal_id"]
        self.assertEqual(self.feedback(proposal, "dismissed", 1).status_code, 422)
        self.assertEqual(self.feedback(proposal, "shown", 0).status_code, 422)
        self.assertEqual(self.feedback(proposal, "shown", 1).status_code, 200)
        self.assertEqual(self.feedback(proposal, "dismissed", 2).status_code, 200)
        self.restart()
        old = prediction_request(sequence=0)
        old.request.request_id = "old-sequence"
        self.assertEqual(self.predict(old).json()["detail"]["code"], "stale")
        request = prediction_request(sequence=3)
        request.request.request_id = "r2"
        request.request.interaction_epoch = 1
        response = self.predict(request)
        self.assertEqual(response.json()["status"], "suppressed", response.text)
        self.assertIsNone(response.json()["result_document"])
        snapshot = contract().parse_snapshot(json.loads(self.adapter.calls[-1][1]["content"]))
        self.assertEqual(snapshot.recent_feedback[0].outcome, "dismissed")

    def test_acceptance_and_undo_require_actual_committed_transitions(self):
        result = self.predict().json()
        proposal = result["proposal_id"]
        self.assertEqual(self.feedback(proposal, "shown", 1).status_code, 200)
        self.assertEqual(self.feedback(proposal, "accepted", 2).status_code, 422)
        transition = contract().committed_transition(
            document(),
            result["result_document"],
            CORRECTION,
            2,
            origin="human",
            transaction_id="accept-1",
        )
        bad = copy.deepcopy(transition)
        bad["after"]["content"][2]["sourceStart"] = 100
        self.assertEqual(self.feedback(proposal, "accepted", 2, transition=bad).status_code, 422)
        self.assertEqual(
            self.feedback(proposal, "accepted", 2, transition=transition).status_code, 200
        )
        self.restart()
        request = prediction_request(document=result["result_document"], sequence=3)
        request.request.request_id = "missing-history"
        self.assertEqual(self.predict(request).status_code, 422)
        undo = contract().committed_transition(
            result["result_document"],
            document(),
            CORRECTION,
            4,
            origin="human",
            transaction_id="undo-1",
            undo_of="accept-1",
        )
        self.assertEqual(self.feedback(proposal, "undone", 4, transition=undo).status_code, 200)
        request = prediction_request(sequence=5, history=[transition, undo])
        request.request.request_id = "after-undo"
        self.assertEqual(self.predict(request).json()["status"], "suppressed")
        events = self.exported()
        accepted = next(
            e for e in events if e["kind"] == "feedback" and e["feedback"]["outcome"] == "accepted"
        )
        undone = next(
            e for e in events if e["kind"] == "feedback" and e["feedback"]["outcome"] == "undone"
        )
        self.assertEqual(accepted["feedback"]["transition"], transition)
        self.assertEqual(undone["feedback"]["transition"], undo)
        self.assertEqual(accepted["prepared_event_hash"], undone["prepared_event_hash"])
        self.assertNotEqual(accepted["process_id"], undone["process_id"])
        self.assertLess(accepted["seq"], undone["seq"])

    def test_feedback_retry_is_idempotent_and_expiry_is_not_a_negative_preference(self):
        body = prediction_request(origin="ai_computer_use", observed_at="2026-10-11T01:00:00Z")
        result = self.predict(body).json()
        args = {
            "event_id": "shown-1",
            "request": body.request.model_dump(),
            "observed_at": "2026-10-11T01:00:02Z",
            "revision": 0,
            "interaction_epoch": 0,
        }
        first = self.feedback(result["proposal_id"], "shown", 1, **args)
        self.assertTrue(first.json()["persisted"])
        self.restart()
        count = len(self.exported())
        retry = self.feedback(result["proposal_id"], "shown", 1, **args)
        self.assertEqual(retry.json(), first.json())
        self.assertEqual(len(self.exported()), count)
        conflict = self.feedback(result["proposal_id"], "expired", 2, **args)
        self.assertEqual(conflict.json()["detail"]["code"], "feedback_conflict")
        self.assertEqual(self.feedback(result["proposal_id"], "expired", 2).status_code, 200)
        self.restart()
        body.request.request_id = "after-expiry"
        body.sequence = 3
        self.assertEqual(self.predict(body).json()["status"], "proposal")
        snapshot = json.loads(self.adapter.calls[-1][1]["content"])
        self.assertEqual([item["outcome"] for item in snapshot["recent_feedback"]], ["expired"])
        exported = self.exported()
        self.assertEqual(exported[0]["actor"], "ai_computer_use")
        observed = [e for e in exported if e["kind"] == "feedback"]
        self.assertEqual([e["feedback"]["outcome"] for e in observed], ["shown", "expired"])
        self.assertEqual(observed[0]["feedback"]["observed_at"], args["observed_at"])
        self.assertTrue(
            all(e["preference_qualification"] == "unqualified_observation" for e in observed)
        )

    def test_failed_write_never_acknowledges_or_updates_feedback(self):
        result = self.predict().json()
        before = self.service.sessions["doc"].feedback.dump()
        with patch.object(self.service.evidence, "append", side_effect=EvidenceError("disk full")):
            response = self.feedback(result["proposal_id"], "shown", 1)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "feedback_storage")
        self.assertNotIn("recorded", response.json())
        self.assertEqual(self.service.sessions["doc"].feedback.dump(), before)
        self.assertFalse(any(e["kind"] == "feedback" for e in self.exported()))
        self.assertEqual(self.feedback(result["proposal_id"], "shown", 1).status_code, 503)
        self.restart()
        self.assertEqual(self.feedback(result["proposal_id"], "shown", 1).status_code, 200)

    def test_failed_fsync_is_not_success_even_if_a_complete_record_survives_restart(self):
        result = self.predict().json()
        before = self.service.sessions["doc"].feedback.dump()
        with patch("app.copilot_store.os.fsync", side_effect=OSError("sync failed")):
            response = self.feedback(result["proposal_id"], "shown", 1, event_id="retry-after-sync")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("recorded", response.json())
        self.assertEqual(self.service.sessions["doc"].feedback.dump(), before)
        self.restart()
        with patch("app.copilot_store.os.fsync", side_effect=OSError("still failing")):
            retry = self.feedback(result["proposal_id"], "shown", 1, event_id="retry-after-sync")
        self.assertEqual(retry.status_code, 503)
        self.restart()
        retry = self.feedback(result["proposal_id"], "shown", 1, event_id="retry-after-sync")
        self.assertTrue(retry.json()["persisted"])
        self.assertEqual(
            len([event for event in self.exported() if event["kind"] == "feedback"]), 1
        )

    def test_cached_writer_rechecks_external_changes_and_preserves_a_corrupt_prefix(self):
        result = self.predict().json()
        other = CopilotService(self.adapter, self.service.evidence.directory)
        other._session("doc")
        self.assertEqual(self.feedback(result["proposal_id"], "shown", 1).status_code, 200)
        with self.assertRaises(GatewayError) as error:
            other.feedback(
                FeedbackRequest(
                    document_id="doc",
                    proposal_id=result["proposal_id"],
                    outcome="shown",
                    sequence=1,
                )
            )
        self.assertEqual(error.exception.code, "feedback_storage")
        path = self.service.evidence.path("doc")
        changed = path.read_bytes().replace(b"requier", b"requiez", 1)
        path.write_bytes(changed)
        response = self.feedback(result["proposal_id"], "expired", 2)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(path.read_bytes(), changed)

    def test_restart_preserves_unfinished_request_and_rejects_corrupt_tail(self):
        self.predict()
        path = self.service.evidence.path("doc")
        prefix = b"".join(path.read_bytes().splitlines(keepends=True)[:2])
        path.write_bytes(prefix)
        self.restart()
        summary = self.client.get("/copilot/feedback/sessions", headers=self.headers).json()[
            "sessions"
        ][0]
        self.assertEqual(summary["unfinished_requests"], ["r1"])
        self.assertEqual(summary["proposal_outcomes"], {})
        self.assertEqual(self.predict().json()["detail"]["code"], "duplicate")
        self.assertEqual(self.service.sessions["doc"].feedback.dump()["records"], [])
        path.write_bytes(prefix + b'{"kind":')
        self.restart()
        response = self.client.get(
            "/copilot/feedback/export", params={"document_id": "doc"}, headers=self.headers
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.predict().json()["detail"]["code"], "feedback_storage")
        self.assertEqual(path.read_bytes(), prefix + b'{"kind":')

    def test_privileged_snapshot_fields_and_bad_selection_are_rejected(self):
        body = prediction_request().model_dump()
        body["task_brief"] = "Invent a better answer"
        self.assertEqual(
            self.client.post("/copilot/predict", headers=self.headers, json=body).status_code, 422
        )
        request = prediction_request(selection=["absent"])
        self.assertEqual(self.predict(request).status_code, 422)


class AllActionCopilotTests(IsolatedEvidence, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.adapter.contract_version = "passive-nep-actions-v1"
        self.service = self.make_service(self.adapter)
        self.module = contract(self.adapter.contract_version)

    def gap_document(self):
        result = document()
        result["content"][2] = {
            "type": "non_text",
            "uuid": "w1",
            "length": 1.0,
            "source": "media",
            "sourceStart": 1.0,
        }
        return result

    async def test_all_cut_actions_produce_source_bound_previews_and_round_trip_feedback(self):
        health = await self.service.health()
        self.assertEqual(health["contract_version"], "passive-nep-actions-v1")
        self.assertEqual(health["allowed_actions"], list(self.module.ALLOWED_ACTIONS))
        self.assertEqual(health["gated_actions"], [])
        for action in ("delete_span", "trim_silence", "remove_silence"):
            with self.subTest(action=action):
                service = self.make_service(self.adapter)
                before = document() if action == "delete_span" else self.gap_document()
                prediction = {"action": action, "target": [1]}
                if action == "trim_silence":
                    prediction["keep_ms"] = 400
                self.adapter.text = json.dumps(prediction)
                result = await service.predict(prediction_request(document=before))
                self.assertEqual(result["status"], "proposal")
                self.assertEqual(result["prediction"], prediction)
                self.assertEqual(result["target_uuids"], ["w1"])
                self.assertEqual(
                    before, document() if action == "delete_span" else self.gap_document()
                )
                saved = self.module.parse_snapshot(json.loads(self.adapter.calls[-1][1]["content"]))
                self.assertEqual(saved.items[1].boundary_status, "unverified")
                if action != "delete_span":
                    self.assertEqual(saved.items[1].audio_class, "unknown")

                def observe(outcome, sequence, transition=None):
                    return service.feedback(
                        FeedbackRequest(
                            document_id="doc",
                            proposal_id=result["proposal_id"],
                            outcome=outcome,
                            sequence=sequence,
                            origin="human",
                            transition=transition,
                        )
                    )

                self.assertTrue(observe("shown", 1)["persisted"])
                transition = self.module.committed_transition(
                    before,
                    result["result_document"],
                    prediction,
                    2,
                    origin="human",
                    transaction_id="accepted-cut",
                )
                self.assertTrue(observe("accepted", 2, transition)["persisted"])
                undo = self.module.committed_transition(
                    result["result_document"],
                    before,
                    prediction,
                    3,
                    origin="human",
                    transaction_id="undo-cut",
                    undo_of="accepted-cut",
                )
                self.assertTrue(observe("undone", 3, undo)["persisted"])
                service = CopilotService(self.adapter, service.evidence.directory)
                request = prediction_request(
                    document=before, sequence=4, history=[transition, undo]
                )
                request.request.request_id = "after-undo"
                self.assertEqual((await service.predict(request))["status"], "suppressed")

    async def test_invalid_cut_parameters_are_errors_not_abstentions(self):
        for index, prediction in enumerate(
            (
                {"action": "trim_silence", "target": [1], "keep_ms": 0},
                {"action": "trim_silence", "target": [1], "keep_ms": 1000},
                {"action": "trim_silence", "target": [1], "keep_ms": 1100},
                {"action": "remove_silence", "target": [0]},
                {"action": "delete_span", "target": [0, 2]},
            )
        ):
            with self.subTest(prediction=prediction):
                self.adapter.text = json.dumps(prediction)
                request = prediction_request(document=self.gap_document())
                request.request.request_id = f"bad-cut-{index}"
                with self.assertRaises(GatewayError) as error:
                    await self.service.predict(request)
                self.assertEqual(error.exception.code, "invalid")


class CopilotLifecycleTests(IsolatedEvidence, unittest.IsolatedAsyncioTestCase):
    async def test_synchronous_validation_cannot_publish_after_deadline(self):
        service = self.make_service(FakeAdapter())
        original = contract().PassivePredictionSession.validate

        def slow_validation(*args, **kwargs):
            time.sleep(0.12)
            return original(*args, **kwargs)

        with patch.object(contract().PassivePredictionSession, "validate", slow_validation):
            with self.assertRaises(GatewayError) as error:
                await service.predict(prediction_request(deadline_ms=100))
        self.assertEqual(error.exception.code, "deadline")
        self.assertEqual(service.sessions["doc"].feedback.dump()["records"], [])

    async def test_durable_write_crossing_deadline_is_failed_and_not_restored_as_a_proposal(self):
        service = self.make_service(FakeAdapter())
        append = service.evidence.append

        def slow_append(document_id, fields, expected_head):
            event = append(document_id, fields, expected_head)
            if fields["kind"] == "prediction_completed":
                time.sleep(0.12)
            return event

        with patch.object(service.evidence, "append", side_effect=slow_append):
            with self.assertRaises(GatewayError) as error:
                await service.predict(prediction_request(deadline_ms=100))
        self.assertEqual(error.exception.code, "deadline")
        restored = CopilotService(FakeAdapter(), service.evidence.directory)
        self.assertEqual(restored._session("doc").feedback.dump()["records"], [])
        summary = restored.evidence.sessions()[0]
        self.assertEqual(summary["failed_requests"], {"r1": "deadline"})
        self.assertEqual(set(summary["proposal_outcomes"].values()), {"request_failed"})

    async def test_one_active_request_cancel_identity_and_recovery(self):
        adapter = FakeAdapter()
        adapter.block = True
        service = self.make_service(adapter)
        body = prediction_request()
        task = asyncio.create_task(service.predict(body))
        await asyncio.wait_for(adapter.started.wait(), timeout=2)
        with self.assertRaises(GatewayError) as busy:
            await service.predict(prediction_request())
        self.assertEqual(busy.exception.code, "busy")
        other = RequestIdentity(**(body.request.model_dump() | {"request_id": "wrong"}))
        self.assertFalse(service.cancel(other)["cancelled"])
        self.assertTrue(service.cancel(body.request)["cancelled"])
        with self.assertRaises(GatewayError) as cancelled:
            await task
        self.assertEqual(cancelled.exception.code, "cancelled")
        self.assertTrue(adapter.finished)
        self.assertIsNone(service.active)
        adapter.block = False
        body.request.request_id = "after-cancel"
        self.assertEqual((await service.predict(body))["status"], "proposal")

    async def test_deadline_and_disconnect_cancel_without_recording_abstention(self):
        for disconnected in (False, True):
            adapter = FakeAdapter()
            adapter.block = True
            service = self.make_service(adapter)

            async def check_disconnect():
                return disconnected and adapter.started.is_set()

            with self.subTest(disconnected=disconnected), self.assertRaises(GatewayError) as error:
                await service.predict(prediction_request(deadline_ms=100), check_disconnect)
            self.assertEqual(error.exception.code, "cancelled" if disconnected else "deadline")
            self.assertTrue(adapter.finished)
            self.assertIsNone(service.active)
            self.assertEqual(service.sessions["doc"].feedback.dump()["records"], [])


class LocalAdapterTests(IsolatedEvidence, unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        model = self.root / "fake-model.gguf"
        model.write_bytes(b"test model bytes")
        template = self.root / "native.jinja"
        template.write_text("native template")
        self.config = {
            "endpoint": "http://127.0.0.1:18080",
            "model_id": "test-model",
            "model_path": str(model),
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "chat_template_path": str(template),
            "chat_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
            "runtime_build": "test-build",
        }
        self.props = {
            "model_alias": "test-model",
            "model_path": str(model),
            "build_info": "test-build",
            "chat_template": template.read_text(),
            "total_slots": 1,
            "default_generation_settings": {
                "n_ctx": 4096,
                "params": {"reasoning_format": "none", "reasoning_in_content": False},
            },
        }

    def test_configuration_disallows_network_and_redirect_targets(self):
        for endpoint in (
            "https://example.com",
            "http://example.com:80",
            "http://127.0.0.1:80/v1",
            "http://user:secret@127.0.0.1:80",
            "http://127.0.0.1:80?remote=1",
            "http://localhost:80",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValidationError):
                LocalConfig.model_validate(self.config | {"endpoint": endpoint})
        LocalConfig.model_validate(self.config)

    def test_prefix_reuse_is_explicit_and_actual_runtime_timings_are_retained(self):
        responses = []

        def handler(request):
            responses.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "test-model",
                    "choices": [
                        {"message": {"content": '{"action":"no_action"}'}, "finish_reason": "stop"}
                    ],
                    "timings": {
                        "prompt_ms": 12.5,
                        "predicted_ms": 3.0,
                        "prompt_n": 4,
                        "predicted_n": 8,
                        "cache_n": 100,
                    },
                    "usage": {"prompt_tokens_details": {"cached_tokens": 100}},
                },
            )

        original_client = httpx.AsyncClient
        for enabled in (False, True):
            config = self.config | (
                {"cache_prompt": True, "capture_runtime_evidence": True} if enabled else {}
            )
            adapter = LocalLlamaAdapter(LocalConfig.model_validate(config))
            client = original_client(
                base_url=self.config["endpoint"], transport=httpx.MockTransport(handler)
            )
            with patch("app.copilot.httpx.AsyncClient", return_value=client):
                completion = asyncio.run(adapter.generate([], {}, 5))
            self.assertIs(responses[-1]["cache_prompt"], enabled)
            self.assertEqual(completion.timings["runtime_cached_tokens"], 100)
            self.assertEqual(completion.timings["runtime_usage_cached_tokens"], 100)
            self.assertEqual(completion.timings["runtime_prompt_ms"], 12.5)
            self.assertEqual(completion.timings["runtime_decode_ms"], 3.0)
            self.assertEqual(completion.timings["runtime_timing_evidence"], "complete")
            self.assertGreaterEqual(completion.timings["runtime_http_ms"], 0)
            if enabled:
                self.assertIs(responses[-1]["verbose"], True)
                self.assertIs(responses[-1]["return_tokens"], True)
                self.assertEqual(completion.runtime_evidence["request"], responses[-1])
                self.assertEqual(completion.runtime_evidence["response"]["model"], "test-model")
                self.assertNotIn("Authorization", json.dumps(completion.runtime_evidence))
            else:
                self.assertNotIn("verbose", responses[-1])
                self.assertIsNone(completion.runtime_evidence)
        for invalid in ("true", 1, None):
            with self.assertRaises(ValidationError):
                LocalConfig.model_validate(self.config | {"cache_prompt": invalid})

    def test_missing_or_malformed_runtime_timings_never_become_zero_cache_hits(self):
        value = inference_timings(
            {"timings": {"prompt_ms": float("nan"), "cache_n": True}}, 1, True
        )
        self.assertEqual(value["runtime_timing_evidence"], "partial")
        self.assertNotIn("runtime_cached_tokens", value)
        self.assertNotIn("runtime_prompt_ms", value)
        value = inference_timings(
            {"timings": {"cache_n": 12}, "usage": {"prompt_tokens_details": {"cached_tokens": 11}}},
            1,
            True,
        )
        self.assertEqual(value["runtime_timing_evidence"], "inconsistent")
        self.assertEqual(value["runtime_cached_tokens"], 12)

    def test_completion_timing_evidence_reaches_gateway_response(self):
        adapter = FakeAdapter()

        async def generate(*args):
            return Completion(
                adapter.text, "stop", {"runtime_cached_tokens": 9, "runtime_prompt_ms": 4.0}
            )

        adapter.generate = generate
        result = asyncio.run(self.make_service(adapter).predict(prediction_request()))
        self.assertEqual(result["timings"]["runtime_cached_tokens"], 9)
        self.assertEqual(result["timings"]["runtime_prompt_ms"], 4.0)
        self.assertGreaterEqual(result["timings"]["total_ms"], 0)

    def mock_client(self):
        def handler(request):
            if request.url.path == "/props":
                return httpx.Response(200, json=self.props)
            if request.url.path == "/apply-template":
                return httpx.Response(200, json={"prompt": "native render"})
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            return httpx.Response(200, json={"status": "ok"})

        return httpx.Client(
            base_url=self.config["endpoint"], transport=httpx.MockTransport(handler)
        )

    def test_model_template_runtime_and_native_token_parity_are_required(self):
        adapter = LocalLlamaAdapter(LocalConfig.model_validate(self.config))
        with patch.object(adapter, "_client", side_effect=self.mock_client):
            self.assertEqual(adapter.health()["status"], "unqualified")
            self.props["total_slots"] = 2
            with self.assertRaisesRegex(ValueError, "slot"):
                adapter.health()
            self.props["total_slots"] = 1
            checks = []
            for language in ("en", "ko"):
                bundle = (
                    contract()
                    .PassiveContextBuilder(FakeAdapter())
                    .build(document(language), "w1", as_of=0)
                )
                checks.append(
                    {
                        "messages": contract().canonical_messages(bundle["input"]),
                        "expected_native_token_ids": [1, 2, 3],
                    }
                )
            receipt = {
                "version": 1,
                "model_sha256": self.config["model_sha256"],
                "chat_template_sha256": self.config["chat_template_sha256"],
                "runtime_build": "test-build",
                "token_source": "native_tokenizer",
                "checks": checks,
            }
            path = self.root / "parity.json"
            path.write_text(json.dumps(receipt))
            adapter.config.token_parity_path = str(path)
            adapter.config.token_parity_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(adapter.health()["status"], "ready")
            other = LocalLlamaAdapter(
                LocalConfig.model_validate(
                    {
                        **adapter.config.model_dump(),
                        "contract_version": "passive-nep-actions-v1",
                    }
                )
            )
            with patch.object(other, "_client", side_effect=self.mock_client):
                with self.assertRaisesRegex(ValueError, "another contract"):
                    other.health()
            receipt["checks"][1]["expected_native_token_ids"] = [2, 3]
            path.write_text(json.dumps(receipt))
            adapter.config.token_parity_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "tokens differ"):
                adapter.health()
        Path(self.config["model_path"]).write_bytes(b"different artifact")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            adapter.health()

    def test_reported_template_may_omit_exactly_one_final_newline_with_fresh_parity(self):
        template = Path(self.config["chat_template_path"])
        native = "native template\n\n"
        template.write_text(native)
        self.config["chat_template_sha256"] = hashlib.sha256(template.read_bytes()).hexdigest()
        self.config["contract_version"] = "passive-nep-actions-v1"
        module = contract(self.config["contract_version"])
        checks = []
        for language in ("en", "ko"):
            bundle = module.PassiveContextBuilder(FakeAdapter()).build(
                document(language), "w1", as_of=0
            )
            checks.append(
                {
                    "messages": module.canonical_messages(bundle["input"]),
                    "expected_native_token_ids": [1, 2, 3],
                }
            )
        receipt = {
            "version": 1,
            "contract_version": self.config["contract_version"],
            "model_sha256": self.config["model_sha256"],
            "chat_template_sha256": self.config["chat_template_sha256"],
            "runtime_build": self.config["runtime_build"],
            "token_source": "native_tokenizer",
            "checks": checks,
        }
        parity = self.root / "parity.json"
        parity.write_text(json.dumps(receipt))
        self.config.update(
            token_parity_path=str(parity),
            token_parity_sha256=hashlib.sha256(parity.read_bytes()).hexdigest(),
        )
        for reported in (native, native[:-1]):
            with self.subTest(reported=reported):
                self.props["chat_template"] = reported
                adapter = LocalLlamaAdapter(LocalConfig.model_validate(self.config))
                with (
                    patch.object(adapter, "_client", side_effect=self.mock_client),
                    patch.object(
                        adapter, "apply_chat_template", wraps=adapter.apply_chat_template
                    ) as tokenize,
                ):
                    self.assertEqual(adapter.health()["status"], "ready")
                    self.assertEqual(tokenize.call_count, 2)

        receipt["checks"][1]["expected_native_token_ids"] = [2, 3]
        parity.write_text(json.dumps(receipt))
        self.config["token_parity_sha256"] = hashlib.sha256(parity.read_bytes()).hexdigest()
        adapter = LocalLlamaAdapter(LocalConfig.model_validate(self.config))
        with patch.object(adapter, "_client", side_effect=self.mock_client):
            with self.assertRaisesRegex(ValueError, "tokens differ"):
                adapter.health()
        template.write_text(native[:-1])
        with self.assertRaisesRegex(ValueError, "hash differs"):
            adapter.health()

    def test_reported_template_rejects_other_content_or_whitespace_changes(self):
        template = Path(self.config["chat_template_path"])
        for native, reported in (
            ("native template\n\n", "native template"),
            ("native template\n", "native template\n\n"),
            ("native template\n", "native template \n"),
            ("native template\n", " native template\n"),
            ("native template\n", "changed template\n"),
            ("native template", "native templat"),
        ):
            with self.subTest(native=native, reported=reported):
                template.write_text(native)
                self.config["chat_template_sha256"] = hashlib.sha256(
                    template.read_bytes()
                ).hexdigest()
                self.props["chat_template"] = reported
                adapter = LocalLlamaAdapter(LocalConfig.model_validate(self.config))
                with patch.object(adapter, "_client", side_effect=self.mock_client):
                    with self.assertRaisesRegex(ValueError, "native template"):
                        adapter.health()


if __name__ == "__main__":
    unittest.main()
