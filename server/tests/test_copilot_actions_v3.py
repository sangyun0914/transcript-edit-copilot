import asyncio
import copy
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
from pydantic import ValidationError
from test_copilot import CORRECTION, FakeAdapter, document, prediction_request

from app.copilot import (
    CopilotService,
    FeedbackRequest,
    GatewayError,
    LocalConfig,
    LocalLlamaAdapter,
    contract,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdapterDeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("base.gguf", "adapter.gguf", "template.jinja"):
            (self.root / name).write_text(name)
        self.config = {
            "version": 2,
            "contract_version": "passive-nep-actions-v1",
            "endpoint": "http://127.0.0.1:18091",
            "model_id": "new-model",
            "model_path": str(self.root / "base.gguf"),
            "model_sha256": digest(self.root / "base.gguf"),
            "adapter_path": str(self.root / "adapter.gguf"),
            "adapter_sha256": digest(self.root / "adapter.gguf"),
            "chat_template_path": str(self.root / "template.jinja"),
            "chat_template_sha256": digest(self.root / "template.jinja"),
            "runtime_build": "test-build",
        }
        self.loaded = [{"id": 0, "path": self.config["adapter_path"], "scale": 1.0}]
        self.props = {
            "model_alias": self.config["model_id"],
            "model_path": self.config["model_path"],
            "build_info": self.config["runtime_build"],
            "chat_template": (self.root / "template.jinja").read_text(),
            "total_slots": 1,
            "default_generation_settings": {
                "n_ctx": 4096,
                "params": {"reasoning_format": "none", "reasoning_in_content": False},
            },
        }

    def make_adapter(self, format_name="json"):
        value = self.config | {"output_format": format_name}
        adapter = LocalLlamaAdapter(LocalConfig.model_validate(value))
        checks = []
        for language in ("en", "ko"):
            bundle = (
                contract("passive-nep-actions-v1")
                .PassiveContextBuilder(FakeAdapter())
                .build(document(language), "w1", as_of=0)
            )
            checks.append(
                {
                    "messages": adapter.request_messages(bundle["input"]),
                    "expected_native_token_ids": [2, 11, 13],
                }
            )
        receipt = {
            "version": 2,
            "contract_version": value["contract_version"],
            "token_source": "native_tokenizer",
            "model_sha256": value["model_sha256"],
            "adapter_sha256": value["adapter_sha256"],
            "adapter_scale": 1.0,
            "output_format": format_name,
            "chat_template_sha256": value["chat_template_sha256"],
            "runtime_build": value["runtime_build"],
            "checks": checks,
        }
        path = self.root / "parity.json"
        path.write_text(json.dumps(receipt))
        adapter.config.token_parity_path = str(path)
        adapter.config.token_parity_sha256 = digest(path)
        return adapter, checks

    def client(self):
        def handler(request):
            if request.url.path == "/props":
                value = self.props
            elif request.url.path == "/lora-adapters":
                value = self.loaded
            elif request.url.path == "/apply-template":
                value = {"prompt": "rendered"}
            elif request.url.path == "/tokenize":
                value = {"tokens": [2, 11, 13]}
            else:
                value = {"status": "ok"}
            return httpx.Response(200, json=value)

        return httpx.Client(
            base_url=self.config["endpoint"], transport=httpx.MockTransport(handler)
        )

    def test_new_mode_requires_explicit_version_paired_hashes_and_fixed_scale(self):
        for changes in (
            {"version": 1},
            {"adapter_sha256": None},
            {"adapter_scale": 0.5},
            {"contract_version": "passive-nep-v1"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                LocalConfig.model_validate(self.config | changes)
        self.assertEqual(LocalConfig.model_validate(self.config).adapter_scale, 1.0)

    def test_health_verifies_loaded_adapter_and_hash_bound_format_prefix(self):
        adapter, _ = self.make_adapter("rewrite")
        with patch.object(adapter, "_client", side_effect=self.client):
            self.assertEqual(adapter.health()["status"], "ready")
            self.assertEqual(adapter.model["adapter_sha256"], self.config["adapter_sha256"])
            self.assertEqual(adapter.model["output_format"], "rewrite")
            for loaded in (
                [],
                [{"id": 0, "path": self.config["model_path"], "scale": 1.0}],
                [{"id": 0, "path": self.config["adapter_path"], "scale": 0}],
                self.loaded * 2,
            ):
                self.loaded = loaded
                with self.assertRaisesRegex(ValueError, "Runtime did not load"):
                    adapter.health()
        (self.root / "adapter.gguf").write_bytes(b"changed adapter")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            adapter.health()

    def test_parity_cannot_be_reused_for_another_adapter_or_output_format(self):
        adapter, _ = self.make_adapter("rewrite")
        path = Path(adapter.config.token_parity_path)
        receipt = json.loads(path.read_text())
        receipt["output_format"] = "json"
        path.write_text(json.dumps(receipt))
        adapter.config.token_parity_sha256 = digest(path)
        with patch.object(adapter, "_client", side_effect=self.client):
            with self.assertRaisesRegex(ValueError, "another adapter or format"):
                adapter.health()

    def test_generate_binds_actual_loaded_adapter_and_rewrite_request(self):
        adapter, checks = self.make_adapter("rewrite")
        posted = []

        def handler(request):
            if request.url.path == "/lora-adapters":
                return httpx.Response(200, json=self.loaded)
            posted.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "new-model",
                    "choices": [
                        {
                            "message": {"content": '{"format_version":1,"scope":[],"after":[]}'},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        original = httpx.AsyncClient
        client = original(base_url=self.config["endpoint"], transport=httpx.MockTransport(handler))
        with patch("app.copilot.httpx.AsyncClient", return_value=client):
            asyncio.run(adapter.generate(checks[0]["messages"], {}, 5))
        self.assertEqual(posted[0]["lora"], [{"id": 0, "scale": 1.0}])
        self.assertEqual(posted[0]["messages"], checks[0]["messages"])
        self.assertIn("scope", posted[0]["response_format"]["json_schema"]["schema"]["properties"])
        self.loaded = []
        client = original(base_url=self.config["endpoint"], transport=httpx.MockTransport(handler))
        with (
            patch("app.copilot.httpx.AsyncClient", return_value=client),
            self.assertRaises(ValueError),
        ):
            asyncio.run(adapter.generate(checks[0]["messages"], {}, 5))
        self.assertEqual(len(posted), 1)

    def test_exact_native_count_reuses_one_client_and_preserves_parity_token_ids(self):
        adapter, checks = self.make_adapter()
        paths = []
        options = {
            "tokenize": True,
            "return_dict": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }

        def handler(request):
            paths.append(request.url.path)
            if request.url.path == "/v1/chat/completions/input_tokens":
                return httpx.Response(200, json={"input_tokens": 3})
            if request.url.path == "/apply-template":
                return httpx.Response(200, json={"prompt": "rendered"})
            return httpx.Response(200, json={"tokens": [2, 11, 13]})

        client = httpx.Client(
            base_url=self.config["endpoint"], transport=httpx.MockTransport(handler)
        )
        with patch.object(adapter, "_client", return_value=client) as create:
            with adapter.tokenization_session():
                for check in checks:
                    count = adapter.count_chat_template_tokens(check["messages"], **options)
                    ids = adapter.apply_chat_template(check["messages"], **options)
                    self.assertEqual(ids, check["expected_native_token_ids"])
                    self.assertEqual(count, len(ids))
            self.assertEqual(create.call_count, 1)
        self.assertTrue(client.is_closed)
        self.assertEqual(paths.count("/v1/chat/completions/input_tokens"), 2)

    def test_malformed_native_count_is_rejected_without_a_budget_or_fallback(self):
        options = {
            "tokenize": True,
            "return_dict": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        for payload in ([3], {"input_tokens": True}, {"input_tokens": -1}):
            adapter, checks = self.make_adapter()
            paths = []

            def handler(request):
                paths.append(request.url.path)
                return httpx.Response(200, json=payload)

            client = httpx.Client(
                base_url=self.config["endpoint"], transport=httpx.MockTransport(handler)
            )
            with (
                patch.object(adapter, "_client", return_value=client),
                adapter.tokenization_session(),
            ):
                with self.assertRaises(ValueError):
                    adapter.count_chat_template_tokens(checks[0]["messages"], **options)
            self.assertEqual(paths, ["/v1/chat/completions/input_tokens"])

    def test_native_count_falls_back_only_for_missing_endpoint(self):
        options = {
            "tokenize": True,
            "return_dict": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        for status in (404, 405, 503):
            adapter, checks = self.make_adapter()
            paths = []

            def handler(request):
                paths.append(request.url.path)
                if request.url.path == "/v1/chat/completions/input_tokens":
                    return httpx.Response(status, json={"error": "unsupported"})
                if request.url.path == "/apply-template":
                    return httpx.Response(200, json={"prompt": "rendered"})
                return httpx.Response(200, json={"tokens": [2, 11, 13]})

            client = httpx.Client(
                base_url=self.config["endpoint"], transport=httpx.MockTransport(handler)
            )
            with (
                patch.object(adapter, "_client", return_value=client),
                adapter.tokenization_session(),
            ):
                if status == 503:
                    with self.assertRaises(httpx.HTTPStatusError):
                        adapter.count_chat_template_tokens(checks[0]["messages"], **options)
                    self.assertEqual(paths, ["/v1/chat/completions/input_tokens"])
                else:
                    for _ in range(2):
                        self.assertEqual(
                            adapter.count_chat_template_tokens(checks[0]["messages"], **options), 3
                        )
                    self.assertEqual(paths.count("/v1/chat/completions/input_tokens"), 1)
                    self.assertEqual(paths.count("/tokenize"), 2)


class IncrementalHistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = contract("passive-nep-actions-v1")
        self.adapter = FakeAdapter()
        self.adapter.contract_version = "passive-nep-actions-v1"
        self.adapter.text = '{"action":"no_action"}'
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.service = CopilotService(self.adapter, self.temporary.name)

    def change(self, before, sequence):
        after = copy.deepcopy(before)
        after["content"][2].update(text=f"require {sequence}", conf=1)
        edit = {"action": "correct_text", "target": [1], "text": f"require {sequence}"}
        return after, self.module.committed_transition(before, after, edit, sequence)

    def request(self, document_, history, sequence, request_id, base=None):
        body = prediction_request(
            document=document_,
            history=history,
            sequence=sequence,
            history_transport="incremental",
            history_base=base,
        )
        body.request.request_id = request_id
        body.request.revision = sequence
        body.request.interaction_epoch = sequence
        return body

    async def test_160_manual_edits_and_210_requests_keep_exact_context_with_bounded_history(self):
        current, history, base = document(), [], None
        for index in range(1, 211):
            delta = []
            if index <= 160:
                current, transition = self.change(current, index)
                history.append(transition)
                delta = [transition]
            body = self.request(current, delta, len(history), f"long-{index}", base)
            result = await self.service.predict(body)
            self.assertEqual(result["status"], "no_action")
            self.assertEqual(
                result["history_tip"],
                {
                    "transaction_id": history[-1]["transaction_id"],
                    "sequence": len(history),
                    "count": len(history),
                },
            )
            base = result["history_tip"]["transaction_id"]
        reference = self.module.PassiveContextBuilder(self.adapter).build(
            current, "w1", as_of=160, history=history
        )
        self.assertEqual(self.adapter.calls[-1], self.module.canonical_messages(reference["input"]))
        state = self.service.sessions["doc"]._history_state
        self.assertEqual(state.limit, 512)
        self.assertEqual(len(state.entries), 160)
        self.assertEqual(len(self.service.used_requests), 210)
        self.assertIs(state.entries[-1].transition.before, state.entries[-2].transition.after)
        self.assertIs(
            state.entries[-1].transition.after["metadata"],
            state.entries[0].transition.before["metadata"],
        )
        with self.assertRaises(ValidationError):
            prediction_request(history=[{}] * 129)
        prediction_request(history_transport="incremental", history=[{}] * 512)
        with self.assertRaises(ValidationError):
            prediction_request(history_transport="incremental", history=[{}] * 513)

    async def test_backend_reset_and_lost_ack_require_explicit_resync_and_reject_forgery(self):
        current, transition = self.change(document(), 1)
        first = await self.service.predict(self.request(current, [transition], 1, "initial"))
        base = first["history_tip"]["transaction_id"]
        with self.assertRaises(GatewayError) as error:
            await self.service.predict(self.request(current, [], 1, "wrong-base", "lost-ack"))
        self.assertEqual(
            (error.exception.status_code, error.exception.code), (409, "history_resync_required")
        )
        synced = await self.service.predict(self.request(current, [transition], 1, "full-sync"))
        self.assertEqual(synced["history_tip"], first["history_tip"])
        restored = CopilotService(self.adapter, self.temporary.name)
        with self.assertRaises(GatewayError) as error:
            await restored.predict(self.request(current, [], 1, "after-reset", base))
        self.assertEqual(error.exception.code, "history_resync_required")
        synced = await restored.predict(self.request(current, [transition], 1, "reset-full-sync"))
        self.assertEqual(synced["history_tip"], first["history_tip"])
        forged = copy.deepcopy(transition)
        forged["prediction"]["text"] = "forged"
        with self.assertRaises(GatewayError) as error:
            await restored.predict(self.request(current, [forged], 1, "forged"))
        self.assertEqual(error.exception.code, "invalid")
        self.assertEqual(restored.sessions["doc"]._history_state.tip, first["history_tip"])

    async def test_cancelled_prediction_does_not_acknowledge_staged_history(self):
        current, transition = self.change(document(), 1)
        self.adapter.block = True
        body = self.request(current, [transition], 1, "cancelled")
        task = asyncio.create_task(self.service.predict(body))
        await self.adapter.started.wait()
        self.assertTrue(self.service.cancel(body.request)["cancelled"])
        with self.assertRaises(GatewayError) as error:
            await task
        self.assertEqual(error.exception.code, "cancelled")
        self.assertEqual(
            self.service.sessions["doc"]._history_state.tip,
            {"transaction_id": None, "sequence": 0, "count": 0},
        )
        self.adapter.block = False
        result = await self.service.predict(self.request(current, [transition], 1, "retry"))
        self.assertEqual(result["history_tip"]["count"], 1)

    async def test_compact_journal_preserves_durable_documents_and_feedback_idempotency(self):
        self.adapter.text = json.dumps(CORRECTION)
        before = document()
        result = await self.service.predict(self.request(before, [], 0, "proposal"))
        shown = FeedbackRequest(
            document_id="doc",
            proposal_id=result["proposal_id"],
            outcome="shown",
            sequence=1,
            event_id="shown",
            request=result["request"],
        )
        self.service.feedback(shown)
        transition = self.module.committed_transition(
            before, result["result_document"], CORRECTION, 2, origin="human"
        )
        accepted = FeedbackRequest(
            document_id="doc",
            proposal_id=result["proposal_id"],
            outcome="accepted",
            sequence=2,
            event_id="accepted",
            transition=transition,
        )
        receipt = self.service.feedback(accepted)
        self.assertEqual(self.service.feedback(accepted), receipt)
        with self.assertRaises(GatewayError) as error:
            self.service.feedback(accepted.model_copy(update={"reason": "timing"}))
        self.assertEqual(error.exception.code, "feedback_conflict")
        events, _ = self.service.evidence.read("doc")
        requested = next(event for event in events if event["kind"] == "prediction_requested")
        completed = next(event for event in events if event["kind"] == "prediction_completed")
        self.assertEqual(requested["input_request"]["document"], before)
        self.assertEqual(completed["response"]["result_document"], result["result_document"])
        self.assertEqual(events[-1]["feedback"]["transition"], transition)
        compact = json.dumps(self.service.journals["doc"])
        for field in ("input_request", "result_document", "ledger_record", '"transition"'):
            self.assertNotIn(field, compact)
        restored = CopilotService(self.adapter, self.temporary.name)
        self.assertEqual(restored.feedback(accepted), receipt)
        self.assertEqual(restored.journals["doc"], self.service.journals["doc"])


class SequencedSessionTests(unittest.IsolatedAsyncioTestCase):
    setUp = IncrementalHistoryTests.setUp
    change = IncrementalHistoryTests.change

    def request(self, document_, history, sequence, request_id=None, base=None):
        return prediction_request(
            request={
                "document_id": "doc",
                "request_id": request_id or f"{uuid4()}:{sequence}",
                "revision": sequence,
                "interaction_epoch": sequence,
            },
            document=document_,
            history=history,
            sequence=sequence,
            request_transport="sequenced-v1",
            history_transport="incremental",
            history_base=base,
        )

    async def test_more_than_512_proposals_and_feedback_keep_replay_and_restore(self):
        self.adapter.text = json.dumps(CORRECTION)
        original, first_feedback, first_receipt = document(), None, None
        for index in range(520):
            sequence = index * 3 + 1
            response = await self.service.predict(self.request(original, [], sequence))
            self.assertEqual(response["status"], "proposal")
            for offset, outcome in ((1, "shown"), (2, "expired")):
                feedback = FeedbackRequest(
                    document_id="doc",
                    proposal_id=response["proposal_id"],
                    outcome=outcome,
                    sequence=sequence + offset,
                    event_id=f"{index}-{outcome}",
                )
                receipt = self.service.feedback(feedback)
                if first_feedback is None:
                    first_feedback, first_receipt = feedback, receipt
        ledger = self.service.sessions["doc"].feedback
        self.assertEqual(len(ledger._ledger.events), 1040)
        self.assertEqual(len(ledger._records), 1)
        self.assertEqual(self.service.used_requests, set())
        self.assertNotIn('"content":', json.dumps(ledger.dump()))
        self.assertEqual(self.service.feedback(first_feedback), first_receipt)
        restored = CopilotService(self.adapter, self.temporary.name)
        self.assertEqual(restored.feedback(first_feedback), first_receipt)
        self.assertEqual(restored.sessions["doc"].feedback.dump(), ledger.dump())
        with self.assertRaises(GatewayError) as error:
            await restored.predict(self.request(original, [], 1558))
        self.assertEqual(error.exception.code, "request_replay")
        with self.assertRaises(GatewayError) as error:
            await restored.predict(prediction_request(sequence=1561))
        self.assertEqual(error.exception.code, "request_transport")
        expected = self.module.PassiveContextBuilder(self.adapter).build(
            original, "w1", as_of=1561, feedback=ledger
        )
        await restored.predict(self.request(original, [], 1561))
        self.assertEqual(self.adapter.calls[-1], self.module.canonical_messages(expected["input"]))
        records, _ = restored.evidence.read("doc")
        self.assertTrue(all(record["capture_mode"] == "product-compact-v1" for record in records))
        for record in records:
            if record["kind"] == "prediction_requested":
                self.assertNotIn("document", record["input_request"])
            if record["kind"] == "prediction_completed":
                self.assertNotIn("result_document", record["response"])
                self.assertNotIn("document", record["ledger_record"])

    async def test_more_than_512_commits_restore_exact_history_and_undo(self):
        current, history, base = document(), [], None
        for sequence in range(1, 521):
            current, transition = self.change(current, sequence)
            history.append(transition)
            result = await self.service.predict(
                self.request(current, [transition], sequence, base=base)
            )
            base = result["history_tip"]["transaction_id"]
        self.assertEqual(result["history_tip"]["count"], 520)
        self.assertTrue(
            all(
                not hasattr(entry.transition, "before")
                for entry in self.service.sessions["doc"]._history_state.entries
            )
        )
        restored = CopilotService(self.adapter, self.temporary.name)
        result = await restored.predict(
            self.request(
                current,
                history[-1:],
                521,
                base=history[-2]["transaction_id"],
            )
        )
        expected = self.module.PassiveContextBuilder(self.adapter).build(
            current, "w1", as_of=521, history=history
        )
        self.assertEqual(self.adapter.calls[-1], self.module.canonical_messages(expected["input"]))
        undo = self.module.committed_transition(
            current,
            history[-1]["before"],
            history[-1]["prediction"],
            522,
            undo_of=history[-1]["transaction_id"],
        )
        current = undo["after"]
        history.append(undo)
        await restored.predict(
            self.request(
                current,
                history[-2:],
                522,
                base=history[-3]["transaction_id"],
            )
        )
        expected = self.module.PassiveContextBuilder(self.adapter).build(
            current, "w1", as_of=522, history=history
        )
        self.assertEqual(self.adapter.calls[-1], self.module.canonical_messages(expected["input"]))

    async def test_failed_and_cancelled_sequences_are_consumed_without_history_commit(self):
        current, transition = self.change(document(), 1)
        self.adapter.block = True
        body = self.request(current, [transition], 1)
        task = asyncio.create_task(self.service.predict(body))
        await self.adapter.started.wait()
        self.service.cancel(body.request)
        with self.assertRaises(GatewayError) as error:
            await task
        self.assertEqual(error.exception.code, "cancelled")
        self.adapter.block = False
        restored = CopilotService(self.adapter, self.temporary.name)
        with self.assertRaises(GatewayError) as error:
            await restored.predict(body)
        self.assertEqual(error.exception.code, "request_replay")
        self.assertEqual(restored.sessions["doc"]._history_state.tip["count"], 0)
        response = await restored.predict(self.request(current, [transition], 2))
        self.assertEqual(response["history_tip"]["count"], 1)
        with self.assertRaises(GatewayError) as error:
            await restored.predict(self.request(current, [], 3, base="wrong"))
        self.assertEqual(error.exception.code, "history_resync_required")
        with self.assertRaises(GatewayError) as error:
            await restored.predict(self.request(current, [], 3))
        self.assertEqual(error.exception.code, "request_replay")

    async def test_product_acceptance_restore_undo_and_failed_write_are_transactional(self):
        from app.copilot_store import EvidenceError

        original = document()
        self.adapter.text = json.dumps(CORRECTION)
        result = await self.service.predict(self.request(original, [], 1))
        shown = FeedbackRequest(
            document_id="doc",
            proposal_id=result["proposal_id"],
            outcome="shown",
            sequence=2,
            event_id="shown",
            request=result["request"],
        )
        before = self.service.sessions["doc"].feedback.dump()
        with patch.object(self.service.evidence, "append", side_effect=EvidenceError("unwritable")):
            with self.assertRaises(GatewayError) as error:
                self.service.feedback(shown)
        self.assertEqual(error.exception.code, "feedback_storage")
        self.assertEqual(self.service.sessions["doc"].feedback.dump(), before)
        self.service = CopilotService(self.adapter, self.temporary.name)
        self.service.feedback(shown)
        accepted = self.module.committed_transition(
            original, result["result_document"], CORRECTION, 3, origin="human"
        )
        event = FeedbackRequest(
            document_id="doc",
            proposal_id=result["proposal_id"],
            outcome="accepted",
            sequence=3,
            event_id="accepted",
            transition=accepted,
        )
        receipt = self.service.feedback(event)
        self.adapter.text = '{"action":"no_action"}'
        accepted_result = await self.service.predict(
            self.request(result["result_document"], [accepted], 4)
        )
        self.service = CopilotService(self.adapter, self.temporary.name)
        self.assertEqual(self.service.feedback(event), receipt)
        with self.assertRaises(GatewayError) as error:
            self.service.feedback(event.model_copy(update={"reason": "timing"}))
        self.assertEqual(error.exception.code, "feedback_conflict")
        undo = self.module.committed_transition(
            result["result_document"],
            original,
            CORRECTION,
            5,
            origin="human",
            undo_of=accepted["transaction_id"],
        )
        self.service.feedback(
            FeedbackRequest(
                document_id="doc",
                proposal_id=result["proposal_id"],
                outcome="undone",
                sequence=5,
                event_id="undone",
                transition=undo,
            )
        )
        await self.service.predict(
            self.request(
                original,
                [undo],
                6,
                base=accepted_result["history_tip"]["transaction_id"],
            )
        )
        expected = self.module.PassiveContextBuilder(self.adapter).build(
            original,
            "w1",
            as_of=6,
            history=[accepted, undo],
            feedback=self.service.sessions["doc"].feedback,
        )
        self.assertEqual(self.adapter.calls[-1], self.module.canonical_messages(expected["input"]))
        records, _ = self.service.evidence.read("doc")
        committed = [event for event in records if event["kind"] == "feedback"]
        self.assertEqual(
            committed[1]["feedback"]["transition"]["before_hash"],
            self.module.session_hash(original),
        )
        self.assertNotIn('"content":', json.dumps(committed))

    async def test_cancel_after_worker_commit_rolls_back_history_before_retry_and_restart(self):
        current, transition = self.change(document(), 1)
        predict = self.service._predict

        async def late_cancel(body, active, started):
            response = await predict(body, active, started)
            active.cancelled.set()
            return response

        with patch.object(self.service, "_predict", late_cancel):
            with self.assertRaises(GatewayError) as error:
                await self.service.predict(self.request(current, [transition], 1))
        self.assertEqual(error.exception.code, "cancelled")
        self.assertEqual(self.service.sessions["doc"]._history_state.tip["count"], 0)
        self.assertFalse(self.service.sessions["doc"]._history_persisted)
        await self.service.predict(self.request(current, [transition], 2))
        restored = CopilotService(self.adapter, self.temporary.name)
        result = await restored.predict(
            self.request(current, [], 3, base=transition["transaction_id"])
        )
        self.assertEqual(result["history_tip"]["count"], 1)

    def test_sequenced_identity_and_per_request_history_limits_are_strict(self):
        for identity in ("plain", f"{uuid4()}:01", f"{uuid4()}:2"):
            with self.subTest(identity=identity), self.assertRaises(ValidationError):
                self.request(document(), [], 1, identity)
        with self.assertRaises(ValidationError):
            self.request(document(), [{}] * 513, 1)


class RewriteServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_context_reuses_tokens_but_changed_content_is_tokenized(self):
        adapter = FakeAdapter()
        adapter.contract_version = "passive-nep-actions-v1"
        adapter.count_chat_template_tokens = lambda *args, **kwargs: len(
            adapter.apply_chat_template(*args, **kwargs)
        )
        with tempfile.TemporaryDirectory() as temporary:
            service = CopilotService(adapter, temporary)
            with patch.object(
                adapter, "apply_chat_template", wraps=adapter.apply_chat_template
            ) as tokenize:
                first = await service.predict(prediction_request())
                body = prediction_request()
                body.request.request_id = "repeat"
                second = await service.predict(body)
                self.assertEqual(first["prediction"], second["prediction"])
                self.assertEqual(first["context"], second["context"])
                self.assertEqual(tokenize.call_count, 1)
                body.request.request_id = "changed"
                body.document.content[3]["text"] = "a changed review."
                await service.predict(body)
                self.assertEqual(tokenize.call_count, 2)

    async def test_cancel_during_validation_does_not_publish_or_mutate_feedback(self):
        module = contract("passive-nep-actions-v1")
        original = module.PassivePredictionSession.validate
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def blocked_validation(*args, **kwargs):
            entered.set()
            try:
                if not release.wait(2):
                    raise AssertionError("Validation was not released after cancellation.")
                return original(*args, **kwargs)
            finally:
                finished.set()

        adapter = FakeAdapter()
        adapter.contract_version = "passive-nep-actions-v1"
        with tempfile.TemporaryDirectory() as temporary:
            service = CopilotService(adapter, temporary)
            body = prediction_request()
            with patch.object(module.PassivePredictionSession, "validate", blocked_validation):
                task = asyncio.create_task(service.predict(body))
                try:
                    self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                    self.assertTrue(service.cancel(body.request)["cancelled"])
                    with self.assertRaises(GatewayError) as error:
                        await asyncio.wait_for(task, 0.5)
                    self.assertEqual(error.exception.code, "cancelled")
                finally:
                    release.set()
                    await asyncio.to_thread(finished.wait, 1)
                    await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(service.sessions["doc"].feedback.dump()["records"], [])
            events, _ = service.evidence.read("doc")
            self.assertNotIn("prediction_completed", [event["kind"] for event in events])

    async def test_raw_rewrite_is_recorded_and_same_typed_edit_is_applied(self):
        contract("passive-nep-actions-v1")
        import passive_formats_actions_v3 as formats

        adapter = FakeAdapter()
        adapter.contract_version = "passive-nep-actions-v1"
        adapter.output_format = "rewrite"
        adapter.model["adapter_sha256"] = "c" * 64
        adapter.request_messages = lambda snapshot: formats.messages(snapshot, "rewrite")
        bundle = formats.contract.PassiveContextBuilder(adapter).build(document(), "w1", as_of=0)
        adapter.text = formats.serialize(bundle["input"], CORRECTION, "rewrite")
        with tempfile.TemporaryDirectory() as temporary:
            service = CopilotService(adapter, temporary)
            result = await service.predict(prediction_request())
            self.assertEqual(result["prediction"], CORRECTION)
            self.assertEqual(result["status"], "proposal")
            events, _ = service.evidence.read("doc")
            prepared = next(value for value in events if value["kind"] == "prediction_prepared")
            generated = next(value for value in events if value["kind"] == "prediction_generated")
            self.assertEqual(prepared["messages"][0]["content"], formats.REWRITE_PROMPT)
            self.assertEqual(generated["completion"]["text"], adapter.text)
            adapter.text = '{"format_version":1,"scope":[999],"after":[]}'
            body = prediction_request()
            body.request.request_id = "invalid-rewrite"
            with self.assertRaises(GatewayError) as error:
                await service.predict(body)
            self.assertEqual(error.exception.code, "invalid")


if __name__ == "__main__":
    unittest.main()
