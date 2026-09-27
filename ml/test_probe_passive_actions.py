"""Evidence and cohort failure checks for the bounded all-action probe."""

import copy
import unittest
from unittest.mock import patch

import httpx

import probe_passive_actions as probe


def response(terminal=106):
    return {
        "model": "test",
        "choices": [
            {"message": {"content": '{"action":"no_action"}'}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        "__verbose": {
            "content": '```json\n{"action":"no_action"}\n```',
            "tokens": [200, terminal],
            "stop_type": "eos",
        },
    }


class EvidenceTests(unittest.TestCase):
    def test_raw_fences_and_tool_terminal_remain_distinct(self):
        evidence = probe.sampled_evidence(response(50))
        self.assertFalse(evidence["raw_format_valid"])
        self.assertTrue(evidence["api_format_valid"])
        self.assertEqual(evidence["terminal_token_id"], 50)
        self.assertFalse(evidence["sampled_content_equals_api_content"])

    def test_inconsistent_counts_tokens_and_terminals_fail(self):
        for mutate in (
            lambda r: r["usage"].update(completion_tokens=1),
            lambda r: r["__verbose"].update(tokens=[True, 106]),
            lambda r: r["__verbose"].update(tokens=[]),
            lambda r: r["__verbose"].update(stop_type="limit"),
            lambda r: r["choices"][0].update(finish_reason="length"),
        ):
            value = response()
            mutate(value)
            with self.assertRaises(ValueError):
                probe.sampled_evidence(value)

    def test_cohort_rejects_duplicate_cells_and_runtime_ids(self):
        cases = [
            {
                "id": f"{language}-{action}",
                "language": language,
                "primary": {"action": action},
                "split": "dev",
                "input": {},
            }
            for language in ("en", "ko")
            for action in probe.contract.ALLOWED_ACTIONS
        ]
        runtime = [{"id": c["id"], "split": "dev"} for c in cases]
        native = [{**r, "messages": []} for r in runtime]
        with (
            patch.object(probe.contract, "parse_snapshot", return_value={}),
            patch.object(probe.contract, "canonical_messages", return_value=[]),
        ):
            probe.validate_cohort(cases, runtime, native)
            with self.assertRaisesRegex(ValueError, "duplicate identities"):
                probe.validate_cohort(cases, [*runtime, runtime[0]], native)
            with self.assertRaisesRegex(ValueError, "cohort differs"):
                probe.validate_cohort(cases, runtime, list(reversed(native)))
            wrong = copy.deepcopy(cases)
            wrong[1]["primary"] = wrong[0]["primary"]
            with self.assertRaisesRegex(ValueError, "one development state"):
                probe.validate_cohort(wrong, runtime, native)

    def test_failed_response_is_not_preservation_or_abstention(self):
        cases = [
            {"id": "a", "language": "en", "primary": {"action": "no_action"}},
            {"id": "b", "language": "en", "primary": {"action": "delete_span"}},
        ]
        result = probe.summarize(
            cases,
            [
                {"case_id": "a", "status": "failed", "outcome_match": False},
                {
                    "case_id": "b",
                    "status": "completed",
                    "gateway": {"status": "no_action"},
                    "outcome_match": False,
                },
            ],
        )
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["preserved"], 0)
        self.assertEqual(result["recall"], 0)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            probe.summarize(cases, [])


class CapturingTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_raw_response_is_retained_before_failure(self):
        config = probe.LocalConfig(
            contract_version=probe.contract.VERSION,
            endpoint="http://127.0.0.1:18087",
            model_id="test",
            model_path="/unused",
            model_sha256="a" * 64,
            chat_template_path="/unused",
            chat_template_sha256="b" * 64,
            runtime_build="test",
        )
        expected = {
            "case": {
                "messages": [],
                "primary": {"input_ids": [3, 4], "completion_start": 1},
            }
        }
        adapter = probe.CapturingAdapter(config, expected)
        adapter.case_id = "case"
        adapter.apply_chat_template = lambda *args, **kwargs: [3]
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=["wrong response root"])
        )
        client = httpx.AsyncClient(base_url=config.endpoint, transport=transport)
        with patch.object(probe.httpx, "AsyncClient", return_value=client):
            with self.assertRaisesRegex(ValueError, "model identity"):
                await adapter.generate([], probe.contract.OUTPUT_SCHEMA, 1)
        self.assertEqual(adapter.observation["response"], ["wrong response root"])
        self.assertIn("response_body", adapter.observation)
        self.assertIn("integrity_failure", adapter.observation)
        self.assertTrue(adapter.observation["request"]["verbose"])
        self.assertTrue(adapter.observation["request"]["return_tokens"])


if __name__ == "__main__":
    unittest.main()
