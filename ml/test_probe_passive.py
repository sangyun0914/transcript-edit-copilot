import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import probe_passive
from passive_contract import canonical_messages
from test_passive_contract import builder, document


class PassiveProbeTests(unittest.TestCase):
    def test_diagnostic_cannot_change_schema_or_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, registry = root / "protocol.json", root / "registry.json"
            original = {"id": "model", "base_revision": "pinned", "tokenizer_id": "model"}
            registry.write_text(json.dumps({
                "candidates": [original], "tokenizers": {"registry_sha256": "pinned"},
                "runtime": {"revision": "pinned", "sha256": "pinned"},
            }))
            protocol.write_text(json.dumps({
                "fixed_conditions": {"schema_constrained": True, "request_timeout_seconds": 120},
                "stable_identity_bindings": {"candidates": [original],
                    "tokenizer_registry": {"registry_sha256": "pinned"},
                    "runtime_revision": "pinned", "runtime_launcher_sha256": "pinned"},
            }))
            execution = root / "execution.json"
            execution.write_text(json.dumps({"protocol_sha256": probe_passive.digest(protocol),
                "registry_sha256": probe_passive.digest(registry), "selected_candidate_ids": ["model"]}))
            self.assertEqual(probe_passive.validate_diagnostic_run(
                protocol, registry, "model", unconstrained=False, timeout=120
            ), execution)
            with self.assertRaisesRegex(ValueError, "schema or timeout"):
                probe_passive.validate_diagnostic_run(protocol, registry, "model", unconstrained=True, timeout=120)
            changed = json.loads(registry.read_text())
            changed["candidates"][0]["base_revision"] = "different"
            registry.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "candidate identity"):
                probe_passive.validate_diagnostic_run(protocol, registry, "model", unconstrained=False, timeout=120)

    def test_diagnostic_subset_requires_frozen_binding_and_registered_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("manifest.json", "screening-protocol.json", "sampling.json"):
                (root / name).write_text("{}")
            sampling = root / "sampling.json"
            protocol = {
                "kind": "native_sampling_diagnostic",
                "release_manifest_sha256": probe_passive.digest(root / "manifest.json"),
                "screening_protocol_sha256": probe_passive.digest(root / "screening-protocol.json"),
                "case_ids": ["b", "c"],
                "configs": {"model": [{"sha256": probe_passive.digest(sampling)}]},
            }
            path = root / "diagnostic.json"
            path.write_text(json.dumps(protocol))
            cases = [{"id": name} for name in ("a", "b", "c")]
            self.assertEqual(probe_passive.diagnostic_cases(cases, root, path, "model", sampling), cases[1:])
            for ids in (["c", "b"], ["b", "b"], ["missing"], []):
                path.write_text(json.dumps({**protocol, "case_ids": ids}))
                with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "ordered subset"):
                    probe_passive.diagnostic_cases(cases, root, path, "model", sampling)
            path.write_text(json.dumps(protocol))
            sampling.write_text('{"seed": 7}')
            with self.assertRaisesRegex(ValueError, "not registered"):
                probe_passive.diagnostic_cases(cases, root, path, "model", sampling)
            (root / "manifest.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                probe_passive.diagnostic_cases(cases, root, path, "model", sampling)

    def test_sampling_config_is_applied_without_changing_frozen_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, override = Path(directory) / "registry.json", Path(directory) / "override.json"
            registry.write_text(json.dumps({"sampling": {"seed": 42, "reasoning": "off"}}))
            override.write_text(json.dumps({"temperature": 0.7, "top_k": 20, "top_p": 0.8}))
            options = probe_passive.sampling_options(registry, override)
            body = probe_passive.request_body(self.case(), "qwen27", sampling=options)
            self.assertEqual((body["temperature"], body["top_k"], body["top_p"]), (0.7, 20, 0.8))
            self.assertEqual(body["max_tokens"], 512)
            for invalid in ({"max_tokens": 1024}, {"reasoning": "on"}, {"seed": True},
                            {"temperature": float("nan")}, {"messages": []}):
                override.write_text(json.dumps(invalid))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    probe_passive.sampling_options(registry, override)

    def test_output_cannot_write_into_frozen_release(self):
        for output in (probe_passive.DEFAULT_RELEASE / "accidental-run",
                       probe_passive.ROOT / "artifacts/datasets/nep-passive-v1/frozen/accidental-run"):
            args = SimpleNamespace(release=probe_passive.DEFAULT_RELEASE, output=output)
            with self.assertRaisesRegex(ValueError, "outside the immutable"):
                probe_passive.run(args)

    def case(self):
        bundle = builder().build(document(), "b", as_of=0)
        return {
            "id": "smoke", "input": bundle["input"],
            "expectedOutput": {"action": "correct_text", "target": [1], "text": "require"},
            "metadata": {"decision_rationale": "PRIVATE ANSWER"},
            "_execution": {"private": "PRIVATE SOURCE"},
        }

    def test_only_frozen_visible_messages_enter_request(self):
        case = self.case()
        body = probe_passive.request_body(case, "gemma26")
        self.assertEqual(body["messages"], canonical_messages(case["input"]))
        self.assertNotIn("PRIVATE", json.dumps(body))
        self.assertNotIn("expectedOutput", json.dumps(body))
        self.assertFalse(body["cache_prompt"])
        self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(body["max_tokens"], 512)
        self.assertNotIn("response_format", probe_passive.request_body(
            case, "gemma26", constrained=False
        ))

    def test_template_mismatch_and_extra_token_are_detected(self):
        body = probe_passive.request_body(self.case(), "gemma26")
        for actual, expected, mismatch in (([1, 3], [1, 2], 1), ([1, 2, 3], [1, 2], 2)):
            with self.subTest(actual=actual), patch.object(
                probe_passive, "server_json", side_effect=[{"prompt": "rendered"}, {"tokens": actual}]
            ) as call:
                result = probe_passive.template_check(None, body, expected)
                self.assertFalse(result["passed"])
                self.assertEqual(result["first_mismatch"], mismatch)
                self.assertTrue(call.call_args.args[2]["add_special"])
                self.assertTrue(call.call_args.args[2]["parse_special"])

    def test_template_failure_stops_before_generation_and_cleans_up(self):
        case = self.case()
        state = {"alive": True, "terminated": False, "completions": 0}

        class Process:
            pid = 123

            def poll(self):
                return None if state["alive"] else 0

            def terminate(self):
                state.update(alive=False, terminated=True)

            def wait(self, timeout=None):
                return 0

        def response(request):
            if request.url.path == "/v1/chat/completions":
                state["completions"] += 1
                raise AssertionError("Inference must not start after a template mismatch")
            payload = {"prompt": "mismatched"} if request.url.path == "/apply-template" else (
                {"tokens": [2]} if request.url.path == "/tokenize" else {}
            )
            return httpx.Response(200, json=payload)

        with tempfile.TemporaryDirectory(dir=probe_passive.ROOT) as temporary:
            root = Path(temporary)
            release = root / "release"
            release.mkdir()
            for name in ("manifest.json", "screening-protocol.json", "registry.json"):
                (release / name).write_text("{}")
            args = SimpleNamespace(
                release=release, output=root / "run", candidate="gemma26", stage="screen",
                registry=release / "registry.json", unconstrained=False, timeout=1,
                editor_state="test",
            )
            client = httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(response))
            with (
                patch.object(probe_passive, "load_screen", return_value=[case]),
                patch.object(probe_passive, "code_bindings", return_value={}),
                patch.object(probe_passive, "inspect_runtime", return_value=(
                    {"runtime": {}}, {}, {}, root / "model", root / "server", root / "template"
                )),
                patch.object(probe_passive, "frozen_prefixes", return_value={"smoke": [1]}),
                patch.object(probe_passive, "system_output", return_value="test"),
                patch.object(probe_passive.platform, "platform", return_value="test"),
                patch.object(probe_passive, "sample_rss"),
                patch.object(probe_passive.socket, "socket") as socket_factory,
                patch.object(probe_passive.subprocess, "Popen", return_value=Process()),
                patch.object(probe_passive.httpx, "Client", return_value=client),
                patch.object(probe_passive, "summarize", return_value={"missing": 1}),
            ):
                socket_factory.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 1234)
                with self.assertRaisesRegex(ValueError, "template/token mismatch"):
                    probe_passive.run(args)
            self.assertTrue(state["terminated"])
            self.assertEqual(state["completions"], 0)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["completed_requests"], 0)
            self.assertEqual(json.loads((args.output / "summary.json").read_text()), {"missing": 1})


if __name__ == "__main__":
    unittest.main()
