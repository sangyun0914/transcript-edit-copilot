"""Portable request and failure-path fixtures; no model weights or held-out bodies."""

import copy
import json
import unittest

import infer_task_test_actions_v3 as native
import prepare_task_test_actions_v3 as sealed
import test_prepare_task_test_actions_v3 as fixtures


class PortableFinalTestTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SealedTaskTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.inputs = self.fixture.prepare()
        self.release = self.fixture.seal(self.inputs)
        self.portable = self.root / "portable"
        self.receipt = sealed.export_execution(
            self.fixture.freeze_path, self.release, "native_base", self.portable
        )

    def remote_failure(self):
        destination = self.root / "remote-failure"
        info = native.run(
            self.portable,
            self.receipt["sha256"],
            self.fixture.inference_config["path"],
            destination,
        )
        return destination, info

    def test_bundle_is_portable_label_free_and_scope_stays_final_test(self):
        self.assertEqual(
            {path.name for path in self.portable.iterdir()},
            {"execution.json", "requests.jsonl", "manifest.json"},
        )
        spec, requests = native.load_bundle(self.portable, self.receipt["sha256"])
        self.assertEqual(len(requests), 2)
        self.assertFalse("/Users/" in (self.portable / "execution.json").read_text())
        self.assertFalse("primary" in requests[0])
        altered = copy.deepcopy(requests)
        altered[0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "final_test"):
            native.validate_requests(altered, spec)
        altered = copy.deepcopy(requests)
        altered[0]["primary"] = {"action": "no_action"}
        with self.assertRaisesRegex(ValueError, "label-free"):
            native.validate_requests(altered, spec)
        altered = copy.deepcopy(requests)
        altered[0]["prefix_ids"][0] = 999
        with self.assertRaisesRegex(ValueError, "binding changed"):
            native.validate_requests(altered, spec)

    def test_real_preflight_failure_preserves_every_request_without_inventing_generation_time(
        self,
    ):
        destination, info = self.remote_failure()
        self.assertEqual(info["requested"], 2)
        self.assertEqual(info["observed"], 2)
        self.assertIsNone(info["actual_model_identity"])
        self.assertFalse(info["model_identity_verified"])
        self.assertTrue(info["completed_with_startup_or_interruption_failure"])
        observations = sealed.rows(destination / "observations.jsonl")
        self.assertTrue(
            all(
                row["status"] == "request_failure"
                and row["generation_ms"] is None
                and not row["generation_attempted"]
                for row in observations
            )
        )
        admitted = self.root / "admitted"
        sealed.admit_observations(
            self.fixture.freeze_path,
            self.release,
            self.portable,
            destination,
            admitted,
            tokenizer=self.fixture.tokenizer,
        )
        report = self.root / "graded-remote"
        sealed.grade(self.fixture.freeze_path, self.release, admitted, report)
        value = sealed.read_json(report / "report.json")
        self.assertEqual(value["summary"]["status_counts"], {"request_failure": 2})
        self.assertIsNone(value["summary"]["latency_ms"]["p50"])
        self.assertIn("predicted_action", value["by"])
        self.assertIn("predicted_action_language", value["by"])

    def test_native_model_configuration_identity_is_exact(self):
        spec, _ = native.load_bundle(self.portable, self.receipt["sha256"])
        config = sealed.read_json(self.fixture.inference_config["path"])
        verified = {
            "preparation_sha256": spec["preparation_manifest_sha256"],
            "environment": {"matched": True},
            "native_assets_verified": True,
        }
        native.verify_loaded_configuration(
            spec,
            config,
            self.fixture.inference_config["path"],
            verified,
            spec["model_identity"],
        )
        with self.assertRaisesRegex(ValueError, "Native model/configuration"):
            native.verify_loaded_configuration(
                spec,
                config,
                self.fixture.inference_config["path"],
                verified,
                {"kind": "another model"},
            )
        verified["environment"]["matched"] = False
        with self.assertRaisesRegex(ValueError, "Native model/configuration"):
            native.verify_loaded_configuration(
                spec,
                config,
                self.fixture.inference_config["path"],
                verified,
                spec["model_identity"],
            )

    def test_local_admission_rejects_missing_observations_even_after_new_transport_hashes(
        self,
    ):
        destination, _ = self.remote_failure()
        observations = sealed.rows(destination / "observations.jsonl")
        (destination / "observations.jsonl").write_text(
            sealed.packed(observations[0]) + "\n"
        )
        manifest = sealed.read_json(destination / "manifest.json")
        manifest["files_sha256"]["observations.jsonl"] = sealed.sha256(
            destination / "observations.jsonl"
        )
        (destination / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            sealed.admit_observations(
                self.fixture.freeze_path,
                self.release,
                self.portable,
                destination,
                self.root / "never-admitted",
                tokenizer=self.fixture.tokenizer,
            )


if __name__ == "__main__":
    unittest.main()
