import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_small_actions_final as evaluation
import train_small_actions_final as training


class SmallEvaluationTests(unittest.TestCase):
    def admission_fixture(self, root, *, native=False):
        data = evaluation.data
        preparation, prepared, run, exported = (
            root / name for name in ("preparation", "evaluation", "run", "export")
        )
        for directory in (preparation, prepared, run, exported):
            directory.mkdir()
        config = {
            "model": "lfm25-1.2b",
            "base": data.PROFILES["lfm25-1.2b"],
            "training": {"rows": 1261, "epochs": 3},
        }
        data.write(preparation / "configuration.json", config)
        data.write(exported / "configuration.json", config)
        configuration_sha = data.sha(preparation / "configuration.json")
        request = {
            "case_id": "engineering-only",
            "format": "json",
            "input_sha256": "input",
            "messages_sha256": "messages",
            "prefix_ids_sha256": "native",
        }
        requests = prepared / "requests-json.jsonl"
        data.write_rows(requests, [request])
        data.write(
            prepared / "manifest.json",
            {
                "small_model_version": evaluation.VERSION,
                "final_test_bodies_read": False,
                "model_preparation": evaluation.grading.file_binding(
                    preparation / "configuration.json"
                ),
                "native_model": config["base"],
                "files_sha256": {requests.name: data.sha(requests)},
            },
        )
        data.write_rows(
            run / "observations.jsonl",
            [
                {
                    **request,
                    "status": "completed",
                    "text": '{"action":"no_action"}',
                    "finish_reason": "stop",
                    "native_token_parity": True,
                    "http_response_ms": 1,
                }
            ],
        )
        identity = {
            "kind": "sft",
            "configuration_sha256": configuration_sha,
            "adapter_sha256": "a" * 64,
            "reload_run_sha256": "r" * 64,
        }
        data.write(
            run / "run.json",
            {
                "version": data.VERSION,
                "status": "inference_complete"
                if native
                else "runtime_preflight_complete",
                "requested": 1,
                "failures": 0,
                "final_test_bodies_read": False,
                "observations_sha256": data.sha(run / "observations.jsonl"),
                "request_sha256": data.sha(requests),
                "configuration_sha256": configuration_sha,
                "kind": "sft",
                "model_identity": identity,
                "schema_constrained": not native,
            },
        )
        model = exported / "model-Q4_K_M.gguf"
        model.write_bytes(b"engineering-only dummy bytes; never loaded")
        data.write(
            exported / "run.json",
            {
                "version": data.VERSION,
                "status": "gguf_exported_runtime_validation_pending",
                "kind": "sft",
                "base": config["base"],
                "quantization": "Q4_K_M",
                "gguf_sha256": data.sha(model),
                "adapter_sha256": identity["adapter_sha256"],
                "training_configuration_sha256": configuration_sha,
            },
        )
        data.write(
            run / "runtime.json",
            {
                "base": config["base"],
                "alias": "fixture",
                "request_sha256": data.sha(requests),
                "props": {"model_path": str(model.resolve()), "model_alias": "fixture"},
            },
        )
        return prepared, run, exported, config, identity

    def test_native_prefix_binding_rejects_other_tokenizer_and_missing_rows(self):
        request = {
            "case_id": "a",
            "format": "json",
            "input_sha256": "input",
            "messages_sha256": "messages",
            "prefix_ids_sha256": "native",
        }
        evaluation.validate_observations([request], [copy.deepcopy(request)])
        with self.assertRaisesRegex(ValueError, "prefix_ids_sha256"):
            evaluation.validate_observations(
                [request], [{**request, "prefix_ids_sha256": "other"}]
            )
        with self.assertRaisesRegex(ValueError, "Every DEV"):
            evaluation.validate_observations([request], [])

    def test_macro_f1_requires_correct_effect_and_counts_failures(self):
        def row(reference, prediction, useful=False, preserved=False, abstained=False):
            return {
                "reference_action": reference,
                "predicted_action": prediction,
                "useful": useful,
                "preserved": preserved,
                "model_abstained": abstained,
            }

        records = [
            row("correct_text", "correct_text", useful=True),
            row("correct_text", "correct_text"),
            row("correct_text", None),
            row("delete_span", "correct_text", useful=True),
            row("no_action", "delete_span", preserved=True),
            row("no_action", "no_action", preserved=True, abstained=True),
        ]
        result = evaluation.semantic_macro_f1(records)
        self.assertEqual(
            result["actions"]["correct_text"], {"tp": 1, "fp": 1, "fn": 2, "f1": 0.4}
        )
        self.assertEqual(result["actions"]["delete_span"]["tp"], 1)
        self.assertEqual(result["actions"]["delete_span"]["fp"], 1)
        self.assertEqual(result["actions"]["no_action"]["tp"], 1)
        self.assertEqual(result["actions"]["no_action"]["fn"], 1)

    def test_gguf_admission_requires_the_frozen_training_configuration(self):
        data = evaluation.data
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, run, exported, config, _ = self.admission_fixture(root)
            self.assertEqual(
                evaluation.admit(prepared, run, root / "valid", exported=exported)[
                    "observed"
                ],
                1,
            )
            original_report = data.read(exported / "run.json")
            other = {**config, "training": {"rows": 16, "epochs": 1}}
            data.write(exported / "configuration.json", other)
            data.write(
                exported / "run.json",
                {
                    **original_report,
                    "training_configuration_sha256": data.sha(
                        exported / "configuration.json"
                    ),
                },
            )
            with self.assertRaises(ValueError):
                evaluation.admit(
                    prepared, run, root / "other-preparation", exported=exported
                )
            data.write(exported / "configuration.json", config)
            with self.assertRaises(ValueError):
                evaluation.admit(
                    prepared, run, root / "stale-export-report", exported=exported
                )

    def test_gguf_admission_rejects_observations_changed_after_the_run(self):
        data = evaluation.data
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, run, exported, _, _ = self.admission_fixture(root)
            self.assertEqual(
                evaluation.admit(prepared, run, root / "valid", exported=exported)[
                    "observed"
                ],
                1,
            )
            rows = data.rows(run / "observations.jsonl")
            rows[0]["text"] = '{"action":"correct_text","target":"t1","text":"changed"}'
            data.write_rows(run / "observations.jsonl", rows)
            with self.assertRaises(ValueError):
                evaluation.admit(
                    prepared, run, root / "mixed-observations", exported=exported
                )

    def test_native_admission_requires_the_actual_training_and_reload_identity(self):
        data = evaluation.data
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, run, _, _, identity = self.admission_fixture(root, native=True)
            trained, reloaded = root / "training", root / "reload"
            trained.mkdir()
            reloaded.mkdir()
            with patch.object(
                training, "model_identity", return_value=identity
            ) as recompute:
                result = evaluation.admit(
                    prepared, run, root / "valid", training=trained, reloaded=reloaded
                )
                self.assertEqual(result["observed"], 1)
                recompute.assert_called_once()
                with self.assertRaises(ValueError):
                    evaluation.admit(prepared, run, root / "unbound-training")
                info = data.read(run / "run.json")
                data.write(
                    run / "run.json",
                    {
                        **info,
                        "model_identity": {**identity, "adapter_sha256": "b" * 64},
                    },
                )
                with self.assertRaises(ValueError):
                    evaluation.admit(
                        prepared,
                        run,
                        root / "other-adapter",
                        training=trained,
                        reloaded=reloaded,
                    )
            wrong_configuration = {**identity, "configuration_sha256": "c" * 64}
            data.write(
                run / "run.json", {**info, "model_identity": wrong_configuration}
            )
            with patch.object(
                training, "model_identity", return_value=wrong_configuration
            ):
                with self.assertRaises(ValueError):
                    evaluation.admit(
                        prepared,
                        run,
                        root / "other-training",
                        training=trained,
                        reloaded=reloaded,
                    )


if __name__ == "__main__":
    unittest.main()
