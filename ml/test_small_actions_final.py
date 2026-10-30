import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import small_actions_final as small
import train_small_actions_final as training
import serve_small_actions_final as serving
from test_gemma_actions_v3 import cases


class Tokenizer:
    pad_token_id, bos_token_id = 0, 1

    def apply_chat_template(self, messages, **kwargs):
        prefix = [1, 20, 21]
        if kwargs["add_generation_prompt"]:
            return prefix
        return prefix + [ord(c) + 256 for c in messages[-1]["content"]] + [7, 266]

    def decode(self, tokens, **kwargs):
        return "".join(chr(token - 256) for token in tokens)


class Linear:
    in_features, out_features = 8, 12


class SmallPreparationTests(unittest.TestCase):
    def test_native_assistant_boundary_mask_and_terminal(self):
        text = '{"action":"no_action"}'
        row = small.encode_completion(
            Tokenizer(), [{"role": "user", "content": "Hi"}], text, [7]
        )
        self.assertEqual(row["completion_start"], 3)
        self.assertEqual(row["labels"][:3], [-100] * 3)
        self.assertEqual(row["labels"][3:], row["input_ids"][3:])
        self.assertEqual(row["input_ids"][-1], 7)
        self.assertNotIn(266, row["input_ids"][3:])
        small.validate_encoded(row, [7], 6144, 512)
        row["labels"][2] = row["input_ids"][2]
        with self.assertRaisesRegex(ValueError, "mask"):
            small.validate_encoded(row, [7], 6144, 512)

    def test_changed_template_boundary_and_special_token_injection_fail(self):
        tokenizer = Tokenizer()
        original = tokenizer.apply_chat_template
        tokenizer.apply_chat_template = lambda messages, **kwargs: (
            [999, *original(messages, **kwargs)]
            if not kwargs["add_generation_prompt"]
            else original(messages, **kwargs)
        )
        with self.assertRaisesRegex(ValueError, "prefix boundary"):
            small.encode_completion(tokenizer, [], "{}", [7])
        tokenizer = Tokenizer()
        tokenizer.apply_chat_template = lambda messages, **kwargs: (
            [1, 20, 21] if kwargs["add_generation_prompt"] else [1, 20, 21, 7, 266]
        )
        with self.assertRaisesRegex(ValueError, "target text"):
            small.encode_completion(tokenizer, [], "{}", [7])

    def test_over_budget_full_cohort_fails_without_excluding_rows(self):
        tokenizer = Tokenizer()
        selected = cases()[:2]
        native = tokenizer.apply_chat_template
        tokenizer.apply_chat_template = lambda messages, **kwargs: (
            [22] * 4096 + native(messages, **kwargs)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            small.write_rows(root / "cases.jsonl", selected)
            with mock.patch.object(small, "tokenizer_from", return_value=tokenizer):
                with self.assertRaisesRegex(ValueError, "Full cohort"):
                    small.prepare(
                        "lfm25-1.2b",
                        root / "cases.jsonl",
                        root,
                        root / "prepared",
                        context_tokens=4096,
                    )
            report = small.read(root / "prepared/preparation.json")
            self.assertEqual(report["requested_rows"], 2)
            self.assertEqual(report["retained_rows"], 2)
            self.assertEqual(report["excluded_rows"], 0)
            self.assertEqual(len(report["over_budget"]), 2)
            self.assertFalse((root / "prepared/configuration.json").exists())

    def test_train_only_saved_messages_and_accepted_reference_binding(self):
        case = cases()[0]
        small.validate_case(case)
        for changed in (
            {**case, "split": "dev"},
            {**case, "messages": [{"role": "user", "content": "changed"}]},
            {
                **case,
                "accepted_predictions": [{"action": "delete_span", "target": [0]}],
            },
        ):
            with self.assertRaises(ValueError):
                small.validate_case(changed)

    def test_lfm_full_hybrid_targets_and_attention_only_exclude_convolution(self):
        modules = [
            *(
                f"model.layers.2.self_attn.{name}"
                for name in ("q_proj", "k_proj", "v_proj", "out_proj")
            ),
            *(f"model.layers.0.conv.{name}" for name in ("in_proj", "out_proj")),
            *(f"model.layers.0.feed_forward.{name}" for name in ("w1", "w2", "w3")),
            "lm_head",
        ]
        model = SimpleNamespace(
            named_modules=lambda: [(name, Linear()) for name in modules]
        )
        full = small.select_targets(model, "lfm25-1.2b", "hybrid-ffn")
        attention = small.select_targets(model, "lfm25-1.2b", "attention")
        self.assertEqual(len(full), 9)
        self.assertEqual(len(attention), 4)
        self.assertIn("model.layers.0.conv.out_proj", full)
        self.assertNotIn("model.layers.0.conv.out_proj", attention)
        modules.remove("model.layers.2.self_attn.out_proj")
        with self.assertRaisesRegex(ValueError, "inventory"):
            small.select_targets(model, "lfm25-1.2b", "attention")

    def test_gemma_targets_do_not_include_vision_or_audio(self):
        suffixes = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        names = [f"model.language_model.layers.0.self_attn.{name}" for name in suffixes]
        names += [
            "model.vision_tower.layers.0.self_attn.q_proj",
            "model.audio_tower.layers.0.self_attn.q_proj",
        ]
        model = SimpleNamespace(
            named_modules=lambda: [(name, Linear()) for name in names]
        )
        selected = small.select_targets(model, "gemma4-e2b", "attention-ffn")
        self.assertEqual(len(selected), 7)
        self.assertTrue(all("language_model" in name for name in selected))


class ReloadAndRuntimeTests(unittest.TestCase):
    def adapter_fixture(self, root, model_name, targets):
        (root / "adapter").mkdir()
        config = {
            "model": model_name,
            "base": small.PROFILES[model_name],
            "training": {
                "rank": 16,
                "alpha": 16,
                "dropout": 0.0,
                "target_mode": small.PROFILES[model_name]["default_targets"],
            },
        }
        small.write(root / "configuration.json", config)
        (root / "adapter/adapter_model.safetensors").write_bytes(
            b"engineering fixture, never loaded"
        )
        small.write(
            root / "run.json",
            {
                "status": "training_complete_reload_pending",
                "configuration_sha256": small.sha(root / "configuration.json"),
                "adapter": {
                    "sha256": small.sha(root / "adapter/adapter_model.safetensors")
                },
            },
        )
        adapter = {
            "base_model_name_or_path": config["base"]["repository"],
            "revision": config["base"]["revision"],
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": targets,
        }
        small.write(root / "adapter/adapter_config.json", adapter)
        return config, adapter

    @unittest.skipUnless(
        importlib.util.find_spec("peft"), "Requires the ML training dependencies."
    )
    def test_minimized_lfm_targets_require_the_exact_resolved_module_inventory(self):
        names = [
            *(
                f"model.layers.2.self_attn.{name}"
                for name in ("q_proj", "k_proj", "v_proj", "out_proj")
            ),
            *(f"model.layers.0.conv.{name}" for name in ("in_proj", "out_proj")),
            *(f"model.layers.0.feed_forward.{name}" for name in ("w1", "w2", "w3")),
            "lm_head",
        ]
        model = SimpleNamespace(
            named_modules=lambda: [(name, Linear()) for name in names]
        )
        suffixes = [
            "in_proj",
            "k_proj",
            "out_proj",
            "q_proj",
            "v_proj",
            "w1",
            "w2",
            "w3",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, adapter = self.adapter_fixture(root, "lfm25-1.2b", suffixes)
            with (
                mock.patch(
                    "peft.PeftModel.from_pretrained", return_value="loaded"
                ) as loader,
                mock.patch.object(training, "verify_adapter_file"),
            ):
                self.assertEqual(
                    training.load_adapter(model, root, config, None), "loaded"
                )
                for changed in (suffixes[:-1], suffixes):
                    if changed == suffixes:
                        names.append("unrelated.q_proj")
                    small.write(
                        root / "adapter/adapter_config.json",
                        {**adapter, "target_modules": changed},
                    )
                    with self.assertRaisesRegex(ValueError, "execution settings"):
                        training.load_adapter(model, root, config, None)
                self.assertEqual(loader.call_count, 1)

    @unittest.skipUnless(
        importlib.util.find_spec("peft"), "Requires the ML training dependencies."
    )
    def test_minimized_gemma_targets_do_not_allow_vision_or_audio_matches(self):
        from peft.tuners.tuners_utils import _find_minimal_target_modules

        suffixes = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        names = [f"model.language_model.layers.0.self_attn.{name}" for name in suffixes]
        names += [
            "model.vision_tower.layers.0.self_attn.q_proj",
            "model.audio_tower.layers.0.self_attn.q_proj",
        ]
        model = SimpleNamespace(
            named_modules=lambda: [(name, Linear()) for name in names]
        )
        expected = set(small.select_targets(model, "gemma4-e2b", "attention-ffn"))
        minimized = sorted(
            _find_minimal_target_modules(expected, set(names) - expected)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, adapter = self.adapter_fixture(root, "gemma4-e2b", minimized)
            with (
                mock.patch(
                    "peft.PeftModel.from_pretrained", return_value="loaded"
                ) as loader,
                mock.patch.object(training, "verify_adapter_file"),
            ):
                self.assertEqual(
                    training.load_adapter(model, root, config, None), "loaded"
                )
                small.write(
                    root / "adapter/adapter_config.json",
                    {**adapter, "target_modules": suffixes},
                )
                with self.assertRaisesRegex(ValueError, "execution settings"):
                    training.load_adapter(model, root, config, None)
                self.assertEqual(loader.call_count, 1)

    def test_reload_requires_same_output_and_finite_matching_loss(self):
        before = [{"case_id": "a", "output_ids": [1, 7], "loss": 0.4}]
        training.verify_diagnostics(before, copy.deepcopy(before))
        for changed in (
            [{"case_id": "a", "output_ids": [2, 7], "loss": 0.4}],
            [{"case_id": "a", "output_ids": [1, 7], "loss": float("nan")}],
            [{"case_id": "a", "output_ids": [1, 7], "loss": 0.5}],
            [],
        ):
            with self.assertRaises(ValueError):
                training.verify_diagnostics(before, changed)

    def test_export_requires_exact_adapter_reload_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train/adapter").mkdir(parents=True)
            (root / "reload").mkdir()
            small.write(root / "train/configuration.json", {"model": "lfm"})
            (root / "train/adapter/adapter_model.safetensors").write_bytes(b"adapter")
            small.write(
                root / "reload/run.json",
                {
                    "status": "independent_reload_passed",
                    "training_configuration_sha256": small.sha(
                        root / "train/configuration.json"
                    ),
                    "adapter_sha256": small.sha(
                        root / "train/adapter/adapter_model.safetensors"
                    ),
                },
            )
            training.verified_reload(root / "train", root / "reload")
            (root / "train/adapter/adapter_model.safetensors").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "same independently"):
                training.verified_reload(root / "train", root / "reload")

    def test_export_rejects_replaced_training_configuration_before_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged = root / "merged"
            merged.mkdir()
            config = {
                "base": small.PROFILES["lfm25-1.2b"],
                "training": {"rows": 16, "epochs": 1},
            }
            small.write(merged / "configuration.json", config)
            small.write(
                merged / "run.json",
                {
                    "kind": "sft",
                    "status": "merged_native_bf16_local_validation_pending",
                    "training_configuration_sha256": small.sha(
                        merged / "configuration.json"
                    ),
                    "files_sha256": {},
                },
            )
            small.write(
                merged / "configuration.json",
                {**config, "training": {"rows": 1261, "epochs": 3}},
            )
            output = root / "export"
            with mock.patch.object(serving.subprocess, "run") as converter:
                with self.assertRaisesRegex(
                    ValueError, "training configuration changed"
                ):
                    serving.export(
                        merged, root / "unused-llama", root / "unused-quantizer", output
                    )
                converter.assert_not_called()
            self.assertFalse(output.exists())

    def test_runtime_requests_are_label_free_and_do_not_accept_heldout_split(self):
        case = cases()[0]
        request = {
            "case_id": case["id"],
            "split": "dev",
            "input": case["input"],
            "messages": case["messages"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "requests.jsonl"
            small.write_rows(path, [request])
            self.assertEqual(serving.request_rows(path), [request])
            for update in ({"primary": case["primary"]}, {"split": "test"}):
                small.write_rows(path, [{**request, **update}])
                with self.assertRaises(ValueError):
                    serving.request_rows(path)

    def test_train_inference_binds_saved_training_inputs_and_excludes_labels(self):
        case = cases()[0]
        request = {
            "case_id": case["id"],
            "split": "train",
            "input": case["input"],
            "messages": case["messages"],
        }
        prepared = [
            {
                "case_id": case["id"],
                "messages": case["messages"],
                "semantic_input_sha256": small.value_hash(
                    small.formats.contract.wire_snapshot(case["input"])
                ),
            }
        ]
        training.validate_requests([request], prepared)
        for update in (
            {"case_id": "not-trained"},
            {"split": "test"},
            {"primary": case["primary"]},
        ):
            with self.assertRaises(ValueError):
                training.validate_requests([{**request, **update}], prepared)
        prepared[0]["semantic_input_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "prepared training cohort"):
            training.validate_requests([request], prepared)

    def test_runtime_command_is_explicit_non_thinking_and_single_slot(self):
        command = serving.server_command(
            "/tmp/server",
            "/tmp/model.gguf",
            "/tmp/chat.jinja",
            alias="lfm",
            port=18092,
            context_tokens=6144,
        )
        self.assertEqual(command[command.index("--reasoning") + 1], "off")
        self.assertEqual(command[command.index("-np") + 1], "1")
        self.assertEqual(command[command.index("-c") + 1], "6144")
        schema = serving.runtime_schema(small.formats.contract.OUTPUT_SCHEMA)
        self.assertIsInstance(schema, dict)
        self.assertIn("oneOf", schema)


class NativeTokenizerTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("transformers"), "Requires the ML training dependencies."
    )
    def test_both_pinned_tokenizers_mask_all_seven_actions_in_english_and_korean(self):
        assets = (
            Path(__file__).resolve().parents[1]
            / "artifacts/model-preparation/small-final-v1"
        )
        if not all((assets / name / "assets.json").exists() for name in small.PROFILES):
            self.skipTest(
                "Pinned tokenizer-only assets are not installed; acquire explicitly before integration checks."
            )
        for model_name, profile in small.PROFILES.items():
            tokenizer = small.tokenizer_from(assets / model_name, model_name)
            for case in cases():
                for language in ("en", "ko"):
                    action = copy.deepcopy(case["primary"])
                    if action["action"] == "correct_text" and language == "ko":
                        action["text"] = "정확한 표현"
                    text = small.compact(action)
                    messages = (
                        case["messages"]
                        if language == "en"
                        else [
                            case["messages"][0],
                            {
                                "role": "user",
                                "content": "한국어 발화의 자막을 편집합니다.",
                            },
                        ]
                    )
                    with self.subTest(
                        model=model_name, action=action["action"], language=language
                    ):
                        row = small.encode_completion(
                            tokenizer, messages, text, profile["terminal_ids"]
                        )
                        small.validate_encoded(row, profile["terminal_ids"], 6144, 512)
                        self.assertEqual(
                            tokenizer.decode(
                                row["input_ids"][row["completion_start"] : -1]
                            ),
                            text,
                        )


if __name__ == "__main__":
    unittest.main()
