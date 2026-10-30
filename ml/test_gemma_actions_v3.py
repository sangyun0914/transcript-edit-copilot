import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import passive_formats_actions_v3 as formats
import prepare_gemma_actions_v3 as prep
import prepare_gemma_final_fallback as fallback
import train_gemma_actions_v3 as trainer
import train_gemma_actions_v3_dpo as dpo
from test_passive_actions import builder, document


class Tokenizer:
    eos_token_id, pad_token_id, bos_token_id = 1, 0, 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "return_dict": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return [
            2,
            *[
                byte + 256
                for byte in hashlib.sha256(formats.compact(messages).encode()).digest()
            ],
        ]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [byte + 256 for byte in text.encode()]


def cases():
    doc = document()
    context = builder()
    state = context.build(doc, "b", as_of=0)
    actions = [
        {"action": "no_action"},
        {"action": "correct_text", "target": [1], "text": "we"},
        {"action": "delete_span", "target": [0]},
        {"action": "trim_silence", "target": [3], "keep_ms": 1000},
        {"action": "remove_silence", "target": [3]},
        {"action": "split_paragraph", "target": [4]},
        {"action": "merge_paragraph", "target": [4]},
    ]
    result = []
    for index, action in enumerate(actions):
        current, bundle = doc, state
        if action["action"] == "merge_paragraph":
            current = formats.contract.apply_passive_edit(doc, state, actions[-2])
            bundle = context.build(
                current, "b", as_of=1, caption_profile={"max_merge_cue_ms": 10000}
            )
        result.append(
            {
                "id": f"case-{index}",
                "split": "train",
                "language": "en",
                "document": current,
                "runtime": {
                    "focus_uuid": "b",
                    "selection": [],
                    "sequence": 1 if action["action"] == "merge_paragraph" else 0,
                    "history": [],
                    "feedback_journal": None,
                    "caption_profile": bundle["input"]["caption_profile"],
                },
                "input": bundle["input"],
                "messages": formats.messages(bundle["input"]),
                "primary": action,
                "acceptable": [action],
                "source": {"split": "train", "id": "s", "source_group": "g"},
                "judgment": {"status": "preserve" if index == 0 else "edit_required"},
            }
        )
    return result


class SevenActionFormatTests(unittest.TestCase):
    def test_all_seven_round_trip_with_identical_visible_input(self):
        for case in cases():
            with self.subTest(action=case["primary"]["action"]):
                self.assertEqual(formats.messages(case["input"]), case["messages"])
                self.assertEqual(
                    formats.messages(case["input"], "rewrite")[1], case["messages"][1]
                )
                for name in formats.FORMATS:
                    raw = formats.serialize(case["input"], case["primary"], name)
                    self.assertEqual(
                        formats.parse(raw, case["input"], name).model_dump(),
                        case["primary"],
                    )
        self.assertIn('"keep_ms":positive_integer', formats.REWRITE_PROMPT)

    def test_scoped_rewrite_rejects_hidden_extra_operations_and_stale_output(self):
        case = cases()[3]
        value = json.loads(formats.serialize(case["input"], case["primary"], "rewrite"))
        for field, replacement in (
            ("text", "invented"),
            ("boundary_before", True),
            ("ids", [99]),
        ):
            bad = copy.deepcopy(value)
            bad["after"][0][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                formats.parse(json.dumps(bad), case["input"], "rewrite")
        with self.assertRaises(ValueError):
            formats.parse(
                json.dumps(value), case["input"], "rewrite", finish_reason="length"
            )
        bad = {"format_version": 1, "scope": [], "after": value["after"]}
        with self.assertRaises(ValueError):
            formats.parse(json.dumps(bad), case["input"], "rewrite")

    def test_native_masking_eos_and_terminal_injection(self):
        case = cases()[1]
        for name in formats.FORMATS:
            row = formats.encode(Tokenizer(), case["input"], case["primary"], name)
            start = row["completion_start"]
            self.assertEqual(row["labels"][:start], [-100] * start)
            self.assertEqual(row["labels"][start:], row["input_ids"][start:])
            self.assertEqual(row["input_ids"][-1], 1)
        tokenizer = Tokenizer()
        tokenizer.encode = lambda *args, **kwargs: [106]
        with self.assertRaises(ValueError):
            formats.encode(tokenizer, case["input"], case["primary"])


class PreparationTests(unittest.TestCase):
    def test_explicit_order_preparation_binds_repeats_and_rejects_resigned_changes(
        self,
    ):
        selected = cases()
        order = [case["id"] for case in selected] + ["case-0", "case-0"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "native"
            native.mkdir()
            for name in prep.TOKENIZER_FILES:
                (native / name).write_text("tokenizer fixture")
            prep.write(
                root / "pin.json",
                {
                    "repository": prep.gemma.REPOSITORY,
                    "revision": prep.gemma.REVISION,
                    "files": {
                        name: {"sha256": prep.digest(native / name)}
                        for name in prep.TOKENIZER_FILES
                    },
                },
            )
            prep.write(
                root / "template.json",
                {
                    "libraries": {},
                    "absent_libraries": [],
                    "model": {},
                    "base": {},
                    "tokenizer": {},
                },
            )
            prep.write_rows(root / "cases.jsonl", selected)
            order_path = root / "order.json"
            prep.write(
                order_path, {"version": prep.DRAW_ORDER_VERSION, "sample_order": order}
            )
            plan = {
                "seed": 3407,
                "passes": 1,
                "gradient_accumulation": 4,
                "learning_rate": 5e-5,
                "draw_order_mode": "explicit",
                "draw_order_sha256": prep.digest(order_path),
                "sample_order": order,
                "replacement": True,
            }
            prep.write(
                root / "protocol.json",
                {
                    "final_test_bodies_read": False,
                    "dataset": {
                        "train_cases_sha256": prep.digest(root / "cases.jsonl")
                    },
                    "training_plan": plan,
                    "eligibility_format": "json",
                },
            )
            kwargs = {
                "cases_path": root / "cases.jsonl",
                "native_pin": root / "pin.json",
                "native": native,
                "runtime_template": root / "template.json",
                "output": root / "bundle",
                "phase": "substantive",
                "protocol": root / "protocol.json",
                "eligibility_format": "json",
            }
            fake = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=lambda *args, **kwargs: Tokenizer()
                )
            )
            with mock.patch.dict("sys.modules", {"transformers": fake}):
                with self.assertRaisesRegex(ValueError, "supplied together"):
                    prep.prepare(**kwargs)
                with self.assertRaisesRegex(ValueError, "Learning rate differs"):
                    prep.prepare(**kwargs, draw_order=order_path)
                prep.prepare(**kwargs, draw_order=order_path, learning_rate=5e-5)
                defaults = {
                    **kwargs,
                    "phase": "engineering",
                    "protocol": None,
                    "output": root / "historical-default",
                }
                prep.prepare(**defaults)
            self.assertEqual(
                prep.read(root / "historical-default/config-json.json")["training"][
                    "learning_rate"
                ],
                1e-4,
            )
            bundle = prep.verify_bundle(root / "bundle")
            training = bundle["configs"]["json"]["training"]
            self.assertEqual(training["learning_rate"], 5e-5)
            self.assertEqual(training["sample_order"], order)
            self.assertEqual(
                (
                    training["draw_count"],
                    training["unique_cases"],
                    training["max_updates"],
                    training["final_update_rows"],
                ),
                (9, 7, 3, 1),
            )
            self.assertEqual(training["action_exposures"]["no_action"], 3)
            self.assertEqual(training["language_exposures"], {"en": 9})
            token_rows = bundle["tokens"]["json"]
            self.assertEqual(
                training["max_total_sequence_tokens"],
                sum(len(row["primary"]["input_ids"]) for row in token_rows)
                + 2 * len(token_rows[0]["primary"]["input_ids"]),
            )
            self.assertEqual(
                (root / "bundle/draw-order.json").read_bytes(), order_path.read_bytes()
            )
            config_path = root / "bundle/config-json.json"
            original_config = config_path.read_bytes()
            changed_config = prep.read(config_path)
            changed_config["training"]["learning_rate"] = 1e-4
            config_path.write_text(json.dumps(changed_config))
            (root / "bundle/manifest.json").unlink()
            prep.manifest(root / "bundle", selected_formats=("json",))
            with self.assertRaisesRegex(ValueError, "learning rate differs"):
                prep.verify_bundle(root / "bundle")
            config_path.write_bytes(original_config)
            changed = root / "bundle/draw-order.json"
            changed.write_text(
                json.dumps(
                    {"version": prep.DRAW_ORDER_VERSION, "sample_order": order[::-1]}
                )
            )
            (root / "bundle/manifest.json").unlink()
            prep.manifest(root / "bundle", selected_formats=("json",))
            with self.assertRaisesRegex(ValueError, "registered plan"):
                prep.verify_bundle(root / "bundle")

    def test_learning_rate_rejects_nonfinite_nonpositive_and_boolean_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            for value in (0, -1e-4, float("nan"), float("inf"), True):
                with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(
                        ValueError, "Learning rate must be finite and positive"
                    ),
                ):
                    prep.prepare(
                        None,
                        None,
                        None,
                        None,
                        Path(temporary) / "unused",
                        learning_rate=value,
                    )

    def test_explicit_order_rejects_missing_foreign_and_changed_registered_input(self):
        eligible = [{"id": "a"}, {"id": "b"}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "order.json"

            def save(order):
                path.write_text(
                    json.dumps(
                        {"version": prep.DRAW_ORDER_VERSION, "sample_order": order}
                    )
                )
                return {
                    "draw_order_mode": "explicit",
                    "draw_order_sha256": prep.digest(path),
                    "passes": 1,
                    "replacement": len(order) != len(set(order)),
                    "sample_order": order,
                }

            for order in (["a", "a"], ["a", "b", "foreign-or-budget-skipped"]):
                with (
                    self.subTest(order=order),
                    self.assertRaisesRegex(ValueError, "every eligible"),
                ):
                    prep.explicit_draw_order(path, eligible, save(order))
            plan = save(["a", "b", "a"])
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash/order/settings"):
                prep.explicit_draw_order(path, eligible, plan)
            plan = save(["a", "b", "a"])
            plan["sample_order"] = ["b", "a", "a"]
            with self.assertRaisesRegex(ValueError, "hash/order/settings"):
                prep.explicit_draw_order(path, eligible, plan)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "Missing registered"):
                prep.explicit_draw_order(path, eligible, plan)

    def test_variable_counts_exact_budgets_and_common_input_exclusion(self):
        admitted, tokens, excluded = prep.tokenize_cases(cases(), Tokenizer())
        self.assertEqual(len(admitted), 7)
        self.assertFalse(excluded)
        for name in formats.FORMATS:
            for case, record in zip(admitted, tokens[name], strict=True):
                prep.validate_record(case, record, name)
            order = [case["id"] for case in admitted] * 2
            budget = prep.budget(admitted, tokens[name], order, 4)
            self.assertEqual(
                (
                    budget["draw_count"],
                    budget["max_updates"],
                    budget["final_update_rows"],
                ),
                (14, 4, 2),
            )
            self.assertEqual(budget["unique_cases"], 7)
            self.assertEqual(
                budget["max_total_sequence_tokens"],
                2 * sum(len(row["primary"]["input_ids"]) for row in tokens[name]),
            )
        tokenizer = Tokenizer()
        old = tokenizer.apply_chat_template
        tokenizer.apply_chat_template = lambda messages, **kwargs: (
            [2] * 3585
            if messages[0]["content"] == formats.REWRITE_PROMPT
            else old(messages, **kwargs)
        )
        with self.assertRaisesRegex(ValueError, "No jointly eligible"):
            prep.tokenize_cases(cases(), tokenizer)
        admitted, tokens, excluded = prep.tokenize_cases(cases(), tokenizer, ("json",))
        self.assertEqual(len(admitted), 7)
        self.assertFalse(excluded)
        self.assertEqual(set(tokens), {"json"})

    def test_json_only_bundle_preserves_full_input_and_verifies_without_rewrite(self):
        admitted, tokens, _ = prep.tokenize_cases(cases(), Tokenizer(), ("json",))
        order = [case["id"] for case in admitted]
        settings = {
            "gradient_accumulation": 4,
            **prep.budget(admitted, tokens["json"], order, 4),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prep.write_rows(root / "train-cases.jsonl", admitted)
            prep.write_rows(root / "tokens-json.jsonl.gz", tokens["json"])
            prep.write(root / "config-json.json", {"training": settings})
            prep.manifest(root, selected_formats=("json",))
            bundle = prep.verify_bundle(root)
            self.assertEqual(bundle["cases"], cases())
            self.assertEqual(set(bundle["configs"]), {"json"})
            self.assertFalse((root / "tokens-rewrite.jsonl.gz").exists())
            changed = copy.deepcopy(admitted)
            changed[0]["input"]["items"][0]["text"] = "different"
            (root / "train-cases.jsonl").write_text("\n".join(map(json.dumps, changed)))
            with self.assertRaises(ValueError):
                prep.verify_bundle(root)

    def test_no_dev_unknown_or_privileged_input_and_detect_mask_corruption(self):
        case = cases()[0]
        for change in (
            lambda value: value.update(split="dev"),
            lambda value: value["judgment"].update(status="uncertain"),
            lambda value: value["input"].update(task_brief="delete all"),
        ):
            bad = copy.deepcopy(case)
            change(bad)
            with self.assertRaises(ValueError):
                prep.validate_case(bad)
        admitted, tokens, _ = prep.tokenize_cases([case], Tokenizer())
        tokens["json"][0]["primary"]["labels"][0] = 2
        with self.assertRaises(ValueError):
            prep.validate_record(admitted[0], tokens["json"][0], "json")

    def test_source_closure_contains_every_runtime_entrypoint(self):
        files = prep.source_closure(
            ["train_gemma_actions_v3.py", "train_gemma_actions_v3_dpo.py"]
        )
        for name in (
            "passive_actions.py",
            "passive_formats_actions_v3.py",
            "train_passive_gemma.py",
            "train_passive_dpo.py",
            "export_passive_gemma.py",
        ):
            self.assertIn(name, files)
        self.assertFalse(any(name.startswith("test_") for name in files))


class ConditionalFallbackTests(unittest.TestCase):
    def test_lr_recovery_cannot_substitute_the_original_run_or_change_its_data(self):
        original = {
            "model": "lfm25-1.2b",
            "source_cases_sha256": "frozen-train",
            "training": {"learning_rate": 5e-5, "sample_order": [2, 0, 1]},
        }
        recovery = copy.deepcopy(original)
        recovery["training"]["learning_rate"] = 2e-4
        fallback.validate_lr_recovery(original, recovery)
        with self.assertRaises(ValueError):
            fallback.validate_lr_recovery(original, original)
        for field, value in (
            ("source_cases_sha256", "other-train"),
            ("model", "gemma4-e2b"),
        ):
            changed = copy.deepcopy(recovery)
            changed[field] = value
            with self.assertRaises(ValueError):
                fallback.validate_lr_recovery(original, changed)
        changed = copy.deepcopy(recovery)
        changed["training"]["sample_order"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            fallback.validate_lr_recovery(original, changed)

    def test_epoch_mapping_preserves_both_small_orders_without_replacement(self):
        config = {
            "training": {
                "epochs": 3,
                "seed": 42,
                "gradient_accumulation": 4,
                "learning_rate": 5e-5,
                "micro_batch_size": 1,
                "packing": False,
                "truncation": False,
                "sample_order": [2, 0, 1, 1, 0, 2, 0, 2, 1],
            }
        }
        expected = ["c", "a", "b", "b", "a", "c", "a", "c", "b"]
        self.assertEqual(
            fallback.epoch_order(["a", "b", "c"], [config, config], 3), expected
        )
        for changed in ([2, 0, 0, 1, 0, 2, 0, 2, 1], [0, 2, 1, 1, 0, 2, 0, 2, 1]):
            bad = copy.deepcopy(config)
            bad["training"]["sample_order"] = changed
            with self.assertRaises(ValueError):
                fallback.epoch_order(["a", "b", "c"], [config, bad], 3)

    def test_runtime_uses_actual_observation_and_refuses_missing_or_insufficient_gpu(
        self,
    ):
        libraries = {key: "1.0" for key in trainer.smoke.LIBRARIES}
        libraries["python"] = "3.13.15"
        historical = {
            "libraries": libraries,
            "absent_libraries": ["torchao"],
            "model": {"gpu_name_contains": "old observed GPU"},
            "base": {},
            "tokenizer": {},
        }
        inspection = {
            "version": fallback.RUNTIME_VERSION,
            "status": "owned_runtime_observed_without_model_load",
            "owned_runtime_id": "synthetic-fixture-not-an-actual-runtime",
            "libraries": {**libraries, "python": "3.13.16"},
            "absent_libraries": ["torchao"],
            "gpu": {
                "name": "synthetic observed GPU",
                "total_memory": 96000000000,
                "bf16_supported": True,
            },
            "final_test_bodies_read": False,
            "weights_loaded": False,
            "training_started": False,
        }
        template = fallback.runtime_template(historical, inspection)
        self.assertEqual(template["libraries"]["python"], "3.13.16")
        self.assertEqual(
            template["model"]["gpu_name_contains"], "synthetic observed GPU"
        )
        self.assertEqual(historical["model"]["gpu_name_contains"], "old observed GPU")
        for change in (
            lambda value: value.update(owned_runtime_id=""),
            lambda value: value["gpu"].update(total_memory=24000000000),
            lambda value: value["libraries"].update(peft="changed-unreviewed-version"),
        ):
            bad = copy.deepcopy(inspection)
            change(bad)
            with self.assertRaises(ValueError):
                fallback.runtime_template(historical, bad)

    def test_fallback_requires_completed_native_failure_not_partial_quantized_or_pass(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = {"kind": "sft", "configuration_sha256": "a" * 64}
            prep.write(root / "run.json", {"status": "inference_complete"})
            prep.write(root / "mapping.json", {})
            report = {
                "model_identity": identity,
                "numerical_regime": "native_bf16_fp32_lora",
                "schema_constrained": False,
                "final_test": False,
                "excluded_before_inference": 0,
                "inference_run": fallback.binding(root / "run.json"),
            }
            prep.write(root / "report.json", report)
            summary = {
                "model_identity": identity,
                "report": fallback.binding(root / "report.json"),
                "cohort_mapping": fallback.binding(root / "mapping.json"),
                "overall": {"requests": 471},
                "cohorts": {
                    "canonical_dev322": {"metrics": {"requests": 322}},
                    "supplemental_dev149": {"metrics": {"requests": 149}},
                },
                "minimum_numeric_quality_gate": {
                    "passed": False,
                    "failures": ["precision"],
                },
            }
            path = root / "summary.json"
            prep.write(path, summary)
            with mock.patch(
                "evaluate_small_actions_final.summarize", return_value=summary
            ):
                self.assertEqual(
                    fallback.failed_development(
                        path, {"sha256": "a" * 64}, root / "checked.json"
                    ),
                    fallback.binding(path),
                )
            for kind in ("partial", "quantized", "passed"):
                changed_report, changed = copy.deepcopy(report), copy.deepcopy(summary)
                if kind == "partial":
                    changed["overall"]["requests"] = 470
                elif kind == "quantized":
                    changed_report["numerical_regime"] = "merged_Q4"
                else:
                    changed["minimum_numeric_quality_gate"]["passed"] = True
                (root / "report.json").write_text(json.dumps(changed_report))
                changed["report"] = fallback.binding(root / "report.json")
                path.write_text(json.dumps(changed))
                with (
                    mock.patch(
                        "evaluate_small_actions_final.summarize", return_value=changed
                    ),
                    self.assertRaises(ValueError),
                ):
                    fallback.failed_development(
                        path, {"sha256": "a" * 64}, root / "checked.json"
                    )


class PreferenceBindingTests(unittest.TestCase):
    def make_pair(self, index=1):
        case = cases()[index]
        candidate = {
            "case_id": case["id"],
            "candidate_id": "c",
            "input_sha256": prep.value_hash(
                formats.contract.wire_snapshot(case["input"])
            ),
            "split": "train",
            "parsed_action": {"action": "no_action"},
            "text": '{"action":"no_action"}',
        }
        pair = {
            "pair_id": "p",
            "case_id": case["id"],
            "input_sha256": candidate["input_sha256"],
            "candidate_id": "c",
            "candidate_run_sha256": "a" * 64,
            "chosen": case["primary"],
            "rejected": candidate["parsed_action"],
            "judgment": {
                "actor": "ai_generated",
                "status": "preferred",
                "rationale": "Observable repetition supports the minimal correction.",
            },
        }
        return case, candidate, pair

    def test_same_state_candidate_provenance_tie_and_unknown_fail_closed(self):
        case, candidate, pair = self.make_pair()
        ready = dpo.prepare_pair(
            pair, case, candidate, Tokenizer(), {"format": "json"}, "a" * 64
        )
        dpo.validate_pair_tokens(ready)
        for change in (
            lambda value: value.update(input_sha256="b" * 64),
            lambda value: value.update(candidate_run_sha256="b" * 64),
            lambda value: value.update(chosen=value["rejected"]),
            lambda value: value["judgment"].update(status="uncertain"),
        ):
            bad = copy.deepcopy(pair)
            change(bad)
            with self.assertRaises(ValueError):
                dpo.prepare_pair(
                    bad, case, candidate, Tokenizer(), {"format": "json"}, "a" * 64
                )

    def test_reference_cache_binds_anchor_and_both_completion_tokens(self):
        case, candidate, pair = self.make_pair()
        ready = dpo.prepare_pair(
            pair, case, candidate, Tokenizer(), {"format": "json"}, "a" * 64
        )
        config = {
            "anchor": {"anchor_adapter_sha256": "c" * 64},
            "pairs_sha256": "d" * 64,
            "format_version": formats.VERSION,
        }
        binding = dpo.cache_binding(config, [ready])
        for side in dpo.SIDES:
            changed = copy.deepcopy(ready)
            changed[side]["input_ids"][-2] += 1
            self.assertNotEqual(binding, dpo.cache_binding(config, [changed]))
        changed = copy.deepcopy(config)
        changed["anchor"]["anchor_adapter_sha256"] = "e" * 64
        self.assertNotEqual(binding, dpo.cache_binding(changed, [ready]))

    def test_preference_rejects_distinct_actions_with_equal_executed_result(self):
        case, candidate, pair = self.make_pair()
        alternative = {
            "action": "correct_text",
            "target": [1, 2],
            "text": "we can begin.",
        }
        candidate.update(
            parsed_action=alternative, text=json.dumps(alternative, indent=2)
        )
        pair["rejected"] = alternative
        with self.assertRaisesRegex(ValueError, "Equivalent executed outcomes"):
            dpo.prepare_pair(
                pair, case, candidate, Tokenizer(), {"format": "json"}, "a" * 64
            )

    def test_saved_cache_rejects_scalar_corruption_and_other_anchor(self):
        case, candidate, pair = self.make_pair()
        ready = dpo.prepare_pair(
            pair, case, candidate, Tokenizer(), {"format": "json"}, "a" * 64
        )
        config = {
            "anchor": {"anchor_adapter_sha256": "c" * 64},
            "pairs_sha256": "d" * 64,
            "format_version": formats.VERSION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prep.write(directory / "configuration.json", config)
            prep.write_rows(
                directory / "reference.jsonl",
                [
                    {
                        "pair_id": ready["pair_id"],
                        "pair_binding": dpo.pair_binding(ready),
                        "chosen_logprob": -1.25,
                        "rejected_logprob": -2.5,
                    }
                ],
            )
            prep.write(
                directory / "run.json",
                {
                    "status": "reference_cache_complete",
                    "configuration_sha256": prep.digest(
                        directory / "configuration.json"
                    ),
                    "binding": dpo.cache_binding(config, [ready]),
                },
            )
            trainer.manifest(directory, "reference_cache_complete", dpo.VERSION)
            self.assertEqual(
                dpo.verify_cache(directory, config, [ready])["p"]["chosen_logprob"],
                -1.25,
            )
            changed = copy.deepcopy(config)
            changed["anchor"]["anchor_adapter_sha256"] = "e" * 64
            with self.assertRaises(ValueError):
                dpo.verify_cache(directory, changed, [ready])
            (directory / "reference.jsonl").write_text(
                (directory / "reference.jsonl").read_text().replace("-1.25", "-1.26")
            )
            with self.assertRaises(ValueError):
                dpo.verify_cache(directory, config, [ready])


class GradientAndSerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            raise unittest.SkipTest(
                "Use the existing Gemma ML environment for real CPU gradients."
            ) from error
        cls.torch = torch

    def model(self):
        import test_train_passive_sft as fixtures

        fixture = fixtures.SmokeLossTests()
        fixture.torch = self.torch
        return fixture.model()

    def test_gemma_shared_objective_gradients_and_actual_partial_update_exposure(self):
        import test_train_passive_dpo as fixtures

        fixture = fixtures.DpoGradientTests()
        fixture.torch = self.torch
        pairs, model = fixture.pairs(3), self.model()
        references = fixture.references(pairs)
        optimizer = self.torch.optim.AdamW(model.parameters(), lr=1e-5)
        exposure, events, updates = dpo.objective.Exposure(), [], []
        before = [parameter.detach().clone() for parameter in model.parameters()]
        dpo.objective.train_pairs(
            model,
            pairs,
            references,
            optimizer,
            self.torch,
            fixture.settings(3),
            "cpu",
            exposure,
            events.append,
            updates.append,
        )
        self.assertEqual(exposure.summary()["phases"]["optimizer_committed"]["rows"], 3)
        self.assertEqual(updates[0]["microbatches"], 3)
        self.assertGreater(updates[0]["gradient_norm_before_clip"], 0)
        self.assertTrue(
            any(
                not self.torch.equal(old, new)
                for old, new in zip(before, model.parameters(), strict=True)
            )
        )
        fixture.test_three_forward_gradient_equals_ordinary_joint_graph_with_summed_eos_loss()

    def test_serialized_tensor_integrity_is_separate_from_reload_claim(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state = {
                "lora_A": self.torch.arange(6, dtype=self.torch.float32).reshape(2, 3)
            }
            save_file(state, str(directory / "adapter_model.safetensors"))
            expected = trainer.smoke.tensor_records(state, self.torch)
            self.assertEqual(
                trainer.verify_adapter_serialization(directory, expected, self.torch)[
                    "status"
                ],
                "passed",
            )
            state["lora_A"][0, 0] = 999
            save_file(state, str(directory / "adapter_model.safetensors"))
            with self.assertRaises(ValueError):
                trainer.verify_adapter_serialization(directory, expected, self.torch)
        self.assertNotEqual(trainer.TRAINED, trainer.RELOADED)

    def test_lora_export_verifier_rejects_wrong_values_and_scaling(self):
        import export_gemma_actions_v3 as exporter
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            state = {"left": self.torch.ones((2, 3)), "right": self.torch.zeros((4, 2))}
            save_file(state, str(directory / "adapter_model.safetensors"))
            gguf_path = directory / "adapter.gguf"
            gguf_path.write_bytes(b"mock GGUF reader fixture")
            pairs = {
                "native": {
                    "gguf_name": "mapped",
                    "tensors": {"A": "left", "B": "right"},
                }
            }
            fields = {"general.architecture": "gemma4", "adapter.lora.alpha": 16}
            tensors = [
                SimpleNamespace(
                    name="mapped.lora_a", tensor_type="F32", data=state["left"].numpy()
                ),
                SimpleNamespace(
                    name="mapped.lora_b", tensor_type="F32", data=state["right"].numpy()
                ),
            ]
            reader = SimpleNamespace(
                tensors=tensors,
                get_field=lambda name: SimpleNamespace(contents=lambda: fields[name]),
            )
            gguf = SimpleNamespace(
                GGUFReader=lambda _: reader,
                GGMLQuantizationType=SimpleNamespace(F32="F32"),
            )
            self.assertTrue(
                exporter.verify_adapter_tensors(directory, gguf_path, pairs, 16, gguf)[
                    "all_tensors_exact"
                ]
            )
            with self.assertRaises(ValueError):
                exporter.verify_adapter_tensors(directory, gguf_path, pairs, 32, gguf)
            tensors[1].data = tensors[1].data + 1
            with self.assertRaises(ValueError):
                exporter.verify_adapter_tensors(directory, gguf_path, pairs, 16, gguf)


if __name__ == "__main__":
    unittest.main()
