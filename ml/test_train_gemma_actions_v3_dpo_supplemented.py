"""Synthetic supplemented DPO regression checks, intended for remote execution."""

import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_gemma_actions_v3_dpo_supplemented as trainer
from test_gemma_actions_v3 import Tokenizer, cases
from test_generate_gemma_actions_v3_supplement import input_bundle


def preference_fixture():
    selected = [cases()[1], cases()[6]]
    registry, raw = {}, []
    for index, case in enumerate(selected):
        sha = ("a" if index == 0 else "b") * 64
        candidate = {
            "case_id": case["id"],
            "candidate_id": "same-id-in-two-runs",
            "split": "train",
            "format": "json",
            "input_sha256": trainer.prep.value_hash(
                trainer.prep.formats.contract.wire_snapshot(case["input"])
            ),
            "text": '{"action":"no_action"}',
            "parsed_action": {"action": "no_action"},
        }
        registry[sha] = trainer.index_run(
            {"manifest_sha256": sha, "candidates": [candidate]},
            [case],
            {
                "kind": "original_sft" if index == 0 else "supplement",
                "manifest_sha256": sha,
            },
        )
        pair = {
            "pair_id": f"pair-{index}",
            "case_id": case["id"],
            "candidate_id": candidate["candidate_id"],
            "candidate_run_sha256": sha,
            "input_sha256": candidate["input_sha256"],
            "chosen": case["primary"],
            "rejected_text": candidate["text"],
            "judgment": {
                "actor": "ai_generated",
                "status": "preferred",
                "rationale": "Synthetic executable edit versus the actual frozen abstention.",
            },
        }
        raw.append((pair, str(index) * 64))
    return registry, raw, Tokenizer(), {"format": "json"}


class SupplementalPreferenceTests(unittest.TestCase):
    def test_same_candidate_id_in_two_runs_uses_the_bound_input_and_survives_preflight_rebuild(
        self,
    ):
        registry, raw, tokenizer, config = preference_fixture()
        pairs = trainer.prepare_preferences(raw, registry, tokenizer, config)
        trainer.verify_prepared_pairs(pairs, raw, registry, tokenizer, config)
        self.assertEqual(
            [pair["case_id"] for pair in pairs], [pair["case_id"] for pair, _ in raw]
        )
        self.assertEqual(
            [pair["candidate_run_sha256"] for pair in pairs], list(registry)
        )
        self.assertEqual(
            [pair["input_bundle"]["kind"] for pair in pairs],
            ["original_sft", "supplement"],
        )
        self.assertEqual(
            [pair["judgments_sha256"] for pair in pairs], [sha for _, sha in raw]
        )
        self.assertEqual(
            [pair["metadata"]["chosen_action"] for pair in pairs],
            ["correct_text", "merge_paragraph"],
        )
        self.assertEqual(
            trainer.pair_census(pairs)["by"]["input_bundle_kind"],
            {"original_sft": 1, "supplement": 1},
        )

    def test_cross_run_case_binding_and_unknown_run_are_rejected(self):
        for run_sha in ("b" * 64, "c" * 64):
            with self.subTest(run=run_sha):
                registry, raw, tokenizer, config = preference_fixture()
                raw[0][0]["candidate_run_sha256"] = run_sha
                with self.assertRaisesRegex(
                    ValueError, "unregistered candidate run|bound run's input roster"
                ):
                    trainer.prepare_preferences(raw, registry, tokenizer, config)

    def test_duplicate_candidate_id_in_one_run_never_overwrites_a_row(self):
        registry, _, _, _ = preference_fixture()
        original = registry["a" * 64]
        with self.assertRaisesRegex(ValueError, "Duplicate candidate IDs"):
            trainer.index_run(
                {"candidates": original["candidates"] * 2},
                list(original["cases_by_id"].values()) * 2,
                original["input_bundle"],
            )

    def test_same_state_pair_cannot_be_counted_twice_even_with_new_case_run_and_pair_ids(
        self,
    ):
        registry, raw, tokenizer, config = preference_fixture()
        pair = trainer.prepare_preferences(raw[:1], registry, tokenizer, config)[0]
        duplicate = copy.deepcopy(pair)
        duplicate.update(
            pair_id="renamed-pair",
            case_id="renamed-case",
            candidate_run_sha256="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate same-state"):
            trainer.reject_duplicate_pairs([pair, duplicate])

    def test_preflight_rebuild_rejects_supplement_token_or_provenance_changes(self):
        registry, raw, tokenizer, config = preference_fixture()
        original = trainer.prepare_preferences(raw, registry, tokenizer, config)
        for mutate in (
            lambda pair: pair["input_bundle"].update(manifest_sha256="c" * 64),
            lambda pair: pair["chosen"]["input_ids"].__setitem__(-2, 777),
            lambda pair: pair.update(candidate_run_sha256="a" * 64),
            lambda pair: pair["judgment"].update(rationale="A different judgment."),
        ):
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(original)
                mutate(changed[1])
                with self.assertRaisesRegex(ValueError, "Prepared pairs differ"):
                    trainer.verify_prepared_pairs(
                        changed, raw, registry, tokenizer, config
                    )

    def test_strict_pair_rules_still_reject_ties_uncertainty_and_fabricated_rejections(
        self,
    ):
        registry, raw, tokenizer, config = preference_fixture()
        for status in ("tie", "uncertain"):
            changed = copy.deepcopy(raw[:1])
            changed[0][0]["judgment"]["status"] = status
            with (
                self.subTest(status=status),
                self.assertRaisesRegex(
                    ValueError, "explicitly justified AI preferences"
                ),
            ):
                trainer.prepare_preferences(changed, registry, tokenizer, config)
        changed = copy.deepcopy(raw[:1])
        changed[0][0]["chosen"] = {"action": "no_action"}
        changed[0][0]["rejected_text"] = '{ "action": "no_action" }'
        with self.assertRaisesRegex(ValueError, "Equivalent executed outcomes"):
            trainer.prepare_preferences(changed, registry, tokenizer, config)
        changed = copy.deepcopy(raw[:1])
        changed[0][0]["rejected_text"] = (
            "A made up rejection absent from this candidate run"
        )
        with self.assertRaisesRegex(ValueError, "actual frozen SFT candidate"):
            trainer.prepare_preferences(changed, registry, tokenizer, config)

    def test_original_51_byte_order_is_fixed_and_zero_new_preferences_is_allowed(self):
        registry = {trainer.ORIGINAL_CANDIDATE_SHA256: {}, "b" * 64: {}}
        with tempfile.TemporaryDirectory() as directory:
            original, extra = (
                Path(directory) / "original.jsonl",
                Path(directory) / "extra.jsonl",
            )
            trainer.prep.write_rows(
                original,
                [
                    {
                        "pair_id": f"pair-{index}",
                        "candidate_run_sha256": trainer.ORIGINAL_CANDIDATE_SHA256,
                    }
                    for index in range(51)
                ],
            )
            extra.write_text("")
            with patch.object(
                trainer, "ORIGINAL_PREFERENCES_SHA256", trainer.digest(original)
            ):
                entries, raw = trainer.read_preferences([original, extra], registry)
                self.assertEqual([entry["pairs"] for entry in entries], [51, 0])
                self.assertEqual(
                    [pair["pair_id"] for pair, _ in raw],
                    [f"pair-{index}" for index in range(51)],
                )
                original.write_bytes(original.read_bytes() + b" ")
                with self.assertRaisesRegex(
                    ValueError, "original 51 preference judgments"
                ):
                    trainer.read_preferences([original, extra], registry)

    def test_registry_verifies_copied_supplement_before_resolving_any_pair(self):
        for name in ("cases.jsonl", "case-roster.jsonl", "evidence/admission.json"):
            with (
                self.subTest(payload=name),
                tempfile.TemporaryDirectory() as directory,
                input_bundle(directory) as path,
            ):
                identity = trainer.supplement.verify_supplement(path)["identity"]
                target = path / name
                target.write_bytes(target.read_bytes() + b" ")
                with patch.object(trainer, "verify_original_candidates") as original:
                    with self.assertRaisesRegex(ValueError, "Bound file changed"):
                        trainer.candidate_registry(
                            {},
                            Path(directory),
                            {},
                            identity,
                            "unused",
                            "unused",
                            path,
                            object(),
                        )
                    original.assert_not_called()


class SupplementalTrainingBindingTests(unittest.TestCase):
    def test_fixed_one_pass_does_not_inherit_the_848_sft_sample_order(self):
        registry, raw, tokenizer, config = preference_fixture()
        pairs = trainer.prepare_preferences(raw, registry, tokenizer, config)
        config["training"] = {
            "sample_order": ["old"] * 848,
            "draw_count": 848,
            "passes": 31,
            "max_total_sequence_tokens": 999999,
            "learning_rate": 1e-4,
            "gradient_accumulation": 8,
            "max_updates": 212,
            "dropout": 0,
        }
        before = copy.deepcopy(config)
        settings = trainer.training_settings(config, pairs)
        self.assertEqual(config, before)
        self.assertEqual(
            {key: settings[key] for key in trainer.FIXED_TRAINING},
            trainer.FIXED_TRAINING,
        )
        self.assertEqual(settings["pair_order"], ["pair-0", "pair-1"])
        self.assertEqual(settings["max_updates"], 1)
        self.assertNotIn("sample_order", settings)
        self.assertNotIn("draw_count", settings)
        self.assertNotIn("max_total_sequence_tokens", settings)

    def test_cache_binding_changes_if_only_supplement_manifest_changes(self):
        registry, raw, tokenizer, config = preference_fixture()
        pairs = trainer.prepare_preferences(raw, registry, tokenizer, config)
        config.update(
            anchor={"adapter": "frozen"},
            pairs_sha256="p" * 64,
            format_version="json",
            supplement={"manifest_sha256": "a" * 64},
        )
        original = trainer.dpo.cache_binding(config, pairs)
        config["supplement"]["manifest_sha256"] = "b" * 64
        self.assertNotEqual(original, trainer.dpo.cache_binding(config, pairs))

    def test_reference_parity_samples_each_candidate_run(self):
        registry, raw, tokenizer, config = preference_fixture()
        pairs = trainer.prepare_preferences(raw, registry, tokenizer, config)
        seen = []

        def parity(model, selected, references, torch, device, beta):
            seen.append([row["pair_id"] for row in selected])
            return [
                {
                    "pair_id": row["pair_id"],
                    "absolute_errors": {"chosen": 0, "rejected": 0},
                }
                for row in selected
            ]

        with patch.object(trainer.dpo, "check_reference_direct", side_effect=parity):
            result = trainer.check_reference_direct(None, pairs, {}, None, "unused")
        self.assertEqual(seen, [["pair-0"], ["pair-1"]])
        self.assertEqual(
            [row["candidate_run_sha256"] for row in result], list(registry)
        )

    def test_protocol_must_bind_actual_count_all_inputs_and_one_pass(self):
        registry, raw, tokenizer, config = preference_fixture()
        pairs = trainer.prepare_preferences(raw, registry, tokenizer, config)
        identity = {"anchor_adapter_sha256": "frozen"}
        inputs = {"bindings": {"manifest_sha256": "supplement"}}
        preferences = [{"sha256": "original"}, {"sha256": "extra"}]
        protocol = {
            "version": trainer.PROTOCOL_VERSION,
            "final_test_bodies_read": False,
            "anchor": identity,
            "candidate_manifest_sha256": list(registry),
            "preference_sha256": ["original", "extra"],
            "supplement_manifest_sha256": "supplement",
            "training_plan": {
                **trainer.FIXED_TRAINING,
                "pair_count": 2,
                "max_updates": 1,
            },
        }
        trainer.validate_protocol(
            protocol, identity, registry, inputs, preferences, pairs
        )
        for key, value in (("passes", 2), ("beta", 0.2), ("pair_count", 51)):
            changed = copy.deepcopy(protocol)
            changed["training_plan"][key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "Protocol must bind"),
            ):
                trainer.validate_protocol(
                    changed, identity, registry, inputs, preferences, pairs
                )

    def test_export_accepts_new_version_and_reuses_exact_tensor_verification(self):
        import export_gemma_actions_v3 as lora
        import export_passive_gemma as exporter

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            training, reloaded, output = (
                root / "training",
                root / "reload",
                root / "export",
            )
            training.mkdir()
            output.mkdir()
            trainer.write(training / "manifest.json", {"version": trainer.VERSION})
            trainer.write(
                training / "configuration.json",
                {
                    "version": trainer.VERSION,
                    "sources_sha256": {
                        Path(module.__file__).name: "source"
                        for module in (lora, exporter)
                    },
                    "contract_version": "seven-actions",
                    "format_version": "json",
                    "format": "json",
                    "phase": "substantive",
                },
            )
            config = {
                "training_record": {"path": str(training / "manifest.json")},
                "adapter": {"path": str(training / "adapter")},
            }
            reload_check = stack.enter_context(
                patch.object(trainer.sft, "verify_reload")
            )
            stack.enter_context(patch.object(trainer.sft.smoke, "check_file"))
            stack.enter_context(
                patch.object(exporter, "configuration", return_value=config)
            )
            stack.enter_context(
                patch.object(
                    exporter,
                    "preflight",
                    return_value={
                        "pairs": {},
                        "alpha": 16,
                        "native_pin_sha256": "native",
                    },
                )
            )
            stack.enter_context(
                patch.object(exporter, "bound_run", return_value=(config, {}))
            )
            stack.enter_context(
                patch.object(
                    exporter,
                    "conversion_commands",
                    return_value=[
                        ("convert-base", ["unused"]),
                        ("convert-adapter", ["converter", "--f32"]),
                    ],
                )
            )
            command = stack.enter_context(patch.object(exporter, "run_command"))
            stack.enter_context(
                patch.object(exporter, "gguf_module", return_value=object())
            )
            stack.enter_context(patch.object(exporter, "sha256", return_value="bound"))
            verify = stack.enter_context(
                patch.object(
                    lora,
                    "verify_adapter_tensors",
                    return_value={"all_tensors_exact": True},
                )
            )
            stack.enter_context(
                patch.object(
                    lora,
                    "run",
                    side_effect=AssertionError(
                        "The closed old version gate must not run."
                    ),
                )
            )
            result = trainer.export_adapter(
                root / "export-config.json", reloaded, output
            )
            reload_check.assert_called_once_with(
                training, reloaded, version=trainer.VERSION
            )
            command.assert_called_once_with(
                output, "convert-adapter", ["converter", "--f32"]
            )
            verify.assert_called_once()
            self.assertTrue(result["all_tensors_exact"])
            self.assertEqual(result["version"], trainer.VERSION + "-lora-export")
            self.assertFalse(result["full_base_or_merged_copy_created"])
            self.assertEqual(
                json.loads((output / "adapter-receipt.json").read_text()), result
            )


if __name__ == "__main__":
    unittest.main()
