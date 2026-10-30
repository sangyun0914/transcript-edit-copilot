import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import prepare_small_actions_final_dpo as prep
import small_actions_final as data
import train_small_actions_final as sft
import train_small_actions_final_dpo as dpo
import serve_small_actions_final as serving
from test_gemma_actions_v3 import cases


class Tokenizer:
    def __init__(self, terminal=7):
        self.terminal = terminal

    def apply_chat_template(self, messages, **kwargs):
        prompt = messages if kwargs["add_generation_prompt"] else messages[:-1]
        prefix = [
            1,
            *[
                byte + 256
                for byte in hashlib.sha256(data.compact(prompt).encode()).digest()
            ],
        ]
        if kwargs["add_generation_prompt"]:
            return prefix
        return (
            prefix
            + [ord(c) + 256 for c in messages[-1]["content"]]
            + [self.terminal, 266]
        )

    def decode(self, tokens, *, skip_special_tokens=False):
        return "".join(
            ("" if skip_special_tokens else "<end>")
            if token in (7, 1, 106, 50)
            else "bad"
            if token == 999
            else chr(token - 256)
            for token in tokens
        )


def fixture(root):
    tokenizer = Tokenizer()
    original = cases()
    for case in original:
        case["accepted_predictions"] = case["acceptable"]
        case["lineage"] = {}
    data.write_rows(root / "cases.jsonl", original)
    config = {
        "model": "lfm25-1.2b",
        "base": data.PROFILES["lfm25-1.2b"],
        "source_cases_sha256": data.sha(root / "cases.jsonl"),
        "assets_path": str(root),
        "assets": {"mock": "native tokenizer assets"},
        "contract_version": data.formats.contract.VERSION,
        "context_tokens": 4096,
        "max_completion_tokens": 512,
        "generation": {"eos_token_id": [7]},
        "training": {"epochs": 3},
    }
    encoded = []
    for case in original:
        row = data.encode_completion(
            tokenizer,
            case["messages"],
            data.formats.serialize(case["input"], case["primary"], "json"),
            [7],
        )
        row.update(
            case_id=case["id"],
            semantic_input_sha256=data.value_hash(
                data.formats.contract.wire_snapshot(case["input"])
            ),
            messages=case["messages"],
        )
        encoded.append(row)
    for name in ("prepared", "training/adapter", "reload", "candidates"):
        (root / name).mkdir(parents=True)
    for name in ("prepared", "training"):
        data.write(root / name / "configuration.json", config)
    (root / "training/adapter/adapter_model.safetensors").write_bytes(
        b"exact parent adapter"
    )
    config_sha = data.sha(root / "prepared/configuration.json")
    adapter_sha = data.sha(root / "training/adapter/adapter_model.safetensors")
    data.write(
        root / "training/run.json",
        {
            "status": "training_complete_reload_pending",
            "pid": -1,
            "configuration_sha256": config_sha,
            "adapter": {"sha256": adapter_sha},
        },
    )
    data.write(
        root / "reload/run.json",
        {
            "status": "independent_reload_passed",
            "pid": -2,
            "training_configuration_sha256": config_sha,
            "adapter_sha256": adapter_sha,
        },
    )
    identity = sft.model_identity(root / "training", root / "reload")
    observations = []
    for case, row in zip(original, encoded, strict=True):
        text = (
            "bad"
            if case["id"] == "case-0"
            else data.formats.serialize(case["input"], {"action": "no_action"}, "json")
        )
        tokens = [ord(c) + 256 for c in text] + [7]
        observations.append(
            {
                "case_id": case["id"],
                "format": "json",
                "status": "completed",
                "finish_reason": "stop",
                "input_sha256": row["semantic_input_sha256"],
                "messages_sha256": data.value_hash(case["messages"]),
                "prefix_ids_sha256": data.value_hash(
                    row["input_ids"][: row["completion_start"]]
                ),
                "text": text,
                "text_raw": tokenizer.decode(tokens),
                "output_ids": tokens,
            }
        )
    data.write_rows(root / "candidates/observations.jsonl", observations)
    data.write(
        root / "candidates/run.json",
        {
            "status": "inference_complete",
            "kind": "sft",
            "splits": ["train"],
            "model_identity": identity,
            "configuration_sha256": config_sha,
            "observations_sha256": data.sha(root / "candidates/observations.jsonl"),
            "requested": len(original),
        },
    )
    return config, encoded, tokenizer, original, observations


def mine_fixture(root, config, encoded, output="mined", **kwargs):
    with mock.patch.object(data, "prepared", return_value=(config, encoded, root)):
        return prep.mine(
            root / "prepared",
            root / "training",
            root / "reload",
            root / "cases.jsonl",
            root / "candidates",
            root / output,
            **kwargs,
        )


class PairMiningTests(unittest.TestCase):
    def test_actual_train_errors_cover_preservation_and_all_sources_without_rewriting(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, encoded, _, original, observations = fixture(root)
            report = mine_fixture(root, config, encoded)
            pairs = data.rows(root / "mined/pairs.jsonl.gz")
            self.assertEqual(report["requested"], 7)
            self.assertEqual(report["preservation_pairs"], 1)
            self.assertEqual(report["pairs"], 7)
            self.assertEqual(report["pair_counts"]["chosen_action"]["no_action"], 1)
            for pair, case, observation in zip(
                pairs, original, observations, strict=True
            ):
                self.assertEqual(pair["input"], case["input"])
                self.assertEqual(pair["messages"], case["messages"])
                self.assertEqual(
                    pair["accepted_predictions"], case["accepted_predictions"]
                )
                self.assertEqual(pair["rejected"], observation)
            with self.assertRaises(FileExistsError):
                mine_fixture(root, config, encoded)

    def test_foreign_parent_dev_run_or_rewritten_observation_fails(self):
        for mutation in ("parent", "split", "observations", "cases"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                config, encoded, _, _, _ = fixture(root)
                run = data.read(root / "candidates/run.json")
                if mutation == "parent":
                    run["model_identity"]["adapter_sha256"] = "base-or-other-parent"
                elif mutation == "split":
                    run["splits"] = ["train", "dev"]
                elif mutation == "observations":
                    with (root / "candidates/observations.jsonl").open("a") as stream:
                        stream.write("{}\n")
                else:
                    with (root / "cases.jsonl").open("a") as stream:
                        stream.write("{}\n")
                data.write(root / "candidates/run.json", run)
                with self.assertRaises(ValueError):
                    mine_fixture(root, config, encoded)

    def test_ties_and_valid_nonreference_candidates_do_not_become_automatic_preferences(
        self,
    ):
        for preserve in (False, True):
            case = cases()[0 if preserve else 1]
            case.update(accepted_predictions=case["acceptable"], lineage={})
            bundle = prep.bound_evaluation_bundle(case)
            accepted = {
                "status": "completed",
                "finish_reason": "stop",
                "text": data.compact(case["primary"]),
            }
            self.assertEqual(prep.classify(case, accepted, bundle)["status"], "tie")
            alternate = {
                **accepted,
                "text": '{"action":"correct_text","target":[1],"text":"you"}',
            }
            decision = prep.classify(case, alternate, bundle)
            self.assertEqual(decision["status"], "needs_review")
            alternate.update(case_id=case["id"], input_sha256="input")
            review = {
                "case_id": case["id"],
                "candidate_run_sha256": "run",
                "input_sha256": "input",
                "actor": "ai_generated",
                "status": "preferred",
                "rationale": "Changes the speaker's intended wording.",
                "chosen": {"action": "delete_span", "target": [0]},
            }
            with self.assertRaisesRegex(ValueError, "original accepted"):
                prep.reviewed_decision(case, alternate, decision, review, "run")
            review.update(chosen=case["primary"], status="uncertain")
            self.assertEqual(
                prep.reviewed_decision(case, alternate, decision, review, "run")[
                    "status"
                ],
                "uncertain",
            )
            with self.assertRaisesRegex(ValueError, "unresolved"):
                prep.reviewed_decision(
                    case, alternate, {"status": "tie"}, review, "run"
                )

    def test_incomplete_generation_is_excluded_even_when_it_looks_like_an_error(self):
        case = cases()[1]
        case.update(accepted_predictions=case["acceptable"], lineage={})
        for observation in (
            {"status": "request_failure"},
            {"status": "completed", "finish_reason": "length", "text": "bad"},
        ):
            self.assertEqual(
                prep.classify(case, observation, prep.bound_evaluation_bundle(case))[
                    "status"
                ],
                "excluded",
            )

    def test_train_source_and_prompt_are_validated_even_with_rebound_file_hash(self):
        for mutation in ("split", "prompt"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                config, encoded, _, original, _ = fixture(root)
                if mutation == "split":
                    original[0]["source"]["split"] = "dev"
                else:
                    original[0]["messages"][0]["content"] += "different contract"
                data.write_rows(root / "cases.jsonl", original)
                config["source_cases_sha256"] = data.sha(root / "cases.jsonl")
                with self.assertRaises(ValueError):
                    mine_fixture(root, config, encoded)


class PreparationAndReloadTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("peft"), "Requires the ML training dependencies."
    )
    def test_unchanged_adapter_tensor_bytes_do_not_allow_changed_lora_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, _, _, _ = fixture(root)
            config["training"].update(
                rank=16, alpha=16, dropout=0.0, target_mode="hybrid-ffn"
            )
            data.write(root / "training/configuration.json", config)
            run = data.read(root / "training/run.json")
            run["configuration_sha256"] = data.sha(root / "training/configuration.json")
            data.write(root / "training/run.json", run)
            adapter = {
                "base_model_name_or_path": config["base"]["repository"],
                "revision": config["base"]["revision"],
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": ["module"],
            }
            data.write(root / "training/adapter/adapter_config.json", adapter)
            model = SimpleNamespace(named_modules=lambda: [("module", object())])
            with (
                mock.patch(
                    "peft.PeftModel.from_pretrained", return_value="loaded"
                ) as loader,
                mock.patch.object(
                    data, "select_targets", return_value={"module": [8, 8]}
                ),
                mock.patch.object(sft, "verify_adapter_file"),
            ):
                with self.assertRaisesRegex(ValueError, "execution settings"):
                    sft.load_adapter(model, root / "training", config, None)
                loader.assert_not_called()
                adapter["lora_alpha"] = 16
                data.write(root / "training/adapter/adapter_config.json", adapter)
                self.assertEqual(
                    sft.load_adapter(model, root / "training", config, None), "loaded"
                )

    def test_native_rejected_tokens_are_retained_and_budget_does_not_truncate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, encoded, tokenizer, _, _ = fixture(root)
            mine_fixture(root, config, encoded)
            pair = data.rows(root / "mined/pairs.jsonl.gz")[0]
            pair["rejected"]["output_ids"] = [999, 7]
            actual = prep.encoded_pair(pair, tokenizer, config)
            self.assertEqual(
                actual["rejected"]["labels"][actual["rejected"]["completion_start"] :],
                [999, 7],
            )
            self.assertEqual(
                actual["chosen"]["labels"][: actual["chosen"]["completion_start"]],
                [-100] * 33,
            )
            config["max_completion_tokens"] = 1
            with self.assertRaisesRegex(ValueError, "budget"):
                prep.encoded_pair(pair, tokenizer, config)
            config["max_completion_tokens"] = 512
            pair["rejected"]["output_ids"] = [999, 50]
            with self.assertRaises(ValueError):
                prep.encoded_pair(pair, tokenizer, config)

    def test_gemma_actual_terminal_can_differ_from_the_native_chosen_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, encoded, _, _, _ = fixture(root)
            mine_fixture(root, config, encoded)
            pair = data.rows(root / "mined/pairs.jsonl.gz")[0]
            config.update(model="gemma4-e2b", base=data.PROFILES["gemma4-e2b"])
            config["generation"]["eos_token_id"] = [1, 106, 50]
            pair["rejected"]["output_ids"] = [999, 50]
            actual = prep.encoded_pair(pair, Tokenizer(106), config)
            self.assertEqual(actual["chosen"]["input_ids"][-1], 106)
            self.assertEqual(actual["rejected"]["input_ids"][-1], 50)

    def test_prepare_freezes_budget_plan_and_requires_actual_preservation_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, encoded, tokenizer, _, _ = fixture(root)
            mine_fixture(root, config, encoded)
            (root / "plan.md").write_text(
                "One epoch; adopt only if full DEV precision and preservation do not regress."
            )
            with (
                mock.patch.object(data, "verify_assets", return_value=config["assets"]),
                mock.patch.object(data, "tokenizer_from", return_value=tokenizer),
            ):
                with self.assertRaisesRegex(ValueError, "Too few"):
                    prep.prepare(root / "mined", root / "plan.md", root / "failed")
                frozen = prep.prepare(
                    root / "mined",
                    root / "plan.md",
                    root / "prepared-dpo",
                    minimum_pairs=7,
                )
                self.assertEqual(frozen["training"]["epochs"], 1)
                self.assertEqual(frozen["training"]["updates"], 2)
                self.assertEqual(frozen["preservation_pairs"], 1)
                self.assertEqual(
                    data.read(root / "prepared-dpo/configuration.json"), config
                )
                prep.prepared(root / "prepared-dpo")
                (root / "prepared-dpo/plan.md").write_text(
                    "Change adoption after seeing results"
                )
                with self.assertRaisesRegex(ValueError, "preparation changed"):
                    prep.prepared(root / "prepared-dpo")
                with self.assertRaisesRegex(ValueError, "preservation"):
                    prep.prepare(
                        root / "mined",
                        root / "plan.md",
                        root / "no-preservation",
                        minimum_pairs=7,
                        minimum_preservation_pairs=0,
                    )

    def test_dpo_requires_separate_reload_diagnostics_for_existing_export_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, encoded, tokenizer, _, _ = fixture(root)
            mine_fixture(root, config, encoded)
            (root / "plan.md").write_text("Frozen adoption plan")
            with (
                mock.patch.object(data, "verify_assets", return_value=config["assets"]),
                mock.patch.object(data, "tokenizer_from", return_value=tokenizer),
            ):
                frozen = prep.prepare(
                    root / "mined", root / "plan.md", root / "dpo", minimum_pairs=7
                )
            (root / "dpo/adapter").mkdir()
            (root / "dpo/adapter/adapter_model.safetensors").write_bytes(b"trained-dpo")
            info = {
                "kind": "dpo",
                "status": "training_complete_reload_pending",
                "rows": 7,
                "configuration_sha256": frozen["parent"]["configuration_sha256"],
                "dpo_configuration_sha256": data.sha(
                    root / "dpo/dpo_configuration.json"
                ),
                "parent_adapter_sha256": frozen["parent"]["adapter_sha256"],
                "adapter": {
                    "sha256": data.sha(root / "dpo/adapter/adapter_model.safetensors")
                },
                "draws_completed": 7,
                "updates_completed": 2,
            }
            data.write(root / "dpo/run.json", info)
            reload = data.read(root / "reload/run.json")
            reload["adapter_sha256"] = info["adapter"]["sha256"]
            data.write(root / "reload/run.json", reload)
            with self.assertRaisesRegex(ValueError, "pair diagnostics"):
                sft.verified_reload(root / "dpo", root / "reload")
            reload.update(
                kind="dpo",
                dpo_configuration_sha256=info["dpo_configuration_sha256"],
                parent_adapter_sha256=info["parent_adapter_sha256"],
                pair_diagnostics=7,
            )
            data.write(root / "reload/run.json", reload)
            sft.verified_reload(root / "dpo", root / "reload")
            info["updates_completed"] = 1
            data.write(root / "dpo/run.json", info)
            with self.assertRaisesRegex(ValueError, "budget"):
                sft.verified_reload(root / "dpo", root / "reload")

    def test_reference_cache_binds_all_pair_tokens_and_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cache").mkdir()
            data.write(root / "dpo_configuration.json", {"parent": "sft"})
            pairs = [{"pair_id": "p", "chosen": [1], "rejected": [2]}]
            values = [
                {
                    "pair_id": "p",
                    "pair_sha256": data.value_hash(pairs[0]),
                    "chosen_logprob": -5.0,
                    "rejected_logprob": -6.0,
                }
            ]
            data.write_rows(root / "cache/reference.jsonl", values)
            config = {
                "parent": {"adapter_sha256": "exact-sft"},
                "pairs_sha256": "pairs",
            }
            run = {
                "status": "frozen_reference_complete",
                **config,
                "scored": 1,
                "dpo_configuration_sha256": data.sha(root / "dpo_configuration.json"),
                "reference_sha256": data.sha(root / "cache/reference.jsonl"),
            }
            data.write(root / "cache/run.json", run)
            dpo.reference_cache(root / "cache", root, config, pairs)
            pairs[0]["rejected"] = [3]
            with self.assertRaisesRegex(ValueError, "score binding"):
                dpo.reference_cache(root / "cache", root, config, pairs)

    def test_existing_export_preserves_dpo_budget_parent_and_rejects_changed_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("merged/model", "llama"):
                (root / directory).mkdir(parents=True)
            (root / "merged/model/chat_template.jinja").write_text("native template")
            (root / "merged/plan.md").write_text("frozen adoption and budget")
            (root / "llama/convert_hf_to_gguf.py").write_text("mock converter")
            (root / "quantizer").write_text("mock quantizer")
            frozen = {
                "parent": {"adapter_sha256": "parent"},
                "plan_sha256": data.sha(root / "merged/plan.md"),
                "training": {"epochs": 1},
            }
            data.write(root / "merged/dpo_configuration.json", frozen)
            data.write(
                root / "merged/configuration.json",
                {"base": data.PROFILES["lfm25-1.2b"], "training": {"epochs": 3}},
            )
            frozen["parent"]["configuration_sha256"] = data.sha(
                root / "merged/configuration.json"
            )
            data.write(root / "merged/dpo_configuration.json", frozen)
            data.write(
                root / "merged/run.json",
                {
                    "kind": "dpo",
                    "status": "merged_native_bf16_local_validation_pending",
                    "files_sha256": {
                        "model/chat_template.jinja": data.sha(
                            root / "merged/model/chat_template.jinja"
                        )
                    },
                    "training_configuration_sha256": frozen["parent"][
                        "configuration_sha256"
                    ],
                    "adapter_sha256": "final-dpo",
                    "parent_adapter_sha256": "parent",
                    "dpo_configuration_sha256": data.sha(
                        root / "merged/dpo_configuration.json"
                    ),
                },
            )

            def convert(command, **kwargs):
                destination = command[-1] if "--outfile" in command else command[2]
                Path(destination).write_bytes(b"mock gguf")

            gguf = SimpleNamespace(
                GGUFReader=lambda _: SimpleNamespace(
                    get_field=lambda _: SimpleNamespace(contents=lambda: "lfm2")
                )
            )
            with (
                mock.patch.dict(sys.modules, {"gguf": gguf}),
                mock.patch.object(
                    serving.subprocess, "run", side_effect=convert
                ) as process,
            ):
                result = serving.export(
                    root / "merged", root / "llama", root / "quantizer", root / "export"
                )
                self.assertEqual(result["kind"], "dpo")
                self.assertEqual(result["parent_adapter_sha256"], "parent")
                self.assertEqual(
                    result["training_configuration_sha256"],
                    frozen["parent"]["configuration_sha256"],
                )
                self.assertEqual(
                    data.read(root / "export/dpo_configuration.json"), frozen
                )
                self.assertEqual(
                    data.read(root / "export/configuration.json")["training"]["epochs"],
                    3,
                )
                self.assertEqual(process.call_count, 2)
                (root / "merged/plan.md").write_text("changed after training")
                with self.assertRaisesRegex(ValueError, "training plan changed"):
                    serving.export(
                        root / "merged",
                        root / "llama",
                        root / "quantizer",
                        root / "bad-export",
                    )
                self.assertEqual(process.call_count, 2)


class ObjectiveTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Requires the ML training dependencies."
    )
    def test_sequential_gradient_matches_standard_dpo_and_uses_completion_sum(self):
        import torch

        parameter = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
        pair = {
            "pair_id": "p",
            "chosen": {"slope": [1.0, 2.0, 3.0]},
            "rejected": {"slope": [-1.0, 2.0]},
        }

        def logps(model, row, torch):
            return model * torch.tensor(row["slope"], dtype=torch.float64) - 5

        reference = {"chosen_logprob": -14.0, "rejected_logprob": -10.0}
        with mock.patch.object(sft, "completion_logps", side_effect=logps):
            chosen, rejected = (
                dpo.summed_logprob(parameter, pair[side], torch) for side in prep.SIDES
            )
            self.assertAlmostEqual(float(chosen.detach()), -13.2)
            loss, _ = dpo.dpo_loss(chosen, rejected, reference, 0.1, torch)
            (loss / 3).backward()
            expected = parameter.grad.detach().clone()
            parameter.grad = None
            actual = dpo.backward_pair(parameter, pair, reference, 0.1, 3, torch)
            self.assertTrue(
                torch.allclose(parameter.grad, expected, atol=1e-12, rtol=1e-12)
            )
            self.assertAlmostEqual(actual["loss"], float(loss.detach()))
            same_parent = {"chosen_logprob": -13.2, "rejected_logprob": -9.7}
            loss, margin = dpo.dpo_loss(chosen, rejected, same_parent, 0.1, torch)
            self.assertAlmostEqual(float(margin.detach()), 0.0)
            self.assertAlmostEqual(float(loss.detach()), math.log(2))

    def test_reload_rejects_changed_scores_and_same_sample_output(self):
        expected = [
            {
                "pair_id": "p",
                "pair_sha256": "h",
                "chosen_logprob": -5.0,
                "rejected_logprob": -6.0,
                "generation": {"output_ids": [100, 7]},
            }
        ]
        dpo.verify_diagnostics(expected, copy.deepcopy(expected))
        for edit in (
            {"chosen_logprob": -5.1},
            {"rejected_logprob": float("nan")},
            {"generation": {"output_ids": [101, 7]}},
        ):
            with self.assertRaises(ValueError):
                dpo.verify_diagnostics(expected, [{**expected[0], **edit}])


if __name__ == "__main__":
    unittest.main()
