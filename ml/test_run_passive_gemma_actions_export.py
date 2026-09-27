import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_passive_gemma_actions_export as actions


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def proof_fixture():
    return {
        "version": "passive-actions-gemma-independent-phase-verification-v1",
        "status": "passed_complete_phase_and_independent_reload",
        "phase": "substantive",
        "configuration_sha256": actions.CONFIG_SHA,
        "verifier_sha256": actions.VERIFIER_SHA,
        "budget": {
            "draws": 42,
            "updates": 11,
            "final_rows": 2,
            "sequence_tokens": 56276,
            "completion_tokens": 598,
        },
        "checks": dict.fromkeys(actions.FINAL_CHECKS, True),
        "initial_t1_adapter_tensors_exact": 410,
        "finite_final_fp32_adapter_tensors": 410,
        "finite_final_fp32_adamw_states": 410,
        "trainable_parameters": 18585600,
        "changed_final_vs_t1_tensors": 410,
        "identical_raw_diagnostic_outputs": 8,
        "train_pid": 10,
        "reload_pid": 11,
        "new_training_updates": 0,
        "model_forward_performed": False,
    }


class ActionsExportTests(unittest.TestCase):
    def setUp(self):
        (actions.ROOT / "artifacts/training").mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(
            prefix="unit-actions-export-", dir=actions.ROOT / "artifacts/training"
        )
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_rejects_wrong_phase_partial_group_and_incomplete_anchor(self):
        original = proof_fixture()
        actions.validate_proof(original, actions.CONFIG_SHA)
        mutations = (
            ("phase", "smoke"),
            ("finite_final_fp32_adamw_states", 409),
            ("initial_t1_adapter_tensors_exact", 0),
            ("reload_pid", 10),
            ("configuration_sha256", "0" * 64),
            ("new_training_updates", 1),
            ("checks", dict.fromkeys(actions.FINAL_CHECKS, False)),
            ("budget", {**original["budget"], "final_rows": 4}),
            ("budget", {**original["budget"], "updates": 201}),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                actions.validate_proof({**original, key: value}, actions.CONFIG_SHA)

    def test_transitive_nested_local_import_closure(self):
        directory = self.root / "ml"
        directory.mkdir()
        main, first, second = [
            directory / name for name in ("main.py", "first.py", "second.py")
        ]
        main.write_text("import first\n")
        first.write_text(
            "def later():\n    from second import value\n    import json\n"
        )
        second.write_text("value = 1\n")
        with (
            patch.object(actions, "ROOT", self.root),
            patch.object(actions, "__file__", str(main)),
            patch.object(actions.reuse, "source_paths", return_value=[]),
            patch.object(actions.trainer, "SOURCE_MODULES", ()),
        ):
            self.assertEqual(actions.source_paths(), sorted([main, first, second]))

    def test_configuration_and_commands_preserve_native_and_original_worker(self):
        canonical = {
            "base": {"path": "native"},
            "native_pin": {"sha256": "p"},
            "runtime": {
                "python": "/lexical/.venv/bin/python",
                "support_files_sha256": {"real.dylib": "h"},
            },
            "adapter": {"path": "old"},
            "training_record": {"path": "old"},
        }
        before = copy.deepcopy(canonical)
        manifest = self.root / "train/manifest.json"
        config = actions.build_configuration(
            canonical,
            manifest,
            {
                "files_sha256": {
                    "adapter/adapter_model.safetensors": "a",
                    "adapter/adapter_config.json": "b",
                    "run.json": "c",
                }
            },
            {"path": "proof", "sha256": "d"},
        )
        self.assertEqual(canonical, before)
        for key in ("base", "native_pin", "runtime"):
            self.assertEqual(config[key], canonical[key])
        self.assertEqual(config["continuation"]["training_updates"], 11)
        self.assertFalse(config["continuation"]["optimizer_state_resumed"])
        self.assertEqual(config["adapter"]["path"], str(manifest.parent / "adapter"))
        commands = actions.commands(
            config, self.root / "config.json", self.root / "out"
        )
        for stage in ("convert", "quantize"):
            self.assertEqual(
                commands[stage][3], str(Path(actions.reuse.__file__).resolve())
            )
            self.assertEqual(
                commands[stage][4:],
                [
                    "worker",
                    "--plan",
                    str(self.root / "out/plan.json"),
                    "--stage",
                    stage,
                ],
            )
        for stage in ("preflight", "verify", "receipt"):
            self.assertEqual(
                commands[stage][3], str(Path(actions.exporter.__file__).resolve())
            )

    def test_configuration_argument_cannot_be_replaced_by_policy(self):
        config = put(self.root / "config.json", {})
        request = put(self.root / "request.json", {})
        policy = put(
            self.root / "policy.json",
            {
                "version": actions.VERSION,
                "mode": actions.MODE,
                "request": actions.binding(request),
                "export_configuration": actions.binding(config),
            },
        )
        context = ({}, {}, {}, {}, {}, {}, {}, set(), 0)
        with (
            patch.object(actions, "load_context", return_value=context),
            self.assertRaisesRegex(ValueError, "argument differs"),
        ):
            actions.load_plan(
                self.root / "different.json",
                policy,
                actions.digest(policy),
                self.root / "out",
            )

    def test_capacity_retires_only_completed_rewrite_and_keeps_reserve(self):
        caps = {
            "old_rewrite_export": actions.STORAGE["new_model_cap_bytes"],
            "new_all_action_export": actions.STORAGE["new_model_cap_bytes"],
            "minimum_free_reserve": actions.STORAGE["minimum_free_bytes"],
            "ordinary_qwen": 48491233824,
        }
        original = {
            "status": "passed_registered_fixed_future_allocations",
            "future_caps_bytes": caps,
        }
        remaining = {k: v for k, v in caps.items() if k != "old_rewrite_export"}
        current = {
            "status": "passed_completed_rewrite_future_reservation_retired_once",
            "previous_registration_sha256": actions.CAPACITY_SHA,
            "retired_newly_completed_rewrite_future_bytes": caps["old_rewrite_export"],
            "remaining_future_caps_bytes": remaining,
            "remaining_future_required_bytes": sum(remaining.values()),
            "new_smoke_or_input_allocation_credits": 0,
            "previous_json_or_training_credits_repeated": 0,
            "next_export_order": [
                "new_all_action_after_final_reload_and_own_lineage_registration",
                "ordinary_qwen",
            ],
            "minimum_export_stage_bytes": {
                "conversion": 95676664704,
                "quantize": 55450723328,
            },
        }
        self.assertEqual(
            actions.validate_capacity(original, current), sum(remaining.values())
        )
        for key, value in (
            ("new_smoke_or_input_allocation_credits", 1),
            ("remaining_future_required_bytes", 1),
            ("next_export_order", ["ordinary_qwen"]),
            ("minimum_export_stage_bytes", {"conversion": 1, "quantize": 1}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                actions.validate_capacity(original, {**current, key: value})

    def header_fixture(self):
        base, adapter = self.root / "native", self.root / "adapter"
        base.mkdir()
        adapter.mkdir()
        native = {
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 30,
                "attention_k_eq_v": True,
                "layer_types": ["sliding_attention"] * 25 + ["full_attention"] * 5,
            },
        }
        targets = sorted(actions.exporter.target_names(native))
        inventory, tensors = {}, {}
        for i, name in enumerate(targets):
            h, w = 2833 + (70 if i == 204 else 0), 2833
            inventory[name] = {
                "shape": [h, w],
                "dtype": "BF16",
                "shard": "model.safetensors",
            }
            for side, shape in (("A", [16, w]), ("B", [h, 16])):
                tensors[f"base_model.model.{name[:-7]}.lora_{side}.weight"] = {
                    "shape": shape,
                    "dtype": "F32",
                }
        put(base / "config.json", native)
        put(
            base / "model.safetensors.index.json",
            {"weight_map": {name: "model.safetensors" for name in inventory}},
        )
        put(base / "generation_config.json", {"eos_token_id": [1, 106, 50]})
        (base / "model.safetensors").write_bytes(b"synthetic header reader stub")
        (adapter / "adapter_model.safetensors").write_bytes(
            b"synthetic header reader stub"
        )
        api = put(self.root / "native-api.json", {"fixture": True})
        pin = put(self.root / "native-pin.json", {"api_sha256": actions.digest(api)})
        config = {
            "base": {
                "path": str(base),
                "files_sha256": {p.name: actions.digest(p) for p in base.iterdir()},
            },
            "native_pin": actions.binding(pin),
            "adapter": {"path": str(adapter), "files_sha256": {}},
        }
        adapter_config = {
            "base_model_name_or_path": actions.exporter.BASE_REPOSITORY,
            "revision": actions.exporter.BASE_REVISION,
            "peft_type": "LORA",
            "bias": "none",
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "target_modules": [name[:-7] for name in targets],
        }
        training = {"model": {"target_modules": adapter_config["target_modules"]}}
        return config, training, adapter_config, inventory, tensors

    def test_header_checks_reject_default_base_revision_or_non_fp32(self):
        config, training, adapter_config, inventory, tensors = self.header_fixture()
        with (
            patch.object(actions.exporter, "validate_native_pin"),
            patch.object(actions.exporter, "native_inventory", return_value=inventory),
            patch.object(actions.exporter, "tensor_header", return_value=tensors),
        ):
            for key, value in (
                (None, None),
                ("base_model_name_or_path", ""),
                ("revision", None),
                ("r", 8),
            ):
                candidate = dict(adapter_config)
                if key is not None:
                    candidate[key] = value
                path = put(
                    Path(config["adapter"]["path"]) / "adapter_config.json", candidate
                )
                config["adapter"]["files_sha256"]["adapter_config.json"] = (
                    actions.digest(path)
                )
                if key is None:
                    self.assertEqual(
                        actions.adapter_headers(config, training, {})[
                            "fp32_parameters"
                        ],
                        18585600,
                    )
                else:
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        actions.adapter_headers(config, training, {})
            path = put(
                Path(config["adapter"]["path"]) / "adapter_config.json", adapter_config
            )
            config["adapter"]["files_sha256"]["adapter_config.json"] = actions.digest(
                path
            )
            next(iter(tensors.values()))["dtype"] = "BF16"
            with self.assertRaisesRegex(ValueError, "header/rank/dtype"):
                actions.adapter_headers(config, training, {})

    def fake_completed_export(self, output):
        output.mkdir()
        receipt = put(
            output / "export/receipt.json",
            {"artifacts": {"adapter": {"bytes": 74370944}}},
        )
        compactions = []
        for name in ("merged", "merged_quantized"):
            proof = put(
                output / "compaction" / f"{name}.reconstruction.json",
                {
                    "status": "exact_bytes_verified",
                    "max_changed_bytes": actions.reuse.LIMITS["max_changed_bytes"][
                        name
                    ],
                    "changed_chunk_bytes": 100,
                    "minimum_free_bytes": actions.STORAGE["minimum_free_bytes"],
                },
            )
            compactions.append(
                {"artifact": name, "reconstruction": actions.binding(proof)}
            )
        result = {
            "status": "paired_export_passed_local_evaluation_pending",
            "exit_code": 0,
            "mode": actions.MODE,
            "implementation_sha256": actions.EXECUTOR_SHA,
            "export_receipt": actions.binding(receipt),
            "compactions": compactions,
        }
        put(output / "result.json", result)
        return result

    def test_retention_rejects_cap_corruption_and_old_mode(self):
        output = self.root / "out"
        result = self.fake_completed_export(output)
        self.assertEqual(actions.retained_bytes(result, output)[0], 74371144)
        with self.assertRaises(ValueError):
            actions.retained_bytes({**result, "mode": "matched_final"}, output)
        proof = output / "compaction/merged.reconstruction.json"
        value = actions.read(proof)
        value["changed_chunk_bytes"] = (
            actions.reuse.LIMITS["max_changed_bytes"]["merged"] + 1
        )
        put(proof, value)
        with self.assertRaisesRegex(ValueError, "proof changed"):
            actions.retained_bytes(result, output)
        result["compactions"][0]["reconstruction"] = actions.binding(proof)
        with self.assertRaisesRegex(ValueError, "cap failed"):
            actions.retained_bytes(result, output)

    def test_execution_delegates_only_original_public_executor_and_keeps_new_identity(
        self,
    ):
        output = self.root / "out"
        metadata = put(self.root / "metadata.json", {"fixture": True})
        plan = {
            "mode": actions.MODE,
            "planner_sha256": actions.digest(Path(actions.__file__)),
            "storage": actions.STORAGE,
            "limits": actions.reuse.LIMITS,
            "space_required_before_stages": actions.STORAGE["stage_requirements"],
            "implementation_sha256": actions.EXECUTOR_SHA,
            "output": str(output),
            "remaining_future_required_bytes": 106492066848,
            "configuration": actions.binding(metadata),
            "policy": actions.binding(metadata),
            "scientific_validation": {"final_verification": actions.binding(metadata)},
        }

        def original_executor(actual):
            self.assertIs(actual, plan)
            result = self.fake_completed_export(output)
            put(output / "plan.json", actual)
            return result

        with (
            patch.object(
                actions.reuse, "execute", side_effect=original_executor
            ) as execute,
            patch.object(
                actions.reuse,
                "load_plan",
                side_effect=AssertionError("No old scientific planner"),
            ),
            patch.object(
                actions.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=200000000000),
            ),
        ):
            result = actions.execute(plan)
        execute.assert_called_once_with(plan)
        self.assertEqual(
            result["status"], "passed_actions_continuation_export_local_prefix_pending"
        )
        self.assertEqual(result["mode"], actions.MODE)
        self.assertEqual(
            result["bindings"]["receipt"],
            actions.binding(output / "export/receipt.json"),
        )
        self.assertEqual(
            actions.read(output / "result.json")["implementation_sha256"],
            actions.EXECUTOR_SHA,
        )
        self.assertTrue((output / "actions-export-result.json").is_file())


if __name__ == "__main__":
    unittest.main()
