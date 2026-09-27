import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from export_passive_sft import (
    BASE_REPOSITORY,
    BASE_REVISION,
    VERSION,
    RUNTIME_REVISION,
    configuration,
    conversion_commands,
    native_inventory,
    preflight,
    quantized_changes,
    refuse_existing,
    tensor_header,
    training_census,
    validate_adapter,
    verify,
)
from export_qwen06 import sha256


def write_tensor_file(path, tensors):
    offset = 0
    header = {}
    for name, shape in tensors.items():
        size = 4
        for dimension in shape:
            size *= dimension
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


class PassiveSftExportTests(unittest.TestCase):
    def setUp(self):
        self.module = "model.language_model.layers.3.self_attn.q_proj"
        self.native = self.module + ".weight"
        self.prefix = "base_model.model." + self.module
        self.inventory = {
            self.native: {
                "shape": [8, 6],
                "dtype": "BF16",
                "shard": "model-1.safetensors",
            },
        }
        self.adapter = {
            self.prefix + ".lora_A.weight": {"shape": [2, 6], "dtype": "F32"},
            self.prefix + ".lora_B.weight": {"shape": [8, 2], "dtype": "F32"},
        }
        self.adapter_config = {
            "base_model_name_or_path": BASE_REPOSITORY,
            "revision": BASE_REVISION,
            "peft_type": "LORA",
            "bias": "none",
            "r": 2,
            "lora_alpha": 4,
            "target_modules": [self.module],
        }

    def validate(self, config=None, inventory=None, adapter=None):
        return validate_adapter(
            self.adapter_config if config is None else config,
            self.inventory if inventory is None else inventory,
            self.adapter if adapter is None else adapter,
            32,
            BASE_REPOSITORY,
        )

    def test_actual_sharded_headers_must_match_the_complete_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensor_file(root / "a.safetensors", {self.native: [8, 6]})
            second = "model.language_model.layers.4.mlp.up_proj.weight"
            write_tensor_file(root / "b.safetensors", {second: [12, 6]})
            weight_map = {self.native: "a.safetensors", second: "b.safetensors"}
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": weight_map}))
            files = {
                name: "unused"
                for name in (index.name, "a.safetensors", "b.safetensors")
            }
            result = native_inventory(root, files)
            self.assertEqual(result[self.native]["shape"], [8, 6])
            self.assertEqual(result[second]["shard"], "b.safetensors")
            weight_map[self.native] = "b.safetensors"
            index.write_text(json.dumps({"weight_map": weight_map}))
            with self.assertRaises(ValueError):
                native_inventory(root, files)
            with (root / "a.safetensors").open("ab") as stream:
                stream.write(b"unexpected trailing bytes")
            with self.assertRaises(ValueError):
                tensor_header(root / "a.safetensors")

    def test_adapter_coverage_shape_and_real_layer_count_are_checked(self):
        pairs = self.validate()
        self.assertEqual(pairs[self.native]["gguf_name"], "blk.3.attn_q.weight")
        for target in (
            ["q_proj"],
            r"model\.language_model\.layers\.3\.self_attn\.q_proj",
        ):
            config = {**self.adapter_config, "target_modules": target}
            self.assertEqual(self.validate(config=config), pairs)
        malformed = copy.deepcopy(self.adapter)
        malformed[self.prefix + ".lora_B.weight"]["shape"] = [7, 2]
        with self.assertRaises(ValueError):
            self.validate(adapter=malformed)
        with self.assertRaises(ValueError):
            self.validate(
                adapter={
                    self.prefix + ".lora_A.weight": self.adapter[
                        self.prefix + ".lora_A.weight"
                    ]
                }
            )
        with self.assertRaises(ValueError):
            validate_adapter(
                self.adapter_config, self.inventory, self.adapter, 3, BASE_REPOSITORY
            )
        extra = {
            **self.inventory,
            "model.language_model.layers.7.self_attn.q_proj.weight": self.inventory[
                self.native
            ],
        }
        with self.assertRaises(ValueError):
            self.validate(
                config={**self.adapter_config, "target_modules": ["q_proj"]},
                inventory=extra,
            )

    def test_saved_peft_suffixes_use_bound_runtime_census_and_explicit_mtp_exclusions(
        self,
    ):
        mtp = [
            "mtp.fc.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.mlp.down_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ]
        inventory = {
            **self.inventory,
            **{name: self.inventory[self.native] for name in mtp},
        }
        adapter_config = {**self.adapter_config, "target_modules": ["q_proj"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {"repository": BASE_REPOSITORY, "revision": BASE_REVISION}
            training = {
                "base": base,
                "training": {"rank": 2, "alpha": 4},
                "model": {
                    "target_modules": [self.module],
                    "ignored_native_tensor_names": mtp,
                },
            }
            runtime = {
                "targets": {
                    self.module: {
                        "class": "Linear4bit",
                        "in_features": 6,
                        "out_features": 8,
                    }
                },
                "ignored_native_tensor_names": mtp,
            }
            (root / "configuration.json").write_text(json.dumps(training))
            (root / "runtime.json").write_text(json.dumps(runtime))
            run = {
                "mode": "train",
                "training_updates": 2,
                "status": "training_complete_fresh_reload_pending",
                "configuration_sha256": sha256(root / "configuration.json"),
            }
            (root / "run.json").write_text(json.dumps(run))
            adapter_hashes = {
                "adapter_config.json": "a" * 64,
                "adapter_model.safetensors": "b" * 64,
            }
            files = {
                name: sha256(root / name)
                for name in ("run.json", "configuration.json", "runtime.json")
            }
            files.update(
                {
                    "adapter/" + name: expected
                    for name, expected in adapter_hashes.items()
                }
            )
            (root / "manifest.json").write_text(json.dumps({"files_sha256": files}))
            config = {
                "training_record": {
                    "path": str(root / "run.json"),
                    "sha256": files["run.json"],
                },
                "base": base,
                "adapter": {"files_sha256": adapter_hashes},
            }
            census = training_census(config, inventory)
            self.assertEqual(len(census["ignored_native_tensor_names"]), 15)
            kwargs = {
                "trained_modules": list(census["targets"]),
                "ignored_native_names": census["ignored_native_tensor_names"],
            }
            pairs = validate_adapter(
                adapter_config, inventory, self.adapter, 32, BASE_REPOSITORY, **kwargs
            )
            self.assertEqual(set(pairs), {self.native})
            with self.assertRaisesRegex(ValueError, "MTP exclusions"):
                validate_adapter(
                    adapter_config,
                    inventory,
                    self.adapter,
                    32,
                    BASE_REPOSITORY,
                    **{**kwargs, "ignored_native_names": mtp[:-1]},
                )
            with self.assertRaisesRegex(ValueError, "trained text module census"):
                validate_adapter(
                    adapter_config,
                    inventory,
                    self.adapter,
                    32,
                    BASE_REPOSITORY,
                    **{**kwargs, "trained_modules": []},
                )
            rogue = {
                **inventory,
                "model.visual.self_attn.q_proj.weight": self.inventory[self.native],
            }
            with self.assertRaisesRegex(ValueError, "Only ordinary text"):
                validate_adapter(
                    adapter_config, rogue, self.adapter, 32, BASE_REPOSITORY, **kwargs
                )
            runtime["targets"][self.module]["out_features"] = 9
            (root / "runtime.json").write_text(json.dumps(runtime))
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                training_census(config, inventory)
            files["runtime.json"] = sha256(root / "runtime.json")
            (root / "manifest.json").write_text(json.dumps({"files_sha256": files}))
            with self.assertRaisesRegex(ValueError, "Native tensor shape differs"):
                training_census(config, inventory)

    def test_unsupported_adapter_semantics_and_nontext_targets_are_rejected(self):
        for key, value in (
            ("revision", "0" * 40),
            ("base_model_name_or_path", "another-model"),
            ("fan_in_fan_out", True),
            ("use_dora", True),
            ("use_rslora", True),
            ("rank_pattern", {"q_proj": 8}),
            ("modules_to_save", ["lm_head"]),
            ("init_lora_weights", "pissa"),
            ("exclude_modules", ["q_proj"]),
            ("r", 0),
            ("lora_alpha", float("nan")),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(config={**self.adapter_config, key: value})
        for module in (
            "model.visual.layers.3.self_attn.q_proj",
            "model.language_model.layers.3.linear_attn.out_proj",
            "model.language_model.embed_tokens",
        ):
            with self.subTest(module=module), self.assertRaises(ValueError):
                self.validate(
                    config={**self.adapter_config, "target_modules": [module]},
                    inventory={module + ".weight": self.inventory[self.native]},
                    adapter={},
                )

    def config(self, root):
        return {
            "version": VERSION,
            "base": {
                "repository": BASE_REPOSITORY,
                "revision": BASE_REVISION,
                "path": "base",
                "files_sha256": {},
            },
            "adapter": {"path": "adapter", "files_sha256": {}},
            "training_record": {"path": "run.json", "sha256": "0" * 64},
            "runtime": {
                "revision": RUNTIME_REVISION,
                "source_path": "llama",
                "python": "python",
                "source_files_sha256": {},
                "support_files_sha256": {},
                "tools": {
                    name: {"path": name, "sha256": "0" * 64}
                    for name in ("llama-export-lora", "llama-quantize")
                },
            },
        }

    def test_plan_uses_exact_native_base_and_paired_f16_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "export.json"
            path.write_text(json.dumps(self.config(root)))
            config = configuration(path)
            commands = conversion_commands(config, root / "output")
            base = commands[0][1]
            adapter = commands[1][1]
            merge = commands[2][1]
            self.assertEqual(base[2], str((root / "base").resolve()))
            self.assertEqual(adapter[adapter.index("--base") + 1], base[2])
            self.assertEqual(base[-1], "f16")
            self.assertEqual(adapter[-1], "f32")
            self.assertEqual(
                merge[merge.index("-m") + 1], base[base.index("--outfile") + 1]
            )
            self.assertEqual(
                merge[merge.index("--lora") + 1],
                adapter[adapter.index("--outfile") + 1],
            )
            config["base"]["revision"] = "0" * 40
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                configuration(path)

    def test_converter_preserves_virtual_environment_interpreter_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base_python = root / "base-python"
            base_python.write_text("fixture executable")
            venv_python = root / ".venv/bin/python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(base_python)
            path = root / "export.json"
            for interpreter in (".venv/bin/../bin/python", str(venv_python)):
                with self.subTest(interpreter=interpreter):
                    value = self.config(root)
                    value["runtime"]["python"] = interpreter
                    path.write_text(json.dumps(value))
                    config = configuration(path)
                    self.assertEqual(config["runtime"]["python"], str(venv_python))
                    self.assertTrue(Path(config["runtime"]["python"]).is_file())
                    self.assertNotEqual(config["runtime"]["python"], str(base_python))
                    for _, command in conversion_commands(config, root / "output")[:2]:
                        self.assertEqual(command[0], str(venv_python))

    def test_failed_preflight_and_existing_outputs_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "export.json"
            path.write_text(json.dumps(self.config(root)))
            output = root / "attempt"
            with patch(
                "export_passive_sft.validate_inputs",
                side_effect=ValueError("invalid shard"),
            ):
                with self.assertRaisesRegex(ValueError, "invalid shard"):
                    preflight(path, output)
            original = (output / "preflight-error.json").read_bytes()
            with self.assertRaises(FileExistsError):
                preflight(path, output)
            self.assertEqual((output / "preflight-error.json").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                refuse_existing(output / "configuration.json")

    @unittest.skipUnless(
        importlib.util.find_spec("numpy"),
        "Use the existing export venv for NumPy tensor checks.",
    )
    def test_quantization_can_erase_target_deltas_but_cannot_change_untouched_tensors(
        self,
    ):
        import numpy as np

        def tensor(values, dtype=12):
            return SimpleNamespace(
                tensor_type=dtype,
                shape=np.array([2]),
                data=np.array(values, dtype=np.uint8),
            )

        base = {
            name: tensor([1, 2]) for name in ("adapted-a", "adapted-b", "untouched")
        }
        merged = {**base, "adapted-a": tensor([1, 3])}
        self.assertEqual(
            quantized_changes(base, merged, {"adapted-a", "adapted-b"}), ["adapted-a"]
        )
        self.assertEqual(quantized_changes(base, base, {"adapted-a", "adapted-b"}), [])
        with self.assertRaises(ValueError):
            quantized_changes(
                base,
                {**merged, "untouched": tensor([1, 3])},
                {"adapted-a", "adapted-b"},
            )
        with self.assertRaises(ValueError):
            quantized_changes(
                base,
                {**merged, "adapted-a": tensor([1, 3], dtype=1)},
                {"adapted-a", "adapted-b"},
            )

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name)
            for name in ("torch", "peft", "safetensors", "numpy")
        ),
        "Use the existing export venv for the tiny native/PEFT oracle check.",
    )
    def test_tiny_native_adapter_merge_oracle_and_untouched_tensors(self):
        import numpy as np
        import torch
        from safetensors.torch import save_file

        def tensor(name, data, dtype):
            return SimpleNamespace(
                name=name,
                data=data,
                shape=np.array(data.shape[::-1]),
                tensor_type=dtype,
            )

        def reader(tensors, **extra):
            values = {
                "general.architecture": "qwen35",
                "tokenizer.chat_template": "template",
                "tokenizer.ggml.eos_token_id": 7,
                **extra,
            }
            fields = {
                name: SimpleNamespace(contents=lambda value=value: value)
                for name, value in values.items()
            }
            return SimpleNamespace(tensors=tensors, fields=fields, get_field=fields.get)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_dir, adapter_dir, output = (
                root / "base",
                root / "adapter",
                root / "export",
            )
            for path in (base_dir, adapter_dir, output):
                path.mkdir()
            (base_dir / "chat_template.jinja").write_text("template")
            base = torch.arange(48).reshape(8, 6).float() / 32
            a, b = torch.ones(2, 6) / 16, torch.ones(8, 2) / 8
            save_file({self.native: base}, base_dir / "model-1.safetensors")
            save_file(
                {self.prefix + ".lora_A.weight": a, self.prefix + ".lora_B.weight": b},
                adapter_dir / "adapter_model.safetensors",
            )
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps(self.adapter_config)
            )
            name = "blk.3.attn_q.weight"
            untouched = tensor(
                "blk.3.attn_norm.weight", np.ones(6, dtype=np.float32), 0
            )
            merged = (base.half().float() + 2 * (b @ a)).half().numpy()
            readers = {
                "base-F16.gguf": reader(
                    [tensor(name, base.half().numpy(), 1), untouched]
                ),
                "adapter-F32.gguf": reader(
                    [
                        tensor(name + ".lora_a", a.numpy(), 0),
                        tensor(name + ".lora_b", b.numpy(), 0),
                    ],
                    **{"adapter.lora.alpha": 4},
                ),
                "merged-F16.gguf": reader([tensor(name, merged, 1), untouched]),
            }
            artifacts = {}
            for key, filename in (
                ("base", "base-F16.gguf"),
                ("adapter", "adapter-F32.gguf"),
                ("merged", "merged-F16.gguf"),
            ):
                path = output / filename
                path.write_bytes(b"mock GGUF transport")
                artifacts[key] = {"path": str(path), "sha256": sha256(path)}
            (output / "conversion.json").write_text(
                json.dumps({"artifacts": artifacts})
            )
            config = {
                "base": {"path": str(base_dir)},
                "adapter": {"path": str(adapter_dir)},
            }
            preflight_report = {
                "pairs": self.validate(),
                "rank": 2,
                "alpha": 4,
                "template_sha256": sha256(base_dir / "chat_template.jinja"),
            }
            fake_gguf = SimpleNamespace(
                GGUFReader=lambda path: readers[Path(path).name],
                GGMLQuantizationType=SimpleNamespace(F16=1, F32=0),
            )
            with (
                patch(
                    "export_passive_sft.bound_run",
                    return_value=(config, preflight_report),
                ),
                patch("export_passive_sft.gguf_module", return_value=fake_gguf),
            ):
                result = verify(root / "config.json", output)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["native_targets_exact"], 1)
            self.assertGreater(result["merged_targets"][0]["changed_elements"], 0)
            self.assertEqual(
                result["unadapted_tensors_exact"], ["blk.3.attn_norm.weight"]
            )


if __name__ == "__main__":
    unittest.main()
