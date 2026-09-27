import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import export_passive_gemma as export


def native_config():
    return {
        "model_type": "gemma4",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 30,
            "attention_k_eq_v": True,
            "layer_types": [
                "full_attention" if index % 6 == 5 else "sliding_attention"
                for index in range(30)
            ],
        },
    }


class GemmaExportTests(unittest.TestCase):
    def fixture(self):
        names = [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.5.self_attn.k_proj",
            "model.language_model.layers.0.mlp.down_proj",
        ]
        config = {
            "base_model_name_or_path": export.BASE_REPOSITORY,
            "revision": export.BASE_REVISION,
            "peft_type": "LORA",
            "bias": "none",
            "r": 2,
            "lora_alpha": 2,
            "target_modules": ["q_proj", "k_proj", "down_proj"],
        }
        inventory, tensors = {}, {}
        for module in names:
            inventory[module + ".weight"] = {
                "shape": [4, 6],
                "dtype": "BF16",
                "shard": "model.safetensors",
            }
            for side, shape in (("A", [2, 6]), ("B", [4, 2])):
                tensors[f"base_model.model.{module}.lora_{side}.weight"] = {
                    "shape": shape,
                    "dtype": "F32",
                }
        inventory["model.vision_tower.layers.0.self_attn.q_proj.weight"] = {
            "shape": [4, 6],
            "dtype": "BF16",
            "shard": "model.safetensors",
        }
        inventory["model.language_model.layers.0.experts.down_proj"] = {
            "shape": [128, 4, 6],
            "dtype": "BF16",
            "shard": "model.safetensors",
        }
        return names, config, inventory, tensors

    def test_exact_target_policy_has_205_modules_with_five_shared_kv_layers(self):
        targets = export.target_names(native_config())
        self.assertEqual(len(targets), 205)
        for layer in (5, 11, 17, 23, 29):
            self.assertIn(
                f"model.language_model.layers.{layer}.self_attn.k_proj.weight", targets
            )
            self.assertNotIn(
                f"model.language_model.layers.{layer}.self_attn.v_proj.weight", targets
            )
        self.assertFalse(
            any(
                "experts" in name or "router" in name or "vision" in name
                for name in targets
            )
        )
        changed = native_config()
        changed["text_config"]["layer_types"][5] = "sliding_attention"
        with self.assertRaisesRegex(ValueError, "205"):
            export.target_names(changed)

    def test_saved_suffixes_expand_only_in_recorded_text_namespace(self):
        names, config, inventory, tensors = self.fixture()
        pairs = export.validate_adapter(
            config,
            inventory,
            tensors,
            30,
            export.BASE_REPOSITORY,
            trained_modules=names,
        )
        self.assertEqual(set(pairs), {name + ".weight" for name in names})
        self.assertEqual(
            pairs[names[1] + ".weight"]["gguf_name"], "blk.5.attn_k.weight"
        )
        self.assertEqual(
            pairs[names[2] + ".weight"]["gguf_name"], "blk.0.ffn_down.weight"
        )
        with self.assertRaisesRegex(ValueError, "module census"):
            export.validate_adapter(
                config,
                inventory,
                tensors,
                30,
                export.BASE_REPOSITORY,
                trained_modules=names[:-1],
            )

    def test_expert_vision_wrong_shape_or_revision_adapters_are_rejected(self):
        names, config, inventory, tensors = self.fixture()
        for unsupported in (
            "model.vision_tower.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.experts.down_proj",
            "model.language_model.layers.0.router.proj",
        ):
            changed = {
                **tensors,
                f"base_model.model.{unsupported}.lora_A.weight": {
                    "shape": [2, 6],
                    "dtype": "F32",
                },
            }
            with self.subTest(unsupported=unsupported), self.assertRaises(ValueError):
                export.validate_adapter(
                    config,
                    inventory,
                    changed,
                    30,
                    export.BASE_REPOSITORY,
                    trained_modules=names,
                )
        changed = copy.deepcopy(tensors)
        changed[next(iter(changed))]["shape"] = [2, 5]
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            export.validate_adapter(
                config,
                inventory,
                changed,
                30,
                export.BASE_REPOSITORY,
                trained_modules=names,
            )
        with self.assertRaisesRegex(ValueError, "revision"):
            export.validate_adapter(
                {**config, "revision": None},
                inventory,
                tensors,
                30,
                export.BASE_REPOSITORY,
                trained_modules=names,
            )

    def test_native_pin_preserves_all_three_stops_without_inventing_license_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "native"
            base.mkdir()
            generation = base / "generation_config.json"
            generation.write_text(
                json.dumps(
                    {"eos_token_id": [1, 106, 50], "bos_token_id": 2, "pad_token_id": 0}
                )
            )
            api = root / "native-api.json"
            api.write_text("{}")
            pin_path = root / "native-pin.json"
            files = {"generation_config.json": {"sha256": export.sha256(generation)}}
            pin = {
                "version": "passive-native-pin-v1",
                "candidate_id": "gemma26",
                "repository": export.BASE_REPOSITORY,
                "revision": export.BASE_REVISION,
                "files": files,
                "api_sha256": export.sha256(api),
            }
            pin_path.write_text(json.dumps(pin))
            config = {
                "native_pin": {
                    "path": str(pin_path),
                    "sha256": export.sha256(pin_path),
                },
                "base": {
                    "path": str(base),
                    "files_sha256": {
                        name: value["sha256"] for name, value in files.items()
                    },
                },
            }
            export.validate_native_pin(config)
            generation.write_text(
                json.dumps(
                    {"eos_token_id": [1, 106], "bos_token_id": 2, "pad_token_id": 0}
                )
            )
            with self.assertRaisesRegex(ValueError, "terminal IDs"):
                export.validate_native_pin(config)

    def test_pinned_converter_preserves_all_410_synthetic_adapter_tensors(self):
        root = Path(__file__).resolve().parent.parent
        python = root / "artifacts/model-preparation/gemma26-v1/.venv/bin/python"
        source = (
            root
            / f"artifacts/model-preparation/qwen06-v1/llama.cpp-{export.RUNTIME_REVISION}"
        )
        metadata = (
            root
            / "artifacts/training/passive-t1-gemma26-smoke-v1/native-acquisition/native/config.json"
        )
        if not (python.is_file() and source.is_dir() and metadata.is_file()):
            self.skipTest("Pinned local converter environment/metadata not staged.")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            script = target / "check.py"
            script.write_text("""import json, sys, subprocess
from pathlib import Path
import torch
from safetensors.torch import save_file
root,source,metadata=map(Path,sys.argv[1:])
sys.path.insert(0,str(source/'gguf-py'))
import gguf
base=root/'base';base.mkdir();base.joinpath('config.json').write_bytes(metadata.read_bytes())
config=json.loads(metadata.read_text());text=config['text_config'];hidden=text['hidden_size'];rank=1
adapter=root/'adapter';adapter.mkdir();tensors={};expected={}
for layer,kind in enumerate(text['layer_types']):
 full=kind=='full_attention';head=text['global_head_dim'] if full else text['head_dim'];kv=text['num_global_key_value_heads'] if full else text['num_key_value_heads']
 targets={'self_attn.q_proj':(text['num_attention_heads']*head,hidden,'attn_q'),'self_attn.k_proj':(kv*head,hidden,'attn_k'),'self_attn.o_proj':(hidden,text['num_attention_heads']*head,'attn_output'),'mlp.gate_proj':(text['intermediate_size'],hidden,'ffn_gate'),'mlp.up_proj':(text['intermediate_size'],hidden,'ffn_up'),'mlp.down_proj':(hidden,text['intermediate_size'],'ffn_down')}
 if not full:targets['self_attn.v_proj']=(kv*head,hidden,'attn_v')
 for name,(outdim,indim,mapped) in targets.items():
  module=f'model.language_model.layers.{layer}.{name}'
  for side,shape in (('A',(rank,indim)),('B',(outdim,rank))):
   value=torch.arange(shape[0]*shape[1],dtype=torch.float32).reshape(shape)/100000
   tensors[f'base_model.model.{module}.lora_{side}.weight']=value
   expected[f'blk.{layer}.{mapped}.weight.lora_{side.lower()}']=value.numpy()
assert len(tensors)==410
save_file(tensors,str(adapter/'adapter_model.safetensors'))
(adapter/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':rank,'lora_alpha':rank,'bias':'none'}))
out=root/'adapter.gguf'
result=subprocess.run([sys.executable,str(source/'convert_lora_to_gguf.py'),str(adapter),'--base',str(base),'--outfile',str(out),'--outtype','f32'],capture_output=True,text=True)
if result.returncode:raise RuntimeError(result.stdout+result.stderr)
reader=gguf.GGUFReader(out)
assert reader.get_field('general.architecture').contents()=='gemma4'
assert {t.name for t in reader.tensors}==set(expected)
for tensor in reader.tensors:
 assert tensor.tensor_type==gguf.GGMLQuantizationType.F32
 assert tensor.data.shape==expected[tensor.name].shape
 assert (tensor.data==expected[tensor.name]).all(),tensor.name
print(json.dumps({'adapter_tensors':len(reader.tensors),'bytes':out.stat().st_size,'status':'passed'}))
""")
            result = subprocess.run(
                [str(python), str(script), str(target), str(source), str(metadata)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)["adapter_tensors"], 410)


if __name__ == "__main__":
    unittest.main()
