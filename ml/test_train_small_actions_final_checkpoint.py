import contextlib
import io
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
    from safetensors.torch import save_file
    import torch
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "safetensors", "torch"}:
        raise
    raise unittest.SkipTest(
        "Checkpoint tests require the ML training dependencies."
    ) from error

import small_actions_final as data
import train_small_actions_final as training


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Module()
        self.base.lora_A = torch.nn.Linear(2, 2, bias=False)
        self.base.lora_B = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.base.lora_B.weight)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, values):
        return values + self.base.lora_B(self.base.lora_A(self.dropout(values)))

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def enable_input_require_grads(self):
        pass

    def save_pretrained(self, path, **kwargs):
        Path(path).mkdir()
        save_file(
            tiny_adapter_state(self, torch),
            str(Path(path) / "adapter_model.safetensors"),
        )


def tiny_adapter_state(model, torch):
    return {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }


def fixture():
    rows = [
        {
            "case_id": f"row-{index}",
            "messages": [],
            "input_ids": [1, index + 2],
            "completion_start": 1,
            "value": index,
        }
        for index in range(3)
    ]
    config = {
        "model": "tiny",
        "base": {"repository": "local/tiny", "revision": "frozen"},
        "training": {
            "seed": 42,
            "epochs": 3,
            "sample_order": [2, 0, 1, 1, 2, 0, 0, 1, 2],
            "gradient_accumulation": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "rank": 2,
            "alpha": 2,
            "dropout": 0.25,
        },
        "source_cases_sha256": "frozen-tiny-cohort",
        "train_rows_sha256": "frozen-tiny-tokens",
    }
    return config, rows


class TrainingCheckpointTests(unittest.TestCase):
    def run_tiny(
        self,
        output,
        *,
        checkpoint_every=0,
        checkpoint_dir=None,
        resume=None,
        fail_after=None,
        config=None,
    ):
        default_config, rows = fixture()
        config = config or default_config
        captured, draws = {}, []
        adamw = torch.optim.AdamW

        def load_base(config, assets, **kwargs):
            random.seed(config["training"]["seed"])
            np.random.seed(config["training"]["seed"])
            torch.manual_seed(config["training"]["seed"])
            captured["model"] = model = TinyAdapter()
            runtime = {
                "environment": {"torch": torch.__version__, "numpy": np.__version__},
                "base": config["base"],
                "dtype": "float32",
                "attention": "tiny",
                "targets": {"base": [2, 2]},
                "parameters": 8,
                "device": "cpu",
            }
            return torch, model, None, runtime

        def optimizer(*args, **kwargs):
            captured["optimizer"] = adamw(*args, **kwargs)
            return captured["optimizer"]

        def logps(model, row, torch):
            draws.append(row["case_id"])
            if fail_after == len(draws):
                raise RuntimeError("simulated loss interruption")
            values = torch.tensor([[row["value"] + 0.5, -0.3]])
            target = random.gauss(0.0, 0.1) + float(np.random.normal(0.0, 0.1))
            return -(model(values) - target).square().flatten()

        with (
            mock.patch.object(data, "prepared", return_value=(config, rows, output)),
            mock.patch.object(data, "prompt_ids", return_value=[1]),
            mock.patch.object(training, "load_base", side_effect=load_base),
            mock.patch.object(
                training, "attach_lora", side_effect=lambda model, *args: model
            ),
            mock.patch.object(
                training, "adapter_state", side_effect=tiny_adapter_state
            ),
            mock.patch.object(training, "completion_logps", side_effect=logps),
            mock.patch.object(training, "diagnostics", return_value=[]),
            mock.patch.object(torch.optim, "AdamW", side_effect=optimizer),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            if fail_after:
                with self.assertRaisesRegex(
                    RuntimeError, "simulated loss interruption"
                ):
                    training.train(
                        output / "unused",
                        output,
                        checkpoint_every=checkpoint_every,
                        checkpoint_dir=checkpoint_dir,
                        resume_checkpoint=resume,
                    )
                report = data.read(output / "run.json")
            else:
                report = training.train(
                    output / "unused",
                    output,
                    checkpoint_every=checkpoint_every,
                    checkpoint_dir=checkpoint_dir,
                    resume_checkpoint=resume,
                )
        return {
            "report": report,
            "draws": draws,
            "weights": tiny_adapter_state(captured["model"], torch),
            "optimizer": training.cpu_state(captured["optimizer"].state_dict(), torch),
            "rng": training.rng_state(torch),
            "next_random": [
                torch.rand(3).tolist(),
                random.random(),
                np.random.random(3).tolist(),
            ],
        }

    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(expected), type(actual))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual, strict=True):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_interrupted_optimizer_resume_matches_uninterrupted_weights_rng_and_budget(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.run_tiny(root / "baseline")
            interrupted = self.run_tiny(
                root / "interrupted", checkpoint_every=1, fail_after=6
            )
            checkpoint = root / "interrupted-checkpoints/checkpoint-000001"
            initial_checkpoint = root / "interrupted-checkpoints/checkpoint-000000"
            resumed_checkpoints = root / "durable-checkpoints"
            resumed = self.run_tiny(
                root / "resumed",
                checkpoint_every=2,
                checkpoint_dir=resumed_checkpoints,
                resume=checkpoint,
            )
            self.assertFalse((root / "baseline-checkpoints").exists())
            self.assertFalse((root / "resumed/checkpoints").exists())
            self.assertFalse((root / "resumed-checkpoints").exists())
            self.assertEqual(interrupted["report"]["status"], "failed_incomplete")
            self.assertEqual(interrupted["report"]["updates_completed"], 1)
            self.assertEqual(interrupted["report"]["draws_completed"], 4)
            self.assertEqual(len(interrupted["draws"]), 6)
            self.assertEqual(resumed["draws"], baseline["draws"][4:])
            self.assertEqual(resumed["report"]["updates_completed"], 3)
            self.assertEqual(resumed["report"]["draws_completed"], 9)
            self.assertEqual(
                resumed["report"]["initialization"], "checkpoint_adapter_optimizer_rng"
            )
            self.assertEqual(len(resumed["report"]["resume_history"]), 1)
            self.assertEqual(
                resumed["report"]["resumed_elapsed_seconds"],
                data.read(checkpoint / "checkpoint.json")["elapsed_seconds"],
            )
            self.assertGreater(
                resumed["report"]["seconds"],
                resumed["report"]["resumed_elapsed_seconds"],
            )
            for key in ("weights", "optimizer", "rng", "next_random"):
                self.assert_state_equal(baseline[key], resumed[key])
            history = data.rows(root / "resumed/updates.jsonl")
            self.assertEqual(history[:1], data.rows(checkpoint / "updates.jsonl"))
            self.assertEqual([len(event["case_ids"]) for event in history], [4, 4, 1])
            self.assertEqual([event["draws_completed"] for event in history], [4, 8, 9])
            self.assertEqual(
                sorted(path.name for path in resumed_checkpoints.iterdir()),
                ["checkpoint-000001", "checkpoint-000002", "checkpoint-000003"],
            )
            initial = torch.load(initial_checkpoint / "state.pt", weights_only=True)
            final = torch.load(
                resumed_checkpoints / "checkpoint-000003/state.pt", weights_only=True
            )
            self.assertEqual(initial["optimizer"]["state"], {})
            self.assert_state_equal(initial["initial_adapter"], initial["trainable"])
            self.assert_state_equal(
                initial["initial_adapter"], final["initial_adapter"]
            )
            self.assert_state_equal(baseline["optimizer"], final["optimizer"])
            restarted = self.run_tiny(root / "from-initial", resume=initial_checkpoint)
            self.assertEqual(restarted["draws"], baseline["draws"])
            self.assert_state_equal(baseline["weights"], restarted["weights"])

    def test_completed_checkpoint_exports_without_retraining_and_preserves_original_initial_adapter(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.run_tiny(root / "baseline", checkpoint_every=2)
            checkpoint = root / "baseline-checkpoints/checkpoint-000003"
            resumed = self.run_tiny(root / "resumed", resume=checkpoint)
            self.assertEqual(resumed["draws"], [])
            self.assertEqual(
                resumed["report"]["status"], "training_complete_reload_pending"
            )
            self.assertEqual(resumed["report"]["updates_completed"], 3)
            for key in ("weights", "optimizer", "rng", "next_random"):
                self.assert_state_equal(baseline[key], resumed[key])
            with self.assertRaises(FileExistsError):
                self.run_tiny(root / "resumed", resume=checkpoint)

    def test_restore_rejects_tampered_incomplete_or_incompatible_checkpoint_before_tensor_copy(
        self,
    ):
        mutations = {
            "tamper": "SHA256 differs",
            "incomplete": "file inventory",
            "configuration": "configuration differs",
            "trainer": "implementation differs",
            "runtime": "runtime settings differ",
            "cursor": "step boundary",
            "history": "history.*incomplete",
            "order": "sample order",
            "extra_tensor": "trainable tensor inventory",
            "missing_tensor": "trainable tensor inventory",
            "optimizer": "optimizer step count",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_tiny(root / "source", checkpoint_every=1, fail_after=6)
            source = root / "source-checkpoints/checkpoint-000001"
            for mutation, message in mutations.items():
                with self.subTest(mutation=mutation):
                    path = root / mutation / source.name
                    shutil.copytree(source, path)
                    manifest = data.read(path / "checkpoint.json")
                    config = None
                    if mutation == "tamper":
                        with (path / "state.pt").open("ab") as stream:
                            stream.write(b"changed")
                    elif mutation == "incomplete":
                        (path / "state.pt").unlink()
                    elif mutation == "configuration":
                        config, _ = fixture()
                        config["training"]["sample_order"] = list(
                            reversed(config["training"]["sample_order"])
                        )
                    elif mutation == "trainer":
                        manifest["implementation_sha256"][
                            "train_small_actions_final.py"
                        ] = "changed"
                    elif mutation == "runtime":
                        manifest["runtime"]["targets"] = {"other": [2, 2]}
                    elif mutation == "cursor":
                        manifest["draws_completed"] += 1
                    elif mutation in ("history", "order"):
                        history = data.rows(path / "updates.jsonl")
                        if mutation == "history":
                            history.clear()
                        else:
                            history[0]["case_ids"][0] = "foreign-row"
                        data.write_rows(path / "updates.jsonl", history)
                        manifest["files_sha256"]["updates.jsonl"] = data.sha(
                            path / "updates.jsonl"
                        )
                    else:
                        state = torch.load(path / "state.pt", weights_only=True)
                        if mutation == "extra_tensor":
                            state["trainable"]["foreign.weight"] = torch.ones(2, 2)
                        elif mutation == "missing_tensor":
                            del state["trainable"][next(iter(state["trainable"]))]
                        else:
                            state["optimizer"]["state"][0]["step"] += 1
                        torch.save(state, path / "state.pt")
                        manifest["files_sha256"]["state.pt"] = data.sha(
                            path / "state.pt"
                        )
                    data.write(path / "checkpoint.json", manifest)
                    with mock.patch.object(
                        torch.nn.Parameter,
                        "copy_",
                        side_effect=AssertionError("Restored before validation"),
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            self.run_tiny(
                                root / f"rejected-{mutation}",
                                resume=path,
                                config=config,
                            )
                    failed = data.read(root / f"rejected-{mutation}/run.json")
                    self.assertEqual(failed["status"], "failed_incomplete")
                    self.assertEqual(failed["updates_completed"], 0)
                    self.assertFalse((root / f"rejected-{mutation}/adapter").exists())

    def test_failed_checkpoint_write_never_publishes_a_directory_or_updates_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with mock.patch.object(
                torch, "save", side_effect=OSError("simulated disk failure")
            ):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    self.run_tiny(output, checkpoint_every=1)
            self.assertEqual(list(output.with_name("failed-checkpoints").iterdir()), [])
            info = data.read(output / "run.json")
            self.assertEqual(info["status"], "failed_incomplete")
            self.assertEqual(info["updates_completed"], 0)
            self.assertEqual(info["draws_completed"], 0)
            output = Path(temporary) / "inside-output"
            with self.assertRaisesRegex(
                ValueError, "outside the final training output"
            ):
                self.run_tiny(
                    output, checkpoint_every=1, checkpoint_dir=output / "checkpoints"
                )


if __name__ == "__main__":
    unittest.main()
