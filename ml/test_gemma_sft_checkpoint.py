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

import gemma_sft_checkpoint as recovery
import train_gemma_actions_v3 as trainer


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Linear(2, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, value):
        return value + self.lora_B(self.lora_A(self.dropout(value)))

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        path.mkdir()
        save_file(adapter(self), str(path / "adapter_model.safetensors"))
        (path / "adapter_config.json").write_text("{}\n")


def adapter(model, **kwargs):
    return {
        name: value.detach().cpu().clone() for name, value in model.named_parameters()
    }


def fixture(count=11):
    rows = [
        {
            "case_id": f"case-{i % 4}",
            "draw_index": i,
            "source_id": "source",
            "source_group_id": "group",
            "episode_id": f"episode-{i % 2}",
            "language": "en" if i % 2 else "ko",
            "action": "correct_text",
            "input_ids": [1, 2, i % 4 + 3],
            "completion_start": 1,
        }
        for i in range(count)
    ]
    config = {
        "format": "json",
        "native_pin_sha256": "b" * 64,
        "sources_sha256": {"trainer.py": "a" * 64},
        "training": {
            "draw_count": count,
            "max_updates": (count + 3) // 4,
            "gradient_accumulation": 4,
            "sample_order": [row["case_id"] for row in rows],
            "seed": 42,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "checkpoint_every_updates": 1,
            "wall_limit_seconds": 10000,
        },
    }
    return config, rows


class GemmaCheckpointTests(unittest.TestCase):
    def run_tiny(
        self,
        output,
        *,
        resume=None,
        fail_after=None,
        publish_failure=None,
        config=None,
        expected_error=None,
    ):
        config, rows = (
            (config, fixture(config["training"]["draw_count"])[1])
            if config
            else fixture()
        )
        preparation = output.parent / (output.name + "-inputs")
        preparation.mkdir()
        path = preparation / "configuration.json"
        trainer.write(path, config)
        config_sha = trainer.digest(path)
        captured, draws = {}, []
        original_adamw, original_write = torch.optim.AdamW, recovery.write

        def load_runtime(*args):
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            model = captured["model"] = TinyAdapter()
            captured["initial"] = adapter(model)
            tokenizer = mock.Mock()
            tokenizer.save_pretrained.side_effect = lambda path: (
                Path(path).mkdir(),
                (Path(path) / "tokenizer.json").write_text("{}"),
            )
            return (
                torch,
                model,
                tokenizer,
                "cpu",
                {"gpu": "cpu-fixture", "targets": ["lora_A", "lora_B"]},
            )

        def optimizer(*args, **kwargs):
            captured["optimizer"] = original_adamw(*args, **kwargs)
            return captured["optimizer"]

        def loss(model, row, *_):
            draws.append(row["draw_index"])
            if fail_after == len(draws):
                raise RuntimeError("simulated mid-accumulation loss interruption")
            target = random.gauss(0, 0.1) + float(np.random.normal(0, 0.1))
            return (
                (model(torch.tensor([[row["draw_index"] + 0.5, -0.3]])) - target)
                .square()
                .mean()
            )

        def checkpoint_write(path, value):
            if (
                Path(path).name == "checkpoint.json"
                and value["updates_completed"] == publish_failure
            ):
                raise OSError("simulated publication interruption")
            return original_write(path, value)

        verified = {
            "configuration_sha256": config_sha,
            "preparation_sha256": "c" * 64,
            "environment": {"matched": True},
            "train_rows": 4,
            "draws": len(rows),
            "final_test_bodies_read": False,
        }
        with (
            mock.patch.object(
                trainer,
                "preflight",
                return_value=(
                    config,
                    preparation,
                    {"tokens": {"json": []}},
                    rows,
                    [],
                    verified,
                ),
            ),
            mock.patch.object(trainer.gemma, "load_runtime", side_effect=load_runtime),
            mock.patch.object(
                trainer.gemma,
                "enable_training",
                side_effect=lambda model, *_: (
                    model,
                    {n: p.numel() for n, p in model.named_parameters()},
                ),
            ),
            mock.patch.object(trainer.gemma, "generate_smoke", return_value=[]),
            mock.patch.object(trainer.smoke, "completion_loss", side_effect=loss),
            mock.patch("peft.get_peft_model_state_dict", side_effect=adapter),
            mock.patch.object(torch.optim, "AdamW", side_effect=optimizer),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0),
            mock.patch.object(torch.cuda, "max_memory_reserved", return_value=0),
            mock.patch.object(recovery, "write", side_effect=checkpoint_write),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            if fail_after or publish_failure is not None or expected_error:
                with self.assertRaises(
                    expected_error
                    or (OSError if publish_failure is not None else RuntimeError)
                ):
                    trainer.run(path, output, resume_checkpoint=resume)
                report = trainer.read(output / "run.json")
            else:
                report = trainer.run(path, output, resume_checkpoint=resume)
        return {
            "report": report,
            "draws": draws,
            "initial": captured["initial"],
            "weights": adapter(captured["model"]),
            "optimizer": recovery.primitives.cpu_state(
                captured["optimizer"].state_dict(), torch
            ),
            "rng": recovery.primitives.rng_state(torch),
            "next_random": [
                torch.rand(3).tolist(),
                random.random(),
                np.random.random(3).tolist(),
            ],
        }

    def assert_state_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_state_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(type(a), type(b))
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b, strict=True):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_mid_group_loss_resume_matches_real_adamw_rng_order_and_last_three_rows(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.run_tiny(root / "baseline")
            interrupted = self.run_tiny(root / "interrupted", fail_after=6)
            checkpoint = root / "interrupted-checkpoints/checkpoint-000001"
            resumed = self.run_tiny(root / "resumed", resume=checkpoint)
            self.assertEqual(interrupted["report"]["status"], "failed")
            self.assertEqual(interrupted["report"]["training_updates"], 1)
            self.assertEqual(interrupted["draws"], list(range(6)))
            self.assertEqual(resumed["draws"], list(range(4, 11)))
            self.assertEqual(resumed["report"]["training_updates"], 3)
            self.assertEqual(resumed["report"]["optimizer_committed_rows"], 11)
            self.assertTrue(resumed["report"]["optimizer_state_resumed"])
            for field in ["weights", "optimizer", "rng", "next_random"]:
                self.assert_state_equal(baseline[field], resumed[field])
            history = trainer.read(
                root / "resumed-checkpoints/checkpoint-000003/history.json"
            )
            self.assertEqual(
                [event["microbatches"] for event in history["updates"]], [4, 4, 3]
            )
            self.assertEqual(
                [event["draw_index"] for event in history["rows"]], list(range(11))
            )
            self.assertEqual(
                history["rows"][:4], trainer.read(checkpoint / "history.json")["rows"]
            )
            self.assertEqual(
                trainer.read(root / "resumed/initialization.json"),
                trainer.read(root / "baseline/initialization.json"),
            )
            trainer.verified_run(root / "resumed")
            self.assertFalse(
                any(
                    p.name.startswith("checkpoint-")
                    for p in (root / "resumed").iterdir()
                )
            )

    def test_initial_and_completed_checkpoint_restore_and_cumulative_wall_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self.run_tiny(root / "baseline")
            initial = self.run_tiny(
                root / "from-initial",
                resume=root / "baseline-checkpoints/checkpoint-000000",
            )
            self.assertEqual(initial["draws"], list(range(11)))
            self.assert_state_equal(initial["weights"], baseline["weights"])
            complete = root / "baseline-checkpoints/checkpoint-000003"
            manifest = trainer.read(complete / "checkpoint.json")
            manifest["elapsed_seconds"] = 10000
            trainer.write(complete / "checkpoint.json", manifest)
            final = self.run_tiny(root / "from-final", resume=complete)
            self.assertEqual(final["draws"], [])
            self.assertEqual(final["report"]["status"], trainer.TRAINED)
            self.assertGreaterEqual(final["report"]["seconds"], 10000)
            for field in ["weights", "optimizer", "rng", "next_random"]:
                self.assert_state_equal(baseline[field], final[field])
            trainer.verified_run(root / "from-final")
            partial = root / "baseline-checkpoints/checkpoint-000001"
            manifest = trainer.read(partial / "checkpoint.json")
            manifest["elapsed_seconds"] = 10000
            trainer.write(partial / "checkpoint.json", manifest)
            stopped = self.run_tiny(
                root / "budget-exhausted",
                resume=partial,
                expected_error=trainer.curriculum.WallLimitExceeded,
            )
            self.assertEqual(stopped["draws"], [])
            self.assertEqual(stopped["report"]["training_updates"], 1)

    def test_invalid_checkpoint_is_rejected_before_weights_or_optimizer_are_restored(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_tiny(root / "baseline")
            source = root / "baseline-checkpoints/checkpoint-000001"
            for mode in [
                "file_tamper",
                "source",
                "runtime",
                "cursor",
                "history",
                "exposure",
                "initialization",
                "optimizer",
                "rng",
                "extra_tensor",
                "missing_tensor",
                "incomplete",
                "configuration",
            ]:
                with self.subTest(mode=mode):
                    checkpoint = root / mode / source.name
                    shutil.copytree(source, checkpoint)
                    manifest = trainer.read(checkpoint / "checkpoint.json")
                    config = None
                    if mode == "file_tamper":
                        with (checkpoint / "state.pt").open("ab") as stream:
                            stream.write(b"changed")
                    elif mode == "incomplete":
                        (checkpoint / "runtime.json").unlink()
                    elif mode == "source":
                        manifest["sources_sha256"]["trainer.py"] = "d" * 64
                    elif mode == "runtime":
                        manifest["runtime"]["gpu"] = "different"
                    elif mode == "cursor":
                        manifest["draws_completed"] = 3
                    elif mode == "exposure":
                        manifest["exposure"]["optimizer_commit_uncertain"] = True
                    elif mode == "initialization":
                        manifest["initialization"]["initialization"] = "unknown"
                    elif mode == "configuration":
                        config = fixture()[0]
                        config["training"]["learning_rate"] = 0.02
                    elif mode == "history":
                        history = trainer.read(checkpoint / "history.json")
                        history["rows"][0]["draw_index"] = 1
                        trainer.write(checkpoint / "history.json", history)
                        manifest["files_sha256"]["history.json"] = trainer.digest(
                            checkpoint / "history.json"
                        )
                    else:
                        payload = torch.load(checkpoint / "state.pt", weights_only=True)
                        if mode == "optimizer":
                            next(iter(payload["optimizer"]["state"].values()))[
                                "step"
                            ].fill_(2)
                        elif mode == "rng":
                            del payload["rng"]["numpy"]
                        elif mode == "extra_tensor":
                            payload["trainable"]["extra"] = torch.zeros(2)
                        elif mode == "missing_tensor":
                            payload["trainable"].pop(next(iter(payload["trainable"])))
                        torch.save(payload, checkpoint / "state.pt")
                        manifest["files_sha256"]["state.pt"] = trainer.digest(
                            checkpoint / "state.pt"
                        )
                    trainer.write(checkpoint / "checkpoint.json", manifest)
                    failed = self.run_tiny(
                        root / ("rejected-" + mode),
                        resume=checkpoint,
                        config=config,
                        expected_error=ValueError,
                    )
                    self.assertEqual(failed["draws"], [])
                    self.assertEqual(failed["optimizer"]["state"], {})
                    self.assert_state_equal(failed["weights"], failed["initial"])

    def test_completed_946_update_checkpoint_finishes_without_any_new_draw_or_optimizer_step(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(3783)
            config["training"]["checkpoint_every_updates"] = 946
            baseline = self.run_tiny(root / "baseline", config=config)
            complete = root / "baseline-checkpoints/checkpoint-000946"
            resumed = self.run_tiny(root / "resumed", config=config, resume=complete)
            self.assertEqual(resumed["draws"], [])
            self.assertEqual(resumed["report"]["training_updates"], 946)
            self.assertEqual(resumed["report"]["optimizer_committed_rows"], 3783)
            history = trainer.read(complete / "history.json")
            self.assertEqual(history["updates"][-1]["microbatches"], 3)
            self.assertEqual(history["rows"][-1]["draw_index"], 3782)
            for field in ["weights", "optimizer", "rng", "next_random"]:
                self.assert_state_equal(baseline[field], resumed[field])
            self.assertTrue(
                all(
                    value["step"].item() == 946
                    for value in resumed["optimizer"]["state"].values()
                )
            )
            trainer.verified_run(root / "resumed")

    def test_atomic_publication_failure_keeps_previous_checkpoint_and_rejects_partial_cursor(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = self.run_tiny(root / "failure", publish_failure=1)
            parent = root / "failure-checkpoints"
            self.assertEqual([p.name for p in parent.iterdir()], ["checkpoint-000000"])
            manifest = trainer.read(parent / "checkpoint-000000/checkpoint.json")
            for name, expected in manifest["files_sha256"].items():
                self.assertEqual(
                    trainer.digest(parent / "checkpoint-000000" / name), expected
                )
            self.assertEqual(failed["report"]["training_updates"], 1)
            config, rows = fixture()
            exposure = trainer.curriculum.Exposure()
            model = TinyAdapter()
            with self.assertRaisesRegex(ValueError, "optimizer boundary"):
                trainer.curriculum.train_draws(
                    model,
                    rows,
                    torch.optim.AdamW(model.parameters()),
                    torch,
                    config["training"],
                    "cpu",
                    exposure,
                    lambda _: None,
                    lambda _: None,
                    start_offset=1,
                )


if __name__ == "__main__":
    unittest.main()
