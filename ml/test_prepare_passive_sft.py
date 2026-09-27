import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from prepare_passive_sft import ACTIONS, SLOTS, json_bytes, jsonl_bytes, prepare, sha256


class PassiveSftPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.release = self.root / "release"
        self.release.mkdir()
        self.registry = self.root / "models.json"
        self.output = self.root / "preparation-only"
        self.revision = "a" * 40
        self.reset_rows()
        self.write_inputs()

    def reset_rows(self):
        self.cases, self.messages, self.tokens = [], [], []
        slots = [
            (f"{action}-{language}", action, language, 5 + index)
            for index, (action, language) in enumerate(SLOTS)
        ]
        slots += [
            ("z-long-tie", "correct_text", "en", 20),
            ("a-long-tie", "correct_text", "en", 20),
        ]
        for identity, action, language, prefix in slots:
            reference = {"action": action}
            serialized = json.dumps(reference, separators=(",", ":"))
            request = [
                {"role": "system", "content": "fixed policy"},
                {"role": "user", "content": f"input {identity}"},
            ]
            self.cases.append(
                {
                    "id": identity,
                    "input": {
                        "language": "kor" if language == "ko" else language,
                        "allowed_actions": list(ACTIONS),
                    },
                    "expectedOutput": reference,
                    "acceptableOutputs": [],
                    "metadata": {
                        "split": "train",
                        "training_eligible": True,
                        "language": language,
                        "source_id": language,
                        "source_group_id": language,
                        "episode_id": identity,
                        "sampling_cluster": f"train:{language}",
                        "sampling_weight": 0.1,
                        "input_sha256": "b" * 64,
                    },
                }
            )
            self.messages.append(
                {
                    "id": identity,
                    "sampling_weight": 0.1,
                    "messages": [
                        *request,
                        {"role": "assistant", "content": serialized},
                    ],
                }
            )
            self.tokens.append(
                {
                    "id": identity,
                    "split": "train",
                    "sampling_weight": 0.1,
                    "input_sha256": "b" * 64,
                    "messages": request,
                    "references": [serialized],
                    "primary": {
                        "input_ids": [5] * prefix + [9, 1],
                        "attention_mask": [1] * (prefix + 2),
                        "labels": [-100] * prefix + [9, 1],
                        "completion_start": prefix,
                    },
                    "alternative_completion_ids": [],
                }
            )

    def write_inputs(self):
        payloads = {
            "train-cases.jsonl": "".join(
                json.dumps(case, ensure_ascii=False) + "\n" for case in self.cases
            ).encode(),
            "train-messages.jsonl": jsonl_bytes(self.messages),
            "tokenized/qwen-test/json-curriculum-train.jsonl.gz": gzip.compress(
                jsonl_bytes(self.tokens), mtime=0
            ),
            "source-groups.json": json_bytes({"en": "train", "ko": "train"}),
            "prompt-json.txt": b"fixed policy\n",
        }
        for name, content in payloads.items():
            path = self.release / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        tokenizer_files = {
            "config.json": json_bytes({"text_config": {"vocab_size": 64}}),
            "tokenizer_config.json": json_bytes({"eos_token": "<eos>"}),
            "tokenizer.json": json_bytes(
                {"added_tokens": [{"id": 1, "content": "<eos>"}]}
            ),
            "chat_template.jinja": b"fixed template",
        }
        tokenizer_dir = self.root / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        for name, content in tokenizer_files.items():
            (tokenizer_dir / name).write_bytes(content)
        file_hashes = {
            name: sha256(content) for name, content in tokenizer_files.items()
        }
        tokenizer = {
            "id": "qwen-test",
            "repository": "org/native",
            "revision": self.revision,
            "path": "tokenizer",
            "files_sha256": file_hashes,
        }
        tokenizer_registry = json_bytes({"candidates": [tokenizer]})
        (self.root / "tokenizers.json").write_bytes(tokenizer_registry)
        self.registry.write_bytes(
            json_bytes(
                {
                    "version": "passive-models-v2",
                    "tokenizers": {
                        "registry_path": "tokenizers.json",
                        "registry_sha256": sha256(tokenizer_registry),
                    },
                    "candidates": [
                        {
                            "id": "qwen-test",
                            "tokenizer_id": "qwen-test",
                            "base_repository": "org/native",
                            "base_revision": self.revision,
                            "tokenizer_path": "tokenizer",
                            "tokenizer_files_sha256": file_hashes,
                            "template_sha256": file_hashes["chat_template.jinja"],
                        }
                    ],
                }
            )
        )
        self.manifest = {
            "version": "passive-dataset-completion-v2",
            "status": "frozen_supported_scope",
            "release_id": "fixture",
            "files_sha256": {
                name: sha256(content) for name, content in payloads.items()
            },
            "dependencies_sha256": {
                "tokenizers.json": sha256(tokenizer_registry),
                **{f"tokenizer/{name}": value for name, value in file_hashes.items()},
            },
        }
        (self.release / "manifest.json").write_bytes(json_bytes(self.manifest))
        self.release_hash = sha256((self.release / "manifest.json").read_bytes())
        self.registry_hash = sha256(self.registry.read_bytes())

    def prepare(self, **changes):
        kwargs = {
            "release": self.release,
            "registry": self.registry,
            "candidate_id": "qwen-test",
            "native_revision": self.revision,
            "output": self.output,
            "release_sha256": self.release_hash,
            "registry_sha256": self.registry_hash,
            "project_root": self.root,
        }
        kwargs.update(changes)
        return prepare(**kwargs)

    def test_pinned_file_hash_and_native_revision_drift_fail_before_output(self):
        for relative in (
            "release/manifest.json",
            "models.json",
            "tokenizers.json",
            "tokenizer/config.json",
            "release/train-cases.jsonl",
            "release/tokenized/qwen-test/json-curriculum-train.jsonl.gz",
        ):
            with self.subTest(path=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "hash drift"):
                    self.prepare()
                self.assertFalse(self.output.exists())
                path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Native revision"):
            self.prepare(native_revision="c" * 40)
        self.assertFalse(self.output.exists())

    def test_mask_attention_eos_and_payload_order_drift_fail_even_when_rehashed(self):
        def prompt_label(rows):
            rows[0]["primary"]["labels"][0] = 5

        def completion_label(rows):
            rows[0]["primary"]["labels"][-1] = -100

        def attention(rows):
            rows[0]["primary"]["attention_mask"].pop()

        def missing_eos(rows):
            rows[0]["primary"]["input_ids"][-1] = 2
            rows[0]["primary"]["labels"][-1] = 2

        for mutate in (
            prompt_label,
            completion_label,
            attention,
            missing_eos,
            lambda rows: rows.reverse(),
        ):
            with self.subTest(mutation=mutate.__name__):
                self.reset_rows()
                mutate(self.tokens)
                self.write_inputs()
                with self.assertRaisesRegex(ValueError, "mask|order"):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_non_train_sources_and_missing_action_language_slots_are_rejected(self):
        for key, value in (
            ("split", "dev"),
            ("training_eligible", False),
            ("source_group_id", "sealed"),
        ):
            with self.subTest(key=key):
                self.reset_rows()
                self.cases[0]["metadata"][key] = value
                self.write_inputs()
                with self.assertRaisesRegex(ValueError, "Non-train"):
                    self.prepare()
        self.reset_rows()
        for collection in (self.cases, self.messages, self.tokens):
            collection.pop(0)
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Missing.*action-language"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_longest_then_lexicographic_selection_is_deterministic_and_portable(self):
        receipt = self.prepare()
        config = json.loads((self.output / "configuration.json").read_text())
        self.assertEqual(config["candidate_selection_status"], "not_selected_for_t1")
        self.assertEqual(config["status"], "data_ready_trainer_not_implemented")
        self.assertFalse(config["training_run"])
        self.assertEqual(config["selection"][2]["case_id"], "a-long-tie")
        self.assertEqual(
            [(row["action"], row["language"]) for row in config["selection"]],
            list(SLOTS),
        )
        self.assertEqual(config["proposed_smoke"]["optimizer_updates"], 2)
        self.assertTrue(
            all(value is None for value in config["unresolved_prerequisites"].values())
        )
        for name, expected in receipt["files_sha256"].items():
            self.assertEqual(sha256((self.output / name).read_bytes()), expected)
        original_lines = {
            json.loads(line)["id"]: line
            for line in (self.release / "train-cases.jsonl")
            .read_bytes()
            .splitlines(keepends=True)
        }
        for line in (
            (self.output / "smoke-cases.jsonl").read_bytes().splitlines(keepends=True)
        ):
            case = json.loads(line)
            self.assertEqual(line, original_lines[case["id"]])
            if case["metadata"]["language"] == "ko":
                self.assertEqual(case["input"]["language"], "kor")
        inputs = [
            json.loads(line)
            for line in (self.output / "smoke-model-inputs.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(len(inputs), 8)
        self.assertEqual(set(inputs[0]), {"input_ids", "attention_mask", "labels"})
        census = json.loads((self.output / "exposure-census.json").read_text())
        budgets = census["full_training_budget_proposals"]
        self.assertEqual(
            [
                (
                    row["proposed_sampled_rows"],
                    row["proposed_optimizer_updates"],
                    row["final_update_microbatches"],
                )
                for row in budgets
            ],
            [(10, 3, 2), (20, 5, 4)],
        )
        for collection in (self.cases, self.messages, self.tokens):
            collection.reverse()
        self.write_inputs()
        second = self.root / "second"
        self.prepare(output=second)
        new_config = json.loads((second / "configuration.json").read_text())
        self.assertEqual(
            new_config["proposed_smoke"]["order"], config["proposed_smoke"]["order"]
        )
        for name in (
            "smoke-cases.jsonl",
            "smoke-messages.jsonl",
            "smoke-tokens.jsonl",
            "smoke-model-inputs.jsonl",
        ):
            self.assertEqual(
                (self.output / name).read_bytes(), (second / name).read_bytes()
            )

    def test_existing_output_and_frozen_descendants_are_never_written(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("do not overwrite")
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            self.prepare()
        self.assertEqual(list(self.output.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(), "do not overwrite")
        for path in (
            self.release / "new-output",
            self.root / "another" / "frozen" / "new-output",
        ):
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ValueError, "frozen release"),
            ):
                self.prepare(output=path)
            self.assertFalse(path.exists())

    def test_changed_message_target_or_weight_does_not_pass_structural_validation(self):
        original = copy.deepcopy(self.messages)
        for mutate in (
            lambda row: row.update(sampling_weight=0.2),
            lambda row: row["messages"][-1].update(content='{"action":"correct_text"}'),
        ):
            self.messages = copy.deepcopy(original)
            mutate(self.messages[0])
            self.write_inputs()
            with self.assertRaisesRegex(ValueError, "weight|target"):
                self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
