import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_passive_gemma as gemma


class GemmaPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.release = self.root / "frozen"
        self.qwen = self.root / "qwen"
        self.output = self.root / "prepared"

    def save(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return gemma.sha256(content)

    def build(self, mutate=None):
        cases, messages, tokens = [], [], []
        for index in range(809):
            action, language = gemma.smoke.SLOTS[index % 8]
            identity = f"case-{index:04d}"
            request = [
                {"role": "system", "content": "policy"},
                {"role": "user", "content": identity},
            ]
            reference = {"action": action}
            serialized = json.dumps(reference, separators=(",", ":"))
            weight = (index % 7 + 1) / 809
            input_hash = gemma.sha256(identity.encode())
            prefix = 3 + index % 13
            cases.append(
                {
                    "id": identity,
                    "input": {
                        "language": language,
                        "allowed_actions": list(gemma.smoke.ACTIONS),
                    },
                    "expectedOutput": reference,
                    "acceptableOutputs": [],
                    "metadata": {
                        "split": "train",
                        "training_eligible": True,
                        "language": language,
                        "source_group_id": language,
                        "source_id": language + str(index % 5),
                        "episode_id": "episode-" + str(index // 3),
                        "sampling_cluster": "train:" + language + str(index % 3),
                        "sampling_weight": weight,
                        "input_sha256": input_hash,
                    },
                }
            )
            messages.append(
                {
                    "id": identity,
                    "sampling_weight": weight,
                    "messages": [
                        *request,
                        {"role": "assistant", "content": serialized},
                    ],
                }
            )
            tokens.append(
                {
                    "id": identity,
                    "split": "train",
                    "sampling_weight": weight,
                    "input_sha256": input_hash,
                    "messages": request,
                    "references": [serialized],
                    "primary": {
                        "input_ids": [2] * prefix + [9, 1],
                        "attention_mask": [1] * (prefix + 2),
                        "labels": [-100] * prefix + [9, 1],
                        "completion_start": prefix,
                    },
                    "alternative_completion_ids": [],
                }
            )
        if mutate:
            mutate(cases, messages, tokens)
        self.cases, self.tokens = cases, tokens
        tokenizer_files = {
            "config.json": gemma.smoke.json_bytes(
                {
                    "model_type": "gemma4",
                    "architectures": ["Gemma4ForConditionalGeneration"],
                    "text_config": {
                        "vocab_size": 262144,
                        "eos_token_id": 1,
                        "pad_token_id": 0,
                    },
                    "eos_token_id": [1, 106],
                }
            ),
            "tokenizer_config.json": gemma.smoke.json_bytes({"eos_token": "<eos>"}),
            "tokenizer.json": gemma.smoke.json_bytes(
                {"added_tokens": [{"content": "<eos>", "id": 1}]}
            ),
            "chat_template.jinja": b"fixture-template",
        }
        tokenizer_pins = {
            name: self.save(self.root / "tokenizer" / name, value)
            for name, value in tokenizer_files.items()
        }
        tokenizer = {
            "id": gemma.CANDIDATE,
            "repository": gemma.REPOSITORY,
            "revision": gemma.REVISION,
            "path": "tokenizer",
            "files_sha256": tokenizer_pins,
        }
        tokenizer_registry = self.root / "tokenizer-registry.json"
        tokenizer_registry_sha = self.save(
            tokenizer_registry, gemma.smoke.json_bytes({"candidates": [tokenizer]})
        )
        candidate = {
            "id": gemma.CANDIDATE,
            "base_repository": gemma.REPOSITORY,
            "base_revision": gemma.REVISION,
            "tokenizer_id": gemma.CANDIDATE,
            "tokenizer_path": "tokenizer",
            "tokenizer_files_sha256": tokenizer_pins,
            "template_sha256": tokenizer_pins["chat_template.jinja"],
            "model_sha256": "a" * 64,
            "quant_repository": "fixture/public-qat",
            "quant_revision": "b" * 40,
        }
        self.registry = self.root / "model-registry.json"
        self.registry_sha = self.save(
            self.registry,
            gemma.smoke.json_bytes(
                {
                    "version": "passive-models-v2",
                    "candidates": [candidate],
                    "tokenizers": {
                        "registry_path": "tokenizer-registry.json",
                        "registry_sha256": tokenizer_registry_sha,
                    },
                }
            ),
        )
        payloads = {
            "train-cases.jsonl": gemma.smoke.jsonl_bytes(cases),
            "train-messages.jsonl": gemma.smoke.jsonl_bytes(messages),
            "tokenized/gemma26/json-curriculum-train.jsonl.gz": gzip.compress(
                gemma.smoke.jsonl_bytes(tokens), mtime=0
            ),
            "source-groups.json": gemma.smoke.json_bytes(
                {"en": "train", "ko": "train", "dev": "dev"}
            ),
            "prompt-json.txt": b"policy\n",
        }
        payload_hashes = {
            name: self.save(self.release / name, value)
            for name, value in payloads.items()
        }
        self.release_sha = self.save(
            self.release / "manifest.json",
            gemma.smoke.json_bytes(
                {
                    "version": "passive-dataset-completion-v2",
                    "status": "frozen_supported_scope",
                    "files_sha256": {
                        **payload_hashes,
                        "dev-cases.jsonl": "0" * 64,
                        "test-cases.jsonl": "0" * 64,
                    },
                    "dependencies_sha256": {
                        "tokenizer-registry.json": tokenizer_registry_sha,
                        **{
                            "tokenizer/" + name: pin
                            for name, pin in tokenizer_pins.items()
                        },
                        "ml/passive_bundle_v2.py": gemma.FROZEN_HELPERS[
                            "passive_bundle_v2.py"
                        ],
                    },
                }
            ),
        )
        self.save(self.release / "dev-cases.jsonl", b"must not be read")
        self.save(self.release / "test-cases.jsonl", b"must not be read")
        indices = gemma.curriculum.weighted_draw_indices(cases, 809, "3407")
        draw_sha = self.save(
            self.qwen / "draw-plan.json",
            gemma.smoke.json_bytes(
                [
                    {
                        "draw_index": draw,
                        "case_id": cases[index]["id"],
                        "train_row_index": index,
                    }
                    for draw, index in enumerate(indices)
                ]
            ),
        )
        self.qwen_sha = self.save(
            self.qwen / "manifest.json",
            gemma.smoke.json_bytes(
                {
                    "candidate_id": "qwen35-9b",
                    "release_manifest_sha256": self.release_sha,
                    "files_sha256": {**payload_hashes, "draw-plan.json": draw_sha},
                }
            ),
        )

    def prepare(self):
        with (
            patch.object(gemma, "RELEASE_SHA256", self.release_sha),
            patch.object(gemma, "QWEN_MANIFEST_SHA256", self.qwen_sha),
        ):
            return gemma.prepare(
                self.release,
                self.registry,
                self.qwen,
                self.output,
                registry_sha256=self.registry_sha,
                project_root=self.root,
            )

    def verify(self):
        with (
            patch.object(gemma, "RELEASE_SHA256", self.release_sha),
            patch.object(gemma, "QWEN_MANIFEST_SHA256", self.qwen_sha),
        ):
            return gemma.verify_bundle(self.output)

    def rewrite_bound_payload(self, name, data):
        self.save(self.output / name, data)
        path = self.output / "manifest.json"
        value = json.loads(path.read_bytes())
        value["files_sha256"][name] = gemma.sha256(data)
        for row in value["inventory"]:
            if row["path"] == name:
                row.update(bytes=len(data), sha256=gemma.sha256(data))
        value["payload_bytes"] = sum(row["bytes"] for row in value["inventory"])
        self.save(path, gemma.smoke.json_bytes(value))

    def test_train_only_common_draws_exact_phase_budgets_and_fresh_output(self):
        self.build()
        read_bytes = Path.read_bytes

        def guarded(path):
            if path.name in {"dev-cases.jsonl", "test-cases.jsonl"}:
                raise AssertionError("Held-out body read")
            return read_bytes(path)

        with patch.object(Path, "read_bytes", guarded):
            self.prepare()
            checked = self.verify()
        plan = checked["draw_plan"]
        self.assertEqual(plan, json.loads((self.qwen / "draw-plan.json").read_bytes()))
        config = checked["configuration"]
        self.assertEqual(config["sampling"]["seed"], "3407")
        self.assertEqual(config["training"]["optimizer_updates"], 203)
        self.assertEqual(config["training"]["final_update_rows"], 1)
        self.assertEqual(config["training"]["max_total_completion_tokens"], 809 * 2)
        selected = checked["smoke_configuration"]["proposed_smoke"]
        self.assertEqual(selected["optimizer_updates"], 2)
        self.assertEqual(selected["max_total_completion_tokens"], 8 * 2)
        self.assertEqual(len(set(selected["order"])), 8)
        self.assertEqual(checked["tokens"], self.tokens)
        self.assertFalse(
            config["launch_gates"]["smoke_training_and_fresh_reload_completed"]
        )
        with self.assertRaisesRegex(ValueError, "fresh output"):
            self.prepare()

    def test_rejects_ineligible_source_or_completion_eos_before_publish(self):
        for mutation, message in (
            (
                lambda c, m, t: c[0]["metadata"].update(source_group_id="dev"),
                "Non-train",
            ),
            (
                lambda c, m, t: t[0]["primary"].update(
                    input_ids=[2] * 3 + [9, 106], labels=[-100] * 3 + [9, 106]
                ),
                "EOS",
            ),
        ):
            with self.subTest(message=message):
                self.build(mutation)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_inventory_rejected_before_unregistered_body_read(self):
        self.build()
        self.prepare()
        (self.output / "dev-cases.jsonl").write_bytes(b"must never be read")
        read_bytes = Path.read_bytes

        def guarded(path):
            if path.name == "dev-cases.jsonl":
                raise AssertionError("Unregistered body read")
            return read_bytes(path)

        with (
            patch.object(Path, "read_bytes", guarded),
            self.assertRaisesRegex(ValueError, "Physical bundle inventory"),
        ):
            self.verify()

    def test_resigned_draw_order_or_seed_drift_fails_semantic_replay(self):
        self.build()
        self.prepare()
        original = (self.output / "draw-plan.json").read_bytes()
        changed = json.loads(original)
        changed[0], changed[1] = changed[1], changed[0]
        self.rewrite_bound_payload("draw-plan.json", gemma.smoke.json_bytes(changed))
        with self.assertRaisesRegex(ValueError, "shared string-seed draw order"):
            self.verify()
        self.rewrite_bound_payload("draw-plan.json", original)
        config = json.loads((self.output / "configuration.json").read_bytes())
        config["sampling"]["seed"] = 3407
        self.rewrite_bound_payload("configuration.json", gemma.smoke.json_bytes(config))
        with self.assertRaisesRegex(ValueError, "configuration or phase budgets"):
            self.verify()

    def test_frozen_input_hash_drift_never_publishes_bundle(self):
        self.build()
        changed = copy.deepcopy(self.tokens)
        changed[0]["primary"]["labels"][0] = 2
        (self.release / "tokenized/gemma26/json-curriculum-train.jsonl.gz").write_bytes(
            gzip.compress(gemma.smoke.jsonl_bytes(changed), mtime=0)
        )
        with self.assertRaisesRegex(ValueError, "Bound file changed"):
            self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
