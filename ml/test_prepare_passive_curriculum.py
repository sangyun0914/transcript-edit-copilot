import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_passive_curriculum as curriculum
import prepare_passive_sft as smoke
from passive_bundle_v2 import weighted_indices


class CurriculumPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.release = self.root / "frozen"
        self.preparation = self.root / "original-smoke"
        self.completed = self.root / "completed-smoke"
        self.output = self.root / "curriculum"
        self.source = self.root / "ml"
        self.source.mkdir()
        for name in curriculum.SOURCE_FILES:
            (self.source / name).write_bytes(
                Path(curriculum.__file__).with_name(name).read_bytes()
            )
        self.source_patch = patch.object(
            curriculum, "__file__", str(self.source / "prepare_passive_curriculum.py")
        )
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        self.build_inputs()

    def save(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return smoke.sha256(content)

    def build_inputs(self, mutate=None):
        cases, messages, tokens = [], [], []
        for index in range(curriculum.TRAIN_ROWS):
            action, language = smoke.SLOTS[index % len(smoke.SLOTS)]
            identity = f"{action}-{language}-{index:04d}"
            reference = {"action": action}
            serialized = json.dumps(reference, separators=(",", ":"))
            input_hash = smoke.sha256(identity.encode())
            request = [
                {"role": "system", "content": "fixed policy"},
                {"role": "user", "content": identity},
            ]
            weight = (1 + index % 7) / curriculum.TRAIN_ROWS
            prefix = 4 + index % 11
            cases.append(
                {
                    "id": identity,
                    "input": {
                        "language": language,
                        "allowed_actions": list(smoke.ACTIONS),
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
                        "release_stratum": "fixture",
                        "kind": "fixture",
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
                        "input_ids": [5] * prefix + [9, 1],
                        "attention_mask": [1] * (prefix + 2),
                        "labels": [-100] * prefix + [9, 1],
                        "completion_start": prefix,
                    },
                    "alternative_completion_ids": [],
                }
            )
        if mutate:
            mutate(cases, messages, tokens)
        self.cases, self.messages, self.tokens = cases, messages, tokens
        content = {
            "train-cases.jsonl": smoke.jsonl_bytes(cases),
            "train-messages.jsonl": smoke.jsonl_bytes(messages),
            "tokenized/qwen35-9b/json-curriculum-train.jsonl.gz": gzip.compress(
                smoke.jsonl_bytes(tokens), mtime=0
            ),
            "source-groups.json": smoke.json_bytes(
                {"en": "train", "ko": "train", "held-out": "dev"}
            ),
            "prompt-json.txt": b"fixed policy\n",
        }
        hashes = {
            name: self.save(self.release / name, value)
            for name, value in content.items()
        }
        release = {
            "version": "passive-dataset-completion-v2",
            "status": "frozen_supported_scope",
            "release_id": "fixture",
            "files_sha256": {**hashes, "dev-cases.jsonl": "0" * 64},
            "dependencies_sha256": {
                "ml/passive_bundle_v2.py": smoke.sha256(
                    (self.source / "passive_bundle_v2.py").read_bytes()
                )
            },
        }
        self.release_sha = self.save(
            self.release / "manifest.json", smoke.json_bytes(release)
        )
        self.save(self.release / "dev-cases.jsonl", b"must never be read\n")
        selected = smoke.select_smoke(cases, tokens)
        tokenizer = {
            "id": curriculum.CANDIDATE,
            "files_sha256": {
                name: smoke.sha256(name.encode())
                for name in (
                    "config.json",
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "chat_template.jinja",
                )
            },
            "eos_token_id": 1,
            "vocab_size": 64,
        }
        native = {
            "repository": "Qwen/Qwen3.5-9B",
            "revision": "a" * 40,
            "weights_verified_by_this_preparation": False,
        }
        config = {
            "version": "passive-sft-preparation-v1",
            "candidate_id": curriculum.CANDIDATE,
            "native_base": native,
            "tokenizer": tokenizer,
            "proposed_smoke": {"order": [cases[i]["id"] for i in selected]},
        }
        prep_files = {
            "smoke-cases.jsonl": smoke.jsonl_bytes(cases[i] for i in selected),
            "smoke-messages.jsonl": smoke.jsonl_bytes(messages[i] for i in selected),
            "smoke-tokens.jsonl": smoke.jsonl_bytes(tokens[i] for i in selected),
            "smoke-model-inputs.jsonl": smoke.jsonl_bytes(
                {
                    key: tokens[i]["primary"][key]
                    for key in ("input_ids", "attention_mask", "labels")
                }
                for i in selected
            ),
            "configuration.json": smoke.json_bytes(config),
            "exposure-census.json": b"{}\n",
            "source/prepare_passive_sft.py": Path(smoke.__file__).read_bytes(),
            "source/test_prepare_passive_sft.py": Path(smoke.__file__)
            .with_name("test_prepare_passive_sft.py")
            .read_bytes(),
            "native-metadata/config.json": b"{}\n",
            "native-metadata/tokenizer_config.json": b"{}\n",
            "native-metadata/chat_template.jinja": b"fixture template",
        }
        prep_hashes = {
            name: self.save(self.preparation / name, value)
            for name, value in prep_files.items()
        }
        prep_manifest = {
            "candidate_id": curriculum.CANDIDATE,
            "release_manifest_sha256": self.release_sha,
            "files_sha256": prep_hashes,
            "inputs_sha256": {
                "frozen/" + name: value for name, value in hashes.items()
            },
        }
        prep_sha = self.save(
            self.preparation / "manifest.json", smoke.json_bytes(prep_manifest)
        )
        selection = {
            "selected_t1_candidate_ids": [curriculum.CANDIDATE],
            "bindings": {
                "release_manifest_sha256": self.release_sha,
                "train_only_preparation": {"sha256": prep_sha},
            },
            "selected_native_identity": {
                "base_repository": native["repository"],
                "base_revision": native["revision"],
                "tokenizer_files_sha256": tokenizer["files_sha256"],
            },
        }
        self.selection = self.root / "selection.json"
        self.selection_sha = self.save(self.selection, smoke.json_bytes(selection))
        run_config = {
            "version": "passive-sft-smoke-v1",
            "base": native,
            "bindings": {"preparation": {"sha256": prep_sha}},
            "training": {"sample_order": config["proposed_smoke"]["order"]},
        }
        run_hash = self.save(
            self.completed / "input/configuration-attempt-03.json",
            smoke.json_bytes(run_config),
        )
        completion = {
            "version": "passive-t1-feasibility-completion-v1",
            "status": "passed_engineering_roundtrip_substantive_sft_preregistration_pending",
            "candidate": native["repository"],
            "revision": native["revision"],
            "gates": dict.fromkeys(curriculum.SMOKE_GATES, True),
            "training": {"training_updates": 2, "unique_cases": 8},
            "files_sha256": {"input/configuration-attempt-03.json": run_hash},
        }
        self.completion = self.completed / "completion.json"
        self.completion_sha = self.save(self.completion, smoke.json_bytes(completion))

    def prepare(self, seed="3407"):
        return curriculum.prepare(
            self.release,
            self.selection,
            self.completion,
            self.preparation,
            self.output,
            release_sha256=self.release_sha,
            selection_sha256=self.selection_sha,
            smoke_completion_sha256=self.completion_sha,
            seed=seed,
            project_root=self.root,
        )

    def rebind(self, name, value):
        content = smoke.json_bytes(value)
        (self.output / name).write_bytes(content)
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["files_sha256"][name] = smoke.sha256(content)
        for entry in manifest["inventory"]:
            if entry["path"] == name:
                entry.update(bytes=len(content), sha256=smoke.sha256(content))
        manifest["payload_bytes"] = sum(
            entry["bytes"] for entry in manifest["inventory"]
        )
        path.write_bytes(smoke.json_bytes(manifest))

    def test_frozen_bytes_string_seed_exposure_and_final_partial_update(self):
        original = {
            path: path.read_bytes()
            for parent in (self.release, self.preparation)
            for path in parent.rglob("*")
            if path.is_file()
        }
        result = self.prepare()
        bundle = curriculum.verify_bundle(self.output)
        indices = weighted_indices(self.cases, 809, "3407")
        self.assertNotEqual(indices, weighted_indices(self.cases, 809, 3407))
        self.assertEqual(
            [row["train_row_index"] for row in bundle["draw_plan"]], indices
        )
        self.assertEqual(result["optimizer_updates"], 203)
        self.assertEqual(result["final_update_rows"], 1)
        self.assertEqual(result["unique_drawn_cases"], len(set(indices)))
        self.assertEqual(result["repeat_draws"], 809 - len(set(indices)))
        self.assertEqual(result["completion_tokens"], 809 * 2)
        self.assertEqual(
            result["sequence_tokens"],
            sum(len(self.tokens[i]["primary"]["input_ids"]) for i in indices),
        )
        for name, source in curriculum.PAYLOADS.items():
            self.assertEqual(
                (self.output / name).read_bytes(), (self.release / source).read_bytes()
            )
        self.assertTrue(
            all(path.read_bytes() == content for path, content in original.items())
        )
        with self.assertRaisesRegex(ValueError, "fresh output"):
            self.prepare()

    def test_nontrain_and_mask_corruption_never_publish(self):
        changes = [
            lambda cases, messages, tokens: cases[0]["metadata"].update(split="dev"),
            lambda cases, messages, tokens: tokens[0]["primary"]["labels"].__setitem__(
                0, 5
            ),
        ]
        for index, mutate in enumerate(changes):
            with self.subTest(index=index):
                self.build_inputs(mutate)
                self.output = self.root / f"bad-{index}"
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_hash_drift_and_incomplete_smoke_do_not_publish(self):
        (self.release / "train-cases.jsonl").write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "Bound file changed"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.build_inputs()
        completion = json.loads(self.completion.read_bytes())
        completion["gates"]["fresh_process_exact_adapter_and_output_reload"] = False
        self.completion_sha = self.save(self.completion, smoke.json_bytes(completion))
        with self.assertRaisesRegex(ValueError, "successful smoke"):
            self.prepare()
        self.assertFalse(self.output.exists())
        failures = list(self.root.glob("curriculum.staging-*/failure.json"))
        self.assertEqual(len(failures), 1)

    def test_self_consistent_manifest_cannot_change_draws_or_seed_type(self):
        self.prepare()
        plan = json.loads((self.output / "draw-plan.json").read_bytes())
        changed = copy.deepcopy(plan)
        changed[0]["train_row_index"] = (changed[0]["train_row_index"] + 1) % 809
        self.rebind("draw-plan.json", changed)
        with self.assertRaisesRegex(ValueError, "draw order"):
            curriculum.verify_bundle(self.output)
        self.rebind("draw-plan.json", plan)
        config = json.loads((self.output / "configuration.json").read_bytes())
        config["sampling"]["seed"] = 3407
        self.rebind("configuration.json", config)
        with self.assertRaisesRegex(ValueError, "explicit string"):
            curriculum.verify_bundle(self.output)

    def test_unregistered_dev_body_is_rejected_before_reading_it(self):
        self.prepare()
        target = self.output / "dev-cases.jsonl"
        content = b"must never be read\n"
        target.write_bytes(content)
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["files_sha256"][target.name] = smoke.sha256(content)
        path.write_bytes(smoke.json_bytes(manifest))
        original = Path.read_bytes

        def guarded(path):
            if path == target:
                self.fail("The verifier read an unregistered development body.")
            return original(path)

        with patch.object(Path, "read_bytes", guarded):
            with self.assertRaisesRegex(ValueError, "train-only payload inventory"):
                curriculum.verify_bundle(self.output)


if __name__ == "__main__":
    unittest.main()
