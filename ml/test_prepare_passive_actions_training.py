import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import struct
import sys
import tempfile
import unittest

import prepare_passive_actions_training as preparation


class TrainingPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = (
            preparation.ROOT / preparation.PILOT / "cases/train.jsonl",
            preparation.ROOT / preparation.PILOT / "cases/dev.jsonl",
            preparation.ROOT
            / preparation.PILOT
            / "tokenized/gemma26/json-curriculum-train.jsonl.gz",
            preparation.ROOT / preparation.STAGE,
            preparation.ROOT / preparation.TRAIN / "adapter-tensors.json",
            preparation.ROOT / preparation.TRAIN / "adapter/adapter_model.safetensors",
        )
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest(
                "Requires the local frozen Gemma training experiment artifacts."
            )
        cls.cases = preparation.rows(
            preparation.ROOT / preparation.PILOT / "cases/train.jsonl"
        )
        cls.tokens = preparation.rows(
            preparation.ROOT
            / preparation.PILOT
            / "tokenized/gemma26/json-curriculum-train.jsonl.gz"
        )
        cls.stage = preparation.read(preparation.ROOT / preparation.STAGE)
        cls.prompt_hash = hashlib.sha256(
            cls.cases[0]["messages"][0]["content"].encode()
        ).hexdigest()

    def test_actual_train_view_masks_order_budget_and_provenance(self):
        phases, rows, diagnostics, plan = preparation.derive(
            self.cases, self.tokens, self.stage, self.prompt_hash
        )
        self.assertEqual(phases["training"]["max_total_sequence_tokens"], 56276)
        self.assertEqual(phases["training"]["max_total_completion_tokens"], 598)
        self.assertEqual(phases["training"]["optimizer_updates"], 11)
        self.assertEqual(phases["training"]["final_update_rows"], 2)
        self.assertEqual(phases["smoke"]["max_total_sequence_tokens"], 10993)
        self.assertEqual(phases["smoke"]["max_total_completion_tokens"], 106)
        self.assertEqual([r["draw_index"] for r in diagnostics], list(range(8)))
        self.assertEqual(len({row["case_id"] for row in rows}), 42)
        for row, draw in zip(rows, plan, strict=True):
            original = self.tokens[draw["train_row_index"]]["primary"]
            self.assertEqual(row["input_ids"], original["input_ids"])
            start = row["completion_start"]
            self.assertEqual(row["labels"][:start], [-100] * start)
            self.assertEqual(row["labels"][start:], row["input_ids"][start:])
            self.assertEqual(row["labels"][-1], 1)
            self.assertEqual(row["attention_mask"], [1] * len(row["input_ids"]))
            self.assertEqual(
                set(row["provenance"]),
                {"source_id", "source_group_id", "episode_id", "language", "action"},
            )

    def test_seed_type_and_smoke_longest_cell_must_not_drift(self):
        stage = copy.deepcopy(self.stage)
        order = sorted(c["id"] for c in self.cases)
        random.Random("3407").shuffle(order)
        stage["proposed_substantive"]["sample_order"] = order
        with self.assertRaisesRegex(ValueError, "integer-seed"):
            preparation.planned(self.cases, self.tokens, stage)
        stage = copy.deepcopy(self.stage)
        stage["proposed_smoke"]["sample_order"][0] = self.cases[0]["id"]
        with self.assertRaisesRegex(ValueError, "diagnostics"):
            preparation.planned(self.cases, self.tokens, stage)

    def test_dev_row_and_reference_prefix_mutations_are_rejected(self):
        cases = copy.deepcopy(self.cases)
        cases[0]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "Train-only"):
            preparation.validate_cases(cases, self.tokens, self.prompt_hash)
        tokens = copy.deepcopy(self.tokens)
        tokens[0]["primary"]["completion_start"] += 1
        with self.assertRaisesRegex(ValueError, "prefix binding"):
            preparation.validate_cases(self.cases, tokens, self.prompt_hash)
        tokens = copy.deepcopy(self.tokens)
        tokens[0]["primary"]["input_ids"][-1] = 106
        with self.assertRaisesRegex(ValueError, "EOS"):
            preparation.validate_cases(self.cases, tokens, self.prompt_hash)

    def test_portable_source_has_only_stdlib_dependencies_and_no_authored_builder(self):
        source = Path(preparation.__file__).read_text()
        tree = ast.parse(source)
        top_level = [
            node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        modules = {
            node.module.split(".")[0]
            for node in top_level
            if isinstance(node, ast.ImportFrom)
        }
        modules |= {
            name.name.split(".")[0]
            for node in top_level
            if isinstance(node, ast.Import)
            for name in node.names
        }
        self.assertTrue(modules <= sys.stdlib_module_names)
        self.assertNotIn("import prepare_passive_actions ", source)
        self.assertNotIn("runtime-inputs.jsonl", preparation.ALLOWED)
        self.assertNotIn("reference-executions.jsonl", preparation.ALLOWED)
        self.assertNotIn("evidence/pilot-manifest.json", preparation.ALLOWED)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-S",
                "-c",
                "import sys;sys.path.insert(0,'ml');import prepare_passive_actions_training",
            ],
            cwd=preparation.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_inventory_rejects_unlisted_or_linked_payload_before_body_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text("not-json")
            (root / "unregistered.jsonl").write_text("private")
            with self.assertRaisesRegex(ValueError, "allowlist"):
                preparation.verify_bundle(root)
            (root / "alias").symlink_to(root / "unregistered.jsonl")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                preparation.verify_bundle(root)

    def test_actual_anchor_record_uses_frozen_typed_tensor_hash(self):
        root = preparation.ROOT / preparation.TRAIN
        records = preparation.read(root / "adapter-tensors.json")
        with (root / "adapter/adapter_model.safetensors").open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
            name = sorted(records)[0]
            first, last = header[name]["data_offsets"]
            stream.seek(8 + size + first)
            raw = stream.read(last - first)
        record = records[name]
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
        self.assertEqual(
            preparation.tensor_sha256(record["dtype"], record["shape"], raw),
            record["sha256"],
        )

    def test_train_payload_and_new_sources_contain_no_pilot_dev_ids_or_transcript_text(
        self,
    ):
        # The deny-list is used only by this local test and is never packaged.
        dev = preparation.rows(preparation.ROOT / preparation.PILOT / "cases/dev.jsonl")
        forbidden = {case["id"] for case in dev}
        forbidden |= {
            item["text"]
            for case in dev
            for item in case["input"]["items"]
            if item["type"] == "text" and item["text"]
        }
        content = "\n".join(
            [
                Path(preparation.__file__).read_text(),
                json.dumps(self.cases, ensure_ascii=False),
                json.dumps(self.tokens, ensure_ascii=False),
                json.dumps(self.stage, ensure_ascii=False),
                json.dumps(preparation.pilot_reference(), ensure_ascii=False),
            ]
        )
        for value in forbidden:
            self.assertNotIn(value, content)


if __name__ == "__main__":
    unittest.main()
