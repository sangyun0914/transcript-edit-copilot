import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import unicodedata

import build_exposed_text_index_actions_v3 as index


def reference_functions(path, names):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Reference functions changed.")
    scope = {"hashlib": hashlib, "json": json, "unicodedata": unicodedata}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class ExposureIndexTests(unittest.TestCase):
    def test_exact_sealer_normalization_and_string_digest_compatibility(self):
        legacy = reference_functions(
            index.ROOT / "ml/prepare_passive_actions_final_test.py",
            {"corpus_text", "packed", "digest"},
        )
        current = reference_functions(
            index.ROOT / "ml/prepare_passive_actions_v3.py", {"packed", "digest"}
        )
        value = {
            "schema_version": "fixture",
            "task": "predict_next_transcript_edit",
            "items": [
                {"text": "한글\t\n café"},
                {"text": "a\u00a0b"},
                {"type": "gap"},
                {"text": '“Quoted” \\ "text"'},
            ],
        }
        compact = {
            **value,
            "item_fields": ["id", "text"],
            "items": [[0, item.get("text", "")] for item in value["items"]],
        }
        for candidate in (value, {"input": value}, compact, {"items": []}):
            text = index.corpus_text(candidate)
            self.assertEqual(text, legacy["corpus_text"](candidate))
            self.assertEqual(index.digest(text), legacy["digest"](text))
            self.assertEqual(index.digest(text), current["digest"](text))
        self.assertNotEqual(
            index.digest(index.corpus_text(value)),
            hashlib.sha256(index.corpus_text(value).encode()).hexdigest(),
        )
        value["expectedOutput"] = {"text": "This label must not enter the index."}
        value["recent_edits"] = [{"after": "History also stays outside corpus_text."}]
        self.assertEqual(index.corpus_text(value), index.corpus_text(compact))

    def test_wrong_sha_or_visible_shape_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "train.jsonl"
            row = {
                "split": "train",
                "input": {
                    "contract_version": "fixture",
                    "items": [{"text": "Visible"}],
                },
            }
            path.write_text(json.dumps(row) + "\n")
            entry = {
                "path": path.name,
                "sha256": index.sha256(path),
                "storage": "jsonl",
                "mode": "visible_inputs",
                "input_paths": [["input"]],
                "split_path": ["split"],
                "allowed_splits": ["train"],
                "input_schemas": [{"contract_version": "fixture"}],
            }
            stats, hashes = index.inspect_entry(root, entry)
            self.assertEqual(
                (stats["text_bearing_rows"], hashes), (1, {index.digest("Visible")})
            )
            changed = copy.deepcopy(row)
            changed["input"] = {
                "contract_version": "fixture",
                "document": {"text": "Hidden"},
            }
            path.write_text(json.dumps(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                index.inspect_entry(root, entry)
            entry["sha256"] = index.sha256(path)
            with self.assertRaisesRegex(ValueError, "input.items"):
                index.inspect_entry(root, entry)
            changed["input"] = {
                "contract_version": "fixture",
                "item_fields": ["text", "id"],
                "items": [["bad width"]],
            }
            path.write_text(json.dumps(changed) + "\n")
            entry["sha256"] = index.sha256(path)
            with self.assertRaisesRegex(ValueError, "Malformed compact"):
                index.inspect_entry(root, entry)

    def test_message_roles_and_metadata_only_classification_cannot_hide_inputs(self):
        entry = {
            "mode": "user_message_inputs",
            "message_roles": ["system", "user"],
            "user_input_fields": ["items"],
        }
        row = {
            "messages": [
                {"role": "system", "content": "ignore this"},
                {"role": "user", "content": '{"items":[{"text":"visible"}]}'},
            ]
        }
        self.assertEqual(index.inputs(row, entry), [{"items": [{"text": "visible"}]}])
        ambiguous = copy.deepcopy(row)
        ambiguous["messages"][1]["content"] = (
            '{"items":[],"input":{"items":[{"text":"hidden"}]}}'
        )
        with self.assertRaisesRegex(ValueError, "user input field schema"):
            index.inputs(ambiguous, entry)
        ambiguous["messages"][1]["content"] = '{"items":[],"items":[{"text":"hidden"}]}'
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            index.inputs(ambiguous, entry)
        with self.assertRaisesRegex(ValueError, "unambiguous visible input"):
            index.validate_input(
                {"contract_version": "fixture", "items": [], "input": {"items": []}},
                [{"contract_version": "fixture"}],
            )
        row["messages"].append({"role": "assistant", "content": "Reference"})
        with self.assertRaisesRegex(ValueError, "role sequence"):
            index.inputs(row, entry)
        with self.assertRaisesRegex(ValueError, "Metadata-only runtime shape"):
            index.inputs(
                {"input": {}},
                {
                    "mode": "runtime_without_visible_input",
                    "row_keys": ["input"],
                    "exclusion_reason": "Not allowed to discard visible input.",
                },
            )

    def test_reserved_or_escaping_paths_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in (
                "artifacts/holdout/cases.jsonl",
                "artifacts/datasets/final-test/inputs.json",
                "../train.jsonl",
            ):
                with self.subTest(path=name), self.assertRaises(ValueError):
                    index.local_path(root, name)
            with self.assertRaisesRegex(ValueError, "Reserved body"):
                index.local_path(root, index.HISTORICAL_METADATA)
            self.assertEqual(
                index.local_path(root, index.HISTORICAL_METADATA, metadata=True),
                (root / index.HISTORICAL_METADATA).resolve(),
            )

    def test_split_resolution_requires_same_input_and_registered_message_hash(self):
        row = {
            "id": "attempt",
            "input": {"items": [{"text": "Original"}]},
            "input_sha256": "a" * 64,
        }
        registered = {
            **copy.deepcopy(row),
            "metadata": {"split": "train", "input_sha256": "a" * 64},
        }
        entry = {"split_authority": {"mode": "exact_registered_input"}}
        self.assertEqual(
            index.original_split(row, entry, {"attempt": registered}), "train"
        )
        self.assertNotIn("split", row)
        registered["input"]["items"][0]["text"] = "Changed"
        with self.assertRaisesRegex(
            ValueError, "input or stored canonical-message hash differs"
        ):
            index.original_split(row, entry, {"attempt": registered})
        registered["input"] = copy.deepcopy(row["input"])
        registered["metadata"]["input_sha256"] = "b" * 64
        with self.assertRaisesRegex(
            ValueError, "input or stored canonical-message hash differs"
        ):
            index.original_split(row, entry, {"attempt": registered})


if __name__ == "__main__":
    unittest.main()
