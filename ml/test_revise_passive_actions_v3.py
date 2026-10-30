import copy
import json
import tempfile
import unittest
from pathlib import Path

from revise_passive_actions_v3 import add_references, clone_and_append
from prepare_passive_actions_v3 import digest


class ActionsV3ReferenceRevisionTests(unittest.TestCase):
    def test_appending_new_proofs_keeps_frozen_parent_bytes_and_inode_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (
                Path(directory) / "old.jsonl",
                Path(directory) / "new.jsonl",
            )
            source.write_text('{"id":"old"}\n')
            receipt = clone_and_append(source, target, [{"id": "new"}])
            self.assertEqual(source.read_text(), '{"id":"old"}\n')
            self.assertEqual(
                [json.loads(line)["id"] for line in target.read_text().splitlines()],
                ["old", "new"],
            )
            self.assertTrue(receipt["distinct_inodes_before_and_after"])

    def test_addition_preserves_input_and_primary_and_cannot_change_necessity(self):
        primary = {"action": "correct_text", "target": [2], "text": "little"}
        original = {
            "input": {"cursor_before": 2},
            "document": {"content": []},
            "runtime": {"sequence": 1},
            "source": {"split": "train"},
            "judgment": {"status": "edit_required"},
            "primary": primary,
            "acceptable": [primary],
            "accepted_predictions": [primary],
            "lineage": {},
        }
        alternate = {
            "prediction": {
                "action": "correct_text",
                "target": [1],
                "text": "two words",
            },
            "input_sha256": digest(original["input"]),
            "rationale": "Visible binary context.",
        }
        changed = add_references(original, [alternate], {"sha256": "frozen"})
        for key in ("input", "document", "runtime", "source", "judgment", "primary"):
            self.assertEqual(changed[key], original[key])
        self.assertEqual(len(original["acceptable"]), 1)
        self.assertEqual(changed["acceptable"], changed["accepted_predictions"])
        with self.assertRaisesRegex(ValueError, "already accepted"):
            add_references(changed, [alternate], {})
        with self.assertRaisesRegex(ValueError, "edit necessity"):
            add_references(
                original,
                [
                    {
                        "prediction": {"action": "no_action"},
                        "input_sha256": digest(original["input"]),
                    }
                ],
                {},
            )
        with self.assertRaisesRegex(ValueError, "different predictor input"):
            add_references(original, [{**alternate, "input_sha256": "other"}], {})
        preserved = copy.deepcopy(original)
        preserved["judgment"]["status"] = "preserve"
        with self.assertRaisesRegex(ValueError, "abstention"):
            add_references(preserved, [alternate], {})


if __name__ == "__main__":
    unittest.main()
