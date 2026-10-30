import json
import tempfile
import unittest
from pathlib import Path

from verify_passive_actions_v3 import IndexedProofs, iter_cases


class ActionsV3ReadbackTests(unittest.TestCase):
    def test_proof_lookup_supports_different_split_order_and_rejects_duplicate_identity(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cases").mkdir()
            (root / "cases/train.jsonl").write_text('{"id":"second"}\n')
            (root / "cases/dev.jsonl").write_text('{"id":"first"}\n')
            path = root / "reference-executions.jsonl"
            rows = [
                {"id": "first", "journal": ["one"]},
                {"id": "second", "journal": ["two"]},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            proofs = IndexedProofs(path)
            try:
                self.assertEqual(
                    [proofs[row["id"]]["journal"] for row in iter_cases(root)],
                    [["two"], ["one"]],
                )
            finally:
                proofs.close()
            with path.open("a") as stream:
                stream.write(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate execution proof"):
                IndexedProofs(path)


if __name__ == "__main__":
    unittest.main()
