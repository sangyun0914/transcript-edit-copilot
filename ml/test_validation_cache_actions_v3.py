import copy
import json
import tempfile
import unittest
from pathlib import Path

from validation_cache_actions_v3 import ReferenceProofCache, digest


class ActionsV3ValidationCacheTests(unittest.TestCase):
    def test_only_identical_runtime_and_implementation_reuse_a_proof(self):
        case = {
            "id": "one",
            "document": {"content": []},
            "input": {"cursor_before": 0},
            "messages": [],
            "runtime": {"sequence": 0},
            "source": {"id": "recording"},
            "split": "train",
            "judgment": {"status": "preserve"},
            "primary": {"action": "no_action"},
            "accepted_predictions": [{"action": "no_action"}],
        }
        calls = []

        def validate(row):
            calls.append(row["id"])
            return {"before": digest(row["document"]), "prediction": row["primary"]}

        with tempfile.TemporaryDirectory() as directory:
            cache = ReferenceProofCache(
                directory, {"contract": "one", "tokenizer": "native"}
            )
            proof = cache.validate(case, validate)
            path = next(Path(directory).glob("*.json"))
            case["execution_sha256"] = digest(proof)
            self.assertEqual(cache.validate(case, validate), proof)
            self.assertEqual(len(calls), 1)
            changed = copy.deepcopy(case)
            changed["runtime"]["sequence"] = 1
            cache.validate(changed, validate)
            ReferenceProofCache(
                directory, {"contract": "two", "tokenizer": "native"}
            ).validate(case, validate)
            self.assertEqual(len(calls), 3)
            saved = json.loads(path.read_text())
            saved["proof"]["prediction"] = {"action": "delete_span", "target": [0]}
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "cache binding changed"):
                cache.validate(case, validate)


if __name__ == "__main__":
    unittest.main()
