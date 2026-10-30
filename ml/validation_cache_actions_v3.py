"""Persist immutable replay proofs bound to inputs, runtime and implementation."""

import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ReferenceProofCache:
    def __init__(self, directory, implementation):
        self.directory = Path(directory)
        self.implementation = implementation
        self.hits = 0
        self.misses = 0

    def validate(self, case, validator):
        bound = {key: case[key] for key in ("document", "input", "messages", "runtime", "source", "split", "judgment", "primary", "accepted_predictions")}
        binding = digest({"case": bound, "implementation": self.implementation})
        path = self.directory / (binding + ".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["binding"] != binding or saved["proof_sha256"] != digest(saved["proof"]):
                raise ValueError("Immutable validation cache binding changed.")
            proof = saved["proof"]
            self.hits += 1
        else:
            proof = validator(case)
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                json.dump({"binding": binding, "case_id": case["id"], "implementation": self.implementation,
                           "proof_sha256": digest(proof), "proof": proof}, stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            self.misses += 1
        if "execution_sha256" in case and case["execution_sha256"] != digest(proof):
            raise ValueError("Cached execution proof differs from the case.")
        return proof
