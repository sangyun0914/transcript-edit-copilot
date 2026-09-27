import ctypes
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import clone_passive_export as clone


@unittest.skipUnless(
    sys.platform == "darwin" and hasattr(ctypes.CDLL(None), "clonefile"),
    "Requires native copy-on-write clones",
)
class CloneExportTests(unittest.TestCase):
    def fixture(self, directory):
        base, target, destination = (
            Path(directory) / name for name in ("base", "target", "clone")
        )
        base.write_bytes(b"a" * 4096 + b"b" * 4096 + b"c" * 4096)
        target.write_bytes(b"a" * 4096 + b"z" * 4096 + b"c" * 4096)
        return base, target, destination

    def reconstruct(self, paths, **overrides):
        base, target, destination = paths
        values = {
            "base_sha256": clone.digest(base),
            "target_sha256": clone.digest(target),
            "max_changed_bytes": 4096,
            "minimum_free_bytes": 0,
            "chunk_bytes": 4096,
        }
        values.update(overrides)
        return clone.rebuild_clone(base, target, destination, **values)

    def test_real_clone_preserves_exact_target_and_both_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory)
            originals = [p.read_bytes() for p in paths[:2]]
            receipt = self.reconstruct(paths)
            self.assertEqual(paths[2].read_bytes(), originals[1])
            self.assertEqual([p.read_bytes() for p in paths[:2]], originals)
            self.assertEqual(receipt["changed_chunk_bytes"], 4096)
            self.assertEqual(receipt["changed_chunks"], 1)
            with paths[2].open("r+b") as stream:
                stream.write(b"other")
            self.assertEqual([p.read_bytes() for p in paths[:2]], originals)

    def test_failed_hash_or_budget_leaves_inputs_and_no_partial_destination(self):
        for overrides in (
            {"target_sha256": hashlib.sha256(b"wrong").hexdigest()},
            {"base_sha256": hashlib.sha256(b"wrong").hexdigest()},
            {"max_changed_bytes": 0},
            {"minimum_free_bytes": 10**30},
        ):
            with (
                self.subTest(overrides=overrides),
                tempfile.TemporaryDirectory() as directory,
            ):
                paths = self.fixture(directory)
                originals = [p.read_bytes() for p in paths[:2]]
                with self.assertRaises(ValueError):
                    self.reconstruct(paths, **overrides)
                self.assertFalse(paths[2].exists())
                self.assertEqual([p.read_bytes() for p in paths[:2]], originals)

    def test_existing_destination_and_symlink_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(directory)
            paths[2].write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                self.reconstruct(paths)
            self.assertEqual(paths[2].read_bytes(), b"keep")
            paths[2].unlink()
            paths[2].symlink_to(paths[0])
            with self.assertRaises(ValueError):
                self.reconstruct(paths)
            self.assertTrue(paths[2].is_symlink())


if __name__ == "__main__":
    unittest.main()
