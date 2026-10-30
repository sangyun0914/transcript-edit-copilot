import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import append_natural_preservation_actions_v3 as append


class AppendIsolationTests(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform == "darwin", "Production preparation requires APFS cloning."
    )
    def test_real_clone_appends_exact_bytes_without_touching_parent(self):
        with tempfile.TemporaryDirectory(dir=append.ROOT) as folder:
            root = Path(folder)
            source, suffix, target = [
                root / name for name in ("source.jsonl", "suffix.jsonl", "target.jsonl")
            ]
            source.write_bytes(b'{"id":"old"}\n')
            suffix.write_bytes(b'{"id":"new"}\n')
            receipt = append.clone_append(source, target, suffix)
            self.assertEqual(source.read_bytes(), b'{"id":"old"}\n')
            self.assertEqual(
                target.read_bytes(), source.read_bytes() + suffix.read_bytes()
            )
            self.assertFalse(os.path.samefile(source, target))
            self.assertTrue(receipt["exact_prefix_and_suffix"])

    def test_hardlink_returned_by_clone_is_rejected_before_append(self):
        with tempfile.TemporaryDirectory(dir=append.ROOT) as folder:
            root = Path(folder)
            source, suffix, target = [
                root / name for name in ("source", "suffix", "target")
            ]
            source.write_bytes(b"{}\n")
            suffix.write_bytes(b'{"new":true}\n')
            with patch.object(append, "clone", side_effect=os.link):
                with self.assertRaisesRegex(ValueError, "parent inode"):
                    append.clone_append(source, target, suffix)
            self.assertEqual(source.read_bytes(), b"{}\n")

    def test_unavailable_clone_never_falls_back_to_a_giant_copy(self):
        with tempfile.TemporaryDirectory(dir=append.ROOT) as folder:
            root = Path(folder)
            source, target = root / "source", root / "target"
            source.write_bytes(b"{}\n")
            with patch.object(append.sys, "platform", "linux"):
                with self.assertRaisesRegex(ValueError, "no full-copy fallback"):
                    append.clone(source, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
