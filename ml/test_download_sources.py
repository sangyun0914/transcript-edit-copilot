import tempfile
import unittest
from pathlib import Path

import httpx

from download_sources import download


class BoundedDownloadTests(unittest.TestCase):
    def test_oversized_body_is_never_finalized_as_a_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            with httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=b"too large")
                )
            ) as client:
                with self.assertRaisesRegex(ValueError, "acquisition limit"):
                    download(client, "https://example.org/source", path, max_bytes=5)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".bin.json").exists())

    def test_reuse_verifies_contents_instead_of_trusting_cached_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            with httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=b"original")
                )
            ) as client:
                download(client, "https://example.org/source", path, max_bytes=8)
                path.write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "differs"):
                    download(client, "https://example.org/source", path, max_bytes=8)


if __name__ == "__main__":
    unittest.main()
