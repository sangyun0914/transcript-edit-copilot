import hashlib
import tempfile
import unittest
from pathlib import Path

import httpx

from acquire_b1_models import acquire


class AcquireB1Tests(unittest.TestCase):
    def test_bad_hash_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.gguf"
            with httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"bad")
            )) as client, self.assertRaisesRegex(ValueError, "SHA256"):
                acquire(client, "https://example.test/model", target, 3, "wrong")
            self.assertFalse(target.exists())
            self.assertEqual(target.with_suffix(".gguf.part").read_bytes(), b"bad")

    def test_valid_resume_and_server_ignoring_range(self):
        payload = b"abcdef"
        for honors_range in (True, False):
            with self.subTest(honors_range=honors_range), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "model.gguf"
                target.with_suffix(".gguf.part").write_bytes(payload[:3])

                def response(request):
                    self.assertEqual(request.headers["Range"], "bytes=3-")
                    return httpx.Response(206, headers={"Content-Range": "bytes 3-5/6"}, content=payload[3:]) if honors_range else httpx.Response(200, content=payload)

                with httpx.Client(transport=httpx.MockTransport(response)) as client:
                    acquire(client, "https://example.test/model", target, 6, hashlib.sha256(payload).hexdigest())
                self.assertEqual(target.read_bytes(), payload)
                self.assertFalse(target.with_suffix(".gguf.part").exists())


if __name__ == "__main__":
    unittest.main()
