import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from acquire_passive_sft import download, pin


class NativeAcquisitionTests(unittest.TestCase):
    def test_changed_pin_is_rejected_before_network_or_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "pin.json"
            receipt.write_text('{"version":"passive-native-pin-v1"}')
            with patch("acquire_passive_sft.httpx.Client") as client:
                with self.assertRaisesRegex(ValueError, "execution binding"):
                    download(receipt, root / "native", "0" * 64)
                client.assert_not_called()
                self.assertFalse((root / "native").exists())

    def test_missing_selected_tokenizer_does_not_publish_pin(self):
        repository, revision = "Qwen/Qwen3.5-9B", "a" * 40
        selection = {
            "selected_t1_candidate_ids": ["qwen35-9b"],
            "selected_native_identity": {
                "base_repository": repository,
                "base_revision": revision,
                "tokenizer_files_sha256": {"tokenizer.json": "b" * 64},
            },
        }
        response = MagicMock()
        response.json.return_value = {
            "id": repository,
            "sha": revision,
            "siblings": [
                {"rfilename": "LICENSE"},
                {"rfilename": "model.safetensors.index.json"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selection.json"
            selected.write_text(json.dumps(selection))
            with patch("acquire_passive_sft.httpx.Client") as client:
                client.return_value.__enter__.return_value.get.return_value = response
                with self.assertRaisesRegex(
                    ValueError, "required model/tokenizer/license"
                ):
                    pin(selected, root / "output")
                self.assertFalse((root / "output/native-pin.json").exists())


if __name__ == "__main__":
    unittest.main()
