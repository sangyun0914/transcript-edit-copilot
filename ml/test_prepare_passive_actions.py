import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import prepare_passive_actions as pilot


class Tokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        rendered = json.dumps(messages, ensure_ascii=False)
        return [8] * 100 if tokenize else rendered

    def encode(self, text, **kwargs):
        return [ord(character) + 1000 for character in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(value - 1000) for value in ids)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer()
        cls.builder = pilot.contract.PassiveContextBuilder(cls.tokenizer)
        cls.cases, cls.runtime, cls.executions, cls.controls = pilot.build_cases(
            cls.builder
        )

    def test_balanced_distinct_groups_and_no_answer_in_observable_input(self):
        self.assertEqual(len(self.cases), 56)
        sources = {case["source"]["id"] for case in self.cases}
        self.assertEqual(len(sources), 56)
        groups = {}
        for split, count in (("train", 3), ("dev", 1)):
            selected = [case for case in self.cases if case["split"] == split]
            cells = Counter(
                (case["primary"]["action"], case["language"]) for case in selected
            )
            self.assertEqual(len(cells), 14)
            self.assertEqual(set(cells.values()), {count})
            groups[split] = {case["source"]["source_group"] for case in selected}
        self.assertFalse(groups["train"] & groups["dev"])
        for case in self.cases:
            user = case["messages"][1]["content"]
            self.assertNotIn(case["id"], user)
            self.assertNotIn(case["source"]["source_group"], user)
            self.assertNotIn(case["rationale"], user)
            self.assertNotIn("task_brief", user)
            self.assertNotIn("accepted_predictions", user)
            self.assertNotIn('"split":', user)
            for item in case["input"]["items"]:
                self.assertRegex(item["source_id"], r"^symbolic:[a-f0-9-]{36}$")
                self.assertEqual(item["boundary_status"], "unverified")
                self.assertIn(item["audio_class"], {"speech", "unknown", "audio_event"})

    def test_runtime_inputs_reproduce_messages_and_real_history(self):
        self.assertEqual(pilot.PROFILE, pilot.contract.CaptionProfile().model_dump())
        count = 0
        for case, runtime in zip(self.cases, self.runtime, strict=True):
            bundle = self.builder.build(
                runtime["document"],
                runtime["focus_uuid"],
                as_of=runtime["sequence"],
                selection=runtime["selection"],
                history=runtime["history"],
                caption_profile=runtime["caption_profile"],
            )
            self.assertEqual(
                pilot.contract.canonical_messages(bundle["input"]), case["messages"]
            )
            if runtime["history"]:
                count += 1
                transition = runtime["history"][0]
                self.assertEqual(transition["origin"], "synthetic")
                self.assertNotEqual(transition["before"], transition["after"])
                self.assertEqual(transition["after"], runtime["document"])
                self.assertEqual(
                    transition["prediction"]["action"], case["primary"]["action"]
                )
                damaged = copy.deepcopy(runtime["history"])
                damaged[0]["after"]["content"][1]["length"] += 0.1
                with self.assertRaises(ValueError):
                    self.builder.build(
                        runtime["document"],
                        runtime["focus_uuid"],
                        as_of=1,
                        history=damaged,
                    )
        self.assertEqual(count, 16)

    def test_reference_preview_apply_undo_redo_and_invalid_controls(self):
        operations = Counter()
        for case, result in zip(self.cases, self.executions, strict=True):
            bound = {key: value for key, value in result.items() if key != "id"}
            self.assertEqual(pilot.digest(bound), case["execution_sha256"])
            if result["changed"]:
                final, summary = pilot.replay(result["journal"])
                self.assertEqual(summary["transactions"], 3)
                self.assertEqual(pilot.session_hash(final), result["after_sha256"])
                self.assertEqual(
                    result["journal"][1]["operation"], case["primary"]["action"]
                )
            else:
                self.assertEqual(result["before_sha256"], result["after_sha256"])
            operations[case["primary"]["action"]] += 1
        self.assertEqual(set(operations.values()), {8})
        probes = [probe for row in self.controls for probe in row["probes"]]
        self.assertEqual(len(probes), 90)
        self.assertTrue(all(probe["rejected"] for probe in probes))
        self.assertEqual(sum(p["name"] == "known_audio_event" for p in probes), 2)

    def test_token_boundary_roundtrip_and_explicit_eos(self):
        row = pilot.encode(self.cases[0], self.tokenizer)
        primary = row["primary"]
        self.assertEqual(primary["completion_start"], 100)
        self.assertEqual(primary["input_ids"][:100], [8] * 100)
        self.assertEqual(primary["input_ids"][-1], 1)
        self.assertEqual(
            self.tokenizer.decode(primary["input_ids"][100:-1]),
            pilot.packed(self.cases[0]["primary"]),
        )
        with patch.object(
            self.tokenizer, "apply_chat_template", return_value=[8, 1, 8]
        ):
            row = pilot.encode(self.cases[0], self.tokenizer)
            self.assertEqual(row["primary"]["input_ids"][:3], [8, 1, 8])
            self.assertEqual(row["primary"]["completion_start"], 3)
        with patch.object(self.tokenizer, "decode", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "round-trip"):
                pilot.encode(self.cases[0], self.tokenizer)

    def test_existing_attempt_is_not_overwritten_and_positive_inventory_is_enforced(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pilot"
            identities = {name: {"eos_token_id": 1} for name in pilot.TOKENIZERS}
            tokenizers = dict.fromkeys(pilot.TOKENIZERS, self.tokenizer)
            with patch.object(
                pilot, "load_tokenizers", return_value=(tokenizers, identities)
            ):
                manifest = pilot.prepare(output)
                result = pilot.verify_bundle(output, tokenizers)
                self.assertEqual(result["native_tokenizer_rows_verified"], 112)
                self.assertEqual(
                    manifest["counts"], {"train": 42, "dev": 14, "test": 0}
                )
                with self.assertRaisesRegex(ValueError, "fresh attempt"):
                    pilot.prepare(output)
                (output / "unexpected.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "inventory"):
                    pilot.verify_bundle(output, tokenizers)


if __name__ == "__main__":
    unittest.main()
