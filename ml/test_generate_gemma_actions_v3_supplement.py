"""Small synthetic checks for remote execution; no model or native tokenizer loads."""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import generate_gemma_actions_v3_supplement as candidates
import prepare_dpo_supplement_actions_v3 as frozen
from test_gemma_actions_v3 import Tokenizer as BaseTokenizer, cases as action_cases
from test_prepare_dpo_supplement_actions_v3 import fixture


class Tokenizer(BaseTokenizer):
    def decode(self, ids, *, skip_special_tokens):
        text = bytes(token - 256 for token in ids if token >= 256).decode()
        if not skip_special_tokens:
            text += "".join(
                {1: "<eos>", 106: "<end_of_turn>", 50: "<fim_suffix>"}[token]
                for token in ids
                if token in (1, 106, 50)
            )
        return text


@contextmanager
def input_bundle(directory):
    root = Path(directory)
    _, _, pins = fixture(root / "source")
    with patch.object(frozen, "TRUSTED_SHA256", pins):
        result = frozen.prepare(root / "source", root / "supplement")
        with patch.object(
            candidates, "SUPPLEMENT_MANIFEST_SHA256", result["manifest_sha256"]
        ):
            yield root / "supplement"


def token_bundle():
    originals = action_cases()
    cases = []
    for index in range(22):
        case = copy.deepcopy(originals[index % len(originals)])
        case["id"] = f"case-{index:02}"
        cases.append(case)
    identity = dict.fromkeys(candidates.ANCHOR_KEYS, "a" * 64)
    supplement = {
        "cases": cases,
        "plan": {"case_order": [case["id"] for case in cases]},
        "identity": identity,
        "bindings": dict.fromkeys(
            (
                "manifest_sha256",
                "plan_sha256",
                "cases_sha256",
                "roster_sha256",
                "configuration_sha256",
            ),
            "a" * 64,
        ),
    }
    tokenizer = Tokenizer()
    records, prompts = candidates.tokenize_supplement(supplement, tokenizer)
    values = []
    for case, record, prompt in zip(cases, records, prompts, strict=True):
        ids = record["primary"]["input_ids"][record["primary"]["completion_start"] :]
        raw = {
            "case_id": case["id"],
            "prompt_ids_sha256": prompt["prompt_ids_sha256"],
            "output_ids": ids,
            "terminal_token_id": ids[-1],
            "text": tokenizer.decode(ids, skip_special_tokens=True),
            "text_raw": tokenizer.decode(ids, skip_special_tokens=False),
            "finish_reason": "stop",
        }
        values.append(candidates.candidate_record(raw, case, prompt, supplement))
    return supplement, records, prompts, values, tokenizer


class SupplementalInputTests(unittest.TestCase):
    def test_exact_input_plan_retains_original_bytes_and_anchor(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            input_bundle(directory) as path,
        ):
            before = (path / "cases.jsonl").read_bytes()
            verified = candidates.verify_supplement(path / "candidate-input-plan.json")
            self.assertEqual(verified["manifest"]["cases"], 22)
            self.assertEqual(
                [case["id"] for case in verified["cases"]],
                verified["plan"]["case_order"],
            )
            self.assertEqual(verified["identity"], verified["manifest"]["anchor"])
            self.assertEqual(
                verified["bindings"]["manifest_sha256"],
                candidates.digest(path / "manifest.json"),
            )
            self.assertEqual((path / "cases.jsonl").read_bytes(), before)
            self.assertTrue(
                all(row["b002_optimizer_exposures"] == 0 for row in verified["roster"])
            )
            with self.assertRaisesRegex(ValueError, "input plan"):
                candidates.verify_supplement(path / "cases.jsonl")

    def test_payload_or_manifest_tampering_is_rejected(self):
        for name in (
            "cases.jsonl",
            "manifest.json",
            "anchor/configuration.json",
            "evidence/exposure.json",
        ):
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as directory,
                input_bundle(directory) as path,
            ):
                target = path / name
                target.write_bytes(target.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "Bound file changed"):
                    candidates.verify_supplement(path)

    def test_missing_or_extra_payload_is_rejected(self):
        for missing in (False, True):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as directory,
                input_bundle(directory) as path,
            ):
                if missing:
                    (path / "case-roster.jsonl").unlink()
                else:
                    (path / "unregistered-cases.jsonl").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "file inventory"):
                    candidates.verify_supplement(path)

    def test_wrong_verified_anchor_is_rejected(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            input_bundle(directory) as path,
        ):
            identity = candidates.read(path / "candidate-input-plan.json")["anchor"]
            identity["anchor_adapter_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "verified SFT anchor"):
                candidates.verify_supplement(path, identity)

    def test_reordered_roster_is_rejected_even_behind_a_resealed_test_pin(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            input_bundle(directory) as path,
        ):
            roster = candidates.prep.rows(path / "case-roster.jsonl")
            (path / "case-roster.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in reversed(roster))
            )
            plan = candidates.read(path / "candidate-input-plan.json")
            plan["case_roster"]["sha256"] = candidates.digest(
                path / "case-roster.jsonl"
            )
            candidates.write(path / "candidate-input-plan.json", plan)
            manifest = candidates.read(path / "manifest.json")
            for name in ("case-roster.jsonl", "candidate-input-plan.json"):
                manifest["files_sha256"][name] = candidates.digest(path / name)
            candidates.write(path / "manifest.json", manifest)
            with patch.object(
                candidates,
                "SUPPLEMENT_MANIFEST_SHA256",
                candidates.digest(path / "manifest.json"),
            ):
                with self.assertRaisesRegex(ValueError, "roster, order"):
                    candidates.verify_supplement(path)


class SupplementalCandidateManifestTests(unittest.TestCase):
    def test_candidate_manifest_binds_nested_input_manifest_for_full_inventory_readback(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as directory,
            input_bundle(directory) as input_path,
        ):
            output = Path(directory) / "candidates"
            output.mkdir()
            shutil.copytree(input_path, output / "supplement")
            shutil.copyfile(
                input_path / "anchor/configuration.json", output / "configuration.json"
            )
            candidates.write(
                output / "run.json",
                {
                    "version": candidates.VERSION,
                    "status": "candidates_complete",
                    "configuration_sha256": candidates.digest(
                        output / "configuration.json"
                    ),
                },
            )
            candidates.run_manifest(output, "candidates_complete")
            receipt, _ = candidates.sft.verified_run(
                output, "candidates_complete", version=candidates.VERSION
            )
            candidates.exact_files(output, receipt)
            inputs = candidates.verify_supplement(output / "supplement")
            self.assertEqual(
                receipt["files_sha256"]["supplement/manifest.json"],
                inputs["bindings"]["manifest_sha256"],
            )
            self.assertNotIn("manifest.json", receipt["files_sha256"])

            del receipt["files_sha256"]["supplement/manifest.json"]
            candidates.write(output / "manifest.json", receipt)
            incomplete, _ = candidates.sft.verified_run(
                output, "candidates_complete", version=candidates.VERSION
            )
            with self.assertRaisesRegex(ValueError, "file inventory"):
                candidates.exact_files(output, incomplete)


class SupplementalTokenAndCandidateTests(unittest.TestCase):
    def test_all_22_model_rows_contain_only_the_native_prompt_prefix(self):
        supplement, records, prompts, values, tokenizer = token_bundle()
        candidates.validate_candidate_rows(
            supplement, records, prompts, values, tokenizer
        )
        rows = candidates.generation_rows(prompts)
        self.assertEqual(
            [row["case_id"] for row in rows], supplement["plan"]["case_order"]
        )
        for row, record in zip(rows, records, strict=True):
            self.assertEqual(set(row), {"case_id", "input_ids", "completion_start"})
            self.assertEqual(
                row["input_ids"],
                record["primary"]["input_ids"][: record["primary"]["completion_start"]],
            )
            self.assertEqual(len(row["input_ids"]), row["completion_start"])
            self.assertLess(len(row["input_ids"]), len(record["primary"]["input_ids"]))

    def test_tokenizer_exclusion_cannot_become_a_smaller_successful_run(self):
        supplement, records, _, _, tokenizer = token_bundle()
        with patch.object(
            candidates.prep,
            "tokenize_cases",
            return_value=(
                supplement["cases"][:-1],
                {"json": records[:-1]},
                [{"case_id": "case-21", "reason": "json_input_budget"}],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "dropping or truncation"):
                candidates.tokenize_supplement(supplement, tokenizer)

    def test_runtime_reconstruction_must_preserve_the_exact_input(self):
        supplement, _, _, _, tokenizer = token_bundle()
        with patch.object(
            candidates.dpo.dataset,
            "runtime_bundle",
            return_value={"input": {"changed": True}},
        ):
            with self.assertRaisesRegex(ValueError, "runtime input changed"):
                candidates.tokenize_supplement(supplement, tokenizer)

    def test_missing_duplicate_or_reordered_candidates_cannot_be_complete(self):
        supplement, records, prompts, values, tokenizer = token_bundle()
        for altered in (
            values[:-1],
            [values[0], *values[2:], values[0]],
            list(reversed(values)),
        ):
            with self.subTest(case_ids=[row["case_id"] for row in altered]):
                with self.assertRaisesRegex(ValueError, "exact 22-case ordered roster"):
                    candidates.validate_candidate_rows(
                        supplement, records, prompts, altered, tokenizer
                    )

    def test_reference_completion_cannot_leak_into_a_self_consistent_saved_prompt(self):
        supplement, records, prompts, values, tokenizer = token_bundle()
        prompt = prompts[0]
        prompt["input_ids"] = list(records[0]["primary"]["input_ids"])
        prompt["completion_start"] = len(prompt["input_ids"])
        prompt["prompt_ids_sha256"] = candidates.prep.value_hash(prompt["input_ids"])
        values[0]["prompt_ids_sha256"] = prompt["prompt_ids_sha256"]
        with self.assertRaisesRegex(ValueError, "reference leakage"):
            candidates.validate_candidate_rows(
                supplement, records, prompts, values, tokenizer
            )

    def test_resealed_prefix_tokens_still_require_native_tokenizer_parity(self):
        supplement, records, prompts, values, tokenizer = token_bundle()
        records[0]["primary"]["input_ids"][0] = 3
        prompts[0] = candidates.prompt_record(supplement["cases"][0], records[0])
        values[0]["prompt_ids_sha256"] = prompts[0]["prompt_ids_sha256"]
        with self.assertRaisesRegex(ValueError, "Actual Gemma tokenizer differs"):
            candidates.validate_candidate_rows(
                supplement, records, prompts, values, tokenizer
            )

    def test_rewritten_candidate_text_is_rejected_even_with_updated_parse_and_hash(
        self,
    ):
        supplement, records, prompts, values, tokenizer = token_bundle()
        value = values[1]
        value["text"] = '{"action":"no_action"}'
        value["text_raw"] = value["text"] + "<eos>"
        value["raw_output_sha256"] = frozen.bytes_hash(value["text_raw"].encode())
        value["parsed_action"], value["validation_error"] = candidates.parsed_output(
            value["text"], supplement["cases"][1], "stop"
        )
        with self.assertRaisesRegex(ValueError, "raw generated tokens"):
            candidates.validate_candidate_rows(
                supplement, records, prompts, values, tokenizer
            )

    def test_parse_failure_is_preserved_as_a_candidate_without_inventing_an_action(
        self,
    ):
        supplement, records, prompts, values, tokenizer = token_bundle()
        ids = tokenizer.encode("not JSON", add_special_tokens=False) + [1]
        raw = {
            "case_id": supplement["cases"][0]["id"],
            "prompt_ids_sha256": prompts[0]["prompt_ids_sha256"],
            "output_ids": ids,
            "terminal_token_id": 1,
            "text": tokenizer.decode(ids, skip_special_tokens=True),
            "text_raw": tokenizer.decode(ids, skip_special_tokens=False),
            "finish_reason": "stop",
        }
        values[0] = candidates.candidate_record(
            raw, supplement["cases"][0], prompts[0], supplement
        )
        candidates.validate_candidate_rows(
            supplement, records, prompts, values, tokenizer
        )
        self.assertIsNone(values[0]["parsed_action"])
        self.assertTrue(values[0]["validation_error"])
        self.assertEqual(values[0]["output_ids"], ids)


if __name__ == "__main__":
    unittest.main()
