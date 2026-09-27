import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import prepare_passive_gemma_matched as preparation
import test_prepare_passive_gemma as gemma_fixture


class GemmaMatchedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = gemma_fixture.GemmaPreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.release = self.fixture.release
        self.reference = self.fixture.qwen
        self.output = self.fixture.output

    def build(self, mutate=None):
        fixture = self.fixture
        fixture.build()
        cases = copy.deepcopy(fixture.cases[:802])
        tokens = {"json": copy.deepcopy(fixture.tokens[:802])}
        tokens["rewrite"] = copy.deepcopy(tokens["json"])
        for index, row in enumerate(tokens["rewrite"]):
            row["messages"][0]["content"] = "rewrite-policy"
            row["references"] = ['{"format_version":1,"scope":[],"after":[]}']
            prefix = 3 + index % 17
            row["primary"] = {
                "input_ids": [2] * prefix + [8, 9, 1],
                "attention_mask": [1] * (prefix + 3),
                "labels": [-100] * prefix + [8, 9, 1],
                "completion_start": prefix,
            }
        if mutate:
            mutate(cases, tokens)
        self.cases, self.tokens = cases, tokens
        manifest_path = self.release / "manifest.json"
        release = json.loads(manifest_path.read_bytes())
        for name in ("passive_bundle_v2.py", "passive_formats_v2.py"):
            release["dependencies_sha256"]["ml/" + name] = fixture.save(
                self.root / "ml" / name,
                Path(preparation.__file__).with_name(name).read_bytes(),
            )
        payloads = {
            "matched/train-cases.jsonl": preparation.smoke.jsonl_bytes(cases),
            "tokenized/gemma26/json-train.jsonl.gz": gzip.compress(
                preparation.smoke.jsonl_bytes(tokens["json"]), mtime=0
            ),
            "tokenized/gemma26/rewrite-train.jsonl.gz": gzip.compress(
                preparation.smoke.jsonl_bytes(tokens["rewrite"]), mtime=0
            ),
            "prompt-rewrite.txt": b"rewrite-policy\n",
            "matched/eligibility.json": preparation.smoke.json_bytes(
                {"qualified": {"train": 802, "dev": 286}}
            ),
        }
        for name, value in payloads.items():
            release["files_sha256"][name] = fixture.save(self.release / name, value)
        for name in (
            "matched/dev-cases.jsonl",
            "tokenized/gemma26/json-dev.jsonl.gz",
            "tokenized/gemma26/rewrite-dev.jsonl.gz",
        ):
            release["files_sha256"][name] = fixture.save(
                self.release / name, b"must not be read"
            )
        self.release_sha = fixture.save(
            manifest_path, preparation.smoke.json_bytes(release)
        )
        indices = preparation.curriculum.weighted_draw_indices(cases, 802, "3407")
        self.unique = len(set(indices))
        draw_plan = [
            {"draw_index": pos, "case_id": cases[index]["id"], "train_row_index": index}
            for pos, index in enumerate(indices)
        ]
        reference_files = {
            name: release["files_sha256"][original]
            for name, original in preparation.PAYLOADS.items()
        }
        reference_files["draw-plan.json"] = fixture.save(
            self.reference / "draw-plan.json", preparation.smoke.json_bytes(draw_plan)
        )
        self.reference_sha = fixture.save(
            self.reference / "manifest.json",
            preparation.smoke.json_bytes(
                {
                    "version": "passive-sft-matched-preparation-v1",
                    "status": preparation.STATUS,
                    "candidate_id": "qwen35-9b",
                    "release_manifest_sha256": self.release_sha,
                    "files_sha256": reference_files,
                }
            ),
        )

    def patches(self):
        return (
            patch.object(preparation, "RELEASE_SHA256", self.release_sha),
            patch.object(preparation, "REFERENCE_SHA256", self.reference_sha),
            patch.object(preparation, "UNIQUE_DRAWS", self.unique),
        )

    def synthetic_semantics(self, cases, tokens):
        self.assertEqual(len(cases), 802)
        self.assertEqual(set(tokens), {"json", "rewrite"})
        references = sum(1 + len(row["acceptableOutputs"]) for row in cases)
        return {
            "status": "passed_local_paired_reference_readback",
            "train_cases": len(cases),
            "references_per_format": references,
            "format_reference_checks": 2 * references,
            "canonical_message_checks": 2 * len(cases),
            "training_primary_targets_only": True,
            "retokenized": False,
            "dev_test_bodies_read": 0,
        }

    def prepare(self):
        a, b, c = self.patches()
        with (
            a,
            b,
            c,
            patch.object(preparation, "semantic_readback", self.synthetic_semantics),
        ):
            return preparation.prepare(
                self.release,
                self.fixture.registry,
                self.reference,
                self.output,
                registry_sha256=self.fixture.registry_sha,
                project_root=self.root,
            )

    def verify(self):
        a, b, c = self.patches()
        with a, b, c:
            return preparation.verify_bundle(self.output)

    def rewrite_payload(self, name, data):
        self.fixture.save(self.output / name, data)
        path = self.output / "manifest.json"
        value = json.loads(path.read_bytes())
        value["files_sha256"][name] = preparation.sha256(data)
        for row in value["inventory"]:
            if row["path"] == name:
                row.update(bytes=len(data), sha256=preparation.sha256(data))
        value["payload_bytes"] = sum(row["bytes"] for row in value["inventory"])
        self.fixture.save(path, preparation.smoke.json_bytes(value))

    @staticmethod
    def deny_held_out(path, original):
        if (
            "dev-cases" in path.name
            or "test-cases" in path.name
            or "-dev.jsonl" in path.name
        ):
            raise AssertionError("Held-out body read")
        return original(path)

    def test_shared_draws_gemma_phase_budgets_and_train_only_inventory(self):
        self.build()
        read_bytes = Path.read_bytes
        with patch.object(
            Path, "read_bytes", lambda p: self.deny_held_out(p, read_bytes)
        ):
            result = self.prepare()
            bundle = self.verify()
        config = bundle["configuration"]
        self.assertEqual(config["candidate_id"], "gemma26")
        self.assertEqual(config["training"]["optimizer_updates"], 201)
        self.assertEqual(config["training"]["final_update_rows"], 2)
        self.assertEqual(config["sampling"]["seed"], "3407")
        self.assertEqual(config["tokenizer"]["eos_token_id"], 1)
        self.assertEqual(
            config["generation_metadata"]["native_generation_stop_ids"], [1, 106, 50]
        )
        self.assertEqual(
            bundle["draw_plan"],
            json.loads((self.reference / "draw-plan.json").read_bytes()),
        )
        selected = preparation.matched.selected_diagnostics(self.cases, self.tokens)
        self.assertNotEqual(
            selected, preparation.smoke.select_smoke(self.cases, self.tokens["json"])
        )
        expected_ids = [self.cases[index]["id"] for index in selected]
        for fmt, completion_length in (("json", 2), ("rewrite", 3)):
            view = config["views"][fmt]
            self.assertEqual(
                view["max_total_completion_tokens"], 802 * completion_length
            )
            self.assertEqual(
                view["diagnostic_completion_tokens"], 8 * completion_length
            )
            self.assertEqual(view["diagnostic_case_order"], expected_ids)
            self.assertEqual(bundle["views"][fmt]["tokens"], self.tokens[fmt])
            self.assertEqual(
                [r["case_id"] for r in bundle["views"][fmt]["diagnostics"]["rows"]],
                expected_ids,
            )
        self.assertEqual(result["dev_test_record_bodies_read"], 0)
        self.assertEqual(result["training_updates_executed"], 0)
        self.assertEqual(preparation.matched.curriculum.CANDIDATE, "qwen35-9b")
        portable = subprocess.run(
            [
                sys.executable,
                "-B",
                "-S",
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import prepare_passive_gemma_matched as p; "
                "p.RELEASE_SHA256=sys.argv[3]; p.REFERENCE_SHA256=sys.argv[4]; "
                "p.UNIQUE_DRAWS=int(sys.argv[5]); "
                "print(p.verify_bundle(sys.argv[2])['verification']['status'])",
                str(self.output / "source"),
                str(self.output),
                self.release_sha,
                self.reference_sha,
                str(self.unique),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(
            portable.stdout.strip(), "passed_matched_train_only_gemma_readback"
        )
        with self.assertRaisesRegex(ValueError, "fresh output"):
            self.prepare()

    def test_rejects_nontrain_rewrite_eos_masks_or_different_user_evidence(self):
        def bad_eos(cases, tokens):
            value = tokens["rewrite"][0]["primary"]
            value["input_ids"][-1] = value["labels"][-1] = 106

        for mutation, message in (
            (lambda c, t: c[0]["metadata"].update(source_group_id="dev"), "Non-train"),
            (bad_eos, "EOS"),
            (
                lambda c, t: t["rewrite"][0]["primary"]["labels"].__setitem__(0, 2),
                "mask",
            ),
            (
                lambda c, t: t["rewrite"][0]["messages"][1].update(content="different"),
                "evidence",
            ),
        ):
            with self.subTest(message=message):
                self.build(mutation)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_rejects_resigned_draw_order_or_format_budget(self):
        self.build()
        self.prepare()
        original = (self.output / "draw-plan.json").read_bytes()
        changed = json.loads(original)
        changed[0], changed[1] = changed[1], changed[0]
        self.rewrite_payload("draw-plan.json", preparation.smoke.json_bytes(changed))
        with self.assertRaisesRegex(ValueError, "draw-plan replay"):
            self.verify()
        self.rewrite_payload("draw-plan.json", original)
        config = json.loads((self.output / "configuration.json").read_bytes())
        config["views"]["rewrite"]["max_total_completion_tokens"] += 1
        self.rewrite_payload("configuration.json", preparation.smoke.json_bytes(config))
        with self.assertRaisesRegex(ValueError, "phase budgets"):
            self.verify()

    def test_extra_body_or_symlink_rejected_before_read(self):
        self.build()
        self.prepare()
        extra = self.output / "dev-cases.jsonl"
        extra.write_bytes(b"must not be read")
        read_bytes = Path.read_bytes
        with (
            patch.object(
                Path, "read_bytes", lambda p: self.deny_held_out(p, read_bytes)
            ),
            self.assertRaisesRegex(ValueError, "Physical bundle inventory"),
        ):
            self.verify()
        extra.unlink()
        target = self.output / "source-groups.json"
        data = target.read_bytes()
        target.unlink()
        replacement = self.root / "groups.json"
        replacement.write_bytes(data)
        target.symlink_to(replacement)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.verify()

    def test_foreign_native_identity_or_frozen_helper_cannot_be_resigned(self):
        self.build()
        self.prepare()
        name = "evidence/model-registry.json"
        original = (self.output / name).read_bytes()
        changed = json.loads(original)
        changed["candidates"][0]["base_revision"] = "0" * 40
        self.rewrite_payload(name, preparation.smoke.json_bytes(changed))
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["bindings"]["model_registry"]["sha256"] = manifest["files_sha256"][
            name
        ]
        manifest_path.write_bytes(preparation.smoke.json_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "native revision"):
            self.verify()
        self.rewrite_payload(name, original)
        manifest = json.loads(manifest_path.read_bytes())
        manifest["bindings"]["model_registry"]["sha256"] = manifest["files_sha256"][
            name
        ]
        manifest_path.write_bytes(preparation.smoke.json_bytes(manifest))
        self.rewrite_payload("source/prepare_passive_matched.py", b"changed helper\n")
        with self.assertRaisesRegex(ValueError, "frozen preparation helper"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
