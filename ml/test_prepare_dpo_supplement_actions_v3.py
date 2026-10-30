import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_dpo_supplement_actions_v3 as prepare


def fixture(root, mutate_case=None):
    """Small synthetic evidence chain; production pins remain non-overridable."""
    cases = []
    actions = (
        ["no_action"] * 14
        + ["delete_span"] * 4
        + ["split_paragraph"] * 2
        + ["merge_paragraph", "correct_text"]
    )
    for index, action in enumerate(actions):
        refs = [{"action": action}]
        if 14 <= index < 20:
            refs.append({"action": action, "alternative": True})
        cases.append(
            {
                "id": f"supplement-{index:02}",
                "split": "train",
                "language": "en",
                "input": {"text": f"Frozen input {index}", "history": []},
                "document": {"original": index},
                "source": {
                    "split": "train",
                    "source_group": "train-group",
                    "id": "train-recording",
                    "natural_audio": True,
                    "acoustically_verified": False,
                    "parent_provenance": "existing_recording",
                },
                "lineage": {
                    "stratum": "natural_recorded",
                    "parent_stratum": "legacy_curriculum",
                    "parent_id": f"parent-{index:03}",
                    "format_eligibility": {"json": {"eligible": True}},
                },
                "judgment": {
                    "status": "preserve" if action == "no_action" else "edit_required",
                    "independent_review": "pending_parent_review",
                },
                "primary": refs[0],
                "accepted_predictions": refs,
            }
        )
    admitted_ids = [case["id"] for case in cases]
    if mutate_case:
        mutate_case(cases[0])
    values, hashes, raw = {}, {}, {}

    def save(key, value):
        content = (
            value
            if isinstance(value, bytes)
            else (json.dumps(value, indent=2) + "\n").encode()
        )
        path = root / prepare.FILES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        raw[key], hashes[key] = content, prepare.bytes_hash(content)
        if not isinstance(value, bytes):
            values[key] = value

    def binding(key):
        return {"path": prepare.FILES[key], "sha256": hashes[key]}

    save("cases", b"".join((json.dumps(case) + "\n").encode() for case in cases))
    save("splits", {"train-group": "train", "dev-group": "dev"})
    parent_train, dev = (
        prepare.bytes_hash(b"old train"),
        prepare.bytes_hash(b"unchanged dev"),
    )
    save(
        "parent",
        {
            "counts": {"train": 848, "dev": 322},
            "files_sha256": {
                "cases/train.jsonl": parent_train,
                "cases/dev.jsonl": dev,
                "source-splits.json": hashes["splits"],
            },
        },
    )
    save(
        "component",
        {
            "counts": {"sampled": 24, "admitted": 22, "excluded": 2},
            "files_sha256": {
                "cases.jsonl": hashes["cases"],
                "source-splits.json": hashes["splits"],
            },
            "by_action": dict(Counter(actions)),
            "by_source_group": {"train-group": 22},
            "by_parent_stratum": {"legacy_curriculum": 22},
        },
    )
    save(
        "admission",
        {
            "status": "admitted_for_train_only_append_release",
            "parent_release": binding("parent"),
            "component": binding("component"),
            "case_ids": admitted_ids,
            "unchanged_development_cases": {"sha256": dev},
            "limitations": ["Dependent post-edit states."],
        },
    )
    save(
        "append",
        {
            "cases/train.jsonl": {
                "parent": {
                    "path": prepare.DATASET + "release-attempt-002/cases/train.jsonl",
                    "sha256": parent_train,
                },
                "suffix": binding("cases"),
                "suffix_bytes": len(raw["cases"]),
                "exact_prefix_and_suffix": True,
                "parent_unchanged": True,
            }
        },
    )
    save(
        "release",
        {
            "parent_release": binding("parent"),
            "components": {
                "parent": binding("parent"),
                "supplement": binding("component"),
            },
            "parent_admission": binding("admission"),
            "counts": {"train": 870, "dev": 322},
            "files_sha256": {
                "cases/dev.jsonl": dev,
                "source-splits.json": hashes["splits"],
                "append-integrity.json": hashes["append"],
            },
            "coverage": {
                "recorded_source_case_counts": [
                    {
                        "split": "train",
                        "source_group": "train-group",
                        "source_id": "train-recording",
                        "cases": 870,
                    },
                    {
                        "split": "dev",
                        "source_group": "dev-group",
                        "source_id": "dev-recording",
                        "cases": 322,
                    },
                ]
            },
        },
    )
    old_ids = [f"parent-{index:03}" for index in range(848)]
    dev_ids = [f"dev-{index:03}" for index in range(322)]
    results = [
        {"id": identity, "status": "passed"}
        for identity in old_ids + dev_ids + admitted_ids
    ]
    results.extend(
        {"id": identity, "status": "passed"}
        for identity in old_ids[:142] + admitted_ids[14:20]
    )
    save(
        "verification",
        {
            "status": "passed",
            "release_manifest_sha256": hashes["release"],
            "parent_release_manifest_sha256": hashes["parent"],
            "cases": 1192,
            "references": 1340,
            "added_cases": 22,
            "added_references": 28,
            "inputs_and_primaries_changed": 0,
            "counts_by_split": {"train": 870, "dev": 322},
            "final_test_bodies_read": False,
            "checks": ["no exact input or source-split collision"],
            "exact_old_bytes": {
                "cases/train.jsonl": parent_train,
                "cases/dev.jsonl": dev,
            },
            "component_admission": binding("admission"),
            "append_integrity": binding("append"),
            "results": results,
        },
    )
    save(
        "configuration",
        {
            "format": "json",
            "generation": {"max_new_tokens": 512, "do_sample": False, "num_beams": 1},
            "training": {"sample_order": old_ids},
            "unchanged_original_setting": "keep me",
        },
    )
    save(
        "preparation",
        {
            "files_sha256": {
                "config-json.json": hashes["configuration"],
                "train-cases.jsonl": parent_train,
            }
        },
    )

    def exposure(counts, updates):
        return {
            "completed_updates": updates,
            "optimizer_commit_uncertain": False,
            "pending_backward_rows_not_committed": 0,
            "phases": {
                name: {
                    "rows": sum(counts.values()),
                    "unique_cases": len(counts),
                    "counts": {"case_id": counts},
                }
                for name in ("forward", "backward", "optimizer_committed")
            },
        }

    save("exposure", exposure(dict.fromkeys(old_ids, 1), 212))
    save(
        "b003_exposure",
        exposure({**dict.fromkeys(old_ids, 1), **dict.fromkeys(admitted_ids, 31)}, 383),
    )
    adapter = prepare.bytes_hash(b"unchanged B002 adapter identity")
    save(
        "training",
        {
            "files_sha256": {
                "configuration.json": hashes["configuration"],
                "exposure.json": hashes["exposure"],
                "adapter/adapter_model.safetensors": adapter,
            }
        },
    )
    save("reload", {"files_sha256": {"configuration.json": hashes["configuration"]}})
    save("b003_training", {"files_sha256": {"exposure.json": hashes["b003_exposure"]}})
    save(
        "selection",
        {
            "status": "frozen",
            "selected_anchor": "B002_full_json",
            "test_bodies_read": False,
            "dataset": binding("parent"),
            "configuration": binding("configuration"),
            "preparation": binding("preparation"),
            "training": binding("training"),
            "reloaded": binding("reload"),
            "identity": {
                "configuration_sha256": hashes["configuration"],
                "training_manifest_sha256": hashes["training"],
                "reload_manifest_sha256": hashes["reload"],
                "adapter_sha256": adapter,
            },
        },
    )
    pins = {key: hashes[key] for key in prepare.TRUSTED_SHA256}
    return values, raw, pins


class SupplementPreparationTests(unittest.TestCase):
    def test_exact_roster_preserves_bytes_anchor_and_refuses_output_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            values, raw, pins = fixture(root)
            output = Path(directory) / "output"
            with patch.object(prepare, "TRUSTED_SHA256", pins):
                result = prepare.prepare(root, output)
                manifest_before = (output / "manifest.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "reuse is forbidden"):
                    prepare.prepare(root, output)
            self.assertEqual((output / "manifest.json").read_bytes(), manifest_before)
            self.assertEqual((output / "cases.jsonl").read_bytes(), raw["cases"])
            self.assertEqual(
                (output / "anchor/configuration.json").read_bytes(),
                raw["configuration"],
            )
            plan = json.loads((output / "candidate-input-plan.json").read_text())
            self.assertEqual(plan["case_order"], values["admission"]["case_ids"])
            self.assertEqual(result["counts"]["language"], {"en": 22, "ko": 0})
            self.assertEqual(
                result["counts"]["judgment"], {"preserve": 14, "edit_required": 8}
            )
            self.assertEqual(
                plan["original_anchor_identity"], values["selection"]["identity"]
            )
            self.assertFalse(plan["exposure_scope"]["b002_trained_on_these_states"])
            self.assertTrue(plan["exposure_scope"]["b003_trained_on_these_states"])
            self.assertFalse(plan["exposure_scope"]["globally_unseen"])
            self.assertFalse(plan["existing_generator_accepts_this_plan"])
            self.assertEqual(plan["generated_candidates"], 0)
            self.assertEqual(plan["generated_preference_pairs"], 0)
            self.assertFalse((output / "candidates.jsonl").exists())
            for name, expected in json.loads(manifest_before)["files_sha256"].items():
                self.assertEqual(prepare.file_hash(output / name), expected)

    def test_dev_case_or_dev_source_cannot_enter_otherwise_bound_suffix(self):
        mutations = (
            lambda case: case.update(split="dev"),
            lambda case: case["source"].update(
                source_group="dev-group", id="dev-recording"
            ),
        )
        for mutate in mutations:
            with (
                self.subTest(mutation=mutate),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory) / "input"
                _, _, pins = fixture(root, mutate)
                output = Path(directory) / "output"
                with patch.object(prepare, "TRUSTED_SHA256", pins):
                    with self.assertRaisesRegex(ValueError, "Non-TRAIN"):
                        prepare.prepare(root, output)
                self.assertFalse(output.exists())

    def test_replaced_case_is_rejected_even_with_self_consistent_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            _, _, pins = fixture(root, lambda case: case.update(id="unadmitted-case"))
            output = Path(directory) / "output"
            with patch.object(prepare, "TRUSTED_SHA256", pins):
                with self.assertRaises(ValueError):
                    prepare.prepare(root, output)
            self.assertFalse(output.exists())

    def test_changed_original_configuration_or_admission_fails_binding(self):
        for key in ("configuration", "admission"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "input"
                values, _, pins = fixture(root)
                altered = copy.deepcopy(values[key])
                if key == "configuration":
                    altered["generation"]["max_new_tokens"] = 1024
                else:
                    altered["case_ids"][0] = "unadmitted-case"
                (root / prepare.FILES[key]).write_text(json.dumps(altered))
                output = Path(directory) / "output"
                with patch.object(prepare, "TRUSTED_SHA256", pins):
                    with self.assertRaisesRegex(ValueError, "binding differs"):
                        prepare.prepare(root, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
