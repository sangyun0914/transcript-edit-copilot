import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import build_b003_final_freeze_actions_v3 as builder


class B003FreezeAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.registration = {
            "implementations": {
                "b003_builder": {"path": "new-builder.py", "sha256": "new-code"}
            },
            "actor_registry": {"path": "actors.json", "sha256": "actual-actors"},
            "exposure_review": {"path": "exposure.json", "sha256": "reviewed-exposure"},
            "dpo_model_receipt": {"path": "dpo.json", "sha256": "b003-dpo"},
            "proposed_selected_role": "sft",
        }
        self.reference = {"path": "registration.json", "sha256": "registration"}
        self.ready = {
            "b003_implementations": self.registration["implementations"],
            "b003_registration": self.reference,
            "actor_registry": self.registration["actor_registry"],
            "exposure_review": self.registration["exposure_review"],
            "dpo_model_receipt": self.registration["dpo_model_receipt"],
            "selected_deployment_receipt": {
                "path": "/workspace/deployment/deployment.json",
                "sha256": "deployment",
            },
            "selected_model_ui_receipt": {
                "path": "/workspace/ui/summary.json",
                "sha256": "ui",
            },
        }

    def core(self, original):
        return SimpleNamespace(
            preflight=original, read_bound=Mock(return_value=self.ready)
        )

    def test_old_readiness_without_exact_new_registration_never_reaches_core(self):
        for key in (
            "b003_implementations",
            "b003_registration",
            "actor_registry",
            "exposure_review",
            "dpo_model_receipt",
        ):
            with self.subTest(missing=key):
                ready = copy.deepcopy(self.ready)
                ready.pop(key)
                original = Mock()
                core = self.core(original)
                core.read_bound.return_value = ready
                builder.protect_preflight(core, self.registration, self.reference)
                with self.assertRaises(ValueError):
                    core.preflight(Path("ready.json"), "ready", Path("new-freeze"))
                original.assert_not_called()

    def test_original_parent_authority_rejection_is_not_bypassed(self):
        original = Mock(
            side_effect=ValueError("Required parent evidence assessments are missing.")
        )
        core = self.core(original)
        builder.protect_preflight(core, self.registration, self.reference)
        with self.assertRaisesRegex(ValueError, "parent evidence assessments"):
            core.preflight(Path("ready.json"), "ready", Path("new-freeze"))
        original.assert_called_once()

    def test_wrong_actual_ui_deployment_still_blocks_after_core_passes(self):
        context = {
            "selected_model": "sft",
            "models": {"sft": {"identity": {"adapter_sha256": "b003"}}},
        }
        core = self.core(Mock(return_value=context))
        core.read_bound.side_effect = [
            self.ready,
            {
                "adapter_export_receipt": {
                    "path": "/workspace/export/adapter-receipt.json"
                }
            },
        ]
        builder.protect_preflight(core, self.registration, self.reference)
        with (
            patch.object(
                builder.inputs,
                "export_binding",
                return_value={"path": "export.json", "sha256": "export"},
            ),
            patch.object(
                builder.inputs,
                "local_evidence",
                side_effect=ValueError("UI ran a different model deployment."),
            ) as check,
            self.assertRaisesRegex(ValueError, "different model deployment"),
        ):
            core.preflight(Path("ready.json"), "ready", Path("new-freeze"))
        check.assert_called_once()

    def test_author_and_grader_cannot_share_identity_even_with_evidence(self):
        registry = {
            "version": "task-test-actions-v3-actor-registry-1",
            "status": "parent_reviewed_roles",
            "actor": "/root",
            "reserved_bodies_read": False,
            "actors": {
                "preparer": "preparer",
                "author": "same",
                "reviewer": "reviewer",
                "grader": "same",
            },
            "access_evidence": {
                role: [{"path": role + ".json", "sha256": "evidence"}]
                for role in ("preparer", "author", "reviewer", "grader")
            },
        }
        reader = Mock()
        with self.assertRaisesRegex(ValueError, "distinct actual actors"):
            builder.actors_from(registry, reader)
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
