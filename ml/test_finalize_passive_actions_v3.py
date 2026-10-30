import unittest

from finalize_passive_actions_v3 import source_neighbors, stratum


def natural_case(identity, source_id, start, end, *, old=False, split="train"):
    return {
        "id": identity,
        "split": split,
        "source": {
            "id": source_id,
            "source_group": "shared-speaker-family",
            "natural_audio": True,
            "acoustically_verified": False,
        },
        "input": {"items": [{"source_start_ms": start, "source_end_ms": end}]},
        "lineage": {
            "selection_method": "source_first",
            "old_release": "completion-v2" if old else None,
            "scenario_family": "source_checkpoint",
        },
    }


class ActionsV3AssemblyTests(unittest.TestCase):
    def test_natural_source_stratum_does_not_invent_acoustic_certification(self):
        case = natural_case("new", "recording-a", 0, 10)
        self.assertEqual(stratum(case), "natural_recorded")
        self.assertFalse(case["source"]["acoustically_verified"])
        self.assertEqual(
            stratum(natural_case("old", "recording-a", 0, 10, old=True)),
            "legacy_source_first",
        )

    def test_neighbor_audit_uses_recording_identity_and_actual_overlap(self):
        cases = [
            natural_case("a", "recording-a", 0, 1000),
            natural_case("a-variant", "recording-a", 100, 1000),
            natural_case("different-recording", "recording-b", 0, 1000),
            natural_case("later", "recording-a", 2000, 3000),
        ]
        neighbors = source_neighbors(cases)
        self.assertEqual(
            [(row["left_id"], row["right_id"]) for row in neighbors],
            [("a", "a-variant")],
        )
        cases[1]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "cross splits"):
            source_neighbors(cases)


if __name__ == "__main__":
    unittest.main()
