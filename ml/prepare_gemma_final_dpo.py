"""Mine final1261 Gemma errors and prepare same-parent DPO on saved projections."""

import argparse
from collections import Counter
import json
from pathlib import Path

import gemma_final_readiness as final
import prepare_gemma_actions_v3 as prep
import prepare_small_actions_final_dpo as semantic
from prepare_final_unified_dataset import bound_evaluation_bundle
import train_gemma_actions_v3_dpo as dpo

VERSION = "gemma26-final-dpo-mining-1"
require, read, sha, write = final.require, final.read, final.sha, final.write


def decisions_for(cases, observations, candidate_sha, reviews=()):
    indexed, reviewed = ({row["case_id"]: row for row in rows} for rows in (observations, reviews))
    require(len(cases) == len(observations) == len(indexed)
            and {case["id"] for case in cases} == set(indexed)
            and len(reviewed) == len(reviews) and set(reviewed) <= set(indexed),
            "Account for the full unique TRAIN candidate census and exact review IDs.")
    decisions, pairs, queue = [], [], []
    for case in cases:
        prep.validate_case(case)
        candidate = indexed[case["id"]]
        require(candidate["split"] == "train" and candidate["format"] == "json"
                and candidate["input_sha256"] == conditions_hash(case)
                and candidate["candidate_id"] == case["id"] + ":greedy:0",
                "Candidate is not the saved original TRAIN state.")
        observation = {**candidate, "status": candidate.get("status", "completed")}
        decision = semantic.classify(case, observation, bound_evaluation_bundle(case))
        if case["id"] in reviewed:
            decision = semantic.reviewed_decision(case, observation, decision, reviewed[case["id"]], candidate_sha)
        value = {"case_id": case["id"], "input_sha256": candidate["input_sha256"],
                 "candidate_id": candidate["candidate_id"], "candidate_run_sha256": candidate_sha,
                 "language": case["language"], "history_length": case.get("history_length", {}).get("length", 3),
                 **decision}
        decisions.append(value)
        if decision["status"] == "needs_review":
            queue.append({**value, "input": case["input"], "accepted_predictions": semantic.references(case),
                          "reference_rationale": case.get("rationale"), "observation": candidate})
        if decision["status"] == "preferred":
            pairs.append({key: value[key] for key in ("case_id", "input_sha256", "candidate_id", "candidate_run_sha256")}
                         | {"pair_id": case["id"] + ":actual-sft-error", "chosen": decision["chosen"],
                            "rejected_text": candidate["text"],
                            "judgment": {"status": "preferred", "actor": "ai_generated", "rationale": decision["reason"]},
                            "decision_actor": decision.get("actor", "contract_or_admitted_reference"),
                            "saved_messages_sha256": prep.value_hash(case["messages"]),
                            "original_case_sha256": prep.value_hash(case)})
    return decisions, pairs, queue


def conditions_hash(case):
    return prep.value_hash(prep.formats.contract.wire_snapshot(case["input"]))


def mine(config_path, training, reloaded, candidates, output, *, judgments=None):
    config, bundle = final.verify_preparation(config_path)
    parent = final.verify_trained(config_path, training, reloaded)["identity"]
    candidates, output = Path(candidates), Path(output).resolve()
    require(not output.exists(), "Use a fresh immutable mining directory.")
    _, run = dpo.sft.verified_run(candidates, "candidates_complete", version=dpo.VERSION)
    expected = dpo.candidate_identity(config_path, training, reloaded)
    require(run["train_only"] is True and run["final_test_bodies_read"] is False
            and all(run[key] == value for key, value in expected.items()),
            "Mine only the complete independently reloaded new SFT's TRAIN candidate run.")
    observations = prep.rows(candidates / "candidates.jsonl")
    require(run["candidates"] == len(observations) == 1261, "Final TRAIN mining must retain all1261 candidates.")
    candidate_sha = sha(candidates / "manifest.json")
    decisions, pairs, queue = decisions_for(bundle["cases"], observations, candidate_sha,
                                           prep.rows(judgments) if judgments else [])
    output.mkdir(parents=True)
    for name, values in (("decisions.jsonl", decisions), ("pairs.jsonl", pairs), ("review-queue.jsonl", queue)):
        prep.write_rows(output / name, values)
    report = {"version": VERSION, "status": "train_errors_mined", "parent": parent,
              "configuration": final.binding(config_path), "candidates": final.binding(candidates / "manifest.json"),
              "candidate_run_sha256": candidate_sha, "requested": 1261, "pairs": len(pairs),
              "preservation_pairs": sum(row["chosen"]["action"] == "no_action" for row in pairs),
              "decisions": dict(Counter(row["status"] for row in decisions)),
              "judgments": final.binding(judgments) if judgments else None,
              "saved_projections_preserved": True, "final_test_bodies_read": False,
              "encoding_policy": "Preserve exact generated rejected text; both DPO completions use the existing Gemma26 EOS1 training convention. This is not native generated-token replay.",
              "files_sha256": {path.name: sha(path) for path in output.iterdir() if path.is_file()}}
    write(output / "run.json", report)
    return report


def verify_mining(mining, parent):
    mining = Path(mining)
    report = read(mining / "run.json")
    require(report["version"] == VERSION and report["status"] == "train_errors_mined"
            and report["parent"] == parent and report["requested"] == 1261
            and report["saved_projections_preserved"] is True and report["final_test_bodies_read"] is False,
            "Mining belongs to another parent, cohort or projection.")
    require(final.binding(final.bound(report["configuration"]))["sha256"] == parent["configuration_sha256"]
            and report["candidates"]["sha256"] == report["candidate_run_sha256"],
            "Mining configuration or candidate manifest differs from the actual parent.")
    candidates = final.bound(report["candidates"]).parent
    _, observed = dpo.sft.verified_run(candidates, "candidates_complete", version=dpo.VERSION)
    require(observed["train_only"] is True and observed["final_test_bodies_read"] is False
            and observed["candidates"] == 1261 and all(observed["anchor_" + key] == parent[key] for key in
                ("training_manifest_sha256", "reload_manifest_sha256", "configuration_sha256", "adapter_sha256")),
            "Retained candidates do not belong to this complete TRAIN-only SFT run.")
    require(set(report["files_sha256"]) == {"decisions.jsonl", "pairs.jsonl", "review-queue.jsonl"},
            "Incomplete final Gemma mining inventory.")
    for name, expected in report["files_sha256"].items():
        require(sha(mining / name) == expected, "Mining evidence changed.")
    decisions, pairs, queue = (prep.rows(mining / name) for name in ("decisions.jsonl", "pairs.jsonl", "review-queue.jsonl"))
    candidates_by_id = {row["case_id"]: row for row in prep.rows(candidates / "candidates.jsonl")}
    require(len(decisions) == len({row["case_id"] for row in decisions}) == 1261
            and set(candidates_by_id) == {row["case_id"] for row in decisions}
            and report["decisions"] == dict(Counter(row["status"] for row in decisions))
            and len(pairs) == report["pairs"] == report["decisions"].get("preferred", 0)
            and report["preservation_pairs"] == sum(row["chosen"]["action"] == "no_action" for row in pairs)
            and {row["case_id"] for row in pairs} == {row["case_id"] for row in decisions if row["status"] == "preferred"}
            and not queue and report["decisions"].get("needs_review", 0) == 0,
            "DPO decision requires every actual candidate and no unresolved semantic review.")
    for row in [*decisions, *pairs]:
        candidate = candidates_by_id[row["case_id"]]
        require(row["candidate_id"] == candidate["candidate_id"]
                and row["input_sha256"] == candidate["input_sha256"]
                and row["candidate_run_sha256"] == report["candidate_run_sha256"]
                and ("rejected_text" not in row or row["rejected_text"] == candidate["text"]),
                "A mined preference or decision changed the actual candidate text or input.")
    return report


def assess(config_path, training, reloaded, mining, output, *, plan=final.PLAN):
    parent = final.verify_trained(config_path, training, reloaded)["identity"]
    report = verify_mining(mining, parent)
    reasons = (["fewer_than_eight_strict_pairs"] if report["pairs"] < 8 else [])
    reasons += ["no_actual_preservation_error_pair"] if report["preservation_pairs"] == 0 else []
    result = {"version": "gemma26-final-dpo-decision-1", "status": "registered_before_test",
              "parent": parent, "mining": final.binding(Path(mining) / "run.json"), "plan": final.binding(plan),
              "decision": "eligible_omission" if reasons else "same_parent_DPO_required", "reason_codes": reasons,
              "counts": {key: report[key] for key in ("requested", "pairs", "preservation_pairs")},
              "body_access_authorized": False, "final_test_bodies_read": False,
              "final_registration_performed": False}
    write(output, result)
    return result


def prepare(config_path, training, reloaded, mining, output, *, plan=final.PLAN):
    parent = final.verify_trained(config_path, training, reloaded)["identity"]
    report = verify_mining(mining, parent)
    require(report["pairs"] >= 8 and report["preservation_pairs"] >= 1,
            "Use an evidence-bound omission if actual strict pairs do not meet the prospective minimum.")
    protocol = Path(output).resolve().with_suffix(".protocol.json")
    require(not protocol.exists(), "Freeze the DPO protocol once beside a fresh output directory.")
    write(protocol, {"version": VERSION, "status": "registered_before_training", "parent": parent,
                     "mining": final.binding(Path(mining) / "run.json"), "plan": final.binding(plan),
                     "epochs": 1, "beta": 0.1, "learning_rate": 1e-6, "gradient_accumulation": 4,
                     "saved_projection": True, "final_test_bodies_read": False})
    return dpo.prepare_pairs(config_path, training, reloaded, final.bound(report["candidates"]).parent,
                             Path(mining) / "pairs.jsonl", output, phase="substantive", beta=0.1,
                             learning_rate=1e-6, accumulation=4, protocol=protocol, saved_projection=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("mine", "prepare", "assess"):
        cmd = sub.add_parser(name)
        for field in ("config-path", "training", "reloaded", "output", "candidates" if name == "mine" else "mining"):
            cmd.add_argument("--" + field, type=Path, required=True)
        if name == "mine":
            cmd.add_argument("--judgments", type=Path)
    args = vars(parser.parse_args())
    print(json.dumps(globals()[args.pop("command")](**args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
