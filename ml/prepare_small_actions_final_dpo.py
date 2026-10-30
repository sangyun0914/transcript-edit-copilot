"""Mine actual selected-SFT TRAIN errors and freeze native, same-input DPO pairs.

Mining never infers a semantic preference merely from a reference mismatch.
The optional review file can resolve that queue using already accepted references.
"""

import argparse
from collections import Counter
import math
from pathlib import Path
import random
import shutil

from grading import transcript_media
from prepare_final_unified_dataset import bound_evaluation_bundle
import small_actions_final as data
import train_small_actions_final as sft

VERSION = "small-actions-final-dpo-v1"
SIDES = ("chosen", "rejected")


def references(case):
    values = case["accepted_predictions"]
    data.require(values and values == case["acceptable"] and values[0] == case["primary"],
                 "Use the exact accepted references, primary first.")
    preserve = case["judgment"]["status"] == "preserve"
    data.require(all((value["action"] == "no_action") == preserve for value in values),
                 "References disagree on whether this state needs an edit.")
    return values


def classify(case, observation, bundle):
    accepted = references(case)
    outcomes = [transcript_media(data.formats.contract.apply_passive_edit(case["document"], bundle, value))
                for value in accepted]
    if observation.get("status") != "completed":
        return {"status": "excluded", "reason": "request_failure"}
    if observation.get("finish_reason") != "stop" or not observation.get("text", "").strip():
        return {"status": "excluded", "reason": "incomplete_or_empty_candidate"}
    try:
        action = data.formats.parse(observation["text"], case["input"], "json", finish_reason="stop").model_dump()
    except (ValueError, TypeError) as error:
        return {"status": "preferred", "category": "contract_violation", "chosen": accepted[0],
                "reason": "Complete sampled output violates the executable contract: " + str(error)}
    try:
        outcome = transcript_media(data.formats.contract.apply_passive_edit(case["document"], bundle, action))
    except (ValueError, KeyError) as error:
        return {"status": "preferred", "category": "execution_violation", "chosen": accepted[0],
                "reason": "Sampled action cannot execute against the saved source mapping: " + str(error)}
    if outcome in outcomes:
        return {"status": "tie", "reason": "Candidate executes to an accepted transcript, grouping and media result."}
    if action["action"] == "no_action" and case["judgment"]["status"] == "edit_required":
        return {"status": "preferred", "category": "missed_required_edit", "chosen": accepted[0],
                "reason": "Candidate abstains where the independently admitted reference requires an edit. "
                          + case.get("rationale", "")}
    journal = case["runtime"].get("feedback_journal")
    if case["judgment"]["status"] == "preserve" and journal:
        ledger = data.formats.contract.PassiveFeedbackLedger.restore(journal)
        if ledger.suppressed(case["document"], bundle, action, case["runtime"]["sequence"]):
            return {"status": "preferred", "category": "repeats_dismissed_or_undone", "chosen": accepted[0],
                    "reason": "The candidate repeats an explicitly dismissed or undone proposal in unchanged relevant context."}
    return {"status": "needs_review", "parsed_action": action,
            "reason": "A valid non-reference result alone does not establish a strict preference."}


def reviewed_decision(case, observation, decision, review, candidate_sha):
    data.require(decision["status"] == "needs_review" and review["case_id"] == case["id"]
                 and review["candidate_run_sha256"] == candidate_sha
                 and review["input_sha256"] == observation["input_sha256"]
                 and review["status"] in ("preferred", "tie", "uncertain")
                 and review["actor"] in ("human", "ai_generated") and review["rationale"].strip(),
                 "Review must bind this unresolved actual candidate and state its evidence.")
    result = {"status": review["status"], "reason": review["rationale"], "actor": review["actor"]}
    if review["status"] == "preferred":
        data.require(review["chosen"] in references(case), "A review may only choose an original accepted reference.")
        result.update(chosen=review["chosen"], category="semantic_review")
    return result


def census(values):
    result = {name: dict(sorted(Counter(str(value[name]) for value in values).items()))
              for name in ("language", "reference_action", "chosen_action", "source_id", "source_group", "source_kind")}
    result["chosen_action_language"] = dict(sorted(Counter(f"{value['chosen_action']}:{value['language']}" for value in values).items()))
    return result


def mine(preparation, training, reloaded, cases_path, candidates, output, *, judgments=None, assets=None):
    config, prepared, _ = data.prepared(preparation, assets)
    identity = sft.model_identity(training, reloaded)
    data.require(identity["kind"] == "sft", "Mine only the selected frozen SFT, never a base or DPO model.")
    data.require(identity["configuration_sha256"] == data.sha(Path(preparation) / "configuration.json")
                 and data.sha(cases_path) == config["source_cases_sha256"],
                 "TRAIN cases or selected SFT configuration differ from the trained cohort.")
    candidates = Path(candidates)
    run = data.read(candidates / "run.json")
    candidate_sha = data.sha(candidates / "run.json")
    data.require(run["status"] == "inference_complete" and run["kind"] == "sft"
                 and run["splits"] == ["train"] and run["model_identity"] == identity
                 and run["configuration_sha256"] == identity["configuration_sha256"]
                 and run["observations_sha256"] == data.sha(candidates / "observations.jsonl"),
                 "Candidates must be an intact TRAIN-only inference run of this independently reloaded SFT.")
    cases, observations = data.rows(cases_path), data.rows(candidates / "observations.jsonl")
    indexed = {row["case_id"]: row for row in observations}
    prepared_by_id = {row["case_id"]: row for row in prepared}
    data.require(cases and len(cases) == len(indexed) == len(observations) == run["requested"]
                 and {case["id"] for case in cases} == set(indexed) == set(prepared_by_id),
                 "Mining must account for every unique original TRAIN row exactly once, including failures.")
    reviews = data.rows(judgments) if judgments else []
    by_review = {row["case_id"]: row for row in reviews}
    data.require(len(reviews) == len(by_review) and set(by_review) <= set(indexed), "Duplicate or foreign reviews.")
    decisions, pairs, queue = [], [], []
    for case in cases:
        data.validate_case(case)
        references(case)
        observation, trained = indexed[case["id"]], prepared_by_id[case["id"]]
        data.require(observation["format"] == "json"
                     and observation["input_sha256"] == trained["semantic_input_sha256"]
                     == data.value_hash(data.formats.contract.wire_snapshot(case["input"]))
                     and observation["messages_sha256"] == data.value_hash(case["messages"])
                     and trained["messages"] == case["messages"]
                     and observation["prefix_ids_sha256"] == data.value_hash(trained["input_ids"][:trained["completion_start"]]),
                     "Candidate, original canonical input, and trained token prefix differ.")
        bundle = bound_evaluation_bundle(case)
        decision = classify(case, observation, bundle)
        if case["id"] in by_review:
            decision = reviewed_decision(case, observation, decision, by_review[case["id"]], candidate_sha)
        metadata = {"case_id": case["id"], "input_sha256": observation["input_sha256"],
                    "language": case["language"], "reference_action": case["primary"]["action"],
                    "chosen_action": decision.get("chosen", {}).get("action", "none"),
                    "source_id": case["source"]["id"], "source_group": case["source"]["source_group"],
                    "source_kind": case["source"].get("provenance", case["source"].get("kind", "unspecified"))}
        decisions.append({**metadata, **decision})
        if decision["status"] == "needs_review":
            queue.append({**metadata, "candidate_run_sha256": candidate_sha, "input": case["input"],
                          "accepted_predictions": references(case), "reference_rationale": case.get("rationale"),
                          "observation": observation, "reason": decision["reason"]})
        if decision["status"] == "preferred":
            data.require(decision["chosen"] in references(case), "Preferred action left the frozen accepted set.")
            pairs.append({**metadata, "pair_id": case["id"] + ":actual-sft-error", "split": "train",
                          "input": case["input"], "messages": case["messages"], "source": case["source"],
                          "accepted_predictions": references(case), "original_case_sha256": data.value_hash(case),
                          "chosen": decision["chosen"], "rejected": observation,
                          "judgment": {key: decision[key] for key in ("status", "category", "reason")},
                          "judgment_actor": decision.get("actor", "contract_or_admitted_reference")})
    output = data.new_directory(output)
    data.write_rows(output / "pairs.jsonl.gz", pairs)
    data.write_rows(output / "decisions.jsonl", decisions)
    data.write_rows(output / "review-queue.jsonl", queue)
    shutil.copyfile(Path(preparation) / "configuration.json", output / "configuration.json")
    result = {"version": VERSION, "status": "train_errors_mined", "parent": identity,
              "candidate_run_sha256": candidate_sha, "source_cases_sha256": config["source_cases_sha256"],
              "judgments_sha256": data.sha(judgments) if judgments else None,
              "requested": len(cases), "pairs": len(pairs), "preservation_pairs": sum(p["chosen_action"] == "no_action" for p in pairs),
              "decisions": dict(sorted(Counter(row["status"] for row in decisions).items())),
              "categories": dict(sorted(Counter(pair["judgment"]["category"] for pair in pairs).items())),
              "all_candidates_counts": census(decisions), "pair_counts": census(pairs),
              "unique_inputs": len({p["input_sha256"] for p in pairs}), "final_test_bodies_read": False,
              "files_sha256": {name: data.sha(output / name) for name in
                               ("pairs.jsonl.gz", "decisions.jsonl", "review-queue.jsonl", "configuration.json")}}
    data.write(output / "run.json", result)
    return result


def encoded_pair(pair, tokenizer, config):
    data.require(pair["split"] == pair["source"]["split"] == "train"
                 and pair["judgment"]["status"] == "preferred" and pair["judgment"]["reason"].strip()
                 and pair["messages"] == data.formats.messages(pair["input"])
                 and pair["chosen"] in pair["accepted_predictions"], "Invalid frozen strict TRAIN preference.")
    chosen = data.encode_completion(tokenizer, pair["messages"],
                                   data.formats.serialize(pair["input"], pair["chosen"], "json"),
                                   config["generation"]["eos_token_id"])
    prefix = chosen["input_ids"][:chosen["completion_start"]]
    rejected = pair["rejected"]
    data.require(rejected["prefix_ids_sha256"] == data.value_hash(prefix)
                 and rejected["status"] == "completed" and rejected["finish_reason"] == "stop",
                 "Rejected completion must retain its original native prefix and terminal.")
    tokens = rejected["output_ids"]
    data.require(tokens and tokenizer.decode(tokens, skip_special_tokens=True) == rejected["text"]
                 and tokenizer.decode(tokens, skip_special_tokens=False) == rejected["text_raw"],
                 "Actual rejected token IDs no longer decode to the frozen observation.")
    encoded_rejected = {"input_ids": prefix + tokens, "attention_mask": [1] * (len(prefix) + len(tokens)),
                        "labels": [-100] * len(prefix) + tokens, "completion_start": len(prefix),
                        "completion_text": rejected["text"], "terminal_token_id": tokens[-1]}
    for row in (chosen, encoded_rejected):
        data.validate_encoded(row, config["generation"]["eos_token_id"], config["context_tokens"], config["max_completion_tokens"])
    data.require(chosen["input_ids"] != encoded_rejected["input_ids"], "Identical completions are ties.")
    return {**{key: pair[key] for key in ("pair_id", "case_id", "language", "reference_action", "chosen_action",
                                        "source_id", "source_group", "source_kind", "input_sha256")},
            "raw_pair_sha256": data.value_hash(pair), "messages": pair["messages"],
            "chosen": chosen, "rejected": encoded_rejected,
            "parent_output_ids": tokens, "parent_text": rejected["text"]}


def prepare(mining, plan, output, *, assets=None, epochs=1, learning_rate=1e-6, beta=0.1,
            accumulation=4, seed=42, minimum_pairs=8, minimum_preservation_pairs=1):
    mining = Path(mining)
    report, config = data.read(mining / "run.json"), data.read(mining / "configuration.json")
    data.require(report["version"] == VERSION and report["status"] == "train_errors_mined", "Incomplete preference mining.")
    for name, expected in report["files_sha256"].items():
        data.require(data.sha(mining / name) == expected, "Mined preference payload changed: " + name)
    data.require(data.sha(mining / "configuration.json") == report["parent"]["configuration_sha256"]
                 and config["base"] == data.PROFILES[config["model"]], "Parent model/configuration changed.")
    plan_text = Path(plan).read_text()
    data.require(plan_text.strip(), "Bind a written DPO budget rationale and DEV adoption rules before training.")
    data.require(type(epochs) is int and 1 <= epochs <= 3 and type(accumulation) is int and accumulation > 0
                 and math.isfinite(learning_rate) and 0 < learning_rate <= 5e-5
                 and math.isfinite(beta) and 0 < beta <= 1
                 and minimum_pairs >= 1 and minimum_preservation_pairs >= 1,
                 "Use explicit bounded DPO settings and require actual preservation errors.")
    pairs = data.rows(mining / "pairs.jsonl.gz")
    data.require(len(pairs) >= minimum_pairs and len(pairs) == report["pairs"]
                 and sum(p["chosen_action"] == "no_action" for p in pairs) >= minimum_preservation_pairs,
                 "Too few strict or preservation pairs; do not invent rejected completions to fill a quota.")
    assets = Path(assets or config["assets_path"])
    data.require(data.verify_assets(assets, config["model"]) == config["assets"], "Parent tokenizer changed.")
    tokenizer = data.tokenizer_from(assets, config["model"])
    encoded = [encoded_pair(pair, tokenizer, config) for pair in pairs]
    data.require(len({pair["pair_id"] for pair in encoded}) == len(encoded), "Duplicate preference pair.")
    order, randomizer = [], random.Random(seed)
    for _ in range(epochs):
        indices = list(range(len(encoded)))
        randomizer.shuffle(indices)
        order.extend(indices)
    output = data.new_directory(output)
    shutil.copyfile(mining / "configuration.json", output / "configuration.json")
    data.write_rows(output / "pairs.jsonl.gz", encoded)
    (output / "plan.md").write_text(plan_text)
    frozen = {"version": VERSION, "status": "frozen_before_training", "parent": report["parent"],
              "mining_run_sha256": data.sha(mining / "run.json"), "pairs_sha256": data.sha(output / "pairs.jsonl.gz"),
              "plan_sha256": data.sha(output / "plan.md"), "pair_counts": report["pair_counts"],
              "pairs": len(encoded), "preservation_pairs": report["preservation_pairs"],
              "minimum_pairs": minimum_pairs, "minimum_preservation_pairs": minimum_preservation_pairs,
              "objective": "standard_dpo_sigmoid_completion_sum_including_native_terminal",
              "reference": "exact_frozen_sft_parent_cached_before_updates",
              "training": {"epochs": epochs, "sample_order": order, "learning_rate": learning_rate,
                           "beta": beta, "gradient_accumulation": accumulation, "seed": seed,
                           "micro_batch_size": 1, "weight_decay": 0.01, "max_grad_norm": 1.0,
                           "updates": math.ceil(len(order) / accumulation), "dropout": 0.0,
                           "truncation": False, "resampling": False, "sft_loss_weight": 0.0},
              "final_test_bodies_read": False}
    data.write(output / "dpo_configuration.json", frozen)
    return frozen


def prepared(path, assets_override=None):
    path = Path(path)
    config, dpo = data.read(path / "configuration.json"), data.read(path / "dpo_configuration.json")
    data.require(dpo["version"] == VERSION and dpo["status"] == "frozen_before_training"
                 and data.sha(path / "configuration.json") == dpo["parent"]["configuration_sha256"]
                 and data.sha(path / "pairs.jsonl.gz") == dpo["pairs_sha256"]
                 and data.sha(path / "plan.md") == dpo["plan_sha256"], "Frozen DPO preparation changed.")
    data.require(config["base"] == data.PROFILES[config["model"]]
                 and config["contract_version"] == data.formats.contract.VERSION,
                 "Pinned model or action contract changed.")
    assets = Path(assets_override or config["assets_path"])
    data.require(data.verify_assets(assets, config["model"]) == config["assets"], "Tokenizer identity changed.")
    pairs = data.rows(path / "pairs.jsonl.gz")
    data.require(len(pairs) == dpo["pairs"] and len({p["pair_id"] for p in pairs}) == len(pairs), "Pair census changed.")
    data.require(Counter(dpo["training"]["sample_order"]) == Counter({i: dpo["training"]["epochs"] for i in range(len(pairs))}),
                 "DPO must cover every admitted pair once per declared epoch.")
    for pair in pairs:
        for side in SIDES:
            data.validate_encoded(pair[side], config["generation"]["eos_token_id"], config["context_tokens"], config["max_completion_tokens"])
        left, right = pair["chosen"], pair["rejected"]
        data.require(left["input_ids"][:left["completion_start"]] == right["input_ids"][:right["completion_start"]],
                     "Chosen and rejected prompts differ.")
    return config, dpo, pairs, assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mining = sub.add_parser("mine")
    for name in ("preparation", "training", "reloaded", "cases", "candidates", "output"):
        mining.add_argument("--" + name, type=Path, required=True)
    mining.add_argument("--judgments", type=Path)
    mining.add_argument("--assets", type=Path)
    prep = sub.add_parser("prepare")
    for name in ("mining", "plan", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--assets", type=Path)
    for name, default in (("epochs", 1), ("accumulation", 4), ("seed", 42), ("minimum-pairs", 8), ("minimum-preservation-pairs", 1)):
        prep.add_argument("--" + name, type=int, default=default)
    prep.add_argument("--learning-rate", type=float, default=1e-6)
    prep.add_argument("--beta", type=float, default=0.1)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "mine":
        args["cases_path"] = args.pop("cases")
    print(data.compact((mine if command == "mine" else prepare)(**args)))


if __name__ == "__main__":
    main()
