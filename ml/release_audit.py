"""Audit source dependence, lexical-masked structure, context limits and token weights."""

from collections import Counter, defaultdict

from replay_session import session_hash
from session_cases import document_records


def dependence_blocks(cases):
    parents = list(range(len(cases)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, left in enumerate(cases):
        lm = left["metadata"]
        for j in range(i):
            rm = cases[j]["metadata"]
            if lm["source_group_id"] != rm["source_group_id"]:
                continue
            same_episode = lm["parent_episode_id"] == rm["parent_episode_id"]
            a, b = lm["source_context_seconds"], rm["source_context_seconds"]
            overlap = lm.get("source_identity_sha256", lm["video_sha256"]) == rm.get(
                "source_identity_sha256", rm["video_sha256"]
            ) and max(a[0], b[0]) < min(a[1], b[1])
            if same_episode or overlap:
                parents[root(i)] = root(j)
    return {case["id"]: f"block-{root(index):03d}" for index, case in enumerate(cases)}


def masked_structure(case):
    value = case["input"]
    para, speakers, runs = {}, {}, {}

    def alias(table, key):
        return table.setdefault(key, len(table))

    indices = {i["id"]: n for n, i in enumerate(value["items"])}
    return {
        "items": [
            (
                i["type"],
                alias(para, i["paragraph"]),
                alias(speakers, i["speaker"]),
                alias(runs, i["source_run"]),
                i["audio_class"],
                i["boundary_status"],
            )
            for i in value["items"]
        ],
        "cursor": indices[value["cursor_before"]],
        "target_positions": [
            indices[t] for t in case["expectedOutput"].get("target", [])
        ],
        "action": case["expectedOutput"]["action"],
        "history_actions": [h["action"] for h in value["recent_edits"]],
        "feedback": [
            (h["action"], h["outcome"], h["same_context"], h["reason"])
            for h in value["recent_feedback"]
        ],
    }


def audit_release(cases, contexts, executions):
    cases = sorted(cases, key=lambda c: c["id"])
    blocks = dependence_blocks(cases)
    contexts = {r["case_id"]: r for r in contexts}
    executions = {r["id"]: r for r in executions}
    structures, mechanisms = defaultdict(list), defaultdict(list)
    for case in cases:
        structures[session_hash(masked_structure(case))].append(case["id"])
        meta = case["metadata"]
        mechanisms[
            (meta["scenario_family"], case["expectedOutput"]["action"], meta["stratum"])
        ].append(case["id"])
        settings = executions[case["id"]]["reconstruction"]
        if any(
            e["sequence"] > settings["as_of"]
            for e in settings.get("feedback", {}).get("events", [])
        ):
            raise ValueError(
                "An episode reconstruction contains a future feedback event."
            )
    shares, weights = [], []
    for split in ("train", "dev"):
        selected = [c for c in cases if c["metadata"]["split"] == split]
        groups = sorted({c["metadata"]["source_group_id"] for c in selected})
        total_tokens = sum(
            contexts[c["id"]]["completion_token_count"] for c in selected
        )
        for group in groups:
            group_cases = [
                c for c in selected if c["metadata"]["source_group_id"] == group
            ]
            group_blocks = sorted({blocks[c["id"]] for c in group_cases})
            tokens = sum(
                contexts[c["id"]]["completion_token_count"] for c in group_cases
            )
            shares.append(
                {
                    "split": split,
                    "source_group_id": group,
                    "rows": len(group_cases),
                    "loss_tokens": tokens,
                    "unweighted_loss_token_share": tokens / total_tokens,
                    "dependence_blocks": len(group_blocks),
                    "proposed_source_mass": 1 / len(groups),
                }
            )
            for block in group_blocks:
                block_cases = [c for c in group_cases if blocks[c["id"]] == block]
                for case in block_cases:
                    mass = 1 / len(groups) / len(group_blocks) / len(block_cases)
                    tokens = contexts[case["id"]]["completion_token_count"]
                    weights.append(
                        {
                            "case_id": case["id"],
                            "split": split,
                            "block": block,
                            "sampling_probability": mass,
                            "completion_tokens": tokens,
                            "per_supervised_token_weight": mass / tokens,
                        }
                    )
    limits = []
    for case in cases:
        document = executions[case["id"]]["before"]
        records, mapping = document_records(document)
        visible = Counter(i["paragraph"] for i in case["input"]["items"])
        paragraphs = []
        for paragraph in visible:
            whole = [i for i in records if i["paragraph"] == paragraph]
            full = len(whole) == visible[paragraph]
            if full != (paragraph in case["input"]["complete_paragraphs"]):
                raise ValueError("A cropped paragraph was labeled complete.")
            paragraphs.append(
                {
                    "paragraph": paragraph,
                    "source_items": len(whole),
                    "visible_items": visible[paragraph],
                    "complete": full,
                    "source_duration_ms": round(
                        sum(
                            document["content"][mapping[str(i["id"])]["index"]][
                                "length"
                            ]
                            for i in whole
                        )
                        * 1000
                    ),
                    "source_text_characters": len(
                        " ".join(i["text"] for i in whole if i["text"])
                    ),
                }
            )
        limits.append(
            {
                "case_id": case["id"],
                "paragraphs": paragraphs,
                "cropped_paragraphs": sum(not p["complete"] for p in paragraphs),
                "caption_policy": "v4 uses trusted full-document facts and visible local evidence; cropped paragraphs remain incomplete"
                if case["input"]["schema_version"] == 4
                else "v3 requires complete affected paragraphs",
            }
        )
    return {
        "status": "deterministic_dependence_audit_not_independent_semantic_judgment",
        "lexical_mask": "All transcript words/names/numbers and replacement content removed; relative item/paragraph/source/target structure and feedback retained.",
        "structurally_identical_clusters": [
            ids for ids in structures.values() if len(ids) > 1
        ],
        "shared_decision_mechanisms": [
            {"family": key[0], "action": key[1], "stratum": key[2], "case_ids": ids}
            for key, ids in mechanisms.items()
            if len(ids) > 1
        ],
        "interpretation": "Shared structures/mechanisms require inspection and do not prove semantic independence or duplication. Correlated source/episode states stay in one split.",
        "source_shares": shares,
        "weights_status": "frozen_for_next_training_not_yet_applied"
        if all(
            c["metadata"].get("release_status") == "frozen_supported_scope"
            for c in cases
        )
        else "proposed_for_release_review_not_applied_to_training",
        "weight_policy": "Equal source mass within each split, then equal overlapping-source-or-episode block mass, then equal case mass; divide by actual completion tokens for token-normalized loss.",
        "case_weights": weights,
        "context_eligibility": limits,
    }
