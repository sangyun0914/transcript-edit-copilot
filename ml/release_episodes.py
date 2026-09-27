"""Execute source-backed synthetic interactions before recording each next-edit state."""

import copy

from context_policy import FeedbackLedger
from dataset_release_inventory import write
from iteration import jsonl
from replay_session import replay, session_hash
from session_cases import apply_edit, replayable_session, transaction
from source_qualification import focus_at
from release_media import verify_media_edit


def committed_history(bundle, action):
    selected = [i for i in bundle["input"]["items"] if i["id"] in action["target"]]
    result = {"action": action["action"], "origin": "synthetic"}
    if action["action"] in {"correct_text", "delete_span"}:
        result.update(
            before=" ".join(i["text"] for i in selected if i["text"]),
            after=action.get("text", ""),
        )
    elif action["action"] in {"trim_silence", "remove_silence"}:
        result.update(
            before_ms=selected[0]["duration_ms"], after_ms=action.get("keep_ms", 0)
        )
    return result


def build_episodes(
    recipe, sources, windows, builder, output, prompt, make_case, action_for
):
    from dataset_release import restore_bundle, source_evidence, timed

    cases, contexts, executions, summaries = [], [], [], []
    (output / "episodes").mkdir()
    for episode in recipe.get("episodes", []):
        source = sources[episode["source_id"]]
        document = copy.deepcopy(source["document"])
        ledger, history = FeedbackLedger(episode["id"]), []
        journal = replayable_session(document, document, "no_action", {})[:1]
        journal[0].update(
            generator=recipe["version"],
            sources={
                source["id"]: {
                    "sha256": source.get(
                        "source_identity_sha256", source["video_sha256"]
                    ),
                    "media_kind": source.get("media_kind", "recording"),
                }
            },
        )
        sequence, revision = 0, 0
        step_records = []
        brief = episode["brief"]
        for index, step in enumerate(episode["steps"]):
            brief = step.get("brief", brief)
            sequence += 10
            settings = {
                "as_of": sequence,
                "history": copy.deepcopy(history),
                "feedback": {
                    "document_id": ledger.document_id,
                    "proposals": copy.deepcopy(ledger.proposals),
                    "events": copy.deepcopy(ledger.events),
                },
            }
            focus = focus_at(timed(document), step["focus_seconds"])
            if step.get("record_case", True):
                spec = {
                    "source_id": source["id"],
                    "window": step.get("window", episode["window"]),
                    "brief": brief,
                    "parent_episode_id": episode["id"],
                    "episode_protocol": episode["protocol"],
                    "episode_step": index,
                    "stratum": episode["stratum"],
                    "preservation": episode["preservation"],
                    "recipe_version": recipe["version"],
                    "label_status": "known_preserve"
                    if step["action"] == "no_action"
                    else "known_edit",
                    **step,
                }
                case, context, execution = make_case(
                    spec,
                    source,
                    windows,
                    builder,
                    output,
                    prompt,
                    {"document": document, "settings": settings},
                )
                cases.append(case)
                contexts.append(context)
                executions.append(execution)
                bundle = restore_bundle(
                    builder, document, focus["uuid"], brief, execution["reconstruction"]
                )
                candidate = case["expectedOutput"]
            else:
                settings["evidence"] = source_evidence(source, document)
                bundle = restore_bundle(
                    builder, document, focus["uuid"], brief, settings
                )
                candidate = action_for(step, document, bundle["mapping"])
            outcome = step.get("outcome")
            before_hash = session_hash(document)
            if candidate["action"] != "no_action" and ledger.suppressed(
                document, bundle, candidate, sequence
            ):
                raise ValueError("Episode reference repeats a suppressed edit.")
            if outcome:
                if candidate["action"] == "no_action":
                    raise ValueError("Abstention cannot receive proposal feedback.")
                ledger.propose(step["id"], sequence + 1, document, bundle, candidate)
                ledger.record(step["id"], "shown", sequence + 2)
                if outcome in {"accepted", "modified"}:
                    applied = (
                        action_for(step["applied"], document, bundle["mapping"])
                        if "applied" in step
                        else candidate
                    )
                    after = apply_edit(document, bundle, applied)
                    verify_media_edit(
                        document,
                        after,
                        bundle["mapping"],
                        applied,
                        source.get("audio_qualifications", []),
                        source.get("media_kind", "recording"),
                    )
                    revision += 1
                    txid = f"{episode['id']}/tx-{revision}"
                    journal.append(
                        transaction(
                            document,
                            after,
                            len(journal),
                            revision,
                            applied["action"],
                            transaction_id=txid,
                        )
                    )
                    ledger.record(
                        step["id"],
                        outcome,
                        sequence + 3,
                        applied=applied,
                        transaction_id=txid,
                        reason=step.get("reason", "unspecified"),
                    )
                    history.append(
                        {
                            "sequence": sequence + 3,
                            "edit": committed_history(bundle, applied),
                        }
                    )
                    if step.get("undo"):
                        revision += 1
                        journal.append(
                            transaction(
                                after,
                                document,
                                len(journal),
                                revision,
                                applied["action"],
                                origin="undo",
                                transaction_id=txid,
                            )
                        )
                        ledger.record(
                            step["id"], "undone", sequence + 4, transaction_id=txid
                        )
                        history.pop()
                    else:
                        document = after
                else:
                    ledger.record(
                        step["id"],
                        outcome,
                        sequence + 3,
                        reason=step.get("reason", "unspecified"),
                    )
            step_records.append(
                {
                    "id": step["id"],
                    "recorded_case": step.get("record_case", True),
                    "sequence": sequence,
                    "before_hash": before_hash,
                    "after_hash": session_hash(document),
                    "brief": brief,
                    "focus_uuid": focus["uuid"],
                    "candidate": candidate,
                    "outcome": outcome,
                    "undo": step.get("undo", False),
                }
            )
        journal.append(
            {
                "kind": "end",
                "seq": len(journal),
                "revision": revision,
                "document_hash": session_hash(document),
                "transactions": revision,
            }
        )
        actual, _ = replay(journal)
        if actual != document:
            raise ValueError("Episode final replay differs from its actual state.")
        path = f"episodes/{episode['id']}"
        jsonl(output / f"{path}.session.jsonl", journal)
        write(
            output / f"{path}.feedback.json",
            {
                "document_id": ledger.document_id,
                "proposals": ledger.proposals,
                "events": ledger.events,
                "steps": step_records,
            },
        )
        summaries.append(
            {
                "id": episode["id"],
                "protocol": episode["protocol"],
                "status": episode["status"],
                "remaining": episode.get("remaining", []),
                "source_group_id": source["source_group_id"],
                "split": source["reserved_split"],
                "steps": len(step_records),
                "cases": sum(s["recorded_case"] for s in step_records),
                "transactions": revision,
                "actor": "synthetic",
                "human_preference_evidence": False,
                "session": f"{path}.session.jsonl",
                "feedback": f"{path}.feedback.json",
            }
        )
    return {
        "episodes": summaries,
        "cases": cases,
        "contexts": contexts,
        "executions": executions,
    }
