"""Register exposed TRAIN/Shuka UI attempts and summarize actual renderer traces.

No UI control, model calls, TEST data, or deployment mutation.
"""

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import wave
import zipfile

from caption_facts import CaptionProfile

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/final-ui-latency-measurement-2026-10-25.md"
ORIGIN_PLAN = ROOT / "docs/plans/final-ui-latency-origin-amendment-2026-10-25.md"
PRIMARY_METRIC = "trusted_committing_dom_event_to_validated_paint_proxy"
TRAIN = (
    ROOT
    / "artifacts/datasets/nep-passive-actions-final-v1/release-attempt-001/cases/train.jsonl"
)
SHUKA = ROOT / "artifacts/audits/shuka-live-20261025-001"
VERSION = "final-ui-latency-measurement-1"
SUMMARY_VERSION = "final-ui-latency-summary-2"
STATUSES = {
    "preview",
    "no_action",
    "suppressed",
    "invalid",
    "request_failure",
    "not_issued",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def binding(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": checksum}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def select_cases(path):
    chosen, counts, seen = {}, Counter(), set()
    profile = CaptionProfile().model_dump()
    with Path(path).open() as stream:
        for line in stream:
            case = json.loads(line)
            require(
                case["split"] == case["source"]["split"] == "train",
                "UI registration accepts TRAIN only.",
            )
            require(case["id"] not in seen, "Duplicate TRAIN case ID.")
            seen.add(case["id"])
            key = (case["language"], case["primary"]["action"])
            if key not in {
                (language, action)
                for language in ("en", "ko")
                for action in ("correct_text", "no_action")
            }:
                continue
            runtime, content = case["runtime"], case["document"]["content"]
            focused = next(
                (item for item in content if item["uuid"] == runtime["focus_uuid"]), {}
            )
            if (
                runtime["history"]
                or runtime.get("feedback_journal") is not None
                or runtime["selection"]
                or runtime["sequence"] != 0
                or runtime["caption_profile"] != profile
                or len(content) > 80
                or len(case["input"]["items"]) > 24
                or sum(len(item.get("text", "")) for item in content) > 1200
                or focused.get("type") != "text"
                or sum(item["type"] == "text" for item in content) < 2
            ):
                continue
            counts[key] += 1
            rank = hashlib.sha256(case["id"].encode()).hexdigest()
            if key not in chosen or rank < chosen[key][0]:
                chosen[key] = (rank, case)
    require(len(chosen) == 4, "Missing eligible EN/KO edit/preserve TRAIN cell.")
    return chosen, counts


def write_ui_archive(path, case):
    require(
        case["source"]["natural_audio"] is False
        and case["source"]["generated_audio"] is False,
        "Do not replace actual source audio with symbolic UI fixture media.",
    )
    sources = {}
    for item in case["document"]["content"]:
        if "source" in item:
            sources[item["source"]] = max(
                sources.get(item["source"], 0), item["sourceStart"] + item["length"]
            )
    document = {
        "version": 3,
        "content": case["document"]["content"],
        "metadata": {
            "display_video": False,
            "display_speaker_names": True,
            **case["document"]["metadata"],
        },
    }
    files = {
        "document.json": json.dumps(
            document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    }
    media = {}
    for source, end in sorted(sources.items()):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(bytes(2 * 16000 * math.ceil(end + 1)))
        files["sources/" + source] = buffer.getvalue()
        media[source] = {
            "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
            "seconds": math.ceil(end + 1),
        }
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 25, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return {
        "kind": "generated_silence_for_UI_only_not_original_or_training_audio",
        "source_ids_unchanged": True,
        "acoustic_quality_evidence": False,
        "sources": media,
    }


def prepare(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(ROOT) and not output.exists(),
        "Use a fresh workspace registration directory.",
    )
    selected, counts = select_cases(TRAIN)
    saved = read(SHUKA / "live-editor-before.json")
    original = {
        "content": saved["document"]["content"],
        "metadata": {
            key: value
            for key, value in saved["document"]["metadata"].items()
            if key in {"transcription", "transcriptions"}
        },
    }
    operations = {}
    previous = original
    for name in ("split", "delete", "trim"):
        path = SHUKA / f"manual-{name}.json"
        state = read(path)
        transition = state["copilot"]["history"][-1]
        require(
            transition["before"] == previous,
            "Saved Shuka edit chain differs from its original state.",
        )
        before, after = transition["before"]["content"], transition["after"]["content"]
        before_ids, after_ids = (
            {item["uuid"] for item in before},
            {item["uuid"] for item in after},
        )
        if name == "split":
            index = next(
                index
                for index, item in enumerate(after)
                if item["type"] == "paragraph_start" and item["uuid"] not in before_ids
            )
            targets = [after[index + 1]["uuid"]]
        elif name == "delete":
            targets = [item["uuid"] for item in before if item["uuid"] not in after_ids]
        else:
            old = {item["uuid"]: item for item in before}
            targets = [
                item["uuid"]
                for item in after
                if item["uuid"] in old and item != old[item["uuid"]]
            ]
        operations[name] = {
            "saved_state": binding(path),
            "target_uuids": targets,
            "prediction": transition["prediction"],
            "before_document_sha256": digest(transition["before"]),
            "reference_after_document_sha256": digest(transition["after"]),
            "focus_uuid": after[state["cursor"]["userIndex"]]["uuid"],
        }
        previous = transition["after"]
    output.mkdir(parents=True)
    seeds = {}
    for (language, action), (rank, case) in sorted(selected.items()):
        name = f"{language}-{'edit' if action == 'correct_text' else 'preserve'}"
        path = output / f"{name}.json"
        write(path, case)
        archive = output / f"{name}.audapolis"
        media = write_ui_archive(archive, case)
        seeds[name] = {
            "case_id": case["id"],
            "case": binding(path),
            "archive": binding(archive),
            "ui_media": media,
            "case_sha256": digest(case),
            "rank_sha256": rank,
            "eligible_pool": counts[(language, action)],
            "document_sha256": digest(case["document"]),
            "runtime_sha256": digest(case["runtime"]),
            "focus_uuid": case["runtime"]["focus_uuid"],
            "alternate_focus_uuid": next(
                item["uuid"]
                for item in case["document"]["content"]
                if item["type"] == "text"
                and item["uuid"] != case["runtime"]["focus_uuid"]
            ),
            "reference_action": action,
        }
    templates = [
        ("en-edit", "en", "en-edit", "focus", "suggestion", None),
        ("en-accept", "en", "en-edit", "accept", "continuation", "en-edit"),
        ("ko-edit", "ko", "ko-edit", "focus", "suggestion", None),
        ("ko-reject", "ko", "ko-edit", "dismiss", "feedback_only", "ko-edit"),
        ("en-preserve", "en", "en-preserve", "focus", "preserve", None),
        ("ko-preserve", "ko", "ko-preserve", "focus", "preserve", None),
        ("shuka-split", "ko", "shuka", "split", "suggestion", None),
        ("shuka-delete", "ko", "shuka", "delete", "suggestion", "shuka-split"),
        ("shuka-trim", "ko", "shuka", "trim", "suggestion", "shuka-delete"),
        ("en-burst-edit", "en", "en-edit", "append_period", "cancel_probe", None),
        (
            "en-burst-undo",
            "en",
            "en-edit",
            "undo_after_100ms",
            "suggestion",
            "en-burst-edit",
        ),
    ]
    attempts = [
        {
            "case_id": f"cold-{language}",
            "condition": "process_cold",
            "language": language,
            "seed": f"{language}-edit",
            "operation": "ordinary_npm_start_and_focus",
            "intent": "startup",
            "depends_on": None,
        }
        for language in ("en", "ko")
    ]
    for repetition in range(1, 4):
        attempts.extend(
            {
                "case_id": f"r{repetition}-{name}",
                "condition": "warm",
                "round": repetition,
                "language": language,
                "seed": seed,
                "operation": operation,
                "intent": intent,
                "depends_on": f"r{repetition}-{dependency}" if dependency else None,
            }
            for name, language, seed, operation, intent, dependency in templates
        )
    roster = {
        "version": VERSION,
        "status": "registered_before_selected_model_ui_observations",
        "scope": "exposed_development_ui",
        "final_test_bodies_read": False,
        "measurement_plan": binding(PLAN),
        "preparation_script": binding(__file__),
        "normal_app_setup": binding(ROOT / "ml/final_ui_setup.cjs"),
        "train": binding(TRAIN),
        "selection": "minimum SHA256(case_id) per eligible TRAIN language/reference-action cell; no model outcome filter",
        "selected_cases": seeds,
        "shuka": {
            "snapshot": binding(SHUKA / "live-editor-before.json"),
            "editor_history": binding(SHUKA / "live-editor-history-before.json"),
            "archive": binding(SHUKA / "shuka-user-state-before.audapolis"),
            "document_sha256": digest(original),
            "history_sha256": digest(saved["copilot"]["history"]),
            "history_count": len(saved["copilot"]["history"]),
            "document_items": len(original["content"]),
            "operations": operations,
        },
        "trace_sources": {
            str(path.relative_to(ROOT)): binding(path)["sha256"]
            for path in (
                ROOT / "app/src/state/copilot.ts",
                ROOT / "app/src/pages/Editor/Copilot.tsx",
            )
        },
        "repetitions": 3,
        "planned_case_ids": [row["case_id"] for row in attempts],
        "attempts": attempts,
    }
    write(output / "roster.json", roster)
    write(
        output / "observations-template.json",
        {
            "version": VERSION,
            "status": "draft_not_observed",
            "measurement_plan": binding(PLAN),
            "measurement_roster": binding(output / "roster.json"),
            "launch_method": "npm start",
            "automatic_only": True,
            "runtime_condition": "cold_and_warm_separate",
            "deployment": None,
            "model": None,
            "environment": None,
            "trace_sources": roster["trace_sources"],
            "setup_evidence": [],
            "planned_case_ids": roster["planned_case_ids"],
            "outcomes": [
                {
                    "case_id": row["case_id"],
                    "language": row["language"],
                    "status": None,
                    "detail": None,
                    "automatic_trigger": None,
                    "trace_id": None,
                    "measurements": [],
                    "feedback_events": [],
                    "preview_useful": None,
                    "usefulness_reason": None,
                    "startup": None,
                    "evidence": [],
                }
                for row in attempts
            ],
        },
    )
    return {
        "roster": binding(output / "roster.json"),
        "attempts": len(attempts),
        "warm": 33,
        "process_cold": 2,
        "selected_case_ids": {name: value["case_id"] for name, value in seeds.items()},
    }


def distribution(values):
    values = sorted(values)
    return {
        "n": len(values),
        "p50_ms": statistics.median(values) if values else None,
        "p95_ms": values[math.ceil(0.95 * len(values)) - 1] if values else None,
        "max_ms": values[-1] if values else None,
        **{
            f"under_{limit}ms": {
                "count": sum(value < limit for value in values),
                "denominator": len(values),
                "share": sum(value < limit for value in values) / len(values)
                if values
                else None,
            }
            for limit in (500, 1000)
        },
    }


def primary_timing(roster, plan, observed, paint, registered):
    def clock(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    def matches(event):
        operation = plan.get("operation")
        if operation == "focus" or plan["condition"] == "process_cold":
            focus = (
                roster.get("selected_cases", {}).get(plan["seed"], {}).get("focus_uuid")
            )
            return (
                focus is not None
                and event.get("type") == "mousedown"
                and event.get("target") == f"item-{focus}"
            )
        key = {
            "split": "Enter",
            "delete": "Backspace",
            "accept": "Tab",
            "dismiss": "Escape",
        }.get(operation)
        if key:
            return (
                event.get("type") == "keydown"
                and event.get("key") == key
                and event.get("in_document") is True
                and event.get("editable") is not True
            )
        if operation == "trim":
            return (
                event.get("type") == "keydown"
                and event.get("key") == "Enter"
                and event.get("aria_label") == "Gap duration in seconds"
            )
        if operation == "append_period":
            return (
                event.get("type") == "keydown"
                and event.get("key") == "Enter"
                and event.get("editable") is True
            )
        return (
            operation == "undo_after_100ms"
            and event.get("type") == "click"
            and event.get("button_icon") == "undo"
        )

    origin, reason, candidates = None, None, []
    window = observed.get("interaction_window")
    if not registered:
        reason = "origin_protocol_not_registered"
    elif (
        not isinstance(window, dict)
        or not clock(window.get("start_ms"))
        or not clock(window.get("end_ms"))
        or window["start_ms"] >= window["end_ms"]
    ):
        reason = "missing_or_invalid_interaction_window"
    else:
        for event in observed.get("input_events", []):
            if (
                not isinstance(event, dict)
                or event.get("trusted") is not True
                or not matches(event)
            ):
                continue
            if not clock(event.get("at_ms")):
                reason = "invalid_committing_dom_clock"
                break
            if window["start_ms"] <= event["at_ms"] < window["end_ms"]:
                candidates.append(event)
        if reason is None:
            if len(candidates) == 1:
                origin = candidates[0]
            else:
                reason = (
                    "ambiguous_committing_dom_event"
                    if candidates
                    else "missing_committing_dom_event"
                )
    trace_origin = paint["at_ms"] - paint["since_trigger_ms"] if paint else None
    if observed.get("action_error"):
        reason = "action_not_verified"
    elif reason is None and not paint:
        reason = "no_validated_painted_preview"
    elif reason is None and (
        paint["at_ms"] < origin["at_ms"] or trace_origin < origin["at_ms"] - 0.01
    ):
        reason = "paint_trace_precedes_committing_event"
    return {
        "origin_event": origin,
        "origin_candidate_count": len(candidates),
        "full_latency_ms": paint["at_ms"] - origin["at_ms"] if reason is None else None,
        "full_latency_unavailable_reason": reason,
        "trace_latency_ms": paint["since_trigger_ms"] if paint else None,
        "trace_origin_after_dom_input_ms": trace_origin - origin["at_ms"]
        if origin and paint
        else None,
    }


def summarize(roster, observations):
    planned, outcomes = roster["attempts"], observations["outcomes"]
    require(
        observations["planned_case_ids"]
        == roster["planned_case_ids"]
        == [row["case_id"] for row in outcomes],
        "Keep every planned attempt exactly once, in order.",
    )
    require(
        observations["status"] == "observed"
        and observations["launch_method"] == "npm start"
        and observations["automatic_only"] is True
        and observations["runtime_condition"] == "cold_and_warm_separate",
        "Actual product launch/automatic measurement regime differs.",
    )
    origin_registered = observations.get("primary_latency_metric") == PRIMARY_METRIC
    if origin_registered:
        require(
            observations.get("latency_origin_amendment") == binding(ORIGIN_PLAN),
            "Primary DOM latency requires the unchanged prospective origin amendment.",
        )
    rows = []
    for plan, observed in zip(planned, outcomes, strict=True):
        require(
            observed["language"] == plan["language"]
            and observed["status"] in STATUSES
            and type(observed["automatic_trigger"]) is bool
            and observed["evidence"],
            "Incomplete observed UI outcome.",
        )
        require(
            observed["preview_useful"] is None
            or type(observed["preview_useful"]) is bool,
            "Use explicit useful/incorrect/unknown preview assessment.",
        )
        require(
            observed["preview_useful"] is None
            or bool(observed.get("usefulness_reason")),
            "Observed usefulness needs a recorded reason.",
        )
        events = observed["measurements"]
        require(
            all(
                event["trace_id"] == observed["trace_id"]
                and event["trigger"] == "automatic"
                for event in events
            ),
            "Mixed or explicit traces cannot establish automatic interaction latency.",
        )
        require(
            all(
                type(event[field]) in (int, float)
                and math.isfinite(event[field])
                and event[field] >= 0
                for event in events
                for field in ("at_ms", "since_trigger_ms")
            ),
            "Invalid renderer trace clock.",
        )
        origins = [event["at_ms"] - event["since_trigger_ms"] for event in events]
        require(
            not origins or max(origins) - min(origins) < 0.01,
            "Trace events do not share the interaction clock.",
        )
        previews = [
            event for event in events if event["event"] == "visible_complete_preview"
        ]
        paint = None
        if observed["status"] == "preview":
            require(
                observed["automatic_trigger"] is True and len(previews) == 1,
                "A preview requires one actual painted-preview trace.",
            )
            paint = previews[0]
            requests = [
                event
                for event in events
                if event["event"] == "request"
                and event.get("request_id") == paint.get("request_id")
            ]
            responses = [
                event
                for event in events
                if event["event"] == "response"
                and event.get("request_id") == paint.get("request_id")
                and event.get("outcome") == "proposal"
            ]
            require(
                requests
                and responses
                and any(event["event"] == "trigger" for event in events)
                and requests[-1]["at_ms"] <= responses[-1]["at_ms"] <= paint["at_ms"],
                "Paint lacks its automatic request and validated proposal response.",
            )
        else:
            require(
                not previews and observed["preview_useful"] is None,
                "Never reclassify a painted preview or assign usefulness to a non-preview.",
            )
            if observed["status"] == "no_action":
                require(
                    any(
                        event["event"] == "response"
                        and event.get("outcome") == "no_action"
                        for event in events
                    ),
                    "No-action requires its actual response; failure is not abstention.",
                )
            if observed["status"] == "not_issued":
                require(
                    not any(event["event"] == "request" for event in events),
                    "An issued cancellation/nonpublication is a request_failure, not not_issued.",
                )
        timing = primary_timing(roster, plan, observed, paint, origin_registered)
        for key in ("origin_event", "full_latency_ms"):
            if key in observed:
                require(
                    observed[key] == timing[key],
                    "Recorded primary timing differs from its actual committing input.",
                )
        rows.append(
            {
                **plan,
                "status": observed["status"],
                "detail": observed.get("detail"),
                "issued": any(event["event"] == "request" for event in events),
                "preview_ms": timing["full_latency_ms"],
                **timing,
                "preview_useful": observed["preview_useful"],
            }
        )
    warm = [row for row in rows if row["condition"] == "warm"]
    previews = [row for row in warm if row["status"] == "preview"]
    timed_previews = [row for row in previews if row["preview_ms"] is not None]
    missing = [row for row in previews if row["preview_ms"] is None]
    eligible = [row for row in warm if row["intent"] in {"suggestion", "continuation"}]
    latency = distribution([row["preview_ms"] for row in timed_previews])
    return {
        "version": SUMMARY_VERSION,
        "scope": "exposed_development_ui",
        "primary_latency_metric": PRIMARY_METRIC,
        "latency_origin_amendment": observations.get("latency_origin_amendment")
        if origin_registered
        else None,
        "attempts": len(rows),
        "warm_attempts": len(warm),
        "outcome_counts": dict(Counter(row["status"] for row in rows)),
        "warm_outcome_counts": dict(Counter(row["status"] for row in warm)),
        "warm_by_intent": {
            intent: dict(
                Counter(row["status"] for row in warm if row["intent"] == intent)
            )
            for intent in sorted({row["intent"] for row in warm})
        },
        "warm_issued": sum(row["issued"] for row in warm),
        "warm_preview_latency": latency,
        "warm_trace_component_latency": distribution(
            [row["trace_latency_ms"] for row in previews]
        ),
        "primary_timing_coverage": {
            "warm_painted_previews": len(previews),
            "available": len(timed_previews),
            "missing": len(missing),
            "share": len(timed_previews) / len(previews) if previews else None,
            "missing_case_ids": [row["case_id"] for row in missing],
            "missing_reasons": dict(
                Counter(row["full_latency_unavailable_reason"] for row in missing)
            ),
        },
        "warm_by_language": {
            language: {
                "attempts": sum(row["language"] == language for row in warm),
                "preview_latency": distribution(
                    [
                        row["preview_ms"]
                        for row in timed_previews
                        if row["language"] == language
                    ]
                ),
            }
            for language in sorted({row["language"] for row in warm})
        },
        "warm_by_seed": {
            seed: {
                "attempts": sum(row["seed"] == seed for row in warm),
                "preview_latency": distribution(
                    [row["preview_ms"] for row in timed_previews if row["seed"] == seed]
                ),
            }
            for seed in sorted({row["seed"] for row in warm})
        },
        "useful_previews": {
            "count": sum(row["preview_useful"] is True for row in previews),
            "painted": len(previews),
            "unknown_usefulness": sum(
                row["preview_useful"] is None for row in previews
            ),
            "all_warm_attempts": len(warm),
            "eligible_attempts": len(eligible),
            "useful_on_eligible": sum(
                row["preview_useful"] is True for row in eligible
            ),
        },
        "preview_under_1000ms_per_warm_attempt": {
            "count": latency["under_1000ms"]["count"],
            "denominator": len(warm),
        },
        "latency_goal_on_observed_previews": None
        if not previews or missing
        else latency["p95_ms"] < 1000,
        "stretch_goal_on_observed_previews": None
        if not previews or missing
        else latency["p50_ms"] < 500,
        "cold_startup": [
            {**row, "startup": observed.get("startup")}
            for row, observed in zip(rows, outcomes, strict=True)
            if row["condition"] == "process_cold"
        ],
        "percentiles": "p50 median; p95 nearest rank ceil(0.95*N), without interpolation; strict <500/<1000ms shares",
        "scope_limit": "Conditional warm trusted committing DOM input to validated two-RAF paint proxy. Missing origins stay unavailable; trace durations are separate diagnostics. All non-previews remain attempts. Excludes OS/CDP delivery latency and does not establish all-interaction latency, independent quality, population p95, optical display latency, or physical desktop visibility.",
        "quality_gates_changed": False,
        "product_complete_asserted": False,
        "attempt_results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize")
    for key in ("roster", "observations", "output"):
        summary.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    else:
        require(not args.output.exists(), "Use a fresh summary path.")
        observations = read(args.observations)
        require(
            observations["measurement_roster"] == binding(args.roster),
            "Observation roster binding changed.",
        )
        roster = read(args.roster)
        require(
            roster["measurement_plan"] == binding(roster["measurement_plan"]["path"]),
            "Registered measurement protocol changed.",
        )
        preparation_source = observations.get(
            "registered_preparation_source", roster["preparation_script"]
        )
        require(
            preparation_source == binding(preparation_source["path"])
            and preparation_source["sha256"] == roster["preparation_script"]["sha256"],
            "Registered preparation source changed; preserve its exact historical bytes.",
        )
        for row in observations["outcomes"]:
            for item in row["evidence"]:
                require(item == binding(item["path"]), "UI evidence binding changed.")
        result = {
            **summarize(roster, observations),
            "measurement_roster": binding(args.roster),
            "observations": binding(args.observations),
            "registered_preparation_source": preparation_source,
            "summary_script": binding(__file__),
        }
        write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
