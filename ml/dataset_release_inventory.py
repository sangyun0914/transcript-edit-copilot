"""Inventory release inputs and lock their existing splits without opening final tests."""

import argparse
import copy
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK_ORDER = "docs/plans/dataset-work-order-v1.json"
WINDOWS = "data/derived/nep/source-windows/iteration-03/windows.jsonl"
CALCULUS = "data/derived/local-tests/calculus-30min-v1"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def resolve_local(root, relative):
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Input path escapes the repository.")
    if candidate.exists():
        return candidate
    current = root
    for part in Path(relative).parts:
        matches = [
            p
            for p in current.iterdir()
            if unicodedata.normalize("NFC", p.name)
            == unicodedata.normalize("NFC", part)
        ]
        if len(matches) != 1:
            raise FileNotFoundError(relative)
        current = matches[0]
    return current


def register_group(groups, group, split, origin, kind):
    if split not in {"train", "dev", "reserved_test", "regression"}:
        raise ValueError(f"Unrecognized split: {split}")
    if group in groups and groups[group]["split"] != split:
        raise ValueError(f"Source group crosses splits: {group}")
    item = groups.setdefault(group, {"split": split, "kind": kind, "origins": []})
    if origin not in item["origins"]:
        item["origins"].append(origin)


PLANNED_SOURCES = [
    (
        "fosdem-2026-python-lightning",
        "English technical recognition and speaker changes",
    ),
    ("fosdem-podlibre", "Workflow reference; no fabricated editing log"),
    ("fosdem-2024-vim-demo", "Workflow reference; no fabricated editing log"),
    ("chimchakman-emotion-science", "Korean dialogue, emphasis and uncertainty"),
    ("chaos-extraterrestrial-life", "Korean scientific dialogue"),
    ("chaos-dark-energy", "Korean scientific explanation"),
    ("ami-additional-series", "Independent conversational development source"),
    ("zeroth-korean", "Independent Korean read-speech recognition"),
    ("ksponspeech", "Spontaneous Korean conversation"),
    ("aihub-71477", "Korean conversation"),
    ("aihub-71560", "Overcorrection controls"),
    ("aihub-71260", "Mixed-language evidence"),
    ("podcastfillers", "Filler, breath and overlap taxonomy"),
    ("seoul-corpus", "Korean conversational boundaries"),
    ("disfluencyspeech", "Acted disfluency; distinguish from natural conversation"),
    ("hyporadise", "N-best reconstruction evidence"),
    ("earnings22", "Numbers and technical vocabulary"),
    ("switchboard", "Additional conversation only if it fills a specific gap"),
    ("sssd", "Additional disfluency evidence only if necessary"),
    ("gigaspeech", "Additional independent source only if necessary"),
    ("coreaspeech", "Additional Korean source only if necessary"),
]


def build_inventory(root, output, recipe_path):
    work = read(root / WORK_ORDER)
    if digest(root / work["scenario_spec_path"]) != work["scenario_spec_sha256"]:
        raise ValueError("Scenario specification changed; update its registered hash.")
    recipe = read(recipe_path)
    checked, groups = {}, {}

    def check(relative, expected=None):
        path = resolve_local(root, relative)
        name = str(path.relative_to(root))
        if name not in checked:
            checked[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
        if expected and checked[name]["sha256"] != expected:
            raise ValueError(f"Source hash mismatch: {name}")
        return {"path": name, **checked[name]}

    for group in work["reserved_test_groups"]:
        register_group(
            groups, group, "reserved_test", WORK_ORDER, "unopened_reserved_source"
        )
    historical = {"train": {}, "dev": {}}
    for number in range(1, 6):
        for split in historical:
            path = f"artifacts/iterations/colab-l4-run-{number:02d}/input-bundle/{split}-cases.jsonl"
            check(path)
            for case in rows(root / path):
                meta = case["metadata"]
                if meta["split"] != split:
                    raise ValueError("Archive split does not match its filename.")
                register_group(
                    groups,
                    meta["source_group_id"],
                    split,
                    path,
                    "recording_derivative"
                    if meta["source_group_id"] == "sedong-local-calibration"
                    else "authored_script",
                )
                pair = json.dumps(
                    [case["input"], case["expectedOutput"]],
                    sort_keys=True,
                    ensure_ascii=False,
                )
                historical[split][hashlib.sha256(pair.encode()).hexdigest()] = case[
                    "id"
                ]
    if historical["train"].keys() & historical["dev"].keys():
        raise ValueError("Historical exact pair crosses splits.")

    fixtures = []
    known = {f["path"]: f for f in work["fixtures"]}
    for path in sorted((root / "ml/fixtures").iterdir()):
        if not path.is_file():
            continue
        name = str(path.relative_to(root))
        spec = known.get(name)
        if spec is None and path.resolve() != recipe_path.resolve():
            raise ValueError(f"Unassigned fixture: {name}")
        fixtures.append(
            {
                **check(name, spec["sha256"] if spec else None),
                "role": spec["assigned_role"]
                if spec
                else "New source-backed release recipe",
                "release_status": "accounted_not_automatically_appended",
                "admitted_case_ids": [],
            }
        )
    if set(known) - {f["path"] for f in fixtures}:
        raise ValueError("A registered fixture is missing.")

    windows = rows(root / WINDOWS)
    check(WINDOWS)
    local = {
        s["id"]: s
        for s in read(root / "data/manifests/local-iteration-03-sources.json")[
            "sources"
        ]
    }
    current = []
    for original in work["sources"]:
        source = copy.deepcopy(original)
        sid = source["source_id"]
        register_group(
            groups,
            source["source_group_id"],
            source["split"],
            source["source_manifest"],
            "real_recording",
        )
        source.update(
            {
                "acquired": True,
                "trained_in_new_release": False,
                "release_admission_status": "draft_only",
                "files": [],
            }
        )
        if sid in {w["source_id"] for w in windows}:
            selected = [w for w in windows if w["source_id"] == sid]
            first = selected[0]
            if any(
                w["reserved_split"] != source["split"]
                or w["source_group_id"] != source["source_group_id"]
                for w in selected
            ):
                raise ValueError("Window lineage crosses the registered split.")
            source["files"].append(check(first["source_video"], first["video_sha256"]))
            transcript = (
                local[sid]["project"]
                if sid in local
                else f"data/raw/ocw/{sid}/captions.vtt"
            )
            source["files"].append(check(transcript, first["transcript_sha256"]))
            source["transcript_origin"] = first["transcript_origin"]
            source["terms"] = first["license"]
            source["language"] = first["language"]
            source["windows"] = [
                {"id": w["id"], "start": w["start_seconds"], "end": w["end_seconds"]}
                for w in selected
            ]
            for w in selected:
                check(
                    f"data/derived/nep/source-windows/iteration-03/{w['audio_preview']}",
                    w["audio_sha256"],
                )
            source["transcribed"] = True
            source["reference_status"] = "prior_references_revalidate_in_draft"
        elif sid == "ami-es2002a":
            folder = "data/raw/ami/ami-pilot-v1"
            acquisition = read(root / folder / "acquisition.json")
            source["files"].append(check(f"{folder}/acquisition.json"))
            for name, record in acquisition["files"].items():
                source["files"].append(check(f"{folder}/{name}", record["sha256"]))
            for name, expected in acquisition["extracted_annotation_sha256"].items():
                check(f"{folder}/annotations/{name}", expected)
            source.update(
                {
                    "transcribed": True,
                    "language": "en",
                    "reference_status": "overlap_aware_draft_candidates"
                    if recipe.get("include_ami")
                    else "source_only_overlap_aware_import_pending",
                    "terms": acquisition["catalog"]["license"],
                }
            )
        elif sid == "local-ko-calculus":
            record = next(
                r
                for r in read(root / "data/manifests/local-tests.json")["files"]
                if r["sha256"].startswith("11c8e9b16fbb")
            )
            source["files"].append(check("data/" + record["path"], record["sha256"]))
            source["files"].append(check("data/manifests/calculus-30min-v1.json"))
            acquisition_path = root / CALCULUS / "acquisition.json"
            if acquisition_path.exists():
                acquisition = read(acquisition_path)
                source["files"].append(check(f"{CALCULUS}/acquisition.json"))
                for clip in acquisition["clips"]:
                    source["files"].append(
                        check(f"{CALCULUS}/{clip['result']}", clip["result_sha256"])
                    )
                    source["files"].append(
                        check(
                            f"{CALCULUS}/{clip['id']}/input.wav", clip["audio_sha256"]
                        )
                    )
                source.update(
                    {
                        "transcribed": True,
                        "transcript_status": "scribe_v2_two_15min_excerpts",
                        "reference_status": "source_only_pending_reference_selection",
                        "transcribed_seconds": acquisition["total_audio_seconds"],
                        "full_recording_status": "quota_blocked_10770_required_4071_available_before_excerpt_requests",
                        "windows": [
                            {
                                "id": c["id"],
                                "start": c["start_seconds"],
                                "end": c["start_seconds"] + c["duration_seconds"],
                            }
                            for c in acquisition["clips"]
                        ],
                    }
                )
            else:
                source.update(
                    {
                        "transcribed": False,
                        "reference_status": "acquisition_pending",
                        "blocking_reason": "Approved 30-minute Scribe request has not produced a verified acquisition manifest.",
                    }
                )
            source["language"] = "ko"
            source["split_registration_status"] = (
                "registered_before_transcript_inspection"
            )
        else:
            raise ValueError(f"Unaccounted source: {sid}")
        source["project_use_authorization"] = (
            "data/manifests/project-source-use-2026-09-25.json"
        )
        current.append(source)

    if recipe.get("include_ami_dev"):
        folder = "data/raw/ami/ami-dev-v1"
        acquisition = read(root / folder / "acquisition.json")
        catalog = acquisition["catalog"]
        register_group(
            groups,
            catalog["source_group_id"],
            catalog["split"],
            "data/manifests/ami-dev-v1.json",
            "real_recording",
        )
        files = [
            check("data/manifests/ami-dev-v1.json", acquisition["catalog_sha256"]),
            check(f"{folder}/acquisition.json"),
        ]
        files.extend(
            check(f"{folder}/{name}", record["sha256"])
            for name, record in acquisition["files"].items()
        )
        files.extend(
            check(f"{folder}/annotations/{name}", expected)
            for name, expected in acquisition["extracted_annotation_sha256"].items()
        )
        current.append(
            {
                "source_id": catalog["source_id"],
                "source_group_id": catalog["source_group_id"],
                "split": catalog["split"],
                "language": catalog["language"],
                "source_manifest": "data/manifests/ami-dev-v1.json",
                "acquired": True,
                "transcribed": True,
                "trained_in_new_release": False,
                "transcript_origin": "publisher_word_event_annotations",
                "split_registration_status": "registered_before_transcript_inspection",
                "terms": catalog["license"],
                "files": files,
            }
        )

    authored = []
    if recipe.get("scripted_sources"):
        registration = read(root / recipe["scripted_sources"])
        check(recipe["scripted_sources"])
        check(registration["sources_file"])
        for group in registration["groups"]:
            register_group(
                groups,
                group["id"],
                group["split"],
                recipe["scripted_sources"],
                "authored_script",
            )
            authored.append(
                {
                    **group,
                    "source_document": registration["sources_file"],
                    "provenance": registration["provenance"],
                    "recorded_audio": False,
                }
            )

    for group in recipe.get("new_script_groups", []):
        register_group(
            groups,
            group["id"],
            group["split"],
            str(recipe_path.relative_to(root)),
            "authored_script",
        )
    check(str(recipe_path.relative_to(root)))
    check(WORK_ORDER)
    check(work["scenario_spec_path"])
    check("data/manifests/project-source-use-2026-09-25.json")
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "source-inventory.json",
        {
            "version": "nep-release-v1-inventory-v1",
            "current_sources": current,
            "authored_sources": authored,
            "fixtures": fixtures,
            "planned_sources": [
                {
                    "source_id": sid,
                    "status": "acquired"
                    if sid == "ami-additional-series" and recipe.get("include_ami_dev")
                    else "not_needed_for_v1",
                    "potential_gap": gap,
                    "reason": "AMI IS1008b independently preregistered and acquired for conversational development."
                    if sid == "ami-additional-series" and recipe.get("include_ami_dev")
                    else recipe.get("source_dispositions", {}).get(
                        sid,
                        "The supported v1 decisions use the seven registered recordings and bounded authored timelines. This additional corpus is not required to validate those admitted references; unqualified natural-cut subcoverage remains explicit rather than being filled by bulk acquisition.",
                    ),
                }
                for sid, gap in PLANNED_SOURCES
            ],
            "reserved_test_sources": work["reserved_test_groups"],
            "historical_unique_pairs": {k: len(v) for k, v in historical.items()},
            "input_files": checked,
            "test_bodies_opened": False,
        },
    )
    write(
        output / "split-manifest.json",
        {
            "version": 1,
            "groups": groups,
            "conflicts": [],
            "lineage_rule": "Recording/course/meeting-series/script parent, neighboring windows, translations and retranscriptions stay in their existing group.",
            "unassigned_groups": [],
            "reserved_test_groups": work["reserved_test_groups"],
        },
    )
    planned = defaultdict(list)
    for case in recipe["cases"] + [
        s
        for e in recipe.get("episodes", [])
        for s in e["steps"]
        if s.get("record_case", True)
    ]:
        for scenario in case["scenario_ids"]:
            planned[scenario].append(case["id"])
    write(
        output / "coverage-plan.json",
        {
            "version": 1,
            "status": "planned_before_draft_generation_not_admissions",
            "decisions": [
                {
                    **d,
                    "planned_case_ids": planned[d["id"]],
                    "candidate_sources": recipe["coverage_sources"].get(d["id"], []),
                    "gap": recipe["gaps"].get(
                        d["id"],
                        "Validate candidate against exact visible context and execution before admission.",
                    ),
                }
                for d in work["content_decisions"]
            ],
        },
    )
    protocol = (
        read(root / recipe["test_protocol"])
        if recipe.get("test_protocol")
        else {
            "version": "release-v1-test-design-draft",
            "status": "design_only_not_final_protocol_freeze",
            "reserved_source_groups": work["reserved_test_groups"],
            "generated_cases": 0,
            "generation_gate": "Freeze model/runtime/prompt/context/display policy, generator/rubric, source selection and metrics before inspecting or generating test bodies.",
            "measures": [
                "action_target_parameter_outcome",
                "raw_invalid_and_timeouts",
                "shown_edit_precision",
                "useful_edit_recall",
                "preserve_and_repeat_errors",
                "replay_undo_caption_integrity",
                "all_request_latency_and_memory",
            ],
            "strata": [
                "language",
                "source_episode",
                "action",
                "natural_vs_constructed",
                "reference_assisted_vs_open_next_edit",
            ],
            "remaining": "Pin sample allocation and thresholds after development, before final-test generation; do not tune from final-test findings.",
        }
    )
    write(output / "test-protocol.json", protocol)
    return {
        "current_sources": len(current),
        "fixtures": len(fixtures),
        "groups": len(groups),
        "files_verified": len(checked),
        "transcribed_sources": sum(s["transcribed"] for s in current),
        "test_bodies_opened": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_inventory(ROOT, args.output.resolve(), args.recipe.resolve()),
            indent=2,
        )
    )
