"""Prepare a small, separately scoped all-action proposal pilot from authored scenes.

No historical passive labels are migrated. Source timing and acoustic annotations
are explicitly symbolic; this preparation establishes reversible editor behavior,
not the acoustic safety or usefulness of a real recording.
"""

import argparse
import ast
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import passive_actions as contract
from replay_session import replay, session_hash
from session_cases import replayable_session

ROOT = Path(__file__).resolve().parents[1]
VERSION = "passive-actions-pilot-preparation-v1"
REGISTRY = ROOT / "ml/passive-models-v2.json"
TOKENIZERS = ("gemma26", "qwen35-9b")
EOS_IDS = {"gemma26": 1, "qwen35-9b": 248046}
GROUPS = (
    ("en-field-recorder", "en", "train"),
    ("en-community-kitchen", "en", "train"),
    ("en-bookbinding", "en", "train"),
    ("ko-paper-folding", "ko", "train"),
    ("ko-science-workshop", "ko", "train"),
    ("ko-library-recording", "ko", "train"),
    ("en-bicycle-light", "en", "dev"),
    ("ko-plant-observation", "ko", "dev"),
)
PROFILE = {
    "version": 1,
    "export_line_length": None,
    "min_cue_ms": 1000,
    "max_merge_cue_ms": 7000,
    "max_merge_text_utf16": 84,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(packed(value) + "\n" for value in values).encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    with path.open("xb") as stream:
        stream.write(raw)


def recipe():
    return {
        "version": VERSION,
        "contract_version": contract.VERSION,
        "status": "bounded_authored_pilot_pending_semantic_review",
        "source_groups_assigned_before_case_construction": [
            {"source_group": group, "language": language, "split": split}
            for group, language, split in GROUPS
        ],
        "counts": {"total": 56, "train": 42, "dev": 14, "test": 0},
        "per_action_language": {"train": 3, "dev": 1},
        "actions": list(contract.ALLOWED_ACTIONS),
        "historical_rows_migrated": 0,
        "natural_audio_rows": 0,
        "authored_symbolic_rows": 56,
        "policy": "One fixed seven-action, source-timed, preview-only policy; no per-case task brief.",
        "limitations": [
            "Generated source identities and intervals are not recordings or acoustic verification.",
            "Gap labels remain unknown; annotations in preservation controls are explicitly synthetic.",
            "Reference previews are authored policy examples, not observed human preferences.",
            "Actual committed synthetic history supplies local pause-style evidence; it is not a natural edit log.",
            "This small balanced pilot does not estimate production action prevalence or generalization.",
            "Old 809-train/289-dev passive labels are not admitted under the expanded action policy.",
        ],
    }


def text(key, value, milliseconds=2200):
    return {"key": key, "text": value, "milliseconds": milliseconds}


def gap(key, milliseconds):
    return {"key": key, "text": None, "milliseconds": milliseconds}


def scene(
    action,
    paragraphs,
    target=None,
    *,
    replacement=None,
    keep_ms=None,
    focus=None,
    history=None,
    note="",
    annotations=(),
    selection=(),
):
    prediction = {"action": action}
    if target is not None:
        prediction["targets"] = [target] if isinstance(target, str) else target
    if replacement is not None:
        prediction["text"] = replacement
    if keep_ms is not None:
        prediction["keep_ms"] = keep_ms
    return {
        "paragraphs": paragraphs,
        "prediction": prediction,
        "focus": focus or paragraphs[-1][0]["key"],
        "history": history,
        "rationale": note,
        "annotations": list(annotations),
        "selection": list(selection),
    }


def pause(action, previous, current, before_ms, current_ms, keep_ms=None):
    first = [
        text("h-left", previous[0], 1600),
        gap("h-gap", before_ms),
        text("h-right", previous[1], 1800),
    ]
    second = [
        text("left", current[0], 1800),
        gap("gap", current_ms),
        text("right", current[1], 2200),
    ]
    prior = {"action": action, "targets": ["h-gap"]}
    if action == "trim_silence":
        prior["keep_ms"] = keep_ms
    return scene(
        action,
        [first, second],
        "gap",
        keep_ms=keep_ms,
        focus="right",
        history=prior,
        note=(
            "The previous genuinely executed synthetic pause edit supplies local pacing "
            "and retention evidence. The next unfinished phrase has the same kind of "
            "interruption. This is a listening-preview proposal for an unknown gap, "
            "not a claim that the interval contains silence."
        ),
    )


def scenes():
    t, g, s, p = text, gap, scene, pause
    return {
        "en-field-recorder": [
            s(
                "no_action",
                [[t("a", "Really, really check the polarity.", 3200)]],
                note="The repeated intensifier is deliberate emphasis; there is no justified repair.",
            ),
            s(
                "correct_text",
                [[t("a", "The microhpone is connected.", 2800)]],
                "a",
                replacement="The microphone is connected.",
                note="Local, unambiguous letter transposition.",
            ),
            s(
                "delete_span",
                [[t("a", "Um,", 450), t("b", "connect the red cable first.", 3200)]],
                "a",
                focus="b",
                note="An isolated leading hesitation is dispensable; the instruction remains complete.",
            ),
            p(
                "trim_silence",
                ("Set the input", "to the lower range."),
                ("Move the microphone", "away from the fan."),
                1500,
                2300,
                400,
            ),
            p(
                "remove_silence",
                ("Plug the cable into", "the left socket."),
                ("Check the label on", "the lower switch."),
                550,
                650,
            ),
            s(
                "split_paragraph",
                [
                    [
                        t("a", "The recorder is now ready.", 3200),
                        t("b", "Next we will test the headphones.", 3800),
                    ]
                ],
                "b",
                note="Two independent setup stages can become two readable complete cues.",
            ),
            s(
                "merge_paragraph",
                [[t("a", "Connect the", 600)], [t("b", "red cable first.", 2000)]],
                "b",
                note="A short incomplete determiner phrase is reunited with its object within the profile.",
            ),
        ],
        "en-community-kitchen": [
            s(
                "no_action",
                [
                    [t("a", "Wait for the timer.", 2000), g("bell", 600)],
                    [t("b", "Then open the lid.", 2300)],
                ],
                annotations=[("bell", "audio_event")],
                note="The annotated timer sound is meaningful; a gap-removal probe must be rejected.",
            ),
            s(
                "correct_text",
                [[t("a", "Keep the lid closde.", 2400)]],
                "a",
                replacement="Keep the lid closed.",
                note="Obvious spelling error, preserving wording and audio.",
            ),
            s(
                "delete_span",
                [
                    [
                        t("a", "Put the—", 650),
                        t("b", "place the tray on the rack.", 3300),
                    ]
                ],
                "a",
                focus="b",
                note="The visibly abandoned construction is followed by a complete restart.",
            ),
            p(
                "trim_silence",
                ("Pour the water", "into the large bowl."),
                ("Stir the mixture", "with the wooden spoon."),
                1800,
                2700,
                500,
            ),
            p(
                "remove_silence",
                ("Take the bowl from", "the upper shelf."),
                ("Place the lid beside", "the empty plate."),
                420,
                780,
            ),
            s(
                "split_paragraph",
                [
                    [
                        t("a", "The soup can simmer here.", 3000),
                        t("b", "Meanwhile prepare the serving bowls.", 4000),
                    ]
                ],
                "b",
                note="Independent concurrent steps are clearer in separate complete cues.",
            ),
            s(
                "merge_paragraph",
                [
                    [t("a", "Into the", 650)],
                    [t("b", "large bowl, pour the water.", 2800)],
                ],
                "b",
                note="The isolated preposition fragment belongs to its following noun phrase.",
            ),
        ],
        "en-bookbinding": [
            s(
                "no_action",
                [[t("a", "The printed title is 'Um'.", 3100)]],
                note="Um is a quoted title, not a spoken filler to remove.",
            ),
            s(
                "correct_text",
                [[t("a", "Fold along the creese.", 2700)]],
                "a",
                replacement="Fold along the crease.",
                note="A spelling repair of a concrete local instruction.",
            ),
            s(
                "delete_span",
                [
                    [
                        t("a", "Keep", 350),
                        t("b", "the", 180),
                        t("c", "the", 190),
                        t("d", "pages aligned.", 1900),
                    ]
                ],
                "b",
                focus="d",
                note="Accidental duplicated article; one occurrence is removed without changing meaning.",
            ),
            p(
                "trim_silence",
                ("Press the fold", "with the smooth handle."),
                ("Hold the cover", "against the straight edge."),
                2100,
                3100,
                600,
            ),
            p(
                "remove_silence",
                ("Keep both edges in", "the same position."),
                ("Slide the thread through", "the first hole."),
                600,
                900,
            ),
            s(
                "split_paragraph",
                [
                    [
                        t("a", "The pages are held firmly now.", 3500),
                        t("b", "We can begin sewing the spine.", 3700),
                    ]
                ],
                "b",
                note="Completed preparation and the next operation form separate useful complete cues.",
            ),
            s(
                "merge_paragraph",
                [
                    [t("a", "The folded", 700)],
                    [t("b", "edge must stay straight.", 2500)],
                ],
                "b",
                note="Merging repairs an adjective/noun boundary and eliminates a subsecond cue.",
            ),
        ],
        "ko-paper-folding": [
            s(
                "no_action",
                [[t("a", "아주 아주 천천히 접으세요.", 3000)]],
                note="아주 아주 is emphatic instruction, not an accidental repeated word.",
            ),
            s(
                "correct_text",
                [[t("a", "모서리를 정학하게 맞추세요.", 2800)]],
                "a",
                replacement="모서리를 정확하게 맞추세요.",
                note="Recoverable local spelling error without style changes.",
            ),
            s(
                "delete_span",
                [[t("a", "어,", 380), t("b", "모서리를 가운데로 접으세요.", 2900)]],
                "a",
                focus="b",
                note="Isolated hesitation before a complete instruction; no instruction content removed.",
            ),
            p(
                "trim_silence",
                ("종이의 끝을", "가운데 선에 맞춥니다."),
                ("반대쪽 모서리도", "같은 선에 맞추세요."),
                1600,
                2400,
                400,
            ),
            p(
                "remove_silence",
                ("오른쪽 끝을", "왼손으로 잡으세요."),
                ("접은 부분을", "손끝으로 눌러 주세요."),
                500,
                700,
            ),
            s(
                "split_paragraph",
                [
                    [
                        t("a", "아래쪽 접기가 끝났습니다.", 3200),
                        t("b", "이제 종이를 뒤집어 주세요.", 3600),
                    ]
                ],
                "b",
                note="The completed step and following instruction have a clear grouping boundary.",
            ),
            s(
                "merge_paragraph",
                [[t("a", "오른쪽", 600)], [t("b", "모서리를 안으로 접습니다.", 2700)]],
                "b",
                note="The detached directional modifier belongs to the following noun and instruction.",
            ),
        ],
        "ko-science-workshop": [
            s(
                "no_action",
                [
                    [t("a", "신호음을 기다리세요.", 2200), g("tone", 700)],
                    [t("b", "그다음 눈금을 읽습니다.", 2600)],
                ],
                annotations=[("tone", "audio_event")],
                note="The annotated signal sound conveys the stated timing and is not removable silence.",
            ),
            s(
                "correct_text",
                [[t("a", "물의 온도를 기룍하세요.", 2600)]],
                "a",
                replacement="물의 온도를 기록하세요.",
                note="Obvious local spelling repair retaining the exact task.",
            ),
            s(
                "delete_span",
                [[t("a", "왼쪽을—", 700), t("b", "오른쪽 손잡이를 돌리세요.", 3000)]],
                "a",
                focus="b",
                note="A visibly abandoned false start is superseded by the complete directional instruction.",
            ),
            p(
                "trim_silence",
                ("첫 번째 눈금을", "표에 기록합니다."),
                ("두 번째 눈금도", "같은 표에 적어 주세요."),
                2000,
                2900,
                500,
            ),
            p(
                "remove_silence",
                ("투명한 용기를", "책상 위에 놓으세요."),
                ("작은 뚜껑을", "오른쪽으로 돌립니다."),
                450,
                850,
            ),
            s(
                "split_paragraph",
                [
                    [
                        t("a", "첫 측정값을 기록했습니다.", 3300),
                        t("b", "이번에는 차가운 물을 넣겠습니다.", 4000),
                    ]
                ],
                "b",
                note="One completed measurement and a new condition should be distinct complete cues.",
            ),
            s(
                "merge_paragraph",
                [[t("a", "투명한", 700)], [t("b", "용기에 물을 담습니다.", 2500)]],
                "b",
                note="A detached adjective and noun become one readable instruction.",
            ),
        ],
        "ko-library-recording": [
            s(
                "no_action",
                [[t("a", "작품 제목은 '음'입니다.", 2800)]],
                note="음 is the stated work title, not hesitation; deletion would change the content.",
            ),
            s(
                "correct_text",
                [[t("a", "다음 문장을 천천히 일으세요.", 3000)]],
                "a",
                replacement="다음 문장을 천천히 읽으세요.",
                note="Context makes the intended reading verb explicit.",
            ),
            s(
                "delete_span",
                [
                    [
                        t("a", "책장을", 550),
                        t("b", "책장을", 580),
                        t("c", "조용히 넘깁니다.", 2200),
                    ]
                ],
                "a",
                focus="c",
                note="Immediate same-object restart before the only predicate; retain the completed occurrence.",
            ),
            p(
                "trim_silence",
                ("표지의 제목을", "한 번 읽어 주세요."),
                ("아래쪽 부제도", "같은 속도로 읽습니다."),
                2200,
                3200,
                600,
            ),
            p(
                "remove_silence",
                ("첫 문장의 끝에서", "마침표를 확인합니다."),
                ("두 번째 줄에서는", "쉼표 위치를 살펴보세요."),
                650,
                950,
            ),
            s(
                "split_paragraph",
                [
                    [
                        t("a", "첫 문단의 낭독이 끝났습니다.", 3500),
                        t("b", "이어서 두 번째 문단을 읽겠습니다.", 4000),
                    ]
                ],
                "b",
                note="Two independent narration stages have a useful complete-sentence boundary.",
            ),
            s(
                "merge_paragraph",
                [[t("a", "표지의", 650)], [t("b", "제목을 먼저 읽어 주세요.", 2600)]],
                "b",
                note="The possessive fragment and its noun should share one readable cue.",
            ),
        ],
        "en-bicycle-light": [
            s(
                "no_action",
                [[t("a", "It may need two washers, not three.", 3800)]],
                note="Uncertainty, number and negation are meaningful and must remain intact.",
            ),
            s(
                "correct_text",
                [[t("a", "Tighten the braket gently.", 2900)]],
                "a",
                replacement="Tighten the bracket gently.",
                note="Local bracket spelling repair; no tightening claim changes.",
            ),
            s(
                "delete_span",
                [
                    [
                        t("a", "We can,", 500),
                        t("b", "we can", 540),
                        t("c", "test the light now.", 2300),
                    ]
                ],
                "a",
                focus="c",
                note="Immediate clause-opening restart is dispensable; the remaining clause is complete.",
            ),
            p(
                "trim_silence",
                ("Place the bracket", "under the handlebar."),
                ("Keep the cable", "away from the brake lever."),
                1900,
                2800,
                450,
            ),
            p(
                "remove_silence",
                ("Slide the strap around", "the narrow tube."),
                ("Fasten the clip onto", "the rubber mount."),
                580,
                820,
            ),
            s(
                "split_paragraph",
                [
                    [
                        t("a", "The light is firmly attached.", 3400),
                        t("b", "Now check it from across the room.", 3900),
                    ]
                ],
                "b",
                note="A completed mounting step and a separate check become readable complete cues.",
            ),
            s(
                "merge_paragraph",
                [
                    [t("a", "The rubber", 650)],
                    [t("b", "strap goes beneath the bar.", 2800)],
                ],
                "b",
                note="The modifier/noun fracture is repaired within duration and text limits.",
            ),
        ],
        "ko-plant-observation": [
            s(
                "no_action",
                [[t("a", "아마 잎이 두 장 더 나온 것 같습니다.", 4000)]],
                note="Tentative observation and count are meaningful; no correction or deletion is justified.",
            ),
            s(
                "correct_text",
                [[t("a", "잎의 색갈을 기록해 주세요.", 2900)]],
                "a",
                replacement="잎의 색깔을 기록해 주세요.",
                note="Unambiguous local spelling error without changing the observation.",
            ),
            s(
                "delete_span",
                [
                    [
                        t("a", "오늘은,", 450),
                        t("b", "오늘은", 480),
                        t("c", "새 잎을 살펴보겠습니다.", 2500),
                    ]
                ],
                "a",
                focus="c",
                note="Immediate opening restart leaves a complete single temporal phrase and predicate.",
            ),
            p(
                "trim_silence",
                ("잎의 가장자리를", "가까이에서 봅니다."),
                ("줄기의 아랫부분도", "같은 거리에서 살펴보세요."),
                1700,
                2600,
                450,
            ),
            p(
                "remove_silence",
                ("작은 화분을", "밝은 쪽으로 옮깁니다."),
                ("새로 나온 잎을", "옆에서 관찰해 주세요."),
                520,
                880,
            ),
            s(
                "split_paragraph",
                [
                    [
                        t("a", "오늘의 잎 관찰을 마쳤습니다.", 3400),
                        t("b", "다음에는 줄기의 높이를 재겠습니다.", 4100),
                    ]
                ],
                "b",
                note="Completed observation and next measurement have distinct complete wording.",
            ),
            s(
                "merge_paragraph",
                [
                    [t("a", "새로 나온", 700)],
                    [t("b", "잎의 길이를 재어 보세요.", 2800)],
                ],
                "b",
                note="The detached noun modifier should share its complete short instruction.",
            ),
        ],
    }


def make_document(spec, source_id, language):
    content, keys, intervals, elapsed_ms = [], {}, {}, 0

    def identity(key):
        return str(uuid5(NAMESPACE_URL, f"{VERSION}/{source_id}/{key}"))

    for n, paragraph in enumerate(spec["paragraphs"]):
        content.append(
            {
                "type": "paragraph_start",
                "uuid": identity(f"p{n}"),
                "speaker": "narrator",
                "language": language,
            }
        )
        for item in paragraph:
            key, milliseconds = item["key"], item["milliseconds"]
            require(
                key not in keys and type(milliseconds) is int and milliseconds > 0,
                "Authored item keys/timing must be unique and positive.",
            )
            keys[key] = identity(key)
            intervals[key] = (elapsed_ms / 1000, (elapsed_ms + milliseconds) / 1000)
            value = {
                "type": "non_text" if item["text"] is None else "text",
                "uuid": keys[key],
                "source": source_id,
                "sourceStart": elapsed_ms / 1000,
                "length": milliseconds / 1000,
            }
            if item["text"] is not None:
                value.update(text=item["text"], conf=1)
            content.append(value)
            elapsed_ms += milliseconds
        content.append({"type": "paragraph_end", "uuid": identity(f"e{n}")})
    metadata = {"provenance": "authored_symbolic_timing_not_audio"}
    if spec["annotations"]:
        metadata["transcriptions"] = [
            {
                "source": source_id,
                "provenance": "synthetic_control_annotation_not_audio_observation",
                "annotations": [
                    {"type": kind, "start": intervals[key][0], "end": intervals[key][1]}
                    for key, kind in spec["annotations"]
                ],
            }
        ]
    return {"content": content, "metadata": metadata}, keys


def resolve(prediction, keys, bundle):
    result = copy.deepcopy(prediction)
    if "targets" in result:
        aliases = {
            item["uuid"]: int(alias) for alias, item in bundle["mapping"].items()
        }
        result["target"] = [aliases[keys[key]] for key in result.pop("targets")]
    return result


def execution(document, bundle, prediction, source):
    after = contract.apply_passive_edit(document, bundle, prediction)
    evidence = {
        "prediction": prediction,
        "before_sha256": session_hash(document),
        "after_sha256": session_hash(after),
        "document_hash_format": "typed-json-f64-v1",
        "source_interval_claim": "symbolic_source_timing_only",
    }
    if prediction["action"] == "no_action":
        require(document == after, "Abstention changed the document.")
        evidence.update(changed=False, replay=None, journal=[])
    else:
        require(document != after, "An edit reference must change the document.")
        transition = contract.committed_transition(document, after, prediction, 1)
        journal = replayable_session(document, after, prediction["action"], source)
        restored, summary = replay(journal)
        require(
            restored == after and summary["transactions"] == 3,
            "Reference apply/undo/redo did not replay exactly.",
        )
        evidence.update(
            changed=True,
            transition_sha256=session_hash(transition),
            replay=summary,
            journal=journal,
        )
    return evidence


def rejection_probes(document, bundle, prediction):
    probes = [("unknown_alias", {"action": "delete_span", "target": [99999]})]
    items = bundle["input"]["items"]
    target = prediction.get("target", [items[0]["id"]])[0]
    if prediction["action"] in {"trim_silence", "remove_silence"}:
        duration = next(item["duration_ms"] for item in items if item["id"] == target)
        probes.extend(
            [
                (
                    "zero_retention",
                    {"action": "trim_silence", "target": [target], "keep_ms": 0},
                ),
                (
                    "nonshortening_trim",
                    {"action": "trim_silence", "target": [target], "keep_ms": duration},
                ),
            ]
        )
    for item in items:
        if item["type"] == "non_text" and item["audio_class"] == "audio_event":
            probes.append(
                (
                    "known_audio_event",
                    {"action": "remove_silence", "target": [item["id"]]},
                )
            )
    results = []
    for name, proposed in probes:
        try:
            contract.apply_passive_edit(document, bundle, proposed)
        except (ValueError, KeyError) as exc:
            results.append(
                {
                    "name": name,
                    "prediction": proposed,
                    "rejected": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            raise ValueError(f"Invalid control was admitted: {name}")
    return results


def build_cases(builder):
    authored = scenes()
    require(
        set(authored) == {g[0] for g in GROUPS},
        "Scenario groups differ from the assigned recipe.",
    )
    cases, runtime_inputs, executions, controls = [], [], [], []
    for group, language, split in GROUPS:
        specifications = authored[group]
        require(
            [s["prediction"]["action"] for s in specifications]
            == list(contract.ALLOWED_ACTIONS),
            "Every assigned group must cover all seven actions exactly once.",
        )
        for index, spec in enumerate(specifications):
            case_id = f"actions-v1-{group}-{index:02d}"
            source_id = "symbolic:" + str(uuid5(NAMESPACE_URL, VERSION + "/" + case_id))
            document, keys = make_document(spec, source_id, language)
            history = []
            if spec["history"] is not None:
                before = copy.deepcopy(document)
                prior_bundle = builder.build(
                    before, keys["h-gap"], as_of=0, caption_profile=PROFILE
                )
                applied = resolve(spec["history"], keys, prior_bundle)
                document = contract.apply_passive_edit(before, prior_bundle, applied)
                history = [contract.committed_transition(before, document, applied, 1)]
            sequence = len(history)
            focus = keys[spec["focus"]]
            selection = [keys[key] for key in spec["selection"]]
            bundle = builder.build(
                document,
                focus,
                as_of=sequence,
                selection=selection,
                history=history,
                caption_profile=PROFILE,
            )
            require(
                not bundle["context"]["budget_reduced"],
                "Pilot snapshots must remain complete.",
            )
            prediction = resolve(spec["prediction"], keys, bundle)
            source = {
                "id": source_id,
                "source_group": group,
                "split": split,
                "language": language,
                "provenance": "authored_symbolic_timing_not_audio",
                "natural_audio": False,
                "acoustically_verified": False,
            }
            result = execution(document, bundle, prediction, source)
            messages = contract.canonical_messages(bundle["input"])
            case = {
                "id": case_id,
                "split": split,
                "language": language,
                "source": source,
                "primary": prediction,
                "accepted_predictions": [prediction],
                "rationale": spec["rationale"],
                "reference_status": "authored_policy_example_pending_independent_review",
                "input": bundle["input"],
                "messages": messages,
                "execution_sha256": digest(result),
            }
            runtime = {
                "id": case_id,
                "split": split,
                "document": document,
                "focus_uuid": focus,
                "selection": selection,
                "sequence": sequence,
                "history": history,
                "feedback_journal": None,
                "caption_profile": PROFILE,
            }
            cases.append(case)
            runtime_inputs.append(runtime)
            executions.append({"id": case_id, **result})
            controls.append(
                {
                    "id": case_id,
                    "probes": rejection_probes(document, bundle, prediction),
                }
            )
    require(
        len(cases) == 56
        and Counter(c["split"] for c in cases) == {"train": 42, "dev": 14},
        "Pilot split census differs.",
    )
    for split, expected in (("train", 3), ("dev", 1)):
        counts = Counter(
            (c["primary"]["action"], c["language"])
            for c in cases
            if c["split"] == split
        )
        require(
            len(counts) == 14 and set(counts.values()) == {expected},
            "Action/language balance differs.",
        )
    return cases, runtime_inputs, executions, controls


def load_tokenizers(registry_path=REGISTRY):
    from transformers import AutoTokenizer

    registry = json.loads(registry_path.read_text())
    selected, identities = {}, {}
    for name in TOKENIZERS:
        candidates = [c for c in registry["candidates"] if c["id"] == name]
        require(len(candidates) == 1, "Native tokenizer candidate must be unique.")
        candidate = candidates[0]
        path = (ROOT / candidate["tokenizer_path"]).resolve()
        require(path.is_relative_to(ROOT), "Tokenizer must be in the local repository.")
        for filename, expected in candidate["tokenizer_files_sha256"].items():
            require(
                sha256(path / filename) == expected,
                "Pinned native tokenizer bytes differ.",
            )
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        require(tokenizer.eos_token_id == EOS_IDS[name], "Native label EOS differs.")
        selected[name] = tokenizer
        identities[name] = {
            "path": str(path.relative_to(ROOT)),
            "files_sha256": candidate["tokenizer_files_sha256"],
            "eos_token_id": tokenizer.eos_token_id,
            "base_repository": candidate["base_repository"],
            "base_revision": candidate["base_revision"],
            "model_weights_loaded": False,
        }
    return selected, identities


def encode(case, tokenizer):
    prefix = tokenizer.apply_chat_template(
        case["messages"],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    completion = packed(case["primary"])
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    require(
        eos not in completion_ids,
        "Native EOS unexpectedly occurs inside the completion.",
    )
    require(
        tokenizer.decode(completion_ids, skip_special_tokens=False) == completion,
        "Completion text does not round-trip through the native tokenizer.",
    )
    require(
        len(prefix) <= 3584
        and len(completion_ids) + 1 <= 512
        and len(prefix) + len(completion_ids) + 1 <= 4096,
        "Native token budget exceeded.",
    )
    rendered = tokenizer.apply_chat_template(
        case["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {
        "id": case["id"],
        "split": case["split"],
        "messages": case["messages"],
        "rendered_prompt": rendered,
        "prefix_ids_sha256": digest(prefix),
        "primary": {
            "input_ids": prefix + completion_ids + [eos],
            "completion_start": len(prefix),
            "completion_text": completion,
            "completion_tokens": len(completion_ids) + 1,
            "eos_token_id": eos,
        },
    }


def source_files():
    pending = [
        Path(__file__),
        Path(contract.__file__),
        ROOT / "ml/test_prepare_passive_actions.py",
    ]
    seen = set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        require(path.is_file(), "A preparation source is missing.")
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module]
                    if isinstance(node, ast.ImportFrom) and node.module
                    else []
                )
            )
            for name in names:
                dependency = ROOT / "ml" / (name.split(".")[0] + ".py")
                if dependency.is_file() and dependency.resolve() not in seen:
                    pending.append(dependency)
    seen.update((contract.PROMPT_PATH.resolve(), REGISTRY.resolve()))
    return sorted(seen)


def prepare(output):
    require(not output.exists(), "Output must be a fresh attempt directory.")
    source_paths = source_files()
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    output.mkdir(parents=True)
    write(output / "recipe.json", recipe())
    tokenizer_map, identities = load_tokenizers()
    builder = contract.PassiveContextBuilder(
        tokenizer_map["gemma26"], budget_tokenizers=tokenizer_map
    )
    cases, runtime, executions, controls = build_cases(builder)
    for split in ("train", "dev"):
        jsonl(
            output / "cases" / f"{split}.jsonl",
            [c for c in cases if c["split"] == split],
        )
    jsonl(output / "runtime-inputs.jsonl", runtime)
    jsonl(output / "reference-executions.jsonl", executions)
    jsonl(output / "rejection-controls.jsonl", controls)
    budgets = {}
    for name, tokenizer in tokenizer_map.items():
        budgets[name] = {}
        for split in ("train", "dev"):
            rows = [encode(case, tokenizer) for case in cases if case["split"] == split]
            jsonl(
                output / "tokenized" / name / f"json-curriculum-{split}.jsonl.gz", rows
            )
            budgets[name][split] = {
                "rows": len(rows),
                "max_prefix_tokens": max(
                    r["primary"]["completion_start"] for r in rows
                ),
                "sequence_tokens": sum(len(r["primary"]["input_ids"]) for r in rows),
                "completion_tokens": sum(
                    r["primary"]["completion_tokens"] for r in rows
                ),
                "case_order": [r["id"] for r in rows],
            }
    write(output / "tokenizer-metadata.json", identities)
    census = {
        split: {
            language: dict(
                Counter(
                    c["primary"]["action"]
                    for c in cases
                    if c["split"] == split and c["language"] == language
                )
            )
            for language in ("en", "ko")
        }
        for split in ("train", "dev")
    }
    write(
        output / "census.json",
        {
            "counts": census,
            "budgets": budgets,
            "mechanically_rejected_controls": sum(len(c["probes"]) for c in controls),
            "executed_synthetic_history_rows": sum(bool(r["history"]) for r in runtime),
            "source_groups": recipe()[
                "source_groups_assigned_before_case_construction"
            ],
        },
    )
    for path in source_paths:
        relative = str(path.relative_to(ROOT))
        require(
            sha256(path) == sources[relative],
            "Preparation source changed during construction.",
        )
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
    paths = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {
        "version": VERSION,
        "contract_version": contract.VERSION,
        "status": "prepared_authored_pilot_pending_independent_semantic_review",
        "counts": {"train": 42, "dev": 14, "test": 0},
        "tokenizers": list(TOKENIZERS),
        "token_budgets": budgets,
        "sources_sha256": sources,
        "files_sha256": {str(p.relative_to(output)): sha256(p) for p in paths},
        "runtime_inputs": "runtime-inputs.jsonl",
        "reference_executions": "reference-executions.jsonl",
        "input_trust_boundary": "Only canonical system policy and observable snapshot; references and rationale stay out of messages.",
        "historical_rows_migrated": 0,
        "natural_audio_rows": 0,
        "training_updates_executed": 0,
        "model_generations_executed": 0,
        "final_test_bodies_read": 0,
        "acoustic_safety_claim": "none; all generated source intervals are symbolic",
    }
    write(output / "manifest.json", manifest)
    return manifest


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_bundle(directory, tokenizer_map=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest["version"] == VERSION
        and manifest["contract_version"] == contract.VERSION,
        "Unknown pilot preparation version.",
    )
    expected = set(manifest["files_sha256"]) | {"manifest.json"}
    actual = {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    }
    require(actual == expected, "Pilot file inventory differs.")
    for relative, expected_hash in manifest["files_sha256"].items():
        path = directory / relative
        require(
            not path.is_symlink()
            and path.resolve().is_relative_to(directory.resolve()),
            "Pilot paths must be local regular files.",
        )
        require(sha256(path) == expected_hash, "Pilot payload hash differs.")
    for relative, expected_hash in manifest["sources_sha256"].items():
        require(
            sha256(ROOT / relative) == expected_hash,
            "Live preparation dependency differs.",
        )
    require(
        json.loads((directory / "recipe.json").read_text()) == recipe(),
        "Assigned pilot recipe differs.",
    )
    if tokenizer_map is None:
        tokenizer_map, _ = load_tokenizers()
    builder = contract.PassiveContextBuilder(
        tokenizer_map["gemma26"], budget_tokenizers=tokenizer_map
    )
    cases, runtime, executions, controls = build_cases(builder)
    require(
        read_rows(directory / "runtime-inputs.jsonl") == runtime,
        "Runtime source documents/history differ from reproducible authored scenes.",
    )
    require(
        read_rows(directory / "reference-executions.jsonl") == executions,
        "Reference execution/replay differs.",
    )
    require(
        read_rows(directory / "rejection-controls.jsonl") == controls,
        "Invalid-proposal controls differ.",
    )
    for split in ("train", "dev"):
        selected = [case for case in cases if case["split"] == split]
        require(
            read_rows(directory / "cases" / f"{split}.jsonl") == selected,
            "Case/reference/snapshot differs from the assigned split.",
        )
        for name, tokenizer in tokenizer_map.items():
            token_path = (
                directory / "tokenized" / name / f"json-curriculum-{split}.jsonl.gz"
            )
            require(
                read_rows(token_path) == [encode(case, tokenizer) for case in selected],
                "Native message/prefix/completion encoding differs.",
            )
    return {
        "status": "passed_authored_pilot_readback_semantic_review_pending",
        "manifest_sha256": sha256(directory / "manifest.json"),
        "files_verified": len(manifest["files_sha256"]),
        "train_rows": 42,
        "dev_rows": 14,
        "reference_previews_verified": 48,
        "preservation_controls": 8,
        "invalid_controls_rejected": sum(len(c["probes"]) for c in controls),
        "native_tokenizer_rows_verified": len(cases) * len(tokenizer_map),
        "model_generations_executed": 0,
        "training_updates_executed": 0,
        "final_test_bodies_read": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_bundle(args.output.resolve()), indent=2))
        return
    manifest = prepare(args.output.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": sha256(args.output / "manifest.json"),
                "status": manifest["status"],
                "counts": manifest["counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
