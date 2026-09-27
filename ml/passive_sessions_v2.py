"""Replayable, explicitly synthetic feedback sessions for the passive contract.

This supplement imports the frozen contract and executor without modifying them.
Its authored timelines inherit existing source-family splits; semantic admission
remains a separate input-first review.
"""

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from passive_contract import (
    Envelope,
    PassiveContextBuilder,
    PassiveFeedbackLedger,
    PassivePredictionSession,
    apply_passive_edit,
    canonical_messages,
    committed_transition,
    parse_snapshot,
    wire_snapshot,
)
from passive_dataset import (
    RESERVED,
    ROOT,
    compact_transaction,
    digest,
    jsonl,
    read,
    write,
)
from replay_session import replay, session_hash
from session_cases import document_records, replayable_session
from verify_passive_dataset import tokenizers
from verify_release_export import Exporter, verify_export_delta, verify_operation

VERSION = "passive-sessions-v2"
PARENT_REGISTRY = ROOT / "data/manifests/passive-authored-sources-v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/datasets/nep-passive-completion-v2/sessions"


def correction(keys, text):
    return {"action": "correct_text", "targets": list(keys), "text": text}


def boundary(action, key):
    return {"action": action, "targets": [key]}


def proposal(name, focus, prediction, outcome, *, reason="unspecified", applied=None):
    return [
        {
            "kind": "propose",
            "proposal_id": name,
            "focus": focus,
            "prediction": prediction,
        },
        {"kind": "outcome", "proposal_id": name, "outcome": "shown"},
        {
            "kind": "outcome",
            "proposal_id": name,
            "outcome": outcome,
            "reason": reason,
            **({"applied": applied} if applied else {}),
        },
    ]


def query(focus, prediction, reason, *, apply=False, alternatives=(), probes=()):
    events = [
        {
            "kind": "query",
            "focus": focus,
            "prediction": prediction,
            "alternatives": list(alternatives),
            "reason": reason,
            "probes": list(probes),
        }
    ]
    if apply:
        if prediction["action"] == "no_action":
            raise ValueError("Abstention cannot become a transaction.")
        events.append({"kind": "commit", "focus": focus, "prediction": prediction})
    return events


def stop(focus, reason, *, probes=()):
    return query(focus, {"action": "no_action"}, reason, probes=probes)


def probe(prediction, suppressed):
    return {"prediction": prediction, "suppressed": suppressed}


def scenes():
    """Small complete scenes, not a cross-product of labels and translated text."""
    c, b = correction, boundary
    return [
        {
            "name": "needle-accepted-then-grouping",
            "parent": "authored-en-workshop-processing",
            "contrasts": ["H09"],
            "protocols": ["P01", "P06"],
            "paragraphs": [
                [("a", "Wait for the needel to settle.", 3.0)],
                [
                    ("b", "Only then turn the dial again and check the reading.", 4.0),
                    (
                        "c",
                        "The second demonstration shows why changing two controls together hides the cause.",
                        5.5,
                    ),
                ],
            ],
            "events": proposal(
                "needle", "a", c(["a"], "Wait for the needle to settle."), "accepted"
            )
            + query(
                "b",
                b("split_paragraph", "c"),
                "The accepted spelling repair is already applied. Two complete instructional units still occupy one 9.5-second caption; separate them without deleting the processing instruction.",
                apply=True,
            )
            + stop(
                "c",
                "The entire three-caption scene is visible. Spelling is repaired, independent instructions are separated, and no further text or boundary change is justified.",
            ),
        },
        {
            "name": "count-modified-retains-hedge",
            "parent": "authored-ko-lab-confidence",
            "contrasts": ["H03-text-analogue", "H09"],
            "protocols": ["P01", "P05"],
            "paragraphs": [
                [("a", "아마 세 번쯤 확닌했습니다.", 2.8)],
                [("b", "횟수는 아직 확정한 것이 아닙니다.", 3.2)],
                [("c", "기룍을 찾아보겠습니다.", 2.4)],
            ],
            "events": proposal(
                "count",
                "a",
                c(["a"], "세 번 확인했습니다."),
                "modified",
                reason="parameter",
                applied=c(["a"], "아마 세 번쯤 확인했습니다."),
            )
            + query(
                "a",
                c(["c"], "기록을 찾아보겠습니다."),
                "The actual modified acceptance retains 아마 and 쯤. Correct the separate visible 기룍 spelling error without copying the more certain proposed wording.",
                apply=True,
            )
            + stop(
                "b",
                "All three sentences are visible and correctly spelled. The approximate count and explicit lack of confirmation are intentional evidence, not defects.",
            ),
        },
        {
            "name": "catalogue-dismissal-is-local",
            "parent": "authored-en-operations-contrast",
            "contrasts": ["H01", "H02", "H10"],
            "protocols": ["P03", "P06"],
            "paragraphs": [
                [("a", "The catalogue uses the older spelling.", 3.3)],
                [("b", "Please send me the catalogue tomorrow.", 3.2)],
                [("c", "The shipping scheudle is on the next page.", 3.5)],
            ],
            "events": proposal(
                "catalogue",
                "b",
                c(["b"], "Please send me the catalog tomorrow."),
                "dismissed",
                reason="target",
            )
            + query(
                "b",
                c(["c"], "The shipping schedule is on the next page."),
                "Dismissal applies to the unchanged catalogue wording at its target. It does not suppress the unrelated clear scheudle error.",
                apply=True,
                probes=[probe(c(["b"], "Please send me the catalog tomorrow."), True)],
            )
            + stop(
                "b",
                "The remaining catalogue spelling is acceptable, explicitly retained locally, and the only clear spelling error has been repaired.",
                probes=[probe(c(["b"], "Please send me the catalog tomorrow."), True)],
            ),
        },
        {
            "name": "timing-dismissal-of-useful-spelling",
            "parent": "authored-en-operations-budget",
            "contrasts": ["H01", "H02", "H10"],
            "protocols": ["P03", "P06"],
            "paragraphs": [
                [("a", "The estimate is probbaly too low.", 3.2)],
                [("b", "We should repeat the trial before comitting.", 3.6)],
            ],
            "events": query(
                "a",
                c(["a"], "The estimate is probably too low."),
                "Before any dismissal, probbaly is an unambiguous visible spelling error. The independent comitting error is another useful option.",
                alternatives=[
                    c(["b"], "We should repeat the trial before committing.")
                ],
            )
            + proposal(
                "estimate",
                "a",
                c(["a"], "The estimate is probably too low."),
                "dismissed",
                reason="timing",
            )
            + query(
                "a",
                c(["b"], "We should repeat the trial before committing."),
                "The unchanged probably repair was explicitly dismissed for timing. Repair the different comitting target instead; a timing dismissal is not a global spelling preference.",
                apply=True,
                probes=[probe(c(["a"], "The estimate is probably too low."), True)],
            )
            + stop(
                "a",
                "The second typo was actually repaired. The only remaining defect is the exact unchanged target just dismissed for timing, so the current feedback gate suppresses it.",
                probes=[probe(c(["a"], "The estimate is probably too low."), True)],
            ),
        },
        {
            "name": "recollection-undo-restores-uncertainty",
            "parent": "authored-ko-recollection-approximation",
            "contrasts": ["H04", "H02"],
            "protocols": ["P05", "P06"],
            "paragraphs": [
                [("a", "서른 명쯤 왔던 것 같아요.", 2.8)],
                [("b", "정확한 명단은 없어요.", 2.3)],
                [("c", "제 기역만으로는 단정하기 어렵습니다.", 3.5)],
            ],
            "events": proposal("count", "a", c(["a"], "서른 명이 왔어요."), "accepted")
            + [{"kind": "undo", "proposal_id": "count"}]
            + query(
                "a",
                c(["c"], "제 기억만으로는 단정하기 어렵습니다."),
                "Undo actually restored the uncertain quantity. Preserve it and repair the independent 기역 typo.",
                apply=True,
                probes=[probe(c(["a"], "서른 명이 왔어요."), True)],
            )
            + stop(
                "c",
                "The complete recollection remains intentionally uncertain and has no remaining clear display or caption defect.",
            ),
        },
        {
            "name": "oral-history-expired-merge",
            "parent": "authored-en-oral-history-dramatic-gap",
            "contrasts": ["H05"],
            "protocols": ["P04", "P06"],
            "paragraphs": [
                [("a", "What I remember", 0.7)],
                [("b", "is the lights going out.", 2.4)],
                [("c", "The platfrom was quiet.", 2.6)],
            ],
            "events": proposal(
                "fragment", "b", b("merge_paragraph", "b"), "expired", reason="timing"
            )
            + query(
                "b",
                b("merge_paragraph", "b"),
                "Expiry is not a rejection. Join the subsecond incomplete subject to its grammatical continuation; the 3.1-second combined caption is short and coherent.",
                apply=True,
                alternatives=[c(["c"], "The platform was quiet.")],
                probes=[probe(b("merge_paragraph", "b"), False)],
            )
            + query(
                "b",
                c(["c"], "The platform was quiet."),
                "The fragment has been merged; the other visible spelling error remains.",
                apply=True,
            )
            + stop(
                "c",
                "The two complete captions are now clean. Preserve the remembered darkness and quiet rather than polishing the account.",
            ),
        },
        {
            "name": "museum-invalidated-merge",
            "parent": "authored-ko-museum-audience-time",
            "contrasts": ["H05", "H10"],
            "protocols": ["P04", "P06"],
            "paragraphs": [
                [("a", "그림 속 손은", 0.7)],
                [("b", "왼쪽을 가리킵니다.", 1.8)],
                [("c", "시선을 오른쪽 인물로 옴겨 보세요.", 3.0)],
            ],
            "events": [
                {
                    "kind": "propose",
                    "proposal_id": "hand",
                    "focus": "b",
                    "prediction": b("merge_paragraph", "b"),
                },
                {"kind": "outcome", "proposal_id": "hand", "outcome": "shown"},
                {"kind": "cursor", "focus": "c"},
                {
                    "kind": "outcome",
                    "proposal_id": "hand",
                    "outcome": "invalidated",
                    "reason": "timing",
                },
                {"kind": "cursor", "focus": "b"},
            ]
            + query(
                "b",
                b("merge_paragraph", "b"),
                "The prior preview was invalidated by cursor movement, not rejected. The 0.7-second subject fragment belongs with its predicate in a 2.5-second caption.",
                apply=True,
                alternatives=[c(["c"], "시선을 오른쪽 인물로 옮겨 보세요.")],
                probes=[probe(b("merge_paragraph", "b"), False)],
            )
            + query(
                "c",
                c(["c"], "시선을 오른쪽 인물로 옮겨 보세요."),
                "Caption grouping is repaired; the visible 옴겨 spelling error is still actionable.",
                apply=True,
            )
            + stop(
                "c",
                "All displayed wording and caption boundaries are adequate after the two distinct repairs.",
            ),
        },
        {
            "name": "connector-modified-not-confident",
            "parent": "authored-en-workshop-uncertainty",
            "contrasts": ["H03-text-analogue", "H09"],
            "protocols": ["P01", "P05"],
            "paragraphs": [
                [("a", "It probaly needs another try.", 2.8)],
                [
                    (
                        "b",
                        "Check the label before choosing a replacement connector.",
                        4.2,
                    ),
                    (
                        "c",
                        "The next test compares both versions while keeping the power switched off.",
                        5.0,
                    ),
                ],
            ],
            "events": proposal(
                "certainty",
                "a",
                c(["a"], "It needs another try."),
                "modified",
                reason="parameter",
                applied=c(["a"], "It probably needs another try."),
            )
            + query(
                "b",
                b("split_paragraph", "c"),
                "Actual acceptance kept probably. The remaining useful edit separates two complete instructions in a long 9.2-second caption.",
                apply=True,
            )
            + stop(
                "a",
                "The complete scene is correctly spelled and grouped. A modified acceptance that retains probably is not permission to strengthen the claim.",
            ),
        },
        {
            "name": "lab-expired-reassurance",
            "parent": "authored-ko-lab-processing",
            "contrasts": ["H05"],
            "protocols": ["P01"],
            "paragraphs": [
                [("a", "분모가 영에 가까워지면 어떻게 될까요?", 3.7)],
                [("b", "바로 답하지 안아도 됩니다.", 2.8)],
                [("c", "그래프의 양쪽을 비교해 보세요.", 2.9)],
            ],
            "events": proposal(
                "reassurance",
                "b",
                c(["b"], "바로 답하지 않아도 됩니다."),
                "expired",
                reason="timing",
            )
            + query(
                "b",
                c(["b"], "바로 답하지 않아도 됩니다."),
                "The expired suggestion remains an unambiguous 안아도 to 않아도 display correction. Keep the invitation to take processing time.",
                apply=True,
                probes=[probe(c(["b"], "바로 답하지 않아도 됩니다."), False)],
            )
            + stop(
                "b",
                "The complete visible explanation is now clean; its question and reassurance are intentional content.",
            ),
        },
        {
            "name": "oral-history-undone-boundary",
            "parent": "authored-en-oral-history-dramatic-gap",
            "contrasts": ["H04", "H02"],
            "protocols": ["P05", "P06"],
            "paragraphs": [
                [
                    ("a", "Nobody said a word.", 2.3),
                    ("b", "I still remember that silence.", 2.5),
                ],
                [("c", "The station clock had stoped.", 2.8)],
            ],
            "events": proposal("silence", "b", b("split_paragraph", "b"), "accepted")
            + [{"kind": "undo", "proposal_id": "silence"}]
            + query(
                "b",
                c(["c"], "The station clock had stopped."),
                "The actual boundary undo restores a reasonable 4.8-second caption. Do not repeat that split; correct the independent stoped error.",
                apply=True,
                probes=[probe(b("split_paragraph", "b"), True)],
            )
            + stop(
                "a",
                "The restored two-sentence caption is short enough and was explicitly kept together; the remaining sentence is now correctly spelled.",
                probes=[probe(b("split_paragraph", "b"), True)],
            ),
        },
        {
            "name": "museum-dismissed-formality",
            "parent": "authored-ko-museum-provenance",
            "contrasts": ["H01", "H02", "H10"],
            "protocols": ["P03", "P04"],
            "paragraphs": [
                [("a", "이 작품은 옛날 방식으로 만들었어요.", 3.6)],
                [("b", "다만 정확한 제작 연대는 확닌되지 않았습니다.", 4.2)],
            ],
            "events": proposal(
                "formality",
                "a",
                c(["a"], "이 작품은 전통적인 방식으로 제작되었습니다."),
                "dismissed",
                reason="preference",
            )
            + query(
                "a",
                c(["b"], "다만 정확한 제작 연대는 확인되지 않았습니다."),
                "The formality rewrite was dismissed and the original spoken register is acceptable. Correct only 확닌 in the other caption.",
                apply=True,
                probes=[
                    probe(c(["a"], "이 작품은 전통적인 방식으로 제작되었습니다."), True)
                ],
            )
            + stop(
                "a",
                "The two-caption account is clean; mixed conversational and formal endings do not themselves justify a rewrite.",
            ),
        },
        {
            "name": "changed-local-evidence-sign",
            "parent": "authored-en-operations-budget",
            "contrasts": ["H01", "H06", "H10"],
            "protocols": ["P05"],
            "paragraphs": [
                [
                    ("a", "Check the sine.", 1.7),
                    ("b", "The note says sine or sign; I cannot read it.", 4.0),
                ],
            ],
            "events": proposal(
                "sign", "a", c(["a"], "Check the sign."), "dismissed", reason="target"
            )
            + stop(
                "a",
                "The visible note expressly leaves sine versus sign unresolved, and the same correction was dismissed. Do not guess the term.",
                probes=[probe(c(["a"], "Check the sign."), True)],
            )
            + [
                {
                    "kind": "commit",
                    "focus": "b",
                    "prediction": c(
                        ["b"],
                        "The sign is plus or minus, not a trigonometric function.",
                    ),
                }
            ]
            + query(
                "a",
                c(["a"], "Check the sign."),
                "An actual later transcript correction makes the local meaning explicit: plus or minus, not trigonometry. The former target evidence changed; reconsider the same term from the new visible evidence.",
                apply=True,
                probes=[probe(c(["a"], "Check the sign."), False)],
            )
            + stop(
                "a",
                "The complete short explanation consistently identifies a mathematical sign and has no remaining clear defect.",
            ),
        },
        {
            "name": "mixed-ending-history-not-global",
            "parent": "authored-ko-recollection-self-repair",
            "contrasts": ["H08", "H09"],
            "protocols": ["P04", "P05"],
            "paragraphs": [
                [("a", "처음에는 부산이라고 생각했어요.", 3.0)],
                [("b", "다시 보니 대구였습니다.", 2.5)],
                [("c", "기억이 꼭 정확한 건 아니에요.", 3.0)],
            ],
            "events": proposal(
                "formal", "a", c(["a"], "처음에는 부산이라고 생각했습니다."), "accepted"
            )
            + proposal("informal", "b", c(["b"], "다시 보니 대구였어요."), "accepted")
            + stop(
                "c",
                "The two actual past edits choose different registers. They do not establish a uniform style rule; the full current recollection is grammatical and its uncertainty and self-correction must remain.",
            ),
        },
        {
            "name": "oral-history-word-fragments",
            "parent": "authored-en-oral-history-uncertain-date",
            "contrasts": ["H05"],
            "protocols": ["P04", "P06"],
            "content_scenarios": ["R07"],
            "paragraphs": [
                [("a", "I remem", 0.9), ("b", "ber the lights going out.", 2.2)],
                [
                    (
                        "c",
                        "The last line in my diary is unfinished: 'We went to the...'.",
                        4.6,
                    )
                ],
            ],
            "events": proposal(
                "memory",
                "a",
                c(["a", "b"], "I remember the lights going out."),
                "invalidated",
                reason="timing",
            )
            + query(
                "a",
                c(["a", "b"], "I remember the lights going out."),
                "Adjacent remem and ber form an unmistakable broken word in a continuous symbolic interval. Correct the complete two-item display span while preserving the explicitly unfinished diary quotation.",
                apply=True,
                probes=[
                    probe(c(["a", "b"], "I remember the lights going out."), False)
                ],
            )
            + stop(
                "c",
                "The recoverable split word is repaired. The quoted diary ends mid-phrase, and no visible evidence supplies the destination; completing it would invent content.",
            ),
        },
        {
            "name": "museum-word-fragments",
            "parent": "authored-ko-museum-provenance",
            "contrasts": ["H09"],
            "protocols": ["P04", "P06"],
            "content_scenarios": ["R07"],
            "paragraphs": [
                [("a", "제작 연", 0.9), ("b", "대는 아직 확인되지 않았습니다.", 2.9)],
                [("c", "서명은 '김…'까지만 남아 있습니다.", 3.1)],
                [("d", "안내문에도 추졍이라고 적혀 있습니다.", 3.2)],
            ],
            "events": query(
                "a",
                c(["a", "b"], "제작 연대는 아직 확인되지 않았습니다."),
                "The adjacent fragments 연 and 대는 clearly form 제작 연대 in this sentence. Restore only that visible word boundary; neither a full signature nor a precise date is known.",
                apply=True,
                alternatives=[c(["d"], "안내문에도 추정이라고 적혀 있습니다.")],
            )
            + query(
                "d",
                c(["d"], "안내문에도 추정이라고 적혀 있습니다."),
                "The date wording is repaired; the independent 추졍 typo remains, while the incomplete signature remains unresolvable.",
                apply=True,
            )
            + stop(
                "c",
                "All visible recoverable display defects are repaired. The incomplete signature and unconfirmed date are explicitly uncertain source content.",
            ),
        },
        {
            "name": "workshop-adjacent-display-errors",
            "parent": "authored-en-workshop-uncertainty",
            "contrasts": [],
            "protocols": ["P04", "P06"],
            "content_scenarios": ["R07"],
            "paragraphs": [
                [
                    ("a", "Hold the conector", 1.8),
                    ("b", "in its orginal position.", 2.4),
                ],
                [
                    (
                        "c",
                        "The label is smudged; I cannot tell whether it says one or seven.",
                        4.8,
                    )
                ],
            ],
            "events": query(
                "a",
                c(["a", "b"], "Hold the connector in its original position."),
                "Two adjacent independently misspelled display items form one short command and continuous source interval. Repairing the complete span is useful; fixing either error separately is also valid. The smudged number remains unknowable.",
                apply=True,
                alternatives=[
                    c(["a"], "Hold the connector"),
                    c(["b"], "in its original position."),
                ],
            )
            + stop(
                "c",
                "The complete command is correctly spelled after the multi-item transaction. The separate smudged-label account explicitly gives two possibilities without evidence to choose either.",
            ),
        },
        {
            "name": "lab-adjacent-display-errors",
            "parent": "authored-ko-lab-number",
            "contrasts": [],
            "protocols": ["P01", "P06"],
            "content_scenarios": ["R07"],
            "paragraphs": [
                [("a", "먼저 물의 온도르", 1.5), ("b", "정학하게 기록하세요.", 2.1)],
                [("c", "기록지의 마지막 숫자는 번져서 읽을 수 없습니다.", 4.1)],
            ],
            "events": query(
                "a",
                c(["a", "b"], "먼저 물의 온도를 정확하게 기록하세요."),
                "The object particle and adverb each have a clear visible spelling error across two adjacent items. One span repair preserves the entire short instruction; either smaller correction is also useful. Do not invent the unreadable recorded number.",
                apply=True,
                alternatives=[
                    c(["a"], "먼저 물의 온도를"),
                    c(["b"], "정확하게 기록하세요."),
                ],
            )
            + stop(
                "c",
                "Both recoverable display errors have been repaired. The complete visible scene supplies no value for the blurred final digit, so retaining that uncertainty is the appropriate stop.",
            ),
        },
    ]


def make_document(scene, source_id, language):
    content, keys, elapsed = [], {}, 0.0

    def identity(key):
        return str(uuid5(NAMESPACE_URL, f"{VERSION}/{source_id}/{key}"))

    for n, paragraph in enumerate(scene["paragraphs"]):
        content.append(
            {
                "type": "paragraph_start",
                "uuid": identity(f"p{n}"),
                "speaker": "narrator",
                "language": language,
            }
        )
        for key, text, seconds in paragraph:
            if key in keys or not text or seconds <= 0:
                raise ValueError(
                    "Authored items need unique keys, text and positive timing."
                )
            keys[key] = identity(key)
            content.append(
                {
                    "type": "text",
                    "uuid": keys[key],
                    "source": "symbolic:" + source_id,
                    "sourceStart": elapsed,
                    "length": seconds,
                    "text": text,
                    "conf": 1,
                }
            )
            elapsed += seconds
        content.append({"type": "paragraph_end", "uuid": identity(f"e{n}")})
    document = {
        "content": content,
        "metadata": {"provenance": "authored_symbolic_timing_not_audio"},
    }
    document_records(document)
    return document, keys, elapsed


def resolve_action(value, keys, bundle):
    value = copy.deepcopy(value)
    if "targets" in value:
        by_uuid = {
            item["uuid"]: int(alias) for alias, item in bundle["mapping"].items()
        }
        value["target"] = [by_uuid[keys[k]] for k in value.pop("targets")]
    return value


def build_session(spec, source, builder, exporter=None):
    """Return standard cases, exact replay evidence and mechanical reference checks."""
    document = copy.deepcopy(source["document"])
    initial, keys = copy.deepcopy(document), spec["item_keys"]
    session = PassivePredictionSession(spec["id"], builder)
    history, cases, checkpoints, checks, proposals, event_checks = (
        [],
        [],
        [],
        [],
        {},
        [],
    )
    journal = [
        {
            "kind": "start",
            "seq": 0,
            "schema_version": 2,
            "hash_format": "typed-json-f64-v1",
            "actor": "synthetic",
            "collection_mode": "synthetic",
            "generator": VERSION,
            "document": initial,
            "document_hash": session_hash(initial),
            "sources": [source["source_group_id"]],
        }
    ]

    def prepare(focus, sequence):
        return session.prepare(document, keys[focus], as_of=sequence, history=history)

    def audit_action(bundle, prediction, label):
        after = apply_passive_edit(document, bundle, prediction)
        operation = verify_operation(document, after, bundle["mapping"], prediction)
        undo = replayable_session(
            document, after, prediction["action"], [source["source_group_id"]]
        )
        if replay(undo)[0] != after:
            raise ValueError("Reference undo/redo did not restore the full state.")
        evidence = {"label": label, "operation": operation, "undo_redo_journal": undo}
        if exporter is not None:
            profile = bundle["input"]["caption_profile"]["export_line_length"]
            evidence["export"] = verify_export_delta(
                exporter.export(document, profile),
                exporter.export(after, profile),
                operation,
                prediction["action"],
            )
        return after, evidence

    def commit(prediction, bundle, sequence):
        nonlocal document
        after, evidence = audit_action(bundle, prediction, f"event-{sequence}")
        transition = committed_transition(
            document,
            after,
            prediction,
            sequence,
            transaction_id=f"{spec['id']}:transaction:{sequence}",
        )
        history.append(transition)
        journal.append(
            compact_transaction(
                document,
                after,
                len(journal),
                len(history),
                prediction["action"],
                session_sequence=sequence,
                transaction_id=transition["transaction_id"],
            )
        )
        document = after
        event_checks.append(evidence)
        return transition

    for sequence, event in enumerate(spec["events"], 1):
        kind = event["kind"]
        if kind == "propose":
            bundle = prepare(event["focus"], sequence - 1)
            prediction = resolve_action(event["prediction"], keys, bundle)
            session.feedback.propose(
                event["proposal_id"], sequence, document, bundle, prediction
            )
            proposals[event["proposal_id"]] = {
                "prediction": prediction,
                "bundle": bundle,
            }
        elif kind == "outcome":
            outcome, kwargs = (
                event["outcome"],
                {"reason": event.get("reason", "unspecified")},
            )
            if outcome in {"accepted", "modified"}:
                proposed = proposals[event["proposal_id"]]
                prediction = (
                    resolve_action(event["applied"], keys, proposed["bundle"])
                    if "applied" in event
                    else proposed["prediction"]
                )
                transition = commit(prediction, proposed["bundle"], sequence)
                proposed["transition"] = transition
                kwargs["transition"] = transition
            session.feedback.record(event["proposal_id"], outcome, sequence, **kwargs)
        elif kind == "undo":
            original = proposals[event["proposal_id"]]["transition"]
            if document != original["after"]:
                raise ValueError(
                    "Authored undo must reverse the current latest transaction."
                )
            restored = copy.deepcopy(original["before"])
            transition = committed_transition(
                document,
                restored,
                original["prediction"],
                sequence,
                transaction_id=f"{spec['id']}:undo:{sequence}",
                undo_of=original["transaction_id"],
            )
            history.append(transition)
            journal.append(
                compact_transaction(
                    document,
                    restored,
                    len(journal),
                    len(history),
                    original["prediction"]["action"],
                    origin="undo",
                    session_sequence=sequence,
                    undo_of=original["transaction_id"],
                )
            )
            session.feedback.record(
                event["proposal_id"], "undone", sequence, transition=transition
            )
            document = restored
        elif kind == "commit":
            bundle = prepare(event["focus"], sequence)
            commit(resolve_action(event["prediction"], keys, bundle), bundle, sequence)
        elif kind == "cursor":
            if keys[event["focus"]] not in {i["uuid"] for i in document["content"]}:
                raise ValueError("Cursor move targets a removed item.")
            journal.append(
                {
                    "kind": "context",
                    "seq": len(journal),
                    "session_sequence": sequence,
                    "focus_uuid": keys[event["focus"]],
                    "reason": "explicit_synthetic_cursor_move",
                }
            )
        elif kind == "query":
            bundle = prepare(event["focus"], sequence)
            if bundle["context"]["visible_items"] != bundle["context"]["total_items"]:
                raise ValueError(
                    "This bounded session stratum requires the entire authored window."
                )
            actions = [
                resolve_action(p, keys, bundle)
                for p in [event["prediction"], *event.get("alternatives", [])]
            ]
            if len({session_hash(p) for p in actions}) != len(actions):
                raise ValueError("Duplicate primary/alternative reference.")
            index = len(cases)
            identifier = f"{spec['id']}-s{index:02d}"
            outputs, references = [], []
            request = Envelope(
                document_id=spec["id"],
                request_id=identifier,
                revision=len(history),
                interaction_epoch=sequence,
            )
            for ref_index, prediction in enumerate(actions):
                runtime = session.validate(
                    json.dumps(prediction, ensure_ascii=False),
                    document,
                    bundle,
                    request,
                    request,
                    "stop",
                    as_of=sequence,
                )
                if runtime["suppressed"] or runtime["show"] != (
                    prediction["action"] != "no_action"
                ):
                    raise ValueError(
                        "A reference contradicts the actual feedback display gate."
                    )
                after, evidence = audit_action(
                    bundle, prediction, f"{identifier}:reference:{ref_index}"
                )
                outputs.append(after)
                references.append(evidence)
                checks.append(
                    {
                        "case_id": identifier,
                        "reference_index": ref_index,
                        "status": "passed",
                        "runtime": runtime,
                        **evidence.get("export", {}),
                    }
                )
            probe_results = []
            for check in event.get("probes", []):
                prediction = resolve_action(check["prediction"], keys, bundle)
                result = session.validate(
                    json.dumps(prediction, ensure_ascii=False),
                    document,
                    bundle,
                    request,
                    request,
                    "stop",
                    as_of=sequence,
                )
                if result["suppressed"] != check["suppressed"]:
                    raise ValueError(
                        "The actual ledger disagrees with a suppression control."
                    )
                probe_results.append({"prediction": prediction, **result})
            case = {
                "id": identifier,
                "input": bundle["input"],
                "expectedOutput": actions[0],
                "acceptableOutputs": actions[1:],
                "metadata": {
                    "split": source["split"],
                    "source_group_id": source["source_group_id"],
                    "source_id": source["id"],
                    "language": source["language"],
                    "episode_id": spec["id"],
                    "episode_step": index,
                    "source_region": spec["source_region"],
                    "source_base": source["source_base"],
                    "actor": "synthetic",
                    "history_origin": "executed_reconstruction",
                    "feedback_origin": "synthetic",
                    "kind": "authored_feedback_session",
                    "media_kind": "authored_symbolic_timeline",
                    "content_origin": "new_authored_derivative_of_registered_parent_not_natural_asr",
                    "decision_rationale": event["reason"],
                    "feedback_contrasts": spec["contrasts"],
                    "session_protocols": spec["protocols"],
                    "content_scenarios": spec.get("content_scenarios", []),
                    "context": bundle["context"],
                    "as_of": sequence,
                    "input_sha256": session_hash(canonical_messages(bundle["input"])),
                    "document_before_hash": session_hash(document),
                    "document_after_hash": session_hash(outputs[0]),
                    "training_eligible": False,
                    "review_status": "pending_independent_admission",
                },
                "reproduction": {
                    "episode": f"episodes/{spec['id']}.json",
                    "step": index,
                    "session_event_sequence": sequence,
                    "checkpoint": index,
                },
            }
            cases.append(case)
            checkpoints.append(
                {
                    "case_id": identifier,
                    "as_of": sequence,
                    "focus_uuid": keys[event["focus"]],
                    "before": copy.deepcopy(document),
                    "after_primary": outputs[0],
                    "bundle": bundle,
                    "committed_history_count": len(history),
                    "references": references,
                    "suppression_probes": probe_results,
                }
            )
            journal.append(
                {
                    "kind": "opportunity",
                    "seq": len(journal),
                    "revision": len(history),
                    "session_sequence": sequence,
                    "case_id": identifier,
                    "document_hash": session_hash(document),
                }
            )
        else:
            raise ValueError("Unknown authored session event.")
    journal.append(
        {
            "kind": "end",
            "seq": len(journal),
            "revision": len(history),
            "document_hash": session_hash(document),
            "transactions": len(history),
        }
    )
    replayed, summary = replay(journal)
    if replayed != document or not summary["complete"]:
        raise ValueError("Whole-session journal failed replay.")
    ledger = PassiveFeedbackLedger.restore(session.feedback.dump())
    for checkpoint, case in zip(checkpoints, cases, strict=True):
        rebuilt = builder.build(
            checkpoint["before"],
            checkpoint["focus_uuid"],
            as_of=checkpoint["as_of"],
            history=history,
            feedback=ledger,
        )
        if canonical_messages(rebuilt["input"]) != canonical_messages(case["input"]):
            raise ValueError("Future ledger/history leaked into an earlier prediction.")
        if parse_snapshot(wire_snapshot(case["input"])).model_dump() != case["input"]:
            raise ValueError("Session compact wire roundtrip was not lossless.")
    artifact = {
        "version": VERSION,
        "spec": spec,
        "source_document": source["document_path"],
        "source_document_sha256": digest(ROOT / source["document_path"]),
        "initial_document": initial,
        "final_document": document,
        "committed_transitions": history,
        "feedback_ledger": session.feedback.dump(),
        "checkpoints": checkpoints,
        "journal": journal,
        "event_operation_checks": event_checks,
        "replay_summary": summary,
        "actor": "synthetic",
        "audio_listening": False,
    }
    return cases, artifact, checks


def register_sources(output):
    parents = {row["id"]: row for row in read(PARENT_REGISTRY)["sources"]}
    authored = scenes()
    registration = {
        "version": VERSION,
        "status": "preregistered_before_session_generation",
        "parent_registry": str(PARENT_REGISTRY.relative_to(ROOT)),
        "parent_registry_sha256": digest(PARENT_REGISTRY),
        "sources": [],
    }
    for scene in authored:
        parent = parents[scene["parent"]]
        if parent["source_group_id"] in RESERVED or parent["split"] not in {
            "train",
            "dev",
        }:
            raise ValueError(
                "Reserved or unassigned parents cannot enter this stratum."
            )
        if digest(ROOT / parent["document_path"]) != parent["document_sha256"]:
            raise ValueError("Registered authored parent changed.")
        registration["sources"].append(
            {
                "id": f"sessions-v2-{scene['name']}",
                "parent": scene["parent"],
                "source_group_id": parent["source_group_id"],
                "split": parent["split"],
                "language": parent["language"],
            }
        )
    write(output / "registration.json", registration)
    sources, specs = [], []
    for scene, allocation in zip(authored, registration["sources"], strict=True):
        parent = parents[scene["parent"]]
        document, keys, duration = make_document(
            scene, allocation["id"], parent["language"]
        )
        path = output / "sources" / allocation["id"] / "document.json"
        write(path, document)
        source = {
            "id": allocation["id"],
            "source_group_id": parent["source_group_id"],
            "split": parent["split"],
            "language": parent["language"],
            "document_path": str(path.relative_to(ROOT)),
            "document_sha256": digest(path),
            "media_path": None,
            "media_kind": "authored_symbolic_timeline",
            "license": "Project-authored text",
            "source_base": {
                "source_id": parent["id"],
                "document_path": parent["document_path"],
                "document_sha256": parent["document_sha256"],
                "adaptation": "New authored continuation scene and explicitly symbolic item timing; not a verbatim excerpt or recorded session.",
            },
        }
        sources.append(source)
        specs.append(
            {
                "id": allocation["id"],
                "source_id": allocation["id"],
                "kind": "authored_feedback_session",
                "source_region": [0, duration],
                "item_keys": keys,
                "events": scene["events"],
                "contrasts": scene["contrasts"],
                "protocols": scene["protocols"],
                "content_scenarios": scene.get("content_scenarios", []),
            }
        )
    write(output / "source-registry.json", {"version": VERSION, "sources": sources})
    write(output / "recipe.json", {"version": VERSION, "episodes": specs})
    return sources, specs


def build(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT):
        raise ValueError("Use a new output directory inside the project.")
    output.mkdir(parents=True)
    sources, specs = register_sources(output)
    loaded = tokenizers()
    builder = PassiveContextBuilder(loaded["qwen06"], budget_tokenizers=loaded)
    exporter, cases = Exporter(), []
    try:
        for source, spec in zip(sources, specs, strict=True):
            source = {**source, "document": read(ROOT / source["document_path"])}
            batch, artifact, _ = build_session(spec, source, builder, exporter)
            cases.extend(batch)
            write(output / "episodes" / f"{spec['id']}.json", artifact)
    finally:
        exporter.close()
    for split in ("train", "dev"):
        jsonl(
            output / f"{split}-candidates.jsonl",
            [c for c in cases if c["metadata"]["split"] == split],
        )
    jsonl(output / "exclusions.jsonl", [])
    census = {
        "status": "candidates_only_not_a_release",
        "version": VERSION,
        "rows": len(cases),
        "episodes": len(specs),
        "splits": dict(Counter(c["metadata"]["split"] for c in cases)),
        "languages": dict(Counter(c["metadata"]["language"] for c in cases)),
        "actions": dict(Counter(c["expectedOutput"]["action"] for c in cases)),
        "feedback_context_rows": sum(
            bool(c["input"]["recent_feedback"]) for c in cases
        ),
        "observed_feedback_outcomes": dict(
            Counter(f["outcome"] for c in cases for f in c["input"]["recent_feedback"])
        ),
        "multi_item_primary_corrections": sum(
            c["expectedOutput"]["action"] == "correct_text"
            and len(c["expectedOutput"]["target"]) > 1
            for c in cases
        ),
        "dev_merge_rows": sum(
            c["metadata"]["split"] == "dev"
            and c["expectedOutput"]["action"] == "merge_paragraph"
            for c in cases
        ),
        "all_windows_complete": True,
        "test_rows": 0,
        "training_eligible": 0,
        "audio_listening": False,
        "limitations": [
            "All content, timing, proposal outcomes and interaction histories are explicitly synthetic.",
            "H03 pause-parameter modification and destructive portions of P02/P03/P04 remain gated; text modification is an analogue only.",
            "H07 session isolation and future-event exclusion are mechanical regressions, not duplicate training rows.",
            "Partial protocols are not full seven-action session coverage; live editor/server integration is not certified.",
        ],
    }
    write(output / "census.json", census)
    return census


def verify_sessions(directory, output=None):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError("Session package must be inside this project.")
    registered = read(directory / "source-registry.json")["sources"]
    sources = {s["id"]: s for s in registered}
    parents = {s["id"]: s for s in read(PARENT_REGISTRY)["sources"]}
    if len(sources) != len(registered):
        raise ValueError("Duplicate registered source identity.")
    specs = read(directory / "recipe.json")["episodes"]
    cases = []
    for split in ("train", "dev"):
        batch = [
            json.loads(line)
            for line in (directory / f"{split}-candidates.jsonl")
            .read_text()
            .splitlines()
            if line
        ]
        if any(case["metadata"]["split"] != split for case in batch):
            raise ValueError("Candidate placement contradicts its registered split.")
        cases.extend(batch)
    if len({c["id"] for c in cases}) != len(cases) or len(
        {s["id"] for s in specs}
    ) != len(specs):
        raise ValueError("Duplicate session/candidate identity.")
    if set(sources) != {s["source_id"] for s in specs}:
        raise ValueError("Source registry and recipe differ.")
    dependencies = {}
    loaded = tokenizers(dependencies)
    builder = PassiveContextBuilder(loaded["qwen06"], budget_tokenizers=loaded)
    exporter, checks, rebuilt_cases, errors = Exporter(), [], [], []
    try:
        for spec in specs:
            source = sources[spec["source_id"]]
            parent = parents[source["source_base"]["source_id"]]
            if any(
                source[key] != parent[key]
                for key in ("source_group_id", "split", "language")
            ) or any(
                source["source_base"][key] != parent[key]
                for key in ("document_path", "document_sha256")
            ):
                raise ValueError(
                    "Derived source changed its parent's allocation or lineage."
                )
            if (
                source["split"] not in {"train", "dev"}
                or source["source_group_id"] in RESERVED
            ):
                raise ValueError("Reserved or invalid source in session registry.")
            for value in (source, source["source_base"]):
                path = ROOT / value["document_path"]
                if digest(path) != value["document_sha256"]:
                    raise ValueError("Session or parent source document changed.")
                dependencies[str(path.relative_to(ROOT))] = digest(path)
            source = {**source, "document": read(ROOT / source["document_path"])}
            batch, artifact, outcomes = build_session(spec, source, builder, exporter)
            saved = read(directory / "episodes" / f"{spec['id']}.json")
            if artifact != saved:
                raise ValueError(
                    "Saved session evidence differs from actual reconstruction."
                )
            rebuilt_cases.extend(batch)
            checks.extend(outcomes)
    finally:
        exporter.close()
    actual = {c["id"]: c for c in cases}
    rebuilt = {c["id"]: c for c in rebuilt_cases}
    if actual != rebuilt:
        raise ValueError(
            "Saved candidate differs from reconstructed serving input/reference."
        )
    groups, signatures = {}, set()
    for case in cases:
        meta = case["metadata"]
        if groups.setdefault(meta["source_group_id"], meta["split"]) != meta["split"]:
            raise ValueError("An authored family crosses splits.")
        signature = session_hash(canonical_messages(case["input"]))
        if signature != meta["input_sha256"] or signature in signatures:
            raise ValueError("Stale or duplicate session predictor input.")
        signatures.add(signature)
    expected = {
        (c["id"], n) for c in cases for n in range(1 + len(c["acceptableOutputs"]))
    }
    if {(r["case_id"], r["reference_index"]) for r in checks} != expected:
        raise ValueError("Reference/export audit is incomplete.")
    audit_path = Path(output).resolve() if output else directory / "verification.json"
    for path in directory.rglob("*"):
        if path.is_file() and path.resolve() != audit_path:
            dependencies[str(path.relative_to(ROOT))] = digest(path)
    dependencies[str(PARENT_REGISTRY.relative_to(ROOT))] = digest(PARENT_REGISTRY)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.is_file():
                dependencies[str(path.relative_to(ROOT))] = digest(path)
    for name in (
        "ml/prompt-passive-v1.txt",
        "ml/export_release_states.cjs",
        "app/src/core/webvtt.ts",
        "ml/test_passive_sessions_v2.py",
    ):
        dependencies[name] = digest(ROOT / name)
    for name, expected_hash in dependencies.items():
        if digest(ROOT / name) != expected_hash:
            raise ValueError("Verification dependency changed during the audit.")
    result = {
        "status": "passed",
        "version": VERSION,
        "directory": str(directory),
        "errors": errors,
        "candidate_hashes": {
            n: digest(directory / n)
            for n in ("train-candidates.jsonl", "dev-candidates.jsonl", "recipe.json")
        },
        "dependencies_sha256": dependencies,
        "case_ids": [c["id"] for c in cases],
        "candidates": len(cases),
        "actual_serving_message_parity_checks": len(cases),
        "full_ledger_as_of_parity_checks": len(cases),
        "replayed_episodes": len(specs),
        "accepted_outcomes_exported": len(checks),
        "outcomes": checks,
        "semantic_admission": "separate_independent_review_required",
        "audio_listening": False,
    }
    write(audit_path, result)
    return {
        k: v
        for k, v in result.items()
        if k not in {"dependencies_sha256", "case_ids", "outcomes"}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = (
        build(args.directory)
        if args.mode == "build"
        else verify_sessions(args.directory, args.output)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
