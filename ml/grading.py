"""Grade exact edits and equivalent text corrections without trusting model judgments."""

import json
import unicodedata

from replay_session import session_hash
from session_cases import apply_edit


def transcript_media(document):
    paragraphs = []
    for item in document["content"]:
        if item["type"] == "paragraph_start":
            current = {
                "speaker": item.get("speaker"),
                "language": item.get("language"),
                "text": [],
                "media": [],
            }
            paragraphs.append(current)
        elif item["type"] in ("text", "non_text", "artificial_silence"):
            if item["type"] == "text":
                current["text"].append(item["text"])
            segment = [item.get("source"), item.get("sourceStart"), item["length"]]
            media = current["media"]
            if (
                media
                and segment[0] is not None
                and media[-1][0] == segment[0]
                and abs(media[-1][1] + media[-1][2] - segment[1]) < 1e-8
            ):
                media[-1][2] += segment[2]
            else:
                media.append(segment)
    for paragraph in paragraphs:
        paragraph["text"] = unicodedata.normalize("NFC", " ".join(paragraph["text"]))
        paragraph["media"] = [
            [s, round(start, 8) if start is not None else None, round(length, 8)]
            for s, start, length in paragraph["media"]
        ]
    return {"paragraphs": paragraphs, "metadata": document["metadata"]}


def grade(case, prediction, execution=None):
    expected = case.get("expectedOutput")
    if expected is None:
        return {
            "exact_match": None,
            "outcome_match": None,
            "method": "unknown_reference",
        }
    exact = prediction == expected
    for alternative in case.get("acceptableOutputs", []):
        alternate_case = {
            **case,
            "expectedOutput": alternative,
            "acceptableOutputs": [],
        }
        if grade(alternate_case, prediction, execution)["outcome_match"]:
            return {
                "exact_match": exact,
                "outcome_match": True,
                "method": "acceptable_reference",
            }
    outcome, method = exact, "exact_edit"
    if (
        not exact
        and execution
        and prediction["action"] == expected["action"] == "correct_text"
    ):
        before = execution["before"]
        bundle = {
            "input": case["input"],
            "mapping": execution["mapping"],
            "document_hash": session_hash(before),
            **(
                {"caption_authority": execution["caption_authority"]}
                if "caption_authority" in execution
                else {}
            ),
        }
        predicted = apply_edit(before, bundle, prediction)
        reference = apply_edit(before, bundle, expected)
        outcome = transcript_media(predicted) == transcript_media(reference)
        method = "transcript_media_equivalence"
    return {"exact_match": exact, "outcome_match": outcome, "method": method}


def load_executions(path):
    text = path.read_text()
    rows = (
        json.loads(text)
        if path.suffix == ".json"
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    return {row["id"]: row for row in rows}
