"""Source-grounded cue import for the frozen E1 recording, with no word-time claims."""

import re
from uuid import NAMESPACE_URL, uuid5


def cues_from_srt(text):
    cues = []
    pattern = r"(\d+):(\d{2}):(\d{2}),(\d{3}) --> (\d+):(\d{2}):(\d{2}),(\d{3})"
    for block in re.split(r"\n\s*\n", text.lstrip("\ufeff").replace("\r\n", "\n").strip()):
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].isdigit():
            raise ValueError("malformed_publisher_caption")
        match = re.fullmatch(pattern, lines[1])
        if not match:
            raise ValueError("malformed_publisher_caption_time")
        values = [int(value) for value in match.groups()]
        times = []
        for hours, minutes, seconds, milliseconds in (values[:4], values[4:]):
            if minutes >= 60 or seconds >= 60:
                raise ValueError("invalid_publisher_caption_time")
            times.append(hours * 3600 + minutes * 60 + seconds + milliseconds / 1000)
        payload = " ".join(line.strip() for line in lines[2:])
        if times[0] >= times[1] or not payload:
            raise ValueError("empty_or_reversed_publisher_caption")
        cues.append({"index": int(lines[0]), "start": times[0], "end": times[1], "text": payload})
    if not cues or len({cue["index"] for cue in cues}) != len(cues):
        raise ValueError("empty_or_duplicate_publisher_captions")
    return cues


def window_cues(cues, window):
    selected = [cue for cue in cues if window["start"] <= cue["start"] and cue["end"] <= window["end"]]
    if not selected or any(right["start"] < left["end"] for left, right in zip(selected, selected[1:])):
        raise ValueError("empty_or_overlapping_fixed_caption_window")
    return selected


def initial_document(cues, window, source_id, language="en"):
    """Preserve each complete publisher cue; gaps have source timing but unknown audio."""
    selected = window_cues(cues, window)
    content = []
    for position, cue in enumerate(selected):
        def uid(kind):
            return str(uuid5(NAMESPACE_URL, f"{source_id}:{cue['index']}:{kind}"))

        content.append({"type": "paragraph_start", "uuid": uid("paragraph"), "speaker": None, "language": language})
        content.append({"type": "text", "uuid": uid("text"), "text": cue["text"], "source": source_id,
                        "sourceStart": cue["start"], "length": cue["end"] - cue["start"],
                        "asr": {"boundary_status": "unverified", "audio_class": "unknown"}})
        if position + 1 < len(selected) and selected[position + 1]["start"] > cue["end"]:
            content.append({"type": "non_text", "uuid": uid("gap"), "source": source_id,
                            "sourceStart": cue["end"], "length": selected[position + 1]["start"] - cue["end"],
                            "asr": {"boundary_status": "unverified", "audio_class": "unknown"}})
        content.append({"type": "paragraph_end", "uuid": uid("end")})
    return {"content": content, "metadata": {"transcription": {"provider": "publisher-caption", "source": source_id}}}


def verify_initial_document(document, cues, window, source_id, language):
    if document != initial_document(cues, window, source_id, language):
        raise ValueError("initial_natural_state_differs_from_frozen_publisher_cues")
