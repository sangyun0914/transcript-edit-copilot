"""Separate reviewed recordings from algebraic checks on authored timelines."""

from reviewed_audio import CUTS, expected_media_after_cut, removed_interval
from reviewed_audio import verify_media_edit as verify_reviewed_media
from source_qualification import media_intervals


def verify_media_edit(
    before, after, mapping, action, qualifications, media_kind="recording"
):
    if media_kind != "authored_symbolic_timeline":
        return verify_reviewed_media(before, after, mapping, action, qualifications)
    metadata = before.get("metadata", {})
    source = f"symbolic:{metadata.get('script_sha256')}"
    if metadata.get("provenance") != "authored_symbolic_timeline" or any(
        i.get("source") != source for i in before["content"] if "sourceStart" in i
    ):
        raise ValueError("A symbolic check cannot authorize edits to a recording.")
    if qualifications:
        raise ValueError(
            "Authored timelines cannot carry natural-audio qualifications."
        )
    if action["action"] in CUTS:
        _, interval = removed_interval(before, mapping, action)
        expected = expected_media_after_cut(before, source, interval)
    else:
        expected = media_intervals(before)
    actual = media_intervals(after)
    if len(actual) != len(expected) or any(
        a[0] != e[0] or any(abs(x - y) > 1.1e-6 for x, y in zip(a[1:], e[1:]))
        for a, e in zip(actual, expected)
    ):
        raise ValueError(
            "Action changed symbolic intervals outside its declared effect."
        )
    return None
