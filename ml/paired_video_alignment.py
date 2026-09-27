"""Measured raw/final audio correspondence; never a predictor or acoustic label."""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfiltfilt, stft


@dataclass(frozen=True)
class AlignmentPolicy:
    sample_rate: int = 8000
    feature_hop: int = 128
    anchor_stride_seconds: float = 3.0
    anchor_seconds: float = 2.0
    anchor_pcm_min: float = 0.62
    anchor_pcm_margin: float = 0.15
    candidate_count: int = 3
    dense_seconds: float = 0.24
    dense_stride_seconds: float = 0.10
    dense_pcm_min: float = 0.68
    dense_pcm_margin: float = 0.15
    offset_merge_seconds: float = 0.006
    max_anchor_distance_seconds: float = 15.0
    local_search_seconds: float = 0.012
    boundary_window_seconds: float = 0.08
    boundary_step_seconds: float = 0.005
    boundary_pcm_min: float = 0.72


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_audio(path, rate):
    actual_rate, pcm = wavfile.read(path)
    if actual_rate != rate or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError("Expected the pinned mono signed-16-bit PCM extraction.")
    return pcm.astype(np.float32) / 32768


def features(pcm, policy):
    _, _, spectrum = stft(
        pcm,
        fs=policy.sample_rate,
        nperseg=2 * policy.feature_hop,
        noverlap=policy.feature_hop,
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum) ** 2
    edges = np.unique(np.round(np.geomspace(2, power.shape[0], 25)).astype(int))
    bands = np.stack(
        [
            np.log(np.maximum(power[a:b].mean(axis=0), 1e-10))
            for a, b in pairwise(edges)
        ],
        axis=1,
    )
    return ((bands - bands.mean(axis=0)) / np.maximum(bands.std(axis=0), 0.2)).astype(
        np.float32
    )


class FeatureSearch:
    def __init__(self, source, target_frames):
        self.source = source
        self.frames = target_frames
        self.size = next_fast_len(len(source) + target_frames - 1)
        self.spectrum = rfft(source, self.size, axis=0)
        self.energy = np.r_[0.0, np.cumsum(np.sum(source.astype(float) ** 2, axis=1))]

    def scores(self, target):
        if len(target) != self.frames:
            raise ValueError("Feature query duration differs from registration.")
        product = self.spectrum * rfft(target[::-1], self.size, axis=0)
        numerator = irfft(product.sum(axis=1), self.size)
        numerator = numerator[len(target) - 1 : len(self.source)]
        denominator = np.sqrt(
            np.maximum(self.energy[len(target) :] - self.energy[: -len(target)], 1e-12)
            * np.sum(target.astype(float) ** 2)
        )
        return numerator / np.maximum(denominator, 1e-12)


def bandpass(pcm, rate):
    return sosfiltfilt(
        butter(4, [250, 2800], btype="bandpass", fs=rate, output="sos"), pcm
    ).astype(np.float32)


def waveform_match(raw, edited, rate, edited_start, raw_start, duration, radius):
    start = round(edited_start * rate)
    length = round(duration * rate)
    target = edited[start : start + length].astype(float)
    low = max(0, round((raw_start - radius) * rate))
    high = min(len(raw), round((raw_start + duration + radius) * rate))
    source = raw[low:high].astype(float)
    if len(target) != length or len(source) < length:
        return None
    target -= target.mean()
    target_energy = np.sum(target**2)
    if target_energy < 1e-10:
        return None
    numerator = fftconvolve(source, target[::-1], mode="valid")
    energy = np.r_[0.0, np.cumsum(source**2)]
    totals = np.r_[0.0, np.cumsum(source)]
    centered_energy = energy[length:] - energy[:-length]
    centered_energy -= (totals[length:] - totals[:-length]) ** 2 / length
    denominator = np.sqrt(np.maximum(centered_energy, 1e-15) * target_energy)
    values = numerator / denominator
    best = int(np.argmax(values))
    return {"raw_start": (low + best) / rate, "pcm_correlation": float(values[best])}


def global_anchors(raw, edited, start, end, policy, progress=None, positions=None):
    raw_features, edited_features = features(raw, policy), features(edited, policy)
    step = policy.feature_hop / policy.sample_rate
    frames = round(policy.anchor_seconds / step)
    search = FeatureSearch(raw_features, frames)
    raw_filtered = bandpass(raw, policy.sample_rate)
    edited_filtered = bandpass(edited, policy.sample_rate)
    anchors = []
    positions = (
        np.arange(start, end - policy.anchor_seconds, policy.anchor_stride_seconds)
        if positions is None
        else positions
    )
    for n, position in enumerate(positions):
        frame = round(position / step)
        position = frame * step
        target = edited_features[frame : frame + frames]
        values = search.scores(target)
        work = values.copy()
        candidates = []
        for _ in range(policy.candidate_count):
            index = int(np.argmax(work))
            raw_position = index * step
            match = waveform_match(
                raw_filtered,
                edited_filtered,
                policy.sample_rate,
                position,
                raw_position,
                policy.anchor_seconds,
                0.6,
            )
            if match:
                candidates.append(
                    {**match, "feature_correlation": float(values[index])}
                )
            distance = round(3 / step)
            work[max(0, index - distance) : index + distance + 1] = -np.inf
        candidates.sort(key=lambda row: row["pcm_correlation"], reverse=True)
        best = candidates[0] if candidates else {"pcm_correlation": -1}
        second = candidates[1]["pcm_correlation"] if len(candidates) > 1 else -1
        accepted = (
            best["pcm_correlation"] >= policy.anchor_pcm_min
            and best["pcm_correlation"] - second >= policy.anchor_pcm_margin
        )
        anchors.append(
            {
                "edited_start": position,
                "duration": policy.anchor_seconds,
                "accepted": bool(accepted),
                "candidates": candidates,
                "reversed_feature_control": float(search.scores(target[::-1]).max()),
                "offset": best.get("raw_start", position) - position
                if accepted
                else None,
            }
        )
        if progress and n % 25 == 0:
            progress(
                {
                    "phase": "global_anchors",
                    "edited_seconds": position,
                    "accepted": sum(a["accepted"] for a in anchors),
                    "total": len(anchors),
                }
            )
    return anchors, raw_filtered, edited_filtered


def candidate_offsets(anchors, position, policy):
    nearby = sorted(
        (
            a
            for a in anchors
            if a["accepted"]
            and abs(a["edited_start"] - position) <= policy.max_anchor_distance_seconds
        ),
        key=lambda a: abs(a["edited_start"] - position),
    )
    offsets = []
    for anchor in nearby:
        if all(
            abs(anchor["offset"] - prior) > policy.offset_merge_seconds
            for prior in offsets
        ):
            offsets.append(anchor["offset"])
    return offsets


def dense_matches(raw, edited, anchors, start, end, policy):
    records = []
    for position in np.arange(
        start, end - policy.dense_seconds, policy.dense_stride_seconds
    ):
        matches = []
        for offset in candidate_offsets(anchors, position, policy):
            match = waveform_match(
                raw,
                edited,
                policy.sample_rate,
                position,
                position + offset,
                policy.dense_seconds,
                policy.local_search_seconds,
            )
            if match:
                matches.append(match)
        matches.sort(key=lambda r: r["pcm_correlation"], reverse=True)
        best = matches[0] if matches else {"pcm_correlation": -1}
        second = matches[1]["pcm_correlation"] if len(matches) > 1 else -1
        accepted = (
            best["pcm_correlation"] >= policy.dense_pcm_min
            and best["pcm_correlation"] - second >= policy.dense_pcm_margin
        )
        records.append(
            {
                "edited_start": round(float(position), 6),
                "duration": policy.dense_seconds,
                "accepted": bool(accepted),
                "pcm_correlation": best["pcm_correlation"],
                "runner_up_pcm": second,
                "offset": best["raw_start"] - position if accepted else None,
            }
        )
    return records


def correspondence_runs(records, policy):
    runs = []
    for row in records:
        if not row["accepted"]:
            continue
        same = (
            runs
            and abs(runs[-1]["offset"] - row["offset"]) <= policy.offset_merge_seconds
            and row["edited_start"] <= runs[-1]["edited_end"] + 0.11
        )
        if same:
            run = runs[-1]
            run["edited_end"] = row["edited_start"] + row["duration"]
            run["observations"] += 1
            run["min_pcm_correlation"] = min(
                run["min_pcm_correlation"], row["pcm_correlation"]
            )
        else:
            runs.append(
                {
                    "edited_start": row["edited_start"],
                    "edited_end": row["edited_start"] + row["duration"],
                    "offset": float(row["offset"]),
                    "observations": 1,
                    "min_pcm_correlation": row["pcm_correlation"],
                }
            )
    return runs


def refine_transition(raw, edited, left, right, policy):
    low = max(left["edited_start"], left["edited_end"] - 0.5)
    high = min(right["edited_end"], right["edited_start"] + 0.5)
    if high - low > 4 or high <= low:
        return {
            "status": "unresolved_transition",
            "edited_bracket": sorted([left["edited_end"], right["edited_start"]]),
            "reason": "Insufficient close supported flanks.",
        }
    support = [[], []]
    for position in np.arange(
        low, high - policy.boundary_window_seconds, policy.boundary_step_seconds
    ):
        scores = []
        for run in (left, right):
            match = waveform_match(
                raw,
                edited,
                policy.sample_rate,
                position,
                position + run["offset"],
                policy.boundary_window_seconds,
                0.002,
            )
            scores.append(match["pcm_correlation"] if match else -1)
        for index in (0, 1):
            if (
                scores[index] >= policy.boundary_pcm_min
                and scores[index] - scores[1 - index] >= 0.2
            ):
                support[index].append(float(position))
    if not all(support):
        return {
            "status": "unresolved_transition",
            "edited_bracket": sorted([left["edited_end"], right["edited_start"]]),
            "reason": "Short-window waveform flanks are not independently supported.",
        }
    lower = max(support[0]) + policy.boundary_window_seconds
    upper = min(support[1])
    # Correlation can persist in a window straddling a splice. Keep its full support width.
    bracket = [
        min(lower, upper) - policy.boundary_window_seconds,
        max(lower, upper) + policy.boundary_window_seconds,
    ]
    estimate = (lower + upper) / 2
    jump = right["offset"] - left["offset"]
    result = {
        "status": "forward_cut_candidate"
        if jump > 0
        else "reorder_or_repeat_candidate",
        "edited_boundary": estimate,
        "edited_bracket": bracket,
        "raw_removed_interval_estimate": [
            estimate + left["offset"],
            estimate + right["offset"],
        ]
        if jump > 0
        else None,
        "raw_left_boundary_bracket": [t + left["offset"] for t in bracket],
        "raw_right_boundary_bracket": [t + right["offset"] for t in bracket],
        "source_jump_seconds": jump,
        "left_offset": left["offset"],
        "right_offset": right["offset"],
        "boundary_precision": "bracketed_audio_correspondence_not_semantic_boundary_certificate",
    }
    profile = boundary_profile(raw, edited, left, right, bracket, policy)
    result["change_point_profile"] = profile
    if profile["status"] == "localized":
        estimate, bracket = profile["edited_boundary"], profile["edited_bracket"]
        result.update(
            edited_boundary=estimate,
            edited_bracket=bracket,
            raw_removed_interval_estimate=[
                estimate + left["offset"],
                estimate + right["offset"],
            ]
            if jump > 0
            else None,
            raw_left_boundary_bracket=[t + left["offset"] for t in bracket],
            raw_right_boundary_bracket=[t + right["offset"] for t in bracket],
            boundary_precision="localized_waveform_change_point_not_semantic_certificate",
        )
    return result


def boundary_profile(raw, edited, left, right, bracket, policy):
    rate = policy.sample_rate
    low = max(0, bracket[0] - 0.15)
    high = min(len(edited) / rate, bracket[1] + 0.15)
    start, stop = round(low * rate), round(high * rate)
    target = edited[start:stop].astype(float)
    source = []
    for run, tail in ((left, False), (right, True)):
        index = start + round(run["offset"] * rate)
        part = raw[index : index + len(target)].astype(float)
        if len(part) != len(target):
            return {"status": "unresolved", "reason": "Source range unavailable."}
        reference = (
            slice(-round(0.1 * rate), None) if tail else slice(None, round(0.1 * rate))
        )
        gain = np.dot(target[reference], part[reference]) / max(
            np.dot(part[reference], part[reference]), 1e-12
        )
        source.append(part * gain)
    errors = [(target - part) ** 2 for part in source]
    difference = np.cumsum(errors[0] - errors[1])
    first = max(0, round((bracket[0] - low) * rate))
    last = min(len(difference), round((bracket[1] - low) * rate))
    if first >= last:
        return {"status": "unresolved", "reason": "Empty change-point bracket."}
    cost = difference[first:last]
    best = int(np.argmin(cost)) + first
    # Keep the entire near-optimal set, including silent/flat portions of the profile.
    scale = float(np.sum(target**2))
    tolerance = max(scale * 0.005, 1e-10)
    near = np.flatnonzero(cost <= difference[best] + tolerance) + first
    interval = [(start + int(near[0])) / rate, (start + int(near[-1]) + 1) / rate]
    separation = min(float(cost[0] - cost.min()), float(cost[-1] - cost.min())) / max(
        scale, 1e-12
    )
    localized = interval[1] - interval[0] <= 0.04 and separation > 0.01
    return {
        "status": "localized" if localized else "unresolved",
        "edited_boundary": (start + best) / rate,
        "edited_bracket": interval,
        "normalized_edge_cost_margin": separation,
        "near_optimal_cost_fraction": 0.005,
        "reason": "Waveform loss separates the source mappings."
        if localized
        else "Flat, weak or multiway change-point evidence; retain wider bracket.",
    }


def find_transitions(raw, edited, runs, policy):
    stable = [r for r in runs if r["observations"] >= 3]
    transitions = []
    for left, right in pairwise(stable):
        if abs(left["offset"] - right["offset"]) <= policy.offset_merge_seconds:
            continue
        transitions.append(
            {
                "id": f"transition-{len(transitions):04d}",
                **refine_transition(raw, edited, left, right, policy),
            }
        )
    return transitions


def uncovered_intervals(runs, start, end):
    cursor, result = start, []
    for run in sorted(runs, key=lambda r: r["edited_start"]):
        if run["edited_start"] > cursor:
            result.append([cursor, run["edited_start"]])
        cursor = max(cursor, run["edited_end"])
    if cursor < end:
        result.append([cursor, end])
    return result


def align_audio(
    raw_pcm, edited_pcm, policy=None, start_seconds=0, end_seconds=600, progress=None
):
    policy = policy or AlignmentPolicy()
    end_seconds = min(end_seconds, len(edited_pcm) / policy.sample_rate)
    if not 0 <= start_seconds < end_seconds:
        raise ValueError("Invalid edited-audio analysis range.")
    anchors, raw, edited = global_anchors(
        raw_pcm, edited_pcm, start_seconds, end_seconds, policy, progress
    )
    dense = dense_matches(raw, edited, anchors, start_seconds, end_seconds, policy)
    runs = correspondence_runs(dense, policy)
    unresolved = uncovered_intervals(runs, start_seconds, end_seconds)
    positions = [float(p) for a, b in unresolved for p in np.arange(a, b - 0.6, 0.4)]
    if positions:
        adaptive, _, _ = global_anchors(
            raw_pcm,
            edited_pcm,
            start_seconds,
            end_seconds,
            replace(
                policy, anchor_seconds=0.6, anchor_pcm_min=0.72, anchor_pcm_margin=0.2
            ),
            progress,
            positions,
        )
        for row in adaptive:
            row["adaptive_short_anchor"] = True
        anchors.extend(adaptive)
        dense = dense_matches(raw, edited, anchors, start_seconds, end_seconds, policy)
        runs = correspondence_runs(dense, policy)
    transitions = find_transitions(raw, edited, runs, policy)
    return {
        "version": "paired-audio-correspondence-v2",
        "policy": asdict(policy),
        "edited_analysis_range": [start_seconds, end_seconds],
        "anchors": anchors,
        "dense_observations": dense,
        "supported_runs": runs,
        "uncertain_edited_intervals": uncovered_intervals(
            runs, start_seconds, end_seconds
        ),
        "transitions": transitions,
        "source_group_id": "shuka-local-recording",
        "split": "dev",
        "training_eligible": False,
        "generated_dataset_rows": 0,
        "agent_listened": False,
        "provenance": "Final export is teacher-only correspondence evidence, not observed editor transactions or predictor input.",
    }


def classify_project_cuts(document, transitions, edited_limit=600):
    """Map teacher cuts to native items without manufacturing boundary certificates."""
    timed = [i for i in document["content"] if "sourceStart" in i]
    result = []
    for cut in transitions:
        if (
            cut["status"] != "forward_cut_candidate"
            or cut["edited_boundary"] > edited_limit
        ):
            continue
        start, end = cut["raw_removed_interval_estimate"]
        selected = [
            i
            for i in timed
            if i["sourceStart"] < end and i["sourceStart"] + i["length"] > start
        ]
        partial = [
            i["uuid"]
            for i in selected
            if i["sourceStart"] < start - 0.001
            or i["sourceStart"] + i["length"] > end + 0.001
        ]
        row = {
            "transition_id": cut["id"],
            "edited_boundary": cut["edited_boundary"],
            "source_interval_estimate": [start, end],
            "duration_seconds": end - start,
            "boundary_uncertainty_seconds": cut["edited_bracket"][1]
            - cut["edited_bracket"][0],
            "items": [
                {
                    k: i[k]
                    for k in ("uuid", "type", "sourceStart", "length", "text")
                    if k in i
                }
                for i in selected
            ],
            "partial_item_ids": partial,
            "natural_cut_qualified": False,
            "training_eligible": False,
            "predictor_input_generated": False,
        }
        if end - start > 2.0:
            row.update(
                status="long_cut_outside_delete_span_policy",
                reason="One observed removal; never subdivide it into artificial microactions.",
            )
        elif cut.get("change_point_profile", {}).get("status") != "localized":
            row.update(
                status="unresolved_boundary",
                reason="Offset jump is measured but its exact source boundary is not localized.",
            )
        elif partial:
            row.update(
                status="native_item_boundary_mismatch",
                reason="The inferred export splice intersects native project items. Do not retime ASR words to force eligibility.",
            )
        else:
            row.update(
                status="native_item_geometry_candidate_only",
                reason="Item geometry is compatible; acoustic class, semantic intent and join quality still lack the required qualification.",
            )
        result.append(row)
    return result


def conservative_mapping(alignment, start, end):
    """Remove mixed-window edges and uncertain transition brackets from support."""
    half = alignment["policy"]["dense_seconds"] / 2
    masks = [sorted(t["edited_bracket"]) for t in alignment["transitions"]]
    runs = alignment["supported_runs"]
    for left, right in pairwise(runs):
        if (
            left["edited_end"] > right["edited_start"]
            and abs(left["offset"] - right["offset"])
            > alignment["policy"]["offset_merge_seconds"]
        ):
            masks.append([right["edited_start"], left["edited_end"]])
    pieces = []
    for run in runs:
        intervals = [
            [max(start, run["edited_start"] + half), min(end, run["edited_end"] - half)]
        ]
        for low, high in masks:
            intervals = [
                piece
                for a, b in intervals
                for piece in ([a, min(low, b)], [max(a, high), b])
                if piece[1] > piece[0]
            ]
        pieces.extend(
            {
                "edited_start": a,
                "edited_end": b,
                "raw_start": a + run["offset"],
                "raw_end": b + run["offset"],
                "offset": run["offset"],
            }
            for a, b in intervals
            if b > a
        )
    pieces.sort(key=lambda p: p["edited_start"])
    for left, right in pairwise(pieces):
        if left["edited_end"] > right["edited_start"] + 1e-6:
            raise ValueError("Conservative mapping intervals overlap.")
    return pieces


def teacher_timeline(alignment, source_hash, start=7.52, end=607.52):
    """A complete estimated preview, with uncertainty separate from NEP evidence."""
    cuts = [
        t
        for t in alignment["transitions"]
        if t["status"] == "forward_cut_candidate" and start < t["edited_boundary"] < end
    ]
    cuts.sort(key=lambda t: t["edited_boundary"])
    if any(
        t["status"] != "forward_cut_candidate"
        and start <= min(t["edited_bracket"]) <= end
        for t in alignment["transitions"]
    ):
        raise ValueError("Cannot flatten unresolved/reordered transitions into cuts.")
    containing = [
        r
        for r in alignment["supported_runs"]
        if r["edited_start"] <= start < r["edited_end"]
    ]
    if not containing:
        raise ValueError("Pilot start needs a supported source correspondence.")
    offset = containing[0]["offset"]
    cursor, segments = start, []
    for cut in cuts:
        if (
            abs(cut["left_offset"] - offset)
            > alignment["policy"]["offset_merge_seconds"]
        ):
            raise ValueError("Unexplained offset change in reconstructed path.")
        boundary = cut["edited_boundary"]
        if boundary <= cursor:
            raise ValueError("Non-increasing candidate boundary.")
        segments.append(
            {
                "edited_start": cursor,
                "edited_end": boundary,
                "raw_start": cursor + offset,
                "raw_end": boundary + offset,
            }
        )
        cursor, offset = boundary, cut["right_offset"]
    segments.append(
        {
            "edited_start": cursor,
            "edited_end": end,
            "raw_start": cursor + offset,
            "raw_end": end + offset,
        }
    )
    content = [
        {
            "type": "paragraph_start",
            "uuid": str(uuid5(NAMESPACE_URL, "paired-teacher/start")),
            "language": "ko",
            "speaker": None,
        }
    ]
    for n, segment in enumerate(segments):
        content.append(
            {
                "type": "non_text",
                "uuid": str(uuid5(NAMESPACE_URL, f"paired-teacher/{start}/{end}/{n}")),
                "source": source_hash,
                "sourceStart": segment["raw_start"],
                "length": segment["raw_end"] - segment["raw_start"],
            }
        )
    content.append(
        {
            "type": "paragraph_end",
            "uuid": str(uuid5(NAMESPACE_URL, "paired-teacher/end")),
        }
    )
    metadata = {
        "provenance": "reconstructed_teacher_audio_source_map_not_original_editor_history",
        "source_group_id": "shuka-local-recording",
        "split": "dev",
        "training_eligible": False,
        "edited_analysis_range": [start, end],
        "caption_status": "not_reconstructed_do_not_infer_transcript_labels",
        "warning": "Audio-only estimated preview. Wide boundary brackets remain uncertain. Non_text means no reconstructed transcript; it does not certify silence.",
    }
    return {"content": content, "metadata": metadata}, segments, cuts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--edited", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Pilot outputs are immutable; choose a new output file.")
    policy = AlignmentPolicy()
    result = align_audio(
        read_audio(args.raw, policy.sample_rate),
        read_audio(args.edited, policy.sample_rate),
        policy,
        end_seconds=args.end_seconds,
        progress=lambda row: print(json.dumps(row), flush=True),
    )
    result["inputs_sha256"] = {
        str(p.resolve()): digest(p) for p in [args.raw, args.edited, Path(__file__)]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runs": len(result["supported_runs"]),
                "transitions": len(result["transitions"]),
                "uncertain_seconds": sum(
                    b - a for a, b in result["uncertain_edited_intervals"]
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
