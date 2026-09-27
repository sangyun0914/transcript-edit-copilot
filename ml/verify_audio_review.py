"""Verify candidate audio artifacts and render a local listening review packet."""

import argparse
import html
import json
import warnings
from pathlib import Path

from audio_review import (
    RATE,
    check_files,
    contract_probe,
    cut_pcm,
    load_pcm,
    source_selection,
)
from dataset_release_inventory import ROOT, digest, read, rows, write


def verify(directory):
    import numpy as np

    manifest = read(directory / "manifest.json")
    check_files(directory, manifest)
    scan = ROOT / manifest["scan_directory"]
    if digest(scan / "manifest.json") != manifest["scan_manifest_sha256"]:
        raise ValueError("Parent scan changed.")
    check_files(scan, read(scan / "manifest.json"))
    windows = {w["id"]: w for w in read(scan / "windows.json")}
    cases = rows(directory / "review-cases.jsonl")
    if len(cases) != manifest["cases"] or manifest["training_eligible"]:
        raise ValueError("Unexpected candidate count or admission flag.")
    render_count = 0
    for case in cases:
        source = read(scan / "sources" / f"{case['source_id']}.json")
        if (
            case["training_eligible"]
            or case["agent_listened"]
            or case["human_listened"]
        ):
            raise ValueError(
                "Unreviewed audio cannot claim listening or training eligibility."
            )
        if (
            case["split"] != source["reserved_split"]
            or case["source_group_id"] != source["source_group_id"]
        ):
            raise ValueError("Candidate source crosses splits.")
        if (
            case["source_sha256"] != source["video_sha256"]
            or digest(ROOT / source["video"]) != case["source_sha256"]
        ):
            raise ValueError("Original media changed.")
        selected = source_selection(source, case)
        if (
            selected != case["selected_items"]
            or contract_probe(source, case, selected) != case["contract"]
        ):
            raise ValueError("Native targets or unverified-cut rejection changed.")
        window = windows[case["window_id"]]
        track = window["tracks"]["mix"]
        if digest(Path(track["path"])) != track["sha256"]:
            raise ValueError("Parent audio preview changed.")
        original = load_pcm(Path(track["path"]))
        offset = round(
            (case["preview_origin_seconds"] - window["start_seconds"]) * RATE
        )
        for render in case["renders"]:
            folder = directory / render["folder"]
            before, after, removed = [
                load_pcm(folder / f"{kind}.wav")
                for kind in ["before", "after", "removed"]
            ]
            expected, cut, indices = cut_pcm(
                before,
                render["removed_source_interval"],
                case["preview_origin_seconds"],
            )
            if not np.array_equal(before, original[offset : offset + len(before)]):
                raise ValueError(
                    "Review audio is not the exact registered source excerpt."
                )
            if not np.array_equal(after, expected) or not np.array_equal(removed, cut):
                raise ValueError("Retained/removed samples differ from the proposal.")
            if not np.array_equal(load_pcm(folder / "ffmpeg-after.wav"), after):
                raise ValueError("FFmpeg render changed.")
            restored = np.concatenate(
                (after[: indices[0]], removed, after[indices[0] :])
            )
            if not np.array_equal(restored, before):
                raise ValueError("PCM reconstruction failed.")
            for kind in ["before", "after", "removed"]:
                if read(folder / f"asr-{kind}.json")["audio_sha256"] != digest(
                    folder / f"{kind}.wav"
                ):
                    raise ValueError("ASR evidence refers to different audio.")
            render_count += 1
    if render_count != manifest["render_variants"]:
        raise ValueError("Variant count changed.")
    return {
        "directory": str(directory.relative_to(ROOT)),
        "manifest_sha256": digest(directory / "manifest.json"),
        "cases": len(cases),
        "renders": render_count,
        "verified": True,
        "checks": [
            "native item boundaries",
            "source group split",
            "original media and preview hashes",
            "sample-exact proposal",
            "independent FFmpeg equality",
            "PCM reconstruction",
            "ASR audio hashes",
            "executor rejects unverified cuts",
            "no invented listening or SFT admission",
        ],
    }


def verify_features(directories):
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.feature_extractor import FeatureExtractor

    extractor, seen, results = FeatureExtractor(), set(), []
    for directory in directories:
        for path in sorted(directory.glob("*/*/asr-*.json")):
            audio = path.with_name(path.stem.replace("asr-", "") + ".wav")
            identity = digest(audio)
            if identity in seen:
                continue
            seen.add(identity)
            values = decode_audio(str(audio))
            with warnings.catch_warnings(record=True) as caught:
                features = extractor(values)
            padded = np.pad(values.astype(np.float32), (0, 160))
            window = np.hanning(extractor.n_fft + 1)[:-1].astype(np.float32)
            spectrum = extractor.stft(
                padded,
                extractor.n_fft,
                extractor.hop_length,
                window=window,
                return_complex=True,
            ).astype(np.complex64)
            magnitudes = np.abs(spectrum[..., :-1]) ** 2
            # Independent accumulation checks the platform BLAS warnings, rather
            # than suppressing a potentially corrupt model input.
            mel = np.einsum(
                "mf,ft->mt",
                extractor.mel_filters.astype(np.float64),
                magnitudes.astype(np.float64),
                optimize=False,
            )
            reference = np.log10(np.clip(mel, 1e-10, None))
            reference = (np.maximum(reference, reference.max() - 8) + 4) / 4
            error = float(np.max(np.abs(features - reference)))
            if not np.isfinite(features).all() or error >= 2e-6:
                raise ValueError(
                    "Whisper features differ from the independent numeric reference."
                )
            results.append(
                {
                    "audio_sha256": identity,
                    "features_finite": True,
                    "max_abs_error": error,
                    "warnings": [str(w.message) for w in caught],
                }
            )
    return {"all_pass": True, "unique_audio_probes": len(results), "checks": results}


def plot_case(case, source_directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 1, figsize=(10, 3.5), constrained_layout=True)
    render = case["renders"][0]
    folder = source_directory / render["folder"]
    for ax, kind in zip(axes, ["before", "after"]):
        pcm = load_pcm(folder / f"{kind}.wav").astype(float) / 32768
        # Min/max envelope preserves short transients when the figure is reduced.
        stride = max(1, len(pcm) // 1800)
        frames = pcm[: len(pcm) // stride * stride].reshape(-1, stride)
        times = np.arange(len(frames)) * stride / RATE
        ax.fill_between(
            times, frames.min(axis=1), frames.max(axis=1), color="#275c94", linewidth=0
        )
        ax.set_ylabel(kind.capitalize())
        ax.grid(alpha=0.15)
        ax.set_xlim(0, len(pcm) / RATE)
    left, right = render["verification"]["cut_samples"]
    axes[0].axvspan(left / RATE, right / RATE, alpha=0.2, color="#e08b43")
    axes[1].axvline(left / RATE, color="#b34635", linestyle="--")
    axes[1].set_xlabel(
        "Seconds within each preview; orange = removed, dashed = new join"
    )
    fig.savefig(output, dpi=130)
    plt.close(fig)


def report(directories, decisions_path, output):
    output.mkdir(parents=True, exist_ok=False)
    decisions = read(decisions_path)
    decision_map = {row["id"]: row for row in decisions["cases"]}
    collected, checks = [], []
    for directory in directories:
        checks.append(verify(directory))
        for case in rows(directory / "review-cases.jsonl"):
            collected.append((case, directory))
    if set(decision_map) != {c["id"] for c, _ in collected}:
        raise ValueError("Every investigated candidate needs an explicit disposition.")
    cards, review_rows = [], []
    for case, directory in sorted(
        collected,
        key=lambda pair: (
            decision_map[pair[0]["id"]]["status"] != "listening_review",
            pair[0]["id"],
        ),
    ):
        decision = decision_map[case["id"]]
        plot_case(case, directory, output / f"{case['id']}.png")
        renders = []
        for render in case["renders"]:
            relative = Path("..") / directory.name / render["folder"]
            players = "".join(
                f'<label>{kind.capitalize()}<audio controls preload="none" src="{html.escape(str(relative / (kind + ".wav")), quote=True)}"></audio></label>'
                for kind in ["before", "after", "removed", "join"]
            )
            asr = "".join(
                f"<dt>{kind.capitalize()}</dt><dd>{html.escape(render['asr'][kind]) or '(no decoded words)'}</dd>"
                for kind in ["before", "after", "removed"]
            )
            renders.append(
                f'<div class="variant"><h3>{html.escape(render["id"])}</h3><div class="players">{players}</div><details><summary>Automatic transcription and signal evidence</summary><dl>{asr}</dl><p>Same normalized before/after text: {render["asr_text_stable"]}. Join jump: {render["verification"]["signal"]["join_jump_dbfs"]} dBFS.</p></details></div>'
            )
        question = ""
        if decision["status"] == "listening_review":
            review_rows.append(
                {
                    "id": case["id"],
                    "source_sha256": case["source_sha256"],
                    "evidence_sha256": digest(directory / case["id"] / "evidence.json"),
                    "decision": None,
                    "notes": "",
                    "human_listened": False,
                }
            )
            question = f'<fieldset data-case="{case["id"]}"><legend>After listening to the original, removed portion and proposed join</legend><select aria-label="Decision for {case["id"]}"><option value="">Unreviewed</option><option value="accept">Cut preserves speech and sounds appropriate for the brief</option><option value="reject">Reject this cut</option><option value="uncertain">Uncertain / needs a different boundary</option></select><textarea aria-label="Listening notes for {case["id"]}" placeholder="Word tail, breath, laughter, pacing, preferred variant, or another issue"></textarea></fieldset>'
        cards.append(
            f'<article id="{case["id"]}"><div class="tag">{html.escape(decision["status"])} · {case["language"]} · {case["split"]}</div><h2>{html.escape(case["id"])}</h2><p>{html.escape(decision["reason"])}</p><p class="brief">{html.escape(case["brief"])}</p><p class="meta">{html.escape(case["source_group_id"])} · original {case["target_start"]:.3f}–{case["target_end"]:.3f}s · {case["action"]}</p><img src="{case["id"]}.png" alt="Original and proposed audio waveform with removed region and new join"/><div>{"".join(renders)}</div>{question}</article>'
        )
    pending = len(review_rows)
    payload = json.dumps(
        {
            "schema_version": 1,
            "scope": "human listening observations only; no automatic SFT admission",
            "cases": review_rows,
        },
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Audio dataset review</title>
<style>body{font:16px/1.5 system-ui;margin:0;background:#f3f5f7;color:#172231}main{max-width:980px;margin:auto;padding:32px 20px}h1{font-size:30px}h2{font-size:21px;margin:8px 0}h3{font-size:16px}article{background:white;border:1px solid #d4dce4;border-radius:12px;padding:24px;margin:24px 0}img{width:100%;height:auto}.tag,.meta{font-size:13px;color:#56697c}.tag{font-weight:700}.brief{border-left:3px solid #87a5c5;padding-left:12px}.players{display:grid;grid-template-columns:1fr 1fr;gap:12px}audio{display:block;width:100%;margin-top:4px}fieldset{margin-top:20px;border:1px solid #c1d2e3}select,textarea{display:block;width:100%;padding:10px;margin:8px 0;box-sizing:border-box;font:inherit}textarea{height:80px}button{padding:12px 20px;border:0;border-radius:8px;background:#245881;color:white;font:inherit;cursor:pointer}dt{font-weight:bold}dd{margin:0 0 12px}summary{cursor:pointer}header{position:sticky;top:0;background:#f3f5f7f2;padding:12px 0;z-index:1}details{margin-top:12px}@media(max-width:600px){.players{grid-template-columns:1fr}article{padding:16px}}</style><main>
<h1>Audio dataset review</h1><p>REVIEW_COUNT short listening checks remain after local Whisper, Silero speech detection, native annotations, source frames and exact render verification. The other candidates below are already excluded or preserved for stated reasons.</p><p>These are proposed edits, not approved training labels. “Removed” plays only what the edit discards; “Join” plays four seconds around the new boundary. Silence-only ASR can invent words. Listen to the original and edited versions before deciding.</p><header><button id="export">Download listening observations</button> <span id="progress">0 / REVIEW_COUNT reviewed</span></header>CARDS
<script>const observations=PAYLOAD;document.querySelectorAll('fieldset').forEach(f=>{f.addEventListener('change',()=>{const r=observations.cases.find(r=>r.id===f.dataset.case);r.decision=f.querySelector('select').value||null;r.notes=f.querySelector('textarea').value;r.human_listened=Boolean(r.decision);document.querySelector('#progress').textContent=observations.cases.filter(r=>r.decision).length+' / '+observations.cases.length+' reviewed';});});document.querySelector('#export').onclick=()=>{document.querySelectorAll('fieldset').forEach(f=>{observations.cases.find(r=>r.id===f.dataset.case).notes=f.querySelector('textarea').value;});const url=URL.createObjectURL(new Blob([JSON.stringify(observations,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='audio-listening-observations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};</script></main></html>"""
    page = (
        page.replace("REVIEW_COUNT", str(pending))
        .replace("CARDS", "".join(cards))
        .replace("PAYLOAD", payload)
    )
    (output / "index.html").write_text(page)
    write(output / "decisions.json", decisions)
    write(
        output / "verification.json",
        {
            "all_pass": True,
            "bundles": checks,
            "listening_checks": pending,
            "observations_auto_promote_training": False,
        },
    )
    write(
        output / "manifest.json",
        {
            "files": {
                str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
            "decision_source_sha256": digest(decisions_path),
            "code_sha256": digest(Path(__file__)),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", action="append", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-features", action="store_true")
    args = parser.parse_args()
    directories = [p.resolve() for p in args.directory]
    if args.decisions:
        report(directories, args.decisions.resolve(), args.output.resolve())
    else:
        result = {"bundles": [verify(p) for p in directories], "all_pass": True}
        if args.check_features:
            result["features"] = verify_features(directories)
        write(args.output, result)


if __name__ == "__main__":
    main()
