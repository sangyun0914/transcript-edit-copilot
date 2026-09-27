import argparse
import json
import time
from pathlib import Path

from app.asr.scribe import ScribeError, transcribe


def main():
    parser = argparse.ArgumentParser(
        description="Upload one PCM WAV to Scribe and save its result."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="ko")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output already exists; choose a new path.")

    started = time.perf_counter()
    try:
        with args.audio.open("rb") as audio:
            content, metadata = transcribe(audio, args.audio.name, args.language, True, None)
    except (ScribeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    elapsed = time.perf_counter() - started
    result = {"content": content, "transcription": metadata, "elapsed_seconds": elapsed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "duration_seconds": metadata["duration"],
                "elapsed_seconds": round(elapsed, 3),
                "language": metadata["language_code"],
                "timed_items": sum(len(paragraph["content"]) for paragraph in content),
                "result": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
