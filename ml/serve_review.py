"""Serve one local review batch using the backend's existing dependencies."""

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from app.review import create_review_app  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--port", type=int, default=43127)
    args = parser.parse_args()
    uvicorn.run(
        create_review_app(args.batch),
        host="127.0.0.1",
        port=args.port,
        access_log=False,
    )
