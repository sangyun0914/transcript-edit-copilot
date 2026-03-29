# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**audapolis** — a desktop app for editing spoken-word media (podcasts, interviews, audiobooks) with automatic transcription. Provides a word-processor-like editing experience for audio/video content. Fully offline, no cloud dependencies.

## Architecture

Two-process architecture: an **Electron + React frontend** (`app/`) and a **Python FastAPI backend** (`server/`).

- **Electron main process** (`app/main_process/`) spawns the Python server as a subprocess. In dev, it runs `uv run python run.py`; in production, a bundled executable.
- **Server startup protocol**: Python server finds an open port, prints JSON messages to stdout (`{msg: "server_starting", port: ...}` then `{msg: "server_started", token: ...}`). The main process reads these to establish connection.
- **Authentication**: Bearer token (base64-encoded 64 random bytes) generated at server startup.
- **Frontend state**: Redux Toolkit with redux-undo for editor undo/redo. Store slices: `nav`, `transcribe`, `editor`, `models`, `server`.
- **Document format (V3)**: ZIP files containing `document.json` + `/sources/` with audio files. Documents have paragraphs with `text`, `non_text`, and `artificial_silence` items, each with source audio references and timing data. V1/V2 auto-migrate to V3.
- **Transcription pipeline**: Audio → pydub (16kHz mono WAV) → optional pydiar diarization → Vosk ASR (2-second blocks) → structured word/silence segments with timing and confidence.

## Commands

### Frontend (run from `app/`)

```bash
npm start                # Dev mode with hot reload
npm run build            # Production build (Vite)
npm run dist             # Package with electron-builder
npm test                 # All tests (fast + puppeteer)
npm run test:fast        # Jest unit tests only (excludes .pup.spec.ts)
npm run test:puppeteer   # E2E tests with Puppeteer/Electron
npm run check            # TypeScript + ESLint
npm run check:tsc        # Type checking only
npm run check:eslint     # Linting only
npm run fmt              # Prettier format
```

### Backend (run from `server/`)

```bash
uv sync                                     # Install dependencies
uv run python run.py                        # Start server
uv run uvicorn app.main:app --reload        # Dev server with reload
```

Backend linting (from `server/`):
```bash
ruff check .
ruff format --check .
mypy .
```

## Key API Endpoints

- `POST /tasks/start_transcription/` — start transcription job
- `POST /tasks/download_model/` — download speech model
- `GET /tasks/{uuid}/` — task status/progress
- `GET /models/available` — list downloadable models
- `GET /models/downloaded` — list cached models
- `POST /util/otio/convert` — export to OpenTimelineIO

## Code Style

- **Frontend**: Prettier + ESLint. `@typescript-eslint/no-explicit-any` is off. Unused imports are errors. Prefix unused vars with `_`.
- **Backend**: Ruff for linting and formatting (line-length: 100, isort enabled). Python 3.13+.
