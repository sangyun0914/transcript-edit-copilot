from importlib.util import find_spec

LOCAL_SETUP = "Local ASR is unavailable. In server/, run: uv sync --extra local-asr"


def local_asr_available() -> bool:
    try:
        return all(
            find_spec(module) is not None
            for module in ("vosk", "soundfile", "torch", "pyannote.audio", "scipy", "numpy")
        )
    except (ImportError, ValueError):
        return False
