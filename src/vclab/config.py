"""Paths and settings. Override the root with VCLAB_HOME (e.g. a mounted Google Drive folder on Colab)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("VCLAB_HOME", Path.cwd())).resolve()
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 22050  # XTTS training sample rate
OUTPUT_SAMPLE_RATE = 24000  # XTTS synthesis sample rate
# Transcription model for training data: tiny (~75MB) < base (~150MB, default) < small (~480MB, more accurate)
WHISPER_MODEL = os.environ.get("VCLAB_WHISPER_MODEL", "base")
LANGUAGE = os.environ.get("VCLAB_LANGUAGE", "en")


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
