"""TTS engines behind one interface, so XTTS can be swapped for another model later."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np

from . import config


class TTSEngine(ABC):
    sample_rate: int

    @abstractmethod
    def synthesize(self, text: str, speaker_wav: list[str] | str, language: str = config.LANGUAGE) -> np.ndarray:
        """Return mono float32 audio at self.sample_rate."""

    def stream(self, text: str, speaker_wav: list[str] | str, language: str = config.LANGUAGE) -> Iterator[np.ndarray]:
        """Default: no real streaming, yield the whole utterance. Engines override for low latency."""
        yield self.synthesize(text, speaker_wav, language)


class XTTSEngine(TTSEngine):
    """Coqui XTTS-v2. Pass model_dir (with model.pth, config.json, vocab.json) to use a fine-tuned checkpoint."""

    sample_rate = config.OUTPUT_SAMPLE_RATE

    def __init__(self, model_dir: Path | None = None, device: str | None = None):
        import torch
        from TTS.api import TTS

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_dir:
            self.tts = TTS(model_path=str(model_dir), config_path=str(Path(model_dir) / "config.json"))
        else:
            self.tts = TTS(config.XTTS_MODEL)
        self.tts.to(device)

    def synthesize(self, text, speaker_wav, language=config.LANGUAGE):
        wavs = [speaker_wav] if isinstance(speaker_wav, str) else list(speaker_wav)
        out = self.tts.tts(text=text, speaker_wav=wavs, language=language)
        return np.asarray(out, dtype=np.float32)


def find_finetuned(models_dir: Path = config.MODELS_DIR) -> Path | None:
    d = models_dir / "finetuned"
    return d if (d / "model.pth").exists() else None


def load_engine(use_finetuned: bool = False) -> XTTSEngine:
    return XTTSEngine(find_finetuned() if use_finetuned else None)
