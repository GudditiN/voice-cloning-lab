"""Turn raw recordings into a training set: mono 22.05kHz clips split on silence + transcripts."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}
MIN_CLIP_S = 2.0
MAX_CLIP_S = 15.0


def load_mono(path: Path, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    import torchaudio

    wav, orig_sr = torchaudio.load(str(path))
    wav = wav.mean(0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav.squeeze(0).numpy()


def normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.abs(audio).max()) if audio.size else 0.0
    return audio if m == 0 else audio * (peak / m)


def split_on_silence(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    min_s: float = MIN_CLIP_S,
    max_s: float = MAX_CLIP_S,
    silence_db: float = -40.0,
    min_silence_s: float = 0.4,
) -> list[np.ndarray]:
    """Energy-based splitter. Cuts at pauses >= min_silence_s, keeps clips within [min_s, max_s]."""
    frame = int(0.02 * sr)
    n = len(audio) // frame
    if n == 0:
        return []
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    voiced = 20 * np.log10(rms) > silence_db
    min_gap = int(min_silence_s / 0.02)

    # voiced runs; a run ends once silence lasts min_gap frames
    segs: list[tuple[int, int]] = []
    start = last = None
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            last = i
        elif start is not None and i - last >= min_gap:
            segs.append((start, last + 1))
            start = None
    if start is not None:
        segs.append((start, last + 1))

    pad = int(0.1 * sr)
    clips: list[np.ndarray] = []
    for s, e in segs:
        clip = audio[max(0, s * frame - pad) : min(len(audio), e * frame + pad)]
        dur = len(clip) / sr
        if dur < min_s:
            continue
        if dur > max_s:  # hard-chop overlong runs
            step = int(max_s * sr)
            for k in range(0, len(clip), step):
                part = clip[k : k + step]
                if len(part) / sr >= min_s:
                    clips.append(part)
        else:
            clips.append(clip)
    return clips


def quality_report(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> dict:
    """Quick checks used by the UI to warn about bad recordings."""
    if audio.size == 0:
        return {"seconds": 0.0, "peak": 0.0, "clipping": False, "too_quiet": True}
    peak = float(np.abs(audio).max())
    return {
        "seconds": len(audio) / sr,
        "peak": peak,
        "clipping": bool((np.abs(audio) > 0.99).mean() > 0.001),
        "too_quiet": peak < 0.1,
    }


def transcribe(clips: list[Path], language: str = config.LANGUAGE) -> list[str]:
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel("small", device=device, compute_type="float16" if device == "cuda" else "int8")
    texts = []
    for p in clips:
        segs, _ = model.transcribe(str(p), language=language)
        texts.append(" ".join(s.text.strip() for s in segs).strip())
    return texts


def prepare(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR, do_transcribe: bool = True) -> Path:
    """Process every recording in raw_dir. Writes out_dir/wavs/*.wav and out_dir/metadata.csv (audio_file|text|speaker_name)."""
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    paths: list[Path] = []
    for f in files:
        for i, clip in enumerate(split_on_silence(normalize(load_mono(f)))):
            out = wav_dir / f"{f.stem}_{i:03d}.wav"
            sf.write(out, clip, config.SAMPLE_RATE)
            paths.append(out)
    texts = transcribe(paths) if do_transcribe else [""] * len(paths)
    meta = out_dir / "metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="|")
        w.writerow(["audio_file", "text", "speaker_name"])
        for p, t in zip(paths, texts):
            if t:
                w.writerow([f"wavs/{p.name}", t, "me"])
    return meta
