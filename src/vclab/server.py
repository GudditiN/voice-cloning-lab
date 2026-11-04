"""FastAPI service: POST /tts returns wav; web UI served at /. Future avatar/realtime layers call this."""
from __future__ import annotations

import io
from functools import lru_cache

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from . import config
from .engine import find_finetuned, load_engine

app = FastAPI(title="vclab")


class TTSRequest(BaseModel):
    text: str
    language: str = config.LANGUAGE
    finetuned: bool = False
    speaker_wav: str | None = None  # path to reference wav; defaults to latest in data/raw


@lru_cache(maxsize=2)
def _engine(finetuned: bool):
    return load_engine(finetuned)


def latest_reference() -> str | None:
    files = sorted(config.RAW_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else None


@app.get("/health")
def health():
    return {"ok": True, "finetuned_available": find_finetuned() is not None}


@app.post("/tts")
def tts(req: TTSRequest):
    ref = req.speaker_wav or latest_reference()
    if not ref:
        raise HTTPException(400, "No reference audio: record one in the web UI or put a wav in data/raw/")
    eng = _engine(req.finetuned)
    audio = eng.synthesize(req.text, ref, req.language)
    buf = io.BytesIO()
    sf.write(buf, audio, eng.sample_rate, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")


def mount_ui() -> None:
    import gradio as gr

    from .webui import build

    gr.mount_gradio_app(app, build(), path="/")


mount_ui()
