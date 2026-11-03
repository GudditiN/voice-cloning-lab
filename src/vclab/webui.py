"""Gradio web UI: record your voice in the browser, train, then speak text in your voice."""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from functools import lru_cache

import gradio as gr
import soundfile as sf

from . import config, data_prep
from .engine import find_finetuned, load_engine

PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Good morning, thank you for calling. How can I help you today?",
    "I'd be happy to look into that for you; could you share your order number?",
    "Please hold on for just a moment while I check the details.",
    "We offer three plans, each with different features and prices.",
    "Is there anything else I can help you with before we wrap up?",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "Numbers matter too: one, two, three, forty-five, and one hundred twenty.",
    "She sells seashells by the seashore, and the shells she sells are surely seashells.",
    "Thanks so much for your patience, and have a wonderful rest of your day!",
]


def _recorded_minutes() -> float:
    return sum(sf.info(p).duration for p in config.RAW_DIR.glob("*.wav")) / 60


def save_clip(path):
    if not path:
        return "Record or upload a clip first.", f"{_recorded_minutes():.1f} min recorded"
    config.ensure_dirs()
    audio = data_prep.normalize(data_prep.load_mono(path))
    q = data_prep.quality_report(audio)
    out = config.RAW_DIR / f"me_{int(time.time())}.wav"
    sf.write(out, audio, config.SAMPLE_RATE)
    warn = []
    if q["too_quiet"]:
        warn.append("very quiet (move closer to the mic)")
    if q["clipping"]:
        warn.append("clipping (lower the input volume)")
    if q["seconds"] < 3:
        warn.append("very short")
    msg = f"Saved {out.name} ({q['seconds']:.1f}s)." + (" Warning: " + ", ".join(warn) if warn else " Looks good.")
    return msg, f"{_recorded_minutes():.1f} min recorded (aim for 15-60 min to fine-tune; 10-30s is enough for zero-shot)"


def next_prompt(i):
    i = (i + 1) % len(PROMPTS)
    return i, PROMPTS[i]


def prepare():
    config.ensure_dirs()
    meta = data_prep.prepare()
    return _read_meta(meta)


def _read_meta(meta=None):
    meta = meta or config.PROCESSED_DIR / "metadata.csv"
    if not meta.exists():
        return []
    with open(meta, encoding="utf-8") as fh:
        return list(csv.reader(fh, delimiter="|"))[1:]


def save_meta(rows):
    rows = rows.values.tolist() if hasattr(rows, "values") else rows
    meta = config.PROCESSED_DIR / "metadata.csv"
    with open(meta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="|")
        w.writerow(["audio_file", "text", "speaker_name"])
        w.writerows(r for r in rows if r and str(r[1]).strip())
    return f"Saved {len(rows)} rows."


def train(epochs):
    import torch

    if not torch.cuda.is_available():
        archive = shutil.make_archive(str(config.OUTPUT_DIR / "dataset"), "zip", config.PROCESSED_DIR)
        yield (
            "No GPU on this machine, so training can't run here. Dataset exported: "
            f"{archive}\nUpload it to Colab (notebooks/00_launch_webui.ipynb) and train there.",
            archive,
        )
        return
    proc = subprocess.Popen(
        [sys.executable, "-m", "vclab.train", "--epochs", str(int(epochs))],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    log = ""
    for line in proc.stdout:
        log = (log + line)[-6000:]
        yield log, None
    proc.wait()
    yield log + ("\nDone. Use the Speak tab with 'fine-tuned'." if proc.returncode == 0 else f"\nFailed (exit {proc.returncode})"), None


@lru_cache(maxsize=2)
def _engine(finetuned: bool):
    return load_engine(finetuned)


def speak(text, use_finetuned, language):
    refs = sorted(config.RAW_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    if not refs:
        raise gr.Error("Record at least one clip in the Record tab first.")
    if use_finetuned and not find_finetuned():
        raise gr.Error("No fine-tuned model yet. Train one first or untick 'fine-tuned'.")
    eng = _engine(bool(use_finetuned))
    audio = eng.synthesize(text, [str(p) for p in refs[-3:]], language)
    config.ensure_dirs()
    out = config.OUTPUT_DIR / f"speech_{int(time.time())}.wav"
    sf.write(out, audio, eng.sample_rate)
    return str(out)


def build() -> gr.Blocks:
    config.ensure_dirs()
    with gr.Blocks(title="Voice Cloning Lab") as demo:
        gr.Markdown("# Voice Cloning Lab\nOnly record your own voice, or a voice you have explicit consent to clone.")
        with gr.Tab("1. Record"):
            idx = gr.State(0)
            prompt = gr.Textbox(PROMPTS[0], label="Read this aloud", interactive=False)
            nxt = gr.Button("Next sentence")
            mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload")
            save = gr.Button("Save clip", variant="primary")
            status, total = gr.Textbox(label="Status", interactive=False), gr.Textbox(f"{_recorded_minutes():.1f} min recorded", label="Total", interactive=False)
            nxt.click(next_prompt, idx, [idx, prompt])
            save.click(save_clip, mic, [status, total])
        with gr.Tab("2. Train"):
            gr.Markdown("Prepare splits clips and transcribes them. Fix any wrong transcripts, save, then train (GPU required).")
            prep = gr.Button("Prepare data")
            table = gr.Dataframe(headers=["audio_file", "text", "speaker_name"], interactive=True, wrap=True)
            save_t = gr.Button("Save edits")
            epochs = gr.Slider(1, 20, 6, step=1, label="Epochs")
            go = gr.Button("Start training", variant="primary")
            log = gr.Textbox(label="Log", lines=12, interactive=False)
            ds = gr.File(label="Dataset export (CPU hosts)")
            prep.click(prepare, None, table)
            save_t.click(save_meta, table, log)
            go.click(train, epochs, [log, ds])
        with gr.Tab("3. Speak"):
            text = gr.Textbox(lines=3, label="Text", value="Hello! This is my cloned voice speaking.")
            ft = gr.Checkbox(label="Use fine-tuned model", value=False)
            lang = gr.Dropdown(["en", "es", "fr", "de", "it", "pt", "hi", "ja", "zh-cn"], value=config.LANGUAGE, label="Language")
            gen = gr.Button("Generate", variant="primary")
            out = gr.Audio(label="Result")
            gen.click(speak, [text, ft, lang], out)
    return demo


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public link (use on Colab)")
    a = ap.parse_args()
    build().queue().launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
