# Voice Cloning Lab

Record your voice in a browser, fine-tune [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2), and generate speech from text in your voice. Cloud-first: Google Colab (GPU), GitHub Codespaces, or Docker.

> Only clone your own voice, or one you have explicit consent to clone. XTTS weights are non-commercial (CPML); see `docs/ROADMAP.md` before any customer-facing use.

## Run it

**Colab (training, needs GPU):** open `notebooks/00_launch_webui.ipynb` in Colab, select a T4 runtime, run both cells, open the share link. Everything persists to `MyDrive/vclab`.

**Docker (CPU: recording, zero-shot speech, data prep):**
```bash
docker compose up webui        # http://localhost:7860
docker compose up api          # REST: POST http://localhost:8000/tts, UI at /ui
docker compose --profile gpu up webui-gpu   # NVIDIA host: can train too
```

**Codespaces:** open the repo in a Codespace (uses `.devcontainer`), run `vclab-webui`, open the forwarded port 7860.

## Workflow
1. **Record** tab: read the prompts. 10-30s is enough for zero-shot cloning; 15-60 min of clean audio for fine-tuning.
2. **Speak** tab: type text, generate. Works immediately with zero-shot.
3. **Train** tab (GPU): Prepare data, fix transcripts, Start training. Then tick "fine-tuned" in Speak. On CPU hosts the Train tab exports a dataset zip to upload to Colab.

CLI: `vclab-infer "Hello" --ref data/raw/me.wav -o outputs/out.wav [--finetuned]`

## Dev
`pip install -e ".[dev]" && pytest` (data prep tests need only numpy/soundfile).
Layout: `src/vclab/` (config, data_prep, engine, train, infer, server, webui), `notebooks/`, `docs/ROADMAP.md`.
