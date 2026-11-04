# Voice Cloning Lab

Record your voice in a browser, fine-tune [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2), and generate speech from text in your voice.

> Only clone your own voice, or one you have explicit consent to clone. XTTS weights are non-commercial (CPML, accepted via `COQUI_TOS_AGREED=1` in the Dockerfile); see `docs/ROADMAP.md` before customer-facing use.

## Run (Docker)

```bash
docker compose up --build
```

Open **http://localhost:7860**. That's it: web UI at `/`, REST API at `POST /tts` on the same port.
Your recordings, models and outputs stay in `data/`, `models/`, `outputs/` next to this README. The first synthesis downloads ~2 GB of model weights into `models/.cache`.

Stop with `Ctrl+C`; run in the background with `docker compose up -d`.

Lightweight by default: CPU-only torch, slim base image, and the small `base` Whisper model for transcripts (set `VCLAB_WHISPER_MODEL=tiny` for even less, `small` for accuracy).

## Use

1. **Record** tab: read the prompts (10-30s is enough for zero-shot; 15-60 min for fine-tuning).
2. **Speak** tab: type text, click Generate.
3. **Train** tab: Prepare data, fix transcripts, Start training. Training needs a GPU:
   - **Google Colab (free T4):** open `notebooks/00_launch_webui.ipynb`, run the two cells, use the link. Files persist on Google Drive.
   - **NVIDIA machine:** `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`
   - On a Mac or other CPU-only Docker, the Train button exports a dataset zip to take to Colab.

API example: `curl -X POST localhost:7860/tts -H 'content-type: application/json' -d '{"text":"Hello"}' -o hello.wav`

Codespaces: open the repo (uses `.devcontainer`), then run `uvicorn vclab.server:app --port 7860`.

## Dev
`pip install -e ".[dev]" && pytest` (not baked into the image, to keep it small).
Code: `src/vclab/`. Roadmap: `docs/ROADMAP.md`.
