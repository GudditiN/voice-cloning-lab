"""Fine-tune XTTS-v2 GPT on data/processed (metadata.csv). Needs a CUDA GPU (Colab T4 is enough).

Adapted from the Coqui TTS recipe recipes/ljspeech/xtts_v2/train_gpt_xtts.py. Not exercised on a GPU in CI.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import config

HF_REPO = "coqui/XTTS-v2"
BASE_FILES = ["dvae.pth", "mel_stats.pth", "model.pth", "vocab.json", "config.json"]


def download_base(dest: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    return {f: Path(hf_hub_download(HF_REPO, f, local_dir=dest)) for f in BASE_FILES}


def train(epochs: int = 6, batch_size: int = 3, grad_accum: int = 84, lr: float = 5e-6, language: str = config.LANGUAGE) -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning needs a GPU. On Colab: Runtime > Change runtime type > T4 GPU.")

    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

    config.ensure_dirs()
    base = download_base(config.MODELS_DIR / "xtts_base")
    run_dir = config.MODELS_DIR / "runs"

    dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="me",
        path=str(config.PROCESSED_DIR),
        meta_file_train="metadata.csv",
        language=language,
    )
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6s
        min_conditioning_length=66150,  # 3s
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6s
        max_text_length=200,
        mel_norm_file=str(base["mel_stats.pth"]),
        dvae_checkpoint=str(base["dvae.pth"]),
        xtts_checkpoint=str(base["model.pth"]),
        tokenizer_file=str(base["vocab.json"]),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_cfg = XttsAudioConfig(sample_rate=config.SAMPLE_RATE, dvae_sample_rate=config.SAMPLE_RATE, output_sample_rate=config.OUTPUT_SAMPLE_RATE)
    cfg = GPTTrainerConfig(
        output_path=str(run_dir),
        model_args=model_args,
        run_name="xtts_ft",
        project_name="vclab",
        audio=audio_cfg,
        batch_size=batch_size,
        batch_group_size=48,
        eval_batch_size=batch_size,
        num_loader_workers=2,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        epochs=epochs,
    )
    train_samples, eval_samples = load_tts_samples([dataset], eval_split=True, eval_split_max_size=cfg.eval_split_max_size, eval_split_size=cfg.eval_split_size)
    if not train_samples:
        raise RuntimeError("No training samples: run data prep and make sure metadata.csv has rows.")

    model = GPTTrainer.init_from_config(cfg)
    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False, grad_accum_steps=grad_accum),
        cfg,
        output_path=str(run_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    ckpts = sorted(run_dir.rglob("best_model*.pth"), key=lambda p: p.stat().st_mtime)
    ckpts = ckpts or sorted(run_dir.rglob("checkpoint_*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise RuntimeError(f"Training finished but no checkpoint found under {run_dir}")
    out = config.MODELS_DIR / "finetuned"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpts[-1], out / "model.pth")
    shutil.copy(base["vocab.json"], out / "vocab.json")
    shutil.copy(base["config.json"], out / "config.json")
    shutil.copy(base["mel_stats.pth"], out / "mel_stats.pth")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=84)
    ap.add_argument("--language", default=config.LANGUAGE)
    a = ap.parse_args()
    print("saved to", train(a.epochs, a.batch_size, a.grad_accum, language=a.language))


if __name__ == "__main__":
    main()
