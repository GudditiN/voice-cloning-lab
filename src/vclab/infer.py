"""CLI: vclab-infer "text" --ref data/raw/me.wav [--finetuned] -o outputs/out.wav"""
import argparse

import soundfile as sf

from . import config
from .engine import load_engine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text")
    ap.add_argument("--ref", nargs="+", required=True, help="reference wav(s) of the target voice")
    ap.add_argument("--language", default=config.LANGUAGE)
    ap.add_argument("--finetuned", action="store_true", help="use models/finetuned if present")
    ap.add_argument("-o", "--output", default=str(config.OUTPUT_DIR / "out.wav"))
    a = ap.parse_args()
    eng = load_engine(a.finetuned)
    audio = eng.synthesize(a.text, a.ref, a.language)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(a.output, audio, eng.sample_rate)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
