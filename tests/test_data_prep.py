import numpy as np

from vclab import data_prep

SR = 22050


def tone(seconds, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_split_on_silence_separates_phrases():
    audio = np.concatenate([silence(0.5), tone(3), silence(1.0), tone(4), silence(0.5)])
    clips = data_prep.split_on_silence(audio, SR)
    assert len(clips) == 2
    assert 3 <= len(clips[0]) / SR <= 3.5


def test_short_bursts_dropped():
    audio = np.concatenate([tone(0.5), silence(1.0), tone(3)])
    assert len(data_prep.split_on_silence(audio, SR)) == 1


def test_long_run_chopped():
    clips = data_prep.split_on_silence(tone(31), SR)
    assert all(len(c) / SR <= data_prep.MAX_CLIP_S + 0.3 for c in clips)
    assert len(clips) >= 2


def test_normalize_and_quality():
    out = data_prep.normalize(tone(1, 0.2))
    assert abs(np.abs(out).max() - 0.95) < 1e-3
    assert data_prep.quality_report(tone(1, 0.05))["too_quiet"]
    assert data_prep.quality_report(np.ones(SR, dtype=np.float32))["clipping"]
