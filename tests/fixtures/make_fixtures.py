"""
Provenance of the test fixtures, and the generator for the synthetic ones.

    python tests/fixtures/make_fixtures.py            # regenerate synthetic fixtures
    python tests/fixtures/make_fixtures.py --sources  # also re-import the recorded ones

The fixtures are committed; this script only matters when one has to be rebuilt.
Changing any fixture invalidates tests/golden.json (rebuild it with
UPDATE_GOLDEN=1 pytest tests/test_golden.py, and review the diff).

| file                         | what                                   | source                        |
|------------------------------|----------------------------------------|-------------------------------|
| tts_chest_pain.wav           | "I have a severe pain in the upper     | macOS `say`, Samantha voice   |
|                              |  left side of my chest", calm          |                               |
| tts_knee_pain.wav            | "My knee hurts when I bend it", calm   | macOS `say`, default voice    |
| vivae_pain_groan.wav         | wordless pain vocalisation, 1.4 s      | VIVAE S02_pain_strong_021,    |
|                              |                                        | CC BY-NC 4.0 (see README)     |
| silence_5ms.wav              | 5 ms of digital zeros                  | generated here                |
| ward_noise_4s.wav            | washing machine, first 4 s             | ESC-50 1-21896-A-35,          |
|                              |                                        | CC BY-NC 3.0 (see README)     |
| noise_floor_4s.wav           | Gaussian noise at -60 dBFS, 4 s        | generated here, seed 0        |

TTS clips are intelligible to Whisper but are not patient speech: they are pipeline
fixtures, not evidence about clinical audio. The VIVAE clip is a positive in the
distress gate's training set and the ESC-50 clip is from the non_speech class's
training categories, so both are in-sample for the acoustic heads by construction.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
SR = 16000


def synthetic():
    sf.write(HERE / "silence_5ms.wav", np.zeros(int(0.005 * SR), dtype=np.float32), SR, subtype="PCM_16")
    noise = np.random.default_rng(0).normal(0, 10 ** (-60 / 20), 4 * SR).astype(np.float32)
    sf.write(HERE / "noise_floor_4s.wav", noise, SR, subtype="FLOAT")


def sources(knee_tts: Path, vivae_clip: Path, esc_clip: Path):
    import librosa
    knee, _ = librosa.load(knee_tts, sr=SR, mono=True)
    sf.write(HERE / "tts_knee_pain.wav", knee, SR, subtype="PCM_16")
    groan, sr = sf.read(vivae_clip)            # unmodified copy
    sf.write(HERE / "vivae_pain_groan.wav", groan, sr)
    ward, sr = sf.read(esc_clip)
    sf.write(HERE / "ward_noise_4s.wav", ward[: 4 * sr], sr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs=3, metavar=("KNEE_TTS", "VIVAE_CLIP", "ESC_CLIP"))
    a = ap.parse_args()
    synthetic()
    if a.sources:
        sources(*map(Path, a.sources))
    print("fixtures written to", HERE)
