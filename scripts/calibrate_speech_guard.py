"""
Fit the minimum-speech guard thresholds (preprocess.SPEECH_GUARD).

The guard must never reject a real call, so both thresholds sit below the
minimum observed across all four corpora, with margin. Reports the margin so
the choice is visible rather than asserted.

Run on Kaggle, where the audio lives:
    python scripts/calibrate_speech_guard.py data/manifest.csv results/speech_guard.json

Reading the output:
- separable: true  -> clean air between real calls and silence; use
  recommended_MIN_RAW_RMS.
- separable: false -> some real clip is quieter than a 50 dB noise floor. Find it
  and look at it; it is probably corrupt and belongs in the integrity sweep, not
  in the threshold.
- dur_min on VIVAE protects demo moment 2 (the wordless groan). If any pain clip
  trims below MIN_TRIMMED_S, lower the threshold to suit it.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import denoise, load_audio, trim  # noqa: E402

SR = 16000


def metrics(path):
    """(raw RMS, seconds surviving trim), measured exactly as pipeline.process does:
    load -> denoise -> trim."""
    wav, meta = load_audio(path, target_sr=SR, return_meta=True)
    wav = trim(denoise(wav, sr=SR), sr=SR)
    return meta["raw_rms"], len(wav) / SR


def main(manifest_csv, out_json):
    np.random.seed(0)
    df = pd.read_csv(manifest_csv)
    rows = []
    for r in df.itertuples():
        try:
            rms, dur = metrics(r.path)
            rows.append({"corpus": r.corpus, "raw_rms": rms, "trimmed_s": dur})
        except Exception as e:
            rows.append({"corpus": r.corpus, "raw_rms": np.nan,
                         "trimmed_s": np.nan, "error": str(e)})
    m = pd.DataFrame(rows)
    n_err = int(m.raw_rms.isna().sum())
    m = m.dropna(subset=["raw_rms"])

    # Negative controls: what the guard has to reject.
    neg = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, sig in {
            "digital_silence": np.zeros(int(0.05 * SR), dtype=np.float32),
            "5ms_silence":     np.zeros(int(0.005 * SR), dtype=np.float32),
            "noise_floor_60db": (np.random.default_rng(0)
                                 .normal(0, 10 ** (-60 / 20), int(2.0 * SR))
                                 .astype(np.float32)),
            "noise_floor_50db": (np.random.default_rng(1)
                                 .normal(0, 10 ** (-50 / 20), int(2.0 * SR))
                                 .astype(np.float32)),
        }.items():
            p = os.path.join(tmp, f"neg_{name}.wav")
            sf.write(p, sig, SR)
            rms, dur = metrics(p)
            neg[name] = {"raw_rms": rms, "trimmed_s": dur}

    per_corpus = (m.groupby("corpus")
                   .agg(n=("raw_rms", "size"),
                        rms_min=("raw_rms", "min"),
                        rms_p01=("raw_rms", lambda s: float(s.quantile(0.01))),
                        rms_median=("raw_rms", "median"),
                        dur_min=("trimmed_s", "min"),
                        dur_p01=("trimmed_s", lambda s: float(s.quantile(0.01))))
                   .reset_index())

    rms_floor_real = float(m.raw_rms.min())
    dur_floor_real = float(m.trimmed_s.min())
    rms_ceil_neg = max(v["raw_rms"] for v in neg.values())
    dur_ceil_neg = max(v["trimmed_s"] for v in neg.values())

    out = {
        "n_clips": int(len(m)),
        "n_load_errors": n_err,
        "per_corpus": per_corpus.to_dict("records"),
        "negative_controls": neg,
        "real_min_raw_rms": rms_floor_real,
        "real_min_trimmed_s": dur_floor_real,
        "negative_max_raw_rms": rms_ceil_neg,
        "negative_max_trimmed_s": dur_ceil_neg,
        "separable": rms_ceil_neg < rms_floor_real,
        # Geometric midpoint when the two sets separate; otherwise fall back to
        # just under the real minimum and record that the margin is thin.
        "recommended_MIN_RAW_RMS": (float(np.sqrt(rms_ceil_neg * rms_floor_real))
                                    if rms_ceil_neg < rms_floor_real
                                    else rms_floor_real * 0.5),
        "recommended_MIN_TRIMMED_S": round(min(0.35, dur_floor_real * 0.8), 3),
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_corpus"}, indent=2))
    print(per_corpus.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else "results/speech_guard.json")
