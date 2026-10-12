"""
Prosodic distress feature: harmonics-to-noise ratio (HNR), scored against the
bed's own calm baseline.

Replaces the WavLM distress gate, which never fired on the team's real voices
(distressed speech or a real groan; see results/groan_check_v2.csv and the
2026-10-11 tests). One interpretable feature instead of a 768-d classifier:
openSMILE eGeMAPSv02 functional HNRdBACF_sma3nz_amean - mean HNR (dB, from the
autocorrelation) over voiced frames.

Two design constraints, both measured:
  - Native sample rate. HNR moves 4-8 dB under resampling to 16 kHz and the
    calm/distress difference can vanish, so it is computed on
    preprocess.load_audio_native output, never on the 16 kHz pipeline audio.
  - Per-bed baseline. Absolute HNR differs a lot between people (calm medians
    5.1 to 9.0 dB across three speakers), so a call is scored as a z-score against
    the same bed's previous calls (src/baseline.py), not against a global norm.

Whether distress is scored one-sided (z) or two-sided (|z|), and at what
threshold, is rules.CONFIG["DISTRESS_SCORE"] / ["DISTRESS_HIGH"]; both are unset
until decided, so no acoustic escalation happens yet.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

CONFIG = {
    "feature_set": "eGeMAPSv02",
    "feature": "HNRdBACF_sma3nz_amean",
}

_SMILE = None


def _smile():
    global _SMILE
    if _SMILE is None:
        import opensmile
        _SMILE = opensmile.Smile(feature_set=getattr(opensmile.FeatureSet, CONFIG["feature_set"]),
                                 feature_level=opensmile.FeatureLevel.Functionals)
    return _SMILE


def hnr_db(wav: np.ndarray, sr: int) -> Optional[float]:
    """Mean HNR in dB over voiced frames, at the given (native) rate.
    None when there is no voiced speech to measure (silence gives NaN, a clip too
    short for openSMILE's windows warns and gives NaN)."""
    if len(wav) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = float(_smile().process_signal(np.asarray(wav, dtype=np.float32), sr)[CONFIG["feature"]].iloc[0])
    return v if np.isfinite(v) else None


def distress_z(hnr: Optional[float], median: Optional[float], std: Optional[float]) -> Optional[float]:
    """(hnr - bed median) / bed std, or None when either side is missing or the
    baseline has no spread."""
    if hnr is None or median is None or not std or std <= 0:
        return None
    return (hnr - median) / std


def distress_score(z: Optional[float], kind: Optional[str]) -> Optional[float]:
    """The single score rules.apply_rules sees: z ("z", one-sided), |z| ("abs_z"),
    or None when unset or unmeasurable."""
    if z is None or kind is None:
        return None
    if kind == "z":
        return z
    if kind == "abs_z":
        return abs(z)
    raise ValueError(f"unknown DISTRESS_SCORE {kind!r}")
