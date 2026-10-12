"""
Prosodic distress: the HNR feature, the per-bed baseline store, and how the
pipeline uses them (cold start, who joins the baseline).
"""

import numpy as np
import pytest
import soundfile as sf

from src import prosody
from src.baseline import BedBaseline
from src.preprocess import load_audio_native
from src.rules import CONFIG
from tests.conftest import FIXTURES

MIN = CONFIG["BASELINE_MIN_CALLS"]


# ----------------------------------------------------------------- feature

def test_native_loader_keeps_sample_rate():
    wav, sr = load_audio_native(FIXTURES / "vivae_pain_groan.wav")
    assert sr == 44100 and wav.ndim == 1 and wav.dtype == np.float32


def test_hnr_on_speech_is_finite():
    wav, sr = load_audio_native(FIXTURES / "tts_chest_pain.wav")
    v = prosody.hnr_db(wav, sr)
    assert v is not None and np.isfinite(v)


def test_hnr_on_silence_is_none():
    wav, sr = load_audio_native(FIXTURES / "silence_5ms.wav")
    assert prosody.hnr_db(wav, sr) is None
    assert prosody.hnr_db(np.zeros(0, dtype=np.float32), 16000) is None


def test_hnr_depends_on_sample_rate():
    """Why the feature is computed at the native rate: resampling moves it."""
    import librosa
    wav, sr = load_audio_native(FIXTURES / "vivae_pain_groan.wav")
    native = prosody.hnr_db(wav, sr)
    down = prosody.hnr_db(librosa.resample(wav, orig_sr=sr, target_sr=16000), 16000)
    assert abs(native - down) > 0.5


def test_z_and_score():
    assert prosody.distress_z(10.0, 8.0, 2.0) == pytest.approx(1.0)
    assert prosody.distress_z(None, 8.0, 2.0) is None
    assert prosody.distress_z(10.0, 8.0, 0.0) is None          # no spread, no score
    assert prosody.distress_score(-1.5, "z") == -1.5
    assert prosody.distress_score(-1.5, "abs_z") == 1.5
    assert prosody.distress_score(-1.5, None) is None           # unset: no score
    assert prosody.distress_score(None, "z") is None
    with pytest.raises(ValueError):
        prosody.distress_score(1.0, "bogus")


# ----------------------------------------------------------------- baseline store

@pytest.fixture
def store(tmp_path):
    return BedBaseline(tmp_path / "b.db")


def test_cold_start_until_min_calls(store):
    for i, v in enumerate([8.0, 9.0, 10.0, 8.5]):
        store.add("7", 48000, v)
        assert not store.stats("7", 48000).ready, i
    store.add("7", 48000, 9.5)
    s = store.stats("7", 48000)
    assert s.n == MIN and s.ready
    assert s.median == pytest.approx(9.0)
    assert s.std == pytest.approx(np.std([8.0, 9.0, 10.0, 8.5, 9.5], ddof=1))


def test_no_spread_is_not_ready(store):
    for _ in range(MIN + 2):
        store.add("7", 48000, 9.0)
    assert not store.stats("7", 48000).ready


def test_keeps_last_max_calls(store):
    for v in range(CONFIG["BASELINE_MAX_CALLS"] + 25):
        store.add("7", 48000, float(v))
    s = store.stats("7", 48000)
    assert s.n == CONFIG["BASELINE_MAX_CALLS"]
    n_rows = store.con.execute("SELECT COUNT(*) FROM prosody_baseline WHERE bed='7'").fetchone()[0]
    assert n_rows == CONFIG["BASELINE_MAX_CALLS"]


def test_keyed_by_bed_and_sample_rate(store):
    for v in (8.0, 9.0, 10.0, 8.5, 9.5):
        store.add("7", 48000, v)
    assert store.stats("8", 48000).n == 0                     # other bed
    assert store.stats("7", 16000).n == 0                     # other rate: different HNR scale
    store.reset("7")
    assert store.stats("7", 48000).n == 0


# ----------------------------------------------------------------- pipeline

@pytest.mark.pipeline
def test_pipeline_cold_start_and_who_joins(pipe, tmp_path):
    """Ordinary calls build the baseline; Emergency and no-speech calls never
    join; during cold start there is no score and no acoustic escalation."""
    pipe.baseline = BedBaseline(tmp_path / "p.db")
    try:
        r = pipe.process(FIXTURES / "tts_chest_pain.wav", bed="3")      # Emergency
        assert not r["baseline_joined"] and r["distress"] is None
        r = pipe.process(FIXTURES / "ward_noise_4s.wav", bed="3")       # no speech
        assert not r["baseline_joined"]
        for i in range(MIN):
            r = pipe.process(FIXTURES / "tts_knee_pain.wav", bed="3")    # ordinary Pain call
            assert r["baseline_joined"] and r["baseline_n"] == i and not r["baseline_ready"]
            assert r["distress"] is None and not r["distress_fired"]
        # identical clips have no spread: still no score, still no escalation
        r = pipe.process(FIXTURES / "tts_knee_pain.wav", bed="3")
        assert r["baseline_n"] == MIN and not r["baseline_ready"] and r["distress"] is None
        r = pipe.process(FIXTURES / "tts_knee_pain.wav", bed=None)       # no bed: no baseline
        assert not r["baseline_joined"] and r["baseline_n"] == 0
    finally:
        pipe.baseline = None
