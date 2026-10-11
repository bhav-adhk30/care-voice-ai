"""
Bad input must never crash the nurse station. Every case below must return a
decision - the call is routed, with a reason that says what went wrong - and log
it, never raise.

Expected behaviour for a call that cannot be processed: route it at P_NO_SPEECH
(P2), the same as a conventional call button. Someone pressed the button; the
system does not know why, so it must not drop the call or rank it below routine
requests.
"""

import logging
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from src.rules import CONFIG as RULES
from tests.conftest import FIXTURES, MODELS

pytestmark = pytest.mark.pipeline


def assert_routed_with_reason(r, caplog, *words):
    assert isinstance(r, dict), "process() must return a result dict"
    assert r["final_priority"] == RULES["P_NO_SPEECH"], r.get("reason")
    assert r["reason"], "a failed call needs a reason the dashboard can show"
    for w in words:
        assert w in r["reason"].lower(), (w, r["reason"])
    logged = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert logged, "the failure must be logged, not only returned"


def test_missing_file(pipe, tmp_path, caplog):
    r = pipe.process(tmp_path / "does_not_exist.wav")
    assert_routed_with_reason(r, caplog, "not found")


def test_zero_length_file(pipe, tmp_path, caplog):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    r = pipe.process(f)
    assert_routed_with_reason(r, caplog)


def test_unreadable_format(pipe, tmp_path, caplog):
    f = tmp_path / "not_audio.wav"
    f.write_text("this is not a wav file\n" * 50)
    r = pipe.process(f)
    assert_routed_with_reason(r, caplog)


def test_missing_model_file_refuses_to_start(tmp_path, monkeypatch):
    """Unlike bad audio, a missing model is a deployment fault: a pipeline without
    its distress gate must refuse to start - immediately, before loading anything
    else, naming the file - rather than run a silently weaker triage all shift."""
    from src.pipeline import CareVoicePipeline
    models = tmp_path / "models"
    models.mkdir()
    for p in MODELS.iterdir():
        if p.name != "distress_gate.joblib":
            os.symlink(p, models / p.name)

    def must_not_load(self):
        raise AssertionError("loaded a model before checking all model files exist")
    monkeypatch.setattr(CareVoicePipeline, "_load_asr", must_not_load)

    with pytest.raises(FileNotFoundError, match="distress_gate.joblib"):
        CareVoicePipeline(model_dir=str(models))


def test_sixty_second_clip(pipe, tmp_path):
    """A long recording (patient keeps talking, button held) is processed, not
    rejected, and the request at its start still classifies."""
    speech, sr = sf.read(FIXTURES / "tts_chest_pain.wav", dtype="float32")
    gap = np.zeros(sr, dtype=np.float32)
    clip = np.tile(np.concatenate([speech, gap]), 20)[: 60 * sr]
    f = tmp_path / "sixty_seconds.wav"
    sf.write(f, clip, sr)
    r = pipe.process(f)
    assert abs(r["duration_s"] - 60) < 0.1
    assert r["request_class"] == "Emergency"
    assert r["final_priority"] == 1
