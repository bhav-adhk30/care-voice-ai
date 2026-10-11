"""
Known gap for the live-microphone path (not the demo's file-upload path).

The acoustic branch crops to the FIRST 4 s and does not trim (the extraction
contract the heads were fitted on). A live recording that starts with a long
silence - the patient presses the button, then speaks - therefore gives WavLM a
window of pure silence. The text branch still hears the speech (Whisper gets the
whole trimmed clip), but the distress gate and the speech guard's non_speech
input are computed on silence, so a distressed caller would not escalate.

Marked xfail(strict=True): when the acoustic branch learns to find the speech
(trim on the acoustic path, VAD-located window, ...) this starts passing, the
strict xfail turns that into a failure, and the marker must be removed - with the
heads refit if the change alters what they see.

    pytest tests/test_live_mic.py
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "tts_chest_pain.wav"   # macOS TTS, 16 kHz
# The spec was "10 s: 8 s of silence then speech". The fixture phrase is 3.4 s,
# so cutting at 10 s would truncate the request itself ("...upper left" became
# Pain instead of Emergency). The clip is 8 s of silence plus the whole phrase.
LEAD_SILENCE_S = 8.0

needs_models = pytest.mark.skipif(
    not (ROOT / "models" / "faster-whisper-small").exists(),
    reason="models/ not present; run scripts/fetch_models.py")


@pytest.fixture(scope="module")
def pipe():
    from src.pipeline import CareVoicePipeline
    return CareVoicePipeline(model_dir=str(ROOT / "models"))


@pytest.fixture
def late_speech_clip(tmp_path):
    """8 s of digital silence, then the full spoken request (~11.4 s total)."""
    speech, sr = sf.read(FIXTURE, dtype="float32")
    lead = np.zeros(int(LEAD_SILENCE_S * sr), dtype=np.float32)
    clip = np.concatenate([lead, speech])
    path = tmp_path / "late_speech.wav"
    sf.write(path, clip, sr)
    return path


@needs_models
@pytest.mark.xfail(strict=True, reason="acoustic branch crops the first 4 s, which are "
                                       "silence; it never sees the speech")
def test_acoustic_branch_sees_speech_after_long_silence(pipe, late_speech_clip):
    r = pipe.process(late_speech_clip)
    # The text branch is expected to work already; checked so that the xfail
    # can only be explained by the acoustic branch.
    assert r["speech_present"]
    assert r["request_class"] == "Emergency"
    # The acoustic branch must have heard the speech, not 4 s of silence.
    assert r["non_speech_prob"] < 0.5
