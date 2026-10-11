"""
Lifecycle: models load once and stay loaded; the same audio gives the same answer.
"""

import pytest

from tests.conftest import FIXTURES, GOLDEN_FIXTURES, LOADERS

pytestmark = pytest.mark.pipeline


def test_models_load_once(pipe, load_counts):
    """Counters wrap every loader (conftest). Construction loads each model once;
    calls and the display-only emotion toggle must never reload anything else."""
    count = lambda name: load_counts[(id(pipe), name)]
    assert {n: count(n) for n in LOADERS} == {
        "_load_asr": 1, "_load_text": 1, "_load_acoustic": 1, "_load_emotion": 0}

    for f in ("tts_chest_pain.wav", "silence_5ms.wav", "tts_chest_pain.wav"):
        pipe.process(FIXTURES / f)
    pipe.keep_warm()
    assert {n: count(n) for n in LOADERS} == {
        "_load_asr": 1, "_load_text": 1, "_load_acoustic": 1, "_load_emotion": 0}

    try:
        pipe.set_emotion(True)
        pipe.set_emotion(True)                      # already on: must not reload
        pipe.process(FIXTURES / "tts_chest_pain.wav")
        assert count("_load_emotion") == 1
        assert {n: count(n) for n in LOADERS if n != "_load_emotion"} == {
            "_load_asr": 1, "_load_text": 1, "_load_acoustic": 1}
    finally:
        pipe.set_emotion(False)                     # leave the session pipeline as found


VOLATILE = ("latency_ms",)


@pytest.mark.parametrize("name", list(GOLDEN_FIXTURES))
def test_pipeline_deterministic(name, pipe, fixture_results):
    """Second run of each fixture equals the first, exactly, every field but timing."""
    again = pipe.process(FIXTURES / GOLDEN_FIXTURES[name])
    first = fixture_results[name]
    diff = {k: (first[k], again[k]) for k in first
            if k not in VOLATILE and first[k] != again[k]}
    assert not diff, f"{name}: non-deterministic fields {diff}"
