"""
Shared test setup.

One CareVoicePipeline per session: loading it costs ~15 s and ~1.5 GB, and the
demo machine has 8 GB. Its model loaders are wrapped with counters before
construction, so test_models_load_once can check that nothing ever reloads.

    pytest tests                     # whole suite (target: under 2 min on CPU)
    pytest tests -m "not pipeline"   # rules only, no models needed, ~1 s
    UPDATE_GOLDEN=1 pytest tests/test_golden.py   # rewrite tests/golden.json
"""

import os
from collections import Counter
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
MODELS = ROOT / "models"

# The six golden fixtures (see tests/fixtures/make_fixtures.py for provenance).
GOLDEN_FIXTURES = {
    "chest": "tts_chest_pain.wav",
    "knee": "tts_knee_pain.wav",
    "groan": "vivae_pain_groan.wav",
    "silence_5ms": "silence_5ms.wav",
    "ward_noise": "ward_noise_4s.wav",
    "noise_floor": "noise_floor_4s.wav",
}

LOADERS = ("_load_asr", "_load_text", "_load_emotion", "_load_acoustic")
LOAD_COUNTS = Counter()


def pytest_configure(config):
    config.addinivalue_line("markers", "pipeline: needs models/ and loads the pipeline")


def pytest_collection_modifyitems(config, items):
    if (MODELS / "faster-whisper-small").exists():
        return
    skip = pytest.mark.skip(reason="models/ not present; run scripts/fetch_models.py")
    for item in items:
        if "pipeline" in item.keywords:
            item.add_marker(skip)


def _counted(name, fn):
    # Keyed per instance: other tests build their own pipelines (missing-model case).
    def wrapper(self, *a, **kw):
        LOAD_COUNTS[(id(self), name)] += 1
        return fn(self, *a, **kw)
    return wrapper


@pytest.fixture(scope="session")
def pipe():
    from src.pipeline import CareVoicePipeline
    for name in LOADERS:
        setattr(CareVoicePipeline, name, _counted(name, getattr(CareVoicePipeline, name)))
    return CareVoicePipeline(model_dir=str(MODELS))


@pytest.fixture(scope="session")
def load_counts():
    return LOAD_COUNTS


@pytest.fixture(scope="session")
def fixture_results(pipe):
    """process() output for each golden fixture, computed once and shared."""
    return {k: pipe.process(FIXTURES / f) for k, f in GOLDEN_FIXTURES.items()}
