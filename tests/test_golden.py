"""
Golden pipeline test: the six fixtures end to end, against tests/golden.json.

Decisions must match exactly (class, priorities, rules fired, every flag). Model
scores get a tolerance, because int8 Whisper and float kernels may differ in the
last digits across CPUs; a real change in a model or threshold moves them far more.

A deliberate change (refit, new threshold) is recorded by regenerating the file
and reviewing its git diff:

    UPDATE_GOLDEN=1 pytest tests/test_golden.py
"""

import json
import os

import pytest

from tests.conftest import GOLDEN_FIXTURES, ROOT

pytestmark = pytest.mark.pipeline

GOLDEN = ROOT / "tests" / "golden.json"

EXACT = ("request_class", "base_priority", "final_priority", "rules_fired", "escalated",
         "distress_fired", "speech_present", "asr_trusted", "energy_ok", "medication")
TOLERANCE = {"distress": 0.02, "non_speech_prob": 0.02, "arousal": 0.02,
             "class_confidence": 0.02, "asr_confidence": 0.05}
# Not asserted, but kept in the file for a human reading the diff.
VOLATILE = ("latency_ms", "audio_path", "bed")


def _strip(r):
    return {k: v for k, v in r.items() if k not in VOLATILE}


@pytest.fixture(scope="module")
def golden(fixture_results):
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps({k: _strip(r) for k, r in fixture_results.items()},
                                     indent=2, sort_keys=True) + "\n")
        pytest.skip(f"wrote {GOLDEN.relative_to(ROOT)}; review the diff")
    if not GOLDEN.exists():
        pytest.fail("tests/golden.json missing; run UPDATE_GOLDEN=1 pytest tests/test_golden.py")
    return json.loads(GOLDEN.read_text())


def _close(a, b, tol):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


@pytest.mark.parametrize("name", list(GOLDEN_FIXTURES))
def test_golden_decision(name, fixture_results, golden):
    got, want = fixture_results[name], golden[name]
    for k in EXACT:
        assert got[k] == want[k], f"{name}: {k} = {got[k]!r}, golden {want[k]!r}"
    for k, tol in TOLERANCE.items():
        assert _close(got[k], want[k], tol), f"{name}: {k} = {got[k]!r}, golden {want[k]!r} (±{tol})"
    assert set(got["class_probs"]) == set(want["class_probs"]), name
    for c, p in want["class_probs"].items():
        assert abs(got["class_probs"][c] - p) <= 0.02, f"{name}: P({c}) {got['class_probs'][c]} vs {p}"


def test_golden_thresholds_and_models_unchanged(fixture_results, golden):
    """A failure here means config or weights moved: regenerate golden deliberately."""
    got = fixture_results["chest"]
    assert got["thresholds"] == golden["chest"]["thresholds"]
    assert got["model_versions"] == golden["chest"]["model_versions"]


# What each fixture is FOR, independent of the recorded numbers: if a regenerated
# golden ever contradicts one of these, the regeneration was wrong.
INTENT = {
    "chest": dict(request_class="Emergency", final_priority=1),
    "knee": dict(request_class="Pain", final_priority=3, escalated=False),
    "groan": dict(final_priority=1, distress_fired=True, asr_trusted=False),
    "silence_5ms": dict(request_class="Unclassified", final_priority=2, speech_present=False,
                        energy_ok=False),
    "ward_noise": dict(request_class="Unclassified", final_priority=2, speech_present=False),
    "noise_floor": dict(request_class="Unclassified", final_priority=2, speech_present=False),
}


@pytest.mark.parametrize("name", list(INTENT))
def test_fixture_intent(name, fixture_results):
    got = fixture_results[name]
    for k, v in INTENT[name].items():
        assert got[k] == v, f"{name}: {k} = {got[k]!r}, intended {v!r}"
