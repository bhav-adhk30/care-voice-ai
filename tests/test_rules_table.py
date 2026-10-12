"""
apply_rules, exhaustively. No models needed: runs in about a second.

The table is 4 classes x distress fired or not x ASR trusted or not x speech
present or not = 32 rows, each with its expected priority and rule ids written out
by hand (not computed), plus the Unclassified rows the pipeline actually produces.

"Speech absent" for a trained class is expressed through the energy pre-filter
(energy_ok=False). The pipeline never produces that combination - a no-speech call
always reaches apply_rules as Unclassified with an empty transcript - but
apply_rules is a public function and the table pins down what it does with it.
"""

import itertools

import pytest

from src.rules import (CONFIG, UNCLASSIFIED, apply_rules, apply_starvation_timer,
                       starvation_priority)

# The shipped config leaves the prosodic threshold unset (no acoustic escalation),
# which would leave the escalation rules untested. Every table and property test
# runs under TCFG, an explicit score and threshold; test_unset_threshold_never_fires
# pins the shipped behaviour.
TCFG = {**CONFIG, "DISTRESS_SCORE": "z", "DISTRESS_HIGH": 1.0}
T = TCFG["DISTRESS_HIGH"]


def rules(*a, **kw):
    return apply_rules(*a, cfg=TCFG, **kw)


BASE = {"Emergency": 1, "Assistance": 2, "Pain": 3, "General": 3}
TEXT = {"Emergency": "i have a severe pain in my chest", "Assistance": "i feel too weak to stand",
        "Pain": "my knee hurts", "General": "could i get some water"}


def call(cls, distress_fired, trusted, speech, conf=0.9):
    return rules(
        base_priority=BASE[cls], request_class=cls, class_confidence=conf,
        distress=2.0 if distress_fired else 0.0,
        asr_confidence=-0.20 if trusted else -0.90,
        energy_ok=speech, non_speech_prob=0.0 if speech else None,
        energy_reason="" if speech else "signal level below floor",
        transcript=TEXT[cls])


R0, R1, R2, R4 = "R0_no_speech", "R1_distress_untrusted_transcript", "R2_distress", "R4_emergency_floor"

# (class, distress fired, ASR trusted, speech present) -> (final priority, rules fired)
TABLE = {
    # Emergency: base P1, floor P2 never needed; distress cannot go above P1, and
    # Rule 0 may not demote it (min(base, P_NO_SPEECH))
    ("Emergency", True,  True,  True):  (1, [R2]),
    ("Emergency", True,  False, True):  (1, [R1]),
    ("Emergency", False, True,  True):  (1, []),
    ("Emergency", False, False, True):  (1, []),
    ("Emergency", True,  True,  False): (1, [R0]),
    ("Emergency", True,  False, False): (1, [R0]),
    ("Emergency", False, True,  False): (1, [R0]),
    ("Emergency", False, False, False): (1, [R0]),
    # Assistance: base P2
    ("Assistance", True,  True,  True):  (1, [R2]),
    ("Assistance", True,  False, True):  (1, [R1]),
    ("Assistance", False, True,  True):  (2, []),
    ("Assistance", False, False, True):  (2, []),
    ("Assistance", True,  True,  False): (2, [R0]),
    ("Assistance", True,  False, False): (2, [R0]),
    ("Assistance", False, True,  False): (2, [R0]),
    ("Assistance", False, False, False): (2, [R0]),
    # Pain: base P3
    ("Pain", True,  True,  True):  (1, [R2]),
    ("Pain", True,  False, True):  (1, [R1]),
    ("Pain", False, True,  True):  (3, []),
    ("Pain", False, False, True):  (3, []),
    ("Pain", True,  True,  False): (2, [R0]),
    ("Pain", True,  False, False): (2, [R0]),
    ("Pain", False, True,  False): (2, [R0]),
    ("Pain", False, False, False): (2, [R0]),
    # General: base P3
    ("General", True,  True,  True):  (1, [R2]),
    ("General", True,  False, True):  (1, [R1]),
    ("General", False, True,  True):  (3, []),
    ("General", False, False, True):  (3, []),
    ("General", True,  True,  False): (2, [R0]),
    ("General", True,  False, False): (2, [R0]),
    ("General", False, True,  False): (2, [R0]),
    ("General", False, False, False): (2, [R0]),
}


def test_table_is_complete():
    assert len(TABLE) == 32
    assert set(TABLE) == set(itertools.product(BASE, [True, False], [True, False], [True, False]))


@pytest.mark.parametrize("key", list(TABLE), ids=lambda k: "-".join(
    [k[0], "distress" if k[1] else "calm", "trusted" if k[2] else "untrusted",
     "speech" if k[3] else "nospeech"]))
def test_rules_table(key):
    want_p, want_rules = TABLE[key]
    d = call(*key)
    assert (d.final_priority, d.rules_fired) == (want_p, want_rules), d.reason


# The rows the pipeline really sends for an empty transcript: Unclassified at P2.
UNCLASSIFIED_ROWS = [
    # (distress, non_speech, energy_ok, asr_confidence) -> final, rules
    (dict(distress=2.0, non_speech_prob=0.99, energy_ok=True, asr_confidence=-10.0), 1, [R1]),  # groan
    (dict(distress=2.0, non_speech_prob=0.05, energy_ok=True, asr_confidence=-10.0), 1, [R1]),
    (dict(distress=0.05, non_speech_prob=0.99, energy_ok=True, asr_confidence=-10.0), 2, [R0]),  # fan
    (dict(distress=0.05, non_speech_prob=0.05, energy_ok=True, asr_confidence=-10.0), 2, []),
    (dict(distress=0.00, non_speech_prob=None, energy_ok=False, asr_confidence=None), 2, [R0]),  # silence
]


@pytest.mark.parametrize("kw,want_p,want_rules", UNCLASSIFIED_ROWS)
def test_unclassified_rows(kw, want_p, want_rules):
    d = rules(base_priority=CONFIG["P_NO_SPEECH"], request_class=UNCLASSIFIED,
                    class_confidence=0.0, transcript="", **kw)
    assert (d.final_priority, d.rules_fired) == (want_p, want_rules), d.reason


def test_audio_error_routes_like_a_call_button():
    d = rules(CONFIG["P_NO_SPEECH"], UNCLASSIFIED, 0.0, 0.0, transcript="",
                    audio_error="file not found: x.wav")
    assert (d.final_priority, d.rules_fired) == (2, ["R0_audio_error"])
    assert "file not found" in d.reason and not d.escalated


def test_absent_asr_confidence_is_untrusted():
    d = rules(3, "Pain", 0.9, 0.0, asr_confidence=None, non_speech_prob=0.0,
                    transcript="my knee hurts")
    assert d.asr_trusted is False


def test_unclassified_invariant_both_directions():
    with pytest.raises(ValueError):
        rules(2, UNCLASSIFIED, 0.0, 0.0, transcript="hello")
    with pytest.raises(ValueError):
        rules(3, "General", 0.9, 0.0, transcript="  ")


# ----------------------------------------------------------------- properties

CLASSES = list(BASE) + [UNCLASSIFIED]
GRID = list(itertools.product(
    CLASSES,
    [0.30, 0.99],                    # class confidence
    [None, 0.0, 0.5, T - 1e-6, T, 2.5],  # distress score, either side of the threshold
    [None, -10.0, -0.9, -0.2],       # asr confidence
    [True, False],                   # energy ok
    [None, 0.0, 0.5, 0.95],          # non_speech
))


def _run(cls, conf, distress, asr, energy, ns):
    unc = cls == UNCLASSIFIED
    base = CONFIG["P_NO_SPEECH"] if unc else BASE[cls]
    return base, rules(base, cls, 0.0 if unc else conf, distress, asr_confidence=asr,
                             energy_ok=energy, non_speech_prob=ns,
                             transcript="" if unc else TEXT[cls])


def test_priority_never_increases_in_number():
    """Rules may only make a call more urgent. Collects every violation."""
    bad = []
    for args in GRID:
        base, d = _run(*args)
        if d.final_priority > base:
            bad.append((args, base, d.final_priority, d.rules_fired))
    assert not bad, f"{len(bad)} inputs made a call LESS urgent, e.g. {bad[:3]}"


def test_confident_calm_class_cannot_cancel_fired_distress():
    for cls in BASE:
        for asr in (None, -0.9, -0.2):
            d = rules(BASE[cls], cls, 0.99, 2.0, asr_confidence=asr,
                            non_speech_prob=0.0, transcript=TEXT[cls])
            assert d.final_priority == 1 and d.distress_fired, (cls, asr, d.reason)


def test_emergency_never_exits_worse_than_p2():
    for args in GRID:
        if args[0] == "Emergency":
            _, d = _run(*args)
            assert d.final_priority <= 2, (args, d.reason)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_starvation_timer_monotonic(p):
    prev = p
    for waited in range(0, 4 * 3600, 5):
        eff, bumps = starvation_priority(p, waited)
        assert 1 <= eff <= prev, (p, waited, eff, prev)   # never less urgent with time
        assert eff == p - len(bumps)
        prev = eff
    assert prev == 1 or p == 1, "a call that waits long enough must reach P1"


def test_starvation_single_step_never_raises_number():
    for p in (1, 2, 3, 4):
        for waited in (0, 59, 61, 299, 301, 899, 901, 1799, 1801, 10**6):
            new_p, bumped = apply_starvation_timer(p, waited)
            assert new_p <= p and bumped == (new_p < p)


# ----------------------------------------------------------------- Rule 5: care requests

from src.mapping import care_request_tag  # noqa: E402

CARE = ["washroom"]


@pytest.mark.parametrize("cls,base,want", [
    ("General", 3, 2), ("Pain", 3, 2), ("Assistance", 2, 2), ("Emergency", 1, 1)])
def test_care_request_floors_at_p2_never_lowers(cls, base, want):
    d = rules(base, cls, 0.9, 0.0, asr_confidence=-0.3, non_speech_prob=0.0,
                    transcript="i need to go to the washroom", care_request=CARE)
    assert d.final_priority == want, d.reason
    assert ("R5_care_request" in d.rules_fired) == (want < base)


def test_care_request_ignored_without_transcript():
    d = rules(CONFIG["P_NO_SPEECH"], UNCLASSIFIED, 0.0, 0.0, asr_confidence=-10.0,
                    non_speech_prob=0.99, transcript="", care_request=CARE)
    assert "R5_care_request" not in d.rules_fired


@pytest.mark.parametrize("text", [
    "I need to go to washroom can somebody get me there", "Can someone help me to the toilet",
    "I need help getting out of bed", "Please help me up", "I need a bedpan", "Help me sit up please"])
def test_care_request_tag_catches_requests(text):
    assert care_request_tag(text), text


@pytest.mark.parametrize("text", [
    "My knee hurts when I get up", "It hurts when I walk", "I have a severe pain in my chest",
    "Could I get some water", "Can I have my painkillers please"])
def test_care_request_tag_ignores_symptoms(text):
    assert not care_request_tag(text), text


def test_care_request_tag_no_false_positives_on_corpus():
    """None of the 706 corpus phrases is a care request, so any match changes a
    symptom's priority by mistake. Measured at 0 when the rule was written."""
    import pandas as pd
    from tests.conftest import ROOT
    ph = pd.read_csv(ROOT / "results" / "phrases.csv")
    hits = [p for p in ph.phrase if care_request_tag(p)]
    assert not hits, hits[:5]



def test_unset_threshold_never_fires():
    """Shipped config: DISTRESS_HIGH is None until a score is chosen, so no
    distress value escalates, and the speech guard treats distress as absent."""
    assert CONFIG["DISTRESS_HIGH"] is None
    for dist in (None, 0.0, 5.0, 100.0):
        d = apply_rules(3, "Pain", 0.9, dist, asr_confidence=-0.9, non_speech_prob=0.0,
                        transcript="my knee hurts")
        assert not d.distress_fired and d.final_priority == 3, d.reason


def test_cold_start_none_score_never_fires():
    d = rules(3, "Pain", 0.9, None, asr_confidence=-0.9, non_speech_prob=0.0,
              transcript="my knee hurts")
    assert not d.distress_fired and d.final_priority == 3
