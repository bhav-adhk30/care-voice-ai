"""
Escalation and safety-override rules for Care-Voice AI.

This is deliberately NOT a trained model. See CLAUDE.md section 2.1: training a
fusion head needs clips labelled with both a request type and an urgency level,
and no such corpus exists. The combination is an explicit rule instead, which in
a clinical system is also easier to audit than learned weights.

Every threshold below was measured, not chosen. Provenance is in the comments.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional


# ---------------------------------------------------------------------------
# Thresholds. All measured on held-out data — see results/*.json.
# ---------------------------------------------------------------------------

CONFIG = {
    # Distress gate, v2 (notebooks/11_head_refit.ipynb, results/head_refit_v2.json).
    # WavLM, no padding, mean-pooled over real frames, RMS 0.05 (extraction
    # contract "no_pad_meanpool_rms0.05"); logistic regression, VIVAE pain/fear
    # against other VIVAE vocalisations, CREMA-D neutral, ESC-50 ward noise and
    # silence, VIVAE deduplicated by content hash (480 duplicates removed).
    # Held-out AUC 0.863 against all negatives, 0.629 within VIVAE alone (pain
    # versus other intense vocalisations - it cannot tell pain from triumph).
    # At 0.4028: 63.7% distress recall; false alarms 0.75% on medical speech,
    # 1.7% on ward noise, 0% on silence (results/operating_curve_distress_v2.csv).
    #
    # Replaces 0.0677, which came from a curve whose negatives were a different
    # corpus (CREMA-D) padded differently: clip duration alone separates VIVAE
    # from CREMA-D at AUC 0.9965 (results/padding_diagnostic.json), so that
    # threshold measured corpus identity, not distress, and sat below every
    # observed score.
    "DISTRESS_HIGH": 0.4028,

    # Speech guard: the arousal head's non_speech class (7th class, trained on
    # ESC-50 ward noise and silence against CREMA-D speech; F1 0.988 out of fold
    # at 0.5). Raised to 0.90 and made one of THREE conditions (speech_guard):
    # at 0.5 alone it classed 7 of 32 clearly intelligible TTS requests as
    # non-speech, including "I can't breathe properly", with scores up to 1.000.
    # The class learned "unlike CREMA-D", not "not speech", so it is never
    # trusted on its own.
    "NON_SPEECH_HIGH": 0.90,

    # The arousal head's fear probability does NOT escalate. At a 2% false-alarm
    # budget on medical speech it recalls 14.5% of fear, against the distress
    # gate's 63.7% at 0.75% (results/operating_curve_arousal_v2.csv). It is
    # reported on the dashboard for context only. There is no AROUSAL_HIGH.

    # Whisper mean token log-probability, 20th percentile of the corpus.
    # Below it: 18.3% of transcripts are badly wrong (WER > 0.5) and request-type
    # accuracy falls to 0.707, against 1.5% / 0.86 above it.
    # Whisper hallucinates fluent unrelated sentences on poor audio, which is a
    # safety failure rather than an accuracy one — hence this gate.
    "ASR_MIN": -0.4934,

    # Priority is floored (made more urgent) at this level, never raised.
    "P_IMMEDIATE": 1,

    # An Emergency classification is never queued worse than P2, however calm
    # the voice sounds.
    "EMERGENCY_FLOOR": 2,

    # Waiting time after which a call is escalated one level, in seconds.
    # Prevents low-priority requests starving behind a busy ward.
    "STARVATION_SECONDS": {1: 60, 2: 300, 3: 900, 4: 1800},

    # A call with no usable speech carries no triage information at all. It is
    # routed at this fixed level: the behaviour of a conventional nurse-call
    # button, which signals a press and nothing more. Neither suppressed to the
    # bottom of the queue nor escalated on an acoustic score computed from a
    # noise floor.
    "P_NO_SPEECH": 2,
}


# The distress gate is BINARY, not graded. On the v1 features its score
# correlated with annotated intensity only weakly (Spearman rho = 0.16,
# n = 1565 including duplicates); not re-measured on v2. It fires or it doesn't.
DISTRESS_IS_BINARY = True

# The v2 gate still cannot separate anger from distress. Out of fold, anger
# averages 0.417 against pain 0.465 and fear 0.480, and anger's false-alarm rate
# is above the overall rate at every threshold, so anger cannot be filtered out
# acoustically.
#
# There is NO anger suppression. A text-emotion suppressor was tried and removed:
# the pretrained emotion model (j-hartmann/emotion-english-distilroberta-base)
# scores "I feel cold" as anger 0.97 but a plainly angry sentence ("This is
# ridiculous, I have been pressing this button for twenty minutes!") only 0.67,
# and 27 of 706 genuine medical requests score anger above 0.5
# (results/text_emotion_on_phrases.csv). Any of those 27 would have lost its distress
# escalation had the gate fired. Its output is displayed on the dashboard and never affects
# priority. The accepted cost: an angry patient whose voice fires the distress
# gate is escalated - a false alarm inside the measured budget, not a miss.
ANGER_NOT_SEPARABLE_ACOUSTICALLY = True


# Pipeline sentinel for a call with no transcript. NOT a trained class: it is
# deliberately absent from mapping.CLASSES and CLASS_BASE_PRIORITY, so nothing that
# indexes the label space can see it.
#
# The condition is an empty transcript, not the speech guard. A wordless groan
# passes the guard on energy but gives Whisper nothing to decode; running
# ClinicalBERT on "" would assign it a class out of distribution (it predicted
# Emergency). Instead it is Unclassified at P_NO_SPEECH and the acoustic rules
# still apply, so a distressed groan goes Unclassified P2 -> distress gate -> P1.
UNCLASSIFIED = "Unclassified"


@dataclass
class Decision:
    """Everything the dashboard needs to explain itself."""
    final_priority: int
    base_priority: int
    reason: str
    escalated: bool
    distress_fired: bool
    asr_trusted: bool
    speech_present: bool = True
    acoustic_used: bool = True
    # Both halves of the speech guard, so the dashboard can show which decided.
    energy_ok: bool = True
    non_speech_prob: Optional[float] = None
    # Which rules fired, as stable ids (R0..R4), so the dashboard and the audit
    # log can say exactly which branch of this function decided.
    rules_fired: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def speech_guard(energy_ok: bool, non_speech_prob: Optional[float],
                 transcript: str, asr_confidence: Optional[float],
                 distress: Optional[float], energy_reason: str = "",
                 cfg: dict = CONFIG) -> tuple[bool, str]:
    """
    Is there speech (or a vocalisation) in this call? Returns (present, reason).

    Conservative by construction: the cost of wrongly calling a real request
    "no speech" is a lost transcript (an Emergency at P2 instead of P1), so a
    call is no-speech only when ALL of these agree:
      1. the arousal head's non_speech class >= NON_SPEECH_HIGH,
      2. Whisper returned nothing, or a transcript below ASR_MIN,
      3. the distress gate did not fire (a groan has no words but must escalate).
    The energy pre-filter (preprocess.energy_present) short-circuits digital
    silence before anything runs; then there is nothing else to read.
    """
    if not energy_ok:
        return False, energy_reason or "signal level below floor"
    if non_speech_prob is None or non_speech_prob < cfg["NON_SPEECH_HIGH"]:
        return True, ""
    if transcript.strip() and asr_confidence is not None and asr_confidence >= cfg["ASR_MIN"]:
        return True, ""
    if distress is not None and distress >= cfg["DISTRESS_HIGH"]:
        return True, ""
    heard = (f"untrusted transcript {transcript.strip()!r}" if transcript.strip()
             else "no transcript")
    return False, (f"non_speech class {non_speech_prob:.3f} >= {cfg['NON_SPEECH_HIGH']}, "
                   f"{heard}, distress below threshold")


def apply_rules(
    base_priority: int,
    request_class: str,
    class_confidence: float,
    distress: float,
    asr_confidence: Optional[float] = None,
    energy_ok: bool = True,
    non_speech_prob: Optional[float] = None,
    energy_reason: str = "",
    cfg: dict = CONFIG,
    *,
    transcript: str,
    asr_heard: Optional[str] = None,
) -> Decision:
    """
    Combine the two branches into a final priority.

    Takes the MAXIMUM urgency of the available signals, never an average.
    Averaging lets a calm-sounding text classification cancel out a distress
    signal, which is the one failure mode this system exists to prevent.

    Lower priority number is more urgent.

    base_priority    1..4, from CLASS_BASE_PRIORITY via the predicted class
    request_class    "Emergency" | "Pain" | "Assistance" | "General", or
                     UNCLASSIFIED exactly when the transcript is empty
    class_confidence softmax probability of the predicted class, 0..1
    distress         distress-gate probability, 0..1
    asr_confidence   mean token log-probability from Whisper, or None
    energy_ok        preprocess.energy_present pre-filter result
    non_speech_prob  arousal head's non_speech probability, or None if the
                     pre-filter stopped the clip before WavLM ran
    energy_reason    why the pre-filter failed, shown on the dashboard
    transcript       the ASR output; required, to enforce the invariant below
    asr_heard        what Whisper decoded before the pipeline blanked a no-speech
                     transcript; used by the speech guard and its reason string.
                     Defaults to `transcript`.

    There is no arousal input: the fear probability is displayed by the
    pipeline but never affects priority (see CONFIG).
    """
    # Invariant, both directions: Unclassified <=> empty transcript. Raised rather
    # than asserted so it also holds under python -O.
    empty = transcript.strip() == ""
    if (request_class == UNCLASSIFIED) != empty:
        raise ValueError(
            f"request_class={request_class!r} with "
            f"{'an empty' if empty else 'a non-empty'} transcript: {UNCLASSIFIED!r} "
            "is required exactly when the transcript is empty")

    # --- Rule 0: no speech ---------------------------------------------------
    # Short-circuits everything: a call with no speech carries no triage
    # information, so it is routed at P_NO_SPEECH like a conventional call
    # button, and Rule 1 cannot fire on it.
    #
    # A wordless groan must NOT land here: it has to pass the guard and reach
    # Rule 1. speech_guard() requires the distress gate to be silent before it
    # calls anything no-speech, so a groan that fires the gate always gets
    # through, whatever the non_speech class says about it.
    heard = transcript if asr_heard is None else asr_heard
    speech_present, speech_reason = speech_guard(energy_ok, non_speech_prob, heard,
                                                 asr_confidence, distress,
                                                 energy_reason, cfg)
    if not speech_present:
        return Decision(
            final_priority=cfg["P_NO_SPEECH"],
            base_priority=base_priority,
            reason=("no speech detected" + (f" ({speech_reason})" if speech_reason else "")
                    + "; routed as an unclassified call, equivalent to a "
                      "conventional call button"),
            escalated=False,
            distress_fired=False,
            asr_trusted=False,
            speech_present=False,
            acoustic_used=False,
            energy_ok=energy_ok,
            non_speech_prob=non_speech_prob,
            rules_fired=["R0_no_speech"],
        )

    priority = base_priority
    reasons, fired = [], []

    asr_trusted = asr_confidence is None or asr_confidence >= cfg["ASR_MIN"]
    if not asr_trusted:
        reasons.append("ASR confidence below threshold; transcript not trusted")

    distress_fired = distress >= cfg["DISTRESS_HIGH"]

    # --- Rule 1: untrusted transcript plus distress -> immediate -------------
    # The text branch cannot be relied on and the acoustic path is the only
    # signal left. This is the wordless-patient case and the hallucination case
    # at the same time.
    if not asr_trusted and distress_fired:
        priority = min(priority, cfg["P_IMMEDIATE"])
        reasons.append("distress detected with an unreliable transcript")
        fired.append("R1_distress_untrusted_transcript")

    # --- Rule 2: distress gate ----------------------------------------------
    # No anger suppression: see ANGER_NOT_SEPARABLE_ACOUSTICALLY above.
    elif distress_fired:
        priority = min(priority, cfg["P_IMMEDIATE"])
        reasons.append(f"distress gate fired ({distress:.3f} >= "
                       f"{cfg['DISTRESS_HIGH']})")
        fired.append("R2_distress")

    # (Rule 3, arousal raises one level, was removed in v2: 14.5% fear recall at
    # a 2% medical false-alarm budget does not justify an escalation path.)

    # --- Rule 4: Emergency floor --------------------------------------------
    # Never queue a predicted Emergency worse than P2, regardless of how the
    # voice sounds. A patient can describe a cardiac event perfectly calmly.
    if request_class == "Emergency":
        if priority > cfg["EMERGENCY_FLOOR"]:
            reasons.append("Emergency class floor applied")
            fired.append("R4_emergency_floor")
        priority = min(priority, cfg["EMERGENCY_FLOOR"])

    if not reasons:
        reasons.append("no escalation; base priority from request class")

    return Decision(
        final_priority=priority,
        base_priority=base_priority,
        reason="; ".join(reasons),
        escalated=priority < base_priority,
        distress_fired=distress_fired,
        asr_trusted=asr_trusted,
        speech_present=True,
        acoustic_used=True,
        energy_ok=energy_ok,
        non_speech_prob=non_speech_prob,
        rules_fired=fired,
    )


def apply_starvation_timer(priority: int, waiting_seconds: float,
                           cfg: dict = CONFIG) -> tuple[int, bool]:
    """
    Escalate a call that has waited too long at its current priority.

    Without this, a P4 request behind a busy ward never gets answered. Called
    by the routing engine on every refresh, not by apply_rules.
    """
    limit = cfg["STARVATION_SECONDS"].get(priority)
    if limit is not None and waiting_seconds > limit and priority > 1:
        return priority - 1, True
    return priority, False


def starvation_priority(final_priority: int, waiting_seconds: float,
                        cfg: dict = CONFIG) -> tuple[int, list[int]]:
    """
    Effective queue priority after waiting: walk apply_starvation_timer level by
    level, each level held for its STARVATION_SECONDS before the call is bumped.
    Returns (priority, levels it was bumped from). Never less urgent than
    final_priority, and never less urgent the longer a call waits.
    """
    p, t, bumps = final_priority, waiting_seconds, []
    while True:
        new_p, bumped = apply_starvation_timer(p, t, cfg)
        if not bumped:
            return p, bumps
        t -= cfg["STARVATION_SECONDS"][p]
        bumps.append(p)
        p = new_p


if __name__ == "__main__":
    cases = [
        ("calm knee pain",
         dict(base_priority=3, request_class="Pain", class_confidence=0.95,
              distress=0.01, non_speech_prob=0.001, asr_confidence=-0.2,
              transcript="my knee hurts when i bend it")),
        ("same phrase, distressed voice (gate fires; v2 has no arousal step)",
         dict(base_priority=3, request_class="Pain", class_confidence=0.95,
              distress=0.55, non_speech_prob=0.001, asr_confidence=-0.2,
              transcript="my knee hurts when i bend it")),
        ("wordless groan, no transcript",
         dict(base_priority=CONFIG["P_NO_SPEECH"], request_class=UNCLASSIFIED,
              class_confidence=0.0, distress=0.85, non_speech_prob=0.10,
              asr_confidence=-10.0, transcript="")),
        ("calm chest pain",
         dict(base_priority=1, request_class="Emergency", class_confidence=0.88,
              distress=0.00, non_speech_prob=0.001, asr_confidence=-0.15,
              transcript="i have chest pain")),
        ("angry, not unwell: gate fires, escalates (accepted false alarm)",
         dict(base_priority=3, request_class="General", class_confidence=0.60,
              distress=0.45, non_speech_prob=0.001, asr_confidence=-0.20,
              transcript="this is ridiculous, i have been pressing this button for "
                         "twenty minutes")),
        ("button press, digital silence (energy pre-filter)",
         dict(base_priority=CONFIG["P_NO_SPEECH"], request_class=UNCLASSIFIED,
              class_confidence=0.0, distress=0.0, non_speech_prob=None,
              asr_confidence=None, transcript="", energy_ok=False,
              energy_reason="signal level below floor (RMS 0.00e+00 < 1e-04)")),
        ("ward noise: non_speech high, no transcript, no distress -> no speech",
         dict(base_priority=CONFIG["P_NO_SPEECH"], request_class=UNCLASSIFIED,
              class_confidence=0.0, distress=0.02, non_speech_prob=0.97,
              asr_confidence=-10.0, transcript="")),
        ("odd voice: non_speech high but a trusted transcript -> speech",
         dict(base_priority=1, request_class="Emergency", class_confidence=0.9,
              distress=0.0, non_speech_prob=0.99, asr_confidence=-0.3,
              transcript="i can't breathe properly")),
        ("groan the non_speech class claims, but distress fires -> P1",
         dict(base_priority=CONFIG["P_NO_SPEECH"], request_class=UNCLASSIFIED,
              class_confidence=0.0, distress=0.80, non_speech_prob=0.99,
              asr_confidence=-10.0, transcript="")),
    ]
    for name, kw in cases:
        d = apply_rules(**kw)
        print(f"\n{name}")
        print(f"  P{d.base_priority} -> P{d.final_priority}"
              f"{'  [ESCALATED]' if d.escalated else ''}")
        print(f"  {d.reason}   {d.rules_fired}")
