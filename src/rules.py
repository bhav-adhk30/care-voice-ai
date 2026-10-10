"""
Escalation and safety-override rules for Care-Voice AI.

This is deliberately NOT a trained model. See CLAUDE.md section 2.1: training a
fusion head needs clips labelled with both a request type and an urgency level,
and no such corpus exists. The combination is an explicit rule instead, which in
a clinical system is also easier to audit than learned weights.

Every threshold below was measured, not chosen. Provenance is in the comments.
"""

from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Thresholds. All measured on held-out data — see results/*.json.
# ---------------------------------------------------------------------------

CONFIG = {
    # Arousal head (WavLM + MLP, trained on CREMA-D with augmentation).
    # At 0.105: 60% fear recall, 3.8% false alarms on calm medical speech.
    # Chosen from the operating curve in results/operating_curve_augmented.csv —
    # the acoustic path is a safety net on top of the text branch, so a low
    # false-alarm rate matters more than recall. Alert fatigue kills these systems.
    "AROUSAL_HIGH": 0.105,

    # Distress gate (WavLM + logistic regression, trained on VIVAE pain/fear
    # against CREMA-D neutral speech and other VIVAE vocalisations).
    # At 0.0677: 90% distress recall, 2.4% false alarms.
    # AUC 0.996 against calm speech, 0.651 against other intense vocalisations —
    # it detects distress reliably but cannot tell pain from triumph. Irrelevant
    # in a ward, stated as a limitation.
    "DISTRESS_HIGH": 0.0677,

    # Whisper mean token log-probability, 20th percentile of the corpus.
    # Below it: 18.3% of transcripts are badly wrong (WER > 0.5) and request-type
    # accuracy falls to 0.707, against 1.5% / 0.86 above it.
    # Whisper hallucinates fluent unrelated sentences on poor audio, which is a
    # safety failure rather than an accuracy one — hence this gate.
    "ASR_MIN": -0.4934,

    # Priority is floored (made more urgent) at these levels, never raised.
    "P_IMMEDIATE": 1,
    "P_URGENT": 2,

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


# The distress gate is BINARY, not graded. Its score correlates with annotated
# intensity only weakly (Spearman rho = 0.16, p < 1e-9, n = 1565), far too
# weakly to support severity levels. It fires or it doesn't.
DISTRESS_IS_BINARY = True

# The gate cannot separate anger (mean score 0.510) from pain (0.585).
# So anger suppression cannot happen acoustically — it has to come from the text
# branch. See `suppress_anger_escalation` below.
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
    arousal_fired: bool
    asr_trusted: bool
    speech_present: bool = True
    acoustic_used: bool = True

    def to_dict(self):
        return asdict(self)


def suppress_anger_escalation(request_class: str, class_confidence: float) -> bool:
    """
    Should acoustic escalation be suppressed for this request?

    An irritated patient is not a medical emergency, but the distress gate
    cannot tell anger from pain acoustically. So the suppression decision uses
    the text branch instead: if the request is confidently General (the class
    that holds Emotional pain, Skin issue, Acne and similar), high arousal is
    most likely frustration rather than clinical distress.

    Deliberately conservative: only suppresses on a confident General, and never
    touches the distress gate, which still fires for a genuine wordless cry.
    """
    return request_class == "General" and class_confidence >= 0.80


def apply_rules(
    base_priority: int,
    request_class: str,
    class_confidence: float,
    arousal: float,
    distress: float,
    asr_confidence: Optional[float] = None,
    speech_present: bool = True,
    speech_reason: str = "",
    cfg: dict = CONFIG,
    *,
    transcript: str,
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
    arousal          calibrated arousal-head score, 0..1
    distress         calibrated distress-gate score, 0..1
    asr_confidence   mean token log-probability from Whisper, or None
    speech_present   False when preprocess.speech_present found no usable signal
    speech_reason    why speech_present was False, shown on the dashboard
    transcript       the ASR output; required, to enforce the invariant below
    """
    # Invariant, both directions: Unclassified <=> empty transcript. Raised rather
    # than asserted so it also holds under python -O.
    empty = transcript.strip() == ""
    if (request_class == UNCLASSIFIED) != empty:
        raise ValueError(
            f"request_class={request_class!r} with "
            f"{'an empty' if empty else 'a non-empty'} transcript: {UNCLASSIFIED!r} "
            "is required exactly when the transcript is empty")

    # --- Rule 0: no usable speech -------------------------------------------
    # Short-circuits everything. Both branches are reading noise: the text
    # branch has no transcript and the acoustic heads were never trained on a
    # normalised noise floor, where the distress gate scores near 1.0.
    #
    # Routed at P_NO_SPEECH and returned immediately, so Rule 1 (untrusted
    # transcript + distress -> P1) cannot fire on an empty clip. A wordless
    # groan still reaches Rule 1, because it passes this guard on energy.
    if not speech_present:
        return Decision(
            final_priority=cfg["P_NO_SPEECH"],
            base_priority=base_priority,
            reason=("no speech detected" + (f" ({speech_reason})" if speech_reason else "")
                    + "; routed as an unclassified call, equivalent to a "
                      "conventional call button"),
            escalated=False,
            distress_fired=False,
            arousal_fired=False,
            asr_trusted=False,
            speech_present=False,
            acoustic_used=False,
        )

    priority = base_priority
    reasons = []

    asr_trusted = asr_confidence is None or asr_confidence >= cfg["ASR_MIN"]
    if not asr_trusted:
        reasons.append("ASR confidence below threshold; transcript not trusted")

    distress_fired = distress >= cfg["DISTRESS_HIGH"]
    arousal_fired = arousal >= cfg["AROUSAL_HIGH"]

    suppress = suppress_anger_escalation(request_class, class_confidence)
    if suppress and (arousal_fired or distress_fired):
        reasons.append("acoustic escalation suppressed: confident General, "
                       "likely frustration rather than clinical distress")

    # --- Rule 1: untrusted transcript plus distress -> immediate -------------
    # The text branch cannot be relied on and the acoustic path is the only
    # signal left. This is the wordless-patient case and the hallucination case
    # at the same time.
    if not asr_trusted and distress_fired:
        priority = min(priority, cfg["P_IMMEDIATE"])
        reasons.append("distress detected with an unreliable transcript")

    # --- Rule 2: distress gate ----------------------------------------------
    elif distress_fired and not suppress:
        priority = min(priority, cfg["P_IMMEDIATE"])
        reasons.append(f"distress gate fired ({distress:.3f} >= "
                       f"{cfg['DISTRESS_HIGH']})")

    # --- Rule 3: high arousal raises by one level ---------------------------
    elif arousal_fired and not suppress:
        priority = min(priority, max(cfg["P_URGENT"], base_priority - 1))
        reasons.append(f"elevated vocal arousal ({arousal:.3f} >= "
                       f"{cfg['AROUSAL_HIGH']})")

    # --- Rule 4: Emergency floor --------------------------------------------
    # Never queue a predicted Emergency worse than P2, regardless of how the
    # voice sounds. A patient can describe a cardiac event perfectly calmly.
    if request_class == "Emergency":
        if priority > cfg["EMERGENCY_FLOOR"]:
            reasons.append("Emergency class floor applied")
        priority = min(priority, cfg["EMERGENCY_FLOOR"])

    if not reasons:
        reasons.append("no escalation; base priority from request class")

    return Decision(
        final_priority=priority,
        base_priority=base_priority,
        reason="; ".join(reasons),
        escalated=priority < base_priority,
        distress_fired=distress_fired,
        arousal_fired=arousal_fired,
        asr_trusted=asr_trusted,
        speech_present=True,
        acoustic_used=True,
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


if __name__ == "__main__":
    cases = [
        ("calm knee pain",
         dict(base_priority=3, request_class="Pain", class_confidence=0.95,
              arousal=0.02, distress=0.01, asr_confidence=-0.2,
              transcript="my knee hurts when i bend it")),
        ("same phrase, distressed voice",
         dict(base_priority=3, request_class="Pain", class_confidence=0.95,
              arousal=0.40, distress=0.02, asr_confidence=-0.2,
              transcript="my knee hurts when i bend it")),
        ("wordless groan, no transcript",
         dict(base_priority=CONFIG["P_NO_SPEECH"], request_class=UNCLASSIFIED,
              class_confidence=0.0, arousal=0.60, distress=0.85,
              asr_confidence=-10.0, transcript="")),
        ("calm chest pain",
         dict(base_priority=1, request_class="Emergency", class_confidence=0.88,
              arousal=0.01, distress=0.00, asr_confidence=-0.15,
              transcript="i have chest pain")),
        ("frustrated but not unwell",
         dict(base_priority=4, request_class="General", class_confidence=0.91,
              arousal=0.35, distress=0.05, asr_confidence=-0.20,
              transcript="my skin has been itching all day")),
        ("button press, no speech",
         dict(base_priority=CONFIG["P_NO_SPEECH"], request_class=UNCLASSIFIED,
              class_confidence=0.0, arousal=0.0, distress=0.0, asr_confidence=None,
              transcript="", speech_present=False,
              speech_reason="signal level below floor (RMS 1.1e-05 < 2.0e-03)")),
    ]
    for name, kw in cases:
        d = apply_rules(**kw)
        print(f"\n{name}")
        print(f"  P{d.base_priority} -> P{d.final_priority}"
              f"{'  [ESCALATED]' if d.escalated else ''}")
        print(f"  {d.reason}")
