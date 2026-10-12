"""
Label mapping for the text branch: dataset prompt -> (request class, base priority).

This table is the project's main contribution and was derived by error analysis on the
Medical Speech, Transcription and Intent corpus (see CLAUDE.md section 5). Do not edit it
without being asked. In particular:

- "Heart hurts" is the correct spelling (cardiac complaint, Emergency P1), not "Head hurts".
- "Internal pain" is Pain, not Emergency: its phrases are unlocalised aches.
- "Blurry vision" is General, not Emergency: its phrases are optometry complaints.
- There is no Medication class. No intent in the dataset is a medication request, so the
  system trains on exactly four classes.

Priority scale (lower number = more urgent):
    P1 immediate, bypass the queue
    P2 urgent, within a few minutes
    P3 standard
    P4 routine
"""

from __future__ import annotations

PROMPT_TO_CLASS = {
    # Emergency
    "Hard to breath":     ("Emergency", 1),
    "Heart hurts":        ("Emergency", 1),
    "Open wound":         ("Emergency", 1),
    "Feeling dizzy":      ("Emergency", 2),
    "Infected wound":     ("Emergency", 2),
    # Assistance
    "Body feels weak":    ("Assistance", 2),
    "Feeling cold":       ("Assistance", 3),
    # Pain
    "Internal pain":      ("Pain", 3),
    "Shoulder pain":      ("Pain", 3),
    "Joint pain":         ("Pain", 3),
    "Knee pain":          ("Pain", 3),
    "Muscle pain":        ("Pain", 3),
    "Ear ache":           ("Pain", 3),
    "Head ache":          ("Pain", 3),
    "Stomach ache":       ("Pain", 3),
    "Back pain":          ("Pain", 3),
    "Neck pain":          ("Pain", 3),
    "Foot ache":          ("Pain", 3),
    "Injury from sports": ("Pain", 3),
    # General
    "Blurry vision":      ("General", 3),
    "Cough":              ("General", 3),
    "Emotional pain":     ("General", 3),
    "Skin issue":         ("General", 4),
    "Acne":               ("General", 4),
    "Hair falling out":   ("General", 4),
}

# Fixed class order. The index of a class in this list is its integer label everywhere
# (training targets, logits, saved metrics), so never reorder it.
CLASSES = ["Emergency", "Pain", "Assistance", "General"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

PRIORITIES = (1, 2, 3, 4)


def lookup(prompt: str) -> tuple[str, int]:
    """Return (request_class, base_priority) for a dataset prompt.

    Leading/trailing whitespace is ignored; spelling and case must otherwise match the
    dataset exactly (e.g. "Hard to breath", "Heart hurts"). Raises KeyError for an unknown
    prompt rather than guessing, so a typo or a new label surfaces immediately.
    """
    key = prompt.strip()
    try:
        return PROMPT_TO_CLASS[key]
    except KeyError:
        raise KeyError(
            f"Unknown prompt {prompt!r}. Expected one of {sorted(PROMPT_TO_CLASS)}"
        ) from None


def prompt_to_class(prompt: str) -> str:
    """Request class for a prompt."""
    return lookup(prompt)[0]


def prompt_to_priority(prompt: str) -> int:
    """Base priority (1..4) for a prompt."""
    return lookup(prompt)[1]


def prompt_to_idx(prompt: str) -> int:
    """Integer class label (index into CLASSES) for a prompt."""
    return CLASS_TO_IDX[lookup(prompt)[0]]


# Sanity checks at import time: the table must stay consistent with CLASSES.
assert len(PROMPT_TO_CLASS) == 25
assert {c for c, _ in PROMPT_TO_CLASS.values()} == set(CLASSES)
assert all(p in PRIORITIES for _, p in PROMPT_TO_CLASS.values())


# Base priority when only the predicted CLASS is known.
# ClinicalBERT predicts one of four classes, but priorities vary within a class
# (Emergency spans P1-P2, Assistance spans P2-P3). We take the most urgent
# priority in the class, because under-triage is the costlier error here.
# PROMPT_TO_CLASS is still used for ground-truth evaluation and the oracle experiment.
CLASS_BASE_PRIORITY = {
    "Emergency":  1,
    "Assistance": 2,
    "Pain":       3,
    "General":    3,
}


def class_base_priority(req_class: str) -> int:
    if req_class not in CLASS_BASE_PRIORITY:
        raise KeyError(f"Unknown class: {req_class!r}")
    return CLASS_BASE_PRIORITY[req_class]


# ---------------------------------------------------------------------------
# Medication tag: a keyword RULE, deliberately not a learned class.
#
# The system design has a Medication request type, but no public corpus labels
# medication requests: none of the 25 intents in the Medical Speech corpus is
# one (section 5), and nothing here is synthesised to fake one. So instead of a
# fifth class, any transcript is tagged by keyword as a sub-label shown under
# its predicted class ("Pain + medication"). It never changes the class or the
# priority. Recall is whatever the keyword list covers; it has not been, and
# cannot yet be, measured.
# ---------------------------------------------------------------------------

import re  # noqa: E402

MEDICATION_KEYWORDS = ("medicine", "medication", "painkiller", "tablet", "dose",
                       "injection", "pill")

_MEDICATION_RE = re.compile(
    r"\b(" + "|".join(MEDICATION_KEYWORDS) + r")(s|es)?\b", re.IGNORECASE)


def medication_tag(transcript: str) -> list[str]:
    """Medication keywords found in the transcript (base forms, sorted); [] if none."""
    return sorted({m.group(1).lower() for m in _MEDICATION_RE.finditer(transcript or "")})


# ---------------------------------------------------------------------------
# Care-request tag: a keyword RULE for toileting and mobility help.
#
# The most common real reasons for pressing a call button - getting to the
# toilet, getting up, being moved - are not in the training data at all: the
# Medical Speech corpus's 25 intents are symptom complaints, not care requests.
# ClinicalBERT therefore files "I need to go to the washroom, can somebody get
# me there" under General (0.93, found with a real team recording), because the
# category does not exist for it. No public corpus labels these requests, so,
# like the medication tag, this is deliberately a rule and not a learned class.
#
# Unlike the medication tag it changes priority: rules.py Rule 5 floors a tagged
# call at P_CARE_REQUEST (P2, the Assistance level). The predicted class is left
# as the classifier said, so the audit shows both.
#
# Measured: 0 of the 706 dataset phrases match (none is a care request, so every
# match would be a false positive). Mobility phrases match only when they ask for
# help ("help me up", "need help getting out of bed"), not symptoms ("my knee
# hurts when I get up"). Recall on real care requests is unmeasured: there is no
# data to measure it on.
# ---------------------------------------------------------------------------

_TOILET_RE = re.compile(
    r"\b(toilet|washroom|bathroom|restroom|loo|lavatory|bedpan|commode)s?\b", re.IGNORECASE)
_HELP_RE = re.compile(
    r"\b(help|take|get|bring|walk|lift|move|turn)\s+me\b"
    r"|\b(need|want)\s+(some\s+)?help\s+(to|with|getting|going|moving|standing|walking)\b"
    r"|\bhelp\s+(getting|going|moving|standing|walking|sitting)\b"
    r"|\bcan(\s+some(one|body)|\s+you|\s+i\s+get\s+some(one|body))\s+(help|take|get|move|lift|walk|turn)\b",
    re.IGNORECASE)
_MOBILITY_RE = re.compile(
    r"\b(up|out of bed|into bed|to bed|stand|sit up|sit|walk|move|turn over|wheelchair|chair|"
    r"shower|wash|dressed)\b", re.IGNORECASE)


def care_request_tag(transcript: str) -> list[str]:
    """Care-request keywords in the transcript: the toileting word found, and/or
    "mobility" for an explicit request for help moving. [] if none."""
    t = transcript or ""
    hits = []
    m = _TOILET_RE.search(t)
    if m:
        hits.append(m.group(1).lower())
    if _HELP_RE.search(t) and _MOBILITY_RE.search(t):
        hits.append("mobility")
    return hits
