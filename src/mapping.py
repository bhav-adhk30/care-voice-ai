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
