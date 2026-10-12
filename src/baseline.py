"""
Per-bed HNR baseline: what this bed's calm voice sounds like.

Prosodic distress is scored relative to the bed's own previous calls, because
absolute HNR varies far more between people than within one person (calm medians
5.1 to 9.0 dB across three speakers). For each bed the store keeps the last
BASELINE_MAX_CALLS HNR values and returns their median and sample standard
deviation; below BASELINE_MIN_CALLS the bed has no baseline yet and no prosodic
score - the cold start, which behaves exactly like a system with no acoustic
escalation ("building baseline (n/5)" on the dashboard).

Entries are keyed by (bed, sample rate). HNR shifts 4-8 dB with sample rate, so a
48 kHz browser recording and a 16 kHz upload must never share a baseline.

What joins the baseline is decided by CareVoicePipeline: only calls with speech, a
measurable HNR, not escalated and not classified Emergency - so a baseline is built
from ordinary calls and an alarming call never becomes the new normal.

Stored in SQLite (the app passes app/calls.db). Unlike the audit log this is
operational state: old entries roll off and a bed can be reset (new patient).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.rules import CONFIG as RULES


@dataclass
class BaselineStats:
    n: int
    median: Optional[float]
    std: Optional[float]

    @property
    def ready(self) -> bool:
        return self.n >= RULES["BASELINE_MIN_CALLS"] and bool(self.std) and self.std > 0


class BedBaseline:
    def __init__(self, db_path, cfg: dict = RULES):
        self.cfg = cfg
        self.con = sqlite3.connect(Path(db_path), check_same_thread=False)
        self.con.execute("""CREATE TABLE IF NOT EXISTS prosody_baseline (
            id INTEGER PRIMARY KEY AUTOINCREMENT, bed TEXT NOT NULL, sr INTEGER NOT NULL,
            hnr REAL NOT NULL, ts REAL NOT NULL)""")
        self.con.execute("CREATE INDEX IF NOT EXISTS prosody_baseline_bed ON prosody_baseline (bed, sr, id)")
        self.con.commit()

    def stats(self, bed, sr: int) -> BaselineStats:
        vals = [r[0] for r in self.con.execute(
            "SELECT hnr FROM prosody_baseline WHERE bed=? AND sr=? ORDER BY id DESC LIMIT ?",
            (str(bed), int(sr), self.cfg["BASELINE_MAX_CALLS"]))]
        if not vals:
            return BaselineStats(0, None, None)
        arr = np.array(vals, dtype=float)
        return BaselineStats(len(arr), float(np.median(arr)),
                             float(arr.std(ddof=1)) if len(arr) > 1 else None)

    def add(self, bed, sr: int, hnr: float):
        """Append one calm call's HNR, keeping only the newest BASELINE_MAX_CALLS."""
        self.con.execute("INSERT INTO prosody_baseline (bed, sr, hnr, ts) VALUES (?,?,?,?)",
                         (str(bed), int(sr), float(hnr), time.time()))
        self.con.execute(
            """DELETE FROM prosody_baseline WHERE bed=? AND sr=? AND id NOT IN (
                   SELECT id FROM prosody_baseline WHERE bed=? AND sr=? ORDER BY id DESC LIMIT ?)""",
            (str(bed), int(sr), str(bed), int(sr), self.cfg["BASELINE_MAX_CALLS"]))
        self.con.commit()

    def reset(self, bed=None):
        """Forget a bed's baseline (new patient), or every bed's."""
        if bed is None:
            self.con.execute("DELETE FROM prosody_baseline")
        else:
            self.con.execute("DELETE FROM prosody_baseline WHERE bed=?", (str(bed),))
        self.con.commit()
