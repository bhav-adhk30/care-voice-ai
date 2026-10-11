"""
Append-only audit log: one row per call, enough to reconstruct the decision.

Every row stores the inputs the rule saw (transcript, ASR confidence, class
probabilities, every acoustic score), what it decided (base and final priority,
rules fired, reason) and the THRESHOLDS IN FORCE at decision time, plus short
hashes of the model files. Thresholds in rules.CONFIG change as the heads are
refit (DISTRESS_HIGH moved 0.0677 -> 0.4028); without the snapshot an old
decision could not be explained once the config moved on.

Append-only is enforced by the database, not by convention: SQLite triggers
abort any UPDATE or DELETE on the table, and this module exposes no path to
either. Queue state (waiting / answered) lives elsewhere and is mutable; this
table is the record.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

COLUMNS = (
    "ts_utc", "mode", "bed", "audio_name", "audio_sha256",
    "transcript", "asr_heard", "asr_confidence", "asr_trusted",
    "energy_ok", "speech_present", "speech_reason",
    "request_class", "class_probs", "class_confidence", "medication",
    "sentiment_label", "text_anger", "sentiment",
    "non_speech_prob", "fear_prob_display_only", "distress", "distress_fired",
    "base_priority", "final_priority", "escalated", "rules_fired", "reason",
    "thresholds", "model_versions", "latency_total_ms",
)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {", ".join(f"{c} TEXT" if c not in ("asr_confidence", "class_confidence", "text_anger",
                                        "non_speech_prob", "fear_prob_display_only", "distress",
                                        "latency_total_ms", "base_priority", "final_priority",
                                        "asr_trusted", "energy_ok", "speech_present",
                                        "distress_fired", "escalated")
               else f"{c} REAL" for c in COLUMNS)}
);
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AuditLog:
    def __init__(self, db_path):
        self.path = Path(db_path)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.executescript(_SCHEMA)
        self.con.commit()

    def append(self, result: dict, *, bed, audio_name: str, audio_bytes: bytes,
               mode: str) -> int:
        """Record one decision. `result` is CareVoicePipeline.process() output."""
        j = lambda v: json.dumps(v, sort_keys=True)
        row = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": mode, "bed": str(bed), "audio_name": audio_name,
            "audio_sha256": sha256_bytes(audio_bytes),
            "transcript": result["transcript"], "asr_heard": result.get("asr_heard", ""),
            "asr_confidence": result["asr_confidence"], "asr_trusted": result["asr_trusted"],
            "energy_ok": result["energy_ok"], "speech_present": result["speech_present"],
            "speech_reason": result["speech_reason"],
            "request_class": result["request_class"], "class_probs": j(result["class_probs"]),
            "class_confidence": result["class_confidence"], "medication": j(result["medication"]),
            "sentiment_label": result["sentiment_label"], "text_anger": result["text_anger"],
            "sentiment": j(result["sentiment"]),
            "non_speech_prob": result["non_speech_prob"],
            "fear_prob_display_only": result["arousal"],
            "distress": result["distress"], "distress_fired": result["distress_fired"],
            "base_priority": result["base_priority"], "final_priority": result["final_priority"],
            "escalated": result["escalated"], "rules_fired": j(result["rules_fired"]),
            "reason": result["reason"],
            "thresholds": j(result["thresholds"]), "model_versions": j(result["model_versions"]),
            "latency_total_ms": result["latency_ms"]["total"],
        }
        cur = self.con.execute(
            f"INSERT INTO audit_log ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
            [row[c] for c in COLUMNS])
        self.con.commit()
        return cur.lastrowid

    def recent(self, limit: int = 200):
        cur = self.con.execute(f"SELECT id, {', '.join(COLUMNS)} FROM audit_log "
                               "ORDER BY id DESC LIMIT ?", (limit,))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]
