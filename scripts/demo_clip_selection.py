"""
Run every raw demo take through CareVoicePipeline and tabulate the outcome, so the
final clip set is chosen from measured results rather than by ear.

    HF_HUB_OFFLINE=1 python scripts/demo_clip_selection.py

Reads app/demo_clips/raw/<clip>_<what>_<speaker>_<take>.<ext>, writes
results/demo_clip_selection.csv, and prints the table plus, for the clips whose
outcome is uncertain (03 distressed knee, 08 irritated), how many takes fired the
distress gate. It does not choose the set.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import CareVoicePipeline  # noqa: E402

CONFIG = {
    "raw_dir": ROOT / "app" / "demo_clips" / "raw",
    "out": ROOT / "results" / "demo_clip_selection.csv",
    "audio_suffixes": {".wav", ".m4a", ".mp3", ".aiff", ".aif", ".flac", ".ogg", ".caf"},
    # What each clip is meant to show. Display only: the table records what happened.
    "expected": {
        "01": "Assistance P2",
        "02": "Pain P3",
        "03": "Pain, distress gate -> P1",
        "04": "Emergency P1",
        "05": "P1 via distress, no trusted transcript",
        "06": "Pain P3 + medication tag",
        "07": "General P3",
        "08": "General P3; P1 if the gate fires (accepted false alarm)",
        "09": "Unclassified P2 (no speech)",
    },
    "fire_count_clips": ("03", "08"),
}


def main() -> int:
    takes = sorted(p for p in CONFIG["raw_dir"].iterdir()
                   if p.suffix.lower() in CONFIG["audio_suffixes"])
    if not takes:
        print(f"no takes in {CONFIG['raw_dir']}")
        return 1
    pipe = CareVoicePipeline(model_dir=str(ROOT / "models"))
    rows = []
    for p in takes:
        r = pipe.process(p)
        clip = p.name.split("_", 1)[0]
        rows.append({
            "filename": p.name,
            "clip": clip,
            "expected": CONFIG["expected"].get(clip, ""),
            "transcript": r["transcript"],
            "asr_heard": r["asr_heard"],
            "class": r["request_class"],
            "medication": ",".join(r["medication"]),
            "base_priority": r["base_priority"],
            "final_priority": r["final_priority"],
            "rules_fired": ",".join(r["rules_fired"]) or "none",
            "distress": r["distress"],
            "distress_fired": r["distress_fired"],
            "non_speech": r["non_speech_prob"],
            "asr_confidence": r["asr_confidence"],
            "asr_trusted": r["asr_trusted"],
            "duration_s": r["duration_s"],
            "latency_ms": r["latency_ms"]["total"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(CONFIG["out"], index=False)

    pd.set_option("display.width", 250, "display.max_colwidth", 50)
    print(df.drop(columns=["asr_heard", "expected"]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    for clip in CONFIG["fire_count_clips"]:
        sub = df[df["clip"] == clip]
        if len(sub):
            print(f"clip {clip}: distress gate fired on {int(sub.distress_fired.sum())} of "
                  f"{len(sub)} takes; final priority counts "
                  f"{sub.final_priority.value_counts().sort_index().to_dict()}")
    print(f"\nwrote {len(df)} rows to {CONFIG['out'].relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
