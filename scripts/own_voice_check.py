"""
Run the shipped pipeline over the team's own recordings (care-voice-own-voices) and
save what it decided, for the app's "Tested on our own voices" section.

    python scripts/own_voice_check.py <folder of speaker_what_label_take.wav>

Writes results/own_voice_check.csv. Room-noise clips are skipped. "emergency" = the
chest-pain and dizziness takes (calm and distressed); "calm request" = the ten
everyday requests each speaker recorded calmly.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd  # noqa: E402

from src.pipeline import CareVoicePipeline  # noqa: E402

CONFIG = {"out": ROOT / "results" / "own_voice_check.csv",
          "emergency": {"chest", "dizzy"}, "symptom": {"chest", "dizzy", "knee", "groan"}}


def kind(what: str, label: str) -> str:
    if what in CONFIG["emergency"]:
        return "emergency"
    if what not in CONFIG["symptom"] and label == "calm":
        return "calm request"
    return "other"


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    pipe = CareVoicePipeline(model_dir=str(ROOT / "models"))
    rows = []
    for f in sorted(folder.glob("*.wav")):
        spk, what, label = f.stem.split("_")[:3]
        if label == "noise":
            continue
        r = pipe.process(f)
        rows.append({"speaker": spk.capitalize(), "clip": what, "delivery": label, "kind": kind(what, label),
                     "transcript": r["transcript"], "request_class": r["request_class"],
                     "final_priority": r["final_priority"], "rules_fired": ";".join(r["rules_fired"]),
                     "latency_s": round(r["latency_ms"]["total"] / 1000, 2)})
    df = pd.DataFrame(rows)
    df.to_csv(CONFIG["out"], index=False)
    em, calm = df[df.kind == "emergency"], df[df.kind == "calm request"]
    print(f"{len(df)} clips | emergencies at P1 {(em.final_priority == 1).sum()}/{len(em)} | "
          f"calm requests at P1 {(calm.final_priority == 1).sum()}/{len(calm)}")
    print("saved", CONFIG["out"])
