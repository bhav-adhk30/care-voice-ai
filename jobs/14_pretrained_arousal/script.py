"""
Job 14 - does a pretrained, naturalistic arousal model detect distress in the team's voices?

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (wav2vec2 fine-tuned on
MSP-Podcast, natural speech from thousands of speakers; outputs arousal, dominance,
valence in ~0..1). Licence CC BY-NC-SA 4.0: fine for this non-commercial project.

Data: care-voice-own-voices - 51 clips (39 calm, 12 distress incl. 3 groans), 3 speakers,
one phone each, 48 kHz. Room-noise clips excluded.

Protocol (identical to the HNR measurement, CLAUDE.md §2.8):
  - Each clip is a z-score against the SAME speaker's calm clips, leave-one-out (a clip
    is never in its own baseline) - the app's per-bed baseline.
  - Threshold leave-one-speaker-out: 95th percentile of the OTHER two speakers' calm
    scores, applied to the held-out speaker.
  - All 12 distress clips positive. No grading, no exclusions.

PRE-REGISTERED (written before the run):
  primary score   one-sided arousal z (distress = more aroused than this speaker's calm)
  audio           native rate -> 16 kHz via src.preprocess (load_audio_native + resample),
                  NO loudness normalisation (loudness is part of arousal; the per-speaker
                  baseline absorbs mic distance)
  PASS            every speaker >= 2/4 distress caught AND calm false alarms <= 2/39
  Anything else printed is exploratory and cannot turn a FAIL into a PASS.

Outputs (/kaggle/working): pretrained_arousal_clips.csv, pretrained_arousal.json
Local smoke test (random scores, no model): CV14_SMOKE=1 python jobs/14_pretrained_arousal/script.py
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

SMOKE = os.environ.get("CV14_SMOKE") == "1"
ON_KAGGLE = Path("/kaggle/input").exists() and not SMOKE
if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyloudnorm"], check=True)


def find_one(pattern, what):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{what}: nothing matches /kaggle/input/**/{pattern}")
    return Path(hits[0])


if ON_KAGGLE:
    REPO = find_one("src/preprocess.py", "care-voice-code").parent.parent
    CLIPS = find_one("bhavishya_knee_calm_1.wav", "care-voice-own-voices").parent
    WORK = Path("/kaggle/working")
else:
    REPO = Path(__file__).resolve().parents[2]
    CLIPS = Path(os.environ["CV14_CLIPS"])
    WORK = Path(os.environ.get("CV14_OUT", "/tmp/cv14"))
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.preprocess import load_audio_native, resample  # noqa: E402

CONFIG = {
    "model": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "seed": 42,
    "fa_percentile": 95,
    "pass_min_per_speaker": 2,      # of 4 distress clips
    "pass_max_false_alarms": 2,     # of 39 calm clips
}
np.random.seed(CONFIG["seed"])
SPEAKERS = ("bhavishya", "sumayah", "vinuthna")

# ----------------------------------------------------------------- clips
rows = []
for f in sorted(CLIPS.glob("*.wav")):
    spk, what, lab = f.stem.split("_")[:3]
    if lab == "noise":
        continue
    rows.append({"file": f.name, "path": str(f), "speaker": spk, "what": what, "label": lab})
df = pd.DataFrame(rows)
print(df.groupby(["speaker", "label"]).size().unstack().to_string())
assert len(df) == 51 and (df.label == "distress").sum() == 12, "expected 51 clips, 12 distress"

# ----------------------------------------------------------------- model scores
if SMOKE:
    rng = np.random.default_rng(0)
    for k in ("arousal", "dominance", "valence"):
        df[k] = rng.random(len(df))
else:
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (Wav2Vec2Model,
                                                                Wav2Vec2PreTrainedModel)

    class RegressionHead(nn.Module):           # as published on the model card
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            x = torch.tanh(self.dense(self.dropout(x)))
            return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            h = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.classifier(h)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Wav2Vec2Processor.from_pretrained(CONFIG["model"])
    model = EmotionModel.from_pretrained(CONFIG["model"]).to(dev).eval()
    print("model labels:", model.config.id2label, "| device", dev, flush=True)
    order = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    assert set(order) == {"arousal", "dominance", "valence"}, order

    out = []
    with torch.no_grad():
        for p in df.path:
            wav, sr = load_audio_native(p)
            wav = resample(wav, sr, 16000).astype(np.float32)   # no loudness normalisation
            x = proc(wav, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
            out.append(model(x)[0].cpu().numpy())
    out = np.array(out)
    for i, k in enumerate(order):
        df[k] = out[:, i]

# ----------------------------------------------------------------- per-speaker z, LOSO threshold
for k in ("arousal", "dominance", "valence"):
    z = np.zeros(len(df))
    for i, r in df.iterrows():
        base = df[(df.speaker == r.speaker) & (df.label == "calm") & (df.index != i)][k]
        z[i] = (r[k] - base.median()) / base.std(ddof=1)
    df[f"{k}_z"] = z
y = (df.label == "distress").values


def loso(score):
    per, tp, fp = {}, 0, 0
    for held in SPEAKERS:
        t = np.percentile(score[(df.speaker != held).values & ~y], CONFIG["fa_percentile"])
        d = int((score[(df.speaker == held).values & y] >= t).sum())
        c = int((score[(df.speaker == held).values & ~y] >= t).sum())
        per[held] = {"threshold": round(float(t), 3), "distress_caught": f"{d}/4", "calm_fa": f"{c}/13"}
        tp, fp = tp + d, fp + c
    return {"auc": round(float(roc_auc_score(y, score)), 3), "recall": f"{tp}/12",
            "false_alarms": f"{fp}/39", "tp": tp, "fp": fp, "per_speaker": per}


primary = loso(df.arousal_z.values)
passed = (all(int(v["distress_caught"].split("/")[0]) >= CONFIG["pass_min_per_speaker"]
              for v in primary["per_speaker"].values())
          and primary["fp"] <= CONFIG["pass_max_false_alarms"])
exploratory = {
    "two_sided_arousal_|z|": loso(np.abs(df.arousal_z.values)),
    "arousal_z_minus_valence_z": loso(df.arousal_z.values - df.valence_z.values),
    "raw_arousal_no_baseline": loso(df.arousal.values),
}

print("\n=== PRIMARY (pre-registered): one-sided arousal z vs own calm baseline ===")
print(json.dumps(primary, indent=1))
print(f"\nPASS MARK: every speaker >= {CONFIG['pass_min_per_speaker']}/4 and false alarms "
      f"<= {CONFIG['pass_max_false_alarms']}/39  ->  {'PASS' if passed else 'FAIL'}")
print("\n=== exploratory (cannot change the verdict) ===")
for k, v in exploratory.items():
    print(f"{k:30s} AUC {v['auc']}  recall {v['recall']}  FA {v['false_alarms']}  "
          + "  ".join(f"{s[:3]} {d['distress_caught']}" for s, d in v["per_speaker"].items()))
print("\nper distress clip:")
print(df[y][["speaker", "what", "arousal", "arousal_z", "valence_z"]].round(2).to_string(index=False))

df.drop(columns="path").to_csv(WORK / "pretrained_arousal_clips.csv", index=False)
json.dump({"model": CONFIG["model"], "verdict": "PASS" if passed else "FAIL",
           "pass_mark": {k: CONFIG[k] for k in ("pass_min_per_speaker", "pass_max_false_alarms")},
           "primary_one_sided_arousal_z": primary, "exploratory": exploratory,
           "smoke": SMOKE}, open(WORK / "pretrained_arousal.json", "w"), indent=2)
print("\nVERDICT:", "PASS" if passed else "FAIL", flush=True)
