"""
Re-derive ASR_MIN with the pipeline's exact decoder (Kaggle CPU job).

ASR_MIN (rules.py) was the 20th percentile of Whisper mean log-probability over
results/transcripts.csv, produced by an earlier decoder setup. The pipeline now
decodes with faster-whisper int8 on CPU, beam 5, temperature 0.0 (no fallback),
condition_on_previous_text=False, vad_filter=False. This job re-transcribes the same
6,661 clips through src.pipeline.transcribe() - the function process() calls, with
src.pipeline.CONFIG["whisper_decode"] - so the settings cannot drift, and
recomputes everything per log-probability quintile.

Audio goes through the pipeline's ASR path: load_audio (-23 LUFS) -> denoise ->
trim. Denoise is disabled, as on the demo machine where DeepFilterNet does not
install.

Request-type accuracy: the original out-of-fold per-fold ClinicalBERT models no
longer exist, so both old and new transcripts are scored with the DEPLOYED model
(fit on all phrases): like-for-like between decoders, but in-sample, so higher than
the original 0.707 / 0.86. The original out-of-fold `correct` column is reported per
old-quintile alongside for reference.

Outputs (/kaggle/working/results/): asr_confidence_v2.json, asr_transcripts_v2.csv.

Local smoke test:  CV12_MED_DIR=<dir with some of the wavs> CV12_LIMIT=8 python script.py
"""

import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Everything src/ imports that the Kaggle image lacks (v1 died on pyloudnorm).
for pkg, mod in (("faster-whisper", "faster_whisper"), ("jiwer", "jiwer"),
                 ("pyloudnorm", "pyloudnorm")):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

ON_KAGGLE = Path("/kaggle/input").exists()


def find_dir(marker, local):
    if ON_KAGGLE:
        hits = sorted(Path("/kaggle/input").glob(f"**/{marker}"))
        if hits:
            return hits[0].parent
    return Path(local)


REPO = find_dir("src/pipeline.py", Path(__file__).resolve().parents[2] / "src").parent \
    if ON_KAGGLE else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONFIG = {
    "seed": 42,
    "med_dir": os.environ.get("CV12_MED_DIR", "/kaggle/input"),
    "bert_dir": find_dir("classes.json", REPO / "models" / "clinicalbert_requestclass"),
    "whisper": (str(REPO / "models" / "faster-whisper-small")
                if (REPO / "models" / "faster-whisper-small").exists()
                else "Systran/faster-whisper-small"),
    "out_dir": Path("/kaggle/working/results" if ON_KAGGLE else os.environ.get(
        "CV12_OUT", REPO / "results")),
    "limit": int(os.environ.get("CV12_LIMIT", 0)) or None,
    "quantile": 0.20,
    "bad_wer": 0.5,
    "checkpoint_every": 500,
}
CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)

import jiwer  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

import src.preprocess as pp  # noqa: E402
from src.pipeline import CONFIG as PIPE, transcribe  # noqa: E402
from src.preprocess import denoise, load_audio, trim  # noqa: E402
from src.rules import CONFIG as RULES  # noqa: E402

pp.CONFIG["enable_denoise"] = False
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
SR = PIPE["sr"]
print(f"repo={REPO} kaggle={ON_KAGGLE} cpus={os.cpu_count()} seed={CONFIG['seed']}")
print("decoder:", PIPE["whisper_decode"], "compute_type:", PIPE["whisper_compute_type"])


def norm(s):
    # Reproduces transcripts.csv's `ref` column on 99.9% of rows.
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


# ---- clips ----------------------------------------------------------------
tr = pd.read_csv(REPO / "results" / "transcripts.csv")
index = {}
for p in glob.glob(f"{CONFIG['med_dir']}/**/*.wav", recursive=True):
    index.setdefault(os.path.basename(p), p)
tr["local_path"] = tr.file_name.map(index)
missing = int(tr.local_path.isna().sum())
tr = tr.dropna(subset=["local_path"]).reset_index(drop=True)
if CONFIG["limit"]:
    tr = tr.head(CONFIG["limit"])
print(f"clips: {len(tr)} (missing audio: {missing})")

# ---- transcribe -----------------------------------------------------------
asr = WhisperModel(CONFIG["whisper"], device="cpu", compute_type=PIPE["whisper_compute_type"],
                   cpu_threads=os.cpu_count() or 4)
rows, t0 = [], time.time()
for k, r in enumerate(tr.itertuples()):
    wav = trim(denoise(load_audio(r.local_path, target_sr=SR), sr=SR), sr=SR)
    text, lp, segs = transcribe(asr, wav)
    rows.append({
        "file_name": r.file_name,
        "new_hyp_raw": text,
        "new_logprob": lp,
        "n_segments": len(segs),
        "n_tokens": int(sum(len(s.tokens) for s in segs)),
        "no_speech_probs": json.dumps([round(s.no_speech_prob, 5) for s in segs]),
        "no_speech_prob_max": max((s.no_speech_prob for s in segs), default=np.nan),
        "temperatures": json.dumps(sorted({s.temperature for s in segs})),
    })
    if (k + 1) % 100 == 0 or k + 1 == len(tr):
        el = time.time() - t0
        print(f"  {k+1}/{len(tr)}  {el:.0f}s  eta {el/(k+1)*(len(tr)-k-1)/60:.0f} min", flush=True)
    if (k + 1) % CONFIG["checkpoint_every"] == 0:
        pd.DataFrame(rows).to_csv(CONFIG["out_dir"] / "asr_transcripts_v2_partial.csv", index=False)

new = pd.DataFrame(rows)
temps = set().union(*[set(json.loads(t)) for t in new.temperatures])
assert temps <= {0.0}, f"fallback ladder fired: {temps}"
df = tr.drop(columns=["local_path"]).merge(new, on="file_name")
df["ref_norm"] = df.phrase.map(norm)
df["old_hyp_norm"] = df.hyp.fillna("").map(norm)
df["new_hyp_norm"] = df.new_hyp_raw.map(norm)
df["old_wer"] = [jiwer.wer(a, b) if a else np.nan for a, b in zip(df.ref_norm, df.old_hyp_norm)]
df["new_wer"] = [jiwer.wer(a, b) if a else np.nan for a, b in zip(df.ref_norm, df.new_hyp_norm)]

# ---- request type, deployed ClinicalBERT (in-sample) ----------------------
bd = CONFIG["bert_dir"]
classes = json.loads((Path(bd) / "classes.json").read_text())["classes"]
tok = AutoTokenizer.from_pretrained(bd)
bert = AutoModelForSequenceClassification.from_pretrained(bd).eval()


@torch.no_grad()
def classify(texts):
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], return_tensors="pt", padding=True, truncation=True,
                  max_length=PIPE["bert_max_length"])
        out += [classes[j] for j in bert(**enc).logits.argmax(-1).tolist()]
    return out


df["new_pred"] = classify([t if t.strip() else " " for t in df.new_hyp_raw])
df["old_pred_deployed"] = classify([t if str(t).strip() else " " for t in df.hyp.fillna("")])
df["new_correct"] = df.new_pred == df.req_class
df["old_correct_deployed"] = df.old_pred_deployed == df.req_class


# ---- quintile tables ------------------------------------------------------
def quintiles(lp_col, wer_col, hyp_col, correct_cols):
    # Rank-based so ties (every empty transcript is -10) cannot collapse a bin edge.
    q = pd.qcut(df[lp_col].rank(method="first"), 5,
                labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    rows = []
    for name, g in df.groupby(q, observed=True):
        row = {"bin": str(name), "n": len(g),
               "logprob_min": float(g[lp_col].min()), "logprob_max": float(g[lp_col].max()),
               "mean_row_wer": float(g[wer_col].mean()),
               "corpus_wer": float(jiwer.wer(g.ref_norm.tolist(), g[hyp_col].tolist())),
               "frac_wer_gt_0.5": float((g[wer_col] > CONFIG["bad_wer"]).mean())}
        for c in correct_cols:
            row[f"acc_{c}"] = float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def split(lp_col, wer_col, thr, correct_cols):
    out = {}
    for side, m in (("below", df[lp_col] < thr), ("above", df[lp_col] >= thr)):
        g = df[m]
        out[side] = {"n": int(len(g)), "frac_wer_gt_0.5": float((g[wer_col] > CONFIG["bad_wer"]).mean()),
                     "mean_row_wer": float(g[wer_col].mean()),
                     **{f"acc_{c}": float(g[c].mean()) for c in correct_cols}}
    return out


new_min = float(df.new_logprob.quantile(CONFIG["quantile"]))
old_min = float(df.asr_logprob.quantile(CONFIG["quantile"]))
q_new = quintiles("new_logprob", "new_wer", "new_hyp_norm", ["new_correct"])
q_old = quintiles("asr_logprob", "old_wer", "old_hyp_norm", ["old_correct_deployed", "correct"])

pd.set_option("display.width", 200)
print("\n=== NEW decoder, by log-prob quintile ===")
print(q_new.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n=== OLD transcripts (transcripts.csv), by log-prob quintile ===")
print("    acc_correct = original out-of-fold; acc_old_correct_deployed = deployed model")
print(q_old.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\nASR_MIN in rules.py {RULES['ASR_MIN']}  | old p20 here {old_min:.4f} | NEW p20 {new_min:.4f}")

result = {
    "n_clips": int(len(df)), "missing_audio": missing,
    "decoder": PIPE["whisper_decode"], "compute_type": PIPE["whisper_compute_type"],
    "device": "cpu", "denoise": False, "audio_path": "load_audio -> trim (pipeline ASR path)",
    "logprob": "segment avg_logprob weighted by segment token count (src.pipeline.transcribe)",
    "ASR_MIN_rules_py": RULES["ASR_MIN"], "old_p20": old_min, "ASR_MIN_new_p20": new_min,
    "wer_corpus": {"old": float(jiwer.wer(df.ref_norm.tolist(), df.old_hyp_norm.tolist())),
                   "new": float(jiwer.wer(df.ref_norm.tolist(), df.new_hyp_norm.tolist()))},
    "perfect_transcripts": {"old": float((df.old_wer == 0).mean()), "new": float((df.new_wer == 0).mean())},
    "quintiles_new": q_new.to_dict("records"),
    "quintiles_old": q_old.to_dict("records"),
    "split_at_new_min": split("new_logprob", "new_wer", new_min, ["new_correct"]),
    "split_old_at_old_min": split("asr_logprob", "old_wer", old_min, ["old_correct_deployed", "correct"]),
    "accuracy_note": "acc_new_correct and acc_old_correct_deployed use the deployed ClinicalBERT "
                     "(fit on all phrases, in-sample); acc_correct is the original out-of-fold.",
    "no_speech_prob_max": {"median": float(df.no_speech_prob_max.median()),
                           "p95": float(df.no_speech_prob_max.quantile(0.95)),
                           "max": float(df.no_speech_prob_max.max()),
                           "frac_no_segments": float((df.n_segments == 0).mean())},
    "paired_logprob_shift_median": float((df.new_logprob - df.asr_logprob).median()),
}
json.dump(result, open(CONFIG["out_dir"] / "asr_confidence_v2.json", "w"), indent=2)
df.to_csv(CONFIG["out_dir"] / "asr_transcripts_v2.csv", index=False)
print("\nwrote", sorted(p.name for p in CONFIG["out_dir"].glob("asr_*")))
