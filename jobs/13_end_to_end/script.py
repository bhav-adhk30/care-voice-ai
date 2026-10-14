"""
Job 13 — end-to-end evaluation on a held-out split (Kaggle, GPU for retraining).

Design (unchanged from the notebook it came from):
  - Split by PHRASE, 80/20, stratified by class, seed 42. 706 phrases, each read by
    ~9 speakers; a clip-level split would put the same sentence on both sides.
  - ClinicalBERT is retrained on the train phrases only and the full pipeline runs
    on the held-out clips. The app keeps shipping the model trained on everything.
  - Whisper was never trained on this corpus and the rules are deterministic; they
    need no split. The acoustic branch is ROUTER-ONLY: WavLM's arousal head decides
    speech / non-speech. The WavLM distress gate is retired and the prosodic gate
    has no threshold (rules.CONFIG DISTRESS_HIGH is None), so no acoustic
    escalation exists to evaluate and none is reported.
  - The rules are the shipped ones: CareVoicePipeline calls src.rules.apply_rules
    (Rules 0-7, including Rule 7 which can lower a priority). The script asserts the
    code dataset is the committed revision (EXPECTED_SHA below).
  - Part B checks the router on constructed non-speech sets (ward noise, silence).

Changes from the pasted notebook, and why:
  1. Every input path is resolved by glob: Kaggle mounts datasets at
     /kaggle/input/datasets/<owner>/<slug>/.
  2. ClinicalBERT is retrained from emilyalsentzer/Bio_ClinicalBERT. The
     care-voice-clinicalbert dataset is the deployed checkpoint, fit on every phrase:
     starting from it would leak the test set. It is used only to check class order.
  3. Training uses the UNIQUE train phrases with class-weighted loss (CLAUDE.md §4,
     §5), not every clip: clip rows repeat each phrase ~9x and reweight by recording
     count. The text-only score is on unique held-out phrases, comparable with the
     published phrase-level 0.855.
  4. Whisper (Systran/faster-whisper-small) and WavLM are downloaded into the models
     dir first; CareVoicePipeline refuses to start without them.
  5. Result keys match process(): non_speech_prob and latency_ms (the notebook read
     non_speech and latency, which do not exist).
  6. The pipeline runs on CPU so Whisper's int8 path matches the deployed app and
     the ASR_MIN derivation. GPU is used only for retraining.
  7. scikit-learn pinned to 1.6.1, the version that pickled the v2 heads.

Ground truth priority, two readings of the project's mapping (not a nurse's judgement;
no corpus labels clinical urgency):
  - class level  CLASS_BASE_PRIORITY of the mapped class - what the system can
                 predict, and the stricter test: HEADLINE under-triage.
  - prompt level the per-prompt priority in PROMPT_TO_CLASS (Dizzy P2, Acne P4...);
                 used for the over-triage / alarm-load view (P1 when P3 or P4).
HEADLINE METRIC: under-triage rate (assigned less urgent than correct), overall and
for Emergency. Accuracy and macro F1 are secondary.

Outputs (/kaggle/working): splits.csv, end_to_end_v3.csv, end_to_end_v3.json,
priority_confusion_class_v3.csv, priority_confusion_prompt_v3.csv, router_set_v3.csv,
router_summary_v3.csv, train_loss_v3.csv.

Local smoke test (no downloads, no training, fixture audio stands in for clips):
    CV13_SMOKE=1 python jobs/13_end_to_end/script.py
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"          # downloads happen here; src.pipeline would default it to 1
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SMOKE = os.environ.get("CV13_SMOKE") == "1"
ON_KAGGLE = Path("/kaggle/input").exists() and not SMOKE
LOCAL_REPO = Path(__file__).resolve().parents[2]

if ON_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper",
                    "pyloudnorm", "jiwer", "opensmile", "scikit-learn==1.6.1"], check=True)


def find_one(pattern, what):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{what}: nothing matches /kaggle/input/**/{pattern}")
    return Path(hits[0])


if ON_KAGGLE:
    REPO = find_one("src/pipeline.py", "care-voice-code").parent.parent
    HEADS = find_one("arousal_head.joblib", "care-voice-heads-v2").parent
    DEPLOYED_BERT = find_one("classes.json", "care-voice-clinicalbert").parent
    WORK = Path("/kaggle/working")
else:
    REPO = LOCAL_REPO
    HEADS = LOCAL_REPO / "models"
    DEPLOYED_BERT = LOCAL_REPO / "models" / "clinicalbert_requestclass"
    WORK = Path(os.environ.get("CV13_OUT", "/tmp/cv13"))
sys.path.insert(0, str(REPO))
WORK.mkdir(parents=True, exist_ok=True)

# The evaluated system must be the shipped system: rules.py, mapping.py and
# pipeline.py must be byte-identical to the committed revision this job was
# written against. Filled in at push time (see jobs/13_end_to_end/push.sh).
EXPECTED_SHA = json.loads(os.environ.get("CV13_EXPECTED_SHA", "null") or "null") or {
    "src/rules.py": "4dd440ffcd4b0ad609d9275c484da6091bc1a67873eecf9cdfa9215d8f2ff436", "src/mapping.py": "3fa9e56cc7c8b9a00f5ad91e0b8ed47b375fc84f0f9b48a7b4529d67f4fcc304",
    "src/pipeline.py": "5d0d53f82b905fac0db423e0bb9bf934a333b6020228c757b22e0a5c2ec63ce9"}
for rel, want in EXPECTED_SHA.items():
    got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
    assert want.startswith("__") or got == want, f"{rel}: code dataset {got[:12]} != committed {want[:12]}"
    print(f"{rel} sha256 {got[:12]}" + ("" if want.startswith("__") else " == committed"))
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

CONFIG = {
    "seed": 42,
    "test_size": 0.20,
    "base_model": "emilyalsentzer/Bio_ClinicalBERT",
    "epochs": 10,                 # unique phrases only (~565), so ~9x fewer rows than clip-level
    "batch_size": 16,
    "lr": 2e-5,
    "max_length": 64,
    "smoke_test_clips": 8,
    "esc_ward": ["washing_machine", "vacuum_cleaner", "clock_tick", "clock_alarm",
                 "footsteps", "door_wood_knock", "keyboard_typing", "water_drops"],
    "esc_max": 160,
}
print(f"repo={REPO} heads={HEADS} deployed_bert={DEPLOYED_BERT} work={WORK} "
      f"kaggle={ON_KAGGLE} smoke={SMOKE} seed={CONFIG['seed']}", flush=True)

import numpy as np  # noqa: E402
from src import rules as RULES  # noqa: E402

# Rule set sanity: the rules the paper describes are the ones being run.
_src = (REPO / "src" / "rules.py").read_text()
for r in ("R0_no_speech", "R4_emergency_floor", "R5_care_request", "R6_distress_words",
          "R7_comfort_request"):
    assert r in _src, f"rule {r} missing from src/rules.py"
assert RULES.CONFIG["DISTRESS_HIGH"] is None, "acoustic distress threshold is set; this job assumes router-only"
print("rules: R0-R7 present, DISTRESS_HIGH None (acoustic branch router-only)", flush=True)
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.mapping import CLASS_BASE_PRIORITY, PROMPT_TO_CLASS  # noqa: E402

np.random.seed(CONFIG["seed"])

# =============================================================================
# 1. the split
# =============================================================================
if ON_KAGGLE:
    ov = pd.read_csv(find_one("overview-of-recordings.csv", "medical speech dataset"))
    paths = {}
    for p in glob.glob("/kaggle/input/**/*.wav", recursive=True):
        if "medical" in p.lower():
            paths.setdefault(os.path.basename(p), p)     # the zip extracts two copies
else:
    ov = pd.read_csv(REPO / "results" / "transcripts.csv")[
        ["file_name", "phrase", "prompt", "speaker_id"]]
    fx = [str(REPO / "tests/fixtures/tts_chest_pain.wav"), str(REPO / "tests/fixtures/tts_knee_pain.wav")]
    paths = {f: fx[i % 2] for i, f in enumerate(ov.file_name)}
ov["path"] = ov.file_name.map(paths)
ov = ov.dropna(subset=["path"]).reset_index(drop=True)
ov["true_class"] = ov.prompt.map(lambda p: PROMPT_TO_CLASS.get(str(p).strip(), (None, None))[0])
unmapped = int(ov.true_class.isna().sum())
if unmapped:
    print(f"WARNING {unmapped} rows with an unmapped prompt:\n"
          + ov[ov.true_class.isna()].prompt.value_counts().to_string())
ov = ov.dropna(subset=["true_class"]).reset_index(drop=True)
ov["true_priority"] = ov.true_class.map(CLASS_BASE_PRIORITY)
ov["prompt_priority"] = ov.prompt.map(lambda p: PROMPT_TO_CLASS[str(p).strip()][1])

ph = ov.groupby("phrase").agg(cls=("true_class", "first"), n=("file_name", "size")).reset_index()
print(f"{len(ph)} unique phrases, {len(ov)} clips\n{ph.cls.value_counts().to_string()}")
tr_ph, te_ph = train_test_split(ph, test_size=CONFIG["test_size"], random_state=CONFIG["seed"],
                                stratify=ph.cls)
ov["split"] = np.where(ov.phrase.isin(set(te_ph.phrase)), "test", "train")
overlap = set(ov[ov.split == "train"].phrase) & set(ov[ov.split == "test"].phrase)
assert not overlap, f"phrase leak: {len(overlap)}"
ov[["file_name", "phrase", "prompt", "true_class", "true_priority", "speaker_id", "split"]].to_csv(
    WORK / "splits.csv", index=False)
tr, te = ov[ov.split == "train"], ov[ov.split == "test"].reset_index(drop=True)
print(f"\ntrain {len(tr)} clips / {tr.phrase.nunique()} phrases")
print(f"test  {len(te)} clips / {te.phrase.nunique()} phrases")
print("test clips by class:\n" + te.true_class.value_counts().to_string())
print("speakers in both splits:", len(set(tr.speaker_id) & set(te.speaker_id)),
      "(expected: the split is by phrase; the text branch reads transcripts)")

# =============================================================================
# 2. retrain ClinicalBERT on the train phrases only (from the base model)
# =============================================================================
import torch  # noqa: E402

CLASSES = sorted(ov.true_class.unique())
deployed_classes = json.loads((DEPLOYED_BERT / "classes.json").read_text())["classes"]
assert CLASSES == deployed_classes, f"class order {CLASSES} != deployed {deployed_classes}"
EVAL_BERT = MODELS / "clinicalbert_requestclass"

train_ph = tr.drop_duplicates("phrase")[["phrase", "true_class"]].reset_index(drop=True)
test_ph = te.drop_duplicates("phrase")[["phrase", "true_class"]].reset_index(drop=True)

if SMOKE:
    # No training locally: the deployed model stands in, so this checks plumbing only.
    # Symlinked, not copied: the dev machine has no room for a second 433 MB copy.
    if not EVAL_BERT.exists():
        os.symlink(DEPLOYED_BERT, EVAL_BERT)
else:
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments)
    c2i = {c: i for i, c in enumerate(CLASSES)}
    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"])

    class DS(Dataset):
        def __init__(self, df):
            self.enc = tok(df.phrase.tolist(), truncation=True, padding="max_length",
                           max_length=CONFIG["max_length"])
            self.y = [c2i[c] for c in df.true_class]

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            d["labels"] = torch.tensor(self.y[i])
            return d

    counts = train_ph.true_class.value_counts()
    weights = torch.tensor([len(train_ph) / (len(CLASSES) * counts[c]) for c in CLASSES],
                           dtype=torch.float)
    print("train phrases by class:", counts.to_dict(), "| class weights:",
          dict(zip(CLASSES, weights.round(decimals=3).tolist())))

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits, labels, weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    from transformers import TrainerCallback

    class LeakGuard(TrainerCallback):
        """Smell test, never fatal. The phrase-overlap assert before training is the
        real leak check. The leaked run sat at 0.0114 by step 50; an honest 4-class
        head from the base model is ~1.39 at step 0 and can be 0.3-0.4 three epochs
        in, so only an absurdly low loss this early is flagged. The run continues
        either way; judge it from train_loss_v3.csv."""
        def __init__(self):
            self.log, self.flagged = [], False

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.log.append({"step": state.global_step, "loss": logs["loss"]})
                print(f"  step {state.global_step} loss {logs['loss']:.4f}", flush=True)
                if state.global_step <= 50 and logs["loss"] < 0.1 and not self.flagged:
                    self.flagged = True
                    print("!" * 70 + f"\nLEAK SMELL: train loss {logs['loss']:.4f} at step "
                          f"{state.global_step}. Check which checkpoint was loaded before trusting "
                          "these numbers. Continuing.\n" + "!" * 70, flush=True)

    torch.manual_seed(CONFIG["seed"])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["base_model"], num_labels=len(CLASSES))
    steps = int(np.ceil(len(train_ph) / CONFIG["batch_size"])) * CONFIG["epochs"]
    args = TrainingArguments(
        output_dir=str(WORK / "bert_eval"), num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"], learning_rate=CONFIG["lr"],
        warmup_steps=int(0.1 * steps),     # warmup_ratio was removed in transformers 5
        weight_decay=0.01, logging_steps=10, save_strategy="no",
        seed=CONFIG["seed"], fp16=torch.cuda.is_available(), report_to=[])
    guard = LeakGuard()
    t0 = time.time()
    WeightedTrainer(model=model, args=args, train_dataset=DS(train_ph), callbacks=[guard]).train()
    pd.DataFrame(guard.log).to_csv(WORK / "train_loss_v3.csv", index=False)
    LEAK_SMELL = guard.flagged
    print(f"train loss: first {guard.log[0]['loss']:.3f}, last {guard.log[-1]['loss']:.3f}", flush=True)
    print(f"retrained in {time.time()-t0:.0f}s", flush=True)
    model.save_pretrained(EVAL_BERT)
    tok.save_pretrained(EVAL_BERT)
    (EVAL_BERT / "classes.json").write_text(json.dumps({"classes": CLASSES}))
    del model
    torch.cuda.empty_cache()

# text-only on held-out unique phrases, ground-truth text
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

_tok = AutoTokenizer.from_pretrained(EVAL_BERT)
_m = AutoModelForSequenceClassification.from_pretrained(EVAL_BERT).eval()
with torch.no_grad():
    enc = _tok(test_ph.phrase.tolist(), truncation=True, padding=True,
               max_length=CONFIG["max_length"], return_tensors="pt")
    text_pred = [CLASSES[i] for i in _m(**enc).logits.argmax(-1).numpy()]
del _m
print("\n=== text branch on held-out phrases (ground-truth text) ===")
print(classification_report(test_ph.true_class, text_pred, digits=3, zero_division=0))
TEXT_ONLY_F1 = float(f1_score(test_ph.true_class, text_pred, average="macro"))
TEXT_ONLY_EMERG_RECALL = float(np.mean([p == "Emergency" for p, t in zip(text_pred, test_ph.true_class)
                                        if t == "Emergency"]))
print(f"macro F1 {TEXT_ONLY_F1:.3f}, Emergency recall {TEXT_ONLY_EMERG_RECALL:.3f} "
      f"(published out-of-fold: 0.855 / 0.826)", flush=True)

# =============================================================================
# 3. stage the models and build the pipeline
# =============================================================================
shutil.copy(HEADS / "arousal_head.joblib", MODELS / "arousal_head.joblib")   # router only
if SMOKE:
    for d in ("faster-whisper-small", "wavlm-base-plus"):
        if not (MODELS / d).exists():
            os.symlink(LOCAL_REPO / "models" / d, MODELS / d)
else:
    from huggingface_hub import snapshot_download
    snapshot_download("Systran/faster-whisper-small", local_dir=MODELS / "faster-whisper-small")
    snapshot_download("microsoft/wavlm-base-plus", local_dir=MODELS / "wavlm-base-plus",
                      allow_patterns=["*.json", "*.bin", "*.safetensors"])
print("models dir:", sorted(p.name for p in MODELS.iterdir()), flush=True)

from src.pipeline import CareVoicePipeline  # noqa: E402

pipe = CareVoicePipeline(model_dir=str(MODELS), device="cpu")
print(f"pipeline loaded in {pipe.load_ms:.0f} ms; versions {pipe.model_versions}", flush=True)


def run(path):
    d = pipe.process(path)
    return {
        "transcript": d["transcript"], "asr_confidence": d["asr_confidence"],
        "asr_trusted": d["asr_trusted"], "pred_class": d["request_class"],
        "class_confidence": d["class_confidence"], "base_priority": d["base_priority"],
        "pred_priority": d["final_priority"], "escalated": d["escalated"],
        "classifier_class": d.get("classifier_class"), "non_speech_prob": d["non_speech_prob"],
        "speech_present": d["speech_present"], "rules_fired": ";".join(d["rules_fired"]),
        "audio_error": d["audio_error"], "latency_ms": d["latency_ms"]["total"],
    }


# =============================================================================
# 4. Part A: the whole system on the held-out clips
# =============================================================================
if SMOKE:
    te = te.head(CONFIG["smoke_test_clips"])
OUT = WORK / "end_to_end_v3.csv"
rows, t0 = [], time.time()
for i, r in enumerate(te.itertuples()):
    base = {"file_name": r.file_name, "phrase": r.phrase, "prompt": r.prompt,
            "true_class": r.true_class, "true_priority": int(r.true_priority),
            "prompt_priority": int(r.prompt_priority)}
    try:
        rows.append({**base, **run(r.path)})
    except Exception as e:  # recorded, never silently dropped
        rows.append({**base, "error": f"{type(e).__name__}: {e}"})
    if (i + 1) % 100 == 0 or i + 1 == len(te):
        pd.DataFrame(rows).to_csv(OUT, index=False)
        el = time.time() - t0
        print(f"  {i+1}/{len(te)}  {el:.0f}s  eta {el/(i+1)*(len(te)-i-1)/60:.0f} min", flush=True)

ee = pd.DataFrame(rows)
errors = int(ee["error"].notna().sum()) if "error" in ee else 0
ok = ee[ee.pred_priority.notna()].copy()
ok["pred_priority"] = ok.pred_priority.astype(int)
ok["delta"] = ok.pred_priority - ok.true_priority            # positive = less urgent (under-triage)
ok["delta_prompt"] = ok.pred_priority - ok.prompt_priority
em = ok[ok.true_class == "Emergency"]

under = float((ok.delta > 0).mean())
em_under = float((em.pred_priority > em.true_priority).mean())
exact, over = float((ok.delta == 0).mean()), float((ok.delta < 0).mean())
acc = float((ok.pred_class == ok.true_class).mean())
macro_f1 = float(f1_score(ok.true_class, ok.pred_class, average="macro",
                          labels=sorted(ok.true_class.unique())))
cm_class = pd.crosstab(ok.true_priority, ok.pred_priority, rownames=["correct (class)"],
                       colnames=["assigned"])
cm_prompt = pd.crosstab(ok.prompt_priority, ok.pred_priority, rownames=["correct (prompt)"],
                        colnames=["assigned"])
alarm = ok[(ok.pred_priority == 1) & (ok.prompt_priority >= 3)]
fired = ok.rules_fired.fillna("")
rule_counts = {r: int(fired.str.contains(r).sum()) for r in
               ("R0_no_speech", "R4_emergency_floor", "R5_care_request", "R6_distress_words",
                "R7_comfort_request")}
r7_on_emergency = int(em.rules_fired.fillna("").str.contains("R7").sum())

print(f"\n=== PART A: {len(ok)} held-out clips, audio in -> priority out (errors: {errors}) ===")
print(f"  UNDER-TRIAGE {under:.1%} overall | Emergency {em_under:.1%} "
      f"({int((em.pred_priority > 1).sum())} of {len(em)} not at P1)")
print(f"  exact {exact:.1%}  over-triage {over:.1%}  class accuracy {acc:.1%}  macro F1 {macro_f1:.3f}")
for p in sorted(ok.true_priority.unique()):
    sub = ok[ok.true_priority == p]
    print(f"  correct P{p}  n={len(sub):4d}  under={float((sub.delta > 0).mean()):.3f}")
print(f"\nEmergency n={len(em)}: P1 {float((em.pred_priority == 1).mean()):.3f}, "
      f"P1-P2 {float((em.pred_priority <= 2).mean()):.3f}, below P2 {int((em.pred_priority > 2).sum())}")
if (em.pred_priority > 1).any():
    print("  not P1, by prompt:\n" + em[em.pred_priority > 1].prompt.value_counts().to_string())
print("\npriority confusion, class-level truth:\n" + cm_class.to_string())
print("\npriority confusion, prompt-level truth:\n" + cm_prompt.to_string())
print(f"\nalarm load: {len(alarm)} calls at P1 whose prompt-level priority is P3 or P4 "
      f"({len(alarm)/len(ok):.1%} of calls; {len(alarm)/max(1,int((ok.pred_priority==1).sum())):.1%} of all P1 alarms)")
if len(alarm):
    print(alarm.prompt.value_counts().head(10).to_string())
print("\nrequest class from audio:")
print(classification_report(ok.true_class, ok.pred_class, digits=3, zero_division=0))
print("rules fired on held-out clips:", rule_counts, f"| R7 on true Emergency clips: {r7_on_emergency}")
u = ok[ok.delta > 0]
loss = {
    "under_triaged": int(len(u)),
    "asr_not_trusted": int((~u.asr_trusted.astype(bool)).sum()),
    "empty_transcript": int(u.transcript.fillna("").str.strip().eq("").sum()),
    "routed_unclassified": int((u.pred_class == "Unclassified").sum()),
    "trusted_transcript_wrong_class": int((u.asr_trusted.astype(bool) & (u.pred_class != "Unclassified")).sum()),
}
print("where the under-triage comes from:", loss)
print(f"latency median {ok.latency_ms.median():.0f} ms, p95 {ok.latency_ms.quantile(0.95):.0f} ms", flush=True)

summary = {
    "HEADLINE_under_triage": round(under, 4), "HEADLINE_emergency_under_triage": round(em_under, 4),
    "n_test_clips": int(len(ok)), "n_test_phrases": int(te.phrase.nunique()), "errors": errors,
    "split": f"phrase-grouped {1-CONFIG['test_size']:.0%}/{CONFIG['test_size']:.0%}, stratified, seed {CONFIG['seed']}",
    "eval_clinicalbert": "retrained from emilyalsentzer/Bio_ClinicalBERT on unique train phrases, "
                         f"class-weighted, {CONFIG['epochs']} epochs" if not SMOKE else "SMOKE: deployed model",
    "train_loss_leak_smell": bool(globals().get("LEAK_SMELL", False)),
    "acoustic_branch": "router only (speech / non-speech); no acoustic escalation (DISTRESS_HIGH None)",
    "text_only_heldout": {"macro_f1": round(TEXT_ONLY_F1, 4), "emergency_recall": round(TEXT_ONLY_EMERG_RECALL, 4),
                          "n_phrases": int(len(test_ph))},
    "exact_match": round(exact, 4), "over_triage": round(over, 4),
    "class_accuracy": round(acc, 4), "class_macro_f1": round(macro_f1, 4),
    "under_triage_by_correct_priority": {int(p): round(float((g.delta > 0).mean()), 4)
                                         for p, g in ok.groupby("true_priority")},
    "emergency_n": int(len(em)), "emergency_p1": round(float((em.pred_priority == 1).mean()), 4),
    "emergency_within_p2": round(float((em.pred_priority <= 2).mean()), 4),
    "emergency_below_p2": int((em.pred_priority > 2).sum()),
    "emergency_not_p1_by_prompt": em[em.pred_priority > 1].prompt.value_counts().to_dict(),
    "alarm_load_p1_when_prompt_p3_or_p4": int(len(alarm)),
    "alarm_load_share_of_p1": round(len(alarm) / max(1, int((ok.pred_priority == 1).sum())), 4),
    "rules_fired": rule_counts, "r7_on_true_emergency": r7_on_emergency,
    "losses": loss,
    "latency_median_ms": float(ok.latency_ms.median()), "latency_p95_ms": float(ok.latency_ms.quantile(0.95)),
    "ground_truth": "class level = CLASS_BASE_PRIORITY (headline); prompt level = PROMPT_TO_CLASS priority",
}
json.dump(summary, open(WORK / "end_to_end_v3.json", "w"), indent=2)
cm_class.to_csv(WORK / "priority_confusion_class_v3.csv")
cm_prompt.to_csv(WORK / "priority_confusion_prompt_v3.csv")

# =============================================================================
# 5. Part B: the router on constructed non-speech sets
#    (VIVAE distress groups removed: there is no acoustic escalation to test)
# =============================================================================
if SMOKE:
    esc_paths = [str(REPO / "tests/fixtures/ward_noise_4s.wav")]
else:
    ESC = WORK / "esc50"
    if not glob.glob(f"{ESC}/**/*.wav", recursive=True):
        urllib.request.urlretrieve("https://github.com/karolpiczak/ESC-50/archive/master.zip",
                                   WORK / "esc50.zip")
        with zipfile.ZipFile(WORK / "esc50.zip") as z:
            z.extractall(ESC)
    meta = pd.read_csv(glob.glob(f"{ESC}/**/esc50.csv", recursive=True)[0])
    adir = os.path.dirname(glob.glob(f"{ESC}/**/audio/*.wav", recursive=True)[0])
    esc_paths = [os.path.join(adir, f) for f in
                 meta[meta.category.isin(CONFIG["esc_ward"])].filename][: CONFIG["esc_max"]]

SIL = WORK / "silence"
SIL.mkdir(exist_ok=True)
rng, sil_paths = np.random.default_rng(CONFIG["seed"]), []
for dur in (0.5, 1, 2, 4):
    q = SIL / f"zeros_{dur}s.wav"
    sf.write(q, np.zeros(int(dur * 16000), dtype=np.float32), 16000)
    sil_paths.append(str(q))
    for db in (-60, -50, -40):
        q = SIL / f"noise_{db}_{dur}s.wav"
        sf.write(q, rng.normal(0, 10 ** (db / 20), int(dur * 16000)).astype(np.float32), 16000)
        sil_paths.append(str(q))

GROUPS = [
    ("ward noise", esc_paths, "Unclassified P2",
     lambda d: d["pred_class"] == "Unclassified" and d["pred_priority"] == 2),
    ("silence / noise floor", sil_paths, "Unclassified P2",
     lambda d: d["pred_class"] == "Unclassified" and d["pred_priority"] == 2),
]
br = []
for name, pl, exp, chk in GROUPS:
    print(f"{name}: {len(pl)} clips, expect {exp}", flush=True)
    for p in pl:
        try:
            d = run(p)
            br.append({"group": name, "expectation": exp, "path": os.path.basename(p), **d,
                       "met": bool(chk(d))})
        except Exception as e:
            br.append({"group": name, "expectation": exp, "path": os.path.basename(p),
                       "error": f"{type(e).__name__}: {e}", "met": False})
bdf = pd.DataFrame(br)
bdf.to_csv(WORK / "router_set_v3.csv", index=False)
summ = (bdf.groupby("group", sort=False)
           .agg(n=("met", "size"), expectation=("expectation", "first"), met=("met", "mean"),
                p1_rate=("pred_priority", lambda s: float((s == 1).mean())))
           .round(3))
summ.to_csv(WORK / "router_summary_v3.csv")
summary["part_b_router"] = summ.reset_index().to_dict("records")
json.dump(summary, open(WORK / "end_to_end_v3.json", "w"), indent=2)

print(f"""
======================= EVALUATION SUMMARY =======================
PART A - {len(ok)} held-out clips ({te.phrase.nunique()} phrases never seen in training)
  UNDER-TRIAGE (headline)       {under:.1%} overall, {em_under:.1%} for Emergency
  text only, held-out phrases   macro F1 {TEXT_ONLY_F1:.3f}, Emergency recall {TEXT_ONLY_EMERG_RECALL:.3f}
  class accuracy / macro F1     {acc:.1%} / {macro_f1:.3f}
  exact priority match          {exact:.1%}
  over-triage                   {over:.1%}
  Emergency reached P1          {float((em.pred_priority == 1).mean()):.1%}
  Emergency reached P1 or P2    {float((em.pred_priority <= 2).mean()):.1%}
  alarm load (P1, correct P3/4) {len(alarm)} calls ({len(alarm)/max(1,int((ok.pred_priority==1).sum())):.1%} of P1 alarms)
  R7 on true Emergency clips    {r7_on_emergency}
  median latency                {ok.latency_ms.median():.0f} ms
PART B - router, constructed non-speech
{summ.to_string()}
""", flush=True)
