"""
Noise robustness: the shipped pipeline on the held-out clips with ward noise mixed in.

CLAUDE.md step 8 - "the most valuable figure in the project". Two passes, deliberately
separate so that no classifier number exists until the right classifier is in hand:

  split        print and save the held-out phrases (deterministic from the phrase list
               and seed 42, recomputed as jobs/13_end_to_end does). Compare with that
               job's splits.csv BEFORE trusting any pass-2 number.
  transcribe   PASS 1. Whisper only, through the pipeline's own ASR path: the noisy clip
               is written to disk and read back by load_audio -> denoise -> trim ->
               CareVoicePipeline._transcribe, exactly as process() does. Output is the
               ASR cache and WER per SNR - model-independent, legitimate today. No
               classifier is called, no rules run, no request-class or priority column
               is written, no plot.
  evaluate     PASS 2. CareVoicePipeline.process() unchanged (rules R0-R7) with the
               ClinicalBERT job 13 retrained on the train phrases only (--bert), Whisper
               replaced by the pass-1 output for the same key. The Whisper input is
               re-hashed and must equal the hash pass 1 stored, so a cache entry can
               never be paired with different audio. Writes the curve and the plot.

Cache key = sha1(source clip) | sha1(noise clip) | SNR | seed, and the seed and the
choice of noise clip derive from the clip's content, never from loop position or row
index. A split that differs from Kaggle's just causes cache misses (re-transcription),
not mispairing; a miss regenerates byte-identical audio.

Noise: ESC-50 categories the deployed router (arousal head, notebook 11) never saw.
nb11 trained its non_speech class on EVERY fold of 12 ward-like categories
(clock_alarm, door_wood_knock, footsteps, clock_tick, washing_machine, vacuum_cleaner,
mouse_click, keyboard_typing, can_opening, door_wood_creaks, water_drops,
toilet_flush), so no unused fold exists; this curve uses disjoint categories instead
(CONFIG["noise_categories"]), mostly human non-speech sounds of a ward. The check is
enforced below. Mixing is src.preprocess.augment at active-speech SNR (CLAUDE.md §2.5).
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.mapping import CLASS_BASE_PRIORITY, PROMPT_TO_CLASS  # noqa: E402
from src.preprocess import (augment, denoise, energy_present, load_audio,  # noqa: E402
                            stable_seed, trim)

CONFIG = {
    "seed": 42,
    "test_size": 0.20,
    "snrs": [None, 20, 10, 5, 0],          # None = clean (no noise added)
    # ESC-50 categories NOT used to train the router (nb11 WARD list below).
    "noise_categories": ["coughing", "sneezing", "breathing", "snoring", "drinking_sipping",
                         "pouring_water", "laughing", "siren"],
    "router_training_categories": ["clock_alarm", "door_wood_knock", "footsteps", "clock_tick",
                                   "washing_machine", "vacuum_cleaner", "mouse_click",
                                   "keyboard_typing", "can_opening", "door_wood_creaks",
                                   "water_drops", "toilet_flush"],
    "download_dir": Path(os.environ.get("NOISE_DL", ROOT / "data" / "downloads")),
    "clips_dir": ROOT / "data" / "medical_heldout",
    "noise_dir": ROOT / "data" / "esc50_noise_unseen",
    "models": ROOT / "models",
    "out_dir": ROOT / "results",
    "cache": ROOT / "data" / "noise_asr_cache.json",
    "sr": 16000,
}
assert not set(CONFIG["noise_categories"]) & set(CONFIG["router_training_categories"])


def sha1_file(p) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def sha1_wav(w) -> str:
    return hashlib.sha1(np.ascontiguousarray(w, dtype=np.float32).tobytes()).hexdigest()


# ----------------------------------------------------------------- split

def heldout_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(held-out clips, held-out phrases), computed exactly as jobs/13_end_to_end does."""
    ov = pd.read_csv(ROOT / "results" / "transcripts.csv")[["file_name", "phrase", "prompt", "speaker_id"]]
    ov["true_class"] = ov.prompt.map(lambda p: PROMPT_TO_CLASS.get(str(p).strip(), (None, None))[0])
    ov = ov.dropna(subset=["true_class"]).reset_index(drop=True)
    ov["true_priority"] = ov.true_class.map(CLASS_BASE_PRIORITY)
    ov["prompt_priority"] = ov.prompt.map(lambda p: PROMPT_TO_CLASS[str(p).strip()][1])
    ph = ov.groupby("phrase").agg(cls=("true_class", "first"), n=("file_name", "size")).reset_index()
    _, te_ph = train_test_split(ph, test_size=CONFIG["test_size"], random_state=CONFIG["seed"],
                                stratify=ph.cls)
    te = ov[ov.phrase.isin(set(te_ph.phrase))].reset_index(drop=True)
    assert not set(te.phrase) & set(ov[~ov.phrase.isin(set(te_ph.phrase))].phrase)
    return te, te_ph.sort_values("phrase").reset_index(drop=True)


def cmd_split():
    te, te_ph = heldout_split()
    out = CONFIG["out_dir"] / "heldout_phrases_local.csv"
    te_ph[["phrase", "cls", "n"]].to_csv(out, index=False)
    digest = hashlib.sha1("\n".join(te_ph.phrase).encode()).hexdigest()
    print(f"{len(te_ph)} held-out phrases, {len(te)} clips  |  phrase-list sha1 {digest}")
    print(te_ph.cls.value_counts().to_string())
    for p in te_ph.phrase:
        print("  ", p)
    print("saved", out)


# ----------------------------------------------------------------- inputs

def extract_inputs(te: pd.DataFrame) -> list[Path]:
    dl = CONFIG["download_dir"]
    CONFIG["clips_dir"].mkdir(parents=True, exist_ok=True)
    want = set(te.file_name)
    have = {p.name for p in CONFIG["clips_dir"].glob("*.wav")}
    if want - have:
        with zipfile.ZipFile(next(dl.glob("medical-speech*.zip"))) as z:
            for n in z.namelist():
                b = os.path.basename(n)
                if b in want and b not in have:
                    (CONFIG["clips_dir"] / b).write_bytes(z.read(n))
                    have.add(b)
    assert not want - have, f"{len(want - have)} held-out clips not in the zip"

    CONFIG["noise_dir"].mkdir(parents=True, exist_ok=True)
    if not any(CONFIG["noise_dir"].glob("*.wav")):
        with zipfile.ZipFile(dl / "esc50.zip") as z:
            meta = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith("meta/esc50.csv"))))
            missing = set(CONFIG["noise_categories"]) - set(meta.category)
            assert not missing, f"not ESC-50 categories: {missing}"
            keep = set(meta[meta.category.isin(CONFIG["noise_categories"])].filename)
            for n in z.namelist():
                if os.path.basename(n) in keep:
                    (CONFIG["noise_dir"] / os.path.basename(n)).write_bytes(z.read(n))
    noise = sorted(CONFIG["noise_dir"].glob("*.wav"), key=sha1_file)   # content order
    print(f"held-out clips {len(want)}, unseen-category noise files {len(noise)}", flush=True)
    return noise


def noisy_clip(src: Path, src_sha: str, snr, noise: list[tuple[str, np.ndarray]]):
    """Return (wav, key). Noise clip and seed derive from the clip's content and SNR."""
    wav = load_audio(src)
    if snr is None:
        return wav, f"{src_sha}|none|clean|0"
    pick = np.random.default_rng(stable_seed(src_sha, snr, "pick")).integers(len(noise))
    n_sha, n_wav = noise[int(pick)]
    seed = stable_seed(src_sha, n_sha, snr)
    return augment(wav, [n_wav], snr, seed=seed), f"{src_sha}|{n_sha}|{snr}|{seed}"


def build_pipeline(bert_dir=None):
    from src.pipeline import CareVoicePipeline
    model_dir = CONFIG["models"]
    if bert_dir:
        tmp = Path(tempfile.mkdtemp(prefix="cv_models_"))
        for p in CONFIG["models"].iterdir():
            if p.name != "clinicalbert_requestclass":
                os.symlink(p, tmp / p.name)
        os.symlink(Path(bert_dir).resolve(), tmp / "clinicalbert_requestclass")
        model_dir = tmp
    return CareVoicePipeline(model_dir=str(model_dir), device="cpu")


def iterate(te, noise_files):
    noise = [(sha1_file(p), load_audio(p)) for p in noise_files]
    tmpdir = Path(tempfile.mkdtemp(prefix="cv_noise_"))
    shas = {f: sha1_file(CONFIG["clips_dir"] / f) for f in te.file_name}
    for snr in CONFIG["snrs"]:
        for r in te.itertuples():
            wav, key = noisy_clip(CONFIG["clips_dir"] / r.file_name, shas[r.file_name], snr, noise)
            path = tmpdir / f"{shas[r.file_name][:16]}_{snr}.wav"
            sf.write(path, wav, CONFIG["sr"])
            yield r, snr, key, path


# ----------------------------------------------------------------- pass 1

def cmd_transcribe():
    """Whisper only, via the pipeline's ASR path. Writes the cache and WER per SNR."""
    import jiwer
    te, _ = heldout_split()
    noise_files = extract_inputs(te)
    pipe = build_pipeline()
    cache = json.loads(CONFIG["cache"].read_text()) if CONFIG["cache"].exists() else {}
    rows, t0, n_new, total = [], time.time(), 0, len(te) * len(CONFIG["snrs"])
    for r, snr, key, path in iterate(te, noise_files):
        if key not in cache:
            # Identical to CareVoicePipeline.process(): load_audio -> energy_present ->
            # denoise -> trim -> _transcribe. Pass 2 verifies the hash of what Whisper got.
            wav, meta = load_audio(path, target_sr=CONFIG["sr"], return_meta=True)
            if energy_present(meta)[0]:
                wav = trim(denoise(wav, sr=CONFIG["sr"]), sr=CONFIG["sr"])
                text, conf = pipe._transcribe(wav)
                cache[key] = {"text": text, "conf": conf, "asr_input_sha1": sha1_wav(wav)}
            else:
                cache[key] = {"text": "", "conf": None, "asr_input_sha1": None}
            n_new += 1
            if n_new % 50 == 0:
                CONFIG["cache"].write_text(json.dumps(cache))
                el = time.time() - t0
                print(f"  {len(rows)+1}/{total} snr={snr}  {el/n_new:.2f}s/clip  "
                      f"eta {el/n_new*(total-len(rows))/60:.0f} min", flush=True)
        path.unlink()
        rows.append({"snr": "clean" if snr is None else f"{snr} dB", "ref": r.phrase.lower(),
                     "hyp": (cache[key]["text"] or "").lower()})
    CONFIG["cache"].write_text(json.dumps(cache))
    df = pd.DataFrame(rows)
    wer = (df.groupby("snr", sort=False)
             .apply(lambda g: pd.Series({"n": len(g), "wer": round(jiwer.wer(g.ref.tolist(), g.hyp.tolist()), 4),
                                         "empty_transcripts": int((g.hyp.str.strip() == "").sum())}))
             .reset_index())
    out = CONFIG["out_dir"] / "noise_wer_pass1.csv"
    wer.to_csv(out, index=False)
    print("\nPASS 1 - Whisper WER under unseen ward noise (model-independent):")
    print(wer.to_string(index=False))
    print("saved", out)


# ----------------------------------------------------------------- pass 2

def cmd_evaluate(bert_dir: str, tag: str):
    import jiwer
    if not bert_dir:
        raise SystemExit("--bert is required: pass 2 runs only with the held-out ClinicalBERT")
    te, _ = heldout_split()
    noise_files = extract_inputs(te)
    pipe = build_pipeline(bert_dir)
    cache = json.loads(CONFIG["cache"].read_text())
    current = {}

    def replay(wav):
        entry = cache[current["key"]]
        got = sha1_wav(wav)
        assert got == entry["asr_input_sha1"], (
            f"Whisper input differs from pass 1 for {current['key']}: {got} != {entry['asr_input_sha1']}")
        return entry["text"], entry["conf"]

    pipe._transcribe = replay
    rows = []
    for r, snr, key, path in iterate(te, noise_files):
        if key not in cache:
            raise KeyError(f"no pass-1 transcript for {r.file_name} at {snr}: run `transcribe` first")
        current["key"] = key
        d = pipe.process(path)
        path.unlink()
        rows.append({"file_name": r.file_name, "snr": "clean" if snr is None else f"{snr} dB",
                     "phrase": r.phrase, "prompt": r.prompt, "true_class": r.true_class,
                     "true_priority": int(r.true_priority), "prompt_priority": int(r.prompt_priority),
                     "transcript": d["transcript"], "asr_trusted": d["asr_trusted"],
                     "pred_class": d["request_class"], "pred_priority": d["final_priority"],
                     "rules_fired": ";".join(d["rules_fired"])})
    df = pd.DataFrame(rows)
    df.to_csv(CONFIG["out_dir"] / f"noise_curve_{tag}_raw.csv", index=False)
    s = []
    for lab, g in df.groupby("snr", sort=False):
        em = g[g.true_class == "Emergency"]
        s.append({"snr": lab, "n": len(g),
                  "wer": round(jiwer.wer(g.phrase.str.lower().tolist(),
                                         g.transcript.fillna("").str.lower().tolist()), 4),
                  "under_triage": round(float((g.pred_priority > g.true_priority).mean()), 4),
                  "emergency_under_triage": round(float((em.pred_priority > 1).mean()), 4),
                  "emergency_within_p2": round(float((em.pred_priority <= 2).mean()), 4),
                  "emergency_recall_class": round(float((em.pred_class == "Emergency").mean()), 4),
                  "class_accuracy": round(float((g.pred_class == g.true_class).mean()), 4),
                  "routed_no_speech": round(float((g.pred_class == "Unclassified").mean()), 4),
                  "alarm_load_p1_when_p3p4": int(((g.pred_priority == 1) & (g.prompt_priority >= 3)).sum())})
    s = pd.DataFrame(s)
    s.to_csv(CONFIG["out_dir"] / f"noise_curve_{tag}.csv", index=False)
    print(s.to_string(index=False))
    plot(s, tag)


def plot(s: pd.DataFrame, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = range(len(s))
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for col, label, style in [("emergency_under_triage", "Emergency under-triage (not P1)", "-o"),
                              ("under_triage", "Under-triage, all calls", "-v"),
                              ("emergency_recall_class", "Emergency recall (class)", "--s"),
                              ("class_accuracy", "Request-class accuracy", "--^"),
                              ("wer", "Whisper WER", ":d")]:
        ax.plot(x, s[col], style, label=label)
    ax.set_xticks(list(x), s.snr)
    ax.set_xlabel("added noise, active-speech SNR (ESC-50 categories unseen by the router)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.set_title(f"Held-out clips under ward noise - ClinicalBERT trained on train phrases only "
                 f"(n={int(s.n.iloc[0])}/level)", fontsize=8)
    ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    out = CONFIG["out_dir"] / f"noise_curve_{tag}.png"
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["split", "transcribe", "evaluate"])
    ap.add_argument("--bert", help="pass 2 only: job 13's held-out ClinicalBERT dir")
    ap.add_argument("--tag", default="heldout")
    a = ap.parse_args()
    np.random.seed(CONFIG["seed"])
    print(f"seed {CONFIG['seed']}, snrs {CONFIG['snrs']}, noise {CONFIG['noise_categories']}", flush=True)
    {"split": cmd_split, "transcribe": cmd_transcribe}.get(a.mode, lambda: cmd_evaluate(a.bert, a.tag))()
