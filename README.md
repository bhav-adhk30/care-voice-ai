# Care-Voice AI

**Intelligent voice-based patient request classification and prioritisation.**
Final-year project, Amrita Vishwa Vidyapeetham, Dept. of CSE (AIE) — A. Bhavishya,
K. Sumayah, C. Vinuthna.

A nurse-call button tells the nurse only that *someone in bed 14 wants something*. Care-Voice AI
lets the patient speak instead. In about two seconds, on a laptop with no internet, the nurse
station shows **what they need, how urgent it is (P1–P4), why, and how they sound**.

```
Patient speaks → Whisper (speech → text) → ClinicalBERT (request type) ─┐
              └→ WavLM, frozen (is it really speech?) ──────────────────┴→ Safety rules R0, R4–R9 → Nurse station
Emergency button ─────────────────────────────────────────────────────────→ P1 (skips every model)
```

- **Request types:** Emergency, Pain, Assistance, General (ClinicalBERT, fine-tuned) plus
  Medication, comfort and toilet requests (tested keyword rules: the training data has only symptoms).
- **Priority:** written, auditable rules — never a black box — with a plain-English reason per call.
- **Never drops a call:** silence, noise or a groan is routed at P2, like a pressed button.
- **Sentiment:** text emotion shown to the nurse; it never changes priority.
- **Nurse station:** live queue, waiting timer (P3→P2 after 15 min, P2→P1 after 5 min), red
  flashing alarm + beep for P1, append-only audit log.

---

## Prerequisites

| Requirement | Version / size | Notes |
|---|---|---|
| macOS or Linux | tested on macOS 15 (Apple Silicon) | Windows untested |
| Python | **3.12** | 3.10–3.11 also work; 3.13 untested |
| ffmpeg | any recent (tested 8.0) | decodes m4a/mp4/mp3. macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg` |
| RAM | 8 GB minimum | close other apps before a demo (models ≈ 1.5 GB in memory) |
| Disk | ≈ 3 GB | ≈ 1.5 GB of models + Python packages |
| Internet | setup only | the app runs fully offline (`HF_HUB_OFFLINE=1`) |
| Browser | Chrome, Arc, Safari, Edge | microphone recording needs `http://localhost:8501` |

No GPU is needed to run the system. Training was done on Kaggle GPUs (`jobs/`).

---

## Setup

```bash
git clone https://github.com/bhav-adhk30/care-voice-ai.git
cd care-voice-ai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Models (`models/`, not in git — too large for GitHub)

| Folder | What | How to get it |
|---|---|---|
| `faster-whisper-small/` | Whisper small (CTranslate2) | `python scripts/fetch_models.py` |
| `wavlm-base-plus/` | WavLM Base+, frozen | `python scripts/fetch_models.py` |
| `emotion-english-distilroberta-base/` | sentiment model | `python scripts/fetch_models.py` |
| `clinicalbert_requestclass/` | **our fine-tuned ClinicalBERT** (433 MB) | trained here — see below |
| `arousal_head.joblib` | **our speech / non-speech head** | trained here — see below |

The two models trained in this project are stored in the team's Kaggle datasets
`care-voice-clinicalbert` and `care-voice-heads-v2`. Ask the team for access, then:

```bash
kaggle datasets download bhavishyaadhikari/care-voice-clinicalbert -p models/clinicalbert_requestclass --unzip
kaggle datasets download bhavishyaadhikari/care-voice-heads-v2 -p models --unzip
```

To retrain them instead, see `jobs/` and `notebooks/` (Kaggle GPU).

---

## Running the demo

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** (use `localhost`, not a `192.168…` address, or the browser blocks the mic).

1. **Submit a Call** — press the red **EMERGENCY** button, or speak a request: *Record*, *Upload a
   file* (any phone format), or *Demo clip*. The decision appears on the right with a
   **How it decided** panel.
2. **Nurse Station** — the live queue, sorted by priority then waiting time; P1 calls flash and beep.
3. **Results** — how it works, measured accuracy, results on our own voices, limitations.

Sidebar (presenter controls): **Cached mode** replays pre-computed results if a model fails on
stage · **Demo clock** speeds up the waiting timer · **Sentiment** · **Alert sound** · **Reset demo**.

The first call takes ~15 s while models load, then ~2 s per call.

> **Demo clips.** The team's voice recordings are kept out of this public repository for privacy
> (`app/demo_clips/*__*.wav` is gitignored). Clone users get the no-speech clip only; record or
> upload your own audio, e.g. *"I have a severe pain in my chest"*, *"Can you close the window?"*,
> *"What time is my next medication?"*.

**Microphone not working?** Click **Allow microphone** above the recorder: it triggers the
browser prompt and says exactly what is blocking it (site permission or macOS
*System Settings → Privacy & Security → Microphone*). File upload always works.

---

## Tests

```bash
python -m pytest tests -q
```

169 tests: the rule table, error handling (missing/empty/corrupt audio still produces a routed
call), every audio format, the emergency button, and a golden file of pipeline outputs. The tests
marked `pipeline` need the models in `models/`.

---

## Results (measured)

| | Emergency recall | Macro F1 |
|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.797 | 0.824 |
| **ClinicalBERT (ours)**, 706 sentences, 5-fold CV grouped by sentence | **0.877** | **0.858** |

- Whisper small word error rate: 9.9% over 6,661 recordings.
- Team's own voices (51 clips, 3 speakers): **11/12 emergency calls → P1**, **0/30 everyday
  requests → P1**, median 1.8 s per call.
- Voice-tone distress detection: five methods tested, none passed for all speakers → disabled;
  distress is detected from the words (Rule R6). Details in the project report.

All metrics tables are in `results/`.

---

## Repository layout

```
app/streamlit_app.py   the demo (imports the same pipeline that was evaluated)
src/preprocess.py      the one shared audio function: decode, mono, 16 kHz, −23 LUFS, trim
src/pipeline.py        CareVoicePipeline: audio → transcript → class → rules → decision
src/rules.py           all thresholds + the rule set (R0, R4–R9) + reasons
src/mapping.py         intent → class mapping, keyword rules (medication, comfort, toilet, urgent words)
src/audit.py           append-only audit log (SQLite)
scripts/               model download, demo cache, own-voice check, noise curve
jobs/, notebooks/      Kaggle training and evaluation
tests/                 pytest suite
results/               every measured table (CSV/JSON)
CLAUDE.md              full design notes and the reasons behind each decision
```

---

## Limitations

- Not clinically validated: priority levels are the team's design and need nurse review.
- Distress in the tone of voice is not detected (measured negative result).
- The training data has only symptoms; medication/comfort/toilet requests rely on keyword rules.
- English only. Denoising (DeepFilterNet3) is specified but disabled (no Python 3.12 build).

## Data and models

[Medical Speech, Transcription and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent) ·
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) ·
[ESC-50](https://github.com/karolpiczak/ESC-50) ·
[VIVAE](https://zenodo.org/records/4066235) ·
[Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) ·
[Whisper small](https://huggingface.co/Systran/faster-whisper-small) ·
[WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus) ·
[emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
