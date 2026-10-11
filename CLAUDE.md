# CLAUDE.md — Care-Voice AI

Read this fully before writing any code. Several decisions here look wrong at first glance
and are deliberate. The reasons are given. Do not "fix" them.

---

## 1. What this project is

**Care-Voice AI: Intelligent Voice-Based Patient Request Classification and Prioritisation.**
Final-year academic project, Amrita Vishwa Vidyapeetham, Dept. of CSE (AIE).
Team: A. Bhavishya, K. Sumayah, C. Vinuthna.

A hospital nurse-call button carries one bit of information: someone in bed 14 wants
something. It does not say what they need or how urgently. Calls are therefore answered in
arrival order rather than clinical order.

This system lets the patient speak instead. It transcribes the request, classifies what
they are asking for, independently assesses how distressed they sound, and produces a
prioritised alert for the nurse station.

Deliverables: a working pipeline, an evaluation with real numbers, a Streamlit demo, and a
paper.

---

## 2. Non-negotiable design decisions

These are settled. If the code you are about to write contradicts one of these, stop and
re-read this section.

### 2.1 The fusion layer is a RULE, not a trained model

The methodology diagram shows a box labelled "Late-Fusion Priority Model". **Do not build a
trained fusion model.** It is a deterministic rule.

Reason: training a fusion head requires audio clips labelled with BOTH a request type AND
an urgency level on the same clip. No such dataset exists, publicly or commercially. We
searched thoroughly. The two branches are therefore trained on separate corpora and
combined by an explicit rule.

This is presented as a design choice, not a compromise. In a clinical system, an auditable
escalation rule is preferable to learned weights nobody can inspect.

Concretely: `rules.py` contains the combination logic. There is no `fusion_model.pt`.

### 2.2 WavLM is FROZEN

Do not fine-tune the WavLM encoder. Extract features once, mean-pool over time, train a
small MLP head on the pooled vectors.

Reason: CREMA-D has 7,442 clips from 91 actors. Fine-tuning 94.7M parameters on that
overfits to those specific actors and those specific microphones, and the whole point is
that the head must transfer to a different corpus (the medical clips). A frozen encoder
plus a small head transfers far better.

### 2.3 The two branches train on different datasets, and never share a loss

- Text branch trains on Medical Speech Intent (has request-type labels, no emotion).
- Acoustic branch trains on CREMA-D (has emotion labels, no request types).
- Distress gate trains on VIVAE (has pain/fear, no words at all).

They never appear in the same training loop. There is no label schema to reconcile. They
meet only at inference, as calibrated scores.

### 2.4 Emergency recall is the headline metric, not accuracy

A missed emergency and a misrouted water request are not the same error. Every evaluation
must report per-class recall with Emergency called out separately, plus its raw
false-negative count. Overall accuracy may be reported but never as the primary number.

### 2.5 One preprocessing function, used everywhere

All audio from all four corpora passes through the same function in `src/preprocess.py`.
Never do preprocessing inline in a notebook or in the app. If the corpora drift apart in
sample rate or loudness, the cross-corpus design silently breaks and the failure is very
hard to find.

### 2.6 Anger is not filtered out, and nothing suppresses escalation

Only the distress gate escalates (arousal/fear was removed as an escalation path in v2:
14.5% fear recall at a 2% false-alarm budget). The gate cannot separate anger from
distress: out of fold, anger averages 0.417 against pain 0.465 and fear 0.480.

Suppressing anger from the text was tried and removed. The pretrained emotion model
scores "I feel cold" as anger 0.97 but a plainly angry sentence ("This is ridiculous, I
have been pressing this button for twenty minutes!") only 0.67, and 27 of 706 genuine
medical requests score anger above 0.5 (`results/text_emotion_on_phrases.csv`). A
suppressor built on it would have removed real escalations. The emotion output is shown
on the dashboard and never affects priority.

Accepted consequence: an angry patient whose voice fires the distress gate is escalated.
That is a false alarm within the measured budget, not a missed emergency. Do not add a
suppressor back without labelled angry-but-well calls to measure it on.

### 2.7 The demo imports the evaluated code

`app/streamlit_app.py` must import `CareVoicePipeline` from `src/pipeline.py`. The
end-to-end evaluation notebook must import the same class. Never write a second
implementation of the pipeline for the app.

---

## 3. The pipeline

```
Bedside unit (button press / wake word)
        ↓
VAD + audio capture — 16 kHz mono, 16-bit PCM
        ↓
Preprocessing: DeepFilterNet denoise → AGC → silence trim
        ↓
        ├─────────────────────────────┬────────────────────────────┐
        ↓                             ↓                            │
ASR: Whisper small            Acoustic: WavLM (frozen)             │
        ↓                             ↓                            │
   Transcript              Arousal head + Distress gate            │
        ↓                             ↓                            │
ClinicalBERT → Request Type           │                            │
Emergency / Pain / Assistance /       │                            │
General  → base priority              │                            │
        └─────────────┬───────────────┘                            │
                      ↓                                            │
        Escalation RULE (not a model) → priority                   │
                      ↓                                            │
        Hard safety overrides (lookup table)                       │
                      ↓                                            │
   Routing: priority queue → dashboard, escalation timer, audit log
```

### Component table

| Stage | Component | Params | Trained here? |
|---|---|---|---|
| Wake word | openWakeWord (optional; button is fine for demo) | small | No |
| VAD | Silero VAD | 1.8 MB | No |
| Denoise | DeepFilterNet3 | ~2 M | No |
| AGC | pyloudnorm, target −23 LUFS | – | No |
| ASR | Whisper small (`openai/whisper-small`), faster-whisper int8 at runtime | 244 M | No |
| Text classifier | `emilyalsentzer/Bio_ClinicalBERT` + 4-class head | 110 M | **Yes** |
| Acoustic encoder | `microsoft/wavlm-base-plus` | 94.7 M | No, FROZEN |
| Arousal head | MLP 768→256→out | ~200 K | **Yes** |
| Distress gate | MLP / logistic regression on 768-d | ~200 K | **Yes** |
| Calibration | Platt scaling (sklearn) | – | Yes, trivial |
| Fusion | Rule | – | **No — not a model** |
| Overrides | Lookup table | – | No |
| Routing | `heapq` + SQLite | – | No |
| UI | Streamlit | – | No |

Three trained components. Two of them are small heads on frozen features.

---

## 4. Datasets

All four are free and publicly downloadable. None requires a form.

| Purpose | Dataset | Size | Source |
|---|---|---|---|
| Request type | Medical Speech, Transcription and Intent | 6,661 clips, 8.5 h, 124 speakers, 25 intents | Kaggle `paultimothymooney/medical-speech-transcription-and-intent` |
| Arousal | CREMA-D | 7,442 clips, 91 actors, 6 emotions, 4 intensity levels | github.com/CheyneyComputerScience/CREMA-D |
| Distress | VIVAE | 1,085 clips, 11 speakers, pain + fear at 4 intensities | zenodo.org/records/4066235 |
| Noise | MUSAN | 109 h | openslr.org/17 |

Optional extras, only if time allows: ESC-50 (lighter noise alternative), PriMock57 and
Svarah (ASR evaluation), MEAD (more graded fear, but it is an audio-visual dataset and the
download is video-heavy — check for an audio-only package before committing).

### Known facts about the medical dataset (already verified, do not re-derive)

- Metadata CSV: `overview-of-recordings.csv`, 6,661 rows, 13 columns.
- Label column is `prompt`, 25 unique values.
- Transcript column is `phrase`.
- **Only 706 unique phrases exist.** Each sentence was read by roughly nine different
  speakers. For the TEXT branch, deduplicate to unique phrases — training on 6,661 rows
  just reweights by recording count and inflates results.
- **Every split must be grouped by `phrase`.** A random split puts the same sentence in
  train and test and the classifier memorises it.
- `speaker_id` exists (124 unique). Use it for speaker-independent splits on the ACOUSTIC
  branch.
- Labels are clean: 705 of 706 phrases carry exactly one prompt.
- Audio quality is good: nothing below 3.0/5, 54 clips with light clipping, 15 quiet
  speakers. About half carry light background noise.
- The provided train/test/validate folders are split 381 / 5,895 / 385. Ignore them and
  make your own splits.

---

## 5. The label mapping

This table is the project's main contribution. Do not alter it without being asked. It was
derived by error analysis, not by intuition — two earlier assignments ("Internal pain" and
"Blurry vision" as Emergency) were measurably wrong and have been corrected.

```python
# src/mapping.py

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
```

Notes:

- **"Heart hurts" is the correct spelling**, not "Head hurts". It is a cardiac complaint
  and maps to Emergency P1.
- `"Internal pain"` is in Pain, not Emergency. The phrases under that label are
  unlocalised aches ("I have a pain in my stomach", "i have pain but i can't specify
  place"), not internal bleeding. It was in Emergency initially and was missed 65% of the
  time, correctly.
- `"Blurry vision"` is in General. Its phrases are optometry complaints ("my television
  seems out of focus even though I have new glasses"), not stroke.
- **There is no Medication class.** No intent in the dataset is a medication request. The
  class exists in the system design but has no training data. Do not invent one, do not
  synthesise one unless explicitly asked. The system trains on four classes.
- Class balance after mapping (unique phrases): Pain 344, General 172, Emergency 138,
  Assistance 52. Assistance is the weak class. Use class-weighted loss. Report it honestly
  rather than merging it away.

---

## 6. Priority scale and the escalation rule

```
P1  immediate, bypass the queue
P2  urgent, within a few minutes
P3  standard
P4  routine
```

Lower number is more urgent.

```python
# src/rules.py

def apply_rules(base_priority, request_class, class_confidence, distress,
                asr_confidence=None, energy_ok=True, non_speech_prob=None,
                energy_reason="", cfg=CONFIG, *, transcript, asr_heard=None) -> Decision:
    """
    Takes the MAXIMUM urgency of the signals. Never an average.
    Returns a Decision: final_priority, reason, rules_fired (R0..R4), and the
    speech-guard inputs, so the dashboard and audit log can explain it.
    """
```

Rules, applied in order (thresholds: `DISTRESS_HIGH` 0.4028, `NON_SPEECH_HIGH` 0.90,
`ASR_MIN` -0.4934; provenance in `rules.py`):

0. **No speech** → P2 (`P_NO_SPEECH`), routed like a plain call button, and nothing else
   runs. Either the energy pre-filter found digital silence, or all three hold:
   `non_speech >= NON_SPEECH_HIGH`, Whisper returned nothing or a transcript below
   `ASR_MIN`, and `distress < DISTRESS_HIGH`. Never decided by `non_speech` alone (it
   dropped 7/32 intelligible TTS requests at 0.5), and never when distress fires (15/20
   VIVAE groans score as non-speech).
1. If `asr_confidence < ASR_MIN` and `distress >= DISTRESS_HIGH`, floor at P1. The
   transcript cannot be trusted and the acoustic path is the only signal left: the
   wordless-groan and the hallucination case.
2. Else if `distress >= DISTRESS_HIGH`, floor at P1.
3. *(Removed in v2.)* Arousal/fear no longer escalates: 14.5% fear recall at a 2%
   false-alarm budget on medical speech. Fear is displayed only. There is no
   `AROUSAL_HIGH` and no `DISTRESS_MED`; the gate is binary.
4. If the predicted class is Emergency, never assign worse than P2 regardless of how calm
   the voice is.
5. Nothing suppresses the distress gate, including anger (see §2.6).

An empty transcript is always `"Unclassified"` (a pipeline sentinel, not a class) at
P2, and `"Unclassified"` is only ever used for an empty transcript; `apply_rules` raises
otherwise.

Thresholds live in a config dict at the top of `rules.py`, not scattered through the code.
Every escalation must return a human-readable `reason` string; the dashboard displays it.

**Calibration.** The plan was a Platt scaler per head. In practice the branches never
meet on a common scale: only the distress gate reaches the rule, as a binary decision at
a threshold chosen on its own held-out operating curve
(`results/operating_curve_distress_v2.csv`). The shipped heads are uncalibrated
probabilities; if a second acoustic score ever feeds the rule again, calibrate first.

---

## 7. Repo layout

```
care-voice-ai/
  CLAUDE.md
  requirements.txt
  src/
    preprocess.py     # the single shared audio function
    mapping.py        # PROMPT_TO_CLASS, class lists
    rules.py          # thresholds, apply_rules, safety overrides
    features.py       # WavLM feature extraction + caching
    pipeline.py       # CareVoicePipeline — the one entry point
  notebooks/
    01_explore.ipynb
    02_transcribe.ipynb
    03_text_branch.ipynb
    04_features.ipynb
    05_arousal_head.ipynb
    06_distress_gate.ipynb
    07_noise_robustness.ipynb
    08_end_to_end.ipynb
  models/             # all checkpoints, gitignored
  results/            # every metrics table as CSV, committed
  app/
    streamlit_app.py
    demo_clips/
    cached_results.json
  data/
    manifest.csv      # unified index across all corpora
```

### The manifest

One CSV indexing every clip from every corpus. Blank cells where a corpus has no such
label. Loaders filter by the `task` column.

```
path, corpus, speaker_id, phrase, transcript, prompt, req_class,
base_priority, emotion, intensity, task, split
```

---

## 8. Build order

Each step must write a file. Kaggle sessions die after 12 hours and time out after 20
minutes idle; nothing should ever need recomputing.

**Step 1 — `src/preprocess.py`**

```python
def load_audio(path, target_sr=16000, target_lufs=-23.0, max_seconds=None):
    """Load → mono → resample to 16 kHz → loudness normalise → optional pad/crop.
    Returns float32 numpy array. Every corpus goes through this. No exceptions."""
```

Plus `augment(wav, noise_bank, snr_db)` for MUSAN mixing and MP3 re-encoding.

**Step 2 — manifest builder.** One script per corpus appending rows to `data/manifest.csv`.

**Step 3 — notebook 02.** Whisper over all 6,661 medical clips. Save
`results/transcripts.csv`. Upload as a private Kaggle Dataset. Never run Whisper in bulk
again.

**Step 4 — notebook 03.** Text branch. Grouped-by-phrase 5-fold CV. Save
`models/clinicalbert/` and `results/text_metrics.csv`. Report both the TF-IDF baseline and
ClinicalBERT.

**Step 5 — notebook 04.** WavLM features for CREMA-D, VIVAE and the medical clips. Then
repeat with MUSAN noise at 0, 5, 10, 20 dB. Five feature sets, each saved as `.npy`. This
is the most important caching step — after it, every remaining experiment is CPU-only and
runs in seconds.

**Step 6 — notebook 05.** Arousal head on CREMA-D features, ordinal target from the
intensity levels, speaker-independent splits. Save head + calibrator.

**Then immediately run the domain-gap check**, in the same notebook:

- Train a throwaway classifier to predict which corpus an embedding came from. If it hits
  99%, the corpora are trivially separable and the head will not transfer.
- Run the trained arousal head over several hundred medical clips and plot the score
  distribution. If the scores collapse to a single value, the domain gap has broken it.
- Fix by increasing augmentation (more MUSAN, MP3 re-encoding, varied SNR). Re-check.

This check gates everything after it. If it cannot be made to pass, stop and report it
rather than building on top of a dead branch.

**Step 7 — notebook 06.** Distress gate. Positives: VIVAE pain and fear, plus CREMA-D fear.
Negatives: CREMA-D neutral, MUSAN, ESC-50. Save head + calibrator.

**Step 8 — notebook 07.** Noise robustness. Load the five cached feature sets, evaluate
both branches on each, plot degradation against SNR. **This is the most valuable figure in
the project.** Save it to `results/`.

**Step 9 — `src/pipeline.py`.** Wire it together.

```python
class CareVoicePipeline:
    def __init__(self, model_dir="models", device="cpu"): ...

    def process(self, audio_path) -> dict:
        """Returns every intermediate value, not just the final priority:
        transcript, asr_confidence, request_class, class_probs, base_priority,
        arousal, distress, final_priority, escalation_reason, latency_ms
        """
```

Returning the intermediates is what lets the demo show *why* a call was escalated, which
is far more convincing than showing the number alone.

**Step 10 — notebook 08.** End-to-end evaluation, importing `CareVoicePipeline`. Final
tables.

**Step 11 — the app.** See below.

---

## 9. The demo

Streamlit, three tabs.

**Tab 1 — Nurse Station.** The main view. A queue of calls sorted by priority showing bed
number, time waiting, request class, transcript, a coloured priority badge, and an
escalation flag with its reason when the acoustic branch fired. Auto-refresh so waiting
times tick up. An escalation timer bumps priority when a call has waited past a threshold,
so low-priority requests do not starve.

**Tab 2 — Submit a Call.** Upload a file or record from the mic, pick a bed number, submit.
Displays the full breakdown from `process()`: transcript, ASR confidence, class
probabilities, arousal, distress, base priority, final priority, escalation reason,
latency.

**Tab 3 — Results.** The metrics tables and the noise degradation plot, loaded from
`results/`. Lets an evaluator's question be answered without leaving the app.

### Demo clips

Prepare about eight clips in `app/demo_clips/`, named so the running order is obvious.
Three moments matter:

1. **A pain phrase spoken calmly** → classified Pain, P3. Then **the same phrase spoken
   distressed** → escalated to P2. This shows the acoustic branch doing something the text
   cannot, with the transcript held constant.
2. **A wordless groan from VIVAE** → no usable transcript, but the distress gate fires and
   the override pushes it to P1. The most memorable moment in the demo; make sure it works.
3. **A calm "I have chest pain"** → shows the Emergency class catching something the voice
   does not betray.

### Demo reliability

Assume no internet in the room.

- Pre-download every model into `models/` and load from local paths. Set `HF_HUB_OFFLINE=1`
  when running the demo so a failed network call cannot hang the app.
- Use **faster-whisper int8** rather than the HF Whisper pipeline. Roughly 4x faster on
  CPU, which is the difference between a 1-second and a 5-second pause on stage.
- Pre-compute `process()` output for every demo clip into `app/cached_results.json`, and
  add a "cached mode" toggle. If a model fails live, flip it and the demo still runs.
- Prefer file upload over live microphone as the primary path. Mic permissions and audio
  drivers fail more often than anything else in the system.
- Record a screen capture of a successful full run as a last-resort backup.

### Memory on the 8 GB demo machine

Measured: a speech call is ~1.5 s when the weights are resident (Whisper ~1.3 s of it)
and ~10 s after 60 s idle, because macOS swaps the PyTorch weights out and pages ~1 GB
back in. That is the whole 9.4 s seen in the app; it is not compute.

- Quit other apps before presenting (the machine sat 9.7 GB into a 10 GB swap with the
  usual apps open). This is the biggest lever.
- The text-emotion model (313 MB, display-only) is a sidebar toggle, off by default and
  not loaded unless switched on.
- "Keep models warm" (sidebar, off by default) runs a dummy inference every 30 s. It only
  helps once there is free RAM; under pressure it evicts the browser instead.
- Available, not applied: int8 dynamic quantisation of ClinicalBERT saves 244 MB (413 →
  169 MB) and halves its latency (38 → 19 ms); 704/706 phrases keep the same prediction.
  It changes the evaluated model (§2.7), so re-run the 5-fold grouped text metrics on the
  int8 model first, and only switch the app over if Emergency recall holds.
  (On Apple Silicon set `torch.backends.quantized.engine = "qnnpack"`.)

---

## 10. Results measured so far

Text branch, 706 unique phrases, 5-fold CV grouped by phrase, four classes:

| Model | Macro F1 | Accuracy | Emergency recall |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.824 | 0.841 | 0.797 |
| ClinicalBERT | 0.855 | 0.884 | 0.826 |

Whisper small WER on this corpus: **10.4%**. 106 of 200 sampled clips transcribed perfectly.

Emergency misses by prompt, ClinicalBERT: Heart hurts 26%, Hard to breath 24%, Feeling
dizzy 19%, Infected wound 15%, Open wound 0%.

**This failure pattern is the project's central finding.** The two most clinically dangerous
categories are the hardest for text alone, because cardiac and respiratory complaints are
described entirely in pain vocabulary ("severe pain in the upper left side of chest").
The acoustic branch exists to recover exactly this measured gap. Keep this framing
throughout the code comments and the paper.

Also observed and worth handling: **Whisper hallucinates**. On quiet or noisy clips it
produces fluent, confident, unrelated sentences — "I struggle to breathe" became "I feel a
sharp pain in my stomach". This is a safety failure, not an accuracy failure, and is why
`apply_rules` takes `asr_confidence` as an input.

---

## 11. Things not to do

- Do not build a trained fusion model.
- Do not fine-tune WavLM.
- Do not pool the datasets into a single training set.
- Do not split the medical data randomly — group by phrase for the text branch, by speaker
  for the acoustic branch.
- Do not train the text branch on all 6,661 rows; deduplicate to 706 phrases first.
- Do not report overall accuracy as the headline number.
- Do not claim, anywhere in code comments or output, that fusion beats text alone. That
  comparison requires joint labels we do not have and cannot be measured.
- Do not invent a Medication class or synthesise data for it without being asked.
- Do not add anger suppression back without labelled angry-but-well calls to measure it
  on (§2.6). Text emotion is display-only.
- Do not write a second implementation of the pipeline for the app.
- Do not put preprocessing logic anywhere except `src/preprocess.py`.

---

## 12. Conventions

- Python 3.10+, PyTorch, transformers, librosa, torchaudio, scikit-learn, pandas, Streamlit.
- Every notebook writes its outputs to `results/` or `models/` before it ends.
- `results/` is committed. `models/` and `data/` are gitignored.
- Thresholds and paths in config dicts at the top of a module, never inline.
- Set and log a random seed everywhere.
- Every metrics table saved as CSV, so the paper can be rebuilt without rerunning anything.
- Log latency for every pipeline stage. Responsiveness is the point of the system and the
  number belongs in the paper.

---

## 13. Running jobs on Kaggle

The local machine has no GPU. All training runs on Kaggle through the API.
Kaggle username: bhavishyaadhikari

Each job is a folder under `jobs/`:

    jobs/03_text_branch/
      kernel-metadata.json
      script.py

kernel-metadata.json:

    {
      "id": "bhavishyaadhikari/care-voice-text-branch",
      "title": "care-voice-text-branch",
      "code_file": "script.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": true,
      "enable_gpu": true,
      "enable_internet": true,
      "dataset_sources": ["paultimothymooney/medical-speech-transcription-and-intent"]
    }

Run with:

    kaggle kernels push -p jobs/03_text_branch
    kaggle kernels status bhavishyaadhikari/care-voice-text-branch
    kaggle kernels output bhavishyaadhikari/care-voice-text-branch -p results/

Rules:
- Scripts write everything to /kaggle/working/ so it comes back in the output.
- Debug logic locally on a 50-row sample before pushing. The push-run-pull cycle costs
  minutes and the GPU quota is 30 hours a week shared by three people.
- Chain jobs by output. Upload a job's results as a private Kaggle Dataset with
  `kaggle datasets create`, then list it in the next job's dataset_sources. Never
  recompute Whisper transcripts or WavLM features.
- Always read the metrics in the output, not just the exit status.
- The medical dataset mounts at:
  /kaggle/input/medical-speech-transcription-and-intent/Medical Speech, Transcription, and Intent/
  containing overview-of-recordings.csv and recordings/{train,test,validate}/
