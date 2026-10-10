# Patch: minimum-speech guard

Three file edits plus one calibration script. Apply in order.

> **Correction (after applying this patch).** The root-cause paragraph below is
> wrong about the mechanism, and is kept only as the original decision record.
> The 5 ms clip was pure digital zeros. `normalize_loudness` returns a signal
> with −inf loudness unchanged, so **no gain was applied**. The pipeline then
> zero-pads to the 4 s acoustic window, so the distress gate scored a WavLM
> embedding of four seconds of silence, at 0.9996.
>
> The actual cause is an out-of-distribution input. The gate's negatives were
> CREMA-D neutral speech and other VIVAE vocalisations. Nothing in its training
> set was an absence of signal. A logistic regression on 768-d features
> extrapolates confidently outside its support, in an arbitrary direction.
> ClinicalBERT on the resulting empty transcript has the same problem: it
> predicted Emergency. The guard below is still the right fix, and the text
> branch is now skipped as well (request class `"Unclassified"`).
>
> A further issue follows for notebook 09. Notebook 04 trained on the same 4 s
> window, but `degrade()` added Gaussian noise over the whole window after
> padding. So the heads saw noise-filled padding during training, never digital
> zeros. Short clips at inference carry zero-padding the heads never saw.

The problem (original text, mechanism superseded by the correction above): a 5
ms near-silent clip scored 0.9996 on the distress gate and was routed to P1.
Root cause is two things stacking — `load_audio` normalises to −23 LUFS, which
applies huge gain to a near-silent signal, and the resulting amplified noise
floor lands somewhere the distress gate was never trained to see. Then
`apply_rules` Rule 1 fires (no transcript → untrusted ASR, distress fired → P1).

The fix routes a no-speech call at a fixed unclassified priority and suppresses
both acoustic scores. Do not fix it by filtering on the empty transcript: a
genuine wordless groan also has no transcript, and that case must still reach
P1.

---

## Edit 1 — `src/preprocess.py`

The speech check must read the signal **before** loudness normalisation.

### 1a. In `load_audio`, immediately before the loudness-normalisation step, add:

```python
    # Measured before normalisation. After −23 LUFS the gain applied to a
    # near-silent clip is enormous, so post-normalisation level carries no
    # information about whether anyone actually spoke.
    _raw_rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    _raw_peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    _raw_dur = len(wav) / CONFIG["sample_rate"]
```

### 1b. Change the signature and the return

```python
def load_audio(path, return_meta: bool = False):
    ...
    # existing final return becomes:
    if return_meta:
        return wav, {"raw_rms": _raw_rms, "raw_peak": _raw_peak,
                     "raw_duration_s": _raw_dur}
    return wav
```

Default `False`, so every existing caller — the notebooks, the feature
extraction — is unaffected. Only `pipeline.py` passes `True`.

### 1c. Append to the end of `src/preprocess.py`

```python
# ---------------------------------------------------------------------------
# Minimum-speech guard
#
# Thresholds below are placeholders. Run scripts/calibrate_speech_guard.py and
# replace them with the measured values before using this in any result.
# ---------------------------------------------------------------------------

SPEECH_GUARD = {
    # Pre-normalisation RMS floor. Set below the minimum observed across all
    # four corpora, with margin.
    "MIN_RAW_RMS": 2.0e-3,
    # Seconds of audio surviving trim(). A VIVAE groan is ~1 s; a button press
    # with no speech leaves almost nothing.
    "MIN_TRIMMED_S": 0.35,
}


def speech_present(meta: dict, trimmed_samples: int, sr: int = 16000,
                   cfg: dict = SPEECH_GUARD) -> tuple[bool, str]:
    """
    Is there enough signal here for either branch to mean anything?

    Returns (present, reason). `meta` is the dict from load_audio(return_meta=True),
    `trimmed_samples` is len(wav) after trim().

    Checks energy and duration, never the transcript — a wordless cry has no
    transcript and must still be treated as a real call.
    """
    trimmed_s = trimmed_samples / sr

    if meta["raw_rms"] < cfg["MIN_RAW_RMS"]:
        return False, (f"signal level below floor "
                       f"(RMS {meta['raw_rms']:.2e} < {cfg['MIN_RAW_RMS']:.2e})")
    if trimmed_s < cfg["MIN_TRIMMED_S"]:
        return False, (f"too little audio after silence trimming "
                       f"({trimmed_s:.2f}s < {cfg['MIN_TRIMMED_S']}s)")
    return True, ""
```

---

## Edit 2 — `src/rules.py`

### 2a. Add to `CONFIG`

```python
    # A call with no usable speech carries no triage information at all. It is
    # routed at this fixed level: the behaviour of a conventional nurse-call
    # button, which signals a press and nothing more. Neither suppressed to the
    # bottom of the queue nor escalated on an acoustic score computed from a
    # noise floor.
    "P_NO_SPEECH": 2,
```

### 2b. Add two fields to `Decision`

```python
    speech_present: bool = True
    acoustic_used: bool = True
```

(Both default `True`, so existing construction sites keep working.)

### 2c. Add the parameter to `apply_rules`

```python
def apply_rules(
    base_priority: int,
    request_class: str,
    class_confidence: float,
    arousal: float,
    distress: float,
    asr_confidence: Optional[float] = None,
    speech_present: bool = True,
    speech_reason: str = "",
    cfg: dict = CONFIG,
) -> Decision:
```

### 2d. Insert as the first branch, before the `asr_trusted` line

```python
    # --- Rule 0: no usable speech -------------------------------------------
    # Short-circuits everything. Both branches are reading noise: the text
    # branch has no transcript and the acoustic heads were never trained on a
    # normalised noise floor, where the distress gate scores near 1.0.
    #
    # Routed at P_NO_SPEECH and returned immediately, so Rule 1 (untrusted
    # transcript + distress -> P1) cannot fire on an empty clip. A wordless
    # groan still reaches Rule 1, because it passes this guard on energy.
    if not speech_present:
        return Decision(
            final_priority=cfg["P_NO_SPEECH"],
            base_priority=base_priority,
            reason=("no speech detected" + (f" ({speech_reason})" if speech_reason else "")
                    + "; routed as an unclassified call, equivalent to a "
                      "conventional call button"),
            escalated=False,
            distress_fired=False,
            arousal_fired=False,
            asr_trusted=False,
            speech_present=False,
            acoustic_used=False,
        )
```

### 2e. Pass the two new fields in the normal return at the end

```python
        speech_present=True,
        acoustic_used=True,
```

### 2f. Add a sixth case to the `__main__` block

```python
        ("button press, no speech",
         dict(base_priority=4, request_class="General", class_confidence=0.0,
              arousal=0.03, distress=0.9996, asr_confidence=-5.0,
              speech_present=False,
              speech_reason="signal level below floor (RMS 1.1e-05 < 2.0e-03)")),
```

Expected: `P4 -> P2`, not escalated, reason naming the guard.

---

## Edit 3 — `src/pipeline.py`

### 3a. Import

```python
from preprocess import load_audio, denoise, trim, speech_present
```

### 3b. Two fields on `Result`

```python
    speech_present: bool = True
    speech_reason: str = ""
```

### 3c. In `process()`, replace the preprocessing and acoustic sections

```python
        wav, meta = load_audio(audio_path, return_meta=True)
        tick("load")
        if self.use_denoise:
            wav = denoise(wav)
            tick("denoise")
        wav = trim(wav)
        tick("trim")

        has_speech, speech_reason = speech_present(meta, len(wav))

        transcript, asr_conf = self._transcribe(wav)
        tick("asr")

        req_class, probs, class_conf = self._classify_text(transcript)
        tick("text_branch")

        # Skipped entirely when there is no speech. Running the heads on a
        # normalised noise floor produces a confident score with no meaning,
        # and the dashboard should show 0.0 rather than that number.
        if has_speech:
            arousal, distress = self._acoustic(wav)
        else:
            arousal, distress = 0.0, 0.0
        tick("acoustic_branch")
```

Then add `speech_present=has_speech, speech_reason=speech_reason` to the
`apply_rules` call and to the `Result` constructor.

---

## Script — `scripts/calibrate_speech_guard.py`

The two thresholds must be measured, like every other number in `rules.py`.
Run this on Kaggle where the audio lives, and put the output in
`results/speech_guard.json`.

```python
"""
Fit the minimum-speech guard thresholds.

The guard must never reject a real call, so both thresholds sit below the
minimum observed across all four corpora, with margin. Reports the margin so
the choice is visible rather than asserted.
"""
import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from src.preprocess import load_audio, trim

SR = 16000


def metrics(path):
    _, meta = load_audio(path, return_meta=True)
    wav = trim(load_audio(path))
    return meta["raw_rms"], len(wav) / SR


def main(manifest_csv, out_json):
    df = pd.read_csv(manifest_csv)
    rows = []
    for r in df.itertuples():
        try:
            rms, dur = metrics(r.path)
            rows.append({"corpus": r.corpus, "raw_rms": rms, "trimmed_s": dur})
        except Exception as e:
            rows.append({"corpus": r.corpus, "raw_rms": np.nan,
                         "trimmed_s": np.nan, "error": str(e)})
    m = pd.DataFrame(rows).dropna(subset=["raw_rms"])

    # Negative controls: what the guard has to reject.
    neg = {}
    for name, sig in {
        "digital_silence": np.zeros(int(0.05 * SR), dtype=np.float32),
        "5ms_silence":     np.zeros(int(0.005 * SR), dtype=np.float32),
        "noise_floor_60db": (np.random.default_rng(0)
                             .normal(0, 10 ** (-60 / 20), int(2.0 * SR))
                             .astype(np.float32)),
        "noise_floor_50db": (np.random.default_rng(1)
                             .normal(0, 10 ** (-50 / 20), int(2.0 * SR))
                             .astype(np.float32)),
    }.items():
        p = f"/tmp/_neg_{name}.wav"
        sf.write(p, sig, SR)
        rms, dur = metrics(p)
        neg[name] = {"raw_rms": rms, "trimmed_s": dur}

    per_corpus = (m.groupby("corpus")
                   .agg(n=("raw_rms", "size"),
                        rms_min=("raw_rms", "min"),
                        rms_p01=("raw_rms", lambda s: float(s.quantile(0.01))),
                        rms_median=("raw_rms", "median"),
                        dur_min=("trimmed_s", "min"),
                        dur_p01=("trimmed_s", lambda s: float(s.quantile(0.01))))
                   .reset_index())

    rms_floor_real = float(m.raw_rms.min())
    dur_floor_real = float(m.trimmed_s.min())
    rms_ceil_neg = max(v["raw_rms"] for v in neg.values())
    dur_ceil_neg = max(v["trimmed_s"] for v in neg.values())

    out = {
        "per_corpus": per_corpus.to_dict("records"),
        "negative_controls": neg,
        "real_min_raw_rms": rms_floor_real,
        "real_min_trimmed_s": dur_floor_real,
        "negative_max_raw_rms": rms_ceil_neg,
        "negative_max_trimmed_s": dur_ceil_neg,
        "separable": rms_ceil_neg < rms_floor_real,
        # Geometric midpoint when the two sets separate; otherwise fall back to
        # just under the real minimum and record that the margin is thin.
        "recommended_MIN_RAW_RMS": (float(np.sqrt(rms_ceil_neg * rms_floor_real))
                                    if rms_ceil_neg < rms_floor_real
                                    else rms_floor_real * 0.5),
        "recommended_MIN_TRIMMED_S": round(min(0.35, dur_floor_real * 0.8), 3),
    }
    json.dump(out, open(out_json, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_corpus"}, indent=2))
    print(per_corpus.to_string(index=False))


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2
         else "results/speech_guard.json")
```

What to read in the output:

- `separable: true` — the guard has clean air between real calls and silence.
  Use `recommended_MIN_RAW_RMS`.
- `separable: false` — some real clip is quieter than a 50 dB noise floor.
  Find it and look at it; it is probably a corrupt file, and belongs in the
  integrity sweep rather than in the threshold.
- `dur_min` on VIVAE is the number that protects Moment 2. If any pain clip
  trims below 0.35 s, lower `MIN_TRIMMED_S` to suit it.

---

## Verification

```bash
.venv/bin/python -m src.rules          # six cases, new one gives P4 -> P2

python - <<'PY'
import numpy as np, soundfile as sf
sf.write("/tmp/silence.wav", np.zeros(int(0.005*16000), dtype=np.float32), 16000)
PY
.venv/bin/python -m src.pipeline /tmp/silence.wav --bed 14   # P2, no speech detected
.venv/bin/python -m src.pipeline <a_vivae_pain_clip>.wav     # still P1
.venv/bin/python -m src.pipeline chest.aiff --bed 14         # still Emergency P1
```

The VIVAE check is the one that matters. If the guard swallows a real groan it
has broken the thing the system exists to catch.

---

## For the paper

Limitations (rewritten after the correction above):

> A call carrying no usable signal, such as a press with no vocalisation, lies
> outside the training distribution of both branches. The distress gate's
> negatives were neutral speech and non-distress vocalisations; it never saw an
> absence of signal, and it scores a WavLM embedding of silence at 0.9996 — a
> confident extrapolation with no meaning. ClinicalBERT likewise assigns a
> class to an empty transcript. The system therefore detects this input from
> pre-normalisation signal level and post-trim duration before either branch
> runs, skips both branches, and routes the call at a fixed unclassified
> priority (P2): the behaviour of the conventional call button it replaces.
> It does not infer urgency from a signal that contains none.

Viva answer: the distress gate scores silence at 0.9996 because silence is
nowhere near either of its training classes. That is the standard
out-of-distribution failure of any discriminative classifier. The standard
answer is to detect the input you were never trained on rather than trust the
score, and that is what the guard does. (Do not mention loudness gain: it plays
no part, since zeros stay zeros through normalisation.)

The guard thresholds are unmeasured placeholders until
`scripts/calibrate_speech_guard.py` has been run on the corpora.

## Demo clip

Add a ninth clip, `09_button_press_no_speech.wav` — 2 s of room tone, or just
digital silence. Show it landing at P2 with "no speech detected" and an
"Unclassified" grey badge. It is the
clip that answers the obvious examiner question about what happens when the
model has nothing to work with.
