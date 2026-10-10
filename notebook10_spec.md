# Notebook 10 — corrected feature extraction and head refit

Specification, not a patch. Written against the behaviour the code should have,
not against line numbers, after the last patch assumed a shape the repo didn't
have. Implement it however the existing code is actually organised.

Runs on Kaggle with GPU. Replaces the feature sets and both acoustic heads, and
re-derives every threshold in `rules.py` that came from them.

---

## Why this exists

Notebook 04's `degrade()` filled the 4-second window's padding with noise.
`X_med.npy`, the false-alarm set, padded with zeros. Live inference pads with
zeros.

VIVAE positives are around 1 s, so a positive example is roughly 1 s of
vocalisation and 3 s of noise. CREMA-D negatives are around 2.5 s, so a
negative is 2.5 s of speech and 1.5 s of noise. Padding amount is correlated
with the label, and the padding carries signal.

Then the false-alarm rate was measured on a set that padded with zeros, which
is a different regime from the one the positives were trained in.

Four observed numbers are all consistent with a gate that has partly learned
the padding rather than the distress:

| Observation | Value |
|---|---|
| Distress gate, 4 s digital silence | 0.9996 |
| Distress gate, 2 s digital silence | 0.43 |
| Distress gate, 1 s digital silence | 3e-7 |
| Arousal head, any noise filling the window | 0.965 |
| Distress gate AUC vs calm speech | 0.996 |

A monotonic relationship between how much of the window is padding and the
score is not what a distress detector should do.

None of this is yet proven to be the cause. Section 1 proves or disproves it
before anything is re-extracted.

---

## Section 1 — the padding diagnostic

Run this first. It decides whether the rest of the notebook is necessary and it
produces a figure for the paper either way.

Hold audio content constant. Vary only the padding.

1. Take 200 VIVAE clips and 200 CREMA-D neutral clips.
2. For each clip, build three 4-second versions:
   - `zeros` — clip then zero padding
   - `noise` — clip then noise padding, exactly as `degrade()` did it
   - `none` — the clip at its own length, mean-pooled over its real frames
3. Extract WavLM features for all of them.
4. Train a logistic regression to predict **padding type** from the embedding,
   with clip identity held out of the split (GroupKFold on clip id). Report AUC.
5. Train a logistic regression to predict **VIVAE vs CREMA-D** separately within
   each padding condition. Report three AUCs.

Read it like this:

- Padding-type AUC near 1.0 → padding content is linearly decodable from the
  embedding, and anything correlated with it is contaminated.
- The VIVAE-vs-CREMA-D AUC under `none` is the honest estimate. The gap between
  it and 0.996 is how much of the published number was padding.
- If the `none` AUC stays high, the gate is real and the padding was a
  confound that happened not to matter much. Say so, keep the fix anyway,
  and report both numbers.

Save to `results/padding_diagnostic.json` and a two-panel figure. This figure
belongs in the paper whichever way it comes out — it is the methodological
check that a reviewer would otherwise ask for.

Also report, for the VIVAE and CREMA-D sets separately: clip duration
distribution, and the Spearman correlation between duration and the current
distress score. A strong correlation is the same finding from the other side.

---

## Section 2 — windowing

Replace fixed-window zero padding everywhere.

- Longer than 4 s: crop. Keep the existing crop rule (from the start) so it
  matches what notebook 04 did for long clips, and record which rule it is.
- Shorter than 4 s: feed at its own length. Do not pad, tile or loop.
- Mean-pool over the clip's actual frames only.

Batch size 1, or bucket by length, whichever is faster on the Kaggle GPU.
Variable length is fine for WavLM at batch size 1; the padding problem comes
back the moment you batch unequal lengths without masking, so if you batch,
mask the pool.

Do not use the feature extractor's `attention_mask` to solve this without
checking `feat_extract_norm` in the WavLM config first. Group-norm models were
pretrained without padding masks and HF warns against passing one. Feeding the
real length sidesteps the question entirely, which is why it is the recommended
route here.

Clips shorter than about 0.5 s give few frames and a noisier embedding. The
speech guard already rejects those, so record the minimum surviving duration
per corpus and confirm nothing in the training set falls below the guard's
`MIN_TRIMMED_S`. If something does, the guard would have rejected a clip the
head was trained on, which is its own inconsistency.

---

## Section 3 — level

Every feature in this notebook goes through `src/preprocess.load_audio`. No
inline normalisation, no RMS 0.05. This is CLAUDE.md §2.5 and the whole reason
the mismatch existed.

Verification cell: re-extract 20 VIVAE clips the way notebook 04 did, compare
against the saved rows of `X_vivae_aug.npy`, report max absolute difference. If
they do not match to float tolerance, the saved features were produced by code
that no longer exists in the repo, and the before/after comparison has no
baseline. Stop and say so rather than reporting a difference that means
nothing.

---

## Section 4 — deduplication

`X_vivae_aug.npy` has 1,565 rows for a 1,085-clip dataset.

Dedup by **file content hash**, not basename. The zip contains overlapping
sets, so the same audio may sit at two paths with two names.

Then check whether any duplicate pair straddled a CV fold boundary in the
original fit. If it did, AUC 0.996 is leakage and not generalisation, and the
corrected number will be lower. Report the original and corrected values side
by side; a drop here is a finding, not a regression.

Keep the augmented copies — those are intentional. The question is whether the
*same source clip* appears on both sides of a split.

---

## Section 5 — negatives

This is the fix for arousal firing at 0.965 on steady noise. A guard rejects
only what it can recognise as absent; the better answer is to make the noise
region part of the training distribution.

**Distress gate negative set, add:**

- ESC-50 clips at several levels, no speech at all. Weight toward the ward
  categories already in the augmentation bank.
- Digital silence at 0.5, 1, 2 and 4 s.
- Gaussian noise at −60, −50 and −40 dBFS, at several durations.
- Any clip the speech guard would reject, so that the guard and the gate agree
  rather than contradicting each other.

**Arousal head:** add a seventh class, `non_speech`, holding the same material.
The head is read through `fear_index`, so a noise clip predicting `non_speech`
drops the fear probability without any change to how the pipeline reads it.
Record the new class order and `fear_index` in the saved joblib, since the index
shifts.

Noise clips get no speaker id. Keep them out of the GroupKFold grouping by
assigning each a unique synthetic group so they never create a fold where one
class is entirely absent.

---

## Section 6 — refit and re-derive

1. Refit the distress gate (logistic regression) and the arousal head (MLP) on
   the corrected features. Same architectures, same CV protocol, same seeds.
2. Refit the Platt calibrators on held-out data, not on the training fold.
3. Re-derive both operating curves. Save `operating_curve_distress_v2.csv` and
   `operating_curve_arousal_v2.csv` alongside the originals; do not overwrite
   them, the comparison is part of the result.
4. Choose `DISTRESS_HIGH` and `AROUSAL_HIGH` again from the new curves at the
   same target operating points (90% distress recall, and the lowest-false-alarm
   point that keeps fear recall near 60%). The numbers will move. Record the old
   and new values with the reason.
5. Re-run the sanity battery and put it in the notebook output, not just in a
   commit message:

| Input | Required behaviour |
|---|---|
| 4 s digital silence | distress below threshold |
| 2 s, 1 s, 0.5 s digital silence | below threshold, and monotonic in nothing |
| 4 s Gaussian noise, −60 to −40 dBFS | distress and fear both below threshold |
| 4 s ESC-50 ward noise | distress and fear both below threshold |
| VIVAE pain clip | distress above threshold |
| CREMA-D fear, HI intensity | fear above threshold |
| medical calm speech | both below threshold, at the measured false-alarm rate |

A refit that fixes silence but breaks the VIVAE case has made things worse.
This table is the gate on the whole notebook.

---

## Section 7 — write back

- `models/arousal_head.joblib` and `models/distress_gate.joblib`, overwritten,
  with the new class order and `fear_index`.
- New `CONFIG` values for `rules.py`, printed as a copyable block with the
  measured provenance comment for each, in the style the file already uses.
- `results/head_refit_v2.json` — every number above.
- A short markdown cell listing what changed and what it changed the answer to.
  This is the viva material; write it while the numbers are in front of you.

Do not edit `rules.py` from the notebook. Print the block and apply it by hand,
so the change shows up as a reviewable diff.

---

## Kaggle inputs

- `src/` (the real one, not a copy — attach the repo or a dataset built from it)
- CREMA-D, VIVAE, ESC-50, the medical corpus
- `X_vivae_aug.npy` and `X_med.npy`, for the section 3 baseline comparison only
- notebook 04, for reading its `degrade()` and window code

---

## What this does not fix

State these in limitations rather than leaving them for a question:

- Acted emotion is not clinical distress. Unchanged by any of this.
- The gate still cannot separate pain from other intense vocalisations
  (AUC 0.651 against other VIVAE categories). Section 5 does not address that;
  it addresses noise and silence.
- Cross-corpus transfer remains a transfer, however the features are extracted.
- The speech guard's thresholds are still unmeasured until
  `calibrate_speech_guard.py` runs on the real corpora.
