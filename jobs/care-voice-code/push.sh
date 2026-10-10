#!/usr/bin/env bash
# Stage the repo files Kaggle jobs import (src/, the current heads, cached
# baseline features, notebook 04, transcripts) and upload them as the PRIVATE
# dataset bhavishyaadhikari/care-voice-code. Creates it the first time,
# versions it afterwards. Run from the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/src" "$STAGE/models" "$STAGE/notebooks" "$STAGE/results"
cp "$ROOT"/src/*.py "$STAGE/src/"
cp "$ROOT"/models/arousal_head.joblib "$ROOT"/models/distress_gate.joblib \
   "$ROOT"/models/X_vivae_aug.npy "$ROOT"/models/X_med.npy "$STAGE/models/"
cp "$ROOT"/notebooks/04_acoustic_branch.ipynb "$STAGE/notebooks/"
cp "$ROOT"/results/transcripts.csv "$STAGE/results/"
cp "$ROOT"/jobs/care-voice-code/dataset-metadata.json "$STAGE/"

KAGGLE="${KAGGLE:-kaggle}"
if "$KAGGLE" datasets status bhavishyaadhikari/care-voice-code >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$STAGE" -m "sync from $(git -C "$ROOT" rev-parse --short HEAD)" -r zip
else
  "$KAGGLE" datasets create -p "$STAGE" -r zip     # private by default
fi
