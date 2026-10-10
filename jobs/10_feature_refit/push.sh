#!/usr/bin/env bash
# Push notebooks/10_feature_refit.ipynb as a Kaggle job. The notebook in
# notebooks/ is the source of truth; this copies it in at push time.
# Pull results with:
#   kaggle kernels output bhavishyaadhikari/care-voice-feature-refit -p results/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/../../notebooks/10_feature_refit.ipynb" "$HERE/"
trap 'rm -f "$HERE/10_feature_refit.ipynb"' EXIT
"${KAGGLE:-kaggle}" kernels push -p "$HERE"
