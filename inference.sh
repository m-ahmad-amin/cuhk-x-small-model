#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:?usage: ./inference.sh <data_dir> [submission.csv]}"
OUT="${2:-$ROOT/submission.csv}"
python "$ROOT/scripts/infer.py" \
  --data-root "$DATA_DIR" \
  --checkpoint "$ROOT/checkpoints/model_v4_all.pt" \
  --aux-checkpoint "$ROOT/checkpoints/model_v4_s44.pt" \
  --tta-flip \
  --tta-time 2 \
  --tta-crop \
  --out "$OUT"
