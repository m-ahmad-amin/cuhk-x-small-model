#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:?usage: ./inference.sh <data_dir> [submission.csv]}"
OUT="${2:-$ROOT/submission.csv}"
python "$ROOT/scripts/infer.py" --data-root "$DATA_DIR" --out "$OUT"
