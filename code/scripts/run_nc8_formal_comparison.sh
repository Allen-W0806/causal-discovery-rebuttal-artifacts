#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$ROOT/review_runs/nc8_formal_comparison}"
cd "$ROOT"

bash scripts/run_nc8.sh "$OUT_ROOT/low_rank_bo"
bash scripts/run_nodewise_greedy.sh "$OUT_ROOT/nodewise_greedy"
