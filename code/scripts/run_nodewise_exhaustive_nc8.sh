#!/usr/bin/env bash
# Run node-wise exhaustive oracle comparator on NC8 benchmark.
# Evaluates all d x 2^d = 8 x 256 = 2048 parent-mask combinations.
# Usage: bash scripts/run_nodewise_exhaustive_nc8.sh [REPLICA=0] [SEED=0]
# Example: bash scripts/run_nodewise_exhaustive_nc8.sh 0 0
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPLICA="${1:-0}"
SEED="${2:-0}"
OUT="$ROOT/outputs/nodewise_exhaustive/replica${REPLICA}"
cd "$ROOT"

python scripts/run_nodewise_exhaustive_nc8.py \
  --dataset NC8 \
  --data_dir data/NC8 \
  --outdir "$OUT" \
  --replica "$REPLICA" \
  --T 2000 \
  --d 8 \
  --lag 16 \
  --mlp_hidden 64 \
  --mlp_max_iter 50 \
  --pretrain_epochs 0 \
  --inner_step_count 10 \
  --inner_lr 0.01 \
  --inner_batch_size 0 \
  --inner_optimizer adam \
  --lambda_sparse 8 \
  --score_type mlp \
  --seed "$SEED" \
  --meta_update_every 10 \
  --meta_batch_size 8 \
  --meta_recent_window 20 \
  --meta_inner_lr 0.001 \
  --meta_outer_lr 0.0001 \
  --no-plot
