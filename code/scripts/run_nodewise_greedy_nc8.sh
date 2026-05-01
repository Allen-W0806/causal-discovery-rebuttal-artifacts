#!/usr/bin/env bash
# Run node-wise greedy baseline on NC8 benchmark.
# Usage: bash scripts/run_nodewise_greedy_nc8.sh [REPLICA=0] [SEED=0]
# Example: bash scripts/run_nodewise_greedy_nc8.sh 0 0
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPLICA="${1:-0}"
SEED="${2:-0}"
OUT="$ROOT/outputs/nodewise_greedy/replica${REPLICA}"
cd "$ROOT"

python scripts/run_nodewise_greedy.py \
  --dataset NC8 \
  --data_dir data/NC8 \
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
  --ts_rank 4 \
  --tau 0.5 \
  --lambda_sparse 8 \
  --score_type mlp \
  --eval 4000 \
  --batch_size 32 \
  --n_cands 10000 \
  --n_grads 10 \
  --hidden_size 64 \
  --dropout 0.1 \
  --tol 1e-9 \
  --seed "$SEED" \
  --meta_update_every 10 \
  --meta_batch_size 8 \
  --meta_recent_window 20 \
  --meta_inner_lr 0.001 \
  --meta_outer_lr 0.0001 \
  --no-plot \
  --outdir "$OUT"
