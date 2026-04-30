#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/review_runs/nodewise_nc8_rep0_seed0_formal}"
cd "$ROOT"

python scripts/run_nodewise_greedy.py \
  --dataset NC8 \
  --data_dir data/NC8 \
  --replica 0 \
  --T 2000 \
  --d 8 \
  --lag 16 \
  --mlp_hidden 64 \
  --mlp_max_iter 50 \
  --pretrain_epochs 0 \
  --inner_step_count 10 \
  --inner_lr 0.01 \
  --inner_batch_size 0 \
  --ts_rank 4 \
  --tau 0.5 \
  --lambda_sparse 8 \
  --score_type mlp \
  --eval 2000 \
  --batch_size 20 \
  --n_cands 10000 \
  --n_grads 10 \
  --hidden_size 64 \
  --dropout 0.1 \
  --tol 1e-9 \
  --seed 0 \
  --no-plot \
  --outdir "$OUT"
