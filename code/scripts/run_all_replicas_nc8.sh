#!/usr/bin/env bash
# Run all three methods on NC8 replicas 0-4.
# Usage: bash scripts/run_all_replicas_nc8.sh [SEED=0]
# Example: bash scripts/run_all_replicas_nc8.sh 0
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE

SEED="${1:-0}"

for REPLICA in 0 1 2 3 4; do
  echo "=== Running low_rank_bo  replica=${REPLICA} seed=${SEED} ==="
  bash scripts/run_nc8.sh "$REPLICA" "$SEED"

  echo "=== Running nodewise_greedy  replica=${REPLICA} seed=${SEED} ==="
  bash scripts/run_nodewise_greedy_nc8.sh "$REPLICA" "$SEED"

  echo "=== Running nodewise_exhaustive  replica=${REPLICA} seed=${SEED} ==="
  bash scripts/run_nodewise_exhaustive_nc8.sh "$REPLICA" "$SEED"
done
