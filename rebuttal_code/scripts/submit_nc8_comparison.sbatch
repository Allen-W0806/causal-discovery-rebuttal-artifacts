#!/bin/bash
#SBATCH --job-name=nc8_comparison
#SBATCH --partition=epyc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

conda run -n base bash scripts/run_nc8_formal_comparison.sh \
  review_runs/nc8_formal_comparison
