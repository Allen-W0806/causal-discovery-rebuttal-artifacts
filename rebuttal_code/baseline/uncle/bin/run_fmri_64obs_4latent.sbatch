#!/bin/bash
#SBATCH --job-name=uncle_fmri64
#SBATCH --partition=epyc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/storage/home/ydk297/projects/meta_causal_discovery/uncle/slurm_logs/%x-%j.out
#SBATCH --error=/storage/home/ydk297/projects/meta_causal_discovery/uncle/slurm_logs/%x-%j.err

set -euo pipefail

cd /storage/home/ydk297/projects/meta_causal_discovery/uncle/bin
export PYTHONPATH="${PYTHONPATH:-}:../"

conda run -n uncle_clean env LD_LIBRARY_PATH=/storage/home/ydk297/miniconda3/envs/uncle_clean/lib python run_grid_search.py \
  --experiment unicsl_fmri_64obs_4latent \
  --num-sim 50 \
  --K 6 \
  --num-hidden-layers 8 \
  --hidden-layer-size 12 \
  --num-epochs-1 1000 \
  --num-epochs-2 2000 \
  --initial-lr 0.00001 \
  --seed 0 \
  --cuda-i 0
