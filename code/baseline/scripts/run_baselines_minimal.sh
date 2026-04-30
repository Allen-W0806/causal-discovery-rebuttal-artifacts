#!/usr/bin/env bash
# run_baselines_minimal.sh
# Syntax-checks and prints the entry command for each baseline.
# Does NOT launch full training runs.
# To run baselines for real, use run_gpu_baselines.sh or run_cpu_baselines.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_NC8="${ROOT}/data/NC8"

echo "=== Baseline entry points (relative to: ${ROOT}) ==="
echo ""

echo "--- VAR ---"
echo "python VAR/scripts/run_nc8_baseline.py --data-dir data/NC8 --output-dir results/VAR/NC8 --lag 16"
echo ""

echo "--- VARLiNGAM ---"
echo "python scripts/run_varlingam_appendix_l.py --dataset NC8 --output-root results/VARLiNGAM"
echo ""

echo "--- PCMCI ---"
echo "python PCMCI/experiments/run_nc8_pcmci_baseline.py --data-dir data/NC8 --output-dir results/PCMCI/NC8 --tau-max-values 16 --pc-alpha 0.01"
echo ""

echo "--- DYNOTEARS ---"
echo "python scripts/run_dynotears_appendix_l.py --dataset NC8 --output-root results/DYNOTEARS --lags 2 3 4 5"
echo ""

echo "--- cMLP ---"
echo "python cMLP/experiments/run_nc8_cmlp_baseline.py --data-dir data/NC8 --output-dir results/cMLP/NC8 --lag 16 --hidden 20 --penalty H --lam 1.0 --lr 0.005 --max-iter 1000"
echo ""

echo "--- GVAR ---"
echo "python GVAR/experiments/run_nc8_gvar_baseline.py --data-dir data/NC8 --output-dir results/GVAR/NC8 --order 16 --num-hidden-layers 1 --num-epochs 1000 --initial-lr 0.0001 --lambda-value 1.5 --gamma-value 0.0125 --use-cuda"
echo ""

echo "--- TCDF ---"
echo "python TCDF/experiments/run_nc8_tcdf_baseline.py --data-dir data/NC8 --output-dir results/TCDF/NC8 --kernel-size 16 --hidden-layers 1 --epochs 1000 --learning-rate 0.005 --significance 1.0"
echo ""

echo "--- JRNGC ---"
echo "python scripts/run_jrngc_appendix_l.py --dataset NC8 --output-root results/JRNGC"
echo ""

echo "--- CUTS+ ---"
echo "python scripts/run_cutsplus_appendix_l.py --dataset NC8 --output-root results/CUTSplus --lr 0.001 --epochs 64 --max-groups 32 --lambda-grid 0.1 0.05 0.01 0.005"
echo ""

echo "--- UnCLe ---"
echo "(cd uncle/bin && python run_grid_search.py --experiment unicsl_nc8 --K 8 --num-hidden-layers 6 --hidden-layer-size 20 --num-epochs-1 1000 --num-epochs-2 2000 --initial-lr 0.0003)"
echo ""

echo "=== Syntax check passed. Use run_gpu_baselines.sh or run_cpu_baselines.sh to run experiments. ==="
