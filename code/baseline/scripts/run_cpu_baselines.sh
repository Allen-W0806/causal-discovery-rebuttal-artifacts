#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-full_cpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/cc_runs/${RUN_ID}}"

mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

run_job() {
  local name="$1"
  local outdir="$2"
  shift 2
  if [[ -e "${outdir}" ]]; then
    echo "SKIP ${name}: output exists at ${outdir}"
    return 0
  fi
  mkdir -p "$(dirname "${outdir}")"
  echo "RUN ${name}"
  echo "OUT ${outdir}"
  "$@"
}

echo "CPU baseline run id: ${RUN_ID}"
echo "CPU baseline root: ${RUN_ROOT}"

for DATASET in NC8 ND8 FINANCE; do
  case "${DATASET}" in
    NC8) DATA_DIR="${ROOT}/data/NC8"; VAR_LAG=16; PCMCI_TAU=16 ;;
    ND8) DATA_DIR="${ROOT}/data/ND8"; VAR_LAG=16; PCMCI_TAU=16 ;;
    FINANCE) DATA_DIR="${ROOT}/data/Finance"; VAR_LAG=5; PCMCI_TAU=5 ;;
  esac

  run_job "VAR/${DATASET}" "${RUN_ROOT}/VAR/${DATASET}" \
    python VAR/scripts/run_nc8_baseline.py \
      --data-dir "${DATA_DIR}" \
      --output-dir "${RUN_ROOT}/VAR/${DATASET}" \
      --lag "${VAR_LAG}"

  run_job "PCMCI/${DATASET}" "${RUN_ROOT}/PCMCI/${DATASET}" \
    python PCMCI/experiments/run_nc8_pcmci_baseline.py \
      --data-dir "${DATA_DIR}" \
      --output-dir "${RUN_ROOT}/PCMCI/${DATASET}" \
      --tau-max-values "${PCMCI_TAU}" \
      --pc-alpha 0.01

  run_job "VARLiNGAM/${DATASET}" "${RUN_ROOT}/VARLiNGAM/${DATASET}" \
    python scripts/run_varlingam_appendix_l.py \
      --dataset "${DATASET}" \
      --output-root "${RUN_ROOT}/VARLiNGAM"

  run_job "DYNOTEARS/${DATASET}" "${RUN_ROOT}/DYNOTEARS/${DATASET}" \
    python scripts/run_dynotears_appendix_l.py \
      --dataset "${DATASET}" \
      --output-root "${RUN_ROOT}/DYNOTEARS" \
      --max-iter 1000 \
      --lambda-w 0.1 \
      --lambda-a 0.1 \
      --lags 2 3 4 5
done

echo "CPU jobs finished or skipped: ${RUN_ROOT}"
