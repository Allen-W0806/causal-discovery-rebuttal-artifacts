#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-full_gpu_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/cc_runs/${RUN_ID}}"
GPU_ID="${GPU_ID:-0}"

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

echo "GPU baseline run id: ${RUN_ID}"
echo "GPU baseline root: ${RUN_ROOT}"
echo "GPU id: ${GPU_ID}"

# UnCLe reads data from uncle/datasets/ (hardcoded upstream); create symlink to data/.
if [[ ! -e "${ROOT}/uncle/datasets" ]]; then
  ln -s "${ROOT}/data" "${ROOT}/uncle/datasets"
fi

run_uncle() {
  local dataset="$1"
  local experiment="$2"
  local kernel="$3"
  local blocks="$4"
  local filters="$5"
  local recon_epochs="$6"
  local joint_epochs="$7"

  echo "RUN UnCLe/${dataset}"
  echo "NOTE UnCLe upstream writes timestamped logs under uncle/bin/logs."
  (
    cd "${ROOT}/uncle/bin"
    python run_grid_search.py \
      --experiment "${experiment}" \
      --K "${kernel}" \
      --num-hidden-layers "${blocks}" \
      --hidden-layer-size "${filters}" \
      --num-epochs-1 "${recon_epochs}" \
      --num-epochs-2 "${joint_epochs}" \
      --initial-lr 0.0003 \
      --cuda-i "${GPU_ID}"
  )
}

run_uncle NC8 unicsl_nc8 8 6 20 1000 2000
run_uncle ND8 unicsl_nd8 8 6 20 1000 2000
run_uncle FINANCE unicsl_finance 2 3 24 500 10000

for DATASET in NC8 ND8 FINANCE; do
  case "${DATASET}" in
    NC8)
      DATA_DIR="${ROOT}/data/NC8"
      CMLP_LAG=16; CMLP_LR=0.005
      TCDF_KERNEL=16; TCDF_EPOCHS=1000; TCDF_LR=0.005
      GVAR_ORDER=16; GVAR_LAYERS=1; GVAR_EPOCHS=1000
      ;;
    ND8)
      DATA_DIR="${ROOT}/data/ND8"
      CMLP_LAG=16; CMLP_LR=0.005
      TCDF_KERNEL=16; TCDF_EPOCHS=1000; TCDF_LR=0.005
      GVAR_ORDER=16; GVAR_LAYERS=1; GVAR_EPOCHS=1000
      ;;
    FINANCE)
      DATA_DIR="${ROOT}/data/Finance"
      CMLP_LAG=3; CMLP_LR=0.001
      TCDF_KERNEL=5; TCDF_EPOCHS=2000; TCDF_LR=0.01
      GVAR_ORDER=3; GVAR_LAYERS=2; GVAR_EPOCHS=500
      ;;
  esac

  run_job "cMLP/${DATASET}" "${RUN_ROOT}/cMLP/${DATASET}" bash -lc "
    set -euo pipefail
    for LAM in 0 0.5 1 1.5 2; do
      python cMLP/experiments/run_nc8_cmlp_baseline.py \
        --data-dir '${DATA_DIR}' \
        --output-dir '${RUN_ROOT}/cMLP/${DATASET}/lambda_'\"\${LAM}\" \
        --lag '${CMLP_LAG}' \
        --hidden 20 \
        --penalty H \
        --lam \"\${LAM}\" \
        --lr '${CMLP_LR}' \
        --max-iter 1000
    done
  "

  run_job "TCDF/${DATASET}" "${RUN_ROOT}/TCDF/${DATASET}" bash -lc "
    set -euo pipefail
    for ALPHA in 0 0.5 1 1.5 2; do
      python TCDF/experiments/run_nc8_tcdf_baseline.py \
        --data-dir '${DATA_DIR}' \
        --output-dir '${RUN_ROOT}/TCDF/${DATASET}/alpha_'\"\${ALPHA}\" \
        --kernel-size '${TCDF_KERNEL}' \
        --hidden-layers 1 \
        --epochs '${TCDF_EPOCHS}' \
        --learning-rate '${TCDF_LR}' \
        --significance \"\${ALPHA}\"
    done
  "

  run_job "GVAR/${DATASET}" "${RUN_ROOT}/GVAR/${DATASET}" bash -lc "
    set -euo pipefail
    for LAM in 0 0.75 1.5 2.25 3; do
      for GAMMA in 0 0.00625 0.0125 0.01875 0.025; do
        python GVAR/experiments/run_nc8_gvar_baseline.py \
          --data-dir '${DATA_DIR}' \
          --output-dir '${RUN_ROOT}/GVAR/${DATASET}/lambda_'\"\${LAM}\"'_gamma_'\"\${GAMMA}\" \
          --order '${GVAR_ORDER}' \
          --num-hidden-layers '${GVAR_LAYERS}' \
          --num-epochs '${GVAR_EPOCHS}' \
          --initial-lr 0.0001 \
          --lambda-value \"\${LAM}\" \
          --gamma-value \"\${GAMMA}\" \
          --use-cuda
      done
    done
  "

  run_job "CUTS+/${DATASET}" "${RUN_ROOT}/CUTSplus/${DATASET}" \
    python scripts/run_cutsplus_appendix_l.py \
      --dataset "${DATASET}" \
      --output-root "${RUN_ROOT}/CUTSplus" \
      --lr 0.001 \
      --epochs 64 \
      --max-groups 32 \
      --lambda-grid 0.1 0.05 0.01 0.005 \
      --gpu "${GPU_ID}"

  run_job "JRNGC/${DATASET}" "${RUN_ROOT}/JRNGC/${DATASET}" \
    python scripts/run_jrngc_appendix_l.py \
      --dataset "${DATASET}" \
      --output-root "${RUN_ROOT}/JRNGC" \
      --gpu "${GPU_ID}"
done

echo "GPU jobs finished or skipped: ${RUN_ROOT}"
