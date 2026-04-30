This directory contains lightweight baseline wrappers, configuration files, and selected source code needed to reproduce the rebuttal baseline comparisons under the unified evaluation protocol. Full third-party baseline repositories are not vendored unless required for execution. Method-specific hyperparameters follow the reported settings in the corresponding prior baseline implementations.

## Baselines

- `VAR/` — Vector Autoregression baseline
- `uncle/` — UnCLe baseline
- `PCMCI/` — PCMCI baseline
- `cMLP/` — cMLP baseline
- `GVAR/` — GVAR baseline
- `DYNOTEAR/` — DYNOTEAR baseline
- `TCDF/` — TCDF baseline
- `JRNGC/` — JRNGC baseline
- `VARLiNGAM/` — VARLiNGAM baseline

## Data setup

Required input data are expected under `data/` before running:

```
data/NC8/        nc8_data_{0..4}.csv, nc8_struct_{0..4}.csv
data/ND8/        nc8_dynamic_{0..4}.csv, nc8_structure_dynamic.npy
data/Finance/    finance_data_{0..4}.csv, finance_struct_{0..4}.csv
```

## Running

GPU baselines (cMLP, TCDF, GVAR, CUTS+, JRNGC, UnCLe):

```bash
bash scripts/run_gpu_baselines.sh
```

CPU baselines (VAR, PCMCI, VARLiNGAM, DYNOTEAR):

```bash
bash scripts/run_cpu_baselines.sh
```
