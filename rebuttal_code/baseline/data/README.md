Place dataset directories here before running the baselines:

    data/NC8/       — nc8_data_{0..4}.csv, nc8_struct_{0..4}.csv
    data/ND8/       — nc8_dynamic_{0..4}.csv, nc8_structure_dynamic.npy
    data/Finance/   — finance_data_{0..4}.csv, finance_struct_{0..4}.csv

The run scripts (scripts/run_gpu_baselines.sh, scripts/run_cpu_baselines.sh) read
from this directory.  UnCLe additionally reads from uncle/datasets/, which the GPU
script creates as a symlink to this directory automatically.
