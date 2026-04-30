# TCDF in base_model

- Baseline: Temporal Causal Discovery Framework
- Legacy path preserved as symlink:
  `/storage/home/ydk297/projects/meta_causal_discovery/TCDF`

Main code locations:

- `TCDF.py`, `model.py`, `depthwise.py`: core implementation files
- `runTCDF.py`: main upstream executable entry
- `experiments/`: local experiment scripts for this workspace
- `scripts/run_nc8_baseline.py`: stable symlink to the nc8 baseline entry
- `scripts/run_tcdf.py`: stable symlink to the upstream runner

Shared datasets:

- `data/shared_datasets/nc8/raw`
- `data/shared_datasets/simulated_EEG_68_static_v2/raw`
- `data/shared_datasets/simulated_fMRI/raw`

Results:

- Non-latent runs go to `results/non_latent/<dataset>/`
- Latent runs go to `results/latent/<dataset>/`

Notes:

- Existing upstream demo data remains under `data/`.
- `configs/task_layout.example.yaml` is a planning template only.
- No experiments were run during the reorganization.

