# GVAR in base_model

- Baseline: Generalised Vector Autoregression (self-explaining neural Granger model)
- Legacy path preserved as symlink:
  `/storage/home/ydk297/projects/meta_causal_discovery/GVAR`

Main code locations:

- `models/`: model implementations
- `training.py`, `utils.py`, `experimental_utils.py`: local training and utilities
- `bin/run_grid_search.py`: main upstream driver
- `experiments/run_nc8_gvar_baseline.py`: local nc8 baseline entry
- `scripts/run_grid_search.py` and `scripts/run_nc8_baseline.py`: stable wrapper symlinks

Shared datasets:

- `data/shared_datasets/nc8/raw`
- `data/shared_datasets/simulated_EEG_68_static_v2/raw`
- `data/shared_datasets/simulated_fMRI/raw`

Results:

- Non-latent runs go to `results/non_latent/<dataset>/`
- Latent runs go to `results/latent/<dataset>/`

Notes:

- Original upstream datasets remain under `datasets/`.
- `configs/task_layout.example.yaml` is a planning template only.
- No experiments were run during the reorganization.

