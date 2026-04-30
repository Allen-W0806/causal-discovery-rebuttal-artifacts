# uncle in base_model

- Baseline: UnCLe dynamic causal discovery repository
- Legacy path preserved as symlink:
  `/storage/home/ydk297/projects/meta_causal_discovery/uncle`

Main code locations:

- `bin/experimental_utils.py`: model/training utilities
- `bin/run_grid_search.py`: main upstream driver
- `scripts/run_grid_search.py`: stable wrapper symlink
- `scripts/run_nc8_mask2_latent_obs`: stable wrapper for the current nc8 latent adapter run
- `scripts/run_fmri_64obs_4latent`: stable wrapper for the current fMRI latent adapter run

Shared datasets:

- `data/shared_datasets/nc8/raw`
- `data/shared_datasets/simulated_EEG_68_static_v2/raw`
- `data/shared_datasets/simulated_fMRI/raw`

Results:

- Non-latent runs go to `results/non_latent/<dataset>/`
- Latent runs go to `results/latent/<dataset>/`

Notes:

- Existing method-specific converted CSV adapters remain under `datasets/`.
- `configs/task_layout.example.yaml` is a planning template only.
- No experiments were run during the reorganization.

