# cMLP in base_model

- Baseline: Neural Granger Causality repository
- Legacy path preserved as symlink:
  `/storage/home/ydk297/projects/meta_causal_discovery/cMLP`

Main code locations:

- `models/`: cMLP / cLSTM / cRNN implementations
- `experiments/run_nc8_cmlp_baseline.py`: local nc8 baseline entry
- `scripts/run_nc8_baseline.py`: stable wrapper symlink

Shared datasets:

- `data/shared_datasets/nc8/raw`
- `data/shared_datasets/simulated_EEG_68_static_v2/raw`
- `data/shared_datasets/simulated_fMRI/raw`

Results:

- Non-latent runs go to `results/non_latent/<dataset>/`
- Latent runs go to `results/latent/<dataset>/`

Notes:

- `configs/task_layout.example.yaml` is a planning template only.
- No experiments were run during the reorganization.

