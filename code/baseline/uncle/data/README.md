# uncle Data Notes

Use `shared_datasets/` as the common entry point for the three benchmark datasets that
will be compared across baselines.

Shared dataset entries:

- `shared_datasets/nc8/raw`
- `shared_datasets/simulated_EEG_68_static_v2/raw`
- `shared_datasets/simulated_fMRI/raw`

Method-specific converted adapters were left in place under `datasets/`, for example:

- `datasets/NC8_mask2_latent_obs/`
- `datasets/sim_fmri_64obs_4latent/`
- `datasets/sim_fmri_68full_4latent/`

Those adapter datasets are baseline-specific and were not used as the new shared source of
truth.

