#!/usr/bin/env python3
"""Convert the simulated fMRI latent-compatible dataset into UnCLe CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE_ROOT = Path(
    "/storage/home/ydk297/projects/meta_causal_discovery/meta_latent_4.20/data/"
    "simulated_fMRI/sim_fmri_64obs_4latent_compatible"
)
DEFAULT_OUTPUT_DIR = Path(
    "/storage/home/ydk297/projects/meta_causal_discovery/uncle/datasets/"
    "sim_fmri_64obs_4latent"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert simulated fMRI samples into the CSV layout expected by UnCLe."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Absolute path to the original latent-compatible simulated fMRI dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to convert. Default uses the 50-sample test split to match UnCLe's run_fmri setup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Absolute path where the converted UnCLe CSV files should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    split_dir = source_root / args.split
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    sample_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under {split_dir}")

    converted_shapes = []
    observed_edge_counts = []

    for i, sample_dir in enumerate(sample_dirs):
        x_obs = np.load(sample_dir / "X_obs.npy")
        a_full = np.load(sample_dir / "A.npy")

        if x_obs.ndim != 2:
            raise ValueError(f"Expected 2D X_obs in {sample_dir}, found shape {x_obs.shape}")
        if a_full.ndim != 2 or a_full.shape[0] != a_full.shape[1]:
            raise ValueError(f"Expected square A.npy in {sample_dir}, found shape {a_full.shape}")
        if a_full.shape[0] < x_obs.shape[1]:
            raise ValueError(
                f"A.npy in {sample_dir} has fewer nodes than X_obs: {a_full.shape} vs {x_obs.shape}"
            )

        a_obs = a_full[: x_obs.shape[1], : x_obs.shape[1]]
        converted_shapes.append({"sample": sample_dir.name, "x_obs": list(x_obs.shape)})
        observed_edge_counts.append(int(a_obs.sum()))

        data_path = output_dir / f"sim_fmri_64obs_4latent_data_{i}.csv"
        struct_path = output_dir / f"sim_fmri_64obs_4latent_struct_{i}.csv"

        pd.DataFrame(x_obs).to_csv(data_path, index=False)
        pd.DataFrame(a_obs).to_csv(struct_path, index=False)

    manifest = {
        "source_root": str(source_root),
        "source_split": args.split,
        "num_simulations": len(sample_dirs),
        "output_dir": str(output_dir),
        "data_file_pattern": "sim_fmri_64obs_4latent_data_{i}.csv",
        "struct_file_pattern": "sim_fmri_64obs_4latent_struct_{i}.csv",
        "observed_node_count": converted_shapes[0]["x_obs"][1],
        "timepoints_per_simulation": converted_shapes[0]["x_obs"][0],
        "observed_edge_count_min": min(observed_edge_counts),
        "observed_edge_count_max": max(observed_edge_counts),
        "note": (
            "Converted from X_obs.npy and the observed-observed block of A.npy. "
            "This preserves UnCLe's default CSV-based loading and evaluation flow."
        ),
    }
    with open(output_dir / "conversion_manifest.json", "w", encoding="ascii") as fp:
        json.dump(manifest, fp, indent=2)

    print(
        f"Converted {len(sample_dirs)} simulations from {split_dir} into {output_dir}. "
        f"Observed shape per simulation: {converted_shapes[0]['x_obs']}"
    )


if __name__ == "__main__":
    main()
