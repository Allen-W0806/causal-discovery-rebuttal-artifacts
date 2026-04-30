#!/usr/bin/env python3
"""Convert the simulated fMRI dataset (all 68 variables) into UnCLe CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE_ROOT = Path("data/simulated_fMRI/sim_fmri_64obs_4latent_compatible")
DEFAULT_OUTPUT_DIR = Path("uncle/datasets/sim_fmri_68full_4latent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert full 68-variable fMRI samples into the CSV layout expected by UnCLe."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    split_dir = source_root / args.split
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    sample_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not sample_dirs:
        raise RuntimeError(f"No sample directories found under {split_dir}")

    for i, sample_dir in enumerate(sample_dirs):
        x_full = np.load(sample_dir / "X.npy")   # [T, 68]
        a_full = np.load(sample_dir / "A.npy")   # [68, 68]

        pd.DataFrame(x_full).to_csv(
            output_dir / f"sim_fmri_68full_4latent_data_{i}.csv", index=False
        )
        pd.DataFrame(a_full).to_csv(
            output_dir / f"sim_fmri_68full_4latent_struct_{i}.csv", index=False
        )

    manifest = {
        "source_root": str(source_root),
        "source_split": args.split,
        "num_simulations": len(sample_dirs),
        "output_dir": str(output_dir),
        "n_variables": 68,
        "note": "Full 68-variable data (64 obs + 4 latent) from X.npy and A.npy.",
    }
    with open(output_dir / "conversion_manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(
        f"Converted {len(sample_dirs)} simulations from {split_dir} into {output_dir}. "
        f"Full shape per simulation: {list(x_full.shape)}"
    )


if __name__ == "__main__":
    main()
