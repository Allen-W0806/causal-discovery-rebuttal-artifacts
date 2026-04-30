#!/usr/bin/env python3
"""Convert the NC8 mask-2 latent benchmark into an observed-only CSV layout for UnCLe.

This keeps the same five NC8 simulations used by meta_latent_4.13, drops the two
masked variables (default: x and a at indices [0, 5]), and writes:

  - observed-only time-series CSVs with columns [y, z, w, o, b, c]
  - observed-only GT adjacency CSVs (the XX block only)

UnCLe will therefore be evaluated on the observed subgraph induced by the latent
benchmark mask. It will not recover latent->observed ZX edges, because those latent
variables are removed from the data and target structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_ROOT = Path(
    "/storage/home/ydk297/projects/meta_causal_discovery/meta_latent_4.13/data/nc8"
)
DEFAULT_OUTPUT_DIR = Path(
    "/storage/home/ydk297/projects/meta_causal_discovery/uncle/datasets/NC8_mask2_latent_obs"
)
DEFAULT_HIDDEN_INDICES = [0, 5]
DATA_PATTERN = "nc8_mask2_latent_obs_data_{i}.csv"
STRUCT_PATTERN = "nc8_mask2_latent_obs_struct_{i}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Absolute path to the shared NC8 CSV dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Absolute path where the observed-only UnCLe CSV files should be written.",
    )
    parser.add_argument(
        "--hidden-indices",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_INDICES,
        help="Indices of variables to mask out as latent nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted(source_root.glob("nc8_data_*.csv"))
    struct_files = sorted(source_root.glob("nc8_struct_*.csv"))
    if not data_files or len(data_files) != len(struct_files):
        raise RuntimeError(
            f"Expected matched nc8_data_*.csv / nc8_struct_*.csv pairs under {source_root}"
        )

    hidden_indices = sorted(int(idx) for idx in args.hidden_indices)
    manifest: dict[str, object] = {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "hidden_indices": hidden_indices,
        "num_simulations": len(data_files),
        "data_file_pattern": DATA_PATTERN,
        "struct_file_pattern": STRUCT_PATTERN,
        "note": (
            "Observed-only NC8 view for the latent benchmark with hidden variables removed. "
            "GT structure is the induced XX block only, because UnCLe predicts over observed variables."
        ),
    }

    observed_variable_names: list[str] | None = None
    offdiag_edge_counts: list[int] = []

    for i, (data_path, struct_path) in enumerate(zip(data_files, struct_files)):
        data_df = pd.read_csv(data_path)
        struct_df = pd.read_csv(struct_path)

        if list(data_df.columns) != list(struct_df.columns):
            raise ValueError(f"Column mismatch between {data_path} and {struct_path}")
        columns = list(data_df.columns)
        struct_df.index = columns
        if max(hidden_indices) >= len(columns):
            raise ValueError(f"Hidden indices {hidden_indices} exceed NC8 width {len(columns)}")

        hidden_columns = [columns[idx] for idx in hidden_indices]
        observed_columns = [name for j, name in enumerate(columns) if j not in hidden_indices]
        if observed_variable_names is None:
            observed_variable_names = observed_columns
            manifest["hidden_variable_names"] = hidden_columns
            manifest["observed_variable_names"] = observed_columns
        elif observed_columns != observed_variable_names:
            raise ValueError("Observed variable ordering changed across NC8 simulations.")

        obs_data_df = data_df[observed_columns].copy()
        obs_struct_df = struct_df.loc[observed_columns, observed_columns].copy()

        offdiag_edges = int(obs_struct_df.values.sum() - obs_struct_df.values.diagonal().sum())
        offdiag_edge_counts.append(offdiag_edges)

        obs_data_df.to_csv(output_dir / DATA_PATTERN.format(i=i), index=False)
        obs_struct_df.to_csv(output_dir / STRUCT_PATTERN.format(i=i), index=False)

    manifest["observed_offdiag_edge_count_min"] = min(offdiag_edge_counts)
    manifest["observed_offdiag_edge_count_max"] = max(offdiag_edge_counts)

    with (output_dir / "conversion_manifest.json").open("w", encoding="ascii") as handle:
        json.dump(manifest, handle, indent=2)

    print(
        f"Converted {len(data_files)} NC8 simulations from {source_root} into {output_dir}. "
        f"Observed variables: {manifest['observed_variable_names']}"
    )


if __name__ == "__main__":
    main()
