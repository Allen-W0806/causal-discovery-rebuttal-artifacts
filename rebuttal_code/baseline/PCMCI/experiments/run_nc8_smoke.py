#!/usr/bin/env python
"""Minimal PCMCI smoke test for the workspace NC8 dataset.

This script intentionally uses the simplest stable PCMCI path first:
PCMCI + ParCorr(significance="analytic") on one NC8 sequence through the
upstream Tigramite package checkout.

The selected NC8 format is the CSV pair layout used by the clean nonlatent
repo:
    nc8_data_*.csv
    nc8_struct_*.csv

That choice is explicit in the loader so it is easy to swap later if you want
to benchmark a different NC8 export.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("/storage/home/ydk297/projects/meta_causal_discovery/data/nc8")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results_nc8_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--tau-min", type=int, default=1)
    parser.add_argument("--tau-max", type=int, default=2)
    parser.add_argument("--pc-alpha", type=float, default=0.2)
    parser.add_argument("--alpha-level", type=float, default=0.01)
    parser.add_argument("--ci-test", choices=["parcorr"], default="parcorr")
    parser.add_argument("--verbosity", type=int, default=0)
    return parser.parse_args()


def _is_numeric_row(tokens: list[str]) -> bool:
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def _read_csv_matrix(path: Path) -> tuple[list[str] | None, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().strip()

    if not first_line:
        raise ValueError(f"CSV file is empty: {path}")

    first_tokens = [token.strip() for token in first_line.split(",")]
    has_header = not _is_numeric_row(first_tokens)
    header = first_tokens if has_header else None
    skiprows = 1 if has_header else 0

    matrix = np.loadtxt(path, delimiter=",", skiprows=skiprows, dtype=np.float64)
    matrix = np.atleast_2d(matrix)
    return header, matrix


def load_nc8_bundle(data_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    data_paths = sorted(data_dir.glob("nc8_data_*.csv"))
    struct_paths = sorted(data_dir.glob("nc8_struct_*.csv"))

    if not data_paths or not struct_paths:
        raise FileNotFoundError(
            f"Expected nc8_data_*.csv and nc8_struct_*.csv under {data_dir}."
        )
    if len(data_paths) != len(struct_paths):
        raise ValueError(
            "NC8 data/structure file counts do not match: "
            f"{len(data_paths)} vs {len(struct_paths)}"
        )

    sequences: list[np.ndarray] = []
    graphs: list[np.ndarray] = []
    variable_names: list[str] | None = None

    for data_path, struct_path in zip(data_paths, struct_paths):
        data_header, data_matrix = _read_csv_matrix(data_path)
        struct_header, struct_matrix = _read_csv_matrix(struct_path)

        if variable_names is None:
            if data_header is not None:
                variable_names = data_header
            else:
                variable_names = [f"X{i}" for i in range(data_matrix.shape[1])]
        elif data_header is not None and data_header != variable_names:
            raise ValueError(f"Inconsistent NC8 data columns in {data_path}.")

        if struct_header is not None and struct_header != variable_names:
            raise ValueError(f"Inconsistent NC8 structure columns in {struct_path}.")

        if data_matrix.ndim != 2:
            raise ValueError(f"Expected 2-D sequence array in {data_path}, got {data_matrix.shape}.")
        if struct_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2-D structure array in {struct_path}, got {struct_matrix.shape}."
            )
        if struct_matrix.shape[0] != struct_matrix.shape[1]:
            raise ValueError(f"Expected square structure matrix in {struct_path}.")
        if data_matrix.shape[1] != struct_matrix.shape[0]:
            raise ValueError(
                f"Sequence width {data_matrix.shape[1]} does not match graph size "
                f"{struct_matrix.shape[0]} for {data_path} / {struct_path}."
            )

        sequences.append(data_matrix)
        graphs.append(struct_matrix)

    base_graph = graphs[0]
    for index, graph in enumerate(graphs[1:], start=1):
        if not np.array_equal(base_graph, graph):
            raise ValueError(
                "NC8 ground-truth graphs differ across replicas. "
                f"Mismatch found at structure index {index}."
            )

    return {
        "data_dir": str(data_dir),
        "data_paths": [str(path) for path in data_paths],
        "struct_paths": [str(path) for path in struct_paths],
        "variable_names": variable_names,
        "sequences": sequences,
        "ground_truth_graph": base_graph.astype(np.int64),
    }


def serializable_parents(
    parents: dict[int, list[tuple[int, int]]],
    graph: np.ndarray,
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    var_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    serialized: dict[str, list[dict[str, Any]]] = {}
    for target_index, parent_list in parents.items():
        target_name = var_names[target_index]
        rows: list[dict[str, Any]] = []
        for source_index, lag in parent_list:
            lag_index = abs(lag)
            rows.append(
                {
                    "source": var_names[source_index],
                    "source_index": int(source_index),
                    "target": target_name,
                    "target_index": int(target_index),
                    "lag": int(lag),
                    "link_type": str(graph[source_index, target_index, lag_index]),
                    "val": float(val_matrix[source_index, target_index, lag_index]),
                    "p_value": float(p_matrix[source_index, target_index, lag_index]),
                }
            )
        serialized[target_name] = rows
    return serialized


def graph_rows(
    graph: np.ndarray,
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    var_names: list[str],
    tau_min: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index in range(graph.shape[0]):
        for target_index in range(graph.shape[1]):
            for lag_index in range(tau_min, graph.shape[2]):
                link_type = str(graph[source_index, target_index, lag_index]).strip()
                if not link_type:
                    continue
                rows.append(
                    {
                        "source": var_names[source_index],
                        "source_index": source_index,
                        "target": var_names[target_index],
                        "target_index": target_index,
                        "lag": -lag_index,
                        "link_type": link_type,
                        "val": float(val_matrix[source_index, target_index, lag_index]),
                        "p_value": float(p_matrix[source_index, target_index, lag_index]),
                    }
                )
    rows.sort(key=lambda row: (row["target_index"], row["source_index"], row["lag"]))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, matrix: np.ndarray, header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(matrix.tolist())


def build_success_summary(config: dict[str, Any], link_rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# NC8 PCMCI Smoke Test",
            "",
            "- Status: success",
            f"- Dataset directory: `{config['data_dir']}`",
            f"- Selected sequence: `{config['selected_sequence_path']}`",
            f"- Selected structure: `{config['selected_structure_path']}`",
            f"- Sequence shape used: `{tuple(config['selected_sequence_shape'])}`",
            f"- Ground-truth adjacency shape: `{tuple(config['ground_truth_graph_shape'])}`",
            f"- Method: `PCMCI` + `ParCorr(significance=\"analytic\")`",
            f"- tau_min / tau_max: `{config['tau_min']}` / `{config['tau_max']}`",
            f"- pc_alpha: `{config['pc_alpha']}`",
            f"- alpha_level: `{config['alpha_level']}`",
            f"- Significant lagged links found: `{len(link_rows)}`",
            "",
            "The goal of this run was only to verify that the PCMCI baseline setup can ingest the",
            "selected NC8 format and complete a minimal end-to-end causal discovery",
            "pass without crashing.",
            "",
        ]
    )


def build_failure_summary(config: dict[str, Any], exc: BaseException) -> str:
    return "\n".join(
        [
            "# NC8 PCMCI Smoke Test",
            "",
            "- Status: failed",
            f"- Dataset directory: `{config.get('data_dir', 'unknown')}`",
            f"- Selected sequence path: `{config.get('selected_sequence_path', 'unknown')}`",
            f"- Method: `PCMCI` + `ParCorr(significance=\"analytic\")`",
            f"- Error: `{type(exc).__name__}: {exc}`",
            "",
            "See `smoke_stderr.txt` for the full traceback captured by the runner.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "output_dir": str(output_dir),
        "sequence_index": args.sequence_index,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "pc_alpha": args.pc_alpha,
        "alpha_level": args.alpha_level,
        "ci_test": args.ci_test,
        "verbosity": args.verbosity,
        "status": "starting",
    }

    try:
        bundle = load_nc8_bundle(args.data_dir)
        config["num_sequences_available"] = len(bundle["sequences"])
        config["all_sequence_paths"] = bundle["data_paths"]
        config["all_structure_paths"] = bundle["struct_paths"]
        config["variable_names"] = bundle["variable_names"]
        config["ground_truth_graph_shape"] = list(bundle["ground_truth_graph"].shape)
        config["ground_truth_edge_count"] = int(bundle["ground_truth_graph"].sum())

        if not 0 <= args.sequence_index < len(bundle["sequences"]):
            raise IndexError(
                f"sequence_index={args.sequence_index} is out of range for "
                f"{len(bundle['sequences'])} NC8 sequences."
            )

        selected_sequence = bundle["sequences"][args.sequence_index]
        selected_sequence_path = bundle["data_paths"][args.sequence_index]
        selected_structure_path = bundle["struct_paths"][args.sequence_index]

        config["selected_sequence_path"] = selected_sequence_path
        config["selected_structure_path"] = selected_structure_path
        config["selected_sequence_shape"] = list(selected_sequence.shape)

        # Use the native Tigramite DataFrame on one faithful NC8 sequence.
        dataframe = pp.DataFrame(selected_sequence, var_names=bundle["variable_names"])
        cond_ind_test = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=args.verbosity)

        results = pcmci.run_pcmci(
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            pc_alpha=args.pc_alpha,
            alpha_level=args.alpha_level,
        )

        parent_dict = pcmci.return_parents_dict(
            graph=results["graph"],
            val_matrix=results["val_matrix"],
        )
        parent_payload = serializable_parents(
            parents=parent_dict,
            graph=results["graph"],
            val_matrix=results["val_matrix"],
            p_matrix=results["p_matrix"],
            var_names=bundle["variable_names"],
        )
        link_rows = graph_rows(
            graph=results["graph"],
            val_matrix=results["val_matrix"],
            p_matrix=results["p_matrix"],
            var_names=bundle["variable_names"],
            tau_min=args.tau_min,
        )

        np.save(output_dir / "discovered_graph.npy", results["graph"])
        np.save(output_dir / "p_matrix.npy", results["p_matrix"])
        np.save(output_dir / "val_matrix.npy", results["val_matrix"])
        np.save(output_dir / "ground_truth_graph.npy", bundle["ground_truth_graph"])
        write_matrix_csv(
            output_dir / "ground_truth_graph.csv",
            bundle["ground_truth_graph"],
            header=bundle["variable_names"],
        )
        write_json(output_dir / "discovered_parents.json", parent_payload)
        write_csv(output_dir / "discovered_links.csv", link_rows)

        config["status"] = "success"
        config["discovered_link_count"] = len(link_rows)
        config["parent_counts"] = {
            bundle["variable_names"][node_index]: len(parent_list)
            for node_index, parent_list in parent_dict.items()
        }
        write_json(output_dir / "config.json", config)

        summary = build_success_summary(config, link_rows)
        (output_dir / "summary.md").write_text(summary, encoding="utf-8")

        print("NC8 smoke test completed successfully.")
        print(f"Dataset directory: {config['data_dir']}")
        print(f"Selected sequence: {selected_sequence_path}")
        print(f"Sequence shape: {tuple(config['selected_sequence_shape'])}")
        print("Method: PCMCI + ParCorr(significance='analytic')")
        print(f"tau_min={args.tau_min}, tau_max={args.tau_max}, pc_alpha={args.pc_alpha}, alpha_level={args.alpha_level}")
        print(f"Discovered lagged links: {len(link_rows)}")
        print(f"Output directory: {output_dir}")
        return 0

    except Exception as exc:  # pragma: no cover - used for smoke-test reporting
        config["status"] = "failed"
        config["error_type"] = type(exc).__name__
        config["error_message"] = str(exc)
        write_json(output_dir / "config.json", config)
        (output_dir / "summary.md").write_text(
            build_failure_summary(config, exc),
            encoding="utf-8",
        )
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
