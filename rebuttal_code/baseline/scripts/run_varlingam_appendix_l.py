#!/usr/bin/env python
"""Run VARLiNGAM with Appendix L lag grid on NC8, ND8, or FINANCE."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from appendix_l_config import DATASET_NAMES, get_dataset_spec, get_params  # noqa: E402
from baseline_common import evaluate_score_matrix, load_replicas, write_json, write_matrix_csv, write_rows_csv  # noqa: E402

from lingam.var_lingam import VARLiNGAM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT / "results" / "VARLiNGAM")
    parser.add_argument("--alpha", type=float, default=0.0, help="Optional absolute-score threshold for binary graph. 0 uses top-k GT edges.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def summarize_lagged_adjacency(adjacency_matrices: np.ndarray) -> np.ndarray:
    adjacency_matrices = np.asarray(adjacency_matrices, dtype=np.float64)
    if adjacency_matrices.ndim != 3:
        raise ValueError(f"Expected adjacency_matrices_ with shape [lag+1,n,n], got {adjacency_matrices.shape}")
    if adjacency_matrices.shape[0] == 1:
        return np.abs(adjacency_matrices[0])
    return np.max(np.abs(adjacency_matrices[1:]), axis=0)


def main() -> int:
    args = parse_args()
    params = get_params("VARLiNGAM", args.dataset)
    dataset_spec = get_dataset_spec(args.dataset)
    output_dir = (args.output_root / args.dataset).resolve()
    predicted_dir = output_dir / "predicted_graphs"
    predicted_dir.mkdir(parents=True, exist_ok=True)

    replicas = load_replicas(
        dataset=args.dataset,
        data_dir=Path(dataset_spec["data_dir"]),
        data_glob=dataset_spec["data_glob"],
        struct_glob=dataset_spec["struct_glob"],
    )
    config = {
        "method": "VARLiNGAM",
        "dataset": args.dataset,
        "params": params,
        "dataset_spec": {k: str(v) for k, v in dataset_spec.items()},
        "output_dir": str(output_dir),
        "command": " ".join(sys.argv),
        "score_definition": "score[i,j] = max_lag abs(model.adjacency_matrices_[lag,i,j]) over lag >= 1",
        "binary_definition": "default top-k by GT edge count unless --alpha > 0",
    }
    write_json(output_dir / "config.json", config)

    rows: list[dict] = []
    aggregate_by_lag: dict[int, list[dict]] = {}

    for replica in replicas:
        sequence = np.asarray(replica["sequence"], dtype=np.float64)
        graph = np.asarray(replica["graph"], dtype=np.int64)
        variable_names = replica["variable_names"]
        replica_index = int(replica["replica_index"])

        for lag in params["lag_grid"]:
            start = time.perf_counter()
            model = VARLiNGAM(lags=int(lag), criterion=None, random_state=args.seed)
            model.fit(sequence)
            runtime = time.perf_counter() - start

            weighted_tensor = np.asarray(model.adjacency_matrices_, dtype=np.float64)
            score_matrix = summarize_lagged_adjacency(weighted_tensor)
            binary_graph = (score_matrix > args.alpha).astype(np.int64) if args.alpha > 0 else None
            metrics = evaluate_score_matrix(score_matrix, graph, binary_graph)
            if binary_graph is None:
                from baseline_common import threshold_by_top_k

                binary_graph = threshold_by_top_k(score_matrix, graph)

            setting_dir = predicted_dir / f"lag_{lag}"
            prefix = setting_dir / f"replica_{replica_index}"
            setting_dir.mkdir(parents=True, exist_ok=True)
            np.save(prefix.with_name(prefix.name + "_weighted_adjacency_tensor.npy"), weighted_tensor)
            np.save(prefix.with_name(prefix.name + "_score_matrix.npy"), score_matrix)
            np.save(prefix.with_name(prefix.name + "_binary_graph.npy"), binary_graph)
            write_matrix_csv(prefix.with_name(prefix.name + "_score_matrix.csv"), score_matrix, variable_names)
            write_matrix_csv(prefix.with_name(prefix.name + "_binary_graph.csv"), binary_graph, variable_names)

            row = {
                "method": "VARLiNGAM",
                "dataset": args.dataset,
                "replica_index": replica_index,
                "lag": int(lag),
                "runtime_sec": float(runtime),
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "f1": float(metrics["f1"]),
                "shd": float(metrics["shd"]),
                "selected_hyperparameters": json.dumps({"lag": int(lag)}, sort_keys=True),
            }
            rows.append(row)
            aggregate_by_lag.setdefault(int(lag), []).append(row)

    aggregate_rows = []
    for lag, lag_rows in sorted(aggregate_by_lag.items()):
        aggregate_rows.append(
            {
                "lag": lag,
                "replicas": len(lag_rows),
                "mean_auroc": float(np.mean([r["auroc"] for r in lag_rows])),
                "mean_auprc": float(np.mean([r["auprc"] for r in lag_rows])),
                "mean_f1": float(np.mean([r["f1"] for r in lag_rows])),
                "mean_shd": float(np.mean([r["shd"] for r in lag_rows])),
                "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in lag_rows])),
            }
        )
    best = max(aggregate_rows, key=lambda r: (r["mean_auroc"], r["mean_auprc"], r["mean_f1"], -r["mean_shd"]))
    write_rows_csv(output_dir / "per_replica_metrics.csv", rows)
    write_rows_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_json(output_dir / "selected_hyperparameters.json", {"selection_rule": "max AUROC, then AUPRC, F1, -SHD", "best": best})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

