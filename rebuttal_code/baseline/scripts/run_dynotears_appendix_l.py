#!/usr/bin/env python
"""Run DYNOTEARS with Appendix L lag grid on NC8, ND8, or FINANCE."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parents[0]
DYNOTEARS_ROOT = BASELINE_ROOT / "DYNOTEAR"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(DYNOTEARS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNOTEARS_ROOT))

from appendix_l_config import DATASET_NAMES, get_dataset_spec, get_params  # noqa: E402
from baseline_common import evaluate_score_matrix, load_replicas, threshold_by_top_k, write_json, write_matrix_csv, write_rows_csv  # noqa: E402
from causalnex.structure.dynotears import from_pandas_dynamic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT / "results" / "DYNOTEARS")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--lambda-w", type=float, default=None)
    parser.add_argument("--lambda-a", type=float, default=None)
    parser.add_argument("--lags", type=int, nargs="+", default=None)
    return parser.parse_args()


def edge_weight_matrix(model, names: list[str], lag: int) -> np.ndarray:
    n = len(names)
    score = np.zeros((n, n), dtype=np.float64)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    for u, v, data in model.edges(data=True):
        if "_lag" not in str(u) or "_lag" not in str(v):
            continue
        u_name, u_lag = str(u).rsplit("_lag", 1)
        v_name, v_lag = str(v).rsplit("_lag", 1)
        if int(v_lag) != 0:
            continue
        source_lag = int(u_lag)
        if source_lag < 1 or source_lag > lag:
            continue
        if u_name in name_to_idx and v_name in name_to_idx:
            score[name_to_idx[v_name], name_to_idx[u_name]] = max(
                score[name_to_idx[v_name], name_to_idx[u_name]],
                abs(float(data.get("weight", 0.0))),
            )
    return score


def main() -> int:
    args = parse_args()
    params = get_params("DYNOTEARS", args.dataset)
    if args.max_iter is not None:
        params["max_iter"] = args.max_iter
    if args.lambda_w is not None:
        params["lambda_w"] = args.lambda_w
    if args.lambda_a is not None:
        params["lambda_a"] = args.lambda_a
    if args.lags is not None:
        params["lag_grid"] = args.lags

    spec = get_dataset_spec(args.dataset)
    output_dir = (args.output_root / args.dataset).resolve()
    predicted_dir = output_dir / "predicted_graphs"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    replicas = load_replicas(args.dataset, Path(spec["data_dir"]), spec["data_glob"], spec["struct_glob"])
    write_json(output_dir / "config.json", {"method": "DYNOTEARS", "dataset": args.dataset, "params": params, "command": " ".join(sys.argv)})

    rows: list[dict] = []
    aggregate: dict[int, list[dict]] = {}
    for replica in replicas:
        names = replica["variable_names"]
        frame = pd.DataFrame(np.asarray(replica["sequence"], dtype=np.float64), columns=names)
        graph = np.asarray(replica["graph"], dtype=np.int64)
        idx = int(replica["replica_index"])
        for lag in params["lag_grid"]:
            start = time.perf_counter()
            model = from_pandas_dynamic(frame, p=int(lag), lambda_w=params["lambda_w"], lambda_a=params["lambda_a"], max_iter=params["max_iter"])
            runtime = time.perf_counter() - start
            score = edge_weight_matrix(model, names, int(lag))
            binary = threshold_by_top_k(score, graph)
            metrics = evaluate_score_matrix(score, graph, binary)
            setting_dir = predicted_dir / f"lag_{lag}"
            prefix = setting_dir / f"replica_{idx}"
            setting_dir.mkdir(parents=True, exist_ok=True)
            np.save(prefix.with_name(prefix.name + "_score_matrix.npy"), score)
            np.save(prefix.with_name(prefix.name + "_binary_graph.npy"), binary)
            write_matrix_csv(prefix.with_name(prefix.name + "_score_matrix.csv"), score, names)
            write_matrix_csv(prefix.with_name(prefix.name + "_binary_graph.csv"), binary, names)
            row = {"method": "DYNOTEARS", "dataset": args.dataset, "replica_index": idx, "lag": int(lag), "runtime_sec": runtime, "auroc": metrics["auroc"], "auprc": metrics["auprc"], "f1": metrics["f1"], "shd": metrics["shd"], "selected_hyperparameters": json.dumps({"lag": int(lag), "max_iter": params["max_iter"], "lambda_w": params["lambda_w"], "lambda_a": params["lambda_a"]}, sort_keys=True)}
            rows.append(row)
            aggregate.setdefault(int(lag), []).append(row)

    aggregate_rows = [{"lag": lag, "replicas": len(vals), "mean_auroc": float(np.mean([r["auroc"] for r in vals])), "mean_auprc": float(np.mean([r["auprc"] for r in vals])), "mean_f1": float(np.mean([r["f1"] for r in vals])), "mean_shd": float(np.mean([r["shd"] for r in vals])), "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in vals]))} for lag, vals in sorted(aggregate.items())]
    best = max(aggregate_rows, key=lambda r: (r["mean_auroc"], r["mean_auprc"], r["mean_f1"], -r["mean_shd"]))
    write_rows_csv(output_dir / "per_replica_metrics.csv", rows)
    write_rows_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_json(output_dir / "selected_hyperparameters.json", {"best": best})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

