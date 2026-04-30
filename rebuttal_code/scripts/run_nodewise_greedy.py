#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bo.data.nc8_loader import load_nc8_data
from bo.eval.metrics import compute_graph_metrics, evaluate_uncle_style, print_comparison_table
from bo.eval.perturbation import compute_parent_loo_perturbation
from bo.score.bic_mlp import NonlinearMultiLagScorer


def _load_dataset(name: str, data_dir: str, replica: int):
    if name == "NC8":
        X, GT, var_names = load_nc8_data(data_dir, replica=replica)
        return X, GT, var_names
    raise ValueError(f"Unsupported dataset={name!r}")


def _local_score(scorer: NonlinearMultiLagScorer, child: int, parents: list[int]) -> float:
    mse = float(scorer._fit_node(child, parents))
    n = int(getattr(scorer, "n", 1))
    penalty = float(getattr(scorer, "lambda_sparse", 1.0)) * len(parents) * np.log(max(n, 2))
    return -n * np.log(max(mse, 1e-12)) - penalty


def greedy_parent_search(scorer: NonlinearMultiLagScorer, d: int, tol: float) -> np.ndarray:
    graph = np.zeros((d, d), dtype=int)
    for child in range(d):
        parents: list[int] = []
        current = _local_score(scorer, child, parents)
        improved = True
        while improved:
            improved = False
            best_parent = None
            best_score = current
            for parent in range(d):
                if parent == child or parent in parents:
                    continue
                trial = parents + [parent]
                score = _local_score(scorer, child, trial)
                if score > best_score + tol:
                    best_score = score
                    best_parent = parent
            if best_parent is not None:
                parents.append(best_parent)
                current = best_score
                improved = True
        graph[parents, child] = 1
    np.fill_diagonal(graph, 0)
    return graph


def _write_matrix_csv(path: str, matrix: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [str(i) for i in range(matrix.shape[1])])
        for i, row in enumerate(matrix):
            writer.writerow([str(i)] + [float(x) for x in row])


def _write_metrics_csv(path: str, row: dict) -> None:
    keys = ["method", "dataset", "replica", "seed", "lag", "runtime", "auroc", "auprc", "f1", "shd"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in keys})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run node-wise greedy parent search with the masked MLP score")
    parser.add_argument("--dataset", type=str, default="NC8", choices=["NC8"])
    parser.add_argument("--data_dir", type=str, default=str(Path("data") / "NC8"))
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--d", type=int, default=0)
    parser.add_argument("--lag", type=int, default=16)
    parser.add_argument("--ts_rank", type=int, default=8)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--lambda_sparse", type=float, default=1.0)
    parser.add_argument("--score_type", type=str, default="mlp", choices=["mlp"])
    parser.add_argument("--mlp_hidden", type=int, nargs="+", default=[64])
    parser.add_argument("--mlp_max_iter", type=int, default=50)
    parser.add_argument("--inner_step_count", type=int, default=10)
    parser.add_argument("--inner_lr", type=float, default=1e-2)
    parser.add_argument("--inner_batch_size", type=int, default=0)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--pretrain_avg_parents", type=float, default=3.0)
    parser.add_argument("--eval", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--n_cands", type=int, default=10000)
    parser.add_argument("--n_grads", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=False)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--outdir", type=str, default="review_runs/nodewise_nc8")
    args = parser.parse_args()

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    X, GT, var_names = _load_dataset(args.dataset, args.data_dir, args.replica)
    if args.T is not None:
        X = X[: int(args.T)]
    if args.d and args.d > 0:
        d = int(args.d)
        X = X[:, :d]
        GT = GT[:d, :d]
        var_names = var_names[:d]
    else:
        d = int(X.shape[1])

    os.makedirs(args.outdir, exist_ok=True)
    scorer = NonlinearMultiLagScorer(
        X,
        lag=args.lag,
        hidden_sizes=tuple(args.mlp_hidden),
        max_iter=args.mlp_max_iter,
        lambda_sparse=args.lambda_sparse,
        device=device,
        inner_step_count=args.inner_step_count,
        inner_lr=args.inner_lr,
        inner_batch_size=args.inner_batch_size,
        inner_optimizer="adam",
        pretrain_epochs=args.pretrain_epochs,
        pretrain_avg_parents=args.pretrain_avg_parents,
        seed=args.seed,
    )

    start = time.perf_counter()
    graph = greedy_parent_search(scorer, d=d, tol=args.tol)
    runtime_sec = time.perf_counter() - start
    np.save(os.path.join(args.outdir, "best_graph.npy"), graph)
    np.save(os.path.join(args.outdir, "predicted_adjacency.npy"), graph)
    np.save(os.path.join(args.outdir, "ground_truth.npy"), GT)
    _write_matrix_csv(os.path.join(args.outdir, "predicted_adjacency.csv"), graph)
    _write_matrix_csv(os.path.join(args.outdir, "ground_truth.csv"), GT)

    perturbation = compute_parent_loo_perturbation(scorer, graph, args.outdir, GT=GT)
    np.fill_diagonal(perturbation, 0.0)
    metrics = compute_graph_metrics(GT, graph)
    weighted = evaluate_uncle_style(perturbation, GT, ignore_diag=True, run_dir=args.outdir)
    print_comparison_table(weighted, dataset_name=f"{args.dataset} node-wise greedy")

    flat_metrics = {
        "method": "nodewise_greedy",
        "dataset": args.dataset,
        "replica": int(args.replica),
        "seed": int(args.seed),
        "lag": int(args.lag),
        "runtime": float(runtime_sec),
        "auroc": weighted.get("AUROC"),
        "auprc": weighted.get("AUPRC"),
        "f1": metrics.get("F1", metrics.get("f1")),
        "shd": metrics.get("SHD", metrics.get("shd")),
    }
    results = {
        "method": "nodewise_greedy",
        "dataset": args.dataset,
        "replica": int(args.replica),
        "seed": int(args.seed),
        "lag": int(args.lag),
        "runtime_sec": float(runtime_sec),
        "runtime": float(runtime_sec),
        "auroc": flat_metrics["auroc"],
        "auprc": flat_metrics["auprc"],
        "f1": flat_metrics["f1"],
        "shd": flat_metrics["shd"],
        "metrics": metrics,
        "weighted_metrics": weighted,
        "loaded_shape": list(X.shape),
        "config": vars(args),
    }
    with open(os.path.join(args.outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(flat_metrics, f, indent=2)
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    _write_metrics_csv(os.path.join(args.outdir, "metrics.csv"), flat_metrics)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
