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
from bo.optim.bo_loop import run_bo
from bo.score.bic_mlp import NonlinearMultiLagScorer

LOGGING_DIR = REPO_ROOT / "logging"
sys.path.insert(0, str(LOGGING_DIR))
from run_logger import RunLogger




def _default_exp_name(args, max_evals: int) -> str:
    lambda_tag = f"{float(args.lambda_sparse):g}"
    return (
        f"rep{args.replica}_lag{args.lag}_eval{max_evals}_"
        f"rank{args.ts_rank}_tau{args.tau}_lambda{lambda_tag}_seed{args.seed}_"
        f"score{args.score_type}"
    )


def _jsonable_score(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr.astype(float).tolist()


def _write_metrics_csv(path: str, row: dict) -> None:
    keys = [
        "method",
        "dataset",
        "replica",
        "seed",
        "lag",
        "runtime",
        "n_evals",
        "auroc",
        "auprc",
        "f1",
        "shd",
        "weighted_f1",
        "weighted_shd",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in keys})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run low-rank BO causal graph search on NC8")
    parser.add_argument("--dataset", type=str, default="NC8", choices=["NC8"])
    parser.add_argument("--data_dir", type=str, default=str(Path("data") / "NC8"))
    parser.add_argument("--replica", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--lag", type=int, default=16)
    parser.add_argument("--ts_rank", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--mlp_hidden", type=int, nargs="+", default=[64])
    parser.add_argument("--mlp_max_iter", type=int, default=50)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--pretrain_avg_parents", type=float, default=3.0)
    parser.add_argument("--inner_step_count", type=int, default=10)
    parser.add_argument("--inner_lr", type=float, default=1e-2)
    parser.add_argument("--inner_batch_size", type=int, default=0)
    parser.add_argument("--inner_optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--d", type=int, default=0)
    parser.add_argument("--lambda_sparse", type=float, default=1.0)
    parser.add_argument("--score_type", type=str, default="mlp", choices=["mlp"])
    parser.add_argument("--n_iters", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_cands", type=int, default=10000)
    parser.add_argument("--n_grads", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_replay", type=int, default=1024)
    parser.add_argument("--max_evals", type=int, default=0)
    parser.add_argument("--eval", type=int, default=0)
    parser.add_argument("--mc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="review_runs/NC8")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--meta_update_every", type=int, default=5)
    parser.add_argument("--meta_history_size", type=int, default=256)
    parser.add_argument("--meta_batch_size", type=int, default=4)
    parser.add_argument("--meta_batch_size_J", type=int, default=0)
    parser.add_argument("--meta_history_only", action="store_true", default=True)
    parser.add_argument("--no_meta_history_only", dest="meta_history_only", action="store_false")
    parser.add_argument("--meta_recent_window", type=int, default=10)
    parser.add_argument("--mix_current_frac", type=float, default=0.0)
    parser.add_argument("--meta_fallback_strategy", type=str, default="replacement_then_hist")
    parser.add_argument("--meta_inner_step_count", type=int, default=1)
    parser.add_argument("--meta_inner_lr", type=float, default=5e-4)
    parser.add_argument("--meta_outer_lr", type=float, default=3e-5)
    parser.add_argument("--meta_warmup_frac", type=float, default=0.0)
    parser.add_argument(
        "--meta_mode",
        type=str,
        default="recent_topB_replay",
        choices=["recent_topB_replay", "random_history", "elite_history", "freeze_meta"],
    )
    parser.add_argument("--meta_elite_frac", type=float, default=0.2)
    parser.add_argument("--meta_recent_size", type=int, default=128)
    parser.add_argument("--meta_log_path", type=str, default="")
    parser.add_argument("--meta_history_snapshot", action="store_true", default=True)
    parser.add_argument("--no_meta_history_snapshot", dest="meta_history_snapshot", action="store_false")
    args = parser.parse_args()

    max_evals = int(args.max_evals) if int(args.max_evals) > 0 else int(args.eval)
    if max_evals <= 0:
        max_evals = int(args.n_iters * args.batch_size)
    if int(args.batch_size) <= 0:
        raise ValueError("--batch_size must be positive")
    args.n_iters = int(np.ceil(max_evals / int(args.batch_size)))

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    X, GT, var_names = load_nc8_data(args.data_dir, replica=args.replica)
    if args.T is not None:
        X = X[: int(args.T)]
    if args.d and args.d > 0:
        d = int(args.d)
        X = X[:, :d]
        GT = GT[:d, :d]
        var_names = var_names[:d]
    else:
        d = int(X.shape[1])

    exp_name = args.exp_name.strip() or _default_exp_name(args, max_evals)
    run_dir = os.path.join(args.logdir, exp_name)
    os.makedirs(run_dir, exist_ok=True)

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
        inner_optimizer=args.inner_optimizer,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_avg_parents=args.pretrain_avg_parents,
        seed=args.seed,
    )

    logger = RunLogger(run_dir)
    start = time.perf_counter()
    bo_result = run_bo(
        X=X,
        GT=GT,
        score_method="BIC",
        score_params={},
        max_evals=max_evals,
        batch_size=args.batch_size,
        scorer=scorer,
        device=device,
        random_state=args.seed,
        ts_rank=args.ts_rank,
        tau=args.tau,
        lambda_sparse=args.lambda_sparse,
        n_cands=args.n_cands,
        n_grads=args.n_grads,
        lr=args.lr,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        n_replay=args.n_replay,
        return_model=True,
        run_logger=logger,
        meta_update_every=args.meta_update_every,
        meta_history_size=args.meta_history_size,
        meta_batch_size=args.meta_batch_size_J if args.meta_batch_size_J > 0 else args.meta_batch_size,
        meta_history_only=args.meta_history_only,
        meta_recent_window=args.meta_recent_window,
        mix_current_frac=args.mix_current_frac,
        meta_fallback_strategy=args.meta_fallback_strategy,
        meta_inner_step_count=args.meta_inner_step_count,
        meta_inner_lr=args.meta_inner_lr,
        meta_outer_lr=args.meta_outer_lr,
        meta_warmup_frac=args.meta_warmup_frac,
        meta_mode_mode=args.meta_mode,
        meta_elite_frac=args.meta_elite_frac,
        meta_recent_size=args.meta_recent_size,
        meta_log_path=args.meta_log_path or None,
        meta_history_snapshot=args.meta_history_snapshot,
    )
    runtime_sec = time.perf_counter() - start

    best_graph = np.asarray(bo_result["best_adj"], dtype=int)
    np.fill_diagonal(best_graph, 0)
    np.save(os.path.join(run_dir, "best_graph.npy"), best_graph)

    perturbation = compute_parent_loo_perturbation(scorer, best_graph, run_dir, GT=GT)
    np.fill_diagonal(perturbation, 0.0)

    metrics = compute_graph_metrics(GT, best_graph)
    weighted = evaluate_uncle_style(perturbation, GT, ignore_diag=True, run_dir=run_dir)
    print_comparison_table(weighted, dataset_name="NC8")

    flat_metrics = {
        "method": "low_rank_bo",
        "dataset": args.dataset,
        "replica": int(args.replica),
        "seed": int(args.seed),
        "lag": int(args.lag),
        "runtime": float(runtime_sec),
        "n_evals": int(max_evals),
        "auroc": weighted.get("AUROC"),
        "auprc": weighted.get("AUPRC"),
        "f1": metrics.get("F1", metrics.get("f1")),
        "shd": metrics.get("SHD", metrics.get("shd")),
        "weighted_f1": weighted.get("F1"),
        "weighted_shd": weighted.get("SHD"),
    }
    results = {
        "method": "low_rank_bo",
        "dataset": "NC8",
        "replica": int(args.replica),
        "seed": int(args.seed),
        "runtime_sec": float(runtime_sec),
        "runtime": float(runtime_sec),
        "best_score": None if bo_result.get("history_best_total_score") is None else float(bo_result["history_best_total_score"]),
        "score_S": _jsonable_score(bo_result.get("score_S")),
        "n_evals": int(max_evals),
        "auroc": flat_metrics["auroc"],
        "auprc": flat_metrics["auprc"],
        "f1": flat_metrics["f1"],
        "shd": flat_metrics["shd"],
        "weighted_f1": flat_metrics["weighted_f1"],
        "weighted_shd": flat_metrics["weighted_shd"],
        "best_consistency": bo_result.get("best_consistency"),
        "metrics": metrics,
        "weighted_metrics": weighted,
        "config": vars(args),
    }
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(flat_metrics, f, indent=2)
    _write_metrics_csv(os.path.join(run_dir, "metrics.csv"), flat_metrics)

    print(f"Saved run to: {run_dir}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
