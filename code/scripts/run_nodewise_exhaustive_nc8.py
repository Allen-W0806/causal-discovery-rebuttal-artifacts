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


def _write_matrix_csv(path: str, matrix: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [str(i) for i in range(matrix.shape[1])])
        for i, row in enumerate(matrix):
            writer.writerow([str(i)] + [float(x) for x in row])


def _write_metrics_csv(path: str, row: dict) -> None:
    keys = [
        "method",
        "dataset",
        "replica",
        "seed",
        "lag",
        "search_type",
        "runtime",
        "node_mask_evals",
        "meta_update_count",
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


def _append_jsonl(path: str, payload: dict) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _mask_to_parents(mask: int, d: int) -> list[int]:
    return [idx for idx in range(d) if (int(mask) >> idx) & 1]


def _child_graph(d: int, child: int, parents: list[int]) -> np.ndarray:
    graph = np.zeros((d, d), dtype=int)
    graph[[int(p) for p in parents if 0 <= int(p) < d], int(child)] = 1
    return graph


class ExhaustiveMetaController:
    def __init__(self, scorer: NonlinearMultiLagScorer, args: argparse.Namespace, outdir: str):
        self.scorer = scorer
        self.args = args
        self.enabled = hasattr(scorer, "meta_update") and str(args.meta_mode) != "freeze_meta"
        self.eval_count = 0
        self.update_count = 0
        self.history: list[dict] = []
        self.rng = np.random.default_rng(int(args.seed))
        self.log_path = args.meta_log_path.strip() or os.path.join(outdir, "nodewise_exhaustive_meta_updates.jsonl")

    def _history_pool(self) -> list[dict]:
        if str(self.args.meta_mode) == "elite_history":
            scored = [r for r in self.history if np.isfinite(float(r["score"]))]
            if scored:
                k = max(1, int(np.ceil(len(scored) * float(self.args.meta_elite_frac))))
                return sorted(scored, key=lambda r: float(r["score"]), reverse=True)[:k]
        if str(self.args.meta_mode) == "random_history":
            return list(self.history)
        return self.history[-int(self.args.meta_recent_window):]

    def _sample_graphs(self, records: list[dict]) -> list[np.ndarray]:
        if len(records) == 0:
            return []
        k = max(1, int(self.args.meta_batch_size_J) if int(self.args.meta_batch_size_J) > 0 else int(self.args.meta_batch_size))
        replace = len(records) < k
        idx = self.rng.choice(len(records), size=k, replace=replace)
        return [np.array(records[int(i)]["graph"], copy=True) for i in np.asarray(idx).tolist()]

    def record(self, graph: np.ndarray, score: float) -> None:
        self.eval_count += 1
        if self.enabled and self.eval_count % max(1, int(self.args.meta_update_every)) == 0:
            pool = self._history_pool()
            graphs_history = self._sample_graphs(pool)
            if graphs_history:
                meta_info = self.scorer.meta_update(
                    graphs_history=graphs_history,
                    inner_step_count=int(self.args.meta_inner_step_count),
                    inner_lr=float(self.args.meta_inner_lr),
                    outer_lr=float(self.args.meta_outer_lr),
                )
                self.update_count += 1
                _append_jsonl(
                    self.log_path,
                    {
                        "eval_count": int(self.eval_count),
                        "meta_update": int(self.update_count),
                        "meta_mode": str(self.args.meta_mode),
                        "history_pool_size": int(len(pool)),
                        "history_graphs": int(len(graphs_history)),
                        "meta_info": meta_info,
                    },
                )
            elif hasattr(self.scorer, "clear_adaptation_buffer"):
                self.scorer.clear_adaptation_buffer()
        self.history.append({"graph": np.array(graph, copy=True), "score": float(score)})

    def flush(self) -> None:
        if not self.enabled:
            return
        graphs_history = self._sample_graphs(self._history_pool())
        if not graphs_history:
            return
        meta_info = self.scorer.meta_update(
            graphs_history=graphs_history,
            inner_step_count=int(self.args.meta_inner_step_count),
            inner_lr=float(self.args.meta_inner_lr),
            outer_lr=float(self.args.meta_outer_lr),
        )
        self.update_count += 1
        _append_jsonl(
            self.log_path,
            {
                "eval_count": int(self.eval_count),
                "meta_update": int(self.update_count),
                "meta_mode": str(self.args.meta_mode),
                "history_graphs": int(len(graphs_history)),
                "flush": True,
                "meta_info": meta_info,
            },
        )


def _local_score(
    scorer: NonlinearMultiLagScorer,
    child: int,
    parents: list[int],
    d: int,
    meta_controller: ExhaustiveMetaController | None,
) -> float:
    graph = _child_graph(d, child, parents)
    _total, local_log_mse = scorer.eval_graph_return_local(graph)
    mse = float(np.exp(local_log_mse[int(child)]))
    offdiag_parent_count = sum(1 for p in parents if int(p) != int(child))
    n = int(getattr(scorer, "n", 1))
    penalty = float(getattr(scorer, "lambda_sparse", 1.0)) * offdiag_parent_count * np.log(max(n, 2))
    score = -float(n) * np.log(max(mse, 1e-12)) - penalty
    if meta_controller is not None:
        meta_controller.record(graph, score)
    return float(score)


def exhaustive_nodewise_search(
    scorer: NonlinearMultiLagScorer,
    d: int,
    meta_controller: ExhaustiveMetaController,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_masks = 2 ** int(d)
    selected_masks = np.zeros((d, d), dtype=np.int32)
    local_scores = np.zeros((d, n_masks), dtype=np.float64)
    graph_raw = np.zeros((d, d), dtype=np.int32)

    for child in range(d):
        best_mask = 0
        best_score = -np.inf
        for mask in range(n_masks):
            parents = _mask_to_parents(mask, d)
            score = _local_score(scorer, child, parents, d, meta_controller)
            local_scores[child, mask] = score
            if score > best_score:
                best_score = score
                best_mask = mask
        parents = _mask_to_parents(best_mask, d)
        selected_masks[parents, child] = 1
        graph_raw[parents, child] = 1

    graph_pred = np.array(graph_raw, copy=True)
    np.fill_diagonal(graph_pred, 0)
    return graph_pred, selected_masks, local_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NC8 exhaustive node-wise parent-mask enumeration")
    parser.add_argument("--dataset", type=str, default="NC8", choices=["NC8"])
    parser.add_argument("--data_dir", type=str, default=str(Path("data") / "NC8"))
    parser.add_argument("--outdir", type=str, default="review_runs/nodewise_exhaustive_nc8")
    parser.add_argument("--replica", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lag", type=int, default=16)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--mlp_hidden", type=int, nargs="+", default=[64])
    parser.add_argument("--mlp_max_iter", type=int, default=50)
    parser.add_argument("--pretrain_epochs", type=int, default=0)
    parser.add_argument("--pretrain_avg_parents", type=float, default=3.0)
    parser.add_argument("--inner_step_count", type=int, default=10)
    parser.add_argument("--inner_lr", type=float, default=1e-2)
    parser.add_argument("--inner_batch_size", type=int, default=0)
    parser.add_argument("--inner_optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lambda_sparse", type=float, default=8.0)
    parser.add_argument("--score_type", type=str, default="mlp", choices=["mlp"])
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=False)
    parser.add_argument("--meta_update_every", type=int, default=10)
    parser.add_argument("--meta_history_size", type=int, default=256)
    parser.add_argument("--meta_batch_size", type=int, default=8)
    parser.add_argument("--meta_batch_size_J", type=int, default=0)
    parser.add_argument("--meta_history_only", action="store_true", default=True)
    parser.add_argument("--no_meta_history_only", dest="meta_history_only", action="store_false")
    parser.add_argument("--meta_recent_window", type=int, default=20)
    parser.add_argument("--mix_current_frac", type=float, default=0.0)
    parser.add_argument("--meta_fallback_strategy", type=str, default="replacement_then_hist")
    parser.add_argument("--meta_inner_step_count", type=int, default=1)
    parser.add_argument("--meta_inner_lr", type=float, default=0.001)
    parser.add_argument("--meta_outer_lr", type=float, default=0.0001)
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
    if d != 8:
        raise ValueError(f"NC8 exhaustive comparator expects d=8 for 2^d=256 masks, got d={d}")

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
        inner_optimizer=args.inner_optimizer,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_avg_parents=args.pretrain_avg_parents,
        seed=args.seed,
    )

    meta_controller = ExhaustiveMetaController(scorer, args, args.outdir)
    start = time.perf_counter()
    graph, selected_masks, local_scores = exhaustive_nodewise_search(
        scorer=scorer,
        d=d,
        meta_controller=meta_controller,
    )
    meta_controller.flush()
    runtime_sec = time.perf_counter() - start

    perturbation = compute_parent_loo_perturbation(scorer, graph, args.outdir, GT=GT)
    np.fill_diagonal(perturbation, 0.0)
    metrics = compute_graph_metrics(GT, graph)
    weighted = evaluate_uncle_style(perturbation, GT, ignore_diag=True, run_dir=args.outdir)
    print_comparison_table(weighted, dataset_name="NC8 node-wise exhaustive")

    np.save(os.path.join(args.outdir, "G_pred.npy"), graph)
    np.save(os.path.join(args.outdir, "best_graph.npy"), graph)
    np.save(os.path.join(args.outdir, "GT.npy"), GT)
    np.save(os.path.join(args.outdir, "ground_truth.npy"), GT)
    np.save(os.path.join(args.outdir, "selected_parent_masks.npy"), selected_masks)
    np.save(os.path.join(args.outdir, "all_local_scores.npy"), local_scores)
    np.save(os.path.join(args.outdir, "A_raw.npy"), perturbation)
    np.save(os.path.join(args.outdir, "score_matrix.npy"), perturbation)
    _write_matrix_csv(os.path.join(args.outdir, "G_pred.csv"), graph)
    _write_matrix_csv(os.path.join(args.outdir, "GT.csv"), GT)
    _write_matrix_csv(os.path.join(args.outdir, "score_matrix.csv"), perturbation)

    flat_metrics = {
        "method": "nodewise_exhaustive",
        "dataset": args.dataset,
        "replica": int(args.replica),
        "seed": int(args.seed),
        "lag": int(args.lag),
        "search_type": "nodewise_exhaustive",
        "runtime": float(runtime_sec),
        "node_mask_evals": int(d * (2 ** d)),
        "meta_update_count": int(meta_controller.update_count),
        "auroc": weighted.get("AUROC"),
        "auprc": weighted.get("AUPRC"),
        "f1": metrics.get("F1", metrics.get("f1")),
        "shd": metrics.get("SHD", metrics.get("shd")),
        "weighted_f1": weighted.get("F1"),
        "weighted_shd": weighted.get("SHD"),
    }
    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["search_type"] = "nodewise_exhaustive"
    config["node_masks_per_target"] = int(2 ** d)
    config["total_node_mask_evals"] = int(d * (2 ** d))
    config["bo_only_parameters"] = {
        "n_cands": "N/A",
        "ts_rank": "N/A",
        "low_rank_k": "N/A",
        "candidate_pool_size": "N/A",
        "bo_eval_budget": "N/A",
    }
    results = {
        "method": "nodewise_exhaustive",
        "dataset": args.dataset,
        "replica": int(args.replica),
        "seed": int(args.seed),
        "lag": int(args.lag),
        "runtime_sec": float(runtime_sec),
        "runtime": float(runtime_sec),
        "search_type": "nodewise_exhaustive",
        "node_masks_per_target": int(2 ** d),
        "node_mask_evals": int(d * (2 ** d)),
        "meta_update_count": int(meta_controller.update_count),
        "auroc": flat_metrics["auroc"],
        "auprc": flat_metrics["auprc"],
        "f1": flat_metrics["f1"],
        "shd": flat_metrics["shd"],
        "weighted_f1": flat_metrics["weighted_f1"],
        "weighted_shd": flat_metrics["weighted_shd"],
        "metrics": metrics,
        "weighted_metrics": weighted,
        "config": config,
    }
    with open(os.path.join(args.outdir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(flat_metrics, f, indent=2)
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    _write_metrics_csv(os.path.join(args.outdir, "metrics.csv"), flat_metrics)
    with open(os.path.join(args.outdir, "runtime.txt"), "w", encoding="utf-8") as f:
        f.write(f"{runtime_sec:.6f}\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
