#!/usr/bin/env python
"""Run JRNGC with Appendix L hyperparameters on NC8, ND8, or FINANCE."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parents[0]
JRNGC_ROOT = BASELINE_ROOT / "JRNGC"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(JRNGC_ROOT) not in sys.path:
    sys.path.insert(0, str(JRNGC_ROOT))

from appendix_l_config import DATASET_NAMES, get_dataset_spec, get_params  # noqa: E402
from baseline_common import evaluate_score_matrix, load_replicas, threshold_by_top_k, write_json, write_matrix_csv, write_rows_csv  # noqa: E402
from tgc.model import JRNGC  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT / "results" / "JRNGC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=10000, help="Implementation default kept because Appendix L does not specify max_iter.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Implementation default kept because Appendix L does not specify dropout.")
    parser.add_argument("--struct-loss-choice", type=str, default="JF", help="Implementation default from F_var.yaml.")
    parser.add_argument("--jfn", type=int, default=1, help="Implementation default from F_var.yaml.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def standardize_time_major(sequence: np.ndarray) -> np.ndarray:
    x = np.asarray(sequence, dtype=np.float32)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def compute_jrngc_scores(model: JRNGC, x_dt: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(x_dt, device=next(model.parameters()).device)
    if len(x.shape) == 2:
        x.unsqueeze_(0)
    x = x.transpose(1, 2).unfold(1, lag, 1)
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
    lagged = model.jacobian_causal(x).detach().cpu().numpy()
    summary = np.max(lagged, axis=2)
    return lagged, summary


def main() -> int:
    args = parse_args()
    params = get_params("JRNGC", args.dataset)
    dataset_spec = get_dataset_spec(args.dataset)
    output_dir = (args.output_root / args.dataset).resolve()
    predicted_dir = output_dir / "predicted_graphs"
    predicted_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    replicas = load_replicas(
        dataset=args.dataset,
        data_dir=Path(dataset_spec["data_dir"]),
        data_glob=dataset_spec["data_glob"],
        struct_glob=dataset_spec["struct_glob"],
    )
    write_json(
        output_dir / "config.json",
        {
            "method": "JRNGC",
            "dataset": args.dataset,
            "params": params,
            "implementation_defaults_kept": {
                "max_iter": args.max_iter,
                "dropout": args.dropout,
                "struct_loss_choice": args.struct_loss_choice,
                "JFn": args.jfn,
            },
            "dataset_spec": {k: str(v) for k, v in dataset_spec.items()},
            "device": device,
            "command": " ".join(sys.argv),
            "score_definition": "lagged=model.jacobian_causal(...), summary=max over lag axis",
            "binary_definition": "top-k by GT edge count for metric reporting",
        },
    )

    rows: list[dict] = []
    aggregate_by_lambda: dict[float, list[dict]] = {}

    for replica in replicas:
        sequence = standardize_time_major(replica["sequence"])
        graph = np.asarray(replica["graph"], dtype=np.int64)
        variable_names = replica["variable_names"]
        replica_index = int(replica["replica_index"])
        x_dt = sequence.T.astype(np.float32)
        split = max(params["lag"] + 2, int(x_dt.shape[1] * 0.8))
        x_train = x_dt[:, :split]
        x_eval = x_dt[:, split - params["lag"] :] if split < x_dt.shape[1] else x_train

        for lam in params["jacobian_lambda_grid"]:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            start = time.perf_counter()
            model, it_list, total_loss, pred_loss, eval_loss, best_loss, mean_pred_loss, eval_loss_last = JRNGC.from_train(
                max_iter=args.max_iter,
                d=x_dt.shape[0],
                lag=params["lag"],
                layers=params["layers"],
                hidden=params["hidden"],
                dropout=args.dropout,
                jacobian_lam=float(lam),
                struct_loss_choice=args.struct_loss_choice,
                JFn=args.jfn,
                x=x_train,
                x_eval=x_eval,
                lr=params["learning_rate"],
                seed=args.seed,
                device=device,
                verbose=False,
            )
            lagged_scores, score_matrix = compute_jrngc_scores(model, x_dt, params["lag"])
            runtime = time.perf_counter() - start
            binary_graph = threshold_by_top_k(score_matrix, graph)
            metrics = evaluate_score_matrix(score_matrix, graph, binary_graph)

            setting_dir = predicted_dir / f"jacobian_lambda_{lam:g}"
            prefix = setting_dir / f"replica_{replica_index}"
            setting_dir.mkdir(parents=True, exist_ok=True)
            np.save(prefix.with_name(prefix.name + "_weighted_adjacency_lagged.npy"), lagged_scores)
            np.save(prefix.with_name(prefix.name + "_weighted_adjacency_summary.npy"), score_matrix)
            np.save(prefix.with_name(prefix.name + "_binary_graph.npy"), binary_graph)
            np.save(prefix.with_name(prefix.name + "_loss.npy"), {"it": it_list, "total": total_loss, "pred": pred_loss, "eval": eval_loss})
            write_matrix_csv(prefix.with_name(prefix.name + "_score_matrix.csv"), score_matrix, variable_names)
            write_matrix_csv(prefix.with_name(prefix.name + "_binary_graph.csv"), binary_graph, variable_names)

            row = {
                "method": "JRNGC",
                "dataset": args.dataset,
                "replica_index": replica_index,
                "jacobian_lambda": float(lam),
                "runtime_sec": float(runtime),
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "f1": float(metrics["f1"]),
                "shd": float(metrics["shd"]),
                "selected_hyperparameters": json.dumps({"hidden": params["hidden"], "lag": params["lag"], "layers": params["layers"], "lr": params["learning_rate"], "jacobian_lambda": float(lam)}, sort_keys=True),
                "best_loss": float(best_loss.detach().cpu()) if hasattr(best_loss, "detach") else float(best_loss),
                "mean_pred_loss": float(mean_pred_loss.detach().cpu()) if hasattr(mean_pred_loss, "detach") else float(mean_pred_loss),
                "eval_loss": float(eval_loss_last.detach().cpu()) if hasattr(eval_loss_last, "detach") else float(eval_loss_last),
            }
            rows.append(row)
            aggregate_by_lambda.setdefault(float(lam), []).append(row)

    aggregate_rows = []
    for lam, lam_rows in sorted(aggregate_by_lambda.items(), reverse=True):
        aggregate_rows.append(
            {
                "jacobian_lambda": lam,
                "replicas": len(lam_rows),
                "mean_auroc": float(np.mean([r["auroc"] for r in lam_rows])),
                "mean_auprc": float(np.mean([r["auprc"] for r in lam_rows])),
                "mean_f1": float(np.mean([r["f1"] for r in lam_rows])),
                "mean_shd": float(np.mean([r["shd"] for r in lam_rows])),
                "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in lam_rows])),
            }
        )
    best = max(aggregate_rows, key=lambda r: (r["mean_auroc"], r["mean_auprc"], r["mean_f1"], -r["mean_shd"]))
    write_rows_csv(output_dir / "per_replica_metrics.csv", rows)
    write_rows_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_json(output_dir / "selected_hyperparameters.json", {"selection_rule": "max AUROC, then AUPRC, F1, -SHD", "best": best})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

