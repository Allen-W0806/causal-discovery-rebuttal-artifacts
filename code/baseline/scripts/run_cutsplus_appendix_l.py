#!/usr/bin/env python
"""Run CUTS+ with Appendix L hyperparameters on NC8, ND8, or FINANCE."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_ROOT = SCRIPT_DIR.parents[0]
CUTS_PLUS_ROOT = BASELINE_ROOT / "CUTS+" / "CUTS_Plus"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CUTS_PLUS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_PLUS_ROOT))

from appendix_l_config import DATASET_NAMES, get_dataset_spec, get_params  # noqa: E402
from baseline_common import evaluate_score_matrix, load_replicas, threshold_by_top_k, write_json, write_matrix_csv, write_rows_csv  # noqa: E402


class NullLogger:
    """Small logger compatible with CUTS+ training without TensorBoard output."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = str(log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.tblogger = self

    def log_metrics(self, metrics_dict, iters) -> None:
        return None

    def log_figures(self, figure, name="figure.png", iters=None, exclude_logger=None) -> None:
        return None

    def add_figure(self, tag, figure, global_step=None) -> None:
        return None

    def log_npz(self, name, data, iters=None) -> None:
        path = Path(self.log_dir) / f"{name}_{iters if iters is not None else 'latest'}.npz"
        np.savez(path, **data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--output-root", type=Path, default=BASELINE_ROOT / "results" / "CUTSplus")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=128, help="Implementation default from CUTS+ example YAML.")
    parser.add_argument("--input-step", type=int, default=1, help="Implementation default from CUTS+ example YAML.")
    parser.add_argument("--mlp-hid", type=int, default=32, help="Implementation default from CUTS+ example YAML.")
    parser.add_argument("--gru-layers", type=int, default=1, help="Implementation default from CUTS+ example YAML.")
    parser.add_argument("--seed", type=int, default=42, help="Implementation default from CUTS+ example YAML.")
    return parser.parse_args()


def build_opt(args: argparse.Namespace, params: dict, n_nodes: int, lam: float) -> SimpleNamespace:
    n_groups = min(int(params["max_groups"]), int(n_nodes))
    # "multiply_1_every_999" is a parseable no-op policy: group_mul=1, group_every=999.
    # When n_groups == n_nodes, the < condition is always False so no splitting fires,
    # but the elif epoch_i==0 branch still initialises self.GT correctly.
    graph_policy = "multiply_1_every_999" if n_groups == n_nodes else "multiply_2_every_20"
    return SimpleNamespace(
        n_nodes=int(n_nodes),
        input_step=int(args.input_step),
        window_step=1,
        stride=1,
        batch_size=int(args.batch_size),
        sample_per_epoch=1,
        data_dim=1,
        total_epoch=int(params["epochs"]),
        patience=0,
        warmup=None,
        show_graph_every=int(params["epochs"]) + 1,
        val_every=int(params["epochs"]) + 1,
        n_groups=n_groups,
        group_policy=graph_policy,
        causal_thres="value_0.5",
        supervision_policy="full",
        fill_policy="None",
        data_pred=SimpleNamespace(
            model="multi_lstm",
            pred_step=1,
            mlp_hid=int(args.mlp_hid),
            gru_layers=int(args.gru_layers),
            shared_weights_decoder=False,
            concat_h=True,
            lr_data_start=float(params["learning_rate"]),
            lr_data_end=float(params["learning_rate"]),
            weight_decay=0,
            prob=True,
        ),
        graph_discov=SimpleNamespace(
            lambda_s_start=float(lam),
            lambda_s_end=float(lam),
            lr_graph_start=float(params["learning_rate"]),
            lr_graph_end=float(params["learning_rate"]),
            start_tau=1.0,
            end_tau=0.1,
            dynamic_sampling_milestones=[0],
            dynamic_sampling_periods=[1],
        ),
    )


def standardize(sequence: np.ndarray) -> np.ndarray:
    x = np.asarray(sequence, dtype=np.float64)
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return ((x - mean) / std).astype(np.float32)


def main() -> int:
    args = parse_args()
    try:
        from cuts_plus import main as run_cuts_plus  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "CUTS+ baseline requires optional dependencies (including tensorboard). "
            "Install from code/requirements.txt, then rerun."
        ) from exc

    params = get_params("CUTS+", args.dataset)
    if args.lr is not None:
        params["learning_rate"] = args.lr
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.max_groups is not None:
        params["max_groups"] = args.max_groups
    if args.lambda_grid is not None:
        params["lambda_grid"] = args.lambda_grid

    spec = get_dataset_spec(args.dataset)
    replicas = load_replicas(args.dataset, Path(spec["data_dir"]), spec["data_glob"], spec["struct_glob"])
    output_dir = (args.output_root / args.dataset).resolve()
    predicted_dir = output_dir / "predicted_graphs"
    predicted_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.gpu == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")

    write_json(
        output_dir / "config.json",
        {
            "method": "CUTS+",
            "dataset": args.dataset,
            "params": params,
            "implementation_defaults_kept": {
                "batch_size": args.batch_size,
                "input_step": args.input_step,
                "mlp_hid": args.mlp_hid,
                "gru_layers": args.gru_layers,
                "seed": args.seed,
                "supervision_policy": "full",
                "fill_policy": "None",
            },
            "dataset_spec": {k: str(v) for k, v in spec.items()},
            "device": device,
            "command": " ".join(sys.argv),
            "score_definition": "raw Graph returned by CUTS_Plus/cuts_plus.py::main",
            "binary_definition": "top-k by GT edge count for metric reporting",
            "max_groups_note": "Appendix L max_groups=32 is used as a cap; actual n_groups=min(32,n_nodes) for 8-variable datasets.",
        },
    )

    rows: list[dict] = []
    aggregate_by_lambda: dict[float, list[dict]] = {}
    for lam in params["lambda_grid"]:
        for replica in replicas:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            idx = int(replica["replica_index"])
            sequence = standardize(replica["sequence"])
            graph = np.asarray(replica["graph"], dtype=np.int64)
            names = replica["variable_names"]
            mask = np.ones_like(sequence, dtype=np.float32)
            setting_dir = predicted_dir / f"lambda_{lam:g}"
            log_dir = setting_dir / f"replica_{idx}_logs"
            opt = build_opt(args, params, sequence.shape[1], float(lam))
            logger = NullLogger(log_dir)

            start = time.perf_counter()
            score_matrix = np.asarray(run_cuts_plus(sequence, mask, None, opt, logger, device=device), dtype=np.float64)
            runtime = time.perf_counter() - start
            if score_matrix.shape != graph.shape:
                score_matrix = np.asarray(score_matrix).reshape(graph.shape)
            binary = threshold_by_top_k(score_matrix, graph)
            metrics = evaluate_score_matrix(score_matrix, graph, binary)

            prefix = setting_dir / f"replica_{idx}"
            setting_dir.mkdir(parents=True, exist_ok=True)
            np.save(prefix.with_name(prefix.name + "_score_matrix.npy"), score_matrix)
            np.save(prefix.with_name(prefix.name + "_binary_graph.npy"), binary)
            write_matrix_csv(prefix.with_name(prefix.name + "_score_matrix.csv"), score_matrix, names)
            write_matrix_csv(prefix.with_name(prefix.name + "_binary_graph.csv"), binary, names)

            row = {
                "method": "CUTS+",
                "dataset": args.dataset,
                "replica_index": idx,
                "lambda": float(lam),
                "runtime_sec": float(runtime),
                "auroc": float(metrics["auroc"]),
                "auprc": float(metrics["auprc"]),
                "f1": float(metrics["f1"]),
                "shd": float(metrics["shd"]),
                "selected_hyperparameters": json.dumps(
                    {
                        "learning_rate": params["learning_rate"],
                        "epochs": params["epochs"],
                        "max_groups": params["max_groups"],
                        "actual_n_groups": opt.n_groups,
                        "lambda": float(lam),
                    },
                    sort_keys=True,
                ),
            }
            rows.append(row)
            aggregate_by_lambda.setdefault(float(lam), []).append(row)

    aggregate_rows = []
    for lam, vals in aggregate_by_lambda.items():
        aggregate_rows.append(
            {
                "lambda": lam,
                "replicas": len(vals),
                "mean_auroc": float(np.mean([r["auroc"] for r in vals])),
                "mean_auprc": float(np.mean([r["auprc"] for r in vals])),
                "mean_f1": float(np.mean([r["f1"] for r in vals])),
                "mean_shd": float(np.mean([r["shd"] for r in vals])),
                "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in vals])),
            }
        )
    best = max(aggregate_rows, key=lambda r: (r["mean_auroc"], r["mean_auprc"], r["mean_f1"], -r["mean_shd"]))
    write_rows_csv(output_dir / "per_replica_metrics.csv", rows)
    write_rows_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_json(output_dir / "selected_hyperparameters.json", {"selection_rule": "max AUROC, then AUPRC, F1, -SHD", "best": best})
    print(json.dumps({"output_dir": str(output_dir), "best": best}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
