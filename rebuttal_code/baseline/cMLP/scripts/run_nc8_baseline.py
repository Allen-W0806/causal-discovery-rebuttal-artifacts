#!/usr/bin/env python
"""Run a reproducible cMLP baseline on all NC8 replicas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cmlp import cMLP, train_model_ista  # noqa: E402


DEFAULT_DATA_DIR = Path("/storage/home/ydk297/projects/meta_causal_discovery/data/nc8")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results_nc8_baseline"
T_CRIT_95_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--hidden", type=int, nargs="+", default=[20])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--penalty", type=str, default="H")
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--lam-ridge", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def _is_numeric_row(tokens: list[str]) -> bool:
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def _is_integer_index_header(tokens: list[str]) -> bool:
    """Detect 0,1,...,n-1 style numeric headers (e.g. Finance CSVs)."""
    try:
        vals = [int(float(t.strip())) for t in tokens]
        return vals == list(range(len(vals)))
    except (ValueError, AttributeError):
        return False


def read_csv_matrix(path: Path) -> tuple[list[str] | None, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().strip()

    if not first_line:
        raise ValueError(f"CSV file is empty: {path}")

    first_tokens = [token.strip() for token in first_line.split(",")]
    has_header = not _is_numeric_row(first_tokens) or _is_integer_index_header(first_tokens)
    header = first_tokens if has_header else None
    skiprows = 1 if has_header else 0

    matrix = np.loadtxt(path, delimiter=",", skiprows=skiprows, dtype=np.float64)
    matrix = np.atleast_2d(matrix)
    return header, matrix


def discover_nc8_replicas(data_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    for data_pat, struct_pat in [
        ("nc8_data_*.csv", "nc8_struct_*.csv"),
        ("nc8_dynamic_*.csv", "nc8_structure_dynamic.npy"),
        ("finance_data_*.csv", "finance_struct_*.csv"),
    ]:
        data_paths = sorted(data_dir.glob(data_pat))
        if data_paths:
            break
    else:
        raise FileNotFoundError(
            f"No recognized dataset files under {data_dir}. "
            "Expected nc8_data_*.csv, nc8_dynamic_*.csv, or finance_data_*.csv."
        )

    if struct_pat.endswith(".npy"):
        npy_paths = sorted(data_dir.glob(struct_pat))
        if not npy_paths:
            raise FileNotFoundError(f"Expected {struct_pat} under {data_dir}.")
        shared_graph = np.load(str(npy_paths[0]), allow_pickle=True)
        if shared_graph.ndim == 3:
            shared_graph = (np.max(np.abs(shared_graph), axis=0) > 0).astype(np.int64)
        shared_graph = np.asarray(shared_graph, dtype=np.int64)
        first_header, first_seq = read_csv_matrix(data_paths[0])
        variable_names = first_header or [f"X{i}" for i in range(first_seq.shape[1])]
        return {
            "data_dir": str(data_dir),
            "data_paths": [str(p) for p in data_paths],
            "struct_paths": [str(npy_paths[0])] * len(data_paths),
            "variable_names": variable_names,
            "first_graph": shared_graph,
            "replica_count": len(data_paths),
        }

    struct_paths = sorted(data_dir.glob(struct_pat))
    if not struct_paths:
        raise FileNotFoundError(f"Expected structure files matching {struct_pat} under {data_dir}.")
    if len(data_paths) != len(struct_paths):
        raise ValueError(
            "Data/structure file counts do not match: "
            f"{len(data_paths)} vs {len(struct_paths)}"
        )

    variable_names: list[str] | None = None
    first_graph: np.ndarray | None = None

    for data_path, struct_path in zip(data_paths, struct_paths):
        data_header, data_matrix = read_csv_matrix(data_path)
        struct_header, struct_matrix = read_csv_matrix(struct_path)

        if variable_names is None:
            variable_names = data_header or [f"X{i}" for i in range(data_matrix.shape[1])]
        elif data_header is not None and data_header != variable_names:
            raise ValueError(f"Inconsistent data columns in {data_path}.")

        if struct_header is not None and struct_header != variable_names:
            raise ValueError(f"Inconsistent structure columns in {struct_path}.")

        # Drop extra numeric header row if present (e.g. Finance struct CSVs)
        n_vars = data_matrix.shape[1]
        if struct_matrix.shape == (n_vars + 1, n_vars):
            struct_matrix = struct_matrix[1:]
        if struct_matrix.shape != (n_vars, n_vars):
            raise ValueError(
                f"GT graph shape {struct_matrix.shape} does not match data width "
                f"{n_vars} in {struct_path}."
            )

        if first_graph is None:
            first_graph = struct_matrix.astype(np.int64)

    return {
        "data_dir": str(data_dir),
        "data_paths": [str(path) for path in data_paths],
        "struct_paths": [str(path) for path in struct_paths],
        "variable_names": variable_names,
        "first_graph": first_graph,
        "replica_count": len(data_paths),
    }


def load_nc8_replica(
    data_path: Path,
    struct_path: Path,
    expected_variable_names: list[str] | None = None,
) -> dict[str, Any]:
    data_header, data_matrix = read_csv_matrix(data_path)

    if str(struct_path).endswith(".npy"):
        graph = np.load(str(struct_path), allow_pickle=True)
        if graph.ndim == 3:
            graph = (np.max(np.abs(graph), axis=0) > 0).astype(np.int64)
        struct_matrix = np.asarray(graph, dtype=np.int64)
    else:
        _struct_header, struct_matrix = read_csv_matrix(struct_path)

    variable_names = expected_variable_names or data_header or [f"X{i}" for i in range(data_matrix.shape[1])]

    n = len(variable_names)
    if struct_matrix.shape == (n + 1, n):
        struct_matrix = struct_matrix[1:]

    if data_header is not None and data_header != variable_names:
        raise ValueError(f"Inconsistent data columns in {data_path}.")

    return {
        "variable_names": variable_names,
        "sequence": data_matrix,
        "graph": struct_matrix.astype(np.int64),
    }


def configure_torch_threads(num_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_matrix_csv(path: Path, matrix: np.ndarray, header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + header)
        for name, row in zip(header, matrix.tolist()):
            writer.writerow([name] + row)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64).reshape(-1)


def binary_classification_stats(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    ap = np.sum(precision * y_sorted) / n_pos
    return float(ap)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, fn, _ = binary_classification_stats(y_true, y_pred)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_shd(gt: np.ndarray, pred: np.ndarray) -> int:
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    if gt.shape != pred.shape:
        raise ValueError(f"SHD expects matching shapes, got {gt.shape} vs {pred.shape}.")

    n = gt.shape[0]
    shd = 0

    for i in range(n):
        shd += abs(int(pred[i, i]) - int(gt[i, i]))

    for i in range(n):
        for j in range(i + 1, n):
            gt_pair = (int(gt[i, j]), int(gt[j, i]))
            pred_pair = (int(pred[i, j]), int(pred[j, i]))
            if gt_pair == pred_pair:
                continue
            if (
                sum(gt_pair) == 1
                and sum(pred_pair) == 1
                and pred_pair == (gt_pair[1], gt_pair[0])
            ):
                shd += 1
            else:
                shd += abs(pred_pair[0] - gt_pair[0]) + abs(pred_pair[1] - gt_pair[1])
    return int(shd)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def ci_95(std_value: float, sample_size: int) -> float:
    if sample_size < 2:
        return 0.0
    return float(T_CRIT_95_DF4 * std_value / math.sqrt(sample_size))


def save_replica_outputs(
    predicted_dir: Path,
    replica_index: int,
    variable_names: list[str],
    lag_score_tensor: np.ndarray,
    score_matrix: np.ndarray,
    binary_graph: np.ndarray,
    train_loss_array: np.ndarray,
) -> None:
    prefix = predicted_dir / f"replica_{replica_index}"
    np.save(prefix.with_name(prefix.name + "_lag_score_tensor.npy"), lag_score_tensor)
    np.save(prefix.with_name(prefix.name + "_score_matrix.npy"), score_matrix)
    np.save(prefix.with_name(prefix.name + "_binary_graph.npy"), binary_graph)
    np.save(prefix.with_name(prefix.name + "_train_loss.npy"), train_loss_array)

    write_matrix_csv(prefix.with_name(prefix.name + "_score_matrix.csv"), score_matrix, variable_names)
    write_matrix_csv(prefix.with_name(prefix.name + "_binary_graph.csv"), binary_graph, variable_names)

    lag_header = [f"lag_{lag_idx + 1}" for lag_idx in range(lag_score_tensor.shape[2])]
    for source_idx, source_name in enumerate(variable_names):
        lag_path = prefix.with_name(prefix.name + f"_lag_scores_from_{source_name}.csv")
        with lag_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["target"] + lag_header)
            for target_name, row in zip(variable_names, lag_score_tensor[:, source_idx, :].tolist()):
                writer.writerow([target_name] + row)


def build_readme(output_dir: Path, data_dir: str, args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# cMLP NC8 Baseline",
            "",
            f"- Repo folder: `{REPO_ROOT}`",
            f"- Results folder: `{output_dir}`",
            f"- Dataset: `{data_dir}`",
            "- Method: `cMLP` from the official Neural-GC repository",
            f"- lag: `{args.lag}`",
            f"- hidden: `{args.hidden}`",
            f"- activation: `{args.activation}`",
            f"- penalty: `{args.penalty}`",
            f"- lam: `{args.lam}`",
            f"- lam_ridge: `{args.lam_ridge}`",
            f"- lr: `{args.lr}`",
            f"- max_iter: `{args.max_iter}`",
            f"- check_every: `{args.check_every}`",
            f"- lookback: `{args.lookback}`",
            f"- torch_threads: `{args.torch_threads}`",
            "",
            "Contents:",
            "- `method_note.md`: exact repo code path used for cMLP",
            "- `data_inspection.md`: NC8 replica layout and selected data path",
            "- `evaluation_protocol.md`: exact score, binary graph, and metric definitions",
            "- `per_replica_metrics.csv`: one row per NC8 replica",
            "- `final_results_table.csv` and `.md`: paper-facing aggregate metrics",
            "- `final_summary_for_paper.md`: concise paper-ready summary",
            "- `predicted_graphs/`: per-replica score matrices, binary graphs, lag scores, and loss traces",
            "- `logs/`: stdout, stderr, and per-replica execution log",
            "",
        ]
    )


def build_method_note(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Method Note",
            "",
            "Selected implementation path from the official Neural-GC repository:",
            "- Model class: `models/cmlp.py::cMLP`",
            "- Training function: `models/cmlp.py::train_model_ista`",
            "- Official usage reference: `cmlp_lagged_var_demo.ipynb`",
            "- Graph extraction API: `cMLP.GC(...)`",
            "",
            "Exact path used in this baseline:",
            "- Instantiate `cMLP(num_series=8, lag=5, hidden=[20], activation=\"relu\")`",
            f"- Train with `train_model_ista(..., lam={args.lam}, lam_ridge={args.lam_ridge}, lr={args.lr}, penalty=\"{args.penalty}\", max_iter={args.max_iter}, check_every={args.check_every}, lookback={args.lookback})`",
            "- Extract continuous graph scores with `cMLP.GC(threshold=False, ignore_lag=True)`",
            "- Extract binary adjacency with `cMLP.GC(threshold=True, ignore_lag=True)`",
            "",
            "Why this path was selected:",
            "- It is the repo's direct cMLP implementation rather than a reimplementation.",
            "- It matches the repository's demo workflow for cMLP more closely than the cLSTM or cRNN alternatives.",
            "- The hierarchical penalty `H` is the demo's default sparse cMLP path and yields exact zeros for binary graph extraction.",
            "",
            "Configuration choice note:",
            "- A weaker regularization pilot with the same architecture family produced a fully dense binary graph on NC8.",
            f"- The final baseline therefore keeps the same cMLP + ISTA + hierarchical path but uses `lam={args.lam}` so the learned zero pattern is meaningful for the binary adjacency.",
            "",
        ]
    )


def build_data_inspection(data_dir: str, variable_names: list[str], replica_count: int) -> str:
    return "\n".join(
        [
            "# NC8 Data Inspection",
            "",
            f"- Selected dataset path: `{data_dir}`",
            "- Time-series file pattern: `nc8_data_*.csv`",
            "- GT structure file pattern: `nc8_struct_*.csv`",
            f"- Replicas enumerated: `{replica_count}`",
            "- Time-series shape per replica: `(2000, 8)`",
            "- GT adjacency shape per replica: `(8, 8)`",
            "- Variable names from CSV header: " + ", ".join(f"`{name}`" for name in variable_names),
            "- GT adjacency contains self-lag positives on the diagonal: `diag_sum = 4`",
            "- Total GT positive entries per replica: `11`",
            "- All five GT structure matrices are identical in this NC8 directory",
            "",
            "Selected version rationale:",
            "- The baseline uses the explicit `data/nc8` directory requested for this run.",
            "- This is the same NC8 path used for the existing PCMCI baseline, which keeps cross-baseline comparison aligned.",
            "",
        ]
    )


def build_evaluation_protocol(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Evaluation Protocol",
            "",
            "## Data",
            "- Dataset path: `/storage/home/ydk297/projects/meta_causal_discovery/data/nc8`",
            "- Replicas used: all five `nc8_data_*.csv` files with their matching `nc8_struct_*.csv` graphs",
            "- GT is evaluated separately for each replica against its corresponding structure CSV",
            "",
            "## cMLP Run",
            "- Model: `cMLP` from `models/cmlp.py`",
            f"- lag: `{args.lag}`",
            f"- hidden: `{args.hidden}`",
            f"- activation: `{args.activation}`",
            f"- penalty: `{args.penalty}`",
            f"- lam: `{args.lam}`",
            f"- lam_ridge: `{args.lam_ridge}`",
            f"- lr: `{args.lr}`",
            f"- max_iter: `{args.max_iter}`",
            f"- check_every: `{args.check_every}`",
            f"- lookback: `{args.lookback}`",
            f"- seed: `{args.seed}`",
            f"- CPU thread setting: `torch.set_num_threads({args.torch_threads})` and `torch.set_num_interop_threads(1)`",
            "",
            "## Continuous Score Matrix",
            "The first cMLP layer is a lagged convolution with weight tensor shape",
            "`(hidden_units, source_series, lag)` for each target series.",
            "",
            "Lag-specific score tensor:",
            "- `lag_score[i, j, k] = ||W_i[:, j, k]||_2`",
            "- This is extracted with `cMLP.GC(threshold=False, ignore_lag=False)`",
            "",
            "Final static 8x8 score matrix:",
            "- `score[i, j] = ||W_i[:, j, :]||_2`",
            "- This is the joint L2 norm over all hidden units and all lags for ordered pair `(j -> i)`",
            "- In code, this is `cMLP.GC(threshold=False, ignore_lag=True)`",
            "",
            "This means lag aggregation is built into the score definition by taking a joint norm over the full lag axis rather than a max or mean over separate lag scores.",
            "",
            "## Binary Graph",
            "- `binary[i, j] = 1[score[i, j] > 0]`",
            "- In code, this is `cMLP.GC(threshold=True, ignore_lag=True)`",
            "- The binary graph therefore comes directly from the exact zero pattern produced by ISTA with hierarchical sparsity, not from an extra post-hoc score threshold",
            "",
            "## Diagonal Handling",
            "- Diagonal/self-lag entries are kept in both the predicted graphs and the evaluation metrics",
            "- The NC8 GT adjacency contains self-lag positives (`diag_sum = 4`), so removing the diagonal would make evaluation incomparable to the provided GT",
            "",
            "## Metrics",
            "- AUROC: computed on the flattened 8x8 continuous score matrix against the flattened 8x8 GT adjacency",
            "- AUPRC: average precision on the same flattened 8x8 continuous score matrix",
            "- F1: computed on the flattened 8x8 binary adjacency",
            "- SHD: directed structural Hamming distance on the 8x8 binary adjacency",
            "",
            "SHD details:",
            "- Diagonal mismatches count directly as one edge addition/deletion each",
            "- For off-diagonal node pairs, an exact orientation reversal counts as 1",
            "- Otherwise SHD counts edge additions/deletions entrywise",
            "",
            "## Runtime Definitions",
            "- `core_runtime_mean_sec`: mean wall-clock time per replica from cMLP model construction through ISTA training and GC score extraction, excluding data loading, metric computation, and file saving",
            "- `end_to_end_runtime_mean_sec`: mean wall-clock time per replica including CSV loading, model setup, training, score extraction, binary graph extraction, metric computation, and per-replica result saving/postprocessing",
            "- For comparison against a full model runtime, `end_to_end_runtime_mean_sec` is the more appropriate number",
            "",
        ]
    )


def build_final_results_table_md(table_row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final cMLP Results",
            "",
            "Results are reported across the `R=5` NC8 replicas as mean `±` 95% confidence interval.",
            "`core_runtime_mean_sec` is the mean wall-clock time per replica for cMLP model construction, training, and GC extraction only.",
            "`end_to_end_runtime_mean_sec` is the mean wall-clock time per replica for the full per-replica baseline pipeline.",
            "",
            "| method | lag | hidden | activation | penalty | lam | lam_ridge | lr | max_iter | replicas_used | AUROC_mean | AUROC_95CI | AUPRC_mean | AUPRC_95CI | F1_mean | F1_95CI | SHD_mean | SHD_95CI | core_runtime_mean_sec | end_to_end_runtime_mean_sec |",
            "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {table_row['method']} | {table_row['lag']} | {table_row['hidden']} | {table_row['activation']} | {table_row['penalty']} | {table_row['lam']:.6f} | {table_row['lam_ridge']:.6f} | {table_row['lr']:.6f} | {table_row['max_iter']} | {table_row['replicas_used']} | {table_row['AUROC_mean']:.6f} | {table_row['AUROC_95CI']} | {table_row['AUPRC_mean']:.6f} | {table_row['AUPRC_95CI']} | {table_row['F1_mean']:.6f} | {table_row['F1_95CI']} | {table_row['SHD_mean']:.6f} | {table_row['SHD_95CI']} | {table_row['core_runtime_mean_sec']:.6f} | {table_row['end_to_end_runtime_mean_sec']:.6f} |",
            "",
            "Notes:",
            "- 95% CI is computed as `t_(0.975, df=4) * s / sqrt(5)` with `t_(0.975,4) = 2.776445`.",
            "- Runtime columns are mean wall-clock seconds per replica only; runtime CI is not included in this table.",
            "- This row uses standard cMLP only; no cLSTM, cRNN, or broad hyperparameter sweep was added.",
            "",
        ]
    )


def build_final_summary_for_paper(
    data_dir: str,
    table_row: dict[str, Any],
    replica_stats: dict[str, float],
    output_dir: Path,
) -> str:
    return "\n".join(
        [
            "# cMLP Baseline Summary For Paper",
            "",
            "## Setup",
            "",
            f"- NC8 uses `R=5` replicas from `{data_dir}`.",
            "- Results are reported as mean `±` 95% confidence interval across the 5 replicas.",
            "- The baseline method is `cMLP` only from the official Neural-GC repository.",
            "- No cLSTM or cRNN variants were used.",
            "- No large hyperparameter sweep was performed.",
            f"- Main cMLP baseline setting: `lag={table_row['lag']}`, `hidden={table_row['hidden']}`, `activation={table_row['activation']}`, `penalty={table_row['penalty']}`, `lam={table_row['lam']:.6f}`, `lam_ridge={table_row['lam_ridge']:.6f}`, `lr={table_row['lr']:.6f}`, `max_iter={table_row['max_iter']}`.",
            "",
            "## Evaluation Protocol",
            "",
            "- Score matrix definition: `score[i, j] = ||W_i[:, j, :]||_2`, extracted with `cMLP.GC(threshold=False, ignore_lag=True)`.",
            "- Binary graph definition: `binary[i, j] = 1[score[i, j] > 0]`, extracted with `cMLP.GC(threshold=True, ignore_lag=True)`.",
            "- Lag aggregation rule: joint L2 norm over the full lag axis rather than a max or mean over lags.",
            "- Diagonal entries are kept because the NC8 GT adjacency includes self-lag positives (`diag_sum = 4`).",
            "- The GT graph is the provided `nc8_struct_*.csv` adjacency for each replica.",
            "- `core_runtime_mean_sec` is the mean wall-clock time per replica for cMLP model construction, training, and GC extraction only.",
            "- `end_to_end_runtime_mean_sec` is the mean wall-clock time per replica for the full per-replica pipeline.",
            "",
            "## Final Results",
            "",
            f"- AUROC: `{table_row['AUROC_mean']:.6f} {table_row['AUROC_95CI']}`",
            f"- AUPRC: `{table_row['AUPRC_mean']:.6f} {table_row['AUPRC_95CI']}`",
            f"- F1: `{table_row['F1_mean']:.6f} {table_row['F1_95CI']}`",
            f"- SHD: `{table_row['SHD_mean']:.6f} {table_row['SHD_95CI']}`",
            f"- `core_runtime_mean_sec`: `{table_row['core_runtime_mean_sec']:.6f}`",
            f"- `end_to_end_runtime_mean_sec`: `{table_row['end_to_end_runtime_mean_sec']:.6f}`",
            "",
            "Replica-level standard deviations:",
            f"- AUROC std: `{replica_stats['auroc_std']:.6f}`",
            f"- AUPRC std: `{replica_stats['auprc_std']:.6f}`",
            f"- F1 std: `{replica_stats['f1_std']:.6f}`",
            f"- SHD std: `{replica_stats['shd_std']:.6f}`",
            "",
            "Runtime comparison guidance:",
            "- For comparison against a full model runtime, use `end_to_end_runtime_mean_sec`.",
            "- Use `core_runtime_mean_sec` only for a narrower algorithm-core comparison that excludes data loading, metrics, and saved-result postprocessing.",
            "",
            "## Saved Files",
            "",
            f"- Final CSV table: `{output_dir / 'final_results_table.csv'}`",
            f"- Final Markdown table: `{output_dir / 'final_results_table.md'}`",
            f"- This paper summary: `{output_dir / 'final_summary_for_paper.md'}`",
            "",
        ]
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> int:
    args = parse_args()
    configure_torch_threads(args.torch_threads)
    device = resolve_device(args.device)

    output_dir = args.output_dir.expanduser().resolve()
    logs_dir = output_dir / "logs"
    predicted_dir = output_dir / "predicted_graphs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predicted_dir.mkdir(parents=True, exist_ok=True)

    bundle = discover_nc8_replicas(args.data_dir)
    variable_names = bundle["variable_names"]

    config = {
        "repo_root": str(REPO_ROOT),
        "data_dir": bundle["data_dir"],
        "data_paths": bundle["data_paths"],
        "struct_paths": bundle["struct_paths"],
        "method": "cMLP",
        "lag": args.lag,
        "hidden": args.hidden,
        "activation": args.activation,
        "penalty": args.penalty,
        "lam": args.lam,
        "lam_ridge": args.lam_ridge,
        "lr": args.lr,
        "max_iter": args.max_iter,
        "check_every": args.check_every,
        "lookback": args.lookback,
        "seed": args.seed,
        "torch_threads": args.torch_threads,
        "device": str(device),
        "verbose": args.verbose,
        "replica_count": bundle["replica_count"],
        "score_definition": "score[i,j] = ||W_i[:, j, :]||_2 from cMLP.GC(threshold=False, ignore_lag=True)",
        "binary_definition": "binary[i,j] = 1[score[i,j] > 0] from cMLP.GC(threshold=True, ignore_lag=True)",
        "runtime_definitions": {
            "core_runtime_mean_sec": "mean wall-clock seconds per replica for model construction, ISTA training, and GC extraction",
            "end_to_end_runtime_mean_sec": "mean wall-clock seconds per replica for loading, model setup, training, score extraction, metrics, and result saving",
        },
    }
    write_json(output_dir / "config.json", config)

    write_matrix_csv(output_dir / "ground_truth_graph_replica0.csv", bundle["first_graph"], variable_names)
    np.save(output_dir / "ground_truth_graph_replica0.npy", bundle["first_graph"])

    log_path = logs_dir / "baseline.log"
    log_path.write_text("", encoding="utf-8")
    append_log(log_path, "cMLP NC8 baseline")
    append_log(log_path, f"Data directory: {bundle['data_dir']}")
    append_log(log_path, f"Replicas: {bundle['replica_count']}")
    append_log(
        log_path,
        " ".join(
            [
                f"lag={args.lag}",
                f"hidden={args.hidden}",
                f"activation={args.activation}",
                f"penalty={args.penalty}",
                f"lam={args.lam}",
                f"lam_ridge={args.lam_ridge}",
                f"lr={args.lr}",
                f"max_iter={args.max_iter}",
                f"check_every={args.check_every}",
                f"lookback={args.lookback}",
                f"seed={args.seed}",
                f"torch_threads={args.torch_threads}",
                f"device={device}",
            ]
        ),
    )

    per_replica_rows: list[dict[str, Any]] = []

    for replica_index, (data_path_str, struct_path_str) in enumerate(
        zip(bundle["data_paths"], bundle["struct_paths"])
    ):
        data_path = Path(data_path_str)
        struct_path = Path(struct_path_str)
        append_log(log_path, f"[replica {replica_index}] data={data_path} struct={struct_path}")

        end_to_end_start = time.perf_counter()
        replica = load_nc8_replica(data_path, struct_path, expected_variable_names=variable_names)
        sequence = replica["sequence"]
        gt_graph = replica["graph"]

        X = torch.tensor(sequence[np.newaxis, :, :], dtype=torch.float32).to(device)

        set_random_seed(args.seed)
        core_start = time.perf_counter()
        cmlp = cMLP(
            num_series=X.shape[-1],
            lag=args.lag,
            hidden=args.hidden,
            activation=args.activation,
        ).to(device)
        loss_list = train_model_ista(
            cmlp,
            X,
            lam=args.lam,
            lam_ridge=args.lam_ridge,
            lr=args.lr,
            penalty=args.penalty,
            max_iter=args.max_iter,
            check_every=args.check_every,
            lookback=args.lookback,
            verbose=args.verbose,
        )
        lag_score_tensor = cmlp.GC(threshold=False, ignore_lag=False).detach().cpu().numpy()
        score_matrix = cmlp.GC(threshold=False, ignore_lag=True).detach().cpu().numpy()
        binary_graph = cmlp.GC(threshold=True, ignore_lag=True).detach().cpu().numpy().astype(np.int64)
        core_runtime_sec = time.perf_counter() - core_start

        train_loss_array = np.asarray([float(value) for value in loss_list], dtype=np.float64)
        save_replica_outputs(
            predicted_dir=predicted_dir,
            replica_index=replica_index,
            variable_names=variable_names,
            lag_score_tensor=lag_score_tensor,
            score_matrix=score_matrix,
            binary_graph=binary_graph,
            train_loss_array=train_loss_array,
        )

        y_true = flatten_matrix(gt_graph).astype(np.int64)
        score_flat = flatten_matrix(score_matrix)
        pred_flat = flatten_matrix(binary_graph).astype(np.int64)

        auroc = compute_auroc(y_true, score_flat)
        auprc = compute_auprc(y_true, score_flat)
        f1 = compute_f1(y_true, pred_flat)
        shd = compute_shd(gt_graph, binary_graph)
        tp, fp, fn, tn = binary_classification_stats(y_true, pred_flat)
        end_to_end_runtime_sec = time.perf_counter() - end_to_end_start

        row = {
            "replica_index": replica_index,
            "data_path": str(data_path),
            "structure_path": str(struct_path),
            "sequence_length": int(sequence.shape[0]),
            "num_variables": int(sequence.shape[1]),
            "gt_edges": int(gt_graph.sum()),
            "predicted_edges": int(binary_graph.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "auroc": float(auroc),
            "auprc": float(auprc),
            "f1": float(f1),
            "shd": int(shd),
            "core_runtime_sec": float(core_runtime_sec),
            "end_to_end_runtime_sec": float(end_to_end_runtime_sec),
            "train_loss_points": int(train_loss_array.size),
            "final_train_loss": float(train_loss_array[-1]) if train_loss_array.size else float("nan"),
        }
        per_replica_rows.append(row)

        append_log(
            log_path,
            " ".join(
                [
                    f"[replica {replica_index}]",
                    f"AUROC={auroc:.6f}",
                    f"AUPRC={auprc:.6f}",
                    f"F1={f1:.6f}",
                    f"SHD={shd}",
                    f"predicted_edges={int(binary_graph.sum())}",
                    f"core_runtime_sec={core_runtime_sec:.6f}",
                    f"end_to_end_runtime_sec={end_to_end_runtime_sec:.6f}",
                ]
            ),
        )

    write_rows_csv(output_dir / "per_replica_metrics.csv", per_replica_rows)

    auroc_values = [float(row["auroc"]) for row in per_replica_rows]
    auprc_values = [float(row["auprc"]) for row in per_replica_rows]
    f1_values = [float(row["f1"]) for row in per_replica_rows]
    shd_values = [float(row["shd"]) for row in per_replica_rows]
    core_runtime_values = [float(row["core_runtime_sec"]) for row in per_replica_rows]
    end_to_end_runtime_values = [float(row["end_to_end_runtime_sec"]) for row in per_replica_rows]

    replica_stats = {
        "auroc_std": sample_std(auroc_values),
        "auprc_std": sample_std(auprc_values),
        "f1_std": sample_std(f1_values),
        "shd_std": sample_std(shd_values),
    }

    table_row = {
        "method": "cMLP",
        "lag": args.lag,
        "hidden": json.dumps(args.hidden),
        "activation": args.activation,
        "penalty": args.penalty,
        "lam": args.lam,
        "lam_ridge": args.lam_ridge,
        "lr": args.lr,
        "max_iter": args.max_iter,
        "replicas_used": bundle["replica_count"],
        "AUROC_mean": float(np.mean(auroc_values)),
        "AUROC_95CI": f"±{ci_95(replica_stats['auroc_std'], bundle['replica_count']):.6f}",
        "AUPRC_mean": float(np.mean(auprc_values)),
        "AUPRC_95CI": f"±{ci_95(replica_stats['auprc_std'], bundle['replica_count']):.6f}",
        "F1_mean": float(np.mean(f1_values)),
        "F1_95CI": f"±{ci_95(replica_stats['f1_std'], bundle['replica_count']):.6f}",
        "SHD_mean": float(np.mean(shd_values)),
        "SHD_95CI": f"±{ci_95(replica_stats['shd_std'], bundle['replica_count']):.6f}",
        "core_runtime_mean_sec": float(np.mean(core_runtime_values)),
        "end_to_end_runtime_mean_sec": float(np.mean(end_to_end_runtime_values)),
    }

    write_rows_csv(output_dir / "final_results_table.csv", [table_row])
    (output_dir / "final_results_table.md").write_text(
        build_final_results_table_md(table_row),
        encoding="utf-8",
    )
    (output_dir / "final_summary_for_paper.md").write_text(
        build_final_summary_for_paper(
            data_dir=bundle["data_dir"],
            table_row=table_row,
            replica_stats=replica_stats,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )
    (output_dir / "README_baseline.md").write_text(
        build_readme(output_dir, bundle["data_dir"], args),
        encoding="utf-8",
    )
    (output_dir / "method_note.md").write_text(
        build_method_note(args),
        encoding="utf-8",
    )
    (output_dir / "data_inspection.md").write_text(
        build_data_inspection(bundle["data_dir"], variable_names, bundle["replica_count"]),
        encoding="utf-8",
    )
    (output_dir / "evaluation_protocol.md").write_text(
        build_evaluation_protocol(args),
        encoding="utf-8",
    )

    append_log(
        log_path,
        " ".join(
            [
                "[aggregate]",
                f"AUROC_mean={table_row['AUROC_mean']:.6f}",
                f"AUPRC_mean={table_row['AUPRC_mean']:.6f}",
                f"F1_mean={table_row['F1_mean']:.6f}",
                f"SHD_mean={table_row['SHD_mean']:.6f}",
                f"core_runtime_mean_sec={table_row['core_runtime_mean_sec']:.6f}",
                f"end_to_end_runtime_mean_sec={table_row['end_to_end_runtime_mean_sec']:.6f}",
            ]
        ),
    )

    print("cMLP NC8 baseline completed successfully.")
    print(f"Replicas used: {bundle['replica_count']}")
    print(
        "Aggregate metrics: "
        f"AUROC={table_row['AUROC_mean']:.6f} {table_row['AUROC_95CI']} "
        f"AUPRC={table_row['AUPRC_mean']:.6f} {table_row['AUPRC_95CI']} "
        f"F1={table_row['F1_mean']:.6f} {table_row['F1_95CI']} "
        f"SHD={table_row['SHD_mean']:.6f} {table_row['SHD_95CI']} "
        f"core_runtime_mean_sec={table_row['core_runtime_mean_sec']:.6f} "
        f"end_to_end_runtime_mean_sec={table_row['end_to_end_runtime_mean_sec']:.6f}"
    )
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
