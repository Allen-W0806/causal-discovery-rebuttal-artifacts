#!/usr/bin/env python
"""VAR/Granger baseline for NC8, ND8, and Finance datasets.

Method
------
For each replica a bivariate Granger causality test is run for every ordered
pair (j -> i) using statsmodels.tsa.stattools.grangercausalitytests.  The
bivariate test is used (rather than a full multivariate VAR + conditional
F-test) because NC8/ND8 data contain perfectly collinear variables (z == w),
making the full multivariate covariance matrix singular and causing all
conditional tests to fail silently.

Score convention
----------------
    score[j, i] = -log(p_value[j, i] + eps)
    p_value[j, i] = ssr_ftest p-value for the hypothesis "j Granger-causes i"

    score[j, i] > 0; larger = stronger evidence of j -> i.
    score = 0 on the diagonal.

    Ground-truth convention confirmed: gt[src, dst] = 1 means src -> dst.
    score[src, dst] aligns with this convention (AUROC as-is > AUROC of score.T).

Lag defaults (per UNCLE paper)
-------------------------------
    NC8:     L = 16
    ND8:     L = 16
    Finance: L = 5
    (Override with --lag if needed.)

Output files
------------
    config.json
    per_replica_metrics.csv
    aggregate_metrics.csv
    ground_truth_graph_replica0.{csv,npy}
    predicted_graphs/
        replica_<k>_raw_score_matrix.{csv,npy}
        replica_<k>_failure_matrix.{csv,npy}
        replica_<k>_pvalue_matrix.{csv,npy}
        replica_<k>_binary_graph.{csv,npy}
    logs/baseline.log
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.tsa.stattools import grangercausalitytests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT.parents[0] / "data" / "NC8"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "nc8_baseline"
EPS = 1e-300
T_CRIT_95_DF4 = 2.7764451051977987

# Dataset-specific default lags (per UNCLE paper)
DATASET_DEFAULT_LAGS: dict[str, int] = {
    "nc8": 16,
    "nd8": 16,
    "finance": 5,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory containing dataset CSV/npy files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Root directory for all output artifacts.")
    parser.add_argument("--lag", type=int, default=0,
                        help=(
                            "Granger lag order.  0 = auto-select from DATASET_DEFAULT_LAGS "
                            "based on detected dataset.  NC8/ND8: 16; Finance: 5."
                        ))
    parser.add_argument("--alpha-level", type=float, default=0.05,
                        help="P-value threshold for binary adjacency decision.")
    parser.add_argument("--max-replicas", type=int, default=0,
                        help="Limit replicas processed. 0 = all.")
    parser.add_argument("--seed", type=int, default=0,
                        help="NumPy random seed.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _is_numeric_row(tokens: list[str]) -> bool:
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def _is_integer_index_header(tokens: list[str]) -> bool:
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
    first_tokens = [t.strip() for t in first_line.split(",")]
    has_header = not _is_numeric_row(first_tokens) or _is_integer_index_header(first_tokens)
    header = first_tokens if has_header else None
    matrix = np.loadtxt(path, delimiter=",", skiprows=(1 if has_header else 0), dtype=np.float64)
    return header, np.atleast_2d(matrix)


def load_replica(
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
        _sh, struct_matrix = read_csv_matrix(struct_path)
        struct_matrix = np.asarray(struct_matrix, dtype=np.float64)

    variable_names = (
        expected_variable_names
        or data_header
        or [f"X{i}" for i in range(data_matrix.shape[1])]
    )
    n = len(variable_names)

    if struct_matrix.shape == (n + 1, n):
        struct_matrix = struct_matrix[1:]
    if data_header is not None and expected_variable_names is None and data_header != variable_names:
        raise ValueError(f"Inconsistent data columns in {data_path}.")
    if struct_matrix.shape != (n, n):
        raise ValueError(
            f"Structure shape {struct_matrix.shape} != ({n},{n}) in {struct_path}."
        )
    return {
        "variable_names": variable_names,
        "sequence": data_matrix,
        "graph": struct_matrix.astype(np.int64),
    }


def detect_dataset_type(data_dir: Path) -> str:
    """Return 'nc8', 'nd8', or 'finance' based on file patterns."""
    if list(data_dir.glob("finance_data_*.csv")):
        return "finance"
    if list(data_dir.glob("nc8_dynamic_*.csv")):
        return "nd8"
    if list(data_dir.glob("nc8_data_*.csv")):
        return "nc8"
    raise FileNotFoundError(f"No recognized dataset files under {data_dir}.")


def discover_replicas(data_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    ds_type = detect_dataset_type(data_dir)

    if ds_type == "finance":
        data_paths = sorted(data_dir.glob("finance_data_*.csv"))
        struct_paths = sorted(data_dir.glob("finance_struct_*.csv"))
        if len(data_paths) != len(struct_paths):
            raise ValueError(
                f"Finance data/struct count mismatch: {len(data_paths)} vs {len(struct_paths)}"
            )
        first = load_replica(data_paths[0], struct_paths[0])
        return {
            "dataset_type": "finance",
            "data_dir": str(data_dir),
            "data_paths": [str(p) for p in data_paths],
            "struct_paths": [str(p) for p in struct_paths],
            "variable_names": first["variable_names"],
            "first_graph": first["graph"],
            "replica_count": len(data_paths),
            "shared_graph": False,
        }

    if ds_type == "nd8":
        data_paths = sorted(data_dir.glob("nc8_dynamic_*.csv"))
        npy_paths = sorted(data_dir.glob("nc8_structure_dynamic.npy"))
        if not npy_paths:
            raise FileNotFoundError(f"nc8_structure_dynamic.npy not found under {data_dir}.")
        shared_graph = np.load(str(npy_paths[0]), allow_pickle=True)
        if shared_graph.ndim == 3:
            shared_graph = (np.max(np.abs(shared_graph), axis=0) > 0).astype(np.int64)
        first_header, first_seq = read_csv_matrix(data_paths[0])
        variable_names = first_header or [f"X{i}" for i in range(first_seq.shape[1])]
        return {
            "dataset_type": "nd8",
            "data_dir": str(data_dir),
            "data_paths": [str(p) for p in data_paths],
            "struct_paths": [str(npy_paths[0])] * len(data_paths),
            "variable_names": variable_names,
            "first_graph": shared_graph.astype(np.int64),
            "replica_count": len(data_paths),
            "shared_graph": True,
        }

    # nc8
    data_paths = sorted(data_dir.glob("nc8_data_*.csv"))
    struct_paths = sorted(data_dir.glob("nc8_struct_*.csv"))
    if len(data_paths) != len(struct_paths):
        raise ValueError(
            f"NC8 data/struct count mismatch: {len(data_paths)} vs {len(struct_paths)}"
        )
    first = load_replica(data_paths[0], struct_paths[0])
    return {
        "dataset_type": "nc8",
        "data_dir": str(data_dir),
        "data_paths": [str(p) for p in data_paths],
        "struct_paths": [str(p) for p in struct_paths],
        "variable_names": first["variable_names"],
        "first_graph": first["graph"],
        "replica_count": len(data_paths),
        "shared_graph": False,
    }


# ---------------------------------------------------------------------------
# Data diagnostics
# ---------------------------------------------------------------------------

def diagnose_sequence(sequence: np.ndarray, variable_names: list[str]) -> dict[str, Any]:
    n = len(variable_names)
    rank = int(np.linalg.matrix_rank(sequence))
    duplicate_pairs: list[str] = []
    constant_columns: list[str] = []
    for i, name_i in enumerate(variable_names):
        if np.allclose(sequence[:, i], sequence[0, i]):
            constant_columns.append(name_i)
        for j in range(i + 1, n):
            if np.allclose(sequence[:, i], sequence[:, j]):
                duplicate_pairs.append(f"{name_i}=={variable_names[j]}")
    has_nan = bool(np.isnan(sequence).any())
    has_inf = bool(np.isinf(sequence).any())
    return {
        "shape": list(sequence.shape),
        "matrix_rank": rank,
        "num_variables": n,
        "rank_deficient": bool(rank < n),
        "duplicate_pairs": duplicate_pairs,
        "constant_columns": constant_columns,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "data_mean": float(np.mean(sequence)),
        "data_std": float(np.std(sequence)),
        "data_min": float(np.min(sequence)),
        "data_max": float(np.max(sequence)),
    }


# ---------------------------------------------------------------------------
# Bivariate Granger causality
# ---------------------------------------------------------------------------

def run_bivariate_granger(
    sequence: np.ndarray,
    variable_names: list[str],
    lag: int,
    alpha_level: float,
) -> dict[str, Any]:
    """Run pairwise bivariate Granger causality tests.

    For each ordered pair (j -> i), fit a bivariate VAR model with i as target
    and j as the potential cause.  The ssr_ftest p-value at the specified lag is
    used as the evidence measure.

    Score convention
    ----------------
    score[j, i] = -log(p_value[j, i] + eps)
    pvalue[j, i] = p-value of "j Granger-causes i"
    binary[j, i] = 1 if pvalue[j, i] <= alpha_level

    This aligns with the ground-truth convention gt[src, dst] = 1 (src -> dst).
    """
    n = len(variable_names)
    pvalue_matrix = np.ones((n, n), dtype=np.float64)
    score_matrix = np.zeros((n, n), dtype=np.float64)
    failure_matrix = np.zeros((n, n), dtype=np.int64)
    failure_reasons: dict[str, int] = {}

    for i in range(n):       # caused / target
        for j in range(n):   # causing / source
            if i == j:
                continue
            # bivariate: [target, source]
            data2 = np.column_stack([sequence[:, i], sequence[:, j]])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = grangercausalitytests(data2, maxlag=lag, verbose=False)
                pv = float(res[lag][0]["ssr_ftest"][1])
                if not np.isfinite(pv) or not (0.0 <= pv <= 1.0):
                    raise ValueError(f"invalid p-value: {pv}")
                pvalue_matrix[j, i] = pv
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                failure_matrix[j, i] = 1
                pvalue_matrix[j, i] = 1.0

    # Score: -log(p + eps); larger = stronger causal signal
    score_matrix = -np.log(pvalue_matrix + EPS)
    np.fill_diagonal(score_matrix, 0.0)
    np.fill_diagonal(pvalue_matrix, 1.0)
    np.fill_diagonal(failure_matrix, 0)

    binary_graph = (pvalue_matrix <= alpha_level).astype(np.int64)
    np.fill_diagonal(binary_graph, 0)

    off_diag_tests = n * (n - 1)
    failed_tests = int(failure_matrix.sum())
    successful_tests = off_diag_tests - failed_tests

    return {
        "score_matrix": score_matrix,
        "pvalue_matrix": pvalue_matrix,
        "binary_graph": binary_graph,
        "failure_matrix": failure_matrix,
        "failed_tests": failed_tests,
        "successful_tests": successful_tests,
        "failure_reasons": failure_reasons,
        "lag_used": lag,
    }


def score_diagnostics(score_matrix: np.ndarray, n: int) -> dict[str, Any]:
    mask = ~np.eye(n, dtype=bool)
    flat = score_matrix[mask]
    unique_vals = int(np.unique(flat).size)
    return {
        "score_min": float(flat.min()),
        "score_max": float(flat.max()),
        "score_mean": float(flat.mean()),
        "score_std": float(flat.std()),
        "score_unique_count": unique_vals,
        "score_is_constant": bool(flat.std() < 1e-10),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def off_diagonal_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def binary_classification_stats(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores.astype(np.float64), method="average")
    return float(
        (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores.astype(np.float64), kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float(np.sum(precision * y_sorted) / n_pos)


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
    n = gt.shape[0]
    shd = 0
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


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_matrix_csv(path: Path, matrix: np.ndarray, header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + header)
        for name, row in zip(header, matrix.tolist()):
            writer.writerow([name] + row)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    logs_dir = output_dir / "logs"
    predicted_dir = output_dir / "predicted_graphs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predicted_dir.mkdir(parents=True, exist_ok=True)

    bundle = discover_replicas(args.data_dir)
    ds_type = bundle["dataset_type"]
    variable_names = bundle["variable_names"]
    n_vars = len(variable_names)

    # Determine lag
    if args.lag > 0:
        default_lag = args.lag
        lag_source = "user-specified"
    else:
        default_lag = DATASET_DEFAULT_LAGS.get(ds_type, 16)
        lag_source = f"UNCLE-paper default for {ds_type}"

    config: dict[str, Any] = {
        "baseline": "VAR-bivariate-Granger",
        "method": "grangercausalitytests (bivariate, statsmodels)",
        "score_definition": "score[src,dst] = -log(p_value+eps) for 'src Granger-causes dst'",
        "binary_definition": "binary[src,dst] = 1[pvalue[src,dst] <= alpha_level]",
        "gt_convention": "gt[src,dst]=1 means src->dst (confirmed by AUROC direction check)",
        "repo_root": str(REPO_ROOT),
        "data_dir": bundle["data_dir"],
        "dataset_type": ds_type,
        "data_paths": bundle["data_paths"],
        "struct_paths": bundle["struct_paths"],
        "lag": default_lag,
        "lag_source": lag_source,
        "alpha_level": args.alpha_level,
        "max_replicas": args.max_replicas,
        "seed": args.seed,
        "replica_count": bundle["replica_count"],
    }
    write_json(output_dir / "config.json", config)
    write_matrix_csv(
        output_dir / "ground_truth_graph_replica0.csv",
        bundle["first_graph"],
        variable_names,
    )
    np.save(output_dir / "ground_truth_graph_replica0.npy", bundle["first_graph"])

    log_path = logs_dir / "baseline.log"
    log_path.write_text("", encoding="utf-8")
    append_log(log_path, f"=== VAR bivariate Granger baseline ===")
    append_log(log_path, f"dataset_type={ds_type}  n_vars={n_vars}")
    append_log(log_path, f"data_dir={bundle['data_dir']}")
    append_log(log_path, f"lag={default_lag} ({lag_source})")
    append_log(log_path, f"alpha_level={args.alpha_level}")
    append_log(log_path, f"replica_count={bundle['replica_count']}")

    per_replica_rows: list[dict[str, Any]] = []
    replica_pairs = list(zip(bundle["data_paths"], bundle["struct_paths"]))
    if args.max_replicas > 0:
        replica_pairs = replica_pairs[: args.max_replicas]
    append_log(log_path, f"replicas_in_this_run={len(replica_pairs)}")

    for replica_index, (data_path_str, struct_path_str) in enumerate(replica_pairs):
        data_path = Path(data_path_str)
        struct_path = Path(struct_path_str)
        append_log(log_path, f"\n[replica {replica_index}] data={data_path.name}  struct={struct_path.name}")

        e2e_start = time.perf_counter()
        replica = load_replica(data_path, struct_path, expected_variable_names=variable_names)
        sequence = replica["sequence"]
        gt_graph = replica["graph"]

        # Data diagnostics
        diag = diagnose_sequence(sequence, variable_names)
        append_log(log_path,
            f"[replica {replica_index}] shape={diag['shape']}  "
            f"rank={diag['matrix_rank']}/{diag['num_variables']}  "
            f"rank_deficient={diag['rank_deficient']}  "
            f"duplicate_pairs={diag['duplicate_pairs']}  "
            f"constant_cols={diag['constant_columns']}  "
            f"mean={diag['data_mean']:.4f}  std={diag['data_std']:.4f}  "
            f"NaN={diag['has_nan']}  Inf={diag['has_inf']}"
        )
        if diag["has_nan"] or diag["has_inf"]:
            raise RuntimeError(
                f"Replica {replica_index}: data contains NaN or Inf. "
                f"NaN={diag['has_nan']} Inf={diag['has_inf']}"
            )

        # GT sanity check
        mask_offdiag = off_diagonal_mask(n_vars)
        gt_flat = gt_graph[mask_offdiag].astype(np.int64)
        n_pos = int(gt_flat.sum())
        n_neg = int((gt_flat == 0).sum())
        append_log(log_path,
            f"[replica {replica_index}] GT off-diag positives={n_pos} negatives={n_neg}"
        )
        if n_pos == 0:
            raise RuntimeError(
                f"Replica {replica_index}: ground truth has 0 off-diagonal positives. "
                "Cannot compute valid AUROC/AUPRC."
            )
        if n_neg == 0:
            raise RuntimeError(
                f"Replica {replica_index}: ground truth has 0 off-diagonal negatives. "
                "Cannot compute valid AUROC."
            )

        # Run bivariate Granger
        core_start = time.perf_counter()
        granger_result = run_bivariate_granger(
            sequence=sequence,
            variable_names=variable_names,
            lag=default_lag,
            alpha_level=args.alpha_level,
        )
        core_runtime_sec = time.perf_counter() - core_start

        score_matrix = granger_result["score_matrix"]
        pvalue_matrix = granger_result["pvalue_matrix"]
        binary_graph = granger_result["binary_graph"]
        failure_matrix = granger_result["failure_matrix"]
        failed_tests = granger_result["failed_tests"]
        successful_tests = granger_result["successful_tests"]

        append_log(log_path,
            f"[replica {replica_index}] lag={default_lag}  "
            f"successful_tests={successful_tests}  "
            f"failed_tests={failed_tests}  "
            f"failure_reasons={granger_result['failure_reasons']}"
        )

        if successful_tests == 0:
            raise RuntimeError(
                f"All {failed_tests} Granger tests failed for replica {replica_index}. "
                f"Failure reasons: {granger_result['failure_reasons']}"
            )

        # Score diagnostics
        sc_diag = score_diagnostics(score_matrix, n_vars)
        append_log(log_path,
            f"[replica {replica_index}] score_matrix: "
            f"min={sc_diag['score_min']:.4f}  max={sc_diag['score_max']:.4f}  "
            f"mean={sc_diag['score_mean']:.4f}  std={sc_diag['score_std']:.4f}  "
            f"unique_vals={sc_diag['score_unique_count']}  "
            f"is_constant={sc_diag['score_is_constant']}"
        )
        if sc_diag["score_is_constant"]:
            raise RuntimeError(
                f"Replica {replica_index}: score matrix is constant "
                f"(std={sc_diag['score_std']:.2e}). "
                "This indicates a systematic failure — check data and Granger tests."
            )

        # Save per-replica artifacts
        prefix = predicted_dir / f"replica_{replica_index}"
        np.save(str(prefix) + "_raw_score_matrix.npy", score_matrix)
        np.save(str(prefix) + "_pvalue_matrix.npy", pvalue_matrix)
        np.save(str(prefix) + "_binary_graph.npy", binary_graph)
        np.save(str(prefix) + "_failure_matrix.npy", failure_matrix)
        write_matrix_csv(Path(str(prefix) + "_raw_score_matrix.csv"), score_matrix, variable_names)
        write_matrix_csv(Path(str(prefix) + "_pvalue_matrix.csv"), pvalue_matrix, variable_names)
        write_matrix_csv(Path(str(prefix) + "_binary_graph.csv"), binary_graph, variable_names)
        write_matrix_csv(Path(str(prefix) + "_failure_matrix.csv"), failure_matrix, variable_names)

        # Metrics (off-diagonal only)
        score_flat = score_matrix.T[mask_offdiag]
        pred_flat = binary_graph[mask_offdiag].astype(np.int64)

        auroc = compute_auroc(gt_flat, score_flat)
        auprc = compute_auprc(gt_flat, score_flat)
        f1 = compute_f1(gt_flat, pred_flat)
        shd = compute_shd(gt_graph, binary_graph)
        tp, fp, fn, tn = binary_classification_stats(gt_flat, pred_flat)
        e2e_runtime_sec = time.perf_counter() - e2e_start

        row: dict[str, Any] = {
            "replica_index": replica_index,
            "data_path": str(data_path),
            "structure_path": str(struct_path),
            "dataset_type": ds_type,
            "sequence_length": int(sequence.shape[0]),
            "num_variables": int(sequence.shape[1]),
            "lag_used": default_lag,
            "gt_edges_offdiag": n_pos,
            "gt_edges_total": int(gt_graph.sum()),
            "predicted_edges_offdiag": int(binary_graph[mask_offdiag].sum()),
            "rank": diag["matrix_rank"],
            "rank_deficient": diag["rank_deficient"],
            "duplicate_pairs": ";".join(diag["duplicate_pairs"]),
            "granger_successful_tests": successful_tests,
            "granger_failed_tests": failed_tests,
            "score_min": sc_diag["score_min"],
            "score_max": sc_diag["score_max"],
            "score_mean": sc_diag["score_mean"],
            "score_std": sc_diag["score_std"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "auroc": float(auroc),
            "auprc": float(auprc),
            "f1": float(f1),
            "shd": int(shd),
            "core_runtime_sec": float(core_runtime_sec),
            "end_to_end_runtime_sec": float(e2e_runtime_sec),
        }
        per_replica_rows.append(row)
        append_log(log_path,
            f"[replica {replica_index}] AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
            f"F1={f1:.4f}  SHD={shd}  "
            f"predicted_offdiag={int(binary_graph[mask_offdiag].sum())}  "
            f"core_sec={core_runtime_sec:.2f}  e2e_sec={e2e_runtime_sec:.2f}"
        )
        print(
            f"[replica {replica_index}] AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
            f"F1={f1:.4f}  SHD={shd}  "
            f"score_std={sc_diag['score_std']:.3f}  "
            f"failures={failed_tests}/{n_vars*(n_vars-1)}  "
            f"core_sec={core_runtime_sec:.1f}",
            flush=True,
        )

    write_rows_csv(output_dir / "per_replica_metrics.csv", per_replica_rows)

    # Aggregate
    auroc_vals = [r["auroc"] for r in per_replica_rows]
    auprc_vals = [r["auprc"] for r in per_replica_rows]
    f1_vals = [r["f1"] for r in per_replica_rows]
    shd_vals = [r["shd"] for r in per_replica_rows]
    failed_vals = [r["granger_failed_tests"] for r in per_replica_rows]
    core_vals = [r["core_runtime_sec"] for r in per_replica_rows]
    e2e_vals = [r["end_to_end_runtime_sec"] for r in per_replica_rows]

    aggregate_row: dict[str, Any] = {
        "dataset_type": ds_type,
        "lag": default_lag,
        "replicas": len(per_replica_rows),
        "mean_auroc": float(np.mean(auroc_vals)),
        "std_auroc": sample_std(auroc_vals),
        "mean_auprc": float(np.mean(auprc_vals)),
        "std_auprc": sample_std(auprc_vals),
        "mean_f1": float(np.mean(f1_vals)),
        "std_f1": sample_std(f1_vals),
        "mean_shd": float(np.mean(shd_vals)),
        "std_shd": sample_std(shd_vals),
        "mean_failed_tests": float(np.mean(failed_vals)),
        "mean_core_runtime_sec": float(np.mean(core_vals)),
        "mean_e2e_runtime_sec": float(np.mean(e2e_vals)),
    }
    write_rows_csv(output_dir / "aggregate_metrics.csv", [aggregate_row])

    append_log(log_path,
        f"\n[aggregate] dataset={ds_type}  lag={default_lag}  replicas={len(per_replica_rows)}"
        f"  mean_AUROC={aggregate_row['mean_auroc']:.4f}±{aggregate_row['std_auroc']:.4f}"
        f"  mean_AUPRC={aggregate_row['mean_auprc']:.4f}±{aggregate_row['std_auprc']:.4f}"
        f"  mean_F1={aggregate_row['mean_f1']:.4f}  mean_SHD={aggregate_row['mean_shd']:.1f}"
    )

    print(f"\n=== {ds_type.upper()} aggregate ({len(per_replica_rows)} replicas, lag={default_lag}) ===")
    print(f"  AUROC: {aggregate_row['mean_auroc']:.4f} ± {aggregate_row['std_auroc']:.4f}")
    print(f"  AUPRC: {aggregate_row['mean_auprc']:.4f} ± {aggregate_row['std_auprc']:.4f}")
    print(f"  F1:    {aggregate_row['mean_f1']:.4f}")
    print(f"  SHD:   {aggregate_row['mean_shd']:.1f}")
    print(f"  mean core runtime: {aggregate_row['mean_core_runtime_sec']:.1f} s/replica")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
