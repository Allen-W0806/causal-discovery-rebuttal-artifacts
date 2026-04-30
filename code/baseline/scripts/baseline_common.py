"""Shared data, metric, and result helpers for Appendix L baseline wrappers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-300


def is_numeric_row(tokens: list[str]) -> bool:
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
    tokens = [token.strip() for token in first_line.split(",")]
    has_header = not is_numeric_row(tokens) or _is_integer_index_header(tokens)
    matrix = np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0, dtype=np.float64)
    return tokens if has_header else None, np.atleast_2d(matrix)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_matrix_csv(path: Path, matrix: np.ndarray, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + header)
        for name, row in zip(header, np.asarray(matrix).tolist()):
            writer.writerow([name] + row)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_replicas(dataset: str, data_dir: Path, data_glob: str, struct_glob: str) -> list[dict[str, Any]]:
    data_paths = sorted(data_dir.glob(data_glob))
    if not data_paths:
        raise FileNotFoundError(f"No data files matching {data_glob} under {data_dir}")

    replicas: list[dict[str, Any]] = []
    variable_names: list[str] | None = None

    if dataset == "ND8":
        struct_paths = sorted(data_dir.glob(struct_glob))
        if len(struct_paths) != 1:
            raise FileNotFoundError(f"Expected one ND8 structure file matching {struct_glob} under {data_dir}")
        structure = np.load(struct_paths[0], allow_pickle=True)
        if structure.ndim == 3:
            structure = (np.max(np.abs(structure), axis=0) > 0).astype(np.int64)
        structure = np.asarray(structure, dtype=np.int64)
        for idx, data_path in enumerate(data_paths):
            header, sequence = read_csv_matrix(data_path)
            if variable_names is None:
                variable_names = header or [f"X{i}" for i in range(sequence.shape[1])]
            replicas.append(
                {
                    "replica_index": idx,
                    "data_path": str(data_path),
                    "struct_path": str(struct_paths[0]),
                    "variable_names": variable_names,
                    "sequence": sequence,
                    "graph": structure,
                }
            )
        return replicas

    struct_paths = sorted(data_dir.glob(struct_glob))
    if len(data_paths) != len(struct_paths):
        raise ValueError(f"Data/structure file counts do not match in {data_dir}: {len(data_paths)} vs {len(struct_paths)}")

    for idx, (data_path, struct_path) in enumerate(zip(data_paths, struct_paths)):
        data_header, sequence = read_csv_matrix(data_path)
        struct_header, graph = read_csv_matrix(struct_path)
        if variable_names is None:
            variable_names = data_header or [f"X{i}" for i in range(sequence.shape[1])]
        if data_header is not None and data_header != variable_names:
            raise ValueError(f"Inconsistent data header in {data_path}")
        if struct_header is not None and struct_header != variable_names:
            raise ValueError(f"Inconsistent structure header in {struct_path}")
        graph = np.asarray(graph, dtype=np.int64)
        # Some CSVs (e.g. Finance) have a numeric index header row that is not
        # detected as a header; drop that extra row if shape is (n+1, n).
        n = len(variable_names)
        if graph.shape == (n + 1, n):
            graph = graph[1:]
        if graph.shape != (n, n):
            raise ValueError(f"Structure shape {graph.shape} does not match data width {n} in {struct_path}")
        replicas.append(
            {
                "replica_index": idx,
                "data_path": str(data_path),
                "struct_path": str(struct_path),
                "variable_names": variable_names,
                "sequence": sequence,
                "graph": graph,
            }
        )
    return replicas


def flatten(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64).reshape(-1)


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))


def compute_shd(gt: np.ndarray, pred: np.ndarray) -> int:
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    if gt.shape != pred.shape:
        raise ValueError(f"SHD expects matching shapes, got {gt.shape} vs {pred.shape}")
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
            if sum(gt_pair) == 1 and sum(pred_pair) == 1 and pred_pair == (gt_pair[1], gt_pair[0]):
                shd += 1
            else:
                shd += abs(pred_pair[0] - gt_pair[0]) + abs(pred_pair[1] - gt_pair[1])
    return int(shd)


def threshold_by_top_k(score_matrix: np.ndarray, graph: np.ndarray) -> np.ndarray:
    scores = flatten(score_matrix)
    k = int(np.asarray(graph).sum())
    if k <= 0:
        return np.zeros_like(graph, dtype=np.int64)
    cutoff = np.partition(scores, -k)[-k]
    return (score_matrix >= cutoff).astype(np.int64)


def evaluate_score_matrix(score_matrix: np.ndarray, graph: np.ndarray, binary_graph: np.ndarray | None = None) -> dict[str, float]:
    graph = np.asarray(graph, dtype=np.int64)
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    if binary_graph is None:
        binary_graph = threshold_by_top_k(score_matrix, graph)
    y_true = flatten(graph).astype(np.int64)
    scores = flatten(score_matrix)
    y_pred = flatten(binary_graph).astype(np.int64)
    return {
        "auroc": compute_auroc(y_true, scores),
        "auprc": compute_auprc(y_true, scores),
        "f1": compute_f1(y_true, y_pred),
        "shd": float(compute_shd(graph, binary_graph)),
        "predicted_edges": float(np.asarray(binary_graph).sum()),
        "gt_edges": float(graph.sum()),
    }


def finite_or_none(value: float) -> float | None:
    return None if isinstance(value, float) and (math.isnan(value) or math.isinf(value)) else value

