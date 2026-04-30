#!/usr/bin/env python
"""Run a reproducible PCMCI baseline on all NC8 replicas.

The baseline uses the upstream Tigramite implementation of PCMCI with
ParCorr(significance="analytic") and evaluates two modest lag settings:
tau_max in {2, 4}.

Lagged outputs are aggregated to a static 8x8 graph because the NC8 ground
truth is provided as a lag-agnostic adjacency matrix:

    min_p[i, j]   = min_tau p_matrix[i, j, tau]     for tau in [tau_min, tau_max]
    score[i, j]   = -log10(min_p[i, j])
    binary[i, j]  = 1[min_p[i, j] <= alpha_level]

This means the continuous score matrix and binary graph are fully aligned:
the binary graph is just thresholding the aggregated significance score.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("data/NC8")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results_nc8_baseline"
EPS = 1e-300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tau-min", type=int, default=1)
    parser.add_argument("--tau-max-values", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--pc-alpha", type=float, default=0.2)
    parser.add_argument("--alpha-level", type=float, default=0.01)
    parser.add_argument("--verbosity", type=int, default=0)
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


def _read_csv_matrix(path: Path) -> tuple[list[str] | None, np.ndarray]:
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


def load_nc8_replica(
    data_path: Path,
    struct_path: Path,
    expected_variable_names: list[str] | None = None,
) -> dict[str, Any]:
    data_header, data_matrix = _read_csv_matrix(data_path)

    if str(struct_path).endswith(".npy"):
        graph = np.load(str(struct_path), allow_pickle=True)
        if graph.ndim == 3:
            graph = (np.max(np.abs(graph), axis=0) > 0).astype(np.int64)
        struct_matrix = np.asarray(graph, dtype=np.int64)
        struct_header = None
    else:
        struct_header, struct_matrix = _read_csv_matrix(struct_path)

    variable_names = expected_variable_names or data_header or [f"X{i}" for i in range(data_matrix.shape[1])]

    if data_header is not None and data_header != variable_names:
        raise ValueError(f"Inconsistent NC8 data columns in {data_path}.")

    if struct_header is not None and struct_header != variable_names:
        raise ValueError(f"Inconsistent NC8 structure columns in {struct_path}.")

    if data_matrix.shape[1] != len(variable_names):
        raise ValueError(
            f"NC8 sequence width {data_matrix.shape[1]} does not match variable count "
            f"{len(variable_names)} in {data_path}."
        )

    # Drop extra numeric header row if present (e.g. Finance struct CSVs)
    n = len(variable_names)
    if struct_matrix.shape == (n + 1, n):
        struct_matrix = struct_matrix[1:]

    if struct_matrix.shape != (n, n):
        raise ValueError(
            f"NC8 structure shape {struct_matrix.shape} does not match expected square "
            f"{(n, n)} in {struct_path}."
        )

    return {
        "variable_names": variable_names,
        "sequence": data_matrix,
        "graph": struct_matrix.astype(np.int64),
    }


def load_nc8_bundle(data_dir: Path) -> dict[str, Any]:
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
        first_header, _ = _read_csv_matrix(data_paths[0])
        variable_names = first_header or [f"X{i}" for i in range(shared_graph.shape[0])]
        return {
            "data_dir": str(data_dir),
            "data_paths": [str(p) for p in data_paths],
            "struct_paths": [str(npy_paths[0])] * len(data_paths),
            "variable_names": variable_names,
            "first_graph": shared_graph,
            "replica_count": len(data_paths),
        }

    struct_paths = sorted(data_dir.glob(struct_pat))
    if not data_paths or not struct_paths:
        raise FileNotFoundError(f"Expected data and structure files under {data_dir}.")
    if len(data_paths) != len(struct_paths):
        raise ValueError(
            "Data/structure file counts do not match: "
            f"{len(data_paths)} vs {len(struct_paths)}"
        )

    first_replica = load_nc8_replica(data_paths[0], struct_paths[0])

    return {
        "data_dir": str(data_dir),
        "data_paths": [str(path) for path in data_paths],
        "struct_paths": [str(path) for path in struct_paths],
        "variable_names": first_replica["variable_names"],
        "first_graph": first_replica["graph"],
        "replica_count": len(data_paths),
    }


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


def aggregate_lagged_outputs(
    p_matrix: np.ndarray,
    tau_min: int,
    alpha_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lagged_p = p_matrix[:, :, tau_min:]
    min_p = lagged_p.min(axis=2)
    best_lag = lagged_p.argmin(axis=2) + tau_min
    score_matrix = -np.log10(np.clip(min_p, EPS, 1.0))
    binary_graph = (min_p <= alpha_level).astype(np.int64)
    return min_p, best_lag.astype(np.int64), score_matrix, binary_graph


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


def setting_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["mean_auroc"]),
        float(row["mean_auprc"]),
        float(row["mean_f1"]),
        -float(row["mean_shd"]),
    )


def build_readme(
    output_dir: Path,
    data_dir: str,
    tau_values: list[int],
    alpha_level: float,
    pc_alpha: float,
) -> str:
    return "\n".join(
        [
            "# PCMCI NC8 Baseline",
            "",
            f"- Repo folder: `{REPO_ROOT}`",
            f"- Results folder: `{output_dir}`",
            f"- Dataset: `{data_dir}`",
            "- Method: `PCMCI` with `ParCorr(significance=\"analytic\")`",
            "- tau_min: `1`",
            f"- tau_max settings: `{tau_values}`",
            f"- pc_alpha: `{pc_alpha}`",
            f"- alpha_level: `{alpha_level}`",
            "",
            "Contents:",
            "- `evaluation_protocol.md`: exact graph aggregation and metric definitions",
            "- `per_replica_metrics.csv`: one row per NC8 replica and tau setting",
            "- `aggregate_metrics.csv`: mean/std metrics across replicas for each tau setting",
            "- `predicted_graphs/`: per-setting per-replica score matrices, binary graphs, and raw PCMCI arrays",
            "- `logs/`: per-setting execution logs",
            "- `summary.md`: concise best-setting summary",
            "",
            "The baseline is intentionally small and reproducible: two tau settings,",
            "all available NC8 replicas, and no broader hyperparameter sweep.",
            "",
        ]
    )


def build_evaluation_protocol(alpha_level: float, tau_min: int, tau_values: list[int]) -> str:
    tau_list = ", ".join(str(value) for value in tau_values)
    return "\n".join(
        [
            "# Evaluation Protocol",
            "",
            "## Data",
            "- Dataset path: `data/NC8/` (relative to baseline root)",
            "- Replicas used: all available `nc8_data_*.csv` files with their matching `nc8_struct_*.csv` graphs",
            "- Ground truth is evaluated separately for each replica against its corresponding structure CSV",
            "",
            "## PCMCI Run",
            "- Conditional independence test: `ParCorr(significance=\"analytic\")`",
            f"- tau_min: `{tau_min}`",
            f"- tau_max settings: `{tau_list}`",
            "- pc_alpha: `0.2`",
            f"- alpha_level: `{alpha_level}`",
            "- Contemporaneous edges are excluded because the run uses `tau_min=1`",
            "- `core_runtime_mean_sec`: mean wall-clock time per replica measured around the `PCMCI.run_pcmci(...)` call only",
            "- `end_to_end_runtime_mean_sec`: mean wall-clock time per replica including NC8 CSV loading, Tigramite dataframe construction, PCMCI, lag aggregation, binary graph construction, evaluation metrics, and per-replica saved-result postprocessing",
            "",
            "## Static 8x8 Aggregation",
            "For each ordered pair `(i, j)` and each tau setting, PCMCI returns lag-specific",
            "p-values `p(i, j, tau)` for `tau in [tau_min, tau_max]`.",
            "",
            "The lagged outputs are aggregated to a single static 8x8 matrix as follows:",
            "- `min_p[i, j] = min_tau p(i, j, tau)`",
            "- `score[i, j] = -log10(max(min_p[i, j], 1e-300))`",
            "- `binary[i, j] = 1[min_p[i, j] <= alpha_level]`",
            "- `best_lag[i, j] = argmin_tau p(i, j, tau)`",
            "",
            "Interpretation:",
            "- The continuous score matrix is an aggregated lagged significance score",
            "- The binary graph is the thresholded version of that same score matrix",
            "- Diagonal entries are kept because the NC8 ground-truth adjacency contains self-lag positives",
            "",
            "## Metrics",
            "- AUROC: computed on the flattened 8x8 score matrix against the flattened 8x8 GT adjacency",
            "- AUPRC: reported as average precision on the same flattened 8x8 score matrix",
            "- F1: computed on the flattened 8x8 binary graph",
            "- SHD: directed structural Hamming distance on the 8x8 binary graph",
            "",
            "SHD details:",
            "- Diagonal mismatches count directly as one edge addition/deletion each",
            "- For off-diagonal node pairs, an exact orientation reversal counts as 1",
            "- Otherwise SHD counts edge additions/deletions entrywise",
            "",
        ]
    )


def build_summary(
    best_row: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    per_replica_rows: list[dict[str, Any]],
    data_dir: str,
    replica_count: int,
) -> str:
    lines = [
        "# PCMCI NC8 Baseline Summary",
        "",
        f"- Dataset: `{data_dir}`",
        f"- Replicas used: `{replica_count}`",
        "- Method: `PCMCI` + `ParCorr(significance=\"analytic\")`",
        "- Static score matrix: `score[i,j] = -log10(min_tau p(i,j,tau))` for tau in `[1, tau_max]`",
        "- Static binary graph: `binary[i,j] = 1[min_tau p(i,j,tau) <= alpha_level]`",
        "",
        "## Aggregate Results",
        "",
        "| tau_max | mean core runtime / replica (s) | mean end-to-end runtime / replica (s) | mean AUROC | mean AUPRC | mean F1 | mean SHD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['tau_max']} | {row['core_runtime_mean_sec']:.6f} | {row['end_to_end_runtime_mean_sec']:.6f} | {row['mean_auroc']:.6f} | "
            f"{row['mean_auprc']:.6f} | {row['mean_f1']:.6f} | {row['mean_shd']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Best Setting",
            "",
            "- Selected by highest mean AUROC, then mean AUPRC, then mean F1, then lowest mean SHD",
            f"- Best tau_max: `{best_row['tau_max']}`",
            f"- Mean AUROC: `{best_row['mean_auroc']:.6f}`",
            f"- Mean AUPRC: `{best_row['mean_auprc']:.6f}`",
            f"- Mean F1: `{best_row['mean_f1']:.6f}`",
            f"- Mean SHD: `{best_row['mean_shd']:.6f}`",
            f"- Mean core runtime per replica: `{best_row['core_runtime_mean_sec']:.6f}` seconds",
            f"- Mean end-to-end runtime per replica: `{best_row['end_to_end_runtime_mean_sec']:.6f}` seconds",
            "",
            f"Per-replica metric rows saved: `{len(per_replica_rows)}`",
            "",
        ]
    )
    return "\n".join(lines)


def save_replica_outputs(
    setting_dir: Path,
    replica_index: int,
    variable_names: list[str],
    min_p_matrix: np.ndarray,
    best_lag_matrix: np.ndarray,
    score_matrix: np.ndarray,
    binary_graph: np.ndarray,
    results: dict[str, Any],
) -> None:
    prefix = setting_dir / f"replica_{replica_index}"
    np.save(prefix.with_name(prefix.name + "_min_p_matrix.npy"), min_p_matrix)
    np.save(prefix.with_name(prefix.name + "_best_lag_matrix.npy"), best_lag_matrix)
    np.save(prefix.with_name(prefix.name + "_score_matrix.npy"), score_matrix)
    np.save(prefix.with_name(prefix.name + "_binary_graph.npy"), binary_graph)
    np.save(prefix.with_name(prefix.name + "_pcmci_graph.npy"), results["graph"])
    np.save(prefix.with_name(prefix.name + "_p_matrix.npy"), results["p_matrix"])
    np.save(prefix.with_name(prefix.name + "_val_matrix.npy"), results["val_matrix"])

    write_matrix_csv(prefix.with_name(prefix.name + "_min_p_matrix.csv"), min_p_matrix, variable_names)
    write_matrix_csv(prefix.with_name(prefix.name + "_best_lag_matrix.csv"), best_lag_matrix, variable_names)
    write_matrix_csv(prefix.with_name(prefix.name + "_score_matrix.csv"), score_matrix, variable_names)
    write_matrix_csv(prefix.with_name(prefix.name + "_binary_graph.csv"), binary_graph, variable_names)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    logs_dir = output_dir / "logs"
    predicted_root = output_dir / "predicted_graphs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predicted_root.mkdir(parents=True, exist_ok=True)

    bundle = load_nc8_bundle(args.data_dir)
    variable_names = bundle["variable_names"]
    tau_values = sorted(set(args.tau_max_values))

    config = {
        "repo_root": str(REPO_ROOT),
        "data_dir": bundle["data_dir"],
        "data_paths": bundle["data_paths"],
        "struct_paths": bundle["struct_paths"],
        "tau_min": args.tau_min,
        "tau_max_values": tau_values,
        "pc_alpha": args.pc_alpha,
        "alpha_level": args.alpha_level,
        "verbosity": args.verbosity,
        "replica_count": bundle["replica_count"],
        "aggregation": {
            "min_p": "min over lagged p-values from tau_min..tau_max",
            "score_matrix": "score[i,j] = -log10(max(min_p[i,j], 1e-300))",
            "binary_graph": "binary[i,j] = 1[min_p[i,j] <= alpha_level]",
        },
        "runtime_definitions": {
            "core_runtime_mean_sec": "wall-clock seconds per replica measured around PCMCI.run_pcmci(...) only",
            "end_to_end_runtime_mean_sec": "wall-clock seconds per replica including CSV loading, dataframe construction, PCMCI, lag aggregation, binary graph construction, evaluation metrics, and per-replica saved-result postprocessing",
        },
    }
    write_json(output_dir / "config.json", config)

    write_matrix_csv(output_dir / "ground_truth_graph_replica0.csv", bundle["first_graph"], variable_names)
    np.save(output_dir / "ground_truth_graph_replica0.npy", bundle["first_graph"])

    per_replica_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for tau_max in tau_values:
        setting_name = f"tau_max_{tau_max}"
        setting_dir = predicted_root / setting_name
        setting_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{setting_name}.log"
        log_path.write_text("", encoding="utf-8")

        append_log(log_path, f"PCMCI NC8 baseline setting: {setting_name}")
        append_log(log_path, f"Dataset directory: {bundle['data_dir']}")
        append_log(log_path, f"Replicas: {bundle['replica_count']}")
        append_log(
            log_path,
            f"tau_min={args.tau_min} tau_max={tau_max} pc_alpha={args.pc_alpha} alpha_level={args.alpha_level}",
        )

        setting_rows: list[dict[str, Any]] = []

        for replica_index, (sequence_path, struct_path) in enumerate(
            zip(bundle["data_paths"], bundle["struct_paths"])
        ):
            append_log(
                log_path,
                f"[replica {replica_index}] sequence={sequence_path} structure={struct_path}",
            )

            replica_start_time = time.perf_counter()
            replica_bundle = load_nc8_replica(
                Path(sequence_path),
                Path(struct_path),
                expected_variable_names=variable_names,
            )
            sequence = replica_bundle["sequence"]
            gt_graph = replica_bundle["graph"]

            dataframe = pp.DataFrame(sequence, var_names=variable_names)
            pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=ParCorr(significance="analytic"),
                verbosity=args.verbosity,
            )
            core_start_time = time.perf_counter()
            results = pcmci.run_pcmci(
                tau_min=args.tau_min,
                tau_max=tau_max,
                pc_alpha=args.pc_alpha,
                alpha_level=args.alpha_level,
            )
            core_runtime_sec = time.perf_counter() - core_start_time

            min_p_matrix, best_lag_matrix, score_matrix, binary_graph = aggregate_lagged_outputs(
                p_matrix=results["p_matrix"],
                tau_min=args.tau_min,
                alpha_level=args.alpha_level,
            )

            save_replica_outputs(
                setting_dir=setting_dir,
                replica_index=replica_index,
                variable_names=variable_names,
                min_p_matrix=min_p_matrix,
                best_lag_matrix=best_lag_matrix,
                score_matrix=score_matrix,
                binary_graph=binary_graph,
                results=results,
            )

            y_true = flatten_matrix(gt_graph).astype(np.int64)
            score_flat = flatten_matrix(score_matrix)
            pred_flat = flatten_matrix(binary_graph).astype(np.int64)

            auroc = compute_auroc(y_true, score_flat)
            auprc = compute_auprc(y_true, score_flat)
            f1 = compute_f1(y_true, pred_flat)
            shd = compute_shd(gt_graph, binary_graph)
            tp, fp, fn, tn = binary_classification_stats(y_true, pred_flat)
            end_to_end_runtime_sec = time.perf_counter() - replica_start_time

            row = {
                "tau_max": tau_max,
                "replica_index": replica_index,
                "sequence_path": sequence_path,
                "structure_path": struct_path,
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
            }
            per_replica_rows.append(row)
            setting_rows.append(row)

            append_log(
                log_path,
                f"[replica {replica_index}] AUROC={auroc:.6f} AUPRC={auprc:.6f} "
                f"F1={f1:.6f} SHD={shd} predicted_edges={int(binary_graph.sum())} "
                f"core_runtime_sec={core_runtime_sec:.6f} "
                f"end_to_end_runtime_sec={end_to_end_runtime_sec:.6f}",
            )

        mean_auroc = float(np.mean([row["auroc"] for row in setting_rows]))
        std_auroc = float(np.std([row["auroc"] for row in setting_rows], ddof=0))
        mean_auprc = float(np.mean([row["auprc"] for row in setting_rows]))
        std_auprc = float(np.std([row["auprc"] for row in setting_rows], ddof=0))
        mean_f1 = float(np.mean([row["f1"] for row in setting_rows]))
        std_f1 = float(np.std([row["f1"] for row in setting_rows], ddof=0))
        mean_shd = float(np.mean([row["shd"] for row in setting_rows]))
        std_shd = float(np.std([row["shd"] for row in setting_rows], ddof=0))
        mean_predicted_edges = float(np.mean([row["predicted_edges"] for row in setting_rows]))
        std_predicted_edges = float(np.std([row["predicted_edges"] for row in setting_rows], ddof=0))
        core_runtime_mean_sec = float(np.mean([row["core_runtime_sec"] for row in setting_rows]))
        core_runtime_std_sec = float(np.std([row["core_runtime_sec"] for row in setting_rows], ddof=0))
        end_to_end_runtime_mean_sec = float(
            np.mean([row["end_to_end_runtime_sec"] for row in setting_rows])
        )
        end_to_end_runtime_std_sec = float(
            np.std([row["end_to_end_runtime_sec"] for row in setting_rows], ddof=0)
        )

        aggregate_row = {
            "tau_max": tau_max,
            "replicas": len(setting_rows),
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "mean_auprc": mean_auprc,
            "std_auprc": std_auprc,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_shd": mean_shd,
            "std_shd": std_shd,
            "mean_predicted_edges": mean_predicted_edges,
            "std_predicted_edges": std_predicted_edges,
            "core_runtime_mean_sec": core_runtime_mean_sec,
            "core_runtime_std_sec": core_runtime_std_sec,
            "end_to_end_runtime_mean_sec": end_to_end_runtime_mean_sec,
            "end_to_end_runtime_std_sec": end_to_end_runtime_std_sec,
        }
        aggregate_rows.append(aggregate_row)

        append_log(
            log_path,
            f"[aggregate] mean_AUROC={mean_auroc:.6f} mean_AUPRC={mean_auprc:.6f} "
            f"mean_F1={mean_f1:.6f} mean_SHD={mean_shd:.6f} "
            f"core_runtime_mean_sec={core_runtime_mean_sec:.6f} "
            f"end_to_end_runtime_mean_sec={end_to_end_runtime_mean_sec:.6f}",
        )

    best_row = max(aggregate_rows, key=setting_sort_key)
    for row in aggregate_rows:
        row["is_best"] = int(row["tau_max"] == best_row["tau_max"])

    write_rows_csv(output_dir / "per_replica_metrics.csv", per_replica_rows)
    write_rows_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)

    (output_dir / "README_baseline.md").write_text(
        build_readme(
            output_dir=output_dir,
            data_dir=bundle["data_dir"],
            tau_values=tau_values,
            alpha_level=args.alpha_level,
            pc_alpha=args.pc_alpha,
        ),
        encoding="utf-8",
    )
    (output_dir / "evaluation_protocol.md").write_text(
        build_evaluation_protocol(
            alpha_level=args.alpha_level,
            tau_min=args.tau_min,
            tau_values=tau_values,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        build_summary(
            best_row=best_row,
            aggregate_rows=aggregate_rows,
            per_replica_rows=per_replica_rows,
            data_dir=bundle["data_dir"],
            replica_count=bundle["replica_count"],
        ),
        encoding="utf-8",
    )

    print("PCMCI NC8 baseline completed successfully.")
    print(f"Replicas used: {bundle['replica_count']}")
    print(f"Best tau_max: {best_row['tau_max']}")
    print(
        "Best aggregate metrics: "
        f"AUROC={best_row['mean_auroc']:.6f} "
        f"AUPRC={best_row['mean_auprc']:.6f} "
        f"F1={best_row['mean_f1']:.6f} "
        f"SHD={best_row['mean_shd']:.6f} "
        f"core_runtime_mean_sec={best_row['core_runtime_mean_sec']:.6f} "
        f"end_to_end_runtime_mean_sec={best_row['end_to_end_runtime_mean_sec']:.6f}"
    )
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
