#!/usr/bin/env python
"""Run a reproducible TCDF baseline on all NC8 replicas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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

import TCDF  # noqa: E402


DEFAULT_DATA_DIR = Path("/storage/home/ydk297/projects/meta_causal_discovery/data/nc8")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results_nc8_baseline"
T_CRIT_95_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--hidden-layers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--dilation-coefficient", type=int, default=4)
    parser.add_argument("--significance", type=float, default=0.8)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--cuda", action="store_true", default=True)
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
        _sh, struct_matrix = read_csv_matrix(struct_path)

    variable_names = expected_variable_names or data_header or [f"X{i}" for i in range(data_matrix.shape[1])]

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
    random.seed(seed)
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


def run_tcdf_on_file(file_path: Path, columns: list[str], args: argparse.Namespace) -> dict[str, Any]:
    num_vars = len(columns)
    score_matrix = np.zeros((num_vars, num_vars), dtype=np.float64)
    binary_graph = np.zeros((num_vars, num_vars), dtype=np.int64)
    delay_matrix = np.full((num_vars, num_vars), -1, dtype=np.int64)
    final_loss_vector = np.zeros(num_vars, dtype=np.float64)
    validated_causes_by_target: dict[int, list[int]] = {}

    core_start = time.perf_counter()
    for effect_idx, target in enumerate(columns):
        causes, causeswithdelay, realloss, scores = TCDF.findcauses(
            target=target,
            cuda=args.cuda,
            epochs=args.epochs,
            kernel_size=args.kernel_size,
            layers=args.hidden_layers + 1,
            log_interval=args.log_interval,
            lr=args.learning_rate,
            optimizername=args.optimizer,
            seed=args.seed,
            dilation_c=args.dilation_coefficient,
            significance=args.significance,
            file=str(file_path),
        )

        score_vector = np.asarray(scores, dtype=np.float64)
        if score_vector.shape[0] != num_vars:
            raise ValueError(
                f"TCDF returned {score_vector.shape[0]} scores for target {target}, expected {num_vars}."
            )

        score_matrix[effect_idx, :] = score_vector
        final_loss_vector[effect_idx] = float(realloss)
        validated_causes_by_target[effect_idx] = [int(cause) for cause in causes]
        for cause_idx in causes:
            binary_graph[effect_idx, int(cause_idx)] = 1

        for (effect_key, cause_key), delay in causeswithdelay.items():
            delay_matrix[int(effect_key), int(cause_key)] = int(delay)

    core_runtime_sec = time.perf_counter() - core_start
    return {
        "score_matrix": score_matrix,
        "binary_graph": binary_graph,
        "delay_matrix": delay_matrix,
        "final_loss_vector": final_loss_vector,
        "validated_causes_by_target": validated_causes_by_target,
        "core_runtime_sec": core_runtime_sec,
    }


def save_replica_outputs(
    predicted_dir: Path,
    replica_index: int,
    variable_names: list[str],
    score_matrix: np.ndarray,
    binary_graph: np.ndarray,
    delay_matrix: np.ndarray,
    final_loss_vector: np.ndarray,
    validated_causes_by_target: dict[int, list[int]],
) -> None:
    prefix = predicted_dir / f"replica_{replica_index}"
    np.save(prefix.with_name(prefix.name + "_score_matrix.npy"), score_matrix)
    np.save(prefix.with_name(prefix.name + "_binary_graph.npy"), binary_graph)
    np.save(prefix.with_name(prefix.name + "_delay_matrix.npy"), delay_matrix)
    np.save(prefix.with_name(prefix.name + "_final_loss_vector.npy"), final_loss_vector)

    write_matrix_csv(prefix.with_name(prefix.name + "_score_matrix.csv"), score_matrix, variable_names)
    write_matrix_csv(prefix.with_name(prefix.name + "_binary_graph.csv"), binary_graph, variable_names)
    write_matrix_csv(prefix.with_name(prefix.name + "_delay_matrix.csv"), delay_matrix, variable_names)

    cause_payload = {
        variable_names[target_idx]: [variable_names[cause_idx] for cause_idx in cause_list]
        for target_idx, cause_list in validated_causes_by_target.items()
    }
    prefix.with_name(prefix.name + "_validated_causes.json").write_text(
        json.dumps(cause_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_readme(output_dir: Path, data_dir: str, args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# TCDF NC8 Baseline",
            "",
            f"- Repo folder: `{REPO_ROOT}`",
            f"- Results folder: `{output_dir}`",
            f"- Dataset: `{data_dir}`",
            "- Method: `TCDF` from the official repository",
            f"- epochs: `{args.epochs}`",
            f"- kernel_size: `{args.kernel_size}`",
            f"- hidden_layers: `{args.hidden_layers}`",
            f"- learning_rate: `{args.learning_rate}`",
            f"- optimizer: `{args.optimizer}`",
            f"- log_interval: `{args.log_interval}`",
            f"- seed: `{args.seed}`",
            f"- dilation_coefficient: `{args.dilation_coefficient}`",
            f"- significance: `{args.significance}`",
            f"- torch_threads: `{args.torch_threads}`",
            f"- cuda: `{args.cuda}`",
            "",
            "Contents:",
            "- `method_note.md`: exact repo code path used for TCDF",
            "- `data_inspection.md`: NC8 replica layout and selected data path",
            "- `evaluation_protocol.md`: exact score, binary graph, delay, and metric definitions",
            "- `per_replica_metrics.csv`: one row per NC8 replica",
            "- `final_results_table.csv` and `.md`: paper-facing aggregate metrics",
            "- `final_summary_for_paper.md`: concise paper-ready summary",
            "- `predicted_graphs/`: per-replica score matrices, binary graphs, delay matrices, and validated causes",
            "- `logs/`: stdout, stderr, and per-replica execution log",
            "",
        ]
    )


def build_method_note(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Method Note",
            "",
            "Selected implementation path from the official TCDF repository:",
            "- CLI entry point described by the repo: `runTCDF.py`",
            "- Core discovery function actually used by the CLI: `TCDF.py::findcauses`",
            "- Model class used internally: `model.py::ADDSTCN`",
            "",
            "Exact path used in this baseline:",
            "- The NC8 runner reproduces the per-target loop from `runTCDF.py::runTCDF(...)`.",
            "- For each target variable, it calls `TCDF.findcauses(...)` directly with the same parameterization style exposed by the official CLI.",
            "- Direct use of `TCDF.findcauses(...)` is necessary because importing `runTCDF.py` would trigger its top-level argument parser.",
            "",
            "Official-style configuration used:",
            f"- `epochs={args.epochs}`",
            f"- `kernel_size={args.kernel_size}`",
            f"- `hidden_layers={args.hidden_layers}` which implies `layers={args.hidden_layers + 1}` in `findcauses(...)`",
            f"- `learning_rate={args.learning_rate}`",
            f"- `optimizer={args.optimizer}`",
            f"- `log_interval={args.log_interval}`",
            f"- `seed={args.seed}`",
            f"- `dilation_coefficient={args.dilation_coefficient}`",
            f"- `significance={args.significance}`",
            f"- `cuda={args.cuda}`",
            "",
            "Why this path was selected:",
            "- It is the official repository's own discovery pipeline rather than a reimplementation of the model.",
            "- It preserves the repo's attention-based candidate selection, PIVM validation, and delay extraction logic.",
            "- The baseline uses one clean default-style configuration from the official CLI instead of a broad sweep.",
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
            "- This is the same NC8 path used for the PCMCI and cMLP baselines, which keeps cross-baseline comparison aligned.",
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
            "## TCDF Run",
            "- Discovery path: `runTCDF.py` logic reproduced via direct calls to `TCDF.findcauses(...)`",
            f"- epochs: `{args.epochs}`",
            f"- kernel_size: `{args.kernel_size}`",
            f"- hidden_layers: `{args.hidden_layers}`",
            f"- effective convolution levels passed to `findcauses(...)`: `{args.hidden_layers + 1}`",
            f"- learning_rate: `{args.learning_rate}`",
            f"- optimizer: `{args.optimizer}`",
            f"- log_interval: `{args.log_interval}`",
            f"- seed: `{args.seed}`",
            f"- dilation_coefficient: `{args.dilation_coefficient}`",
            f"- significance: `{args.significance}`",
            f"- cpu thread setting: `torch.set_num_threads({args.torch_threads})` and `torch.set_num_interop_threads(1)`",
            f"- cuda: `{args.cuda}`",
            "",
            "## Continuous Score Matrix",
            "- For each target variable, `TCDF.findcauses(...)` returns a length-`p` list of raw attention scores from the trained model",
            "- These are the pre-softmax attention parameters `fs_attention` returned by TCDF after training",
            "- Final static 8x8 score matrix definition: `score[i, j] = raw_attention_score(j -> i)`",
            "- This is the closest official confidence-like signal available from the repository for ranking edges",
            "",
            "## Binary Graph",
            "- TCDF first proposes potential causes from the raw attention scores using the repository's gap heuristic with a `score >= 1` cutoff",
            "- TCDF then applies its validation step (PIVM) by permuting each candidate cause and checking whether the loss increase is significant enough",
            f"- A binary edge is present iff source `j` remains a validated cause of target `i` after this official validation step, using `significance = {args.significance}`",
            "- Final binary graph definition: `binary[i, j] = 1` iff `j` is in the validated cause list returned by `TCDF.findcauses(...)` for target `i`",
            "",
            "## Delay Handling",
            "- For each validated edge, TCDF returns one discovered delay by interpreting the trained convolutional kernel weights across layers",
            "- Because the NC8 GT used here is a static 8x8 adjacency without lag-specific labels, delays are saved for inspection but are not used to expand the graph into multiple lag-specific edges",
            "- Delay aggregation rule for evaluation: any validated edge counts as present in the static graph regardless of the discovered delay value",
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
            "- `core_runtime_mean_sec`: mean wall-clock time per replica for the TCDF discovery loop itself, from the first per-target `findcauses(...)` call through the final target's returned outputs",
            "- `end_to_end_runtime_mean_sec`: mean wall-clock time per replica including CSV loading, model setup, TCDF discovery, graph extraction, delay aggregation, metric computation, and saved-result postprocessing",
            "- For comparison against a full model runtime, `end_to_end_runtime_mean_sec` is the more appropriate number",
            "",
        ]
    )


def build_final_results_table_md(table_row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final TCDF Results",
            "",
            "Results are reported across the `R=5` NC8 replicas as mean `±` 95% confidence interval.",
            "`core_runtime_mean_sec` is the mean wall-clock time per replica for the TCDF discovery loop only.",
            "`end_to_end_runtime_mean_sec` is the mean wall-clock time per replica for the full per-replica baseline pipeline.",
            "",
            "| method | epochs | kernel_size | hidden_layers | learning_rate | optimizer | significance | dilation_coefficient | replicas_used | AUROC_mean | AUROC_95CI | AUPRC_mean | AUPRC_95CI | F1_mean | F1_95CI | SHD_mean | SHD_95CI | core_runtime_mean_sec | end_to_end_runtime_mean_sec |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {table_row['method']} | {table_row['epochs']} | {table_row['kernel_size']} | {table_row['hidden_layers']} | {table_row['learning_rate']:.6f} | {table_row['optimizer']} | {table_row['significance']:.6f} | {table_row['dilation_coefficient']} | {table_row['replicas_used']} | {table_row['AUROC_mean']:.6f} | {table_row['AUROC_95CI']} | {table_row['AUPRC_mean']:.6f} | {table_row['AUPRC_95CI']} | {table_row['F1_mean']:.6f} | {table_row['F1_95CI']} | {table_row['SHD_mean']:.6f} | {table_row['SHD_95CI']} | {table_row['core_runtime_mean_sec']:.6f} | {table_row['end_to_end_runtime_mean_sec']:.6f} |",
            "",
            "Notes:",
            "- 95% CI is computed as `t_(0.975, df=4) * s / sqrt(5)` with `t_(0.975,4) = 2.776445`.",
            "- Runtime columns are mean wall-clock seconds per replica only; runtime CI is not included in this table.",
            "- This row uses official-style TCDF only; no method substitution or broad hyperparameter sweep was added.",
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
            "# TCDF Baseline Summary For Paper",
            "",
            "## Setup",
            "",
            f"- NC8 uses `R=5` replicas from `{data_dir}`.",
            "- Results are reported as mean `±` 95% confidence interval across the 5 replicas.",
            "- The baseline method is TCDF only from the official repository.",
            "- No method substitution was used.",
            "- No large hyperparameter sweep was performed.",
            f"- Main TCDF baseline setting: `epochs={table_row['epochs']}`, `kernel_size={table_row['kernel_size']}`, `hidden_layers={table_row['hidden_layers']}`, `learning_rate={table_row['learning_rate']:.6f}`, `optimizer={table_row['optimizer']}`, `significance={table_row['significance']:.6f}`, `dilation_coefficient={table_row['dilation_coefficient']}`.",
            "",
            "## Evaluation Protocol",
            "",
            "- Score matrix definition: `score[i, j] = raw_attention_score(j -> i)` from the trained TCDF model output returned by `TCDF.findcauses(...)`.",
            "- Binary graph definition: `binary[i, j] = 1` iff source `j` is a validated cause of target `i` after TCDF's official candidate-selection and PIVM validation steps.",
            "- Delay handling: discovered delays are saved for each validated edge, but evaluation is performed on a static 8x8 graph, so any validated delayed edge counts as present regardless of delay magnitude.",
            "- Diagonal entries are kept because the NC8 GT adjacency includes self-lag positives (`diag_sum = 4`).",
            "- The GT graph is the provided `nc8_struct_*.csv` adjacency for each replica.",
            "- `core_runtime_mean_sec` is the mean wall-clock time per replica for the TCDF discovery loop only.",
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


def main() -> int:
    args = parse_args()
    configure_torch_threads(args.torch_threads)

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
        "method": "TCDF",
        "epochs": args.epochs,
        "kernel_size": args.kernel_size,
        "hidden_layers": args.hidden_layers,
        "effective_layers": args.hidden_layers + 1,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "log_interval": args.log_interval,
        "seed": args.seed,
        "dilation_coefficient": args.dilation_coefficient,
        "significance": args.significance,
        "torch_threads": args.torch_threads,
        "cuda": args.cuda,
        "replica_count": bundle["replica_count"],
        "score_definition": "score[i,j] = raw attention score returned by TCDF.findcauses(...) for source j when predicting target i",
        "binary_definition": "binary[i,j] = 1 iff source j is a validated cause returned by TCDF.findcauses(...) for target i",
        "delay_definition": "delay[i,j] = discovered total delay for validated edge j -> i, else -1",
        "runtime_definitions": {
            "core_runtime_mean_sec": "mean wall-clock seconds per replica for the TCDF discovery loop only",
            "end_to_end_runtime_mean_sec": "mean wall-clock seconds per replica for loading, discovery, metric computation, and result saving",
        },
    }
    write_json(output_dir / "config.json", config)

    write_matrix_csv(output_dir / "ground_truth_graph_replica0.csv", bundle["first_graph"], variable_names)
    np.save(output_dir / "ground_truth_graph_replica0.npy", bundle["first_graph"])

    log_path = logs_dir / "baseline.log"
    log_path.write_text("", encoding="utf-8")
    append_log(log_path, "TCDF NC8 baseline")
    append_log(log_path, f"Data directory: {bundle['data_dir']}")
    append_log(log_path, f"Replicas: {bundle['replica_count']}")
    append_log(
        log_path,
        " ".join(
            [
                f"epochs={args.epochs}",
                f"kernel_size={args.kernel_size}",
                f"hidden_layers={args.hidden_layers}",
                f"learning_rate={args.learning_rate}",
                f"optimizer={args.optimizer}",
                f"log_interval={args.log_interval}",
                f"seed={args.seed}",
                f"dilation_coefficient={args.dilation_coefficient}",
                f"significance={args.significance}",
                f"torch_threads={args.torch_threads}",
                f"cuda={args.cuda}",
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

        set_random_seed(args.seed)
        run_result = run_tcdf_on_file(data_path, variable_names, args)

        score_matrix = run_result["score_matrix"]
        binary_graph = run_result["binary_graph"]
        delay_matrix = run_result["delay_matrix"]
        final_loss_vector = run_result["final_loss_vector"]
        validated_causes_by_target = run_result["validated_causes_by_target"]
        core_runtime_sec = float(run_result["core_runtime_sec"])

        save_replica_outputs(
            predicted_dir=predicted_dir,
            replica_index=replica_index,
            variable_names=variable_names,
            score_matrix=score_matrix,
            binary_graph=binary_graph,
            delay_matrix=delay_matrix,
            final_loss_vector=final_loss_vector,
            validated_causes_by_target=validated_causes_by_target,
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
            "core_runtime_sec": core_runtime_sec,
            "end_to_end_runtime_sec": float(end_to_end_runtime_sec),
            "validated_edge_count": int(np.sum(binary_graph)),
            "delay_edge_count": int(np.sum(delay_matrix >= 0)),
            "mean_final_loss": float(np.mean(final_loss_vector)),
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
        "method": "TCDF",
        "epochs": args.epochs,
        "kernel_size": args.kernel_size,
        "hidden_layers": args.hidden_layers,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "significance": args.significance,
        "dilation_coefficient": args.dilation_coefficient,
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

    print("TCDF NC8 baseline completed successfully.")
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
