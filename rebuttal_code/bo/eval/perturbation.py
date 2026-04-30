"""
Parent-pool leave-one-out (LOO) perturbation on the BO-selected graph.
"""

import csv
import json
import os
import numpy as np


def _write_csv(path, perturbation):
    d = perturbation.shape[0]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + [str(j) for j in range(d)])
        for i in range(d):
            writer.writerow([str(i)] + [float(perturbation[i, j]) for j in range(d)])


def _append_trace_check(run_dir, lines):
    if not run_dir:
        return
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "trace_check.txt")
    with open(path, "a") as f:
        for line in lines:
            f.write(str(line) + "\n")
        f.write("\n")


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _percentiles(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {
            "min": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _print_summary(perturbation, npy_path, csv_path):
    d = perturbation.shape[0]
    w_min = float(np.min(perturbation))
    w_max = float(np.max(perturbation))
    w_mean = float(np.mean(perturbation))
    print(
        "perturbation_parent_loo: "
        f"{npy_path}, {csv_path} "
        f"min={w_min:.6f} max={w_max:.6f} mean={w_mean:.6f}"
    )

    edges = [(i, j, float(perturbation[i, j]))
             for j in range(d) for i in range(d) if i != j]
    edges.sort(key=lambda x: x[2], reverse=True)
    topk = edges[:20]
    print("top-20 edges [parent_loo] (i,j,value)")
    for i, j, v in topk:
        print(f"  {i},{j},{v:.6f}")


def compute_parent_loo_perturbation(
    scorer,
    parent_adj,
    run_dir,
    *,
    eps=1e-12,
    GT=None,
):
    """
    Parent-pool leave-one-out (LOO) perturbation conditioned on the BO-selected graph.

    For each node j:
      - parents_j = {i | parent_adj[i,j] == 1, i != j}
      - Fit full model with parents_j -> MSE_base
      - For each i in parents_j:
          Fit without i -> MSE_minus
          perturbation[i,j] = log(MSE_minus + eps) - log(MSE_base + eps)
      - Non-parents have perturbation = 0
      - Diagonal must be zeroed
    """
    parent_adj = np.asarray(parent_adj)
    if parent_adj.ndim != 2 or parent_adj.shape[0] != parent_adj.shape[1]:
        raise ValueError(f"parent_adj must be square 2D, got shape {parent_adj.shape}")

    d = int(parent_adj.shape[0])
    parent_bin = (parent_adj != 0).astype(int)
    np.fill_diagonal(parent_bin, 0)
    GT_bin = None
    if GT is not None:
        GT_bin = (np.asarray(GT) != 0).astype(int)
        if GT_bin.shape != parent_bin.shape:
            raise ValueError(f"GT shape mismatch: GT={GT_bin.shape}, parent_adj={parent_bin.shape}")

    if os.getenv("TS_TRACE_MSE") == "1":
        lines = ["[EDGE_IMPORTANCE_TRACE] TS_TRACE_MSE=1"]
        try:
            if d <= 0:
                lines.append("[ALERT] d <= 0")
            else:
                j = 0
                parents_full = list(np.where(parent_bin[:, j] == 1)[0])
                if len(parents_full) == 0:
                    lines.append("[ALERT] parents_full empty")
                else:
                    removed_parent = parents_full[0]
                    parents_minus = [p for p in parents_full if p != removed_parent]

                    mse_base_dbg = scorer._fit_node(j, parents_full)
                    mse_minus_dbg = scorer._fit_node(j, parents_minus)
                    delta = mse_minus_dbg - mse_base_dbg

                    n = getattr(scorer, "n", None)
                    if n is None:
                        Xp = getattr(scorer, "Xp", None)
                        if Xp is None:
                            Xp = getattr(scorer, "Xp_t", None)
                        if Xp is not None and hasattr(Xp, "shape"):
                            n = int(Xp.shape[0])
                    effective_samples = n if n is not None else "unknown"

                    full_shape = "unknown"
                    minus_shape = "unknown"
                    if hasattr(scorer, "_get_parent_columns"):
                        cols_full = scorer._get_parent_columns(parents_full)
                        cols_minus = scorer._get_parent_columns(parents_minus)
                        full_shape = (effective_samples, int(len(cols_full)))
                        minus_shape = (effective_samples, int(len(cols_minus)))

                    lines.append(f"node={j} effective_samples={effective_samples}")
                    lines.append(f"parents_full={parents_full} removed_parent={removed_parent}")
                    lines.append(f"X_full_shape={full_shape} X_minus_shape={minus_shape}")
                    lines.append(
                        f"mse_base={float(mse_base_dbg):.6g} "
                        f"mse_minus={float(mse_minus_dbg):.6g} "
                        f"delta={float(delta):.6g}"
                    )
                    if abs(delta) < 1e-8:
                        lines.append("[ALERT] |delta| < 1e-8")
        except Exception as e:
            lines.append(f"[ALERT] trace_failed: {e}")
        _append_trace_check(run_dir, lines)

    perturbation = np.zeros((d, d), dtype=np.float64)
    trace_rows = []
    zero_delta_examples = []
    for j in range(d):
        parents_j = list(np.where(parent_bin[:, j] == 1)[0])
        if len(parents_j) == 0:
            continue
        mse_base = scorer._fit_node(j, parents_j)
        log_mse_base = np.log(mse_base + eps)

        for i in parents_j:
            parents_minus = [p for p in parents_j if p != i]
            mse_minus = scorer._fit_node(j, parents_minus)
            log_mse_minus = np.log(mse_minus + eps)
            weight = log_mse_minus - log_mse_base
            perturbation[i, j] = weight
            delta_mse = float(mse_minus - mse_base)
            is_exact_zero = bool(weight == 0.0)
            is_near_zero = bool(abs(weight) <= 1e-12)
            trace_row = {
                "parent": int(i),
                "child": int(j),
                "is_gt_edge": None if GT_bin is None else int(GT_bin[i, j]),
                "num_parents_full": int(len(parents_j)),
                "mse_base": float(mse_base),
                "mse_minus": float(mse_minus),
                "delta_mse": delta_mse,
                "log_mse_base": float(log_mse_base),
                "log_mse_minus": float(log_mse_minus),
                "weight": float(weight),
                "is_exact_zero": int(is_exact_zero),
                "is_near_zero": int(is_near_zero),
            }
            trace_rows.append(trace_row)
            if is_near_zero and len(zero_delta_examples) < 10:
                zero_delta_examples.append({
                    **trace_row,
                    "parents_full": [int(p) for p in parents_j],
                    "parents_minus": [int(p) for p in parents_minus],
                })

    np.fill_diagonal(perturbation, 0.0)

    mask = ~np.eye(d, dtype=bool)
    offdiag_weights = perturbation[mask]
    parent_offdiag = parent_bin[mask].astype(bool)
    selected_weights = offdiag_weights[parent_offdiag]
    nonparent_weights = offdiag_weights[~parent_offdiag]
    zero_total = int(np.sum(offdiag_weights == 0.0))
    zero_selected = int(np.sum(selected_weights == 0.0)) if selected_weights.size > 0 else 0
    zero_nonparent = int(np.sum(nonparent_weights == 0.0)) if nonparent_weights.size > 0 else 0

    zero_summary = {
        "offdiag_total": int(offdiag_weights.size),
        "offdiag_zero_count": int(zero_total),
        "offdiag_zero_ratio": float(zero_total / max(int(offdiag_weights.size), 1)),
        "selected_parent_edges": int(selected_weights.size),
        "selected_parent_zero_count": int(zero_selected),
        "selected_parent_zero_ratio": float(zero_selected / max(int(selected_weights.size), 1)),
        "nonparent_edges": int(nonparent_weights.size),
        "nonparent_zero_count": int(zero_nonparent),
        "nonparent_zero_ratio": float(zero_nonparent / max(int(nonparent_weights.size), 1)),
        "offdiag_weight_percentiles": _percentiles(offdiag_weights),
        "selected_parent_weight_percentiles": _percentiles(selected_weights),
        "trace_row_count": int(len(trace_rows)),
    }
    if GT_bin is not None:
        gt_true_mask = (GT_bin == 1) & mask
        gt_zero_mask = gt_true_mask & (perturbation == 0.0)
        gt_missing_parent_zero = gt_true_mask & (parent_bin == 0) & (perturbation == 0.0)
        gt_selected_parent_zero = gt_true_mask & (parent_bin == 1) & (perturbation == 0.0)
        zero_summary.update({
            "gt_true_edges_offdiag": int(np.sum(gt_true_mask)),
            "gt_true_zero_count": int(np.sum(gt_zero_mask)),
            "gt_true_zero_ratio": float(np.sum(gt_zero_mask) / max(int(np.sum(gt_true_mask)), 1)),
            "gt_true_zero_due_to_missing_from_best_graph": int(np.sum(gt_missing_parent_zero)),
            "gt_true_zero_within_selected_parents": int(np.sum(gt_selected_parent_zero)),
            "gt_true_weight_percentiles": _percentiles(perturbation[gt_true_mask]),
            "gt_false_weight_percentiles": _percentiles(perturbation[(GT_bin == 0) & mask]),
        })

    if zero_selected == 0 and zero_total == int(nonparent_weights.size):
        zero_summary["zero_source_hint"] = (
            "All exact zeros come from edges absent from best_graph; no clamp/rounding path is zeroing selected parents."
        )
    elif zero_selected > 0:
        zero_summary["zero_source_hint"] = (
            "Some selected parent edges have near-zero LOO effect; inspect parent_loo_trace.csv and parent_loo_zero_examples.json."
        )
    else:
        zero_summary["zero_source_hint"] = (
            "Zeros are dominated by edges left at initialization because parent-LOO only scores edges present in best_graph."
        )

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        npy_path = os.path.join(run_dir, "perturbation_parent_loo.npy")
        csv_path = os.path.join(run_dir, "perturbation_parent_loo.csv")
        np.save(npy_path, perturbation)
        _write_csv(csv_path, perturbation)
        trace_path = os.path.join(run_dir, "parent_loo_trace.csv")
        with open(trace_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "parent",
                "child",
                "is_gt_edge",
                "num_parents_full",
                "mse_base",
                "mse_minus",
                "delta_mse",
                "log_mse_base",
                "log_mse_minus",
                "weight",
                "is_exact_zero",
                "is_near_zero",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace_rows)
        _write_json(os.path.join(run_dir, "parent_loo_summary.json"), zero_summary)
        _write_json(os.path.join(run_dir, "parent_loo_zero_examples.json"), {
            "zero_delta_examples": zero_delta_examples,
            "note": zero_summary.get("zero_source_hint"),
        })
    else:
        npy_path = "<no-save>"
        csv_path = "<no-save>"

    print(
        "[PARENT_LOO_TRACE] "
        f"offdiag_zero_ratio={zero_summary['offdiag_zero_ratio']:.6f} "
        f"selected_parent_zero_ratio={zero_summary['selected_parent_zero_ratio']:.6f}"
    )
    if GT_bin is not None:
        print(
            "[PARENT_LOO_TRACE] "
            f"gt_true_zero_count={zero_summary['gt_true_zero_count']} "
            f"gt_true_zero_due_to_missing_from_best_graph={zero_summary['gt_true_zero_due_to_missing_from_best_graph']} "
            f"gt_true_zero_within_selected_parents={zero_summary['gt_true_zero_within_selected_parents']}"
        )
        print(
            "[PARENT_LOO_TRACE] "
            f"gt_true_p50={zero_summary['gt_true_weight_percentiles']['p50']} "
            f"gt_true_p90={zero_summary['gt_true_weight_percentiles']['p90']} "
            f"gt_true_p99={zero_summary['gt_true_weight_percentiles']['p99']}"
        )
        print(
            "[PARENT_LOO_TRACE] "
            f"gt_false_p50={zero_summary['gt_false_weight_percentiles']['p50']} "
            f"gt_false_p90={zero_summary['gt_false_weight_percentiles']['p90']} "
            f"gt_false_p99={zero_summary['gt_false_weight_percentiles']['p99']}"
        )
    print(f"[PARENT_LOO_TRACE] zero_source_hint={zero_summary['zero_source_hint']}")

    _print_summary(perturbation, npy_path, csv_path)
    return perturbation
