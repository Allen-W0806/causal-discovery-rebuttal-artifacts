"""

1. ACC: scan all thresholds, find the one maximizing accuracy  
2. F1/SHD: computed at the best-ACC threshold
3. All metrics ignore diagonal elements

"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


_TS_TRACE_PRINTED = False


def _append_trace_check(run_dir, lines):
    if not run_dir:
        return False
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "trace_check.txt")
    with open(path, "a") as f:
        for line in lines:
            f.write(str(line) + "\n")
        f.write("\n")
    return True


def _summarize_scores(scores):
    arr = np.asarray(scores).ravel()
    num_nan = int(np.isnan(arr).sum())
    num_inf = int(np.isinf(arr).sum())
    num_zero = int((arr == 0).sum())
    finite = np.isfinite(arr)
    arr_f = arr[finite]
    if arr_f.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "unique_rounded": 0,
            "percentiles": [float("nan")] * 7,
            "num_nan": num_nan,
            "num_inf": num_inf,
            "num_zero": num_zero,
            "near_constant": True,
        }
    uniq = np.unique(np.round(arr_f, 6))
    percentiles = np.percentile(arr_f, [0, 1, 5, 50, 95, 99, 100]).tolist()
    near_constant = (arr_f.max() == arr_f.min()) or (len(uniq) <= 2)
    return {
        "min": float(arr_f.min()),
        "max": float(arr_f.max()),
        "mean": float(arr_f.mean()),
        "std": float(arr_f.std()),
        "unique_rounded": int(len(uniq)),
        "percentiles": [float(x) for x in percentiles],
        "num_nan": num_nan,
        "num_inf": num_inf,
        "num_zero": num_zero,
        "near_constant": bool(near_constant),
    }


def _evaluation_mask(d: int, ignore_diag: bool) -> np.ndarray:
    mask = ~np.eye(d, dtype=bool) if ignore_diag else np.ones((d, d), dtype=bool)
    if ignore_diag:
        assert not bool(np.diag(mask).any()), "Diagonal entries must be excluded from evaluation."
        assert int(mask.sum()) == int(d * max(d - 1, 0)), (
            "Unexpected evaluation mask size while excluding diagonal entries."
        )
    return mask


def to_binary_adj(A, GT=None, method="threshold", threshold=0.0, zero_diag=False):
    A_arr = np.asarray(A)
    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A_arr.shape}")
    d = A_arr.shape[0]
    if A_arr.shape[1] != d:
        raise ValueError(f"A must be square, got shape {A_arr.shape}")

    if method == "threshold":
        A_bin = (A_arr > threshold).astype(int)
    elif method == "topk_gt":
        if GT is None:
            raise ValueError("GT is required for method='topk_gt'")
        GT_arr = (np.asarray(GT) != 0).astype(int)
        if GT_arr.shape != A_arr.shape:
            raise ValueError("GT shape must match A shape")
        mask = np.ones_like(A_arr, dtype=bool)
        if zero_diag:
            mask = mask & ~np.eye(d, dtype=bool)
        scores = A_arr[mask]
        k = int(GT_arr[mask].sum())
        A_bin = np.zeros_like(A_arr, dtype=int)
        if k > 0 and scores.size > 0:
            k = min(k, scores.size)
            top_idx = np.argpartition(scores, -k)[-k:]
            flat_mask_idx = np.flatnonzero(mask)
            A_bin.flat[flat_mask_idx[top_idx]] = 1
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    if zero_diag:
        np.fill_diagonal(A_bin, 0)
    return A_bin


def _binary_metrics(A_pred, A_gt, mask):
    tp = int(((A_pred == 1) & (A_gt == 1) & mask).sum())
    fp = int(((A_pred == 1) & (A_gt == 0) & mask).sum())
    fn = int(((A_pred == 0) & (A_gt == 1) & mask).sum())
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = (2 * tp + fp + fn)
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    shd = fp + fn
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'fdr': float(fdr),
        'tpr': float(tpr),
        'f1': float(f1),
        'shd': float(shd),
    }


def compute_graph_metrics(GT, A_pred, ignore_diag=True):
    A_gt = to_binary_adj(GT, method="threshold", threshold=0.0, zero_diag=ignore_diag)
    A_pred = to_binary_adj(A_pred, method="threshold", threshold=0.0, zero_diag=ignore_diag)
    d = A_gt.shape[0]
    mask = _evaluation_mask(d, ignore_diag)
    if ignore_diag:
        assert not bool(np.diag(A_gt).any()), "Ground-truth diagonal must be zero before metric aggregation."
        assert not bool(np.diag(A_pred).any()), "Predicted diagonal must be zero before metric aggregation."
    metrics = _binary_metrics(A_pred, A_gt, mask)
    metrics['diag_edges'] = int(np.diag(A_pred).sum())
    metrics['total_edges'] = int(A_pred.sum())
    metrics["ACC"] = float((A_pred[mask] == A_gt[mask]).mean())
    metrics["TP"] = metrics["tp"]
    metrics["FP"] = metrics["fp"]
    metrics["FN"] = metrics["fn"]
    metrics["FDR"] = metrics["fdr"]
    metrics["TPR"] = metrics["tpr"]
    metrics["F1"] = metrics["f1"]
    metrics["SHD"] = metrics["shd"]
    return metrics


def evaluate_uncle_style(W_hat, GT, ignore_diag=True, run_dir=None):
    """
    Parameters
    W_hat : ndarray, shape (d, d)
        Weighted summary graph. Signed importance of edge i->j.
    GT : ndarray, shape (d, d)
        Ground truth binary adjacency.
    ignore_diag : bool
        Whether to ignore diagonal (self-loops). 
    
    Returns

    results : dict
        AUROC, AUPRC, ACC, F1, SHD, best_threshold, TP, FP, FN, TPR, FDR
    """
    d = GT.shape[0]
    GT_bin = (np.asarray(GT) != 0).astype(int)

    mask = _evaluation_mask(d, ignore_diag)
    y_true = GT_bin[mask].astype(int)

    edge_scores = np.asarray(W_hat).copy()
    if ignore_diag:
        np.fill_diagonal(edge_scores, 0.0)
        assert not bool(np.diag(edge_scores).any()), "Diagonal scores must be zeroed before weighted evaluation."
    scores = edge_scores[mask]
    if ignore_diag:
        assert y_true.shape[0] == d * (d - 1), "Diagonal entries leaked into weighted evaluation labels."
        assert scores.shape[0] == d * (d - 1), "Diagonal entries leaked into weighted evaluation scores."

    # Debug stats for AUROC/AUPRC inputs
    stats = _summarize_scores(scores)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    pos_rate = float(pos / max(pos + neg, 1))
    trace_lines = [
        "[AUROC_AUPRC_TRACE]",
        f"W_shape={edge_scores.shape} W_dtype={edge_scores.dtype}",
        f"scores_shape={scores.shape} scores_dtype={scores.dtype}",
        f"min={stats['min']:.6g} max={stats['max']:.6g} mean={stats['mean']:.6g} std={stats['std']:.6g}",
        f"unique_count(rounded)={stats['unique_rounded']}",
        "percentiles(0,1,5,50,95,99,100)=" + ",".join(f"{p:.6g}" for p in stats['percentiles']),
        f"num_nan={stats['num_nan']} num_inf={stats['num_inf']} num_zero={stats['num_zero']}",
        f"y_true_pos={pos} y_true_neg={neg} pos_rate={pos_rate:.6g}",
    ]
    if stats["near_constant"]:
        trace_lines.append("[ALERT] W scores near-constant")
    wrote = _append_trace_check(run_dir, trace_lines)
    global _TS_TRACE_PRINTED
    if wrote and not _TS_TRACE_PRINTED:
        print("[TS_TRACE] wrote trace_check.txt")
        _TS_TRACE_PRINTED = True
    
    # AUROC / AUPRC
    if len(np.unique(y_true)) < 2:
        auroc = 0.5
        auprc = float(y_true.mean())
    else:
        try:
            auroc = roc_auc_score(y_true, scores)
        except ValueError:
            auroc = 0.5
        try:
            auprc = average_precision_score(y_true, scores)
        except ValueError:
            auprc = 0.0
    
    # ACC sweep
    unique_thresholds = np.unique(scores)
    thresholds = np.concatenate([[0.0], unique_thresholds,
                                  [np.max(scores) + 1e-6]])
    
    best_acc = 0.0
    best_thresh = 0.0
    
    for thresh in thresholds:
        y_pred = (scores > thresh).astype(int)
        acc = float((y_true == y_pred).mean())
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    #  F1 / SHD at best-ACC threshold 
    A_best = (edge_scores > best_thresh).astype(int)
    if ignore_diag:
        np.fill_diagonal(A_best, 0)
        assert not bool(np.diag(A_best).any()), "Thresholded graph must keep diagonal at zero."
    
    y_pred_best = A_best[mask].astype(int)
    
    tp = int(((y_pred_best == 1) & (y_true == 1)).sum())
    fp = int(((y_pred_best == 1) & (y_true == 0)).sum())
    fn = int(((y_pred_best == 0) & (y_true == 1)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    tpr = recall
    shd = fp + fn
    
    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'ACC': float(best_acc),
        'F1': float(f1),
        'SHD': int(shd),
        'best_threshold': float(best_thresh),
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TPR': float(tpr),
        'FDR': float(fdr),
        'precision': float(precision),
        'recall': float(recall),
    }


def print_comparison_table(results, dataset_name="NC8"):
    print(f"\n{'='*60}")
    print(f"  {dataset_name} Results (UNCLE-style evaluation)")
    print(f"{'='*60}")
    print(f"  AUROC:  {results['AUROC']:.3f}")
    print(f"  AUPRC:  {results['AUPRC']:.3f}")
    print(f"  ACC:    {results['ACC']:.3f}")
    print(f"  F1:     {results['F1']:.3f}")
    print(f"  SHD:    {results['SHD']}")
    print(f"  TPR:    {results['TPR']:.3f}")
    print(f"  FDR:    {results['FDR']:.3f}")
    print(f"  Threshold: {results['best_threshold']:.6f}")
    print(f"{'='*60}")
    
    # UNCLE baselines for comparison
    print(f"\n  UNCLE Table 1 Baselines ({dataset_name}):")
    print(f"  {'Method':<12} {'AUROC':>8} {'AUPRC':>8}")
    print(f"  {'-'*30}")
    print(f"  {'UnCLe(P)':<12} {'0.975':>8} {'0.835':>8}")
    print(f"  {'UnCLe(A)':<12} {'0.952':>8} {'0.952':>8}")
    print(f"  {'GVAR':<12} {'0.956':>8} {'0.831':>8}")
    print(f"  {'cMLP':<12} {'0.928':>8} {'0.717':>8}")
    print(f"  {'BO (ours)':<12} {results['AUROC']:>8.3f} {results['AUPRC']:>8.3f}")
    print()
