from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


def _json_default(v: Any):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (set, tuple)):
        return list(v)
    return str(v)


def append_jsonl(path: str | None, payload: Dict[str, Any]) -> None:
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def _iter_params(obj: Any) -> List[torch.nn.Parameter]:
    if obj is None:
        return []
    if isinstance(obj, torch.nn.Parameter):
        return [obj]
    if isinstance(obj, torch.nn.Module):
        return list(obj.parameters())
    if isinstance(obj, (list, tuple)):
        out: List[torch.nn.Parameter] = []
        for item in obj:
            out.extend(_iter_params(item))
        return out
    if isinstance(obj, Iterable):
        out: List[torch.nn.Parameter] = []
        for item in obj:
            out.extend(_iter_params(item))
        return out
    return []


def collect_param_stats(model: Any) -> Dict[str, Any]:
    params = _iter_params(model)
    total = int(sum(int(p.numel()) for p in params))
    trainable = int(sum(int(p.numel()) for p in params if p.requires_grad))
    return {
        "n_tensors": int(len(params)),
        "total_params": total,
        "trainable_params": trainable,
    }


def collect_grad_stats(model: Any) -> Dict[str, Any]:
    params = _iter_params(model)
    grad_tensors = [p.grad.detach() for p in params if p.grad is not None]
    if len(grad_tensors) == 0:
        return {
            "n_grad_tensors": 0,
            "grad_l2": 0.0,
            "grad_max_abs": 0.0,
            "grad_mean_abs": 0.0,
        }

    l2_sq = 0.0
    max_abs = 0.0
    abs_sum = 0.0
    n = 0
    for g in grad_tensors:
        gg = g.float()
        l2_sq += float(torch.sum(gg * gg).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(gg)).item()))
        abs_sum += float(torch.sum(torch.abs(gg)).item())
        n += int(gg.numel())
    return {
        "n_grad_tensors": int(len(grad_tensors)),
        "grad_l2": float(np.sqrt(l2_sq)),
        "grad_max_abs": float(max_abs),
        "grad_mean_abs": float(abs_sum / max(n, 1)),
    }


def snapshot_params(model: Any) -> List[torch.Tensor]:
    params = _iter_params(model)
    return [p.detach().cpu().clone() for p in params]


def param_delta_from_snapshot(snapshot: Sequence[torch.Tensor], model: Any) -> Dict[str, float]:
    params = _iter_params(model)
    if len(snapshot) == 0 or len(params) == 0:
        return {"param_delta_l2": 0.0, "param_delta_max_abs": 0.0}

    n = min(len(snapshot), len(params))
    l2_sq = 0.0
    max_abs = 0.0
    for i in range(n):
        before = snapshot[i].to(params[i].device)
        after = params[i].detach()
        delta = (after - before).float()
        l2_sq += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(delta)).item()))
    return {
        "param_delta_l2": float(np.sqrt(l2_sq)),
        "param_delta_max_abs": float(max_abs),
    }


def flatten_offdiag(W: np.ndarray, ignore_diag: bool = True) -> np.ndarray:
    arr = np.asarray(W, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square matrix, got {arr.shape}")
    if not ignore_diag:
        return arr.reshape(-1)
    mask = np.ones_like(arr, dtype=bool)
    np.fill_diagonal(mask, False)
    return arr[mask]


def _true_false_weights(W: np.ndarray, GT: np.ndarray, ignore_diag: bool = True):
    w = np.asarray(W, dtype=np.float64)
    gt = (np.asarray(GT) != 0).astype(np.int8)
    if w.shape != gt.shape:
        raise ValueError(f"Shape mismatch: W {w.shape} vs GT {gt.shape}")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"Expected square matrix, got {w.shape}")

    mask = np.ones_like(gt, dtype=bool)
    if ignore_diag:
        np.fill_diagonal(mask, False)
    true_vals = w[(gt == 1) & mask]
    false_vals = w[(gt == 0) & mask]
    return true_vals.astype(np.float64), false_vals.astype(np.float64)


def _percentiles(x: np.ndarray, ps=(50, 90, 95, 99)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def collect_weight_stats(
    W: np.ndarray,
    GT: np.ndarray,
    eps: float = 1e-6,
    top_ks: Sequence[int] = (10, 20),
    ignore_diag: bool = True,
) -> Dict[str, Any]:
    true_vals, false_vals = _true_false_weights(W, GT, ignore_diag=ignore_diag)

    def _m(x):
        return float(np.mean(x)) if x.size else float("nan")

    out: Dict[str, Any] = {
        "n_true": int(true_vals.size),
        "n_false": int(false_vals.size),
        "mean_true": _m(true_vals),
        "mean_false": _m(false_vals),
        "mean_gap_true_minus_false": _m(true_vals) - _m(false_vals),
        "median_true": float(np.median(true_vals)) if true_vals.size else float("nan"),
        "median_false": float(np.median(false_vals)) if false_vals.size else float("nan"),
        "true_lt_eps": int(np.sum(true_vals < float(eps))),
    }
    out.update({f"true_{k}": v for k, v in _percentiles(true_vals).items()})
    out.update({f"false_{k}": v for k, v in _percentiles(false_vals).items()})

    cmp = true_vals[:, None] - false_vals[None, :] if true_vals.size and false_vals.size else None
    if cmp is None:
        out["sep_prob_true_gt_false"] = float("nan")
    else:
        gt_prob = float(np.mean(cmp > 0))
        eq_prob = float(np.mean(cmp == 0))
        out["sep_prob_true_gt_false"] = float(gt_prob + 0.5 * eq_prob)

    w = np.asarray(W, dtype=np.float64)
    gt = (np.asarray(GT) != 0).astype(np.int8)
    d = w.shape[0]
    pairs = []
    for i in range(d):
        for j in range(d):
            if ignore_diag and i == j:
                continue
            pairs.append((i, j, float(w[i, j]), int(gt[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)

    for k in top_ks:
        top = pairs[: int(k)]
        out[f"top{k}_false_count"] = int(sum(1 for _, _, _, g in top if g == 0))
        out[f"top{k}_true_count"] = int(sum(1 for _, _, _, g in top if g == 1))

    return out


def dump_top_edges(W: np.ndarray, GT: np.ndarray, K: int = 20, ignore_diag: bool = True) -> List[Dict[str, Any]]:
    w = np.asarray(W, dtype=np.float64)
    gt = (np.asarray(GT) != 0).astype(np.int8)
    if w.shape != gt.shape:
        raise ValueError(f"Shape mismatch: W {w.shape} vs GT {gt.shape}")
    d = w.shape[0]
    rows = []
    for i in range(d):
        for j in range(d):
            if ignore_diag and i == j:
                continue
            rows.append({
                "i": int(i),
                "j": int(j),
                "weight": float(w[i, j]),
                "gt": int(gt[i, j]),
            })
    rows.sort(key=lambda r: r["weight"], reverse=True)
    top = rows[: int(K)]
    for rank, row in enumerate(top, start=1):
        row["rank"] = int(rank)
    return top


def determinism_banner(seed: int) -> Dict[str, Any]:
    return {
        "timestamp": time.time(),
        "seed": int(seed),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
