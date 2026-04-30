#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.tsa.vector_ar.var_model import VAR as SMVAR

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_DIR = REPO_ROOT.parents[0] / "data"

DATASET_LAG: dict[str, int] = {"nc8": 16, "nd8": 16, "finance": 5}

DATASET_PATTERNS = {
    "nc8":     {"data_glob": "nc8_data_*.csv",     "struct_glob": "nc8_struct_*.csv",    "struct_npy": None},
    "nd8":     {"data_glob": "nc8_dynamic_*.csv",  "struct_glob": None,                  "struct_npy": "nc8_structure_dynamic.npy"},
    "finance": {"data_glob": "finance_data_*.csv", "struct_glob": "finance_struct_*.csv","struct_npy": None},
}

DIR_MAP = {"nc8": "NC8", "nd8": "ND8", "finance": "Finance"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["nc8", "nd8", "finance"],
                   choices=["nc8", "nd8", "finance"])
    p.add_argument("--data-dir", type=Path, default=BASE_DATA_DIR)
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "results" / "var_statsmodels_faithful")
    p.add_argument("--replicas", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _is_numeric_row(tokens: list[str]) -> bool:
    try:
        [float(t) for t in tokens]
        return True
    except ValueError:
        return False


def _is_integer_index_header(tokens: list[str]) -> bool:
    try:
        vals = [int(float(t.strip())) for t in tokens]
        return vals == list(range(len(vals)))
    except (ValueError, AttributeError):
        return False


def read_csv_matrix(path: Path) -> np.ndarray:
    with path.open() as fh:
        first_tokens = [t.strip() for t in fh.readline().strip().split(",")]
    has_header = not _is_numeric_row(first_tokens) or _is_integer_index_header(first_tokens)
    return np.loadtxt(path, delimiter=",", skiprows=int(has_header), dtype=np.float64)


def load_dataset(ds_name: str, data_dir: Path, max_replicas: int) -> list[dict[str, Any]]:
    ds_dir = (data_dir / DIR_MAP[ds_name]).expanduser().resolve()
    pat = DATASET_PATTERNS[ds_name]

    data_paths = sorted(ds_dir.glob(pat["data_glob"]))
    if not data_paths:
        raise FileNotFoundError(f"No {pat['data_glob']} under {ds_dir}")
    if max_replicas > 0:
        data_paths = data_paths[:max_replicas]

    if pat["struct_npy"]:
        g = np.load(str(ds_dir / pat["struct_npy"]), allow_pickle=True)
        if g.ndim == 3:
            g = (np.max(np.abs(g), axis=0) > 0).astype(int)
        shared_gt = g.astype(int)
        struct_paths = [None] * len(data_paths)
    else:
        struct_paths = sorted(ds_dir.glob(pat["struct_glob"]))[:len(data_paths)]
        shared_gt = None

    replicas = []
    for k, dp in enumerate(data_paths):
        df = pd.read_csv(dp)
        X = df.values.astype(np.float64)
        if shared_gt is not None:
            gt = shared_gt
        else:
            raw_gt = read_csv_matrix(struct_paths[k])
            N = X.shape[1]
            if raw_gt.shape == (N + 1, N):
                raw_gt = raw_gt[1:]
            gt = raw_gt.astype(int)
        replicas.append({"index": k, "data_path": str(dp), "X": X, "gt": gt,
                         "col_names": list(df.columns)})
    return replicas


def run_var_coef(X: np.ndarray, lag: int) -> dict[str, Any]:
    N = X.shape[1]
    safe_names = [f"x{i}" for i in range(N)]
    score_mat = np.zeros((N, N), dtype=np.float64)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = SMVAR(pd.DataFrame(X, columns=safe_names)).fit(lag)
    except Exception as e:
        return {"score_matrix": score_mat, "fit_failed": True, "fit_error": str(e),
                "coefs_finite": False, "coefs_max_abs": float("nan")}

    coefs = fitted.coefs
    coefs_finite = bool(np.isfinite(coefs).all())

    if coefs_finite:
        for i in range(N):
            for j in range(N):
                if i != j:
                    score_mat[j, i] = float(np.sqrt(np.sum(coefs[:, i, j] ** 2)))
    else:
        for i in range(N):
            for j in range(N):
                if i != j:
                    v = np.abs(coefs[:, i, j])
                    v = v[np.isfinite(v)]
                    score_mat[j, i] = float(np.sqrt(np.sum(v ** 2))) if len(v) > 0 else 0.0

    return {"score_matrix": score_mat, "fit_failed": False,
            "coefs_finite": coefs_finite,
            "coefs_max_abs": float(np.abs(coefs).max()) if coefs_finite else float("nan")}


def off_diag_mask(N: int) -> np.ndarray:
    return ~np.eye(N, dtype=bool)


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores.astype(float), method="average")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores.astype(float), kind="mergesort")
    y_s = y_true[order]
    tp = np.cumsum(y_s)
    prec = tp / np.arange(1, len(y_s) + 1)
    return float(np.sum(prec * y_s) / n_pos)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True,
                               default=lambda x: float(x) if isinstance(x, np.floating) else x)
                    + "\n", encoding="utf-8")


def write_matrix_csv(path: Path, m: np.ndarray, names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + names)
        for name, row in zip(names, m.tolist()):
            w.writerow([name] + row)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict] = []

    for ds_name in args.datasets:
        lag = DATASET_LAG[ds_name]
        print(f"\n[{ds_name.upper()}] lag={lag}", flush=True)

        replicas = load_dataset(ds_name, args.data_dir, args.replicas)
        print(f"replicas={len(replicas)}", flush=True)

        ds_out = out_root / ds_name
        (ds_out / "predicted_graphs").mkdir(parents=True, exist_ok=True)

        per_replica_rows: list[dict] = []

        for rep in replicas:
            k = rep["index"]
            X = rep["X"]
            gt = rep["gt"]
            N = X.shape[1]
            safe_names = [f"x{i}" for i in range(N)]
            mask = off_diag_mask(N)
            gt_f = gt[mask].astype(int)

            t0 = time.perf_counter()
            result = run_var_coef(X, lag)
            runtime = time.perf_counter() - t0

            score = result["score_matrix"]
            sc_f = score.T[mask]

            auroc = compute_auroc(gt_f, sc_f)
            auprc = compute_auprc(gt_f, sc_f)
            score_std = float(sc_f.std())

            prefix = ds_out / "predicted_graphs" / f"replica_{k}"
            np.save(str(prefix) + "_raw_score_matrix.npy", score)
            write_matrix_csv(Path(str(prefix) + "_raw_score_matrix.csv"), score, safe_names)

            row: dict[str, Any] = {
                "dataset": ds_name,
                "replica": k,
                "lag": lag,
                "T": X.shape[0],
                "N": N,
                "fit_failed": result["fit_failed"],
                "coefs_finite": result.get("coefs_finite", False),
                "coefs_max_abs": result.get("coefs_max_abs", float("nan")),
                "gt_offdiag_pos": int(gt_f.sum()),
                "gt_offdiag_neg": int((gt_f == 0).sum()),
                "score_std": score_std,
                "auroc": float(auroc),
                "auprc": float(auprc),
                "runtime_sec": float(runtime),
            }
            per_replica_rows.append(row)

            print(f"  replica {k}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                  f"score_std={score_std:.4g}  fit_failed={result['fit_failed']}  "
                  f"coefs_finite={result.get('coefs_finite')}  t={runtime:.1f}s",
                  flush=True)

        write_rows_csv(ds_out / "per_replica_metrics.csv", per_replica_rows)

        valid_aurocs = [r["auroc"] for r in per_replica_rows if not np.isnan(r["auroc"])]
        valid_auprcs = [r["auprc"] for r in per_replica_rows if not np.isnan(r["auprc"])]
        agg: dict[str, Any] = {
            "dataset": ds_name,
            "lag": lag,
            "replicas": len(per_replica_rows),
            "mean_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else float("nan"),
            "std_auroc": float(np.std(valid_aurocs, ddof=1)) if len(valid_aurocs) > 1 else 0.0,
            "mean_auprc": float(np.mean(valid_auprcs)) if valid_auprcs else float("nan"),
            "std_auprc": float(np.std(valid_auprcs, ddof=1)) if len(valid_auprcs) > 1 else 0.0,
            "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in per_replica_rows])),
        }
        write_rows_csv(ds_out / "aggregate_metrics.csv", [agg])
        all_summary_rows.append(agg)

        print(f"  [aggregate] mean_AUROC={agg['mean_auroc']:.4f}±{agg['std_auroc']:.4f}  "
              f"mean_AUPRC={agg['mean_auprc']:.4f}±{agg['std_auprc']:.4f}", flush=True)

    write_rows_csv(out_root / "summary_all_datasets.csv", all_summary_rows)
    write_json(out_root / "config.json", {
        "datasets": args.datasets,
        "dataset_lags": {ds: DATASET_LAG[ds] for ds in args.datasets},
        "replicas_arg": args.replicas,
        "seed": args.seed,
    })

    print(f"\noutput: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
