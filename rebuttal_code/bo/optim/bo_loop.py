from tqdm import trange
import os
import time
import random
import json
import hashlib
from collections import deque
import numpy as np
import torch
from .graph_optim import GraphOptimBO
from ..graph.graph_spaces import LowRankGraphSpace
from ..utils.logger import Logger
from ..reporters import NullReporter
from ..eval.perturbation import compute_parent_loo_perturbation
from sklearn.preprocessing import StandardScaler


def _append_csv(path_csv, header, row):
    if path_csv is None:
        return
    out_dir = os.path.dirname(path_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.exists(path_csv)
    import csv
    with open(path_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def _graph_summary(adj):
    A = np.asarray(adj)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square adjacency, got shape={A.shape}")
    d = int(A.shape[0])
    A_bin = (A != 0).astype(np.uint8, copy=False)
    A_offdiag = A_bin.copy()
    np.fill_diagonal(A_offdiag, 0)
    denom = d * (d - 1) if d > 1 else 1
    packed_full = np.packbits(np.ascontiguousarray(A_bin).reshape(-1))
    packed_offdiag = np.packbits(np.ascontiguousarray(A_offdiag).reshape(-1))
    return {
        "shape": [int(d), int(d)],
        "edges_total": int(A_bin.sum()),
        "diag_edges": int(np.diag(A_bin).sum()),
        "offdiag_edge_count": int(A_offdiag.sum()),
        "offdiag_density": float(A_offdiag.sum() / denom),
        "hash_full_md5": hashlib.md5(packed_full.tobytes()).hexdigest(),
        "hash_offdiag_md5": hashlib.md5(packed_offdiag.tobytes()).hexdigest(),
    }


def _adj_offdiag_edges(adj):
    A = np.asarray(adj)
    A_bin = (A != 0).astype(np.uint8, copy=False)
    return int(np.count_nonzero(A_bin) - int(np.diag(A_bin).sum()))


def run_bo(
        X: np.ndarray,
        score_method: str,
        score_params: dict,
        max_evals: int,
        batch_size: int = 64,
        normalize: bool = False,
        graph_rank: int = 8,
        ts_rank: int | None = None,
        k_latent: int = 0,
        lambda_sparse: float = 1.0,
        return_model: bool = False,
        pruner=None,
        n_cands: int = 10000,
        n_grads: int = 10,
        lr: float = 0.1,
        hidden_size: int = 64,
        dropout: float = 0.1,
        n_replay: int = 1024,
        device: str | torch.device = 'cpu',
        verbose: bool = False,
        random_state: int | None = 0,
        GT: np.ndarray | None = None,
        init_adjs=None,
        tau: float = 0.8,
        scorer=None,
        reporter=None,
        run_logger=None,
        track_edge_freq: bool = False,
        meta_update_every: int = 1,
        meta_history_size: int = 256,
        meta_batch_size: int | None = None,
        meta_batch_size_J: int | None = None,
        meta_history_only: bool | None = None,
        meta_use_history_only: bool = True,
        meta_mode_mode: str | None = None,
        meta_recent_window: int = 10,
        mix_current_frac: float = 0.0,
        meta_fallback_strategy: str = "replacement_then_hist",
        meta_elite_frac: float = 0.2,
        meta_recent_size: int = 128,
        meta_history_snapshot: bool = True,
        meta_log_path: str | None = None,
        meta_inner_step_count: int | None = None,
        meta_inner_lr: float | None = None,
        meta_outer_lr: float | None = None,
        meta_warmup_frac: float = 0.0,
        **kwargs
) -> dict:
    """Graph recovery via Bayesian Optimization (DrBO dropout surrogate)."""
    np.random.seed(random_state)
    random.seed(random_state)
    torch.random.fork_rng()
    torch.random.manual_seed(random_state)
    torch.backends.cuda.matmul.allow_tf32 = True

    if isinstance(score_method, str):
        if score_method.lower() != 'bic':
            raise ValueError("Only score_method='BIC'")

    if normalize:
        X = StandardScaler().fit_transform(X)

    n, d = X.shape
    if k_latent not in (0, None):
        raise ValueError("Latent confounders are not supported; k_latent must be 0.")
    if "use_latent" in kwargs and kwargs.get("use_latent"):
        raise ValueError("Latent confounders are not supported; use_latent must be False.")

    if ts_rank is not None:
        if graph_rank != 8:
            raise ValueError("Provide only ts_rank or graph_rank, not both.")
        rank = int(ts_rank)
    else:
        rank = int(graph_rank)

    if scorer is None:
        raise ValueError("scorer must be provided for nonlinear runs.")

    meta_update_every = max(1, int(meta_update_every))
    meta_history_size = max(1, int(meta_history_size))
    if meta_batch_size_J is not None:
        meta_batch_size = int(meta_batch_size_J)
    if meta_batch_size is None:
        meta_batch_size = min(32, meta_history_size)
    meta_batch_size = max(1, min(int(meta_batch_size), meta_history_size))

    if meta_history_only is None:
        meta_history_only = bool(meta_use_history_only)
    meta_history_only = bool(meta_history_only)

    meta_recent_window = max(1, int(meta_recent_window))
    mix_current_frac = float(np.clip(mix_current_frac, 0.0, 1.0))
    meta_elite_frac = float(np.clip(meta_elite_frac, 0.0, 1.0))
    meta_recent_size = max(1, int(meta_recent_size))
    meta_warmup_frac = float(np.clip(meta_warmup_frac, 0.0, 1.0))
    warmup_eval_cutoff = int(np.floor(float(max_evals) * meta_warmup_frac))

    if meta_mode_mode is None:
        meta_mode_mode = "recent_topB_replay"
    else:
        meta_mode_mode = str(meta_mode_mode).strip()
    if meta_mode_mode == "legacy":
        meta_mode_mode = "recent_topB_replay"

    valid_meta_modes = {"recent_topB_replay", "random_history", "elite_history", "freeze_meta"}
    if meta_mode_mode not in valid_meta_modes:
        raise ValueError(
            f"Invalid meta_mode_mode={meta_mode_mode!r}. "
            f"Expected one of {sorted(valid_meta_modes)}."
        )

    meta_fallback_strategy = str(meta_fallback_strategy).strip()
    valid_fallbacks = {"replacement_then_hist", "replacement_only", "hist_only", "skip"}
    if meta_fallback_strategy not in valid_fallbacks:
        raise ValueError(
            f"Invalid meta_fallback_strategy={meta_fallback_strategy!r}. "
            f"Expected one of {sorted(valid_fallbacks)}."
        )

    graph_space = LowRankGraphSpace(
        nodes=d,
        rank=rank,
        tau=tau,
        device=device,
    )

    logger = Logger(verbose=verbose)
    if reporter is None:
        reporter = NullReporter()
    config = dict(
        max_evals=max_evals,
        batch_size=batch_size,
        normalize=normalize,
        graph_rank=graph_rank,
        ts_rank=ts_rank,
        lambda_sparse=lambda_sparse,
        n_cands=n_cands,
        n_grads=n_grads,
        lr=lr,
        hidden_size=hidden_size,
        dropout=dropout,
        n_replay=n_replay,
        tau=tau,
        device=str(device),
        random_state=random_state,
        meta_update_every=meta_update_every,
        meta_history_size=meta_history_size,
        meta_batch_size=meta_batch_size,
        meta_batch_size_J=meta_batch_size,
        meta_history_only=bool(meta_history_only),
        meta_use_history_only=bool(meta_history_only),
        meta_mode_mode=meta_mode_mode,
        meta_recent_window=meta_recent_window,
        mix_current_frac=mix_current_frac,
        meta_fallback_strategy=meta_fallback_strategy,
        meta_elite_frac=meta_elite_frac,
        meta_recent_size=meta_recent_size,
        meta_history_snapshot=bool(meta_history_snapshot),
        meta_log_path=meta_log_path,
        meta_inner_step_count=meta_inner_step_count,
        meta_inner_lr=meta_inner_lr,
        meta_outer_lr=meta_outer_lr,
        meta_warmup_frac=meta_warmup_frac,
        meta_warmup_eval_cutoff=warmup_eval_cutoff,
    )
    reporter.on_start(config)

    trace_instrument = os.getenv("BO_TRACE") == "1"
    trace_meta_trace = os.getenv("TRACE_META_TRACE") == "1"
    # Keep tqdm progress controlled by `verbose`, while noisy per-iteration
    # console lines are disabled by default unless explicitly enabled.
    iter_console_log = bool(kwargs.pop("iter_console_log", False))
    max_size = max_evals
    model = GraphOptimBO(
        X=X,
        GT=GT,
        graph_space=graph_space,
        max_size=max_size,
        lr=lr,
        n_grads=n_grads,
        n_cands=n_cands,
        dropout=dropout,
        hidden_size=hidden_size,
        n_replay=n_replay,
        device=device,
        logger=logger,
        pruner=pruner,
        scorer=scorer,
        init_adjs=init_adjs,
    )
    trace_bo_path = None
    gt_offdiag_edges = None
    if trace_instrument:
        if getattr(model, "_run_dir", None) is None and run_logger is not None and hasattr(run_logger, "run_dir"):
            model._run_dir = str(run_logger.run_dir)
        if getattr(model, "_run_dir", None) is None and os.getenv("BO_RUN_DIR"):
            model._run_dir = os.getenv("BO_RUN_DIR")
        if getattr(model, "_run_dir", None):
            trace_bo_path = os.path.join(model._run_dir, "trace_bo.csv")
        if GT is not None:
            gt_offdiag_edges = _adj_offdiag_edges(GT)

    def _extract_complexity(stats, scorer_obj, adj):
        if stats is not None and "bic_penalty" in stats:
            try:
                return float(stats["bic_penalty"])
            except Exception:
                pass
        n = getattr(scorer_obj, "n", None)
        if n is None:
            return None
        if stats is not None and "edges_offdiag" in stats:
            try:
                return float(stats["edges_offdiag"]) * float(np.log(float(n)))
            except Exception:
                pass
        if adj is None:
            return None
        A = np.asarray(adj)
        diag_edges = int(np.diag(A).sum())
        edges_offdiag = int(np.count_nonzero(A) - diag_edges)
        return float(edges_offdiag) * float(np.log(float(n)))

    def _extract_mse(stats, local_log_mse):
        if stats is not None and "sigma2_dyn_hat" in stats:
            try:
                return float(stats["sigma2_dyn_hat"])
            except Exception:
                pass
        if local_log_mse is None:
            return None
        try:
            return float(np.mean(np.exp(local_log_mse)))
        except Exception:
            return None

    def _select_adj(adjs, idx):
        if isinstance(adjs, list):
            return adjs[idx]
        arr = np.asarray(adjs)
        if arr.ndim == 2:
            return arr
        return arr[idx]

    def _iter_adjs(adjs):
        if isinstance(adjs, list):
            return adjs
        arr = np.asarray(adjs)
        if arr.ndim == 2:
            return [arr]
        return [arr[i] for i in range(arr.shape[0])]

    def _to_history_adj(adj):
        A = (np.asarray(adj) != 0).astype(np.int8)
        np.fill_diagonal(A, 0)
        return A

    def _adj_signature(adj_bin):
        arr = np.asarray(adj_bin, dtype=np.int8, order="C")
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def _make_history_record(adj, total_score, iter_num, rank_in_batch, uid):
        adj_bin = _to_history_adj(adj)
        return {
            "uid": int(uid),
            "iter": int(iter_num),
            "rank": int(rank_in_batch),
            "score": None if total_score is None else float(total_score),
            "adj": adj_bin,
            "sig": _adj_signature(adj_bin),
            "mask_hash": None,
            "seed": None if random_state is None else int(random_state),
        }

    def _record_meta_brief(records):
        out = []
        for r in records:
            out.append({
                "uid": int(r.get("uid", -1)),
                "iter": int(r.get("iter", -1)),
                "rank": int(r.get("rank", -1)),
                "score": None if r.get("score") is None else float(r.get("score")),
                "sig": str(r.get("sig", "")),
                "mask_hash": r.get("mask_hash"),
                "seed": r.get("seed"),
            })
        return out

    def _select_meta_pool(records):
        records = list(records)
        if meta_mode_mode != "elite_history":
            return records
        scored = [r for r in records if r.get("score") is not None and np.isfinite(float(r["score"]))]
        if scored:
            k = max(1, int(np.ceil(len(scored) * meta_elite_frac)))
            scored = sorted(scored, key=lambda r: float(r["score"]), reverse=True)
            return scored[:k]
        return records[-meta_recent_size:]

    def _json_default(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (set, tuple)):
            return list(v)
        return str(v)

    def _append_jsonl(path_jsonl, payload):
        if path_jsonl is None:
            return
        out_dir = os.path.dirname(path_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")

    def _mean_or_none(values):
        vals = []
        for v in values:
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                vals.append(fv)
        if not vals:
            return None
        return float(np.mean(vals))

    def _fmt(v):
        return "nan" if v is None else f"{float(v):.6f}"

    def _current_scorer_score(graph):
        if graph is None:
            return None
        try:
            return float(scorer(np.asarray(graph)))
        except Exception:
            return None

    best_total_score = None
    best_fit_term = None
    best_mse = None
    best_complexity = None
    run_start = time.perf_counter()
    edge_freq_counts = np.zeros((d, d), dtype=np.float64) if track_edge_freq else None
    edge_freq_total = 0
    meta_rng = np.random.default_rng(random_state if random_state is not None else 0)
    meta_history = deque(maxlen=meta_history_size)
    meta_replay_buffer = deque()
    meta_uid = 0
    if meta_log_path is None and meta_mode_mode != "recent_topB_replay":
        meta_log_path = os.path.join("results", "mode_meta_log.jsonl")

    def _sample_records(pool, k, replace=False):
        if len(pool) == 0 or k <= 0:
            return []
        idx = meta_rng.choice(len(pool), size=int(k), replace=bool(replace))
        return [pool[int(i)] for i in np.asarray(idx).tolist()]

    def _prune_replay_buffer(cur_iter):
        cutoff = int(cur_iter) - int(meta_recent_window) + 1
        while len(meta_replay_buffer) > 0 and int(meta_replay_buffer[0]["iter"]) < cutoff:
            meta_replay_buffer.popleft()

    with trange(max_evals, disable=not verbose) as pbar:
        pbar.refresh()
        eval_count = 0
        iter_idx = 0

        while eval_count < max_evals:
            logger.reset()
            if hasattr(scorer, "clear_adaptation_buffer"):
                scorer.clear_adaptation_buffer()
            _prune_replay_buffer(iter_idx - 1)
            history_snapshot = list(meta_history)
            replay_snapshot = list(meta_replay_buffer)

            cur_batch = min(batch_size, max_evals - eval_count)
            if trace_instrument and hasattr(scorer, "_used_surrogate"):
                scorer._used_surrogate = False

            next_zs, next_adjs, est_scores = model.suggest(cur_batch)

            if track_edge_freq:
                for adj_i in _iter_adjs(next_adjs):
                    A_bin = (np.asarray(adj_i) != 0).astype(np.float64)
                    np.fill_diagonal(A_bin, 0.0)
                    edge_freq_counts += A_bin
                    edge_freq_total += 1

            if trace_instrument:
                bic_model = getattr(model, "bic_model", None)
                sur_scorer = getattr(bic_model, "scorer", None) if bic_model is not None else None
                if sur_scorer is None:
                    sur_scorer = scorer
                if model.best_adj is not None:
                    assert getattr(sur_scorer, "_used_surrogate", False), "aggregate_batch did not use surrogate locals!"
                if bic_model is not None:
                    assert getattr(bic_model, "scorer", scorer) is scorer, "BUG: bic_model.scorer is not the same scorer instance used by BO loop"

            true_eval_before = getattr(scorer, "_true_eval_calls", 0)
            if trace_instrument:
                t0 = time.perf_counter()

            stats_list = None
            if hasattr(scorer, "eval_batch_return_local"):
                true_scores, stats_list = scorer.eval_batch_return_local(next_adjs, return_stats=True)
            else:
                true_scores = scorer.batch_eval(next_adjs)

            if trace_instrument and hasattr(model, "record_true_eval_time"):
                model.record_true_eval_time(time.perf_counter() - t0)

            if trace_instrument:
                true_eval_after = getattr(scorer, "_true_eval_calls", 0)
                print(f"true_eval_calls={true_eval_after - true_eval_before} expected={cur_batch}")
                adj0 = next_adjs[0] if isinstance(next_adjs, (list, np.ndarray)) else next_adjs
                local = np.asarray(true_scores)[0, :d]
                approx = scorer.aggregate_batch([adj0], local[None, :])
                approx_val = float(np.asarray(approx).reshape(-1)[0])
                total_true = float(np.asarray(true_scores)[0, -1])
                print(f"true_total={total_true:.6f} approx_from_true_local={approx_val:.6f} diff={total_true - approx_val:.6f}")

            stats = getattr(scorer, "last_stats", None)
            logger.add(**(stats if stats is not None else {}))

            scores_arr = np.asarray(true_scores, dtype=np.float64)
            if scores_arr.ndim == 1:
                scores_arr = scores_arr.reshape(1, -1)
            stats_batch = stats_list if stats_list is not None else [None] * scores_arr.shape[0]

            batch_total = _mean_or_none(scores_arr[:, -1] if scores_arr.shape[1] >= 1 else [])
            batch_logmse = _mean_or_none(scores_arr[:, :d].reshape(-1) if scores_arr.shape[1] >= (d + 1) else [])
            batch_mse = _mean_or_none([(s.get("sigma2_dyn_hat") if isinstance(s, dict) else None) for s in stats_batch])
            batch_edges = _mean_or_none([(s.get("edges_offdiag") if isinstance(s, dict) else None) for s in stats_batch])
            batch_penalty = _mean_or_none([(s.get("bic_penalty") if isinstance(s, dict) else None) for s in stats_batch])
            batch_fit = None
            if batch_total is not None and batch_penalty is not None:
                batch_fit = batch_total + batch_penalty

            logger.add(
                batch_mean_mse=batch_mse,
                batch_mean_logmse=batch_logmse,
                batch_mean_edges=batch_edges,
                batch_mean_fit=batch_fit,
                batch_mean_penalty=batch_penalty,
                batch_mean_total=batch_total,
            )

            if verbose and iter_console_log:
                top_b_scores = [round(float(v), 6) for v in scores_arr[:, -1].tolist()]
                print(f"[BO] iter={iter_idx} topB_true_scores={top_b_scores}")
                print(
                    f"[SCORER] iter={iter_idx} "
                    f"mean_mse={_fmt(batch_mse)} mean_logmse={_fmt(batch_logmse)} "
                    f"edges={_fmt(batch_edges)} fit={_fmt(batch_fit)} "
                    f"penalty={_fmt(batch_penalty)} total={_fmt(batch_total)}"
                )

            batch_items = []
            for i in range(scores_arr.shape[0]):
                eval_so_far = eval_count + i + 1
                total_score = float(scores_arr[i, -1])
                local_log_mse = None
                if scores_arr.shape[1] >= (d + 1):
                    local_log_mse = scores_arr[i, :d]
                stats_i = stats_batch[i] if i < len(stats_batch) else None
                adj_i = _select_adj(next_adjs, i) if isinstance(next_adjs, (list, np.ndarray)) else None
                complexity_term = _extract_complexity(stats_i, scorer, adj_i)
                mse_val = _extract_mse(stats_i, local_log_mse)
                fit_term = total_score + complexity_term if complexity_term is not None else None
                batch_items.append({
                    "rank_in_batch": int(i),
                    "eval_so_far": int(eval_so_far),
                    "total_score": float(total_score),
                    "fit_term": fit_term,
                    "mse": mse_val,
                    "complexity_term": complexity_term,
                    "adj": adj_i,
                })

                if best_total_score is None or total_score > best_total_score:
                    best_total_score = total_score
                    best_fit_term = fit_term
                    best_mse = mse_val
                    best_complexity = complexity_term

            if run_logger is not None:
                for item in batch_items:
                    run_logger.log_progress({
                        "true_evals_so_far": item["eval_so_far"],
                        "total_score": item["total_score"],
                        "fit_term": item["fit_term"],
                        "mse": item["mse"],
                        "complexity_term": item["complexity_term"],
                        "time_sec": time.perf_counter() - run_start,
                    })
                    run_logger.log_history({
                        "true_evals_so_far": item["eval_so_far"],
                        "best_total_score": best_total_score,
                        "best_fit_term": best_fit_term,
                        "best_mse": best_mse,
                        "best_complexity_term": best_complexity,
                    })

            reporter.on_batch_evaluated(eval_count, next_adjs, true_scores, stats_list, GT, d)

            current_records = []
            for item in batch_items:
                record = _make_history_record(
                    adj=item["adj"],
                    total_score=item["total_score"],
                    iter_num=iter_idx,
                    rank_in_batch=item["rank_in_batch"],
                    uid=meta_uid,
                )
                meta_uid += 1
                current_records.append(record)
            pending_topB = list(current_records)

            eval_count += cur_batch
            logger.set_step(eval_count)

            # Surrogate update remains unchanged.
            model.add_data(next_zs, next_adjs, true_scores)
            if trace_bo_path is not None and len(batch_items) > 0:
                batch_edge_counts = np.asarray(
                    [_adj_offdiag_edges(item["adj"]) for item in batch_items],
                    dtype=np.float64,
                )
                batch_best_idx = int(np.argmax(scores_arr[:, -1]))
                batch_best_item = batch_items[batch_best_idx]
                batch_best_offdiag_edges = int(batch_edge_counts[batch_best_idx])
                margin_min = margin_p50 = margin_p95 = margin_max = margin_pos_frac = score_fro = score_absmax = ""
                try:
                    score_mats = np.asarray(graph_space.vec2score(next_zs))
                    if score_mats.ndim == 2:
                        score_mat_best = score_mats
                    else:
                        score_mat_best = score_mats[batch_best_idx]
                    score_mat_best = np.asarray(score_mat_best, dtype=np.float64)
                    score_offdiag = score_mat_best[~np.eye(d, dtype=bool)]
                    margin = score_offdiag - float(tau)
                    margin_min = float(np.min(margin))
                    margin_p50 = float(np.percentile(margin, 50))
                    margin_p95 = float(np.percentile(margin, 95))
                    margin_max = float(np.max(margin))
                    margin_pos_frac = float(np.mean(margin > 0.0))
                    score_fro = float(np.linalg.norm(score_mat_best))
                    score_absmax = float(np.max(np.abs(score_mat_best)))
                except Exception:
                    pass

                best_graph_summary = None
                if getattr(model, "best_adj_snapshot", None) is not None:
                    best_graph_summary = _graph_summary(model.best_adj_snapshot)
                elif getattr(model, "best_adj", None) is not None:
                    best_graph_summary = _graph_summary(model.best_adj)

                _append_csv(
                    trace_bo_path,
                    [
                        "iter",
                        "eval_count_after",
                        "best_total_score",
                        "best_score_datafit",
                        "best_score_complexity",
                        "best_offdiag_edges",
                        "batch_offdiag_edges_min",
                        "batch_offdiag_edges_median",
                        "batch_offdiag_edges_max",
                        "batch_best_total_score",
                        "batch_best_offdiag_edges",
                        "gt_offdiag_edges",
                        "batch_best_vs_gt_offdiag_gap",
                        "score_margin_min",
                        "score_margin_p50",
                        "score_margin_p95",
                        "score_margin_max",
                        "score_margin_pos_frac",
                        "score_mat_fro_norm",
                        "score_mat_absmax",
                    ],
                    [
                        int(iter_idx),
                        int(eval_count),
                        best_total_score,
                        best_fit_term,
                        best_complexity,
                        "" if best_graph_summary is None else int(best_graph_summary["offdiag_edge_count"]),
                        float(np.min(batch_edge_counts)),
                        float(np.median(batch_edge_counts)),
                        float(np.max(batch_edge_counts)),
                        float(batch_best_item["total_score"]),
                        int(batch_best_offdiag_edges),
                        "" if gt_offdiag_edges is None else int(gt_offdiag_edges),
                        "" if gt_offdiag_edges is None else int(batch_best_offdiag_edges - int(gt_offdiag_edges)),
                        margin_min,
                        margin_p50,
                        margin_p95,
                        margin_max,
                        margin_pos_frac,
                        score_fro,
                        score_absmax,
                    ],
                )

            meta_triggered = False
            meta_info = None
            meta_reason = "not_supported"
            meta_source = "skipped_empty"
            sampled_history_records = []
            sampled_history_scores = []
            history_only_violation = False
            overlap_uid_count = 0
            overlap_sig_count = 0
            current_mix_count = 0
            current_uid_set = {int(r["uid"]) for r in current_records}
            current_sig_set = {r["sig"] for r in current_records}
            current_score_mean = _mean_or_none([r.get("score") for r in current_records])

            history_size_before = int(len(history_snapshot))
            history_source = list(history_snapshot)
            history_pool = list(history_source)
            if meta_mode_mode == "elite_history":
                history_pool = _select_meta_pool(history_pool)
            history_pool_size = int(len(history_pool))

            replay_recent_cutoff = int(iter_idx) - int(meta_recent_window)
            replay_pool = [
                r for r in replay_snapshot
                if int(r["iter"]) >= replay_recent_cutoff and int(r["iter"]) < int(iter_idx)
            ]
            replay_pool_size = int(len(replay_pool))

            if hasattr(scorer, "meta_update"):
                if meta_mode_mode == "freeze_meta":
                    meta_reason = "freeze_meta"
                elif warmup_eval_cutoff > 0 and int(eval_count) <= int(warmup_eval_cutoff):
                    meta_reason = "warmup"
                elif ((iter_idx + 1) % meta_update_every) != 0:
                    meta_reason = "not_due"
                else:
                    meta_kwargs = {}
                    if meta_inner_step_count is not None:
                        meta_kwargs["inner_step_count"] = int(meta_inner_step_count)
                    if meta_inner_lr is not None:
                        meta_kwargs["inner_lr"] = float(meta_inner_lr)
                    if meta_outer_lr is not None:
                        meta_kwargs["outer_lr"] = float(meta_outer_lr)

                    if meta_mode_mode == "random_history":
                        if history_pool_size > 0:
                            replace = history_pool_size < meta_batch_size
                            sampled_history_records = _sample_records(history_pool, meta_batch_size, replace=replace)
                            meta_source = "hist_fallback"
                        else:
                            meta_reason = "empty_history"
                    elif meta_mode_mode == "elite_history":
                        if history_pool_size > 0:
                            replace = history_pool_size < meta_batch_size
                            sampled_history_records = _sample_records(history_pool, meta_batch_size, replace=replace)
                            meta_source = "hist_fallback"
                        else:
                            meta_reason = "empty_history"
                    else:
                        # Sample from historical buffer H_{t-1} (one-iteration delayed)
                        # to decouple meta updates from current acquisition choices.
                        if replay_pool_size >= meta_batch_size:
                            sampled_history_records = _sample_records(replay_pool, meta_batch_size, replace=False)
                            meta_source = "recent_topB_replay"
                        elif replay_pool_size > 0 and meta_fallback_strategy in {"replacement_then_hist", "replacement_only"}:
                            sampled_history_records = _sample_records(replay_pool, meta_batch_size, replace=True)
                            meta_source = "replay_replacement"
                        elif history_pool_size > 0 and meta_fallback_strategy in {"replacement_then_hist", "hist_only"}:
                            replace = history_pool_size < meta_batch_size
                            sampled_history_records = _sample_records(history_pool, meta_batch_size, replace=replace)
                            meta_source = "hist_fallback"
                        elif meta_fallback_strategy == "skip":
                            meta_reason = "skipped_by_config"
                        else:
                            meta_reason = "empty_buffer"

                    if len(sampled_history_records) > 0:
                        hist_batch = [np.array(r["adj"], copy=True) for r in sampled_history_records]
                        if trace_meta_trace:
                            for _rec in sampled_history_records:
                                sig_recomputed = _adj_signature(_rec["adj"])
                                if str(_rec.get("sig")) != str(sig_recomputed):
                                    raise AssertionError(
                                        f"meta history signature mismatch uid={_rec.get('uid')} "
                                        f"stored={_rec.get('sig')} recomputed={sig_recomputed}"
                                    )
                        sampled_history_scores = [
                            float(r["score"]) for r in sampled_history_records
                            if r.get("score") is not None and np.isfinite(float(r["score"]))
                        ]
                        overlap_uid_count = sum(1 for r in sampled_history_records if int(r["uid"]) in current_uid_set)
                        overlap_sig_count = sum(1 for r in sampled_history_records if r["sig"] in current_sig_set)
                        history_only_violation = bool(meta_history_only) and overlap_uid_count > 0
                        if history_only_violation and verbose and iter_console_log:
                            print(
                                f"[META-WARN] iter={iter_idx} strict_history_only violated with overlap_uid={overlap_uid_count}"
                            )

                        current_batch = None
                        if mix_current_frac > 0.0 and len(current_records) > 0:
                            current_mix_count = min(
                                len(current_records),
                                max(1, int(round(float(meta_batch_size) * float(mix_current_frac))))
                            )
                            current_selected = _sample_records(current_records, current_mix_count, replace=False)
                            current_batch = [np.array(r["adj"], copy=True) for r in current_selected]
                            meta_kwargs["current_task_frac"] = float(mix_current_frac)

                        meta_info = scorer.meta_update(
                            graphs_current=current_batch,
                            graphs_history=hist_batch,
                            **meta_kwargs,
                        )
                        meta_triggered = True
                        meta_reason = "triggered"
                    elif meta_reason == "not_supported":
                        meta_reason = "skipped_empty"

            meta_info = {} if meta_info is None else dict(meta_info)
            meta_current_count = int(meta_info.get("current_graphs", current_mix_count))
            meta_history_count = int(meta_info.get("history_graphs", len(sampled_history_records)))
            meta_num_graphs_used = int(meta_current_count + meta_history_count)
            sampled_history_score_mean = _mean_or_none(sampled_history_scores)

            # Append current Top-B AFTER meta update so next iteration sees one-step delay.
            for record in pending_topB:
                meta_history.append(record)
                meta_replay_buffer.append(record)
            _prune_replay_buffer(iter_idx)
            history_size_after_append = int(len(meta_history))
            replay_size_after_append = int(len(meta_replay_buffer))

            if hasattr(scorer, "meta_update") and verbose and iter_console_log:
                if meta_triggered:
                    print(
                        f"[META] iter={iter_idx} mode={meta_mode_mode} source={meta_source} "
                        f"history_only={bool(meta_history_only)} overlap_uid={overlap_uid_count} "
                        f"history_before={history_size_before} history_after={history_size_after_append} "
                        f"replay_size={replay_size_after_append} current_used={meta_current_count} "
                        f"history_used={meta_history_count} info={meta_info}"
                    )
                else:
                    print(
                        f"[META] iter={iter_idx} mode={meta_mode_mode} source={meta_source} skipped={meta_reason} "
                        f"history_only={bool(meta_history_only)} history_before={history_size_before} "
                        f"history_after={history_size_after_append} replay_size={replay_size_after_append}"
                    )

            iter_log = {
                "iter": int(iter_idx),
                "mode": meta_mode_mode,
                "eval_count_before": int(eval_count - cur_batch),
                "eval_count_after": int(eval_count),
                "batch_size": int(cur_batch),
                "topb_true_scores": [float(v) for v in scores_arr[:, -1].tolist()],
                "batch_mean_total": batch_total,
                "batch_mean_fit": batch_fit,
                "batch_mean_mse": batch_mse,
                "best_total_score_so_far": best_total_score,
                "best_mse_so_far": best_mse,
                "meta_triggered": bool(meta_triggered),
                "meta_reason": meta_reason,
                "meta_source": meta_source,
                "meta_num_graphs_used": int(meta_num_graphs_used),
                "meta_batch_size": int(meta_batch_size),
                "meta_batch_current_count": int(meta_current_count),
                "meta_batch_history_count": int(meta_history_count),
                "meta_history_only": bool(meta_history_only),
                "mix_current_frac": float(mix_current_frac),
                "history_size_before": int(history_size_before),
                "history_size_after_append": int(history_size_after_append),
                "replay_size_before": int(replay_pool_size),
                "replay_size_after_append": int(replay_size_after_append),
                "replay_recent_window": int(meta_recent_window),
                "meta_warmup_frac": float(meta_warmup_frac),
                "meta_warmup_eval_cutoff": int(warmup_eval_cutoff),
                "meta_warmup_active": bool(warmup_eval_cutoff > 0 and int(eval_count) <= int(warmup_eval_cutoff)),
                "history_pool_size": int(history_pool_size),
                "history_overlap_current_uid": int(overlap_uid_count),
                "history_overlap_current_sig": int(overlap_sig_count),
                "history_only_violation": bool(history_only_violation),
                "current_topb_mean_score": current_score_mean,
                "sampled_history_mean_score": sampled_history_score_mean,
                "meta_info": meta_info,
                "scorer_last_stats": dict(getattr(scorer, "last_stats", {}) or {}),
            }
            if trace_meta_trace:
                history_list = list(meta_history)
                sampled_list = list(sampled_history_records)
                iter_log["meta_history_head3"] = _record_meta_brief(history_list[:3])
                iter_log["meta_history_tail3"] = _record_meta_brief(history_list[-3:])
                iter_log["sampled_history_head3"] = _record_meta_brief(sampled_list[:3])
                iter_log["sampled_history_tail3"] = _record_meta_brief(sampled_list[-3:])
                iter_log["current_head3"] = _record_meta_brief(current_records[:3])
                iter_log["current_tail3"] = _record_meta_brief(current_records[-3:])
            _append_jsonl(meta_log_path, iter_log)

            pbar.set_postfix_str(logger.log_str())
            pbar.update(cur_batch)
            pbar.refresh()
            iter_idx += 1

    best_adj_live = None if model.best_adj is None else np.asarray(model.best_adj)
    best_adj_snapshot = None if getattr(model, "best_adj_snapshot", None) is None else np.array(model.best_adj_snapshot, copy=True)
    best_adj_source = "model.best_adj_snapshot" if best_adj_snapshot is not None else "model.best_adj"
    best_adj = None
    if best_adj_snapshot is not None:
        best_adj = np.array(best_adj_snapshot, copy=True)
        model.best_adj = np.array(best_adj_snapshot, copy=True)
    elif best_adj_live is not None:
        best_adj = np.array(best_adj_live, copy=True)
        model.best_adj = np.array(best_adj_live, copy=True)

    consistency = None
    if best_adj is not None:
        live_summary = None if best_adj_live is None else _graph_summary(best_adj_live)
        selected_summary = _graph_summary(best_adj)
        update_summary = dict(getattr(model, "best_summary", {}) or {})
        history_best_score = None if best_total_score is None else float(best_total_score)
        model_best_score = None if getattr(model, "best_score", None) is None else float(model.best_score)
        recomputed_at_update = getattr(model, "best_recomputed_score_at_update", None)
        current_scorer_score = _current_scorer_score(best_adj)
        model_vs_selected_hash_match = (
            True
            if live_summary is None
            else str(live_summary.get("hash_offdiag_md5")) == str(selected_summary.get("hash_offdiag_md5"))
        )
        update_vs_selected_hash_match = (
            True
            if not update_summary
            else str(update_summary.get("hash_offdiag_md5")) == str(selected_summary.get("hash_offdiag_md5"))
        )
        diff_history_vs_model = None if history_best_score is None or model_best_score is None else float(model_best_score - history_best_score)
        diff_history_vs_update = None if history_best_score is None or recomputed_at_update is None else float(recomputed_at_update - history_best_score)
        diff_history_vs_current = None if history_best_score is None or current_scorer_score is None else float(current_scorer_score - history_best_score)
        scorer_nonstationary = bool(
            diff_history_vs_current is not None
            and abs(diff_history_vs_current) > 1e-6
            and diff_history_vs_update is not None
            and abs(diff_history_vs_update) <= 1e-6
            and update_vs_selected_hash_match
        )
        consistency = {
            "status": "PASS",
            "history_best_total_score": history_best_score,
            "model_best_score": model_best_score,
            "recomputed_score_at_best_update": recomputed_at_update,
            "current_scorer_score_on_selected_best": current_scorer_score,
            "diff_history_vs_model_best_score": diff_history_vs_model,
            "diff_history_vs_recomputed_at_best_update": diff_history_vs_update,
            "diff_history_vs_current_scorer_score": diff_history_vs_current,
            "scorer_nonstationary_detected": scorer_nonstationary,
            "best_update_eval_idx": getattr(model, "best_update_eval_idx", None),
            "final_best_graph_source": best_adj_source,
            "live_model_best_graph_summary": live_summary,
            "selected_best_graph_summary": selected_summary,
            "best_update_graph_summary": update_summary,
            "live_vs_selected_hash_match": model_vs_selected_hash_match,
            "best_update_vs_selected_hash_match": update_vs_selected_hash_match,
        }

        fail_reason = None
        if diff_history_vs_model is not None and abs(diff_history_vs_model) > 1e-6:
            fail_reason = (
                f"model.best_score and history best_total_score diverged by {diff_history_vs_model:.6f}"
            )
        elif diff_history_vs_update is not None and abs(diff_history_vs_update) > 1e-6:
            fail_reason = (
                f"recomputed score at best update diverged from history by {diff_history_vs_update:.6f}"
            )
        elif not update_vs_selected_hash_match:
            fail_reason = (
                "selected final best graph does not match the graph snapshot taken at the last best update"
            )

        if fail_reason is not None:
            consistency["status"] = "FAIL"
            consistency["fail_reason"] = fail_reason
            if getattr(model, "_run_dir", None):
                path = os.path.join(model._run_dir, "best_consistency_check.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(consistency, f, indent=2)
            raise RuntimeError(f"[BEST_CONSISTENCY_FAIL] {fail_reason}")

        if getattr(model, "_run_dir", None):
            path = os.path.join(model._run_dir, "best_consistency_check.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(consistency, f, indent=2)
            if scorer_nonstationary:
                print(
                    "[BEST_CONSISTENCY] PASS with scorer drift: "
                    f"history={history_best_score:.6f} current={current_scorer_score:.6f} "
                    f"delta={diff_history_vs_current:.6f}"
                )
            else:
                print("[BEST_CONSISTENCY] PASS")

    result = dict(best_adj=best_adj)
    result["raw"] = result["best_adj"]
    result["best_graph_source"] = best_adj_source
    result["history_best_total_score"] = None if best_total_score is None else float(best_total_score)
    if consistency is not None:
        result["best_consistency"] = consistency
    best_score = None
    if getattr(model, "zs", None) is not None and getattr(model, "best_idx", None) is not None:
        try:
            best_z = model.zs[model.best_idx]
            best_score = graph_space.vec2score(best_z)
        except Exception:
            best_score = None
    if best_score is not None:
        result["score_S"] = best_score
    if best_adj is not None and hasattr(scorer, "_fit_node"):
        perturbation_dir = None
        if run_logger is not None and hasattr(run_logger, "run_dir"):
            perturbation_dir = run_logger.run_dir
        elif os.getenv("BO_RUN_DIR"):
            perturbation_dir = os.getenv("BO_RUN_DIR")
        t0 = time.perf_counter()
        perturbation = compute_parent_loo_perturbation(
            scorer=scorer,
            parent_adj=best_adj,
            run_dir=perturbation_dir,
            GT=GT,
        )
        result["perturbation_parent_loo"] = perturbation
        result["perturbation_time"] = float(time.perf_counter() - t0)
    if track_edge_freq and edge_freq_counts is not None and edge_freq_total > 0:
        result["edge_freq"] = edge_freq_counts / float(edge_freq_total)
        result["edge_freq_num_candidates"] = int(edge_freq_total)
    result = reporter.on_end(result, best_adj, scorer, GT, d)
    if return_model:
        result["model"] = model
    return result
