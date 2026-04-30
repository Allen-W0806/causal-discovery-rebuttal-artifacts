
import csv
import hashlib
import json
import os
import time
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from ..surrogate.nets import Dropout_Local_BIC

def _append_csv(path, header, row):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def _write_text(path, text):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))


def _write_json(path, payload):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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

class BaseGraphOptim:
    def __init__(self, graph_space, X, GT, logger, pruner, scorer):
        self.graph_space = graph_space
        self.X = X
        self.GT = GT
        self.logger = logger
        self.pruner = (lambda X, adj: adj) if pruner is None else pruner
        self.scorer = scorer
        self.nodes = self.graph_space.nodes
        self.best_adj = self.best_score = None
        self.cnt = 0
        self.n_steps = 0
        self.unique_adjs = set()
        self.best_adj_snapshot = None
        self.best_summary = None
        self.best_update_eval_idx = None
        self.best_score_at_update = None
        self.best_recomputed_score_at_update = None
        self.best_update_log_path = None
        self._best_source = "model.best_adj"

    def _snapshot_best_update(self, best_adj, best_score):
        if getattr(self, "_run_dir", None) is None:
            return

        eval_idx = int(self.best_idx) + 1
        best_adj_np = np.array(best_adj, copy=True)
        summary = _graph_summary(best_adj_np)
        summary.update({
            "eval_idx": int(eval_idx),
            "best_total_score": float(best_score),
            "source": "BaseGraphOptim.add_data",
        })

        graph_path = os.path.join(self._run_dir, f"best_graph_at_update_iter{eval_idx}.npy")
        score_path = os.path.join(self._run_dir, f"best_total_score_at_update_iter{eval_idx}.txt")
        summary_path = os.path.join(self._run_dir, f"best_graph_at_update_iter{eval_idx}_summary.json")
        np.save(graph_path, best_adj_np)
        _write_text(score_path, f"{float(best_score):.16f}\n")

        recomputed_score = None
        scorer_obj = getattr(self, "scorer", None)
        if scorer_obj is not None and callable(scorer_obj):
            log_flag = None
            if hasattr(scorer_obj, "_log_scores"):
                log_flag = scorer_obj._log_scores
                scorer_obj._log_scores = False
            try:
                recomputed_score = float(scorer_obj(best_adj_np))
            finally:
                if log_flag is not None:
                    scorer_obj._log_scores = log_flag

        summary["recomputed_score_at_update"] = recomputed_score
        summary["recomputed_score_diff"] = (
            None if recomputed_score is None else float(recomputed_score - float(best_score))
        )
        _write_json(summary_path, summary)

        self.best_adj_snapshot = best_adj_np
        self.best_summary = dict(summary)
        self.best_update_eval_idx = int(eval_idx)
        self.best_score_at_update = float(best_score)
        self.best_recomputed_score_at_update = recomputed_score
        self.best_update_log_path = os.path.join(self._run_dir, "best_update_log.csv")
        _append_csv(
            self.best_update_log_path,
            [
                "eval_idx",
                "best_total_score",
                "recomputed_score_at_update",
                "recomputed_score_diff",
                "offdiag_edge_count",
                "offdiag_density",
                "diag_edges",
                "hash_offdiag_md5",
            ],
            [
                int(eval_idx),
                float(best_score),
                "" if recomputed_score is None else float(recomputed_score),
                "" if recomputed_score is None else float(recomputed_score - float(best_score)),
                int(summary["offdiag_edge_count"]),
                float(summary["offdiag_density"]),
                int(summary["diag_edges"]),
                str(summary["hash_offdiag_md5"]),
            ],
        )

    def add_data(self, zs, adjs, scores, **kwargs):
        if self.best_score is None or self.best_score < scores[:, -1].max():
            self.best_score = float(scores[:, -1].max())
            best_idx = np.argmax(scores[:, -1]).item()
            best_adj = np.array(self._select_adj(adjs, best_idx), copy=True)
            best_score = float(scores[best_idx, -1])
            self.best_adj = np.array(best_adj, copy=True)
            self.best_idx = best_idx + self.cnt
            self._best_source = "BaseGraphOptim.add_data(scores[:, -1])"
            if getattr(self, "_run_dir", None):
                self._snapshot_best_update(best_adj, best_score)
                stats = getattr(self.scorer, "last_stats", None) or {}
                _append_csv(
                    os.path.join(self._run_dir, "best_progress.csv"),
                    ["eval_idx", "best_total_score", "edges_total", "rss_dyn", "bic_penalty"],
                    [
                        int(self.best_idx) + 1,
                        float(best_score),
                        stats.get("edges_total"),
                        stats.get("rss_dyn"),
                        stats.get("bic_penalty"),
                    ],
                )

        self.cnt += len(zs)
        self.n_steps += 1
        if self.best_score is not None:
            if self.logger.verbose:
                self.logger(f'Best={self.best_score:.6f}@{self.best_idx}')
                self.logger.add(best_score=self.best_score)
        if hasattr(self, "_maybe_log_timing"):
            self._maybe_log_timing()
        self.update(zs, adjs, scores, **kwargs)

    def _adj_matrix(self, adj):
        return adj

    def _select_adj(self, adjs, idx):
        if isinstance(adjs, list):
            return adjs[idx]
        arr = np.asarray(adjs)
        if arr.ndim == 2:
            return arr
        return arr[idx]

    def _select_adjs(self, adjs, indices):
        if isinstance(adjs, list):
            return [adjs[i] for i in indices]
        arr = np.asarray(adjs)
        if arr.ndim == 2:
            return arr
        return arr[indices]

# from TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/utils.py#L15
def to_unit_cube(x, lb, ub):
    xx = (x - lb) / (ub - lb)
    return xx

# from TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/utils.py#L22
def from_unit_cube(x, lb, ub):
    xx = x * (ub - lb) + lb
    return xx

# from TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/utils.py#L29
def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X

class GraphOptimBO(BaseGraphOptim):
    def __init__(self, graph_space, X, GT, lr, n_cands, n_grads, n_replay, dropout, hidden_size, device, max_size, logger, pruner, scorer, init_adjs=None):
        super().__init__(graph_space, X, GT, logger, pruner, scorer)
        for k, v in locals().items():
            if k not in ['self', 'unused', '__class__'] and k not in self.__dict__:
                setattr(self, k, v)

        self.init_adjs = init_adjs
        self.init_used = False
        self.bic_model = Dropout_Local_BIC(
            adj_space=graph_space,
            scorer=scorer,
            max_size=max_size,
            lr=lr,
            n_replay=n_replay,
            dropout=dropout,
            hidden_size=hidden_size,
            GT=GT,
            logger=logger,
            n_grads=n_grads,
            device=device
        )
        self.zs = self.bics = None
        self.length = 1
        self.succcount = self.failcount = 0
        self.suctol, self.failtol = 3, 5
        self.min_length, self.max_length = .01, 2
        self._trace = os.getenv("BO_TRACE") == "1"
        self._run_dir = os.getenv("BO_RUN_DIR") if self._trace else None
        self._timing = {
            "vec2adj": 0.0,
            "surrogate_sample": 0.0,
            "true_eval": 0.0,
            "surrogate_train": 0.0,
        } if self._run_dir else None
        self._timing_last_report = 0

    def _inject_init_adjs(self, adj_cands, init_adjs, batch_size):
        n_init = min(len(init_adjs), batch_size)
        Gxxs = np.array(adj_cands, copy=True)
        if Gxxs.ndim == 2:
            return init_adjs[0]
        for i in range(n_init):
            Gxxs[i] = init_adjs[i]
        return Gxxs

    def get_center(self):
        return self.zs[self.best_idx]

    def update(self, zs, adjs, scores):
        t0 = time.perf_counter() if self._timing is not None else None
        self.bic_model.train(zs, adjs, scores)
        if t0 is not None:
            self._timing["surrogate_train"] += time.perf_counter() - t0
        self.zs = zs if self.zs is None else np.concatenate((self.zs, zs))

    def add_data(self, zs, adjs, scores, **kwargs):
        # based on TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/turbo_1.py#L137
        if self.best_score is not None:
            if self.best_score < scores[:, -1].max():
                self.succcount += 1
                self.failcount = 0
            else:
                self.succcount = 0
                self.failcount += 1
        
        if self.succcount >= self.suctol:
            self.length = min(2 * self.length, self.max_length)
            self.succcount = 0
        elif self.failcount >= self.failtol:
            self.length = self.length / 2
            self.failcount = 0
        
        if self.length < self.min_length:
            self.length = self.max_length

        return super().add_data(zs, adjs, scores, **kwargs)

    def suggest(self, batch_size, **unused):
        if self.best_adj is None:
            zs = latin_hypercube(batch_size, self.graph_space.dim)
            zs = from_unit_cube(zs, -1, 1)
            t0 = time.perf_counter() if self._timing is not None else None
            adj_cands = self.graph_space.vec2adj(zs)
            if t0 is not None:
                self._timing["vec2adj"] += time.perf_counter() - t0
            if self.init_adjs is not None and not self.init_used:
                adj_cands = self._inject_init_adjs(adj_cands, self.init_adjs, batch_size)
                self.init_used = True
            return zs, adj_cands, np.zeros(batch_size)
        else:
            # based on TurBO (Eriksson et al., 2019): https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/turbo_1.py#L181
            x_center_unit = to_unit_cube(self.get_center(), -1, 1)  # dim
            weights = 1
            length = self.length
            lb = np.clip(x_center_unit - weights * length / 2.0, 0.0, 1.0)
            ub = np.clip(x_center_unit + weights * length / 2.0, 0.0, 1.0)

            seed = np.random.randint(int(1e6))
            sobol = SobolEngine(self.graph_space.dim, scramble=True, seed=seed)
            pert = sobol.draw(self.n_cands).numpy()
            pert = lb + (ub - lb) * pert

            prob_perturb = min(20.0 / self.graph_space.dim, 1.0)
            mask = (np.random.rand(self.n_cands, self.graph_space.dim) <= prob_perturb)


            ind = np.where(np.sum(mask, axis=1) == 0)[0]
            mask[ind, np.random.randint(0, self.graph_space.dim - 1, size=len(ind))] = 1

            X_cand_unit = x_center_unit.copy() * np.ones((self.n_cands, self.graph_space.dim))  # n_cand x dim
            X_cand_unit[mask] = pert[mask]
            zs = from_unit_cube(X_cand_unit, -1, 1)

            t0 = time.perf_counter() if self._timing is not None else None
            adj_cands = self.graph_space.vec2adj(zs)
            if t0 is not None:
                self._timing["vec2adj"] += time.perf_counter() - t0

            t1 = time.perf_counter() if self._timing is not None else None
            est_scores = self.bic_model.sample(zs=zs, adjs=adj_cands, num_samples=1)
            if t1 is not None:
                self._timing["surrogate_sample"] += time.perf_counter() - t1
            est_scores = np.asarray(est_scores)

            if est_scores.ndim == 2:
                if est_scores.shape[0] != self.n_cands:
                    raise ValueError(f"Unexpected est_scores shape: {est_scores.shape}, expected first dim {self.n_cands}")
                if est_scores.shape[1] == 1:
                    est_scores = est_scores.reshape(-1)
                else:
                    est_scores = est_scores.max(axis=1)
            elif est_scores.ndim == 1:
                if est_scores.shape[0] != self.n_cands:
                    raise ValueError(f"Unexpected est_scores shape: {est_scores.shape}, expected ({self.n_cands},)")
            else:
                raise ValueError(f"Unexpected est_scores ndim: {est_scores.ndim}, shape={est_scores.shape}")

            topk = np.argpartition(-est_scores, kth=batch_size)[:batch_size]
            return zs[topk], self._select_adjs(adj_cands, topk), est_scores[topk]

    def record_true_eval_time(self, dt):
        if self._timing is not None:
            self._timing["true_eval"] += dt

    def _maybe_log_timing(self):
        if self._timing is None:
            return
        if (self.cnt - self._timing_last_report) < 20:
            return
        total = sum(self._timing.values())
        if total <= 0:
            return
        ratios = {k: self._timing[k] / total for k in self._timing}
        _append_csv(
            os.path.join(self._run_dir, "timing.csv"),
            [
                "eval_idx",
                "vec2adj",
                "surrogate_sample",
                "true_eval",
                "surrogate_train",
                "vec2adj_ratio",
                "surrogate_sample_ratio",
                "true_eval_ratio",
                "surrogate_train_ratio",
            ],
            [
                int(self.cnt),
                self._timing["vec2adj"],
                self._timing["surrogate_sample"],
                self._timing["true_eval"],
                self._timing["surrogate_train"],
                ratios["vec2adj"],
                ratios["surrogate_sample"],
                ratios["true_eval"],
                ratios["surrogate_train"],
            ],
        )
        if os.getenv("BO_TIMING_PRINT") == "1":
            print(
                "timing@{} vec2adj={:.2%} sample={:.2%} true_eval={:.2%} train={:.2%}".format(
                    self.cnt,
                    ratios["vec2adj"],
                    ratios["surrogate_sample"],
                    ratios["true_eval"],
                    ratios["surrogate_train"],
                )
            )
        self._timing_last_report = self.cnt

# Backward-compatible aliases were removed to avoid DAG-specific naming.
