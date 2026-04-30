"""
MAML/Reptile-style meta-learning scorer for nonlinear time-series causal discovery.

Provides identical BO-loop interfaces: __call__, eval_graph_return_local,
eval_batch_return_local, aggregate_batch, last_stats, etc.

Key difference from old scorer:
  New:  maintain meta-parameters theta_meta per node.  For each candidate graph,
        copy theta_meta -> inner-loop fine-tune K steps -> evaluate adapted MSE.
        After a BO round, outer-loop Reptile update moves theta_meta toward the
        adapted parameters.
"""

import csv
import logging
import os
import time
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..trace.meta_trace import (
    append_jsonl as _trace_append_jsonl,
    collect_grad_stats,
    collect_param_stats,
    param_delta_from_snapshot,
    snapshot_params,
)

LOGGER = logging.getLogger(__name__)

# Utilities

def _append_csv(path: Optional[str], header: list, row: list) -> None:
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def _default_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

# Node MLP

class _NodeMLP(nn.Module):
    """Per-node regressor: R^{d*lag} -> R^1."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        dims = [int(input_dim)] + [int(h) for h in hidden_sizes] + [1]
        layers: list = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Main scorer

class NonlinearMultiLagScorer:
    """
    MAML/Reptile meta-learning scorer for BO-based time-series causal discovery.

    Lifecycle
    1. __init__        build lagged design, create meta_models (random init).
    2. pretrain()      (optional) warm-start theta_meta with random-mask training.
    3. BO loop calls   eval_batch_return_local / __call__ / aggregate_batch.
    4. meta_update()   Reptile outer-loop after each BO round.
    """

    # Construction

    def __init__(
        self,
        data: np.ndarray,
        lag: int = 1,
        hidden_sizes: Optional[Sequence[int]] = None,
        lambda_sparse: float = 1.0,
        mlp_random_state: int = 0,
        # inner-loop defaults
        inner_step_count: int = 5,
        inner_lr: float = 1e-3,
        inner_batch_size: int = 256,
        inner_optimizer: str = "sgd",
        # outer-loop defaults
        outer_lr: float = 0.01,
        # pre-train defaults
        pretrain_epochs: int = 0,
        pretrain_lr: float = 1e-3,
        pretrain_batch_size: int = 256,
        pretrain_avg_parents: int = 3,
        pretrain_weight_decay: float = 1e-5,
        pretrain_early_stop_patience: int = 10,
        pretrain_val_ratio: float = 0.1,
        # misc
        device: Optional[str] = None,
        **unused,
    ):
        X = np.asarray(data, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Data must be 2D (T, d), got {X.ndim}D")
        T, d = X.shape
        self.lag = int(lag)
        self.d = int(d)
        if self.lag < 1 or T <= self.lag:
            raise ValueError(f"Need T > lag >= 1, got T={T}, lag={self.lag}")

        hs = list(hidden_sizes or [64, 64])
        if len(hs) == 0:
            hs = [64, 64]
        if len(hs) == 1:
            hs = [hs[0], hs[0]]
        self.hidden_sizes = [int(h) for h in hs]

        self.lambda_sparse = float(lambda_sparse)
        self.inner_step_count = int(inner_step_count)
        self.inner_lr = float(inner_lr)
        self.inner_batch_size = int(inner_batch_size)
        self.inner_optimizer = str(inner_optimizer).strip().lower()
        if self.inner_optimizer not in {"sgd", "adam"}:
            raise ValueError(f"inner_optimizer must be 'sgd' or 'adam', got {inner_optimizer!r}")
        self.outer_lr = float(outer_lr)
        self.device = _default_device(device)
        self.input_dim = self.d * self.lag

        # reproducibility
        self._seed = int(mlp_random_state)
        self._rng = np.random.default_rng(self._seed)
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

        # build lagged design
        # Xp columns: [all_vars_lag1 | all_vars_lag2 | ... | all_vars_lagL]
        # col index for variable v at lag l: l * d + v   (l = 0 .. lag-1)
        Xp_list = []
        for l_idx in range(1, self.lag + 1):
            Xp_list.append(X[self.lag - l_idx: T - l_idx])
        self.Xp = np.hstack(Xp_list)          # (n, d*lag)
        self.Yn = X[self.lag:]                 # (n, d)
        self.n = int(self.Xp.shape[0])

        self.Xp_t = torch.as_tensor(self.Xp, dtype=torch.float32, device=self.device)
        self.Yn_t = torch.as_tensor(self.Yn, dtype=torch.float32, device=self.device)

        # baseline MSE per node (variance of targets, used for empty-parent fallback)
        self._baseline_mse = np.var(self.Yn, axis=0)  # (d,)

        # -- meta-parameters: one MLP per target node --
        self.models = nn.ModuleList(
            [_NodeMLP(self.input_dim, self.hidden_sizes) for _ in range(self.d)]
        ).to(self.device)

        # -- adaptation buffer (filled during eval, consumed by meta_update) --
        self._adaptation_diffs: List[List[Dict[str, torch.Tensor]]] = [
            [] for _ in range(self.d)
        ]

        # -- BO-loop compatibility fields --
        self._cache_hits = 0
        self._cache_misses = 0
        self._used_surrogate = False
        self._true_eval_calls = 0
        self.last_stats: dict = {}
        self.last_meta_stats: dict = {}
        self._log_scores = True
        self._trace = os.getenv("BO_TRACE") == "1"
        self._run_dir = os.getenv("BO_RUN_DIR") if self._trace else None
        self._printed_score_trace = False

        # -- meta trace hooks (no-op unless TRACE_META_TRACE=1) --
        self._trace_enabled = os.getenv("TRACE_META_TRACE") == "1"
        trace_log_env = os.getenv("TRACE_META_TRACE_LOG", "").strip()
        if trace_log_env:
            self._trace_log_path = trace_log_env
        elif self._run_dir:
            self._trace_log_path = os.path.join(self._run_dir, "meta_trace_log.jsonl")
        else:
            self._trace_log_path = None
        self._trace_inner_events = 0
        self._trace_mode_events = 0
        self._trace_optimizer_checked = False

        if self._trace_enabled:
            pstats = collect_param_stats(self.models)
            dropout_layers = sum(1 for m in self.models.modules() if isinstance(m, nn.Dropout))
            self._trace_event(
                "scorer_init",
                lag=int(self.lag),
                d=int(self.d),
                n=int(self.n),
                input_dim=int(self.input_dim),
                hidden_sizes=[int(h) for h in self.hidden_sizes],
                base_total_params=int(pstats.get("total_params", 0)),
                base_trainable_params=int(pstats.get("trainable_params", 0)),
                meta_total_params=int(pstats.get("total_params", 0)),
                meta_trainable_params=int(pstats.get("trainable_params", 0)),
                dropout_layers=int(dropout_layers),
                scorer_training=bool(self.models.training),
            )

        # -- optional pre-training --
        if pretrain_epochs > 0:
            self.pretrain(
                n_epochs=pretrain_epochs,
                lr=pretrain_lr,
                batch_size=pretrain_batch_size,
                avg_parents=pretrain_avg_parents,
                weight_decay=pretrain_weight_decay,
                early_stop_patience=pretrain_early_stop_patience,
                val_ratio=pretrain_val_ratio,
            )

    def _trace_mode_snapshot(self) -> dict:
        dropout_train = 0
        dropout_eval = 0
        dropout_total = 0
        for m in self.models.modules():
            if isinstance(m, nn.Dropout):
                dropout_total += 1
                if m.training:
                    dropout_train += 1
                else:
                    dropout_eval += 1
        return {
            "scorer_training": bool(self.models.training),
            "dropout_total": int(dropout_total),
            "dropout_train": int(dropout_train),
            "dropout_eval": int(dropout_eval),
        }

    def _trace_event(self, event: str, **payload) -> None:
        if not self._trace_enabled:
            return
        row = {
            "event": str(event),
            "time": float(time.time()),
            "seed": int(self._seed),
        }
        row.update(payload)
        _trace_append_jsonl(self._trace_log_path, row)

    # Mask construction  (layout: col = lag_index * d + var)

    def _build_parent_mask(self, parents: Sequence[int]) -> torch.Tensor:
        """Binary mask (d*lag,) with 1s at columns of parent variables across all lags."""
        mask = torch.zeros(self.input_dim, dtype=torch.float32, device=self.device)
        parent_set = sorted({int(p) for p in parents if 0 <= int(p) < self.d})
        for l in range(self.lag):
            base = l * self.d
            for p in parent_set:
                mask[base + p] = 1.0
        return mask

    @staticmethod
    def _adj_to_parents(adj: np.ndarray) -> List[List[int]]:
        """adj[j,i]=1 means j->i.  Returns parents[i] = [j, ...]."""
        d = adj.shape[0]
        return [np.where(adj[:, i] != 0)[0].tolist() for i in range(d)]

    def _get_parent_columns(self, parents: Sequence[int]) -> np.ndarray:
        """Return lagged feature-column indices for a parent set (compat helper)."""
        parent_set = sorted({int(p) for p in parents if 0 <= int(p) < self.d})
        if len(parent_set) == 0:
            return np.array([], dtype=int)
        cols = []
        for l in range(self.lag):
            base = l * self.d
            for p in parent_set:
                cols.append(base + p)
        return np.asarray(cols, dtype=int)

    def _fit_node(self, node: int, parents: Sequence[int], eps: float = 1e-12) -> float:
        """Compatibility API used by parent-LOO perturbation code.

        Runs the same inner-loop adaptation used during true scoring, but for a
        single (node, parent-set) query. This method does not write adaptation
        diffs, so it does not affect meta-update buffers.
        """
        j = int(node)
        if j < 0 or j >= self.d:
            raise ValueError(f"node index out of range: {j} for d={self.d}")

        parent_set = sorted({int(p) for p in parents if 0 <= int(p) < self.d})
        if len(parent_set) == 0:
            return max(float(self._baseline_mse[j]), eps)

        mask = self._build_parent_mask(parent_set)
        if float(mask.sum().item()) <= 0.0:
            return max(float(self._baseline_mse[j]), eps)

        mse, _adapted_state = self._inner_loop_node(
            node=j,
            mask=mask,
            inner_step_count=self.inner_step_count,
            inner_lr=self.inner_lr,
            batch_size=self.inner_batch_size,
            eps=eps,
        )
        return max(float(mse), eps)

    # Pre-training (meta initialisation)

    def pretrain(
        self,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 256,
        avg_parents: int = 3,
        weight_decay: float = 1e-5,
        early_stop_patience: int = 10,
        val_ratio: float = 0.1,
    ) -> None:
        """
        Warm-start theta_meta using structured random masks.  This is the 'meta initialisation' step that ensures
        inner-loop fine-tuning starts from a reasonable point.
        """
        avg_parents = max(1, min(avg_parents, self.d))

        # train / val split (shared across nodes, deterministic given seed)
        perm = self._rng.permutation(self.n)
        n_val = max(1, int(round(self.n * val_ratio)))
        n_val = min(n_val, self.n - 1)
        val_idx = perm[:n_val].astype(np.int64)
        train_idx = perm[n_val:].astype(np.int64)

        for node in range(self.d):
            model = self.models[node]
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            X_full = self.Xp_t
            y_full = self.Yn_t[:, node: node + 1]

            best_val = float("inf")
            best_state = None
            stale = 0

            for _epoch in range(n_epochs):
                # -- train --
                model.train()
                order = self._rng.permutation(len(train_idx))
                for start in range(0, len(order), batch_size):
                    batch = train_idx[order[start: start + batch_size]]
                    idx_t = torch.as_tensor(batch, dtype=torch.long, device=self.device)
                    mask = self._sample_random_mask(avg_parents)
                    pred = model(X_full[idx_t] * mask.unsqueeze(0))
                    loss = torch.mean((pred - y_full[idx_t]) ** 2)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # -- validate with random masks (MC) --
                model.eval()
                val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=self.device)
                val_losses = []
                with torch.no_grad():
                    for _ in range(4):
                        mask = self._sample_random_mask(avg_parents)
                        pred = model(X_full[val_idx_t] * mask.unsqueeze(0))
                        val_losses.append(torch.mean((pred - y_full[val_idx_t]) ** 2).item())
                val_loss = float(np.mean(val_losses))

                if val_loss + 1e-4 < best_val:
                    best_val = val_loss
                    stale = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= early_stop_patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

        LOGGER.info("Pre-training complete for %d nodes.", self.d)

    def _sample_random_mask(self, avg_parents: int) -> torch.Tensor:
        """Sample a structured variable-group mask with geometric parent-set size."""
        geom_p = 1.0 / max(1, avg_parents)
        n_vars = min(int(self._rng.geometric(geom_p)), self.d)
        if n_vars == 0:
            return torch.zeros(self.input_dim, dtype=torch.float32, device=self.device)
        selected = self._rng.choice(self.d, size=n_vars, replace=False)
        mask = torch.zeros(self.input_dim, dtype=torch.float32, device=self.device)
        for var in selected:
            for l in range(self.lag):
                mask[l * self.d + int(var)] = 1.0
        return mask

    # Inner loop: copy theta_meta -> fine-tune -> evaluate

    def _inner_loop_node(
        self,
        node: int,
        mask: torch.Tensor,
        inner_step_count: int,
        inner_lr: float,
        batch_size: int,
        eps: float = 1e-12,
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """
        MAML inner loop for one (graph, node) task.

        1. Deep-copy theta_meta for this node.
        2. Fine-tune K steps on the full training set under the given parent mask.
        3. Evaluate MSE on the full training set with adapted params.

        Returns (mse, adapted_state_dict).
        """
        meta_model = self.models[node]
        X_full = self.Xp_t                     # (n, d*lag)
        y_full = self.Yn_t[:, node: node + 1]  # (n, 1)
        mask_2d = mask.unsqueeze(0)             # (1, d*lag) for broadcasting

        # --- step 1: deep-copy theta_meta ---
        adapted_model = deepcopy(meta_model).to(self.device)

        # --- step 2: fine-tune K steps ---
        adapted_model.train()
        if self.inner_optimizer == "adam":
            opt = torch.optim.Adam(adapted_model.parameters(), lr=inner_lr)
        else:
            opt = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)

        if self._trace_enabled and not self._trace_optimizer_checked:
            opt_param_count = int(sum(int(p.numel()) for g in opt.param_groups for p in g["params"]))
            adapted_stats = collect_param_stats(adapted_model)
            self._trace_event(
                "inner_optimizer_check",
                optimizer=str(self.inner_optimizer),
                lr=float(inner_lr),
                optimizer_param_count=opt_param_count,
                adapted_total_params=int(adapted_stats.get("total_params", 0)),
                adapted_trainable_params=int(adapted_stats.get("trainable_params", 0)),
            )
            self._trace_optimizer_checked = True

        mse_before = None
        if self._trace_enabled:
            prev_training = bool(meta_model.training)
            meta_model.eval()
            with torch.no_grad():
                pred_before = meta_model(X_full * mask_2d)
                mse_before = float(torch.mean((pred_before - y_full) ** 2).item())
            if prev_training:
                meta_model.train()

        if self._trace_enabled:
            meta_model.zero_grad(set_to_none=True)

        n_samples = int(X_full.shape[0])
        eff_bs = int(batch_size) if batch_size is not None else n_samples
        if eff_bs <= 0:
            eff_bs = n_samples
        eff_bs = min(eff_bs, n_samples)

        for step_idx in range(inner_step_count):
            # Inner-loop optimization can run on a mini-batch; evaluation stays full-data.
            if eff_bs < n_samples:
                idx = self._rng.choice(n_samples, size=eff_bs, replace=False)
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
                x_step = X_full[idx_t] * mask_2d
                y_step = y_full[idx_t]
            else:
                x_step = X_full * mask_2d
                y_step = y_full

            pred = adapted_model(x_step)
            loss = torch.mean((pred - y_step) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            if self._trace_enabled and self._trace_inner_events < 500 and (
                step_idx == 0 or step_idx == (inner_step_count - 1)
            ):
                grad_base = collect_grad_stats(adapted_model)
                grad_meta = collect_grad_stats(meta_model)
                self._trace_event(
                    "inner_step_grad",
                    node=int(node),
                    step=int(step_idx),
                    inner_step_count=int(inner_step_count),
                    inner_lr=float(inner_lr),
                    inner_batch_size=int(eff_bs),
                    loss_mse=float(loss.item()),
                    grad_base_l2=float(grad_base.get("grad_l2", 0.0)),
                    grad_meta_l2=float(grad_meta.get("grad_l2", 0.0)),
                    grad_base_tensors=int(grad_base.get("n_grad_tensors", 0)),
                    grad_meta_tensors=int(grad_meta.get("n_grad_tensors", 0)),
                )
                self._trace_inner_events += 1

            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
            opt.step()

        # --- step 3: evaluate with adapted params ---
        adapted_model.eval()
        with torch.no_grad():
            pred_all = adapted_model(X_full * mask_2d)
            mse = torch.mean((pred_all - y_full) ** 2).item()
        if not np.isfinite(mse):
            mse = float(self._baseline_mse[node])
        mse = max(float(mse), eps)

        if self._trace_enabled and mse_before is not None and self._trace_inner_events < 1000:
            self._trace_event(
                "inner_loop_summary",
                node=int(node),
                mse_before=float(mse_before),
                mse_after=float(mse),
                mse_delta=float(mse - mse_before),
                **self._trace_mode_snapshot(),
            )
            self._trace_inner_events += 1

        # save adapted state (for outer loop)
        adapted_state = {k: v.detach().clone() for k, v in adapted_model.state_dict().items()}

        return mse, adapted_state

    # Score one graph  (inner-loop adapt all nodes, compute BIC)

    def _score_components(
        self,
        G: np.ndarray,
        eps: float = 1e-12,
        record_diffs: bool = True,
    ) -> Tuple[float, np.ndarray, float, float, int, int, np.ndarray]:
        """
        Evaluate one graph: inner-loop adapt per node, compute BIC.

        Returns (total_score, local_scores, fit_term, complexity_term,
                 edges_offdiag, diag_edges, mse_per_node).
        """
        G = np.asarray(G, dtype=int)
        parents_list = self._adj_to_parents(G)
        mse_per_node = np.zeros(self.d, dtype=np.float64)

        for node in range(self.d):
            parents = parents_list[node]

            # empty parent set -> baseline variance (no adaptation needed)
            if len(parents) == 0:
                mse_per_node[node] = max(float(self._baseline_mse[node]), eps)
                continue

            mask = self._build_parent_mask(parents)
            if mask.sum() == 0:
                mse_per_node[node] = max(float(self._baseline_mse[node]), eps)
                continue

            mse, adapted_state = self._inner_loop_node(
                node=node,
                mask=mask,
                inner_step_count=self.inner_step_count,
                inner_lr=self.inner_lr,
                batch_size=self.inner_batch_size,
                eps=eps,
            )
            mse_per_node[node] = mse

            # record adaptation diff for outer loop
            if record_diffs:
                meta_sd = self.models[node].state_dict()
                diff = {k: adapted_state[k] - meta_sd[k].detach() for k in meta_sd}
                self._adaptation_diffs[node].append(diff)

        # --- BIC assembly (identical to old scorer) ---
        local_scores = -float(self.n) * np.log(mse_per_node + eps)   # (d,)
        diag_edges = int(np.diag(G).sum())
        edges_offdiag = int(np.count_nonzero(G)) - diag_edges
        fit_term = float(np.sum(local_scores))
        complexity_term = float(self.lambda_sparse * edges_offdiag * np.log(self.n))
        total_score = fit_term - complexity_term

        return (total_score, local_scores, fit_term, complexity_term,
                edges_offdiag, diag_edges, mse_per_node)

    # BO-loop compatible interfaces

    def __call__(self, graph, return_components: bool = False):
        G = np.asarray(graph)
        if G.ndim != 2:
            raise ValueError(f"Graph must be 2D, got shape {G.shape}")

        (total, local_scores, fit_term, complexity_term,
         edges_offdiag, diag_edges, mse_per_node) = self._score_components(
            G, record_diffs=False
        )

        self.last_stats = {
            "n": int(self.n),
            "d": int(self.d),
            "lag": int(self.lag),
            "rss_dyn": float(np.sum(mse_per_node) * self.n),
            "sigma2_dyn_hat": float(np.mean(mse_per_node)),
            "mean_log_mse": float(np.mean(np.log(mse_per_node + 1e-12))),
            "fit_term": float(fit_term),
            "penalty": float(complexity_term),
            "total_score": float(total),
            "edges_total": int(edges_offdiag),
            "edges_offdiag": int(edges_offdiag),
            "diag_edges": int(diag_edges),
            "bic_penalty": float(complexity_term),
            "bic_lambda": float(self.lambda_sparse),
            "cache_hits": int(self._cache_hits),
            "cache_misses": int(self._cache_misses),
        }

        if self._trace and self._run_dir and self._log_scores:
            _append_csv(
                os.path.join(self._run_dir, "score_stats.csv"),
                ["rss_dyn", "edges_offdiag", "bic_penalty", "total_score",
                 "cache_hits", "cache_misses"],
                [self.last_stats["rss_dyn"], edges_offdiag, complexity_term,
                 total, self._cache_hits, self._cache_misses],
            )

        if return_components:
            return float(total), float(fit_term), float(complexity_term)
        return float(total)

    def eval_graph_return_local(
        self, dag, eps: float = 1e-12, return_stats: bool = False,
    ):
        """Return (total_score, local_log_mse) and optionally stats dict."""
        if self._trace_enabled and self._trace_mode_events < 200:
            self._trace_event(
                "eval_graph_mode",
                true_eval_call=int(self._true_eval_calls + 1),
                **self._trace_mode_snapshot(),
            )
            self._trace_mode_events += 1
        self._true_eval_calls += 1
        G = np.asarray(dag, dtype=int)

        (total, local_scores, fit_term, complexity_term,
         edges_offdiag, diag_edges, mse_per_node) = self._score_components(G, eps=eps)

        local_log_mse = np.log(mse_per_node + eps)

        self.last_stats = {
            "n": int(self.n),
            "d": int(self.d),
            "lag": int(self.lag),
            "rss_dyn": float(np.sum(mse_per_node) * self.n),
            "sigma2_dyn_hat": float(np.mean(mse_per_node)),
            "mean_log_mse": float(np.mean(local_log_mse)),
            "fit_term": float(fit_term),
            "penalty": float(complexity_term),
            "total_score": float(total),
            "edges_total": int(edges_offdiag),
            "edges_offdiag": int(edges_offdiag),
            "diag_edges": int(diag_edges),
            "bic_penalty": float(complexity_term),
            "bic_lambda": float(self.lambda_sparse),
            "cache_hits": int(self._cache_hits),
            "cache_misses": int(self._cache_misses),
        }

        if return_stats:
            return float(total), local_log_mse, dict(self.last_stats)
        return float(total), local_log_mse

    def eval_batch_return_local(
        self, dags, eps: float = 1e-12, return_stats: bool = False,
    ):
        """Batch version of eval_graph_return_local.  Returns (B, d+1) array."""
        dags_arr = np.asarray(dags)
        if dags_arr.ndim == 2:
            dags_arr = dags_arr[None, :]
        if dags_arr.ndim != 3:
            raise ValueError(f"Unexpected dags shape: {dags_arr.shape}")

        rows, stats_list = [], []
        for dag in dags_arr:
            if return_stats:
                total, llm, stats = self.eval_graph_return_local(dag, eps=eps, return_stats=True)
                rows.append(np.concatenate([llm, [total]]))
                stats_list.append(stats)
            else:
                total, llm = self.eval_graph_return_local(dag, eps=eps)
                rows.append(np.concatenate([llm, [total]]))

        result = np.asarray(rows, dtype=np.float64)
        if return_stats:
            return result, stats_list
        return result

    def batch_eval(self, dags):
        """Legacy interface used by some BO code paths."""
        dags_arr = np.asarray(dags)
        if dags_arr.ndim == 2:
            return np.asarray([[self(dags_arr)]], dtype=np.float64)
        if dags_arr.ndim == 3:
            return np.asarray([[self(dag)] for dag in dags_arr], dtype=np.float64)
        raise ValueError(f"Unexpected dags shape: {dags_arr.shape}")

    def aggregate_batch(self, dags, est_local_scores, eps: float = 1e-12):
        """Reconstruct total BIC from surrogate-predicted local log-MSE.
        Identical logic to old scorer so that Dropout_Local_BIC works unchanged."""
        self._used_surrogate = True
        est_local_scores = np.asarray(est_local_scores)
        if est_local_scores.ndim != 2:
            raise ValueError(f"est_local_scores must be (B,d), got {est_local_scores.shape}")
        B, _d = est_local_scores.shape
        n = self.n
        fit_terms = -float(n) * np.sum(est_local_scores, axis=1)
        dags_arr = np.asarray(dags)
        if dags_arr.ndim == 2:
            dags_arr = dags_arr[None, :]
        edge_counts = np.zeros(B, dtype=np.float64)
        for b in range(B):
            A = dags_arr[b]
            diag_edges = int(np.diag(A).sum())
            edge_counts[b] = float(np.count_nonzero(A) - diag_edges)
        return fit_terms - edge_counts * np.log(float(n) + eps) * self.lambda_sparse

    # ------------------------------------------------------------------
    # Meta outer-loop  (Reptile)
    # ------------------------------------------------------------------

    def meta_update(
        self,
        graphs_current: Optional[Sequence[np.ndarray]] = None,
        graphs_history: Optional[Sequence[np.ndarray]] = None,
        outer_lr: Optional[float] = None,
        inner_step_count: Optional[int] = None,
        inner_lr: Optional[float] = None,
        inner_batch_size: Optional[int] = None,
        current_task_frac: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Reptile outer-loop update of theta_meta.

        Two sources of tasks:
          graphs_current  - diffs already recorded during eval_batch_return_local
                            in this BO round (pass the adjs for bookkeeping only;
                            the diffs are in self._adaptation_diffs).
          graphs_history  - additional historical graphs: inner-loop is run fresh
                            for these to compute new diffs.

        Either or both may be provided.  If graphs_current is given, we use the
        diffs that were already accumulated.  If graphs_history is given, we run
        inner-loop for each and add their diffs to the batch.
        """
        if outer_lr is None:
            outer_lr = self.outer_lr
        if inner_step_count is None:
            inner_step_count = self.inner_step_count
        if inner_lr is None:
            inner_lr = self.inner_lr
        if inner_batch_size is None:
            inner_batch_size = self.inner_batch_size
        if current_task_frac is None:
            current_task_frac = 1.0
        current_task_frac = float(np.clip(float(current_task_frac), 0.0, 1.0))

        meta_snapshot_before = snapshot_params(self.models) if self._trace_enabled else []
        mode_before = self._trace_mode_snapshot() if self._trace_enabled else {}
        history_mse_before: list[float] = []
        history_mse_after: list[float] = []
        if self._trace_enabled:
            self._trace_event(
                "meta_update_start",
                graphs_current_count=int(len(graphs_current)) if graphs_current is not None else 0,
                graphs_history_count=int(len(graphs_history)) if graphs_history is not None else 0,
                outer_lr=float(outer_lr),
                inner_step_count=int(inner_step_count),
                inner_lr=float(inner_lr),
                inner_batch_size=int(inner_batch_size),
                current_task_frac=float(current_task_frac),
                **mode_before,
            )

        current_count = int(len(graphs_current)) if graphs_current is not None else 0
        use_current = current_count > 0 and current_task_frac > 0.0
        update_diffs: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(self.d)]
        current_tasks_per_node: List[int] = [0 for _ in range(self.d)]
        if use_current:
            for node in range(self.d):
                node_diffs = list(self._adaptation_diffs[node])
                if len(node_diffs) == 0:
                    continue
                if current_task_frac < 1.0:
                    keep = max(1, int(round(len(node_diffs) * current_task_frac)))
                    keep = min(len(node_diffs), keep)
                    if keep < len(node_diffs):
                        idx = self._rng.choice(len(node_diffs), size=keep, replace=False)
                        node_diffs = [node_diffs[int(i)] for i in np.asarray(idx).tolist()]
                current_tasks_per_node[node] = int(len(node_diffs))
                update_diffs[node].extend(node_diffs)

        # --- process historical graphs (run fresh inner loops) ---
        history_count = 0
        if graphs_history is not None and len(graphs_history) > 0:
            for adj in graphs_history:
                history_count += 1
                G = np.asarray(adj, dtype=int)
                parents_list = self._adj_to_parents(G)
                for node in range(self.d):
                    parents = parents_list[node]
                    if len(parents) == 0:
                        continue
                    mask = self._build_parent_mask(parents)
                    if mask.sum() == 0:
                        continue
                    mse_before_local = None
                    if self._trace_enabled:
                        meta_model = self.models[node]
                        prev_training = bool(meta_model.training)
                        meta_model.eval()
                        with torch.no_grad():
                            pred_before = meta_model(self.Xp_t * mask.unsqueeze(0))
                            mse_before_local = float(torch.mean((pred_before - self.Yn_t[:, node: node + 1]) ** 2).item())
                        if prev_training:
                            meta_model.train()

                    _mse, adapted_state = self._inner_loop_node(
                        node=node, mask=mask,
                        inner_step_count=inner_step_count, inner_lr=inner_lr,
                        batch_size=inner_batch_size,
                    )
                    if self._trace_enabled and mse_before_local is not None:
                        history_mse_before.append(float(mse_before_local))
                        history_mse_after.append(float(_mse))

                    meta_sd = self.models[node].state_dict()
                    diff = {k: adapted_state[k] - meta_sd[k].detach() for k in meta_sd}
                    update_diffs[node].append(diff)

        # --- Reptile update: theta_meta += outer_lr * mean(diffs) ---
        n_tasks_per_node: List[int] = []
        mean_l2_per_node: List[float] = []
        mean_norm_per_node: List[float] = []
        ratio_per_node: List[float] = []

        for node in range(self.d):
            diffs = update_diffs[node]
            n_tasks = len(diffs)
            n_tasks_per_node.append(n_tasks)
            if n_tasks == 0:
                mean_l2_per_node.append(0.0)
                mean_norm_per_node.append(0.0)
                ratio_per_node.append(0.0)
                continue

            meta_sd = self.models[node].state_dict()
            new_sd = {}
            avg_delta_by_key: Dict[str, torch.Tensor] = {}
            for k, v_meta in meta_sd.items():
                avg_delta = torch.stack([d[k] for d in diffs]).mean(dim=0)
                avg_delta_by_key[k] = avg_delta
                new_sd[k] = v_meta.detach() + float(outer_lr) * avg_delta

            mean_l2_sq = 0.0
            for avg_delta in avg_delta_by_key.values():
                mean_l2_sq += float(torch.sum(avg_delta * avg_delta).item())
            mean_l2 = float(np.sqrt(mean_l2_sq))

            task_norms = []
            for diff in diffs:
                norm_sq = 0.0
                for dv in diff.values():
                    norm_sq += float(torch.sum(dv * dv).item())
                task_norms.append(float(np.sqrt(norm_sq)))
            mean_norm = float(np.mean(task_norms)) if task_norms else 0.0
            ratio = float(mean_l2 / (mean_norm + 1e-12))

            mean_l2_per_node.append(mean_l2)
            mean_norm_per_node.append(mean_norm)
            ratio_per_node.append(ratio)
            self.models[node].load_state_dict(new_sd)

        # --- clear buffer used by eval_batch_return_local ---
        self.clear_adaptation_buffer()

        mean_tasks = float(np.mean(n_tasks_per_node)) if n_tasks_per_node else 0.0
        total_tasks = int(sum(n_tasks_per_node)) if n_tasks_per_node else 0
        nodes_updated = int(np.sum(np.asarray(n_tasks_per_node) > 0)) if n_tasks_per_node else 0
        active_idx = [i for i, n_tasks in enumerate(n_tasks_per_node) if n_tasks > 0]
        mean_l2 = float(np.mean([mean_l2_per_node[i] for i in active_idx])) if active_idx else 0.0
        mean_norm = float(np.mean([mean_norm_per_node[i] for i in active_idx])) if active_idx else 0.0
        mean_ratio = float(np.mean([ratio_per_node[i] for i in active_idx])) if active_idx else 0.0

        current_tasks_total = int(sum(current_tasks_per_node)) if current_tasks_per_node else 0
        used_current_tasks = bool(current_tasks_total > 0)

        meta_grad_stats = collect_grad_stats(self.models) if self._trace_enabled else {}
        mode_after = self._trace_mode_snapshot() if self._trace_enabled else {}
        meta_delta_stats = (
            param_delta_from_snapshot(meta_snapshot_before, self.models)
            if self._trace_enabled else {"param_delta_l2": 0.0, "param_delta_max_abs": 0.0}
        )
        hist_before_mean = float(np.mean(history_mse_before)) if history_mse_before else None
        hist_after_mean = float(np.mean(history_mse_after)) if history_mse_after else None

        self.last_meta_stats = {
            "mean_tasks_per_node": mean_tasks,
            "total_tasks": total_tasks,
            "nodes_updated": nodes_updated,
            "outer_lr": float(outer_lr),
            "inner_step_count": int(inner_step_count),
            "inner_lr": float(inner_lr),
            "inner_batch_size": int(inner_batch_size),
            "inner_optimizer": str(self.inner_optimizer),
            "current_graphs": int(current_count),
            "history_graphs": int(history_count),
            "current_task_frac": float(current_task_frac),
            "current_tasks_total": int(current_tasks_total),
            "current_tasks_per_node": [int(v) for v in current_tasks_per_node],
            "used_current_graphs": bool(used_current_tasks),
            "num_tasks_per_node": [int(v) for v in n_tasks_per_node],
            "mean_l2_per_node": [float(v) for v in mean_l2_per_node],
            "mean_norm_per_node": [float(v) for v in mean_norm_per_node],
            "ratio_per_node": [float(v) for v in ratio_per_node],
            "mean_l2": mean_l2,
            "mean_norm": mean_norm,
            "mean_ratio": mean_ratio,
            "avg_meta_ratio": mean_ratio,
            "avg_num_tasks": mean_tasks,
            "meta_param_delta_l2": float(meta_delta_stats.get("param_delta_l2", 0.0)),
            "meta_param_delta_max_abs": float(meta_delta_stats.get("param_delta_max_abs", 0.0)),
            "meta_grad_l2": float(meta_grad_stats.get("grad_l2", 0.0)),
            "meta_grad_tensors": int(meta_grad_stats.get("n_grad_tensors", 0)),
            "history_mse_before_mean": hist_before_mean,
            "history_mse_after_mean": hist_after_mean,
            "history_mse_delta_mean": (
                None if (hist_before_mean is None or hist_after_mean is None)
                else float(hist_after_mean - hist_before_mean)
            ),
        }

        if self._trace_enabled:
            self._trace_event(
                "meta_update_trace",
                mean_tasks_per_node=float(mean_tasks),
                total_tasks=int(total_tasks),
                nodes_updated=int(nodes_updated),
                meta_param_delta_l2=float(meta_delta_stats.get("param_delta_l2", 0.0)),
                meta_param_delta_max_abs=float(meta_delta_stats.get("param_delta_max_abs", 0.0)),
                grad_meta_l2=float(meta_grad_stats.get("grad_l2", 0.0)),
                grad_meta_tensors=int(meta_grad_stats.get("n_grad_tensors", 0)),
                history_mse_before_mean=hist_before_mean,
                history_mse_after_mean=hist_after_mean,
                history_mse_delta_mean=(
                    None if (hist_before_mean is None or hist_after_mean is None)
                    else float(hist_after_mean - hist_before_mean)
                ),
                **mode_before,
                **{f"after_{k}": v for k, v in mode_after.items()},
            )

        return dict(self.last_meta_stats)

    def clear_adaptation_buffer(self) -> None:
        """Discard accumulated diffs without applying an update."""
        self._adaptation_diffs = [[] for _ in range(self.d)]

    def get_last_stats(self) -> dict:
        """Compatibility accessor for BO-loop instrumentation/traceging."""
        return dict(self.last_stats)
