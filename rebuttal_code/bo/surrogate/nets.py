import csv
import gc
import os
import numpy as np
from torch import nn
import torch

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

def SingleNodeMLP(input_dim, hidden_sizes, batch_norm, dropout, activation, dtype, **unused):
    last_dim = input_dim
    act = {
        'relu': lambda: nn.ReLU(inplace=True),
        'tanh': lambda: nn.Tanh(),
    }[activation]
    layers = []
    for h in hidden_sizes:
        layer = [nn.Linear(last_dim, h, dtype=dtype), nn.Dropout(p=dropout, inplace=True), act()]
        if batch_norm:
            layer.append(nn.BatchNorm1d(h))
        layers.extend(layer)
        last_dim = h
    layers.append(nn.Linear(last_dim, 1, dtype=dtype))
    return nn.Sequential(*layers)

class MultiSingleNodeMLP(nn.Module):
    def __init__(self, nodes_out, input_dim, hidden_sizes, batch_norm, dropout, activation, dtype, **unused):
        super().__init__()
        self.mlps = nn.ModuleList(
            (SingleNodeMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, batch_norm=batch_norm, dropout=dropout, activation=activation, dtype=dtype, **unused) for i in range(nodes_out))
        )
    
    def forward(self, X):
        xs = torch.unbind(X, dim=-1)
        xs = [mlp(x) for mlp, x in zip(self.mlps, xs)]
        return torch.concat(xs, dim=-1)

class Dropout_Local_BIC:
    def __init__(self, max_size, adj_space, hidden_size, dropout, n_replay, GT, logger, scorer, device, n_grads, lr, sampling_chunksize=None, **unused):
        for k, v in locals().items():
            if k not in ['self', 'unused', '__class__'] and k not in self.__dict__:
                setattr(self, k, v)
        self.logs = []
        self._trace = os.getenv("BO_TRACE") == "1"
        self._run_dir = os.getenv("BO_RUN_DIR")
        self._loss_step = 0
        self.nodes = self.adj_space.nodes
        self.in_dim = self.nodes
        self.dtype = torch.float32
        # Only use pin_memory for CUDA devices
        pin_memory = isinstance(device, torch.device) and device.type == 'cuda' or (isinstance(device, str) and device == 'cuda')
        self.adjs = torch.empty((max_size, self.in_dim, self.nodes), dtype=torch.uint8, pin_memory=pin_memory)
        self.local_scores = torch.empty((max_size, adj_space.nodes), dtype=self.dtype, pin_memory=pin_memory)
        self.idx = 0
        self.model = MultiSingleNodeMLP(nodes_out=self.nodes, input_dim=self.in_dim, dtype=self.dtype, hidden_sizes=[hidden_size], batch_norm=True, dropout=dropout, activation='relu').to(device=self.device, dtype=self.dtype)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._trace_shape = os.getenv("BO_TRACE") == "1"
        self._printed_shape = False

    def _maybe_trace_print(self, out):
        if self._trace_shape and not self._printed_shape:
            try:
                shape = tuple(out.shape)
            except Exception:
                shape = "unknown"
            print("SURROGATE_OUT_SHAPE", shape)
            try:
                if out.ndim >= 2:
                    print("SURROGATE_OUT_HEAD", out[0][:5])
                else:
                    print("SURROGATE_OUT_HEAD", out[:5])
            except Exception:
                pass
            self._printed_shape = True

    def _feat_from_adjs(self, adjs):
        Gxx = np.asarray(adjs)
        if Gxx.ndim == 2:
            Gxx = Gxx[None]
        if Gxx.ndim != 3:
            raise ValueError("Expected adjacencies with shape (B,d,d) or (d,d).")
        return Gxx.astype(np.uint8)
    
    def sample(self, zs, adjs, num_samples, **kwargs):
        est_scores = []
        if isinstance(adjs, np.ndarray):
            batch_len = 1 if adjs.ndim == 2 else adjs.shape[0]
        else:
            raise ValueError("Unsupported adjacency format.")
        chunksize = batch_len if self.sampling_chunksize is None else self.sampling_chunksize
        for i in range(0, batch_len, chunksize):
            if adjs.ndim == 2:
                adjs_chunk = adjs
            else:
                adjs_chunk = adjs[i: i + chunksize]
            feats = self._feat_from_adjs(adjs_chunk)
            adjs_torch = torch.as_tensor(feats, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                est_scores_ = np.stack([self.model(adjs_torch).cpu().numpy() for i in range(num_samples)], axis=1)
            if num_samples == 1:
                self._maybe_trace_print(est_scores_[:, 0, :] if est_scores_.ndim == 3 else est_scores_)
            del adjs_torch
            est_scores.append(est_scores_)
            torch.cuda.empty_cache()
            gc.collect()

        est_scores = np.concatenate(est_scores)

        if num_samples == 1:
            est_local_scores = est_scores[:, 0, :] if est_scores.ndim == 3 else est_scores
            est_scores = self.scorer.aggregate_batch(adjs, est_local_scores)
            est_scores = np.asarray(est_scores).reshape(-1)
            return est_scores
        est_local_scores = est_scores
        out = []
        for s in range(num_samples):
            est_scores_s = self.scorer.aggregate_batch(adjs, est_local_scores[:, s, :])
            out.append(np.asarray(est_scores_s).reshape(-1))
        return np.stack(out, axis=-1)

    def predict_local(self, zs, adjs, num_samples=1):
        est_scores = []
        if isinstance(adjs, np.ndarray):
            batch_len = 1 if adjs.ndim == 2 else adjs.shape[0]
        else:
            raise ValueError("Unsupported adjacency format.")
        chunksize = batch_len if self.sampling_chunksize is None else self.sampling_chunksize
        for i in range(0, batch_len, chunksize):
            if adjs.ndim == 2:
                adjs_chunk = adjs
            else:
                adjs_chunk = adjs[i: i + chunksize]
            feats = self._feat_from_adjs(adjs_chunk)
            adjs_torch = torch.as_tensor(feats, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                est_scores_ = np.stack([self.model(adjs_torch).cpu().numpy() for i in range(num_samples)], axis=1)
            if num_samples == 1:
                self._maybe_trace_print(est_scores_[:, 0, :] if est_scores_.ndim == 3 else est_scores_)
            del adjs_torch
            est_scores.append(est_scores_)
            torch.cuda.empty_cache()
            gc.collect()

        est_scores = np.concatenate(est_scores)
        if num_samples == 1:
            est_local_scores = est_scores[:, 0, :] if est_scores.ndim == 3 else est_scores
            est_local_scores = np.asarray(est_local_scores)
            assert est_local_scores.ndim == 2 and est_local_scores.shape[1] == self.nodes
            return est_local_scores
        est_local_scores = np.asarray(est_scores)
        assert est_local_scores.ndim == 3 and est_local_scores.shape[2] == self.nodes
        return est_local_scores
    
    def prepare_data(self, adjs, scores, new_adjs, new_scores):
        old_size = self.idx - len(new_adjs)
        if old_size == 0:
            return new_adjs.to(self.device), new_scores.to(self.device)
        buffer = torch.multinomial(torch.as_tensor([1 / old_size] * old_size), num_samples=min(old_size, self.n_replay))
        buffer = torch.concat((buffer, torch.arange(old_size, self.idx)))
        X, y = adjs[buffer].to(self.device, dtype=self.dtype), scores[buffer].to(self.device)
        return X, y
    
    def train(self, zs, adjs, scores, **unused):
        scores = scores[:, :self.nodes]
        feats = self._feat_from_adjs(adjs)
        feats_torch = torch.as_tensor(feats, dtype=self.dtype)
        scores_torch = torch.as_tensor(scores, dtype=self.dtype)
        self.adjs[self.idx: self.idx + len(feats_torch)] = feats_torch
        self.local_scores[self.idx: self.idx + len(scores_torch)] = scores_torch
        self.idx += len(feats_torch)
            
        for step in range(self.n_grads):
            X, y = self.prepare_data(self.adjs, self.local_scores, feats_torch, scores_torch)
            # BatchNorm requires batch_size > 1; duplicate when only one sample.
            if X.shape[0] == 1:
                X = torch.cat([X, X], dim=0)
                y = torch.cat([y, y], dim=0)
            pred = self.model(X)
            loss = torch.nn.functional.mse_loss(pred, y)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)
            self.opt.step()
            loss = loss.item()
            self.logs.append(loss)
            if self._run_dir:
                _append_csv(
                    os.path.join(self._run_dir, "surrogate_fit.csv"),
                    ["step", "loss_name", "loss_value"],
                    [int(self._loss_step), "dropout_nn_mse", float(loss)],
                )
            if self._trace and self._loss_step % 10 == 0:
                print(f"surrogate_fit@{self._loss_step} dropout_nn_mse={loss:.6f}")
            self._loss_step += 1
            del X, y
            torch.cuda.empty_cache()
            gc.collect()
