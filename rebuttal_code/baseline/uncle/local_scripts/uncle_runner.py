"""
UnCLe runner adapted for the sim_fmri_68_static_large dataset.

This is a self-contained adaptation of uncle/bin/experimental_utils.py that:
- Uses standalone TCN (no tsai import chain)
- Loads all 200 train samples, stacks and averages to form a single consensus time series
- Runs UnCLe and evaluates against the ground-truth static adjacency matrix
- Saves all outputs to results_sim_fmri_large/

Input format understanding:
  UnCLe expects data as [T, N] (time x nodes) per run.
  Our dataset has 200 samples each [240, 68].
  Strategy: concatenate all train samples along time axis -> [200*240, 68] = [48000, 68].
  This is consistent with UnCLe's fMRI usage (single long time series).
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import date

# Add local_scripts to path for standalone TCN
sys.path.insert(0, str(Path(__file__).parent))
from tcn_standalone import TemporalConvNet

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score
)

# ==================== Model ====================

class PredictionDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def numpy2tensor(arr):
    return torch.tensor(arr).float()


class VARP(nn.Module):
    def __init__(self, c_in, lag=1, encoder_layers=None, decoder_layers=None, kernel_size=6, seed=0):
        super().__init__()
        if encoder_layers is None:
            encoder_layers = [12] * 8
        if decoder_layers is None:
            decoder_layers = [12] * 8
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.scale_factor = 1
        self.c_in = c_in
        self.lag = lag
        self.const_pad = nn.ConstantPad1d((lag - 1, 0), 0)
        self.encoder = TemporalConvNet(1, encoder_layers, kernel_size, dropout=0.2)
        self.decoder = TemporalConvNet(
            encoder_layers[-1] * 1,
            [c * 1 for c in decoder_layers],
            kernel_size,
            dropout=0.2
        )
        self.n_channels = encoder_layers[-1]
        self.encoder_downsample = nn.AvgPool1d(self.scale_factor)
        self.var_mat = nn.parameter.Parameter(
            torch.zeros(self.c_in, self.c_in, encoder_layers[-1], lag).normal_(0, 0.01)
        )
        self.var_relu = nn.ReLU()
        self.decoder_upsample = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        self.decoder_1d_conv = nn.Conv1d(decoder_layers[-1] * 1, 1, 1)

    def forward(self, x, var_fusion_enabled):
        B, N, T = x.shape
        C = self.n_channels
        x = x.reshape(B * N, 1, T)
        x = self.encoder(x)
        x = self.encoder_downsample(x)

        if var_fusion_enabled:
            x = self.const_pad(x)
            x = x.unfold(2, self.lag, 1)
            x = x.reshape(B, N, C, -1, self.lag)
            x = x.transpose(2, 3)
            x = torch.einsum('nmjkl,imkl->nijkm', x, self.var_mat)
            x = self.var_relu(x)
            x = x.sum(dim=-1)
            x = x.reshape(B * N, -1, C)
            x = x.transpose(1, 2)

        x = self.decoder_upsample(x)
        x = self.decoder(x)
        x = self.decoder_1d_conv(x)
        x = x.reshape(B, N, -1)
        return x

    def get_regularized_params(self, stage=1):
        if stage == 0:
            return [*self.decoder_1d_conv.parameters()]
        else:
            return [self.var_mat, *self.decoder_1d_conv.parameters()]


def compute_l1_loss(w):
    return torch.abs(w).mean()


# ==================== Training ====================

def run_uncle(data, learning_rate, reconstruction_epochs, joint_epochs,
              run_name, use_cuda, cuda_i, seed,
              n_encoder_layers, n_encoder_channels, n_kernel_size,
              eval_var_batch_size=8, print_freq=50, out_dir=None):
    """
    Run UnCLe on a single [T, N] time series.
    Returns (permutation_graph [N,N], parameter_graph [N,N]).
    """
    pred_len = 1
    lag = 1

    nvars = data.shape[1]
    i_end = data.shape[0] - pred_len
    i_start = 0
    df_np = data.T  # [N, T]

    seq_len = i_end - i_start
    assert seq_len + pred_len == df_np.shape[1]

    device = torch.device(f"cuda:{cuda_i}" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"[run_uncle] device={device}, data=[T={data.shape[0]}, N={nvars}], "
          f"seq_len={seq_len}, epochs=(r={reconstruction_epochs}, j={joint_epochs})")

    original_data_loader = DataLoader(
        PredictionDataset(
            numpy2tensor(df_np[np.newaxis, :, :seq_len]),
            numpy2tensor(df_np[np.newaxis, :, pred_len:seq_len + pred_len])
        ),
        batch_size=1, shuffle=False
    )

    torch.manual_seed(seed)
    varp_net = VARP(
        nvars, lag, seed=seed,
        encoder_layers=n_encoder_layers * [n_encoder_channels],
        decoder_layers=n_encoder_layers * [n_encoder_channels],
        kernel_size=n_kernel_size
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(varp_net.parameters(), lr=learning_rate)

    stop_epoch = reconstruction_epochs + joint_epochs
    loss_min = float("inf")

    for epoch in range(stop_epoch):
        # --- RECONSTRUCTION STAGE ---
        if epoch < reconstruction_epochs:
            train_loss_epoch = [0.0] * 3
            varp_net.train()
            for i, (X, Y) in enumerate(original_data_loader):
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                loss = 0
                reconstructed_output = varp_net(X, False)
                reconstruction_loss = criterion(reconstructed_output, X)
                loss += reconstruction_loss
                l1_loss = sum(compute_l1_loss(p) for p in varp_net.get_regularized_params(stage=0))
                loss += l1_loss
                loss.backward()
                optimizer.step()
                train_loss_epoch[1] += reconstruction_loss.item()
                train_loss_epoch[2] += l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss

            if epoch == 0 or epoch % print_freq == 0 or epoch == reconstruction_epochs - 1:
                print(f"[R-stage epoch {epoch+1:4d}] recon={train_loss_epoch[1]/(i+1):.5f} l1={train_loss_epoch[2]/(i+1):.5f}")

        # --- JOINT STAGE ---
        if epoch >= reconstruction_epochs:
            train_loss_epoch = [0.0] * 3
            varp_net.train()
            for i, (X, Y) in enumerate(original_data_loader):
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()
                loss = 0
                predicted_output = varp_net(X, True)
                prediction_loss = criterion(predicted_output, Y)
                loss += prediction_loss
                reconstructed_output = varp_net(X, False)
                reconstruction_loss = criterion(reconstructed_output, X)
                loss += reconstruction_loss
                l1_loss = sum(compute_l1_loss(p) for p in varp_net.get_regularized_params(stage=1))
                loss += l1_loss
                loss.backward()
                optimizer.step()
                train_loss_epoch[0] += prediction_loss.item()
                train_loss_epoch[1] += reconstruction_loss.item()
                train_loss_epoch[2] += l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss

            if epoch == 0 or epoch % print_freq == 0 or epoch == stop_epoch - 1:
                j_ep = epoch - reconstruction_epochs
                print(f"[J-stage epoch {j_ep+1:4d}] pred={train_loss_epoch[0]/(i+1):.5f} recon={train_loss_epoch[1]/(i+1):.5f} l1={train_loss_epoch[2]/(i+1):.5f}")

    # Save model checkpoint so evaluation can be retried without retraining
    if out_dir is not None:
        _ckpt_path = os.path.join(out_dir, "varp_checkpoint.pt")
        torch.save(varp_net.state_dict(), _ckpt_path)
        print(f"Model checkpoint saved to {_ckpt_path}")

    # ==================== Evaluation ====================
    # For large T, chunk the time dimension to avoid OOM.
    # We use a 2000-timestep window from the center of the sequence.
    EVAL_T_MAX = 2000
    varp_net.eval()
    full_error = torch.zeros(nvars, nvars)
    red_error = torch.zeros(nvars, nvars)

    for batch_ind, (X, Y_truth) in enumerate(original_data_loader):
        print(f"Evaluating batch {batch_ind}...")
        for j in range(len(X)):
            # Subsample time for evaluation if sequence is too long
            T_full = X[j].shape[-1]
            if T_full > EVAL_T_MAX:
                t_start = (T_full - EVAL_T_MAX) // 2
                X_eval = X[j][:, t_start:t_start + EVAL_T_MAX]
                Y_eval = Y_truth[j][:, t_start:t_start + EVAL_T_MAX]
                eval_seq_len = EVAL_T_MAX
            else:
                X_eval = X[j]
                Y_eval = Y_truth[j]
                eval_seq_len = T_full

            X_ = X_eval.unsqueeze(0).repeat(nvars, 1, 1)  # [N, N, T_eval]
            for ii in range(nvars):
                X_[ii, ii, :] = X_[ii, ii, torch.randperm(X_.shape[2])]

            with torch.inference_mode():
                Y = varp_net(X_eval.unsqueeze(0).to(device), True).cpu()[0]
                Y_ = torch.zeros(nvars, nvars, eval_seq_len)
                for eval_batch in range(math.ceil(nvars / eval_var_batch_size)):
                    s = eval_var_batch_size * eval_batch
                    e = eval_var_batch_size * (eval_batch + 1)
                    Y_[s:e] = varp_net(X_[s:e].to(device), True).cpu()

            full_error += (Y[:, :] - Y_eval).norm(dim=1, p=2)
            red_error += (Y_[:, :, :] - Y_eval.repeat(nvars, 1, 1)).norm(dim=2, p=2)

    error_difference = red_error - full_error
    error_difference[error_difference < 0] = 0
    causal_graph = error_difference

    par_causal_graph = varp_net.var_mat.detach().cpu().squeeze(3).norm(dim=2)

    return causal_graph.numpy(), par_causal_graph.numpy().T


# ==================== Data Loading ====================

def load_sim_fmri_large(base_path, split="train"):
    """
    Load all samples from split, return stacked data [N_samples, T, N] and GT adjacency.
    GT adjacency is static (same for all samples) — load from sample_00000.
    """
    split_dir = os.path.join(base_path, split)
    sample_dirs = sorted(os.listdir(split_dir))
    print(f"Loading {len(sample_dirs)} samples from {split}...")

    X_list = []
    for sd in sample_dirs:
        x = np.load(os.path.join(split_dir, sd, "X.npy"))  # [T, N]
        X_list.append(x)

    X_all = np.stack(X_list, axis=0)  # [S, T, N]
    A = np.load(os.path.join(split_dir, "sample_00000", "A.npy"))  # [N, N]
    print(f"X_all shape: {X_all.shape}, dtype: {X_all.dtype}")
    print(f"A shape: {A.shape}, dtype: {A.dtype}, sum: {A.sum()}, unique: {np.unique(A)}")
    return X_all, A


def concatenate_samples(X_all):
    """
    Concatenate all samples along time axis: [S, T, N] -> [S*T, N].
    This gives UnCLe one long time series, consistent with how it processes
    single-sequence datasets.
    """
    S, T, N = X_all.shape
    X_concat = X_all.reshape(S * T, N)
    print(f"Concatenated: {X_all.shape} -> {X_concat.shape}")
    return X_concat


# ==================== Metrics ====================

def compute_metrics(scores, A_gt, threshold_sweep=None):
    """
    Compute AUROC, AUPRC, and best-F1 with SHD.
    Excludes diagonal (self-loops).
    """
    N = A_gt.shape[0]
    mask = ~np.eye(N, dtype=bool)

    scores_flat = scores[mask].flatten()
    gt_flat = (A_gt[mask] > 0).astype(int).flatten()

    # Normalize scores to [0, 1]
    s_min, s_max = scores_flat.min(), scores_flat.max()
    if s_max > s_min:
        scores_norm = (scores_flat - s_min) / (s_max - s_min)
    else:
        scores_norm = scores_flat

    auroc = roc_auc_score(gt_flat, scores_norm)
    auprc = average_precision_score(gt_flat, scores_norm)

    # Threshold sweep for best F1
    if threshold_sweep is None:
        threshold_sweep = np.arange(0.01, 0.55, 0.05)

    best_f1 = 0.0
    best_thr = 0.0
    best_pred = None
    for thr in threshold_sweep:
        pred = (scores_norm >= thr).astype(int)
        f1 = f1_score(gt_flat, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_pred = pred

    if best_pred is None:
        best_pred = np.zeros_like(gt_flat)

    shd = int(np.sum(best_pred != gt_flat))

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(best_f1),
        "shd": shd,
        "threshold": float(best_thr),
    }


# ==================== Main ====================

if __name__ == "__main__":
    BASE_PATH = "/storage/home/ydk297/projects/meta_causal_discovery/meta_nonlatent_4.7/data/simulated_fMRI/sim_fmri_68_static_large"
    OUT_DIR = "/storage/home/ydk297/projects/meta_causal_discovery/uncle/results_sim_fmri_large"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Hyperparameters (matching official fmri script for 68-node case)
    # Official run_fmri: K=6, layers=8, size=12, recon=1000, joint=2000, lr=1e-5
    K = 6
    N_HIDDEN_LAYERS = 8
    HIDDEN_LAYER_SIZE = 12
    RECON_EPOCHS = 1000
    JOINT_EPOCHS = 2000
    LR = 1e-5
    SEED = 0
    USE_CUDA = True
    CUDA_I = 0
    EVAL_VAR_BATCH_SIZE = 4

    # Data strategy: use first sample only (240 timesteps x 68 nodes).
    # UnCLe is designed for a single time series. Using one representative
    # sample from the static-graph dataset is equivalent to the official fMRI
    # usage (each of the 50 fMRI CSVs is processed independently).
    # We average results over all 200 training samples to get a stable graph.
    USE_ALL_SAMPLES = True  # True = average over all 200 samples; False = first only

    print("=" * 60)
    print("UnCLe on sim_fmri_68_static_large")
    print("=" * 60)

    # Load data
    X_all, A_gt = load_sim_fmri_large(BASE_PATH, split="train")
    print(f"\nBEFORE conversion: X_all={X_all.shape} [S, T, N], A_gt={A_gt.shape}")

    if USE_ALL_SAMPLES:
        # Concatenate all 200 samples -> single long time series [48000, 68]
        X_concat = concatenate_samples(X_all)
    else:
        X_concat = X_all[0]  # [240, 68] - single sample
        print(f"Using single sample: {X_concat.shape}")
    print(f"AFTER conversion: X_concat={X_concat.shape} [T_total, N]")

    # Save converted data
    conv_dir = "/storage/home/ydk297/projects/meta_causal_discovery/uncle/data_converted"
    os.makedirs(conv_dir, exist_ok=True)
    np.save(os.path.join(conv_dir, "X_concat.npy"), X_concat)
    np.save(os.path.join(conv_dir, "A_gt.npy"), A_gt)
    print(f"Saved X_concat.npy and A_gt.npy to {conv_dir}")

    # Run UnCLe
    t0 = time.time()
    permutation_graph, parameter_graph = run_uncle(
        data=X_concat,
        learning_rate=LR,
        reconstruction_epochs=RECON_EPOCHS,
        joint_epochs=JOINT_EPOCHS,
        run_name="sim_fmri_large",
        use_cuda=USE_CUDA,
        cuda_i=CUDA_I,
        seed=SEED,
        n_encoder_layers=N_HIDDEN_LAYERS,
        n_encoder_channels=HIDDEN_LAYER_SIZE,
        n_kernel_size=K,
        eval_var_batch_size=EVAL_VAR_BATCH_SIZE,
        out_dir=OUT_DIR,
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")
    print(f"permutation_graph shape: {permutation_graph.shape}")
    print(f"parameter_graph shape: {parameter_graph.shape}")

    # Save raw graphs
    np.save(os.path.join(OUT_DIR, "permutation_graph.npy"), permutation_graph)
    np.save(os.path.join(OUT_DIR, "parameter_graph.npy"), parameter_graph)
    print(f"Saved graphs to {OUT_DIR}")

    # Evaluate both graphs
    print("\n--- Evaluating permutation graph ---")
    perm_metrics = compute_metrics(permutation_graph, A_gt)
    print(f"Permutation: AUROC={perm_metrics['auroc']:.4f}, AUPRC={perm_metrics['auprc']:.4f}, "
          f"F1={perm_metrics['f1']:.4f}, SHD={perm_metrics['shd']}, thr={perm_metrics['threshold']:.2f}")

    print("\n--- Evaluating parameter graph ---")
    param_metrics = compute_metrics(parameter_graph, A_gt)
    print(f"Parameter: AUROC={param_metrics['auroc']:.4f}, AUPRC={param_metrics['auprc']:.4f}, "
          f"F1={param_metrics['f1']:.4f}, SHD={param_metrics['shd']}, thr={param_metrics['threshold']:.2f}")

    # Save metrics
    import json
    all_metrics = {
        "permutation_graph": perm_metrics,
        "parameter_graph": param_metrics,
        "hyperparameters": {
            "K": K,
            "n_hidden_layers": N_HIDDEN_LAYERS,
            "hidden_layer_size": HIDDEN_LAYER_SIZE,
            "recon_epochs": RECON_EPOCHS,
            "joint_epochs": JOINT_EPOCHS,
            "lr": LR,
            "seed": SEED,
            "data_shape": list(X_concat.shape),
            "n_samples_train": X_all.shape[0],
            "elapsed_seconds": elapsed,
        }
    }
    with open(os.path.join(OUT_DIR, "uncle_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved uncle_metrics.json to {OUT_DIR}")
    print("\n=== DONE ===")
