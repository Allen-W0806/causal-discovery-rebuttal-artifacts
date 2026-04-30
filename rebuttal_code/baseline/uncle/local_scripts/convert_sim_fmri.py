"""
Adapter: convert sim_fmri_68_static_large dataset to UnCLe format.

UnCLe expects a single time series [T, N] (time x nodes).
Our dataset: 200 train samples, each [240, 68].
Strategy: concatenate all train samples along time axis -> [200*240, 68] = [48000, 68].

GT adjacency: static (same across all samples) — load from sample_00000.
GT is binary (int8, values 0/1), shape [68, 68].
"""

import os
import numpy as np

BASE_PATH = (
    "/storage/home/ydk297/projects/meta_causal_discovery"
    "/meta_nonlatent_4.7/data/simulated_fMRI/sim_fmri_68_static_large"
)
OUT_DIR = (
    "/storage/home/ydk297/projects/meta_causal_discovery"
    "/uncle/data_converted"
)
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load all train samples ---
train_dir = os.path.join(BASE_PATH, "train")
sample_dirs = sorted(os.listdir(train_dir))
print(f"Found {len(sample_dirs)} train samples")

X_list = []
for sd in sample_dirs:
    x = np.load(os.path.join(train_dir, sd, "X.npy"))  # [240, 68]
    X_list.append(x)

X_all = np.stack(X_list, axis=0)  # [200, 240, 68]
print(f"BEFORE conversion: X_all shape = {X_all.shape}  (S x T x N)")
print(f"  dtype={X_all.dtype}, mean={X_all.mean():.4f}, std={X_all.std():.4f}")

# --- Load GT adjacency (static, from sample_00000) ---
A_gt = np.load(os.path.join(train_dir, "sample_00000", "A.npy"))  # [68, 68]
print(f"A_gt shape = {A_gt.shape}, dtype={A_gt.dtype}, sum={A_gt.sum()}, unique={np.unique(A_gt)}")

# --- Concatenate along time ---
S, T, N = X_all.shape
X_concat = X_all.reshape(S * T, N)  # [48000, 68]
print(f"AFTER conversion: X_concat shape = {X_concat.shape}  (T_total x N)")

# --- Save ---
np.save(os.path.join(OUT_DIR, "X_concat.npy"), X_concat)
np.save(os.path.join(OUT_DIR, "A_gt.npy"), A_gt)
print(f"\nSaved to {OUT_DIR}:")
print(f"  X_concat.npy  {X_concat.shape}  float32")
print(f"  A_gt.npy      {A_gt.shape}   int8")
