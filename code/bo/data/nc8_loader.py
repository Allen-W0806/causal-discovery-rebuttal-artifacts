import os
import numpy as np
import pandas as pd


def load_nc8_data(data_dir, replica=0):
    """Load NC8 dataset."""
    data_path = os.path.join(data_dir, f"nc8_data_{replica}.csv")
    struct_path = os.path.join(data_dir, f"nc8_struct_{replica}.csv")

    df_data = pd.read_csv(data_path)
    df_struct = pd.read_csv(struct_path)

    X = df_data.values.astype(np.float64)
    GT = df_struct.values.astype(np.float64)
    var_names = list(df_data.columns)

    print(f"NC8 replica {replica}: X shape={X.shape}, GT shape={GT.shape}")

    GT_bin = (GT != 0).astype(int)
    np.fill_diagonal(GT_bin, 0)
    n_edges = GT_bin.sum()
    print(f"Ground truth edges (off-diagonal): {n_edges}")
    for i in range(GT.shape[0]):
        for j in range(GT.shape[1]):
            if GT_bin[i, j] == 1:
                print(f"  {var_names[i]} -> {var_names[j]}")

    return X, GT, var_names
