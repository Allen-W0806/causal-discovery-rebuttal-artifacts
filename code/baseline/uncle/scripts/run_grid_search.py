# This script evaluates UnCLe
import argparse
import os
import numpy as np
import time
from datetime import date
from experimental_utils import run_grid_search

parser = argparse.ArgumentParser(description='UnCLe Runner')


# Simulation model parameters
parser.add_argument('--experiment', type=str, default="lorenz96_0", help="Experiment to be performed (default: "
                                                                       "'lorenz96_0')")

# Model specification
parser.add_argument('--K', type=int, default=5, help='Kernel size (default: 5)')
parser.add_argument('--num-hidden-layers', type=int, default=1, help='Number of hidden layers (default: 1)')
parser.add_argument('--hidden-layer-size', type=int, default=50, help='Number of units in the hidden layer '
                                                                      '(default: 50)')

# Training procedure
parser.add_argument('--num-epochs-1', type=int, default=10, help='Number of epochs to train phase1 (default: 10)')
parser.add_argument('--num-epochs-2', type=int, default=10, help='Number of epochs to train phase2 (default: 10)')
parser.add_argument('--initial-lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')


# Meta
parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
parser.add_argument('--num-sim', type=int, default=1, help='Number of simulations (default: 1)')
parser.add_argument('--use-cuda', type=bool, default=True, help='Use GPU? (default: true)')
parser.add_argument('--cuda-i', type=int, default=0, help='Cuda device number to use (default: 0)')



# Parsing args
args = parser.parse_args()

datasets = []
structures = []
signed_structures = None

print(str(args.num_sim) + " " + str(args.experiment) + " datasets...")

if args.experiment == "unicsl_lorenz96_0":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var20_force10_t250_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var20_force10_t250_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_lorenz96_1":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var20_force40_t250_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var20_force40_t250_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_lorenz96_2":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var100_force40_t500_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/Lorenz96/Lorenz96_var100_force40_t500_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_finance":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/Finance/finance_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/Finance/finance_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_nd8":
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    # ND8 has a single shared time-varying structure; collapse to static binary graph
    nd8_struct_raw = np.load("../datasets/ND8/nc8_structure_dynamic.npy", allow_pickle=True)
    if nd8_struct_raw.ndim == 3:
        nd8_struct = (np.max(np.abs(nd8_struct_raw), axis=0) > 0).astype(float)
    else:
        nd8_struct = nd8_struct_raw.astype(float)
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/ND8/nc8_dynamic_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        datasets.append(data_i.to_numpy())
        structures.append(nd8_struct)
elif args.experiment == "unicsl_nc8":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/NC8/nc8_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/NC8/nc8_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_nc8_mask2_latent_obs":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(
            f"../datasets/NC8_mask2_latent_obs/nc8_mask2_latent_obs_data_{i}.csv",
            index_col=None,
        )
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(
            f"../datasets/NC8_mask2_latent_obs/nc8_mask2_latent_obs_struct_{i}.csv",
            index_col=None,
        )
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_fmri":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(f"../datasets/fMRI/fMRI_data_{i}.csv", index_col=None)
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(f"../datasets/fMRI/fMRI_struct_{i}.csv", index_col=None)
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_fmri_68full_4latent":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(
            f"../datasets/sim_fmri_68full_4latent/sim_fmri_68full_4latent_data_{i}.csv",
            index_col=None,
        )
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(
            f"../datasets/sim_fmri_68full_4latent/sim_fmri_68full_4latent_struct_{i}.csv",
            index_col=None,
        )
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
elif args.experiment == "unicsl_fmri_64obs_4latent":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    for i in range(args.num_sim):
        data_i = pd.read_csv(
            f"../datasets/sim_fmri_64obs_4latent/sim_fmri_64obs_4latent_data_{i}.csv",
            index_col=None,
        )
        data_i[:] = StandardScaler().fit_transform(data_i[:])
        a_i = pd.read_csv(
            f"../datasets/sim_fmri_64obs_4latent/sim_fmri_64obs_4latent_struct_{i}.csv",
            index_col=None,
        )
        datasets.append(data_i.to_numpy())
        structures.append(a_i.to_numpy())
else:
    NotImplementedError("ERROR: This experiment is not supported!")

run_grid_search(datasets=datasets, K=args.K, structures=structures,
                num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size,
                num_epochs_1=args.num_epochs_1, num_epochs_2=args.num_epochs_2, initial_lr=args.initial_lr,
                seed=args.seed, use_cuda=args.use_cuda,
                cuda_i=args.cuda_i, experiment_name=args.experiment)
