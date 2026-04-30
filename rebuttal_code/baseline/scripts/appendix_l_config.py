"""Appendix L baseline configuration for the local UnCLe benchmark tree."""

from __future__ import annotations

from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BASELINE_ROOT / "data"

DATASETS = {
    "NC8": {
        "data_dir": DATA_ROOT / "NC8",
        "data_glob": "nc8_data_*.csv",
        "struct_glob": "nc8_struct_*.csv",
        "source": "Appendix L NC8 setting",
    },
    "ND8": {
        "data_dir": DATA_ROOT / "ND8",
        "data_glob": "nc8_dynamic_*.csv",
        "struct_glob": "nc8_structure_dynamic.npy",
        "source": "Appendix L does not provide separate rows for several methods; NC8 defaults are used where specified.",
    },
    "FINANCE": {
        "data_dir": DATA_ROOT / "Finance",
        "data_glob": "finance_data_*.csv",
        "struct_glob": "finance_struct_*.csv",
        "source": "Appendix L FINANCE setting",
    },
}


HYPERPARAMETERS = {
    "UnCLe": {
        "NC8": {
            "lag": 1,
            "kernel_size": 8,
            "tcn_blocks": 6,
            "kernel_filters": 20,
            "reconstruction_epochs": 1000,
            "joint_epochs": 2000,
            "learning_rate": 3e-4,
            "source": "Appendix L",
        },
        "ND8": {
            "lag": 1,
            "kernel_size": 8,
            "tcn_blocks": 6,
            "kernel_filters": 20,
            "reconstruction_epochs": 1000,
            "joint_epochs": 2000,
            "learning_rate": 3e-4,
            "source": "ND8 follows the same Appendix-L setting as NC8 for the 8-variable synthetic benchmark setting.",
        },
        "FINANCE": {
            "lag": 2,
            "kernel_size": 2,
            "tcn_blocks": 3,
            "kernel_filters": 24,
            "reconstruction_epochs": 500,
            "joint_epochs": 10000,
            "learning_rate": 3e-4,
            "source": "Appendix L",
        },
    },
    "VAR": {
        "NC8": {"lag": 16, "source": "Appendix L"},
        "ND8": {"lag": 16, "source": "ND8 follows the same Appendix-L setting as NC8 for the 8-variable synthetic benchmark setting."},
        "FINANCE": {"lag": 5, "source": "Appendix L"},
    },
    "PCMCI": {
        "NC8": {"tau_min": 1, "tau_max": 16, "pc_alpha": 0.01, "independence_test": "ParCorr", "source": "Appendix L"},
        "ND8": {"tau_min": 1, "tau_max": 16, "pc_alpha": 0.01, "independence_test": "ParCorr", "source": "Appendix L instruction: use L=16"},
        "FINANCE": {"tau_min": 1, "tau_max": 5, "pc_alpha": 0.01, "independence_test": "ParCorr", "source": "Appendix L"},
    },
    "cMLP": {
        "NC8": {"lag": 16, "hidden_layers": 1, "epochs": 1000, "learning_rate": 5e-3, "penalty": "H", "lambda_grid": [0.0, 0.5, 1.0, 1.5, 2.0], "source": "Appendix L"},
        "ND8": {"lag": 16, "hidden_layers": 1, "epochs": 1000, "learning_rate": 5e-3, "penalty": "H", "lambda_grid": [0.0, 0.5, 1.0, 1.5, 2.0], "source": "ND8 follows the same Appendix-L setting as NC8 for the 8-variable synthetic benchmark setting."},
        "FINANCE": {"lag": 3, "hidden_layers": 1, "epochs": 1000, "learning_rate": 1e-3, "penalty": "H", "lambda_grid": [0.0, 0.5, 1.0, 1.5, 2.0], "source": "Appendix L"},
    },
    "TCDF": {
        "NC8": {"kernel_size": 16, "hidden_layers": 1, "epochs": 1000, "learning_rate": 5e-3, "alpha_grid": [0.0, 0.5, 1.0, 1.5, 2.0], "source": "Appendix L"},
        "ND8": {"kernel_size": 16, "hidden_layers": 1, "epochs": 1000, "learning_rate": 5e-3, "alpha_grid": [0.0, 0.5, 1.0, 1.5, 2.0], "source": "ND8 follows the same Appendix-L setting as NC8 for the 8-variable synthetic benchmark setting."},
        "FINANCE": {"kernel_size": 5, "hidden_layers": 1, "epochs": 2000, "learning_rate": 1e-2, "alpha_grid": [0.0, 0.5, 1.0, 1.5, 2.0], "source": "Appendix L"},
    },
    "GVAR": {
        "NC8": {"lag": 16, "hidden_layers": 1, "epochs": 1000, "learning_rate": 1e-4, "lambda_grid": [0.0, 0.75, 1.5, 2.25, 3.0], "gamma_grid": [0.0, 0.00625, 0.0125, 0.01875, 0.025], "source": "Appendix L"},
        "ND8": {"lag": 16, "hidden_layers": 1, "epochs": 1000, "learning_rate": 1e-4, "lambda_grid": [0.0, 0.75, 1.5, 2.25, 3.0], "gamma_grid": [0.0, 0.00625, 0.0125, 0.01875, 0.025], "source": "ND8 follows the same Appendix-L setting as NC8 for the 8-variable synthetic benchmark setting."},
        "FINANCE": {"lag": 3, "hidden_layers": 2, "epochs": 500, "learning_rate": 1e-4, "lambda_grid": [0.0, 0.75, 1.5, 2.25, 3.0], "gamma_grid": [0.0, 0.00625, 0.0125, 0.01875, 0.025], "source": "Appendix L"},
    },
    "VARLiNGAM": {
        dataset: {"lag_grid": [2, 3, 4, 5], "source": "Appendix L"} for dataset in DATASETS
    },
    "DYNOTEARS": {
        dataset: {"max_iter": 1000, "lambda_w": 0.1, "lambda_a": 0.1, "lag_grid": [2, 3, 4, 5], "source": "Appendix L"} for dataset in DATASETS
    },
    "CUTS+": {
        dataset: {"learning_rate": 1e-3, "epochs": 64, "max_groups": 32, "lambda_grid": [0.1, 0.05, 0.01, 0.005], "source": "Appendix L with explicit suggested grid"} for dataset in DATASETS
    },
    "JRNGC": {
        dataset: {"hidden": 100, "lag": 5, "layers": 5, "learning_rate": 1e-3, "jacobian_lambda_grid": [1e-3, 5e-4, 2e-4, 1e-4], "source": "Appendix L with explicit suggested grid"} for dataset in DATASETS
    },
}


METHODS = tuple(HYPERPARAMETERS.keys())
DATASET_NAMES = tuple(DATASETS.keys())


def get_params(method: str, dataset: str) -> dict:
    try:
        return dict(HYPERPARAMETERS[method][dataset])
    except KeyError as exc:
        raise ValueError(f"Unsupported method/dataset combination: {method}/{dataset}") from exc


def get_dataset_spec(dataset: str) -> dict:
    try:
        spec = dict(DATASETS[dataset])
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc
    return spec
