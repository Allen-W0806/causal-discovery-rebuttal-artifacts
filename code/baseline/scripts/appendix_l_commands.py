#!/usr/bin/env python
"""Print Appendix L launch commands for all local baselines.

This script does not run experiments unless --execute is supplied. It is intended
as a consistent launch surface while method-specific wrappers are incrementally
connected to the common NC8/ND8/FINANCE data layout.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from appendix_l_config import BASELINE_ROOT, DATASET_NAMES, METHODS, get_dataset_spec, get_params


def q(value: object) -> str:
    return shlex.quote(str(value))


def command_for(method: str, dataset: str) -> list[str]:
    params = get_params(method, dataset)
    spec = get_dataset_spec(dataset)
    out = BASELINE_ROOT / "results" / method.replace("+", "plus") / dataset
    data_dir = Path(spec["data_dir"])

    if method == "VARLiNGAM":
        return ["python", str(BASELINE_ROOT / "scripts" / "run_varlingam_appendix_l.py"), "--dataset", dataset, "--output-root", str(BASELINE_ROOT / "results" / "VARLiNGAM")]
    if method == "JRNGC":
        return ["python", str(BASELINE_ROOT / "scripts" / "run_jrngc_appendix_l.py"), "--dataset", dataset, "--output-root", str(BASELINE_ROOT / "results" / "JRNGC")]
    if method == "VAR":
        return ["python", str(BASELINE_ROOT / "VAR" / "scripts" / "run_nc8_baseline.py"), "--data-dir", str(data_dir), "--output-dir", str(out), "--lag", str(params["lag"])]
    if method == "PCMCI":
        return ["python", str(BASELINE_ROOT / "PCMCI" / "experiments" / "run_nc8_pcmci_baseline.py"), "--data-dir", str(data_dir), "--output-dir", str(out), "--tau-max-values", str(params["tau_max"]), "--pc-alpha", str(params["pc_alpha"])]
    if method == "cMLP":
        commands = []
        for lam in params["lambda_grid"]:
            commands.append(
                ["python", str(BASELINE_ROOT / "cMLP" / "experiments" / "run_nc8_cmlp_baseline.py"), "--data-dir", str(data_dir), "--output-dir", str(out / f"lambda_{lam:g}"), "--lag", str(params["lag"]), "--hidden", *["20"] * int(params["hidden_layers"]), "--penalty", params["penalty"], "--lam", str(lam), "--lr", str(params["learning_rate"]), "--max-iter", str(params["epochs"])]
            )
        return [" && ".join(" ".join(q(part) for part in cmd) for cmd in commands)]
    if method == "TCDF":
        commands = []
        for alpha in params["alpha_grid"]:
            commands.append(
                ["python", str(BASELINE_ROOT / "TCDF" / "experiments" / "run_nc8_tcdf_baseline.py"), "--data-dir", str(data_dir), "--output-dir", str(out / f"alpha_{alpha:g}"), "--kernel-size", str(params["kernel_size"]), "--hidden-layers", str(params["hidden_layers"]), "--epochs", str(params["epochs"]), "--learning-rate", str(params["learning_rate"]), "--significance", str(alpha)]
            )
        return [" && ".join(" ".join(q(part) for part in cmd) for cmd in commands)]
    if method == "GVAR":
        commands = []
        for lam in params["lambda_grid"]:
            for gamma in params["gamma_grid"]:
                commands.append(
                    ["python", str(BASELINE_ROOT / "GVAR" / "experiments" / "run_nc8_gvar_baseline.py"), "--data-dir", str(data_dir), "--output-dir", str(out / f"lambda_{lam:g}_gamma_{gamma:g}"), "--order", str(params["lag"]), "--num-hidden-layers", str(params["hidden_layers"]), "--num-epochs", str(params["epochs"]), "--initial-lr", str(params["learning_rate"]), "--lambda-value", str(lam), "--gamma-value", str(gamma)]
                )
        return [" && ".join(" ".join(q(part) for part in cmd) for cmd in commands)]
    if method == "UnCLe":
        experiment = {"NC8": "unicsl_nc8", "ND8": "unicsl_nd8", "FINANCE": "unicsl_finance"}[dataset]
        return ["python", str(BASELINE_ROOT / "uncle" / "bin" / "run_grid_search.py"), "--experiment", experiment, "--K", str(params["kernel_size"]), "--num-hidden-layers", str(params["tcn_blocks"]), "--hidden-layer-size", str(params["kernel_filters"]), "--num-epochs-1", str(params["reconstruction_epochs"]), "--num-epochs-2", str(params["joint_epochs"]), "--initial-lr", str(params["learning_rate"])]
    if method == "DYNOTEARS":
        return ["python", str(BASELINE_ROOT / "scripts" / "run_dynotears_appendix_l.py"), "--dataset", dataset, "--max-iter", str(params["max_iter"]), "--lambda-w", str(params["lambda_w"]), "--lambda-a", str(params["lambda_a"]), "--lags", *[str(v) for v in params["lag_grid"]]]
    if method == "CUTS+":
        return ["python", str(BASELINE_ROOT / "scripts" / "run_cutsplus_appendix_l.py"), "--dataset", dataset, "--lr", str(params["learning_rate"]), "--epochs", str(params["epochs"]), "--max-groups", str(params["max_groups"]), "--lambda-grid", *[str(v) for v in params["lambda_grid"]]]
    raise ValueError(f"Unsupported method: {method}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--dataset", choices=DATASET_NAMES, action="append")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run the selected command(s). Use with care.")
    args = parser.parse_args()

    methods = args.method or list(METHODS)
    datasets = args.dataset or list(DATASET_NAMES)
    payload = []
    for method in methods:
        for dataset in datasets:
            cmd = command_for(method, dataset)
            payload.append({"method": method, "dataset": dataset, "params": get_params(method, dataset), "command": cmd})

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in payload:
            cmd = item["command"]
            if len(cmd) == 1 and " && " in cmd[0]:
                command_text = cmd[0]
            else:
                command_text = " ".join(q(part) for part in cmd)
            print(f"# {item['method']} / {item['dataset']}")
            print(command_text)
            print()

    if args.execute:
        for item in payload:
            cmd = item["command"]
            if len(cmd) == 1 and " && " in cmd[0]:
                subprocess.run(cmd[0], shell=True, check=True, cwd=BASELINE_ROOT)
            else:
                subprocess.run(cmd, check=True, cwd=BASELINE_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

