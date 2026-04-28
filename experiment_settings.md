# Experiment Settings

This file summarizes the experimental settings used for the rebuttal-stage artifacts.

## Datasets

### NC8

NC8 is an 8-variable nonlinear time-series benchmark. Results are reported over independent replicas using the same dataset and ground-truth graph protocol across methods.

### ND8

ND8 is an 8-variable nonlinear time-series benchmark with a time-varying graph setting. Results are reported using the same evaluation protocol across methods.

### Finance

Finance is a synthetic time-series benchmark. Results are reported under the same train/evaluation protocol across compared methods.

## Metrics

- **AUROC**: threshold-free ranking metric for edge recovery.
- **AUPRC**: threshold-free ranking metric emphasizing sparse edge recovery.
- **F1**: thresholded graph recovery metric.
- **SHD**: structural Hamming distance after thresholding.
- **Runtime**: wall-clock runtime in seconds when reported.

## Reporting

For replicated experiments, values are reported as:

`mean ± 95% confidence interval`

The same dataset replicas and ground-truth graphs are used across compared methods whenever applicable.
