# Anonymous Rebuttal Artifacts

This repository contains anonymized artifacts supporting the additional experimental results reported in the rebuttal.

## Contents

| File | Description |
|---|---|
| `unified_baseline_results.md` | Unified comparison with additional baselines on NC8, ND8, and Finance. |
| `bo_vs_random_search.md` | Fixed-budget comparison between Bayesian optimization and random search on NC8. |
| `nodewise_greedy_vs_ours.md` | Comparison between node-wise greedy search and the proposed BO-based method. |
| `low_rank_ablation.md` | Ablation study removing the low-rank graph parameterization. |
| `rank_diagnostic_full.md` | Diagnostic showing the chosen low-rank dimension is sufficient for benchmark graphs. |

## Reporting Protocol

Unless otherwise specified, results are reported as mean ± 95% confidence interval over independent replicas. AUROC and AUPRC are threshold-free metrics. F1 and SHD are reported after thresholding when applicable. Runtime is reported in seconds when included.

## Anonymization

All artifacts are anonymized for double-blind review. User names, institution names, local machine paths, and private identifiers have been removed or replaced with generic placeholders.

## Code Availability

This repository is intended as a rebuttal-stage artifact package. The full implementation will be cleaned and released upon acceptance, following the double-blind review policy.
