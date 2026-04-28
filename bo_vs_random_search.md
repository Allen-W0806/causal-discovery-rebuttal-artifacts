# Fixed-Budget BO vs. Random Search on NC8

Table: Fixed-budget comparison between Bayesian optimization and random search on NC8.
| Eval Budget | Method | AUROC ↑ | AUPRC ↑ | F1 ↑ | SHD ↓ |
|---:|---|---:|---:|---:|---:|
| 200 | Ours (BO) | 0.767 ± 0.058 | 0.487 ± 0.122 | 0.525 ± 0.073 | 9.60 ± 3.85 |
| 200 | Random Search | 0.616 ± 0.078 | 0.274 ± 0.092 | 0.303 ± 0.056 | 16.40 ± 3.14 |
| 500 | Ours (BO) | 0.862 ± 0.032 | 0.502 ± 0.111 | 0.585 ± 0.084 | 9.00 ± 3.28 |
| 500 | Random Search | 0.683 ± 0.084 | 0.266 ± 0.081 | 0.364 ± 0.080 | 14.20 ± 3.58 |
