# GVAR Provisional Results — Completed Runs Only

> **⚠️ PROVISIONAL SUMMARY**
> These results are computed from **completed grid-point runs only**.
> No hyperparameter best-selection was applied.
> Failed, incomplete, and still-running grid points are excluded.
> Remaining jobs (NC8: 13/25, ND8: 13/25, FINANCE: 25/25) are still running or pending.
> This summary is intended for deciding whether to continue the GVAR grid.

---

## Method

- CI formula: **mean ± 1.96 × std / sqrt(n)**  where n = number of completed settings
- Each "observation" is the per-setting mean across 5 replicas (NC8/ND8) or 8 replicas (FINANCE)
- Aggregation is across hyperparameter settings, **not** across replicas
- Grid: λ ∈ {0, 0.75, 1.5, 2.25, 3} × γ ∈ {0, 0.00625, 0.0125, 0.01875, 0.025} = 25 combinations

---

## NC8  (12 / 25 settings completed, 5 replicas each)

Completed settings: λ ∈ {0, 0.75, 1.5} × all γ, plus λ=1.5 γ=0 and λ=1.5 γ=0.00625

| Metric       | Mean    | ± 95% CI | Std     |
|--------------|--------:|----------:|--------:|
| AUROC        | 0.6521  | ± 0.0407  | 0.0720  |
| AUPRC        | 0.3033  | ± 0.0368  | 0.0651  |
| F1           | 0.2933  | ± 0.0572  | 0.1011  |
| SHD          | 18.40   | ± 5.21    | 9.20    |
| runtime (s)  | 1287.5  | ± 4.29    | 7.58    |

---

## ND8  (12 / 25 settings completed, 5 replicas each)

| Metric       | Mean    | ± 95% CI | Std     |
|--------------|--------:|----------:|--------:|
| AUROC        | 0.7392  | ± 0.0290  | 0.0512  |
| AUPRC        | 0.4364  | ± 0.0420  | 0.0742  |
| F1           | 0.4400  | ± 0.0269  | 0.0476  |
| SHD          | 21.35   | ± 5.54    | 9.79    |
| runtime (s)  | 1302.0  | ± 6.08    | 10.74   |

---

## FINANCE  (0 / 25 settings completed)

No completed runs yet. Job 7829 is pending and will start after NC8/ND8 finish.

---

## Assessment

### Are the current results usable as provisional paper numbers?

**Possibly, with caveats.**

- NC8: 12/25 settings completed. The completed λ values are 0, 0.75, 1.5 — covering
  the weaker-regularisation half of the grid. The two strongest-regularisation values
  (λ=2.25, λ=3) are not yet done. The std across completed settings is moderately high
  (AUROC std=0.072), suggesting the remaining settings could shift the mean noticeably.
  Current AUROC 0.652 ± 0.041 should be treated as provisional.

- ND8: same coverage (12/25). AUROC 0.739 ± 0.029 with lower std — more stable across
  the completed settings.

- FINANCE: no data at all; cannot be used.

### Is continuing the remaining GVAR jobs necessary?

**Recommendation: yes, let the jobs finish.**

1. The grid is already ~48% done on NC8/ND8. Stopping now would leave λ=2.25 and λ=3
   (the strongest regularisation) completely unsampled.
2. AUROC std across settings is 0.072 for NC8 — wide enough that the missing half
   of the grid could meaningfully change the mean, especially if heavier regularisation
   improves performance.
3. FINANCE is 0% complete; the result would be entirely missing for comparison tables.
4. Runtime per grid point is ~1287 s (~21 min). The remaining 13 NC8 + 13 ND8 points
   will take roughly 26 hours more; FINANCE another ~9 hours.

If a hard deadline makes waiting impossible, the provisional NC8/ND8 numbers above can
be cited as "partial grid (12/25 settings)" with the caveat that high-λ settings are
not included.

---

*Generated: 2026-04-26*
*Detail file: `gvar_completed_runs_metrics.csv` (120 rows, per replica × per setting)*
