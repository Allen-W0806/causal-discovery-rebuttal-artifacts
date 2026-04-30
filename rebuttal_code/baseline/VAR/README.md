# VAR Baseline

Vector Autoregression (VAR) with Granger-style causality testing, implemented via
[`statsmodels`](https://www.statsmodels.org/stable/vector_ar.html).

## Method

A VAR(p) model is fit to the observed multivariate time series.  For each ordered
variable pair (j → i) a Granger F-test is run using
`VARResults.test_causality(caused=i, causing=j, kind='f')`.

| Output | Definition |
|--------|-----------|
| `score[i,j]` | F-statistic of the test "does j Granger-cause i?" |
| `pvalue[i,j]` | Corresponding p-value |
| `binary[i,j]` | `1` if `pvalue[i,j] <= alpha_level`, else `0` |

The lag order `p` is either fixed by `--lag` or selected automatically by AIC over a
search range.

**Diagonal handling:** VAR Granger tests cannot reliably detect self-lag; diagonal
entries are set to 0.  The NC8 ground-truth graph contains self-lag positives on the
diagonal, so VAR will under-report recall compared with methods that explicitly model
autoregressive self-connections.

## Dependencies

- `statsmodels >= 0.14`
- `numpy`, `scipy` (standard scientific stack)

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_nc8_baseline.py` | Full self-contained VAR baseline on NC8 replicas |

**Usage:**

```bash
# Use AIC-selected lag, default alpha = 0.05
python scripts/run_nc8_baseline.py \
    --data-dir ../data/NC8 \
    --output-dir results/nc8_baseline

# Fix lag to 3, tighten threshold
python scripts/run_nc8_baseline.py \
    --data-dir ../data/NC8 \
    --output-dir results/nc8_lag3_a01 \
    --lag 3 \
    --alpha-level 0.01
```

Run `python scripts/run_nc8_baseline.py --help` for the full argument list.

## Results layout

```
results/
  nc8_baseline/               ← default output from run_nc8_baseline.py
    config.json
    per_replica_metrics.csv
    aggregate_metrics.csv
    ground_truth_graph_replica0.{csv,npy}
    evaluation_protocol.md
    summary.md
    logs/
      baseline.log
    predicted_graphs/
      replica_<k>_score_matrix.{csv,npy}
      replica_<k>_pvalue_matrix.{csv,npy}
      replica_<k>_binary_graph.{csv,npy}
```

The `results/` tree is also pre-structured for the broader benchmark layout used by
other experiments in this repository:

```
results/
  non_latent/
    nc8/
    simulated_EEG_68_static_v2/
    simulated_fMRI/
  latent/
    nc8/
    simulated_EEG_68_static_v2/
    simulated_fMRI/
```
