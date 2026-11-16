# ODDS benchmark

The manuscript reports a 23-dataset ODDS comparison. The source dataset files are excluded from this repository because they are third-party benchmark resources with their own distribution terms.

The routing profile in `project/src/odds_adaptive_anomaly.py` was calibrated on the paper's 23-dataset ODDS suite. The resulting summary is therefore benchmark-calibration evidence, not an independent external-generalization estimate.

The generic scoring implementation is `project/src/odds_adaptive_anomaly.py`. To evaluate a locally obtained ODDS snapshot:

```bash
python experiments/scripts/run_odds_benchmark.py \
  --data-dir /path/to/odds \
  --output runtime_outputs/odds_auc.csv
```

Supported containers:

- CSV with a numeric feature table and a label column named `label`, `y`, `target`, `outlier`, or `class`
- NPZ with a two-dimensional `X` array and a one-dimensional label array
- MAT with a two-dimensional feature array under `X`, `x`, or `data` and a label array
- NPY with the label in the final column

Labels must be `0` for normal records and `1` for anomaly records. The script uses only features to compute scores and uses labels only to calculate AUC-ROC.

Benchmark sources:

- ODDS Library: <http://odds.cs.stonybrook.edu/>
- AnoLLM benchmark code and dataset instructions: <https://github.com/amazon-science/AnoLLM-large-language-models-for-tabular-anomaly-detection>

Record the benchmark version and upstream source in any reuse or comparison.
