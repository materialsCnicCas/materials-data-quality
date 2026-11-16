# Reproducibility notes

The release targets the experiment protocol described in the 23-page KBS manuscript dated 21 October 2026.

## Local reference environment

- Python 3.9.7
- NumPy 1.26.4
- pandas 2.3.3
- SciPy 1.13.1
- scikit-learn 1.6.1
- PyTorch 2.7.1 CPU
- torch-geometric 2.6.1
- four CPU threads on an AMD Ryzen 5 9500F

The exact environment used for the recorded local timing and GAT runs is saved in `experiments/results/environment.json`.

## Data splits and evaluation

The repeated `mp_20` experiments use five independently sampled 16,000-record subsets. Records are deduplicated by material ID and grouped by formula plus the nine unrounded lattice-matrix entries. Grouped splits prevent the same structure/composition group from appearing in both fitting and test data.

The primary evaluation uses 10,000 fitting records and a separate 2,000-record calibration batch inside the training partition. The 4,000-record clean outer-test records are used only for downstream evaluation. The calibration batch selects the threshold; it does not train the confidence score. Injected corruption labels are used only for metrics.

The main result files are:

- `experiments/results/validation_summary.csv`
- `experiments/results/validation_all_runs.csv`
- `experiments/results/validation_paired_r2.csv`
- `experiments/results/robustness_summary.csv`
- `experiments/results/sensitivity_summary.csv`
- `experiments/results/memory_summary.csv`
- `experiments/results/runtime_all_runs.csv`
- `experiments/results/alloy_case_audit.json`
- `experiments/results/mace_summary.csv`

## Known scope limits

- The MACE study is five grouped 4,000-record subsets, not a full 45,229-record ten-fold study.
- The MACE evidence includes commands, predictions, and metrics. Multi-gigabyte model weights remain on the compute host described in `experiments/results/mace_environment.json`.
- The ODDS benchmark summary is reported in the manuscript. The 23 third-party input files are not redistributed; use the official benchmark source.
- The alloy audit separates missing fields and unusual-value candidates from source-confirmed errors. The supplied workbooks do not establish a confirmed-error label for every row.
- Synthetic corruption labels and the material rules overlap by construction. Results from this benchmark should not be interpreted as performance on unknown real-world error types.
