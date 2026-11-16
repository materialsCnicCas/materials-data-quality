# Experiment code and outputs

The scripts under `scripts/` are the reference implementation used to regenerate the manuscript analyses.

## Main analyses

| Script | Purpose | Primary outputs |
|---|---|---|
| `run_suite.py` | Main mixed-contamination and sensitivity experiment | `results/main_*.csv`, `results/sensitivity_*.csv`, `results/timing_*.json` |
| `run_validation.py` | Five-seed evaluation with separate internal calibration | `results/validation_*.csv` |
| `run_robustness.py` | Contamination ratio x noise scale analysis | `results/robustness_*.csv` |
| `run_additional.py` | Independent propagation check, Memory-GAT, and runtime measurements | `results/independent_propagation_check.json`, `results/memory_*.csv`, `results/timing_*.json` |
| `run_history_cost.py` | Fresh-process historical-reference cost measurements | `results/history_cost_*.json` |
| `run_case_audit.py` | Record-level audit of the two alloy workbooks | `results/alloy*_audit.csv`, `results/alloy_case_audit.json` |
| `check_protocol.py` | Split independence, retention, and target-independence checks | `results/protocol_checks.json` |
| `export_mace.py` | Export grouped MACE inputs from `mp_20` | `mace_inputs/seed*/` |
| `remote_mace.py` | Train and evaluate the MACE conditions on a CUDA host | MACE run directories |
| `remote_collect.py` | Collect commands, metrics, predictions, and hashes | `mace_evidence/` |
| `build_artifacts.py` | Rebuild result figures and replacement tables | `artifacts/` |
| `run_odds_benchmark.py` | Evaluate a user-supplied ODDS numeric benchmark | CSV summary with AUC-ROC |

The scripts write into this folder. Run them from a fresh copy when comparing regenerated results against the checked-in outputs.

`frozen_source/` preserves the helper implementations used by the paper code so that later changes in the broader application pipeline do not silently alter the reported analyses.

The `results/` and `artifacts/` directories contain the recorded outputs used for the manuscript. The `mace_evidence/` directory contains the external GPU execution records. Machine-specific remote roots are replaced by `<REMOTE_HOME>` in the archived command logs.
