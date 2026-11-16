# Materials Data Quality Assessment

Reference code and research data for:

**Confidence-Based Materials Data Quality Assessment via Domain Knowledge and Graph-Structured Learning**

Sihan Liang, Ji Zhang, Kehan Wang, Zihao Gao, Wei Ma, Yangang Wang, and Zongguo Wang

This repository contains the implementation used for the manuscript revision, the data needed for the reported local experiments, run-level results, and the execution records for the external MACE experiments. The release is aligned with the 23-page KBS manuscript dated 21 October 2026.

## Repository layout

```text
.
|-- main_v2.py                 # General data-quality pipeline entry point
|-- project/                   # Pipeline implementation and material rules
|-- experiments/
|   |-- scripts/               # Paper experiment and verification scripts
|   |-- frozen_source/         # Frozen helper implementations used by the paper code
|   |-- results/               # Published run-level and aggregate results
|   |-- artifacts/             # Generated figures and replacement tables
|   |-- mace_evidence/         # MACE commands, predictions, logs, and summaries
|   `-- mace_inputs.tar.gz     # Frozen structure-level inputs for the MACE study
|-- data/
|   |-- mp20/                  # mp_20 input and controlled-contamination snapshots
|   `-- alloys/                # 185- and 315-record alloy workbooks
|-- examples/                  # Small deterministic smoke-test inputs
|-- scripts/                   # Release verification and data-subset utilities
`-- docs/                      # Reproduction and data-source notes
```

Two implementations are included for different uses:

- `experiments/scripts/` is the paper-focused reference implementation. It contains the scoring, ablation, robustness, sensitivity, propagation check, alloy audit, Memory-GAT, and MACE input-export code.
- `main_v2.py` with `project/` is the broader application pipeline. It accepts general CSV/XLSX inputs and produces sample- and feature-level outputs, audit reports, and rule evidence.

## Quick start

Python 3.9 is the reference runtime for the paper experiments.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python examples/run_demo.py
```

The demo does not require an LLM API key. It runs the manual, non-interactive quality pipeline on a deterministic 1,000-record `mp_20` subset and writes results under `runtime_outputs/`.

To run the general pipeline directly:

```bash
python main_v2.py examples/mp20_demo_1000.csv \
  --profile paper \
  --system-mode manual \
  --non-interactive \
  --disable-quality-memory \
  --output-dir runtime_outputs/manual
```

Set `SRD_CLUSTER_MODEL_DIR` if reusable clustering models should be written outside the source tree:

```bash
SRD_CLUSTER_MODEL_DIR=runtime_outputs/cluster_model python examples/run_demo.py
```

On PowerShell:

```powershell
$env:SRD_CLUSTER_MODEL_DIR = "runtime_outputs\cluster_model"
python examples/run_demo.py
```

## Paper experiments

Install the graph and analysis dependencies before running the full experiments:

```bash
python -m pip install -r requirements-analysis.txt
```

Primary commands:

```bash
# Main 16,000-record experiment and summaries
python experiments/scripts/run_suite.py

# Primary five-seed evaluation with independent internal calibration
python experiments/scripts/run_validation.py

# Contamination ratio x mixed-input scale experiment
python experiments/scripts/run_robustness.py

# Independent propagation check, Memory-GAT study, and runtime scaling
python experiments/scripts/run_additional.py reproduction
python experiments/scripts/run_additional.py memory
python experiments/scripts/run_additional.py timing
python experiments/scripts/run_history_cost.py

# Alloy workbook audit and split/retention checks
python experiments/scripts/run_case_audit.py
python experiments/scripts/check_protocol.py

# Rebuild figures and replacement tables from recorded results
python experiments/scripts/build_artifacts.py
```

The experiment scripts are intentionally sequential where an intermediate corruption table is shared. Use separate output copies for parallel runs.

The MACE experiment uses a separate Python 3.10/CUDA environment and pretrained-model workflow. See `docs/MACE_PROTOCOL.md`. Full MACE model weights are not included because they are several gigabytes; commands, input hashes, predictions, and evaluation records are provided in `experiments/mace_evidence/`.

The 23-dataset ODDS comparison uses third-party benchmark files that are not redistributed here. `experiments/scripts/run_odds_benchmark.py` evaluates numeric datasets supplied by the user and reports AUC-ROC. See `docs/ODDS_BENCHMARK.md`.

## Included data

| File | Rows | Role |
|---|---:|---|
| `data/mp20/mp_20.csv` | 45,229 | Materials Project-derived `mp_20` input |
| `data/mp20/benchmarks/mp20_mixed_pollution_dataset_16000.csv` | 16,000 | Mixed-contamination benchmark snapshot |
| `data/mp20/benchmarks/mp20_robustness_clean_12000.csv` | 12,000 | Robustness clean input |
| `data/mp20/benchmarks/mp20_robustness_noisy_12000.csv` | 12,000 | Robustness noisy input |
| `data/alloys/alloy_case_study_185_records.xlsx` | 185 | Literature-derived alloy case study |
| `data/alloys/alloy_reference_315_records.xlsx` | 315 | Literature-derived auxiliary workbooks |

The SHA-256 hashes are recorded in `manifests/SHA256SUMS.txt`. The `mp_20` hash used by the reported local experiments is:

```text
c0f55956e7d0615c15d0c2044865465cdb02846d69e0b3091d7b9c8ffb65c63a
```

See `DATA_NOTICE.md` before reusing or republishing the data. No additional license is asserted over third-party source data or literature-derived tables.

## Optional LLM features

The manual quality-assessment workflow and all reported local scoring experiments run without an LLM. Optional agent-assisted features read `DEEPSEEK_API_KEY` from the environment:

```bash
export DEEPSEEK_API_KEY="your-key"
```

No API key is stored in this repository. Unsupported or unavailable optional GAT components are reported by the code rather than silently substituted.

## Verification

Run:

```bash
python scripts/verify_release.py
python examples/run_demo.py
```

The first command checks imports, data hashes, the absence of hard-coded credentials and machine-specific absolute paths, and the presence of core result files. The second command performs a no-API-key end-to-end pipeline smoke test.

## Citation

Citation metadata is provided in `CITATION.cff`. If the repository is used in a publication, cite the associated manuscript and the upstream data sources listed in `DATA_NOTICE.md`.
