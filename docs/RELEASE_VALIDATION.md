# Release validation

Validation date: 2026-11-16

## Passed checks

- All 91 Python files in the release were parsed successfully.
- `main_v2.py` and the core `project/` package imported successfully under Python 3.9.7.
- The no-API-key manual pipeline completed on `examples/mp20_demo_1000.csv` with exit code 0 and generated the consolidated JSON, HTML, and Markdown reports.
- The paper-focused reference implementation completed a 300-record scoring smoke test, including static propagation and both rule-GAT branches. Static propagation converged in 37 iterations.
- The alloy audit script completed on the 185- and 315-record workbooks and reproduced the bundled audit counts.
- The ODDS evaluator completed on `examples/odds_demo.csv` and produced an AUC-ROC of 0.981389 on the synthetic smoke dataset.
- The checksum manifest was generated for all release files.

## Not executed locally during this release check

- The full five-seed MACE study requires the separate Python 3.10/CUDA environment described in `docs/MACE_PROTOCOL.md`.
- The 23-dataset ODDS aggregate was not rerun because the third-party ODDS files are not part of the local workspace and are not redistributed.
- The Docker daemon was unavailable on the packaging host. The Dockerfile is included, but the container image was not built during this validation pass.

The checked-in `experiments/results/` and `experiments/mace_evidence/` files preserve the recorded runs used for the manuscript tables.
