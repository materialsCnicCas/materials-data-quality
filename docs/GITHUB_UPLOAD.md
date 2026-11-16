# GitHub upload checklist

1. Revoke and replace the API key that was previously embedded in the source workspace. The release contains no key, but the prior value should be treated as exposed.
2. Create a new public repository under the `materialsCnicCas` organization. The suggested name is `materials-data-quality`.
3. Upload the contents of this directory, not its parent `github_upload` directory.
4. Confirm that the uploaded repository shows the files `README.md`, `LICENSE`, `DATA_NOTICE.md`, `requirements-core.txt`, `main_v2.py`, `project/`, `experiments/`, and `data/`.
5. Run the GitHub Actions `smoke-test` workflow and confirm that the manual pipeline completes without `DEEPSEEK_API_KEY`.
6. Use the resulting repository URL in the manuscript's code-availability statement.

Suggested repository description:

> Reference implementation and reproducibility materials for confidence-based materials data quality assessment.

Suggested topics:

```text
materials-science
data-quality
anomaly-detection
graph-neural-networks
reproducibility
```

Do not upload `output/`, `work/`, `runtime_outputs/`, the original private workspaces, submission packages, or files from `github_upload/.build`.
