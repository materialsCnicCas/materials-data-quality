# Data index

## `mp20/mp_20.csv`

The `mp_20` table used in the manuscript experiments. It contains 45,229 records and 123 source columns. SHA-256:

```text
c0f55956e7d0615c15d0c2044865465cdb02846d69e0b3091d7b9c8ffb65c63a
```

The table is derived from third-party Materials Project data. The repository does not assert a new license over the source records.

## `mp20/benchmarks/`

- `mp20_mixed_pollution_dataset_16000.csv`: 16,000-record controlled mixed-contamination snapshot.
- `mp20_robustness_clean_12000.csv`: 12,000-record clean robustness input.
- `mp20_robustness_noisy_12000.csv`: 12,000-record noisy robustness input.

Their SHA-256 values are:

```text
2f4e17ec8bd7583cb608ada957a2b5b4947e9bceb1d35a8aca08fdad72c11268  mp20_mixed_pollution_dataset_16000.csv
6ee3117ab17dd0509bbbec5ae499f713b8f7f04e849a1c310535acf396aa1708  mp20_robustness_clean_12000.csv
e2c7312322a3a4607596f7ebab3cbc8926df6feda66a11d9d4a889056171cdaf9  mp20_robustness_noisy_12000.csv
```

These files are research snapshots generated for the paper analyses. The experiment scripts can also regenerate corruption from the frozen `mp_20` input.

## `alloys/`

- `alloy_case_study_185_records.xlsx`: 185 records and 50 fields.
- `alloy_reference_315_records.xlsx`: 315 records and 61 fields.

```text
f54a3aca388e3aab625989391d2886cc3321f70e1fa81378f815b5fad80941e8  alloy_case_study_185_records.xlsx
a00c890676bb413879508fe0758323c8e6b52f631e469a1e281c52485670a9b5  alloy_reference_315_records.xlsx
```

The two workbooks are structured data compiled from public literature. They are used for an audit of observable missing fields, unusual values, and context-incomplete property spreads. They are not distributed as source-confirmed error annotations. See `DATA_NOTICE.md`.

## Excluded source data

The 23 ODDS benchmark files, NASA battery archives, MACE model weights, and other unrelated source files are not included. See `docs/ODDS_BENCHMARK.md` and `DATA_NOTICE.md`.
