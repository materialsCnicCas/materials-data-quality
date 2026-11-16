"""Data ingestion and preprocessing for the materials quality pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from project.src.input_adapter import MaterialInputAdapter


class DataProcessor:
    """Run the active input-adaptation-based preprocessing pipeline."""

    def __init__(self, random_seed: int = 42, sample_n: int | None = None):
        self.random_seed = random_seed
        self.sample_n = sample_n
        self.input_adapter = MaterialInputAdapter()

    def ingest_input_source(self, source_path, dataset_type=None):
        """Parse a file or directory into unified raw records."""
        return self.input_adapter.ingest_source(source_path, dataset_type=dataset_type)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop empty/structured columns and coerce numeric-looking values."""
        print("Running data cleaning...")

        numeric_colnames = [
            col
            for col in df.columns
            if str(col)
            .replace(".", "")
            .replace("-", "")
            .replace("e", "")
            .replace("+", "")
            .isdigit()
        ]
        if len(numeric_colnames) > 10:
            print(
                f"Warning: detected {len(numeric_colnames)} numeric-looking column names."
            )
            print(f"Examples: {numeric_colnames[:10]}")

        print("Dropping fully empty rows/columns and coercing numeric strings...")
        original_shape = df.shape

        df = df.dropna(how="all", axis=0)
        df = df.dropna(how="all", axis=1)

        clean_cols = []
        for col in df.columns:
            try:
                sample_values = df[col].dropna().head(20).tolist()
                has_complex_types = any(
                    isinstance(val, (dict, list)) for val in sample_values
                )
                if has_complex_types:
                    print(f"  Skip nested/structured column: {col}")
                    continue
                clean_cols.append(col)
            except Exception as exc:
                print(f"  Failed to inspect column {col}, skipped: {exc}")

        if len(clean_cols) < len(df.columns):
            print(f"  Removed {len(df.columns) - len(clean_cols)} complex columns.")
            df = df[clean_cols]

        for col in df.columns:
            if df[col].dtype != "object":
                continue
            try:
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                if not numeric_col.isna().all():
                    df[col] = numeric_col
            except (TypeError, ValueError):
                pass

        print(f"Cleaning finished: {original_shape} -> {df.shape}")
        return df

    def identify_features(self, df: pd.DataFrame):
        """Create a stable sample id and select numeric feature columns."""
        print("Identifying numeric features...")

        identifier_col = "sample_id"
        if identifier_col in df.columns:
            print(f"Existing {identifier_col} detected; overwriting it.")
            df = df.drop(columns=[identifier_col])

        df.insert(0, identifier_col, range(1, len(df) + 1))
        print(f"Using stable identifier column: {identifier_col}")

        if "source_sample_id" in df.columns:
            print("Preserving source_sample_id as the upstream record id.")

        all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # ``elements`` in the MP20 CSV is a duplicated numeric copy of
        # ``e_above_hull`` (verified row-by-row); it is not an element-count
        # feature and must not enter quality scoring as an independent field.
        excluded_quality_columns = {identifier_col, "elements"}
        num_cols = [col for col in all_numeric_cols if col not in excluded_quality_columns]

        # Literature tensile-property tables contain many engineered descriptors
        # (products, tags, duplicate-management fields).  They are useful for
        # downstream modelling but are not independent evidence for data quality.
        # For UTS/FE tables, restrict the quality graph to raw composition,
        # test condition, targets, and a small set of physically interpretable
        # descriptors.  This prevents zero-variance and derived columns from
        # generating spurious SDAD violations.
        if {"UTS", "FE", "Tem"}.issubset(set(df.columns)):
            raw_composition = {
                c for c in [
                    "Co", "Cr", "Fe", "Mn", "Ni", "Al", "Ti", "V", "Cu",
                    "W", "Mo", "Zr", "Nb", "Ta", "C", "Hf", "N", "Si",
                    "Y", "B", "O", "P", "S",
                ] if c in df.columns
            }
            interpretable = {
                "Tem", "UTS", "FE", "composition_sum_raw",
                "composition_sum_atpct", "n_elements", "max_element_atpct",
                "min_nonzero_element_atpct", "composition_std", "DeltaR",
                "DeltaSmix", "DeltaHmix_metal_only", "Omega_metal_only",
                "homologous_T",
            }
            allowed = raw_composition | interpretable
            num_cols = [c for c in num_cols if c in allowed]
            print(f"Literature tensile table detected: restricted quality features to {len(num_cols)} raw/interpretable fields.")

        # Coordinate slots beyond a material's atom count are structural
        # missingness. The handler computes the per-row mask; expose the
        # candidate columns for audit without globally exempting real gaps.
        structural_missing_cols = {
            col for col in df.columns
            if str(col).startswith("coord_") or str(col).startswith("atom_element_")
        }

        print(f"Total numeric columns: {len(all_numeric_cols)}")
        print(f"Numeric quality features: {len(num_cols)}")
        if len(num_cols) <= 20:
            print(f"Feature columns: {num_cols}")
        else:
            print(f"First 20 feature columns: {num_cols[:20]}")

        final_cols = [identifier_col] + num_cols
        available_cols = [col for col in final_cols if col in df.columns]
        if len(available_cols) != len(final_cols):
            missing_cols = sorted(set(final_cols) - set(available_cols))
            print(f"Warning: skipped missing columns: {missing_cols}")

        df_processed = df[available_cols].copy()

        print(f"Processed shape: {df_processed.shape}")
        return identifier_col, num_cols, df_processed, structural_missing_cols

    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optionally subsample rows for a faster run."""
        if self.sample_n is not None and self.sample_n < len(df):
            df = df.sample(self.sample_n, random_state=self.random_seed).reset_index(
                drop=True
            )
            print(f"Sampled shape: {df.shape}")
        return df

    def process_data(self, data_path, output_dir, dataset_type=None):
        """Run ingestion, adaptation, cleaning, feature selection, and persistence."""
        print("Running source ingestion and format adaptation...")
        ingestion = self.ingest_input_source(data_path, dataset_type=dataset_type)
        data_file = ingestion.selected_file
        print(f"Detected input format: {ingestion.detected_format}")
        print(f"Selected input file: {data_file}")
        print(f"Unified raw record count: {len(ingestion.raw_records)}")

        raw_df = ingestion.to_dataframe()
        if raw_df is None:
            raise ValueError("Failed to load input data.")
        if raw_df.empty and not ingestion.raw_records:
            raise ValueError("Input dataset is empty.")

        raw_shape = raw_df.shape
        print(f"Raw shape: {raw_shape}")
        print(f"First 5 raw columns: {list(raw_df.columns)[:5]}")

        print("Running semantic input adaptation...")
        adaptation = self.input_adapter.adapt(raw_df, source_path=data_file)
        standardized_df = adaptation.normalized_df.copy()
        print(f"Adaptation finished: {raw_shape} -> {standardized_df.shape}")

        cleaned_df = self.clean_data(standardized_df.copy())
        sampled_df = self.sample_data(cleaned_df).reset_index(drop=True)

        (
            identifier_col,
            num_cols,
            df_processed,
            skip_missing_check_cols,
        ) = self.identify_features(sampled_df)
        if not num_cols:
            raise ValueError("No usable numeric quality features were detected.")

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        input_filename = Path(data_file).stem
        processed_csv = output_root / f"{input_filename}_processed.csv"
        standardized_csv = (
            output_root / f"{input_filename}_standardized_materials_table.csv"
        )
        processing_info_json = output_root / f"{input_filename}_processing_info.json"
        adaptation_summary_json = (
            output_root / f"{input_filename}_input_adaptation_summary.json"
        )
        adaptation_bundle_json = (
            output_root / f"{input_filename}_input_adaptation_bundle.json"
        )

        standardized_df.to_csv(standardized_csv, index=False)
        df_processed.to_csv(processed_csv, index=False)
        print(f"Processed dataset saved to: {processed_csv}")

        adaptation_summary = self._build_input_adaptation_summary(
            ingestion=ingestion.to_dict(),
            adaptation=adaptation.to_dict(),
            processed_df=df_processed,
            standardized_df=standardized_df,
        )
        adaptation_bundle = {
            "summary": adaptation_summary,
            "ingestion": ingestion.to_dict(),
            "raw_records_preview": ingestion.raw_records_preview(limit=5),
            "reports": adaptation.to_dict().get("reports", {}),
            "warnings": sorted(
                {
                    *ingestion.warnings,
                    *(adaptation.to_dict().get("warnings", []) or []),
                }
            ),
        }

        processing_info = {
            "source": {
                "requested_path": str(data_path),
                "selected_file": str(data_file),
                "detected_format": ingestion.detected_format,
                "input_kind": ingestion.input_kind,
            },
            "shapes": {
                "raw": {
                    "rows": int(raw_shape[0]),
                    "columns": int(raw_shape[1]),
                },
                "standardized": {
                    "rows": int(standardized_df.shape[0]),
                    "columns": int(standardized_df.shape[1]),
                },
                "processed": {
                    "rows": int(df_processed.shape[0]),
                    "columns": int(df_processed.shape[1]),
                },
            },
            "features": {
                "identifier_column": identifier_col,
                "numeric_feature_count": len(num_cols),
                "numeric_feature_columns": list(num_cols),
                "sample_n": self.sample_n,
            },
            "artifacts": {
                "processed_csv": str(processed_csv),
                "standardized_csv": str(standardized_csv),
                "processing_info_json": str(processing_info_json),
                "input_adaptation_summary_json": str(adaptation_summary_json),
                "input_adaptation_bundle_json": str(adaptation_bundle_json),
            },
        }

        processing_info_json.write_text(
            json.dumps(processing_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        adaptation_summary_json.write_text(
            json.dumps(adaptation_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        adaptation_bundle_json.write_text(
            json.dumps(adaptation_bundle, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "processed_df": df_processed,
            "standardized_df": standardized_df,
            "identifier_col": identifier_col,
            "num_cols": num_cols,
            "original_df": raw_df,
            "processing_info": processing_info,
            "output_file": str(processed_csv),
            "standardized_output_file": str(standardized_csv),
            "skip_missing_check_cols": skip_missing_check_cols,
            "input_adaptation": {
                "summary": adaptation_summary,
                "bundle": adaptation_bundle,
                "files": {
                    "summary": str(adaptation_summary_json),
                    "bundle": str(adaptation_bundle_json),
                },
            },
        }

    @staticmethod
    def _build_input_adaptation_summary(
        *,
        ingestion: dict,
        adaptation: dict,
        processed_df: pd.DataFrame,
        standardized_df: pd.DataFrame,
    ) -> dict:
        input_summary = adaptation.get("summary", {})
        field_mapping = (
            adaptation.get("reports", {}).get("field_mapping", {}).get("summary", {})
        )
        return {
            "source": {
                "selected_file": ingestion.get("selected_file"),
                "detected_format": ingestion.get("detected_format"),
                "record_count": ingestion.get("record_count"),
            },
            "shape_summary": {
                "raw_rows": input_summary.get("raw_rows"),
                "raw_columns": input_summary.get("raw_columns"),
                "standardized_rows": int(standardized_df.shape[0]),
                "standardized_columns": int(standardized_df.shape[1]),
                "processed_rows": int(processed_df.shape[0]),
                "processed_columns": int(processed_df.shape[1]),
            },
            "field_mapping": {
                "mapping_count": field_mapping.get("mapping_count", 0),
                "agent_mapping_count": field_mapping.get("agent_mapping_count", 0),
                "unresolved_column_count": field_mapping.get(
                    "unresolved_column_count", 0
                ),
                "passthrough_column_count": field_mapping.get(
                    "passthrough_column_count", 0
                ),
            },
            "unit_normalization": {
                "operation_count": input_summary.get("unit_operation_count", 0),
            },
            "coverage": {
                "recommended_rule_packs": input_summary.get(
                    "recommended_rule_packs", {}
                ),
            },
        }
