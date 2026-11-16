"""Build agent-facing dataset contracts from processed pipeline artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from project.agents.contracts import DatasetContract, FieldProfile


MATERIAL_DOMAIN_SCOPE = "materials_database"

_SEMANTIC_KEYWORDS = {
    "materials": (
        "band_gap",
        "e_above_hull",
        "formation_energy",
        "spacegroup",
        "lattice",
        "abc_",
        "angle_",
        "coord_",
        "atom_element",
        "element",
        "formula",
    ),
    "materials_thermodynamic": (
        "formation_energy",
        "e_above_hull",
        "energy",
        "enthalpy",
        "stability",
    ),
    "materials_electronic": (
        "band_gap",
        "dos",
        "fermi",
        "conductivity",
        "dielectric",
    ),
    "materials_structure": (
        "spacegroup",
        "lattice",
        "abc_",
        "angle_",
    ),
    "materials_atomic_geometry": (
        "coord_",
        "atom_",
        "site_",
        "cartesian",
    ),
    "materials_composition": (
        "element",
        "formula",
        "composition",
        "species",
    ),
    "identifier": (
        "id",
        "material_id",
        "sample_id",
    ),
}


@dataclass(frozen=True)
class ContractBuildInputs:
    processed_df: pd.DataFrame
    identifier_col: str
    num_cols: list[str]
    source_path: str | None = None
    source_format: str | None = None
    metadata: dict[str, Any] | None = None
    skip_missing_check_cols: set[str] | None = None


class DatasetContractBuilder:
    """Convert processed tables into a compact, agent-facing contract."""

    def build(self, inputs: ContractBuildInputs) -> DatasetContract:
        df = inputs.processed_df
        field_profiles = tuple(
            self._build_field_profile(df, column, inputs.num_cols, inputs.metadata)
            for column in df.columns
        )
        dataset_hints = self._aggregate_dataset_hints(field_profiles)
        material_applicability = self._material_applicability(
            field_profiles,
            dataset_hints,
        )
        source_path = inputs.source_path
        source_format = inputs.source_format or self._infer_source_format(source_path)
        source_meta = self._build_source_meta(inputs, df)
        domain_scope = MATERIAL_DOMAIN_SCOPE

        return DatasetContract(
            dataset_name=self._infer_dataset_name(source_path),
            row_count=int(df.shape[0]),
            column_count=int(df.shape[1]),
            source_format=source_format,
            identifier_column=inputs.identifier_col,
            field_profiles=field_profiles,
            source_meta=source_meta,
            semantic_hints=dataset_hints,
            domain_scope=domain_scope,
            material_applicability=material_applicability,
        )

    def _build_field_profile(
        self,
        df: pd.DataFrame,
        column: str,
        num_cols: list[str],
        metadata: dict[str, Any] | None,
    ) -> FieldProfile:
        series = df[column]
        logical_type = self._infer_logical_type(series, column, num_cols)
        semantic_hints = self._infer_semantic_hints(column, logical_type)
        null_fraction = float(series.isna().mean()) if len(series) else 0.0
        distinct_fraction = (
            float(series.nunique(dropna=True) / len(series)) if len(series) else 0.0
        )
        sample_values = tuple(self._sample_values(series))

        if metadata and column in metadata:
            meta_entry = metadata[column]
            semantic_hints = tuple(
                dict.fromkeys(
                    semantic_hints
                    + tuple(self._metadata_hints(meta_entry))
                )
            )

        return FieldProfile(
            name=column,
            logical_type=logical_type,
            null_fraction=null_fraction,
            distinct_fraction=distinct_fraction,
            sample_values=sample_values,
            semantic_hints=semantic_hints,
        )

    @staticmethod
    def _sample_values(series: pd.Series, limit: int = 5) -> list[str]:
        non_null = series.dropna()
        if non_null.empty:
            return []
        seen: list[str] = []
        for value in non_null.head(limit * 3):
            rendered = str(value)
            if rendered not in seen:
                seen.append(rendered)
            if len(seen) >= limit:
                break
        return seen

    @staticmethod
    def _metadata_hints(meta_entry: dict[str, Any]) -> list[str]:
        hints: list[str] = []
        unit = str(meta_entry.get("unit", "")).lower()
        description = str(
            meta_entry.get("desc", meta_entry.get("description", ""))
        ).lower()
        if "ev" in unit or "crystal" in description or "material" in description:
            hints.append("materials")
        if "ev" in unit or "stability" in description:
            hints.append("materials_thermodynamic")
        if "angstrom" in unit or "lattice" in description:
            hints.append("materials_structure")
        return hints

    @staticmethod
    def _infer_logical_type(
        series: pd.Series,
        column: str,
        num_cols: list[str],
    ) -> str:
        if is_bool_dtype(series):
            return "boolean"
        if is_datetime64_any_dtype(series):
            return "datetime"
        if is_integer_dtype(series):
            return "integer"
        if is_float_dtype(series):
            return "float"
        if is_numeric_dtype(series) or column in num_cols:
            return "numeric"
        if is_string_dtype(series) or is_object_dtype(series):
            return "string"
        return "unknown"

    def _infer_semantic_hints(self, column: str, logical_type: str) -> tuple[str, ...]:
        lowered = column.lower()
        hints: list[str] = []
        for hint, keywords in _SEMANTIC_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                hints.append(hint)
        if logical_type in {"integer", "float", "numeric"}:
            hints.append("numeric")
        return tuple(dict.fromkeys(hints))

    @staticmethod
    def _aggregate_dataset_hints(field_profiles: tuple[FieldProfile, ...]) -> tuple[str, ...]:
        counts = Counter(
            hint
            for profile in field_profiles
            for hint in profile.semantic_hints
        )
        ordered = [
            hint for hint, _ in counts.most_common()
            if hint not in {"numeric", "identifier"}
        ]
        if counts.get("materials", 0) >= 3:
            ordered.insert(0, "materials")
        return tuple(dict.fromkeys(ordered))

    @staticmethod
    def _material_applicability(
        field_profiles: tuple[FieldProfile, ...],
        dataset_hints: tuple[str, ...],
    ) -> dict[str, Any]:
        counts = Counter(
            hint
            for profile in field_profiles
            for hint in profile.semantic_hints
            if hint.startswith("materials")
        )
        subdomains = tuple(
            hint
            for hint in dataset_hints
            if hint.startswith("materials_")
        )
        matched_fields = tuple(
            profile.name
            for profile in field_profiles
            if any(hint.startswith("materials") for hint in profile.semantic_hints)
        )
        score = int(sum(counts.values()))
        has_material_hints = bool(counts.get("materials", 0) >= 1 and subdomains)
        if has_material_hints:
            reason = (
                "materials subdomain field semantics detected: "
                + ", ".join(subdomains)
            )
        elif counts:
            reason = (
                "materials-generic hints detected; no specific subdomain coverage"
            )
        else:
            reason = (
                "no material subdomain fields detected; baseline material checks "
                "remain available"
            )
        return {
            "assumed_materials_dataset": True,
            "has_material_subdomain_evidence": has_material_hints,
            "score": score,
            "subdomains": list(subdomains),
            "matched_fields": list(matched_fields),
            "hint_counts": dict(counts),
            "reason": reason,
        }

    @staticmethod
    def _infer_dataset_name(source_path: str | None) -> str:
        if not source_path:
            return "in_memory_dataset"
        return Path(source_path).stem

    @staticmethod
    def _infer_source_format(source_path: str | None) -> str:
        if not source_path:
            return "unknown"
        suffix = Path(source_path).suffix.lower().lstrip(".")
        return suffix or "unknown"

    @staticmethod
    def _build_source_meta(inputs: ContractBuildInputs, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "source_path": inputs.source_path,
            "numeric_columns": list(inputs.num_cols),
            "skip_missing_check_cols": sorted(inputs.skip_missing_check_cols or []),
            "column_names": list(df.columns),
        }
