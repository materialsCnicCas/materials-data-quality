"""Input adaptation for materials data.

This module normalizes heterogeneous user inputs into a stable internal
tabular form before the existing quality pipeline runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from project.agents.field_mapping_agent import FieldMappingAgent, NO_MAPPING_TARGET
from project.rule_registry.registry import build_rule_registry

try:
    import bson

    decode_all = bson.decode_all
    BSON = bson.BSON
    BSON_AVAILABLE = True
except (ImportError, AttributeError):
    decode_all = None
    BSON = None
    BSON_AVAILABLE = False


def _canonical_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _semantic_tokens(name: Any) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name).strip())
    parts = [
        token.lower()
        for token in re.split(r"[^a-zA-Z0-9]+", text)
        if token.strip()
    ]
    return set(parts)


def _is_json_like_text(value: str) -> bool:
    text = value.strip()
    return (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    )


def _maybe_parse_value(value: Any) -> Any:
    if isinstance(value, str) and _is_json_like_text(value):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


@dataclass
class InputAdaptationResult:
    normalized_df: pd.DataFrame
    input_format_report: dict[str, Any]
    schema_profile: dict[str, Any]
    field_mapping_manifest: dict[str, Any]
    adaptive_flattening_rules: dict[str, Any]
    unit_normalization_report: dict[str, Any]
    coverage_analysis_report: dict[str, Any]
    pipeline_report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        mapping_summary = self.field_mapping_manifest.get("summary", {})
        coverage_summary = self.coverage_analysis_report.get(
            "recommended_rule_packs", {}
        )
        return {
            "summary": {
                "raw_rows": self.input_format_report.get("raw_row_count"),
                "raw_columns": self.input_format_report.get("raw_column_count"),
                "normalized_rows": self.input_format_report.get(
                    "normalized_row_count"
                ),
                "normalized_columns": self.input_format_report.get(
                    "normalized_column_count"
                ),
                "mapping_count": mapping_summary.get("mapping_count", 0),
                "unresolved_column_count": mapping_summary.get(
                    "unresolved_column_count", 0
                ),
                "passthrough_column_count": mapping_summary.get(
                    "passthrough_column_count", 0
                ),
                "agent_mapping_count": mapping_summary.get(
                    "agent_mapping_count", 0
                ),
                "unit_operation_count": self.unit_normalization_report.get(
                    "operation_count", 0
                ),
                "recommended_rule_packs": {
                    "enable": coverage_summary.get("enable", []),
                    "manual_review": coverage_summary.get("manual_review", []),
                },
            },
            "reports": {
                "field_mapping": self.field_mapping_manifest,
                "flattening": {
                    "nested_source_columns": self.adaptive_flattening_rules.get(
                        "nested_source_columns", []
                    ),
                    "flattened_field_count": self.adaptive_flattening_rules.get(
                        "flattened_field_count", 0
                    ),
                    "truncated_list_fields": self.adaptive_flattening_rules.get(
                        "truncated_list_fields", []
                    ),
                },
                "unit_normalization": {
                    "operation_count": self.unit_normalization_report.get(
                        "operation_count", 0
                    ),
                    "operations": self.unit_normalization_report.get(
                        "operations", []
                    ),
                    "warnings": self.unit_normalization_report.get("warnings", []),
                },
                "coverage": {
                    "available_field_count": self.coverage_analysis_report.get(
                        "available_field_count", 0
                    ),
                    "missing_standard_fields": self.coverage_analysis_report.get(
                        "missing_standard_fields", []
                    ),
                    "recommended_rule_packs": self.coverage_analysis_report.get(
                        "recommended_rule_packs", {}
                    ),
                    "warnings": self.coverage_analysis_report.get(
                        "coverage_warnings", []
                    ),
                },
                "pipeline": {
                    "record_count": self.pipeline_report.get("record_count", 0),
                    "stages": self.pipeline_report.get("stages", {}),
                    "warnings": self.pipeline_report.get("warnings", []),
                },
            },
            "warnings": sorted(
                {
                    *self.pipeline_report.get("warnings", []),
                    *self.unit_normalization_report.get("warnings", []),
                    *self.coverage_analysis_report.get("coverage_warnings", []),
                    *list(self.warnings),
                }
            ),
        }


@dataclass
class SourceIngestionResult:
    source_path: str
    resolved_path: str
    selected_file: str
    input_kind: str
    detected_format: str
    detection_strategy: str
    raw_records: list[dict[str, Any]]
    detection_report: dict[str, Any]
    parse_report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.raw_records:
            return pd.DataFrame()
        return pd.DataFrame(self.raw_records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "resolved_path": self.resolved_path,
            "selected_file": self.selected_file,
            "input_kind": self.input_kind,
            "detected_format": self.detected_format,
            "detection_strategy": self.detection_strategy,
            "record_count": len(self.raw_records),
            "detection_report": self.detection_report,
            "parse_report": self.parse_report,
            "warnings": list(self.warnings),
        }

    def raw_records_preview(self, limit: int = 5) -> list[dict[str, Any]]:
        return self.raw_records[:limit]


class MaterialInputAdapter:
    """Adapt raw user input into a stable materials schema."""

    CONTRACT_VERSION = "1.0"

    STANDARD_ALIASES: dict[str, list[str]] = {
        "source_sample_id": [
            "sampleid",
            "materialid",
            "material_id",
            "id",
            "mpid",
            "sample",
            "recordid",
            "entryid",
            "datasetid",
        ],
        "formula": [
            "prettyformula",
            "chemicalformula",
            "chemformula",
            "composition",
            "reducedformula",
            "formula_pretty",
            "pretty_formula",
        ],
        "band_gap": [
            "bandgap",
            "egap",
            "egapev",
            "gap",
            "gapev",
            "bandgapev",
            "band_gap_ev",
            "electronicgap",
            "pbebandgap",
        ],
        "formation_energy_per_atom": [
            "formationenergyperatom",
            "formationenergy",
            "eform",
            "formationenergyatom",
            "eformperatom",
            "formationenergyevatom",
            "formationenergyevperatom",
            "e_form_per_atom",
        ],
        "e_above_hull": ["eabovehull", "abovehull", "ehull", "energyabovehull", "hullenergy"],
        "spacegroup_number": [
            "spacegroupnumber",
            "spacegroup",
            "spacegroupno",
            "spacegroup_num",
            "space_group_no",
            "space_group_number",
            "spg",
            "spgno",
            "sgnumber",
        ],
        "abc_0": ["latticea", "lattice_a", "cella", "cell_a", "aaxis", "latticeparametera"],
        "abc_1": ["latticeb", "lattice_b", "cellb", "cell_b", "baxis", "latticeparameterb"],
        "abc_2": ["latticec", "lattice_c", "cellc", "cell_c", "caxis", "latticeparameterc"],
        "angle_0": ["alpha", "anglealpha", "latticealpha", "angle_alpha"],
        "angle_1": ["beta", "anglebeta", "latticebeta", "angle_beta"],
        "angle_2": ["gamma", "anglegamma", "latticegamma", "angle_gamma"],
    }

    PATTERN_MAPPERS: list[tuple[re.Pattern[str], Any]] = [
        (re.compile(r"^coord[_]?(\d+)[_]?([xyz])$"), lambda m: f"coord_{m.group(1)}_{m.group(2)}"),
        (re.compile(r"^abc[_]?(\d+)$"), lambda m: f"abc_{m.group(1)}"),
        (re.compile(r"^angle[_]?(\d+)$"), lambda m: f"angle_{m.group(1)}"),
        (re.compile(r"^atomelement[_]?(\d+)$"), lambda m: f"atom_element_{m.group(1)}"),
        (re.compile(r"^prop[_]?(\d+)$"), lambda m: f"prop_{m.group(1)}"),
        (re.compile(r"^latticemat[_]?(\d+)[_]?(\d+)$"), lambda m: f"lattice_mat_{m.group(1)}_{m.group(2)}"),
        (re.compile(r"^structureangles[_]?(\d+)$"), lambda m: f"angle_{m.group(1)}"),
        (re.compile(r"^sites[_]?(\d+)[_]?element$"), lambda m: f"atom_element_{m.group(1)}"),
        (re.compile(r"^sites[_]?(\d+)[_]?coord[_]?([012])$"), lambda m: f"coord_{m.group(1)}_{['x', 'y', 'z'][int(m.group(2))]}"),
    ]

    UNIT_CONVERSIONS: dict[str, dict[str, Any]] = {
        "band_gap": {
            "standard_unit": "eV",
            "conversions": {
                "eV": lambda series: series,
                "meV": lambda series: series / 1000.0,
            },
        },
        "formation_energy_per_atom": {
            "standard_unit": "eV/atom",
            "conversions": {
                "eV/atom": lambda series: series,
                "meV/atom": lambda series: series / 1000.0,
            },
        },
        "e_above_hull": {
            "standard_unit": "eV",
            "conversions": {
                "eV": lambda series: series,
                "meV": lambda series: series / 1000.0,
            },
        },
        "abc_0": {
            "standard_unit": "angstrom",
            "conversions": {"angstrom": lambda series: series},
        },
        "abc_1": {
            "standard_unit": "angstrom",
            "conversions": {"angstrom": lambda series: series},
        },
        "abc_2": {
            "standard_unit": "angstrom",
            "conversions": {"angstrom": lambda series: series},
        },
        "angle_0": {
            "standard_unit": "degree",
            "conversions": {"degree": lambda series: series},
        },
        "angle_1": {
            "standard_unit": "degree",
            "conversions": {"degree": lambda series: series},
        },
        "angle_2": {
            "standard_unit": "degree",
            "conversions": {"degree": lambda series: series},
        },
    }

    def __init__(
        self,
        *,
        max_list_items: int = 20,
        enable_field_mapping_agent: bool = True,
        field_mapping_agent: FieldMappingAgent | None = None,
        field_mapping_agent_min_confidence: float = 0.72,
    ):
        self.max_list_items = max_list_items
        self.enable_field_mapping_agent = enable_field_mapping_agent
        self.field_mapping_agent = field_mapping_agent or FieldMappingAgent()
        self.field_mapping_agent_min_confidence = field_mapping_agent_min_confidence

    def ingest_source(
        self,
        source_path: str | Path,
        *,
        dataset_type: str | None = None,
    ) -> SourceIngestionResult:
        detection = self._detect_input_source(source_path)
        records, parse_report, warnings = self._parse_records(
            Path(detection["selected_file"]),
            detection["detected_format"],
            dataset_type=dataset_type,
        )

        return SourceIngestionResult(
            source_path=detection["source_path"],
            resolved_path=detection["resolved_path"],
            selected_file=detection["selected_file"],
            input_kind=detection["input_kind"],
            detected_format=detection["detected_format"],
            detection_strategy=detection["detection_strategy"],
            raw_records=records,
            detection_report=detection,
            parse_report=parse_report,
            warnings=warnings,
        )

    def adapt(
        self,
        df: pd.DataFrame,
        *,
        source_path: str | Path | None = None,
    ) -> InputAdaptationResult:
        df = self._normalize_dataframe_columns(df.copy())
        source_path = Path(source_path) if source_path else None
        raw_shape = tuple(df.shape)
        raw_columns = [str(col) for col in df.columns]
        nested_columns = self._detect_nested_columns(df)

        flattened_df, flattening_rules = self._expand_structured_records(df)
        mapped_df, mapping_manifest = self._apply_field_mapping(flattened_df)
        normalized_df, unit_normalization_report = self._normalize_units(
            mapped_df,
            mapping_manifest,
        )
        schema_profile = self._build_schema_profile(normalized_df)
        coverage_analysis_report = self._analyze_rule_coverage(normalized_df)

        input_format_report = {
            "source_path": str(source_path) if source_path else None,
            "input_extension": source_path.suffix.lower() if source_path else None,
            "raw_row_count": int(raw_shape[0]),
            "raw_column_count": int(raw_shape[1]),
            "raw_columns_sample": raw_columns[:40],
            "nested_columns": nested_columns,
            "normalized_row_count": int(normalized_df.shape[0]),
            "normalized_column_count": int(normalized_df.shape[1]),
            "has_nested_cells": bool(nested_columns),
        }

        pipeline_report = {
            "record_count": int(normalized_df.shape[0]),
            "input_profile": {
                "path": str(source_path) if source_path else None,
                "input_type": source_path.suffix.lower().lstrip(".") if source_path else None,
                "is_nested": bool(nested_columns),
                "source_format": source_path.suffix.lower().lstrip(".") if source_path else None,
            },
            "stages": {
                "module_3_structure_normalization": {
                    "status": "completed",
                    "output_column_count": int(flattened_df.shape[1]),
                },
                "module_4_field_mapping": {
                    "status": "completed",
                    "mapped_field_count": int(
                        mapping_manifest.get("summary", {}).get("mapping_count", 0)
                    ),
                    "agent_mapping_count": int(
                        mapping_manifest.get("summary", {}).get(
                            "agent_mapping_count", 0
                        )
                    ),
                    "unresolved_column_count": int(
                        mapping_manifest.get("summary", {}).get(
                            "unresolved_column_count", 0
                        )
                    ),
                },
                "module_5_unit_normalization": {
                    "status": "completed",
                    "operation_count": int(unit_normalization_report.get("operation_count", 0)),
                },
                "module_6_rule_coverage": {
                    "status": "completed",
                    "recommended_enable_count": len(
                        coverage_analysis_report.get("recommended_rule_packs", {}).get("enable", [])
                    ),
                },
                "module_7_pipeline_orchestrator": {
                    "status": "completed",
                    "final_standardized_column_count": int(normalized_df.shape[1]),
                },
            },
            "warnings": [
                *mapping_manifest.get("agent_review", {}).get("warnings", []),
                *unit_normalization_report.get("warnings", []),
                *coverage_analysis_report.get("coverage_warnings", []),
            ],
        }

        return InputAdaptationResult(
            normalized_df=normalized_df,
            input_format_report=input_format_report,
            schema_profile=schema_profile,
            field_mapping_manifest=mapping_manifest,
            adaptive_flattening_rules=flattening_rules,
            unit_normalization_report=unit_normalization_report,
            coverage_analysis_report=coverage_analysis_report,
            pipeline_report=pipeline_report,
            warnings=[],
        )

    def _detect_input_source(self, source_path: str | Path) -> dict[str, Any]:
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Input path does not exist: {source}")

        if source.is_dir():
            selected_file, detection_strategy, candidates = self._select_input_file_from_directory(source)
            detected_format = self._detect_format_from_path(selected_file)
            return {
                "source_path": str(source_path),
                "resolved_path": str(source),
                "input_kind": "directory",
                "selected_file": str(selected_file),
                "detected_format": detected_format,
                "detection_strategy": detection_strategy,
                "candidate_files": [str(item) for item in candidates],
            }

        detected_format = self._detect_format_from_path(source)
        return {
            "source_path": str(source_path),
            "resolved_path": str(source),
            "input_kind": "file",
            "selected_file": str(source),
            "detected_format": detected_format,
            "detection_strategy": "file_extension",
            "candidate_files": [str(source)],
        }

    def _select_input_file_from_directory(
        self,
        directory: Path,
    ) -> tuple[Path, str, list[Path]]:
        supported_extensions = [".csv", ".json", ".jsonl", ".xlsx", ".xls", ".parquet"]
        if BSON_AVAILABLE:
            supported_extensions.append(".bson")

        candidates: list[Path] = []
        for extension in supported_extensions:
            candidates.extend(sorted(directory.glob(f"*{extension}")))

        if not candidates:
            raise FileNotFoundError(f"No supported input file found in directory: {directory}")

        selected = max(candidates, key=lambda item: item.stat().st_size)
        return selected, "directory_scan_largest_supported_file", candidates

    @staticmethod
    def _detect_format_from_path(path: Path) -> str:
        extension = path.suffix.lower()
        if extension in {".csv", ".json", ".jsonl", ".bson", ".xlsx", ".xls", ".parquet"}:
            return extension.lstrip(".")
        raise ValueError(f"Unsupported input format: {extension}")

    def _parse_records(
        self,
        file_path: Path,
        detected_format: str,
        *,
        dataset_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        parser = getattr(self, f"_parse_{detected_format}_records", None)
        if parser is None:
            raise ValueError(f"No parser implemented for format: {detected_format}")
        return parser(file_path, dataset_type=dataset_type)

    def _parse_csv_records(
        self,
        file_path: Path,
        *,
        dataset_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        del dataset_type
        warnings: list[str] = []
        attempts: list[dict[str, Any]] = []

        try:
            df = pd.read_csv(file_path, low_memory=False)
            attempts.append({"strategy": "read_csv_default", "success": True})
            return (
                self._dataframe_to_records(df),
                {
                    "selected_parser": "csv",
                    "record_count": int(len(df)),
                    "column_count": int(len(df.columns)),
                    "attempts": attempts,
                },
                warnings,
            )
        except Exception as exc:
            attempts.append({"strategy": "read_csv_default", "success": False, "error": str(exc)})

        for separator in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(file_path, sep=separator, low_memory=False)
                if df.shape[1] <= 1:
                    attempts.append(
                        {
                            "strategy": f"read_csv_sep_{repr(separator)}",
                            "success": False,
                            "error": "single column result",
                        }
                    )
                    continue
                attempts.append({"strategy": f"read_csv_sep_{repr(separator)}", "success": True})
                return (
                    self._dataframe_to_records(df),
                    {
                        "selected_parser": "csv",
                        "record_count": int(len(df)),
                        "column_count": int(len(df.columns)),
                        "attempts": attempts,
                    },
                    warnings,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "strategy": f"read_csv_sep_{repr(separator)}",
                        "success": False,
                        "error": str(exc),
                    }
                )

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            for index, line in enumerate(lines[:10]):
                if line.strip() and not line.startswith("#") and any(char.isalpha() for char in line[:50]):
                    df = pd.read_csv(StringIO("".join(lines[index:])), low_memory=False)
                    attempts.append({"strategy": f"skip_preface_lines_{index}", "success": True})
                    warnings.append(f"Skipped {index} leading lines before CSV header detection.")
                    return (
                        self._dataframe_to_records(df),
                        {
                            "selected_parser": "csv",
                            "record_count": int(len(df)),
                            "column_count": int(len(df.columns)),
                            "attempts": attempts,
                        },
                        warnings,
                    )
        except Exception as exc:
            attempts.append({"strategy": "skip_preface_lines", "success": False, "error": str(exc)})

        raise ValueError(f"Unable to parse CSV file: {file_path}")

    def _parse_json_records(
        self,
        file_path: Path,
        *,
        dataset_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        del dataset_type
        warnings: list[str] = []
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        records, structure_type = self._coerce_json_payload_to_records(payload)
        return (
            records,
            {
                "selected_parser": "json",
                "record_count": len(records),
                "top_level_structure": structure_type,
            },
            warnings,
        )

    def _parse_jsonl_records(
        self,
        file_path: Path,
        *,
        dataset_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        del dataset_type
        warnings: list[str] = []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
            else:
                records.append({"value": obj})

        return (
            records,
            {
                "selected_parser": "jsonl",
                "record_count": len(records),
                "line_count": len(records),
            },
            warnings,
        )

    def _parse_bson_records(
        self,
        file_path: Path,
        *,
        dataset_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        del dataset_type
        if not BSON_AVAILABLE:
            raise ImportError("BSON parsing requires pymongo-provided bson support.")

        warnings: list[str] = []
        content = file_path.read_bytes()

        try:
            documents = decode_all(content)
            if documents:
                return (
                    [dict(item) for item in documents],
                    {
                        "selected_parser": "bson",
                        "strategy": "decode_all",
                        "record_count": len(documents),
                    },
                    warnings,
                )
        except Exception as exc:
            warnings.append(f"decode_all failed: {exc}")

        documents = []
        offset = 0
        while offset < len(content):
            if offset + 4 > len(content):
                break
            doc_length = int.from_bytes(content[offset : offset + 4], "little")
            if doc_length <= 0 or offset + doc_length > len(content):
                break
            document = BSON(content[offset : offset + doc_length]).decode()
            documents.append(document)
            offset += doc_length

        if not documents:
            raise ValueError(f"Unable to parse BSON file: {file_path}")

        return (
            [dict(item) for item in documents],
            {
                "selected_parser": "bson",
                "strategy": "stream_decode",
                "record_count": len(documents),
            },
            warnings,
        )

    def _parse_xlsx_records(
        self,
        file_path: Path,
        *,
        dataset_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        del dataset_type
        df = pd.read_excel(file_path)
        return (
            self._dataframe_to_records(df),
            {
                "selected_parser": "xlsx",
                "record_count": int(len(df)),
                "column_count": int(len(df.columns)),
            },
            [],
        )

    def _parse_xls_records(
        self,
        file_path: Path,
        *,
        dataset_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        return self._parse_xlsx_records(file_path, dataset_type=dataset_type)

    def _parse_parquet_records(
        self,
        file_path: Path,
        *,
        dataset_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
        del dataset_type
        df = pd.read_parquet(file_path)
        return (
            self._dataframe_to_records(df),
            {
                "selected_parser": "parquet",
                "record_count": int(len(df)),
                "column_count": int(len(df.columns)),
            },
            [],
        )

    def _coerce_json_payload_to_records(
        self,
        payload: Any,
    ) -> tuple[list[dict[str, Any]], str]:
        if isinstance(payload, list):
            if all(isinstance(item, dict) for item in payload):
                return [dict(item) for item in payload], "list_of_objects"
            return [{"value": item} for item in payload], "list_of_scalars"

        if isinstance(payload, dict):
            for key in ("data", "records", "rows", "items", "entries"):
                value = payload.get(key)
                if isinstance(value, list):
                    if all(isinstance(item, dict) for item in value):
                        return [dict(item) for item in value], f"object.{key}"
                    return [{"value": item} for item in value], f"object.{key}"
            return [dict(payload)], "single_object"

        return [{"value": payload}], "scalar"

    @staticmethod
    def _dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        df = MaterialInputAdapter._normalize_dataframe_columns(df.copy())
        sanitized = df.where(pd.notna(df), None)
        return sanitized.to_dict(orient="records")

    @staticmethod
    def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
        normalized_names: list[str] = []
        seen: dict[str, int] = {}
        for column in df.columns:
            base = str(column).strip()
            if not base:
                base = "unnamed"
            count = seen.get(base, 0)
            seen[base] = count + 1
            normalized_names.append(base if count == 0 else f"{base}__dup{count}")
        df.columns = normalized_names
        return df

    def _detect_nested_columns(self, df: pd.DataFrame) -> list[str]:
        nested = []
        for column in df.columns:
            sample_values = df[column].dropna().head(8).tolist()
            if not sample_values:
                continue
            if any(isinstance(value, (dict, list)) for value in sample_values):
                nested.append(str(column))
                continue
            if any(
                isinstance(value, str) and _is_json_like_text(value)
                for value in sample_values
            ):
                nested.append(str(column))
        return nested

    def _expand_structured_records(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        flat_rows: list[dict[str, Any]] = []
        object_paths: dict[str, set[str]] = {}
        list_handling: dict[str, dict[str, Any]] = {}
        nested_source_columns = self._detect_nested_columns(df)

        for row_index, record in enumerate(df.to_dict("records")):
            flat_record: dict[str, Any] = {}
            for key, value in record.items():
                self._flatten_value(
                    flat_record,
                    prefix=str(key),
                    value=value,
                    object_paths=object_paths,
                    list_handling=list_handling,
                )
            flat_rows.append(flat_record)

        flat_df = pd.DataFrame(flat_rows)
        merged_df = df.reset_index(drop=True).copy()
        if nested_source_columns:
            merged_df = merged_df.drop(columns=nested_source_columns, errors="ignore")

        for column in flat_df.columns:
            merged_df[column] = flat_df[column]

        return merged_df, {
            "record_count": int(len(merged_df)),
            "flattened_field_count": int(len(flat_df.columns)),
            "nested_source_columns": nested_source_columns,
            "truncated_list_fields": sorted(
                key for key, value in list_handling.items() if value.get("truncated_items", 0) > 0
            ),
        }

    def _flatten_value(
        self,
        out: dict[str, Any],
        prefix: str,
        value: Any,
        *,
        object_paths: dict[str, set[str]],
        list_handling: dict[str, dict[str, Any]],
    ) -> None:
        value = _maybe_parse_value(value)
        if isinstance(value, dict):
            if not value:
                out[prefix] = {}
                return
            for key, item in value.items():
                child_prefix = f"{prefix}_{key}" if prefix else str(key)
                object_paths.setdefault(prefix, set()).add(child_prefix)
                self._flatten_value(
                    out,
                    child_prefix,
                    item,
                    object_paths=object_paths,
                    list_handling=list_handling,
                )
            return

        if isinstance(value, list):
            if not value:
                out[f"{prefix}_len"] = 0
                list_handling[prefix] = {
                    "strategy": "indexed",
                    "reason": "empty_list",
                    "generated_fields": [f"{prefix}_len"],
                    "max_observed_length": 0,
                    "truncated_items": 0,
                }
                return
            if all(isinstance(item, dict) for item in value):
                generated_fields = []
                truncated_items = max(len(value) - self.max_list_items, 0)
                for idx, item in enumerate(value[: self.max_list_items]):
                    indexed_prefix = f"{prefix}_{idx}"
                    generated_fields.append(indexed_prefix)
                    self._flatten_value(
                        out,
                        indexed_prefix,
                        item,
                        object_paths=object_paths,
                        list_handling=list_handling,
                    )
                out[f"{prefix}_len"] = len(value)
                generated_fields.append(f"{prefix}_len")
                list_handling[prefix] = {
                    "strategy": "indexed",
                    "reason": "list_of_objects",
                    "generated_fields": generated_fields,
                    "max_observed_length": len(value),
                    "truncated_items": truncated_items,
                }
                return
            if all(not isinstance(item, (dict, list)) for item in value):
                generated_fields = []
                truncated_items = max(len(value) - self.max_list_items, 0)
                for idx, item in enumerate(value[: self.max_list_items]):
                    out[f"{prefix}_{idx}"] = item
                    generated_fields.append(f"{prefix}_{idx}")
                out[f"{prefix}_len"] = len(value)
                generated_fields.append(f"{prefix}_len")
                list_handling[prefix] = {
                    "strategy": "indexed",
                    "reason": "list_of_scalars",
                    "generated_fields": generated_fields,
                    "max_observed_length": len(value),
                    "truncated_items": truncated_items,
                }
                return
            out[prefix] = json.dumps(value, ensure_ascii=False)
            list_handling[prefix] = {
                "strategy": "preserved_json_string",
                "reason": "mixed_nested_list",
                "generated_fields": [prefix],
                "max_observed_length": len(value),
                "truncated_items": 0,
            }
            return

        out[prefix] = value

    def _apply_field_mapping(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        normalized = df.copy()
        mappings: list[dict[str, Any]] = []
        alias_lookup = self._build_alias_lookup()
        chosen_sources: set[str] = set()
        unresolved_candidates: list[dict[str, Any]] = []

        for column in list(normalized.columns):
            if column in chosen_sources:
                continue
            target, confidence, reason = self._infer_target_column(column, normalized[column], alias_lookup)
            if not target:
                unresolved_candidates.append(
                    self._build_mapping_review_item(
                        source_column=str(column),
                        series=normalized[column],
                        deterministic_target=None,
                        deterministic_confidence=0.0,
                        deterministic_reason="no mapping",
                    )
                )
                continue
            if target == column:
                continue
            if confidence < self.field_mapping_agent_min_confidence:
                unresolved_candidates.append(
                    self._build_mapping_review_item(
                        source_column=str(column),
                        series=normalized[column],
                        deterministic_target=target,
                        deterministic_confidence=confidence,
                        deterministic_reason=reason,
                    )
                )
                continue
            unit_hint = self._infer_unit_hint_from_column_name(column)
            normalized, mapping_record = self._apply_single_mapping(
                normalized,
                source_column=str(column),
                target_column=target,
                confidence=confidence,
                reason=reason,
                source_unit_hint=unit_hint,
            )
            chosen_sources.add(str(mapping_record["target_column"]))
            mappings.append(mapping_record)

        agent_report = self._run_field_mapping_agent(
            df=normalized,
            review_items=unresolved_candidates,
        )
        for item in agent_report.get("mappings", []):
            source_column = str(item["source_column"])
            target_column = str(item["target_column"])
            if (
                target_column == NO_MAPPING_TARGET
                or source_column not in normalized.columns
                or source_column == target_column
                or float(item.get("confidence", 0.0))
                < self.field_mapping_agent_min_confidence
            ):
                continue
            normalized, mapping_record = self._apply_single_mapping(
                normalized,
                source_column=source_column,
                target_column=target_column,
                confidence=float(item.get("confidence", 0.0)),
                reason=f"agent-reviewed: {item.get('reason') or 'accepted'}",
                source_unit_hint=item.get("source_unit_hint"),
            )
            mapping_record["mapping_source"] = "agent"
            mappings.append(mapping_record)

        agent_choices = {
            str(item.get("source_column")): item
            for item in agent_report.get("mappings", [])
            if isinstance(item, dict)
        }
        for item in unresolved_candidates:
            source_column = str(item["source_column"])
            if source_column not in normalized.columns:
                continue
            agent_choice = agent_choices.get(source_column)
            if (
                agent_choice
                and str(agent_choice.get("target_column")) == NO_MAPPING_TARGET
            ):
                continue
            if (
                agent_choice
                and str(agent_choice.get("target_column") or "").strip()
                and str(agent_choice.get("target_column")) != NO_MAPPING_TARGET
                and float(agent_choice.get("confidence", 0.0))
                >= self.field_mapping_agent_min_confidence
            ):
                continue
            deterministic_target = item.get("deterministic_candidate")
            if not deterministic_target or str(deterministic_target) == source_column:
                continue
            normalized, mapping_record = self._apply_single_mapping(
                normalized,
                source_column=source_column,
                target_column=str(deterministic_target),
                confidence=float(item.get("deterministic_confidence", 0.0)),
                reason=(
                    "deterministic fallback after agent review: "
                    f"{item.get('deterministic_reason') or 'heuristic candidate'}"
                ),
                source_unit_hint=item.get("source_unit_hint"),
            )
            mappings.append(mapping_record)

        mapped_targets = {str(item["target_column"]) for item in mappings}
        mapped_sources = {str(item["source_column"]) for item in mappings}

        passthrough_columns = [
            str(column)
            for column in normalized.columns
            if str(column) not in mapped_targets
            and str(column) not in mapped_sources
        ]

        unresolved_items = []
        for item in unresolved_candidates:
            source_column = str(item["source_column"])
            agent_choice = agent_choices.get(source_column)
            final_target = None
            if (
                agent_choice
                and str(agent_choice.get("target_column")) != NO_MAPPING_TARGET
                and float(agent_choice.get("confidence", 0.0))
                >= self.field_mapping_agent_min_confidence
            ):
                final_target = str(agent_choice.get("target_column"))
            elif item.get("deterministic_candidate"):
                final_target = str(item["deterministic_candidate"])
            unresolved_items.append(
                {
                    "source_column": source_column,
                    "deterministic_candidate": item.get("deterministic_candidate"),
                    "deterministic_confidence": item.get("deterministic_confidence"),
                    "agent_target": (
                        None
                        if not agent_choice
                        or str(agent_choice.get("target_column")) == NO_MAPPING_TARGET
                        else str(agent_choice.get("target_column"))
                    ),
                    "agent_confidence": (
                        None if not agent_choice else agent_choice.get("confidence")
                    ),
                    "final_target": final_target,
                    "left_unmapped": source_column in passthrough_columns,
                    "sample_values": item.get("sample_values", []),
                }
            )

        unresolved_review_items = [
            item for item in unresolved_items if item.get("left_unmapped")
        ]
        agent_mapping_count = sum(
            1 for item in mappings if str(item.get("mapping_source")) == "agent"
        )

        return normalized, {
            "mappings": mappings,
            "passthrough_columns": passthrough_columns,
            "unresolved_review_columns": [
                str(item.get("source_column")) for item in unresolved_review_items
            ],
            "summary": {
                "mapping_count": len(mappings),
                "agent_mapping_count": agent_mapping_count,
                "unresolved_column_count": len(unresolved_review_items),
                "passthrough_column_count": len(passthrough_columns),
            },
            "agent_review": {
                "enabled_by_default": self.enable_field_mapping_agent,
                "status": agent_report.get("status"),
                "warnings": list(agent_report.get("warnings", [])),
                "review_items": unresolved_items,
            },
        }

    def _build_alias_lookup(self) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = {}
        for target, aliases in self.STANDARD_ALIASES.items():
            lookup[target] = sorted(
                {_canonical_name(target), *(_canonical_name(alias) for alias in aliases)},
                key=len,
                reverse=True,
            )
        return lookup

    def _build_mapping_review_item(
        self,
        *,
        source_column: str,
        series: pd.Series,
        deterministic_target: str | None,
        deterministic_confidence: float,
        deterministic_reason: str,
    ) -> dict[str, Any]:
        non_null = series.dropna().head(5).tolist()
        sample_values = [
            value if not isinstance(value, str) else value[:80]
            for value in non_null
        ]
        return {
            "source_column": source_column,
            "dtype": str(series.dtype),
            "sample_values": sample_values,
            "deterministic_candidate": deterministic_target,
            "deterministic_confidence": deterministic_confidence,
            "deterministic_reason": deterministic_reason,
            "source_unit_hint": self._infer_unit_hint_from_column_name(source_column),
        }

    def _run_field_mapping_agent(
        self,
        *,
        df: pd.DataFrame,
        review_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.enable_field_mapping_agent:
            return {
                "status": "disabled",
                "mappings": [],
                "warnings": [],
            }
        allowed_targets = sorted(self.STANDARD_ALIASES.keys()) + sorted(
            {
                str(column)
                for column in df.columns
                if any(
                    str(column).startswith(prefix)
                    for prefix in (
                        "coord_",
                        "abc_",
                        "angle_",
                        "lattice_mat_",
                        "atom_element_",
                        "prop_",
                    )
                )
            }
        )
        try:
            return self.field_mapping_agent.propose_mappings(
                review_items=review_items,
                allowed_targets=allowed_targets,
            )
        except Exception as exc:
            return {
                "status": "agent_failed",
                "mappings": [],
                "warnings": [f"Field mapping agent failed: {exc}"],
            }

    def _apply_single_mapping(
        self,
        df: pd.DataFrame,
        *,
        source_column: str,
        target_column: str,
        confidence: float,
        reason: str,
        source_unit_hint: str | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        normalized = df.copy()
        merged_into_existing_target = False
        if target_column in normalized.columns and target_column != source_column:
            left = normalized[target_column]
            right = normalized[source_column]
            if left.isna().all():
                normalized[target_column] = right
            else:
                merged = left.copy()
                left_missing = merged.isna()
                merged.loc[left_missing] = right.loc[left_missing]
                normalized[target_column] = merged
            normalized = normalized.drop(columns=[source_column])
            merged_into_existing_target = True
        else:
            normalized = normalized.rename(columns={source_column: target_column})

        return normalized, {
            "source_column": source_column,
            "target_column": target_column,
            "confidence": confidence,
            "source_unit_hint": source_unit_hint,
            "merged_into_existing_target": merged_into_existing_target,
            "mapping_source": "deterministic",
        }

    def _infer_target_column(
        self,
        column: str,
        series: pd.Series,
        alias_lookup: dict[str, list[str]],
    ) -> tuple[str | None, float, str]:
        canonical = _canonical_name(column)

        if canonical in {"sampleid", "materialid", "materialid", "mpid", "id"}:
            return "source_sample_id", 0.98, "identifier-like source column"

        for pattern, builder in self.PATTERN_MAPPERS:
            match = pattern.match(canonical)
            if match:
                return builder(match), 0.95, "pattern match"

        for target, aliases in alias_lookup.items():
            if canonical == _canonical_name(target):
                return target, 1.0, "exact standard name"
            for alias in aliases:
                if canonical == alias:
                    return target, 0.95, f"alias match: {alias}"
                if len(alias) >= 4 and alias in canonical:
                    return target, 0.75, f"substring alias match: {alias}"

        numeric_target = self._infer_numeric_semantic_target(column, series)
        if numeric_target:
            return numeric_target[0], numeric_target[1], numeric_target[2]

        return None, 0.0, "no mapping"

    def _infer_numeric_semantic_target(
        self,
        column: str,
        series: pd.Series,
    ) -> tuple[str, float, str] | None:
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = numeric.dropna()
        if non_null.empty:
            return None

        canonical = _canonical_name(column)
        tokens = _semantic_tokens(column)
        unit_hint = self._infer_unit_hint_from_column_name(column)
        median = float(non_null.median())
        min_value = float(non_null.min())
        max_value = float(non_null.max())

        gap_like = (
            "gap" in tokens
            or "bandgap" in canonical
            or "electronicgap" in canonical
            or "pbebandgap" in canonical
            or "egap" in canonical
        )
        formation_like = (
            any(token in tokens for token in ("form", "formation", "eform"))
            or "formationenergy" in canonical
        )
        per_atom_like = (
            "atom" in tokens
            or "peratom" in canonical
            or "atom" in canonical
        )
        hull_like = (
            "hull" in tokens
            or "abovehull" in canonical
            or "ehull" in canonical
        )
        spacegroup_like = (
            "spacegroup" in tokens
            or "spg" in tokens
            or "sg" in tokens
            or "spacegroup" in canonical
        )

        if gap_like:
            if unit_hint == "meV" and 0 <= median <= 20000:
                return (
                    "band_gap",
                    0.62,
                    f"gap-like meV series [{min_value}, {max_value}]",
                )
            if 0 <= median <= 20:
                return (
                    "band_gap",
                    0.62,
                    f"gap-like numeric series [{min_value}, {max_value}]",
                )
        if formation_like and per_atom_like:
            return "formation_energy_per_atom", 0.6, "formation-energy-like name"
        if hull_like:
            return "e_above_hull", 0.6, "hull-like name"
        if spacegroup_like and 1 <= median <= 230:
            return "spacegroup_number", 0.7, "space-group-like integer range"

        return None

    @staticmethod
    def _infer_unit_hint_from_column_name(column: str) -> str | None:
        canonical = _canonical_name(column)
        suffix_hints = {
            "mevperatom": "meV/atom",
            "evperatom": "eV/atom",
            "mevatom": "meV/atom",
            "evatom": "eV/atom",
            "mevperformula": "meV/atom",
            "evperformula": "eV/atom",
            "mev": "meV",
            "mevs": "meV",
            "ev": "eV",
            "evs": "eV",
            "angstrom": "angstrom",
            "angstroms": "angstrom",
            "ang": "angstrom",
            "degree": "degree",
            "degrees": "degree",
            "deg": "degree",
            "degs": "degree",
        }
        for suffix, unit in sorted(suffix_hints.items(), key=lambda item: len(item[0]), reverse=True):
            if canonical.endswith(suffix):
                return unit
        return None

    def _normalize_units(
        self,
        df: pd.DataFrame,
        mapping_manifest: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        normalized = df.copy()
        operations: list[dict[str, Any]] = []
        warnings: list[str] = []

        mapping_by_target: dict[str, list[dict[str, Any]]] = {}
        for item in mapping_manifest.get("mappings", []):
            mapping_by_target.setdefault(str(item.get("target_column")), []).append(item)

        for target_field, config in self.UNIT_CONVERSIONS.items():
            if target_field not in normalized.columns:
                continue

            source_hints = [
                str(item.get("source_unit_hint"))
                for item in mapping_by_target.get(target_field, [])
                if item.get("source_unit_hint")
            ]
            unique_hints = sorted(set(source_hints))
            unit_hint = unique_hints[0] if unique_hints else None
            standard_unit = str(config["standard_unit"])
            conversions = config["conversions"]

            if len(unique_hints) > 1:
                warnings.append(
                    f"Field {target_field} received mixed unit hints {unique_hints}; kept merged values unchanged."
                )
                operations.append(
                    {
                        "field": target_field,
                        "status": "mixed_source_unit_hints",
                        "source_unit": unique_hints,
                        "target_unit": standard_unit,
                    }
                )
                continue

            if unit_hint is None:
                operations.append(
                    {
                        "field": target_field,
                        "status": "assumed_standard_unit",
                        "source_unit": None,
                        "target_unit": standard_unit,
                    }
                )
                continue

            converter = conversions.get(unit_hint)
            if converter is None:
                warnings.append(
                    f"Field {target_field} has unsupported unit hint {unit_hint}; kept original values."
                )
                operations.append(
                    {
                        "field": target_field,
                        "status": "unsupported_unit_hint",
                        "source_unit": unit_hint,
                        "target_unit": standard_unit,
                    }
                )
                continue

            series = pd.to_numeric(normalized[target_field], errors="coerce")
            normalized[target_field] = converter(series)
            operations.append(
                {
                    "field": target_field,
                    "status": "converted" if unit_hint != standard_unit else "already_standard",
                    "source_unit": unit_hint,
                    "target_unit": standard_unit,
                }
            )

        return normalized, {
            "operation_count": len(operations),
            "operations": operations,
            "warnings": warnings,
        }

    def _analyze_rule_coverage(self, df: pd.DataFrame) -> dict[str, Any]:
        available_fields = [
            str(column)
            for column in df.columns
            if df[column].notna().any()
        ]
        available_lower = [item.lower() for item in available_fields]

        known_standard_fields = set(self.STANDARD_ALIASES.keys())
        known_standard_fields.update(
            column
            for column in available_fields
            if any(
                column.startswith(prefix)
                for prefix in ("coord_", "abc_", "angle_", "lattice_mat_", "atom_element_", "prop_")
            )
        )

        coverage: dict[str, float] = {}
        requirements: dict[str, Any] = {}
        recommended_enable: list[str] = []
        recommended_disable: list[str] = []
        manual_review: list[str] = []
        warnings: list[str] = []
        decision_reasons: dict[str, str] = {}

        for pack in build_rule_registry():
            required_tokens = list(pack.required_feature_substrings)
            matched_tokens = [
                token
                for token in required_tokens
                if any(token in field for field in available_lower)
            ]
            match_count = len(matched_tokens)
            denominator = max(len(required_tokens), 1)
            pack_coverage = match_count / denominator if required_tokens else 1.0
            coverage[pack.pack_id] = round(pack_coverage, 4)
            requirements[pack.pack_id] = {
                "required_feature_substrings": required_tokens,
                "required_hints": list(pack.required_hints),
                "minimum_feature_matches": pack.min_feature_matches,
                "minimum_coverage": round(pack.min_feature_matches / denominator, 4) if required_tokens else 0.0,
                "matched_feature_substrings": matched_tokens,
            }

            if pack.enabled_by_default or match_count >= pack.min_feature_matches:
                recommended_enable.append(pack.pack_id)
                if pack.enabled_by_default:
                    decision_reasons[pack.pack_id] = "enabled_by_default_baseline_pack"
                else:
                    decision_reasons[pack.pack_id] = (
                        f"matched {match_count}/{denominator} required feature substrings"
                    )
            elif match_count > 0:
                manual_review.append(pack.pack_id)
                decision_reasons[pack.pack_id] = (
                    f"partial coverage: matched {match_count}/{denominator} required feature substrings"
                )
                warnings.append(
                    f"Rule pack {pack.pack_id} only partially covered: {match_count}/{denominator}."
                )
            else:
                recommended_disable.append(pack.pack_id)
                decision_reasons[pack.pack_id] = "no required feature substrings matched"

        missing_standard_fields = sorted(known_standard_fields - set(available_fields))
        return {
            "missing_standard_fields": missing_standard_fields,
            "available_field_count": len(available_fields),
            "recommended_rule_packs": {
                "enable": recommended_enable,
                "disable": recommended_disable,
                "manual_review": manual_review,
            },
            "coverage_warnings": warnings,
        }

    def _build_schema_profile(self, df: pd.DataFrame) -> dict[str, Any]:
        columns = []
        for column in df.columns:
            series = df[column]
            sample_values = [
                self._profile_value(value)
                for value in series.dropna().head(5).tolist()
            ]
            columns.append(
                {
                    "name": str(column),
                    "dtype": str(series.dtype),
                    "non_null_count": int(series.notna().sum()),
                    "unique_count": int(self._safe_unique_count(series)),
                    "sample_values": sample_values,
                }
            )

        return {
            "column_count": int(df.shape[1]),
            "row_count": int(df.shape[0]),
            "columns": columns,
        }

    @staticmethod
    def _profile_value(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
            return text[:120]
        if isinstance(value, str):
            return value[:120]
        if pd.isna(value):
            return None
        return value

    @staticmethod
    def _safe_unique_count(series: pd.Series) -> int:
        try:
            return int(series.nunique(dropna=True))
        except TypeError:
            normalized = [
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for value in series.dropna().tolist()
            ]
            return len(set(normalized))
