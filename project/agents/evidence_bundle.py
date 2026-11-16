"""Evidence bundles for materials anomaly explanation agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ConfidenceSummary:
    count: int
    mean: float | None
    std: float | None
    min: float | None
    median: float | None
    max: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaterialsEvidenceBundle:
    dataset_name: str
    run_mode: str
    dataset_contract: dict[str, Any] | None
    rule_activation_manifest: dict[str, Any] | None
    agent_rule_activation_manifest: dict[str, Any] | None
    layer_summaries: dict[str, Any]
    confidence_summaries: dict[str, ConfidenceSummary]
    quality_memory_summary: dict[str, Any] | None
    low_confidence_samples: list[dict[str, Any]]
    low_confidence_features: list[dict[str, Any]]
    representative_records: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "run_mode": self.run_mode,
            "dataset_contract": self.dataset_contract,
            "rule_activation_manifest": self.rule_activation_manifest,
            "agent_rule_activation_manifest": self.agent_rule_activation_manifest,
            "layer_summaries": self.layer_summaries,
            "confidence_summaries": {
                key: value.to_dict()
                for key, value in self.confidence_summaries.items()
            },
            "quality_memory_summary": self.quality_memory_summary,
            "low_confidence_samples": self.low_confidence_samples,
            "low_confidence_features": self.low_confidence_features,
            "representative_records": self.representative_records,
        }


def build_materials_evidence_bundle(
    pipeline_results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    dataset_name: str | None = None,
    run_mode: str = "expert_validation",
    max_items: int = 20,
) -> MaterialsEvidenceBundle:
    """Build a compact post-run evidence object for explanation agents."""

    data_processing = pipeline_results.get("data_processing", {})
    processed_df = data_processing.get("processed_df")
    inferred_name = dataset_name or _infer_dataset_name(data_processing, output_dir)

    preflight = pipeline_results.get("preflight_probe") or {}
    dkadad = pipeline_results.get("dkadad_v2") or {}
    nine_dimension = pipeline_results.get("nine_dimension") or {}

    low_samples = _read_or_extract_low_samples(
        pipeline_results,
        output_dir=output_dir,
        max_items=max_items,
    )
    low_features = _read_or_extract_low_features(
        pipeline_results,
        output_dir=output_dir,
        max_items=max_items,
    )

    return MaterialsEvidenceBundle(
        dataset_name=inferred_name,
        run_mode=run_mode,
        dataset_contract=preflight.get("dataset_contract"),
        rule_activation_manifest=preflight.get("rule_activation_manifest"),
        agent_rule_activation_manifest=preflight.get("agent_rule_activation_manifest"),
        layer_summaries={
            "nine_dimension": _nine_dimension_summary(nine_dimension),
            "missing_handler": _missing_summary(pipeline_results),
            "sdad": _safe_summary(dkadad.get("sdad_results")),
            "mdad": _safe_summary(dkadad.get("mdad_results")),
            "srd": _safe_summary(dkadad.get("srd_results")),
            "bipartite_walk": _safe_summary(pipeline_results.get("bipartite_walk")),
        },
        confidence_summaries={
            "final_rows": _confidence_summary(dkadad.get("row_conf_df")),
            "final_features": _confidence_summary(dkadad.get("col_conf_df")),
        },
        quality_memory_summary=(
            (pipeline_results.get("quality_memory") or {}).get("summary")
            if pipeline_results.get("quality_memory")
            else None
        ),
        low_confidence_samples=low_samples,
        low_confidence_features=low_features,
        representative_records=_representative_records(
            processed_df,
            low_samples,
            max_items=min(max_items, 10),
        ),
    )


def _infer_dataset_name(
    data_processing: dict[str, Any],
    output_dir: str | Path | None,
) -> str:
    info = data_processing.get("processing_info") or {}
    source_info = info.get("source") or {}
    source = source_info.get("selected_file") or source_info.get("requested_path")
    if source:
        return Path(str(source)).stem
    if output_dir:
        return Path(output_dir).name
    return "materials_dataset"


def _confidence_summary(df: Any) -> ConfidenceSummary:
    if not isinstance(df, pd.DataFrame) or "confidence_score" not in df.columns:
        return ConfidenceSummary(0, None, None, None, None, None)
    scores = pd.to_numeric(df["confidence_score"], errors="coerce").dropna()
    if scores.empty:
        return ConfidenceSummary(0, None, None, None, None, None)
    return ConfidenceSummary(
        count=int(scores.shape[0]),
        mean=float(scores.mean()),
        std=float(scores.std()) if scores.shape[0] > 1 else 0.0,
        min=float(scores.min()),
        median=float(scores.median()),
        max=float(scores.max()),
    )


def _read_or_extract_low_samples(
    pipeline_results: dict[str, Any],
    *,
    output_dir: str | Path | None,
    max_items: int,
) -> list[dict[str, Any]]:
    final_outputs = pipeline_results.get("final_outputs") or {}
    path = final_outputs.get("low_confidence_samples")
    if path:
        rows = _read_csv_records(path, max_items=max_items)
        if rows:
            return rows

    dkadad = pipeline_results.get("dkadad_v2") or {}
    row_conf_df = dkadad.get("row_conf_df")
    if isinstance(row_conf_df, pd.DataFrame):
        return _df_records(
            row_conf_df.sort_values("confidence_score", ascending=True),
            max_items=max_items,
        )
    return []


def _read_or_extract_low_features(
    pipeline_results: dict[str, Any],
    *,
    output_dir: str | Path | None,
    max_items: int,
) -> list[dict[str, Any]]:
    final_outputs = pipeline_results.get("final_outputs") or {}
    path = final_outputs.get("low_confidence_features")
    if path:
        rows = _read_csv_records(path, max_items=max_items)
        if rows:
            return rows

    dkadad = pipeline_results.get("dkadad_v2") or {}
    col_conf_df = dkadad.get("col_conf_df")
    if isinstance(col_conf_df, pd.DataFrame):
        return _df_records(
            col_conf_df.sort_values("confidence_score", ascending=True),
            max_items=max_items,
        )
    return []


def _read_csv_records(path: str | Path, *, max_items: int) -> list[dict[str, Any]]:
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return []
    return _df_records(df, max_items=max_items)


def _df_records(df: pd.DataFrame, *, max_items: int) -> list[dict[str, Any]]:
    records = df.head(max_items).where(pd.notna(df.head(max_items)), None).to_dict("records")
    return [_jsonable_record(record) for record in records]


def _representative_records(
    processed_df: Any,
    low_samples: list[dict[str, Any]],
    *,
    max_items: int,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(processed_df, pd.DataFrame) or processed_df.empty:
        return {"anomalous": [], "normal": []}

    anomalous_indices = [
        int(item["row_index"])
        for item in low_samples
        if "row_index" in item and str(item["row_index"]).lstrip("-").isdigit()
    ][:max_items]
    if anomalous_indices:
        existing_indices = [
            index for index in anomalous_indices
            if index in processed_df.index
        ]
        anomalous = processed_df.loc[existing_indices]
    else:
        anomalous = processed_df.iloc[0:0]

    normal = processed_df.drop(index=anomalous.index, errors="ignore").head(max_items)
    return {
        "anomalous": _df_records(anomalous, max_items=max_items),
        "normal": _df_records(normal, max_items=max_items),
    }


def _nine_dimension_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "column_confidence": _confidence_summary(result.get("col_conf_df")).to_dict(),
        "row_confidence": _confidence_summary(result.get("row_conf_df")).to_dict(),
        "identifier_col": result.get("identifier_col"),
        "output_dir": result.get("output_dir"),
    }


def _missing_summary(pipeline_results: dict[str, Any]) -> dict[str, Any]:
    data_processing = pipeline_results.get("data_processing") or {}
    return {
        "skip_missing_check_cols": sorted(
            data_processing.get("skip_missing_check_cols") or []
        ),
    }


def _safe_summary(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return {
            "rows": int(value.shape[0]),
            "columns": list(value.columns),
        }
    if isinstance(value, dict):
        summary: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, pd.DataFrame):
                summary[key] = {
                    "rows": int(item.shape[0]),
                    "columns": list(item.columns),
                }
            elif isinstance(item, (str, int, float, bool)) or item is None:
                summary[key] = item
            elif isinstance(item, list):
                summary[key] = item[:20]
            elif isinstance(item, dict):
                summary[key] = _safe_summary(item)
        return summary
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def _jsonable_record(record: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            clean[str(key)] = None
        elif hasattr(value, "item"):
            clean[str(key)] = value.item()
        else:
            clean[str(key)] = value
    return clean
