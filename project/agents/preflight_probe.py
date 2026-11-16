"""Pre-detection probe for materials data quality runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from project.agents.contract_builder import (
    ContractBuildInputs,
    DatasetContractBuilder,
)
from project.agents.contracts import PreflightProbeResult
from project.rule_registry.activation import build_activation_manifest
from project.rule_registry.registry import RulePack, build_rule_registry


def build_preflight_probe(
    data_results: dict[str, Any],
    *,
    metadata_results: dict[str, Any] | None = None,
    source_path: str | None = None,
    source_format: str | None = None,
    rule_packs: tuple[RulePack, ...] | None = None,
) -> PreflightProbeResult:
    """Build dataset and rule-activation artifacts before layered detection."""

    metadata = None
    if metadata_results:
        metadata = metadata_results.get("metadata")

    contract = DatasetContractBuilder().build(
        ContractBuildInputs(
            processed_df=data_results["processed_df"],
            identifier_col=data_results["identifier_col"],
            num_cols=list(data_results["num_cols"]),
            source_path=source_path or _source_path_from_results(data_results),
            source_format=source_format,
            metadata=metadata,
            skip_missing_check_cols=set(
                data_results.get("skip_missing_check_cols", set())
            ),
        )
    )
    manifest = build_activation_manifest(
        contract,
        rule_packs=rule_packs or build_rule_registry(),
    )
    return PreflightProbeResult(
        dataset_contract=contract,
        rule_activation_manifest=manifest,
    )


def save_preflight_probe(
    result: PreflightProbeResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Persist probe outputs for agents, audit, and human preflight review."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    contract_path = target / "dataset_contract.json"
    manifest_path = target / "rule_activation_manifest.json"
    bundle_path = target / "preflight_probe.json"
    report_path = target / "materials_preflight_report.md"

    contract_path.write_text(
        json.dumps(result.dataset_contract.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            result.rule_activation_manifest.to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    bundle_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths = {
        "dataset_contract": str(contract_path),
        "rule_activation_manifest": str(manifest_path),
        "preflight_probe": str(bundle_path),
    }
    report_path.write_text(
        render_materials_preflight_report(
            result,
            artifact_paths={**artifact_paths, "materials_preflight_report": str(report_path)},
        ),
        encoding="utf-8",
    )
    artifact_paths["materials_preflight_report"] = str(report_path)
    return artifact_paths


def render_materials_preflight_report(
    result: PreflightProbeResult,
    *,
    artifact_paths: dict[str, str] | None = None,
) -> str:
    """Render a curator-readable report for materials subdomain rule switching."""

    contract = result.dataset_contract
    manifest = result.rule_activation_manifest
    applicability = contract.material_applicability or {}
    subdomains = _format_sequence(applicability.get("subdomains"))
    matched_fields = _format_sequence(applicability.get("matched_fields"))
    assumed_materials = _yes_no(applicability.get("assumed_materials_dataset", True))
    has_subdomain_evidence = _yes_no(
        applicability.get("has_material_subdomain_evidence", False)
    )

    lines = [
        "# Materials Preflight Report",
        "",
        "This report assumes the uploaded dataset is a materials database. "
        "It summarizes visible materials subdomain coverage and the material "
        "rule packs selected for this run.",
        "",
        "## Dataset Contract",
        "",
        f"- Dataset: `{_format_scalar(contract.dataset_name)}`",
        f"- Source format: `{_format_scalar(contract.source_format)}`",
        f"- Rows: `{contract.row_count}`",
        f"- Columns: `{contract.column_count}`",
        f"- Identifier column: `{_format_scalar(contract.identifier_column)}`",
        f"- Domain scope: `{_format_scalar(contract.domain_scope)}`",
        f"- Assumed materials dataset: `{assumed_materials}`",
        f"- Visible materials subdomain evidence: `{has_subdomain_evidence}`",
        f"- Materials subdomains: {subdomains}",
        f"- Matched fields: {matched_fields}",
        f"- Applicability reason: {_format_scalar(applicability.get('reason'))}",
        "",
        "## Rule Activation Manifest",
        "",
        f"- Profiler: `{_format_scalar(manifest.profile_agent)}`",
        f"- Enabled rule packs: {_format_sequence(manifest.enabled_rule_packs())}",
        f"- Global hints: {_format_sequence(manifest.global_hints)}",
        "",
        "| Rule pack | State | Subdomains | Confidence | Matched fields | Score / threshold | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for decision in manifest.decisions:
        criteria = decision.activation_criteria or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(decision.rule_pack),
                    "enabled" if decision.enabled else "disabled",
                    _md_cell(_format_sequence(decision.materials_subdomains)),
                    _md_cell(f"{decision.confidence:.2f}"),
                    _md_cell(_format_sequence(decision.matched_fields)),
                    _md_cell(_score_threshold(criteria)),
                    _md_cell(decision.reason),
                ]
            )
            + " |"
        )

    if artifact_paths:
        lines.extend(["", "## Generated Artifacts", ""])
        for key, value in artifact_paths.items():
            lines.append(f"- `{key}`: `{value}`")

    return "\n".join(lines) + "\n"


def _format_sequence(value: Any, *, limit: int = 12) -> str:
    if not value:
        return "`none`"
    if isinstance(value, str):
        return f"`{value}`"
    values = list(value)
    rendered = ", ".join(f"`{_format_scalar(item)}`" for item in values[:limit])
    if len(values) > limit:
        rendered = f"{rendered}, ... (+{len(values) - limit} more)"
    return rendered


def _format_scalar(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _score_threshold(criteria: dict[str, Any]) -> str:
    score = criteria.get("matched_feature_score", 0)
    threshold = criteria.get("min_feature_matches", 0)
    return f"{score} / {threshold}"


def _md_cell(value: Any) -> str:
    text = _format_scalar(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _source_path_from_results(data_results: dict[str, Any]) -> str | None:
    processing_info = data_results.get("processing_info") or {}
    source = processing_info.get("source") or {}
    return source.get("selected_file") or source.get("requested_path") or data_results.get(
        "output_file"
    )
