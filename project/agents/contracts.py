"""Structured contracts for agent-assisted data quality workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldProfile:
    name: str
    logical_type: str
    null_fraction: float
    distinct_fraction: float
    sample_values: tuple[str, ...] = ()
    semantic_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetContract:
    dataset_name: str
    row_count: int
    column_count: int
    source_format: str
    identifier_column: str | None
    field_profiles: tuple[FieldProfile, ...]
    source_meta: dict[str, Any] = field(default_factory=dict)
    semantic_hints: tuple[str, ...] = ()
    domain_scope: str = "materials_database"
    material_applicability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["field_profiles"] = [asdict(item) for item in self.field_profiles]
        return payload


@dataclass(frozen=True)
class RuleActivationDecision:
    rule_pack: str
    enabled: bool
    reason: str
    confidence: float
    materials_subdomains: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()
    activation_criteria: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleActivationManifest:
    dataset_name: str
    profile_agent: str
    decisions: tuple[RuleActivationDecision, ...]
    global_hints: tuple[str, ...] = ()

    def enabled_rule_packs(self) -> list[str]:
        return [item.rule_pack for item in self.decisions if item.enabled]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decisions"] = [asdict(item) for item in self.decisions]
        return payload


@dataclass(frozen=True)
class PreflightProbeResult:
    dataset_contract: DatasetContract
    rule_activation_manifest: RuleActivationManifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_contract": self.dataset_contract.to_dict(),
            "rule_activation_manifest": self.rule_activation_manifest.to_dict(),
        }
