"""Contracts for agent-proposed materials quality rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ALLOWED_RULE_LAYERS = {"sdad", "mdad", "srd"}
ALLOWED_RULE_STATUS = {"draft"}
ALLOWED_MATERIAL_SUBDOMAINS = {
    "materials",
    "materials_composition",
    "materials_structure",
    "materials_atomic_geometry",
    "materials_thermodynamic",
    "materials_electronic",
    "materials_experimental",
    "materials_computational",
}


@dataclass(frozen=True)
class RuleApplicability:
    materials_subdomains: tuple[str, ...]
    required_fields: tuple[str, ...] = ()
    excluded_conditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleEvidence:
    source_layers: tuple[str, ...]
    supporting_samples: tuple[str, ...] = ()
    supporting_features: tuple[str, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class CandidateRule:
    rule_id: str
    title: str
    layer: str
    status: str
    severity: str
    rule_type: str
    expression: dict[str, Any]
    applicability: RuleApplicability
    evidence: RuleEvidence
    expected_effect: str
    risk_notes: str
    requires_human_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRuleSet:
    dataset_name: str
    agent: str
    rules: tuple[CandidateRule, ...] = field(default_factory=tuple)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rules"] = [rule.to_dict() for rule in self.rules]
        return payload
