"""Validation gate for materials rule-feedback candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from project.rule_feedback.contracts import (
    ALLOWED_MATERIAL_SUBDOMAINS,
    ALLOWED_RULE_LAYERS,
    ALLOWED_RULE_STATUS,
    CandidateRule,
    CandidateRuleSet,
)
from project.rule_feedback.replay import replay_candidate_rule


@dataclass(frozen=True)
class CandidateRuleValidation:
    rule_id: str
    accepted_for_review: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    replay_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRuleValidationReport:
    dataset_name: str
    total_rules: int
    accepted_for_review: int
    validations: tuple[CandidateRuleValidation, ...]
    promotion_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_rules": self.total_rules,
            "accepted_for_review": self.accepted_for_review,
            "promotion_allowed": self.promotion_allowed,
            "validations": [item.to_dict() for item in self.validations],
        }


def validate_candidate_rule_set(
    candidate_set: CandidateRuleSet,
    *,
    known_fields: set[str] | None = None,
    low_confidence_sample_ids: set[str] | None = None,
    normal_sample_ids: set[str] | None = None,
    replay_df: Any = None,
    identifier_col: str | None = None,
) -> CandidateRuleValidationReport:
    validations = tuple(
        validate_candidate_rule(
            rule,
            known_fields=known_fields,
            low_confidence_sample_ids=low_confidence_sample_ids,
            normal_sample_ids=normal_sample_ids,
            replay_df=replay_df,
            identifier_col=identifier_col,
        )
        for rule in candidate_set.rules
    )
    accepted = sum(1 for item in validations if item.accepted_for_review)
    return CandidateRuleValidationReport(
        dataset_name=candidate_set.dataset_name,
        total_rules=len(candidate_set.rules),
        accepted_for_review=accepted,
        validations=validations,
        promotion_allowed=False,
    )


def validate_candidate_rule(
    rule: CandidateRule,
    *,
    known_fields: set[str] | None = None,
    low_confidence_sample_ids: set[str] | None = None,
    normal_sample_ids: set[str] | None = None,
    replay_df: Any = None,
    identifier_col: str | None = None,
) -> CandidateRuleValidation:
    errors: list[str] = []
    warnings: list[str] = []

    if not rule.rule_id.strip():
        errors.append("rule_id is required")
    if not rule.rule_id.startswith("candidate.materials."):
        errors.append("rule_id must start with candidate.materials.")
    if rule.layer not in ALLOWED_RULE_LAYERS:
        errors.append(f"layer must be one of {sorted(ALLOWED_RULE_LAYERS)}")
    if rule.status not in ALLOWED_RULE_STATUS:
        errors.append("candidate rules must remain draft")
    if not rule.requires_human_review:
        errors.append("candidate rules must require human review")
    if not isinstance(rule.expression, dict) or not rule.expression:
        errors.append("expression must be a non-empty object")

    subdomains = set(rule.applicability.materials_subdomains)
    if not subdomains:
        errors.append("at least one materials subdomain is required")
    unknown_subdomains = subdomains - ALLOWED_MATERIAL_SUBDOMAINS
    if unknown_subdomains:
        errors.append(
            "unknown materials subdomains: " + ", ".join(sorted(unknown_subdomains))
        )

    required_fields = set(rule.applicability.required_fields)
    field_coverage = None
    if known_fields is not None:
        missing_fields = required_fields - known_fields
        field_coverage = (
            1.0
            if not required_fields
            else (len(required_fields) - len(missing_fields)) / len(required_fields)
        )
        if missing_fields:
            warnings.append(
                "required fields not present in current dataset: "
                + ", ".join(sorted(missing_fields))
            )

    if not rule.evidence.source_layers:
        errors.append("evidence.source_layers is required")
    invalid_layers = set(rule.evidence.source_layers) - ALLOWED_RULE_LAYERS
    if invalid_layers:
        errors.append(
            "evidence.source_layers contains invalid layers: "
            + ", ".join(sorted(invalid_layers))
        )
    if not rule.evidence.rationale.strip():
        warnings.append("evidence rationale is empty")
    if not rule.risk_notes.strip():
        warnings.append("risk_notes is empty")

    replay_metrics = _build_replay_metrics(
        rule,
        known_fields=known_fields,
        field_coverage=field_coverage,
        low_confidence_sample_ids=low_confidence_sample_ids,
        normal_sample_ids=normal_sample_ids,
        replay_df=replay_df,
        identifier_col=identifier_col,
    )

    if replay_metrics.get("false_positive_pressure") == "high":
        warnings.append(
            "supporting samples overlap normal samples; false-positive pressure is high"
        )

    return CandidateRuleValidation(
        rule_id=rule.rule_id,
        accepted_for_review=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        replay_metrics=replay_metrics,
    )


def _build_replay_metrics(
    rule: CandidateRule,
    *,
    known_fields: set[str] | None,
    field_coverage: float | None,
    low_confidence_sample_ids: set[str] | None,
    normal_sample_ids: set[str] | None,
    replay_df: Any,
    identifier_col: str | None,
) -> dict[str, Any]:
    supporting_samples = {str(item) for item in rule.evidence.supporting_samples}
    supporting_features = set(rule.evidence.supporting_features)
    replay_result = replay_candidate_rule(
        rule,
        replay_df,
        identifier_col=identifier_col,
    )
    replay_matches = set(replay_result.matched_sample_ids)

    if known_fields is None:
        supporting_feature_coverage = None
    elif not supporting_features:
        supporting_feature_coverage = 1.0
    else:
        supporting_feature_coverage = (
            len(supporting_features & known_fields) / len(supporting_features)
        )

    low_overlap = (
        len((replay_matches or supporting_samples) & low_confidence_sample_ids)
        if low_confidence_sample_ids is not None
        else None
    )
    normal_overlap = (
        len((replay_matches or supporting_samples) & normal_sample_ids)
        if normal_sample_ids is not None
        else None
    )
    if normal_overlap is None:
        false_positive_pressure = "unknown"
    elif normal_overlap == 0:
        false_positive_pressure = "low"
    elif low_overlap and normal_overlap <= low_overlap:
        false_positive_pressure = "medium"
    else:
        false_positive_pressure = "high"

    return {
        "expression_executed": replay_result.executed,
        "expression_type": replay_result.expression_type,
        "expression_errors": list(replay_result.errors),
        "expression_matched_count": replay_result.matched_count,
        "expression_matched_sample_ids": list(replay_result.matched_sample_ids[:20]),
        "field_coverage": field_coverage,
        "supporting_feature_coverage": supporting_feature_coverage,
        "supporting_sample_count": len(supporting_samples),
        "low_confidence_support_overlap": low_overlap,
        "normal_support_overlap": normal_overlap,
        "false_positive_pressure": false_positive_pressure,
        "note": (
            "safe declarative replay only; candidate rules are not promoted"
        ),
    }
