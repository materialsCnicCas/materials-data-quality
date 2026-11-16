"""Curator decision summaries for learned materials rule candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateRuleDecisionItem:
    rule_id: str
    title: str
    layer: str
    severity: str
    review_status: str
    recommended_action: str
    risk_level: str
    rationale: str
    materials_subdomains: tuple[str, ...] = field(default_factory=tuple)
    required_fields: tuple[str, ...] = field(default_factory=tuple)
    supporting_samples: tuple[str, ...] = field(default_factory=tuple)
    supporting_features: tuple[str, ...] = field(default_factory=tuple)
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    validation_warnings: tuple[str, ...] = field(default_factory=tuple)
    replay_metrics: dict[str, Any] = field(default_factory=dict)
    preflight_traceability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRuleDecisionSummary:
    rule_set_path: str
    dataset_name: str
    agent: str
    total_rules: int
    action_counts: dict[str, int]
    risk_counts: dict[str, int]
    subdomain_counts: dict[str, int]
    layer_counts: dict[str, int]
    items: tuple[CandidateRuleDecisionItem, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["items"] = [item.to_dict() for item in self.items]
        return payload


def build_candidate_rule_decision_summary(
    rule_set_path: str | Path,
    *,
    preflight_probe_path: str | Path | None = None,
) -> CandidateRuleDecisionSummary:
    """Build a one-page curator decision model for candidate materials rules."""

    path = Path(rule_set_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validations = _validations_by_rule_id(payload.get("validation_report") or {})
    preflight = _load_preflight_context(preflight_probe_path)
    items: list[CandidateRuleDecisionItem] = []

    for rule in payload.get("rules") or []:
        rule_id = str(rule.get("rule_id") or "")
        validation = validations.get(rule_id, {})
        lifecycle = rule.get("lifecycle") or {}
        applicability = rule.get("applicability") or {}
        evidence = rule.get("evidence") or {}
        replay_metrics = validation.get("replay_metrics") or {}
        errors = tuple(str(item) for item in validation.get("errors") or ())
        warnings = tuple(str(item) for item in validation.get("warnings") or ())
        materials_subdomains = tuple(
            str(item) for item in applicability.get("materials_subdomains") or ()
        )
        required_fields = tuple(
            str(item) for item in applicability.get("required_fields") or ()
        )
        supporting_features = tuple(
            str(item) for item in evidence.get("supporting_features") or ()
        )
        risk_level = _risk_level(rule, errors, warnings, replay_metrics)
        action = _recommended_action(
            accepted_for_review=validation.get("accepted_for_review"),
            review_status=str(lifecycle.get("review_status") or "draft"),
            risk_level=risk_level,
            errors=errors,
            warnings=warnings,
            replay_metrics=replay_metrics,
        )
        items.append(
            CandidateRuleDecisionItem(
                rule_id=rule_id,
                title=str(rule.get("title") or ""),
                layer=str(rule.get("layer") or ""),
                severity=str(rule.get("severity") or ""),
                review_status=str(lifecycle.get("review_status") or "draft"),
                recommended_action=action,
                risk_level=risk_level,
                rationale=_rationale(action, risk_level, errors, warnings, replay_metrics),
                materials_subdomains=materials_subdomains,
                required_fields=required_fields,
                supporting_samples=tuple(
                    str(item) for item in evidence.get("supporting_samples") or ()
                ),
                supporting_features=supporting_features,
                preflight_traceability=_trace_rule_to_preflight(
                    rule_subdomains=materials_subdomains,
                    rule_fields=required_fields + supporting_features,
                    rule_layer=str(rule.get("layer") or ""),
                    preflight=preflight,
                ),
                validation_errors=errors,
                validation_warnings=warnings,
                replay_metrics=dict(replay_metrics),
            )
        )

    return CandidateRuleDecisionSummary(
        rule_set_path=str(path),
        dataset_name=str(payload.get("dataset_name") or ""),
        agent=str(payload.get("agent") or ""),
        total_rules=len(items),
        action_counts=_count_by(items, "recommended_action"),
        risk_counts=_count_by(items, "risk_level"),
        subdomain_counts=_count_many(items, "materials_subdomains"),
        layer_counts=_count_by(items, "layer"),
        items=tuple(items),
    )


def _validations_by_rule_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("rule_id") or ""): item
        for item in report.get("validations") or []
    }


def _load_preflight_context(preflight_probe_path: str | Path | None) -> dict[str, Any]:
    if not preflight_probe_path:
        return {}
    source = Path(preflight_probe_path)
    if source.is_dir():
        return {
            "dataset_contract": _read_json(source / "dataset_contract.json"),
            "rule_activation_manifest": _read_first_json(
                (
                    source / "reviewed_rule_activation_manifest.json",
                    source / "agent_rule_activation_manifest.json",
                    source / "rule_activation_manifest.json",
                )
            ),
        }
    payload = _read_json(source)
    if not payload:
        return {}
    if "dataset_contract" in payload or "rule_activation_manifest" in payload:
        return {
            "dataset_contract": payload.get("dataset_contract") or {},
            "rule_activation_manifest": (
                payload.get("reviewed_rule_activation_manifest")
                or payload.get("agent_rule_activation_manifest")
                or payload.get("rule_activation_manifest")
                or {}
            ),
        }
    return {"rule_activation_manifest": payload}


def _read_first_json(paths: tuple[Path, ...]) -> dict[str, Any]:
    for path in paths:
        payload = _read_json(path)
        if payload:
            return payload
    return {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _trace_rule_to_preflight(
    *,
    rule_subdomains: tuple[str, ...],
    rule_fields: tuple[str, ...],
    rule_layer: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    contract = preflight.get("dataset_contract") or {}
    manifest = preflight.get("rule_activation_manifest") or {}
    applicability = contract.get("material_applicability") or {}
    decisions = manifest.get("decisions") or []
    rule_subdomain_set = set(rule_subdomains)
    rule_field_set = {field.lower() for field in rule_fields}
    matched_packs: list[dict[str, Any]] = []

    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        pack_id = str(decision.get("rule_pack") or "")
        pack_subdomains = {
            str(item) for item in decision.get("materials_subdomains") or ()
        }
        pack_fields = {
            str(item).lower() for item in decision.get("matched_fields") or ()
        }
        subdomain_overlap = sorted(rule_subdomain_set & pack_subdomains)
        field_overlap = sorted(rule_field_set & pack_fields)
        layer_match = bool(rule_layer and f".{rule_layer}." in pack_id)
        if not (subdomain_overlap or field_overlap or layer_match):
            continue
        matched_packs.append(
            {
                "rule_pack": pack_id,
                "enabled": bool(decision.get("enabled")),
                "subdomain_overlap": subdomain_overlap,
                "field_overlap": field_overlap,
                "layer_match": layer_match,
                "confidence": decision.get("confidence"),
                "activation_criteria": decision.get("activation_criteria") or {},
            }
        )

    return {
        "domain_scope": contract.get("domain_scope"),
        "assumed_materials_dataset": (
            applicability.get("assumed_materials_dataset")
        ),
        "contract_subdomain_overlap": sorted(
            rule_subdomain_set
            & {str(item) for item in applicability.get("subdomains") or ()}
        ),
        "contract_field_overlap": sorted(
            rule_field_set
            & {str(item).lower() for item in applicability.get("matched_fields") or ()}
        ),
        "matched_rule_packs": matched_packs,
        "has_preflight_evidence": bool(contract or manifest),
    }


def _risk_level(
    rule: dict[str, Any],
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    replay_metrics: dict[str, Any],
) -> str:
    if errors:
        return "high"
    false_positive_pressure = replay_metrics.get("false_positive_pressure")
    if false_positive_pressure == "high":
        return "high"
    if warnings or str(rule.get("severity") or "").lower() == "high":
        return "medium"
    return "low"


def _recommended_action(
    *,
    accepted_for_review: bool | None,
    review_status: str,
    risk_level: str,
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    replay_metrics: dict[str, Any],
) -> str:
    if review_status == "approved":
        return "compile_overlay"
    if review_status == "rejected" or errors:
        return "reject"
    if replay_metrics.get("expression_executed") is False:
        return "investigate"
    if accepted_for_review and risk_level == "low":
        return "approve_for_overlay_review"
    if accepted_for_review and risk_level == "medium":
        return "review_evidence"
    if warnings:
        return "review_evidence"
    return "defer"


def _rationale(
    action: str,
    risk_level: str,
    errors: tuple[str, ...],
    warnings: tuple[str, ...],
    replay_metrics: dict[str, Any],
) -> str:
    if errors:
        return "Validation errors block promotion: " + "; ".join(errors[:3])
    if action == "investigate":
        return "Expression was not replayed successfully; inspect fields and expression."
    false_positive = replay_metrics.get("false_positive_pressure", "unknown")
    matched = replay_metrics.get("expression_matched_count", "unknown")
    if warnings:
        return (
            f"{risk_level} risk with false-positive pressure={false_positive}; "
            f"matched={matched}; warnings: {'; '.join(warnings[:2])}"
        )
    return (
        f"{risk_level} risk with false-positive pressure={false_positive}; "
        f"matched={matched}; action={action}"
    )


def _count_by(items: list[CandidateRuleDecisionItem], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(getattr(item, attr) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _count_many(items: list[CandidateRuleDecisionItem], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        values = getattr(item, attr) or ("unknown",)
        for value in values:
            key = str(value or "unknown")
            counts[key] = counts.get(key, 0) + 1
    return counts
