"""Compile approved learned rules into inactive overlay files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from project.rule_feedback.promotion import LEARNED_RULE_DIR, REVIEW_STATUS_APPROVED


OVERLAY_DIR = LEARNED_RULE_DIR / "overlays"


@dataclass(frozen=True)
class OverlayCompileReport:
    source_rule_set_path: str
    output_dir: str
    compiled_rule_ids: tuple[str, ...]
    skipped_rule_ids: tuple[str, ...]
    overlay_files: dict[str, str]
    rollback_manifest: str
    active_rules_mutated: bool = False
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_approved_rules_to_overlays(
    rule_set_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> OverlayCompileReport:
    """Compile approved learned rules into layer-specific inactive overlays."""

    source_path = Path(rule_set_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    target_dir = Path(output_dir) if output_dir is not None else OVERLAY_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    overlays = _empty_overlays(payload)
    compiled: list[str] = []
    skipped: list[str] = []

    for rule in payload.get("rules") or []:
        rule_id = str(rule.get("rule_id") or "")
        lifecycle = rule.get("lifecycle") or {}
        if lifecycle.get("review_status") != REVIEW_STATUS_APPROVED:
            skipped.append(rule_id)
            continue

        compiled_rule = _compile_rule(rule)
        if compiled_rule is None:
            skipped.append(rule_id)
            continue
        layer = str(rule.get("layer") or "").lower()
        overlays[layer]["learned_rules"][rule_id] = compiled_rule
        compiled.append(rule_id)

    overlay_files: dict[str, str] = {}
    for layer, overlay in overlays.items():
        if overlay["learned_rules"]:
            path = target_dir / f"{layer}_learned_overlay.json"
            path.write_text(
                json.dumps(overlay, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            overlay_files[layer] = str(path)

    rollback_path = target_dir / "rollback_manifest.json"
    rollback = {
        "source_rule_set_path": str(source_path),
        "overlay_files": overlay_files,
        "active_rules_mutated": False,
        "rollback_action": "delete overlay files; no active expert rules were changed",
    }
    rollback_path.write_text(
        json.dumps(rollback, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = OverlayCompileReport(
        source_rule_set_path=str(source_path),
        output_dir=str(target_dir),
        compiled_rule_ids=tuple(compiled),
        skipped_rule_ids=tuple(skipped),
        overlay_files=overlay_files,
        rollback_manifest=str(rollback_path),
        active_rules_mutated=False,
    )
    report_path = target_dir / "compile_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _empty_overlays(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base = {
        "dataset_name": payload.get("dataset_name"),
        "source_agent": payload.get("agent"),
        "overlay_status": "inactive",
        "active_rules_mutated": False,
    }
    return {
        "sdad": dict(base, layer="sdad", learned_rules={}),
        "mdad": dict(base, layer="mdad", learned_rules={}),
        "srd": dict(base, layer="srd", learned_rules={}),
    }


def _compile_rule(rule: dict[str, Any]) -> dict[str, Any] | None:
    layer = str(rule.get("layer") or "").lower()
    expression = rule.get("expression") or {}
    if layer == "sdad":
        return _compile_sdad_rule(rule, expression)
    if layer == "mdad":
        return _compile_mdad_rule(rule, expression)
    if layer == "srd":
        return _compile_srd_rule(rule, expression)
    return None


def _compile_sdad_rule(
    rule: dict[str, Any],
    expression: dict[str, Any],
) -> dict[str, Any] | None:
    if expression.get("type") != "field_range":
        return _metadata_rule(rule, expression)
    field = expression.get("field")
    if not field:
        return None
    valid_range = [
        expression.get("min", "-Infinity"),
        expression.get("max", "Infinity"),
    ]
    return {
        "name": field,
        "data_type": "float",
        "valid_range": valid_range,
        "source": "learned_agent_candidate",
        "status": "approved_inactive_overlay",
        "title": rule.get("title"),
        "severity": rule.get("severity"),
        "evidence": rule.get("evidence"),
        "risk_notes": rule.get("risk_notes"),
    }


def _compile_mdad_rule(
    rule: dict[str, Any],
    expression: dict[str, Any],
) -> dict[str, Any] | None:
    if expression.get("type") == "field_comparison":
        left = expression.get("left")
        right = expression.get("right")
        if not left or not right:
            return None
        return {
            "rule_name": rule.get("title"),
            "feature_pair": [left, right],
            "relationship_type": _comparison_to_relationship(
                expression.get("operator")
            ),
            "threshold_conditions": {
                "tolerance": expression.get("tolerance", 0.0)
            },
            "source": "learned_agent_candidate",
            "status": "approved_inactive_overlay",
            "expression": expression,
            "evidence": rule.get("evidence"),
            "risk_notes": rule.get("risk_notes"),
        }
    return _metadata_rule(rule, expression)


def _compile_srd_rule(
    rule: dict[str, Any],
    expression: dict[str, Any],
) -> dict[str, Any]:
    return _metadata_rule(rule, expression)


def _metadata_rule(rule: dict[str, Any], expression: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_name": rule.get("title"),
        "rule_type": rule.get("rule_type"),
        "severity": rule.get("severity"),
        "expression": expression,
        "applicability": rule.get("applicability"),
        "evidence": rule.get("evidence"),
        "expected_effect": rule.get("expected_effect"),
        "risk_notes": rule.get("risk_notes"),
        "source": "learned_agent_candidate",
        "status": "approved_inactive_overlay",
    }


def _comparison_to_relationship(operator: Any) -> str:
    if operator in {"lt", "lte", "gt", "gte"}:
        return "ordered_constraint"
    if operator == "eq":
        return "equality_constraint"
    if operator == "neq":
        return "inequality_constraint"
    return "custom_constraint"
