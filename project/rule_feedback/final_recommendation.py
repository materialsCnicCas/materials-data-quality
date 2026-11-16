"""Final curator recommendations for learned materials rule packages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuleFinalRecommendation:
    rule_id: str
    final_recommendation: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    decision_action: str = ""
    risk_level: str = ""
    overlay_action: str | None = None
    overlay_conflict: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CuratorFinalRecommendation:
    package_dir: str
    overall_recommendation: str
    summary: dict[str, Any]
    rules: tuple[RuleFinalRecommendation, ...] = field(default_factory=tuple)
    active_rules_mutated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_dir": self.package_dir,
            "overall_recommendation": self.overall_recommendation,
            "summary": self.summary,
            "rules": [item.to_dict() for item in self.rules],
            "active_rules_mutated": self.active_rules_mutated,
        }


def build_curator_final_recommendation(
    package_dir: str | Path,
) -> CuratorFinalRecommendation:
    """Combine review artifacts into one package-level curator conclusion."""

    root = Path(package_dir)
    decision_summary = _read_json(root / "candidate_rule_decision_summary.json")
    diff_report = _read_json(root / "overlays" / "overlay_diff_report.json")
    dry_run = _read_json(root / "overlays" / "overlay_apply_dry_run_report.json")
    guarded = _read_json(root / "overlays" / "guarded_apply_report.json")
    diffs_by_rule = {
        str(item.get("learned_rule_id") or ""): item
        for item in diff_report.get("diffs") or []
    }
    rules = tuple(
        _recommend_rule(item, diffs_by_rule.get(str(item.get("rule_id") or "")))
        for item in decision_summary.get("items") or []
    )
    overall = _overall_recommendation(rules, dry_run)
    summary = {
        "total_rules": len(rules),
        "rule_recommendation_counts": _count_recommendations(rules),
        "dry_run_apply_allowed": dry_run.get("apply_allowed"),
        "dry_run_blocked_reasons": dry_run.get("blocked_reasons") or [],
        "overlay_actions": diff_report.get("actions") or {},
        "guarded_apply_confirmed": guarded.get("confirmed", False),
        "guarded_apply_blocked_reasons": guarded.get("blocked_reasons") or [],
    }
    return CuratorFinalRecommendation(
        package_dir=str(root),
        overall_recommendation=overall,
        summary=summary,
        rules=rules,
        active_rules_mutated=bool(guarded.get("active_rules_mutated", False)),
    )


def _recommend_rule(
    item: dict[str, Any],
    diff: dict[str, Any] | None,
) -> RuleFinalRecommendation:
    action = str(item.get("recommended_action") or "")
    risk = str(item.get("risk_level") or "")
    reasons: list[str] = []
    final = "defer"

    if action in {"reject", "investigate"}:
        final = action
        reasons.append(f"decision summary action is {action}")
    elif risk == "high":
        final = "reject"
        reasons.append("decision summary risk is high")
    elif diff and diff.get("conflict"):
        final = "blocked_by_overlay_conflict"
        reasons.append(str(diff.get("reason") or "overlay conflict"))
    elif diff and diff.get("action") == "unsupported":
        final = "blocked_by_unsupported_overlay"
        reasons.append(str(diff.get("reason") or "unsupported overlay action"))
    elif action == "compile_overlay":
        if diff and diff.get("action") in {"add", "duplicate"}:
            final = "ready_for_guarded_apply"
            reasons.append(f"overlay action is {diff.get('action')}")
        else:
            final = "review_overlay"
            reasons.append("approved rule still needs overlay review")
    elif action == "approve_for_overlay_review":
        if diff and diff.get("action") == "add":
            final = "ready_for_approval"
            reasons.append("low-risk candidate with additive overlay")
        elif diff and diff.get("action") == "duplicate":
            final = "already_covered"
            reasons.append("equivalent expert rule or overlay target already exists")
        else:
            final = "review_overlay"
            reasons.append("candidate needs overlay diff review")
    elif action == "review_evidence":
        final = "needs_evidence_review"
        reasons.append("decision summary requests curator evidence review")
    else:
        reasons.append("no promotion-ready action was established")

    traceability = item.get("preflight_traceability") or {}
    if traceability.get("has_preflight_evidence") and not traceability.get(
        "matched_rule_packs"
    ):
        reasons.append("preflight evidence present but no direct rule-pack match")
    elif traceability.get("matched_rule_packs"):
        reasons.append("preflight rule-pack trace is present")

    return RuleFinalRecommendation(
        rule_id=str(item.get("rule_id") or ""),
        final_recommendation=final,
        reasons=tuple(reasons),
        decision_action=action,
        risk_level=risk,
        overlay_action=(str(diff.get("action")) if diff else None),
        overlay_conflict=(bool(diff.get("conflict")) if diff else None),
    )


def _overall_recommendation(
    rules: tuple[RuleFinalRecommendation, ...],
    dry_run: dict[str, Any],
) -> str:
    if not rules:
        return "no_candidate_rules"
    finals = {item.final_recommendation for item in rules}
    if any(item.startswith("blocked") for item in finals):
        return "blocked"
    if "reject" in finals:
        return "reject_some"
    if dry_run.get("apply_allowed") is False:
        return "blocked"
    if finals <= {"ready_for_guarded_apply", "already_covered"}:
        return "ready_for_guarded_apply"
    if finals & {"ready_for_approval", "review_overlay", "needs_evidence_review"}:
        return "curator_review_required"
    return "defer"


def _count_recommendations(
    rules: tuple[RuleFinalRecommendation, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rules:
        counts[item.final_recommendation] = (
            counts.get(item.final_recommendation, 0) + 1
        )
    return counts


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
