"""Lifecycle storage for agent-proposed materials rules.

Candidate rules are stored separately from active expert rules. Approval here
means "accepted by a human for later compilation", not active in the pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from project.config import RULE_DIR
from project.rule_feedback.contracts import CandidateRuleSet
from project.rule_feedback.validator import CandidateRuleValidationReport


LEARNED_RULE_DIR = RULE_DIR / "learned"
CANDIDATE_RULE_SET_DIR = LEARNED_RULE_DIR / "candidate_rule_sets"
ARCHIVED_RULE_SET_DIR = LEARNED_RULE_DIR / "archived_rule_sets"

REVIEW_STATUS_DRAFT = "draft"
REVIEW_STATUS_APPROVED = "approved"
REVIEW_STATUS_REJECTED = "rejected"
REVIEW_STATUS_PROMOTED = "promoted"
REVIEW_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class StoredCandidateRuleSet:
    dataset_name: str
    agent: str
    created_at: str
    rules: list[dict[str, Any]]
    validation_report: dict[str, Any] | None = None
    notes: str = ""
    lifecycle: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionResult:
    rule_set_path: str
    approved_rule_ids: tuple[str, ...]
    rejected_rule_ids: tuple[str, ...]
    pending_rule_ids: tuple[str, ...]
    active_rules_mutated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def store_candidate_rule_set(
    candidate_set: CandidateRuleSet,
    *,
    validation_report: CandidateRuleValidationReport | None = None,
    source_context: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    created_at: datetime | None = None,
) -> str:
    """Persist candidate rules for review without activating them."""

    created_at = created_at or datetime.now()
    target_dir = Path(output_dir) if output_dir is not None else CANDIDATE_RULE_SET_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(candidate_set.dataset_name)
    timestamp = created_at.strftime("%Y%m%d-%H%M%S")
    path = target_dir / f"{slug}-{timestamp}.json"
    payload = _stored_payload(
        candidate_set,
        validation_report=validation_report,
        source_context=source_context,
        created_at=created_at,
    )
    path.write_text(
        json.dumps(payload.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def approve_candidate_rules(
    rule_set_path: str | Path,
    *,
    approved_rule_ids: set[str],
    reviewer: str,
    review_notes: str = "",
) -> PromotionResult:
    """Mark selected candidate rules as approved for later compilation.

    This function intentionally does not modify active SDAD/MDAD/SRD rule
    files. It only updates the learned-rule lifecycle document.
    """

    return review_candidate_rules(
        rule_set_path,
        approved_rule_ids=approved_rule_ids,
        rejected_rule_ids=set(),
        reviewer=reviewer,
        review_notes=review_notes,
    )


def review_candidate_rules(
    rule_set_path: str | Path,
    *,
    approved_rule_ids: set[str] | None = None,
    rejected_rule_ids: set[str] | None = None,
    reviewer: str,
    review_notes: str = "",
) -> PromotionResult:
    """Mark selected candidate rules as approved or rejected.

    Review is a lifecycle update only. It never activates learned rules and
    never mutates active SDAD/MDAD/SRD expert rule files.
    """

    approved_rule_ids = approved_rule_ids or set()
    rejected_rule_ids = rejected_rule_ids or set()
    overlap = approved_rule_ids & rejected_rule_ids
    if overlap:
        raise ValueError(
            "rule ids cannot be both approved and rejected: "
            + ", ".join(sorted(overlap))
        )

    path = Path(rule_set_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload.get("rules") or []
    approved: list[str] = []
    rejected: list[str] = []
    pending: list[str] = []
    reviewed_at = datetime.now().isoformat(timespec="seconds")

    for rule in rules:
        rule_id = str(rule.get("rule_id") or "")
        lifecycle = dict(rule.get("lifecycle") or {})
        if rule_id in approved_rule_ids:
            lifecycle.update(
                {
                    "review_status": REVIEW_STATUS_APPROVED,
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                    "review_notes": review_notes,
                    "active": False,
                }
            )
            approved.append(rule_id)
        elif rule_id in rejected_rule_ids:
            lifecycle.update(
                {
                    "review_status": REVIEW_STATUS_REJECTED,
                    "reviewer": reviewer,
                    "reviewed_at": reviewed_at,
                    "review_notes": review_notes,
                    "active": False,
                }
            )
            rejected.append(rule_id)
        elif lifecycle.get("review_status") == REVIEW_STATUS_REJECTED:
            rejected.append(rule_id)
        elif lifecycle.get("review_status") == REVIEW_STATUS_APPROVED:
            approved.append(rule_id)
        else:
            lifecycle.setdefault("review_status", REVIEW_STATUS_DRAFT)
            lifecycle.setdefault("active", False)
            pending.append(rule_id)
        rule["lifecycle"] = lifecycle

    payload.setdefault("lifecycle", {})
    payload["lifecycle"].update(
        {
            "last_reviewed_at": reviewed_at,
            "last_reviewer": reviewer,
            "active_rules_mutated": False,
        }
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PromotionResult(
        rule_set_path=str(path),
        approved_rule_ids=tuple(approved),
        rejected_rule_ids=tuple(rejected),
        pending_rule_ids=tuple(pending),
        active_rules_mutated=False,
    )


def update_candidate_rule_lifecycle_after_apply(
    rule_set_path: str | Path,
    *,
    applied_rule_ids: set[str],
    blocked_rule_ids: set[str],
    apply_report_path: str | Path,
    reviewer: str,
) -> str:
    """Record final apply state and archive fully promoted rule sets."""

    path = Path(rule_set_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    applied_rule_ids = set(applied_rule_ids)
    blocked_rule_ids = set(blocked_rule_ids)
    updated_at = datetime.now().isoformat(timespec="seconds")
    apply_report = _load_json(Path(apply_report_path))
    blocked_reasons = apply_report.get("blocked_reasons") or []
    diff_by_rule_id = _diffs_by_rule_id(apply_report)

    for rule in payload.get("rules") or []:
        rule_id = str(rule.get("rule_id") or "")
        lifecycle = dict(rule.get("lifecycle") or {})
        if rule_id in applied_rule_ids:
            diff = diff_by_rule_id.get(rule_id, {})
            lifecycle.update(
                {
                    "review_status": REVIEW_STATUS_PROMOTED,
                    "active": True,
                    "promoted_at": updated_at,
                    "promoted_by": reviewer,
                    "promoted_reason": _promoted_reason(diff),
                    "already_covered_by_active_rule": (
                        diff.get("action") == "duplicate"
                    ),
                    "target_rule_id": diff.get("target_rule_id"),
                    "target_file": diff.get("target_file"),
                    "apply_report_path": str(apply_report_path),
                }
            )
        elif rule_id in blocked_rule_ids:
            diff = diff_by_rule_id.get(rule_id, {})
            lifecycle.update(
                {
                    "review_status": REVIEW_STATUS_BLOCKED,
                    "active": False,
                    "blocked_at": updated_at,
                    "blocked_reason": _blocked_reason(blocked_reasons, diff),
                    "blocked_reasons": blocked_reasons,
                    "blocked_action": diff.get("action"),
                    "blocked_target_rule_id": diff.get("target_rule_id"),
                    "blocked_target_file": diff.get("target_file"),
                    "apply_report_path": str(apply_report_path),
                }
            )
        rule["lifecycle"] = lifecycle

    lifecycle = payload.setdefault("lifecycle", {})
    lifecycle.update(
        {
            "last_apply_checked_at": updated_at,
            "last_reviewer": reviewer,
            "active_rules_mutated": bool(applied_rule_ids),
            "applied_rule_ids": sorted(applied_rule_ids),
            "blocked_rule_ids": sorted(blocked_rule_ids),
            "apply_report_path": str(apply_report_path),
        }
    )

    if _all_rules_promoted(payload):
        archive_dir = _archive_dir_for(path)
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / path.name
        payload["lifecycle"]["storage_status"] = REVIEW_STATUS_PROMOTED
        payload["lifecycle"]["archived_from"] = str(path)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        path.unlink()
        return str(target)

    payload["lifecycle"]["storage_status"] = (
        REVIEW_STATUS_BLOCKED if blocked_rule_ids else REVIEW_STATUS_APPROVED
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def _stored_payload(
    candidate_set: CandidateRuleSet,
    *,
    validation_report: CandidateRuleValidationReport | None,
    source_context: dict[str, Any] | None,
    created_at: datetime,
) -> StoredCandidateRuleSet:
    rules = []
    for rule in candidate_set.rules:
        payload = rule.to_dict()
        payload["lifecycle"] = {
            "review_status": REVIEW_STATUS_DRAFT,
            "active": False,
            "created_at": created_at.isoformat(timespec="seconds"),
        }
        rules.append(payload)

    return StoredCandidateRuleSet(
        dataset_name=candidate_set.dataset_name,
        agent=candidate_set.agent,
        created_at=created_at.isoformat(timespec="seconds"),
        rules=rules,
        validation_report=(
            validation_report.to_dict() if validation_report is not None else None
        ),
        notes=candidate_set.notes,
        lifecycle={
            "storage_status": REVIEW_STATUS_DRAFT,
            "active_rules_mutated": False,
            "source_context": source_context or {},
        },
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "materials-dataset"


def _all_rules_promoted(payload: dict[str, Any]) -> bool:
    rules = payload.get("rules") or []
    return bool(rules) and all(
        (rule.get("lifecycle") or {}).get("review_status")
        == REVIEW_STATUS_PROMOTED
        for rule in rules
    )


def _archive_dir_for(path: Path) -> Path:
    if path.parent.resolve() == CANDIDATE_RULE_SET_DIR.resolve():
        return ARCHIVED_RULE_SET_DIR
    if path.parent.name in {"candidate_rule_sets", "candidate_sets"}:
        return path.parent.parent / "archived_rule_sets"
    return path.parent / "archived_rule_sets"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _diffs_by_rule_id(apply_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dry_run = apply_report.get("dry_run_report") or {}
    diff_report = dry_run.get("diff_report") or {}
    return {
        str(item.get("learned_rule_id") or ""): item
        for item in diff_report.get("diffs") or []
        if isinstance(item, dict) and item.get("learned_rule_id")
    }


def _blocked_reason(
    blocked_reasons: list[Any],
    diff: dict[str, Any],
) -> str:
    reason = str(diff.get("reason") or "").strip()
    if reason:
        return reason
    if blocked_reasons:
        return "; ".join(str(item) for item in blocked_reasons)
    return "guarded apply did not allow activation"


def _promoted_reason(diff: dict[str, Any]) -> str:
    if diff.get("action") == "duplicate":
        return "equivalent active expert rule already exists"
    return "guarded apply wrote rule to active expert library"
