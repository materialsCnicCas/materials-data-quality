"""One-step promotion for reviewed learned materials rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from project.rule_feedback.overlay_apply import CONFIRMATION_TOKEN
from project.rule_feedback.promotion import update_candidate_rule_lifecycle_after_apply
from project.rule_feedback.review_package import build_review_package


@dataclass(frozen=True)
class PromoteResult:
    rule_set_path: str
    output_dir: str
    approved_rule_ids: tuple[str, ...]
    rejected_rule_ids: tuple[str, ...]
    apply_requested: bool
    active_rules_mutated: bool
    review_package: dict[str, Any]
    one_click_semantics: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def promote_candidate_rules(
    rule_set_path: str | Path,
    *,
    approved_rule_ids: set[str] | None = None,
    rejected_rule_ids: set[str] | None = None,
    approved_update_rule_ids: set[str] | None = None,
    approve_all: bool = False,
    reviewer: str,
    review_notes: str = "",
    output_dir: str | Path,
    apply: bool = False,
    preflight_probe_path: str | Path | None = None,
    active_rule_files: dict[str, str | Path] | None = None,
) -> PromoteResult:
    """Approve selected candidates and optionally write them into active rules."""

    approved = set(approved_rule_ids or set())
    if approve_all:
        approved.update(candidate_rule_ids(rule_set_path))
    rejected = set(rejected_rule_ids or set())
    report = build_review_package(
        rule_set_path,
        output_dir=output_dir,
        approved_rule_ids=approved,
        rejected_rule_ids=rejected,
        approved_update_rule_ids=approved_update_rule_ids,
        reviewer=reviewer,
        review_notes=review_notes,
        attempt_apply=apply,
        apply_confirmation=CONFIRMATION_TOKEN if apply else "",
        preflight_probe_path=preflight_probe_path,
        active_rule_files=active_rule_files,
    )
    payload = report.to_dict()
    final_rule_set_path = str(rule_set_path)
    if apply:
        final_rule_set_path = _sync_lifecycle_after_apply(
            rule_set_path,
            review_package=payload,
            selected_rule_ids=approved,
            reviewer=reviewer,
        )
    return PromoteResult(
        rule_set_path=final_rule_set_path,
        output_dir=str(output_dir),
        approved_rule_ids=tuple(sorted(approved)),
        rejected_rule_ids=tuple(sorted(rejected)),
        apply_requested=bool(apply),
        active_rules_mutated=bool(payload.get("active_rules_mutated", False)),
        review_package=payload,
        one_click_semantics=(
            "applied_to_active_rule_library"
            if apply
            else "review_package_only_dry_run"
        ),
    )


def candidate_rule_ids(rule_set_path: str | Path) -> set[str]:
    payload = json.loads(Path(rule_set_path).read_text(encoding="utf-8"))
    return {
        str(rule.get("rule_id"))
        for rule in payload.get("rules") or []
        if rule.get("rule_id")
    }


def _sync_lifecycle_after_apply(
    rule_set_path: str | Path,
    *,
    review_package: dict[str, Any],
    selected_rule_ids: set[str],
    reviewer: str,
) -> str:
    apply_report = review_package.get("guarded_overlay_apply_report_path")
    if not apply_report:
        return str(rule_set_path)
    path = Path(apply_report)
    if not path.exists():
        return str(rule_set_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    applied = set(payload.get("applied_rule_ids") or [])
    skipped = set(payload.get("skipped_rule_ids") or [])
    covered = _duplicate_skipped_rule_ids(payload, skipped)
    finalized = applied | covered
    blocked = set(selected_rule_ids) - finalized
    return update_candidate_rule_lifecycle_after_apply(
        rule_set_path,
        applied_rule_ids=finalized,
        blocked_rule_ids=blocked,
        apply_report_path=path,
        reviewer=reviewer,
    )


def _duplicate_skipped_rule_ids(
    apply_report: dict[str, Any],
    skipped_rule_ids: set[str],
) -> set[str]:
    dry_run = apply_report.get("dry_run_report") or {}
    diff_report = dry_run.get("diff_report") or {}
    return {
        str(item.get("learned_rule_id") or "")
        for item in diff_report.get("diffs") or []
        if item.get("action") == "duplicate"
        and str(item.get("learned_rule_id") or "") in skipped_rule_ids
    }
