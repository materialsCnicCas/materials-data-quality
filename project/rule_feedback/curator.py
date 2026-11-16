"""Curator workflow for learned materials rules."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from project.rule_feedback.compiler import (
    OverlayCompileReport,
    compile_approved_rules_to_overlays,
)
from project.rule_feedback.promotion import (
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_DRAFT,
    REVIEW_STATUS_REJECTED,
    PromotionResult,
    review_candidate_rules,
)


@dataclass(frozen=True)
class CandidateRuleReviewItem:
    rule_id: str
    title: str
    layer: str
    rule_type: str
    severity: str
    review_status: str
    active: bool
    materials_subdomains: tuple[str, ...] = field(default_factory=tuple)
    required_fields: tuple[str, ...] = field(default_factory=tuple)
    accepted_for_review: bool | None = None
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    validation_warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRuleSetReview:
    rule_set_path: str
    dataset_name: str
    agent: str
    total_rules: int
    counts_by_status: dict[str, int]
    active_rules_mutated: bool
    items: tuple[CandidateRuleReviewItem, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["items"] = [item.to_dict() for item in self.items]
        return payload


@dataclass(frozen=True)
class CuratorWorkflowResult:
    rule_set_path: str
    promotion: dict[str, Any]
    compile_report: dict[str, Any] | None = None
    active_rules_mutated: bool = False
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_candidate_rule_set(rule_set_path: str | Path) -> CandidateRuleSetReview:
    """Build a compact curator-facing review summary for a learned rule set."""

    path = Path(rule_set_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validations = _validations_by_rule_id(payload.get("validation_report") or {})
    items: list[CandidateRuleReviewItem] = []
    counts = {
        REVIEW_STATUS_DRAFT: 0,
        REVIEW_STATUS_APPROVED: 0,
        REVIEW_STATUS_REJECTED: 0,
    }

    for rule in payload.get("rules") or []:
        lifecycle = rule.get("lifecycle") or {}
        status = str(lifecycle.get("review_status") or REVIEW_STATUS_DRAFT)
        counts[status] = counts.get(status, 0) + 1
        applicability = rule.get("applicability") or {}
        validation = validations.get(str(rule.get("rule_id") or ""), {})
        items.append(
            CandidateRuleReviewItem(
                rule_id=str(rule.get("rule_id") or ""),
                title=str(rule.get("title") or ""),
                layer=str(rule.get("layer") or ""),
                rule_type=str(rule.get("rule_type") or ""),
                severity=str(rule.get("severity") or ""),
                review_status=status,
                active=bool(lifecycle.get("active", False)),
                materials_subdomains=tuple(
                    str(item)
                    for item in applicability.get("materials_subdomains") or ()
                ),
                required_fields=tuple(
                    str(item) for item in applicability.get("required_fields") or ()
                ),
                accepted_for_review=validation.get("accepted_for_review"),
                validation_errors=tuple(
                    str(item) for item in validation.get("errors") or ()
                ),
                validation_warnings=tuple(
                    str(item) for item in validation.get("warnings") or ()
                ),
            )
        )

    lifecycle = payload.get("lifecycle") or {}
    return CandidateRuleSetReview(
        rule_set_path=str(path),
        dataset_name=str(payload.get("dataset_name") or ""),
        agent=str(payload.get("agent") or ""),
        total_rules=len(items),
        counts_by_status=counts,
        active_rules_mutated=bool(lifecycle.get("active_rules_mutated", False)),
        items=tuple(items),
    )


def review_and_optionally_compile(
    rule_set_path: str | Path,
    *,
    approved_rule_ids: set[str] | None = None,
    rejected_rule_ids: set[str] | None = None,
    reviewer: str,
    review_notes: str = "",
    compile_overlays: bool = False,
    output_dir: str | Path | None = None,
) -> CuratorWorkflowResult:
    """Review learned rules and optionally compile approved rules to overlays."""

    promotion = review_candidate_rules(
        rule_set_path,
        approved_rule_ids=approved_rule_ids or set(),
        rejected_rule_ids=rejected_rule_ids or set(),
        reviewer=reviewer,
        review_notes=review_notes,
    )
    compile_report: OverlayCompileReport | None = None
    if compile_overlays:
        compile_report = compile_approved_rules_to_overlays(
            rule_set_path,
            output_dir=output_dir,
        )
    return CuratorWorkflowResult(
        rule_set_path=str(rule_set_path),
        promotion=promotion.to_dict(),
        compile_report=compile_report.to_dict() if compile_report else None,
        active_rules_mutated=False,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Curate learned materials quality rules without activating them."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("rule_set_path")

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("rule_set_path")
    review_parser.add_argument("--approve", action="append", default=[])
    review_parser.add_argument("--reject", action="append", default=[])
    review_parser.add_argument("--reviewer", required=True)
    review_parser.add_argument("--notes", default="")
    review_parser.add_argument("--compile-overlays", action="store_true")
    review_parser.add_argument("--output-dir")

    args = parser.parse_args(argv)
    if args.command == "summary":
        result = summarize_candidate_rule_set(args.rule_set_path).to_dict()
    else:
        result = review_and_optionally_compile(
            args.rule_set_path,
            approved_rule_ids=set(args.approve),
            rejected_rule_ids=set(args.reject),
            reviewer=args.reviewer,
            review_notes=args.notes,
            compile_overlays=args.compile_overlays,
            output_dir=args.output_dir,
        ).to_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _validations_by_rule_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("rule_id") or ""): item
        for item in report.get("validations") or []
    }


if __name__ == "__main__":
    main()
