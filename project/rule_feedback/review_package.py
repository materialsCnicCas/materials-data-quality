"""End-to-end review package for learned materials rules."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from project.reporting.agent_audit_report import build_agent_audit_markdown
from project.reporting.lifecycle_artifacts import collect_lifecycle_artifact_files
from project.rule_feedback.decision_summary import (
    build_candidate_rule_decision_summary,
)
from project.rule_feedback.final_recommendation import (
    build_curator_final_recommendation,
)
from project.rule_feedback.curator import (
    review_and_optionally_compile,
    summarize_candidate_rule_set,
)
from project.rule_feedback.overlay_apply import apply_overlays_guarded
from project.rule_feedback.overlay_review import (
    diff_inactive_overlays,
    dry_run_apply_overlays,
)
from project.rule_feedback.review_index import build_review_index


@dataclass(frozen=True)
class ReviewPackageReport:
    rule_set_path: str
    output_dir: str
    summary_path: str
    curator_result_path: str
    overlay_dir: str
    overlay_diff_report_path: str
    overlay_apply_dry_run_report_path: str
    guarded_overlay_apply_report_path: str
    audit_report_path: str
    review_index_path: str
    apply_attempted: bool
    active_rules_mutated: bool = False
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_review_package(
    rule_set_path: str | Path,
    *,
    output_dir: str | Path,
    approved_rule_ids: set[str] | None = None,
    rejected_rule_ids: set[str] | None = None,
    approved_update_rule_ids: set[str] | None = None,
    reviewer: str,
    review_notes: str = "",
    apply_confirmation: str = "",
    attempt_apply: bool = False,
    active_rule_files: dict[str, str | Path] | None = None,
    preflight_probe_path: str | Path | None = None,
    rule_activation_manifest_path: str | Path | None = None,
    agent_rule_activation_manifest_path: str | Path | None = None,
) -> ReviewPackageReport:
    """Build a curator review package for a learned candidate rule set."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = target_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    _copy_preflight_artifacts(
        target_dir,
        rule_set_path=rule_set_path,
        preflight_probe_path=preflight_probe_path,
        rule_activation_manifest_path=rule_activation_manifest_path,
        agent_rule_activation_manifest_path=agent_rule_activation_manifest_path,
    )

    curator_result = review_and_optionally_compile(
        rule_set_path,
        approved_rule_ids=approved_rule_ids or set(),
        rejected_rule_ids=rejected_rule_ids or set(),
        reviewer=reviewer,
        review_notes=review_notes,
        compile_overlays=True,
        output_dir=overlay_dir,
    )
    curator_result_path = target_dir / "curator_workflow_result.json"
    _write_json(curator_result_path, curator_result.to_dict())

    summary = summarize_candidate_rule_set(rule_set_path)
    summary_path = target_dir / "candidate_rule_review_summary.json"
    _write_json(summary_path, summary.to_dict())
    decision_summary = build_candidate_rule_decision_summary(
        rule_set_path,
        preflight_probe_path=target_dir / "preflight_probe",
    )
    decision_summary_path = target_dir / "candidate_rule_decision_summary.json"
    _write_json(decision_summary_path, decision_summary.to_dict())

    diff_report = diff_inactive_overlays(
        overlay_dir,
        active_rule_files=active_rule_files,
    )
    diff_path = overlay_dir / "overlay_diff_report.json"
    _write_json(diff_path, diff_report.to_dict())

    dry_run_report = dry_run_apply_overlays(
        overlay_dir,
        active_rule_files=active_rule_files,
        approved_update_rule_ids=approved_update_rule_ids,
    )
    dry_run_path = overlay_dir / "overlay_apply_dry_run_report.json"
    _write_json(dry_run_path, dry_run_report.to_dict())

    guarded_path = overlay_dir / "guarded_apply_report.json"
    active_rules_mutated = False
    if attempt_apply:
        guarded_report = apply_overlays_guarded(
            overlay_dir,
            confirmation=apply_confirmation,
            reviewer=reviewer,
            backup_dir=target_dir / "backups",
            rollback_manifest_path=target_dir / "backups" / "rollback_manifest.json",
            active_rule_files=active_rule_files,
            approved_update_rule_ids=approved_update_rule_ids,
        )
        active_rules_mutated = guarded_report.active_rules_mutated
        _write_json(guarded_path, guarded_report.to_dict())
    else:
        _write_json(
            guarded_path,
            {
                "confirmed": False,
                "apply_allowed": False,
                "applied_rule_ids": [],
                "skipped_rule_ids": [],
                "blocked_reasons": ["guarded apply was not requested"],
                "active_rules_mutated": False,
                "backup_files": {},
                "rollback_manifest": "",
            },
        )

    final_recommendation = build_curator_final_recommendation(target_dir)
    final_recommendation_path = target_dir / "curator_final_recommendation.json"
    _write_json(final_recommendation_path, final_recommendation.to_dict())

    lifecycle_files = collect_lifecycle_artifact_files(
        output_dir=target_dir,
        learned_candidate_rule_set=rule_set_path,
        overlay_dir=overlay_dir,
        existing_files={
            "learned_candidate_rule_set": str(rule_set_path),
            "candidate_rule_review_summary": str(summary_path),
            "candidate_rule_decision_summary": str(decision_summary_path),
            "curator_final_recommendation": str(final_recommendation_path),
            "curator_workflow_result": str(curator_result_path),
            "overlay_diff_report": str(diff_path),
            "overlay_apply_dry_run_report": str(dry_run_path),
            "guarded_overlay_apply_report": str(guarded_path),
        },
    )
    audit_markdown = build_agent_audit_markdown(
        {"agent_outputs": {"files": lifecycle_files}}
    )
    audit_path = target_dir / "materials_rule_review_package.md"
    audit_path.write_text(audit_markdown, encoding="utf-8")

    review_index_path = build_review_index(target_dir)

    report = ReviewPackageReport(
        rule_set_path=str(rule_set_path),
        output_dir=str(target_dir),
        summary_path=str(summary_path),
        curator_result_path=str(curator_result_path),
        overlay_dir=str(overlay_dir),
        overlay_diff_report_path=str(diff_path),
        overlay_apply_dry_run_report_path=str(dry_run_path),
        guarded_overlay_apply_report_path=str(guarded_path),
        audit_report_path=str(audit_path),
        review_index_path=review_index_path,
        apply_attempted=attempt_apply,
        active_rules_mutated=active_rules_mutated,
    )
    _write_json(target_dir / "review_package_report.json", report.to_dict())
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a learned materials-rule curator review package."
    )
    parser.add_argument("rule_set_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--approve", action="append", default=[])
    parser.add_argument("--reject", action="append", default=[])
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--attempt-apply", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument(
        "--preflight-probe",
        help="Path to a pipeline preflight_probe directory or preflight_probe.json file.",
    )
    parser.add_argument(
        "--rule-activation-manifest",
        help="Path to deterministic rule_activation_manifest.json.",
    )
    parser.add_argument(
        "--agent-rule-activation-manifest",
        help="Path to agent_rule_activation_manifest.json.",
    )

    args = parser.parse_args(argv)
    report = build_review_package(
        args.rule_set_path,
        output_dir=args.output_dir,
        approved_rule_ids=set(args.approve),
        rejected_rule_ids=set(args.reject),
        reviewer=args.reviewer,
        review_notes=args.notes,
        attempt_apply=args.attempt_apply,
        apply_confirmation=args.confirmation,
        preflight_probe_path=args.preflight_probe,
        rule_activation_manifest_path=args.rule_activation_manifest,
        agent_rule_activation_manifest_path=args.agent_rule_activation_manifest,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _copy_preflight_artifacts(
    target_dir: Path,
    *,
    rule_set_path: str | Path,
    preflight_probe_path: str | Path | None,
    rule_activation_manifest_path: str | Path | None,
    agent_rule_activation_manifest_path: str | Path | None,
) -> None:
    artifact_dir = target_dir / "preflight_probe"
    sources: dict[str, Path] = {}
    if preflight_probe_path:
        source = Path(preflight_probe_path)
        if source.is_dir():
            for name in (
                "dataset_contract.json",
                "rule_activation_manifest.json",
                "agent_rule_activation_manifest.json",
                "agent_rule_activation_review.json",
                "reviewed_rule_activation_manifest.json",
                "rule_activation_comparison_report.json",
                "rule_activation_application_audit.json",
                "preflight_probe.json",
                "materials_preflight_report.md",
            ):
                candidate = source / name
                if candidate.exists():
                    sources[name] = candidate
        elif source.exists():
            sources[source.name] = source

    if rule_activation_manifest_path:
        sources["rule_activation_manifest.json"] = Path(rule_activation_manifest_path)
    if agent_rule_activation_manifest_path:
        sources["agent_rule_activation_manifest.json"] = Path(
            agent_rule_activation_manifest_path
        )

    if not sources:
        sources.update(_preflight_sources_from_rule_set(rule_set_path))

    existing_sources = {
        name: path
        for name, path in sources.items()
        if path.exists()
        and (path.suffix.lower() == ".json" or name == "materials_preflight_report.md")
    }
    if not existing_sources:
        return

    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, source in existing_sources.items():
        shutil.copy2(source, artifact_dir / name)


def _preflight_sources_from_rule_set(rule_set_path: str | Path) -> dict[str, Path]:
    try:
        payload = json.loads(Path(rule_set_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    lifecycle = payload.get("lifecycle") or {}
    source_context = lifecycle.get("source_context") or {}
    preflight_files = source_context.get("preflight_files") or {}
    if not isinstance(preflight_files, dict):
        return {}

    file_names = {
        "dataset_contract": "dataset_contract.json",
        "rule_activation_manifest": "rule_activation_manifest.json",
        "agent_rule_activation_manifest": "agent_rule_activation_manifest.json",
        "agent_rule_activation_review": "agent_rule_activation_review.json",
        "reviewed_rule_activation_manifest": "reviewed_rule_activation_manifest.json",
        "rule_activation_comparison_report": "rule_activation_comparison_report.json",
        "rule_activation_application_audit": "rule_activation_application_audit.json",
        "preflight_probe": "preflight_probe.json",
        "materials_preflight_report": "materials_preflight_report.md",
    }
    return {
        file_names[key]: Path(str(value))
        for key, value in preflight_files.items()
        if key in file_names and value
    }


if __name__ == "__main__":
    main()
