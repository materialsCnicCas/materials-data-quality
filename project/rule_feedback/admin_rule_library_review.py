"""Administrator-only rule-library self review.

This module is intentionally separate from the normal data-quality pipeline.
It produces reports for system maintainers and never mutates active rule files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project.agents.rule_library_review_agent import (
    RuleLibraryReviewAgent,
    build_rule_library_review_payload,
)
from project.rule_feedback.overlay_review import diff_inactive_overlays


def build_admin_rule_library_review(
    *,
    candidate_rules_payload: dict[str, Any],
    overlay_path: str | Path,
    dataset_name: str = "materials-rule-library",
    output_dir: str | Path,
    use_agent: bool = False,
    active_rule_files: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Build an administrator self-review report for learned vs active rules."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    diff_report = diff_inactive_overlays(
        overlay_path,
        active_rule_files=active_rule_files,
    ).to_dict()
    payload = build_rule_library_review_payload(
        dataset_name=dataset_name,
        candidate_rules_payload=candidate_rules_payload,
        diff_report=diff_report,
    )

    diff_path = target_dir / "admin_rule_library_overlay_diff.json"
    payload_path = target_dir / "admin_rule_library_review_payload.json"
    _write_json(diff_path, diff_report)
    _write_json(payload_path, payload)

    report: dict[str, Any] = {
        "dataset_name": dataset_name,
        "mode": "administrator_rule_library_self_review",
        "active_rules_mutated": False,
        "overlay_path": str(overlay_path),
        "diff_report_path": str(diff_path),
        "review_payload_path": str(payload_path),
        "issue_count": len(payload.get("issues") or []),
        "issues": payload.get("issues") or [],
    }

    if use_agent and payload.get("issues"):
        agent_report = RuleLibraryReviewAgent().review_library(payload).to_dict()
        agent_path = target_dir / "admin_rule_library_agent_review.json"
        _write_json(agent_path, agent_report)
        report["agent_review_path"] = str(agent_path)
        report["agent_review"] = agent_report

    report_path = target_dir / "admin_rule_library_review_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Administrator-only materials rule-library self review. "
            "This command writes reports only and never changes rule files."
        )
    )
    parser.add_argument("--candidate-rules", required=True)
    parser.add_argument("--overlay-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="materials-rule-library")
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Call DeepSeek function-calling reviewer for issue summaries.",
    )
    args = parser.parse_args(argv)

    candidate_rules = json.loads(
        Path(args.candidate_rules).read_text(encoding="utf-8")
    )
    result = build_admin_rule_library_review(
        candidate_rules_payload=candidate_rules,
        overlay_path=args.overlay_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        use_agent=args.use_agent,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
