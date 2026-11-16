"""Discovery helpers for learned-rule lifecycle audit artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


LIFECYCLE_ARTIFACT_FILENAMES = {
    "candidate_rule_decision_summary": "candidate_rule_decision_summary.json",
    "curator_final_recommendation": "curator_final_recommendation.json",
    "overlay_compile_report": "compile_report.json",
    "overlay_rollback_manifest": "rollback_manifest.json",
    "overlay_diff_report": "overlay_diff_report.json",
    "overlay_apply_dry_run_report": "overlay_apply_dry_run_report.json",
    "guarded_overlay_apply_report": "guarded_apply_report.json",
}

ALTERNATE_LIFECYCLE_FILENAMES = {
    "guarded_overlay_apply_report": (
        "guarded_apply_blocked_report.json",
        "guarded_overlay_apply_report.json",
    ),
}


def collect_lifecycle_artifact_files(
    *,
    output_dir: str | Path | None = None,
    learned_candidate_rule_set: str | Path | None = None,
    overlay_dir: str | Path | None = None,
    existing_files: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Collect known learned-rule lifecycle artifact paths if they exist."""

    collected: dict[str, str] = {
        str(key): str(value)
        for key, value in (existing_files or {}).items()
        if value
    }
    if learned_candidate_rule_set:
        path = Path(learned_candidate_rule_set)
        if path.exists():
            collected["learned_candidate_rule_set"] = str(path)

    roots = _candidate_roots(
        output_dir=output_dir,
        learned_candidate_rule_set=learned_candidate_rule_set,
        overlay_dir=overlay_dir,
        existing_files=collected,
    )
    for key, filename in LIFECYCLE_ARTIFACT_FILENAMES.items():
        if key in collected and Path(collected[key]).exists():
            continue
        found = _find_first_existing(roots, (filename,) + ALTERNATE_LIFECYCLE_FILENAMES.get(key, ()))
        if found is not None:
            collected[key] = str(found)
    _attach_promote_package_artifacts(collected)
    return collected


def attach_lifecycle_artifacts(
    pipeline_results: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    overlay_dir: str | Path | None = None,
) -> dict[str, str]:
    """Attach discovered lifecycle artifacts to `agent_outputs.files`."""

    agent_outputs = pipeline_results.setdefault("agent_outputs", {})
    files = agent_outputs.setdefault("files", {})
    discovered = collect_lifecycle_artifact_files(
        output_dir=output_dir,
        learned_candidate_rule_set=files.get("learned_candidate_rule_set"),
        overlay_dir=overlay_dir,
        existing_files=files,
    )
    files.update(discovered)
    return discovered


def _candidate_roots(
    *,
    output_dir: str | Path | None,
    learned_candidate_rule_set: str | Path | None,
    overlay_dir: str | Path | None,
    existing_files: dict[str, str],
) -> list[Path]:
    roots: list[Path] = []
    if overlay_dir:
        roots.append(Path(overlay_dir))
    for key in (
        "overlay_compile_report",
        "overlay_diff_report",
        "overlay_apply_dry_run_report",
        "guarded_overlay_apply_report",
    ):
        value = existing_files.get(key)
        if value:
            roots.append(Path(value).parent)
    learned_path = (
        Path(learned_candidate_rule_set)
        if learned_candidate_rule_set is not None
        else (
            Path(existing_files["learned_candidate_rule_set"])
            if existing_files.get("learned_candidate_rule_set")
            else None
        )
    )
    if learned_path is not None:
        roots.append(learned_path.parent)
        roots.append(learned_path.parent / "overlays")
    if output_dir:
        output_path = Path(output_dir)
        roots.extend(
            [
                output_path,
                output_path / "agent_outputs",
                output_path / "overlays",
                output_path / "learned",
                output_path / "learned" / "overlays",
                output_path / "agent_outputs" / "rule_feedback_review_package",
                output_path / "agent_outputs" / "rule_feedback_review_package" / "overlays",
                output_path / "agent_outputs" / "rule_feedback_promote_package",
                output_path / "agent_outputs" / "rule_feedback_promote_package" / "overlays",
            ]
        )
    return _dedupe_existing_roots(roots)


def _attach_promote_package_artifacts(collected: dict[str, str]) -> None:
    """Attach explicit promote artifact aliases when a promote package exists."""

    package = collected.get("rule_feedback_promote_package")
    if not package:
        promote_report = collected.get("rule_feedback_promote_report")
        if promote_report:
            package = str(Path(promote_report).parent)
    if not package:
        apply_report = collected.get("rule_feedback_promote_apply_report")
        if apply_report:
            package = str(Path(apply_report).parent.parent)
    if not package:
        return

    package_path = Path(package)
    promote_paths = {
        "rule_feedback_promote_package": package_path,
        "rule_feedback_promote_report": package_path / "review_package_report.json",
        "rule_feedback_promote_review_index": package_path / "review_index.html",
        "rule_feedback_promote_overlay_diff_report": package_path / "overlays" / "overlay_diff_report.json",
        "rule_feedback_promote_apply_dry_run_report": package_path / "overlays" / "overlay_apply_dry_run_report.json",
        "rule_feedback_promote_apply_report": package_path / "overlays" / "guarded_apply_report.json",
    }
    for key, path in promote_paths.items():
        if path.exists():
            collected[key] = str(path)


def _find_first_existing(roots: list[Path], filenames: tuple[str, ...]) -> Path | None:
    for root in roots:
        for filename in filenames:
            path = root / filename
            if path.exists():
                return path
    return None


def _dedupe_existing_roots(roots: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen or not root.exists():
            continue
        seen.add(key)
        deduped.append(root)
    return deduped
