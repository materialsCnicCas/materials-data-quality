"""Markdown audit reports for materials quality and agent outputs."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def build_agent_audit_markdown(
    pipeline_results: dict[str, Any],
    *,
    generated_at: datetime | None = None,
) -> str:
    """Render a compact human-readable audit report."""

    generated_at = generated_at or datetime.now()
    lines: list[str] = [
        "# Materials Data Quality Audit Report",
        "",
        f"- Generated at: `{generated_at.isoformat(timespec='seconds')}`",
    ]

    _append_dataset_overview(lines, pipeline_results)
    _append_rule_activation(lines, pipeline_results)
    _append_quality_summary(lines, pipeline_results)
    _append_agent_explanation(lines, pipeline_results)
    _append_rule_feedback_tool_analysis(lines, pipeline_results)
    _append_candidate_rules(lines, pipeline_results)
    _append_learned_rule_lifecycle(lines, pipeline_results)
    _append_files(lines, pipeline_results)

    return "\n".join(lines).rstrip() + "\n"


def _append_dataset_overview(lines: list[str], pipeline_results: dict[str, Any]) -> None:
    data_processing = pipeline_results.get("data_processing") or {}
    info = data_processing.get("processing_info") or {}
    preflight = pipeline_results.get("preflight_probe") or {}
    contract = preflight.get("dataset_contract") or {}
    applicability = contract.get("material_applicability") or {}

    lines.extend(
        [
            "",
            "## Dataset",
            "",
            f"- Name: `{contract.get('dataset_name') or 'unknown'}`",
            f"- Source format: `{contract.get('source_format') or (info.get('source') or {}).get('detected_format') or 'unknown'}`",
            f"- Rows: `{contract.get('row_count') or ((info.get('shapes') or {}).get('processed') or {}).get('rows') or 'unknown'}`",
            f"- Columns: `{contract.get('column_count') or ((info.get('shapes') or {}).get('processed') or {}).get('columns') or 'unknown'}`",
            f"- Identifier column: `{contract.get('identifier_column') or ((info.get('features') or {}).get('identifier_column')) or 'unknown'}`",
            f"- Semantic hints: `{', '.join(contract.get('semantic_hints') or []) or 'none'}`",
        ]
    )
    lines.extend(
        [
            f"- Domain scope: `{contract.get('domain_scope') or 'unknown'}`",
            f"- Assumed materials dataset: `{applicability.get('assumed_materials_dataset', 'unknown')}`",
            f"- Has material subdomain evidence: `{applicability.get('has_material_subdomain_evidence', 'unknown')}`",
            f"- Material subdomains: `{', '.join(applicability.get('subdomains') or []) or 'none'}`",
            f"- Applicability reason: `{applicability.get('reason') or 'n/a'}`",
        ]
    )


def _append_rule_activation(lines: list[str], pipeline_results: dict[str, Any]) -> None:
    preflight = pipeline_results.get("preflight_probe") or {}
    manifest = preflight.get("rule_activation_manifest") or {}
    agent_manifest = preflight.get("agent_rule_activation_manifest")
    comparison = preflight.get("rule_activation_comparison")
    application = preflight.get("rule_activation_application")

    lines.extend(["", "## Rule Activation", ""])
    _append_manifest_table(lines, manifest, "Deterministic baseline")
    if agent_manifest:
        _append_manifest_table(lines, agent_manifest, "Agent proposal")
    if comparison:
        lines.extend(
            [
                "",
                f"- Agent comparison changed packs: `{comparison.get('changed_count', 0)}`",
            ]
        )
        diffs = comparison.get("diffs") or []
        for diff in diffs[:10]:
            if diff.get("status") != "same":
                lines.append(
                    "- "
                    f"`{diff.get('rule_pack')}`: `{diff.get('status')}` "
                    f"(det={diff.get('deterministic_enabled')}, agent={diff.get('agent_enabled')})"
                )
    if application:
        lines.extend(["", "### Applied This Run", ""])
        for decision in (application.get("decisions") or [])[:10]:
            lines.append(
                "- "
                f"`{decision.get('rule_pack')}` -> "
                f"`{decision.get('rule_config_key') or 'reported-only'}` = "
                f"`{decision.get('new_value')}`; applied=`{decision.get('applied')}`"
            )


def _append_manifest_table(lines: list[str], manifest: dict[str, Any], title: str) -> None:
    if not manifest:
        return
    lines.extend(["", f"### {title}", ""])
    decisions = manifest.get("decisions") or []
    if not decisions:
        lines.append("- No decisions recorded.")
        return
    for decision in decisions[:12]:
        subdomains = decision.get("materials_subdomains") or []
        matched_fields = decision.get("matched_fields") or []
        criteria = decision.get("activation_criteria") or {}
        criteria_summary = _activation_criteria_summary(criteria)
        lines.append(
            "- "
            f"`{decision.get('rule_pack')}`: "
            f"`{'enabled' if decision.get('enabled') else 'disabled'}` "
            f"(confidence={_fmt_float(decision.get('confidence'))}; "
            f"subdomains={_inline_list(subdomains)})"
        )
        if matched_fields or criteria_summary:
            lines.append(
                "  - "
                f"matched_fields={_inline_list(matched_fields, limit=6)}; "
                f"criteria={criteria_summary or '`n/a`'}"
            )


def _append_quality_summary(lines: list[str], pipeline_results: dict[str, Any]) -> None:
    agent_outputs = pipeline_results.get("agent_outputs") or {}
    evidence = agent_outputs.get("evidence_bundle") or {}
    confidence = evidence.get("confidence_summaries") or {}
    final_rows = confidence.get("final_rows") or {}
    final_features = confidence.get("final_features") or {}
    low_samples = evidence.get("low_confidence_samples") or []
    low_features = evidence.get("low_confidence_features") or []
    layers = evidence.get("layer_summaries") or {}

    lines.extend(
        [
            "",
            "## Quality Summary",
            "",
            f"- Final row confidence mean/min: `{_fmt_float(final_rows.get('mean'))}` / `{_fmt_float(final_rows.get('min'))}`",
            f"- Final feature confidence mean/min: `{_fmt_float(final_features.get('mean'))}` / `{_fmt_float(final_features.get('min'))}`",
            f"- Low-confidence samples: `{len(low_samples)}`",
            f"- Low-confidence features: `{len(low_features)}`",
        ]
    )
    if low_samples:
        lines.append("")
        lines.append("### Lowest-Confidence Samples")
        for sample in low_samples[:10]:
            lines.append(
                f"- row `{sample.get('row_index')}` confidence=`{_fmt_float(sample.get('confidence_score'))}`"
            )
    if low_features:
        lines.append("")
        lines.append("### Lowest-Confidence Features")
        for feature in low_features[:10]:
            name = feature.get("column") or feature.get("feature")
            lines.append(
                f"- `{name}` confidence=`{_fmt_float(feature.get('confidence_score'))}`"
            )

    lines.extend(["", "### Layer Evidence", ""])
    for layer in ("nine_dimension", "sdad", "mdad", "srd", "bipartite_walk", "gat"):
        summary = layers.get(layer)
        if summary is not None:
            lines.append(f"- `{layer}`: {_compact(summary)}")


def _append_agent_explanation(lines: list[str], pipeline_results: dict[str, Any]) -> None:
    agent_outputs = pipeline_results.get("agent_outputs") or {}
    explanation = agent_outputs.get("explanation")
    if not explanation:
        return

    lines.extend(["", "## Agent Explanation", ""])
    if explanation.get("summary"):
        lines.append(explanation["summary"])
    for title, key in (
        ("Common Issue Patterns", "common_issue_patterns"),
        ("Recommended Actions", "recommended_actions"),
        ("Limitations", "limitations"),
    ):
        values = explanation.get(key) or []
        if values:
            lines.extend(["", f"### {title}", ""])
            for value in values[:10]:
                lines.append(f"- {value}")


def _append_rule_feedback_tool_analysis(
    lines: list[str],
    pipeline_results: dict[str, Any],
) -> None:
    agent_outputs = pipeline_results.get("agent_outputs") or {}
    analysis = agent_outputs.get("rule_feedback_tool_analysis")
    if not analysis:
        files = agent_outputs.get("files") or {}
        analysis = _load_json_artifact(files.get("rule_feedback_tool_analysis"))
    if not analysis:
        return

    cohort = analysis.get("cohort_summary") or {}
    feature_rows = analysis.get("feature_separation") or []
    threshold_rows = analysis.get("numeric_threshold_candidates") or []
    relationship_rows = analysis.get("pairwise_relationship_candidates") or []

    lines.extend(["", "## Rule Feedback Tool Analysis", ""])
    lines.extend(
        [
            f"- Abnormal samples analyzed: `{cohort.get('abnormal_count', 0)}`",
            f"- Normal samples analyzed: `{cohort.get('normal_count', 0)}`",
            f"- Numeric features analyzed: `{cohort.get('numeric_feature_count', 0)}`",
            f"- Threshold candidates: `{len(threshold_rows)}`",
            f"- Relationship candidates: `{len(relationship_rows)}`",
            f"- Active expert rules mutated by tools: `{analysis.get('active_rules_mutated', False)}`",
        ]
    )
    if feature_rows:
        lines.extend(["", "### Top Separating Features", ""])
        for item in feature_rows[:8]:
            lines.append(
                "- "
                f"`{item.get('feature')}` effect=`{_fmt_float(item.get('effect_size'))}` "
                f"direction=`{item.get('direction')}`"
            )
    if threshold_rows:
        lines.extend(["", "### Mined Threshold Candidates", ""])
        for item in threshold_rows[:8]:
            expression = item.get("expression") or {}
            lines.append(
                "- "
                f"`{item.get('feature')}` expression=`{expression}` "
                f"abnormal_coverage=`{_fmt_float(item.get('abnormal_coverage'))}` "
                f"normal_hit_rate=`{_fmt_float(item.get('normal_hit_rate'))}`"
            )


def _append_candidate_rules(lines: list[str], pipeline_results: dict[str, Any]) -> None:
    agent_outputs = pipeline_results.get("agent_outputs") or {}
    candidate_rules = agent_outputs.get("candidate_rules") or {}
    validation = agent_outputs.get("candidate_rule_validation") or {}
    rules = candidate_rules.get("rules") or []
    validations = validation.get("validations") or []
    validations_by_id = {
        item.get("rule_id"): item
        for item in validations
        if isinstance(item, dict)
    }

    if not rules and not validation:
        return
    lines.extend(
        [
            "",
            "## Candidate Rules",
            "",
            f"- Total proposed: `{candidate_rules.get('rules') and len(rules) or validation.get('total_rules', 0)}`",
            f"- Accepted for review: `{validation.get('accepted_for_review', 0)}`",
            f"- Promotion allowed: `{validation.get('promotion_allowed', False)}`",
        ]
    )
    for rule in rules[:10]:
        rule_id = rule.get("rule_id")
        item = validations_by_id.get(rule_id, {})
        metrics = item.get("replay_metrics") or {}
        lines.append(
            "- "
            f"`{rule_id}` ({rule.get('layer')}, {rule.get('severity')}): "
            f"accepted=`{item.get('accepted_for_review')}`; "
            f"matched=`{metrics.get('expression_matched_count')}`; "
            f"false_positive_pressure=`{metrics.get('false_positive_pressure')}`"
        )


def _append_learned_rule_lifecycle(
    lines: list[str],
    pipeline_results: dict[str, Any],
) -> None:
    agent_outputs = pipeline_results.get("agent_outputs") or {}
    files = agent_outputs.get("files") or {}
    stored_rule_set = _load_json_artifact(
        agent_outputs.get("learned_candidate_rule_set")
        or files.get("learned_candidate_rule_set")
    )
    overlay_diff = _load_json_artifact(
        agent_outputs.get("rule_feedback_promote_overlay_diff_report")
        or files.get("rule_feedback_promote_overlay_diff_report")
        or
        agent_outputs.get("overlay_diff_report")
        or files.get("overlay_diff_report")
    )
    dry_run = _load_json_artifact(
        agent_outputs.get("rule_feedback_promote_apply_dry_run_report")
        or files.get("rule_feedback_promote_apply_dry_run_report")
        or
        agent_outputs.get("overlay_apply_dry_run_report")
        or files.get("overlay_apply_dry_run_report")
    )
    guarded_apply = _load_json_artifact(
        agent_outputs.get("rule_feedback_promote_apply_report")
        or files.get("rule_feedback_promote_apply_report")
        or
        agent_outputs.get("guarded_overlay_apply_report")
        or files.get("guarded_overlay_apply_report")
    )
    promote_result = agent_outputs.get("rule_feedback_promote_result") or {}
    promote_summary = promote_result.get("review_package") or {}
    promote_report = _load_json_artifact(
        agent_outputs.get("rule_feedback_promote_report")
        or files.get("rule_feedback_promote_report")
    )
    user_decision = agent_outputs.get("rule_feedback_user_decision") or {}

    if not any(
        (
            stored_rule_set,
            overlay_diff,
            dry_run,
            guarded_apply,
            promote_result,
            promote_report,
            user_decision,
        )
    ):
        return

    lines.extend(["", "## Learned Rule Lifecycle", ""])
    if user_decision or promote_result or promote_report:
        selected_rule_ids = (
            user_decision.get("selected_rule_ids")
            or promote_result.get("approved_rule_ids")
            or []
        )
        final_rule_set_path = (
            promote_result.get("rule_set_path")
            or promote_report.get("rule_set_path")
            or files.get("learned_candidate_rule_set")
        )
        lines.extend(
            [
                "- User requested active-library promotion: "
                f"`{user_decision.get('join_active_rule_library', bool(promote_result))}`",
                f"- User-selected candidate rules: `{len(selected_rule_ids)}`",
                f"- Promotion apply requested: `{promote_result.get('apply_requested', promote_report.get('apply_attempted', False))}`",
                f"- Final learned rule set path: `{final_rule_set_path or 'n/a'}`",
            ]
        )
        if promote_summary:
            lines.append(
                "- Promote package active expert rules mutated: "
                f"`{promote_summary.get('active_rules_mutated', False)}`"
            )
    if stored_rule_set:
        rules = stored_rule_set.get("rules") or []
        counts = _count_review_statuses(rules)
        lifecycle = stored_rule_set.get("lifecycle") or {}
        lines.extend(
            [
                f"- Stored candidate rule set: `{stored_rule_set.get('dataset_name') or 'unknown'}`",
                f"- Review status counts: `{counts}`",
                f"- Last reviewer: `{lifecycle.get('last_reviewer') or 'n/a'}`",
                f"- Active expert rules mutated by storage/review: `{lifecycle.get('active_rules_mutated', False)}`",
            ]
        )
        for rule in rules[:10]:
            lifecycle = rule.get("lifecycle") or {}
            extra = ""
            if lifecycle.get("blocked_target_rule_id"):
                extra = (
                    f"; blocked_target=`{lifecycle.get('blocked_target_rule_id')}`"
                )
            if lifecycle.get("blocked_reason"):
                extra += f"; reason=`{lifecycle.get('blocked_reason')}`"
            lines.append(
                "- "
                f"`{rule.get('rule_id')}` -> "
                f"`{lifecycle.get('review_status', 'draft')}`; "
                f"active=`{lifecycle.get('active', False)}`"
                f"{extra}"
            )

    if overlay_diff:
        lines.extend(["", "### Overlay Diff", ""])
        lines.extend(
            [
                f"- Total overlay rules: `{overlay_diff.get('total_overlay_rules', 0)}`",
                f"- Actions: `{overlay_diff.get('actions') or {}}`",
                f"- Active expert rules mutated by diff: `{overlay_diff.get('active_rules_mutated', False)}`",
            ]
        )
        for diff in (overlay_diff.get("diffs") or [])[:10]:
            lines.append(
                "- "
                f"`{diff.get('learned_rule_id')}` -> "
                f"`{diff.get('action')}` on `{diff.get('layer')}`/"
                f"`{diff.get('target_rule_id')}`; "
                f"conflict=`{diff.get('conflict')}`"
            )

    if dry_run:
        lines.extend(["", "### Overlay Apply Dry Run", ""])
        lines.extend(
            [
                f"- Apply allowed: `{dry_run.get('apply_allowed', False)}`",
                f"- Would mutate files: `{len(dry_run.get('would_mutate_files') or [])}`",
                f"- Blocked reasons: `{dry_run.get('blocked_reasons') or []}`",
                f"- Active expert rules mutated by dry-run: `{dry_run.get('active_rules_mutated', False)}`",
            ]
        )

    if guarded_apply:
        lines.extend(["", "### Guarded Overlay Apply", ""])
        lines.extend(
            [
                f"- Confirmed: `{guarded_apply.get('confirmed', False)}`",
                f"- Apply allowed: `{guarded_apply.get('apply_allowed', False)}`",
                f"- Applied rules: `{len(guarded_apply.get('applied_rule_ids') or [])}`",
                f"- Skipped rules: `{len(guarded_apply.get('skipped_rule_ids') or [])}`",
                f"- Blocked reasons: `{guarded_apply.get('blocked_reasons') or []}`",
                f"- Active expert rules mutated: `{guarded_apply.get('active_rules_mutated', False)}`",
                f"- Backup files: `{len(guarded_apply.get('backup_files') or {})}`",
                f"- Rollback manifest: `{guarded_apply.get('rollback_manifest') or 'n/a'}`",
            ]
        )


def _append_files(lines: list[str], pipeline_results: dict[str, Any]) -> None:
    files: dict[str, Any] = {}
    for section in ("preflight_probe", "agent_outputs", "final_outputs"):
        section_files = (pipeline_results.get(section) or {}).get("files")
        if isinstance(section_files, dict):
            files.update(section_files)
        elif section == "final_outputs" and isinstance(pipeline_results.get(section), dict):
            files.update(pipeline_results[section])

    if not files:
        return
    lines.extend(["", "## Files", ""])
    for name, path in files.items():
        lines.append(f"- `{name}`: `{path}`")


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _compact(value: Any, limit: int = 220) -> str:
    rendered = str(value).replace("\n", " ")
    if len(rendered) <= limit:
        return f"`{rendered}`"
    return f"`{rendered[:limit]}...`"


def _inline_list(values: Any, *, limit: int = 5) -> str:
    if not isinstance(values, (list, tuple)) or not values:
        return "`none`"
    rendered = [str(value) for value in values[:limit]]
    if len(values) > limit:
        rendered.append("...")
    return "`" + ", ".join(rendered) + "`"


def _activation_criteria_summary(criteria: Any) -> str:
    if not isinstance(criteria, dict) or not criteria:
        return ""
    parts: list[str] = []
    if "matched_feature_score" in criteria:
        parts.append(f"matched_score={criteria.get('matched_feature_score')}")
    if "min_feature_matches" in criteria:
        parts.append(f"threshold={criteria.get('min_feature_matches')}")
    if "enabled_by_default" in criteria:
        parts.append(f"default={criteria.get('enabled_by_default')}")
    if not parts:
        return _compact(criteria, limit=140)
    return "`" + "; ".join(parts) + "`"


def _load_json_artifact(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        path = Path(str(value))
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _count_review_statuses(rules: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rule in rules:
        lifecycle = rule.get("lifecycle") or {}
        status = str(lifecycle.get("review_status") or "draft")
        counts[status] = counts.get(status, 0) + 1
    return counts
