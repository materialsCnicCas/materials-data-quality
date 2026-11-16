"""Static HTML index for learned-rule review packages."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from typing import Any


def build_review_index(
    package_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> str:
    """Render a static HTML index for a review package."""

    root = Path(package_dir)
    report = _read_json(root / "review_package_report.json")
    summary = _read_json(root / "candidate_rule_review_summary.json")
    decision_summary = _read_json(root / "candidate_rule_decision_summary.json")
    final_recommendation = _read_json(root / "curator_final_recommendation.json")
    diff = _read_json(root / "overlays" / "overlay_diff_report.json")
    dry_run = _read_json(root / "overlays" / "overlay_apply_dry_run_report.json")
    guarded = _read_json(root / "overlays" / "guarded_apply_report.json")
    activation_manifest = _read_activation_manifest(root)

    html = _render_html(
        root,
        report,
        summary,
        decision_summary,
        final_recommendation,
        diff,
        dry_run,
        guarded,
        activation_manifest,
    )
    target = Path(output_path) if output_path else root / "review_index.html"
    target.write_text(html, encoding="utf-8")
    return str(target)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a static HTML index for a learned-rule review package."
    )
    parser.add_argument("package_dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(build_review_index(args.package_dir, output_path=args.output))


def _render_html(
    root: Path,
    report: dict[str, Any],
    summary: dict[str, Any],
    decision_summary: dict[str, Any],
    final_recommendation: dict[str, Any],
    diff: dict[str, Any],
    dry_run: dict[str, Any],
    guarded: dict[str, Any],
    activation_manifest: dict[str, Any],
) -> str:
    dataset = summary.get("dataset_name") or "unknown"
    apply_allowed = dry_run.get("apply_allowed", False)
    active_mutated = guarded.get("active_rules_mutated", False)
    status_class = "ok" if apply_allowed and not active_mutated else "warn"
    rules = summary.get("items") or []
    decision_items = decision_summary.get("items") or []
    final_items = final_recommendation.get("rules") or []
    diffs = diff.get("diffs") or []
    files = _package_files(root, report)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Materials Rule Review Package</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #202124; background: #f6f7f9; }}
    header {{ background: #17324d; color: white; padding: 24px 32px; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ background: white; border: 1px solid #d8dde6; border-radius: 6px; padding: 18px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #e2e6ee; border-radius: 6px; padding: 12px; background: #fbfcfe; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 6px; }}
    .ok {{ color: #0f6b3e; }}
    .warn {{ color: #9a5b00; }}
    .bad {{ color: #a32020; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e7ebf0; padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f3f7; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
    a {{ color: #165d9c; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <header>
    <h1>Materials Rule Review Package</h1>
    <div>Dataset: <strong>{escape(str(dataset))}</strong></div>
  </header>
  <main>
    <section>
      <h2>Review Status</h2>
      <div class="grid">
        <div class="metric">Total rules<strong>{summary.get('total_rules', 0)}</strong></div>
        <div class="metric">Review counts<strong>{escape(str(summary.get('counts_by_status') or {}))}</strong></div>
        <div class="metric">Diff actions<strong>{escape(str(diff.get('actions') or {}))}</strong></div>
        <div class="metric">Dry-run apply<strong class="{status_class}">{escape(str(apply_allowed))}</strong></div>
        <div class="metric">Active rules mutated<strong class="{('bad' if active_mutated else 'ok')}">{escape(str(active_mutated))}</strong></div>
      </div>
    </section>
    <section>
      <h2>Final Recommendation</h2>
      <div class="grid">
        <div class="metric">Overall<strong>{escape(str(final_recommendation.get('overall_recommendation') or 'n/a'))}</strong></div>
        <div class="metric">Rule recommendations<strong>{escape(str((final_recommendation.get('summary') or {}).get('rule_recommendation_counts') or {}))}</strong></div>
        <div class="metric">Dry-run allowed<strong>{escape(str((final_recommendation.get('summary') or {}).get('dry_run_apply_allowed')))}</strong></div>
        <div class="metric">Active mutated<strong class="{('bad' if final_recommendation.get('active_rules_mutated') else 'ok')}">{escape(str(final_recommendation.get('active_rules_mutated', False)))}</strong></div>
      </div>
      {_render_final_recommendation_table(final_items)}
    </section>
    <section>
      <h2>Candidate Rules</h2>
      {_render_rules_table(rules)}
    </section>
    <section>
      <h2>Curator Decision Summary</h2>
      <div class="grid">
        <div class="metric">Actions<strong>{escape(str(decision_summary.get('action_counts') or {}))}</strong></div>
        <div class="metric">Risks<strong>{escape(str(decision_summary.get('risk_counts') or {}))}</strong></div>
        <div class="metric">Subdomains<strong>{escape(str(decision_summary.get('subdomain_counts') or {}))}</strong></div>
        <div class="metric">Layers<strong>{escape(str(decision_summary.get('layer_counts') or {}))}</strong></div>
      </div>
      {_render_decision_table(decision_items)}
    </section>
    <section>
      <h2>Overlay Diff</h2>
      {_render_diffs_table(diffs)}
    </section>
    <section>
      <h2>Rule Activation Evidence</h2>
      {_render_activation_table(activation_manifest)}
    </section>
    <section>
      <h2>Apply Gates</h2>
      <ul>
        <li>Dry-run blocked reasons: <code>{escape(str(dry_run.get('blocked_reasons') or []))}</code></li>
        <li>Guarded apply confirmed: <code>{escape(str(guarded.get('confirmed', False)))}</code></li>
        <li>Guarded apply blocked reasons: <code>{escape(str(guarded.get('blocked_reasons') or []))}</code></li>
        <li>Rollback manifest: <code>{escape(str(guarded.get('rollback_manifest') or 'n/a'))}</code></li>
      </ul>
    </section>
    <section>
      <h2>Files</h2>
      {_render_files_list(root, files)}
    </section>
  </main>
</body>
</html>
"""


def _render_rules_table(rules: list[dict[str, Any]]) -> str:
    if not rules:
        return "<p>No candidate rules found.</p>"
    rows = []
    for rule in rules:
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(rule.get('rule_id')))}</code></td>"
            f"<td>{escape(str(rule.get('layer')))}</td>"
            f"<td>{escape(str(rule.get('review_status')))}</td>"
            f"<td>{escape(str(rule.get('active')))}</td>"
            f"<td>{escape(', '.join(rule.get('required_fields') or []))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Rule</th><th>Layer</th><th>Status</th>"
        "<th>Active</th><th>Required Fields</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_final_recommendation_table(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<p>No final recommendation found.</p>"
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(item.get('rule_id')))}</code></td>"
            f"<td>{escape(str(item.get('final_recommendation')))}</td>"
            f"<td>{escape(str(item.get('decision_action')))}</td>"
            f"<td>{escape(str(item.get('risk_level')))}</td>"
            f"<td>{escape(str(item.get('overlay_action')))}</td>"
            f"<td>{escape(_join_limited(item.get('reasons') or [], limit=4))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Rule</th><th>Final</th><th>Decision Action</th>"
        "<th>Risk</th><th>Overlay</th><th>Reasons</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_decision_table(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<p>No decision summary found.</p>"
    rows = []
    for item in items:
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(item.get('rule_id')))}</code></td>"
            f"<td>{escape(str(item.get('recommended_action')))}</td>"
            f"<td>{escape(str(item.get('risk_level')))}</td>"
            f"<td>{escape(str(item.get('layer')))}</td>"
            f"<td>{escape(_join_limited(item.get('materials_subdomains') or []))}</td>"
            f"<td>{escape(_join_limited(item.get('supporting_features') or [], limit=6))}</td>"
            f"<td>{escape(_traceability_text(item.get('preflight_traceability') or {}))}</td>"
            f"<td>{escape(str(item.get('rationale') or ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Rule</th><th>Action</th><th>Risk</th>"
        "<th>Layer</th><th>Subdomains</th><th>Supporting Features</th>"
        "<th>Preflight Trace</th><th>Rationale</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_diffs_table(diffs: list[dict[str, Any]]) -> str:
    if not diffs:
        return "<p>No overlay diffs found.</p>"
    rows = []
    for diff in diffs:
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(diff.get('learned_rule_id')))}</code></td>"
            f"<td>{escape(str(diff.get('layer')))}</td>"
            f"<td>{escape(str(diff.get('action')))}</td>"
            f"<td>{escape(str(diff.get('conflict')))}</td>"
            f"<td><code>{escape(str(diff.get('target_rule_id')))}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Rule</th><th>Layer</th><th>Action</th>"
        "<th>Conflict</th><th>Target</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_activation_table(manifest: dict[str, Any]) -> str:
    decisions = manifest.get("decisions") or []
    if not decisions:
        return "<p>No material rule activation evidence found in this package.</p>"
    rows = []
    for decision in decisions[:20]:
        criteria = decision.get("activation_criteria") or {}
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(decision.get('rule_pack')))}</code></td>"
            f"<td>{escape('enabled' if decision.get('enabled') else 'disabled')}</td>"
            f"<td>{escape(_join_limited(decision.get('materials_subdomains') or []))}</td>"
            f"<td>{escape(_join_limited(decision.get('matched_fields') or [], limit=8))}</td>"
            f"<td>{escape(_criteria_text(criteria))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Rule Pack</th><th>Status</th>"
        "<th>Materials Subdomains</th><th>Matched Fields</th>"
        "<th>Activation Criteria</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_files_list(root: Path, files: dict[str, str]) -> str:
    items = []
    for name, path in files.items():
        target = Path(path)
        try:
            href = target.relative_to(root).as_posix()
        except ValueError:
            href = target.resolve().as_uri()
        items.append(
            f"<li><a href=\"{escape(href)}\">{escape(name)}</a> "
            f"<code>{escape(str(path))}</code></li>"
        )
    return "<ul>" + "".join(items) + "</ul>"


def _package_files(root: Path, report: dict[str, Any]) -> dict[str, str]:
    files = {
        "candidate_rule_review_summary": root / "candidate_rule_review_summary.json",
        "candidate_rule_decision_summary": root / "candidate_rule_decision_summary.json",
        "curator_final_recommendation": root / "curator_final_recommendation.json",
        "curator_workflow_result": root / "curator_workflow_result.json",
        "review_package_report": root / "review_package_report.json",
        "materials_rule_review_package": root / "materials_rule_review_package.md",
        "overlay_diff_report": root / "overlays" / "overlay_diff_report.json",
        "overlay_apply_dry_run_report": root / "overlays" / "overlay_apply_dry_run_report.json",
        "guarded_apply_report": root / "overlays" / "guarded_apply_report.json",
        "rule_activation_manifest": root / "preflight_probe" / "rule_activation_manifest.json",
        "agent_rule_activation_manifest": root / "preflight_probe" / "agent_rule_activation_manifest.json",
        "agent_rule_activation_review": root / "preflight_probe" / "agent_rule_activation_review.json",
        "reviewed_rule_activation_manifest": root / "preflight_probe" / "reviewed_rule_activation_manifest.json",
        "materials_preflight_report": root / "preflight_probe" / "materials_preflight_report.md",
    }
    if report.get("rule_set_path"):
        files["learned_candidate_rule_set"] = Path(str(report["rule_set_path"]))
    return {name: str(path) for name, path in files.items() if Path(path).exists()}


def _read_activation_manifest(root: Path) -> dict[str, Any]:
    candidates = (
        root / "preflight_probe" / "reviewed_rule_activation_manifest.json",
        root / "preflight_probe" / "agent_rule_activation_manifest.json",
        root / "preflight_probe" / "rule_activation_manifest.json",
        root / "reviewed_rule_activation_manifest.json",
        root / "agent_rule_activation_manifest.json",
        root / "rule_activation_manifest.json",
    )
    for path in candidates:
        payload = _read_json(path)
        if payload:
            return payload
    return {}


def _join_limited(values: list[Any], *, limit: int = 5) -> str:
    rendered = [str(value) for value in values[:limit]]
    if len(values) > limit:
        rendered.append("...")
    return ", ".join(rendered) if rendered else "none"


def _criteria_text(criteria: dict[str, Any]) -> str:
    parts: list[str] = []
    if "matched_feature_score" in criteria:
        parts.append(f"matched_score={criteria.get('matched_feature_score')}")
    if "min_feature_matches" in criteria:
        parts.append(f"threshold={criteria.get('min_feature_matches')}")
    if "enabled_by_default" in criteria:
        parts.append(f"default={criteria.get('enabled_by_default')}")
    return "; ".join(parts) if parts else "n/a"


def _traceability_text(traceability: dict[str, Any]) -> str:
    if not traceability or not traceability.get("has_preflight_evidence"):
        return "no preflight evidence"
    packs = traceability.get("matched_rule_packs") or []
    pack_ids = [str(item.get("rule_pack")) for item in packs[:3]]
    parts = []
    if pack_ids:
        parts.append("packs=" + ", ".join(pack_ids))
    subdomains = traceability.get("contract_subdomain_overlap") or []
    if subdomains:
        parts.append("subdomains=" + ", ".join(str(item) for item in subdomains[:4]))
    fields = traceability.get("contract_field_overlap") or []
    if fields:
        parts.append("fields=" + ", ".join(str(item) for item in fields[:4]))
    return "; ".join(parts) if parts else "preflight present; no direct match"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    main()
